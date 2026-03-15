# src/openbrowser/daemon/server.py
"""OpenBrowser daemon -- persistent browser session over Unix socket.

Holds a browser session and CodeAgent namespace in memory.
CLI clients connect via Unix socket to execute code.

Usage:
    python -m openbrowser.daemon.server
"""

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

from openbrowser.daemon import IS_WINDOWS, WINDOWS_PORT, get_pid_path, get_socket_path

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT = 600  # 10 minutes
DEFAULT_EXEC_TIMEOUT = 300  # 5 minutes max per code execution


def _read_pid() -> int | None:
    """Read PID from file, return None if stale or missing."""
    pid_path = get_pid_path()
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        # Check if process is alive
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        pid_path.unlink(missing_ok=True)
        return None


def _write_pid() -> None:
    pid_path = get_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    pid_path.chmod(0o600)


def _cleanup_pid() -> None:
    get_pid_path().unlink(missing_ok=True)
    get_socket_path().unlink(missing_ok=True)


class DaemonServer:
    """Persistent browser automation daemon."""

    def __init__(self, idle_timeout: int = DEFAULT_IDLE_TIMEOUT, exec_timeout: int = DEFAULT_EXEC_TIMEOUT):
        self._idle_timeout = idle_timeout
        self._exec_timeout = exec_timeout
        self._last_activity = time.time()
        self._executor = None  # lazy
        self._session = None
        self._running = False
        self._server = None
        self._stop_event = asyncio.Event()
        self._exec_lock = asyncio.Lock()  # serialize code execution (stdout safety)
        self._init_lock = asyncio.Lock()  # guard lazy initialization

    def _build_browser_profile(self):
        """Build a BrowserProfile from config with daemon defaults."""
        from openbrowser.browser import BrowserProfile
        from openbrowser.config import get_default_profile, load_openbrowser_config

        config = load_openbrowser_config()
        profile_config = get_default_profile(config)
        profile_data = {
            'downloads_path': str(Path.home() / 'Downloads' / 'openbrowser-daemon'),
            'wait_between_actions': 0.5,
            'keep_alive': True,
            'user_data_dir': '~/.config/openbrowser/profiles/daemon',
            'device_scale_factor': 1.0,
            'disable_security': False,
            'headless': False,
            **profile_config,
        }
        return BrowserProfile(**profile_data)

    async def _ensure_executor(self):
        """Lazy-initialize browser + namespace on first request."""
        if self._executor is not None:
            return
        async with self._init_lock:
            if self._executor is not None:
                return  # another coroutine initialized while we waited

            # Suppress verbose logging for clean daemon output
            os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'critical'
            os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'
            logging.getLogger('openbrowser').setLevel(logging.ERROR)

            from openbrowser.browser import BrowserSession
            from openbrowser.code_use.executor import DEFAULT_MAX_OUTPUT_CHARS, CodeExecutor
            from openbrowser.code_use.namespace import create_namespace
            from openbrowser.tools.service import CodeAgentTools

            profile = self._build_browser_profile()
            session = BrowserSession(browser_profile=profile)
            await session.start()
            try:
                tools = CodeAgentTools()
                namespace = create_namespace(browser_session=session, tools=tools)

                try:
                    max_output = int(os.environ.get('OPENBROWSER_MAX_OUTPUT', '0'))
                except (ValueError, TypeError):
                    max_output = 0
                self._executor = CodeExecutor(max_output_chars=max_output if max_output > 0 else DEFAULT_MAX_OUTPUT_CHARS)
                self._executor.set_namespace(namespace)
                self._session = session
            except Exception:
                # Kill the browser if namespace/executor setup fails to prevent leak
                try:
                    await session.kill()
                except Exception:
                    pass
                raise

    _CDP_ERROR_KEYWORDS = ('connectionclosederror', 'no close frame', 'websocket', 'connection closed')

    def _is_connection_error(self, text: str) -> bool:
        """Return True if the error indicates a dead CDP/browser connection."""
        lower = text.lower()
        return any(kw in lower for kw in self._CDP_ERROR_KEYWORDS)

    async def _recover_browser_session(self):
        """Tear down dead browser and rebuild session + namespace."""
        logger.info('CDP connection lost -- recovering browser session')
        if self._session:
            try:
                await self._session.kill()
            except Exception:
                pass

        from openbrowser.browser import BrowserSession
        from openbrowser.code_use.namespace import create_namespace
        from openbrowser.tools.service import CodeAgentTools

        profile = self._build_browser_profile()
        session = BrowserSession(browser_profile=profile)
        await session.start()
        try:
            tools = CodeAgentTools()
            namespace = create_namespace(browser_session=session, tools=tools)
            self._executor.set_namespace(namespace)
            self._session = session
        except Exception:
            try:
                await session.kill()
            except Exception:
                pass
            raise
        logger.info('Browser session recovered successfully')

    async def _handle_request(self, data: dict) -> dict:
        """Handle a single JSON request."""
        self._last_activity = time.time()
        action = data.get('action', '')
        req_id = data.get('id', 0)

        if action == 'execute':
            code = data.get('code', '')
            if not code.strip():
                return {'id': req_id, 'success': False, 'output': '', 'error': 'No code provided'}
            await self._ensure_executor()
            try:
                async with self._exec_lock:
                    result = await asyncio.wait_for(
                        self._executor.execute(code), timeout=self._exec_timeout
                    )
            except asyncio.TimeoutError:
                return {
                    'id': req_id,
                    'success': False,
                    'output': '',
                    'error': f'Execution timed out after {self._exec_timeout}s',
                }

            # If error looks like dead CDP connection, recover and retry once
            error_text = result.error or result.output
            if not result.success and self._is_connection_error(error_text):
                try:
                    await self._recover_browser_session()
                    async with self._exec_lock:
                        result = await asyncio.wait_for(
                            self._executor.execute(code), timeout=self._exec_timeout
                        )
                except Exception as recovery_err:
                    logger.error('CDP recovery failed: %s', recovery_err)

            return {
                'id': req_id,
                'success': result.success,
                'output': result.output,
                'error': result.error,
            }

        elif action == 'status':
            return {
                'id': req_id,
                'success': True,
                'output': json.dumps({
                    'pid': os.getpid(),
                    'initialized': self._executor is not None,
                    'idle_timeout': self._idle_timeout,
                }),
                'error': None,
            }

        elif action == 'stop':
            self._running = False
            self._stop_event.set()
            return {'id': req_id, 'success': True, 'output': 'Daemon stopping', 'error': None}

        elif action == 'reset':
            if self._session:
                try:
                    await self._session.kill()
                except Exception:
                    pass
            self._executor = None
            self._session = None
            return {'id': req_id, 'success': True, 'output': 'Session reset', 'error': None}

        return {'id': req_id, 'success': False, 'output': '', 'error': f'Unknown action: {action}'}

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single client connection."""
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not raw:
                return
            data = json.loads(raw.decode())
            if not isinstance(data, dict):
                raise ValueError('Request must be a JSON object')
            response = await self._handle_request(data)
            writer.write(json.dumps(response).encode() + b'\n')
            await writer.drain()
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            try:
                err = {'id': 0, 'success': False, 'output': '', 'error': str(e)}
                writer.write(json.dumps(err).encode() + b'\n')
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _signal_shutdown(self):
        """Signal handler: mark daemon for shutdown."""
        self._running = False
        self._stop_event.set()

    async def _idle_check_loop(self):
        """Shut down daemon if idle beyond timeout."""
        while self._running:
            await asyncio.sleep(30)
            if time.time() - self._last_activity > self._idle_timeout:
                logger.info('Idle timeout reached, shutting down daemon')
                self._running = False
                self._stop_event.set()
                break

    async def run(self):
        """Start the daemon and listen for connections."""
        self._running = True
        self._last_activity = time.time()

        sock_path = get_socket_path()
        sock_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if another daemon is already running
        existing_pid = _read_pid()
        if existing_pid and existing_pid != os.getpid():
            logger.error('Another daemon is already running (PID %d)', existing_pid)
            return

        # Clean up stale socket
        sock_path.unlink(missing_ok=True)

        _write_pid()

        try:
            if IS_WINDOWS:
                # Windows: TCP on loopback only (no Unix sockets available).
                # Any local process can connect; consider a shared-secret
                # auth token if multi-user security is required.
                self._server = await asyncio.start_server(
                    self._handle_client, '127.0.0.1', WINDOWS_PORT
                )
            else:
                self._server = await asyncio.start_unix_server(
                    self._handle_client, path=str(sock_path)
                )
                # Restrict socket permissions to owner only
                os.chmod(str(sock_path), 0o600)

            # Register signal handlers on the running event loop
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, self._signal_shutdown)
                except (NotImplementedError, RuntimeError):
                    pass  # Windows or non-main thread

            # Start idle timeout checker
            idle_task = asyncio.create_task(self._idle_check_loop())

            async with self._server:
                await self._stop_event.wait()

            idle_task.cancel()

        finally:
            if self._session:
                try:
                    await self._session.kill()
                except Exception:
                    pass
            _cleanup_pid()


async def _main():
    daemon = DaemonServer()
    await daemon.run()


if __name__ == '__main__':
    asyncio.run(_main())
