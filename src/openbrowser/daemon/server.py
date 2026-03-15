# src/openbrowser/daemon/server.py
"""OpenBrowser daemon -- persistent browser session over Unix socket.

Holds a browser session and CodeAgent namespace in memory.
CLI clients connect via Unix socket to execute code.

Usage:
    python -m openbrowser.daemon.server          # foreground
    python -m openbrowser.daemon.server --bg     # background (called by client auto-start)
"""

import asyncio
import json
import logging
import os
import platform
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Daemon files directory
DAEMON_DIR = Path.home() / '.openbrowser'
SOCKET_PATH = DAEMON_DIR / 'daemon.sock'
PID_PATH = DAEMON_DIR / 'daemon.pid'

# Windows fallback: TCP on localhost
IS_WINDOWS = platform.system() == 'Windows'
WINDOWS_PORT = 19222

DEFAULT_IDLE_TIMEOUT = 600  # 10 minutes


def _get_socket_path() -> Path:
    return Path(os.environ.get('OPENBROWSER_SOCKET', str(SOCKET_PATH)))


def _read_pid() -> int | None:
    """Read PID from file, return None if stale or missing."""
    pid_path = DAEMON_DIR / 'daemon.pid'
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
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    (DAEMON_DIR / 'daemon.pid').write_text(str(os.getpid()))


def _cleanup_pid() -> None:
    (DAEMON_DIR / 'daemon.pid').unlink(missing_ok=True)
    sock = _get_socket_path()
    sock.unlink(missing_ok=True)


class DaemonServer:
    """Persistent browser automation daemon."""

    def __init__(self, idle_timeout: int = DEFAULT_IDLE_TIMEOUT):
        self._idle_timeout = idle_timeout
        self._last_activity = time.time()
        self._executor = None  # lazy
        self._session = None
        self._running = False
        self._server = None

    async def _ensure_executor(self):
        """Lazy-initialize browser + namespace on first request."""
        if self._executor is not None:
            return

        # Suppress logging for clean daemon output
        os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'critical'
        os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'
        logging.disable(logging.CRITICAL)

        from openbrowser.browser import BrowserProfile, BrowserSession
        from openbrowser.code_use.executor import CodeExecutor
        from openbrowser.code_use.namespace import create_namespace
        from openbrowser.config import get_default_profile, load_openbrowser_config
        from openbrowser.tools.service import CodeAgentTools

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
        profile = BrowserProfile(**profile_data)
        session = BrowserSession(browser_profile=profile)
        await session.start()
        self._session = session

        tools = CodeAgentTools()
        namespace = create_namespace(browser_session=session, tools=tools)

        self._executor = CodeExecutor()
        self._executor.set_namespace(namespace)

    async def _handle_request(self, data: dict) -> dict:
        """Handle a single JSON request."""
        action = data.get('action', '')
        req_id = data.get('id', 0)

        if action == 'execute':
            code = data.get('code', '')
            if not code.strip():
                return {'id': req_id, 'success': False, 'output': '', 'error': 'No code provided'}
            await self._ensure_executor()
            result = await self._executor.execute(code)
            self._last_activity = time.time()
            return {
                'id': req_id,
                'success': result.success,
                'output': result.output,
                'error': None if result.success else result.output,
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

    async def _idle_check_loop(self):
        """Shut down daemon if idle beyond timeout."""
        while self._running:
            await asyncio.sleep(30)
            if time.time() - self._last_activity > self._idle_timeout:
                logger.info('Idle timeout reached, shutting down daemon')
                self._running = False
                if self._server:
                    self._server.close()
                break

    async def run(self):
        """Start the daemon and listen for connections."""
        self._running = True
        self._last_activity = time.time()

        sock_path = _get_socket_path()
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)

        # Clean up stale socket
        sock_path.unlink(missing_ok=True)

        _write_pid()

        try:
            if IS_WINDOWS:
                self._server = await asyncio.start_server(
                    self._handle_client, '127.0.0.1', WINDOWS_PORT
                )
            else:
                self._server = await asyncio.start_unix_server(
                    self._handle_client, path=str(sock_path)
                )

            # Start idle timeout checker
            idle_task = asyncio.create_task(self._idle_check_loop())

            async with self._server:
                while self._running:
                    await asyncio.sleep(0.1)

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

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: setattr(daemon, '_running', False))
        except NotImplementedError:
            pass  # Windows

    await daemon.run()


if __name__ == '__main__':
    asyncio.run(_main())
