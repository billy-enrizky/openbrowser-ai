# src/openbrowser/daemon/server.py
"""OpenBrowser daemon -- persistent browser session over Unix socket.

Holds a browser session and CodeAgent namespace in memory.
CLI clients connect via Unix socket to execute code.

Usage:
    python -m openbrowser.daemon.server
"""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import signal
import time
from pathlib import Path

from openbrowser.browser.lifecycle import LifecycleMachine, LifecyclePhase, LifecycleTransitionError
from openbrowser.daemon import IS_WINDOWS, WINDOWS_PORT, get_pid_path, get_socket_path

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT = 600  # 10 minutes
DEFAULT_EXEC_TIMEOUT = 110  # Leave room for the client to receive a timeout response
DAEMON_INIT_TIMEOUT = 30  # Bound browser startup so clients never wait indefinitely
DAEMON_CLEANUP_TIMEOUT = 5  # Bound cleanup after failed startup


def _init_timeout() -> float:
    """Return the browser startup bound, allowing an environment override."""
    raw = os.environ.get('OPENBROWSER_DAEMON_INIT_TIMEOUT', '')
    try:
        return max(1.0, float(raw)) if raw else float(DAEMON_INIT_TIMEOUT)
    except ValueError:
        return float(DAEMON_INIT_TIMEOUT)


def _read_pid() -> int | None:
    """Read PID from file, return None if stale or missing.

    A PID file with a live process is only considered valid if the daemon
    socket is also connectable. A process killed with SIGKILL skips
    finally/_cleanup_pid(), leaving a stale PID + dead socket.
    """
    pid_path = get_pid_path()
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)  # raises OSError if dead
    except (ValueError, OSError):
        pid_path.unlink(missing_ok=True)
        return None

    if not IS_WINDOWS:
        import socket as _socket
        sock_path = str(get_socket_path())
        try:
            s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(sock_path)
            s.close()
        except OSError:
            logger.warning('PID %d alive but socket unresponsive -- cleaning stale lock', pid)
            pid_path.unlink(missing_ok=True)
            get_socket_path().unlink(missing_ok=True)
            return None

    return pid


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

    def __init__(self, idle_timeout: int = DEFAULT_IDLE_TIMEOUT, exec_timeout: float = DEFAULT_EXEC_TIMEOUT):
        self._idle_timeout = idle_timeout
        self._exec_timeout = exec_timeout
        self._init_timeout = _init_timeout()
        self._last_activity = time.time()
        self._started_at = time.time()
        self._executor = None  # lazy
        self._session = None
        self._running = False
        self._server = None
        self._stop_event = asyncio.Event()
        self._exec_lock = asyncio.Lock()  # serialize code execution (stdout safety)
        self._init_lock = asyncio.Lock()  # guard lazy initialization
        self._lifecycle = LifecycleMachine('daemon')
        self._browser_error: str | None = None
        self._profile_summary: dict[str, object] | None = None
        self._browser_generation = 0

    def _default_profile_dir(self) -> Path:
        """Choose a stable profile directory for this daemon socket."""
        from openbrowser.config import CONFIG

        explicit = os.environ.get('OPENBROWSER_DAEMON_PROFILE_DIR')
        if explicit:
            return Path(explicit).expanduser()
        isolate = os.environ.get('OPENBROWSER_DAEMON_PROFILE_ISOLATION', '').lower() in ('true', '1', 'yes')
        if os.environ.get('OPENBROWSER_SOCKET') and isolate:
            digest = hashlib.sha256(str(get_socket_path()).encode('utf-8')).hexdigest()[:12]
            return CONFIG.OPENBROWSER_PROFILES_DIR / f'daemon-{digest}'
        return CONFIG.OPENBROWSER_PROFILES_DIR / 'daemon'

    def _build_browser_profile(self):
        """Build a BrowserProfile from config with daemon defaults."""
        from openbrowser.browser import BrowserProfile
        from openbrowser.config import CONFIG, apply_managed_browser_profile_defaults, get_default_profile, load_openbrowser_config

        config = load_openbrowser_config()
        profile_config = dict(get_default_profile(config))
        default_profile_dir = self._default_profile_dir()
        profile_data = {
            'downloads_path': str(Path.home() / 'Downloads' / 'openbrowser-daemon'),
            'wait_between_actions': 0.5,
            'keep_alive': True,
            'device_scale_factor': 1.0,
            'disable_security': False,
            'headless': True,
            **profile_config,
        }
        isolate = os.environ.get('OPENBROWSER_DAEMON_PROFILE_ISOLATION', '').lower() in ('true', '1', 'yes')
        daemon_profile_override = bool(os.environ.get('OPENBROWSER_DAEMON_PROFILE_DIR')) or (
            bool(os.environ.get('OPENBROWSER_SOCKET')) and isolate
        )
        if daemon_profile_override and not os.environ.get('OPENBROWSER_USER_DATA_DIR'):
            profile_data['user_data_dir'] = str(default_profile_dir)
        elif not profile_data.get('user_data_dir'):
            profile_data['user_data_dir'] = str(default_profile_dir)
        profile_data = apply_managed_browser_profile_defaults(profile_data, default_profile_dir)
        # The daemon already persists cookies and storage in its Chrome
        # user-data directory. Replaying a second snapshot on every startup
        # can be enormous and blocks the first request while CDP applies it.
        # Keep storage-state loading only when the user explicitly configured
        # a snapshot path.
        if not profile_config.get('storage_state'):
            profile_data.pop('storage_state', None)

        # Stored config (profile_config) may have headless=False from a previous
        # non-headless session, overriding the daemon default above. Apply the env
        # var override last so OPENBROWSER_HEADLESS=false is the only way to opt
        # into a visible window; omitting the var keeps daemon headless by default.
        env_headless = os.environ.get('OPENBROWSER_HEADLESS', '').lower()
        if env_headless in ('true', '1', 'yes'):
            profile_data['headless'] = True
        elif env_headless in ('false', '0', 'no'):
            profile_data['headless'] = False
        else:
            profile_data['headless'] = True  # daemon default: always headless
        profile = BrowserProfile(**profile_data)
        storage_state = profile.storage_state
        self._profile_summary = {
            'user_data_dir': str(profile.user_data_dir) if profile.user_data_dir else None,
            'headless': profile.headless,
            'storage_state': str(storage_state) if isinstance(storage_state, (str, Path)) else None,
        }
        return profile

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
            try:
                init_timeout = getattr(self, '_init_timeout', _init_timeout())
                try:
                    await asyncio.wait_for(session.start(), timeout=init_timeout)
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(f'Browser initialization timed out after {init_timeout:g}s') from exc
                seeded = session.seed_generation(getattr(self, '_browser_generation', 0))
                if inspect.isawaitable(seeded):
                    await seeded

                tools = CodeAgentTools()
                namespace = create_namespace(browser_session=session, tools=tools)

                try:
                    max_output = int(os.environ.get('OPENBROWSER_MAX_OUTPUT', '0'))
                except (ValueError, TypeError):
                    max_output = 0
                self._executor = CodeExecutor(max_output_chars=max_output if max_output > 0 else DEFAULT_MAX_OUTPUT_CHARS)
                self._executor.set_namespace(namespace)
                self._session = session
                self._browser_error = None
            except BaseException as error:
                # Kill the browser if startup or namespace setup fails. Bound
                # cleanup too, so a broken browser cannot block the client.
                self._browser_error = f'{type(error).__name__}: {error}'
                await self._cleanup_failed_session(session)
                raise

    async def _cleanup_failed_session(self, session) -> None:
        """Bound failed-start cleanup without cancelling BrowserSession.kill()."""
        cleanup_task = asyncio.create_task(session.kill())
        try:
            done, pending = await asyncio.wait({cleanup_task}, timeout=DAEMON_CLEANUP_TIMEOUT)
        except BaseException:
            cleanup_task.add_done_callback(self._log_cleanup_task_failure)
            cleanup_task.cancel()
            self._request_force_stop(session)
            raise

        if pending:
            cleanup_task.add_done_callback(self._log_cleanup_task_failure)
            cleanup_task.cancel()
            self._request_force_stop(session)
            logger.warning(
                'Browser cleanup exceeded %ss after daemon initialization failure; force-stop requested',
                DAEMON_CLEANUP_TIMEOUT,
            )
            return

        # Retrieve any cleanup error so the detached-task callback does not
        # produce an unhandled-task warning or replace the startup exception.
        done_task = done.pop()
        try:
            done_task.result()
        except BaseException:
            logger.warning('Failed to clean up browser after daemon initialization failure', exc_info=True)

    @staticmethod
    def _request_force_stop(session) -> None:
        """Queue the normal forced browser-stop event without awaiting it."""
        from openbrowser.browser.events import BrowserStopEvent

        session.event_bus.dispatch(BrowserStopEvent(force=True))

    @staticmethod
    def _log_cleanup_task_failure(task: asyncio.Task) -> None:
        """Consume a detached cleanup task's eventual exception."""
        if task.cancelled():
            return
        try:
            error = task.exception()
        except BaseException:
            logger.warning('Detached daemon browser cleanup task failed', exc_info=True)
        else:
            if error is not None:
                logger.warning('Detached daemon browser cleanup task failed', exc_info=(type(error), error, error.__traceback__))

    _CDP_ERROR_KEYWORDS = ('connectionclosederror', 'no close frame', 'websocket', 'connection closed')

    def _is_connection_error(self, text: str) -> bool:
        """Return True if the error indicates a dead CDP/browser connection."""
        lower = text.lower()
        return any(kw in lower for kw in self._CDP_ERROR_KEYWORDS)

    async def _recover_browser_session(self, *, expected_session: object | None = None):
        """Tear down a dead browser and rebuild the session and namespace once."""
        async with self._init_lock:
            if expected_session is not None and self._session is not expected_session:
                logger.info('Skipping browser recovery because another request already replaced the session')
                return

            logger.info('CDP connection lost -- recovering browser session')
            old_namespace = self._executor.namespace if self._executor else {}
            if self._session:
                closed = await self._close_session_bounded(self._session)
                if not closed:
                    raise RuntimeError('Unable to close the old browser session before recovery')
                self._session = None

            from openbrowser.browser import BrowserSession
            from openbrowser.code_use.namespace import create_namespace
            from openbrowser.tools.service import CodeAgentTools

            profile = self._build_browser_profile()
            session = BrowserSession(browser_profile=profile)
            try:
                await asyncio.wait_for(session.start(), timeout=getattr(self, '_init_timeout', _init_timeout()))
                self._browser_generation = getattr(self, '_browser_generation', 0) + 1
                seeded = session.seed_generation(self._browser_generation)
                if inspect.isawaitable(seeded):
                    await seeded
                tools = CodeAgentTools()
                namespace = create_namespace(browser_session=session, tools=tools)
                for key, value in (old_namespace or {}).items():
                    if not key.startswith('__') and key not in namespace:
                        namespace[key] = value
                self._executor.set_namespace(namespace)
                self._session = session
                self._browser_error = None
            except Exception as error:
                self._browser_error = f'{type(error).__name__}: {error}'
                try:
                    await self._cleanup_failed_session(session)
                except Exception:
                    pass
                raise
            logger.info('Browser session recovered successfully')

    def _status_payload(self) -> dict:
        """Build the daemon status shape shared with MCP diagnostics."""
        if self._profile_summary is None:
            try:
                # Resolve the effective profile without starting a browser so
                # `daemon ensure` can compare configuration before first use.
                self._build_browser_profile()
            except Exception as error:
                self._browser_error = f'Profile configuration failed: {type(error).__name__}: {error}'

        browser: dict
        if self._session is not None and hasattr(self._session, 'lifecycle_snapshot'):
            snapshot_value = self._session.lifecycle_snapshot()
            browser = snapshot_value if isinstance(snapshot_value, dict) else {'phase': 'unknown'}
        elif self._session is not None:
            browser = {'phase': getattr(self._session, 'state', 'unknown')}
        else:
            browser = {'phase': 'stopped', 'last_error': self._browser_error}

        try:
            from openbrowser.utils import get_openbrowser_version

            version = get_openbrowser_version()
        except Exception:
            version = None

        snapshot = self._lifecycle.snapshot()
        return {
            'pid': os.getpid(),
            'state': snapshot.phase.value,
            'initialized': self._executor is not None,
            'socket': str(get_socket_path()),
            'profile': self._profile_summary,
            'idle_timeout': self._idle_timeout,
            'init_timeout': self._init_timeout,
            'exec_timeout': self._exec_timeout,
            'last_error': snapshot.last_error or self._browser_error,
            'generation': snapshot.generation,
            'uptime_seconds': max(0.0, time.time() - self._started_at),
            'version': version,
            'browser': browser,
        }

    def _error_response(
        self,
        req_id: object,
        error: str,
        *,
        code: str | None = None,
        hint: str | None = None,
    ) -> dict:
        return {
            'id': req_id,
            'success': False,
            'output': '',
            'error': error,
            'error_code': code,
            'hint': hint,
            'state': self._lifecycle.phase.value,
        }

    async def _close_session_bounded(self, session) -> bool:
        """Close a session without letting a broken browser block the daemon."""
        task = asyncio.create_task(session.kill())
        done, pending = await asyncio.wait({task}, timeout=DAEMON_CLEANUP_TIMEOUT)
        if pending:
            task.add_done_callback(self._log_cleanup_task_failure)
            task.cancel()
            self._request_force_stop(session)
            logger.warning('Browser cleanup exceeded %ss', DAEMON_CLEANUP_TIMEOUT)
            return False
        done_task = done.pop()
        try:
            done_task.result()
        except BaseException:
            logger.warning('Failed to close browser session', exc_info=True)
            return False
        return True

    async def _handle_request(self, data: dict) -> dict:
        """Handle a single JSON request."""
        self._last_activity = time.time()
        action = data.get('action', '')
        req_id = data.get('id', 0)

        if action == 'execute':
            code = data.get('code', '')
            if not code.strip():
                return self._error_response(req_id, 'No code provided', code='empty_code')
            try:
                async with self._exec_lock:
                    try:
                        await self._ensure_executor()
                    except Exception as error:
                        return self._error_response(
                            req_id,
                            f'Browser initialization failed: {type(error).__name__}: {error}',
                            code='daemon_start_failed',
                            hint='Run `openbrowser-ai daemon doctor` and check the profile path and daemon log.',
                        )
                    failed_session = self._session
                    result = await asyncio.wait_for(
                        self._executor.execute(code), timeout=self._exec_timeout
                    )

                    # Recover while still holding the execution lock so the next
                    # request cannot run user code against the dead browser.
                    error_text = result.error or result.output
                    if not result.success and self._is_connection_error(error_text):
                        try:
                            await self._recover_browser_session(expected_session=failed_session)
                        except Exception as recovery_err:
                            logger.error('CDP recovery failed: %s', recovery_err)
                            return self._error_response(
                                req_id,
                                result.error or result.output,
                                code='cdp_recovery_failed',
                                hint=f'Browser recovery failed: {type(recovery_err).__name__}: {recovery_err}',
                            )
                        return self._error_response(
                            req_id,
                            result.error or result.output,
                            code='cdp_connection_lost',
                            hint='The browser was recovered. The code may have partially run; verify the page before retrying.',
                        )
            except asyncio.TimeoutError:
                return self._error_response(
                    req_id,
                    f'Execution timed out after {self._exec_timeout}s',
                    code='execution_timeout',
                    hint='Run `openbrowser-ai daemon status` before retrying. The code may still be running.',
                )

            return {
                'id': req_id,
                'success': result.success,
                'output': result.output,
                'error': result.error,
                'error_code': None,
                'state': self._lifecycle.phase.value,
            }

        elif action == 'status':
            return {
                'id': req_id,
                'success': True,
                'output': json.dumps(self._status_payload()),
                'error': None,
                'state': self._lifecycle.phase.value,
            }

        elif action == 'stop':
            if self._lifecycle.phase is not LifecyclePhase.STOPPING:
                try:
                    self._lifecycle.transition(LifecyclePhase.STOPPING, reason='stop requested')
                except LifecycleTransitionError:
                    pass
            self._running = False
            self._stop_event.set()
            return {
                'id': req_id,
                'success': True,
                'output': 'Daemon stopping',
                'error': None,
                'state': self._lifecycle.phase.value,
            }

        elif action == 'reset':
            async with self._exec_lock:
                async with self._init_lock:
                    if self._session:
                        closed = await self._close_session_bounded(self._session)
                        if not closed:
                            return self._error_response(
                                req_id,
                                'Unable to close the current browser session',
                                code='reset_failed',
                                hint='Run `openbrowser-ai daemon doctor` and restart the daemon if the browser process is stuck.',
                            )
                    self._executor = None
                    self._session = None
                    self._browser_error = None
                    self._browser_generation += 1
            return {
                'id': req_id,
                'success': True,
                'output': json.dumps(
                    {
                        'message': 'Session reset',
                        'state': self._lifecycle.phase.value,
                        'browser': {'phase': 'stopped'},
                    }
                ),
                'error': None,
                'state': self._lifecycle.phase.value,
            }

        return self._error_response(req_id, f'Unknown action: {action}', code='unknown_action')

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
                err = {
                    'id': 0,
                    'success': False,
                    'output': '',
                    'error': str(e),
                    'error_code': 'daemon_request_failed',
                    'state': self._lifecycle.phase.value,
                }
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
        if self._lifecycle.phase is not LifecyclePhase.STOPPING:
            try:
                self._lifecycle.transition(LifecyclePhase.STOPPING, reason='signal received')
            except LifecycleTransitionError:
                pass
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
        self._lifecycle.transition(LifecyclePhase.STARTING, reason='daemon starting')
        self._running = True
        self._last_activity = time.time()

        sock_path = get_socket_path()
        sock_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if another daemon is already running
        existing_pid = _read_pid()
        if existing_pid and existing_pid != os.getpid():
            logger.error('Another daemon is already running (PID %d)', existing_pid)
            self._lifecycle.fail(f'Another daemon is already running (PID {existing_pid})', reason='daemon start failed')
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
            self._lifecycle.transition(LifecyclePhase.READY, reason='daemon listening')

            async with self._server:
                await self._stop_event.wait()

            idle_task.cancel()
            await asyncio.gather(idle_task, return_exceptions=True)

        finally:
            if self._lifecycle.phase is not LifecyclePhase.STOPPING:
                try:
                    self._lifecycle.transition(LifecyclePhase.STOPPING, reason='daemon stopping')
                except LifecycleTransitionError:
                    pass
            if self._session:
                await self._close_session_bounded(self._session)
            _cleanup_pid()
            if self._lifecycle.phase is not LifecyclePhase.STOPPED:
                try:
                    self._lifecycle.transition(
                        LifecyclePhase.STOPPED,
                        reason='daemon stopped',
                        increment_generation=True,
                    )
                except LifecycleTransitionError:
                    pass


async def _main():
    def _env_number(name: str, default: float) -> float:
        raw = os.environ.get(name, '')
        try:
            return max(1.0, float(raw)) if raw else default
        except ValueError:
            return default

    daemon = DaemonServer(
        idle_timeout=int(_env_number('OPENBROWSER_IDLE_TIMEOUT', DEFAULT_IDLE_TIMEOUT)),
        exec_timeout=_env_number('OPENBROWSER_EXEC_TIMEOUT', DEFAULT_EXEC_TIMEOUT),
    )
    await daemon.run()


if __name__ == '__main__':
    asyncio.run(_main())
