# src/openbrowser/daemon/client.py
"""Thin client for the OpenBrowser daemon.

Connects to the daemon Unix socket, sends code, returns output.
Auto-starts the daemon if it is not running.

This module intentionally avoids importing any heavy openbrowser
modules so the -c CLI path stays fast (<50ms import time).
"""

import asyncio
import errno
import json
import logging
import os
import stat
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from openbrowser.daemon import (
    DAEMON_DIR,
    EXIT_OWNER_BUSY,
    IS_WINDOWS,
    WINDOWS_PORT,
    get_log_path,
    get_pid_path,
    get_socket_path,
)
from openbrowser.daemon.ownership import (
    DaemonOwnerLock,
    SocketEntryIdentity,
    owner_lock_path_for_pid_file,
    socket_entry_identity,
)

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 140.0  # Slightly above the server's default execution timeout
STATUS_TIMEOUT = 5.0
DAEMON_START_TIMEOUT = 20.0
# How long to wait for a competing starter before retrying our own startup.
OWNER_ADOPT_GRACE = 5.0


class _StartRetry(Exception):
    """A transient owner held the lock; retry the whole startup once."""


def _start_timeout() -> float:
    raw = os.environ.get('OPENBROWSER_DAEMON_START_TIMEOUT', '')
    try:
        return max(1.0, float(raw)) if raw else DAEMON_START_TIMEOUT
    except ValueError:
        return DAEMON_START_TIMEOUT


def _read_timeout() -> float:
    raw = os.environ.get('OPENBROWSER_READ_TIMEOUT', '')
    try:
        return max(1.0, float(raw)) if raw else READ_TIMEOUT
    except ValueError:
        return READ_TIMEOUT


@dataclass
class DaemonResponse:
    success: bool
    output: str
    error: str | None
    error_code: str | None = None
    hint: str | None = None
    state: str | None = None


class DaemonClient:
    """Client that communicates with the OpenBrowser daemon."""

    def __init__(self) -> None:
        self._last_process: subprocess.Popen | None = None

    async def _connect(self):
        sock = get_socket_path()
        if IS_WINDOWS:
            return await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', WINDOWS_PORT),
                timeout=CONNECT_TIMEOUT,
            )
        return await asyncio.wait_for(
            asyncio.open_unix_connection(str(sock)),
            timeout=CONNECT_TIMEOUT,
        )

    @staticmethod
    def _response(payload: dict) -> DaemonResponse:
        return DaemonResponse(
            success=bool(payload.get('success', False)),
            output=str(payload.get('output') or ''),
            error=payload.get('error'),
            error_code=payload.get('error_code'),
            hint=payload.get('hint'),
            state=payload.get('state'),
        )

    async def _exchange(self, reader, writer, request: dict, timeout: float) -> dict:
        try:
            writer.write(json.dumps(request).encode() + b'\n')
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not raw:
                raise ConnectionResetError('Daemon closed the connection without responding')
            value = json.loads(raw.decode())
            if not isinstance(value, dict):
                raise ValueError('Daemon returned a non-object response')
            return value
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _send(self, request: dict, *, timeout: float = READ_TIMEOUT) -> DaemonResponse:
        """Send a request and return a structured response."""
        reader, writer = await self._connect()
        try:
            payload = await self._exchange(reader, writer, request, timeout)
        except asyncio.TimeoutError:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Daemon did not respond within {timeout:g}s',
                error_code='daemon_timeout',
                hint='Run `openbrowser-ai daemon status` or `openbrowser-ai daemon doctor`.',
            )
        except (ConnectionResetError, OSError) as error:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Daemon connection lost: {error}',
                error_code='daemon_connection_lost',
                hint='The request may have started. Check `openbrowser-ai daemon status` before retrying.',
            )
        except (ValueError, json.JSONDecodeError) as error:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Invalid daemon response: {error}',
                error_code='daemon_invalid_response',
            )
        return self._response(payload)

    async def _execute_once(self, request: dict) -> DaemonResponse | None:
        """Run one execute request. Return None only when no request was sent."""
        try:
            reader, writer = await self._connect()
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return None

        try:
            read_timeout = _read_timeout()
            payload = await self._exchange(reader, writer, request, read_timeout)
        except asyncio.TimeoutError:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Daemon did not respond within {_read_timeout():g}s',
                error_code='daemon_timeout',
                hint='The code may still be running. Run `openbrowser-ai daemon status` before retrying.',
            )
        except (ConnectionResetError, OSError) as error:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Daemon connection lost: {error}',
                error_code='daemon_connection_lost',
                hint='The code may have partially run. Verify the page before retrying.',
            )
        except (ValueError, json.JSONDecodeError) as error:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Invalid daemon response: {error}',
                error_code='daemon_invalid_response',
            )
        return self._response(payload)

    def _log_path(self) -> Path:
        return get_log_path()

    @staticmethod
    def _tail_log(path: Path, lines: int = 20) -> str:
        try:
            content = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            return ''
        return '\n'.join(content[-lines:])

    async def _probe(self, timeout: float = 1.0) -> bool:
        try:
            reader, writer = await self._connect()
        except (ConnectionRefusedError, FileNotFoundError, OSError, asyncio.TimeoutError):
            return False
        try:
            payload = await self._exchange(
                reader,
                writer,
                {'id': 1, 'action': 'status'},
                timeout,
            )
        except Exception:
            return False
        return bool(payload.get('success'))

    async def _probe_within(self, deadline: float) -> bool:
        """Probe status without exceeding the remaining startup budget."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            return await asyncio.wait_for(self._probe(), timeout=remaining)
        except asyncio.TimeoutError:
            return False

    async def _socket_state(self) -> str:
        """Classify the daemon socket as 'live', 'stale', or 'unknown'.

        Only a refused or missing socket proves that a socket file is stale.
        Permission errors and timeouts leave its state unknown, so callers
        must not delete the path.
        """
        try:
            reader, writer = await self._connect()
        except (FileNotFoundError, ConnectionRefusedError):
            return 'stale'
        except (OSError, asyncio.TimeoutError):
            return 'unknown'
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return 'live'

    async def _socket_accepts_connection(self) -> bool:
        """Return True when the daemon socket accepts a connection."""
        return await self._socket_state() == 'live'

    def _daemon_owner_for_socket(self) -> dict | None:
        """Return the current user's daemon process that owns this socket."""
        try:
            from openbrowser.daemon.diagnostics import daemon_processes

            rows = daemon_processes()
            if IS_WINDOWS:
                for row in rows:
                    if row.get('socket_live'):
                        return row
                return None
            socket_path = str(get_socket_path())
            for row in rows:
                if row.get('socket') == socket_path:
                    return row
        except Exception:
            pass
        return None

    def _verified_daemon_pid(self) -> int | None:
        """Return the PID file value only when it names an OpenBrowser daemon."""
        pid = self._read_pid()
        if pid is None:
            return None
        try:
            import psutil

            from openbrowser.daemon.diagnostics import is_daemon_cmdline

            process = psutil.Process(pid)
            if is_daemon_cmdline(process.cmdline()):
                return pid
        except Exception:
            return None
        return None

    async def _stop_verification_error(self) -> DaemonResponse | None:
        """Return an error when a listener or verified daemon survives cleanup."""
        if await self._socket_accepts_connection():
            return DaemonResponse(
                success=False,
                output='',
                error='Daemon socket still accepts connections',
                error_code='daemon_stop_timeout',
                hint='The daemon is still alive. Inspect it or run `openbrowser-ai daemon reap`.',
            )
        owner = self._daemon_owner_for_socket()
        if owner is not None:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Daemon PID {owner["pid"]} is still running',
                error_code='daemon_stop_timeout',
                hint='Run `openbrowser-ai daemon reap` to verify and stop the owning daemon process.',
            )
        return None

    async def _start_daemon(self):
        """Serialize daemon startup across processes before spawning."""
        async with self._start_lock():
            try:
                await self._start_daemon_unlocked()
            except _StartRetry:
                # The lock holder released without leaving a daemon behind
                # (for example a concurrent stale cleanup). Start once more.
                try:
                    await self._start_daemon_unlocked()
                except _StartRetry:
                    raise OSError(
                        'Daemon startup is already in progress for this socket. '
                        'Wait for it to finish or run `openbrowser-ai daemon doctor`.'
                    ) from None

    @asynccontextmanager
    async def _start_lock(self):
        """Hold an OS start lock for this daemon socket across client processes."""
        socket_path = get_socket_path()
        lock_path = socket_path.parent / f'{socket_path.stem}.start.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, 'a+b')
        if IS_WINDOWS and handle.tell() == 0:
            handle.write(b'\0')
            handle.flush()
        locked = False
        try:
            locked = await asyncio.to_thread(
                self._lock_start_file, handle, time.monotonic() + _start_timeout()
            )
            if not locked:
                raise OSError(
                    'Another process is starting or stopping the daemon for this socket. '
                    'Wait for it to finish or run `openbrowser-ai daemon doctor`.'
                )
            yield
        finally:
            try:
                if locked:
                    await asyncio.to_thread(self._unlock_start_file, handle)
            finally:
                handle.close()

    @staticmethod
    def _lock_start_file(handle, deadline: float) -> bool:
        """Try to take the cross-process start lock until the deadline."""
        if IS_WINDOWS:
            import msvcrt

            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    return True
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.1)
        import fcntl

        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    @staticmethod
    def _unlock_start_file(handle) -> None:
        if IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    async def _start_daemon_unlocked(self):
        """Spawn the daemon process and wait until it answers a status probe."""
        get_socket_path().parent.mkdir(parents=True, exist_ok=True)
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)
        log_file = self._log_path()

        if IS_WINDOWS and await self._probe(timeout=STATUS_TIMEOUT):
            return

        socket_path = get_socket_path()
        socket_observed: SocketEntryIdentity | None = None
        if not IS_WINDOWS:
            socket_observed = socket_entry_identity(socket_path)
            if socket_observed is not None and not socket_observed.is_socket:
                raise OSError(
                    'Daemon socket path exists but is not a Unix socket file. '
                    'Inspect the path before retrying.'
                )
        if socket_observed is not None:
            if await self._probe(timeout=STATUS_TIMEOUT):
                return
            socket_state = await self._socket_state()
            if socket_state == 'live':
                raise OSError(
                    'Daemon is running but did not answer status in time. '
                    'Wait for the current execution to finish or run `openbrowser-ai daemon restart`.'
                )
            if socket_state == 'unknown':
                raise OSError(
                    'Daemon socket state cannot be verified safely. '
                    'Inspect the path, then run `openbrowser-ai daemon doctor`.'
                )

        owner_lock = DaemonOwnerLock(owner_lock_path_for_pid_file(get_pid_path()))
        lock_deadline = time.monotonic() + _start_timeout()
        while not owner_lock.acquire():
            # Another process (for example a direct `python -m
            # openbrowser.daemon.server` launch) owns startup. Adopt its
            # daemon once it answers instead of reporting a false failure.
            if await self._probe_within(lock_deadline):
                return
            if time.monotonic() >= lock_deadline:
                raise OSError(
                    'Daemon startup is already in progress for this socket. '
                    'Wait for it to finish or run `openbrowser-ai daemon doctor`.'
                )
            await asyncio.sleep(0.2)
        try:
            owner = self._daemon_owner_for_socket()
            if owner is not None:
                raise OSError(
                    f'Daemon PID {owner["pid"]} owns this socket but is not accepting connections. '
                    'Run `openbrowser-ai daemon reap` or `openbrowser-ai daemon restart`.'
                )
            verified_pid = self._verified_daemon_pid()
            if verified_pid is not None:
                raise OSError(
                    f'Daemon PID {verified_pid} owns a PID file for this socket but is not accepting connections. '
                    'Run `openbrowser-ai daemon reap` or `openbrowser-ai daemon restart`.'
                )
            if not IS_WINDOWS:
                current_identity = socket_entry_identity(socket_path)
                if current_identity is not None:
                    if current_identity != socket_observed:
                        raise OSError(
                            'Daemon socket path changed during startup. '
                            'Inspect the path before retrying.'
                        )
                    self._remove_stale_socket(socket_path)
            get_pid_path().unlink(missing_ok=True)
        finally:
            owner_lock.release()

        deadline = time.monotonic() + _start_timeout()
        with open(os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as log_handle:
            self._last_process = subprocess.Popen(
                [sys.executable, '-m', 'openbrowser.daemon.server'],
                stdout=subprocess.DEVNULL,
                stderr=log_handle,
                start_new_session=True,
            )

        while time.monotonic() < deadline:
            if self._last_process.poll() is not None:
                if await self._probe_within(deadline):
                    return
                if self._last_process.returncode == EXIT_OWNER_BUSY:
                    # Another starter won the ownership race. Wait briefly for
                    # its daemon; if the lock holder was a cleanup rather than
                    # a starter, retry startup from the top once.
                    grace_deadline = min(deadline, time.monotonic() + OWNER_ADOPT_GRACE)
                    while time.monotonic() < grace_deadline:
                        if await self._probe_within(grace_deadline):
                            return
                        await asyncio.sleep(0.2)
                    raise _StartRetry()
                tail = self._tail_log(log_file)
                raise TimeoutError(
                    f'Daemon exited with code {self._last_process.returncode} during startup. '
                    f'Log: {log_file}\n{tail}'
                )
            if IS_WINDOWS or get_socket_path().exists():
                if await self._probe_within(deadline):
                    return
            await asyncio.sleep(0.02)
        if await self._probe_within(time.monotonic() + 1.0):
            return
        await self._terminate_started_child()
        tail = self._tail_log(log_file)
        raise TimeoutError(
            f'Daemon did not become ready within {_start_timeout():g}s. Log: {log_file}\n{tail}'
        )

    async def _terminate_started_child(self) -> None:
        """Stop and reap the child this client launched, with bounded escalation."""
        process = self._last_process
        if process is None or process.poll() is not None:
            return
        await asyncio.to_thread(self._terminate_process, process)

    @staticmethod
    def _terminate_process(process) -> None:
        """Terminate then kill a child without blocking the event loop."""
        try:
            process.terminate()
        except OSError:
            logger.warning('Could not terminate the daemon child', exc_info=True)
            return
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning('Daemon child did not exit after kill; it may still hold the owner lock')

    async def execute(self, code: str) -> DaemonResponse:
        """Execute code via the daemon. Auto-starts only before the request is sent."""
        request = {'id': 1, 'action': 'execute', 'code': code}
        response = await self._execute_once(request)
        if response is not None:
            return response

        try:
            await self._start_daemon()
        except (TimeoutError, OSError) as error:
            return DaemonResponse(
                success=False,
                output='',
                error=str(error),
                error_code='daemon_start_failed',
                hint='Run `openbrowser-ai daemon doctor` and check the daemon log and profile path.',
            )

        response = await self._execute_once(request)
        if response is not None:
            return response
        return DaemonResponse(
            success=False,
            output='',
            error='Daemon is not reachable after startup',
            error_code='daemon_unavailable',
            hint='Run `openbrowser-ai daemon doctor`.',
        )

    async def status(self) -> DaemonResponse:
        try:
            return await self._send({'id': 1, 'action': 'status'}, timeout=STATUS_TIMEOUT)
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(
                success=False,
                output='',
                error='Daemon not running',
                error_code='daemon_not_running',
            )

    async def stop(self) -> DaemonResponse:
        try:
            return await self._send({'id': 1, 'action': 'stop'}, timeout=STATUS_TIMEOUT)
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(
                success=False,
                output='',
                error='Daemon not running',
                error_code='daemon_not_running',
            )

    async def reset(self) -> DaemonResponse:
        try:
            return await self._send({'id': 1, 'action': 'reset'}, timeout=_read_timeout())
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(
                success=False,
                output='',
                error='Daemon not running',
                error_code='daemon_not_running',
            )

    def _read_pid(self) -> int | None:
        pid_path = get_pid_path()
        try:
            return int(pid_path.read_text().strip())
        except (OSError, ValueError):
            return None

    async def force_stop(self, *, timeout: float = 5.0) -> DaemonResponse:
        """Stop the daemon, terminating only the verified daemon process if needed."""
        try:
            async with self._start_lock():
                return await self._force_stop_unlocked(timeout=timeout)
        except OSError as error:
            return DaemonResponse(
                success=False,
                output='',
                error=str(error),
                error_code='daemon_busy',
                hint='Wait for the current daemon operation to finish, then retry.',
            )

    async def _force_stop_unlocked(self, *, timeout: float) -> DaemonResponse:
        """Force-stop the daemon while the cross-process start lock is held."""
        status = await self.status()
        status_payload: dict = {}
        if status.success:
            try:
                value = json.loads(status.output)
                if isinstance(value, dict):
                    status_payload = value
            except (TypeError, json.JSONDecodeError):
                pass
        profile = status_payload.get('profile') or {}
        profile_dir = profile.get('user_data_dir') if isinstance(profile, dict) else None
        profile_dirs = [profile_dir] if profile_dir else []
        status_pid = status_payload.get('pid')

        response = await self.stop()
        if response.success:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not get_socket_path().exists():
                    still_alive = False
                    if isinstance(status_pid, int):
                        import psutil

                        still_alive = psutil.pid_exists(status_pid)
                    else:
                        still_alive = self._daemon_owner_for_socket() is not None
                    if still_alive:
                        await asyncio.sleep(0.05)
                        continue
                    verification = await self._stop_verification_error()
                    if verification is not None:
                        return verification
                    browser_ok, browser_error = await self._kill_recorded_browsers(
                        profile_dirs,
                        timeout,
                        expected_owner_pid=status_pid if isinstance(status_pid, int) else None,
                    )
                    if not browser_ok:
                        return DaemonResponse(
                            success=False,
                            output='',
                            error=f'Daemon stopped but browser cleanup failed: {browser_error}',
                            error_code='browser_cleanup_failed',
                        )
                    return DaemonResponse(success=True, output='Daemon stopped', error=None)
                await asyncio.sleep(0.2)
            response = DaemonResponse(
                success=False,
                output='',
                error='Daemon did not exit after a graceful stop',
                error_code='daemon_stop_timeout',
            )

        pid = self._read_pid()
        if pid is None:
            if await self._socket_accepts_connection():
                return DaemonResponse(
                    success=False,
                    output='',
                    error='Daemon socket is still accepting connections without a PID file',
                    error_code='daemon_stop_timeout',
                    hint='The daemon is still alive. Run `openbrowser-ai daemon reap` or inspect the listening process.',
                )
            cleanup_error = await self._remove_stale_daemon_files()
            if cleanup_error is not None:
                return cleanup_error
            browser_ok, browser_error = await self._kill_recorded_browsers(profile_dirs, timeout)
            if not browser_ok:
                return DaemonResponse(
                    success=False,
                    output='',
                    error=f'Stale daemon files removed but browser cleanup failed: {browser_error}',
                    error_code='browser_cleanup_failed',
                )
            return DaemonResponse(
                success=True,
                output='Removed stale daemon files',
                error=None,
            )

        try:
            import psutil

            from openbrowser.daemon.diagnostics import is_daemon_cmdline

            process = psutil.Process(pid)
            if not is_daemon_cmdline(process.cmdline()):
                return DaemonResponse(
                    success=False,
                    output='',
                    error=f'PID {pid} is not the OpenBrowser daemon',
                    error_code='daemon_pid_mismatch',
                    hint='Remove the stale PID and socket files manually after checking the process.',
                )
            if profile_dir:
                try:
                    from openbrowser.browser.watchdogs.profile_lease import ProfileLease

                    metadata = ProfileLease.read_metadata(profile_dir)
                    if metadata:
                        if metadata.get('owner_pid') != pid or not ProfileLease.process_matches_identity(
                            process,
                            pid,
                            metadata.get('owner_start_time'),
                        ):
                            return DaemonResponse(
                                success=False,
                                output='',
                                error=f'PID {pid} does not match the recorded daemon owner',
                                error_code='daemon_pid_mismatch',
                                hint='Do not kill the process. Inspect the PID, socket, and profile metadata.',
                            )
                except Exception:
                    pass
            if not profile_dirs:
                profile_dirs = self._profile_dirs_for_daemon(process, pid)
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
        except psutil.NoSuchProcess:
            pass
        except Exception as error:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Could not stop daemon PID {pid}: {type(error).__name__}: {error}',
                error_code='daemon_force_stop_failed',
            )

        cleanup_error = await self._remove_stale_daemon_files()
        if cleanup_error is not None and cleanup_error.error_code not in (
            'daemon_socket_foreign',
            'daemon_cleanup_failed',
        ):
            # Another owner or listener may be live. Do not signal any browser.
            return cleanup_error
        # The daemon is gone and no other owner is proven live, so a refused
        # file cleanup must not hide a browser that still needs killing. The
        # lease must still name the daemon PID that was stopped.
        browser_ok, browser_error = await self._kill_recorded_browsers(
            profile_dirs,
            timeout,
            expected_owner_pid=pid,
        )
        if cleanup_error is not None:
            if not browser_ok:
                cleanup_error.error = f'{cleanup_error.error}. Browser cleanup also failed: {browser_error}'
            return cleanup_error
        if not browser_ok:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Daemon stopped but browser cleanup failed: {browser_error}',
                error_code='browser_cleanup_failed',
            )
        return DaemonResponse(success=True, output='Daemon force-stopped', error=None)

    def _profile_dirs_for_daemon(self, process, pid: int) -> list[str]:
        """Find profile leases owned by the verified daemon process."""
        try:
            from openbrowser.browser.watchdogs.profile_lease import ProfileLease
            from openbrowser.config import CONFIG
        except Exception:
            return []

        profiles_root = Path(CONFIG.OPENBROWSER_PROFILES_DIR)
        expanded: set[Path] = set()
        try:
            if profiles_root.is_dir():
                expanded.update(path for path in profiles_root.iterdir() if path.is_dir())
        except OSError:
            pass

        try:
            daemon_environment = process.environ() or {}
        except Exception:
            daemon_environment = {}

        explicit: set[Path] = set()
        for env_name in ('OPENBROWSER_DAEMON_PROFILE_DIR', 'OPENBROWSER_USER_DATA_DIR'):
            value = daemon_environment.get(env_name) or os.environ.get(env_name)
            if value:
                explicit.add(Path(value).expanduser())
        expanded.update(explicit)

        matched: list[str] = []
        for profile_dir in sorted(expanded):
            metadata = ProfileLease.read_metadata(profile_dir)
            if not metadata or metadata.get('owner_pid') != pid:
                continue
            if not ProfileLease.process_matches_identity(process, pid, metadata.get('owner_start_time')):
                continue
            matched.append(str(profile_dir.expanduser().resolve()))
        return matched

    async def _remove_stale_daemon_files(self) -> DaemonResponse | None:
        """Remove PID and socket files only while no other owner can claim them.

        Returns an error response instead of deleting a path that this client
        cannot prove is the stale daemon socket.
        """
        try:
            owner_lock = DaemonOwnerLock(owner_lock_path_for_pid_file(get_pid_path()))
            acquired = owner_lock.acquire()
        except OSError as error:
            return DaemonResponse(
                success=False,
                output='',
                error=f'Could not inspect daemon ownership: {type(error).__name__}: {error}',
                error_code='daemon_cleanup_failed',
                hint='Check the daemon directory permissions and the socket path.',
            )
        if not acquired:
            return DaemonResponse(
                success=False,
                output='',
                error='Another process owns daemon startup or shutdown for this socket',
                error_code='daemon_busy',
                hint='Wait for the current daemon operation to finish, then retry.',
            )
        try:
            socket_path = get_socket_path()
            try:
                observed = socket_entry_identity(socket_path)
            except OSError as error:
                return DaemonResponse(
                    success=False,
                    output='',
                    error=f'Could not inspect the daemon socket path: {type(error).__name__}: {error}',
                    error_code='daemon_cleanup_failed',
                    hint='Check permissions on the daemon directory and socket path.',
                )
            state = await self._socket_state()
            if state == 'live':
                return DaemonResponse(
                    success=False,
                    output='',
                    error='Daemon socket still accepts connections',
                    error_code='daemon_stop_timeout',
                    hint='The daemon is still alive. Inspect it or run `openbrowser-ai daemon reap`.',
                )
            if state == 'unknown':
                return DaemonResponse(
                    success=False,
                    output='',
                    error='Daemon socket state cannot be verified safely',
                    error_code='daemon_socket_unverified',
                    hint='Inspect the socket path before removing it.',
                )
            owner = self._daemon_owner_for_socket()
            if owner is not None:
                return DaemonResponse(
                    success=False,
                    output='',
                    error=f'Daemon PID {owner["pid"]} is still running',
                    error_code='daemon_stop_timeout',
                    hint='Run `openbrowser-ai daemon reap` to verify and stop the owning daemon process.',
                )
            if socket_entry_identity(socket_path) != observed:
                return DaemonResponse(
                    success=False,
                    output='',
                    error='Daemon socket path changed during cleanup',
                    error_code='daemon_socket_changed',
                    hint='Inspect the path before removing it.',
                )
            try:
                self._remove_stale_socket(socket_path)
            except OSError as error:
                return DaemonResponse(
                    success=False,
                    output='',
                    error=str(error),
                    error_code='daemon_socket_foreign',
                    hint='Inspect the path before removing it. It is not a daemon socket.',
                )
            get_pid_path().unlink(missing_ok=True)
        finally:
            owner_lock.release()
        return None

    @staticmethod
    def _remove_stale_socket(path: Path) -> None:
        """Remove a stale Unix socket without deleting a regular file."""
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise OSError(f'Refusing to remove non-socket daemon path: {path}')
        path.unlink(missing_ok=True)

    async def _kill_recorded_browsers(
        self,
        profile_dirs: list[str],
        timeout: float,
        *,
        expected_owner_pid: int | None = None,
    ) -> tuple[bool, str | None]:
        """Kill the verified browser recorded by each matching profile lease."""
        seen: set[str] = set()
        for profile_dir in profile_dirs:
            if not profile_dir or profile_dir in seen:
                continue
            seen.add(profile_dir)
            browser_ok, browser_error = await self._kill_recorded_browser(
                profile_dir,
                timeout,
                expected_owner_pid=expected_owner_pid,
            )
            if not browser_ok:
                return False, browser_error
        return True, None

    async def _kill_recorded_browser(
        self,
        profile_dir: str | None,
        timeout: float,
        *,
        expected_owner_pid: int | None = None,
    ) -> tuple[bool, str | None]:
        """Kill the browser recorded by the profile lease, using verified identity."""
        if not profile_dir:
            return True, None
        try:
            from openbrowser.browser.watchdogs.profile_lease import ProfileLease

            metadata = ProfileLease.read_metadata(profile_dir)
            if expected_owner_pid is not None and (
                not isinstance(metadata, dict) or metadata.get('owner_pid') != expected_owner_pid
            ):
                # A replacement owner rewrote this lease. Never signal its browser.
                return True, None
            browser = metadata.get('browser') if isinstance(metadata, dict) else None
            if not isinstance(browser, dict):
                return True, None

            import psutil

            try:
                process = psutil.Process(int(browser['pid']))
            except psutil.NoSuchProcess:
                return True, None
            if not ProfileLease.process_matches_browser(process, browser):
                return True, None

            # Capture only descendants that carry this browser's ownership
            # marker. Never signal an arbitrary child of a reused PID.
            owned_children = []
            try:
                descendants = process.children(recursive=True)
            except psutil.NoSuchProcess:
                return True, None
            except (psutil.AccessDenied, psutil.ZombieProcess):
                return False, 'Unable to inspect browser descendants'
            for child in descendants:
                try:
                    child_record = {
                        **browser,
                        'pid': int(child.pid),
                        'start_time': float(child.create_time()),
                    }
                    if ProfileLease.process_matches_browser(child, child_record):
                        owned_children.append(child)
                except psutil.NoSuchProcess:
                    continue
                except (psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError, OSError):
                    return False, 'Unable to verify browser descendant identity'

            for child in owned_children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                for child in owned_children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait(timeout=timeout)
            return True, None
        except psutil.NoSuchProcess:
            return True, None
        except Exception as error:
            return False, f'{type(error).__name__}: {error}'


async def execute_code_via_daemon(code: str) -> DaemonResponse:
    """Convenience function for CLI usage."""
    client = DaemonClient()
    return await client.execute(code)
