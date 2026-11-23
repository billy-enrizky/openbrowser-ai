# src/openbrowser/daemon/server.py
"""OpenBrowser daemon -- persistent browser session over Unix socket.

Holds a browser session and CodeAgent namespace in memory.
CLI clients connect via Unix socket to execute code.

Usage:
    python -m openbrowser.daemon.server
"""

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import os
import signal
import stat
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from openbrowser.browser.cdp_connection import MAX_ERROR_CHARS, bound_text_with_suffix, sanitize_error_detail
from openbrowser.browser.download_state import downloads_summary
from openbrowser.browser.lifecycle import (
    LifecycleMachine,
    LifecyclePhase,
    LifecycleTransitionError,
)
from openbrowser.browser.storage_state import (
    StorageStatePhase,
    StorageStateSavePhase,
    StorageStateSaveSnapshot,
    StorageStateSnapshot,
)
from openbrowser.code_use.executor import output_limit_for_executor
from openbrowser.code_use.namespace_transfer import transfer_namespace
from openbrowser.config import CONFIG, ConfigStartupError, apply_managed_browser_profile_defaults, get_default_profile
from openbrowser.config_lifecycle import ConfigLoadResult, restart_required
from openbrowser.daemon import (
    EXIT_OWNER_BUSY,
    IS_WINDOWS,
    WINDOWS_PORT,
    get_pid_path,
    get_socket_path,
)
from openbrowser.daemon.ownership import (
    DaemonOwnerLock,
    SocketEntryIdentity,
    owner_lock_path_for_pid_file,
    socket_entry_identity,
)
from openbrowser.tools.action_availability import tool_availability_payload

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT = 600  # 10 minutes
DEFAULT_EXEC_TIMEOUT = 110  # Leave room for the client to receive a timeout response
DAEMON_INIT_TIMEOUT = 30  # Bound browser startup so clients never wait indefinitely
DAEMON_CLEANUP_TIMEOUT = 5  # Bound cleanup after failed startup
PID_FILE_WRITE_GRACE_SECONDS = 2.0  # Let another owner finish publishing an empty file


class RetainedBrowserSessionError(RuntimeError):
    """The daemon must reset a failed session before starting another browser."""


def _init_timeout() -> float:
    """Return the browser startup bound, allowing an environment override."""
    raw = os.environ.get('OPENBROWSER_DAEMON_INIT_TIMEOUT', '')
    try:
        return max(1.0, float(raw)) if raw else float(DAEMON_INIT_TIMEOUT)
    except ValueError:
        return float(DAEMON_INIT_TIMEOUT)


class DaemonStartupPhase(str, Enum):
    """Explicit phases of daemon ownership and socket startup."""

    IDLE = 'idle'
    CHECKING_OWNERSHIP = 'checking_ownership'
    CLEANING_STALE_LOCK = 'cleaning_stale_lock'
    ACQUIRING_OWNERSHIP = 'acquiring_ownership'
    BINDING_SOCKET = 'binding_socket'
    LISTENING = 'listening'
    FAILED = 'failed'
    STOPPED = 'stopped'


class PidOwnerState(str, Enum):
    """Ownership classification for a daemon PID and socket pair."""

    ABSENT = 'absent'
    ACTIVE = 'active'
    UNRESPONSIVE = 'unresponsive'
    STALE = 'stale'


class SocketProbeState(str, Enum):
    """Safety classification for a daemon socket path."""

    ABSENT = 'absent'
    ACTIVE = 'active'
    FOREIGN = 'foreign'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class PidInspection:
    """Ownership inspection result, including stale-file cleanup actions."""

    state: PidOwnerState
    pid: int | None = None
    detail: str | None = None
    stale_pid_removed: bool = False


# Names kept for the existing status/tests; the implementation is shared with
# the daemon client so both sides classify and compare the same entry.
SocketPathIdentity = SocketEntryIdentity
_socket_path_identity = socket_entry_identity


def _probe_socket(path: Path) -> SocketProbeState:
    """Classify a socket without treating an unprovable failure as stale."""
    if IS_WINDOWS:
        return SocketProbeState.UNKNOWN
    import socket as _socket

    try:
        if not stat.S_ISSOCK(path.lstat().st_mode):
            return SocketProbeState.FOREIGN
    except FileNotFoundError:
        return SocketProbeState.ABSENT
    except OSError:
        return SocketProbeState.UNKNOWN

    handle = None
    try:
        handle = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        handle.settimeout(1.0)
        handle.connect(str(path))
        return SocketProbeState.ACTIVE
    except (FileNotFoundError, ConnectionRefusedError, ValueError):
        return SocketProbeState.ABSENT
    except OSError:
        return SocketProbeState.UNKNOWN
    finally:
        if handle is not None:
            handle.close()


def _is_openbrowser_daemon_process(pid: int) -> bool | None:
    """Return whether this PID is the daemon for this socket path."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        from openbrowser.daemon.diagnostics import is_daemon_cmdline

        process = psutil.Process(pid)
        if not is_daemon_cmdline(process.cmdline()):
            return False
        try:
            environment = process.environ()
        except (psutil.AccessDenied, OSError):
            return None
        configured = str(environment.get('OPENBROWSER_SOCKET') or '').strip()
        if not configured:
            from openbrowser.daemon import SOCKET_PATH

            owner_socket = SOCKET_PATH.expanduser()
        else:
            owner_socket = Path(configured).expanduser()
            if not owner_socket.is_absolute():
                return None
        return owner_socket.resolve() == get_socket_path().resolve()
    except (psutil.Error, OSError, RuntimeError):
        return None


def _pid_file_age_seconds(path: Path) -> float | None:
    """Return the age of a PID file, or None when its metadata is unavailable."""
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _daemon_pid_exists(pid: int) -> bool | None:
    """Return process liveness without sending a signal on any platform."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.pid_exists(pid)
    except (psutil.Error, OSError):
        return None


_LOOPBACK_ADDRESSES = frozenset({'127.0.0.1', '::1'})


def _windows_pid_owns_listener(pid: int) -> bool | None:
    """Return whether a Windows PID owns the daemon loopback listener."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        for connection in psutil.net_connections(kind='tcp'):
            address = connection.laddr
            if (
                connection.pid == pid
                and connection.status == psutil.CONN_LISTEN
                and address
                and address.port == WINDOWS_PORT
                and address.ip in _LOOPBACK_ADDRESSES
            ):
                return True
        return False
    except (psutil.Error, OSError):
        return None


def _inspect_pid() -> PidInspection:
    """Inspect PID/socket ownership and remove stale files.

    A live OpenBrowser daemon is never treated as stale only because its
    socket is not yet accepting connections. That window exists while the
    owner is still binding and must not be stolen by another server.
    """
    pid_path = get_pid_path()
    if not pid_path.exists():
        return PidInspection(state=PidOwnerState.ABSENT)
    try:
        content = pid_path.read_bytes().strip()
    except FileNotFoundError:
        return PidInspection(state=PidOwnerState.ABSENT)
    except OSError as error:
        return PidInspection(
            state=PidOwnerState.UNRESPONSIVE,
            detail=f'PID file cannot be read: {type(error).__name__}: {error}',
        )
    try:
        pid = int(content)
        if not 0 < pid < 2**31:
            raise ValueError('PID is outside the supported range')
    except (ValueError, OverflowError):
        age = _pid_file_age_seconds(pid_path)
        if age is None:
            return PidInspection(
                state=PidOwnerState.UNRESPONSIVE,
                detail='PID file is empty or incomplete and its age cannot be checked safely',
            )
        if age < PID_FILE_WRITE_GRACE_SECONDS:
            return PidInspection(
                state=PidOwnerState.UNRESPONSIVE,
                detail='PID file is empty or incomplete and may still be written',
            )
        try:
            pid_path.unlink(missing_ok=True)
        except OSError as error:
            return PidInspection(
                state=PidOwnerState.UNRESPONSIVE,
                detail=f'Invalid PID file could not be removed: {type(error).__name__}: {error}',
            )
        return PidInspection(
            state=PidOwnerState.STALE,
            detail='PID file was invalid and was removed',
            stale_pid_removed=True,
        )

    process_exists = _daemon_pid_exists(pid)
    if process_exists is False:
        pid_path.unlink(missing_ok=True)
        return PidInspection(
            state=PidOwnerState.STALE,
            detail=f'PID {pid} no longer exists',
            stale_pid_removed=True,
        )
    if process_exists is None:
        return PidInspection(
            state=PidOwnerState.UNRESPONSIVE,
            pid=pid,
            detail=f'PID {pid} could not be checked safely',
        )

    if not IS_WINDOWS:
        socket_state = _probe_socket(get_socket_path())
        if socket_state is SocketProbeState.UNKNOWN:
            return PidInspection(
                state=PidOwnerState.UNRESPONSIVE,
                pid=pid,
                detail=f'PID {pid} may own the daemon socket, but its state cannot be proven safely',
            )
        if socket_state is SocketProbeState.ACTIVE:
            return PidInspection(state=PidOwnerState.ACTIVE, pid=pid)
        if socket_state is SocketProbeState.FOREIGN:
            return PidInspection(
                state=PidOwnerState.UNRESPONSIVE,
                pid=pid,
                detail=f'PID {pid} exists but the socket path is occupied by a non-socket file',
            )
        daemon_process = _is_openbrowser_daemon_process(pid)
        if daemon_process is False:
            pid_path.unlink(missing_ok=True)
            return PidInspection(
                state=PidOwnerState.STALE,
                detail=f'PID {pid} was reused by a non-daemon process',
                stale_pid_removed=True,
            )
        return PidInspection(
            state=PidOwnerState.UNRESPONSIVE,
            pid=pid,
            detail=f'PID {pid} owns the daemon lock but its socket is not accepting connections',
        )

    daemon_process = _is_openbrowser_daemon_process(pid)
    if daemon_process is False:
        pid_path.unlink(missing_ok=True)
        return PidInspection(
            state=PidOwnerState.STALE,
            detail=f'PID {pid} was reused by a non-daemon process',
            stale_pid_removed=True,
        )
    listener = _windows_pid_owns_listener(pid)
    if listener is True:
        return PidInspection(state=PidOwnerState.ACTIVE, pid=pid)
    if listener is False:
        return PidInspection(
            state=PidOwnerState.UNRESPONSIVE,
            pid=pid,
            detail=f'PID {pid} owns the daemon PID file but not the loopback listener',
        )
    return PidInspection(
        state=PidOwnerState.UNRESPONSIVE,
        pid=pid,
        detail=f'PID {pid} listener ownership could not be verified safely',
    )


def _read_pid() -> int | None:
    """Return a live daemon PID, or None after cleaning proven stale files."""
    inspection = _inspect_pid()
    if inspection.state in (PidOwnerState.ACTIVE, PidOwnerState.UNRESPONSIVE):
        return inspection.pid
    return None


def _peek_pid() -> int | None:
    """Read the PID file without changing ownership or deleting anything."""
    try:
        return int(get_pid_path().read_text().strip())
    except (OSError, ValueError):
        return None


def _acquire_pid_file(pid: int | None = None) -> bool:
    """Publish a complete PID file without exposing an empty owner file."""
    owner = os.getpid() if pid is None else pid
    pid_path = get_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{pid_path.name}.',
        suffix='.tmp',
        dir=pid_path.parent,
    )
    temporary_path = Path(temporary)
    try:
        try:
            if hasattr(os, 'fchmod'):
                os.fchmod(descriptor, 0o600)
            os.write(descriptor, str(owner).encode('utf-8'))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary_path, pid_path)
        except FileExistsError:
            return False
        except OSError:
            # Some filesystems do not support hard links. Fall back to an
            # exclusive create while keeping the complete-content window small.
            try:
                handle = os.open(pid_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                return False
            try:
                os.write(handle, str(owner).encode('utf-8'))
                os.fsync(handle)
            except BaseException:
                pid_path.unlink(missing_ok=True)
                raise
            finally:
                os.close(handle)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_pid() -> None:
    """Write this process PID, raising if another owner exists."""
    if not _acquire_pid_file():
        raise FileExistsError(f'PID file already exists: {get_pid_path()}')


def _cleanup_pid(
    expected_pid: int | None = None,
    *,
    remove_socket: bool = True,
    socket_identity: SocketPathIdentity | None = None,
) -> None:
    """Remove daemon files only when this process still owns them.

    ``socket_identity`` is the identity captured after binding. When given,
    the socket path is removed only while it still names that same socket
    entry, so a replacement file at the same path survives shutdown.
    """
    pid_path = get_pid_path()
    owner_verified = expected_pid is None
    if expected_pid is not None and pid_path.exists():
        try:
            owner = int(pid_path.read_text().strip())
        except (ValueError, OSError):
            logger.warning('Skipping daemon cleanup; PID file ownership cannot be verified')
            return
        if owner != expected_pid:
            logger.warning('Skipping daemon cleanup; PID file now belongs to %d', owner)
            return
        owner_verified = True
    if remove_socket and owner_verified:
        socket_path = get_socket_path()
        current_identity = _socket_path_identity(socket_path)
        if socket_identity is None:
            logger.warning('Skipping daemon socket cleanup; no bound socket identity was captured')
        elif current_identity is None:
            pass
        elif not current_identity.is_socket:
            logger.warning('Skipping daemon socket cleanup; path is not a socket')
        elif current_identity != socket_identity:
            logger.warning('Skipping daemon socket cleanup; path no longer names the bound socket')
        else:
            try:
                socket_path.unlink(missing_ok=True)
            except OSError:
                logger.warning('Failed to remove daemon socket', exc_info=True)
    pid_path.unlink(missing_ok=True)


_STARTUP_TRANSITIONS: dict[DaemonStartupPhase, frozenset[DaemonStartupPhase]] = {
    DaemonStartupPhase.IDLE: frozenset({DaemonStartupPhase.CHECKING_OWNERSHIP, DaemonStartupPhase.FAILED}),
    DaemonStartupPhase.CHECKING_OWNERSHIP: frozenset(
        {
            DaemonStartupPhase.CLEANING_STALE_LOCK,
            DaemonStartupPhase.ACQUIRING_OWNERSHIP,
            DaemonStartupPhase.FAILED,
        }
    ),
    DaemonStartupPhase.CLEANING_STALE_LOCK: frozenset(
        {DaemonStartupPhase.ACQUIRING_OWNERSHIP, DaemonStartupPhase.FAILED}
    ),
    DaemonStartupPhase.ACQUIRING_OWNERSHIP: frozenset({DaemonStartupPhase.BINDING_SOCKET, DaemonStartupPhase.FAILED}),
    DaemonStartupPhase.BINDING_SOCKET: frozenset({DaemonStartupPhase.LISTENING, DaemonStartupPhase.FAILED}),
    DaemonStartupPhase.LISTENING: frozenset({DaemonStartupPhase.STOPPED, DaemonStartupPhase.FAILED}),
    DaemonStartupPhase.FAILED: frozenset(),
    DaemonStartupPhase.STOPPED: frozenset(),
}


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
        self._tools = None
        self._running = False
        self._server = None
        self._stop_event = asyncio.Event()
        self._exec_lock = asyncio.Lock()  # serialize code execution (stdout safety)
        self._init_lock = asyncio.Lock()  # guard lazy initialization
        self._lifecycle = LifecycleMachine('daemon')
        self._browser_error: str | None = None
        self._needs_browser_recovery = False
        self._last_namespace_transfer: dict[str, object] = {}
        self._last_closed_session: object | None = None
        self._profile_summary: dict[str, object] | None = None
        self._browser_generation = 0
        self._cleanup_required = False
        self._startup_phase = DaemonStartupPhase.IDLE
        self._startup_detail: str | None = None
        self._startup_history: list[DaemonStartupPhase] = []
        self._startup_stale_pid_removed = False
        self._startup_stale_socket_removed = False
        self._startup_owner_pid: int | None = None
        self._startup_owner_state: PidOwnerState | None = None
        self._startup_owner_busy = False
        self._bound_socket_identity: SocketPathIdentity | None = None
        self._storage_state_preload = StorageStateSnapshot(phase=StorageStatePhase.NOT_CONFIGURED)
        self._storage_state_save_preload = StorageStateSaveSnapshot(phase=StorageStateSavePhase.NOT_CONFIGURED)
        self._startup_config_result: ConfigLoadResult = CONFIG.load_result()
        self._startup_config: dict[str, Any] = CONFIG.config_dict_from_result(self._startup_config_result)

    def _set_startup(self, phase: DaemonStartupPhase, detail: str | None = None) -> None:
        """Move the startup machine through an explicitly allowed transition."""
        if phase is self._startup_phase:
            self._startup_detail = detail
            return
        if phase not in _STARTUP_TRANSITIONS[self._startup_phase]:
            raise RuntimeError(f'Invalid daemon startup transition: {self._startup_phase.value} -> {phase.value}')
        self._startup_phase = phase
        self._startup_detail = detail
        self._startup_history.append(phase)

    def _fail_startup(self, detail: str, error: BaseException | str | None = None) -> None:
        """Record one terminal startup failure without hiding it as a clean stop."""
        bounded = sanitize_error_detail(detail, max_chars=500) or ''
        self._set_startup(DaemonStartupPhase.FAILED, bounded)
        if self._lifecycle.phase is not LifecyclePhase.FAILED:
            try:
                self._lifecycle.fail(error or bounded, reason='daemon start failed')
            except LifecycleTransitionError:
                pass

    def _record_storage_state(self, storage_state: object) -> None:
        """Record profile storage-state metadata for pre-browser status."""
        if isinstance(storage_state, (str, Path)):
            path = str(storage_state)
            expanded: Path | None = None
            try:
                expanded = Path(path).expanduser().resolve()
                size = expanded.stat().st_size
            except FileNotFoundError:
                resolved = str(expanded) if expanded is not None else path
                self._storage_state_preload = StorageStateSnapshot(
                    phase=StorageStatePhase.SKIPPED,
                    path=resolved,
                    detail='storage-state file does not exist',
                )
                self._storage_state_save_preload = StorageStateSaveSnapshot(
                    phase=StorageStateSavePhase.IDLE,
                    path=resolved,
                    detail='storage-state file does not exist yet',
                )
            except (OSError, RuntimeError) as error:
                error_text = f'{type(error).__name__}: {error}'
                self._storage_state_preload = StorageStateSnapshot(
                    phase=StorageStatePhase.FAILED,
                    path=path,
                    error=error_text,
                    detail='storage-state path could not be resolved or inspected',
                )
                self._storage_state_save_preload = StorageStateSaveSnapshot(
                    phase=StorageStateSavePhase.FAILED,
                    path=path,
                    error=error_text,
                    detail='storage-state path could not be resolved',
                )
            else:
                self._storage_state_preload = StorageStateSnapshot(
                    phase=StorageStatePhase.PENDING,
                    path=str(expanded),
                    bytes=size,
                )
                self._storage_state_save_preload = StorageStateSaveSnapshot(
                    phase=StorageStateSavePhase.IDLE,
                    path=str(expanded),
                )
            return
        if isinstance(storage_state, dict):
            self._storage_state_preload = StorageStateSnapshot(
                phase=StorageStatePhase.SKIPPED,
                detail='in-memory storage state is not loaded from a file',
            )
            self._storage_state_save_preload = StorageStateSaveSnapshot(
                phase=StorageStateSavePhase.SKIPPED,
                detail='in-memory storage state is not saved to a file',
            )
            return
        self._storage_state_preload = StorageStateSnapshot(phase=StorageStatePhase.NOT_CONFIGURED)
        self._storage_state_save_preload = StorageStateSaveSnapshot(phase=StorageStateSavePhase.NOT_CONFIGURED)

    def _downloads_snapshot(self) -> dict:
        """Return a bounded download outcome summary without starting a browser."""
        return downloads_summary(self._session)

    def _storage_state_snapshot(self) -> StorageStateSnapshot:
        """Return the watchdog-owned load state, or the pre-load view."""
        watchdog = getattr(self._session, '_storage_state_watchdog', None)
        snapshot = getattr(watchdog, 'storage_state_snapshot', None)
        if callable(snapshot):
            try:
                value = snapshot()
            except Exception:
                logger.warning('Could not read storage-state snapshot', exc_info=True)
            else:
                if isinstance(value, StorageStateSnapshot):
                    return value
        return self._storage_state_preload

    def _storage_state_save_snapshot(self) -> StorageStateSaveSnapshot:
        """Return the watchdog-owned save state, or the pre-save view."""
        watchdog = getattr(self._session, '_storage_state_watchdog', None)
        snapshot = getattr(watchdog, 'storage_state_save_snapshot', None)
        if callable(snapshot):
            try:
                value = snapshot()
            except Exception:
                logger.warning('Could not read storage-state save snapshot', exc_info=True)
            else:
                if isinstance(value, StorageStateSaveSnapshot):
                    return value
        return self._storage_state_save_preload

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
        """Build a BrowserProfile from the frozen startup config."""
        from openbrowser.browser import BrowserProfile
        from openbrowser.browser.profile_provenance import ProfileProvenance, ProfileValueSource

        assert self._startup_config is not None
        profile_config = dict(get_default_profile(self._startup_config))
        default_profile_dir = self._default_profile_dir()
        defaults = {
            'downloads_path': str(Path.home() / 'Downloads' / 'openbrowser-daemon'),
            'wait_between_actions': 0.5,
            'keep_alive': True,
            'device_scale_factor': 1.0,
            'disable_security': False,
            'headless': True,
        }
        provenance = ProfileProvenance()
        provenance.record_mapping(defaults, ProfileValueSource.DEFAULT)
        provenance.record_mapping(profile_config, ProfileValueSource.CONFIG)
        provenance.record_defaults(
            {'channel': None, 'executable_path': None, 'args': [], 'proxy': None}
        )
        env_user_data = os.environ.get('OPENBROWSER_USER_DATA_DIR')
        if env_user_data and str(profile_config.get('user_data_dir') or '') == env_user_data:
            provenance.record(
                'user_data_dir', profile_config.get('user_data_dir'), ProfileValueSource.ENVIRONMENT
            )
        env_storage = os.environ.get('OPENBROWSER_STORAGE_STATE')
        if env_storage and str(profile_config.get('storage_state') or '') == env_storage:
            provenance.record(
                'storage_state', profile_config.get('storage_state'), ProfileValueSource.ENVIRONMENT
            )
        if isinstance(profile_config.get('proxy'), dict) and any(
            os.environ.get(name)
            for name in (
                'OPENBROWSER_PROXY_URL',
                'OPENBROWSER_NO_PROXY',
                'OPENBROWSER_PROXY_USERNAME',
                'OPENBROWSER_PROXY_PASSWORD',
            )
        ):
            provenance.record('proxy', profile_config.get('proxy'), ProfileValueSource.ENVIRONMENT)
        profile_data = {**defaults, **profile_config}
        isolate = os.environ.get('OPENBROWSER_DAEMON_PROFILE_ISOLATION', '').lower() in ('true', '1', 'yes')
        daemon_profile_override = bool(os.environ.get('OPENBROWSER_DAEMON_PROFILE_DIR')) or (
            bool(os.environ.get('OPENBROWSER_SOCKET')) and isolate
        )
        if daemon_profile_override and not os.environ.get('OPENBROWSER_USER_DATA_DIR'):
            profile_data['user_data_dir'] = str(default_profile_dir)
            provenance.record('user_data_dir', profile_data['user_data_dir'], ProfileValueSource.MANAGED)
        elif not profile_data.get('user_data_dir'):
            profile_data['user_data_dir'] = str(default_profile_dir)
            provenance.record('user_data_dir', profile_data['user_data_dir'], ProfileValueSource.MANAGED)
        before_managed = dict(profile_data)
        profile_data = apply_managed_browser_profile_defaults(profile_data, default_profile_dir)
        provenance.record_changes(before_managed, profile_data, ProfileValueSource.MANAGED)
        # The daemon already persists cookies and storage in its Chrome
        # user-data directory. Replaying a second snapshot on every startup
        # can be enormous and blocks the first request while CDP applies it.
        # Keep storage-state loading only when the user explicitly configured
        # a snapshot path.
        if not profile_config.get('storage_state'):
            profile_data.pop('storage_state', None)
            provenance.record('storage_state', None, ProfileValueSource.DEFAULT)

        # Stored config (profile_config) may have headless=False from a previous
        # non-headless session, overriding the daemon default above. Apply the env
        # var override last so OPENBROWSER_HEADLESS=false is the only way to opt
        # into a visible window; omitting the var keeps daemon headless by default.
        env_headless = os.environ.get('OPENBROWSER_HEADLESS', '').lower()
        if env_headless in ('true', '1', 'yes'):
            profile_data['headless'] = True
            provenance.record('headless', True, ProfileValueSource.ENVIRONMENT)
        elif env_headless in ('false', '0', 'no'):
            profile_data['headless'] = False
            provenance.record('headless', False, ProfileValueSource.ENVIRONMENT)
        else:
            profile_data['headless'] = True  # daemon default: always headless
            provenance.record('headless', True, ProfileValueSource.DEFAULT)
        profile = BrowserProfile(**profile_data)
        provenance.refresh_from_profile(profile)
        storage_state = profile.storage_state
        self._profile_summary = {
            'user_data_dir': str(profile.user_data_dir) if profile.user_data_dir else None,
            'headless': profile.headless,
            'storage_state': str(storage_state) if isinstance(storage_state, (str, Path)) else None,
            'resolution': provenance.to_dict(),
        }
        self._record_storage_state(storage_state)
        return profile

    async def _ensure_executor(self):
        """Lazy-initialize browser + namespace on first request."""
        if getattr(self, '_cleanup_required', False):
            raise RetainedBrowserSessionError(
                'A previous daemon reset failed; run daemon reset again before retrying'
            )
        if self._executor is not None:
            return
        async with self._init_lock:
            if self._executor is not None:
                return  # another coroutine initialized while we waited
            if self._session is not None:
                raise RetainedBrowserSessionError(
                    'A failed browser session still requires cleanup; run daemon reset before retrying'
                )

            # Suppress verbose logging for clean daemon output
            os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'critical'
            os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'
            logging.getLogger('openbrowser').setLevel(logging.ERROR)

            from openbrowser.browser import BrowserSession
            from openbrowser.code_use.executor import DEFAULT_MAX_OUTPUT_CHARS, CodeExecutor
            from openbrowser.code_use.namespace import create_namespace
            from openbrowser.tools.service import CodeAgentTools

            CONFIG.require_browser_start_allowed(self._startup_config_result)
            profile = self._build_browser_profile()
            session = BrowserSession(browser_profile=profile)
            try:
                init_timeout = getattr(self, '_init_timeout', _init_timeout())
                try:
                    await asyncio.wait_for(session.start(), timeout=init_timeout)
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(f'Browser initialization timed out after {init_timeout:g}s') from exc
                seeded: Any = session.seed_generation(getattr(self, '_browser_generation', 0))
                if inspect.isawaitable(seeded):
                    await seeded

                tools = CodeAgentTools()
                namespace = create_namespace(browser_session=session, tools=tools)

                try:
                    max_output = int(os.environ.get('OPENBROWSER_MAX_OUTPUT', '0'))
                except (ValueError, TypeError):
                    max_output = 0
                executor = CodeExecutor(max_output_chars=max_output if max_output > 0 else DEFAULT_MAX_OUTPUT_CHARS)
                executor.set_namespace(namespace)
                self._executor = executor
                self._session = session
                self._tools = tools
                self._browser_error = None
                self._needs_browser_recovery = False
            except BaseException as error:
                # Kill the browser if startup or namespace setup fails. Bound
                # cleanup too, so a broken browser cannot block the client.
                self._browser_error = f'{type(error).__name__}: {sanitize_error_detail(error) or type(error).__name__}'
                self._tools = None
                self._session = session
                try:
                    cleaned = await self._cleanup_failed_session(session)
                except asyncio.CancelledError:
                    self._cleanup_required = True
                    raise
                except BaseException:
                    logger.warning('Failed to clean up browser after daemon initialization failure', exc_info=True)
                    cleaned = False
                if cleaned and self._session is session:
                    self._session = None
                raise

    async def _cleanup_failed_session(self, session) -> bool:
        """Bound daemon startup cleanup.

        Cancel the wrapper at the deadline or on caller cancellation. The daemon
        retains the BrowserSession owner, refuses execution, and lets a later
        daemon reset retry the cancellation-resistant cleanup.
        """
        cleanup_task = asyncio.create_task(session.kill())
        try:
            done, pending = await asyncio.wait({cleanup_task}, timeout=DAEMON_CLEANUP_TIMEOUT)
        except BaseException:
            cleanup_task.add_done_callback(self._log_cleanup_task_failure)
            cleanup_task.cancel()
            raise

        if pending:
            cleanup_task.add_done_callback(self._log_cleanup_task_failure)
            cleanup_task.cancel()
            logger.warning(
                'Browser cleanup exceeded %ss after daemon initialization failure; cleanup retained',
                DAEMON_CLEANUP_TIMEOUT,
            )
            return False

        # Consume the completed cleanup result before the startup failure is re-raised.
        done_task = done.pop()
        try:
            done_task.result()
        except BaseException:
            logger.warning('Failed to clean up browser after daemon initialization failure', exc_info=True)
            return False
        return True

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
                logger.warning(
                    'Detached daemon browser cleanup task failed', exc_info=(type(error), error, error.__traceback__)
                )

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
                self._needs_browser_recovery = False
                return

            CONFIG.require_browser_start_allowed(self._startup_config_result)
            logger.info('CDP connection lost -- recovering browser session')
            self._needs_browser_recovery = True
            self._tools = None
            old_namespace = getattr(self._executor, 'namespace', None) if self._executor else {}
            if not isinstance(old_namespace, dict):
                old_namespace = {}
            old_session = self._session or self._last_closed_session
            transfer = transfer_namespace(old_namespace, old_browser_session=old_session)
            self._last_namespace_transfer = transfer.to_dict()
            if self._session:
                session_to_close = self._session
                closed = await self._close_session_bounded(session_to_close)
                if not closed:
                    raise RuntimeError('Unable to close the old browser session before recovery')
                self._session = None
                self._tools = None
                self._last_closed_session = session_to_close
                self._needs_browser_recovery = True

            from openbrowser.browser import BrowserSession
            from openbrowser.code_use.namespace import create_namespace
            from openbrowser.tools.service import CodeAgentTools

            session = None
            try:
                profile = self._build_browser_profile()
                session = BrowserSession(browser_profile=profile)
                await asyncio.wait_for(session.start(), timeout=getattr(self, '_init_timeout', _init_timeout()))
                self._browser_generation = getattr(self, '_browser_generation', 0) + 1
                seeded: Any = session.seed_generation(self._browser_generation)
                if inspect.isawaitable(seeded):
                    await seeded
                tools = CodeAgentTools()
                namespace = create_namespace(browser_session=session, tools=tools)
                for key, value in transfer.variables.items():
                    if key not in namespace:
                        namespace[key] = value
                if transfer.warnings:
                    logger.warning('Namespace transfer: %s', '; '.join(transfer.warnings))
                executor = self._executor
                if executor is None:
                    raise RuntimeError('Cannot recover a browser without an executor')
                executor.set_namespace(namespace)
                self._session = session
                self._tools = tools
                self._browser_error = None
                self._needs_browser_recovery = False
                self._last_closed_session = None
            except BaseException as error:
                self._browser_error = f'{type(error).__name__}: {sanitize_error_detail(error) or type(error).__name__}'
                self._tools = None
                if session is not None:
                    self._session = session
                    try:
                        cleaned = await self._cleanup_failed_session(session)
                    except asyncio.CancelledError:
                        self._cleanup_required = True
                        raise
                    except BaseException:
                        logger.warning('Failed to clean up replacement browser session', exc_info=True)
                        cleaned = False
                    if cleaned and self._session is session:
                        self._session = None
                if self._executor is not None:
                    # The old session is gone. Remove stale browser references
                    # from the live namespace while keeping user scalars for the
                    # next recovery.
                    self._executor.set_namespace(dict(transfer.variables))
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
                self._browser_error = (
                    f'Profile configuration failed: {type(error).__name__}: '
                    f'{sanitize_error_detail(error) or type(error).__name__}'
                )

        browser: dict
        if self._session is not None and hasattr(self._session, 'lifecycle_snapshot'):
            try:
                snapshot_value = self._session.lifecycle_snapshot()
            except Exception:
                browser = {'phase': 'unknown'}
                health_snapshot = getattr(self._session, 'cdp_connection_health', None)
                if callable(health_snapshot):
                    with contextlib.suppress(Exception):
                        health_value = health_snapshot()
                        to_dict = getattr(health_value, 'to_dict', None)
                        if callable(to_dict):
                            browser['cdp'] = to_dict()
                            browser['cdp_connected'] = bool(getattr(health_value, 'client_present', False))
                focus_snapshot = getattr(self._session, 'focus_snapshot', None)
                if callable(focus_snapshot):
                    with contextlib.suppress(Exception):
                        focus_value = focus_snapshot()
                        to_dict = getattr(focus_value, 'to_dict', None)
                        if callable(to_dict):
                            browser['focus'] = to_dict()
            else:
                browser = snapshot_value if isinstance(snapshot_value, dict) else {'phase': 'unknown'}
        elif self._session is not None:
            browser = {'phase': getattr(self._session, 'state', 'unknown')}
        else:
            browser = {'phase': 'stopped', 'last_error': self._browser_error}
        if 'focus' not in browser:
            from openbrowser.browser.focus import FocusSnapshot

            browser['focus'] = FocusSnapshot.uninitialized().to_dict()

        try:
            from openbrowser.utils import get_openbrowser_version

            version = get_openbrowser_version()
        except Exception:
            version = None

        snapshot = self._lifecycle.snapshot()
        storage_state = self._storage_state_snapshot()
        downloads = self._downloads_snapshot()
        tools_snapshot = tool_availability_payload(self._tools, browser)
        startup_config = self._startup_config_result.snapshot
        try:
            current_config = CONFIG.load_result().snapshot
            needs_restart = restart_required(startup_config, current_config)
        except Exception:
            current_config = startup_config
            needs_restart = True
        return {
            'pid': os.getpid(),
            'state': snapshot.phase.value,
            'initialized': self._executor is not None,
            'needs_browser_recovery': self._needs_browser_recovery,
            'socket': str(get_socket_path()),
            'profile': self._profile_summary,
            'idle_timeout': self._idle_timeout,
            'init_timeout': self._init_timeout,
            'exec_timeout': self._exec_timeout,
            'last_error': sanitize_error_detail(snapshot.last_error or self._browser_error),
            'generation': snapshot.generation,
            'uptime_seconds': max(0.0, time.time() - self._started_at),
            'version': version,
            'browser': browser,
            'startup': {
                'phase': self._startup_phase.value,
                'detail': sanitize_error_detail(self._startup_detail, max_chars=500),
                'history': [phase.value for phase in self._startup_history],
                'stale_pid_removed': self._startup_stale_pid_removed,
                'stale_socket_removed': self._startup_stale_socket_removed,
                'owner_pid': self._startup_owner_pid,
                'owner_state': self._startup_owner_state.value if self._startup_owner_state else None,
            },
            'storage_state': storage_state.to_dict(),
            'storage_state_save': self._storage_state_save_snapshot().to_dict(),
            'namespace_transfer': self._last_namespace_transfer,
            'tools': tools_snapshot,
            'downloads': downloads,
            'config': {
                'startup': startup_config.to_dict(),
                'current': current_config.to_dict(),
                'restart_required': needs_restart,
            },
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
        """Close a session without letting a broken browser block the daemon.

        A timed-out kill wrapper is cancelled and the retained session remains the
        reset owner. Caller cancellation also cancels and observes the wrapper.
        """
        task = asyncio.create_task(session.kill())
        try:
            done, pending = await asyncio.wait({task}, timeout=DAEMON_CLEANUP_TIMEOUT)
        except asyncio.CancelledError:
            task.add_done_callback(self._log_cleanup_task_failure)
            task.cancel()
            raise
        if pending:
            task.add_done_callback(self._log_cleanup_task_failure)
            task.cancel()
            logger.warning('Browser cleanup exceeded %ss; cleanup retained', DAEMON_CLEANUP_TIMEOUT)
            return False
        done_task = done.pop()
        try:
            done_task.result()
        except BaseException:
            logger.warning('Failed to close browser session', exc_info=True)
            return False
        return True

    async def _close_server_on_startup_failure(self) -> None:
        """Close a listener that was created before the listen loop was entered."""
        server = self._server
        if server is None:
            return
        try:
            closed: Any = server.close()
            if inspect.isawaitable(closed):
                await closed
            waited: Any = server.wait_closed()
            if inspect.isawaitable(waited):
                await waited
        except BaseException:
            logger.warning('Failed to close daemon listener after startup failure', exc_info=True)

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
                    except ConfigStartupError as error:
                        safe_error = sanitize_error_detail(error) or error.code
                        safe_hint = sanitize_error_detail(error.hint) or error.hint
                        return self._error_response(req_id, safe_error, code=error.code, hint=safe_hint)
                    except RetainedBrowserSessionError as error:
                        safe_error = sanitize_error_detail(error) or type(error).__name__
                        return self._error_response(
                            req_id,
                            safe_error,
                            code='daemon_reset_required',
                            hint='Run `openbrowser-ai daemon reset`, then retry the request.',
                        )
                    except Exception as error:
                        safe_error = sanitize_error_detail(error) or type(error).__name__
                        return self._error_response(
                            req_id,
                            f'Browser initialization failed: {type(error).__name__}: {safe_error}',
                            code='daemon_start_failed',
                            hint='Run `openbrowser-ai daemon doctor` and check the profile path and daemon log.',
                        )
                    if self._needs_browser_recovery:
                        try:
                            await self._recover_browser_session()
                        except ConfigStartupError as error:
                            safe_error = sanitize_error_detail(error) or error.code
                            safe_hint = sanitize_error_detail(error.hint) or error.hint
                            return self._error_response(req_id, safe_error, code=error.code, hint=safe_hint)
                        except Exception as recovery_error:
                            safe_error = sanitize_error_detail(recovery_error) or type(recovery_error).__name__
                            return self._error_response(
                                req_id,
                                f'Browser recovery failed: {type(recovery_error).__name__}: {safe_error}',
                                code='cdp_recovery_failed',
                                hint=(
                                    'Run `openbrowser-ai daemon doctor` or `openbrowser-ai daemon reset` '
                                    'before retrying.'
                                ),
                            )
                    executor = self._executor
                    if executor is None:
                        return self._error_response(
                            req_id,
                            'Browser executor is unavailable after initialization',
                            code='daemon_start_failed',
                            hint='Run `openbrowser-ai daemon reset` and retry.',
                        )
                    failed_session = self._session
                    result = await asyncio.wait_for(executor.execute(code), timeout=self._exec_timeout)

                    # Recover while still holding the execution lock so the next
                    # request cannot run user code against the dead browser.
                    error_text = result.error or result.output
                    safe_output = result.output
                    safe_error = result.error
                    if not result.success:
                        output_limit = output_limit_for_executor(executor)
                        safe_output = sanitize_error_detail(
                            result.output, max_chars=output_limit, preserve_lines=True
                        )
                        safe_error = sanitize_error_detail(
                            result.error, max_chars=output_limit, preserve_lines=True
                        )
                        safe_output = (
                            safe_output or safe_error or 'Execution failed without diagnostic text'
                        )
                    if not result.success and self._is_connection_error(error_text):
                        try:
                            await self._recover_browser_session(expected_session=failed_session)
                        except ConfigStartupError as error:
                            recovery_budget = max(1, output_limit // 4)
                            recovery_detail = sanitize_error_detail(
                                error.detail, max_chars=recovery_budget, preserve_lines=True
                            )
                            recovery_text = recovery_detail or sanitize_error_detail(
                                error, max_chars=recovery_budget, preserve_lines=True
                            )
                            if recovery_text:
                                partial_budget = max(0, output_limit - len(recovery_text) - 1)
                                partial_detail = (
                                    sanitize_error_detail(
                                        result.output or result.error,
                                        max_chars=partial_budget,
                                        preserve_lines=True,
                                    )
                                    if partial_budget > 0
                                    else None
                                )
                                typed_error = (
                                    f'{partial_detail}\n{recovery_text}'
                                    if partial_detail
                                    else recovery_text
                                )
                            else:
                                typed_error = (
                                    sanitize_error_detail(
                                        result.output or result.error,
                                        max_chars=output_limit,
                                        preserve_lines=True,
                                    )
                                    or safe_output
                                    or error.code
                                )
                            safe_hint = bound_text_with_suffix(
                                error.hint,
                                'The code may have partially run; verify the page before retrying.',
                                max_chars=MAX_ERROR_CHARS,
                            )
                            return self._error_response(
                                req_id,
                                typed_error,
                                code=error.code,
                                hint=safe_hint or error.code,
                            )
                        except Exception as recovery_err:
                            logger.error('CDP recovery failed: %s', recovery_err)
                            safe_recovery_error = (
                                sanitize_error_detail(recovery_err) or type(recovery_err).__name__
                            )
                            safe_recovery_hint = bound_text_with_suffix(
                                f'Browser recovery failed: {type(recovery_err).__name__}: {safe_recovery_error}',
                                'The code may have partially run; verify the page before retrying.',
                                max_chars=MAX_ERROR_CHARS,
                            )
                            return self._error_response(
                                req_id,
                                safe_output,
                                code='cdp_recovery_failed',
                                hint=safe_recovery_hint or type(recovery_err).__name__,
                            )
                        return self._error_response(
                            req_id,
                            safe_output,
                            code='cdp_connection_lost',
                            hint=(
                                'The browser was recovered. The code may have partially run; '
                                'verify the page before retrying.'
                            ),
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
                'output': safe_output,
                'error': safe_error,
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
                        try:
                            closed = await self._close_session_bounded(self._session)
                        except asyncio.CancelledError:
                            self._cleanup_required = True
                            raise
                        if not closed:
                            self._cleanup_required = True
                            return self._error_response(
                                req_id,
                                'Unable to close the current browser session',
                                code='reset_failed',
                                hint=(
                                    'Run `openbrowser-ai daemon reset` again, then `daemon doctor` '
                                    'if cleanup remains stuck.'
                                ),
                            )
                    self._cleanup_required = False
                    self._executor = None
                    self._session = None
                    self._tools = None
                    self._browser_error = None
                    self._needs_browser_recovery = False
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
                safe_error = sanitize_error_detail(e) or type(e).__name__
                err = {
                    'id': 0,
                    'success': False,
                    'output': '',
                    'error': safe_error,
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

    async def serve_daemon(self):
        """Start the daemon and listen for connections."""
        self._lifecycle.transition(LifecyclePhase.STARTING, reason='daemon starting')
        self._running = True
        self._last_activity = time.time()

        sock_path = get_socket_path()
        owner_lock = DaemonOwnerLock(owner_lock_path_for_pid_file(get_pid_path()))
        owner_lock_acquired = False
        acquired_pid = False
        bound_socket = False
        idle_task: asyncio.Task | None = None
        try:
            sock_path.parent.mkdir(parents=True, exist_ok=True)
            self._set_startup(DaemonStartupPhase.CHECKING_OWNERSHIP, f'pid_file={get_pid_path()}')
            if not owner_lock.acquire():
                existing_pid = _peek_pid()
                self._startup_owner_pid = existing_pid
                self._startup_owner_state = PidOwnerState.UNRESPONSIVE
                self._startup_owner_busy = True
                message = f'Another daemon owns the startup lock (PID {existing_pid or "unknown"})'
                logger.error('%s', message)
                self._fail_startup(message)
                return
            owner_lock_acquired = True
            inspection = _inspect_pid()
            self._startup_owner_state = inspection.state
            self._startup_stale_pid_removed = inspection.stale_pid_removed
            if inspection.stale_pid_removed:
                self._set_startup(
                    DaemonStartupPhase.CLEANING_STALE_LOCK,
                    f'removed_stale_pid={inspection.stale_pid_removed}',
                )
            if inspection.state in (PidOwnerState.ACTIVE, PidOwnerState.UNRESPONSIVE):
                self._startup_owner_pid = inspection.pid
                message = inspection.detail or f'PID {inspection.pid} already owns the daemon lock'
                logger.error('%s', message)
                self._fail_startup(message)
                return

            self._set_startup(DaemonStartupPhase.ACQUIRING_OWNERSHIP)
            if not _acquire_pid_file(os.getpid()):
                existing_pid = _peek_pid()
                self._startup_owner_pid = existing_pid
                message = f'Another daemon acquired the PID file (PID {existing_pid or "unknown"})'
                logger.error('%s', message)
                self._fail_startup(message)
                return

            acquired_pid = True
            self._startup_owner_pid = os.getpid()
            self._set_startup(DaemonStartupPhase.BINDING_SOCKET, str(sock_path))
            socket_probe = SocketProbeState.ABSENT
            socket_identity = _socket_path_identity(sock_path)
            if not IS_WINDOWS:
                socket_probe = _probe_socket(sock_path)
                if socket_probe is SocketProbeState.FOREIGN:
                    message = 'Daemon socket path exists but is not a Unix socket file'
                    logger.error('%s', message)
                    self._fail_startup(message)
                    return
                if socket_probe is not SocketProbeState.ABSENT:
                    message = (
                        'Daemon socket accepts connections but has no live PID owner'
                        if socket_probe is SocketProbeState.ACTIVE
                        else 'Daemon socket state is unknown and cannot be replaced safely'
                    )
                    logger.error('%s', message)
                    self._fail_startup(message)
                    return

            if IS_WINDOWS:
                # Windows: TCP on loopback only (no Unix sockets available).
                # Any local process can connect; consider a shared-secret
                # auth token if multi-user security is required.
                self._server = await asyncio.start_server(self._handle_client, '127.0.0.1', WINDOWS_PORT)
            else:
                current_identity = _socket_path_identity(sock_path)
                if socket_identity is None:
                    if current_identity is not None:
                        message = 'Daemon socket path appeared during startup'
                        logger.error('%s', message)
                        self._fail_startup(message)
                        return
                elif current_identity != socket_identity or not socket_identity.is_socket:
                    message = 'Daemon socket path changed during startup'
                    logger.error('%s', message)
                    self._fail_startup(message)
                    return
                else:
                    # The PID file and owner lock prove this is the stale path.
                    sock_path.unlink(missing_ok=True)
                    self._startup_stale_socket_removed = True
                self._server = await asyncio.start_unix_server(self._handle_client, path=str(sock_path))
                bound_socket = True
                # Captured immediately after the bind. On macOS a Unix socket
                # fd reports the socket object, not the filesystem entry, so
                # fstat cannot corroborate this identity portably.
                self._bound_socket_identity = _socket_path_identity(sock_path)
                self._startup_owner_state = PidOwnerState.ACTIVE
                # Restrict socket permissions to owner only
                os.chmod(str(sock_path), 0o600)
            if IS_WINDOWS:
                self._startup_owner_state = PidOwnerState.ACTIVE

            # Register signal handlers on the running event loop
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, self._signal_shutdown)
                except (NotImplementedError, RuntimeError):
                    pass  # Windows or non-main thread

            # Start idle timeout checker
            idle_task = asyncio.create_task(self._idle_check_loop())
            self._set_startup(DaemonStartupPhase.LISTENING, str(sock_path))
            self._lifecycle.transition(LifecyclePhase.READY, reason='daemon listening')

            async with self._server:
                await self._stop_event.wait()
        except BaseException as error:
            await self._close_server_on_startup_failure()
            self._fail_startup(f'{type(error).__name__}: {error}', error)
            raise
        finally:
            self._running = False
            try:
                if idle_task is not None:
                    idle_task.cancel()
                    await asyncio.gather(idle_task, return_exceptions=True)
                if (
                    self._lifecycle.phase is not LifecyclePhase.FAILED
                    and self._lifecycle.phase is not LifecyclePhase.STOPPING
                ):
                    try:
                        self._lifecycle.transition(LifecyclePhase.STOPPING, reason='daemon stopping')
                    except LifecycleTransitionError:
                        pass
                if self._session:
                    await self._close_session_bounded(self._session)
            finally:
                try:
                    if acquired_pid:
                        try:
                            _cleanup_pid(
                                expected_pid=os.getpid(),
                                remove_socket=bound_socket,
                                socket_identity=self._bound_socket_identity,
                            )
                        except OSError:
                            logger.warning('Failed to remove daemon ownership files', exc_info=True)
                    if self._startup_phase is not DaemonStartupPhase.FAILED:
                        self._set_startup(DaemonStartupPhase.STOPPED, 'daemon stopped')
                    if self._lifecycle.phase not in (LifecyclePhase.FAILED, LifecyclePhase.STOPPED):
                        try:
                            self._lifecycle.transition(
                                LifecyclePhase.STOPPED,
                                reason='daemon stopped',
                                increment_generation=True,
                            )
                        except LifecycleTransitionError:
                            pass
                finally:
                    if owner_lock_acquired:
                        owner_lock.release()


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
    await daemon.serve_daemon()
    if daemon._startup_phase is DaemonStartupPhase.FAILED or daemon._lifecycle.phase is LifecyclePhase.FAILED:
        raise SystemExit(EXIT_OWNER_BUSY if daemon._startup_owner_busy is True else 1)


if __name__ == '__main__':
    asyncio.run(_main())
