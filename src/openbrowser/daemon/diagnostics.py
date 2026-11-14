"""Diagnostics for OpenBrowser daemon sockets and orphaned processes."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any

import psutil

from openbrowser.daemon import DAEMON_DIR, IS_WINDOWS, SOCKET_PATH, WINDOWS_PORT, get_socket_path

DAEMON_MODULE = 'openbrowser.daemon.server'


def is_daemon_cmdline(cmdline: Any) -> bool:
    """Return True only for an exact ``python -m openbrowser.daemon.server`` argv."""
    if not isinstance(cmdline, (list, tuple)):
        return False
    arguments = [str(part) for part in cmdline]
    return any(
        arguments[index] == '-m' and arguments[index + 1] == DAEMON_MODULE
        for index in range(len(arguments) - 1)
    )


def socket_candidates() -> list[Path]:
    """Return daemon sockets under the OpenBrowser directory plus the current socket."""
    sockets = {path for path in DAEMON_DIR.glob('*.sock')}
    sockets.add(get_socket_path())
    return sorted(sockets)


def _unix_socket_is_live(path: Path) -> bool:
    """Return False only when a Unix socket is provably absent or refused."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(0.2)
    try:
        client.connect(str(path))
        return True
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    except OSError:
        # A timeout or permission error does not prove the daemon is dead.
        return True
    finally:
        client.close()


def _windows_daemon_is_listening(pid: int) -> bool:
    """Return True when the process owns the daemon's loopback listener."""
    try:
        for connection in psutil.net_connections(kind='tcp'):
            if (
                connection.pid == pid
                and connection.status == psutil.CONN_LISTEN
                and connection.laddr
                and connection.laddr.port == WINDOWS_PORT
            ):
                return True
    except (psutil.AccessDenied, OSError):
        return True
    return False


def daemon_processes() -> list[dict[str, Any]]:
    """Return this user's OpenBrowser daemon processes and their socket context."""
    try:
        current_user = psutil.Process().username()
    except (psutil.Error, OSError):
        return []

    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(['pid', 'create_time', 'cmdline']):
        try:
            cmdline = process.info.get('cmdline') or []
            if not is_daemon_cmdline(cmdline):
                continue
            if process.username() != current_user:
                continue
            environment: dict[str, str] = {}
            socket_known = True
            try:
                environment = process.environ() or {}
            except (psutil.AccessDenied, OSError):
                socket_known = False

            if IS_WINDOWS:
                socket_path = None
                socket_exists = socket_known and _windows_daemon_is_listening(process.pid)
                socket_live = socket_exists
            elif socket_known:
                raw_socket = environment.get('OPENBROWSER_SOCKET')
                resolved_socket = Path(raw_socket).expanduser() if raw_socket else SOCKET_PATH
                if resolved_socket.is_absolute():
                    socket_path = str(resolved_socket)
                    socket_exists = resolved_socket.exists()
                    socket_live = socket_exists and _unix_socket_is_live(resolved_socket)
                else:
                    socket_known = False
                    socket_path = None
                    socket_exists = False
                    socket_live = True
            else:
                socket_path = None
                socket_exists = False
                socket_live = True
            created = float(process.info.get('create_time') or time.time())
            rows.append(
                {
                    'pid': process.pid,
                    'create_time': created,
                    'age_seconds': max(0.0, time.time() - created),
                    'socket': socket_path,
                    'socket_known': socket_known,
                    'socket_exists': socket_exists,
                    'socket_live': socket_live,
                    'cmdline': ' '.join(str(part) for part in cmdline),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return sorted(rows, key=lambda row: row['pid'])


def reap_orphan_processes(*, min_age_seconds: float = 30.0, timeout: float = 5.0) -> list[dict[str, Any]]:
    """Terminate this user's daemon processes that have no live socket."""
    results: list[dict[str, Any]] = []
    for row in daemon_processes():
        orphaned = row['socket_known'] and not row['socket_live']
        if not orphaned:
            results.append({**row, 'reaped': False, 'reason': 'socket is live or unknown'})
            continue
        if row['age_seconds'] < min_age_seconds:
            results.append({**row, 'reaped': False, 'reason': 'process is too young'})
            continue
        try:
            process = psutil.Process(row['pid'])
            if abs(float(process.create_time()) - float(row['create_time'])) > 1e-6:
                results.append({**row, 'reaped': False, 'reason': 'process identity changed'})
                continue
            cmdline_parts = process.cmdline()
            if not is_daemon_cmdline(cmdline_parts):
                results.append({**row, 'reaped': False, 'reason': 'process identity changed'})
                continue
            if row['socket'] and _unix_socket_is_live(Path(row['socket'])):
                results.append({**row, 'reaped': False, 'reason': 'socket is live or unknown'})
                continue
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
            results.append({**row, 'reaped': True, 'reason': 'orphaned'})
        except psutil.NoSuchProcess:
            results.append({**row, 'reaped': True, 'reason': 'already gone'})
        except Exception as error:
            results.append({**row, 'reaped': False, 'reason': f'{type(error).__name__}: {error}'})
    return results
