# src/openbrowser/daemon/__init__.py
"""Shared constants for daemon server and client."""

import os
import platform
from pathlib import Path

DAEMON_DIR = Path.home() / '.openbrowser'
SOCKET_PATH = DAEMON_DIR / 'daemon.sock'
PID_PATH = DAEMON_DIR / 'daemon.pid'

IS_WINDOWS = platform.system() == 'Windows'
WINDOWS_PORT = 19222


def get_socket_path() -> Path:
    """Return the daemon socket path, respecting OPENBROWSER_SOCKET env var."""
    return Path(os.environ.get('OPENBROWSER_SOCKET', str(SOCKET_PATH)))
