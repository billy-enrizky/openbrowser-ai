# src/openbrowser/daemon/client.py
"""Thin client for the OpenBrowser daemon.

Connects to the daemon Unix socket, sends code, returns output.
Auto-starts the daemon if not running.

This module intentionally avoids importing any heavy openbrowser
modules so the -c CLI path stays fast (<50ms import time).
"""

import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DAEMON_DIR = Path.home() / '.openbrowser'
SOCKET_PATH = DAEMON_DIR / 'daemon.sock'
IS_WINDOWS = platform.system() == 'Windows'
WINDOWS_PORT = 19222

CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 300.0  # 5 min for long-running code
DAEMON_START_TIMEOUT = 15.0


@dataclass
class DaemonResponse:
    success: bool
    output: str
    error: str | None


def _get_socket_path() -> Path:
    return Path(os.environ.get('OPENBROWSER_SOCKET', str(SOCKET_PATH)))


def _daemon_is_running() -> bool:
    pid_path = DAEMON_DIR / 'daemon.pid'
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        return False


class DaemonClient:
    """Client that communicates with the OpenBrowser daemon."""

    async def _connect(self):
        sock = _get_socket_path()
        if IS_WINDOWS:
            return await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', WINDOWS_PORT),
                timeout=CONNECT_TIMEOUT,
            )
        return await asyncio.wait_for(
            asyncio.open_unix_connection(str(sock)),
            timeout=CONNECT_TIMEOUT,
        )

    async def _start_daemon(self):
        """Spawn the daemon process in the background."""
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)
        log_file = DAEMON_DIR / 'daemon.log'
        subprocess.Popen(
            [sys.executable, '-m', 'openbrowser.daemon.server'],
            stdout=subprocess.DEVNULL,
            stderr=open(log_file, 'w'),
            start_new_session=True,
        )
        # Wait for socket to appear
        sock = _get_socket_path()
        deadline = time.time() + DAEMON_START_TIMEOUT
        while time.time() < deadline:
            if IS_WINDOWS or sock.exists():
                try:
                    reader, writer = await self._connect()
                    writer.close()
                    await writer.wait_closed()
                    return
                except (ConnectionRefusedError, FileNotFoundError, OSError):
                    pass
            await asyncio.sleep(0.2)
        raise TimeoutError('Daemon did not start within timeout')

    async def _send(self, request: dict) -> dict:
        """Send a request and return the response."""
        reader, writer = await self._connect()
        try:
            writer.write(json.dumps(request).encode() + b'\n')
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), timeout=READ_TIMEOUT)
            return json.loads(raw.decode())
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def execute(self, code: str) -> DaemonResponse:
        """Execute code via the daemon. Auto-starts if needed."""
        try:
            resp = await self._send({'id': 1, 'action': 'execute', 'code': code})
        except (ConnectionRefusedError, FileNotFoundError, ConnectionResetError):
            await self._start_daemon()
            resp = await self._send({'id': 1, 'action': 'execute', 'code': code})

        return DaemonResponse(
            success=resp.get('success', False),
            output=resp.get('output', ''),
            error=resp.get('error'),
        )

    async def status(self) -> DaemonResponse:
        try:
            resp = await self._send({'id': 1, 'action': 'status'})
            return DaemonResponse(success=True, output=resp.get('output', ''), error=None)
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(success=False, output='', error='Daemon not running')

    async def stop(self) -> DaemonResponse:
        try:
            resp = await self._send({'id': 1, 'action': 'stop'})
            return DaemonResponse(success=True, output='Daemon stopped', error=None)
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(success=False, output='', error='Daemon not running')

    async def reset(self) -> DaemonResponse:
        try:
            resp = await self._send({'id': 1, 'action': 'reset'})
            return DaemonResponse(success=True, output=resp.get('output', ''), error=None)
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(success=False, output='', error='Daemon not running')


async def execute_code_via_daemon(code: str) -> DaemonResponse:
    """Convenience function for CLI usage."""
    client = DaemonClient()
    return await client.execute(code)
