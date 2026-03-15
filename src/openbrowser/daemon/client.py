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
import subprocess
import sys
import time
from dataclasses import dataclass

from openbrowser.daemon import DAEMON_DIR, IS_WINDOWS, WINDOWS_PORT, get_socket_path

CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 300.0  # 5 min for long-running code
DAEMON_START_TIMEOUT = 15.0


@dataclass
class DaemonResponse:
    success: bool
    output: str
    error: str | None


class DaemonClient:
    """Client that communicates with the OpenBrowser daemon."""

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

    async def _start_daemon(self):
        """Spawn the daemon process in the background."""
        get_socket_path().parent.mkdir(parents=True, exist_ok=True)
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)
        log_file = DAEMON_DIR / 'daemon.log'
        with open(os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as log_handle:
            subprocess.Popen(
                [sys.executable, '-m', 'openbrowser.daemon.server'],
                stdout=subprocess.DEVNULL,
                stderr=log_handle,
                start_new_session=True,
            )
        # Wait for socket to appear
        sock = get_socket_path()
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
            if not raw:
                raise ConnectionResetError('Daemon closed the connection without responding')
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
        except (ConnectionRefusedError, FileNotFoundError, ConnectionResetError, OSError):
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
            return DaemonResponse(
                success=resp.get('success', False),
                output=resp.get('output', ''),
                error=resp.get('error'),
            )
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(success=False, output='', error='Daemon not running')

    async def stop(self) -> DaemonResponse:
        try:
            resp = await self._send({'id': 1, 'action': 'stop'})
            return DaemonResponse(
                success=resp.get('success', False),
                output=resp.get('output', 'Daemon stopped'),
                error=resp.get('error'),
            )
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(success=False, output='', error='Daemon not running')

    async def reset(self) -> DaemonResponse:
        try:
            resp = await self._send({'id': 1, 'action': 'reset'})
            return DaemonResponse(
                success=resp.get('success', False),
                output=resp.get('output', ''),
                error=resp.get('error'),
            )
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return DaemonResponse(success=False, output='', error='Daemon not running')


async def execute_code_via_daemon(code: str) -> DaemonResponse:
    """Convenience function for CLI usage."""
    client = DaemonClient()
    return await client.execute(code)
