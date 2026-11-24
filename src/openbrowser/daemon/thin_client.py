"""Synchronous warm-daemon client for the ``-c`` CLI fast path.

The full daemon client owns daemon startup and uses asyncio. That import cost
is unnecessary when a daemon already listens on the socket. This module keeps
the warm path to the standard library and returns ``None`` when no daemon is
reachable so the caller can fall back to the full client.
"""

from __future__ import annotations

import json
import os
import socket
import time

from openbrowser.daemon import IS_WINDOWS, WINDOWS_PORT, get_socket_path

DEFAULT_CONNECT_TIMEOUT = 1.0
DEFAULT_READ_TIMEOUT = 140.0


class ThinResponse:
    """Fields the CLI prints from a daemon response."""

    __slots__ = ("error", "error_code", "hint", "output", "state", "success")

    def __init__(
        self,
        *,
        success: bool,
        output: str = "",
        error: str | None = None,
        error_code: str | None = None,
        hint: str | None = None,
        state: str | None = None,
    ) -> None:
        self.success = success
        self.output = output
        self.error = error
        self.error_code = error_code
        self.hint = hint
        self.state = state


def _positive_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        value = float(raw) if raw else default
    except ValueError:
        return default
    if value <= 0:
        return default
    return max(0.05, value)


def _read_line(sock: socket.socket, timeout: float) -> bytes | None:
    """Read one newline-terminated response, or None when the peer closes."""
    deadline = time.monotonic() + timeout
    buffer = bytearray()
    while not buffer.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        sock.settimeout(remaining)
        chunk = sock.recv(65536)
        if not chunk:
            return None if not buffer else bytes(buffer)
        buffer.extend(chunk)
    return bytes(buffer)


def execute_code_sync(code: str) -> ThinResponse | None:
    """Execute code on a warm daemon, or return None when none is reachable.

    A connection failure before the request is sent returns None so the caller
    can use the full client and start a daemon. Once the request is sent, every
    failure is a typed response because the code may have run.
    """
    if IS_WINDOWS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address: str | tuple[str, int] = ("127.0.0.1", WINDOWS_PORT)
    else:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = str(get_socket_path())

    sock.settimeout(_positive_seconds("OPENBROWSER_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT))
    try:
        sock.connect(address)
    except (ConnectionRefusedError, FileNotFoundError, TimeoutError, OSError):
        sock.close()
        return None

    read_timeout = _positive_seconds("OPENBROWSER_READ_TIMEOUT", DEFAULT_READ_TIMEOUT)
    request = json.dumps({"id": 1, "action": "execute", "code": code}).encode() + b"\n"
    try:
        sock.sendall(request)
        raw = _read_line(sock, read_timeout)
    except TimeoutError:
        return ThinResponse(
            success=False,
            error=f"Daemon did not respond within {read_timeout:g}s",
            error_code="daemon_timeout",
            hint="The code may still be running. Run `openbrowser-ai daemon status` before retrying.",
        )
    except (ConnectionResetError, OSError) as error:
        return ThinResponse(
            success=False,
            error=f"Daemon connection lost: {error}",
            error_code="daemon_connection_lost",
            hint="The code may have partially run. Verify the page before retrying.",
        )
    finally:
        sock.close()

    if raw is None:
        return ThinResponse(
            success=False,
            error="Daemon closed the connection without responding",
            error_code="daemon_connection_lost",
            hint="The code may have partially run. Verify the page before retrying.",
        )
    try:
        payload = json.loads(raw.decode())
    except (UnicodeDecodeError, ValueError) as error:
        return ThinResponse(
            success=False,
            error=f"Invalid daemon response: {error}",
            error_code="daemon_invalid_response",
        )
    if not isinstance(payload, dict):
        return ThinResponse(
            success=False,
            error="Daemon returned a non-object response",
            error_code="daemon_invalid_response",
        )
    return ThinResponse(
        success=bool(payload.get("success", False)),
        output=str(payload.get("output") or ""),
        error=payload.get("error"),
        error_code=payload.get("error_code"),
        hint=payload.get("hint"),
        state=payload.get("state"),
    )
