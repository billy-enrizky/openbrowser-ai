"""Bounded loopback-only discovery of Chromium's ``/json/version`` payload."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
from collections.abc import Awaitable, Callable
from typing import Any

MAX_HEADER_BYTES = 16 * 1024
MAX_BODY_BYTES = 64 * 1024
DEFAULT_TIMEOUT = 2.0
_LOOPBACK_ADDRESS = "127.0.0.1"
_DISCOVERY_HOST = "localhost"
_READ_CHUNK = 4096
_HEADER_TERMINATOR = b"\r\n\r\n"
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")

type Connector = Callable[
    [str, int],
    Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]],
]


class LocalDiscoveryError(RuntimeError):
    """Raised when a loopback Chromium version response is incomplete or invalid."""


def _request(port: int) -> bytes:
    return (
        f"GET /json/version HTTP/1.1\r\n"
        f"Host: {_DISCOVERY_HOST}:{port}\r\n"
        "Accept: application/json\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")


async def _read_headers(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    data = bytearray()
    while True:
        chunk = await reader.read(_READ_CHUNK)
        if not chunk:
            raise LocalDiscoveryError("loopback discovery closed before HTTP headers completed")
        data.extend(chunk)
        marker = data.find(_HEADER_TERMINATOR)
        if marker >= 0:
            end = marker + len(_HEADER_TERMINATOR)
            if end > MAX_HEADER_BYTES:
                raise LocalDiscoveryError("loopback discovery headers exceed the size limit")
            return bytes(data[:end]), bytes(data[end:])
        if len(data) > MAX_HEADER_BYTES:
            raise LocalDiscoveryError("loopback discovery headers exceed the size limit")


def _parse_headers(head: bytes) -> int:
    try:
        text = head[: -len(_HEADER_TERMINATOR)].decode("latin-1")
    except UnicodeDecodeError as error:
        raise LocalDiscoveryError("loopback discovery headers are not valid Latin-1") from error
    lines = text.split("\r\n")
    if not lines:
        raise LocalDiscoveryError("loopback discovery response has no status line")
    status = lines[0].split(" ", 2)
    if len(status) < 2 or status[0] != "HTTP/1.1" or status[1] != "200":
        raise LocalDiscoveryError("loopback discovery did not return HTTP/1.1 200")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, separator, value = line.partition(":")
        if not separator or _HEADER_NAME.fullmatch(name) is None:
            raise LocalDiscoveryError("loopback discovery header is malformed")
        key = name.lower()
        if key in headers:
            raise LocalDiscoveryError("loopback discovery header is duplicated")
        headers[key] = value.strip()
    if "transfer-encoding" in headers:
        raise LocalDiscoveryError("loopback discovery transfer encoding is unsupported")
    content_length = headers.get("content-length")
    if content_length is None or not content_length.isascii() or not content_length.isdecimal():
        raise LocalDiscoveryError("loopback discovery content length is missing or malformed")
    length = int(content_length)
    if length > MAX_BODY_BYTES:
        raise LocalDiscoveryError("loopback discovery body exceeds the size limit")
    return length


async def _read_body(reader: asyncio.StreamReader, prefix: bytes, length: int) -> bytes:
    body = bytearray(prefix[:length])
    while len(body) < length:
        chunk = await reader.read(min(_READ_CHUNK, length - len(body)))
        if not chunk:
            raise LocalDiscoveryError("loopback discovery closed before the HTTP body completed")
        body.extend(chunk[: length - len(body)])
    return bytes(body)


def _decode_version(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LocalDiscoveryError("loopback discovery response is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise LocalDiscoveryError("loopback discovery response is not a JSON object")
    return value


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(OSError, RuntimeError):
        writer.close()
    with contextlib.suppress(OSError, RuntimeError):
        await writer.wait_closed()


async def fetch_loopback_version(
    port: int,
    *,
    timeout: float = DEFAULT_TIMEOUT,  # noqa: ASYNC109
    open_connection: Connector | None = None,
) -> dict[str, Any]:
    """Fetch one bounded Chromium ``/json/version`` response from loopback.

    The TCP connection uses the numeric IPv4 loopback address while the HTTP
    ``Host`` header remains ``localhost:{port}``. That keeps Chromium's returned
    WebSocket host stable without adding a DNS lookup. ``open_connection`` is
    injectable for fake-only tests.
    """
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise TimeoutError("loopback discovery timeout must be finite and positive")

    connector = open_connection or asyncio.open_connection
    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(timeout):
            reader, writer = await connector(_LOOPBACK_ADDRESS, port)
            writer.write(_request(port))
            await writer.drain()
            head, prefix = await _read_headers(reader)
            length = _parse_headers(head)
            body = await _read_body(reader, prefix, length)
            return _decode_version(body)
    except TimeoutError:
        raise
    except asyncio.CancelledError:
        raise
    except LocalDiscoveryError:
        raise
    except (OSError, asyncio.IncompleteReadError, ValueError) as error:
        raise LocalDiscoveryError("loopback discovery request failed") from error
    finally:
        if writer is not None:
            await _close_writer(writer)


__all__ = [
    "DEFAULT_TIMEOUT",
    "MAX_BODY_BYTES",
    "MAX_HEADER_BYTES",
    "LocalDiscoveryError",
    "fetch_loopback_version",
]
