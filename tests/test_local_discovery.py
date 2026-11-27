"""Fake-only tests for bounded local Chromium version discovery."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from openbrowser.browser import local_discovery

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/openbrowser/browser/local_discovery.py"
URL = "ws://localhost:9222/devtools/browser/test"
BODY = json.dumps({"webSocketDebuggerUrl": URL}).encode()
REQUEST = b"GET /json/version HTTP/1.1\r\nHost: localhost:9222\r\nAccept: application/json\r\nConnection: close\r\n\r\n"


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local discovery tests must not use native I/O")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "listen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(asyncio, "open_connection", forbidden)


class FakeReader:
    def __init__(self, chunks, *, block_after=False):
        self.chunks = list(chunks)
        self.block_after = block_after
        self.reads = []

    async def read(self, size=-1):
        self.reads.append(size)
        if self.chunks:
            value = self.chunks.pop(0)
            if isinstance(value, BaseException):
                raise value
            if value is None:
                await asyncio.sleep(10)
                return b""
            return value
        if self.block_after:
            await asyncio.sleep(10)
        return b""


class FakeWriter:
    def __init__(self, *, drain_error=None, close_error=None, wait_error=None, wait_block=False):
        self.buffer = bytearray()
        self.drain_error = drain_error
        self.close_error = close_error
        self.wait_error = wait_error
        self.wait_block = wait_block
        self.wait_started: asyncio.Event | None = None
        self.closed = False
        self.close_calls = 0
        self.wait_calls = 0

    def write(self, data):
        self.buffer.extend(data)

    async def drain(self):
        if self.drain_error is not None:
            raise self.drain_error

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self.close_error is not None:
            raise self.close_error

    async def wait_closed(self):
        self.wait_calls += 1
        if self.wait_started is not None:
            self.wait_started.set()
        if self.wait_block:
            await asyncio.Event().wait()
        if self.wait_error is not None:
            raise self.wait_error


def response(body=BODY, *, status="HTTP/1.1 200 OK", extra_headers=(), content_length=None):
    length = len(body) if content_length is None else content_length
    headers = [status, "Content-Type: application/json; charset=UTF-8", f"Content-Length: {length}"]
    headers.extend(extra_headers)
    return ("\r\n".join(headers) + "\r\n\r\n").encode("latin-1") + body


def fetch_coro(*, chunks, timeout=1.0, writer=None, connector=None):
    state = {"calls": []}
    fake_writer = writer or FakeWriter()
    fake_reader = FakeReader(chunks)

    async def default_connector(host, port):
        state["calls"].append((host, port))
        return fake_reader, fake_writer

    async def runner():
        result = await local_discovery.fetch_loopback_version(
            9222,
            timeout=timeout,
            open_connection=cast(local_discovery.Connector, connector or default_connector),
        )
        return result

    return runner(), fake_reader, fake_writer, state


def run_fetch(*, chunks, timeout=1.0, writer=None, connector=None):
    coro, fake_reader, fake_writer, state = fetch_coro(
        chunks=chunks, timeout=timeout, writer=writer, connector=connector
    )
    return asyncio.run(coro), fake_reader, fake_writer, state


async def bounded(coro, *, bound=0.5):
    task = asyncio.create_task(coro)
    _done, pending = await asyncio.wait({task}, timeout=bound)
    if pending:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise AssertionError("helper exceeded the cleanup bound")
    return task.result()


def test_exact_request_and_valid_response():
    result, _reader, writer, state = run_fetch(chunks=[response()])
    assert result == {"webSocketDebuggerUrl": URL}
    assert bytes(writer.buffer) == REQUEST
    assert state["calls"] == [("127.0.0.1", 9222)]
    assert writer.close_calls == 1
    assert writer.wait_calls == 1
    assert writer.closed is True


@pytest.mark.parametrize("split", range(1, len(response())))
def test_fragmented_response_at_every_byte_boundary(split):
    payload = response()
    result, _reader, writer, state = run_fetch(chunks=[payload[:split], payload[split:]])
    assert result == {"webSocketDebuggerUrl": URL}
    assert state["calls"] == [("127.0.0.1", 9222)]
    assert writer.close_calls == 1


def test_headers_are_case_insensitive_and_extra_headers_are_ignored():
    payload = response(
        extra_headers=("X-Test: yes", "CONNECTION: close"),
        status="HTTP/1.1 200 OK",
    )
    result, _, writer, _ = run_fetch(chunks=[payload])
    assert result["webSocketDebuggerUrl"] == URL
    assert writer.closed is True


@pytest.mark.parametrize(
    "payload, error_type",
    [
        (response(status="HTTP/1.1 503 Service Unavailable"), local_discovery.LocalDiscoveryError),
        (response(status="HTTP/1.0 200 OK"), local_discovery.LocalDiscoveryError),
        (response(status="NOTHTTP 200 OK"), local_discovery.LocalDiscoveryError),
        (b"", local_discovery.LocalDiscoveryError),
        (
            response(content_length=None).replace(b"Content-Length: " + str(len(BODY)).encode(), b"Content-Length: 4"),
            local_discovery.LocalDiscoveryError,
        ),
        (response(body=b"{}", content_length=9), local_discovery.LocalDiscoveryError),
        (response(body=b"{}", extra_headers=("Transfer-Encoding: chunked",)), local_discovery.LocalDiscoveryError),
        (response(body=b"{}", content_length=-1), local_discovery.LocalDiscoveryError),
        (response(body=b"{}", content_length="+2"), local_discovery.LocalDiscoveryError),
        (response(body=b"{}", content_length=local_discovery.MAX_BODY_BYTES + 1), local_discovery.LocalDiscoveryError),
        (response(body=b"{", content_length=1), local_discovery.LocalDiscoveryError),
        (response(body=b"\xff", content_length=1), local_discovery.LocalDiscoveryError),
        (response(body=b"[]", content_length=2), local_discovery.LocalDiscoveryError),
    ],
)
def test_malformed_framing_and_payloads_fail_closed(payload, error_type):
    with pytest.raises(error_type):
        run_fetch(chunks=[payload])


def test_missing_and_duplicate_content_length_fail_closed():
    body = b"{}"
    for headers in (
        ("Content-Type: application/json",),
        ("Content-Length: 2", "content-length: 2"),
    ):
        payload = b"HTTP/1.1 200 OK\r\n" + b"\r\n".join(h.encode() for h in headers) + b"\r\n\r\n" + body
        with pytest.raises(local_discovery.LocalDiscoveryError):
            run_fetch(chunks=[payload])


def test_header_and_body_caps_are_enforced():
    too_large_headers = b"HTTP/1.1 200 OK\r\nX-Fill: " + b"a" * local_discovery.MAX_HEADER_BYTES + b"\r\n\r\n"
    with pytest.raises(local_discovery.LocalDiscoveryError):
        run_fetch(chunks=[too_large_headers])

    too_large_body = response(body=b"{}", content_length=local_discovery.MAX_BODY_BYTES + 1)
    with pytest.raises(local_discovery.LocalDiscoveryError):
        run_fetch(chunks=[too_large_body])


def test_short_body_and_eof_fail_closed():
    header = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n"
    with pytest.raises(local_discovery.LocalDiscoveryError):
        run_fetch(chunks=[header, b"{}"])


def test_connection_and_write_failures_fail_closed_and_close_writer():
    async def bad_connection(host, port):
        raise OSError("connect failed")

    with pytest.raises(local_discovery.LocalDiscoveryError):
        run_fetch(chunks=[], connector=bad_connection)

    writer = FakeWriter(drain_error=OSError("write failed"))
    with pytest.raises(local_discovery.LocalDiscoveryError):
        run_fetch(chunks=[response()], writer=writer)
    assert writer.close_calls == 1


def test_timeout_before_connect_and_drain_timeout():
    with pytest.raises(TimeoutError):
        run_fetch(chunks=[], timeout=0)

    writer = FakeWriter(drain_error=None)

    async def slow_drain():
        await asyncio.sleep(10)

    writer.drain = slow_drain
    with pytest.raises(TimeoutError):
        run_fetch(chunks=[response()], timeout=0.01, writer=writer)
    assert writer.close_calls == 1


def test_blocked_close_after_success_is_bounded():
    writer = FakeWriter(wait_block=True)
    coro, _reader, _writer, _state = fetch_coro(chunks=[response()], timeout=0.05, writer=writer)
    result = asyncio.run(bounded(coro))
    assert result["webSocketDebuggerUrl"] == URL
    assert writer.close_calls == 1
    assert writer.wait_calls == 1


def test_blocked_close_does_not_mask_timeout():
    writer = FakeWriter(wait_block=True)

    class SlowReader:
        async def read(self, size=-1):
            if not hasattr(self, "sent"):
                self.sent = True
                return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n"
            await asyncio.sleep(10)
            return b""

    async def connector(host, port):
        return SlowReader(), writer

    async def scenario():
        coro, _reader, _writer, _state = fetch_coro(chunks=[], timeout=0.02, writer=writer, connector=connector)
        with pytest.raises(TimeoutError):
            await coro
        return True

    assert asyncio.run(bounded(scenario())) is True
    assert writer.close_calls == 1


def test_blocked_close_does_not_mask_cancellation():
    writer = FakeWriter(wait_block=True)
    chunks = [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n", asyncio.CancelledError()]

    async def scenario():
        coro, _reader, _writer, _state = fetch_coro(chunks=chunks, timeout=0.05, writer=writer)
        with pytest.raises(asyncio.CancelledError):
            await coro
        return True

    assert asyncio.run(bounded(scenario())) is True
    assert writer.close_calls == 1


def test_second_cancellation_during_blocked_close_propagates():
    async def scenario():
        writer = FakeWriter(wait_block=True)
        writer.wait_started = asyncio.Event()
        coro, _reader, _writer, _state = fetch_coro(chunks=[response()], timeout=5, writer=writer)
        task = asyncio.create_task(coro)
        assert writer.wait_started is not None
        await writer.wait_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert writer.close_calls == 1
        return True

    assert asyncio.run(bounded(scenario())) is True


def test_body_timeout_closes_writer():
    writer = FakeWriter()

    class SlowReader:
        async def read(self, size=-1):
            if not hasattr(self, "sent"):
                self.sent = True
                return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n"
            await asyncio.sleep(10)
            return b""

    async def connector(host, port):
        return SlowReader(), writer

    with pytest.raises(TimeoutError):
        run_fetch(chunks=[], timeout=0.01, writer=writer, connector=connector)
    assert writer.close_calls == 1


@pytest.mark.parametrize("stage", ["connect", "drain", "body", "close"])
def test_cancellation_propagates_and_closes_writer(stage):
    writer = FakeWriter(close_error=asyncio.CancelledError() if stage == "close" else None)
    if stage == "drain":
        writer.drain_error = asyncio.CancelledError()
    chunks = (
        [response()] if stage != "body" else [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n", asyncio.CancelledError()]
    )

    async def cancelled_connector(host, port):
        raise asyncio.CancelledError()

    connector = cancelled_connector if stage == "connect" else None
    with pytest.raises(asyncio.CancelledError):
        run_fetch(chunks=chunks, writer=writer, connector=connector)
    if stage != "connect":
        assert writer.close_calls == 1


def test_cleanup_errors_do_not_mask_success():
    writer = FakeWriter(close_error=OSError("close failed"), wait_error=OSError("wait failed"))
    result, _, _, _ = run_fetch(chunks=[response()], writer=writer)
    assert result["webSocketDebuggerUrl"] == URL
    assert writer.close_calls == 1


@pytest.mark.parametrize("port", [0, 65536, True, "9222"])
def test_invalid_port_is_rejected_without_connecting(port):
    async def connector(host, port):
        pytest.fail("connector must not run for an invalid port")

    with pytest.raises((TypeError, ValueError)):
        asyncio.run(local_discovery.fetch_loopback_version(port, open_connection=connector))


def test_source_parses_as_python_311():
    ast.parse(MODULE_PATH.read_text(), filename=str(MODULE_PATH), feature_version=(3, 11))


def test_inert_import_does_not_import_httpx():
    script = """
import importlib.util
import sys
from pathlib import Path
path = Path(r"__MODULE_PATH__")
spec = importlib.util.spec_from_file_location("local_discovery_probe", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "httpx" not in sys.modules
assert module.MAX_HEADER_BYTES == 16384
assert module.MAX_BODY_BYTES == 65536
"""
    script = script.replace("__MODULE_PATH__", str(MODULE_PATH))
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
