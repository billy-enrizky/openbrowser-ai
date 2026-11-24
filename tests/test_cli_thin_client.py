"""Warm-daemon thin client and CLI execute-code fast path."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


class OneShotServer:
    """Serve exactly one line-delimited JSON request on a Unix socket."""

    def __init__(self, response: object, *, raw: bytes | None = None, delay: float = 0.0, close: bool = False):
        self.response = response
        self.raw = raw
        self.delay = delay
        self.close = close
        self.request: dict | None = None
        self.error: BaseException | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(self.path)
                server.listen(1)
                server.settimeout(10)
                self._ready.set()
                connection, _ = server.accept()
                with connection:
                    data = b""
                    while not data.endswith(b"\n"):
                        chunk = connection.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    if data:
                        self.request = json.loads(data.decode())
                    if self.delay:
                        import time

                        time.sleep(self.delay)
                    if self.close:
                        return
                    payload = self.raw if self.raw is not None else json.dumps(self.response).encode() + b"\n"
                    # The client may time out or close first in the reset and
                    # timeout cases; that is the behavior under test.
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        connection.sendall(payload)
        except BaseException as error:  # surfaced by start()
            self.error = error
            self._ready.set()

    def start(self, path: Path) -> None:
        self.path = str(path)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("server did not start")
        if self.error is not None:
            raise self.error

    def join(self) -> None:
        self._thread.join(10)
        if self._thread.is_alive():
            raise RuntimeError("server did not stop")
        if self.error is not None:
            raise self.error


@pytest.fixture()
def socket_path(monkeypatch):
    import uuid

    path = Path("/tmp") / f"obt-{uuid.uuid4().hex[:8]}.sock"
    monkeypatch.setenv("OPENBROWSER_SOCKET", str(path))
    yield path
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def test_execute_code_sync_returns_success_response(socket_path):
    from openbrowser.daemon.thin_client import execute_code_sync

    server = OneShotServer({"id": 1, "success": True, "output": "42", "error": None})
    server.start(socket_path)
    response = execute_code_sync("print(42)")
    server.join()

    assert response is not None
    assert response.success is True
    assert response.output == "42"
    assert server.request == {"id": 1, "action": "execute", "code": "print(42)"}


def test_execute_code_sync_returns_none_without_daemon(socket_path):
    from openbrowser.daemon.thin_client import execute_code_sync

    assert execute_code_sync("print(1)") is None


def test_execute_code_sync_timeout_is_a_typed_error(socket_path, monkeypatch):
    from openbrowser.daemon.thin_client import execute_code_sync

    monkeypatch.setenv("OPENBROWSER_READ_TIMEOUT", "0.2")
    server = OneShotServer({"success": True, "output": "late", "error": None}, delay=1.0)
    server.start(socket_path)
    response = execute_code_sync("print(1)")
    server.join()

    assert response is not None
    assert response.success is False
    assert response.error_code == "daemon_timeout"
    assert "0.2" in (response.error or "")


def test_execute_code_sync_connection_lost_is_a_typed_error(socket_path):
    from openbrowser.daemon.thin_client import execute_code_sync

    server = OneShotServer({"success": True}, close=True)
    server.start(socket_path)
    response = execute_code_sync("print(1)")
    server.join()

    assert response is not None
    assert response.success is False
    assert response.error_code == "daemon_connection_lost"


def test_execute_code_sync_invalid_json_is_a_typed_error(socket_path):
    from openbrowser.daemon.thin_client import execute_code_sync

    server = OneShotServer({}, raw=b"not-json\n")
    server.start(socket_path)
    response = execute_code_sync("print(1)")
    server.join()

    assert response is not None
    assert response.success is False
    assert response.error_code == "daemon_invalid_response"


def test_execute_code_sync_reads_large_output(socket_path):
    from openbrowser.daemon.thin_client import execute_code_sync

    output = "x" * 200_000
    server = OneShotServer({"success": True, "output": output, "error": None})
    server.start(socket_path)
    response = execute_code_sync("print(1)")
    server.join()

    assert response is not None
    assert response.output == output


def _run_cli(args: list[str], *, env: dict[str, str], timeout: int = 30) -> subprocess.CompletedProcess:
    merged = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **env}
    return subprocess.run(
        [sys.executable, "-m", "openbrowser.cli_entry", *args],
        check=False,
        cwd=ROOT,
        env=merged,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_entry_executes_warm_code_without_loading_full_cli(socket_path):
    server = OneShotServer({"id": 1, "success": True, "output": "thin-path", "error": None})
    server.start(socket_path)
    result = _run_cli(
        ["-c", 'print("thin-path")'],
        env={
            "OPENBROWSER_SOCKET": str(socket_path),
            "OPENBROWSER_SETUP_LOGGING": "false",
            "PYTHONPROFILEIMPORTTIME": "1",
        },
    )
    server.join()

    assert result.returncode == 0, result.stderr
    assert "thin-path" in result.stdout
    # The fast path returns before the Click command tree is imported.
    assert "openbrowser.cli\n" not in result.stderr
    assert server.request is not None
    assert server.request.get("code") == 'print("thin-path")'


def test_cli_entry_falls_back_when_daemon_is_absent(tmp_path, monkeypatch):
    # No server exists at this socket. The normal CLI path must run and start a
    # daemon, so the observed failure comes from cold startup, not from a
    # silently dropped request.
    socket_path = tmp_path / "missing.sock"
    result = _run_cli(
        ["-c", "print(1)"],
        env={
            "OPENBROWSER_SOCKET": str(socket_path),
            "OPENBROWSER_SETUP_LOGGING": "false",
            "OPENBROWSER_DAEMON_START_TIMEOUT": "0.1",
        },
        timeout=60,
    )
    assert result.returncode != 0
    assert "daemon" in (result.stderr + result.stdout).lower()
