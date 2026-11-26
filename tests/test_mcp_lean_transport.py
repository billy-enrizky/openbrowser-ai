"""Behavior tests for the MCP stdio transport that does not import the SDK."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil
import pytest
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).parents[1]
LEAN_FIXTURE = ROOT / "tests" / "support" / "mcp_lean_protocol_server.py"
META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {"name": "lean-test", "version": "1"},
}


def child_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "PYTHONPATH": str(ROOT / "src"),
        "OPENBROWSER_ANONYMIZED_TELEMETRY": "false",
        "ANONYMIZED_TELEMETRY": "false",
        "OPENBROWSER_MCP_TRANSPORT": "lean",
        "OPENBROWSER_SETUP_LOGGING": "false",
    }


class LeanWire:
    def __init__(self, process):
        self.process = process
        self.stderr_task = asyncio.create_task(process.stderr.read())

    async def request(self, ident, method, params=None, *, modern=True):
        await self.send(ident, method, params, modern=modern)
        return await self.receive(ident)

    async def send(self, ident, method, params=None, *, modern=True):
        message = {"jsonrpc": "2.0", "id": ident, "method": method}
        payload = dict(params or {})
        if modern:
            payload["_meta"] = META.copy()
        message["params"] = payload
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def send_envelope(self, message: dict):
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def receive(self, ident):
        raw = await asyncio.wait_for(self.process.stdout.readline(), timeout=10)
        assert raw, "lean MCP server closed stdout before replying"
        response = json.loads(raw)
        assert response["id"] == ident, response
        assert "error" not in response, response
        return response["result"]

    async def notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method, "params": dict(params or {})}
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def finish(self):
        if self.process.stdin and not self.process.stdin.is_closing():
            self.process.stdin.close()
        await asyncio.wait_for(self.process.wait(), timeout=10)
        stderr = (await self.stderr_task).decode(errors="replace")
        assert self.process.returncode == 0, stderr
        return stderr


@asynccontextmanager
async def lean_wire(home: Path, *, fixture: Path | None = None, extra_env: dict[str, str] | None = None):
    args = ["-m", "openbrowser.mcp.lean"]
    if fixture is not None:
        args = [str(fixture)]
    environment = child_env(home)
    environment.update(extra_env or {})
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        *args,
        cwd=ROOT,
        env=environment,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    connection = LeanWire(process)
    try:
        yield connection
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await connection.stderr_task


def test_lean_module_does_not_import_sdk_or_server_at_import_time():
    source = (
        "import sys\n"
        "import openbrowser.mcp.lean\n"
        "print('sdk', 'mcp' in sys.modules)\n"
        "print('server', 'openbrowser.mcp.server' in sys.modules)\n"
    )
    result = __import__("subprocess").run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "sdk False" in result.stdout, result.stdout
    assert "server False" in result.stdout, result.stdout


def test_invalid_transport_value_exits_with_usage_error(tmp_path):
    env = child_env(tmp_path)
    env["OPENBROWSER_MCP_TRANSPORT"] = "invalid"
    result = __import__("subprocess").run(
        [sys.executable, "-m", "openbrowser.mcp"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "OPENBROWSER_MCP_TRANSPORT" in result.stderr


@pytest.mark.asyncio
async def test_lean_transport_modern_catalog_and_context_lifecycle(tmp_path):
    async with lean_wire(tmp_path) as c:
        discover = await c.request(1, "server/discover")
        assert discover["supportedVersions"] == ["2026-07-28"]
        assert discover["capabilities"]["tools"]["listChanged"] is False
        catalog = await c.request(2, "tools/list")
        assert [tool["name"] for tool in catalog["tools"]] == [
            "create_context",
            "execute_code",
            "get_status",
            "reset_browser",
            "close_context",
            "web_search",
            "scrape_web",
        ]
        assert catalog["resultType"] == "complete"
        assert catalog["ttlMs"] == 60000
        created = await c.request(3, "tools/call", {"name": "create_context", "arguments": {}})
        context_id = created["structuredContent"]["context_id"]
        status = await c.request(
            4,
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": context_id}},
        )
        status_payload = json.loads(status["content"][0]["text"])
        assert status_payload["browser"]["phase"] == "stopped"
        assert status_payload["server"]["phase"] == "ready"
        closed = await c.request(
            5,
            "tools/call",
            {"name": "close_context", "arguments": {"context_id": context_id}},
        )
        assert closed["structuredContent"]["status"] == "closed"
        await c.finish()


@pytest.mark.asyncio
async def test_lean_transport_legacy_wire_shape_and_era_gates(tmp_path):
    async with lean_wire(tmp_path) as c:
        await c.send(1, "tools/list", modern=False)
        preinit = await asyncio.wait_for(c.process.stdout.readline(), timeout=10)
        assert json.loads(preinit)["error"]["code"] == -32600

        initialized = await c.request(
            2,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "legacy-lean", "version": "1"},
            },
            modern=False,
        )
        assert initialized["protocolVersion"] == "2024-11-05"
        assert "resultType" not in initialized
        await c.send_envelope({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

        catalog = await c.request(3, "tools/list", modern=False)
        assert len(catalog["tools"]) == 4
        assert not {"ttlMs", "cacheScope", "resultType"} & catalog.keys()

        status = await c.request(4, "tools/call", {"name": "get_status", "arguments": {}}, modern=False)
        assert not {"resultType", "_meta"} & status.keys()
        assert json.loads(status["content"][0]["text"])["server"]["phase"] == "ready"

        await c.send(
            5,
            "tools/list",
        )
        mixed = await asyncio.wait_for(c.process.stdout.readline(), timeout=10)
        assert json.loads(mixed)["error"]["code"] == -32600
        await c.finish()


@pytest.mark.asyncio
async def test_lean_transport_modern_envelope_and_version_errors(tmp_path):
    async with lean_wire(tmp_path) as c:
        await c.send_envelope(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}},
            }
        )
        missing_capabilities = await asyncio.wait_for(c.process.stdout.readline(), timeout=10)
        assert json.loads(missing_capabilities)["error"]["code"] == -32602

        unsupported = {
            **META,
            "io.modelcontextprotocol/protocolVersion": "9999-01-01",
        }
        await c.send(2, "tools/list", {"_meta": unsupported}, modern=False)
        unsupported_response = await asyncio.wait_for(c.process.stdout.readline(), timeout=10)
        assert json.loads(unsupported_response)["error"]["code"] == -32022

        await c.send(3, "server/discover")
        assert (await c.receive(3))["supportedVersions"] == ["2026-07-28"]
        await c.send(
            4,
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "late-legacy", "version": "1"},
            },
            modern=False,
        )
        late_initialize = await asyncio.wait_for(c.process.stdout.readline(), timeout=10)
        assert json.loads(late_initialize)["error"]["code"] == -32022
        await c.finish()


def test_generated_catalog_matches_the_live_server_catalog():
    from openbrowser.code_use.descriptions import (
        EXECUTE_CODE_DESCRIPTION,
        EXECUTE_CODE_DESCRIPTION_COMPACT,
    )
    from openbrowser.mcp.catalog import load_tool_definitions
    from openbrowser.mcp.server import _tool_catalog

    for compact, description in (
        (False, EXECUTE_CODE_DESCRIPTION),
        (True, EXECUTE_CODE_DESCRIPTION_COMPACT),
    ):
        for modern in (False, True):
            generated = load_tool_definitions(modern=modern, compact=compact)
            live = [
                tool.model_dump(by_alias=True, exclude_none=True, mode="json")
                for tool in _tool_catalog(description, modern=modern)
            ]
            assert {tool["name"]: tool for tool in generated} == {tool["name"]: tool for tool in live}


class _StaticPageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<html><head><title>Static title</title></head><body><p>Static body</p></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):
        return


@pytest.mark.asyncio
async def test_lean_scrape_web_never_starts_a_browser(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StaticPageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        async with lean_wire(tmp_path) as c:
            result = await c.request(
                1,
                "tools/call",
                {"name": "scrape_web", "arguments": {"url": url, "format": "text"}},
            )
            payload = json.loads(result["content"][0]["text"])
            assert payload["ok"] is True
            assert payload["engine_used"] == "http"
            assert "Static body" in payload["content"]
            assert psutil.Process(c.process.pid).children(recursive=True) == []
            await c.finish()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_latest_sdk_client_uses_the_lean_transport(tmp_path, mode):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "openbrowser.mcp"],
        cwd=ROOT,
        env=child_env(tmp_path),
    )
    async with Client(params, mode=mode) as client:
        catalog = await client.list_tools()
        names = [tool.name for tool in catalog.tools]
        if mode == "legacy":
            assert names == ["execute_code", "get_status", "reset_browser", "web_search"]
            status = await client.call_tool("get_status", {})
            assert not status.is_error
            assert json.loads(status.content[0].text)["browser"]["phase"] == "stopped"
        else:
            assert names == [
                "create_context",
                "execute_code",
                "get_status",
                "reset_browser",
                "close_context",
                "web_search",
                "scrape_web",
            ]
            created = await client.call_tool("create_context", {})
            context_id = created.structured_content["context_id"]
            status = await client.call_tool("get_status", {"context_id": context_id})
            assert not status.is_error
            assert json.loads(status.content[0].text)["browser"]["phase"] == "stopped"
            closed = await client.call_tool("close_context", {"context_id": context_id})
            assert closed.structured_content["status"] == "closed"


@pytest.mark.asyncio
async def test_lean_transport_executes_and_cancels_owned_work(tmp_path):
    entered = tmp_path / "entered"
    unwound = tmp_path / "unwound"
    async with lean_wire(
        tmp_path,
        fixture=LEAN_FIXTURE,
        extra_env={
            "OPENBROWSER_TEST_CANCEL_ENTERED": str(entered),
            "OPENBROWSER_TEST_CANCEL_UNWOUND": str(unwound),
        },
    ) as c:
        created = await c.request(1, "tools/call", {"name": "create_context", "arguments": {}})
        context_id = created["structuredContent"]["context_id"]
        first = await c.request(
            2,
            "tools/call",
            {"name": "execute_code", "arguments": {"context_id": context_id, "code": "value = 41\nprint(value)"}},
        )
        assert first["content"][0]["text"].strip() == "41"
        second = await c.request(
            3,
            "tools/call",
            {"name": "execute_code", "arguments": {"context_id": context_id, "code": "print(value + 1)"}},
        )
        assert second["content"][0]["text"].strip() == "42"

        await c.send(
            4,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"context_id": context_id, "code": "await wait_for_cancel()"},
            },
        )
        for _ in range(200):
            if entered.exists():
                break
            await asyncio.sleep(0.01)
        assert entered.exists()
        await c.notify("notifications/cancelled", {"requestId": 4})
        for _ in range(200):
            if unwound.exists():
                break
            await asyncio.sleep(0.01)
        assert unwound.exists()
        status = await c.request(
            5,
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": context_id}},
        )
        assert json.loads(status["content"][0]["text"])["browser"]["phase"] == "stopped"
        await c.finish()


@pytest.mark.asyncio
async def test_lean_transport_returns_errors_without_falling_through(tmp_path):
    async with lean_wire(tmp_path) as c:
        await c.send(1, "unknown/method")
        raw = await asyncio.wait_for(c.process.stdout.readline(), timeout=10)
        error = json.loads(raw)
        assert error["id"] == 1
        assert error["error"]["code"] == -32601
        ping = await c.request(2, "ping")
        assert ping["resultType"] == "complete"

        await c.send_envelope({"id": 3, "method": "tools/list", "params": {}})
        raw = await asyncio.wait_for(c.process.stdout.readline(), timeout=10)
        invalid = json.loads(raw)
        assert invalid["id"] == 3
        assert invalid["error"]["code"] == -32600
        ping = await c.request(4, "ping")
        assert ping["resultType"] == "complete"
        await c.finish()


@pytest.mark.asyncio
async def test_lean_transport_keeps_numeric_and_string_ids_distinct(tmp_path):
    entered = tmp_path / "typed-entered"
    unwound = tmp_path / "typed-unwound"
    async with lean_wire(
        tmp_path,
        fixture=LEAN_FIXTURE,
        extra_env={
            "OPENBROWSER_TEST_CANCEL_ENTERED": str(entered),
            "OPENBROWSER_TEST_CANCEL_UNWOUND": str(unwound),
        },
    ) as c:
        created = await c.request(1, "tools/call", {"name": "create_context", "arguments": {}})
        context_id = created["structuredContent"]["context_id"]
        await c.send(
            1,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"context_id": context_id, "code": "await wait_for_cancel()"},
            },
        )
        for _ in range(200):
            if entered.exists():
                break
            await asyncio.sleep(0.01)
        assert entered.exists()

        await c.send("1", "ping")
        ping = await c.receive("1")
        assert ping["resultType"] == "complete"

        await c.notify("notifications/cancelled", {"requestId": 1})
        for _ in range(200):
            if unwound.exists():
                break
            await asyncio.sleep(0.01)
        assert unwound.exists()
        ping = await c.request(2, "ping")
        assert ping["resultType"] == "complete"
        await c.finish()


@pytest.mark.asyncio
async def test_lean_transport_reports_pending_request_on_eof(tmp_path):
    entered = tmp_path / "eof-entered"
    async with lean_wire(
        tmp_path,
        fixture=LEAN_FIXTURE,
        extra_env={"OPENBROWSER_TEST_CANCEL_ENTERED": str(entered)},
    ) as c:
        created = await c.request(1, "tools/call", {"name": "create_context", "arguments": {}})
        context_id = created["structuredContent"]["context_id"]
        await c.send(
            10,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"context_id": context_id, "code": "await wait_for_cancel()"},
            },
        )
        for _ in range(200):
            if entered.exists():
                break
            await asyncio.sleep(0.01)
        assert entered.exists()
        assert c.process.stdin is not None
        c.process.stdin.close()
        raw = await asyncio.wait_for(c.process.stdout.readline(), timeout=10)
        response = json.loads(raw)
        assert response["id"] == 10
        assert response["error"]["code"] == -32000
        await asyncio.wait_for(c.process.wait(), timeout=10)
        stderr = (await c.stderr_task).decode(errors="replace")
        assert c.process.returncode == 0, stderr


@pytest.mark.asyncio
async def test_lean_transport_keeps_stdout_private_during_execution(tmp_path):
    async with lean_wire(tmp_path, fixture=LEAN_FIXTURE) as c:
        created = await c.request(1, "tools/call", {"name": "create_context", "arguments": {}})
        context_id = created["structuredContent"]["context_id"]
        await c.send(
            2,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {
                    "context_id": context_id,
                    "code": ("os.write(1, b'NOISE\\n')\nawait asyncio.sleep(0.5)\nprint('done')"),
                },
            },
        )
        await c.send(3, "ping")
        first = json.loads(await asyncio.wait_for(c.process.stdout.readline(), timeout=10))
        assert first["id"] == 3
        assert first["result"]["resultType"] == "complete"
        second = json.loads(await asyncio.wait_for(c.process.stdout.readline(), timeout=10))
        assert second["id"] == 2
        assert second["result"]["content"][0]["text"].strip() == "done"
        stderr = await c.finish()
        assert "NOISE" in stderr
