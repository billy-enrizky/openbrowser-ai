"""Actual app/SDK protocol tests, with no DummyServer or DummyTypes.

Only the browser-dependent execute-code cases use a namespace-startup fixture.
They do not establish browser-agent integration correctness.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import Client, StdioServerParameters

from openbrowser.mcp import server as module

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "support" / "mcp_protocol_server.py"
META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {
        "name": "openbrowser-protocol-test",
        "version": "1",
    },
}


def modern_final(stderr: str) -> dict | None:
    line = next(
        (line for line in stderr.splitlines() if line.startswith("MODERN_FINAL ")),
        None,
    )
    return json.loads(line.removeprefix("MODERN_FINAL ")) if line else None


def child_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "PYTHONPATH": str(ROOT / "src"),
        "OPENBROWSER_ANONYMIZED_TELEMETRY": "false",
        "ANONYMIZED_TELEMETRY": "false",
        "OPENBROWSER_MCP_TRANSPORT": "sdk",
    }


class Wire:
    def __init__(self, process):
        self.process = process
        self.stderr_task = asyncio.create_task(process.stderr.read())
        self.context_id = None

    async def send(self, ident, method, params=None, *, modern=True):
        message = {"jsonrpc": "2.0", "method": method}
        if ident is not None:
            message["id"] = ident
        if params is not None:
            message["params"] = params
        if modern and ident is not None:
            message.setdefault("params", {})["_meta"] = META.copy()
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def receive(self, ident):
        raw = await asyncio.wait_for(self.process.stdout.readline(), timeout=10)
        assert raw, "MCP server closed stdout before replying"
        message = json.loads(raw)
        assert message["id"] == ident, message
        return message

    async def _raw_request(self, ident, method, params=None, *, modern=True):
        await self.send(ident, method, params, modern=modern)
        message = await self.receive(ident)
        assert "error" not in message, message
        return message["result"]

    async def ensure_context(self):
        if self.context_id is None:
            created = await self._raw_request(
                "__context__",
                "tools/call",
                {"name": "create_context", "arguments": {}},
            )
            self.context_id = created["structuredContent"]["context_id"]
        return self.context_id

    async def request(self, ident, method, params=None, *, modern=True, inject_context=True):
        if modern and inject_context and method == "tools/call":
            name = (params or {}).get("name")
            if name in {"execute_code", "get_status", "reset_browser"}:
                context_id = await self.ensure_context()
                params = dict(params or {})
                arguments = dict(params.get("arguments") or {})
                arguments["context_id"] = context_id
                params["arguments"] = arguments
        return await self._raw_request(ident, method, params, modern=modern)

    async def send_tool(self, ident, name, arguments=None, *, modern=True):
        params = {"name": name, "arguments": dict(arguments or {})}
        if modern and name in {"execute_code", "get_status", "reset_browser"}:
            params["arguments"]["context_id"] = await self.ensure_context()
        await self.send(ident, "tools/call", params, modern=modern)

    async def finish(self, *, pending_ids=()):
        if self.process.stdin and not self.process.stdin.is_closing():
            self.process.stdin.close()
        await asyncio.wait_for(self.process.wait(), timeout=10)
        stderr = (await self.stderr_task).decode(errors="replace")
        assert self.process.returncode == 0, stderr
        remaining = await self.process.stdout.read()
        for line in remaining.splitlines():
            message = json.loads(line)
            assert message["id"] in pending_ids, message
            assert "result" in message or "error" in message
        return stderr


@asynccontextmanager
async def wire(home: Path, *, browser_fixture=False, fixture=None, extra_env=None):
    args = [str(fixture or FIXTURE)] if browser_fixture or fixture else ["-m", "openbrowser.mcp"]
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
    connection = Wire(process)
    try:
        yield connection
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await connection.stderr_task


@pytest.mark.asyncio
async def test_real_entrypoint_modern_first_catalog_and_status_without_handshake(
    tmp_path,
):
    assert version("mcp") == "2.3.0"
    assert Path(module.__file__).resolve() == ROOT / "src/openbrowser/mcp/server.py"
    assert version("mcp-types") == "2.3.0"
    async with wire(tmp_path) as c:
        catalog = await c.request(1, "tools/list")
        assert catalog["resultType"] == "complete"
        assert catalog["ttlMs"] == 60000
        assert catalog["cacheScope"] == "private"
        assert [tool["name"] for tool in catalog["tools"]] == [
            "create_context",
            "execute_code",
            "get_status",
            "reset_browser",
            "close_context",
            "web_search",
            "scrape_web",
        ]
        identity = catalog["_meta"]["io.modelcontextprotocol/serverInfo"]
        assert identity == {
            "name": "openbrowser",
            "version": module.get_openbrowser_version(),
        }
        status = await c.request(2, "tools/call", {"name": "get_status"})
        payload = json.loads(status["content"][0]["text"])
        assert payload["browser"]["phase"] == "stopped"
        assert payload["namespace_variables"] == 0
        assert "ttlMs" not in status and "cacheScope" not in status
        assert await c.request(3, "tools/list") == catalog
        await c.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "2026-07-28"])
async def test_latest_sdk_real_stdio_discovery_and_direct_calls(tmp_path, mode):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "openbrowser.mcp"],
        cwd=ROOT,
        env=child_env(tmp_path),
    )
    async with Client(params, mode=mode) as client:
        assert client.protocol_version == "2026-07-28"
        catalog = await client.list_tools()
        assert len(catalog.tools) == 7
        created = await client.call_tool("create_context", {})
        assert not created.is_error
        context_id = created.structured_content["context_id"]
        status = await client.call_tool("get_status", {"context_id": context_id})
        assert not status.is_error
        assert json.loads(status.content[0].text)["browser"]["phase"] == "stopped"
        closed = await client.call_tool("close_context", {"context_id": context_id})
        assert not closed.is_error
        if mode == "auto":
            assert client.server_info.name == "openbrowser"
            caps = client.server_capabilities
            assert caps.tools.list_changed is False
            assert caps.completions is None


@pytest.mark.asyncio
async def test_modern_first_call_and_catalog_cache_metadata(tmp_path):
    async with wire(tmp_path) as c:
        status = await c.request(1, "tools/call", {"name": "get_status", "arguments": {}})
        assert json.loads(status["content"][0]["text"])["browser"]["phase"] == "stopped"
        for ident, method, key in [
            (2, "resources/list", "resources"),
            (3, "resources/templates/list", "resourceTemplates"),
            (4, "prompts/list", "prompts"),
        ]:
            result = await c.request(ident, method)
            assert result[key] == []
            assert (result["ttlMs"], result["cacheScope"], result["resultType"]) == (
                60000,
                "private",
                "complete",
            )
        discover = await c.request(5, "server/discover")
        assert discover["supportedVersions"] == ["2026-07-28"]
        assert discover["capabilities"]["tools"]["listChanged"] is False
        assert not {"completions", "extensions"} & discover["capabilities"].keys()
        await c.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("execute_code", {}),
        ("execute_code", {"code": 1}),
        ("execute_code", {"code": "  "}),
        ("get_status", {"extra": True}),
        ("reset_browser", {"clear_variables": "false"}),
        ("reset_browser", {"clear_variables": 1}),
        ("reset_browser", {"extra": True}),
    ],
)
async def test_schema_failures_are_tool_errors_before_browser_effects(tmp_path, name, arguments):
    async with wire(tmp_path) as c:
        result = await c.request(1, "tools/call", {"name": name, "arguments": arguments})
        assert result["isError"] is True
        status = await c.request(2, "tools/call", {"name": "get_status", "arguments": {}})
        payload = json.loads(status["content"][0]["text"])
        assert payload["browser"]["phase"] == "stopped"
        assert payload["namespace_variables"] == 0
        await c.finish()


@pytest.mark.asyncio
async def test_unknown_tool_and_mixed_era_are_protocol_errors(tmp_path):
    async with wire(tmp_path) as c:
        await c.send(1, "tools/call", {"name": "unknown"})
        assert (await c.receive(1))["error"]["code"] == -32602
        await c.send(
            2,
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
            modern=False,
        )
        assert "error" in await c.receive(2)
        await c.finish()


@pytest.mark.asyncio
async def test_missing_modern_capabilities_are_rejected(tmp_path):
    async with wire(tmp_path) as c:
        await c.send(
            1,
            "tools/list",
            {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}},
            modern=False,
        )
        assert (await c.receive(1))["error"]["code"] == -32602
        await c.finish()


@pytest.mark.asyncio
async def test_legacy_2024_raw_stdio_preserves_catalog_and_status(tmp_path):
    async with wire(tmp_path) as c:
        init = await c.request(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "old-client", "version": "1"},
            },
            modern=False,
        )
        assert init["protocolVersion"] == "2024-11-05"
        await c.send(None, "notifications/initialized", modern=False)
        catalog = await c.request(2, "tools/list", modern=False)
        assert len(catalog["tools"]) == 4
        assert not {"ttlMs", "cacheScope", "resultType"} & catalog.keys()
        status = await c.request(3, "tools/call", {"name": "get_status", "arguments": {}}, modern=False)
        assert json.loads(status["content"][0]["text"])["browser"]["phase"] == "stopped"
        await c.finish()


@pytest.mark.asyncio
async def test_real_executor_state_print_extra_keys_reset_and_process_isolation(
    tmp_path,
):
    async with (
        wire(tmp_path / "one", browser_fixture=True) as first,
        wire(tmp_path / "two", browser_fixture=True) as second,
    ):
        a = await first.request(
            1,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {
                    "code": "answer = 42\nprint('hello')",
                    "extra": "permitted",
                },
            },
        )
        assert not a["isError"] and "hello" in a["content"][0]["text"]
        b = await first.request(
            2,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": "print(answer)"}},
        )
        assert "42" in b["content"][0]["text"]
        other = await second.request(
            1,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"code": "print('answer' in globals())"},
            },
        )
        assert "False" in other["content"][0]["text"]
        first_status = await first.request(3, "tools/call", {"name": "get_status"})
        second_status = await second.request(2, "tools/call", {"name": "get_status"})
        assert (
            json.loads(first_status["content"][0]["text"])["instance_id"]
            != json.loads(second_status["content"][0]["text"])["instance_id"]
        )
        catalog = await first.request(4, "tools/list")
        reset = await first.request(
            5,
            "tools/call",
            {"name": "reset_browser", "arguments": {"clear_variables": True}},
        )
        assert json.loads(reset["content"][0]["text"])["ok"] is True
        assert await first.request(6, "tools/list") == catalog
        printed_error = await first.request(
            7,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"code": "print('Error: ordinary user output')"},
            },
        )
        assert not printed_error["isError"]
        assert "Error: ordinary user output" in printed_error["content"][0]["text"]
        await first.finish()
        await second.finish()


@pytest.mark.asyncio
async def test_cancelled_execution_has_no_reply_and_releases_execution_lock(tmp_path):
    async with wire(tmp_path, browser_fixture=True) as c:
        await c.send_tool(1, "execute_code", {"code": "await wait_for_cancel()"})
        for ident in range(2, 22):
            status = await c.request(ident, "tools/call", {"name": "get_status"})
            if json.loads(status["content"][0]["text"])["namespace_variables"]:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("Execution did not enter namespace")
        await c.send(None, "notifications/cancelled", {"requestId": 1})
        result = await c.request(
            30,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": "print('after cancel')"}},
        )
        assert "after cancel" in result["content"][0]["text"]
        status = await c.request(31, "tools/call", {"name": "get_status"})
        assert json.loads(status["content"][0]["text"])["pending_variables"] == 1
        await c.finish()


async def wait_for_file(path: Path, message: str, timeout: float = 10) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not path.exists():
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(message)
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_active_eof_stops_real_protocol_process(tmp_path):
    entered = tmp_path / "cancel-entered"
    unwound = tmp_path / "cancel-unwound"
    async with wire(
        tmp_path,
        browser_fixture=True,
        extra_env={
            "OPENBROWSER_TEST_CANCEL_ENTERED": str(entered),
            "OPENBROWSER_TEST_CANCEL_UNWOUND": str(unwound),
        },
    ) as c:
        await c.send_tool(1, "execute_code", {"code": "await wait_for_cancel()"})
        await wait_for_file(entered, "execute_code never entered wait_for_cancel")
        await c.finish(pending_ids=(1,))
    assert unwound.exists(), "active call did not unwind before the process exited"


@pytest.mark.asyncio
async def test_catalog_is_snapshot_and_cannot_mutate_validation(monkeypatch):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    monkeypatch.setenv("OPENBROWSER_COMPACT_DESCRIPTION", "true")
    server = module.OpenBrowserServer()
    async with Client(server.server) as client:
        catalog = await client.list_tools(cache_mode="refresh")
        execute = next(tool for tool in catalog.tools if tool.name == "execute_code")
        original = execute.description
        execute.input_schema["properties"]["code"]["type"] = "number"
        monkeypatch.delenv("OPENBROWSER_COMPACT_DESCRIPTION")
        fresh = await client.list_tools(cache_mode="refresh")
        fresh_execute = next(tool for tool in fresh.tools if tool.name == "execute_code")
        assert fresh_execute.description == original == module._EXECUTE_CODE_DESCRIPTION_COMPACT
        assert fresh_execute.input_schema["properties"]["code"]["type"] == "string"
        created = await client.call_tool("create_context", {})
        context_id = created.structured_content["context_id"]
        result = await client.call_tool("execute_code", {"context_id": context_id, "code": 5})
        assert result.is_error
        assert "$.code" in result.content[0].text
        assert "not of type" in result.content[0].text
        assert server.browser_session is None
        owner = server._contexts._entries[context_id].owner
        assert owner.browser_session is None


@pytest.mark.asyncio
async def test_actual_sdk_126_client_uses_app_legacy_stdio(tmp_path):
    legacy_python = os.environ.get("OPENBROWSER_TEST_MCP1_PYTHON")
    if not legacy_python:
        if os.environ.get("CI"):
            pytest.fail("CI must configure the isolated mcp==1.26.0 compatibility interpreter")
        pytest.skip(
            "Set OPENBROWSER_TEST_MCP1_PYTHON to an isolated mcp==1.26.0 interpreter for cross-version acceptance"
        )
    process = await asyncio.create_subprocess_exec(
        legacy_python,
        str(ROOT / "tests" / "support" / "mcp_legacy_client.py"),
        sys.executable,
        str(FIXTURE),
        cwd=ROOT,
        env=child_env(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, stderr.decode(errors="replace")
    assert json.loads(stdout) == {
        "sdk": "1.26.0",
        "protocol": "2025-11-25",
        "state": 42,
    }


@pytest.mark.asyncio
async def test_dispatch_exception_is_explicit_tool_error(monkeypatch):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()

    async def fail(name, arguments):
        raise RuntimeError("dispatcher failed")

    monkeypatch.setattr(server, "_handle_tool", fail)
    handler = server.server.get_request_handler("tools/call").handler
    result = await handler(None, module.types.CallToolRequestParams(name="get_status", arguments={}))
    assert result.is_error
    assert result.content[0].text == "Error: dispatcher failed"


@pytest.mark.asyncio
async def test_tool_error_sanitizes_credentials(monkeypatch):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()

    async def fail(name, arguments):
        raise RuntimeError("Authorization: Bearer mcp-response-secret")

    monkeypatch.setattr(server, "_handle_tool", fail)
    handler = server.server.get_request_handler("tools/call").handler
    result = await handler(None, module.types.CallToolRequestParams(name="get_status", arguments={}))

    assert result.is_error
    text = result.content[0].text
    assert "mcp-response-secret" not in text
    assert "authorization=<redacted>" in text.lower()


@pytest.mark.asyncio
async def test_typed_recovery_keeps_long_detail_and_guidance_at_tool_boundary(monkeypatch):
    from types import SimpleNamespace

    from openbrowser.config_lifecycle import ConfigStartupError

    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    server._executor = MagicMock()
    server._executor.initialized = True
    server._executor.max_output_chars = 12_000
    server._namespace = {"__builtins__": __builtins__}
    server.browser_session = MagicMock()
    server._executor.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=False,
            output=(
                "websocket connection closed\nAuthorization: Bearer typed-output-secret\n" + "partial-output-" * 1000
            ),
            error=None,
        )
    )
    long_hint = "Fix the config, then restart the MCP server. " + "Follow the config guide. " * 20
    recovery_error = ConfigStartupError(
        SimpleNamespace(
            path="config.toml",
            error_code="config_restart_required",
            hint=long_hint,
        ),
        code="config_restart_required",
        hint=long_hint,
        detail="recovery-specific diagnostic detail",
    )
    server._recover_browser_session = AsyncMock(side_effect=recovery_error)

    handler = server.server.get_request_handler("tools/call").handler
    with patch.object(server, "_is_cdp_alive", AsyncMock(return_value=True)):
        result = await handler(
            None,
            module.types.CallToolRequestParams(
                name="execute_code",
                arguments={"code": "print(1)"},
            ),
        )

    assert result.is_error
    text = result.content[0].text
    assert len(text) > 240
    assert "partial-output-" in text
    assert "recovery-specific diagnostic detail" in text
    assert "config_restart_required" in text
    assert "Fix the config, then restart the MCP server." in text
    assert "The code may have partially run" in text
    assert "get_status before retrying" in text
    assert "typed-output-secret" not in text
    assert "authorization=<redacted>" in text.lower()


@pytest.mark.asyncio
async def test_typed_recovery_tool_boundary_respects_a_64_char_cap(monkeypatch):
    from types import SimpleNamespace

    from openbrowser.config_lifecycle import ConfigStartupError

    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    server._executor = MagicMock()
    server._executor.initialized = True
    server._executor.max_output_chars = 64
    server._namespace = {"__builtins__": __builtins__}
    server.browser_session = MagicMock()
    server._executor.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=False,
            output="partial-output Authorization: Bearer typed-output-secret",
            error="websocket connection closed",
        )
    )
    long_hint = "Fix the config. Authorization: Bearer typed-output-secret " + "Follow the guide. " * 30
    recovery_error = ConfigStartupError(
        SimpleNamespace(
            path="config.toml",
            error_code="config_restart_required",
            hint=long_hint,
        ),
        code="config_restart_required",
        hint=long_hint,
        detail="recovery-specific diagnostic detail",
    )
    server._recover_browser_session = AsyncMock(side_effect=recovery_error)

    handler = server.server.get_request_handler("tools/call").handler
    with patch.object(server, "_is_cdp_alive", AsyncMock(return_value=True)):
        result = await handler(
            None,
            module.types.CallToolRequestParams(
                name="execute_code",
                arguments={"code": "print(1)"},
            ),
        )

    assert result.is_error
    text = result.content[0].text
    body = text.removeprefix("Error: ")
    detail, separator, message = body.partition("\nconfig_restart_required:")
    assert separator
    assert len(detail) <= 64
    assert "partial-output" in detail
    assert "recovery-spec" in detail
    assert "Fix the config." in message
    assert "config_restart_required" in text
    assert "The code may have partially run" in text
    assert "get_status before retrying" in text
    assert "typed-output-secret" not in text
    assert "authorization=<redacted>" in text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("output_cap", [1, 2, 3, 64])
async def test_typed_recovery_detail_stays_within_small_output_caps(monkeypatch, output_cap):
    from types import SimpleNamespace

    from openbrowser.browser.cdp_connection import MAX_ERROR_CHARS
    from openbrowser.config_lifecycle import ConfigStartupError

    server = module.OpenBrowserServer()
    server.browser_session = None
    server._executor = MagicMock()
    server._executor.initialized = True
    server._executor.max_output_chars = output_cap
    server._namespace = {"__builtins__": __builtins__}
    server._executor.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=False,
            output="partial-output",
            error="websocket connection closed",
        )
    )
    recovery_error = ConfigStartupError(
        SimpleNamespace(
            path="config.toml",
            error_code="config_restart_required",
            hint="Fix the config, then restart the MCP server.",
        ),
        code="config_restart_required",
        hint="Fix the config, then restart the MCP server.",
        detail="recovery-detail",
    )
    server._recover_browser_session = AsyncMock(side_effect=recovery_error)

    with pytest.raises(ConfigStartupError) as raised:
        await server._execute_code_locked("print(1)")

    detail = raised.value.detail or ""
    assert len(detail) <= output_cap
    if output_cap in (1, 2):
        assert detail == "r"
    elif output_cap == 3:
        assert detail == "p\nr"
    else:
        assert "partial-output" in detail
        assert "recovery-detail" in detail
    assert raised.value.code == "config_restart_required"
    assert "The code may have partially run" in raised.value.hint
    assert len(raised.value.hint) <= MAX_ERROR_CHARS


@pytest.mark.asyncio
async def test_typed_config_error_sanitizes_credentials(monkeypatch):
    from types import SimpleNamespace

    from openbrowser.config_lifecycle import ConfigStartupError

    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    error = ConfigStartupError(
        SimpleNamespace(
            path="config.toml",
            error_code="config_invalid",
            hint="Authorization: Bearer mcp-typed-hint",
        ),
        hint="Authorization: Bearer mcp-typed-hint",
        detail="Authorization: Bearer mcp-typed-detail",
    )

    async def fail(name, arguments):
        raise error

    monkeypatch.setattr(server, "_handle_tool", fail)
    handler = server.server.get_request_handler("tools/call").handler
    result = await handler(None, module.types.CallToolRequestParams(name="get_status", arguments={}))

    text = result.content[0].text
    assert "mcp-typed" not in text
    assert "authorization=<redacted>" in text.lower()


@pytest.mark.asyncio
async def test_exact_catalog_input_contracts_and_annotations(monkeypatch):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    async with Client(server.server) as client:
        catalog = await client.list_tools()
    assert [tool.name for tool in catalog.tools] == [
        "create_context",
        "execute_code",
        "get_status",
        "reset_browser",
        "close_context",
        "web_search",
        "scrape_web",
    ]
    create, execute, status, reset, close, search_tool, scrape_tool = catalog.tools
    context_property = {
        "type": "string",
        "description": (
            "Opaque context handle returned by create_context. Pass the same value "
            "on every stateful call for that context."
        ),
    }
    assert create.input_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert create.output_schema == {
        "type": "object",
        "properties": {"context_id": context_property},
        "required": ["context_id"],
        "additionalProperties": False,
    }
    assert execute.input_schema == {
        "type": "object",
        "properties": {
            "context_id": context_property,
            "code": {
                "type": "string",
                "description": "Python code to execute. All browser functions are async (use await).",
            },
        },
        "required": ["context_id", "code"],
    }
    assert status.input_schema == {
        "type": "object",
        "properties": {"context_id": context_property},
        "required": ["context_id"],
        "additionalProperties": False,
    }
    assert scrape_tool.name == "scrape_web"
    assert scrape_tool.input_schema["required"] == ["url"]
    assert "no-Blink" in scrape_tool.description
    assert "no-V8" in scrape_tool.description
    legacy_status = server._legacy_by_name["get_status"]
    for status_tool in (status, legacy_status):
        assert "does not start a browser or probe CDP health" in status_tool.description
        assert "last accepted connect or probe observation" in status_tool.description
        assert "cdp.healthy" in status_tool.description
        assert "cdp.phase" in status_tool.description
        assert "disconnected" in status_tool.description
        assert "cdp.last_error_code" in status_tool.description
        assert "session_not_found" in status_tool.description
        assert "cdp.last_probe_at" in status_tool.description
        assert "cdp_connected only means a client object is present" in status_tool.description
        assert "does not prove a target session, tab, or page can accept commands" in status_tool.description
        assert "browser.health" in status_tool.description
        assert "last_focus_recovery_outcome" in status_tool.description
        assert "process_present is null" in status_tool.description
        assert "process identity is inconclusive" in status_tool.description
        assert "Wait while recovery is pending" in status_tool.description
        assert "Reset only when no usable target exists" in status_tool.description
        assert "missing health block means no browser session" in status_tool.description
    assert reset.input_schema == {
        "type": "object",
        "properties": {
            "context_id": context_property,
            "clear_variables": {
                "type": "boolean",
                "description": "Also discard user-defined Python variables.",
                "default": False,
            },
        },
        "required": ["context_id"],
        "additionalProperties": False,
    }
    assert close.input_schema == {
        "type": "object",
        "properties": {"context_id": context_property},
        "required": ["context_id"],
        "additionalProperties": False,
    }
    assert close.output_schema == {
        "type": "object",
        "properties": {
            "context_id": context_property,
            "status": {"type": "string", "enum": ["closed", "close_failed"]},
            "error_code": {"type": ["string", "null"]},
            "retryable": {"type": "boolean"},
        },
        "required": ["context_id", "status", "retryable"],
        "additionalProperties": False,
    }
    assert search_tool.input_schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 2048},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "default": 5,
                "description": "Maximum results. Values above 20 are capped at 20.",
            },
            "backend": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "default": "auto",
                "description": "DDGS backend or comma-delimited backends.",
            },
            "region": {"type": "string", "minLength": 1, "maxLength": 32, "default": "us-en"},
            "safe_search": {
                "type": "string",
                "enum": ["on", "moderate", "off"],
                "default": "moderate",
            },
            "timelimit": {
                "type": "string",
                "maxLength": 64,
                "description": "Optional DDGS time limit, such as d, w, m, or y.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    assert "context_id" not in search_tool.input_schema["properties"]
    assert "Snippets are discovery previews." in search_tool.description
    assert [tool.annotations.model_dump(by_alias=True, exclude_none=True) for tool in catalog.tools] == [
        {
            "readOnlyHint": False,
            "destructiveHint": False,
            "openWorldHint": False,
            "idempotentHint": False,
        },
        {
            "readOnlyHint": False,
            "destructiveHint": True,
            "openWorldHint": True,
            "idempotentHint": False,
        },
        {
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
            "idempotentHint": True,
        },
        {
            "readOnlyHint": False,
            "destructiveHint": True,
            "openWorldHint": False,
            "idempotentHint": True,
        },
        {
            "readOnlyHint": False,
            "destructiveHint": True,
            "openWorldHint": False,
            "idempotentHint": True,
        },
        {
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": True,
            "idempotentHint": False,
        },
            {
                "readOnlyHint": True,
                "destructiveHint": False,
                "openWorldHint": True,
                "idempotentHint": True,
            },
    ]

    legacy = await server.server.get_request_handler("tools/list").handler(None, None)
    assert [tool.name for tool in legacy.tools] == [
        "execute_code",
        "get_status",
        "reset_browser",
        "web_search",
    ]
    legacy_execute, legacy_status, legacy_reset, legacy_search = legacy.tools
    assert "context_id" not in legacy_search.input_schema["properties"]
    assert legacy_execute.input_schema == {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. All browser functions are async (use await).",
            }
        },
        "required": ["code"],
    }
    assert legacy_status.input_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert legacy_reset.input_schema == {
        "type": "object",
        "properties": {
            "clear_variables": {
                "type": "boolean",
                "description": "Also discard user-defined Python variables.",
                "default": False,
            }
        },
        "additionalProperties": False,
    }


@pytest.mark.asyncio
async def test_subprocess_stdout_is_off_protocol_wire(tmp_path):
    async with wire(tmp_path, browser_fixture=True) as c:
        result = await c.request(
            1,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {
                    "code": "import subprocess, sys\nsubprocess.run([sys.executable, '-c', \"print('child output is not protocol')\"], check=True)\nprint('captured user output')"
                },
            },
        )
        assert not result["isError"]
        assert "captured user output" in result["content"][0]["text"]
        stderr = await c.finish()
        assert "child output is not protocol" in stderr


async def wait_for_barrier(path):
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [True, False], ids=["modern", "legacy"])
@pytest.mark.parametrize("mode", ["reset", "cancel-start", "start-failure", "replacement-failure"])
async def test_owned_cleanup_cancellation_preserves_real_stdio_usability(tmp_path, modern, mode):
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    fixture = ROOT / "tests/support/mcp_owned_browser_server.py"
    async with wire(
        tmp_path,
        fixture=fixture,
        extra_env={
            "OPENBROWSER_TEST_BARRIERS": str(barriers),
            "OPENBROWSER_TEST_OWNED_MODE": mode,
        },
    ) as c:
        if not modern:
            await c.request(
                90,
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "cleanup-test", "version": "1"},
                },
                modern=False,
            )
            await c.send(None, "notifications/initialized", modern=False)
        if mode in ("reset", "replacement-failure"):
            await c.request(
                1,
                "tools/call",
                {"name": "execute_code", "arguments": {"code": "answer = 42"}},
                modern=modern,
            )
        if mode == "replacement-failure":
            (barriers / "recover").touch()
        operation = (
            {"name": "reset_browser"}
            if mode == "reset"
            else {"name": "execute_code", "arguments": {"code": "answer = 99"}}
        )
        await c.send_tool(2, operation["name"], operation.get("arguments"), modern=modern)
        if mode == "cancel-start":
            await wait_for_barrier(barriers / "start-1")
            await c.send(None, "notifications/cancelled", {"requestId": 2}, modern=modern)
        blocked_owner = 2 if mode == "replacement-failure" else 1
        await wait_for_barrier(barriers / f"kill-{blocked_owner}")
        if mode != "cancel-start":
            await c.send(None, "notifications/cancelled", {"requestId": 2}, modern=modern)
        # The response proves the dispatcher consumed the cancellation before kill can complete.
        during = await c.request(3, "tools/call", {"name": "get_status"}, modern=modern)
        assert json.loads(during["content"][0]["text"])["browser"]["phase"] in (
            "ready",
            "failed",
        )
        (barriers / "recover").unlink(missing_ok=True)
        (barriers / f"release-{blocked_owner}").touch()
        following = await c.request(
            4,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"code": "print(browser.ident)\nprint(globals().get('answer', 'missing'))"},
            },
            modern=modern,
        )
        text = following["content"][0]["text"]
        assert str(blocked_owner + 1) in text
        # A failed replacement still keeps scalar user values for the next start.
        assert ("42" if mode in ("reset", "replacement-failure") else "missing") in text
        status = await c.request(5, "tools/call", {"name": "get_status"}, modern=modern)
        assert json.loads(status["content"][0]["text"])["browser"]["phase"] == "ready"
        reset = await c.request(
            6,
            "tools/call",
            {"name": "reset_browser", "arguments": {"clear_variables": True}},
            modern=modern,
        )
        assert json.loads(reset["content"][0]["text"])["ok"] is True
        stderr = await c.finish()
        final = next(
            line.removeprefix("OWNED_FINAL ") for line in stderr.splitlines() if line.startswith("OWNED_FINAL ")
        )
        state = json.loads(final)
        assert state["owner"] is None
        assert not state["execution_locked"] and not state["lifecycle_locked"]
        assert blocked_owner in state["killed"]
        if modern:
            modern_state = modern_final(stderr)
            assert modern_state is not None
            assert modern_state["owner"] is None
            assert not modern_state["execution_locked"]
            assert not modern_state["lifecycle_locked"]


@pytest.mark.asyncio
async def test_eof_during_owned_cleanup_finishes_bounded_and_releases_owner(tmp_path):
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    async with wire(
        tmp_path,
        fixture=ROOT / "tests/support/mcp_owned_browser_server.py",
        extra_env={
            "OPENBROWSER_TEST_BARRIERS": str(barriers),
            "OPENBROWSER_TEST_OWNED_MODE": "reset",
        },
    ) as c:
        await c.request(
            1,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": "answer = 42"}},
        )
        await c.send_tool(2, "reset_browser")
        await wait_for_barrier(barriers / "kill-1")
        c.process.stdin.close()
        (barriers / "release-1").touch()
        stderr = await c.finish(pending_ids=(2,))
        final = next(
            line.removeprefix("OWNED_FINAL ") for line in stderr.splitlines() if line.startswith("OWNED_FINAL ")
        )
        assert json.loads(final) == {
            "owner": None,
            "execution_locked": False,
            "lifecycle_locked": False,
            "killed": [1],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [True, False], ids=["modern", "legacy"])
@pytest.mark.parametrize("mode", ["reset-failure", "reset-child-cancel", "reset-both-cancel"])
async def test_failed_owned_cleanup_retains_owner_and_cancellation_keeps_transport_usable(tmp_path, modern, mode):
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    async with wire(
        tmp_path,
        fixture=ROOT / "tests/support/mcp_owned_browser_server.py",
        extra_env={
            "OPENBROWSER_TEST_BARRIERS": str(barriers),
            "OPENBROWSER_TEST_OWNED_MODE": mode,
        },
    ) as c:
        if not modern:
            await c.request(
                90,
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "cleanup-test", "version": "1"},
                },
                modern=False,
            )
            await c.send(None, "notifications/initialized", modern=False)
        await c.request(
            1,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": "answer = 42"}},
            modern=modern,
        )
        await c.send_tool(2, "reset_browser", modern=modern)
        await wait_for_barrier(barriers / "kill-1")
        if mode != "reset-child-cancel":
            await c.send(None, "notifications/cancelled", {"requestId": 2}, modern=modern)
        await c.request(3, "tools/call", {"name": "get_status"}, modern=modern)
        (barriers / "release-1").touch()
        if mode == "reset-child-cancel":
            failed = await c.receive(2)
            assert "result" in failed
            assert json.loads(failed["result"]["content"][0]["text"])["ok"] is False
        existing = await c.request(
            4,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"code": "print(browser.ident)\nprint(answer)"},
            },
            modern=modern,
        )
        assert existing["content"][0]["text"] == "1\n42\n"
        status = await c.request(5, "tools/call", {"name": "get_status"}, modern=modern)
        browser = json.loads(status["content"][0]["text"])["browser"]
        assert browser["phase"] == "failed" and browser["last_error"]
        reset = await c.request(6, "tools/call", {"name": "reset_browser"}, modern=modern)
        assert json.loads(reset["content"][0]["text"])["ok"]
        fresh = await c.request(
            7,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"code": "print(browser.ident)\nprint(answer)"},
            },
            modern=modern,
        )
        assert fresh["content"][0]["text"] == "2\n42\n"
        await c.request(
            8,
            "tools/call",
            {"name": "reset_browser", "arguments": {"clear_variables": True}},
            modern=modern,
        )
        stderr = await c.finish()
        final = next(
            line.removeprefix("OWNED_FINAL ") for line in stderr.splitlines() if line.startswith("OWNED_FINAL ")
        )
        assert json.loads(final) == {
            "owner": None,
            "execution_locked": False,
            "lifecycle_locked": False,
            "killed": [1, 2],
        }
        if modern:
            modern_state = modern_final(stderr)
            assert modern_state is not None
            assert modern_state["owner"] is None
            assert not modern_state["execution_locked"]
            assert not modern_state["lifecycle_locked"]


@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [True, False], ids=["modern", "legacy"])
@pytest.mark.parametrize("stage", ["initial-start", "replacement-start", "health", "executor"])
async def test_owned_tool_self_cancellation_is_error_without_losing_request_transport(tmp_path, modern, stage):
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    (barriers / "release-1").touch()
    (barriers / "release-2").touch()
    async with wire(
        tmp_path,
        fixture=ROOT / "tests/support/mcp_owned_browser_server.py",
        extra_env={
            "OPENBROWSER_TEST_BARRIERS": str(barriers),
            "OPENBROWSER_TEST_OWNED_MODE": stage + "-self-cancel",
        },
    ) as c:
        if not modern:
            await c.request(
                90,
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "self-cancel-test", "version": "1"},
                },
                modern=False,
            )
            await c.send(None, "notifications/initialized", modern=False)
        if stage != "initial-start":
            await c.request(
                1,
                "tools/call",
                {"name": "execute_code", "arguments": {"code": "answer = 42"}},
                modern=modern,
            )
        if stage == "replacement-start":
            (barriers / "recover").touch()
        if stage == "health":
            (barriers / "health-self-cancel").touch()
        code = (
            "partial = 7\nraise asyncio.CancelledError('executor terminated itself')"
            if stage == "executor"
            else "should_not_run = True"
        )
        failed = await c.request(
            2,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": code}},
            modern=modern,
        )
        assert failed["isError"] is True
        (barriers / "recover").unlink(missing_ok=True)
        status = await c.request(3, "tools/call", {"name": "get_status"}, modern=modern)
        state = json.loads(status["content"][0]["text"])
        assert state["browser"]["phase"] == ("failed" if stage in ("initial-start", "replacement-start") else "ready")
        inspect = "print(globals().get('answer', 'missing'))\nprint(globals().get('partial', 'missing'))\nprint('should_not_run' in globals())"
        following = await c.request(
            4,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": inspect}},
            modern=modern,
        )
        text = following["content"][0]["text"]
        assert ("42" if stage in ("health", "executor") else "missing") in text
        if stage == "executor":
            assert "7" in text
        assert "False" in text
        await c.request(
            5,
            "tools/call",
            {"name": "reset_browser", "arguments": {"clear_variables": True}},
            modern=modern,
        )
        cleared = await c.request(
            6,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"code": "print('answer' in globals())\nprint('partial' in globals())"},
            },
            modern=modern,
        )
        assert cleared["content"][0]["text"] == "False\nFalse\n"
        await c.request(
            7,
            "tools/call",
            {"name": "reset_browser", "arguments": {"clear_variables": True}},
            modern=modern,
        )
        stderr = await c.finish()
        final = next(
            line.removeprefix("OWNED_FINAL ") for line in stderr.splitlines() if line.startswith("OWNED_FINAL ")
        )
        final_state = json.loads(final)
        assert final_state["owner"] is None
        assert not final_state["execution_locked"]
        assert not final_state["lifecycle_locked"]
        if modern:
            modern_state = modern_final(stderr)
            assert modern_state is not None
            assert modern_state["owner"] is None
            assert not modern_state["execution_locked"]
            assert not modern_state["lifecycle_locked"]


@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [True, False], ids=["modern", "legacy"])
async def test_executor_timeout_keeps_existing_partial_state_and_transport_usable(tmp_path, modern):
    async with wire(tmp_path, browser_fixture=True, extra_env={"OPENBROWSER_MCP_EXEC_TIMEOUT": "1"}) as c:
        if not modern:
            await c.request(
                90,
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "timeout-test", "version": "1"},
                },
                modern=False,
            )
            await c.send(None, "notifications/initialized", modern=False)
        result = await c.request(
            1,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"code": "partial = 7\nawait asyncio.sleep(3600)"},
            },
            modern=modern,
        )
        assert "timed out" in result["content"][0]["text"].lower()
        following = await c.request(
            2,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": "print(partial)"}},
            modern=modern,
        )
        assert following["content"][0]["text"] == "7\n"
        await c.request(
            3,
            "tools/call",
            {"name": "reset_browser", "arguments": {"clear_variables": True}},
            modern=modern,
        )
        cleared = await c.request(
            4,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"code": "print('partial' in globals())"},
            },
            modern=modern,
        )
        assert cleared["content"][0]["text"] == "False\n"
        await c.finish()


@pytest.mark.asyncio
async def test_two_modern_contexts_are_isolated_and_close_invalidates_handle(tmp_path):
    async with wire(tmp_path, browser_fixture=True) as c:
        first = await c._raw_request("ctx-a", "tools/call", {"name": "create_context", "arguments": {}})
        second = await c._raw_request("ctx-b", "tools/call", {"name": "create_context", "arguments": {}})
        first_id = first["structuredContent"]["context_id"]
        second_id = second["structuredContent"]["context_id"]
        assert first_id != second_id
        assert json.loads(first["content"][0]["text"]) == {"context_id": first_id}

        before = await c._raw_request(
            "status-a",
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": first_id}},
        )
        assert json.loads(before["content"][0]["text"])["browser"]["phase"] == "stopped"

        await c._raw_request(
            "exec-a",
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"context_id": first_id, "code": "alpha = 'first'"},
            },
        )
        isolated = await c._raw_request(
            "exec-b",
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {
                    "context_id": second_id,
                    "code": "print('alpha' in globals())",
                },
            },
        )
        assert isolated["content"][0]["text"] == "False\n"

        first_status = await c._raw_request(
            "status-a2",
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": first_id}},
        )
        second_status = await c._raw_request(
            "status-b2",
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": second_id}},
        )
        first_count = json.loads(first_status["content"][0]["text"])["namespace_variables"]
        second_count = json.loads(second_status["content"][0]["text"])["namespace_variables"]
        assert first_count == second_count + 1

        closed = await c._raw_request(
            "close-a",
            "tools/call",
            {"name": "close_context", "arguments": {"context_id": first_id}},
        )
        assert closed["structuredContent"]["status"] == "closed"
        assert not closed["isError"]

        missing = await c._raw_request(
            "status-a3",
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": first_id}},
        )
        assert missing["isError"] is True
        assert "context_not_found" in missing["content"][0]["text"]

        after = await c._raw_request(
            "exec-b2",
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"context_id": second_id, "code": "print('still open')"},
            },
        )
        assert "still open" in after["content"][0]["text"]
        await c._raw_request(
            "close-b",
            "tools/call",
            {"name": "close_context", "arguments": {"context_id": second_id}},
        )
        await c.finish()


@pytest.mark.asyncio
async def test_modern_context_capacity_returns_before_allocation(tmp_path):
    async with wire(
        tmp_path,
        browser_fixture=True,
        extra_env={"OPENBROWSER_MCP_MAX_CONTEXTS": "2"},
    ) as c:
        first = await c._raw_request("ctx-a", "tools/call", {"name": "create_context", "arguments": {}})
        await c._raw_request("ctx-b", "tools/call", {"name": "create_context", "arguments": {}})
        limited = await c._raw_request("ctx-c", "tools/call", {"name": "create_context", "arguments": {}})
        assert limited["isError"] is True
        assert "context_limit_reached" in limited["content"][0]["text"]
        first_id = first["structuredContent"]["context_id"]
        await c._raw_request(
            "close-a",
            "tools/call",
            {"name": "close_context", "arguments": {"context_id": first_id}},
        )
        replacement = await c._raw_request("ctx-d", "tools/call", {"name": "create_context", "arguments": {}})
        assert replacement["isError"] is False
        await c.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"])
async def test_all_allowlisted_legacy_versions_keep_four_tools(tmp_path, version):
    async with wire(tmp_path, browser_fixture=True) as c:
        init = await c.request(
            1,
            "initialize",
            {
                "protocolVersion": version,
                "capabilities": {},
                "clientInfo": {"name": "legacy-matrix", "version": "1"},
            },
            modern=False,
        )
        assert init["protocolVersion"] == version
        await c.send(None, "notifications/initialized", modern=False)
        catalog = await c.request(2, "tools/list", modern=False)
        assert [tool["name"] for tool in catalog["tools"]] == [
            "execute_code",
            "get_status",
            "reset_browser",
            "web_search",
        ]
        status_tool = next(tool for tool in catalog["tools"] if tool["name"] == "get_status")
        assert "does not start a browser or probe CDP health" in status_tool["description"]
        assert "last accepted connect or probe observation" in status_tool["description"]
        assert "cdp.phase" in status_tool["description"]
        assert "disconnected" in status_tool["description"]
        assert "cdp.last_error_code" in status_tool["description"]
        assert "session_not_found" in status_tool["description"]
        assert "cdp.last_probe_at" in status_tool["description"]
        assert "cdp_connected only means a client object is present" in status_tool["description"]
        assert "does not prove a target session, tab, or page can accept commands" in status_tool["description"]
        assert "browser.health" in status_tool["description"]
        assert "last_focus_recovery_outcome" in status_tool["description"]
        assert "process_present is null" in status_tool["description"]
        assert "process identity is inconclusive" in status_tool["description"]
        assert "Wait while recovery is pending" in status_tool["description"]
        assert "Reset only when no usable target exists" in status_tool["description"]
        assert "missing health block means no browser session" in status_tool["description"]
        status = await c.request(3, "tools/call", {"name": "get_status", "arguments": {}}, modern=False)
        assert json.loads(status["content"][0]["text"])["browser"]["phase"] == "stopped"
        await c.finish()


@pytest.mark.asyncio
async def test_unsupported_protocol_version_is_rejected(tmp_path):
    async with wire(tmp_path) as c:
        meta = {
            **META,
            "io.modelcontextprotocol/protocolVersion": "9999-01-01",
        }
        await c.send(1, "tools/list", {"_meta": meta}, modern=False)
        message = await c.receive(1)
        assert "error" in message
        await c.finish()


@pytest.mark.asyncio
async def test_modern_stateful_calls_without_handle_are_rejected_before_effect(tmp_path):
    async with wire(tmp_path) as c:
        for ident, (name, arguments) in enumerate(
            [
                ("execute_code", {"code": "pass"}),
                ("get_status", {}),
                ("reset_browser", {}),
            ],
            1,
        ):
            result = await c.request(
                ident,
                "tools/call",
                {"name": name, "arguments": arguments},
                inject_context=False,
            )
            assert result["isError"] is True
            assert "context_id" in result["content"][0]["text"]
        created = await c._raw_request("ctx", "tools/call", {"name": "create_context", "arguments": {}})
        context_id = created["structuredContent"]["context_id"]
        status = await c._raw_request(
            "status",
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": context_id}},
        )
        assert json.loads(status["content"][0]["text"])["browser"]["phase"] == "stopped"
        await c.finish()


@pytest.mark.asyncio
async def test_close_failure_is_explicit_and_retryable_over_stdio(tmp_path):
    barriers = tmp_path / "barriers"
    barriers.mkdir()
    async with wire(
        tmp_path,
        fixture=ROOT / "tests/support/mcp_owned_browser_server.py",
        extra_env={
            "OPENBROWSER_TEST_BARRIERS": str(barriers),
            "OPENBROWSER_TEST_OWNED_MODE": "reset-failure",
        },
    ) as c:
        created = await c._raw_request("ctx", "tools/call", {"name": "create_context", "arguments": {}})
        context_id = created["structuredContent"]["context_id"]
        await c._raw_request(
            "exec",
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"context_id": context_id, "code": "answer = 42"},
            },
        )
        (barriers / "release-1").touch()
        first = await c._raw_request(
            "close-1",
            "tools/call",
            {"name": "close_context", "arguments": {"context_id": context_id}},
        )
        assert first["isError"] is True
        assert first["structuredContent"] == json.loads(first["content"][0]["text"])
        assert first["structuredContent"] == {
            "context_id": context_id,
            "status": "close_failed",
            "error_code": "cleanup_failed",
            "retryable": True,
        }
        blocked = await c._raw_request(
            "status-1",
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": context_id}},
        )
        assert blocked["isError"] is True
        assert "context_close_failed" in blocked["content"][0]["text"]

        second = await c._raw_request(
            "close-2",
            "tools/call",
            {"name": "close_context", "arguments": {"context_id": context_id}},
        )
        assert second["isError"] is False
        assert second["structuredContent"] == json.loads(second["content"][0]["text"])
        assert second["structuredContent"]["status"] == "closed"
        missing = await c._raw_request(
            "status-2",
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": context_id}},
        )
        assert missing["isError"] is True
        assert "context_not_found" in missing["content"][0]["text"]
        await c.finish()


@pytest.mark.asyncio
async def test_context_errors_do_not_leak_handles_to_telemetry(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", True)
    server = module.OpenBrowserServer()
    events = []
    server._telemetry = SimpleNamespace(capture=events.append)
    handler = server.server.get_request_handler("tools/call").handler
    result = await handler(
        SimpleNamespace(protocol_version="2026-07-28"),
        module.types.CallToolRequestParams(name="get_status", arguments={"context_id": "secret-handle"}),
    )
    assert result.is_error
    assert len(events) == 1
    assert events[0].error_message == "context_not_found"
    assert "secret-handle" not in str(events[0])


@pytest.mark.asyncio
async def test_unknown_tool_telemetry_redacts_caller_name(monkeypatch):
    from types import SimpleNamespace

    from mcp import MCPError

    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", True)
    server = module.OpenBrowserServer()
    events = []
    server._telemetry = SimpleNamespace(capture=events.append)
    handler = server.server.get_request_handler("tools/call").handler
    bad_name = "Authorization: Bearer telemetry-secret"

    with pytest.raises(MCPError):
        await handler(
            SimpleNamespace(protocol_version="2026-07-28"),
            module.types.CallToolRequestParams(name=bad_name, arguments={}),
        )

    assert len(events) == 1
    event = events[0]
    assert event.error_message == "unknown_tool"
    assert event.tool_name == "unknown_tool"
    assert "telemetry-secret" not in str(event)


@pytest.mark.asyncio
async def test_real_stdio_invalid_config_reports_tool_error(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("NOT JSON {{{")
    async with wire(
        tmp_path,
        extra_env={"OPENBROWSER_CONFIG_PATH": str(config_path)},
    ) as c:
        created = await c.request(
            1,
            "tools/call",
            {"name": "create_context", "arguments": {}},
        )
        context_id = created["structuredContent"]["context_id"]
        status = await c.request(
            2,
            "tools/call",
            {"name": "get_status", "arguments": {"context_id": context_id}},
        )
        payload = json.loads(status["content"][0]["text"])
        assert payload["config"]["startup"]["phase"] == "invalid"
        assert payload["config"]["current"]["phase"] == "invalid"
        assert payload["config"]["restart_required"] is True

        failure = await c.request(
            3,
            "tools/call",
            {
                "name": "execute_code",
                "arguments": {"context_id": context_id, "code": "print(1)"},
            },
        )
        assert failure["isError"] is True
        text = failure["content"][0]["text"]
        assert "config_restart_required" in text
        assert str(config_path) in text
        assert config_path.read_text() == "NOT JSON {{{"
        await c.finish()


@pytest.mark.asyncio
async def test_config_errors_do_not_leak_paths_to_telemetry(isolated_config_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from openbrowser.config_lifecycle import ConfigStartupError

    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", True)
    server = module.OpenBrowserServer()
    events = []
    server._telemetry = SimpleNamespace(capture=events.append)
    context_id = await server._contexts.create()
    owner = server._contexts._entries[context_id].owner
    owner._ensure_namespace = AsyncMock(
        side_effect=ConfigStartupError(
            server._startup_config_result.snapshot,
            code="config_restart_required",
            hint="Fix the config, then restart the daemon or MCP server.",
        )
    )
    handler = server.server.get_request_handler("tools/call").handler
    result = await handler(
        SimpleNamespace(protocol_version="2026-07-28"),
        module.types.CallToolRequestParams(
            name="execute_code",
            arguments={"context_id": context_id, "code": "print(1)"},
        ),
    )

    assert result.is_error
    assert len(events) == 1
    assert events[0].error_message == "config_restart_required"
    assert str(isolated_config_path) not in str(events[0])
    await server._contexts.close(context_id)
