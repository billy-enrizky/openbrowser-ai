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


def child_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "PYTHONPATH": str(ROOT / "src"),
        "OPENBROWSER_ANONYMIZED_TELEMETRY": "false",
        "ANONYMIZED_TELEMETRY": "false",
    }


class Wire:
    def __init__(self, process):
        self.process = process
        self.stderr_task = asyncio.create_task(process.stderr.read())

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

    async def request(self, ident, method, params=None, *, modern=True):
        await self.send(ident, method, params, modern=modern)
        message = await self.receive(ident)
        assert "error" not in message, message
        return message["result"]

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
    args = (
        [str(fixture or FIXTURE)]
        if browser_fixture or fixture
        else ["-m", "openbrowser.mcp"]
    )
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
            "execute_code",
            "get_status",
            "reset_browser",
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
        assert len(catalog.tools) == 3
        status = await client.call_tool("get_status", {})
        assert not status.is_error
        assert json.loads(status.content[0].text)["browser"]["phase"] == "stopped"
        if mode == "auto":
            assert client.server_info.name == "openbrowser"
            caps = client.server_capabilities
            assert caps.tools.list_changed is False
            assert caps.completions is None


@pytest.mark.asyncio
async def test_modern_first_call_and_catalog_cache_metadata(tmp_path):
    async with wire(tmp_path) as c:
        status = await c.request(
            1, "tools/call", {"name": "get_status", "arguments": {}}
        )
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
async def test_schema_failures_are_tool_errors_before_browser_effects(
    tmp_path, name, arguments
):
    async with wire(tmp_path) as c:
        result = await c.request(
            1, "tools/call", {"name": name, "arguments": arguments}
        )
        assert result["isError"] is True
        status = await c.request(
            2, "tools/call", {"name": "get_status", "arguments": {}}
        )
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
        assert len(catalog["tools"]) == 3
        assert not {"ttlMs", "cacheScope", "resultType"} & catalog.keys()
        status = await c.request(
            3, "tools/call", {"name": "get_status", "arguments": {}}, modern=False
        )
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
        await c.send(
            1,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": "await wait_for_cancel()"}},
        )
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


@pytest.mark.asyncio
async def test_active_eof_stops_real_protocol_process(tmp_path):
    async with wire(tmp_path, browser_fixture=True) as c:
        await c.send(
            1,
            "tools/call",
            {"name": "execute_code", "arguments": {"code": "await wait_for_cancel()"}},
        )
        await c.request(2, "tools/call", {"name": "get_status"})
        await c.finish(pending_ids=(1,))


@pytest.mark.asyncio
async def test_catalog_is_snapshot_and_cannot_mutate_validation(monkeypatch):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    monkeypatch.setenv("OPENBROWSER_COMPACT_DESCRIPTION", "true")
    server = module.OpenBrowserServer()
    async with Client(server.server) as client:
        catalog = await client.list_tools(cache_mode="refresh")
        original = catalog.tools[0].description
        catalog.tools[0].input_schema["properties"]["code"]["type"] = "number"
        monkeypatch.delenv("OPENBROWSER_COMPACT_DESCRIPTION")
        fresh = await client.list_tools(cache_mode="refresh")
        assert (
            fresh.tools[0].description
            == original
            == module._EXECUTE_CODE_DESCRIPTION_COMPACT
        )
        assert fresh.tools[0].input_schema["properties"]["code"]["type"] == "string"
        result = await client.call_tool("execute_code", {"code": 5})
        assert result.is_error
        assert server.browser_session is None


@pytest.mark.asyncio
async def test_actual_sdk_126_client_uses_app_legacy_stdio(tmp_path):
    legacy_python = os.environ.get("OPENBROWSER_TEST_MCP1_PYTHON")
    if not legacy_python:
        if os.environ.get("CI"):
            pytest.fail(
                "CI must configure the isolated mcp==1.26.0 compatibility interpreter"
            )
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
    async with Client(server.server) as client:
        result = await client.call_tool("get_status", {})
        assert result.is_error
        assert result.content[0].text == "Error: dispatcher failed"


@pytest.mark.asyncio
async def test_exact_catalog_input_contracts_and_annotations(monkeypatch):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    async with Client(server.server) as client:
        catalog = await client.list_tools()
    execute, status, reset = catalog.tools
    assert execute.input_schema == {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. All browser functions are async (use await).",
            }
        },
        "required": ["code"],
    }
    assert status.input_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert reset.input_schema == {
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
    assert [
        tool.annotations.model_dump(by_alias=True, exclude_none=True)
        for tool in catalog.tools
    ] == [
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
    ]


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
@pytest.mark.parametrize(
    "mode", ["reset", "cancel-start", "start-failure", "replacement-failure"]
)
async def test_owned_cleanup_cancellation_preserves_real_stdio_usability(
    tmp_path, modern, mode
):
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
        await c.send(2, "tools/call", operation, modern=modern)
        if mode == "cancel-start":
            await wait_for_barrier(barriers / "start-1")
            await c.send(
                None, "notifications/cancelled", {"requestId": 2}, modern=modern
            )
        blocked_owner = 2 if mode == "replacement-failure" else 1
        await wait_for_barrier(barriers / f"kill-{blocked_owner}")
        if mode != "cancel-start":
            await c.send(
                None, "notifications/cancelled", {"requestId": 2}, modern=modern
            )
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
                "arguments": {
                    "code": "print(browser.ident)\nprint(globals().get('answer', 'missing'))"
                },
            },
            modern=modern,
        )
        text = following["content"][0]["text"]
        assert str(blocked_owner + 1) in text
        assert ("42" if mode == "reset" else "missing") in text
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
            line.removeprefix("OWNED_FINAL ")
            for line in stderr.splitlines()
            if line.startswith("OWNED_FINAL ")
        )
        state = json.loads(final)
        assert state["owner"] is None
        assert not state["execution_locked"] and not state["lifecycle_locked"]
        assert blocked_owner in state["killed"]


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
        await c.send(2, "tools/call", {"name": "reset_browser"})
        await wait_for_barrier(barriers / "kill-1")
        c.process.stdin.close()
        (barriers / "release-1").touch()
        stderr = await c.finish(pending_ids=(2,))
        final = next(
            line.removeprefix("OWNED_FINAL ")
            for line in stderr.splitlines()
            if line.startswith("OWNED_FINAL ")
        )
        assert json.loads(final) == {
            "owner": None,
            "execution_locked": False,
            "lifecycle_locked": False,
            "killed": [1],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [True, False], ids=["modern", "legacy"])
@pytest.mark.parametrize(
    "mode", ["reset-failure", "reset-child-cancel", "reset-both-cancel"]
)
async def test_failed_owned_cleanup_retains_owner_and_cancellation_keeps_transport_usable(
    tmp_path, modern, mode
):
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
        await c.send(2, "tools/call", {"name": "reset_browser"}, modern=modern)
        await wait_for_barrier(barriers / "kill-1")
        if mode != "reset-child-cancel":
            await c.send(
                None, "notifications/cancelled", {"requestId": 2}, modern=modern
            )
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
        reset = await c.request(
            6, "tools/call", {"name": "reset_browser"}, modern=modern
        )
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
            line.removeprefix("OWNED_FINAL ")
            for line in stderr.splitlines()
            if line.startswith("OWNED_FINAL ")
        )
        assert json.loads(final) == {
            "owner": None,
            "execution_locked": False,
            "lifecycle_locked": False,
            "killed": [1, 2],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [True, False], ids=["modern", "legacy"])
@pytest.mark.parametrize(
    "stage", ["initial-start", "replacement-start", "health", "executor"]
)
async def test_owned_tool_self_cancellation_is_error_without_losing_request_transport(
    tmp_path, modern, stage
):
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
        assert state["browser"]["phase"] == (
            "failed" if stage in ("initial-start", "replacement-start") else "ready"
        )
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
                "arguments": {
                    "code": "print('answer' in globals())\nprint('partial' in globals())"
                },
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
            line.removeprefix("OWNED_FINAL ")
            for line in stderr.splitlines()
            if line.startswith("OWNED_FINAL ")
        )
        assert json.loads(final)["owner"] is None
        assert (
            not json.loads(final)["execution_locked"]
            and not json.loads(final)["lifecycle_locked"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [True, False], ids=["modern", "legacy"])
async def test_executor_timeout_keeps_existing_partial_state_and_transport_usable(
    tmp_path, modern
):
    async with wire(
        tmp_path, browser_fixture=True, extra_env={"OPENBROWSER_MCP_EXEC_TIMEOUT": "1"}
    ) as c:
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
