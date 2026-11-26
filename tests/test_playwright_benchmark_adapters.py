"""Offline contracts for the native Playwright MCP and CLI benchmark arms.

No native process, browser, profile, or network is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from benchmarks.interactive.adapters import playwright_cli, playwright_mcp
from benchmarks.interactive.adapters.base import RunContext
from benchmarks.interactive.adapters.playwright_common import (
    SECRETS_ENV,
    last_json,
    wait_for_download,
    write_secrets,
)
from benchmarks.interactive.adapters.playwright_tasks import (
    SCRAPING_TASKS,
    SECRET_PASSWORD,
    login_program,
    task_program,
)
from benchmarks.interactive.fixtures.server import BENCH_PASSWORD, BENCH_USER, REPORT_BYTES


def make_context(tmp_path: Path, *, task: str = "wizard") -> RunContext:
    root = tmp_path / "run"
    for name in ("downloads", "profile", "home", "tmp", "config"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return RunContext(
        task=task,
        arm="playwright",
        sample=0,
        base_url="http://127.0.0.1:9999",
        root=root,
        downloads=root / "downloads",
        profile=root / "profile",
        home=root / "home",
        tmp=root / "tmp",
        config=root / "config",
        session="pw-test",
    )


@pytest.mark.parametrize("task", ["wizard", "modal", "download", "tabs", "auth"])
def test_interactive_programs_are_async_functions_without_secrets(task):
    code = task_program(task, "http://127.0.0.1:9999", None)
    assert code.startswith("async (page) => {")
    assert "process.env" not in code
    assert BENCH_PASSWORD not in code
    assert "SECRET_BENCH_PASSWORD" not in code


@pytest.mark.parametrize("task", SCRAPING_TASKS)
def test_scraping_programs_target_public_fixtures(task):
    code = task_program(task, "http://127.0.0.1:9999", None)
    assert code.startswith("async (page) => {")
    assert "process.env" not in code


def test_task_program_routes_and_unknown_task():
    assert "/wizard" in task_program("wizard", "http://x", None)
    assert "#download-report" in task_program("download", "http://x", None)
    assert "waitForEvent('popup')" in task_program("tabs", "http://x", None)
    assert "#secret" in task_program("auth", "http://x", None)
    assert "/static" in task_program("static_article", "http://x", None)
    assert "/js-list" in task_program("js_list", "http://x", None)
    assert "/pagination?page=1" in task_program("pagination", "http://x", None)
    assert "/table" in task_program("table", "http://x", None)
    assert "/assets" in task_program("asset_heavy", "http://x", None)
    with pytest.raises(ValueError):
        task_program("unknown", "http://x", None)


def test_login_program_uses_username_only():
    code = login_program("http://x")
    assert BENCH_USER in code
    assert BENCH_PASSWORD not in code
    assert SECRET_PASSWORD not in code


def test_last_json_and_missing_result():
    assert last_json('### Result\n{"quote": "Q-1"}') == {"quote": "Q-1"}
    with pytest.raises(ValueError):
        last_json("### Result\nnot json")


def test_secrets_file_permissions(tmp_path):
    ctx = make_context(tmp_path)
    path = write_secrets(ctx)
    assert f"BENCH_PASSWORD={BENCH_PASSWORD}" in path.read_text()
    assert (path.stat().st_mode & 0o777) == 0o600


def test_wait_for_download_requires_exact_bytes(tmp_path):
    directory = tmp_path / "downloads"
    directory.mkdir()
    (directory / "wrong.csv").write_bytes(b"wrong")
    (directory / "report.csv").write_bytes(REPORT_BYTES)
    assert wait_for_download(directory, timeout=1.0).name == "report.csv"
    (directory / "report.csv").unlink()
    with pytest.raises(TimeoutError):
        wait_for_download(directory, timeout=0.2)


class FakeMcpClient:
    instances: ClassVar[list[FakeMcpClient]] = []

    def __init__(self, command, *, env=None, timeout=120.0):
        self.command = list(command)
        self.env = dict(env or {})
        self.timeout = timeout
        self.calls: list[tuple[str, dict]] = []
        self.stderr_lines: list[str] = []
        self.closed = False
        FakeMcpClient.instances.append(self)

    def initialize(self):
        self.calls.append(("initialize", {}))
        return {}

    def call_tool(self, name, arguments, *, timeout=None):
        self.calls.append((name, dict(arguments)))
        if name == "browser_run_code_unsafe":
            code = arguments.get("code", "")
            if "navigator.userAgent" in code:
                return {"content": [{"type": "text", "text": '{"browser_version": "TestUA"}'}]}
            if "#secret" in code and "textContent" in code:
                return {"content": [{"type": "text", "text": '{"secret": "bench-secret"}'}]}
            if "#quote" in code:
                return {"content": [{"type": "text", "text": '{"quote": "Q-1"}'}]}
            return {"content": [{"type": "text", "text": '{"ok": true}'}]}
        return {"content": [{"type": "text", "text": ""}]}

    def close(self):
        self.closed = True


def test_mcp_server_command_and_execute_program(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    FakeMcpClient.instances.clear()
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: True)

    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    result = arm.execute_program(ctx, 'async (page) => { return {quote: "Q-1"}; }')
    assert result.ok is True
    assert result.reported == {"ok": True} or result.reported == {"quote": "Q-1"}
    client = FakeMcpClient.instances[0]
    assert client.command[0] == "/bin/playwright-mcp"
    assert "--headless" in client.command
    assert "--user-data-dir" in client.command and str(ctx.profile) in client.command
    assert "--output-dir" in client.command and str(ctx.downloads) in client.command
    assert "--secrets" in client.command
    assert client.command[client.command.index("--codegen") + 1] == "none"
    assert client.command[client.command.index("--snapshot-mode") + 1] == "none"
    arm.cleanup(ctx)
    assert client.closed is True


def test_mcp_auth_uses_native_secret_fill_and_restart(tmp_path, monkeypatch):
    ctx = make_context(tmp_path, task="auth")
    FakeMcpClient.instances.clear()
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: True)

    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    result = arm.run(ctx, cleanup=True, restart_for_auth=True)
    assert result.ok is True
    assert result.reported.get("secret") == "bench-secret"
    first = FakeMcpClient.instances[0]
    names = [name for name, _ in first.calls]
    assert names.count("browser_run_code_unsafe") == 2  # login, ready
    fill = next(arguments for name, arguments in first.calls if name == "browser_type")
    assert fill["text"] == SECRET_PASSWORD
    assert fill["target"] == "#password"
    assert fill["submit"] is True
    assert len(FakeMcpClient.instances) == 2  # restart started a new server
    assert any(name == "browser_close" for name, _ in FakeMcpClient.instances[1].calls) or first.closed is True


def test_mcp_warmup_payload(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    FakeMcpClient.instances.clear()
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    payload, elapsed = arm.warmup(ctx)
    assert payload["browser_version"] == "TestUA"
    assert payload["server"] == "playwright-mcp"
    assert elapsed >= 0


class FakeProcess:
    @classmethod
    def run(cls, command, *, env=None, cwd=None, input_text=None, timeout=120.0):
        cls.commands.append(list(command))
        cls.envs.append(dict(env or {}))
        if "run-code" in command and "--filename" in command:
            path = Path(command[command.index("--filename") + 1])
            code = path.read_text()
            if "navigator.userAgent" in code:
                return 0, '{"browser_version": "TestUA"}', "", 0.1
            if "#secret" in code and "textContent" in code:
                return 0, '{"secret": "bench-secret"}', "", 0.1
            if "#quote" in code:
                return 0, '{"quote": "Q-1"}', "", 0.1
            return 0, '{"ok": true}', "", 0.1
        return 0, "", "", 0.1

    commands: ClassVar[list[list[str]]] = []
    envs: ClassVar[list[dict]] = []


def test_cli_commands_and_execute_program(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    FakeProcess.commands.clear()
    FakeProcess.envs.clear()
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    monkeypatch.setattr(playwright_cli, "keychain", lambda ctx: True)

    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    result = arm.execute_program(ctx, 'async (page) => { return {quote: "Q-1"}; }')
    assert result.ok is True
    open_command = FakeProcess.commands[0]
    assert open_command[:3] == ["/bin/playwright-cli", "-s", "pw-test"]
    assert open_command[3] == "open"
    assert "--profile" in open_command and str(ctx.profile) in open_command
    assert "--config" in open_command
    code_commands = [command for command in FakeProcess.commands if "run-code" in command]
    assert code_commands and "--filename" in code_commands[0]
    assert code_commands[0][-1] == "--raw"
    assert FakeProcess.envs[0][SECRETS_ENV].endswith("playwright-secrets.env")
    config = json.loads((ctx.config / "playwright-cli.json").read_text())
    assert config["outputDir"] == str(ctx.downloads)
    assert config["codegen"] == "none"


def test_cli_auth_uses_secret_fill_and_restart(tmp_path, monkeypatch):
    ctx = make_context(tmp_path, task="auth")
    FakeProcess.commands.clear()
    FakeProcess.envs.clear()
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    monkeypatch.setattr(playwright_cli, "keychain", lambda ctx: True)

    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    result = arm.run(ctx, cleanup=True, restart_for_auth=True)
    assert result.ok is True
    assert result.reported.get("secret") == "bench-secret"
    fill = next(command for command in FakeProcess.commands if "fill" in command)
    assert fill[fill.index("fill") + 1] == "#password"
    assert fill[fill.index("fill") + 2] == SECRET_PASSWORD
    assert "--submit" in fill
    assert any(command[3] == "close" for command in FakeProcess.commands if len(command) > 3)
    opens = [command for command in FakeProcess.commands if len(command) > 3 and command[3] == "open"]
    assert len(opens) == 2  # initial and post-restart


def test_cli_cleanup_closes_session(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    FakeProcess.commands.clear()
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    arm._open(ctx)
    arm.cleanup(ctx)
    assert any(command[3] == "close" for command in FakeProcess.commands if len(command) > 3)


def test_secret_leak_guard_rejects_literal():
    from benchmarks.interactive.adapters.playwright_common import assert_no_secret

    assert_no_secret("clean native output")
    with pytest.raises(RuntimeError, match="leaked"):
        assert_no_secret("clean", f"value {BENCH_PASSWORD}")


def test_mcp_run_fails_when_native_output_leaks_password(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)

    class LeakyClient(FakeMcpClient):
        def call_tool(self, name, arguments, *, timeout=None):
            if name == "browser_run_code_unsafe":
                return {"content": [{"type": "text", "text": json.dumps({"quote": BENCH_PASSWORD})}]}
            return {"content": [{"type": "text", "text": ""}]}

    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", LeakyClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: True)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    result = arm.run(ctx, cleanup=True)
    assert result.ok is False
    assert "leaked" in result.error
