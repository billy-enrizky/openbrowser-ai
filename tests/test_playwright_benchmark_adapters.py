"""Unit contracts for the native Playwright MCP and CLI benchmark arms.

No browser, profile, socket, or native process is used. Constructors and the
Chrome version probe are replaced with metadata fakes by an autouse guard.
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

from benchmarks.interactive.adapters import playwright_cli, playwright_common, playwright_mcp
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


@pytest.fixture(autouse=True)
def _no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fake test attempted native I/O")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(playwright_mcp, "chrome_version", lambda: "Guarded/1.0")
    monkeypatch.setattr(playwright_cli, "chrome_version", lambda: "Guarded/1.0")
    monkeypatch.setattr(playwright_mcp, "terminate_matching", lambda marker: [])
    monkeypatch.setattr(playwright_cli, "terminate_matching", lambda marker: [])
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: True)
    monkeypatch.setattr(playwright_cli, "keychain", lambda ctx: True)


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
    assert f"SECRET_BENCH_PASSWORD={BENCH_PASSWORD}" in path.read_text()
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
        self.sequence: list[tuple[str, str]] = []
        self.stderr_lines: list[str] = []
        self.closed = False
        FakeMcpClient.instances.append(self)

    def initialize(self):
        self.calls.append(("initialize", {}))
        return {}

    def call_tool(self, name, arguments, *, timeout=None):
        self.calls.append((name, dict(arguments)))
        self.sequence.append(("tool", name))
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
        self.sequence.append(("close", ""))


def test_mcp_server_command_and_execute_program(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    FakeMcpClient.instances.clear()
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: True)

    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    arm.warmup(ctx)
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
    assert client.command[client.command.index("--viewport-size") + 1] == "1280x800"
    config = json.loads((ctx.config / "playwright-config.json").read_text())
    assert config["browser"]["launchOptions"]["executablePath"] == playwright_common.matched_chrome()
    assert config["browser"]["launchOptions"]["args"] == ["--use-mock-keychain"]
    assert config["browser"]["launchOptions"]["headless"] is True
    assert config["browser"]["launchOptions"]["downloadsPath"] == str(ctx.downloads)
    assert config["browser"]["contextOptions"]["viewport"] == {"width": 1280, "height": 800}
    assert config["outputDir"] == str(ctx.downloads)
    assert config["codegen"] == "none"
    assert config["timeouts"]["settle"] == 0
    arm.cleanup(ctx)
    assert client.closed is True


def test_shared_config_sets_zero_settle_for_both_native_paths(tmp_path):
    ctx = make_context(tmp_path)
    mcp = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    cli = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    command = mcp._server_command(ctx)
    mcp_config = Path(command[command.index("--config") + 1])
    cli_config = cli._config(ctx)
    assert mcp_config == cli_config
    config = json.loads(cli_config.read_text())
    assert config["timeouts"]["settle"] == 0
    assert "browser" in config and "contextOptions" in config["browser"]


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
    assert names.count("browser_run_code_unsafe") == 3  # login, ready, account
    fill = next(arguments for name, arguments in first.calls if name == "browser_type")
    assert fill["text"] == SECRET_PASSWORD
    assert fill["target"] == "#password"
    assert fill["submit"] is True
    assert len(FakeMcpClient.instances) == 1
    assert result.process_starts == 1
    assert names.count("browser_close") == 2  # restart and final teardown
    assert first.closed is True


def test_mcp_warmup_payload(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    FakeMcpClient.instances.clear()
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    payload, elapsed = arm.warmup(ctx)
    assert payload["user_agent"] == "TestUA"
    assert payload["server"] == "playwright-mcp"
    assert payload["keychain_verified"] is True
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
    arm.warmup(ctx)
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
    config = json.loads((ctx.config / "playwright-config.json").read_text())
    assert config["outputDir"] == str(ctx.downloads)
    assert config["codegen"] == "none"
    assert config["timeouts"]["settle"] == 0
    assert config["browser"]["launchOptions"]["executablePath"] == playwright_common.matched_chrome()
    assert config["browser"]["launchOptions"]["args"] == ["--use-mock-keychain"]
    assert config["browser"]["contextOptions"]["viewport"] == {"width": 1280, "height": 800}


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


def test_mcp_initialize_failure_closes_partial_server(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)

    class FailingInitClient(FakeMcpClient):
        def initialize(self):
            raise RuntimeError("initialize failed")

    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FailingInitClient)
    FailingInitClient.instances.clear()
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    with pytest.raises(RuntimeError):
        arm._start(ctx)
    assert FailingInitClient.instances[0].closed is True
    assert arm._client is None


def test_mcp_cleanup_is_idempotent_and_ctx_scoped(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    markers: list[str] = []
    monkeypatch.setattr(playwright_mcp, "terminate_matching", lambda marker: markers.append(marker) or [])
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    arm.cleanup(ctx)
    arm.cleanup(ctx)
    assert markers == [str(ctx.profile), str(ctx.downloads), str(ctx.profile), str(ctx.downloads)]


def test_cli_cleanup_closes_after_partial_open_and_is_idempotent(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    FakeProcess.commands.clear()
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    monkeypatch.setattr(playwright_cli, "terminate_matching", lambda marker: [])
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    arm._attempted_open = True
    arm.cleanup(ctx)
    closes = [command for command in FakeProcess.commands if len(command) > 3 and command[3] == "close"]
    assert len(closes) == 1
    arm.cleanup(ctx)
    closes = [command for command in FakeProcess.commands if len(command) > 3 and command[3] == "close"]
    assert len(closes) == 1


def test_modal_program_waits_for_hidden_state():
    code = task_program("modal", "http://x", None)
    assert "state: 'hidden'" in code
    assert "#modal[hidden]" not in code


def test_arms_expose_pinned_browser_version(monkeypatch):
    monkeypatch.setattr(playwright_mcp, "chrome_version", lambda: "TestChrome/1.0")
    monkeypatch.setattr(playwright_cli, "chrome_version", lambda: "TestChrome/1.0")
    mcp = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    cli = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    assert mcp.browser_version == "TestChrome/1.0"
    assert cli.browser_version == "TestChrome/1.0"


def test_mcp_run_sets_browser_version(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    monkeypatch.setattr(playwright_mcp, "chrome_version", lambda: "TestChrome/1.0")
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: True)
    FakeMcpClient.instances.clear()
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    result = arm.run(ctx, cleanup=False)
    assert result.browser_version == "TestChrome/1.0"


def test_mcp_archives_stderr_before_cleanup_and_fails_leak(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)

    class StderrLeakClient(FakeMcpClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stderr_lines.append(f"server logged {BENCH_PASSWORD}")

        def call_tool(self, name, arguments, *, timeout=None):
            if name == "browser_run_code_unsafe":
                return {"content": [{"type": "text", "text": '{"quote": "Q-1"}'}]}
            return {"content": [{"type": "text", "text": ""}]}

    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", StderrLeakClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: True)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    result = arm.run(ctx, cleanup=True)
    assert BENCH_PASSWORD in result.stderr
    assert result.ok is False
    assert "leaked" in result.error


def test_mcp_initialize_failure_archives_stderr(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)

    class FailingInitClient(FakeMcpClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stderr_lines.append("startup failure trace")

        def initialize(self):
            raise RuntimeError("initialize failed")

    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FailingInitClient)
    FailingInitClient.instances.clear()
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    with pytest.raises(RuntimeError):
        arm._start(ctx)
    assert "startup failure trace" in arm._result.stderr


def test_explicit_browser_version_skips_native_probe(monkeypatch):
    def boom():
        raise AssertionError("probe must not run when a preflight version is supplied")

    monkeypatch.setattr(playwright_mcp, "chrome_version", boom)
    monkeypatch.setattr(playwright_cli, "chrome_version", boom)
    mcp = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp", browser_version="Preflight/1.0")
    cli = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli", browser_version="Preflight/1.0")
    assert mcp.browser_version == "Preflight/1.0"
    assert cli.browser_version == "Preflight/1.0"


def test_browser_version_env_preflight_wins(monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BENCH_BROWSER_VERSION", "EnvPreflight/2.0")

    def boom():
        raise AssertionError("probe must not run when the env preflight is set")

    monkeypatch.setattr(playwright_common, "_VERSION_CACHE", {})
    monkeypatch.setattr(playwright_common.subprocess, "run", lambda *a, **k: boom())
    assert playwright_common.chrome_version() == "EnvPreflight/2.0"


class FakeCompleted:
    stdout = "Google Chrome 999.0.0.0\n"
    stderr = ""


def test_chrome_version_uses_injected_metadata_and_caches(monkeypatch):
    monkeypatch.delenv("PLAYWRIGHT_BENCH_BROWSER_VERSION", raising=False)
    monkeypatch.setattr(playwright_common, "_VERSION_CACHE", {})
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return FakeCompleted()

    monkeypatch.setattr(playwright_common.subprocess, "run", fake_run)
    assert playwright_common.chrome_version() == "Google Chrome 999.0.0.0"
    assert playwright_common.chrome_version() == "Google Chrome 999.0.0.0"
    assert len(calls) == 1


def test_cli_process_starts_counts_every_native_invocation(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    FakeProcess.commands.clear()
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    arm._open(ctx)
    assert arm._result.process_starts == 1
    arm._code(ctx, "async (page) => { return {ok: true}; }")
    assert arm._result.process_starts == 2
    arm._run(ctx, "close")
    assert arm._result.process_starts == 3


def test_mcp_restart_closes_browser_before_transport(tmp_path, monkeypatch):
    ctx = make_context(tmp_path, task="auth")
    FakeMcpClient.instances.clear()
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: True)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    arm._start(ctx)
    arm._restart(ctx)
    assert arm._result.tool_calls == 1  # the in-task browser_close is a real native call
    first = FakeMcpClient.instances[0]
    assert ("tool", "browser_close") in first.sequence
    assert len(FakeMcpClient.instances) == 1
    assert first.closed is False
    assert arm._client is first
    arm.cleanup(ctx)
    assert first.sequence.index(("tool", "browser_close")) < first.sequence.index(("close", ""))
    assert len(FakeMcpClient.instances) == 1


@pytest.mark.parametrize(
    ("task", "key"),
    [
        ("static_article", "content"),
        ("js_list", "skus"),
        ("pagination", "ids"),
        ("table", "rows"),
        ("asset_heavy", "content"),
    ],
)
def test_scraping_program_reports_oracle_key(task, key):
    code = task_program(task, "http://x", None)
    assert f"{key}" in code.split("return", 1)[1]


def test_mcp_warmup_caches_keychain_for_warm_runs(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: calls.append(1) or True)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    arm.warmup(ctx)
    assert calls == [1]
    first = arm.run(ctx, cleanup=False)
    second = arm.run(ctx, cleanup=False)
    assert first.ok is True and first.keychain_verified is True
    assert second.ok is True and second.keychain_verified is True
    assert calls == [1]
    arm.cleanup(ctx)
    assert arm._keychain_verified is None


def test_cli_warmup_caches_keychain_for_warm_runs(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    monkeypatch.setattr(playwright_cli, "keychain", lambda ctx: calls.append(1) or True)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    arm.warmup(ctx)
    assert calls == [1]
    first = arm.run(ctx, cleanup=False)
    second = arm.run(ctx, cleanup=False)
    assert first.ok is True and first.keychain_verified is True
    assert second.ok is True and second.keychain_verified is True
    assert calls == [1]
    arm.cleanup(ctx)
    assert arm._keychain_verified is None


@pytest.mark.parametrize("value", [False, None])
def test_mcp_warmup_rejects_failed_or_unknown_keychain(tmp_path, monkeypatch, value):
    ctx = make_context(tmp_path)
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: value)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    with pytest.raises(RuntimeError, match="keychain"):
        arm.warmup(ctx)
    assert arm._keychain_verified is None


@pytest.mark.parametrize("value", [False, None])
def test_cli_warmup_rejects_failed_or_unknown_keychain(tmp_path, monkeypatch, value):
    ctx = make_context(tmp_path)
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    monkeypatch.setattr(playwright_cli, "keychain", lambda ctx: value)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    with pytest.raises(RuntimeError, match="keychain"):
        arm.warmup(ctx)
    assert arm._keychain_verified is None


def test_mcp_cold_run_still_scans_keychain(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: calls.append(1) or True)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    result = arm.run(ctx, cleanup=False)
    assert result.ok is True and result.keychain_verified is True
    assert calls == [1]


def test_cli_cold_run_still_scans_keychain(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    monkeypatch.setattr(playwright_cli, "keychain", lambda ctx: calls.append(1) or True)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    result = arm.run(ctx, cleanup=False)
    assert result.ok is True and result.keychain_verified is True
    assert calls == [1]


def test_mcp_restart_invalidates_cached_keychain(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    monkeypatch.setattr(playwright_mcp, "keychain", lambda ctx: calls.append(1) or True)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    arm.warmup(ctx)
    assert calls == [1]
    arm._restart(ctx)
    assert arm._keychain_verified is None
    result = arm.run(ctx, cleanup=False)
    assert result.ok is True
    assert calls == [1, 1]


def test_cli_restart_invalidates_cached_keychain(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    monkeypatch.setattr(playwright_cli, "keychain", lambda ctx: calls.append(1) or True)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    arm.warmup(ctx)
    assert calls == [1]
    arm._restart(ctx)
    assert arm._keychain_verified is None
    result = arm.run(ctx, cleanup=False)
    assert result.ok is True
    assert calls == [1, 1]


def test_mcp_execute_program_requires_verified_launch(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    result = arm.execute_program(ctx, "async (page) => { return {ok: true}; }")
    assert result.ok is False
    assert "readiness" in result.error


def test_cli_execute_program_requires_verified_launch(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    monkeypatch.setattr(playwright_cli, "run_process", FakeProcess.run)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    result = arm.execute_program(ctx, "async (page) => { return {ok: true}; }")
    assert result.ok is False
    assert "readiness" in result.error


class StartupLeakClient(FakeMcpClient):
    def __init__(self, command, *, env=None, timeout=120.0):
        super().__init__(command, env=env, timeout=timeout)
        self.stderr_lines.append(f"startup debug {BENCH_PASSWORD}")


class ProgramStderrLeakClient(FakeMcpClient):
    def call_tool(self, name, arguments, *, timeout=None):
        result = super().call_tool(name, arguments, timeout=timeout)
        code = arguments.get("code", "")
        if name == "browser_run_code_unsafe" and "navigator.userAgent" not in code:
            self.stderr_lines.append(f"program debug {BENCH_PASSWORD}")
        return result


def test_mcp_warmup_rejects_startup_stderr_leak(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", StartupLeakClient)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    with pytest.raises(RuntimeError, match="leaked"):
        arm.warmup(ctx)


def test_cli_warmup_rejects_startup_stderr_leak(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)

    def leaky(command, *, env=None, cwd=None, input_text=None, timeout=120.0):
        if "run-code" in command and "--filename" in command:
            return 0, '{"browser_version": "TestUA"}', f"startup debug {BENCH_PASSWORD}", 0.1
        return 0, "", "", 0.1

    monkeypatch.setattr(playwright_cli, "run_process", leaky)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    with pytest.raises(RuntimeError, match="leaked"):
        arm.warmup(ctx)


def test_mcp_execute_program_rejects_stderr_leak(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", ProgramStderrLeakClient)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    arm.warmup(ctx)
    result = arm.execute_program(ctx, 'async (page) => { return {quote: "Q-1"}; }')
    assert result.ok is False
    assert "leaked" in result.error
    assert BENCH_PASSWORD in result.stderr


def test_cli_execute_program_rejects_stderr_leak(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)

    def leaky(command, *, env=None, cwd=None, input_text=None, timeout=120.0):
        if "run-code" in command and "--filename" in command:
            path = Path(command[command.index("--filename") + 1])
            code = path.read_text()
            if "navigator.userAgent" in code:
                return 0, '{"browser_version": "TestUA"}', "", 0.1
            return 0, '{"quote": "Q-1"}', f"program debug {BENCH_PASSWORD}", 0.1
        return 0, "", "", 0.1

    monkeypatch.setattr(playwright_cli, "run_process", leaky)
    arm = playwright_cli.PlaywrightCliArm(binary="/bin/playwright-cli")
    arm.warmup(ctx)
    result = arm.execute_program(ctx, 'async (page) => { return {quote: "Q-1"}; }')
    assert result.ok is False
    assert "leaked" in result.error
    assert BENCH_PASSWORD in result.stderr


def test_mcp_run_without_cleanup_rejects_stderr_leak(tmp_path, monkeypatch):
    ctx = make_context(tmp_path)
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", ProgramStderrLeakClient)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    arm.warmup(ctx)
    result = arm.run(ctx, cleanup=False)
    assert result.ok is False
    assert "leaked" in result.error
    assert BENCH_PASSWORD in result.stderr


@pytest.mark.parametrize("flag", [True, None, 0, "false"])
def test_mcp_restart_rejects_tool_error_or_malformed_error_flag(tmp_path, monkeypatch, flag):
    ctx = make_context(tmp_path, task="auth")
    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", FakeMcpClient)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    client = arm._start(ctx)
    assert isinstance(client, FakeMcpClient)
    monkeypatch.setattr(client, "call_tool", lambda *a, **kw: {"isError": flag, "content": []})
    with pytest.raises(RuntimeError, match="tool returned an error"):
        arm._restart(ctx)
    assert arm._keychain_verified is None
    assert client.closed is False
    assert arm._client is client


def test_fake_io_guard_cannot_be_swallowed_by_adapter():
    with pytest.raises(pytest.fail.Exception):
        subprocess.Popen(["forbidden-native-program"])


def test_mcp_tool_error_text_is_retained_and_secret_checked(tmp_path, monkeypatch):
    class ErrorClient(FakeMcpClient):
        def call_tool(self, name, arguments, *, timeout=None):
            return {"isError": True, "content": [{"type": "text", "text": BENCH_PASSWORD}]}

    monkeypatch.setattr(playwright_mcp, "StdioMcpClient", ErrorClient)
    arm = playwright_mcp.PlaywrightMcpArm(binary="/bin/playwright-mcp")
    result = arm.run(make_context(tmp_path), cleanup=False)
    assert result.ok is False
    assert "leaked" in result.error
    assert BENCH_PASSWORD in result.stdout
    assert BENCH_PASSWORD not in result.error
