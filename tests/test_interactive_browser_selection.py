"""Declared browser selection must be call-time and probe-free."""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import psutil
import pytest

from benchmarks.interactive.adapters import base, openbrowser, openbrowser_mcp


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("browser selection attempted native I/O")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)
    yield
    assert not attempts


def context(tmp_path: Path) -> base.RunContext:
    return base.RunContext(
        task="wizard",
        arm="openbrowser-fast",
        sample=0,
        base_url="http://127.0.0.1:0",
        root=tmp_path,
        downloads=tmp_path / "downloads",
        profile=tmp_path / "profile",
        home=tmp_path / "home",
        tmp=tmp_path / "tmp",
        config=tmp_path / "config",
        session="unit",
    )


def test_browser_resolver_reads_the_current_environment(monkeypatch):
    monkeypatch.setenv("CHROME_BIN", "/fake/engine one")
    assert base.matched_chrome() == "/fake/engine one"
    monkeypatch.setenv("CHROME_BIN", "/fake/engine two")
    assert base.matched_chrome() == "/fake/engine two"


def test_browser_resolver_preserves_the_unconfigured_default(monkeypatch):
    monkeypatch.delenv("CHROME_BIN", raising=False)
    assert base.matched_chrome() == base.DEFAULT_CHROME


@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_explicit_empty_browser_choice_is_rejected(monkeypatch, value):
    monkeypatch.setenv("CHROME_BIN", value)
    with pytest.raises(ValueError, match="CHROME_BIN"):
        base.matched_chrome()


def test_declared_browser_path_reaches_the_shared_config_model(monkeypatch, tmp_path):
    from openbrowser.config import CONFIG, DBStyleConfigJSON

    path = "/fake/browser cache/chrome-headless-shell"
    monkeypatch.setenv("CHROME_BIN", path)
    ctx = context(tmp_path)
    openbrowser._write_config(ctx, extensions=False)
    data = json.loads((ctx.config / "config.json").read_text())
    profile = data["browser_profile"]["bench"]
    assert profile["executable_path"] == path
    assert "channel" not in profile
    flat = CONFIG._get_default_profile(DBStyleConfigJSON.model_validate(data))
    assert flat["executable_path"] == path
    assert openbrowser_mcp._write_config is openbrowser._write_config


def test_unconfigured_profile_keeps_automatic_engine_selection(monkeypatch, tmp_path):
    monkeypatch.delenv("CHROME_BIN", raising=False)
    ctx = context(tmp_path)
    openbrowser._write_config(ctx, extensions=False)
    profile = json.loads((ctx.config / "config.json").read_text())["browser_profile"]["bench"]
    assert "executable_path" not in profile
    assert profile["channel"] == "chrome"


def test_blank_declared_browser_does_not_write_a_config(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROME_BIN", "")
    ctx = context(tmp_path)
    with pytest.raises(ValueError, match="CHROME_BIN"):
        openbrowser._write_config(ctx, extensions=False)
    assert not ctx.config.exists()


def test_agent_browser_command_uses_call_time_engine(monkeypatch, tmp_path):
    from benchmarks.interactive.adapters.agent_browser import AgentBrowserArm

    arm = AgentBrowserArm()
    ctx = context(tmp_path)
    for path in ("/fake/first-engine", "/fake/second-engine"):
        monkeypatch.setenv("CHROME_BIN", path)
        command = arm._command(ctx, "open", "about:blank")
        assert command[command.index("--executable-path") + 1] == path


def test_chrome_devtools_start_uses_call_time_engine(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from benchmarks.interactive.adapters import chrome_devtools

    commands = []

    def client(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(initialize=lambda: {"serverInfo": {"version": "fake"}})

    monkeypatch.setattr(chrome_devtools, "StdioMcpClient", client)
    monkeypatch.setattr(chrome_devtools, "verify_mock_keychain", lambda *args: True)
    monkeypatch.setattr(chrome_devtools, "_write_download_preferences", lambda *args: None)
    arm = chrome_devtools.ChromeDevToolsArm()
    monkeypatch.setattr(arm, "_pages", lambda: [(1, "about:blank")])
    for path in ("/fake/first-engine", "/fake/second-engine"):
        monkeypatch.setenv("CHROME_BIN", path)
        arm._start(context(tmp_path))
        assert commands[-1][commands[-1].index("--executablePath") + 1] == path


def test_browser_use_launch_uses_call_time_engine(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from benchmarks.interactive.adapters import browser_use

    paths = []

    def chrome(path, *args, **kwargs):
        paths.append(path)
        return SimpleNamespace(start=lambda: None)

    monkeypatch.setattr(browser_use, "ChromeInstance", chrome)
    monkeypatch.setattr(browser_use, "verify_mock_keychain", lambda *args: True)
    arm = browser_use.BrowserUseArm()
    for path in ("/fake/first-engine", "/fake/second-engine"):
        monkeypatch.setenv("CHROME_BIN", path)
        arm._start_chrome(context(tmp_path))
        assert paths[-1] == path


def test_both_playwright_configurations_use_call_time_engine(monkeypatch, tmp_path):
    from benchmarks.interactive.adapters import playwright_common

    ctx = context(tmp_path)
    ctx.config.mkdir()
    for path in ("/fake/first-engine", "/fake/second-engine"):
        monkeypatch.setenv("CHROME_BIN", path)
        config = json.loads(playwright_common.write_browser_config(ctx).read_text())
        assert config["browser"]["launchOptions"]["executablePath"] == path


def test_playwright_version_cache_is_scoped_to_the_current_engine(monkeypatch):
    from types import SimpleNamespace

    from benchmarks.interactive.adapters import playwright_common

    monkeypatch.delenv("PLAYWRIGHT_BENCH_BROWSER_VERSION", raising=False)
    monkeypatch.setattr(playwright_common, "_VERSION_CACHE", {})
    commands = []

    def version(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="Chromium 159.0.0.1", stderr="")

    monkeypatch.setattr(playwright_common.subprocess, "run", version)
    for path in ("/fake/first-engine", "/fake/second-engine"):
        monkeypatch.setenv("CHROME_BIN", path)
        assert playwright_common.chrome_version() == "Chromium 159.0.0.1"
    assert commands == [["/fake/first-engine", "--version"], ["/fake/second-engine", "--version"]]
