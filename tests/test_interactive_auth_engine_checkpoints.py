"""Auth observes the login browser while preserving every native task program."""

from __future__ import annotations

import socket
import subprocess
from types import SimpleNamespace

import psutil
import pytest

from benchmarks.interactive.adapters import (
    agent_browser,
    browser_use,
    chrome_devtools,
    openbrowser,
    openbrowser_mcp,
    playwright_cli,
    playwright_mcp,
)
from benchmarks.interactive.fixtures.server import ACCOUNT_SECRET
from benchmarks.interactive.run_benchmark import make_context

MODULES = {
    module.__name__.rsplit(".", 1)[-1]: module
    for module in (
        agent_browser,
        browser_use,
        chrome_devtools,
        openbrowser,
        openbrowser_mcp,
        playwright_cli,
        playwright_mcp,
    )
}


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("auth checkpoint tests require injected native work")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)


def build_arm(name, ctx, events, monkeypatch):
    module = MODULES[name]
    monkeypatch.setattr(
        module, "browser_launch_checkpoint", lambda marker: events.append(("checkpoint", marker)), raising=False
    )
    if hasattr(module, "verify_mock_keychain"):
        monkeypatch.setattr(module, "verify_mock_keychain", lambda *a, **k: True)
    if name == "openbrowser":
        arm = module.OpenBrowserArm(binary="/fake/openbrowser-ai", fast_selectors=True)
        monkeypatch.setattr(module, "_write_config", lambda *a, **k: None)
        monkeypatch.setattr(arm, "_invoke", lambda *a: events.append(("native", None)))
        monkeypatch.setattr(arm, "_reset_browser", lambda *a: events.append(("reset", None)))
        monkeypatch.setattr(arm, "_parse_last_json", dict)
        monkeypatch.setattr(arm, "_keychain_profile", lambda *a: str(ctx.profile))
    elif name == "openbrowser_mcp":
        arm = module.OpenBrowserMcpArm(binary="/fake/openbrowser-ai", fast_selectors=True)
        monkeypatch.setattr(module, "_write_config", lambda *a, **k: None)
        monkeypatch.setattr(arm, "_start", lambda *a: SimpleNamespace(process=None))
        monkeypatch.setattr(arm, "_create_context", lambda *a: "context")
        monkeypatch.setattr(arm, "_ensure_context", lambda *a: "context")
        monkeypatch.setattr(arm, "_execute", lambda *a: events.append(("native", None)) or "{}\n")
        monkeypatch.setattr(arm, "_reset_browser", lambda *a: events.append(("reset", None)))
        monkeypatch.setattr(arm, "_parse_last_json", dict)
        monkeypatch.setattr(arm, "_mock_keychain_profile", lambda *a: str(ctx.profile))
    elif name in {"playwright_cli", "playwright_mcp"}:
        cls = module.PlaywrightCliArm if name.endswith("cli") else module.PlaywrightMcpArm
        arm = cls(binary="/fake/playwright", browser_version="159.0.8154.6")

        def program(*args, **kwargs):
            events.append(("native", None))
            arm._result.stdout += "{}\n"
            return "{}"

        monkeypatch.setattr(arm, "_ensure_keychain", lambda *a: True)
        monkeypatch.setattr(arm, "_restart", lambda *a: events.append(("reset", None)))
        if name.endswith("cli"):
            monkeypatch.setattr(arm, "_open", lambda *a: None)
            monkeypatch.setattr(arm, "_run", lambda *a, **k: "")
            monkeypatch.setattr(arm, "_code", program)
        else:
            monkeypatch.setattr(arm, "_call", program)
    elif name == "agent_browser":
        arm = module.AgentBrowserArm()
        monkeypatch.setattr(
            arm, "_run", lambda context, *a, **k: events.append(("reset", None)) if a[0] == "close" else None
        )
        monkeypatch.setattr(arm, "_text", lambda *a: events.append(("native", None)) or ACCOUNT_SECRET)
        monkeypatch.setattr(arm, "_run_wizard", lambda *a: None)
    elif name == "browser_use":
        arm = module.BrowserUseArm()

        def start(*args):
            arm._chrome = SimpleNamespace(process=None)

        monkeypatch.setattr(arm, "_start_chrome", start)
        monkeypatch.setattr(arm, "_stop_chrome", lambda: events.append(("reset", None)))
        monkeypatch.setattr(arm, "_script", lambda *a: events.append(("native", None)) or {"secret": ACCOUNT_SECRET})
    else:
        arm = module.ChromeDevToolsArm()
        monkeypatch.setattr(arm, "_start", lambda *a: None)
        monkeypatch.setattr(arm, "_navigate", lambda *a, **k: None)
        monkeypatch.setattr(arm, "_snapshot", list)
        monkeypatch.setattr(module, "_find", lambda *a, **k: "uid")
        monkeypatch.setattr(arm, "_fill", lambda *a: None)
        monkeypatch.setattr(arm, "_click", lambda *a: None)
        monkeypatch.setattr(arm, "_wait_js", lambda *a: None)
        monkeypatch.setattr(arm, "_evaluate", lambda *a: events.append(("native", None)) or ACCOUNT_SECRET)
        monkeypatch.setattr(arm, "_stop_client", lambda: events.append(("reset", None)))
        monkeypatch.setattr(arm, "_run_wizard", lambda *a: None)
    return arm


@pytest.mark.parametrize(
    "name",
    [
        "openbrowser",
        "openbrowser_mcp",
        "playwright_cli",
        "playwright_mcp",
        "agent_browser",
        "browser_use",
        "chrome_devtools",
    ],
)
@pytest.mark.parametrize("task", ["auth", "wizard"])
def test_initial_auth_browser_is_observed_before_native_reset(name, task, monkeypatch, tmp_path):
    ctx = make_context(task, name, 0, "http://fixture.invalid", tmp_path)
    events = []
    arm = build_arm(name, ctx, events, monkeypatch)
    result = arm.run(ctx, cleanup=False)
    assert result.ok is True, result.error
    checkpoints = [event for event in events if event[0] == "checkpoint"]
    if task == "wizard":
        assert checkpoints == []
        return
    assert checkpoints == [("checkpoint", str(ctx.profile))]
    checkpoint_index = events.index(checkpoints[0])
    reset_index = next(index for index, event in enumerate(events) if event[0] == "reset")
    assert checkpoint_index < reset_index
    if name != "browser_use":
        assert any(event[0] == "native" for event in events[:checkpoint_index])
    else:
        assert events[0][0] == "checkpoint"
    assert any(event[0] == "native" for event in events[reset_index + 1 :])
    assert sum(event[0] == "reset" for event in events) == 1
