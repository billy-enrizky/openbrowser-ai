"""Auth uses a browser reset inside one native context and persistent profile."""

from __future__ import annotations

import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.interactive.adapters import openbrowser_mcp
from benchmarks.interactive.adapters.base import RunContext
from benchmarks.interactive.fixtures.server import ACCOUNT_SECRET


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("auth tests require fake native I/O")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def context(tmp_path):
    return RunContext(
        task="auth",
        arm="openbrowser-mcp-fast",
        sample=0,
        base_url="http://fixture.invalid",
        root=tmp_path,
        downloads=tmp_path / "downloads",
        profile=tmp_path / "profile",
        home=tmp_path / "home",
        tmp=tmp_path / "tmp",
        config=tmp_path / "config",
        session="auth",
    )


def setup_arm(monkeypatch, ctx, reset):
    events = []

    def call(name, args, **kwargs):
        events.append((name, dict(args)))
        if name == "create_context":
            return json.dumps({"context_id": "context"})
        if name == "reset_browser":
            return json.dumps(reset)
        if name == "close_context":
            return json.dumps({"status": "closed"})
        return json.dumps({"secret": ACCOUNT_SECRET})

    client = SimpleNamespace(process=None, call_tool_text=call)
    arm = openbrowser_mcp.OpenBrowserMcpArm(binary="/fake/openbrowser-ai", fast_selectors=True)
    monkeypatch.setattr(openbrowser_mcp, "_write_config", lambda *a, **kw: None)
    monkeypatch.setattr(openbrowser_mcp, "verify_mock_keychain", lambda *a: True)
    monkeypatch.setattr(openbrowser_mcp, "browser_launch_checkpoint", lambda marker: events.append(("checkpoint", {})))
    monkeypatch.setattr(arm, "_start", lambda *a: client)
    monkeypatch.setattr(arm, "_mock_keychain_profile", lambda *a: str(ctx.profile))
    return arm, events


def test_auth_resets_browser_in_same_context(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    arm, events = setup_arm(monkeypatch, ctx, {"ok": True, "browser": {"phase": "stopped"}})
    result = arm.run(ctx, cleanup=False)
    assert result.ok is True
    names = [name for name, _ in events]
    assert names == ["create_context", "execute_code", "checkpoint", "reset_browser", "execute_code"]
    assert arm._context_id == "context"
    assert [args["context_id"] for name, args in events if name in {"execute_code", "reset_browser"}] == ["context"] * 3
    assert result.tool_calls == 4


@pytest.mark.parametrize(
    "reset",
    [
        {"ok": False},
        {"ok": True, "browser": {"phase": "running"}},
        {"ok": "true", "browser": {"phase": "stopped"}},
        {"ok": True, "browser": None},
    ],
)
def test_auth_rejects_unverified_browser_reset(tmp_path, monkeypatch, reset):
    ctx = context(tmp_path)
    arm, events = setup_arm(monkeypatch, ctx, reset)
    result = arm.run(ctx, cleanup=False)
    assert result.ok is False
    assert sum(name == "execute_code" for name, _ in events) == 1
