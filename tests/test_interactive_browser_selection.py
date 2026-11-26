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
