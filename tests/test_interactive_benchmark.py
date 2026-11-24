from __future__ import annotations

import json
from pathlib import Path

from benchmarks.interactive.adapters.base import RunContext
from benchmarks.interactive.adapters.openbrowser import _env, _write_config


def _context(tmp_path: Path) -> RunContext:
    root = tmp_path / "run"
    for name in ("downloads", "profile", "home", "tmp", "config"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return RunContext(
        task="wizard",
        arm="openbrowser",
        sample=0,
        base_url="http://127.0.0.1:1234",
        root=root,
        downloads=root / "downloads",
        profile=root / "profile",
        home=root / "home",
        tmp=root / "tmp",
        config=root / "config",
        session="bench",
    )


def test_openbrowser_benchmark_uses_lean_profile(tmp_path: Path) -> None:
    ctx = _context(tmp_path)

    _write_config(ctx, extensions=False)

    config = json.loads((ctx.config / "config.json").read_text())
    profile = config["browser_profile"]["bench"]
    assert profile["enable_default_extensions"] is False
    assert profile["args"] == ["--use-mock-keychain"]


def test_openbrowser_benchmark_pins_the_daemon_socket(tmp_path: Path) -> None:
    ctx = _context(tmp_path)

    env = _env(ctx)

    assert env["HOME"] == str(ctx.home)
    assert env["OPENBROWSER_CONFIG_DIR"] == str(ctx.config)
    assert env["OPENBROWSER_SOCKET"] == str(ctx.home / ".openbrowser" / "daemon.sock")
