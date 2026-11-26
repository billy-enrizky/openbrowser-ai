from __future__ import annotations

import json
from pathlib import Path

from benchmarks.interactive.adapters.base import RunContext, host_playwright_cache
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


def test_benchmark_environment_exposes_the_host_browser_cache(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    cache = host_playwright_cache()

    env = ctx.env

    if cache is None:
        assert "PLAYWRIGHT_BROWSERS_PATH" not in env
    else:
        assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(cache)


def test_cold_context_uses_short_private_direct_scope_for_every_arm(tmp_path):
    from benchmarks.interactive.arms import CANONICAL_ARMS
    from benchmarks.interactive.run_benchmark import make_context

    roots = []
    for arm in CANONICAL_ARMS:
        ctx = make_context("wizard", arm, 0, "http://fixture.invalid", tmp_path)
        assert ctx.root.parent == tmp_path.resolve()
        assert len(ctx.root.name) <= 12
        assert ctx.root.stat().st_mode & 0o777 == 0o700
        assert ctx.tmp.parent == ctx.root
        roots.append(ctx.root)
    repeated = make_context("wizard", CANONICAL_ARMS[0], 0, "http://fixture.invalid", tmp_path)
    assert repeated.root not in roots
    assert all(root.is_dir() for root in roots)
