"""Matched engine configuration is explicit and process-free in workers."""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.interactive import engine_preflight
from benchmarks.interactive.engine_binding import ENGINE_PIN_ENV, EngineBindingError, capture_engine_pin


@pytest.fixture
def pin(tmp_path):
    launcher = tmp_path / "chromium_headless_shell-1255/chrome-headless-shell-mac-arm64/chrome-headless-shell"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"fixture browser image")
    return capture_engine_pin(
        "matched-shell-1255",
        tmp_path,
        platform_name="darwin",
        machine="arm64",
        executable_checker=lambda path: True,
        version_runner=lambda path: "Google Chrome for Testing 159.0.8154.6",
    )


def test_configure_sets_one_pin_and_the_same_executable_for_every_arm(pin):
    env = {"CHROME_BIN": "old", "UNRELATED": "preserved"}
    engine_preflight.configure_engine_environment(pin, env)
    assert env["CHROME_BIN"] == str(pin.launcher)
    assert env["PLAYWRIGHT_BENCH_BROWSER_VERSION"] == pin.version
    assert env["OPENBROWSER_USE_HEADLESS_SHELL"] == "0"
    assert env["UNRELATED"] == "preserved"
    assert engine_preflight.configured_engine_pin(env) == pin


def test_unconfigured_mode_performs_no_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unconfigured engine read native state")

    monkeypatch.setattr(engine_preflight, "read_pin_environment", forbidden)
    assert engine_preflight.configured_engine_pin({"CHROME_BIN": "/explicit/legacy/browser"}) is None


@pytest.mark.parametrize("key", ["CHROME_BIN", "PLAYWRIGHT_BENCH_BROWSER_VERSION", "OPENBROWSER_USE_HEADLESS_SHELL"])
@pytest.mark.parametrize("mutation", [None, "", "wrong"])
def test_missing_or_mixed_configuration_fails_closed(pin, key, mutation):
    env = {}
    engine_preflight.configure_engine_environment(pin, env)
    if mutation is None:
        del env[key]
    else:
        env[key] = mutation
    with pytest.raises(EngineBindingError):
        engine_preflight.configured_engine_pin(env)


def test_equivalent_executable_alias_is_accepted_then_retarget_is_rejected(pin, tmp_path):
    env = {}
    engine_preflight.configure_engine_environment(pin, env)
    alias = tmp_path / "browser-alias"
    alias.symlink_to(pin.launcher)
    env["CHROME_BIN"] = str(alias)
    assert engine_preflight.configured_engine_pin(env) == pin
    alias.unlink()
    other = tmp_path / "different-browser"
    other.write_bytes(b"other")
    alias.symlink_to(other)
    with pytest.raises(EngineBindingError, match="engine_configuration_mismatch"):
        engine_preflight.configured_engine_pin(env)


def test_browser_file_drift_is_rejected_in_the_worker(pin):
    env = {}
    engine_preflight.configure_engine_environment(pin, env)
    pin.launcher.write_bytes(b"changed image")
    with pytest.raises(EngineBindingError, match="pin_changed"):
        engine_preflight.configured_engine_pin(env)


def test_malformed_config_is_never_downgraded_to_unconfigured():
    with pytest.raises(EngineBindingError):
        engine_preflight.configured_engine_pin({ENGINE_PIN_ENV: "{}"})


def test_invalid_pin_cannot_partially_mutate_launch_environment():
    env = {"CHROME_BIN": "unchanged"}
    invalid: Any = None
    with pytest.raises(EngineBindingError):
        engine_preflight.configure_engine_environment(invalid, env)
    assert env == {"CHROME_BIN": "unchanged"}
