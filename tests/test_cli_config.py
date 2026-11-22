"""Tests for the config status CLI fast path and prompt gate."""

import json
import os
import subprocess
import sys
from pathlib import Path

from openbrowser.cli import CliMode, select_cli_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], *, env: dict[str, str], cwd: Path | None = None):
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _base_env(config_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENBROWSER_CONFIG_PATH"] = str(config_path)
    for name in (
        "OPENBROWSER_CONFIG_DIR",
        "OPENBROWSER_HEADLESS",
        "OPENBROWSER_ALLOWED_DOMAINS",
        "OPENBROWSER_USER_DATA_DIR",
        "OPENBROWSER_STORAGE_STATE",
        "OPENBROWSER_PROXY_URL",
        "OPENBROWSER_NO_PROXY",
        "OPENBROWSER_PROXY_USERNAME",
        "OPENBROWSER_PROXY_PASSWORD",
    ):
        env.pop(name, None)
    return env


def test_mode_selection_prefers_prompt_and_recognizes_config():
    assert select_cli_mode(["config", "status"]) is CliMode.CONFIG
    assert select_cli_mode(["-p", "config status"]) is None
    assert select_cli_mode(["-p", "config"]) is None


def test_config_status_through_console_entrypoint(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("NOT JSON {{{")
    result = _run(["-m", "openbrowser.cli_entry", "config", "status"], env=_base_env(config_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["phase"] == "invalid"
    assert payload["error_code"] == "config_invalid_json"
    assert payload["browser_start_allowed"] is False
    assert config_path.read_text() == "NOT JSON {{{"


def test_config_status_through_module_fast_path(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("NOT JSON {{{")
    result = _run(["-m", "openbrowser.cli", "config", "status"], env=_base_env(config_path))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["phase"] == "invalid"


def test_unknown_config_subcommand_exits_2(tmp_path):
    config_path = tmp_path / "config.json"
    result = _run(["-m", "openbrowser.cli", "config", "unknown"], env=_base_env(config_path))

    assert result.returncode == 2
    assert "Usage: openbrowser-ai config status" in result.stderr


def test_prompt_mode_rejects_invalid_config_before_start(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("NOT JSON {{{")
    result = _run(["-m", "openbrowser.cli", "-p", "hello"], env=_base_env(config_path))

    assert result.returncode == 1
    assert "Config error [config_invalid_json]" in result.stderr
    assert str(config_path) in result.stderr
    assert "unique backup" in result.stderr
    assert config_path.read_text() == "NOT JSON {{{"


def test_config_status_does_not_echo_rejected_values(tmp_path):
    config_path = tmp_path / "config.json"
    secret = "SENTINEL-CLI-SECRET-3fd7"
    config_path.write_text(
        json.dumps(
            {
                "browser_profile": {"profile-a": {"id": "profile-a", "default": True, "headless": secret}},
                "llm": {},
                "agent": {},
            }
        )
    )
    result = _run(["-m", "openbrowser.cli", "config", "status"], env=_base_env(config_path))

    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert json.loads(result.stdout)["phase"] == "invalid"
