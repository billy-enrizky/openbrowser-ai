"""Tests for the non-destructive config file lifecycle."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from openbrowser.config import (
    CONFIG,
    OldConfig,
    _config_cache,
    _get_env_bool_cached,
    _get_env_cached,
    _get_path_cached,
    load_and_migrate_config,
    load_and_migrate_config_result,
)
from openbrowser.config_lifecycle import ConfigFilePhase, ConfigStartupError


def _reset_config_caches() -> None:
    _config_cache.clear()
    CONFIG._old_config = None
    CONFIG._env_config = None
    OldConfig._dirs_created = False
    _get_env_cached.cache_clear()
    _get_env_bool_cached.cache_clear()
    _get_path_cached.cache_clear()


@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setenv("OPENBROWSER_CONFIG_PATH", str(path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("OPENBROWSER_CONFIG_DIR", raising=False)
    for name in (
        "OPENBROWSER_HEADLESS",
        "OPENBROWSER_ALLOWED_DOMAINS",
        "OPENBROWSER_USER_DATA_DIR",
        "OPENBROWSER_STORAGE_STATE",
        "OPENBROWSER_PROXY_URL",
        "OPENBROWSER_NO_PROXY",
        "OPENBROWSER_PROXY_USERNAME",
        "OPENBROWSER_PROXY_PASSWORD",
        "OPENBROWSER_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    _reset_config_caches()
    yield path
    _reset_config_caches()


def _db_config(*, headless: bool = True, model: str = "gpt-5-mini") -> dict:
    return {
        "browser_profile": {
            "profile-a": {
                "id": "profile-a",
                "default": True,
                "headless": headless,
                "user_data_dir": "/tmp/openbrowser-test-profile",
            }
        },
        "llm": {"llm-a": {"id": "llm-a", "default": True, "model": model}},
        "agent": {"agent-a": {"id": "agent-a", "default": True, "max_steps": 5}},
    }


def _write(path: Path, value: object) -> bytes:
    raw = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_missing_config_is_created_once(config_path):
    first = load_and_migrate_config_result(config_path)
    second = load_and_migrate_config_result(config_path)

    assert first.snapshot.phase is ConfigFilePhase.CREATED
    assert first.snapshot.browser_start_allowed is True
    assert config_path.is_file()
    assert second.snapshot.phase is ConfigFilePhase.READY
    assert second.snapshot.file_version == first.snapshot.file_version
    json.loads(config_path.read_text())


def test_valid_config_is_ready(config_path):
    raw = _write(config_path, _db_config())
    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.READY
    assert result.snapshot.error_code is None
    assert result.snapshot.browser_start_allowed is True
    assert config_path.read_bytes() == raw


def test_invalid_json_is_not_rewritten(config_path):
    raw = b"NOT JSON {{{"
    config_path.write_bytes(raw)

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert result.snapshot.error_code == "config_invalid_json"
    assert result.snapshot.browser_start_allowed is False
    assert "unique backup" in (result.snapshot.hint or "")
    assert config_path.read_bytes() == raw


def test_invalid_schema_is_not_rewritten(config_path):
    raw = _write(
        config_path,
        {
            "browser_profile": {"profile-a": {"id": "profile-a", "default": {}}},
            "llm": {},
            "agent": {},
        },
    )

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert result.snapshot.error_code == "config_invalid_schema"
    assert config_path.read_bytes() == raw


def test_legacy_flat_format_is_not_rewritten(config_path):
    raw = _write(
        config_path,
        {
            "browser_profile": {"headless": False},
            "llm": {"model": "gpt-4"},
            "agent": {"max_steps": 10},
        },
    )

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.LEGACY_FORMAT
    assert result.snapshot.error_code == "config_legacy_format"
    assert config_path.read_bytes() == raw


def test_uuid_scalar_profile_is_invalid_not_legacy(config_path):
    _write(
        config_path,
        {
            "browser_profile": {"550e8400-e29b-41d4-a716-446655440000": "not-an-entry"},
            "llm": {},
            "agent": {},
        },
    )

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert result.snapshot.error_code == "config_invalid_schema"


def test_empty_browser_profile_is_invalid(config_path):
    _write(config_path, {"browser_profile": {}, "llm": {}, "agent": {}})

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert result.snapshot.error_code == "config_invalid_schema"


def test_partial_db_config_is_invalid(config_path):
    _write(config_path, {"browser_profile": {"p": {"id": "p", "default": True}}})

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert result.snapshot.error_code == "config_invalid_schema"


def test_non_regular_config_path_is_read_failed(config_path):
    config_path.mkdir()

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.READ_FAILED
    assert result.snapshot.error_code == "config_read_failed"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is not available on this platform")
def test_fifo_config_path_returns_without_blocking(config_path):
    os.mkfifo(config_path)
    started = time.monotonic()

    result = load_and_migrate_config_result(config_path)

    assert time.monotonic() - started < 1.0
    assert result.snapshot.phase is ConfigFilePhase.READ_FAILED


def test_cache_reuses_ready_result_until_file_changes(config_path):
    _write(config_path, _db_config(headless=True))
    first = load_and_migrate_config_result(config_path)
    second = load_and_migrate_config_result(config_path)
    assert first is second

    _write(config_path, _db_config(headless=False, model="gpt-5-mini-extra"))
    third = load_and_migrate_config_result(config_path)

    assert third.snapshot.file_version != first.snapshot.file_version
    assert CONFIG._get_default_profile(third.config)["headless"] is False


def test_concurrent_first_start_observes_one_complete_file(config_path):
    barrier = config_path.parent / "publish-barrier"
    barrier.mkdir()
    script = """
import json
import os
import time
from pathlib import Path
import openbrowser.config as config_mod
barrier = Path(os.environ["OPENBROWSER_TEST_LINK_BARRIER"])
original_link = config_mod.os.link

def coordinated_link(source, destination):
    (barrier / f"link-ready-{os.getpid()}").write_text("")
    deadline = time.monotonic() + 10
    while not (barrier / "link-go").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("publish barrier timed out")
        time.sleep(0.001)
    return original_link(source, destination)

config_mod.os.link = coordinated_link
path = Path(os.environ["OPENBROWSER_CONFIG_PATH"])
result = config_mod.load_and_migrate_config_result(path)
print(json.dumps({
    "phase": result.snapshot.phase.value,
    "profile_ids": sorted(result.config.browser_profile),
}))
"""
    env = os.environ.copy()
    env["OPENBROWSER_CONFIG_PATH"] = str(config_path)
    env["OPENBROWSER_TEST_LINK_BARRIER"] = str(barrier)
    project_root = Path(__file__).resolve().parents[1]
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    deadline = time.monotonic() + 10
    while len(list(barrier.glob("link-ready-*"))) < 2:
        if time.monotonic() > deadline:
            for process in processes:
                process.kill()
                process.wait()
            raise AssertionError("processes did not reach the publication barrier")
        time.sleep(0.01)
    (barrier / "link-go").write_text("")
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        outputs.append(json.loads(stdout))
    assert {output["phase"] for output in outputs} <= {"created", "ready"}
    assert [output["phase"] for output in outputs].count("created") <= 1
    assert len({tuple(output["profile_ids"]) for output in outputs}) == 1
    json.loads(config_path.read_text())


def test_write_failed_before_publish(config_path, monkeypatch):
    def fail_fsync(_descriptor):
        raise OSError("fsync failed")

    monkeypatch.setattr("openbrowser.config.os.fsync", fail_fsync)

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.WRITE_FAILED
    assert not config_path.exists()


def test_write_failed_after_publish_leaves_complete_file(config_path, monkeypatch):
    def fail_directory_fsync(_path):
        raise OSError("directory fsync failed")

    monkeypatch.setattr("openbrowser.config._fsync_directory", fail_directory_fsync)
    failed = load_and_migrate_config_result(config_path)
    assert failed.snapshot.phase is ConfigFilePhase.WRITE_FAILED
    assert config_path.is_file()
    json.loads(config_path.read_text())

    monkeypatch.undo()
    ready = load_and_migrate_config_result(config_path)
    assert ready.snapshot.phase is ConfigFilePhase.READY


def test_link_collision_reads_winner(config_path, monkeypatch):
    winner = _db_config(headless=False)

    def collide(source, destination):
        Path(destination).write_bytes(json.dumps(winner).encode("utf-8"))
        raise FileExistsError

    monkeypatch.setattr("openbrowser.config.os.link", collide)
    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.READY
    assert CONFIG._get_default_profile(result.config)["headless"] is False


def test_require_browser_start_allowed_returns_exact_invalid_code(config_path):
    _write(config_path, {"bad": True})

    with pytest.raises(ConfigStartupError) as error:
        CONFIG.require_browser_start_allowed()

    assert error.value.code == "config_invalid_schema"
    assert error.value.path == str(config_path)
    assert "unique backup" in error.value.hint


def test_startup_result_requires_restart_after_valid_change(config_path):
    _write(config_path, _db_config(headless=True))
    startup = load_and_migrate_config_result(config_path)
    _write(config_path, _db_config(headless=False, model="gpt-5-mini-extra"))

    with pytest.raises(ConfigStartupError) as error:
        CONFIG.require_browser_start_allowed(startup)

    assert error.value.code == "config_restart_required"


def test_startup_blocked_requires_restart_after_repair(config_path):
    _write(config_path, b"NOT JSON {{{")
    startup = load_and_migrate_config_result(config_path)
    _write(config_path, _db_config())

    with pytest.raises(ConfigStartupError) as error:
        CONFIG.require_browser_start_allowed(startup)

    assert error.value.code == "config_restart_required"


def test_config_dict_preserves_profile_selection_and_env(config_path, monkeypatch):
    config = _db_config(headless=True)
    config["browser_profile"]["profile-a"]["default"] = False
    config["browser_profile"]["profile-b"] = {
        "id": "profile-b",
        "default": True,
        "headless": False,
        "allowed_domains": ["example.com"],
    }
    _write(config_path, config)
    monkeypatch.setenv("OPENBROWSER_HEADLESS", "true")
    monkeypatch.setenv("OPENBROWSER_ALLOWED_DOMAINS", "env.example, second.example")
    monkeypatch.setenv("OPENBROWSER_PROXY_URL", "http://proxy.example:8080")
    monkeypatch.setenv("OPENBROWSER_PROXY_USERNAME", "user")
    _reset_config_caches()

    result = load_and_migrate_config_result(config_path)
    config_dict = CONFIG.config_dict_from_result(result)

    assert CONFIG._get_default_profile(result.config)["id"] == "profile-b"
    assert config_dict["browser_profile"]["headless"] is True
    assert config_dict["browser_profile"]["allowed_domains"] == ["env.example", "second.example"]
    assert config_dict["browser_profile"]["proxy"] == {
        "server": "http://proxy.example:8080",
        "username": "user",
    }


def test_snapshot_is_json_safe(config_path):
    _write(config_path, b"NOT JSON {{{")
    snapshot = load_and_migrate_config_result(config_path).snapshot
    serialized = json.dumps(snapshot.to_dict())

    assert "config_invalid_json" in serialized
    assert "NOT JSON" not in serialized


def test_legacy_wrapper_returns_config_without_rewriting(config_path):
    raw = b'{"browser_profile": {"headless": false}, "llm": {}, "agent": {}}'
    config_path.write_bytes(raw)

    config = load_and_migrate_config(config_path)

    assert config.browser_profile
    assert config_path.read_bytes() == raw


def test_config_path_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.json"
    config_dir = tmp_path / "config-dir"
    xdg = tmp_path / "xdg"
    _write(explicit, _db_config(model="explicit-model"))
    (config_dir).mkdir(parents=True)
    (xdg / "openbrowser").mkdir(parents=True)
    _write(config_dir / "config.json", _db_config(model="dir-model"))
    _write(xdg / "openbrowser" / "config.json", _db_config(model="xdg-model"))
    monkeypatch.setenv("OPENBROWSER_CONFIG_PATH", str(explicit))
    monkeypatch.setenv("OPENBROWSER_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    _reset_config_caches()

    selected = CONFIG.load_result()
    assert selected.snapshot.path == str(explicit)
    assert selected.config.llm["llm-a"].model == "explicit-model"

    monkeypatch.delenv("OPENBROWSER_CONFIG_PATH")
    _reset_config_caches()
    selected = CONFIG.load_result()
    assert selected.snapshot.path == str(config_dir / "config.json")
    assert selected.config.llm["llm-a"].model == "dir-model"

    monkeypatch.delenv("OPENBROWSER_CONFIG_DIR")
    _reset_config_caches()
    selected = CONFIG.load_result()
    assert selected.snapshot.path == str(xdg / "openbrowser" / "config.json")
    assert selected.config.llm["llm-a"].model == "xdg-model"


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges on Windows")
def test_symlinked_config_path_targets_the_file_and_preserves_the_link(config_path):
    target = config_path.parent / "target.json"
    _write(target, _db_config())
    config_path.symlink_to(target)
    _reset_config_caches()

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.READY
    assert result.snapshot.path == str(config_path)
    assert result.snapshot.resolved_path == str(target.resolve())
    assert config_path.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges on Windows")
def test_symlink_loop_is_read_failed_without_hanging(config_path):
    peer = config_path.parent / "peer.json"
    config_path.symlink_to(peer)
    peer.symlink_to(config_path)
    _reset_config_caches()

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.READ_FAILED
    assert result.snapshot.error_code == "config_read_failed"


def test_legacy_classifier_accepts_any_known_browser_profile_field(config_path):
    raw = _write(
        config_path,
        {
            "browser_profile": {"keep_alive": True},
            "llm": {"model": "gpt-4"},
            "agent": {"max_steps": 10},
        },
    )

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.LEGACY_FORMAT
    assert config_path.read_bytes() == raw


def test_unknown_scalar_profile_key_is_invalid_not_legacy(config_path):
    _write(
        config_path,
        {
            "browser_profile": {"unreviewed_setting": True},
            "llm": {},
            "agent": {},
        },
    )

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert result.snapshot.error_code == "config_invalid_schema"


def test_tempfile_creation_failure_returns_write_failed(config_path, monkeypatch):
    def fail_tempfile(*args, **kwargs):
        raise OSError("tempfile unavailable")

    monkeypatch.setattr("openbrowser.config.tempfile.NamedTemporaryFile", fail_tempfile)

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.WRITE_FAILED
    assert result.snapshot.error_code == "config_write_failed"
    assert not config_path.exists()


def test_rejected_secret_never_reaches_snapshot_or_logs(config_path, caplog):
    secret = "SENTINEL-SECRET-71f0e5"
    config_path.write_text(
        json.dumps(
            {
                "browser_profile": {
                    "profile-a": {
                        "id": "profile-a",
                        "default": True,
                        "headless": secret,
                    }
                },
                "llm": {},
                "agent": {},
            }
        )
    )

    with caplog.at_level("DEBUG"):
        result = load_and_migrate_config_result(config_path)

    serialized = json.dumps(result.snapshot.to_dict())
    assert secret not in serialized
    assert secret not in caplog.text
    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert secret in config_path.read_text()


def test_unreadable_config_path_reports_read_failed(config_path, monkeypatch):
    _write(config_path, _db_config())
    original_stat = Path.stat

    def deny_target(self, *args, **kwargs):
        if self == config_path:
            raise PermissionError("denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", deny_target)

    result = load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.READ_FAILED
    assert result.snapshot.error_code == "config_read_failed"


def test_invalid_json_secret_never_reaches_snapshot(config_path):
    secret = "SENTINEL-JSON-SECRET-b42d"
    config_path.write_text('{"api_key": "' + secret + '"')

    result = load_and_migrate_config_result(config_path)

    serialized = json.dumps(result.snapshot.to_dict())
    assert secret not in serialized
    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert result.snapshot.error_code == "config_invalid_json"


def test_file_change_during_parse_is_not_cached_as_ready(config_path, monkeypatch):
    from openbrowser import config as config_mod

    _write(config_path, _db_config())
    original_parse = config_mod._parse_existing
    replaced = False

    def parse_then_replace(*args, **kwargs):
        nonlocal replaced
        result = original_parse(*args, **kwargs)
        if not replaced:
            replaced = True
            config_path.write_text("NOT JSON {{{")
        return result

    monkeypatch.setattr(config_mod, "_parse_existing", parse_then_replace)

    result = config_mod.load_and_migrate_config_result(config_path)

    assert result.snapshot.phase is ConfigFilePhase.INVALID
    assert result.snapshot.error_code == "config_invalid_json"
    assert config_path.read_text() == "NOT JSON {{{"
    with pytest.raises(ConfigStartupError) as error:
        CONFIG.require_browser_start_allowed()
    assert error.value.code == "config_invalid_json"
