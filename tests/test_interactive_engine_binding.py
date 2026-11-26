"""Browser-free tests for matched-engine pinning and process binding."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import psutil
import pytest


def engine_module():
    from benchmarks.interactive import engine_binding

    return engine_binding


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    import socket
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("native I/O is forbidden in fake tests")

    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


SHELL_RELATIVE = Path("chromium_headless_shell-1255/chrome-headless-shell-mac-arm64/chrome-headless-shell")
FULL_APP_RELATIVE = Path("chromium-1255/chrome-mac-arm64/Google Chrome for Testing.app")
FULL_LAUNCHER_RELATIVE = FULL_APP_RELATIVE / "Contents/MacOS/Google Chrome for Testing"
FULL_IMAGE_RELATIVE = (
    FULL_APP_RELATIVE
    / "Contents/Frameworks/Google Chrome for Testing Framework.framework"
    / "Versions/159.0.8154.6/Google Chrome for Testing Framework"
)
VERSION_OUTPUT = "Google Chrome for Testing 159.0.8154.6 "


def write_executable(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o755)
    return path


def shell_cache(root: Path, data: bytes = b"shell-bytes") -> Path:
    write_executable(root / SHELL_RELATIVE, data)
    return root


def full_cache(root: Path, launcher_data: bytes = b"launcher-bytes", image_data: bytes = b"framework-bytes") -> Path:
    write_executable(root / FULL_LAUNCHER_RELATIVE, launcher_data)
    image = root / FULL_IMAGE_RELATIVE
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(image_data)
    image.chmod(0o755)
    return root


def capture_shell(root: Path, **overrides):
    engine = engine_module()
    kwargs = {
        "version_runner": lambda path: VERSION_OUTPUT,
        "platform_name": "darwin",
        "machine": "arm64",
    }
    kwargs.update(overrides)
    return engine.capture_engine_pin(engine.LANE_MATCHED_SHELL_1255, root, **kwargs)


def capture_full(root: Path, **overrides):
    engine = engine_module()
    kwargs = {
        "version_runner": lambda path: VERSION_OUTPUT,
        "platform_name": "darwin",
        "machine": "arm64",
    }
    kwargs.update(overrides)
    return engine.capture_engine_pin(engine.LANE_MATCHED_FULL_1255, root, **kwargs)


def raises_reason(reason: str):
    engine = engine_module()
    return pytest.raises(engine.EngineBindingError, match=f"^{reason}$")


class FakeProcess:
    def __init__(
        self,
        exe,
        *,
        created_at=1000.0,
        created_sequence=None,
        running=True,
        status=psutil.STATUS_RUNNING,
        exe_error=None,
    ):
        self._exe = exe
        self._created_at = created_at
        self._created_sequence = list(created_sequence or [])
        self._running = running
        self._status = status
        self._exe_error = exe_error

    def create_time(self):
        if self._created_sequence:
            return self._created_sequence.pop(0)
        return self._created_at

    def exe(self):
        if self._exe_error is not None:
            raise self._exe_error
        return self._exe

    def is_running(self):
        return self._running

    def status(self):
        return self._status


def factory_for(process):
    return lambda pid: process


def test_shell_pin_public_proof_hides_paths_and_env_round_trips(tmp_path):
    engine = engine_module()
    root = shell_cache(tmp_path / "cache")
    pin = capture_shell(root)

    assert pin.lane == engine.LANE_MATCHED_SHELL_1255
    assert pin.revision == "1255"
    assert pin.version == "159.0.8154.6"
    assert pin.launcher == (root / SHELL_RELATIVE).resolve()
    assert pin.engine_image == pin.launcher
    assert pin.launcher_sha256 == pin.engine_image_sha256
    assert len(pin.launcher_sha256) == 64

    proof = pin.public_proof()
    text = json.dumps(proof, sort_keys=True)
    assert str(tmp_path) not in text
    assert "/Users/" not in text
    assert "worktrees" not in text
    assert proof["launcher_kind"] == "headless-shell"
    assert proof["engine_image_kind"] == "headless-shell"
    assert proof["version"] == "159.0.8154.6"

    env_value = pin.to_env()
    assert str(pin.launcher) in env_value
    restored = engine.EnginePin.from_env(env_value)
    assert restored.public_proof() == proof
    assert restored.launcher == pin.launcher
    assert restored.engine_image == pin.engine_image


def test_full_pin_uses_framework_image(tmp_path):
    engine = engine_module()
    root = full_cache(tmp_path / "cache")
    pin = capture_full(root)

    assert pin.lane == engine.LANE_MATCHED_FULL_1255
    assert pin.launcher.name == "Google Chrome for Testing"
    assert pin.engine_image.name == "Google Chrome for Testing Framework"
    assert pin.engine_image != pin.launcher
    assert pin.engine_image_sha256 != pin.launcher_sha256
    assert pin.launcher_kind == "chrome-for-testing"
    assert pin.engine_image_kind == "framework"
    proof = pin.public_proof()
    assert str(tmp_path) not in json.dumps(proof)


def test_build_lane_paths_rejects_unsupported_inputs(tmp_path):
    engine = engine_module()
    with raises_reason("unsupported_lane"):
        engine.build_lane_paths("matched-system-chrome", tmp_path)
    with raises_reason("unsupported_platform"):
        engine.build_lane_paths(
            engine.LANE_MATCHED_SHELL_1255,
            tmp_path,
            platform_name="win32",
            machine="amd64",
        )
    with raises_reason("path_not_absolute"):
        engine.build_lane_paths(engine.LANE_MATCHED_SHELL_1255, Path("relative-cache"))


def test_capture_rejects_missing_launcher_and_engine_image(tmp_path):
    with raises_reason("launcher_missing"):
        capture_shell(tmp_path / "empty-cache")

    partial = tmp_path / "partial"
    write_executable(partial / FULL_LAUNCHER_RELATIVE, b"launcher")
    with raises_reason("engine_image_missing"):
        capture_full(partial)


def test_capture_rejects_non_executable_launcher(tmp_path):
    root = tmp_path / "cache"
    launcher = root / SHELL_RELATIVE
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_bytes(b"shell")
    launcher.chmod(0o644)
    with raises_reason("launcher_not_executable"):
        capture_shell(root)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("Google Chrome for Testing 159.0.8154.5", "version_mismatch"),
        ("Google Chrome 159.0.8154.6", "version_malformed"),
        ("Google Chrome for Testing 159.0.8154", "version_malformed"),
        ("Google Chrome for Testing 159.0.8154.6\nGoogle Chrome for Testing 159.0.8154.6", "version_malformed"),
        ("", "version_malformed"),
    ],
)
def test_capture_rejects_bad_version_output(tmp_path, output, reason):
    root = shell_cache(tmp_path / "cache")
    with raises_reason(reason):
        capture_shell(root, version_runner=lambda path: output)


def test_capture_rejects_version_runner_failure(tmp_path):
    engine = engine_module()
    root = shell_cache(tmp_path / "cache")

    def broken(path):
        raise engine.EngineBindingError("version_unavailable")

    with raises_reason("version_unavailable"):
        capture_shell(root, version_runner=broken)

    def unexpected(path):
        raise ValueError("bad runner")

    with raises_reason("version_unavailable"):
        capture_shell(root, version_runner=unexpected)


@pytest.mark.parametrize("value", ["abc", "A" * 64, "a" * 63, "a" * 65])
def test_capture_rejects_malformed_sha(tmp_path, value):
    root = shell_cache(tmp_path / "cache")
    with raises_reason("malformed_sha"):
        capture_shell(root, sha256_reader=lambda path: value)


def test_capture_rejects_sha_read_error(tmp_path):
    root = shell_cache(tmp_path / "cache")

    def broken(path):
        raise OSError("denied")

    with raises_reason("sha_unavailable"):
        capture_shell(root, sha256_reader=broken)

    def unexpected(path):
        raise ValueError("bad reader")

    with raises_reason("sha_unavailable"):
        capture_shell(root, sha256_reader=unexpected)


def test_capture_rejects_malformed_or_unavailable_stat(tmp_path):
    root = shell_cache(tmp_path / "cache")

    def broken(path):
        raise OSError("denied")

    with raises_reason("stat_unavailable"):
        capture_shell(root, stat_reader=broken)

    def malformed(path):
        return SimpleStat(device=True, inode=1, size=2, mtime_ns=3)

    with raises_reason("malformed_stat"):
        capture_shell(root, stat_reader=malformed)


class SimpleStat:
    def __init__(self, *, device=1, inode=2, size=3, mtime_ns=4):
        self.st_dev = device
        self.st_ino = inode
        self.st_size = size
        self.st_mtime_ns = mtime_ns


def test_pin_fence_detects_launcher_and_engine_image_change(tmp_path):
    root = full_cache(tmp_path / "cache")
    pin = capture_full(root)
    pin.fence()

    (root / FULL_LAUNCHER_RELATIVE).write_bytes(b"launcher-changed")
    with raises_reason("pin_changed"):
        pin.fence()

    launcher_root = full_cache(tmp_path / "cache-two")
    second = capture_full(launcher_root)
    (launcher_root / FULL_IMAGE_RELATIVE).write_bytes(b"framework-changed")
    with raises_reason("pin_changed"):
        second.fence()


def test_refresh_detects_content_change_with_unchanged_stat(tmp_path):
    root = shell_cache(tmp_path / "cache", b"aaaa")
    pin = capture_shell(root)
    launcher = root / SHELL_RELATIVE
    original = launcher.stat()
    launcher.write_bytes(b"bbbb")
    os.utime(launcher, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert launcher.stat().st_size == original.st_size
    assert launcher.stat().st_mtime_ns == original.st_mtime_ns
    with raises_reason("pin_changed"):
        pin.refresh()
    assert pin.refresh(sha256_reader=lambda path: pin.launcher_sha256) is pin


def test_env_round_trip_and_strict_malformed_rejection(tmp_path):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))
    payload = json.loads(pin.to_env())

    for value in ("not-json", "[]", json.dumps({**payload, "schema": "other"})):
        with pytest.raises(engine.EngineBindingError):
            engine.EnginePin.from_env(value)

    mutations = []
    missing = dict(payload)
    missing.pop("launcher")
    mutations.append(missing)
    extra = dict(payload)
    extra["unexpected"] = True
    mutations.append(extra)
    mutations.append({**payload, "launcher": "relative/path"})
    mutations.append({**payload, "launcher": str(pin.launcher) + "/../x"})
    mutations.append({**payload, "launcher_sha256": "A" * 64})
    mutations.append({**payload, "launcher_sha256": "a" * 63})
    mutations.append({**payload, "lane": "matched-full-1255"})
    mutations.append({**payload, "revision": "1254"})
    mutations.append({**payload, "version": "158.0.8127.18"})
    mutations.append({**payload, "platform": ""})
    mutations.append({**payload, "launcher_kind": "framework"})
    bad_stat = dict(payload["launcher_stat"])
    bad_stat["size"] = True
    mutations.append({**payload, "launcher_stat": bad_stat})
    zero_stat = dict(payload["launcher_stat"])
    zero_stat["size"] = 0
    mutations.append({**payload, "launcher_stat": zero_stat})
    extra_stat = dict(payload["launcher_stat"])
    extra_stat["unexpected"] = 1
    mutations.append({**payload, "launcher_stat": extra_stat})
    mutations.append({**payload, "launcher_stat": {**payload["launcher_stat"], "device": -1}})
    for mutated in mutations:
        with raises_reason("malformed_pin"):
            engine.EnginePin.from_env(json.dumps(mutated))


def test_environment_helpers_round_trip_and_fail_closed(tmp_path):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))
    env = {}
    engine.write_pin_environment(pin, env)
    assert env[engine.ENGINE_PIN_ENV] == pin.to_env()
    assert engine.read_pin_environment(env).public_proof() == pin.public_proof()
    with raises_reason("malformed_pin"):
        engine.read_pin_environment({})
    with raises_reason("malformed_pin"):
        engine.write_pin_environment(object(), {})  # pyright: ignore[reportArgumentType]


def test_process_binding_valid_and_public_proof_has_no_paths(tmp_path):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))
    process = FakeProcess(str(pin.launcher))
    binding = engine.capture_process_binding(pin, 4242, process_factory=factory_for(process))

    assert binding.pid == 4242
    assert binding.created_at == 1000.0
    assert binding.executable_basename == "chrome-headless-shell"
    assert binding.launcher_sha256 == pin.launcher_sha256
    proof = json.dumps(binding.public_proof(), sort_keys=True)
    assert str(tmp_path) not in proof
    assert "/Users/" not in proof
    assert "worktrees" not in proof
    assert engine.verify_process_binding(binding, pin, process_factory=factory_for(process)) == "valid"


@pytest.mark.parametrize(
    ("process", "reason"),
    [
        (FakeProcess("/usr/bin/true"), "executable_mismatch"),
        (FakeProcess(""), "executable_missing"),
        (FakeProcess("/usr/bin/true", exe_error=psutil.AccessDenied(4242)), "process_denied"),
        (FakeProcess("/usr/bin/true", exe_error=psutil.NoSuchProcess(4242)), "process_gone"),
        (FakeProcess("/usr/bin/true", exe_error=psutil.ZombieProcess(4242)), "process_gone"),
        (FakeProcess("/usr/bin/true", running=False), "process_gone"),
        (FakeProcess("/usr/bin/true", status=psutil.STATUS_ZOMBIE), "process_gone"),
        (FakeProcess("/usr/bin/true", status=psutil.STATUS_DEAD), "process_gone"),
        (FakeProcess("/usr/bin/true", status="unknown-status"), "process_unknown"),
        (FakeProcess("/usr/bin/true", created_at=0.0), "process_unknown"),
        (FakeProcess("/usr/bin/true", created_at=math.nan), "process_unknown"),
        (FakeProcess("/usr/bin/true", created_at=math.inf), "process_unknown"),
        (FakeProcess("/usr/bin/true", created_at=True), "process_unknown"),
        (FakeProcess("/usr/bin/true", created_sequence=[1000.0, 1001.0]), "process_replaced"),
    ],
)
def test_capture_process_binding_rejects_bad_process(tmp_path, process, reason):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))
    with raises_reason(reason):
        engine.capture_process_binding(pin, 4242, process_factory=factory_for(process))


@pytest.mark.parametrize("pid", [0, -1, True, "4242", None])
def test_capture_process_binding_rejects_bad_pid(tmp_path, pid):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))
    with raises_reason("malformed_pid"):
        engine.capture_process_binding(pin, pid, process_factory=factory_for(FakeProcess(str(pin.launcher))))


def test_capture_process_binding_maps_factory_errors(tmp_path):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))

    def missing(pid):
        raise psutil.NoSuchProcess(pid)

    with raises_reason("process_gone"):
        engine.capture_process_binding(pin, 4242, process_factory=missing)


def test_verify_detects_replacement_executable_and_pin_change(tmp_path):
    engine = engine_module()
    root = shell_cache(tmp_path / "cache")
    pin = capture_shell(root)
    process = FakeProcess(str(pin.launcher))
    binding = engine.capture_process_binding(pin, 4242, process_factory=factory_for(process))

    process._created_at = 1001.0
    assert engine.verify_process_binding(binding, pin, process_factory=factory_for(process)) == "process_replaced"

    process._created_at = 1000.0
    process._exe = "/usr/bin/true"
    assert engine.verify_process_binding(binding, pin, process_factory=factory_for(process)) == "executable_changed"

    process._exe = str(pin.launcher)
    (root / SHELL_RELATIVE).write_bytes(b"changed")
    assert engine.verify_process_binding(binding, pin, process_factory=factory_for(process)) == "pin_changed"


def test_verify_detects_missing_denied_and_binding_mismatch(tmp_path):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))
    process = FakeProcess(str(pin.launcher))
    binding = engine.capture_process_binding(pin, 4242, process_factory=factory_for(process))

    def missing(pid):
        raise psutil.NoSuchProcess(pid)

    def denied(pid):
        raise psutil.AccessDenied(pid)

    assert engine.verify_process_binding(binding, pin, process_factory=missing) == "process_missing"
    assert engine.verify_process_binding(binding, pin, process_factory=denied) == "process_denied"
    assert engine.verify_process_binding(object(), pin) == "malformed_binding"  # pyright: ignore[reportArgumentType]
    assert engine.verify_process_binding(binding, object()) == "malformed_binding"  # pyright: ignore[reportArgumentType]

    mismatched = engine.ProcessBinding(
        pid=binding.pid,
        created_at=binding.created_at,
        executable_basename=binding.executable_basename,
        lane=binding.lane,
        revision=binding.revision,
        version=binding.version,
        launcher_sha256="a" * 64,
        engine_image_sha256=binding.engine_image_sha256,
    )
    assert engine.verify_process_binding(mismatched, pin, process_factory=factory_for(process)) == "binding_mismatch"


def test_malformed_pin_fails_closed_in_capture_and_verify(tmp_path):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))
    process = FakeProcess(str(pin.launcher))
    binding = engine.capture_process_binding(pin, 4242, process_factory=factory_for(process))
    bad = replace(pin, launcher=Path("relative-launcher"))
    with raises_reason("malformed_pin"):
        engine.capture_process_binding(bad, 4242, process_factory=factory_for(process))
    assert engine.verify_process_binding(binding, bad, process_factory=factory_for(process)) == "malformed_binding"


def test_verify_does_not_recache_or_mutate_binding(tmp_path):
    engine = engine_module()
    pin = capture_shell(shell_cache(tmp_path / "cache"))
    process = FakeProcess(str(pin.launcher))
    binding = engine.capture_process_binding(pin, 4242, process_factory=factory_for(process))
    before = binding.public_proof()
    process._exe = "/usr/bin/true"
    assert engine.verify_process_binding(binding, pin, process_factory=factory_for(process)) == "executable_changed"
    assert binding.public_proof() == before


def test_module_import_is_inert():
    engine = engine_module()
    for name in ("psutil", "subprocess", "socket"):
        assert name not in engine.__dict__


def test_symlink_alias_is_canonicalized_and_retarget_fails_closed(tmp_path):
    engine = engine_module()
    real_root = shell_cache(tmp_path / "real-cache", b"first")
    alias_root = tmp_path / "alias-cache"
    alias_root.symlink_to(real_root, target_is_directory=True)
    pin = capture_shell(alias_root)
    assert pin.launcher == (real_root / SHELL_RELATIVE).resolve()

    second_root = shell_cache(tmp_path / "second-cache", b"second")
    alias_root.unlink()
    alias_root.symlink_to(second_root, target_is_directory=True)
    second_launcher = (second_root / SHELL_RELATIVE).resolve()
    process = FakeProcess(str(second_launcher))
    with raises_reason("executable_mismatch"):
        engine.capture_process_binding(pin, 4242, process_factory=factory_for(process))
