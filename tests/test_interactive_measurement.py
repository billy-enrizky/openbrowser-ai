"""Deterministic clock tests for the native task measurement boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.interactive import measurement


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance(self, value):
        self.now += value


def test_task_timer_keeps_native_work_and_exposes_validation_separately():
    clock = Clock()

    @measurement.validation_only
    def check():
        clock.advance(30)
        return "proof"

    with measurement.TaskTimer(clock=clock) as timer:
        clock.advance(10)
        assert check() == "proof"
        clock.advance(20)
    assert timer.public() == {"wall_ns": 60, "validation_ns": 30, "task_ns": 30, "validation_calls": 1}


def test_nested_checks_exclude_each_interval_once():
    clock = Clock()

    @measurement.validation_only
    def inner():
        clock.advance(20)

    @measurement.validation_only
    def outer():
        clock.advance(10)
        inner()
        clock.advance(5)

    with measurement.TaskTimer(clock=clock) as timer:
        clock.advance(40)
        outer()
    assert timer.public() == {"wall_ns": 75, "validation_ns": 35, "task_ns": 40, "validation_calls": 2}


def test_exception_remains_visible_and_restores_the_active_timer():
    clock = Clock()

    @measurement.validation_only
    def fail():
        clock.advance(7)
        raise LookupError("expected")

    with pytest.raises(LookupError), measurement.TaskTimer(clock=clock) as timer:
        clock.advance(11)
        fail()
    assert timer.public()["task_ns"] == 11
    assert timer.public()["validation_ns"] == 7
    with pytest.raises(LookupError):
        fail()
    assert timer.public()["wall_ns"] == 18


def test_validation_decorator_is_inert_without_a_task_timer():
    calls = []

    @measurement.validation_only
    def check(value, *, required):
        calls.append((value, required))
        return None

    assert check(1, required=False) is None
    assert calls == [(1, False)]
    assert check.__name__ == "check"


def test_timer_rejects_nesting_and_can_measure_a_later_task():
    clock = Clock()
    with (
        measurement.TaskTimer(clock=clock),
        pytest.raises(RuntimeError, match="already active"),
        measurement.TaskTimer(clock=clock),
    ):
        pytest.fail("a second timer replaced the active timer")
    with measurement.TaskTimer(clock=clock) as later:
        clock.advance(17)
    assert later.public()["task_ns"] == 17


def test_unfinished_timer_has_no_publishable_duration():
    timer = measurement.TaskTimer(clock=Clock())
    with pytest.raises(RuntimeError, match="finished"):
        timer.public()


def test_all_existing_keychain_checks_use_the_validation_boundary(monkeypatch):
    from benchmarks.interactive.adapters import base

    clock = Clock()

    def table(marker):
        clock.advance(13)
        return ["chrome --user-data-dir=/profile --use-mock-keychain"]

    monkeypatch.setattr(base, "process_commands_matching", table)
    with measurement.TaskTimer(clock=clock) as timer:
        assert base.verify_mock_keychain("/profile-marker") is True
        clock.advance(19)
    assert timer.public()["validation_ns"] == 13
    assert timer.public()["task_ns"] == 19


def test_failed_keychain_checks_keep_their_false_result(monkeypatch):
    from benchmarks.interactive.adapters import base

    clock = Clock()

    def table(marker):
        clock.advance(13)
        return ["chrome --user-data-dir=/profile"]

    monkeypatch.setattr(base, "process_commands_matching", table)
    with measurement.TaskTimer(clock=clock) as timer:
        assert base.verify_mock_keychain("/profile-marker") is False
    assert timer.public()["validation_ns"] == 13


def test_cached_proof_identity_checks_use_the_same_boundary(monkeypatch):
    from benchmarks.interactive.adapters import launch_proof

    clock = Clock()

    def canonical(marker):
        clock.advance(23)
        return None

    monkeypatch.setattr(launch_proof, "canonical_profile_marker", canonical)
    proof = launch_proof.LaunchProof(
        launch_proof.ProcessIdentity(10, 1.0), launch_proof.ProcessIdentity(20, 2.0), "/profile", None, True
    )
    with measurement.TaskTimer(clock=clock) as timer:
        result = launch_proof.proof_status(proof, profile_marker="/profile", server_pid=10, context_id=None)
    assert result != "valid"
    assert timer.public()["validation_ns"] == 23


def test_unknown_keychain_proof_remains_unknown():
    from benchmarks.interactive.adapters import base

    clock = Clock()
    with measurement.TaskTimer(clock=clock) as timer:
        assert base.verify_mock_keychain("/profile-marker", timeout=0) is None
    assert timer.public()["validation_calls"] == 0


def test_keychain_polling_keeps_browser_startup_wait_inside_the_task(monkeypatch):
    from benchmarks.interactive.adapters import base

    clock = Clock()
    deadlines = iter([0.0, 0.0, 0.21])

    def commands(marker):
        clock.advance(22_000_000)
        return []

    monkeypatch.setattr(base, "process_commands_matching", commands)
    monkeypatch.setattr(
        base,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(deadlines), sleep=lambda seconds: clock.advance(int(seconds * 1_000_000_000))
        ),
    )
    with measurement.TaskTimer(clock=clock) as timer:
        assert base.verify_mock_keychain("/profile-marker", timeout=0.1) is None
    assert timer.public()["validation_ns"] == 22_000_000
    assert timer.public()["task_ns"] == 200_000_000


def test_cold_runner_preserves_native_task_time_and_separates_proof(monkeypatch, tmp_path):
    from benchmarks.interactive import run_benchmark
    from benchmarks.interactive.adapters import base

    clock = Clock()
    server = SimpleNamespace(base_url="http://fixture.invalid", state=SimpleNamespace(snapshot=dict))
    monkeypatch.setattr(run_benchmark, "start_fixture", lambda: (server, None))
    monkeypatch.setattr(run_benchmark, "stop_fixture", lambda *args: None)
    monkeypatch.setattr(run_benchmark, "terminate_matching", lambda *args: None)
    monkeypatch.setattr(run_benchmark, "_binary_version", lambda *args: "fake")
    monkeypatch.setitem(run_benchmark.ORACLES, "wizard", lambda *args: SimpleNamespace(ok=True, detail="exact"))

    def commands(marker):
        clock.advance(30_000_000)
        return ["chrome --user-data-dir=/profile --use-mock-keychain"]

    def run(*args, **kwargs):
        clock.advance(10_000_000)
        verified = base.verify_mock_keychain("/profile-marker")
        clock.advance(20_000_000)
        return base.ArmResult(ok=True, keychain_verified=verified)

    monkeypatch.setattr(base, "process_commands_matching", commands)
    monkeypatch.setattr(run_benchmark, "TaskTimer", lambda: measurement.TaskTimer(clock=clock), raising=False)
    monkeypatch.setattr(
        run_benchmark,
        "make_arm",
        lambda *a, **k: SimpleNamespace(run=run, cleanup=lambda *a: clock.advance(90_000_000)),
    )
    row = run_benchmark.run_one("wizard", "openbrowser-fast", 0, tmp_path, False)
    assert row["ok"] is True
    assert row["latency_ms"] == 30
    assert row["measurement"] == {
        "wall_ns": 60_000_000,
        "validation_ns": 30_000_000,
        "task_ns": 30_000_000,
        "validation_calls": 1,
    }


def test_warm_runner_uses_the_identical_measurement_boundary(monkeypatch, tmp_path):
    import sys

    from benchmarks.interactive import warm_benchmark
    from benchmarks.interactive.adapters import base

    clock = Clock()
    server = SimpleNamespace(base_url="http://fixture.invalid", state=SimpleNamespace(snapshot=dict))
    monkeypatch.setattr(warm_benchmark, "start_fixture", lambda: (server, None))
    monkeypatch.setattr(warm_benchmark, "stop_fixture", lambda *args: None)
    monkeypatch.setattr(warm_benchmark, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(warm_benchmark, "_loads", list)
    monkeypatch.setitem(warm_benchmark.ORACLES, "wizard", lambda *args: SimpleNamespace(ok=True, detail="exact"))

    def commands(marker):
        clock.advance(30_000_000)
        return ["chrome --user-data-dir=/profile --use-mock-keychain"]

    def run(*args, **kwargs):
        clock.advance(10_000_000)
        verified = base.verify_mock_keychain("/profile-marker")
        clock.advance(20_000_000)
        return base.ArmResult(ok=True, keychain_verified=verified)

    monkeypatch.setattr(base, "process_commands_matching", commands)
    monkeypatch.setattr(warm_benchmark, "TaskTimer", lambda: measurement.TaskTimer(clock=clock), raising=False)
    arm = SimpleNamespace(
        run=run, cleanup=lambda *a: None, warmup=lambda *a: ({"ok": True, "browser_version": "fake"}, 0.0)
    )
    monkeypatch.setattr(warm_benchmark, "make_arm", lambda *a: arm)
    output = tmp_path / "warm.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warm_benchmark",
            "--arms",
            "openbrowser-fast",
            "--tasks",
            "wizard",
            "--samples",
            "1",
            "--workdir",
            str(tmp_path / "runs"),
            "--output",
            str(output),
        ],
    )
    assert warm_benchmark.main() == 0
    import json

    row = json.loads(output.read_text())
    assert row["ok"] is True
    assert row["latency_ms"] == 30
    assert row["measurement"] == {
        "wall_ns": 60_000_000,
        "validation_ns": 30_000_000,
        "task_ns": 30_000_000,
        "validation_calls": 1,
    }
