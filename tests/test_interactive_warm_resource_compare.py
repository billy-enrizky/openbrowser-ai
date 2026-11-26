"""Unit contracts for the warm-session resource harness. No native I/O."""

from __future__ import annotations

import json
import queue
import socket
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

from benchmarks.interactive import warm_resource_compare as warm

WIZARD_STATE = {
    "wizard_submissions": [
        {
            "name": "Alice Bench",
            "email": "alice@example.com",
            "country": "CA",
            "province": "ON",
            "plan": "pro",
            "seats": 5,
            "terms": True,
            "quote": "Q-1",
        }
    ],
    "wizard_step1": 1,
    "wizard_step2": 1,
    "wizard_submits": 1,
}


class FakeState:
    def snapshot(self) -> dict:
        return dict(WIZARD_STATE)


class FakeServer:
    base_url = "http://127.0.0.1:9999"

    def __init__(self) -> None:
        self.state = FakeState()


class FakeSampler:
    def __init__(self, script: list[dict]) -> None:
        self.script = script
        self.values: list[dict] = []

    def sample(self) -> None:
        index = min(len(self.values), len(self.script) - 1)
        self.values.append(self.script[index])

    def result(self) -> dict:
        if not self.values:
            return {}
        footprints = [item["footprint"] for item in self.values]
        return {
            "peak_tree_rss_bytes": max(item["rss"] for item in self.values),
            "peak_child_footprint_bytes": max(footprints) if all(value is not None for value in footprints) else None,
            "observed_tree_cpu_seconds": max(item["cpu"] for item in self.values),
            "observed_child_processes": 1,
            "observed_browser_during_native_run": 1,
            "read_errors": 0,
            "fatal_read_errors": 0,
            "footprint_status": "observed" if all(value is not None for value in footprints) else "unavailable",
        }


class FakeSession:
    instances: ClassVar[list[FakeSession]] = []

    def __init__(self, arm: str, *, record: dict | None = None, proactive_fatal: bool = False) -> None:
        self.arm = arm
        self.pid = 5000 + len(FakeSession.instances)
        self.inbox: queue.Queue[dict] = queue.Queue()
        self.sent: list[dict] = []
        self.received: list[dict] = []
        self.closed = False
        self.record = record or {
            "ok": True,
            "adapter_ok": True,
            "keychain_verified": True,
            "reported": {"quote": "Q-1"},
        }
        self.proactive_fatal = proactive_fatal
        self.inbox.put({"type": "ready", "payload": {"browser_version": "Test"}, "warmup_ms": 1.0, "root": "/tmp/root"})
        FakeSession.instances.append(self)

    def send(self, message: dict) -> None:
        self.sent.append(message)
        if message.get("type") == "task":
            self.inbox.put({"type": "started"})
            if self.proactive_fatal:
                self.inbox.put({"type": "fatal", "error": "boom"})
                return
            path = Path(message["record"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "task": message["task"],
                        "arm": self.arm,
                        "sample": message["sample"],
                        **self.record,
                    }
                )
            )
            self.inbox.put({"type": "done", "record": str(path)})
        elif message.get("type") == "finish":
            self.inbox.put({"type": "finished"})

    def receive(self, timeout: float) -> dict:
        try:
            message = self.inbox.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("fake session timeout") from error
        self.received.append(message)
        return message

    def wait(self, timeout: float) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fake test attempted native I/O")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(warm, "source_manifest", lambda: {"source_digest": "same"})
    monkeypatch.setattr(
        warm,
        "binding_snapshot",
        lambda arms: {
            "verified": True,
            "entry_sha256": "e",
            "actual_app_digest": "a",
            "expected_app_digest": "a",
            "dist_version": "1",
        },
    )
    monkeypatch.setattr(warm, "arm_versions", lambda arms: {arm: "test" for arm in arms})
    monkeypatch.setattr(
        warm, "cleanup_owned", lambda sampler, **kwargs: {"verified": True, "remaining": [], "errors": []}
    )


def run_lane(
    tmp_path: Path, *, arms=("a", "b"), samples=1, session_factory=None, sampler_factory=None, cleanup=None, record=None
):
    output = tmp_path / "warm-resource.jsonl"
    sessions: list[FakeSession] = []

    def factory(arm, workdir, diagnostics):
        session = FakeSession(arm, record=record)
        sessions.append(session)
        return session

    def sampler(pid, marker):
        return FakeSampler(
            [
                {"cpu": 1.0, "rss": 100, "footprint": 50},
                {"cpu": 1.5, "rss": 120, "footprint": 60},
                {"cpu": 2.0, "rss": 110, "footprint": 55},
            ]
        )

    records = warm.run_warm_lane(
        list(arms),
        ["wizard"],
        samples,
        output,
        workdir=tmp_path,
        interval=0.01,
        worker_timeout=5.0,
        session_factory=session_factory or factory,
        fixture_factory=lambda: (FakeServer(), None),
        fixture_stop=lambda server, thread: None,
        sampler_factory=sampler_factory or sampler,
        cleanup=cleanup or (lambda sampler, **kwargs: {"verified": True, "remaining": [], "errors": []}),
    )
    return output, records, sessions


def test_warm_plan_rotates_every_arm_through_each_position():
    arms = [f"arm{i}" for i in range(7)]
    plan = warm.warm_plan(arms, ["wizard", "modal", "download", "tabs"], 7)
    assert len(plan) == 196
    for task in ("wizard", "modal", "download", "tabs"):
        positions = {arm: set() for arm in arms}
        for entry in plan:
            if entry.task == task:
                positions[entry.arm].add(entry.position)
        assert all(values == set(range(7)) for values in positions.values())


def test_lane_records_resource_metrics_and_cpu_delta(tmp_path):
    output, records, sessions = run_lane(tmp_path)
    assert len(records) == 2
    assert all(record["ok"] and record["oracle_ok"] and record["resource_ok"] for record in records)
    assert all(record["warm_peak_tree_rss_bytes"] == 120 for record in records)
    assert all(record["warm_peak_child_footprint_bytes"] == 60 for record in records)
    assert all(record["warm_task_tree_cpu_delta_seconds"] == 1.0 for record in records)
    assert all(record["latency_claim_allowed"] is False for record in records)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["record_type"] == "end"
    assert terminal["cleanup_verified"] is True
    assert terminal["provenance_verified"] is True
    assert terminal["observations"] == 2
    assert all(session.closed for session in sessions)


def test_ready_arrives_before_any_task(tmp_path):
    _output, _records, sessions = run_lane(tmp_path)
    for session in sessions:
        assert session.received[0]["type"] == "ready"
        assert session.sent[0]["type"] == "task"


def test_unknown_footprint_is_null(tmp_path):
    def sampler(pid, marker):
        return FakeSampler(
            [
                {"cpu": 1.0, "rss": 100, "footprint": None},
                {"cpu": 1.5, "rss": 120, "footprint": 60},
                {"cpu": 2.0, "rss": 110, "footprint": 55},
            ]
        )

    _output, records, _ = run_lane(tmp_path, sampler_factory=sampler)
    assert records[0]["warm_peak_child_footprint_bytes"] is None
    assert records[0]["warm_footprint_status"] == "unavailable"


def test_failed_observation_is_retained(tmp_path):
    def factory(arm, workdir, diagnostics):
        return FakeSession(arm, proactive_fatal=True)

    output, records, _ = run_lane(tmp_path, session_factory=factory)
    assert all(record["ok"] is False for record in records)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["failures"] == 2


def test_existing_output_is_rejected(tmp_path):
    output = tmp_path / "warm-resource.jsonl"
    output.write_text("existing\n")
    with pytest.raises(FileExistsError):
        warm.run_warm_lane(
            ["a"],
            ["wizard"],
            1,
            output,
            workdir=tmp_path,
            interval=0.01,
            worker_timeout=5.0,
            session_factory=lambda arm, workdir, diagnostics: FakeSession(arm),
            fixture_factory=lambda: (FakeServer(), None),
            fixture_stop=lambda server, thread: None,
            sampler_factory=lambda pid, marker: FakeSampler([{"cpu": 1.0, "rss": 1, "footprint": 1}]),
            cleanup=lambda sampler, **kwargs: {"verified": True},
        )


def test_unverified_cleanup_aborts_lane(tmp_path):
    output, _records, _ = run_lane(
        tmp_path, cleanup=lambda sampler, **kwargs: {"verified": False, "errors": ["survivor"]}
    )
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["aborted"] is True
    assert terminal["cleanup_verified"] is False


def test_provenance_mismatch_withholds_verified(tmp_path, monkeypatch):
    calls = {"count": 0}

    def source():
        calls["count"] += 1
        return {"source_digest": "before" if calls["count"] == 1 else "after"}

    monkeypatch.setattr(warm, "source_manifest", source)
    output, _records, _ = run_lane(tmp_path)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["provenance_verified"] is False
    assert terminal["source_unchanged"] is False
