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
from benchmarks.interactive.fixtures.server import REPORT_BYTES

COMBINED_STATE = {
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
    "modal_events": [
        {"event": "open"},
        {"event": "escape"},
        {"event": "open"},
        {"event": "select", "value": "Project Atlas", "escapes": 1},
    ],
    "download_requests": 1,
    "download_completed": 1,
    "tab_events": [
        {"event": "detail_loaded", "opener": True},
        {"event": "approve"},
        {"event": "status", "status": "approved:42"},
        {"event": "detail_closed"},
    ],
}

REPORTED = {
    "wizard": {"quote": "Q-1"},
    "modal": {"selected": "Project Atlas"},
    "tabs": {"original_title": "Tabs Bench"},
    "download": {},
}


class FakeState:
    def __init__(self, state: dict) -> None:
        self.state = dict(state)

    def snapshot(self) -> dict:
        return dict(self.state)


class FakeServer:
    base_url = "http://127.0.0.1:9999"

    def __init__(self, state: dict = COMBINED_STATE) -> None:
        self.state = FakeState(state)


class FakeSampler:
    def __init__(self, script: list[dict], *, unverified: int = 0) -> None:
        self.script = script
        self.values: list[dict] = []
        self.events: list[str] = []
        self.unverified = unverified

    def discover_orphans(self) -> None:
        self.events.append("discover")

    def sample(self) -> None:
        self.events.append("sample")
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
            "unverified_processes": self.unverified,
            "read_errors": 0,
            "fatal_read_errors": 0,
            "footprint_status": "observed" if all(value is not None for value in footprints) else "unavailable",
        }


class ScriptedResultSampler(FakeSampler):
    """Return each scripted resource result in order, not a running max."""

    def __init__(self, results: list[dict]) -> None:
        super().__init__([])
        self.results = results

    def sample(self) -> None:
        self.events.append("sample")
        index = min(len(self.values), len(self.results) - 1)
        self.values.append(self.results[index])

    def result(self) -> dict:
        return dict(self.values[-1]) if self.values else {}


class FakeSession:
    instances: ClassVar[list[FakeSession]] = []

    def __init__(
        self,
        arm: str,
        workdir: Path,
        *,
        record_override: dict | None = None,
        proactive_fatal: bool = False,
        ready_timeout: bool = False,
        bad_nonce: bool = False,
        bad_record: bool = False,
        bad_root: bool = False,
    ) -> None:
        self.arm = arm
        self.pid = 5000 + len(FakeSession.instances)
        self.inbox: queue.Queue[dict] = queue.Queue()
        self.sent: list[dict] = []
        self.received: list[dict] = []
        self.closed = False
        self.task_downloads: list[Path] = []
        self.marker_seen_at_task: list[bool] = []
        self.record_override = record_override or {}
        self.proactive_fatal = proactive_fatal
        self.ready_timeout = ready_timeout
        self.bad_nonce = bad_nonce
        self.bad_record = bad_record
        self.root = Path("/tmp/not-owned") if bad_root else (workdir / f"fake-{arm}").resolve()
        self.downloads = self.root / "downloads"
        if not bad_root:
            self.downloads.mkdir(parents=True, exist_ok=True)
        if not ready_timeout:
            self.inbox.put(
                {
                    "type": "ready",
                    "payload": {"browser_version": "Test"},
                    "warmup_ms": 1.0,
                    "root": str(self.root),
                    "downloads": str(self.downloads),
                }
            )
        FakeSession.instances.append(self)

    def send(self, message: dict) -> None:
        self.sent.append(message)
        if message.get("type") == "task":
            nonce = message.get("nonce")
            message_nonce = "wrong" if self.bad_nonce else nonce
            self.task_downloads.append(Path(message["downloads"]))
            self.marker_seen_at_task.append((Path(message["downloads"]) / "stale.tmp").exists())
            self.inbox.put(
                {
                    "type": "started",
                    "arm": self.arm,
                    "task": message["task"],
                    "sample": message["sample"],
                    "nonce": message_nonce,
                }
            )
            if self.proactive_fatal:
                self.inbox.put({"type": "fatal", "error": "boom"})
                return
            path = Path(message["record"])
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "task": message["task"],
                "arm": "stale" if self.bad_record else self.arm,
                "sample": message["sample"],
                "nonce": nonce,
                "ok": True,
                "adapter_ok": True,
                "keychain_verified": True,
                "reported": REPORTED[message["task"]],
                **self.record_override,
            }
            path.write_text(json.dumps(record))
            if message["task"] == "download":
                (Path(message["downloads"]) / "report.csv").write_bytes(REPORT_BYTES)
            self.inbox.put(
                {
                    "type": "done",
                    "arm": self.arm,
                    "task": message["task"],
                    "sample": message["sample"],
                    "nonce": message_nonce,
                    "record": str(path),
                }
            )
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
    tmp_path: Path,
    *,
    arms=("a", "b"),
    tasks=("wizard",),
    samples=1,
    session_options: dict | None = None,
    session_factory=None,
    sampler_factory=None,
    cleanup=None,
    fixture_factory=None,
):
    output = tmp_path / "warm-resource.jsonl"
    sessions: list[FakeSession] = []

    def factory(arm, workdir, diagnostics):
        session = FakeSession(arm, Path(workdir), **(session_options or {}))
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
        list(tasks),
        samples,
        output,
        workdir=tmp_path,
        interval=0.01,
        worker_timeout=5.0,
        session_factory=session_factory or factory,
        fixture_factory=fixture_factory or (lambda: (FakeServer(), None)),
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


def test_all_four_warm_tasks_record_eligible_metrics(tmp_path):
    output, records, sessions = run_lane(tmp_path, tasks=("wizard", "modal", "download", "tabs"))
    assert len(records) == 8
    assert all(
        record["ok"] and record["oracle_ok"] and record["keychain_verified"] is True and record["resource_ok"]
        for record in records
    )
    assert all(record["warm_peak_tree_rss_bytes"] == 120 for record in records)
    assert all(record["warm_peak_child_footprint_bytes"] == 60 for record in records)
    assert all(record["warm_task_tree_cpu_delta_seconds"] == 1.0 for record in records)
    assert all(record["latency_claim_allowed"] is False for record in records)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["valid"] is True
    assert terminal["cleanup_verified"] is True
    assert terminal["provenance_verified"] is True
    assert terminal["observations"] == 8
    assert all(session.closed for session in sessions)


def test_ready_arrives_before_any_task(tmp_path):
    _output, _records, sessions = run_lane(tmp_path)
    for session in sessions:
        assert session.received[0]["type"] == "ready"
        assert session.sent[0]["type"] == "task"


def test_downloads_directory_is_stable_and_cleaned(tmp_path):
    _output, _records, sessions = run_lane(tmp_path, samples=2)
    session = sessions[0]
    assert len(set(session.task_downloads)) == 1
    assert session.task_downloads[0] == session.downloads
    assert all(seen is False for seen in session.marker_seen_at_task)


def test_sampler_discovers_orphans_before_every_sample(tmp_path):
    samplers: list[FakeSampler] = []

    def sampler(pid, marker):
        value = FakeSampler(
            [
                {"cpu": 1.0, "rss": 100, "footprint": 50},
                {"cpu": 1.5, "rss": 120, "footprint": 60},
                {"cpu": 2.0, "rss": 110, "footprint": 55},
            ]
        )
        samplers.append(value)
        return value

    run_lane(tmp_path, sampler_factory=sampler)
    for value in samplers:
        assert value.events[0] == "discover"
        for index, event in enumerate(value.events):
            if event == "sample":
                assert "discover" in value.events[:index]


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
    _output, records, _ = run_lane(tmp_path, session_options={"proactive_fatal": True})
    assert all(record["ok"] is False for record in records)
    terminal = json.loads((tmp_path / "warm-resource.jsonl").read_text().splitlines()[-1])
    assert terminal["failures"] == 2
    assert terminal["valid"] is False


def test_ready_timeout_closes_spawned_session_and_writes_terminal(tmp_path):
    output, records, sessions = run_lane(tmp_path, session_options={"ready_timeout": True})
    assert records == []
    assert all(session.closed for session in sessions)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["valid"] is False
    assert "setup_error" in terminal


def test_root_outside_workdir_is_rejected(tmp_path):
    output, records, sessions = run_lane(tmp_path, session_options={"bad_root": True})
    assert records == []
    assert all(session.closed for session in sessions)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["valid"] is False
    assert "setup_error" in terminal


def test_wrong_nonce_and_stale_record_are_rejected(tmp_path):
    _output, records, _ = run_lane(tmp_path, session_options={"bad_nonce": True})
    assert all(record["ok"] is False for record in records)
    _output, records, _ = run_lane(tmp_path / "stale", session_options={"bad_record": True})
    assert all(record["ok"] is False for record in records)


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
            session_factory=lambda arm, workdir, diagnostics: FakeSession(arm, Path(workdir)),
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
    assert terminal["valid"] is False


def test_provenance_mismatch_withholds_verified(tmp_path, monkeypatch):
    calls = {"count": 0}

    def source():
        calls["count"] += 1
        return {"source_digest": "before" if calls["count"] == 1 else "after"}

    monkeypatch.setattr(warm, "source_manifest", source)
    output, _records, _ = run_lane(tmp_path)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["provenance_verified"] is False
    assert terminal["valid"] is False


def test_preflight_failure_writes_terminal_footer(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(warm, "source_manifest", broken)
    output = tmp_path / "warm-resource.jsonl"
    records = warm.run_warm_lane(
        ["a"],
        ["wizard"],
        1,
        output,
        workdir=tmp_path,
        interval=0.01,
        worker_timeout=5.0,
        session_factory=lambda arm, workdir, diagnostics: FakeSession(arm, Path(workdir)),
        fixture_factory=lambda: (FakeServer(), None),
        fixture_stop=lambda server, thread: None,
        sampler_factory=lambda pid, marker: FakeSampler([{"cpu": 1.0, "rss": 1, "footprint": 1}]),
        cleanup=lambda sampler, **kwargs: {"verified": True},
    )
    assert records == []
    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines[0]["preflight_error"] == "RuntimeError"
    assert lines[-1]["valid"] is False


def test_unverified_identity_and_invalid_metrics_reject_resource(tmp_path):
    def unverified(pid, marker):
        return FakeSampler([{"cpu": 1.0, "rss": 100, "footprint": 50}], unverified=1)

    _output, records, _ = run_lane(tmp_path, sampler_factory=unverified)
    assert records[0]["resource_ok"] is False
    assert records[0]["warm_unverified_processes"] == 1

    def zero_rss(pid, marker):
        return FakeSampler(
            [
                {"cpu": 1.0, "rss": 0, "footprint": 50},
                {"cpu": 1.5, "rss": 0, "footprint": 60},
                {"cpu": 2.0, "rss": 0, "footprint": 55},
            ]
        )

    _output, records, _ = run_lane(tmp_path / "zero", sampler_factory=zero_rss)
    assert records[0]["resource_ok"] is False

    def reset_cpu(pid, marker):
        base = {
            "peak_tree_rss_bytes": 100,
            "peak_child_footprint_bytes": 50,
            "observed_child_processes": 1,
            "observed_browser_during_native_run": 1,
            "unverified_processes": 0,
            "read_errors": 0,
            "fatal_read_errors": 0,
            "footprint_status": "observed",
        }
        return ScriptedResultSampler(
            [
                {**base, "observed_tree_cpu_seconds": 2.0},
                {**base, "observed_tree_cpu_seconds": 1.5},
                {**base, "observed_tree_cpu_seconds": 1.0},
            ]
        )

    _output, records, _ = run_lane(tmp_path / "reset", sampler_factory=reset_cpu)
    assert records[0]["warm_task_tree_cpu_delta_seconds"] is None
    assert records[0]["warm_cpu_delta_error"] == "invalid_counter"
    assert records[0]["resource_ok"] is False


def test_terminal_counts_keychain_failure(tmp_path):
    output, _records, _ = run_lane(tmp_path, session_options={"record_override": {"keychain_verified": False}})
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["failures"] == 2
    assert terminal["valid"] is False


def test_lane_valid_requires_every_gate():
    records = [{"ok": True, "adapter_ok": True, "oracle_ok": True, "keychain_verified": True, "resource_ok": True}]
    terminal = {
        "valid": True,
        "failures": 0,
        "expected_observations": 1,
        "provenance_verified": True,
        "cleanup_verified": True,
    }
    assert warm.lane_valid(records, terminal) is True
    assert warm.lane_valid(records, {**terminal, "valid": False}) is False
    assert warm.lane_valid(records, {**terminal, "failures": 1}) is False
    assert warm.lane_valid(records, {**terminal, "provenance_verified": False}) is False
    assert warm.lane_valid(records, {**terminal, "cleanup_verified": False}) is False
    assert warm.lane_valid([], terminal) is False
