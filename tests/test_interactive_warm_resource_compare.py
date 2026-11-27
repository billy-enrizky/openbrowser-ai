"""Unit contracts for the warm-session resource harness. No native I/O."""

from __future__ import annotations

import io
import json
import queue
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from benchmarks.interactive import resource_compare
from benchmarks.interactive import warm_resource_compare as warm
from benchmarks.interactive.fixtures.server import BENCH_PASSWORD, REPORT_BYTES
from benchmarks.interactive.resource_compare import ProcessTreeSampler, cleanup_owned

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
        wrong_record_path: bool = False,
        sibling_root: bool = False,
        wrong_downloads: bool = False,
        event_log: list[str] | None = None,
        no_native_done: bool = False,
        no_verify: bool = False,
        bad_native_nonce: bool = False,
        bad_done_nonce: bool = False,
    ) -> None:
        self.arm = arm
        self.workdir = Path(workdir)
        self.pid = 5000 + len(FakeSession.instances)
        self.inbox: queue.Queue[dict] = queue.Queue()
        self.sent: list[dict] = []
        self.received: list[dict] = []
        self.closed = False
        self.receive_timeouts: list[float] = []
        self.task_downloads: list[Path] = []
        self.marker_seen_at_task: list[bool] = []
        self.record_override = record_override or {}
        self.event_log = event_log
        self.no_native_done = no_native_done
        self.no_verify = no_verify
        self.bad_native_nonce = bad_native_nonce
        self.bad_done_nonce = bad_done_nonce
        self.pending_records: dict[str, Path] = {}
        self.pending_downloads: dict[str, Path] = {}
        self.proactive_fatal = proactive_fatal
        self.ready_timeout = ready_timeout
        self.bad_nonce = bad_nonce
        self.bad_record = bad_record
        self.wrong_record_path = wrong_record_path
        if bad_root:
            self.root = Path("/tmp/not-owned")
        elif sibling_root:
            self.root = (Path(workdir).parent / "warm-sibling").resolve()
        else:
            self.root = Path(workdir).resolve()
        self.downloads = self.root / ("other-downloads" if wrong_downloads else "downloads")
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

    def _log(self, value: str) -> None:
        if self.event_log is not None:
            self.event_log.append(value)

    def send(self, message: dict) -> None:
        self.sent.append(message)
        self._log(f"send:{message.get('type')}")
        if message.get("type") == "task":
            nonce = message.get("nonce")
            message_nonce = "wrong" if self.bad_nonce else nonce
            self.task_downloads.append(Path(message["downloads"]))
            self.marker_seen_at_task.append((Path(message["downloads"]) / "stale.tmp").exists())
            self.pending_records[str(nonce)] = Path(message["record"])
            self.pending_downloads[str(nonce)] = Path(message["downloads"])
            self.inbox.put(
                {
                    "type": "started",
                    "arm": self.arm,
                    "task": message["task"],
                    "sample": message["sample"],
                    "nonce": message_nonce,
                    "before": {"status": "valid", "eligible": True},
                }
            )
            if self.proactive_fatal:
                self.inbox.put({"type": "fatal", "error": "boom"})
                return
        elif message.get("type") == "capture":
            nonce = message.get("nonce")
            path = self.pending_records[str(nonce)]
            path.parent.mkdir(parents=True, exist_ok=True)
            emitted = path.with_name(path.stem + "-foreign.json") if self.wrong_record_path else path
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
            emitted.write_text(json.dumps(record))
            if message["task"] == "download":
                downloads = self.pending_downloads[str(nonce)]
                downloads.mkdir(parents=True, exist_ok=True)
                (downloads / "report.csv").write_bytes(REPORT_BYTES)
            if not self.no_native_done:
                self.inbox.put(
                    {
                        "type": "native_done",
                        "arm": self.arm,
                        "task": message["task"],
                        "sample": message["sample"],
                        "nonce": "wrong" if (self.bad_nonce or self.bad_native_nonce) else nonce,
                    }
                )
        elif message.get("type") == "verify":
            nonce = message.get("nonce")
            if self.no_verify:
                return
            path = Path(self.pending_records[str(nonce)])
            emitted = path.with_name(path.stem + "-foreign.json") if self.wrong_record_path else path
            self.inbox.put(
                {
                    "type": "done",
                    "arm": self.arm,
                    "task": message["task"],
                    "sample": message["sample"],
                    "nonce": "wrong" if (self.bad_nonce or self.bad_done_nonce) else nonce,
                    "record": str(emitted),
                }
            )
        elif message.get("type") == "finish":
            self.inbox.put({"type": "finished"})

    def receive(self, timeout: float) -> dict:
        self.receive_timeouts.append(timeout)
        if self.ready_timeout:
            raise TimeoutError("fake session timeout")
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
    fixture_stop=None,
    worker_timeout=5.0,
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
                {"cpu": 2.0, "rss": 120, "footprint": 60},
            ]
        )

    records = warm.run_warm_lane(
        list(arms),
        list(tasks),
        samples,
        output,
        workdir=tmp_path,
        interval=0.01,
        worker_timeout=worker_timeout,
        session_factory=session_factory or factory,
        fixture_factory=fixture_factory or (lambda: (FakeServer(), None)),
        fixture_stop=fixture_stop or (lambda server, thread: None),
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


def test_sibling_ready_root_is_rejected(tmp_path):
    output, records, sessions = run_lane(
        tmp_path,
        arms=("a",),
        session_options={"sibling_root": True},
        worker_timeout=0.05,
    )
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert records == []
    assert terminal["setup_error"] == "ValueError"
    assert all(session.closed for session in sessions)


def test_wrong_ready_downloads_is_rejected(tmp_path):
    output, records, sessions = run_lane(
        tmp_path,
        arms=("a",),
        session_options={"wrong_downloads": True},
        worker_timeout=0.05,
    )
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert records == []
    assert terminal["setup_error"] == "ValueError"
    assert all(session.closed for session in sessions)


def test_wrong_nonce_and_stale_record_are_rejected(tmp_path):
    _output, records, _ = run_lane(tmp_path, session_options={"bad_nonce": True})
    assert all(record["ok"] is False for record in records)
    _output, records, _ = run_lane(tmp_path / "stale", session_options={"bad_record": True})
    assert all(record["ok"] is False for record in records)


def test_foreign_record_path_is_rejected(tmp_path):
    _output, records, _ = run_lane(tmp_path, session_options={"wrong_record_path": True})
    assert all(record["ok"] is False for record in records)
    assert all("record path" in (record["error"] or "") for record in records)


def test_task_downloads_must_match_warmed_context(tmp_path):
    ctx = warm.make_owned_context(tmp_path / "owned", "a")
    assert warm._task_downloads({"downloads": str(ctx.downloads)}, ctx) == ctx.downloads
    with pytest.raises(ValueError):
        warm._task_downloads({"downloads": str(tmp_path / "other")}, ctx)
    with pytest.raises(ValueError):
        warm._task_downloads({}, ctx)


def test_main_returns_nonzero_for_invalid_lane(tmp_path, monkeypatch):
    output = tmp_path / "warm-resource.jsonl"
    sample = {
        "record_type": "sample",
        "task": "wizard",
        "arm": "a",
        "sample": 0,
        "ok": True,
        "adapter_ok": True,
        "oracle_ok": True,
        "keychain_verified": True,
        "resource_ok": True,
    }
    terminal = {
        "record_type": "end",
        "valid": True,
        "failures": 0,
        "observations": 1,
        "expected_observations": 1,
        "provenance_verified": True,
        "cleanup_verified": True,
        "terminal": True,
    }

    def write_lane(records, end):
        def fake_run(arms, tasks, samples, path, **kwargs):
            path.write_text("\n".join(json.dumps(item) for item in [{"record_type": "run"}, *records, end]) + "\n")
            return records

        return fake_run

    monkeypatch.setattr(warm, "run_warm_lane", write_lane([sample], terminal))
    assert warm.main(["--arms", "a", "--tasks", "wizard", "--samples", "1", "--output", str(output)]) == 0

    invalid = {**terminal, "valid": False, "failures": 1}
    monkeypatch.setattr(warm, "run_warm_lane", write_lane([{**sample, "resource_ok": False}], invalid))
    assert warm.main(["--arms", "a", "--tasks", "wizard", "--samples", "1", "--output", str(output)]) == 1


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
    records = [
        {
            "task": "wizard",
            "arm": "a",
            "sample": 0,
            "ok": True,
            "adapter_ok": True,
            "oracle_ok": True,
            "keychain_verified": True,
            "resource_ok": True,
        }
    ]
    terminal = {
        "terminal": True,
        "valid": True,
        "failures": 0,
        "observations": 1,
        "expected_observations": 1,
        "provenance_verified": True,
        "cleanup_verified": True,
    }
    assert warm.lane_valid(records, terminal) is True
    assert warm.lane_valid(records, {**terminal, "valid": False}) is False
    assert warm.lane_valid(records, {**terminal, "terminal": False}) is False
    assert warm.lane_valid(records, {**terminal, "failures": 1}) is False
    assert warm.lane_valid(records, {**terminal, "observations": 0}) is False
    assert warm.lane_valid(records, {**terminal, "provenance_verified": False}) is False
    assert warm.lane_valid(records, {**terminal, "cleanup_verified": False}) is False
    assert warm.lane_valid(records, {**terminal, "setup_error": "boom"}) is False
    assert warm.lane_valid(records, {**terminal, "fixture_errors": [{"phase": "stop"}]}) is False
    assert warm.lane_valid([], terminal) is False
    assert warm.lane_valid("not-a-list", terminal) is False  # pyright: ignore[reportArgumentType]
    assert warm.lane_valid([{**records[0], "arm": ""}], terminal) is False
    duplicate_terminal = {**terminal, "observations": 2, "expected_observations": 2}
    assert warm.lane_valid([*records, *records], duplicate_terminal) is False
    assert warm.lane_valid(records, terminal, arms=("a",), tasks=("wizard",), samples=1) is True
    assert warm.lane_valid(records, terminal, arms=("b",), tasks=("wizard",), samples=1) is False
    assert warm.lane_valid(records, terminal, arms=("a",), tasks=("modal",), samples=1) is False
    assert warm.lane_valid(records, terminal, arms=("a",), tasks=("wizard",), samples=2) is False


class ExitedPopen:
    def __init__(self):
        self.pid = 123456
        self.stdin = __import__("io").StringIO()
        self.stdout = iter([])
        self.stderr = None

    def poll(self):
        return 0

    def terminate(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


def test_process_session_fails_fast_when_worker_exited(tmp_path, monkeypatch):
    fake = ExitedPopen()
    monkeypatch.setattr(warm.subprocess, "Popen", lambda *args, **kwargs: fake)
    session = warm.ProcessSession(
        ["python", "-c", "pass"],
        cwd=tmp_path,
        stderr_path=tmp_path / "worker.stderr.log",
        env={},
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited"):
        session.receive(1.0)
    assert time.monotonic() - started < 0.5


def test_ready_failure_cleans_preallocated_owned_sampler(tmp_path):
    sessions = []
    samplers = []
    cleanups = []

    def factory(arm, workdir, diagnostics):
        session = FakeSession(arm, Path(workdir), ready_timeout=True)
        sessions.append(session)
        return session

    def sampler_factory(pid, marker):
        sampler = FakeSampler([{"cpu": 1.0, "rss": 1, "footprint": 1}])
        samplers.append((marker, sampler))
        return sampler

    def cleanup(sampler, **kwargs):
        cleanups.append(sampler)
        return {"verified": True, "errors": []}

    output, records, _ = run_lane(
        tmp_path,
        arms=("a",),
        session_factory=factory,
        sampler_factory=sampler_factory,
        cleanup=cleanup,
        worker_timeout=0.05,
    )
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert records == []
    assert len(samplers) == 1
    marker, sampler = samplers[0]
    assert marker == str(sessions[0].workdir.resolve())
    assert sessions[0].workdir.name.startswith("warm-")
    assert len(sessions[0].workdir.name) == len("warm-") + 10
    assert cleanups == [sampler]
    assert terminal["cleanup_verified"] is True
    assert terminal["setup_error"] is not None
    assert terminal["valid"] is False


def test_preallocated_marker_cleans_reparented_descendant(tmp_path, monkeypatch):
    owned = (tmp_path / "warm-a-marker").resolve()
    owned.mkdir()

    class Descendant:
        pid = 200

        def __init__(self):
            self.alive = True
            self.terminated = False
            self.killed = False

        def create_time(self):
            return 2.0

        def is_running(self):
            return self.alive

        def status(self):
            return "running" if self.alive else "zombie"

        def children(self, recursive=True):
            return []

        def memory_info(self):
            return SimpleNamespace(rss=1024)

        def cpu_times(self):
            return SimpleNamespace(user=0.0, system=0.0)

        def name(self):
            return "chrome-headless-shell"

        def cmdline(self):
            return [f"--user-data-dir={owned}"]

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.killed = True
            self.alive = False

    class ProcessInfo:
        pid = 200
        info: ClassVar[dict] = {
            "pid": 200,
            "cmdline": [f"--user-data-dir={owned}"],
            "create_time": 2.0,
        }

    child = Descendant()

    def factory(pid):
        if pid == 200:
            return child
        raise resource_compare.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(resource_compare.psutil, "process_iter", lambda attrs=None: [ProcessInfo()])
    sampler = ProcessTreeSampler(
        100,
        process_factory=factory,
        footprint_reader=lambda pid: None,
        footprint_supported=False,
        ownership_marker=str(owned),
    )
    report = cleanup_owned(sampler, interval=0.01, timeout=0.2)
    assert report["verified"] is True
    assert report["initial_leftovers"] == 1
    assert child.terminated is True
    assert report["remaining"] == []


def test_owned_context_has_no_nested_timestamp(tmp_path):
    owned = tmp_path / "warm-abcdef1234"
    ctx = warm.make_owned_context(owned, "playwright-mcp")
    assert ctx.root == owned.resolve()
    assert ctx.downloads.parent == ctx.root
    assert ctx.profile.parent == ctx.root
    assert ctx.tmp.parent == ctx.root
    assert sorted(path.name for path in ctx.root.iterdir()) == [
        "config",
        "downloads",
        "home",
        "profile",
        "tmp",
    ]


def test_worker_warmup_failure_emits_fatal_and_cleans(tmp_path, monkeypatch):
    messages = []
    cleaned = []

    class Arm:
        def warmup(self, ctx):
            raise RuntimeError(f"warmup exploded {BENCH_PASSWORD}")

        def cleanup(self, ctx):
            cleaned.append(True)

    monkeypatch.setattr(warm, "make_arm", lambda name: Arm())
    monkeypatch.setattr(warm, "_emit", messages.append)
    assert warm.run_worker("a", tmp_path) == 2
    assert len(messages) == 1
    assert messages[0]["type"] == "fatal"
    assert messages[0]["error"] == "warmup failed: RuntimeError"
    assert BENCH_PASSWORD not in messages[0]["error"]
    assert cleaned == [True]


def test_fixture_start_failure_is_recorded_and_workers_closed(tmp_path):
    def broken_fixture():
        raise RuntimeError("fixture start failed")

    output, records, sessions = run_lane(tmp_path, fixture_factory=broken_fixture)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert records == []
    assert terminal["fixture_errors"][0]["phase"] == "start"
    assert terminal["valid"] is False
    assert all(session.closed for session in sessions)


def test_fixture_stop_failure_invalidates_lane(tmp_path):
    def broken_stop(server, thread):
        raise RuntimeError("fixture stop failed")

    output, records, _ = run_lane(tmp_path, fixture_stop=broken_stop)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert len(records) == 2
    assert all(item["phase"] == "stop" for item in terminal["fixture_errors"])
    assert terminal["valid"] is False


def test_fixture_setup_error_stops_fixture(tmp_path):
    stops = []
    calls = []

    def sampler_factory(pid, marker):
        calls.append(marker)
        if len(calls) > 1:
            raise RuntimeError("sampler setup failed")
        return FakeSampler([{"cpu": 1.0, "rss": 100, "footprint": 50}])

    def stop(server, thread):
        stops.append(True)

    output, records, _ = run_lane(
        tmp_path,
        arms=("a",),
        sampler_factory=sampler_factory,
        fixture_stop=stop,
    )
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert records == []
    assert stops == [True]
    assert terminal["fixture_errors"][0]["phase"] == "setup"
    assert terminal["valid"] is False


class SecretErrorSession(FakeSession):
    def receive(self, timeout: float) -> dict:
        self.receive_timeouts.append(timeout)
        raise RuntimeError(f"secret {BENCH_PASSWORD}")


def test_setup_error_is_type_only(tmp_path):
    def factory(arm, workdir, diagnostics):
        return SecretErrorSession(arm, Path(workdir))

    output, records, _ = run_lane(
        tmp_path,
        arms=("a",),
        session_factory=factory,
        worker_timeout=0.05,
    )
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert records == []
    assert terminal["setup_error"] == "RuntimeError"
    assert BENCH_PASSWORD not in json.dumps(terminal)


def test_finish_receive_is_bounded(tmp_path):
    output, records, sessions = run_lane(
        tmp_path,
        arms=("a",),
        session_options={"ready_timeout": True},
        worker_timeout=180.0,
    )
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert records == []
    assert sessions[0].receive_timeouts
    assert sessions[0].receive_timeouts[-1] <= warm.FINISH_GRACE_SECONDS
    assert terminal["valid"] is False


def test_default_sampler_retries_once(monkeypatch):
    calls = []

    def flaky(pid, marker):
        calls.append((pid, marker))
        if len(calls) == 1:
            raise RuntimeError("transient sampler failure")
        return FakeSampler([{"cpu": 1.0, "rss": 1, "footprint": 1}])

    monkeypatch.setattr(warm, "_default_sampler", flaky)
    sampler = warm._make_owned_sampler(123, "/tmp/owned-marker", warm._default_sampler)
    assert len(calls) == 2
    assert isinstance(sampler, FakeSampler)


def test_non_default_sampler_failure_does_not_retry():
    calls = []

    def broken(pid, marker):
        calls.append((pid, marker))
        raise RuntimeError("persistent sampler failure")

    with pytest.raises(RuntimeError, match="persistent"):
        warm._make_owned_sampler(123, "/tmp/owned-marker", broken)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changed_key", ["backend_entry_sha256", "native_manifest_sha256", "native_source_digest", "native_target"]
)
def test_warm_lane_rejects_any_native_binding_change(monkeypatch, tmp_path, changed_key):
    before = {
        "verified": True,
        "entry_sha256": "entry",
        "actual_app_digest": "app",
        "expected_app_digest": "app",
        "dist_version": "v",
        changed_key: "before",
    }
    bindings = iter([before, {**before, changed_key: "after"}])
    monkeypatch.setattr(warm, "binding_snapshot", lambda arms: next(bindings))
    output, records, _ = run_lane(tmp_path, arms=("a",))
    end = json.loads(output.read_text().splitlines()[-1])
    assert end["binding_unchanged"] is False
    assert end["provenance_verified"] is False
    assert end["valid"] is False
    assert all(row["provenance_verified"] is False for row in records)


class BarrierSampler(FakeSampler):
    def __init__(self, log: list[str]):
        super().__init__([{"cpu": 1.0, "rss": 100, "footprint": 50}, {"cpu": 2.0, "rss": 120, "footprint": 60}])
        self.log = log

    def sample(self) -> None:
        self.log.append("sample")
        super().sample()


def test_barriers_place_baseline_after_started_and_final_before_verify(tmp_path):
    log: list[str] = []
    _output, records, sessions = run_lane(
        tmp_path,
        arms=("a",),
        session_options={"event_log": log},
        sampler_factory=lambda pid, marker: BarrierSampler(log),
    )
    assert [session.sent[0]["type"] for session in sessions] == ["task"]
    assert sessions[0].sent[1]["type"] == "capture"
    assert sessions[0].sent[2]["type"] == "verify"
    assert log == ["sample", "send:task", "sample", "send:capture", "sample", "send:verify", "send:finish"]
    assert records[0]["warm_task_tree_cpu_delta_seconds"] == 1.0
    assert records[0]["resource_ok"] is True


def test_method_marks_the_corrected_boundary(tmp_path):
    output, _records, _sessions = run_lane(tmp_path, arms=("a",))
    header = json.loads(output.read_text().splitlines()[0])
    assert header["method"] == "warm-session-resources-v2"


def test_native_done_timeout_is_retained_as_failure(tmp_path):
    output, records, _sessions = run_lane(
        tmp_path,
        arms=("a",),
        session_options={"no_native_done": True},
        worker_timeout=0.05,
    )
    assert records[0]["ok"] is False
    assert records[0]["resource_ok"] is False
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["valid"] is False


def test_parent_rejects_native_done_binding(tmp_path):
    _output, records, _sessions = run_lane(
        tmp_path,
        arms=("a",),
        session_options={"bad_native_nonce": True},
    )
    assert records[0]["ok"] is False


def test_parent_rejects_done_binding(tmp_path):
    _output, records, _sessions = run_lane(
        tmp_path,
        arms=("a",),
        session_options={"bad_done_nonce": True},
    )
    assert records[0]["ok"] is False


def test_verify_timeout_is_retained_as_failure(tmp_path):
    output, records, _sessions = run_lane(
        tmp_path,
        arms=("a",),
        session_options={"no_verify": True},
        worker_timeout=0.05,
    )
    assert records[0]["ok"] is False
    assert records[0]["resource_ok"] is False
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["valid"] is False


class WorkerArm:
    def __init__(self, events: list[str], *, fail: bool = False, invalid_before=None) -> None:
        self.binary = ""
        self.events = events
        self.fail = fail
        self.invalid_before = invalid_before

    def warmup(self, ctx):
        self.events.append("warmup")
        return {"ok": True, "browser_version": "fake"}, 0.0

    def run(self, ctx, *, cleanup=True):
        self.events.append("native")
        if self.invalid_before is not None:
            raise AssertionError("native must not run")
        if self.fail:
            raise RuntimeError("task failed")
        return warm.ArmResult(ok=True, reported={"quote": "Q-1"}, keychain_verified=True)

    def cleanup(self, ctx):
        self.events.append("cleanup")


class WorkerEngine:
    def __init__(self, events: list[str], *, valid: bool = True) -> None:
        self.events = events
        self.valid = valid

    def capture_warm(self, arm, ctx):
        return SimpleNamespace(public_proof=lambda: {"status": "valid"}, eligible=True, status="valid")

    def verify_warm(self, arm, ctx):
        self.events.append("proof")
        if self.valid:
            return SimpleNamespace(
                eligible=True,
                status="valid",
                keychain_verified=True,
                public_proof=lambda: {"status": "valid", "eligible": True},
            )
        return SimpleNamespace(
            eligible=False,
            status="matched_engine_before",
            keychain_verified=None,
            public_proof=lambda: {"status": "matched_engine_before", "eligible": False},
        )

    def stamp(self, row, after, *, before=None):
        verified = bool(after.eligible and (before is None or before.eligible))
        row["engine"] = {"verified": verified}
        row["ok"] = row.get("ok") is True and verified
        if not verified:
            row["error"] = "; ".join(value for value in (row.get("error", ""), "matched_engine") if value)


def _worker_task(workdir: Path, record: Path, nonce: str = "n") -> dict:
    ctx = warm.make_owned_context(workdir, "a")
    return {
        "type": "task",
        "arm": "a",
        "task": "wizard",
        "sample": 0,
        "base_url": "http://fixture.invalid",
        "downloads": str(ctx.downloads),
        "record": str(record),
        "nonce": nonce,
    }


def _worker_messages(task: dict, *types: str) -> str:
    lines = [json.dumps(task)]
    for kind in types:
        lines.append(
            json.dumps(
                {
                    "type": kind,
                    "arm": "a",
                    "task": "wizard",
                    "sample": 0,
                    "nonce": task["nonce"],
                }
            )
        )
    return "\n".join(lines) + "\n"


def test_worker_task_exception_still_completes_handshake(tmp_path, monkeypatch):
    events: list[str] = []
    emitted: list[dict] = []
    record = tmp_path / "record.json"
    task = _worker_task(tmp_path, record)
    arm = WorkerArm(events, fail=True)
    monkeypatch.setattr(warm, "make_arm", lambda name: arm)
    monkeypatch.setattr(warm, "EngineSession", lambda pin: WorkerEngine(events))
    monkeypatch.setattr(warm, "configured_engine_pin", lambda: "pin")
    monkeypatch.setattr(warm, "_emit", emitted.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO(_worker_messages(task, "capture", "verify")))
    assert warm.run_worker("a", tmp_path) == 0
    assert events == ["warmup", "proof", "native", "proof", "cleanup"]
    assert [item["type"] for item in emitted] == ["ready", "started", "native_done", "done"]
    row = json.loads(record.read_text())
    assert row["ok"] is False
    assert "RuntimeError" in row["error"]
    assert row["engine"]["verified"] is True


def test_worker_invalid_before_skips_native_but_completes_handshake(tmp_path, monkeypatch):
    events: list[str] = []
    emitted: list[dict] = []
    record = tmp_path / "record.json"
    task = _worker_task(tmp_path, record)
    arm = WorkerArm(events, invalid_before=False)
    engine = WorkerEngine(events, valid=False)
    monkeypatch.setattr(warm, "make_arm", lambda name: arm)
    monkeypatch.setattr(warm, "EngineSession", lambda pin: engine)
    monkeypatch.setattr(warm, "configured_engine_pin", lambda: "pin")
    monkeypatch.setattr(warm, "_emit", emitted.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO(_worker_messages(task, "capture", "verify")))
    assert warm.run_worker("a", tmp_path) == 0
    assert "native" not in events
    assert [item["type"] for item in emitted] == ["ready", "started", "native_done", "done"]
    row = json.loads(record.read_text())
    assert row["ok"] is False
    assert "matched_engine_before" in row["error"]


def test_worker_capture_gate_timeout_is_bounded(tmp_path, monkeypatch):
    events: list[str] = []
    emitted: list[dict] = []
    task = _worker_task(tmp_path, tmp_path / "record.json")
    monkeypatch.setattr(warm, "make_arm", lambda name: WorkerArm(events))
    monkeypatch.setattr(warm, "EngineSession", lambda pin: WorkerEngine(events))
    monkeypatch.setattr(warm, "configured_engine_pin", lambda: "pin")
    monkeypatch.setattr(warm, "_emit", emitted.append)
    monkeypatch.setattr(warm, "WORKER_GATE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(task) + "\n"))
    assert warm.run_worker("a", tmp_path) == 2
    assert emitted[-1]["type"] == "fatal"
    assert "capture" in emitted[-1]["error"]
    assert "cleanup" in events


def test_worker_verify_gate_timeout_is_bounded(tmp_path, monkeypatch):
    events: list[str] = []
    emitted: list[dict] = []
    task = _worker_task(tmp_path, tmp_path / "record.json")
    monkeypatch.setattr(warm, "make_arm", lambda name: WorkerArm(events))
    monkeypatch.setattr(warm, "EngineSession", lambda pin: WorkerEngine(events))
    monkeypatch.setattr(warm, "configured_engine_pin", lambda: "pin")
    monkeypatch.setattr(warm, "_emit", emitted.append)
    monkeypatch.setattr(warm, "WORKER_GATE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(sys, "stdin", io.StringIO(_worker_messages(task, "capture")))
    assert warm.run_worker("a", tmp_path) == 2
    assert "native" in events
    assert emitted[-1]["type"] == "fatal"
    assert "verify" in emitted[-1]["error"]
    assert "cleanup" in events


def test_worker_rejects_stale_verify_binding(tmp_path, monkeypatch):
    events: list[str] = []
    emitted: list[dict] = []
    task = _worker_task(tmp_path, tmp_path / "record.json")
    monkeypatch.setattr(warm, "make_arm", lambda name: WorkerArm(events))
    monkeypatch.setattr(warm, "EngineSession", lambda pin: WorkerEngine(events))
    monkeypatch.setattr(warm, "configured_engine_pin", lambda: "pin")
    monkeypatch.setattr(warm, "_emit", emitted.append)
    stale = {
        "type": "verify",
        "arm": "a",
        "task": "wizard",
        "sample": 0,
        "nonce": "other",
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(_worker_messages(task, "capture") + json.dumps(stale) + "\n"),
    )
    assert warm.run_worker("a", tmp_path) == 2
    assert "native" in events
    assert emitted[-1]["type"] == "fatal"
    assert "verify" in emitted[-1]["error"]


def test_worker_gate_timeout_is_closed_and_bounded(monkeypatch):
    emitted: list[dict] = []
    monkeypatch.setattr(warm, "_emit", emitted.append)

    class TimeoutChannel:
        def next(self, timeout):
            raise TimeoutError("timeout")

    assert (
        warm._gate_message(
            TimeoutChannel(),  # pyright: ignore[reportArgumentType]
            "capture",
            arm_name="a",
            task="wizard",
            sample=0,
            nonce="n",
        )
        is None
    )
    assert emitted[-1] == {"type": "fatal", "error": "capture gate timeout"}


def test_worker_rejects_stale_capture_binding(tmp_path, monkeypatch):
    events: list[str] = []
    emitted: list[dict] = []
    task = _worker_task(tmp_path, tmp_path / "record.json")
    monkeypatch.setattr(warm, "make_arm", lambda name: WorkerArm(events))
    monkeypatch.setattr(warm, "EngineSession", lambda pin: WorkerEngine(events))
    monkeypatch.setattr(warm, "configured_engine_pin", lambda: "pin")
    monkeypatch.setattr(warm, "_emit", emitted.append)
    stale = {
        "type": "capture",
        "arm": "a",
        "task": "wizard",
        "sample": 0,
        "nonce": "other",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(task) + "\n" + json.dumps(stale) + "\n"))
    assert warm.run_worker("a", tmp_path) == 2
    assert "native" not in events
    assert emitted[-1]["type"] == "fatal"


@pytest.mark.parametrize("fault", ["baseline_sample", "baseline_result", "native_poll", "final_sample", "final_result"])
def test_sampling_failure_preserves_handshake_and_next_task(tmp_path, fault):
    samplers = []
    sessions = []

    class FaultSampler(FakeSampler):
        def __init__(self, broken):
            super().__init__([{"cpu": 1.0, "rss": 100, "footprint": 50}, {"cpu": 2.0, "rss": 120, "footprint": 60}])
            self.broken = broken
            self.sample_calls = 0
            self.result_calls = 0

        def sample(self):
            self.sample_calls += 1
            target = 1 if fault == "baseline_sample" else 2
            if self.broken and fault.endswith(("sample", "poll")) and self.sample_calls == target:
                raise OSError(0, f"inspection {BENCH_PASSWORD}")
            super().sample()

        def result(self):
            self.result_calls += 1
            target = 1 if fault == "baseline_result" else 2
            if self.broken and fault.endswith("result") and self.result_calls == target:
                raise ValueError(f"result {BENCH_PASSWORD}")
            return super().result()

    class PollSession(FakeSession):
        delayed = False

        def receive(self, timeout):
            if (
                fault == "native_poll"
                and not self.delayed
                and self.inbox.queue
                and self.inbox.queue[0].get("type") == "native_done"
            ):
                self.delayed = True
                raise TimeoutError("poll again")
            return super().receive(timeout)

    def sampler_factory(pid, marker):
        sampler = FaultSampler(broken=len(samplers) == 1)
        samplers.append(sampler)
        return sampler

    def session_factory(arm, workdir, diagnostics):
        session = PollSession(arm, workdir)
        sessions.append(session)
        return session

    output, rows, _ = run_lane(
        tmp_path,
        arms=("a",),
        tasks=("wizard", "modal"),
        sampler_factory=sampler_factory,
        session_factory=session_factory,
    )
    first, second = rows
    assert first["ok"] is first["adapter_ok"] is first["oracle_ok"] is True
    assert first["resource_ok"] is False
    for key in (
        "warm_peak_tree_rss_bytes",
        "warm_peak_child_footprint_bytes",
        "warm_task_tree_cpu_delta_seconds",
        "warm_unverified_processes",
        "warm_read_errors",
        "warm_fatal_read_errors",
    ):
        assert first[key] is None
    expected_phase = "native_run" if fault == "native_poll" else fault.split("_")[0]
    assert first["warm_sampling_error"] == {
        "phase": expected_phase,
        "operation": "result" if fault.endswith("result") else "sample",
        "error_type": "ValueError" if fault.endswith("result") else "OSError",
        "errno": None if fault.endswith("result") else 0,
    }
    assert BENCH_PASSWORD not in json.dumps(first)
    assert second["ok"] is second["oracle_ok"] is second["resource_ok"] is True
    assert second["warm_sampling_error"] is None
    assert [message["type"] for message in sessions[0].sent] == [
        "task",
        "capture",
        "verify",
        "task",
        "capture",
        "verify",
        "finish",
    ]
    assert samplers[1].sample_calls == (1 if fault.startswith("baseline") else 2)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["observations"] == 2
    assert terminal["failures"] == 1
    assert terminal["valid"] is False
    assert terminal["cleanup_verified"] is True


def test_protocol_failure_retires_only_its_worker_without_rewarm(tmp_path):
    sessions = {}
    cleanups = []

    class CheckedSession(FakeSession):
        def send(self, message):
            if self.closed and message.get("type") == "task":
                raise AssertionError("task sent to retired worker")
            super().send(message)

    def session_factory(arm, workdir, diagnostics):
        assert arm not in sessions
        session = CheckedSession(arm, workdir, bad_native_nonce=arm == "a")
        sessions[arm] = session
        return session

    def cleanup(sampler, **kwargs):
        cleanups.append(sampler)
        return {"verified": True, "remaining": [], "errors": []}

    def sampler_factory(pid, marker):
        assert not any(session.pid == pid and session.closed for session in sessions.values())
        return FakeSampler([{"cpu": 1.0, "rss": 100, "footprint": 50}, {"cpu": 2.0, "rss": 120, "footprint": 60}])

    output, rows, _ = run_lane(
        tmp_path,
        tasks=("wizard", "modal"),
        session_factory=session_factory,
        sampler_factory=sampler_factory,
        cleanup=cleanup,
    )
    assert len(sessions) == 2
    assert len(rows) == 4
    assert len([m for m in sessions["a"].sent if m["type"] == "task"]) == 1
    assert len([m for m in sessions["b"].sent if m["type"] == "task"]) == 2
    assert all(not row["ok"] and not row["resource_ok"] for row in rows if row["arm"] == "a")
    assert all(row["ok"] and row["resource_ok"] for row in rows if row["arm"] == "b")
    assert len(cleanups) == 3
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["failures"] == 2
    assert terminal["valid"] is False
    assert set(terminal["retired_workers"]) == {"a"}
    assert terminal["retired_workers"]["a"]["cleanup"]["verified"] is True


def test_orphan_discovery_exception_keeps_capture_and_verify_in_sync(tmp_path):
    created = []

    class BrokenDiscovery(FakeSampler):
        def discover_orphans(self):
            raise OSError(0, "sysctl(KERN_PROCARGS2)")

    def sampler_factory(pid, marker):
        kind = BrokenDiscovery if len(created) == 1 else FakeSampler
        sampler = kind([{"cpu": 1.0, "rss": 100, "footprint": 50}, {"cpu": 2.0, "rss": 120, "footprint": 60}])
        created.append(sampler)
        return sampler

    _output, rows, sessions = run_lane(
        tmp_path, arms=("a",), tasks=("wizard", "modal"), sampler_factory=sampler_factory
    )
    assert rows[0]["ok"] is rows[0]["oracle_ok"] is True
    assert rows[0]["resource_ok"] is False
    assert rows[0]["warm_peak_tree_rss_bytes"] is None
    assert rows[0]["warm_sampling_error"]["phase"] == "baseline"
    assert rows[0]["warm_sampling_error"]["error_type"] == "OSError"
    assert rows[1]["ok"] is rows[1]["resource_ok"] is True
    assert [message["type"] for message in sessions[0].sent].count("capture") == 2
    assert [message["type"] for message in sessions[0].sent].count("verify") == 2


def test_unverified_retirement_cleanup_aborts_before_other_native_tasks(tmp_path):
    sessions = {}
    cleanups = []

    def session_factory(arm, workdir, diagnostics):
        session = FakeSession(arm, workdir, bad_native_nonce=arm == "a")
        sessions[arm] = session
        return session

    def cleanup(sampler, **kwargs):
        cleanups.append(sampler)
        return {"verified": len(cleanups) != 1, "remaining": [], "errors": []}

    output, rows, _ = run_lane(tmp_path, tasks=("wizard", "modal"), session_factory=session_factory, cleanup=cleanup)
    assert len(rows) == 1
    assert not any(message["type"] == "task" for message in sessions["b"].sent)
    terminal = json.loads(output.read_text().splitlines()[-1])
    assert terminal["aborted"] is True
    assert terminal["valid"] is False
    assert terminal["retired_workers"]["a"]["cleanup"]["verified"] is False
