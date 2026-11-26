"""Fake native arms prove the uniform engine-check placement in all lanes."""

from __future__ import annotations

import io
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmarks.interactive import run_benchmark, warm_benchmark, warm_resource_compare
from benchmarks.interactive.adapters.base import ArmResult


@pytest.fixture
def probe(monkeypatch):
    events = []
    observation = SimpleNamespace(eligible=True, status="valid", public_proof=lambda: {"eligible": True})

    class Session:
        def __init__(self, pin):
            assert pin == "configured-pin"

        def task_scope(self):
            return nullcontext()

        def observe(self, arm, ctx):
            events.append("observe")
            return observation

        def capture_warm(self, arm, ctx):
            events.append("capture_warm")
            return observation

        def verify_warm(self, arm, ctx):
            events.append("verify_warm")
            return observation

        def stamp(self, row, after, *, before=None):
            assert after is observation
            if before is not None:
                assert before is observation
            events.append("stamp")
            row["engine"] = {"verified": True}

    for module in (run_benchmark, warm_benchmark, warm_resource_compare):
        monkeypatch.setattr(module, "EngineSession", Session, raising=False)
        monkeypatch.setattr(module, "configured_engine_pin", lambda: "configured-pin", raising=False)

    class Arm:
        binary = ""

        def warmup(self, ctx):
            events.append("warmup")
            return {"ok": True, "browser_version": "fake"}, 0.0

        def run(self, ctx, *, cleanup=True):
            assert cleanup is False
            events.append("native_task")
            return ArmResult(ok=True, reported={"quote": "Q-1"}, keychain_verified=True)

        def cleanup(self, ctx):
            events.append("cleanup")

    return SimpleNamespace(events=events, arm=Arm(), observation=observation)


def test_cold_proof_runs_after_native_task_and_before_cleanup(monkeypatch, tmp_path, probe):
    module = run_benchmark
    server = SimpleNamespace(base_url="http://fixture.invalid", state=SimpleNamespace(snapshot=dict))
    monkeypatch.setattr(module, "start_fixture", lambda: (server, None))
    monkeypatch.setattr(module, "stop_fixture", lambda *args: None)
    monkeypatch.setattr(module, "terminate_matching", lambda *args: None)
    monkeypatch.setattr(module, "make_arm", lambda *args, **kwargs: probe.arm)
    monkeypatch.setattr(module, "_binary_version", lambda *args: "fake")
    monkeypatch.setitem(module.ORACLES, "wizard", lambda *args: SimpleNamespace(ok=True, detail="exact"))
    row = module.run_one("wizard", "openbrowser-fast", 0, tmp_path, False)
    assert probe.events == ["native_task", "observe", "cleanup", "stamp"]
    assert row["engine"]["verified"] is True


def test_warm_latency_checks_both_sides_and_keeps_warmup_proof(monkeypatch, tmp_path, probe):
    module = warm_benchmark
    server = SimpleNamespace(base_url="http://fixture.invalid", state=SimpleNamespace(snapshot=dict))
    monkeypatch.setattr(module, "start_fixture", lambda: (server, None))
    monkeypatch.setattr(module, "stop_fixture", lambda *args: None)
    monkeypatch.setattr(module, "make_arm", lambda *args, **kwargs: probe.arm)
    monkeypatch.setattr(module, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setitem(module.ORACLES, "wizard", lambda *args: SimpleNamespace(ok=True, detail="exact"))
    path = tmp_path / "warm.jsonl"
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
            "--output",
            str(path),
            "--workdir",
            str(tmp_path / "runs"),
        ],
    )
    assert module.main() == 0
    assert probe.events == ["warmup", "capture_warm", "verify_warm", "native_task", "verify_warm", "stamp", "cleanup"]
    assert json.loads(path.read_text())["engine"]["verified"] is True


def test_warm_resource_worker_preserves_the_same_engine_policy(monkeypatch, tmp_path, probe):
    module = warm_resource_compare
    monkeypatch.setattr(module, "make_arm", lambda *args, **kwargs: probe.arm)
    emitted = []
    records = []
    monkeypatch.setattr(module, "_emit", emitted.append)
    monkeypatch.setattr(module, "_write_record", lambda path, row: records.append(row))
    root = tmp_path.resolve()
    task = {
        "type": "task",
        "arm": "openbrowser-fast",
        "task": "wizard",
        "sample": 0,
        "nonce": "fake-nonce",
        "base_url": "http://fixture.invalid",
        "downloads": str(root / "downloads"),
        "record": str(tmp_path / "record.json"),
    }
    binding = {key: task[key] for key in ("arm", "task", "sample", "nonce")}
    messages = [task, {"type": "capture", **binding}, {"type": "verify", **binding}, {"type": "finish"}]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(message) + "\n" for message in messages)))
    assert module.run_worker("openbrowser-fast", root) == 0
    assert probe.events == ["warmup", "capture_warm", "verify_warm", "native_task", "verify_warm", "stamp", "cleanup"]
    assert records[0]["engine"]["verified"] is True
    assert emitted[0]["payload"]["engine"] == {"eligible": True}
    assert [message["type"] for message in emitted] == ["ready", "started", "native_done", "done", "finished"]
