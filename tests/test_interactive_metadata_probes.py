"""Metadata probes stay outside captured native resource observations."""

from __future__ import annotations

import json
from types import SimpleNamespace

from benchmarks.interactive import resource_compare, run_benchmark
from benchmarks.interactive.adapters.base import ArmResult


def test_resource_worker_does_not_query_a_tool_version(monkeypatch, tmp_path):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "binary_version": None}

    monkeypatch.setattr(run_benchmark, "run_one", run)
    path = tmp_path / "record.json"
    resource_compare.run_worker("wizard", "openbrowser-fast", 0, tmp_path, path)
    assert calls == [{"record_binary_version": False}]
    assert json.loads(path.read_text())["binary_version"] is None


def test_cold_runner_can_omit_post_task_metadata_without_native_work(monkeypatch, tmp_path):
    server = SimpleNamespace(base_url="http://127.0.0.1:1", state=SimpleNamespace(snapshot=lambda: {}))
    monkeypatch.setattr(run_benchmark, "start_fixture", lambda: (server, None))
    monkeypatch.setattr(run_benchmark, "stop_fixture", lambda *args: None)
    monkeypatch.setattr(run_benchmark, "terminate_matching", lambda *args: None)
    result = ArmResult(ok=True, reported={"quote": "Q-1"}, keychain_verified=True)
    arm = SimpleNamespace(binary="/fake/native-tool", run=lambda *a, **k: result, cleanup=lambda *a: None)
    monkeypatch.setattr(run_benchmark, "make_arm", lambda *a, **k: arm)
    monkeypatch.setitem(run_benchmark.ORACLES, "wizard", lambda *args: SimpleNamespace(ok=True, detail="exact"))

    def forbidden(*args):
        raise AssertionError("a metadata subprocess entered the captured worker")

    monkeypatch.setattr(run_benchmark, "_binary_version", forbidden)
    row = run_benchmark.run_one("wizard", "openbrowser-fast", 0, tmp_path, True, record_binary_version=False)
    assert row["ok"] is True
    assert row["binary_version"] is None


def test_parent_attaches_preflight_version_after_resource_capture(monkeypatch, tmp_path):
    monkeypatch.setattr(resource_compare, "source_manifest", lambda: {"source_digest": "same"})
    monkeypatch.setattr(resource_compare, "binding_snapshot", lambda arms: {"verified": True})
    monkeypatch.setattr(
        resource_compare, "arm_versions", lambda arms: {"openbrowser-fast": {"version": "preflight-version"}}
    )
    monkeypatch.setattr(resource_compare, "summarize_resources", lambda rows: [])
    captured = []

    def sample(task, arm, number, **kwargs):
        row = {
            "task": task,
            "arm": arm,
            "sample": number,
            "ok": True,
            "resource_ok": True,
            "binary_version": None,
            "worker_timed_out": False,
            "cleanup": {"verified": True, "remaining": [], "unverified_processes": 0, "errors": []},
        }
        captured.append(row.copy())
        return row

    monkeypatch.setattr(resource_compare, "run_sample", sample)
    output = tmp_path / "resource.jsonl"
    rows = resource_compare.run_lane(
        ["openbrowser-fast"],
        ["wizard"],
        1,
        output,
        workdir=tmp_path,
        interval=0.02,
        worker_timeout=180,
    )
    assert captured[0]["binary_version"] is None
    assert rows[0]["binary_version"] == "preflight-version"
    assert rows[0]["binary_version_source"] == "preflight"
    raw = [json.loads(line) for line in output.read_text().splitlines()]
    assert raw[0]["arm_versions"]["openbrowser-fast"]["version"] == rows[0]["binary_version"]
