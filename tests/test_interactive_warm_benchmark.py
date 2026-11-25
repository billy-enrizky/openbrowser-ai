"""Tests for the warm interactive benchmark lane."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.interactive import warm_benchmark
from benchmarks.interactive.adapters.base import ArmResult
from benchmarks.interactive.adapters.openbrowser import OpenBrowserArm
from benchmarks.interactive.run_benchmark import make_arm
from benchmarks.interactive.tasks import oracle_tabs
from benchmarks.interactive.warm_benchmark import DEFAULT_ARMS, WARM_TASKS, cold_p50, redact, summarize


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        task="wizard",
        base_url="http://127.0.0.1:1",
        root=Path("/tmp/warm-test"),
        config=Path("/tmp/warm-test/config"),
    )


def test_warm_tasks_exclude_auth():
    assert "auth" not in WARM_TASKS
    assert set(WARM_TASKS) == {"wizard", "modal", "download", "tabs"}


def test_openbrowser_warm_run_skips_cleanup_and_resets_result():
    arm = OpenBrowserArm(binary="/bin/true")
    ctx = _ctx()
    arm._result.ok = True
    arm._result.error = "stale"

    with (
        patch("benchmarks.interactive.adapters.openbrowser._write_config"),
        patch.object(arm, "_invoke", return_value=(0, "", "")) as invoke,
        patch("benchmarks.interactive.adapters.openbrowser.verify_mock_keychain", return_value=True),
        patch.object(arm, "cleanup") as cleanup,
    ):
        result = arm.run(ctx, cleanup=False)

    cleanup.assert_not_called()
    assert result.ok is True
    assert result.error == ""
    assert "await input_text(index=await find" in invoke.call_args.args[1]
    assert "await fill_selector" not in invoke.call_args.args[1]


def test_openbrowser_fast_is_a_distinct_warm_arm():
    semantic_arm = make_arm("openbrowser")
    fast_arm = make_arm("openbrowser-fast")

    assert "openbrowser" in DEFAULT_ARMS
    assert "openbrowser-fast" in DEFAULT_ARMS
    assert isinstance(semantic_arm, OpenBrowserArm)
    assert isinstance(fast_arm, OpenBrowserArm)
    assert semantic_arm.fast_selectors is False
    assert fast_arm.fast_selectors is True
    assert semantic_arm.name == "openbrowser"
    assert fast_arm.name == "openbrowser-fast"


def test_fast_selector_arm_uses_only_stable_fixture_ids():
    arm = make_arm("openbrowser-fast")
    cases = (
        ("wizard", "fill_selector('#name'"),
        ("modal", "click_selector('#open-modal'"),
        ("download", "click_selector('#download-report'"),
        ("tabs", "click_selector_and_wait_for_new_tab('#compare'"),
    )

    for task, expected_call in cases:
        ctx = _ctx()
        ctx.task = task
        with (
            patch("benchmarks.interactive.adapters.openbrowser._write_config"),
            patch.object(arm, "_invoke", return_value=(0, "", "")) as invoke,
            patch("benchmarks.interactive.adapters.openbrowser.verify_mock_keychain", return_value=True),
        ):
            result = arm.run(ctx, cleanup=False)

        assert result.ok is True
        program = invoke.call_args.args[1]
        assert expected_call in program
        assert "await click(index=await find(" not in program
        assert "await input_text(index=await find(" not in program


def test_openbrowser_warmup_parses_json():
    arm = OpenBrowserArm(binary="/bin/true")
    payload = {"ok": True, "elapsed_ms": 12.5, "url": "about:blank"}
    with (
        patch("benchmarks.interactive.adapters.openbrowser._env", return_value={}),
        patch(
            "benchmarks.interactive.adapters.openbrowser.run_process",
            return_value=(0, json.dumps(payload), "", 1.0),
        ),
    ):
        result, elapsed_ms = arm.warmup(_ctx())

    assert result == payload
    assert elapsed_ms >= 0


def test_summarize_uses_paired_cold_p50():
    records = [
        {"task": "wizard", "arm": "openbrowser", "ok": True, "latency_ms": 100.0, "error": ""},
        {"task": "wizard", "arm": "openbrowser", "ok": True, "latency_ms": 200.0, "error": ""},
        {"task": "wizard", "arm": "openbrowser", "ok": True, "latency_ms": 300.0, "error": ""},
    ]

    rows = summarize(records, {("wizard", "openbrowser"): 400.0})

    assert rows[0]["warm_p50_ms"] == 200.0
    assert rows[0]["speedup"] == 2.0


def test_tabs_oracle_rejects_a_flow_without_a_new_popup():
    events = [
        {"event": "detail_loaded", "opener": True},
        {"event": "approve"},
        {"event": "status", "status": "approved:42"},
        {"event": "detail_closed"},
    ]
    reported = {"original_title": "Tabs Bench"}

    assert oracle_tabs({"tab_events": events}, reported).ok
    result = oracle_tabs({"tab_events": events[1:]}, reported)

    assert result.ok is False
    assert result.detail == "detail tab never loaded"

    events[0]["opener"] = False
    result = oracle_tabs({"tab_events": events}, reported)

    assert result.ok is False
    assert result.detail == "detail tab did not see window.opener"


def test_tabs_summary_excludes_samples_that_fail_the_oracle():
    records = [
        {"task": "tabs", "arm": "openbrowser", "ok": False, "latency_ms": 1.0, "error": "popup missing"},
        {"task": "tabs", "arm": "openbrowser", "ok": True, "latency_ms": 100.0, "error": ""},
    ]

    row = next(row for row in summarize(records, {}) if row["task"] == "tabs" and row["arm"] == "openbrowser")

    assert row["runs"] == 2
    assert row["ok"] == 1
    assert row["warm_p50_ms"] == 100.0


def test_cold_p50_reads_openbrowser_records(tmp_path: Path):
    path = tmp_path / "cold.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"task": "wizard", "arm": "openbrowser", "ok": True, "latency_ms": 100.0}),
                json.dumps({"task": "wizard", "arm": "openbrowser", "ok": True, "latency_ms": 200.0}),
                json.dumps({"task": "wizard", "arm": "openbrowser", "ok": True, "latency_ms": 300.0}),
                json.dumps({"task": "wizard", "arm": "other", "ok": True, "latency_ms": 1.0}),
            ]
        )
        + "\n"
    )

    assert cold_p50(path)[("wizard", "openbrowser")] == 200.0


def test_redact_hides_home_and_worktrees():
    assert redact(str(Path.home() / "bin" / "openbrowser-ai")) == "$HOME/bin/openbrowser-ai"
    assert redact("/repo/.worktrees/example/.venv/bin/openbrowser-ai") == "working copy"


def test_no_cold_reference_loads_no_samples():
    assert cold_p50(None) == {}


def test_summary_without_reference_has_no_cold_ratio_fields():
    record = {"task": "wizard", "arm": "openbrowser-fast", "ok": True, "latency_ms": 250.0, "error": ""}
    row = summarize([record], {})[0]
    assert "cold_p50_ms" not in row
    assert "speedup" not in row


def _fake_warm_cli(tmp_path, monkeypatch, *, cold_file=None):
    def forbidden(*args, **kwargs):
        raise AssertionError("fake warm CLI attempted a real process or socket")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    submission = {
        "name": "Alice Bench",
        "email": "alice@example.com",
        "country": "CA",
        "province": "ON",
        "plan": "pro",
        "seats": 5,
        "terms": True,
        "quote": "Q-1",
    }
    state = {"wizard_submissions": [submission], "wizard_step1": 1, "wizard_step2": 1, "wizard_submits": 1}
    server = SimpleNamespace(base_url="http://fixture.invalid", state=SimpleNamespace(snapshot=lambda: state))

    class FakeArm:
        binary = ""

        def warmup(self, ctx):
            return {"ok": True, "browser_version": "fake"}, 0.0

        def run(self, ctx, *, cleanup=True):
            assert cleanup is False
            return ArmResult(ok=True, reported={"quote": "Q-1"}, keychain_verified=True)

        def cleanup(self, ctx):
            assert ctx.root.is_relative_to(tmp_path)

    clock = iter([1000.0, 1000.25])
    monkeypatch.setattr(warm_benchmark, "time", SimpleNamespace(monotonic=lambda: next(clock), time_ns=lambda: 1))
    monkeypatch.setattr(warm_benchmark, "platform", SimpleNamespace(platform=lambda: "fake", machine=lambda: "fake"))
    monkeypatch.setattr(warm_benchmark, "_loads", lambda: (0, 0, 0))
    monkeypatch.setattr(warm_benchmark, "make_arm", lambda name: FakeArm())
    monkeypatch.setattr(warm_benchmark, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(warm_benchmark, "start_fixture", lambda: (server, None))
    monkeypatch.setattr(warm_benchmark, "stop_fixture", lambda *args: None)
    output = tmp_path / "warm.jsonl"
    argv = [
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
    ]
    if cold_file is not None:
        argv.extend(["--cold-results", str(cold_file)])
    monkeypatch.setattr(sys, "argv", argv)
    assert warm_benchmark.main() == 0
    return json.loads(output.with_suffix(".summary.json").read_text())


def test_default_warm_cli_has_no_cold_reference_or_ratio(tmp_path, monkeypatch):
    report = _fake_warm_cli(tmp_path, monkeypatch)
    assert warm_benchmark.DEFAULT_COLD_RESULTS is None
    assert report["environment"]["cold_results"] is None
    assert "cold_p50_ms" not in report["rows"][0]
    assert "speedup" not in report["rows"][0]


def test_explicit_cold_reference_remains_opt_in(tmp_path, monkeypatch):
    cold = tmp_path / "cold.jsonl"
    cold.write_text(json.dumps({"task": "wizard", "arm": "openbrowser-fast", "ok": True, "latency_ms": 500.0}) + "\n")
    report = _fake_warm_cli(tmp_path, monkeypatch, cold_file=cold)
    assert report["environment"]["cold_results"] == str(cold)
    assert report["rows"][0]["cold_p50_ms"] == 500.0
    assert report["rows"][0]["warm_p50_ms"] == 250.0
    assert report["rows"][0]["speedup"] == 2.0
