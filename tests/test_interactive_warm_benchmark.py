"""Tests for the warm interactive benchmark lane."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.interactive.adapters.openbrowser import OpenBrowserArm
from benchmarks.interactive.tasks import oracle_tabs
from benchmarks.interactive.warm_benchmark import WARM_TASKS, cold_p50, redact, summarize


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
        patch.object(arm, "_invoke", return_value=(0, "", "")),
        patch("benchmarks.interactive.adapters.openbrowser.verify_mock_keychain", return_value=True),
        patch.object(arm, "cleanup") as cleanup,
    ):
        result = arm.run(ctx, cleanup=False)

    cleanup.assert_not_called()
    assert result.ok is True
    assert result.error == ""


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
