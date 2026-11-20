"""Tests for the CI audit math, using a committed fixture."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ci.audit_runs import normalize_job, percentile, summary


def records() -> list[dict[str, object]]:
    path = Path(__file__).parent / "fixtures" / "ci_audit_sample.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_normalize_job_removes_shard_and_python_noise() -> None:
    assert normalize_job("Test Python 3.12 on ubuntu-latest shard 1/4") == "Test ubuntu-latest shard"
    assert normalize_job("Lint") == "Lint"


def test_percentile_is_linear() -> None:
    assert percentile([1.0], 0.5) == 1.0
    assert percentile([1.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == 2.9


def test_summary_reports_test_job_and_run_wall() -> None:
    payload = summary(records())
    test_job = payload["jobs"]["Test ubuntu-latest"]
    assert test_job["n"] == 2
    assert test_job["main_p50"] == 600.0
    assert test_job["main_p95"] == 690.0
    event = payload["events"]["pull_request"]
    assert event["n"] == 2
    assert event["p50"] == 645.0
    assert event["p95"] == 739.5


def test_main_step_ignores_dependency_sync() -> None:
    from scripts.ci.audit_runs import main_step

    job = {
        "steps": [
            {"name": "Run uv sync --locked --all-extras"},
            {"name": "Run selected tests"},
            {"name": "Post Run astral-sh/setup-uv@v5"},
        ]
    }
    assert main_step(job)["name"] == "Run selected tests"
