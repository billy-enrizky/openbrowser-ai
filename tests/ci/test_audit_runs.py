"""Tests for the CI audit math, using a committed fixture."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ci.audit_runs import fetch_jobs, normalize_job, percentile, summary


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
            {
                "name": "Run uv sync --locked --all-extras",
                "conclusion": "success",
                "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:00:01Z",
            },
            {
                "name": "Run selected tests",
                "conclusion": "success",
                "started_at": "2026-01-01T00:00:01Z",
                "completed_at": "2026-01-01T00:00:06Z",
            },
            {
                "name": "Post Run astral-sh/setup-uv@v5",
                "conclusion": "success",
                "started_at": "2026-01-01T00:00:06Z",
                "completed_at": "2026-01-01T00:00:07Z",
            },
        ]
    }
    assert main_step(job)["name"] == "Run selected tests"


def test_main_step_skips_a_skipped_candidate() -> None:
    from scripts.ci.audit_runs import main_step

    job = {
        "steps": [
            {
                "name": "Run selector and sharder tests",
                "conclusion": "success",
                "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:00:05Z",
            },
            {
                "name": "Prove selection catches a planted break",
                "conclusion": "skipped",
                "started_at": "2026-01-01T00:00:05Z",
                "completed_at": "2026-01-01T00:00:05Z",
            },
        ]
    }
    assert main_step(job)["name"] == "Run selector and sharder tests"


def test_fetch_jobs_uses_the_run_attempt_and_paginates(monkeypatch) -> None:
    paths: list[str] = []

    def fake_api_get(path: str, auth: str):
        paths.append(path)
        page = int(path.rsplit("page=", 1)[1])
        count = 100 if page == 1 else (1 if page == 2 else 0)
        return {"jobs": [{"id": page * 1000 + index} for index in range(count)]}

    monkeypatch.setattr("scripts.ci.audit_runs.api_get", fake_api_get)
    jobs = fetch_jobs("owner/repo", 42, 3, "token")

    assert len(jobs) == 101
    assert all("/runs/42/attempts/3/jobs" in path for path in paths)
    assert len(paths) == 2
