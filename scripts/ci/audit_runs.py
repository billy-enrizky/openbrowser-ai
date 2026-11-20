#!/usr/bin/env python3
"""Audit recent GitHub Actions runs: waiting, setup, and running time.

Reads the GitHub REST API with the token from ``GITHUB_TOKEN`` or
``gh auth token``. Read-only. No retries and no timeouts are widened.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

API = "https://api.github.com"
MAIN_STEP_PREFIX = "run "


def token() -> str:
    value = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if value:
        return value
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("no GitHub token: set GITHUB_TOKEN or run gh auth login") from exc


def api_get(path: str, auth: str) -> Any:
    request = urllib.request.Request(
        f"{API}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {auth}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "openbrowser-ci-audit",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds(start: dt.datetime | None, end: dt.datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def normalize_job(name: str) -> str:
    name = re.sub(r" shard \d+/\d+", " shard", name)
    name = re.sub(r"Python [\d.]+ on ", "", name)
    return name


def main_step(job: dict[str, Any]) -> dict[str, Any] | None:
    for step in job.get("steps") or []:
        if (step.get("name") or "").lower().startswith(MAIN_STEP_PREFIX):
            return step
    return None


def fetch_runs(repo: str, limit: int, auth: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    page = 1
    while len(runs) < limit:
        payload = api_get(f"/repos/{repo}/actions/runs?per_page=100&page={page}", auth)
        batch = payload.get("workflow_runs", [])
        if not batch:
            break
        runs.extend(batch)
        page += 1
    return runs[:limit]


def fetch_jobs(repo: str, run_id: int, auth: str) -> list[dict[str, Any]]:
    payload = api_get(f"/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100", auth)
    return payload.get("jobs", [])


def build_records(repo: str, runs: list[dict[str, Any]], auth: str, workers: int) -> list[dict[str, Any]]:
    completed = [run for run in runs if run.get("status") == "completed"]
    jobs_by_run: dict[int, list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_jobs, repo, run["id"], auth): run["id"] for run in completed}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            run_id = futures[future]
            jobs_by_run[run_id] = future.result()
            if index % 25 == 0:
                print(f"  fetched jobs for {index}/{len(completed)} runs", file=sys.stderr)

    records: list[dict[str, Any]] = []
    for run in completed:
        created = parse_time(run.get("created_at"))
        finished = parse_time(run.get("updated_at"))
        for job in jobs_by_run.get(run["id"], []):
            if job.get("status") != "completed":
                continue
            started = parse_time(job.get("started_at"))
            completed_at = parse_time(job.get("completed_at"))
            step = main_step(job)
            step_started = parse_time(step.get("started_at")) if step else None
            step_finished = parse_time(step.get("completed_at")) if step else None
            records.append(
                {
                    "run_id": run["id"],
                    "event": run.get("event"),
                    "branch": run.get("head_branch"),
                    "job": job.get("name"),
                    "job_group": normalize_job(job.get("name") or ""),
                    "conclusion": job.get("conclusion"),
                    "waiting": seconds(created, started),
                    "setup": seconds(started, step_started),
                    "main": seconds(step_started, step_finished),
                    "job_total": seconds(started, completed_at),
                    "run_total": seconds(created, finished),
                }
            )
    return records


def summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["job_group"]].append(record)
    jobs: dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        jobs[name] = {"n": len(items)}
        for key in ("waiting", "setup", "main", "job_total"):
            values = [item[key] for item in items if item[key] is not None]
            jobs[name][f"{key}_p50"] = percentile(values, 0.5)
            jobs[name][f"{key}_p95"] = percentile(values, 0.95)
    events: dict[str, Any] = {}
    seen: set[tuple[int, str]] = set()
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        key = (record["run_id"], record["event"])
        if key in seen:
            continue
        seen.add(key)
        if record["run_total"] is not None:
            by_event[record["event"]].append(record["run_total"])
    for event, values in by_event.items():
        events[event] = {
            "n": len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    return {"jobs": jobs, "events": events}


def print_table(payload: dict[str, Any]) -> None:
    print("\nPer job, seconds (p50 / p95):")
    header = (
        f"{'job':52} {'n':>4} {'wait50':>7} {'wait95':>7} "
        f"{'setup50':>8} {'setup95':>8} {'main50':>7} {'main95':>7} "
        f"{'total50':>8} {'total95':>8}"
    )
    print(header)
    for name, item in payload["jobs"].items():

        def value(key: str, item: dict[str, Any] = item) -> str:
            number = item.get(key)
            return "n/a" if number is None else f"{number:7.1f}"

        print(
            f"{name[:52]:52} {item['n']:4d} {value('waiting_p50')} {value('waiting_p95')} "
            f"{value('setup_p50'):>8} {value('setup_p95'):>8} "
            f"{value('main_p50')} {value('main_p95')} "
            f"{value('job_total_p50'):>8} {value('job_total_p95'):>8}"
        )
    print("\nRun wall time by event, seconds (p50 / p95):")
    for event, item in sorted(payload["events"].items()):
        print(
            f"  {event}: n={item['n']} p50={item['p50']:.1f} "
            f"p95={item['p95']:.1f} min={item['min']:.1f} max={item['max']:.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="billy-enrizky/openbrowser-ai")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=None, help="write records and summary JSON")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args(argv)

    auth = token()
    runs = fetch_runs(args.repo, args.limit, auth)
    print(f"runs fetched: {len(runs)}", file=sys.stderr)
    records = build_records(args.repo, runs, auth, args.workers)
    payload = {"records": records, "summary": summary(records)}
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(payload["summary"], indent=2))
    else:
        print_table(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
