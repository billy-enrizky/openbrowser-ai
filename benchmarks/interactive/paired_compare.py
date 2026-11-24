"""Paired warm comparison for two interactive browser arms.

The full warm lane runs every sample for one arm before the next arm. Under a
shared host, load drift can dominate a small difference. This runner warms both
arms once, then alternates their order for every sample, so each pair sees the
same host conditions. A fresh fixture server and oracle check still guard every
run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters.openbrowser import _write_config
from .tasks import ORACLES
from .warm_benchmark import (
    DEFAULT_ARMS,
    DEFAULT_COLD_RESULTS,
    DEFAULT_WORKDIR,
    clean_downloads,
    make_arm,
    make_context,
    percentile,
    redact,
    start_fixture,
    stop_fixture,
)


def _loads() -> list[float]:
    try:
        return [round(value, 3) for value in os.getloadavg()]
    except OSError:
        return []


def _summarize(records: list[dict[str, Any]], arms: list[str], task: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = {arm: [] for arm in arms}
    for record in records:
        if record["ok"]:
            groups[record["arm"]].append(record["latency_ms"])
    rows = []
    for arm in arms:
        values = sorted(groups[arm])
        rows.append(
            {
                "arm": arm,
                "task": task,
                "ok": len(values),
                "runs": sum(1 for record in records if record["arm"] == arm),
                "p50_ms": round(statistics.median(values), 3) if values else None,
                "p95_ms": round(percentile(values, 0.95), 3) if values else None,
                "min_ms": round(values[0], 3) if values else None,
                "max_ms": round(values[-1], 3) if values else None,
            }
        )
    by_sample: dict[int, dict[str, float]] = {}
    for record in records:
        if record["ok"]:
            by_sample.setdefault(record["sample"], {})[record["arm"]] = record["latency_ms"]
    deltas = [sample[arms[1]] - sample[arms[0]] for sample in by_sample.values() if all(arm in sample for arm in arms)]
    paired = None
    if deltas:
        paired = {
            "faster_arm": arms[0] if statistics.median(deltas) > 0 else arms[1],
            "delta_ms_arm1_minus_arm0": {
                "p50": round(statistics.median(deltas), 3),
                "min": round(min(deltas), 3),
                "max": round(max(deltas), 3),
            },
            "arm0_faster_pairs": sum(1 for delta in deltas if delta > 0),
            "arm1_faster_pairs": sum(1 for delta in deltas if delta < 0),
            "pairs": len(deltas),
        }
    return {"rows": rows, "paired": paired}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs=2, required=True, choices=list(DEFAULT_ARMS))
    parser.add_argument("--task", required=True, choices=sorted(ORACLES))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--cold-results", type=Path, default=DEFAULT_COLD_RESULTS)
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()
    if args.task == "auth":
        parser.error("auth belongs to the cold lane because it restarts the browser")
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_before = _loads()
    arms: list[str] = args.arms
    roots: dict[str, Any] = {}
    instances: dict[str, Any] = {}
    environments: dict[str, Any] = {}
    try:
        for arm_name in arms:
            root_ctx = make_context(args.workdir, arm_name)
            if arm_name in {"openbrowser", "openbrowser-fast"}:
                _write_config(root_ctx, extensions=False)
            arm = make_arm(arm_name)
            payload, warmup_ms = arm.warmup(root_ctx)
            roots[arm_name] = root_ctx
            instances[arm_name] = arm
            environments[arm_name] = {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "binary": redact(getattr(arm, "binary", "")),
                "version": payload.get("browser_version") or (arm.binary if hasattr(arm, "binary") else ""),
                "warmup_ms": round(warmup_ms, 3),
                "warmup_payload": payload,
            }
            print(f"{arm_name:<20} warmup {warmup_ms:.0f} ms", flush=True)

        records: list[dict[str, Any]] = []
        for sample in range(args.samples):
            order = list(arms) if sample % 2 == 0 else list(reversed(arms))
            for arm_name in order:
                root_ctx = roots[arm_name]
                server, thread = start_fixture()
                ctx = replace(
                    root_ctx,
                    task=args.task,
                    sample=sample,
                    base_url=server.base_url,
                )
                if args.task == "download":
                    clean_downloads(ctx.downloads)
                try:
                    started = time.monotonic()
                    result = instances[arm_name].run(ctx, cleanup=False)
                    latency_ms = (time.monotonic() - started) * 1000.0
                    state = server.state.snapshot()
                    oracle = ORACLES[args.task](state, result.reported, ctx.downloads)
                finally:
                    stop_fixture(server, thread)
                record = {
                    "run_id": f"paired-{args.task}-s{sample}-{arm_name}",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "lane": "paired",
                    "task": args.task,
                    "arm": arm_name,
                    "sample": sample,
                    "order": order.index(arm_name),
                    "ok": bool(result.ok and oracle.ok),
                    "adapter_ok": result.ok,
                    "oracle_ok": oracle.ok,
                    "oracle_detail": oracle.detail,
                    "error": result.error,
                    "latency_ms": latency_ms,
                    "tool_calls": result.tool_calls,
                    "process_starts": result.process_starts,
                    "output_bytes": len(result.stdout.encode()),
                    "stderr_bytes": len(result.stderr.encode()),
                    "browser_version": result.browser_version,
                    "reported": result.reported,
                    "requests": state.get("request_count", 0),
                    "request_bytes": state.get("request_bytes", 0),
                    "keychain_verified": result.keychain_verified,
                }
                records.append(record)
                status = "ok" if record["ok"] else "fail"
                print(
                    f"{arm_name:<20} {args.task:<10} sample {sample} {status} "
                    f"{latency_ms:.0f} ms calls={result.tool_calls} procs={result.process_starts}",
                    flush=True,
                )
                if record["error"]:
                    print(f"    error: {record['error'][:200]}", flush=True)
    finally:
        for arm_name in arms:
            arm = instances.get(arm_name)
            root_ctx = roots.get(arm_name)
            if arm is not None and root_ctx is not None:
                arm.cleanup(root_ctx)
                if not args.keep_runs:
                    shutil.rmtree(root_ctx.root, ignore_errors=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "environment": {
            "cold_results": str(args.cold_results),
            "task": args.task,
            "samples": args.samples,
            "load_before": load_before,
            "load_after": _loads(),
            "arms": environments,
        },
        **_summarize(records, arms, args.task),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print()
    for row in summary["rows"]:
        print(
            f"{row['arm']:<20} {row['task']:<10} {row['ok']}/{row['runs']} "
            f"p50={row['p50_ms']} p95={row['p95_ms']} min={row['min_ms']} max={row['max_ms']}"
        )
    if summary["paired"] is not None:
        print("paired:", json.dumps(summary["paired"]))
    print(f"\njsonl: {args.output}\nsummary: {summary_path}")
    return 0 if all(row["ok"] == row["runs"] for row in summary["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
