"""Alternating A/B for the warm CLI ``-c`` startup path.

Both binaries talk to the same warm daemon. Each sample runs one command from
each binary in alternating order, so host drift hits both arms. The payload is
a no-op cell, which isolates client startup and daemon handshake cost from page
work.
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

from .adapters.base import run_process
from .adapters.openbrowser import _env, _write_config
from .warm_benchmark import make_context, percentile, redact


def _loads() -> list[float]:
    try:
        return [round(value, 3) for value in os.getloadavg()]
    except OSError:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-binary", required=True)
    parser.add_argument("--new-binary", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/openbrowser-cli-startup-ab"))
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ctx = replace(make_context(args.workdir, "openbrowser-fast"), task="wizard")
    _write_config(ctx, extensions=False)
    env = _env(ctx)
    records: list[dict[str, Any]] = []
    load_before = _loads()
    try:
        returncode, _stdout, stderr, warm_ms = run_process(
            [args.new_binary, "warm"],
            env=env,
            timeout=180,
        )
        if returncode != 0:
            raise RuntimeError(f"warm failed: {stderr[:600]}")
        for sample in range(args.samples):
            order = [("old", args.old_binary), ("new", args.new_binary)]
            if sample % 2:
                order.reverse()
            for position, (label, binary) in enumerate(order):
                started = time.monotonic()
                code, out, err, _elapsed = run_process(
                    [binary, "-c", "print(1)"],
                    env=env,
                    timeout=60,
                )
                latency_ms = (time.monotonic() - started) * 1000.0
                ok = code == 0 and "1" in out
                records.append(
                    {
                        "run_id": f"cli-startup-{sample}-{label}",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "lane": "cli-startup",
                        "arm": label,
                        "sample": sample,
                        "order": position,
                        "latency_ms": latency_ms,
                        "ok": ok,
                        "error": "" if ok else (err or out)[:300],
                        "stdout_bytes": len(out.encode()),
                    }
                )
                print(f"{label:<4} sample {sample} {'ok' if ok else 'fail'} {latency_ms:.1f} ms", flush=True)
    finally:
        run_process([args.new_binary, "daemon", "force-stop"], env=env, timeout=60)
        if not args.keep_runs:
            shutil.rmtree(ctx.root, ignore_errors=True)

    summary: dict[str, Any] = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "old_binary": redact(args.old_binary),
            "new_binary": redact(args.new_binary),
            "warmup_ms": round(warm_ms, 3),
            "load_before": load_before,
            "load_after": _loads(),
        },
        "rows": [],
    }
    by_arm: dict[str, list[float]] = {"old": [], "new": []}
    by_sample: dict[int, dict[str, float]] = {}
    for record in records:
        if record["ok"]:
            by_arm[record["arm"]].append(record["latency_ms"])
            by_sample.setdefault(record["sample"], {})[record["arm"]] = record["latency_ms"]
    for label in ("old", "new"):
        values = sorted(by_arm[label])
        summary["rows"].append(
            {
                "arm": label,
                "ok": len(values),
                "runs": sum(1 for record in records if record["arm"] == label),
                "p50_ms": round(statistics.median(values), 3) if values else None,
                "p95_ms": round(percentile(values, 0.95), 3) if values else None,
                "min_ms": round(values[0], 3) if values else None,
                "max_ms": round(values[-1], 3) if values else None,
            }
        )
    deltas = [sample["old"] - sample["new"] for sample in by_sample.values() if len(sample) == 2]
    if deltas:
        summary["paired"] = {
            "delta_ms_old_minus_new": {
                "p50": round(statistics.median(deltas), 3),
                "min": round(min(deltas), 3),
                "max": round(max(deltas), 3),
            },
            "new_faster_pairs": sum(1 for delta in deltas if delta > 0),
            "old_faster_pairs": sum(1 for delta in deltas if delta < 0),
            "pairs": len(deltas),
        }
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print()
    for row in summary["rows"]:
        print(f"{row['arm']:<4} {row['ok']}/{row['runs']} p50={row['p50_ms']} p95={row['p95_ms']}")
    if "paired" in summary:
        print("paired:", json.dumps(summary["paired"]))
    print(f"\njsonl: {args.output}\nsummary: {summary_path}")
    return 0 if all(row["ok"] == row["runs"] for row in summary["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
