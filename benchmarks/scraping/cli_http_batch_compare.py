"""A/B the HTTP-only CLI batch path, including wall time and peak RSS.

Both binaries scrape the same five local pagination pages with
``scrape --engine http --format json --selector 'tbody tr' --jsonl``. The
pre-change binary loads the full Click command tree; the new binary takes the
light console entry path. The script alternates order per sample and polls the
child RSS while it runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from benchmarks.interactive.adapters.openbrowser import _env, _write_config
from benchmarks.interactive.warm_benchmark import make_context, percentile
from benchmarks.scraping.batch_compare import PAGE_COUNT, PAGE_SIZE, FixtureServer


def _run(binary: str, args: list[str], env: dict[str, str]) -> dict[str, Any]:
    started = time.monotonic()
    process = psutil.Popen(
        [binary, *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak = 0
    while process.poll() is None:
        try:
            peak = max(peak, process.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.005)
    stdout, stderr = process.communicate()
    return {
        "code": process.returncode,
        "latency_ms": (time.monotonic() - started) * 1000.0,
        "peak_rss_mb": peak / (1024 * 1024),
        "stdout": stdout or "",
        "stderr": (stderr or "")[:500],
    }


def _rows_ok(stdout: str) -> bool:
    try:
        rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except json.JSONDecodeError:
        return False
    ids = []
    for row in rows:
        if not row.get("ok"):
            return False
        for item in row.get("content") or []:
            row_id = (item.get("attributes") or {}).get("data-id")
            if row_id:
                ids.append(row_id)
    return len(ids) == PAGE_COUNT * PAGE_SIZE and len(set(ids)) == len(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-binary", required=True)
    parser.add_argument("--new-binary", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/openbrowser-cli-http-ab"))
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ctx = make_context(args.workdir, "openbrowser-fast")
    _write_config(ctx, extensions=False)
    env = _env(ctx)
    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    urls = [f"{server.base_url}/pagination?page={page}" for page in range(1, PAGE_COUNT + 1)]
    cli_args = ["scrape", *urls, "--engine", "http", "--format", "json", "--selector", "tbody tr", "--jsonl"]
    arms = [("old", args.old_binary), ("new", args.new_binary)]
    load_before = [round(value, 3) for value in os.getloadavg()] if hasattr(os, "getloadavg") else []
    records: list[dict[str, Any]] = []
    try:
        for _label, binary in arms:
            _run(binary, cli_args, env)
        for sample in range(args.samples):
            order = list(arms)
            if sample % 2:
                order.reverse()
            for position, (label, binary) in enumerate(order):
                result = _run(binary, cli_args, env)
                ok = result["code"] == 0 and _rows_ok(result["stdout"])
                records.append(
                    {
                        "run_id": f"cli-http-{sample}-{label}",
                        "arm": label,
                        "sample": sample,
                        "order": position,
                        "ok": ok,
                        "latency_ms": result["latency_ms"],
                        "peak_rss_mb": result["peak_rss_mb"],
                        "error": result["stderr"],
                    }
                )
                print(
                    f"{label} sample {sample} {'ok' if ok else 'fail'} "
                    f"{result['latency_ms']:.1f} ms {result['peak_rss_mb']:.1f} MB",
                    flush=True,
                )
    finally:
        server.shutdown()
        server.server_close()
        if not args.keep_runs:
            shutil.rmtree(ctx.root, ignore_errors=True)

    summary: dict[str, Any] = {
        "samples": args.samples,
        "tasks": {"pages": PAGE_COUNT},
        "load_before": load_before,
        "load_after": [round(value, 3) for value in os.getloadavg()] if hasattr(os, "getloadavg") else [],
        "rows": [],
    }
    by_arm: dict[str, list[dict[str, Any]]] = {"old": [], "new": []}
    for record in records:
        if record["ok"]:
            by_arm[record["arm"]].append(record)
    for label in ("old", "new"):
        group = by_arm[label]
        latencies = [record["latency_ms"] for record in group]
        rss = [record["peak_rss_mb"] for record in group]
        summary["rows"].append(
            {
                "arm": label,
                "ok": len(group),
                "runs": sum(1 for record in records if record["arm"] == label),
                "latency_p50_ms": round(statistics.median(latencies), 3) if latencies else None,
                "latency_p95_ms": round(percentile(latencies, 0.95), 3) if latencies else None,
                "peak_rss_p50_mb": round(statistics.median(rss), 3) if rss else None,
                "peak_rss_max_mb": round(max(rss), 3) if rss else None,
            }
        )
    by_sample: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["ok"]:
            by_sample.setdefault(record["sample"], {})[record["arm"]] = record
    pairs = [pair for pair in by_sample.values() if len(pair) == 2]
    if pairs:
        latencies = [pair["old"]["latency_ms"] - pair["new"]["latency_ms"] for pair in pairs]
        rss = [pair["old"]["peak_rss_mb"] - pair["new"]["peak_rss_mb"] for pair in pairs]
        summary["paired"] = {
            "old_minus_new_latency_p50_ms": round(statistics.median(latencies), 3),
            "old_minus_new_rss_p50_mb": round(statistics.median(rss), 3),
            "new_faster_pairs": sum(1 for value in latencies if value > 0),
            "new_lower_rss_pairs": sum(1 for value in rss if value > 0),
            "pairs": len(pairs),
        }
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(row["ok"] == row["runs"] for row in summary["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
