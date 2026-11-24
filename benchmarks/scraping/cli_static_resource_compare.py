"""Measure the real one-shot CLI static path, including peak RSS.

The resource lane starts a browser eagerly for every arm, so its static rows
include a running Chromium that the CLI never starts. This script measures the
surface a user actually runs: one `openbrowser-ai scrape` process per sample,
auto engine against the HTTP-only engine, with peak child RSS sampled every
5 ms.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from benchmarks.interactive.adapters.openbrowser import _env
from benchmarks.interactive.warm_benchmark import make_context, percentile
from benchmarks.scraping.scrape_compare import FixtureServer


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
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/openbrowser-cli-static"))
    args = parser.parse_args()

    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = server.base_url
    records: list[dict[str, Any]] = []
    arms = {
        "auto": [f"{base}/static", "--format", "markdown"],
        "http": [f"{base}/static", "--format", "markdown", "--engine", "http"],
    }
    try:
        for sample in range(args.samples):
            for name, arm_args in arms.items():
                ctx = make_context(args.workdir, f"cli-static-{name}")
                env = _env(ctx)
                record = _run(args.binary, ["scrape", *arm_args], env)
                record.update(
                    {
                        "arm": name,
                        "sample": sample,
                        "ok": record["code"] == 0 and "Static Bench" in record["stdout"],
                        "output_bytes": len(record["stdout"].encode()),
                    }
                )
                record.pop("stdout", None)
                record.pop("stderr", None)
                records.append(record)
                print(
                    f"{name:<5} sample {sample} ok={record['ok']} "
                    f"{record['latency_ms']:.1f} ms {record['peak_rss_mb']:.1f} MB",
                    flush=True,
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    summary = {}
    for name in arms:
        items = [r for r in records if r["arm"] == name and r["ok"]]
        summary[name] = {
            "ok": len(items),
            "runs": sum(1 for r in records if r["arm"] == name),
            "latency_p50_ms": round(percentile([r["latency_ms"] for r in items], 0.5) or 0.0, 3),
            "rss_p50_mb": round(percentile([r["peak_rss_mb"] for r in items], 0.5) or 0.0, 3),
            "rss_max_mb": round(max((r["peak_rss_mb"] for r in items), default=0.0), 3),
            "output_bytes": sorted({r["output_bytes"] for r in items}),
        }
    payload = {
        "environment": {
            "binary": args.binary,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "load_average": list(os.getloadavg()),
            "samples": args.samples,
        },
        "arms": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
