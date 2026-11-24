"""A/B the headless shell against full Chrome on the CLI scrape browser path.

Both arms run the same `openbrowser-ai scrape --engine browser` command on the
local fixture, alternating order per sample. Peak RSS is the sum of the process
tree, sampled every 20 ms while the child runs.

Set OPENBROWSER_USE_HEADLESS_SHELL=1 or 0 per arm; the CLI reads it.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from benchmarks.scraping.scrape_compare import FixtureServer

TASKS = ("js-list", "assets")


def _run(binary: str, url: str, *, shell: bool) -> dict[str, float]:
    env = dict(os.environ)
    env["OPENBROWSER_USE_HEADLESS_SHELL"] = "1" if shell else "0"
    env["OPENBROWSER_DISABLE_TELEMETRY"] = "true"
    env["ANONYMIZED_TELEMETRY"] = "false"
    started = time.monotonic()
    process = psutil.Popen(
        [binary, "scrape", url, "--engine", "browser", "--format", "text"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    peak = 0
    while process.poll() is None:
        try:
            total = process.memory_info().rss + sum(
                child.memory_info().rss for child in process.children(recursive=True)
            )
            peak = max(peak, total)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.02)
    stdout, _stderr = process.communicate()
    return {
        "latency_ms": (time.monotonic() - started) * 1000.0,
        "peak_rss_mb": peak / (1024 * 1024),
        "output_bytes": len(stdout.encode()),
    }


def main() -> int:
    binary = os.environ.get("OPENBROWSER_BIN") or os.path.abspath(
        sys.executable.replace("python3", "openbrowser-ai")
    )
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    default_output = Path("benchmarks/scraping/results/results-headless-shell-ab.json")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output
    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records: list[dict] = []
    try:
        for task in TASKS:
            for sample in range(samples):
                order = (True, False) if sample % 2 == 0 else (False, True)
                for shell in order:
                    result = _run(binary, f"{server.base_url}/{task}", shell=shell)
                    records.append(
                        {
                            "task": task,
                            "shell": shell,
                            "sample": sample,
                            **{key: round(value, 3) for key, value in result.items()},
                        }
                    )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    summary = {}
    for task in TASKS:
        shell_rows = [row for row in records if row["task"] == task and row["shell"]]
        full_rows = [row for row in records if row["task"] == task and not row["shell"]]
        summary[task] = {
            "shell_p50_ms": round(statistics.median(row["latency_ms"] for row in shell_rows), 3),
            "shell_p50_rss_mb": round(statistics.median(row["peak_rss_mb"] for row in shell_rows), 3),
            "full_p50_ms": round(statistics.median(row["latency_ms"] for row in full_rows), 3),
            "full_p50_rss_mb": round(statistics.median(row["peak_rss_mb"] for row in full_rows), 3),
        }
        summary[task]["rss_saved_mb"] = round(
            summary[task]["full_p50_rss_mb"] - summary[task]["shell_p50_rss_mb"], 3
        )
    payload = {
        "environment": {
            "binary": binary,
            "platform": platform.platform(),
            "samples": samples,
            "load_average": [round(value, 3) for value in os.getloadavg()],
            "chrome_paths": {"shell": "Playwright headless shell", "full": "system Chrome"},
        },
        "summary": summary,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
