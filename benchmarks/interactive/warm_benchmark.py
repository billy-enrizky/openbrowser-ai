"""Run the warm-session interactive benchmark for OpenBrowser.

The cold lane starts a new daemon, browser, and fixture for every task. This
lane starts the daemon and browser once with `openbrowser-ai warm`, then runs
each task against a fresh fixture server. It measures steady-state task time
and keeps the cold-start cost separate.

The auth task remains in the cold lane because it deliberately restarts the
daemon to prove profile reuse across processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .adapters.base import RunContext
from .adapters.openbrowser import OpenBrowserArm, _write_config
from .fixtures.server import FixtureServer
from .tasks import ORACLES, TASKS

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "results-warm.jsonl"
DEFAULT_WORKDIR = Path("/tmp/obib-warm-runs")
DEFAULT_COLD_RESULTS = HERE / "results" / "results-2026-11-23-lean.jsonl"
WARM_TASKS = ("wizard", "modal", "download", "tabs")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def make_context(workdir: Path) -> RunContext:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = workdir / stamp / "warm-openbrowser"
    for name in ("downloads", "profile", "home", "tmp", "config"):
        (root / name).mkdir(parents=True, exist_ok=True)
    token = hashlib.sha1(f"warm-{time.time_ns()}".encode()).hexdigest()[:10]
    return RunContext(
        task="wizard",
        arm="openbrowser",
        sample=0,
        base_url="http://127.0.0.1:1",
        root=root,
        downloads=root / "downloads",
        profile=root / "profile",
        home=root / "home",
        tmp=root / "tmp",
        config=root / "config",
        session=f"w{token}",
    )


def start_fixture() -> tuple[FixtureServer, threading.Thread]:
    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_fixture(server: FixtureServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def clean_downloads(path: Path) -> None:
    for child in path.rglob("*"):
        if child.is_file():
            child.unlink()


def summarize(records: list[dict], cold: dict[str, float | None]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["task"], []).append(record)
    rows: list[dict] = []
    for task in WARM_TASKS:
        items = groups.get(task, [])
        successes = [item for item in items if item["ok"]]
        latencies = [item["latency_ms"] for item in successes]
        warm_p50 = percentile(latencies, 0.50)
        cold_p50 = cold.get(task)
        rows.append(
            {
                "task": task,
                "runs": len(items),
                "ok": len(successes),
                "warm_p50_ms": warm_p50,
                "warm_p95_ms": percentile(latencies, 0.95),
                "cold_p50_ms": cold_p50,
                "speedup": (cold_p50 / warm_p50) if cold_p50 and warm_p50 else None,
                "errors": sorted({item["error"] for item in items if item["error"]})[:3],
            }
        )
    return rows


def cold_p50(path: Path) -> dict[str, float | None]:
    result: dict[str, list[float]] = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("arm") != "openbrowser" or not record.get("ok"):
            continue
        result.setdefault(str(record.get("task")), []).append(float(record["latency_ms"]))
    return {task: percentile(values, 0.50) for task, values in result.items()}


def print_table(rows: list[dict]) -> None:
    header = f"{'task':<10} {'ok':>5} {'cold p50':>10} {'warm p50':>10} {'warm p95':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        cold = f"{row['cold_p50_ms']:.0f}" if row["cold_p50_ms"] else "n/a"
        warm = f"{row['warm_p50_ms']:.0f}" if row["warm_p50_ms"] else "n/a"
        p95 = f"{row['warm_p95_ms']:.0f}" if row["warm_p95_ms"] else "n/a"
        speedup = f"{row['speedup']:.2f}x" if row["speedup"] else "n/a"
        print(f"{row['task']:<10} {row['ok']:>2}/{row['runs']:<2} {cold:>10} {warm:>10} {p95:>10} {speedup:>8}")
        for error in row["errors"]:
            print(f"    error: {error[:160]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(WARM_TASKS))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--cold-results", type=Path, default=DEFAULT_COLD_RESULTS)
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()

    unknown = [task for task in args.tasks if task not in ORACLES]
    if unknown:
        parser.error(f"unknown tasks: {unknown}")
    if "auth" in args.tasks:
        parser.error("auth belongs to the cold lane because it restarts the daemon")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root_ctx = make_context(args.workdir)
    _write_config(root_ctx, extensions=False)
    arm = OpenBrowserArm(extensions=False)
    version_result = subprocess.run(
        [arm.binary, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "binary": str(arm.binary),
        "version": version_result.stdout.strip() if version_result.returncode == 0 else "",
        "cold_results": str(args.cold_results),
        "tasks": args.tasks,
        "samples": args.samples,
    }
    records: list[dict] = []
    warmup_payload: dict = {}
    warmup_ms = 0.0
    try:
        warmup_payload, warmup_ms = arm.warmup(root_ctx)
        environment["warmup_ms"] = round(warmup_ms, 3)
        environment["warmup_payload"] = warmup_payload
        print(
            f"warmup {warmup_ms:.0f} ms url={warmup_payload.get('url', '')} "
            f"reported={warmup_payload.get('elapsed_ms', '')} ms",
            flush=True,
        )
        for sample in range(args.samples):
            for task in args.tasks:
                server, thread = start_fixture()
                ctx = replace(
                    root_ctx,
                    task=task,
                    sample=sample,
                    base_url=server.base_url,
                    session=f"w{sample}-{task}",
                )
                if task == "download":
                    clean_downloads(ctx.downloads)
                try:
                    started = time.monotonic()
                    result = arm.run(ctx, cleanup=False, restart_for_auth=False)
                    latency_ms = (time.monotonic() - started) * 1000.0
                    state = server.state.snapshot()
                    oracle = ORACLES[task](state, result.reported, ctx.downloads)
                finally:
                    stop_fixture(server, thread)
                record = {
                    "run_id": f"warm-{task}-s{sample}",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "lane": "warm",
                    "task": task,
                    "arm": "openbrowser",
                    "sample": sample,
                    "ok": bool(result.ok and oracle.ok),
                    "adapter_ok": result.ok,
                    "oracle_ok": oracle.ok,
                    "oracle_detail": oracle.detail,
                    "error": result.error,
                    "latency_ms": latency_ms,
                    "warmup_ms": warmup_ms,
                    "warmup_elapsed_ms": warmup_payload.get("elapsed_ms"),
                    "warmup_url": warmup_payload.get("url", ""),
                    "tool_calls": result.tool_calls,
                    "process_starts": result.process_starts,
                    "output_bytes": len(result.stdout.encode()),
                    "stderr_bytes": len(result.stderr.encode()),
                    "reported": result.reported,
                    "requests": state.get("request_count", 0),
                    "request_bytes": state.get("request_bytes", 0),
                    "keychain_verified": result.keychain_verified,
                }
                records.append(record)
                status = "ok" if record["ok"] else "fail"
                print(
                    f"{task:<10} sample {sample} {status} {latency_ms:.0f} ms "
                    f"calls={result.tool_calls} procs={result.process_starts}",
                    flush=True,
                )
                if record["error"]:
                    print(f"    error: {record['error'][:200]}", flush=True)
    finally:
        arm.cleanup(root_ctx)
        if not args.keep_runs:
            shutil.rmtree(root_ctx.root, ignore_errors=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    cold = cold_p50(args.cold_results)
    rows = summarize(records, cold)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"environment": environment, "rows": rows}, indent=2) + "\n")
    print()
    print_table(rows)
    print()
    print(f"jsonl: {args.output}")
    print(f"summary: {summary_path}")
    return 0 if all(row["ok"] == row["runs"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
