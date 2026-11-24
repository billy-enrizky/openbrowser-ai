"""Run warm-session interactive benchmarks for OpenBrowser and competitors.

The cold lane starts a new process tree, daemon, and browser for every task.
This lane starts each arm once, then runs every task against a fresh fixture
server. It measures steady-state task latency and keeps browser startup
separate. The auth task remains in the cold lane because it deliberately
restarts the browser to prove profile reuse across processes.
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
from .adapters.openbrowser import _write_config
from .fixtures.server import FixtureServer
from .run_benchmark import make_arm
from .tasks import ORACLES, TASKS

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "results-warm.jsonl"
DEFAULT_WORKDIR = Path("/tmp/obib-warm-runs")
DEFAULT_COLD_RESULTS = HERE / "results" / "results-2026-11-23-lean.jsonl"
WARM_TASKS = ("wizard", "modal", "download", "tabs")
DEFAULT_ARMS = ("openbrowser", "openbrowser-fast", "chrome-devtools-mcp", "agent-browser", "browser-use")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def make_context(workdir: Path, arm_name: str) -> RunContext:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = workdir / stamp / f"warm-{arm_name}"
    for name in ("downloads", "profile", "home", "tmp", "config"):
        (root / name).mkdir(parents=True, exist_ok=True)
    token = hashlib.sha1(f"warm-{arm_name}-{time.time_ns()}".encode()).hexdigest()[:10]
    return RunContext(
        task="wizard",
        arm=arm_name,
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


def summarize(records: list[dict], cold: dict[tuple[str, str], float | None]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["task"], record["arm"]), []).append(record)
    rows: list[dict] = []
    for task in WARM_TASKS:
        for arm in DEFAULT_ARMS:
            items = groups.get((task, arm), [])
            if not items:
                continue
            successes = [item for item in items if item["ok"]]
            latencies = [item["latency_ms"] for item in successes]
            warm_p50 = percentile(latencies, 0.50)
            cold_p50 = cold.get((task, arm))
            rows.append(
                {
                    "task": task,
                    "arm": arm,
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


def cold_p50(path: Path) -> dict[tuple[str, str], float | None]:
    result: dict[tuple[str, str], list[float]] = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not record.get("ok"):
            continue
        key = (str(record.get("task")), str(record.get("arm")))
        result.setdefault(key, []).append(float(record["latency_ms"]))
    return {key: percentile(values, 0.50) for key, values in result.items()}


def print_table(rows: list[dict]) -> None:
    header = f"{'task':<10} {'arm':<20} {'ok':>5} {'cold p50':>10} {'warm p50':>10} {'warm p95':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        cold = f"{row['cold_p50_ms']:.0f}" if row["cold_p50_ms"] else "n/a"
        warm = f"{row['warm_p50_ms']:.0f}" if row["warm_p50_ms"] else "n/a"
        p95 = f"{row['warm_p95_ms']:.0f}" if row["warm_p95_ms"] else "n/a"
        speedup = f"{row['speedup']:.2f}x" if row["speedup"] else "n/a"
        print(
            f"{row['task']:<10} {row['arm']:<20} {row['ok']:>2}/{row['runs']:<2} "
            f"{cold:>10} {warm:>10} {p95:>10} {speedup:>8}"
        )
        for error in row["errors"]:
            print(f"    error: {error[:160]}")


def redact(value: str) -> str:
    """Hide the user home and never publish a worktree path."""
    text = str(value)
    if ".worktrees" in text:
        return "working copy"
    home = str(Path.home())
    return text.replace(home, "$HOME")


def _binary_version(binary: str) -> str:
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--tasks", nargs="+", default=list(WARM_TASKS))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--cold-results", type=Path, default=DEFAULT_COLD_RESULTS)
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()

    unknown_tasks = [task for task in args.tasks if task not in ORACLES]
    if unknown_tasks:
        parser.error(f"unknown tasks: {unknown_tasks}")
    if "auth" in args.tasks:
        parser.error("auth belongs to the cold lane because it restarts the browser")
    unknown_arms = [arm for arm in args.arms if arm not in DEFAULT_ARMS]
    if unknown_arms:
        parser.error(f"unknown arms: {unknown_arms}")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    environments: dict[str, dict] = {}
    for arm_name in args.arms:
        root_ctx = make_context(args.workdir, arm_name)
        if arm_name in {"openbrowser", "openbrowser-fast"}:
            _write_config(root_ctx, extensions=False)
        arm = make_arm(arm_name)
        warmup_payload: dict = {}
        warmup_ms = 0.0
        try:
            warmup_payload, warmup_ms = arm.warmup(root_ctx)
            environments[arm_name] = {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "binary": redact(getattr(arm, "binary", "")),
                "version": warmup_payload.get("browser_version")
                or (_binary_version(getattr(arm, "binary", "")) if getattr(arm, "binary", "") else ""),
                "warmup_ms": round(warmup_ms, 3),
                "warmup_payload": warmup_payload,
            }
            print(
                f"{arm_name:<20} warmup {warmup_ms:.0f} ms {json.dumps(warmup_payload, ensure_ascii=False)[:180]}",
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
                    )
                    if task == "download":
                        clean_downloads(ctx.downloads)
                    try:
                        started = time.monotonic()
                        result = arm.run(ctx, cleanup=False)
                        latency_ms = (time.monotonic() - started) * 1000.0
                        state = server.state.snapshot()
                        oracle = ORACLES[task](state, result.reported, ctx.downloads)
                    finally:
                        stop_fixture(server, thread)
                    record = {
                        "run_id": f"warm-{arm_name}-{task}-s{sample}",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "lane": "warm",
                        "task": task,
                        "arm": arm_name,
                        "sample": sample,
                        "ok": bool(result.ok and oracle.ok),
                        "adapter_ok": result.ok,
                        "oracle_ok": oracle.ok,
                        "oracle_detail": oracle.detail,
                        "error": result.error,
                        "latency_ms": latency_ms,
                        "warmup_ms": warmup_ms,
                        "warmup_payload": warmup_payload,
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
                        f"{arm_name:<20} {task:<10} sample {sample} {status} {latency_ms:.0f} ms "
                        f"calls={result.tool_calls} procs={result.process_starts}",
                        flush=True,
                    )
                    if record["error"]:
                        print(f"    error: {record['error'][:200]}", flush=True)
        except FileNotFoundError as error:
            print(f"{arm_name:<20} unavailable: {error}", flush=True)
            environments[arm_name] = {"unavailable": str(error)}
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
    summary_path.write_text(
        json.dumps(
            {
                "environment": {
                    "cold_results": str(args.cold_results),
                    "tasks": args.tasks,
                    "samples": args.samples,
                    "arms": environments,
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print()
    print_table(rows)
    print()
    print(f"jsonl: {args.output}")
    print(f"summary: {summary_path}")
    return 0 if all(row["ok"] == row["runs"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
