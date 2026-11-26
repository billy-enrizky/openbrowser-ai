"""Run the deterministic interactive browser benchmark.

Usage:
  uv run --no-project python benchmarks/interactive/run_benchmark.py \
    --arms openbrowser chrome-devtools-mcp agent-browser browser-use \
    --tasks wizard modal download tabs auth --samples 1

The harness only calls local fixtures. It never calls AWS, a cloud browser,
or a model provider. Every browser launch includes --use-mock-keychain, and
every adapter cleans up its daemons and browser processes before returning.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from .adapters.agent_browser import AgentBrowserArm
from .adapters.base import RunContext, host_playwright_cache, terminate_matching
from .adapters.browser_use import BrowserUseArm
from .adapters.chrome_devtools import ChromeDevToolsArm
from .adapters.openbrowser import OpenBrowserArm
from .adapters.openbrowser_mcp import OpenBrowserMcpArm
from .arms import CANONICAL_ARMS
from .engine_preflight import configured_engine_pin
from .engine_session import EngineSession
from .fixtures.server import FixtureServer
from .measurement import MEASUREMENT_METHOD, TaskTimer
from .tasks import ORACLES, TASKS

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "results-smoke.jsonl"
DEFAULT_WORKDIR = Path("/tmp/obib-runs")
_VERSION_CACHE: dict[str, str] = {}


def _loads() -> list[float]:
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return []
    try:
        return [round(value, 3) for value in getloadavg()]
    except OSError:
        return []


def _binary_version(binary: str) -> str:
    """Return a cached product version for one arm binary."""
    if binary in _VERSION_CACHE:
        return _VERSION_CACHE[binary]
    try:
        completed = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        value = completed.stdout.strip() or completed.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        value = ""
    _VERSION_CACHE[binary] = value
    return value


def make_arm(name: str, *, openbrowser_extensions: bool = False):
    if name == "openbrowser":
        return OpenBrowserArm(extensions=openbrowser_extensions)
    if name == "openbrowser-fast":
        return OpenBrowserArm(extensions=openbrowser_extensions, fast_selectors=True)
    if name == "openbrowser-mcp":
        return OpenBrowserMcpArm()
    if name == "openbrowser-mcp-fast":
        return OpenBrowserMcpArm(fast_selectors=True)
    if name == "chrome-devtools-mcp":
        return ChromeDevToolsArm()
    if name == "agent-browser":
        return AgentBrowserArm()
    if name == "browser-use":
        return BrowserUseArm()
    if name == "playwright-mcp":
        from .adapters.playwright_mcp import PlaywrightMcpArm

        return PlaywrightMcpArm()
    if name == "playwright-cli":
        from .adapters.playwright_cli import PlaywrightCliArm

        return PlaywrightCliArm()
    raise ValueError(f"unknown arm {name}")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def start_fixture() -> tuple[FixtureServer, threading.Thread]:
    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_fixture(server: FixtureServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def make_context(task: str, arm: str, sample: int, base_url: str, workdir: Path) -> RunContext:
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="c-", dir=workdir))
    for name in ("downloads", "profile", "home", "tmp", "config"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return RunContext(
        task=task,
        arm=arm,
        sample=sample,
        base_url=base_url,
        root=root,
        downloads=root / "downloads",
        profile=root / "profile",
        home=root / "home",
        tmp=root / "tmp",
        config=root / "config",
        session="b" + hashlib.sha1(f"{task}-{arm}-{sample}-{time.time_ns()}".encode()).hexdigest()[:10],
    )


def run_one(
    task: str,
    arm_name: str,
    sample: int,
    workdir: Path,
    keep: bool,
    *,
    openbrowser_extensions: bool = False,
    record_binary_version: bool = True,
) -> dict:
    engine_session = EngineSession(configured_engine_pin())
    engine_observation = None
    server, thread = start_fixture()
    ctx = make_context(task, arm_name, sample, server.base_url, workdir)
    arm = None
    timer = TaskTimer()
    try:
        with timer:
            arm = make_arm(arm_name, openbrowser_extensions=openbrowser_extensions)
            result = arm.run(ctx, cleanup=False)
    except FileNotFoundError as exc:
        result = None
        error = f"unavailable: {exc}"
    except Exception as exc:
        result = None
        error = f"adapter crashed: {exc}"
    finally:
        measurement = timer.public()
        elapsed_ms = measurement["task_ns"] / 1_000_000
        if arm is not None:
            engine_observation = engine_session.observe(arm, ctx)
        if arm is not None:
            with contextlib.suppress(Exception):
                arm.cleanup(ctx)
        terminate_matching(str(ctx.root))
        state = server.state.snapshot()
        stop_fixture(server, thread)

    if result is None:
        record = {
            "run_id": f"{task}-{arm_name}-s{sample}",
            "timestamp": datetime.now(UTC).isoformat(),
            "task": task,
            "arm": arm_name,
            "sample": sample,
            "ok": False,
            "adapter_ok": False,
            "oracle_ok": False,
            "unavailable": error.startswith("unavailable:"),
            "error": error,
            "oracle_detail": "",
            "latency_ms": elapsed_ms,
            "tool_calls": 0,
            "process_starts": 0,
            "output_bytes": 0,
            "stderr_bytes": 0,
            "browser_version": "",
            "reported": {},
            "requests": state.get("request_count", 0),
            "request_bytes": state.get("request_bytes", 0),
            "keychain_flag": "--use-mock-keychain",
            "keychain_verified": None,
            "openbrowser_extensions": openbrowser_extensions if arm_name == "openbrowser" else None,
            "binary_version": "",
        }
    else:
        oracle = ORACLES[task](state, result.reported, ctx.downloads)
        record = {
            "run_id": f"{task}-{arm_name}-s{sample}",
            "timestamp": datetime.now(UTC).isoformat(),
            "task": task,
            "arm": arm_name,
            "sample": sample,
            "ok": bool(result.ok and oracle.ok),
            "adapter_ok": result.ok,
            "oracle_ok": oracle.ok,
            "unavailable": result.unavailable,
            "error": result.error,
            "oracle_detail": oracle.detail,
            "latency_ms": elapsed_ms,
            "tool_calls": result.tool_calls,
            "process_starts": result.process_starts,
            "output_bytes": len(result.stdout.encode()),
            "stderr_bytes": len(result.stderr.encode()),
            "browser_version": result.browser_version,
            "reported": result.reported,
            "requests": state.get("request_count", 0),
            "request_bytes": state.get("request_bytes", 0),
            "keychain_flag": "--use-mock-keychain",
            "keychain_verified": result.keychain_verified,
            "openbrowser_extensions": openbrowser_extensions if arm_name == "openbrowser" else None,
            "binary_version": (_binary_version(getattr(arm, "binary", "")) if arm is not None else "")
            if record_binary_version
            else None,
        }
    if not keep:
        shutil.rmtree(ctx.root, ignore_errors=True)
    record["measurement_method"] = MEASUREMENT_METHOD
    record["measurement"] = measurement
    engine_session.stamp(record, engine_observation)
    return record


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["task"], record["arm"]), []).append(record)
    rows = []
    for (task, arm), items in groups.items():
        successes = [item for item in items if item["ok"]]
        latencies = [item["latency_ms"] for item in successes]
        rows.append(
            {
                "task": task,
                "arm": arm,
                "runs": len(items),
                "ok": len(successes),
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "median_tool_calls": statistics.median([item["tool_calls"] for item in successes])
                if successes
                else None,
                "median_process_starts": statistics.median([item["process_starts"] for item in successes])
                if successes
                else None,
                "median_output_bytes": statistics.median([item["output_bytes"] for item in successes])
                if successes
                else None,
                "errors": sorted({item["error"] for item in items if item["error"]})[:3],
            }
        )
    return sorted(rows, key=lambda row: (list(TASKS).index(row["task"]), row["arm"]))


def print_table(rows: list[dict]) -> None:
    header = f"{'task':<10} {'arm':<20} {'ok':>5} {'p50 ms':>8} {'p95 ms':>8} {'calls':>6} {'proc':>5} {'bytes':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        p50 = f"{row['p50_ms']:.0f}" if row["p50_ms"] is not None else "n/a"
        p95 = f"{row['p95_ms']:.0f}" if row["p95_ms"] is not None else "n/a"
        calls = f"{row['median_tool_calls']:.0f}" if row["median_tool_calls"] is not None else "n/a"
        procs = f"{row['median_process_starts']:.0f}" if row["median_process_starts"] is not None else "n/a"
        out = f"{row['median_output_bytes']:.0f}" if row["median_output_bytes"] is not None else "n/a"
        print(
            f"{row['task']:<10} {row['arm']:<20} {row['ok']:>2}/{row['runs']:<2} "
            f"{p50:>8} {p95:>8} {calls:>6} {procs:>5} {out:>8}"
        )
        for error in row["errors"]:
            print(f"    error: {error[:160]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(CANONICAL_ARMS),
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--rotate-arms",
        action="store_true",
        help="Rotate the arm order for each sample so host-load drift is balanced across arms.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument(
        "--openbrowser-extensions",
        action="store_true",
        help="Measure the OpenBrowser default extension bundle instead of the lean automation profile.",
    )
    args = parser.parse_args()

    unknown_tasks = [task for task in args.tasks if task not in TASKS]
    if unknown_tasks:
        parser.error(f"unknown tasks: {unknown_tasks}")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    load_before = _loads()
    for warmup in range(args.warmups):
        for task in args.tasks:
            for arm in args.arms:
                print(f"[warmup {warmup + 1}] {task} {arm}", flush=True)
                run_one(
                    task,
                    arm,
                    1000 + warmup,
                    args.workdir,
                    False,
                    openbrowser_extensions=args.openbrowser_extensions,
                )

    total = len(args.tasks) * len(args.arms) * args.samples
    counter = 0
    for sample in range(args.samples):
        for task in args.tasks:
            arm_order = list(args.arms)
            if args.rotate_arms and arm_order:
                shift = sample % len(arm_order)
                arm_order = arm_order[shift:] + arm_order[:shift]
            for order_index, arm in enumerate(arm_order):
                counter += 1
                print(f"[{counter}/{total}] {task} {arm} sample {sample}", flush=True)
                record = run_one(
                    task,
                    arm,
                    sample,
                    args.workdir,
                    args.keep_runs,
                    openbrowser_extensions=args.openbrowser_extensions,
                )
                record["arm_order"] = arm_order
                record["order"] = order_index
                records.append(record)
                status = "ok" if record["ok"] else ("unavailable" if record["unavailable"] else "fail")
                print(
                    f"    {status} latency={record['latency_ms']:.0f}ms "
                    f"calls={record['tool_calls']} procs={record['process_starts']} "
                    f"bytes={record['output_bytes']}"
                )
                if record["error"]:
                    print(f"    error: {record['error'][:220]}")

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    rows = summarize(records)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "arms": args.arms,
                "tasks": args.tasks,
                "samples": args.samples,
                "warmups": args.warmups,
                "rotate_arms": args.rotate_arms,
                "load_before": load_before,
                "load_after": _loads(),
                "mcp_transport": os.environ.get("OPENBROWSER_MCP_TRANSPORT", "lean"),
                "playwright_browsers_path_available": host_playwright_cache() is not None,
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
    print(f"meta: {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
