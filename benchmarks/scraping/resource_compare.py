"""Measure peak RSS and CPU time for the local scraping arms.

The task oracle and arm implementations come from scrape_compare.py. This lane
adds a process-tree sampler around each measured task. It is a resource lane,
not a latency lane: the sampler adds a small amount of overhead. Use the normal
scraping lane for clean wall-clock comparisons.

No LLM, cloud service, Docker, or external network is used.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import resource as resource_module
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil

from .scrape_compare import (
    FIRECRAWL_ASSET_WAIT_MS,
    TASKS,
    BrowserArm,
    FirecrawlArm,
    FirecrawlServer,
    FixtureServer,
    LightpandaArm,
    _measure,
    _resolve_firecrawl_binary,
    _summary,
)

DEFAULT_ARMS = (
    "openbrowser-full",
    "openbrowser-lightweight",
    "openbrowser-no-render",
    "firecrawl-local",
)


class ResourceSampler:
    """Sample the benchmark process tree while one task runs."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self._root = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._start_cpu: dict[int, float] = {}
        self._start_children = resource_module.getrusage(resource_module.RUSAGE_CHILDREN)

    def _tree(self) -> list[psutil.Process]:
        processes = [self._root]
        try:
            processes.extend(self._root.children(recursive=True))
        except psutil.Error:
            pass
        return processes

    @staticmethod
    def _cpu_seconds(process: psutil.Process) -> float:
        times = process.cpu_times()
        return float(times.user + times.system)

    def _sample_once(self) -> None:
        total = 0
        for process in self._tree():
            try:
                if process.is_running():
                    total += process.memory_info().rss
            except psutil.Error:
                continue
        self._peak_rss = max(self._peak_rss, total)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._sample_once()
        self._start_cpu = {process.pid: self._cpu_seconds(process) for process in self._tree()}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample_once()
        end_cpu = {process.pid: self._cpu_seconds(process) for process in self._tree()}
        live_cpu = 0.0
        for pid, started in self._start_cpu.items():
            if pid in end_cpu:
                live_cpu += max(0.0, end_cpu[pid] - started)
        end_children = resource_module.getrusage(resource_module.RUSAGE_CHILDREN)
        reaped_cpu = max(
            0.0,
            (end_children.ru_utime - self._start_children.ru_utime)
            + (end_children.ru_stime - self._start_children.ru_stime),
        )
        if sys.platform == "darwin":
            maxrss_mb = end_children.ru_maxrss / (1024 * 1024)
        else:
            maxrss_mb = end_children.ru_maxrss / 1024
        return {
            "peak_rss_mb": round(self._peak_rss / (1024 * 1024), 3),
            "live_tree_cpu_ms": round(live_cpu * 1000, 3),
            "reaped_child_cpu_ms": round(reaped_cpu * 1000, 3),
            "child_maxrss_mb": round(maxrss_mb, 3),
        }


def _build_arm(name: str, args: argparse.Namespace, firecrawl_server: FirecrawlServer | None):
    if name == "openbrowser-full":
        return BrowserArm(name, "full", chrome_path=args.chrome)
    if name == "openbrowser-lightweight":
        return BrowserArm(name, "lightweight", chrome_path=args.chrome)
    if name == "openbrowser-no-render":
        return BrowserArm(name, "no-render", chrome_path=args.chrome)
    if name == "firecrawl-local":
        assert firecrawl_server is not None
        return FirecrawlArm(name, firecrawl_server, {"asset_heavy": FIRECRAWL_ASSET_WAIT_MS})
    if name == "firecrawl-local-auto":
        assert firecrawl_server is not None
        return FirecrawlArm(name, firecrawl_server, {})
    if name == "lightpanda":
        binary = args.lightpanda_bin or os.environ.get("LIGHTPANDA_BIN")
        if not binary:
            raise SystemExit("lightpanda needs --lightpanda-bin or LIGHTPANDA_BIN")
        return LightpandaArm(binary)
    raise SystemExit(f"unknown arm: {name}")


def _group(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["task"], record["arm"]), []).append(record)
    groups = []
    for (task, arm), items in sorted(grouped.items()):
        ok = [item for item in items if item["ok"]]
        groups.append(
            {
                "task": task,
                "arm": arm,
                "successes": len(ok),
                "runs": len(items),
                "ok": len(ok) == len(items) and bool(items),
                "latency": _summary([item["latency_ms"] for item in ok]),
                "peak_rss_mb_max": max((item["peak_rss_mb"] for item in ok), default=None),
                "peak_rss_mb_p50": statistics.median([item["peak_rss_mb"] for item in ok]) if ok else None,
                "live_tree_cpu_ms_p50": statistics.median([item["live_tree_cpu_ms"] for item in ok]) if ok else None,
                "reaped_child_cpu_ms_p50": statistics.median([item["reaped_child_cpu_ms"] for item in ok])
                if ok
                else None,
                "child_maxrss_mb_max": max((item["child_maxrss_mb"] for item in ok), default=None),
                "mean_requests": round(statistics.fmean(item["requests"] for item in items), 1) if items else 0,
                "mean_response_bytes": round(statistics.fmean(item["response_bytes"] for item in items), 1)
                if items
                else 0,
                "errors": sorted({item["detail"] for item in items if item["detail"]}),
            }
        )
    return groups


async def _run(args: argparse.Namespace) -> int:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = server.base_url
    records: list[dict] = []
    environments: dict[str, dict] = {}
    try:
        for arm_name in args.arms:
            firecrawl_server: FirecrawlServer | None = None
            try:
                if arm_name.startswith("firecrawl"):
                    if args.firecrawl_url:
                        firecrawl_server = FirecrawlServer(url=args.firecrawl_url, chrome_path=args.chrome)
                    else:
                        binary = _resolve_firecrawl_binary(args.firecrawl_bin)
                        if not binary:
                            raise SystemExit(
                                "Firecrawl arms need firecrawl-local. Pass --firecrawl-bin, "
                                "set FIRECRAWL_LOCAL_BIN, or attach with --firecrawl-url."
                            )
                        firecrawl_server = FirecrawlServer(binary=binary, chrome_path=args.chrome)
                    firecrawl_server.start()
                    print(f"firecrawl-local {firecrawl_server.version} at {firecrawl_server.url}", flush=True)
                arm = _build_arm(arm_name, args, firecrawl_server)
                environments[arm_name] = {
                    "loaded": True,
                    "version": getattr(firecrawl_server, "version", "") if firecrawl_server else "",
                }
                await arm.start()
                try:
                    for task_name in args.tasks:
                        method = TASKS[task_name]
                        for _ in range(args.warmups):
                            await _measure(arm, task_name, method, base, server.state)
                        for sample in range(args.samples):
                            sampler = ResourceSampler()
                            sampler.start()
                            result = await _measure(arm, task_name, method, base, server.state)
                            resources = sampler.stop()
                            record = {
                                "task": task_name,
                                "arm": arm_name,
                                "sample": sample,
                                "ok": result.ok,
                                "latency_ms": round(result.latency_ms, 3),
                                "output_bytes": result.output_bytes,
                                "requests": result.requests,
                                "response_bytes": result.response_bytes,
                                "engine": result.engine,
                                "detail": result.detail,
                                **resources,
                            }
                            records.append(record)
                            print(
                                f"{arm_name:<24} {task_name:<15} sample {sample} "
                                f"{'ok' if result.ok else 'fail'} "
                                f"rss={resources['peak_rss_mb']:.1f}MB "
                                f"cpu={resources['live_tree_cpu_ms']:.0f}ms "
                                f"child_cpu={resources['reaped_child_cpu_ms']:.0f}ms "
                                f"latency={result.latency_ms:.0f}ms",
                                flush=True,
                            )
                            if result.detail:
                                print(f"    error: {result.detail[:200]}", flush=True)
                finally:
                    await arm.stop()
            except FileNotFoundError as error:
                environments[arm_name] = {"loaded": False, "error": str(error)}
                print(f"{arm_name} unavailable: {error}", flush=True)
            finally:
                if firecrawl_server is not None:
                    firecrawl_server.stop()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    groups = _group(records)
    output = {
        "samples": args.samples,
        "warmups": args.warmups,
        "tasks": args.tasks,
        "arms": args.arms,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "chrome_path": args.chrome or "",
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "arms": environments,
            "note": "Resource lane. The sampler adds overhead; use scrape_compare for clean latency.",
        },
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"{'task':<16} {'arm':<24} {'ok':>5} {'rss max':>9} {'rss p50':>9} "
        f"{'cpu p50':>8} {'child cpu':>10} {'lat p50':>8}"
    )
    for group in groups:
        print(
            f"{group['task']:<16} {group['arm']:<24} "
            f"{group['successes']}/{group['runs']:<3} "
            f"{group['peak_rss_mb_max'] or 0:9.1f} {group['peak_rss_mb_p50'] or 0:9.1f} "
            f"{group['live_tree_cpu_ms_p50'] or 0:8.0f} {group['reaped_child_cpu_ms_p50'] or 0:10.0f} "
            f"{group['latency'].get('p50_ms') or 0:8.0f}"
        )
        for error in group["errors"]:
            print(f"    error: {error[:160]}")
    print(f"Wrote {args.output}")
    # Some arms are expected to fail JavaScript tasks. Correctness is recorded
    # per group; the benchmark command itself completes after writing evidence.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--tasks", nargs="+", default=sorted(TASKS))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--chrome", default=None)
    parser.add_argument("--firecrawl-bin", default=None)
    parser.add_argument("--firecrawl-url", default=None)
    parser.add_argument("--lightpanda-bin", default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/scraping/results-resources.json"))
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
