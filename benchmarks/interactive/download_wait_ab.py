"""Compare baseline and event-wakeup builds on the exact warm download task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import textwrap
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.interactive import warm_benchmark
from benchmarks.interactive.adapters import openbrowser
from benchmarks.interactive.fixtures.server import REPORT_SHA256, REPORT_SIZE
from benchmarks.interactive.tasks import ORACLES


@contextmanager
def download_wait_ab_lock() -> Iterator[None]:
    lock_path = Path("/tmp/openbrowser-bench.lock")
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            time.sleep(1)
    try:
        yield
    finally:
        lock_path.rmdir()


def validate_download_wait_ab_workdir(workdir: Path) -> Path:
    resolved = workdir.expanduser().resolve()
    tmp_root = Path("/tmp").resolve()
    if not resolved.is_relative_to(tmp_root):
        raise ValueError("--workdir must be under /tmp")
    socket_path = resolved / "20261124T000000Z" / "warm-download-baseline" / "home" / ".openbrowser" / "daemon.sock"
    if len(os.fsencode(socket_path)) >= 100:
        raise ValueError("choose a shorter path under /tmp for the daemon socket")
    return resolved


@contextmanager
def instrument_download_wait_ab_timing() -> Iterator[None]:
    original_builder = openbrowser._fast_download
    marker = textwrap.dedent(
        """\
        waiter = asyncio.create_task(wait_for_download(timeout=90))
        await click_selector('#download-report')
        outcome = await waiter
        print(_json.dumps({"download": outcome}))
        """
    )
    replacement = textwrap.dedent(
        """\
        import time as _time
        _timing = {}
        async def _measured_wait():
            _outcome = await wait_for_download(timeout=90)
            _timing['wait_returned_at'] = _time.time()
            return _outcome
        waiter = asyncio.create_task(_measured_wait())
        _timing['click_started_at'] = _time.time()
        await click_selector('#download-report')
        _timing['click_returned_at'] = _time.time()
        outcome = await waiter
        _timing['terminal_at'] = outcome.get('finished_at')
        if not isinstance(_timing['terminal_at'], (int, float)):
            raise RuntimeError('download outcome has no terminal timestamp')
        _timing['click_to_terminal_ms'] = (_timing['terminal_at'] - _timing['click_started_at']) * 1000
        _timing['terminal_to_wait_return_ms'] = (_timing['wait_returned_at'] - _timing['terminal_at']) * 1000
        _timing['click_call_ms'] = (_timing['click_returned_at'] - _timing['click_started_at']) * 1000
        _timing['wait_returned_before_click_return'] = _timing['wait_returned_at'] <= _timing['click_returned_at']
        print(_json.dumps({"download": outcome, "timing": _timing}))
        """
    )

    def build_instrumented_download(url: str) -> str:
        code = original_builder(url)
        count = code.count(marker)
        if count != 1:
            raise RuntimeError(f"expected one fast-download block, found {count}")
        return code.replace(marker, replacement, 1)

    openbrowser._fast_download = build_instrumented_download
    try:
        yield
    finally:
        openbrowser._fast_download = original_builder


def download_wait_ab_load_average() -> list[float]:
    return [float(value) for value in os.getloadavg()]


def download_wait_ab_sample_stats(sample_records: list[dict], field: str) -> dict:
    values = sorted(float(record[field]) for record in sample_records)
    return {
        "n": len(values),
        "p50_ms": statistics.median(values),
        "min_ms": values[0],
        "max_ms": values[-1],
        "range_ms": values[-1] - values[0],
    }


def run_download_wait_ab_sample(arm, root_ctx, label: str, sample: int, sequence: int, source_sha: str) -> dict:
    server, thread = warm_benchmark.start_fixture()
    ctx = replace(root_ctx, task="download", sample=sample, base_url=server.base_url)
    warm_benchmark.clean_downloads(ctx.downloads)
    try:
        load_before = download_wait_ab_load_average()
        with instrument_download_wait_ab_timing():
            started = time.monotonic()
            result = arm.run(ctx, cleanup=False)
            latency_ms = (time.monotonic() - started) * 1000
        state = server.state.snapshot()
        oracle = ORACLES["download"](state, result.reported, ctx.downloads)
        reported_download = result.reported.get("download", {})
        report_files = [path for path in ctx.downloads.rglob("report.csv") if path.is_file()]
        actual_hash = None
        actual_size = None
        if report_files:
            data = report_files[0].read_bytes()
            actual_hash = hashlib.sha256(data).hexdigest()
            actual_size = len(data)
        exact_outcome = (
            reported_download.get("phase") == "completed"
            and reported_download.get("file_size") == REPORT_SIZE
            and actual_hash == REPORT_SHA256
            and actual_size == REPORT_SIZE
        )
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "sequence": sequence,
            "sample": sample,
            "arm": label,
            "source_sha": source_sha,
            "ok": bool(result.ok and oracle.ok and exact_outcome),
            "adapter_ok": result.ok,
            "oracle_ok": oracle.ok,
            "oracle_detail": oracle.detail,
            "exact_outcome": exact_outcome,
            "latency_ms": latency_ms,
            "timing": result.reported.get("timing", {}),
            "file_size": actual_size,
            "file_sha256": actual_hash,
            "expected_file_size": REPORT_SIZE,
            "expected_file_sha256": REPORT_SHA256,
            "download_requests": state.get("download_requests"),
            "download_completed": state.get("download_completed"),
            "request_count": state.get("request_count"),
            "request_bytes": state.get("request_bytes"),
            "stderr_bytes": len(result.stderr.encode()),
            "browser_version": result.browser_version,
            "load_before": load_before,
            "load_after": download_wait_ab_load_average(),
            "error": result.error,
            "reported": result.reported,
        }
        return record
    finally:
        warm_benchmark.stop_fixture(server, thread)


def _run(args: argparse.Namespace) -> int:
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arms = {
        "baseline": openbrowser.OpenBrowserArm(binary=str(args.baseline_bin), fast_selectors=True),
        "treatment": openbrowser.OpenBrowserArm(binary=str(args.treatment_bin), fast_selectors=True),
    }
    contexts = {label: warm_benchmark.make_context(args.workdir, f"download-{label}") for label in arms}
    download_wait_ab_records: list[dict] = []
    loads: list[list[float]] = [download_wait_ab_load_average()]
    try:
        for label, arm in arms.items():
            openbrowser._write_config(contexts[label], extensions=False)
            payload, warmup_ms = arm.warmup(contexts[label])
            print(f"{label} warmup={warmup_ms:.1f} ms {json.dumps(payload)[:180]}", flush=True)
            loads.append(download_wait_ab_load_average())

        sequence = 0
        with args.output.open("w", encoding="utf-8") as handle:
            for sample in range(args.samples):
                order = ("baseline", "treatment") if sample % 2 == 0 else ("treatment", "baseline")
                for label in order:
                    source_sha = args.baseline_sha if label == "baseline" else args.treatment_sha
                    row = run_download_wait_ab_sample(arms[label], contexts[label], label, sample, sequence, source_sha)
                    sequence += 1
                    download_wait_ab_records.append(row)
                    loads.extend([row["load_before"], row["load_after"]])
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    status = "ok" if row["ok"] else "fail"
                    timing = row["timing"]
                    print(
                        f"{label:<10} sample={sample} {status} task={row['latency_ms']:.3f} ms "
                        f"click-terminal={timing.get('click_to_terminal_ms')} ms "
                        f"terminal-return={timing.get('terminal_to_wait_return_ms')} ms "
                        f"bytes={row['file_size']} hash_ok={row['file_sha256'] == REPORT_SHA256}",
                        flush=True,
                    )
        loads.append(download_wait_ab_load_average())
    finally:
        for label, arm in arms.items():
            try:
                arm.cleanup(contexts[label])
            finally:
                shutil.rmtree(contexts[label].root, ignore_errors=True)

    grouped = {label: [record for record in download_wait_ab_records if record["arm"] == label] for label in arms}
    metrics = {}
    for label, items in grouped.items():
        metrics[label] = {
            "all_ok": all(row["ok"] for row in items),
            "ok": sum(row["ok"] for row in items),
            "runs": len(items),
            "task": download_wait_ab_sample_stats(items, "latency_ms"),
            "click_to_terminal": download_wait_ab_sample_stats(
                [{"value": row["timing"]["click_to_terminal_ms"]} for row in items], "value"
            ),
            "terminal_to_wait_return": download_wait_ab_sample_stats(
                [{"value": row["timing"]["terminal_to_wait_return_ms"]} for row in items], "value"
            ),
            "click_call": download_wait_ab_sample_stats(
                [{"value": row["timing"]["click_call_ms"]} for row in items], "value"
            ),
        }
    baseline = metrics["baseline"]["task"]
    treatment = metrics["treatment"]["task"]
    improvement_ms = baseline["p50_ms"] - treatment["p50_ms"]
    noise_floor_ms = max(baseline["range_ms"], treatment["range_ms"])
    max_load = max(max(values) for values in loads)
    summary = {
        "baseline_sha": args.baseline_sha,
        "treatment_sha": args.treatment_sha,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "samples_per_arm": args.samples,
        "order": "alternating baseline/treatment by sample, reversed on odd samples",
        "expected_file_size": REPORT_SIZE,
        "expected_file_sha256": REPORT_SHA256,
        "metrics": metrics,
        "improvement_ms": improvement_ms,
        "noise_floor_ms": noise_floor_ms,
        "improvement_exceeds_sample_spread": improvement_ms > noise_floor_ms,
        "load_samples_1_5_15m": loads,
        "max_load_average": max_load,
        "load_within_limit": max_load <= 4.0,
        "claimable": (
            all(metric["all_ok"] for metric in metrics.values()) and improvement_ms > noise_floor_ms and max_load <= 4.0
        ),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(metric["all_ok"] for metric in metrics.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-bin", type=Path, required=True)
    parser.add_argument("--treatment-bin", type=Path, required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--treatment-sha", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 5:
        parser.error("at least five samples are required")
    if not args.baseline_bin.is_file() or not args.treatment_bin.is_file():
        parser.error("both CLI binaries must exist")
    try:
        args.workdir = validate_download_wait_ab_workdir(args.workdir)
    except ValueError as error:
        parser.error(str(error))
    with download_wait_ab_lock():
        return _run(args)


if __name__ == "__main__":
    sys.exit(main())
