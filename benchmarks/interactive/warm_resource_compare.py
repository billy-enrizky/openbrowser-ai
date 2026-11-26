"""Warm-session process-tree resources for the seven interactive paths.

Each arm keeps one native session for the whole lane. Warmup runs once,
outside every capture window. Every task gets a fresh fixture server, a
baseline process-tree sample, the task, a final sample, and the unchanged
oracle. Metrics are warm peak tree RSS, warm peak child footprint, and the
sampled task CPU delta. Unknown values stay null. No latency claim is made.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .resource_compare import (
    ProcessTreeSampler,
    arm_versions,
    binding_snapshot,
    cleanup_owned,
    source_manifest,
)
from .run_benchmark import make_arm
from .tasks import ORACLES
from .warm_benchmark import (
    DEFAULT_ARMS,
    WARM_TASKS,
    clean_downloads,
    make_context,
    start_fixture,
    stop_fixture,
)

LOGGER = logging.getLogger(__name__)
METHOD = "warm-session-resources-v1"
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class WarmPlanEntry:
    sample: int
    task: str
    arm: str
    position: int
    order: tuple[str, ...]


def warm_plan(
    arms: list[str] | tuple[str, ...], tasks: list[str] | tuple[str, ...], samples: int
) -> list[WarmPlanEntry]:
    if not arms or not tasks or samples < 1:
        raise ValueError("arms, tasks, and at least one sample are required")
    plan: list[WarmPlanEntry] = []
    for sample in range(samples):
        for task in tasks:
            order = list(arms)
            shift = sample % len(order)
            order = order[shift:] + order[:shift]
            for position, arm in enumerate(order):
                plan.append(WarmPlanEntry(sample, task, arm, position, tuple(order)))
    return plan


class ProcessSession:
    """One persistent native worker with a JSON-line control channel."""

    def __init__(self, command: list[str], *, cwd: Path, stderr_path: Path, env: dict[str, str]):
        self._queue: queue.Queue[dict] = queue.Queue()
        self._stderr = stderr_path.open("w")
        self.process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            env=env,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def send(self, message: dict) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def receive(self, timeout: float) -> dict:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("warm worker message timeout") from error

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self.process.stdin:
                self.process.stdin.close()
        with contextlib.suppress(Exception):
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
        with contextlib.suppress(Exception):
            if self.process.poll() is None:
                self.process.kill()
        self._reader.join(timeout=2)
        self._stderr.close()

    def wait(self, timeout: float) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None


def spawn_session(arm: str, workdir: Path, diagnostics: Path) -> ProcessSession:
    diagnostics.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "benchmarks.interactive.warm_resource_compare",
        "--worker",
        "--arm",
        arm,
        "--workdir",
        str(workdir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE.parent.parent), str(HERE.parent.parent / "src")])
    return ProcessSession(command, cwd=HERE.parent.parent, stderr_path=diagnostics / f"{arm}.stderr.log", env=env)


def _task_downloads(message: dict, ctx) -> Path:
    """Return the task download directory only when it matches the warmed one.

    Warmed adapters cache the output directory at warmup. Echoing a different
    per-task path would make the audit trail describe a directory the worker
    never writes to, so the worker rejects the mismatch before it starts.
    """
    value = message.get("downloads")
    if not isinstance(value, str) or Path(value).resolve() != ctx.downloads.resolve():
        raise ValueError("task downloads path does not match the warmed session")
    return Path(value)


def _emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record))


def run_worker(arm_name: str, workdir: Path) -> int:
    arm = make_arm(arm_name)
    root_ctx = make_context(workdir, arm_name)
    try:
        payload, warmup_ms = arm.warmup(root_ctx)
        _emit(
            {
                "type": "ready",
                "payload": payload,
                "warmup_ms": warmup_ms,
                "root": str(root_ctx.root),
                "downloads": str(root_ctx.downloads),
            }
        )
        while True:
            line = sys.stdin.readline()
            if not line:
                return 0
            message = json.loads(line)
            if message.get("type") == "finish":
                _emit({"type": "finished"})
                return 0
            if message.get("type") != "task":
                _emit({"type": "fatal", "error": "unknown worker message"})
                return 2
            nonce = str(message.get("nonce") or "")
            record_path = Path(message["record"])
            ctx = replace(
                root_ctx,
                task=message["task"],
                sample=int(message["sample"]),
                base_url=message["base_url"],
            )
            try:
                _task_downloads(message, ctx)
            except ValueError as error:
                _emit({"type": "fatal", "error": str(error)})
                return 2
            _emit(
                {
                    "type": "started",
                    "arm": arm_name,
                    "task": ctx.task,
                    "sample": ctx.sample,
                    "nonce": nonce,
                }
            )
            try:
                if ctx.downloads.exists():
                    clean_downloads(ctx.downloads)
                result = arm.run(ctx, cleanup=False)
                record = {
                    "task": ctx.task,
                    "arm": arm_name,
                    "sample": ctx.sample,
                    "nonce": nonce,
                    "ok": result.ok,
                    "adapter_ok": result.ok,
                    "error": result.error,
                    "reported": result.reported,
                    "keychain_verified": result.keychain_verified,
                    "browser_version": result.browser_version,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except Exception as error:
                record = {
                    "task": ctx.task,
                    "arm": arm_name,
                    "sample": ctx.sample,
                    "nonce": nonce,
                    "ok": False,
                    "adapter_ok": False,
                    "error": f"worker exception: {type(error).__name__}",
                    "reported": {},
                    "keychain_verified": None,
                }
            try:
                _write_record(record_path, record)
            except Exception as error:
                _emit({"type": "fatal", "error": f"record write failed: {type(error).__name__}"})
                return 2
            _emit(
                {
                    "type": "done",
                    "arm": arm_name,
                    "task": ctx.task,
                    "sample": ctx.sample,
                    "nonce": nonce,
                    "record": str(record_path),
                }
            )
    finally:
        with contextlib.suppress(Exception):
            arm.cleanup(root_ctx)


def _default_sampler(pid: int, marker: str) -> ProcessTreeSampler:
    return ProcessTreeSampler(pid, ownership_marker=marker)


def _sample_tree(sampler) -> None:
    discover = getattr(sampler, "discover_orphans", None)
    if discover is not None:
        discover()
    sampler.sample()


def _receive_until_done(session, sampler, *, interval: float, deadline: float) -> dict:
    while time.monotonic() < deadline:
        remaining = max(0.01, min(interval, deadline - time.monotonic()))
        try:
            message = session.receive(remaining)
        except TimeoutError:
            _sample_tree(sampler)
            continue
        if message.get("type") == "done":
            return message
        if message.get("type") == "fatal":
            raise RuntimeError(message.get("error", "worker fatal"))
    raise TimeoutError("warm task exceeded its deadline")


def _validated_root(value, workdir: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("ready root is missing")
    root = Path(value)
    work = Path(workdir).resolve()
    resolved = root.resolve()
    if not root.is_absolute() or resolved == work or not resolved.is_relative_to(work):
        raise ValueError("ready root is outside the private workdir")
    return str(resolved)


def _validated_downloads(value, root: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("ready downloads is missing")
    path = Path(value)
    resolved = path.resolve()
    if not path.is_absolute() or not resolved.is_relative_to(Path(root)):
        raise ValueError("ready downloads is outside the private root")
    return str(resolved)


def run_warm_lane(
    arms: list[str],
    tasks: list[str],
    samples: int,
    output: Path,
    *,
    workdir: Path,
    interval: float,
    worker_timeout: float,
    session_factory: Callable[[str, Path, Path], Any] = spawn_session,
    fixture_factory: Callable[[], Any] = start_fixture,
    fixture_stop: Callable[[Any, Any], None] = stop_fixture,
    sampler_factory: Callable[[int, str], Any] = _default_sampler,
    cleanup: Callable[..., dict] = cleanup_owned,
) -> list[dict]:
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = output.parent / f"{output.stem}.diagnostics"
    expected_observations = len(arms) * len(tasks) * samples
    records: list[dict] = []
    with output.open("x") as handle:
        header: dict = {
            "record_type": "run",
            "method": METHOD,
            "arms": arms,
            "tasks": tasks,
            "samples": samples,
            "interval_seconds": interval,
            "worker_timeout_seconds": worker_timeout,
            "expected_observations": expected_observations,
            "conclusion_policy": "Warm resource observations only. No latency claim.",
            "latency_claim_allowed": False,
        }
        try:
            before_source = source_manifest()
            before_binding = binding_snapshot(arms)
            header.update(
                {
                    **before_source,
                    "binding_before": before_binding,
                    "arm_versions": arm_versions(arms),
                }
            )
        except Exception as error:
            header["preflight_error"] = type(error).__name__
            handle.write(json.dumps(header) + "\n")
            handle.write(
                json.dumps(
                    {
                        "record_type": "end",
                        "method": METHOD,
                        "observations": 0,
                        "expected_observations": expected_observations,
                        "failures": 0,
                        "provenance_verified": False,
                        "cleanup_verified": False,
                        "valid": False,
                        "terminal": True,
                        "latency_claim_allowed": False,
                        "preflight_error": type(error).__name__,
                    }
                )
                + "\n"
            )
            handle.flush()
            return []
        handle.write(json.dumps(header) + "\n")
        handle.flush()
        sessions: dict[str, Any] = {}
        warmups: dict[str, dict] = {}
        lane_samplers: dict[str, Any] = {}
        setup_error: str | None = None
        aborted = False
        try:
            try:
                for arm in arms:
                    session = session_factory(arm, workdir, diagnostics)
                    sessions[arm] = session
                    ready = session.receive(worker_timeout)
                    if ready.get("type") != "ready":
                        raise RuntimeError(f"{arm} did not become ready")
                    root = _validated_root(ready.get("root"), workdir)
                    downloads = _validated_downloads(ready.get("downloads"), root)
                    warmups[arm] = {
                        "payload": ready.get("payload", {}),
                        "warmup_ms": ready.get("warmup_ms"),
                        "root": root,
                        "downloads": downloads,
                    }
                    lane_samplers[arm] = sampler_factory(session.pid, root)
                    _sample_tree(lane_samplers[arm])
            except Exception as error:
                setup_error = f"{type(error).__name__}: {error}"
                LOGGER.error("warm lane setup failed: %s", setup_error)
            if setup_error is None:
                for entry in warm_plan(arms, tasks, samples):
                    session = sessions[entry.arm]
                    server, thread = fixture_factory()
                    ctx_downloads = Path(warmups[entry.arm]["downloads"])
                    ctx_downloads.mkdir(parents=True, exist_ok=True)
                    clean_downloads(ctx_downloads)
                    sampler = sampler_factory(session.pid, warmups[entry.arm]["root"])
                    _sample_tree(sampler)
                    baseline = sampler.result()
                    nonce = uuid.uuid4().hex
                    record_path = diagnostics / f"record-{entry.task}-{entry.arm}-s{entry.sample}-{nonce}.json"
                    started = time.monotonic()
                    failure = None
                    try:
                        session.send(
                            {
                                "type": "task",
                                "task": entry.task,
                                "sample": entry.sample,
                                "base_url": server.base_url,
                                "downloads": str(ctx_downloads),
                                "record": str(record_path),
                                "nonce": nonce,
                            }
                        )
                        ack = session.receive(worker_timeout)
                        if (
                            ack.get("type") != "started"
                            or ack.get("arm") != entry.arm
                            or ack.get("task") != entry.task
                            or ack.get("sample") != entry.sample
                            or ack.get("nonce") != nonce
                        ):
                            raise RuntimeError("worker acknowledgement does not match the task")
                        _sample_tree(sampler)
                        done = _receive_until_done(
                            session, sampler, interval=interval, deadline=time.monotonic() + worker_timeout
                        )
                        if (
                            done.get("arm") != entry.arm
                            or done.get("task") != entry.task
                            or done.get("sample") != entry.sample
                            or done.get("nonce") != nonce
                        ):
                            raise RuntimeError("worker completion does not match the task")
                        _sample_tree(sampler)
                        emitted = done.get("record")
                        if not isinstance(emitted, str) or Path(emitted).resolve() != record_path.resolve():
                            raise ValueError("worker record path does not match the task")
                        record = json.loads(record_path.read_text())
                        if (
                            record.get("arm") != entry.arm
                            or record.get("task") != entry.task
                            or record.get("sample") != entry.sample
                            or record.get("nonce") != nonce
                        ):
                            raise ValueError("worker record is stale or foreign")
                    except Exception as error:
                        failure = f"{type(error).__name__}: {error}"
                        record = {
                            "task": entry.task,
                            "arm": entry.arm,
                            "sample": entry.sample,
                            "nonce": nonce,
                            "ok": False,
                            "adapter_ok": False,
                            "reported": {},
                            "keychain_verified": None,
                            "error": failure,
                        }
                    finally:
                        with contextlib.suppress(Exception):
                            fixture_stop(server, thread)
                    resources = sampler.result() if failure is None else {}
                    final_cpu = resources.get("observed_tree_cpu_seconds")
                    baseline_cpu = baseline.get("observed_tree_cpu_seconds")
                    if (
                        isinstance(final_cpu, (int, float))
                        and not isinstance(final_cpu, bool)
                        and isinstance(baseline_cpu, (int, float))
                        and not isinstance(baseline_cpu, bool)
                        and math.isfinite(float(final_cpu))
                        and math.isfinite(float(baseline_cpu))
                        and final_cpu >= baseline_cpu
                    ):
                        cpu_delta = float(final_cpu) - float(baseline_cpu)
                        cpu_error = None
                    else:
                        cpu_delta = None
                        cpu_error = "invalid_counter"
                    rss = resources.get("peak_tree_rss_bytes")
                    unverified = resources.get("unverified_processes", 0)
                    rss_valid = (
                        isinstance(rss, (int, float))
                        and not isinstance(rss, bool)
                        and math.isfinite(float(rss))
                        and float(rss) > 0
                    )
                    resource_ok = bool(
                        rss_valid
                        and resources.get("fatal_read_errors") == 0
                        and unverified == 0
                        and resources.get("observed_browser_during_native_run", 0) > 0
                        and cpu_delta is not None
                    )
                    oracle = ORACLES[entry.task](
                        server.state.snapshot() if hasattr(server, "state") else {},
                        record.get("reported", {}),
                        ctx_downloads,
                    )
                    row = {
                        **record,
                        "record_type": "sample",
                        "order": entry.position,
                        "arm_order": list(entry.order),
                        "oracle_ok": oracle.ok,
                        "oracle_detail": oracle.detail,
                        "warm_peak_tree_rss_bytes": rss,
                        "warm_peak_child_footprint_bytes": resources.get("peak_child_footprint_bytes"),
                        "warm_task_tree_cpu_delta_seconds": cpu_delta,
                        "warm_cpu_delta_error": cpu_error,
                        "warm_unverified_processes": unverified,
                        "warm_observed_child_processes": resources.get("observed_child_processes"),
                        "warm_observed_browser_during_native_run": resources.get("observed_browser_during_native_run"),
                        "warm_read_errors": resources.get("read_errors"),
                        "warm_fatal_read_errors": resources.get("fatal_read_errors"),
                        "warm_footprint_status": resources.get("footprint_status"),
                        "resource_ok": resource_ok,
                        "elapsed_seconds": round(time.monotonic() - started, 6),
                        "latency_claim_allowed": False,
                        "provisional": True,
                    }
                    records.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    LOGGER.info(
                        "%s %s s%s ok=%s oracle=%s",
                        entry.task,
                        entry.arm,
                        entry.sample,
                        row.get("ok"),
                        oracle.ok,
                    )
        finally:
            cleanup_reports: dict[str, dict] = {}
            for arm, session in sessions.items():
                with contextlib.suppress(Exception):
                    session.send({"type": "finish"})
                    session.receive(worker_timeout)
                with contextlib.suppress(Exception):
                    session.wait(10)
                with contextlib.suppress(Exception):
                    session.close()
                sampler = lane_samplers.get(arm)
                if sampler is None:
                    cleanup_reports[arm] = {"verified": False, "errors": ["sampler_unavailable"]}
                else:
                    try:
                        cleanup_reports[arm] = cleanup(sampler, interval=interval)
                    except Exception as error:
                        cleanup_reports[arm] = {"verified": False, "errors": [type(error).__name__]}
                if not cleanup_reports[arm].get("verified", False):
                    aborted = True
            try:
                after_source = source_manifest()
                after_binding = binding_snapshot(arms)
            except Exception as error:
                after_source = {}
                after_binding = {"verified": False, "reason": f"postflight_{type(error).__name__}"}
            source_unchanged = before_source.get("source_digest") == after_source.get("source_digest")
            binding_unchanged = all(
                before_binding.get(name) == after_binding.get(name)
                for name in ("entry_sha256", "actual_app_digest", "expected_app_digest", "dist_version")
            )
            verified = bool(
                before_binding.get("verified")
                and after_binding.get("verified")
                and source_unchanged
                and binding_unchanged
            )
            for row in records:
                row["provenance_verified"] = verified
            failures = sum(
                not item.get("ok")
                or not item.get("adapter_ok")
                or not item.get("oracle_ok")
                or item.get("keychain_verified") is not True
                or not item.get("resource_ok")
                for item in records
            )
            terminal = {
                "record_type": "end",
                "method": METHOD,
                "observations": len(records),
                "expected_observations": expected_observations,
                "failures": failures,
                "warmups": warmups,
                "cleanup": cleanup_reports,
                "cleanup_verified": bool(cleanup_reports)
                and all(item.get("verified", False) for item in cleanup_reports.values()),
                "provenance_verified": verified,
                "source_unchanged": source_unchanged,
                "binding_unchanged": binding_unchanged,
                "binding_after": after_binding,
                "latency_claim_allowed": False,
                "terminal": True,
            }
            if setup_error is not None:
                terminal["setup_error"] = setup_error
            if aborted:
                terminal.update({"aborted": True, "abort_reason": "cleanup_unverified"})
            terminal["valid"] = bool(
                terminal["provenance_verified"]
                and terminal["cleanup_verified"]
                and not aborted
                and setup_error is None
                and failures == 0
                and len(records) == expected_observations
            )
            handle.write(json.dumps(terminal) + "\n")
            handle.flush()
    return records


def lane_valid(records: list[dict], terminal: dict) -> bool:
    """Strict gate: complete cohort, every correctness gate, provenance and cleanup."""
    return bool(
        terminal.get("valid") is True
        and terminal.get("failures") == 0
        and len(records) == terminal.get("expected_observations")
        and terminal.get("provenance_verified") is True
        and terminal.get("cleanup_verified") is True
        and not terminal.get("aborted", False)
        and terminal.get("setup_error") is None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--workdir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--tasks", nargs="+", default=list(WARM_TASKS))
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--run-workdir", type=Path, default=Path("/tmp/obib-warm-resource-runs"))
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--worker-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.arm or not args.workdir:
            raise SystemExit("worker mode needs --arm and --workdir")
        return run_worker(args.arm, args.workdir)
    if args.output is None:
        raise SystemExit("--output is required for a lane run")
    if args.samples < 1 or not (0 < args.interval < args.worker_timeout):
        raise SystemExit("samples, interval, and worker timeout are invalid")
    records = run_warm_lane(
        args.arms,
        args.tasks,
        args.samples,
        args.output,
        workdir=args.run_workdir,
        interval=args.interval,
        worker_timeout=args.worker_timeout,
    )
    lines = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    terminal = lines[-1]
    return 0 if lane_valid(records, terminal) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
