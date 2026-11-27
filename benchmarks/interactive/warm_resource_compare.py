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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .adapters.base import ArmResult, RunContext
from .engine_preflight import configured_engine_pin
from .engine_session import EngineSession
from .resource_compare import (
    SAMPLER_METHOD_VERSION,
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
    start_fixture,
    stop_fixture,
)

LOGGER = logging.getLogger(__name__)
METHOD = "warm-session-resources-v2"
FINISH_GRACE_SECONDS = 5.0
FINISH_WAIT_SECONDS = 5.0
WORKER_GATE_TIMEOUT_SECONDS = 120.0
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
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("warm worker message timeout")
            try:
                return self._queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                code = self.process.poll()
                if code is None:
                    continue
                try:
                    return self._queue.get(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
                except queue.Empty:
                    raise RuntimeError(f"warm worker exited with code {code}") from None

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


def make_owned_context(root: Path, arm_name: str) -> RunContext:
    """Build the context directly in the preallocated private arm scope."""
    root = Path(root).resolve()
    for name in ("downloads", "profile", "home", "tmp", "config"):
        (root / name).mkdir(parents=True, exist_ok=True)
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
        session=uuid.uuid4().hex[:10],
    )


class _WorkerInbox:
    """Read JSON-line parent messages off stdin with bounded gate waits."""

    def __init__(self, stream=None) -> None:
        self._stream = stream or sys.stdin
        self._queue: queue.Queue[dict | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        try:
            for line in self._stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._queue.put({"type": "fatal", "error": "invalid worker message"})
                    continue
                self._queue.put(message)
        finally:
            self._queue.put(None)

    def next(self, timeout: float | None = None) -> dict:
        try:
            message = self._queue.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("worker message timeout") from error
        if message is None:
            raise EOFError
        if not isinstance(message, dict):
            raise ValueError("worker message is not an object")
        return message


def _task_context(message: dict, arm_name: str) -> tuple[str, int, str, Path]:
    if message.get("type") != "task" or message.get("arm") != arm_name:
        raise ValueError("worker task message does not match the arm")
    task = message.get("task")
    sample = message.get("sample")
    nonce = message.get("nonce")
    base_url = message.get("base_url")
    record = message.get("record")
    if not isinstance(task, str) or task not in ORACLES:
        raise ValueError("worker task is invalid")
    if type(sample) is not int or sample < 0:
        raise ValueError("worker sample is invalid")
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("worker nonce is invalid")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("worker base URL is invalid")
    if not isinstance(record, str) or not record:
        raise ValueError("worker record path is invalid")
    return task, sample, nonce, Path(record)


def _gate_message(
    channel: _WorkerInbox,
    expected_type: str,
    *,
    arm_name: str,
    task: str,
    sample: int,
    nonce: str,
) -> dict | None:
    try:
        message = channel.next(WORKER_GATE_TIMEOUT_SECONDS)
    except TimeoutError:
        _emit({"type": "fatal", "error": f"{expected_type} gate timeout"})
        return None
    except EOFError:
        _emit({"type": "fatal", "error": f"{expected_type} gate EOF"})
        return None
    if (
        message.get("type") != expected_type
        or message.get("arm") != arm_name
        or message.get("task") != task
        or message.get("sample") != sample
        or message.get("nonce") != nonce
    ):
        _emit({"type": "fatal", "error": f"{expected_type} gate mismatch"})
        return None
    return message


def run_worker(arm_name: str, workdir: Path) -> int:
    try:
        engine_session = EngineSession(configured_engine_pin())
        arm = make_arm(arm_name)
        root_ctx = make_owned_context(workdir, arm_name)
    except Exception as error:
        _emit({"type": "fatal", "error": f"worker setup failed: {type(error).__name__}"})
        return 2
    try:
        try:
            payload, warmup_ms = arm.warmup(root_ctx)
            engine_warmup = engine_session.capture_warm(arm, root_ctx)
            if engine_warmup is not None:
                payload = {**payload, "engine": engine_warmup.public_proof()}
        except Exception as error:
            _emit({"type": "fatal", "error": f"warmup failed: {type(error).__name__}"})
            return 2
        _emit(
            {
                "type": "ready",
                "payload": payload,
                "warmup_ms": warmup_ms,
                "root": str(root_ctx.root),
                "downloads": str(root_ctx.downloads),
            }
        )
        channel = _WorkerInbox()
        while True:
            try:
                message = channel.next()
            except EOFError:
                return 0
            except ValueError:
                _emit({"type": "fatal", "error": "invalid worker message"})
                return 2
            if message.get("type") == "finish":
                _emit({"type": "finished"})
                return 0
            try:
                task, sample, nonce, record_path = _task_context(message, arm_name)
                ctx = replace(
                    root_ctx,
                    task=task,
                    sample=sample,
                    base_url=message["base_url"],
                )
                _task_downloads(message, ctx)
            except Exception:
                _emit({"type": "fatal", "error": "invalid worker task"})
                return 2
            setup_error: str | None = None
            engine_before = None
            try:
                engine_before = engine_session.verify_warm(arm, ctx)
                if ctx.downloads.exists():
                    clean_downloads(ctx.downloads)
            except Exception as error:
                setup_error = f"worker exception: {type(error).__name__}"
            if engine_before is not None and not engine_before.eligible:
                setup_error = setup_error or f"matched_engine_before:{engine_before.status}"
            _emit(
                {
                    "type": "started",
                    "arm": arm_name,
                    "task": task,
                    "sample": sample,
                    "nonce": nonce,
                    "before": engine_before.public_proof() if engine_before is not None else None,
                }
            )
            if _gate_message(channel, "capture", arm_name=arm_name, task=task, sample=sample, nonce=nonce) is None:
                return 2
            if setup_error is None:
                try:
                    result = arm.run(ctx, cleanup=False)
                except Exception as error:
                    result = ArmResult(ok=False, error=f"worker exception: {type(error).__name__}")
            else:
                result = ArmResult(ok=False, error=setup_error)
            _emit(
                {
                    "type": "native_done",
                    "arm": arm_name,
                    "task": task,
                    "sample": sample,
                    "nonce": nonce,
                }
            )
            if _gate_message(channel, "verify", arm_name=arm_name, task=task, sample=sample, nonce=nonce) is None:
                return 2
            engine_after = None
            try:
                engine_after = engine_session.verify_warm(arm, ctx)
                record = {
                    "task": task,
                    "arm": arm_name,
                    "sample": sample,
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
                engine_session.stamp(record, engine_after, before=engine_before)
            except Exception as error:
                record = {
                    "task": task,
                    "arm": arm_name,
                    "sample": sample,
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
                    "task": task,
                    "sample": sample,
                    "nonce": nonce,
                    "record": str(record_path),
                }
            )
    finally:
        with contextlib.suppress(Exception):
            arm.cleanup(root_ctx)


def _default_sampler(pid: int, marker: str) -> ProcessTreeSampler:
    return ProcessTreeSampler(pid, ownership_marker=marker)


def _make_owned_sampler(pid: int, marker: str, factory: Callable[[int, str], Any]):
    try:
        return factory(pid, marker)
    except Exception:
        if factory is not _default_sampler:
            raise
        LOGGER.warning("sampler construction failed; retrying the default sampler once")
        return factory(pid, marker)


def _sample_tree(sampler) -> None:
    discover = getattr(sampler, "discover_orphans", None)
    if discover is not None:
        discover()
    sampler.sample()


class _ResourceCapture:
    """Latch capture uncertainty while the native worker finishes its protocol."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.phase = "baseline"
        self.error: dict | None = None

    def _failed(self, operation: str, error: Exception) -> None:
        known = (OSError, PermissionError, ProcessLookupError, TimeoutError, RuntimeError, ValueError, TypeError)
        self.error = {
            "phase": self.phase,
            "operation": operation,
            "error_type": type(error).__name__ if type(error) in known else "Exception",
            "errno": error.errno if isinstance(error, OSError) and type(error.errno) is int else None,
        }

    def sample(self) -> None:
        if self.error is not None:
            return
        try:
            set_phase = getattr(self.sampler, "set_capture_phase", None)
            if callable(set_phase):
                set_phase(self.phase)
            _sample_tree(self.sampler)
        except Exception as error:
            self._failed("sample", error)

    def finish(self) -> None:
        if self.error is not None:
            return
        try:
            probe = getattr(self.sampler, "record_discovery_end_fences", None)
            if callable(probe):
                probe()
        except Exception as error:
            self._failed("end_probe", error)

    def result(self) -> dict:
        if self.error is not None:
            return {}
        try:
            return self.sampler.result()
        except Exception as error:
            self._failed("result", error)
            return {}


def _receive_until(session, sampler, expected_type: str, *, interval: float, deadline: float) -> dict:
    while time.monotonic() < deadline:
        remaining = max(0.01, min(interval, deadline - time.monotonic()))
        try:
            message = session.receive(remaining)
        except TimeoutError:
            _sample_tree(sampler)
            continue
        if message.get("type") == "fatal":
            raise RuntimeError(message.get("error", "worker fatal"))
        if message.get("type") == expected_type:
            return message
        raise RuntimeError(f"unexpected worker message: {message.get('type')}")
    raise TimeoutError("warm task exceeded its deadline")


def _validated_root(value, workdir: Path, *, expected: Path | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("ready root is missing")
    root = Path(value)
    work = Path(workdir).resolve()
    resolved = root.resolve()
    if not root.is_absolute() or resolved == work or not resolved.is_relative_to(work):
        raise ValueError("ready root is outside the private workdir")
    if expected is not None and resolved != expected.resolve():
        raise ValueError("ready root does not match the preallocated arm scope")
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
    fixture_errors: list[dict] = []
    with output.open("x") as handle:
        header: dict = {
            "record_type": "run",
            "method": METHOD,
            "sampler_method_version": SAMPLER_METHOD_VERSION,
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
            header["method"] = METHOD
            header["sampler_method_version"] = SAMPLER_METHOD_VERSION
        except Exception as error:
            header["preflight_error"] = type(error).__name__
            handle.write(json.dumps(header) + "\n")
            handle.write(
                json.dumps(
                    {
                        "record_type": "end",
                        "method": METHOD,
                        "sampler_method_version": SAMPLER_METHOD_VERSION,
                        "observations": 0,
                        "expected_observations": expected_observations,
                        "failures": 0,
                        "provenance_verified": False,
                        "cleanup_verified": False,
                        "fixture_errors": [],
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
        retired_arms: set[str] = set()
        retired_workers: dict[str, dict] = {}
        setup_error: str | None = None
        aborted = False
        try:
            try:
                for arm in arms:
                    owned = (Path(workdir).resolve() / f"warm-{uuid.uuid4().hex[:10]}").resolve()
                    owned.mkdir(parents=True, exist_ok=True)
                    session = session_factory(arm, owned, diagnostics)
                    sessions[arm] = session
                    lane_samplers[arm] = _make_owned_sampler(session.pid, str(owned), sampler_factory)
                    _sample_tree(lane_samplers[arm])
                    ready = session.receive(worker_timeout)
                    if ready.get("type") == "fatal":
                        raise RuntimeError(f"{arm} warmup failed: {ready.get('error')}")
                    if ready.get("type") != "ready":
                        raise RuntimeError(f"{arm} did not become ready")
                    root = _validated_root(ready.get("root"), workdir, expected=owned)
                    downloads = _validated_downloads(ready.get("downloads"), root)
                    if Path(downloads).resolve() != (owned / "downloads").resolve():
                        raise ValueError("ready downloads does not match the preallocated arm scope")
                    warmups[arm] = {
                        "payload": ready.get("payload", {}),
                        "warmup_ms": ready.get("warmup_ms"),
                        "root": root,
                        "downloads": downloads,
                        "marker": str(owned),
                    }
            except Exception as error:
                setup_error = type(error).__name__
                LOGGER.error("warm lane setup failed: %s", setup_error)
            if setup_error is None:
                for entry in warm_plan(arms, tasks, samples):
                    session = sessions[entry.arm]
                    try:
                        server, thread = fixture_factory()
                    except Exception as error:
                        fixture_errors.append(
                            {
                                "task": entry.task,
                                "arm": entry.arm,
                                "sample": entry.sample,
                                "phase": "start",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                        break
                    try:
                        try:
                            ctx_downloads = Path(warmups[entry.arm]["downloads"])
                            ctx_downloads.mkdir(parents=True, exist_ok=True)
                            clean_downloads(ctx_downloads)
                            sampler = (
                                None
                                if entry.arm in retired_arms
                                else sampler_factory(session.pid, warmups[entry.arm]["marker"])
                            )
                            capture = _ResourceCapture(sampler)
                            nonce = uuid.uuid4().hex
                            record_path = diagnostics / f"record-{entry.task}-{entry.arm}-s{entry.sample}-{nonce}.json"
                            started = time.monotonic()
                            baseline: dict = {}
                            resources: dict = {}
                        except Exception as error:
                            fixture_errors.append(
                                {
                                    "task": entry.task,
                                    "arm": entry.arm,
                                    "sample": entry.sample,
                                    "phase": "setup",
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                            continue
                        failure = None
                        try:
                            if entry.arm in retired_arms:
                                raise RuntimeError("warm worker retired after protocol failure")
                            session.send(
                                {
                                    "type": "task",
                                    "arm": entry.arm,
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
                            capture.sample()
                            baseline = capture.result()
                            session.send(
                                {
                                    "type": "capture",
                                    "arm": entry.arm,
                                    "task": entry.task,
                                    "sample": entry.sample,
                                    "nonce": nonce,
                                }
                            )
                            capture.phase = "native_run"
                            native_done = _receive_until(
                                session,
                                capture,
                                "native_done",
                                interval=interval,
                                deadline=time.monotonic() + worker_timeout,
                            )
                            if (
                                native_done.get("arm") != entry.arm
                                or native_done.get("task") != entry.task
                                or native_done.get("sample") != entry.sample
                                or native_done.get("nonce") != nonce
                            ):
                                raise RuntimeError("worker native completion does not match the task")
                            capture.phase = "final"
                            capture.sample()
                            capture.finish()
                            resources = capture.result()
                            session.send(
                                {
                                    "type": "verify",
                                    "arm": entry.arm,
                                    "task": entry.task,
                                    "sample": entry.sample,
                                    "nonce": nonce,
                                }
                            )
                            done = session.receive(worker_timeout)
                            if (
                                done.get("type") != "done"
                                or done.get("arm") != entry.arm
                                or done.get("task") != entry.task
                                or done.get("sample") != entry.sample
                                or done.get("nonce") != nonce
                            ):
                                raise RuntimeError("worker completion does not match the task")
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
                            if entry.arm not in retired_arms:
                                retired_arms.add(entry.arm)
                                with contextlib.suppress(Exception):
                                    session.close()
                                try:
                                    retired_cleanup = cleanup(lane_samplers[entry.arm], interval=interval)
                                except Exception:
                                    retired_cleanup = {"verified": False}
                                retired_workers[entry.arm] = {
                                    "reason": "protocol_failure",
                                    "cleanup": retired_cleanup,
                                }
                                if retired_cleanup.get("verified") is not True:
                                    aborted = True
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
                        resources = capture.result() if failure is None else {}
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
                        unverified = resources.get("unverified_processes")
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
                            "sampler_method_version": SAMPLER_METHOD_VERSION,
                            "order": entry.position,
                            "arm_order": list(entry.order),
                            "oracle_ok": oracle.ok,
                            "oracle_detail": oracle.detail,
                            "warm_baseline_resources": baseline or None,
                            "warm_final_resources": resources or None,
                            "warm_peak_tree_rss_bytes": rss,
                            "warm_peak_child_footprint_bytes": resources.get("peak_child_footprint_bytes"),
                            "warm_task_tree_cpu_delta_seconds": cpu_delta,
                            "warm_cpu_delta_error": cpu_error,
                            "warm_unverified_processes": unverified,
                            "warm_observed_child_processes": resources.get("observed_child_processes"),
                            "warm_observed_browser_during_native_run": resources.get(
                                "observed_browser_during_native_run"
                            ),
                            "warm_read_errors": resources.get("read_errors"),
                            "warm_fatal_read_errors": resources.get("fatal_read_errors"),
                            "warm_footprint_status": resources.get("footprint_status"),
                            "warm_sampling_error": capture.error,
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
                        if aborted:
                            break
                    finally:
                        try:
                            fixture_stop(server, thread)
                        except Exception as error:
                            fixture_errors.append(
                                {
                                    "task": entry.task,
                                    "arm": entry.arm,
                                    "sample": entry.sample,
                                    "phase": "stop",
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
        finally:
            cleanup_reports: dict[str, dict] = {}
            for arm, session in sessions.items():
                with contextlib.suppress(Exception):
                    session.send({"type": "finish"})
                    session.receive(min(worker_timeout, FINISH_GRACE_SECONDS))
                with contextlib.suppress(Exception):
                    session.wait(FINISH_WAIT_SECONDS)
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
            binding_unchanged = before_binding == after_binding
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
                "sampler_method_version": SAMPLER_METHOD_VERSION,
                "observations": len(records),
                "expected_observations": expected_observations,
                "failures": failures,
                "warmups": warmups,
                "cleanup": cleanup_reports,
                "retired_workers": retired_workers,
                "fixture_errors": fixture_errors,
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
                and not fixture_errors
            )
            handle.write(json.dumps(terminal) + "\n")
            handle.flush()
    return records


def lane_valid(
    records: list[dict],
    terminal: dict,
    *,
    arms: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    samples: int | None = None,
) -> bool:
    """Strict gate: types, terminal fields, complete unique cohort, provenance and cleanup."""
    if not isinstance(records, list) or not isinstance(terminal, dict):
        return False
    if terminal.get("terminal") is not True or terminal.get("valid") is not True:
        return False
    failures = terminal.get("failures")
    if isinstance(failures, bool) or not isinstance(failures, int) or failures != 0:
        return False
    expected = terminal.get("expected_observations")
    observations = terminal.get("observations")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        return False
    if isinstance(observations, bool) or not isinstance(observations, int):
        return False
    if expected != len(records) or observations != len(records):
        return False
    if terminal.get("provenance_verified") is not True:
        return False
    if terminal.get("cleanup_verified") is not True:
        return False
    if terminal.get("aborted", False) or terminal.get("setup_error") is not None:
        return False
    if terminal.get("fixture_errors"):
        return False
    seen: set[tuple[str, str, int]] = set()
    for row in records:
        if not isinstance(row, dict):
            return False
        task = row.get("task")
        arm = row.get("arm")
        sample = row.get("sample")
        if not isinstance(task, str) or not task:
            return False
        if not isinstance(arm, str) or not arm:
            return False
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            return False
        key = (task, arm, sample)
        if key in seen:
            return False
        seen.add(key)
    plan_values = (arms, tasks, samples)
    if any(value is not None for value in plan_values):
        if arms is None or tasks is None or samples is None:
            return False
        if not all(isinstance(value, str) and value for value in (*arms, *tasks)):
            return False
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            return False
        planned = {(task, arm, sample) for task in tasks for arm in arms for sample in range(samples)}
        if planned != seen:
            return False
        if len(planned) != expected or len(planned) != len(records):
            return False
    return True


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
    return 0 if lane_valid(records, terminal, arms=args.arms, tasks=args.tasks, samples=args.samples) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
