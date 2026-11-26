"""Isolated interactive resource lane, using the unchanged native task oracles.

Resource observations include the common Python worker and task driver. Summed
RSS can count shared pages more than once. CPU is a sampled lower bound, not a
complete operating-system charge. Use the latency lane for timing comparisons.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from .arms import CANONICAL_ARMS

ROOT = Path(__file__).resolve().parents[2]
MODULE = "benchmarks.interactive.resource_compare"
LOGGER = logging.getLogger(__name__)
DEFAULT_WORKDIR = Path("/tmp") if os.name != "nt" else Path(tempfile.gettempdir())
SAMPLE_PREFIX = "r-"
SAMPLER_METHOD_VERSION = 3
MAX_READ_ERROR_DETAILS = 32

DEFAULT_ARMS = list(CANONICAL_ARMS)
METRICS = (
    "peak_tree_rss_bytes",
    "peak_worker_rss_bytes",
    "peak_child_tree_rss_bytes",
    "peak_child_footprint_bytes",
    "observed_tree_cpu_seconds",
    "observed_child_cpu_seconds",
)


@dataclass(frozen=True)
class ProcessReading:
    pid: int
    created_at: float
    rss_bytes: int
    cpu_seconds: float
    footprint_bytes: int | None
    name: str | None = None
    browser_main: bool = False

    @property
    def identity(self) -> tuple[int, float]:
        return self.pid, self.created_at


class ResourceLedger:
    """Accumulate only identity-fenced readings from one worker's cohort."""

    def __init__(self, root_identity: tuple[int, float], *, footprint_supported: bool):
        self.root_identity = root_identity
        self.footprint_supported = footprint_supported
        self._cpu: dict[tuple[int, float], float] = {}
        self._details: dict[tuple[int, float], dict] = {}
        self._peak_tree_rss = 0
        self._peak_worker_rss = 0
        self._peak_child_rss = 0
        self._peak_child_footprint = 0
        self._children: set[tuple[int, float]] = set()
        self._missing_footprint = False
        self.samples = 0
        self.read_errors = 0
        self.fatal_read_errors = 0
        self.expected_exit_races = 0
        self.read_error_types: dict[str, int] = {}
        self.read_error_details: list[dict] = []
        self.read_error_details_truncated = 0
        self.fatal_read_identities: set[tuple[int, float]] = set()

    def note_read_error(
        self,
        error: BaseException,
        *,
        identity: tuple[int, float],
        operation: str,
        phase: str,
        sample_index: int,
    ) -> bool:
        self.read_errors += 1
        name = type(error).__name__
        errno_value = getattr(error, "errno", None)
        if isinstance(errno_value, bool) or not isinstance(errno_value, int):
            errno_value = None
        self.read_error_types[name] = self.read_error_types.get(name, 0) + 1
        if len(self.read_error_details) < MAX_READ_ERROR_DETAILS:
            self.read_error_details.append(
                {
                    "pid": identity[0],
                    "created_at": identity[1],
                    "operation": operation,
                    "phase": phase,
                    "sample_index": sample_index,
                    "error_type": name,
                    "errno": errno_value,
                }
            )
        else:
            self.read_error_details_truncated += 1
        if isinstance(error, (psutil.NoSuchProcess, psutil.ZombieProcess)):
            self.expected_exit_races += 1
            return False
        self.fatal_read_errors += 1
        self.fatal_read_identities.add(identity)
        return True

    def observe(self, readings: list[ProcessReading], *, phase: str = "native_run") -> None:
        self.samples += 1
        unique = {reading.identity: reading for reading in readings}
        children = [item for key, item in unique.items() if key != self.root_identity]
        self._children.update(item.identity for item in children)
        for key, item in unique.items():
            self._cpu[key] = max(self._cpu.get(key, 0.0), item.cpu_seconds)
            previous = self._details.get(key, {})
            self._details[key] = {
                "pid": item.pid,
                "created_at": item.created_at,
                "name": item.name,
                "role": "worker" if key == self.root_identity else "child",
                "peak_rss_bytes": max(previous.get("peak_rss_bytes", 0), item.rss_bytes),
                "observed_cpu_seconds": self._cpu[key],
                "footprint_observed": previous.get("footprint_observed", False) or item.footprint_bytes is not None,
                "browser_main": previous.get("browser_main", False) or item.browser_main,
                "observed_during_native_run": previous.get("observed_during_native_run", False)
                or phase == "native_run",
            }
        self._peak_tree_rss = max(self._peak_tree_rss, sum(item.rss_bytes for item in unique.values()))
        self._peak_child_rss = max(self._peak_child_rss, sum(item.rss_bytes for item in children))
        root = unique.get(self.root_identity)
        if root is not None:
            self._peak_worker_rss = max(self._peak_worker_rss, root.rss_bytes)
        if any(item.footprint_bytes is None for item in children):
            self._missing_footprint = True
        self._peak_child_footprint = max(
            self._peak_child_footprint,
            sum(item.footprint_bytes or 0 for item in children),
        )

    def result(self) -> dict:
        footprint_status = "unsupported"
        if self.footprint_supported:
            footprint_status = "unavailable" if self._missing_footprint or not self._children else "observed"
        return {
            "samples": self.samples,
            "observed_processes": len(self._cpu),
            "observed_child_processes": len(self._children),
            "observed_browser_processes": sum(
                item["browser_main"] for item in self._details.values() if item["role"] == "child"
            ),
            "observed_browser_during_native_run": sum(
                item["browser_main"] and item["observed_during_native_run"]
                for item in self._details.values()
                if item["role"] == "child"
            ),
            "processes": [item for _, item in sorted(self._details.items())],
            "peak_tree_rss_bytes": self._peak_tree_rss if self._cpu else None,
            "peak_worker_rss_bytes": self._peak_worker_rss if self.root_identity in self._cpu else None,
            "peak_child_tree_rss_bytes": self._peak_child_rss if self._children else None,
            "observed_tree_cpu_seconds": sum(self._cpu.values()) if self._cpu else None,
            "observed_child_cpu_seconds": sum(value for key, value in self._cpu.items() if key in self._children)
            if self._children
            else None,
            "peak_child_footprint_bytes": self._peak_child_footprint if footprint_status == "observed" else None,
            "footprint_status": footprint_status,
            "cpu_accounting": "sampled_lower_bound",
            "read_errors": self.read_errors,
            "fatal_read_errors": self.fatal_read_errors,
            "expected_exit_races": self.expected_exit_races,
            "read_error_types": dict(sorted(self.read_error_types.items())),
            "sampler_method_version": SAMPLER_METHOD_VERSION,
            "read_error_details": list(self.read_error_details),
            "read_error_details_truncated": self.read_error_details_truncated,
            "fatal_read_identities": [
                {"pid": pid, "created_at": created_at} for pid, created_at in sorted(self.fatal_read_identities)
            ],
            "rss_accounting": "sampled_sum_includes_worker_shared_pages_may_repeat",
        }


class RUsageInfoV0(ctypes.Structure):
    """The stable rusage_info_v0 ABI in Apple's sys/resource.h."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
            )
        ],
    ]


def make_footprint_reader(*, platform_name: str | None = None) -> Callable[[int], int | None] | None:
    if (platform_name or sys.platform) != "darwin":
        return None
    try:
        function = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pid_rusage
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
    except (OSError, AttributeError):
        return None

    def read(pid: int) -> int | None:
        information = RUsageInfoV0()
        if function(pid, 0, ctypes.byref(information)) != 0:
            return None
        return int(information.ri_phys_footprint)

    return read


class ProcessTreeSampler:
    """Retain owned descendants after reparenting without following reused PIDs."""

    def __init__(
        self,
        root_pid: int,
        *,
        process_factory: Callable[[int], Any] | None = None,
        footprint_reader: Callable[[int], int | None] | None = None,
        footprint_supported: bool | None = None,
        ownership_marker: str | None = None,
    ):
        if ownership_marker is not None and (len(ownership_marker) < 12 or not Path(ownership_marker).is_absolute()):
            raise ValueError("The ownership marker must be a private absolute run path")
        self._ownership_marker = ownership_marker
        self._marker_unverified: set[int] = set()
        self._process = process_factory or psutil.Process
        self._footprint = footprint_reader if footprint_reader is not None else make_footprint_reader()
        try:
            root_identity = root_pid, self._process(root_pid).create_time()
        except psutil.Error:
            root_identity = root_pid, -1.0
        self._known = {root_identity}
        self._unverified: set[tuple[int, float]] = set()
        self._ledger = ResourceLedger(
            root_identity,
            footprint_supported=self._footprint is not None if footprint_supported is None else footprint_supported,
        )

    def discover_orphans(self) -> None:
        """Recover visible profile-marked children that escaped ancestry sampling."""
        if self._ownership_marker is None:
            return
        self._marker_unverified.clear()
        for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                command = process.info.get("cmdline") or []
                if process.pid == os.getpid() or not any(self._ownership_marker in value for value in command):
                    continue
                created = process.info.get("create_time")
                if created is None:
                    created = process.create_time()
                self._known.add((process.pid, created))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied:
                self._marker_unverified.add(process.pid)

    def live_processes(self, *, phase: str = "native_run") -> dict[tuple[int, float], Any]:
        self._unverified.clear()
        live: dict[tuple[int, float], Any] = {}
        queue = list(self._known)
        checked: set[tuple[int, float]] = set()
        while queue:
            identity = queue.pop()
            if identity in checked:
                continue
            checked.add(identity)
            operation = "process_lookup"
            try:
                process = self._process(identity[0])
                operation = "is_running"
                if not process.is_running():
                    continue
                operation = "create_time"
                if process.create_time() != identity[1]:
                    continue
                operation = "status"
                if process.status() == psutil.STATUS_ZOMBIE:
                    continue
                operation = "children"
                children = process.children(recursive=True)
                live[identity] = process
                for child in children:
                    try:
                        child_identity = child.pid, child.create_time()
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        continue
                    except (psutil.Error, OSError) as error:
                        child_identity = (child.pid, -1.0)
                        self._ledger.note_read_error(
                            error,
                            identity=child_identity,
                            operation="child_identity",
                            phase=phase,
                            sample_index=self._ledger.samples,
                        )
                        self._unverified.add(child_identity)
                        continue
                    self._known.add(child_identity)
                    queue.append(child_identity)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.Error, OSError) as error:
                self._ledger.note_read_error(
                    error,
                    identity=identity,
                    operation=operation,
                    phase=phase,
                    sample_index=self._ledger.samples,
                )
                self._unverified.add(identity)
        return live

    def sample(self, *, phase: str = "native_run") -> None:
        readings = []
        for identity, process in self.live_processes(phase=phase).items():
            operation = "memory_info"
            try:
                rss = process.memory_info().rss
                operation = "cpu_times"
                cpu = process.cpu_times()
                operation = "name"
                name = process.name()
                lowered = name.lower()
                browser_like = (
                    any(token in lowered for token in ("chrome", "chromium", "headless_shell", "brave", "msedge"))
                    and "helper" not in lowered
                    and "crashpad" not in lowered
                )
                command: list[str] = []
                if browser_like:
                    operation = "cmdline"
                    command = process.cmdline()
                browser_main = browser_like and not any(argument.startswith("--type=") for argument in command)
                operation = "footprint"
                footprint = self._footprint(process.pid) if self._footprint is not None else None
                operation = "identity_recheck"
                if not process.is_running() or process.create_time() != identity[1]:
                    continue
                readings.append(
                    ProcessReading(process.pid, identity[1], rss, cpu.user + cpu.system, footprint, name, browser_main)
                )
            except (psutil.Error, OSError) as error:
                self._ledger.note_read_error(
                    error,
                    identity=identity,
                    operation=operation,
                    phase=phase,
                    sample_index=self._ledger.samples,
                )
        self._ledger.observe(readings, phase=phase)

    def result(self) -> dict:
        live_unknown = {pid for pid, _ in self._unverified} | self._marker_unverified
        observation_unknown = live_unknown | {pid for pid, _ in self._ledger.fatal_read_identities}
        return {
            **self._ledger.result(),
            "unverified_processes": len(observation_unknown),
            "cleanup_unverified_processes": len(live_unknown),
        }


def sample_plan(arms: list[str], tasks: list[str], samples: int):
    if not arms or not tasks or samples < 1:
        raise ValueError("At least one arm, task, and sample is required")
    for sample in range(samples):
        shift = sample % len(arms)
        order = arms[shift:] + arms[:shift]
        for task in tasks:
            for position, arm in enumerate(order):
                yield sample, task, arm, position, order


def summarize_resources(records: list[dict], *, provenance_verified: bool = False) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["task"], record["arm"]), []).append(record)
    result = []
    for (task, arm), items in sorted(groups.items()):
        correct = [item for item in items if item["ok"]]
        eligible = [item for item in correct if item.get("resource_ok", True) and provenance_verified]
        metrics = {}
        for name in METRICS:
            values = [item.get("resources", {}).get(name) for item in eligible]
            available = [value for value in values if value is not None]
            metrics[name] = {
                "count": len(available),
                "median": statistics.median(available) if available else None,
                "max": max(available) if available else None,
            }
        result.append(
            {
                "task": task,
                "arm": arm,
                "runs": len(items),
                "successes": len(correct),
                "failures": len(items) - len(correct),
                "all_correct": len(correct) == len(items),
                "resource_eligible": len(eligible),
                "provenance_verified": provenance_verified,
                "metrics": metrics,
                "errors": [item.get("error", "") for item in items if not item["ok"]],
            }
        )
    return result


def source_manifest() -> dict:
    from .resource_provenance import application_digest

    digest = hashlib.sha256()
    for path in sorted((ROOT / "benchmarks" / "interactive").rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.suffix not in {".py", ".json"} or "results" in relative.parts:
            continue
        digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    harness_digest = digest.hexdigest()
    app_digest = application_digest(ROOT / "src" / "openbrowser")
    configuration = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        path = ROOT / name
        if path.is_file():
            configuration.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    source_digest = hashlib.sha256((harness_digest + app_digest + configuration.hexdigest()).encode()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_digest": source_digest,
        "harness_digest": harness_digest,
        "expected_app_digest": app_digest,
        "configuration_digest": configuration.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "psutil": psutil.__version__,
        "mcp_transport": os.environ.get("OPENBROWSER_MCP_TRANSPORT", "package default"),
        "method": "fresh worker plus owned descendants, through native runner cleanup",
        "sampler_method_version": SAMPLER_METHOD_VERSION,
        "limitations": [
            "Includes common Python worker and task-driver overhead.",
            "Summed RSS can double-count shared pages.",
            "Sampling can miss short-lived processes and final CPU increments.",
            "Expected process-exit read races are counted separately; access and other read failures remain fatal.",
            "Child footprint is null when unsupported or unavailable.",
            "Resource-instrumented wall time is not a latency benchmark.",
        ],
    }


def binding_snapshot(arms: list[str]) -> dict:
    from .resource_provenance import bind_openbrowser

    if not any(name.startswith("openbrowser") for name in arms):
        return {"verified": True, "reason": "openbrowser_not_selected"}
    binary = os.environ.get("OPENBROWSER_BIN")
    if not binary:
        return {"verified": False, "reason": "entry_not_configured"}
    from .adapters.openbrowser import _env
    from .run_benchmark import make_context

    with tempfile.TemporaryDirectory(prefix="ob-resource-binding-") as directory:
        context = make_context("provenance", "openbrowser-fast", 0, "http://127.0.0.1:0", Path(directory))
        return bind_openbrowser(binary, ROOT / "src" / "openbrowser", env=_env(context), cwd=ROOT)


def arm_versions(arms: list[str]) -> dict:
    from .run_benchmark import make_arm

    versions = {}
    for name in arms:
        try:
            arm = make_arm(name)
            binary = arm.binary
            command = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15, check=False)
            versions[name] = {"version": command.stdout.strip() if command.returncode == 0 else None}
        except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired) as error:
            versions[name] = {"version": None, "error": type(error).__name__}
    return versions


def run_worker(task: str, arm: str, sample: int, workdir: Path, output: Path) -> None:
    from .run_benchmark import run_one

    record = run_one(task, arm, sample, workdir, False)
    output.write_text(json.dumps(record) + "\n")


def observation_ok(resources: dict, cleanup: dict) -> bool:
    """Return whether the resource observation is comparable and cleanup is verified."""
    return bool(
        resources.get("peak_tree_rss_bytes") is not None
        and resources.get("fatal_read_errors") == 0
        and resources.get("observed_child_processes", 0) > 0
        and resources.get("observed_browser_during_native_run", 0) > 0
        and cleanup["verified"]
    )


def cleanup_owned(sampler: ProcessTreeSampler, *, interval: float = 0.02, timeout: float = 4.0) -> dict:
    started = time.monotonic()
    deadline = started + timeout
    terminate_sent: set[tuple[int, float]] = set()
    kill_sent: set[tuple[int, float]] = set()
    errors = []

    def cleanup_uncertainty() -> int:
        result = sampler.result()
        value = result.get("cleanup_unverified_processes")
        if value is None:
            value = result.get("unverified_processes", 0)
        return int(value)

    sampler.discover_orphans()
    live = sampler.live_processes(phase="cleanup")
    initial = len(live)
    while live or cleanup_uncertainty():
        now = time.monotonic()
        for identity, process in live.items():
            try:
                if identity not in terminate_sent:
                    terminate_sent.add(identity)
                    process.terminate()
                if now >= started + timeout / 2 and identity not in kill_sent:
                    kill_sent.add(identity)
                    process.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.Error as error:
                if len(errors) < 64:
                    errors.append(type(error).__name__)
        sampler.sample(phase="cleanup")
        sampler.discover_orphans()
        live = sampler.live_processes(phase="cleanup")
        remaining = deadline - time.monotonic()
        if not live and not cleanup_uncertainty():
            break
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    sampler.discover_orphans()
    sampler.sample(phase="cleanup")
    live = sampler.live_processes(phase="cleanup")
    uncertain = cleanup_uncertainty()
    return {
        "initial_leftovers": initial,
        "remaining": [{"pid": pid, "created_at": created} for pid, created in sorted(live)],
        "unverified_processes": uncertain,
        "errors": errors,
        "verified": not live and uncertain == 0,
        "scope": "observed_identity_fenced_cohort",
    }


def worker_diagnostics(root: Path) -> dict:
    result = {}
    for name in ("stdout", "stderr"):
        path = root / f"{name}.log"
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                handle.seek(max(0, size - 4096))
                tail = handle.read(4096).decode(errors="replace")
            tail = tail.replace(str(ROOT), "working copy").replace(str(Path.home()), "$HOME")
            tail = re.sub(r"\.worktrees[/\\][^/\\\s]+", "working copy", tail)
            result[name] = {"bytes": size, "tail": tail}
        except OSError as error:
            result[name] = {"error": type(error).__name__}
    return result


def worker_command(root: Path, task: str, arm: str, sample: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        MODULE,
        "--worker",
        "--task",
        task,
        "--arm",
        arm,
        "--sample",
        str(sample),
        "--workdir",
        str(root),
        "--worker-output",
        str(root / "record.json"),
    ]


def run_sample(task: str, arm: str, sample: int, *, workdir: Path, interval: float, worker_timeout: float) -> dict:
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=SAMPLE_PREFIX, dir=workdir) as directory:
        root = Path(directory)
        destination = root / "record.json"
        command = worker_command(root, task, arm, sample)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
        worker = None
        sampler = None
        timed_out = False
        failure = None
        cleanup = {"verified": True, "initial_leftovers": 0, "remaining": [], "errors": []}
        started = time.monotonic()
        with (root / "stdout.log").open("w") as stdout, (root / "stderr.log").open("w") as stderr:
            try:
                worker = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=os.name != "nt"
                )
                sampler = ProcessTreeSampler(worker.pid, ownership_marker=str(root))
                sampler.sample()
                while worker.poll() is None:
                    remaining = worker_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        timed_out = True
                        worker.kill()
                        break
                    time.sleep(min(interval, remaining))
                    sampler.sample()
                worker.wait(timeout=5)
            except Exception as error:
                failure = type(error).__name__
            finally:
                try:
                    if worker is not None and worker.poll() is None:
                        worker.kill()
                        worker.wait(timeout=5)
                    if sampler is not None:
                        cleanup = cleanup_owned(sampler, interval=interval)
                    elif worker is not None:
                        cleanup = {"verified": False, "remaining": [], "errors": ["sampler_unavailable"]}
                except Exception as error:
                    cleanup = {"verified": False, "remaining": [], "errors": [type(error).__name__]}
            elapsed = time.monotonic() - started
        try:
            record = json.loads(destination.read_text())
            if not isinstance(record, dict) or record.get("task") != task or record.get("arm") != arm:
                raise ValueError("Worker returned a different task or arm")
        except (OSError, ValueError) as error:
            record = {
                "task": task,
                "arm": arm,
                "sample": sample,
                "ok": False,
                "oracle_ok": False,
                "adapter_ok": False,
                "error": f"Worker record unavailable: {type(error).__name__}",
            }
        if failure or timed_out or worker is None or worker.returncode != 0:
            record["ok"] = False
            record["error"] = (
                f"Parent supervision failed: {failure}"
                if failure
                else "Worker exceeded its deadline"
                if timed_out
                else f"Worker exited with code {worker.returncode if worker else 'unavailable'}"
            )
        resources = sampler.result() if sampler is not None else {}
        record.update(
            {
                "resources": resources,
                "resource_window_seconds": elapsed,
                "resource_ok": observation_ok(resources, cleanup),
                "cleanup": cleanup,
                "diagnostics": worker_diagnostics(root),
                "worker_timed_out": timed_out,
                "sampling_interval_seconds": interval,
            }
        )
        return record


def run_lane(
    arms: list[str],
    tasks: list[str],
    samples: int,
    output: Path,
    *,
    workdir: Path,
    interval: float,
    worker_timeout: float,
) -> list[dict]:
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        before_source = source_manifest()
        before_binding = binding_snapshot(arms)
        expected_observations = len(arms) * len(tasks) * samples
        metadata = {
            "record_type": "run",
            **before_source,
            "binding_before": before_binding,
            "conclusion_policy": "Samples are provisional until the end record verifies unchanged source and binding.",
            "arms": arms,
            "tasks": tasks,
            "samples": samples,
            "expected_observations": expected_observations,
            "interval_seconds": interval,
            "worker_timeout_seconds": worker_timeout,
            "arm_versions": arm_versions(arms),
        }
        handle.write(json.dumps(metadata) + "\n")
        handle.flush()
        aborted = False
        for sample, task, arm, position, order in sample_plan(arms, tasks, samples):
            try:
                record = run_sample(
                    task, arm, sample, workdir=workdir, interval=interval, worker_timeout=worker_timeout
                )
            except Exception as error:
                record = {
                    "task": task,
                    "arm": arm,
                    "sample": sample,
                    "ok": False,
                    "resource_ok": False,
                    "error": f"Parent observation failed: {type(error).__name__}",
                    "resources": {},
                    "cleanup": {"verified": False, "errors": [type(error).__name__]},
                }
            record.update({"record_type": "sample", "order": position, "arm_order": order, "provisional": True})
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            LOGGER.info(
                "%s %s sample=%s correct=%s resource_ok=%s", task, arm, sample, record["ok"], record.get("resource_ok")
            )
            if not record.get("cleanup", {"verified": True})["verified"]:
                aborted = True
                break
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
            before_binding.get("verified") and after_binding.get("verified") and source_unchanged and binding_unchanged
        )
        for record in records:
            record["provenance_verified"] = verified
        cleanup_verified = bool(records) and all(
            isinstance(item.get("cleanup"), dict)
            and item["cleanup"].get("verified") is True
            and not item["cleanup"].get("remaining")
            and item["cleanup"].get("unverified_processes") == 0
            and not item["cleanup"].get("errors")
            and item.get("worker_timed_out") is False
            for item in records
        )
        valid = bool(
            not aborted
            and len(records) == expected_observations
            and cleanup_verified
            and verified
            and all(
                item["ok"] and item.get("resource_ok", False) and item.get("provenance_verified", False)
                for item in records
            )
        )
        terminal = {
            "record_type": "end",
            "observations": len(records),
            "expected_observations": expected_observations,
            "failures": sum(not item["ok"] or not item.get("resource_ok", True) for item in records),
            "provenance_verified": verified,
            "source_unchanged": source_unchanged,
            "binding_unchanged": binding_unchanged,
            "binding_after": after_binding,
            "cleanup_verified": cleanup_verified,
            "fixture_errors": [],
            "valid": valid,
            "terminal": True,
        }
        if aborted:
            terminal.update({"aborted": True, "abort_reason": "cleanup_unverified"})
        handle.write(json.dumps(terminal) + "\n")
    return records


def main() -> int:
    from .tasks import TASKS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--worker-timeout", type=float, default=180)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task", help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--sample", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.task or not args.arm or args.worker_output is None:
            parser.error("Worker mode needs task, arm, and worker-output")
        run_worker(args.task, args.arm, args.sample, args.workdir, args.worker_output)
        return 0
    if (
        args.output is None
        or args.samples < 1
        or not math.isfinite(args.interval)
        or not math.isfinite(args.worker_timeout)
        or args.interval <= 0
        or args.worker_timeout <= 0
    ):
        parser.error("Provide --output, positive samples, and finite positive interval and worker-timeout")
    if any(arm.startswith("openbrowser") for arm in args.arms) and not os.environ.get("OPENBROWSER_BIN"):
        parser.error("Set OPENBROWSER_BIN to the exact build under test")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    records = run_lane(
        args.arms,
        args.tasks,
        args.samples,
        args.output,
        workdir=args.workdir,
        interval=args.interval,
        worker_timeout=args.worker_timeout,
    )
    with args.output.with_suffix(".summary.json").open("x") as handle:
        verified = bool(records) and all(item.get("provenance_verified", False) for item in records)
        json.dump(summarize_resources(records, provenance_verified=verified), handle, indent=2)
        handle.write("\n")
    return int(
        not records
        or any(
            not item["ok"] or not item.get("resource_ok", False) or not item.get("provenance_verified", False)
            for item in records
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
