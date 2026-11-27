"""Warm-session process-tree resources for the seven interactive paths.

Each arm keeps one native session for the whole lane. Warmup runs once,
outside every capture window. Every task gets a fresh fixture server, a
baseline process-tree sample, the task, a final sample, and the unchanged
oracle. Metrics are warm peak tree RSS, warm peak child footprint, and the
sampled task CPU delta. Unknown values stay null. No latency claim is made.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import psutil

from .adapters.base import ArmResult, RunContext
from .adapters.native_service_readers import (
    HostReaders,
    NativeArm,
    NativeRunScope,
    NativeServiceRequest,
    capture_native_service,
    verify_native_service,
)
from .adapters.native_service_sources import PidRecord, ProcessCommand
from .adapters.service_roots import ServiceIdentity, ServiceSource
from .engine_preflight import configured_engine_pin
from .engine_session import EngineSession
from .resource_compare import (
    NATIVE_SERVICE_METHOD_VERSION,
    SAMPLER_METHOD_VERSION,
    ProcessTreeSampler,
    arm_versions,
    binding_snapshot,
    cleanup_owned,
    source_manifest,
)
from .resource_provenance import _launcher_interpreter
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
METHOD = "warm-session-resources-v3"
FINISH_GRACE_SECONDS = 5.0
FINISH_WAIT_SECONDS = 5.0
WORKER_GATE_TIMEOUT_SECONDS = 120.0
HERE = Path(__file__).resolve().parent
_MCP_ARMS = frozenset({"openbrowser-mcp-fast", "playwright-mcp", "chrome-devtools-mcp"})
_FILE_SERVICE_ARMS = frozenset({"openbrowser-fast", "agent-browser", "browser-use"})
_OPENBROWSER_ARMS = frozenset({"openbrowser-fast", "openbrowser-mcp-fast"})
_NODE_ARMS = frozenset({"playwright-mcp", "playwright-cli", "chrome-devtools-mcp"})


def _valid_service_time(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value)) and value > 0
    except (OverflowError, ValueError):
        return False


def _absolute_binary(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("native_service_binary")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("native_service_binary")
    return path


def _expected_service_executable(arm: str, binary: object) -> str:
    """Derive the native executable from the declared launcher, not the PID."""
    path = _absolute_binary(binary)
    if arm in _OPENBROWSER_ARMS:
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError("native_service_binary") from error
        interpreter = _launcher_interpreter(content)
        if interpreter is not None:
            return str(Path(interpreter).resolve())
        backend_name = "openbrowser-ai-py.exe" if os.name == "nt" else "openbrowser-ai-py"
        backend = path.resolve().parent / backend_name
        try:
            backend_content = backend.read_bytes()
        except OSError as error:
            raise ValueError("native_service_backend") from error
        interpreter = _launcher_interpreter(backend_content)
        if interpreter is None:
            raise ValueError("native_service_backend")
        return str(Path(interpreter).resolve())
    if arm == "browser-use":
        try:
            first_line = path.read_bytes().splitlines()[0]
        except (IndexError, OSError) as error:
            raise ValueError("native_service_binary") from error
        if not first_line.startswith(b"#!"):
            raise ValueError("native_service_binary")
        try:
            interpreter = first_line[2:].decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ValueError("native_service_binary") from error
        if not interpreter:
            raise ValueError("native_service_binary")
        interpreter_path = Path(interpreter)
        if not interpreter_path.is_absolute() or not interpreter_path.is_file():
            raise ValueError("native_service_binary")
        if not os.access(interpreter_path, os.X_OK):
            raise ValueError("native_service_binary")
        return str(interpreter_path.resolve())
    if arm in _NODE_ARMS:
        node = shutil.which("node", path=os.environ.get("PATH", os.defpath))
        if not node:
            raise ValueError("native_service_node")
        return str(Path(node).resolve())
    if arm == "agent-browser":
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError("native_service_binary")
        return str(resolved)
    raise ValueError("native_service_arm")


def _playwright_cli_daemon_entry(binary: object) -> str:
    entry = _absolute_binary(binary).resolve()
    daemon = entry.parent.parent.parent / "playwright-core" / "lib" / "entry" / "cliDaemon.js"
    if not daemon.is_file():
        raise ValueError("native_service_daemon_entry")
    return str(daemon.resolve())


def _native_run_scope(root: str, session: object) -> NativeRunScope:
    base = Path(root)
    session_value = session if isinstance(session, str) else ""
    return NativeRunScope(
        root=base,
        home=base / "home",
        config=base / "config",
        profile=base / "profile",
        session=session_value,
    )


def _native_request(arm: str, ready: dict, scope: NativeRunScope) -> NativeServiceRequest:
    try:
        arm_value = NativeArm(arm)
    except ValueError as error:
        raise ValueError("native_service_arm") from error
    executable = _expected_service_executable(arm, ready.get("binary"))
    native_identity = None
    if arm not in _FILE_SERVICE_ARMS:
        native = ready.get("native")
        if not isinstance(native, dict):
            raise ValueError("native_service_source")
        pid = native.get("pid")
        created_at = native.get("created_at")
        if type(pid) is not int or pid <= 0 or not _valid_service_time(created_at):
            raise ValueError("native_service_source")
        assert type(created_at) is int or type(created_at) is float
        native_identity = ServiceIdentity(pid, created_at)
    daemon_entry = _playwright_cli_daemon_entry(ready.get("binary")) if arm == "playwright-cli" else None
    return NativeServiceRequest(
        arm=arm_value,
        scope=scope,
        executable=executable,
        native_identity=native_identity,
        daemon_entry=daemon_entry,
    )


def _public_service_read(read: Any) -> dict:
    try:
        eligible = read.eligible is True
        failure = read.failure.value if read.failure is not None else None
        proof = read.proof.proof if eligible else None
    except Exception:
        return {"eligible": False, "failure": "malformed_proof", "proof": None}
    if proof is None:
        return {"eligible": False, "failure": failure or "invalid_proof", "proof": None}
    return {
        "eligible": True,
        "failure": None,
        "proof": {
            "pid": proof.identity.pid,
            "created_at": proof.identity.created_at,
            "source": proof.source.value,
            "executable_basename": Path(proof.executable).name,
            "scope_digest": hashlib.sha256(os.fsencode(proof.scope)).hexdigest(),
        },
    }


def _request_evidence(request: NativeServiceRequest) -> dict:
    return {
        "arm": request.arm.value,
        "scope": {
            "root": str(request.scope.root),
            "home": str(request.scope.home),
            "config": str(request.scope.config),
            "profile": str(request.scope.profile),
            "session": request.scope.session,
        },
        "executable": request.executable,
        "native_identity": (
            None
            if request.native_identity is None
            else {
                "pid": request.native_identity.pid,
                "created_at": request.native_identity.created_at,
            }
        ),
        "daemon_entry": request.daemon_entry,
    }


def _candidate_evidence(candidate: object) -> dict | None:
    if candidate is None:
        return None
    if isinstance(candidate, PidRecord):
        return {
            "type": "pid_record",
            "pid": candidate.pid,
            "record_format": candidate.record_format,
            "started": candidate.started,
        }
    if isinstance(candidate, ProcessCommand):
        return {
            "type": "process_command",
            "pid": candidate.pid,
            "executable": candidate.executable,
            "argv": list(candidate.argv),
        }
    return {"type": type(candidate).__name__}


def _source_read_evidence(read: Any) -> dict | None:
    if read is None:
        return None
    return {
        "source": read.source.value if read.source is not None else None,
        "candidate": _candidate_evidence(read.candidate),
        "record_bytes_base64": (
            base64.b64encode(read.record_bytes).decode("ascii") if read.record_bytes is not None else None
        ),
        "failure": read.failure.value if read.failure is not None else None,
    }


def _proof_result_evidence(result: Any) -> dict | None:
    if result is None:
        return None
    proof = result.proof
    return {
        "failure": result.failure.value if result.failure is not None else None,
        "proof": (
            None
            if proof is None
            else {
                "identity": {"pid": proof.identity.pid, "created_at": proof.identity.created_at},
                "executable": proof.executable,
                "scope": proof.scope,
                "source": proof.source.value,
            }
        ),
    }


def _service_read_evidence(read: Any) -> dict:
    return {
        "eligible": read.eligible is True,
        "failure": read.failure.value if read.failure is not None else None,
        "request": _request_evidence(read.before.request),
        "before": _source_read_evidence(read.before),
        "after": _source_read_evidence(read.after),
        "initial_proof": _proof_result_evidence(read.initial_proof),
        "closing_proof": _proof_result_evidence(read.closing_proof),
        "effective_proof": _proof_result_evidence(read.proof),
    }


def _write_private_evidence(directory: Path, name: str, payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _worker_native_source(arm_name: str, arm: Any) -> dict:
    """Serialize the worker-owned native source without crossing a Popen object."""
    binary = getattr(arm, "binary", None)
    if not isinstance(binary, str) or not binary:
        return {"kind": "unsupported", "binary": None}
    if arm_name in _MCP_ARMS:
        client = getattr(arm, "_client", None)
        process = getattr(client, "process", None)
        pid = getattr(process, "pid", None)
        if type(pid) is not int or pid <= 0:
            raise ValueError("native_service_pid")
        created_at = psutil.Process(pid).create_time()
        if not _valid_service_time(created_at):
            raise ValueError("native_service_create_time")
        return {"kind": "popen", "binary": binary, "pid": pid, "created_at": created_at}
    if arm_name == "playwright-cli":
        pid = getattr(arm, "session_pid", None)
        if type(pid) is not int or pid <= 0:
            raise ValueError("native_service_pid")
        created_at = psutil.Process(pid).create_time()
        if not _valid_service_time(created_at):
            raise ValueError("native_service_create_time")
        return {"kind": "native_status", "binary": binary, "pid": pid, "created_at": created_at}
    return {"kind": "pid_file", "binary": binary}


def _fresh_identity(pid: int) -> ServiceIdentity:
    if type(pid) is not int or pid <= 0:
        raise ValueError("native_service_identity")
    process = psutil.Process(pid)
    if type(process.pid) is not int or process.pid != pid:
        raise ValueError("native_service_identity")
    created_at = process.create_time()
    if not _valid_service_time(created_at):
        raise ValueError("native_service_identity")
    return ServiceIdentity(pid, created_at)


def _session_identity(session: Any) -> ServiceIdentity:
    return _fresh_identity(session.pid)


class _WarmNativeRoot:
    def __init__(
        self,
        arm: str,
        ready: dict,
        scope: NativeRunScope,
        controller: ServiceIdentity,
        worker: ServiceIdentity,
        *,
        host: HostReaders | None = None,
    ) -> None:
        self.request = _native_request(arm, ready, scope)
        self.host = host or HostReaders()
        self.controller = controller
        self.worker = worker
        self.ready_read = capture_native_service(
            self.request,
            self.host,
            controller=controller,
            worker=worker,
        )

    @property
    def ready_eligible(self) -> bool:
        return self.ready_read.eligible

    def verify_before(self):
        return verify_native_service(
            self.ready_read,
            self.request,
            self.host,
            controller=self.controller,
            worker=self.worker,
        )

    def verify_after(self, before):
        return verify_native_service(
            before,
            self.request,
            self.host,
            controller=self.controller,
            worker=self.worker,
        )

    def public_ready(self) -> dict:
        return _public_service_read(self.ready_read)

    def public_before(self, read) -> dict:
        return _public_service_read(read)

    def public_after(self, read) -> dict:
        return _public_service_read(read)

    def evidence_ready(self) -> dict:
        return _service_read_evidence(self.ready_read)

    def evidence_before(self, read) -> dict:
        return _service_read_evidence(read)

    def evidence_after(self, read) -> dict:
        return _service_read_evidence(read)


def _default_native_root(
    arm: str,
    ready: dict,
    scope: NativeRunScope,
    controller: ServiceIdentity,
    worker: ServiceIdentity,
) -> _WarmNativeRoot:
    return _WarmNativeRoot(arm, ready, scope, controller, worker)


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
            native_source = _worker_native_source(arm_name, arm)
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
                "session": root_ctx.session,
                "native": native_source,
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


def _default_sampler(
    pid: int,
    marker: str,
    *,
    controller_identity: ServiceIdentity | None = None,
) -> ProcessTreeSampler:
    if controller_identity is None:
        return ProcessTreeSampler(pid, ownership_marker=marker)
    return ProcessTreeSampler(pid, ownership_marker=marker, controller_identity=controller_identity)


def _make_owned_sampler(
    pid: int,
    marker: str,
    factory: Callable[[int, str], Any],
    controller_identity: ServiceIdentity | None = None,
):
    def build():
        if controller_identity is not None and factory is _default_sampler:
            return _default_sampler(pid, marker, controller_identity=controller_identity)
        sampler = factory(pid, marker)
        if controller_identity is not None:
            setter = getattr(sampler, "set_native_controller", None)
            if callable(setter):
                setter(controller_identity)
        return sampler

    try:
        return build()
    except Exception:
        if factory is not _default_sampler:
            raise
        LOGGER.warning("sampler construction failed; retrying the default sampler once")
        return build()


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
    native_root_factory: Callable[..., Any] = _default_native_root,
    controller_identity: ServiceIdentity | None = None,
    worker_identity_factory: Callable[[Any], ServiceIdentity] | None = None,
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
            "native_service_method_version": NATIVE_SERVICE_METHOD_VERSION,
            "arms": arms,
            "tasks": tasks,
            "samples": samples,
            "interval_seconds": interval,
            "worker_timeout_seconds": worker_timeout,
            "expected_observations": expected_observations,
            "conclusion_policy": "Warm resource observations only. No latency claim.",
            "latency_claim_allowed": False,
        }
        controller: ServiceIdentity | None = None
        worker_identity_for = worker_identity_factory or _session_identity
        try:
            controller = controller_identity or _fresh_identity(os.getpid())
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
            header["native_service_method_version"] = NATIVE_SERVICE_METHOD_VERSION
        except Exception as error:
            header["preflight_error"] = type(error).__name__
            handle.write(json.dumps(header) + "\n")
            handle.write(
                json.dumps(
                    {
                        "record_type": "end",
                        "method": METHOD,
                        "sampler_method_version": SAMPLER_METHOD_VERSION,
                        "native_service_method_version": NATIVE_SERVICE_METHOD_VERSION,
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
        assert controller is not None
        handle.write(json.dumps(header) + "\n")
        handle.flush()
        sessions: dict[str, Any] = {}
        warmups: dict[str, dict] = {}
        lane_samplers: dict[str, Any] = {}
        native_roots: dict[str, Any] = {}
        owned_scopes: dict[str, str] = {}
        native_root_pending: set[str] = set(arms)
        native_ready_failures: dict[str, dict] = {}
        retired_arms: set[str] = set()
        retired_workers: dict[str, dict] = {}
        setup_error: str | None = None
        aborted = False
        try:
            try:
                for arm in arms:
                    owned = (Path(workdir).resolve() / f"warm-{uuid.uuid4().hex[:10]}").resolve()
                    owned.mkdir(parents=True, exist_ok=True)
                    owned_scopes[arm] = str(owned)
                    session = session_factory(arm, owned, diagnostics)
                    sessions[arm] = session
                    lane_samplers[arm] = _make_owned_sampler(
                        session.pid,
                        str(owned),
                        sampler_factory,
                        controller,
                    )
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
                    worker_identity = worker_identity_for(session)
                    scope = _native_run_scope(root, ready.get("session"))
                    native_root = native_root_factory(
                        arm,
                        ready,
                        scope,
                        controller,
                        worker_identity,
                    )
                    ready_evidence_name = f"native-ready-{arm}.json"
                    ready_evidence_sha256 = _write_private_evidence(
                        diagnostics,
                        ready_evidence_name,
                        native_root.evidence_ready(),
                    )
                    if not native_root.ready_eligible:
                        native_ready_failures[arm] = {
                            "evidence_sha256": ready_evidence_sha256,
                            "evidence_artifact": ready_evidence_name,
                        }
                        raise ValueError("native_service_ready")
                    lane_samplers[arm].enroll_native_service(native_root.ready_read, controller=controller)
                    native_root_pending.discard(arm)
                    native_roots[arm] = native_root
                    native_service_public = native_root.public_ready()
                    native_service_public.update(
                        {
                            "evidence_sha256": ready_evidence_sha256,
                            "evidence_artifact": ready_evidence_name,
                        }
                    )
                    warmups[arm] = {
                        "payload": ready.get("payload", {}),
                        "warmup_ms": ready.get("warmup_ms"),
                        "root": root,
                        "downloads": downloads,
                        "marker": str(owned),
                        "native_service": native_service_public,
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
                            if entry.arm not in native_roots:
                                raise ValueError("native_service_root")
                            sampler = (
                                None
                                if entry.arm in retired_arms
                                else _make_owned_sampler(
                                    session.pid,
                                    warmups[entry.arm]["marker"],
                                    sampler_factory,
                                    controller,
                                )
                            )
                            capture = _ResourceCapture(sampler)
                            nonce = uuid.uuid4().hex
                            record_path = diagnostics / f"record-{entry.task}-{entry.arm}-s{entry.sample}-{nonce}.json"
                            started = time.monotonic()
                            baseline: dict = {}
                            resources: dict = {}
                            native_before = None
                            native_after = None
                            native_before_evidence_sha256 = None
                            native_before_evidence_artifact = None
                            native_after_evidence_sha256 = None
                            native_after_evidence_artifact = None
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
                            native_root = native_roots[entry.arm]
                            native_before = native_root.verify_before()
                            native_before_evidence_artifact = (
                                f"native-before-{entry.task}-{entry.arm}-s{entry.sample}-{nonce}.json"
                            )
                            native_before_evidence_sha256 = _write_private_evidence(
                                diagnostics,
                                native_before_evidence_artifact,
                                native_root.evidence_before(native_before),
                            )
                            if getattr(native_before, "eligible", False) is not True:
                                raise RuntimeError("native_service_before")
                            assert sampler is not None
                            sampler.enroll_native_service(native_before, controller=controller)
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
                            native_after = native_root.verify_after(native_before)
                            native_after_evidence_artifact = (
                                f"native-after-{entry.task}-{entry.arm}-s{entry.sample}-{nonce}.json"
                            )
                            native_after_evidence_sha256 = _write_private_evidence(
                                diagnostics,
                                native_after_evidence_artifact,
                                native_root.evidence_after(native_after),
                            )
                            if getattr(native_after, "eligible", False) is not True:
                                raise RuntimeError("native_service_after")
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
                        final_cpu = resources.get("observed_child_cpu_seconds")
                        baseline_cpu = baseline.get("observed_child_cpu_seconds")
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
                        final_tree_cpu = resources.get("observed_tree_cpu_seconds")
                        baseline_tree_cpu = baseline.get("observed_tree_cpu_seconds")
                        if (
                            cpu_delta is not None
                            and isinstance(final_cpu, (int, float))
                            and not isinstance(final_cpu, bool)
                            and isinstance(baseline_cpu, (int, float))
                            and not isinstance(baseline_cpu, bool)
                            and isinstance(final_tree_cpu, (int, float))
                            and not isinstance(final_tree_cpu, bool)
                            and isinstance(baseline_tree_cpu, (int, float))
                            and not isinstance(baseline_tree_cpu, bool)
                            and math.isfinite(float(final_tree_cpu))
                            and math.isfinite(float(baseline_tree_cpu))
                            and final_tree_cpu >= final_cpu
                            and baseline_tree_cpu >= baseline_cpu
                        ):
                            worker_cpu_delta = (float(final_tree_cpu) - float(final_cpu)) - (
                                float(baseline_tree_cpu) - float(baseline_cpu)
                            )
                            worker_cpu_error = None
                        else:
                            worker_cpu_delta = None
                            worker_cpu_error = "invalid_counter"
                        rss = resources.get("peak_child_tree_rss_bytes")
                        worker_rss = resources.get("peak_worker_rss_bytes")
                        unverified = resources.get("unverified_processes")
                        rss_valid = (
                            isinstance(rss, (int, float))
                            and not isinstance(rss, bool)
                            and math.isfinite(float(rss))
                            and float(rss) > 0
                        )
                        worker_rss_valid = (
                            isinstance(worker_rss, (int, float))
                            and not isinstance(worker_rss, bool)
                            and math.isfinite(float(worker_rss))
                            and float(worker_rss) > 0
                        )
                        native_ok = bool(
                            resources.get("native_service_method_version") == NATIVE_SERVICE_METHOD_VERSION
                            and resources.get("native_service_roots_observed") is True
                            and resources.get("native_controller_excluded") is True
                            and getattr(native_before, "eligible", False) is True
                            and getattr(native_after, "eligible", False) is True
                        )
                        resource_ok = bool(
                            rss_valid
                            and worker_rss_valid
                            and native_ok
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
                            "native_service_method_version": NATIVE_SERVICE_METHOD_VERSION,
                            "order": entry.position,
                            "arm_order": list(entry.order),
                            "oracle_ok": oracle.ok,
                            "oracle_detail": oracle.detail,
                            "warm_baseline_resources": baseline or None,
                            "warm_final_resources": resources or None,
                            "warm_peak_tree_rss_bytes": rss,
                            "warm_peak_worker_rss_bytes": worker_rss,
                            "warm_peak_child_footprint_bytes": resources.get("peak_child_footprint_bytes"),
                            "warm_task_tree_cpu_delta_seconds": cpu_delta,
                            "warm_worker_task_cpu_delta_seconds": worker_cpu_delta,
                            "warm_cpu_delta_error": cpu_error,
                            "warm_worker_cpu_delta_error": worker_cpu_error,
                            "warm_native_service_roots": resources.get("native_service_roots"),
                            "warm_native_roots_observed": resources.get("native_service_roots_observed") is True,
                            "warm_native_controller_excluded": resources.get("native_controller_excluded") is True,
                            "native_service_before": (
                                native_root.public_before(native_before) if native_before is not None else None
                            ),
                            "native_service_after": (
                                native_root.public_after(native_after) if native_after is not None else None
                            ),
                            "native_service_before_evidence_sha256": native_before_evidence_sha256,
                            "native_service_before_evidence_artifact": native_before_evidence_artifact,
                            "native_service_after_evidence_sha256": native_after_evidence_sha256,
                            "native_service_after_evidence_artifact": native_after_evidence_artifact,
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
                            "provenance_verified": before_binding.get("verified") is True,
                            "provenance_scope": "preflight",
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
                if arm in native_root_pending:
                    pending_report = {
                        "verified": False,
                        "errors": ["native_service_root_unknown"],
                        "scope": owned_scopes.get(arm),
                        "scope_preserved": True,
                    }
                    if arm in native_ready_failures:
                        pending_report["evidence"] = dict(native_ready_failures[arm])
                    cleanup_reports[arm] = pending_report
                elif sampler is None:
                    cleanup_reports[arm] = {"verified": False, "errors": ["sampler_unavailable"]}
                else:
                    try:
                        cleanup_reports[arm] = cleanup(sampler, interval=interval)
                    except Exception as error:
                        cleanup_reports[arm] = {"verified": False, "errors": [type(error).__name__]}
                if not cleanup_reports[arm].get("verified", False):
                    aborted = True
            if native_root_pending:
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
                "native_service_method_version": NATIVE_SERVICE_METHOD_VERSION,
                "observations": len(records),
                "expected_observations": expected_observations,
                "failures": failures,
                "warmups": warmups,
                "cleanup": cleanup_reports,
                "retired_workers": retired_workers,
                "fixture_errors": fixture_errors,
                "cleanup_verified": bool(cleanup_reports)
                and not native_root_pending
                and all(item.get("verified", False) for item in cleanup_reports.values()),
                "provenance_verified": verified,
                "source_unchanged": source_unchanged,
                "binding_unchanged": binding_unchanged,
                "binding_after": after_binding,
                "latency_claim_allowed": False,
                "terminal": True,
            }
            if native_root_pending:
                terminal["native_root_pending"] = {
                    arm: {"scope": owned_scopes.get(arm), "scope_preserved": True}
                    for arm in sorted(native_root_pending)
                }
            if native_ready_failures:
                terminal["native_ready_failures"] = native_ready_failures
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


def _finite_counter(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value)) and (value > 0 if positive else value >= 0)
    except (OverflowError, ValueError):
        return False


def _integer_counter(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _sha256_value(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _evidence_link(sha256: object, artifact: object) -> bool:
    return (
        _sha256_value(sha256)
        and isinstance(artifact, str)
        and artifact.endswith(".json")
        and artifact not in {".json", "..json"}
        and not any(char in artifact for char in ("/", "\\", "\x00"))
    )


def _native_public_proof(read: object) -> dict | None:
    if not isinstance(read, dict) or read.get("eligible") is not True or read.get("failure") is not None:
        return None
    if not {"eligible", "failure", "proof"} <= set(read):
        return None
    proof = read.get("proof")
    keys = {"pid", "created_at", "source", "executable_basename", "scope_digest"}
    if not isinstance(proof, dict) or set(proof) != keys:
        return None
    name = proof["executable_basename"]
    if (
        not _integer_counter(proof["pid"], minimum=1)
        or not _valid_service_time(proof["created_at"])
        or proof["source"] not in tuple(source.value for source in ServiceSource)
        or not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or any(char in name for char in ("/", "\\", "\x00"))
        or not _sha256_value(proof["scope_digest"])
    ):
        return None
    return proof


def _native_snapshot_valid(snapshot: object, proof: dict, *, final: bool) -> bool:
    if not isinstance(snapshot, dict):
        return False
    for key, version in (
        ("sampler_method_version", SAMPLER_METHOD_VERSION),
        ("native_service_method_version", NATIVE_SERVICE_METHOD_VERSION),
    ):
        if type(snapshot.get(key)) is not int or snapshot[key] != version:
            return False
    if any(
        not _integer_counter(snapshot.get(key), minimum=1)
        for key in ("samples", "observed_child_processes", "observed_browser_processes")
    ):
        return False
    if any(
        not _finite_counter(snapshot.get(key), positive=True)
        for key in ("peak_child_tree_rss_bytes", "peak_worker_rss_bytes")
    ) or not _finite_counter(snapshot.get("observed_child_cpu_seconds")):
        return False
    roots = snapshot.get("native_service_roots")
    if not isinstance(roots, list) or len(roots) != 1 or not isinstance(roots[0], dict):
        return False
    root = roots[0]
    if (
        root.get("observed") is not True
        or _native_public_proof(
            {
                "eligible": True,
                "failure": None,
                "proof": {key: value for key, value in root.items() if key != "observed"},
            }
        )
        != proof
    ):
        return False
    if (
        snapshot.get("native_service_roots_observed") is not True
        or snapshot.get("native_controller_excluded") is not True
    ):
        return False
    inventory = snapshot.get("processes")
    if not isinstance(inventory, list) or any(not isinstance(item, dict) for item in inventory):
        return False
    matches = [
        item
        for item in inventory
        if type(item.get("pid")) is int
        and item["pid"] == proof["pid"]
        and _valid_service_time(item.get("created_at"))
        and item["created_at"] == proof["created_at"]
    ]
    if len(matches) != 1 or matches[0].get("role") != "child":
        return False
    if not _finite_counter(matches[0].get("peak_rss_bytes"), positive=True) or not _finite_counter(
        matches[0].get("observed_cpu_seconds")
    ):
        return False
    if any(
        not _integer_counter(snapshot.get(key))
        for key in ("read_errors", "fatal_read_errors", "unverified_processes", "read_error_details_truncated")
    ):
        return False
    footprint = snapshot.get("peak_child_footprint_bytes")
    status = snapshot.get("footprint_status")
    if status == "observed":
        if not _finite_counter(footprint, positive=True):
            return False
    elif status not in ("unsupported", "unavailable") or footprint is not None:
        return False
    # Baseline discovery failures may have linked sampler-7 resolutions at the final barrier.
    # The publication consumer independently checks every raw read proof and artifact hash.
    return not final or (
        snapshot["samples"] >= 2
        and snapshot["fatal_read_errors"]
        == snapshot["unverified_processes"]
        == snapshot["read_error_details_truncated"]
        == 0
        and _integer_counter(snapshot.get("observed_browser_during_native_run"), minimum=1)
    )


def _warm_resource_row_valid(row: dict, ready: object) -> bool:
    if any(
        row.get(key) is not True
        for key in (
            "ok",
            "adapter_ok",
            "oracle_ok",
            "keychain_verified",
            "resource_ok",
            "provenance_verified",
            "warm_native_roots_observed",
            "warm_native_controller_excluded",
        )
    ):
        return False
    # The row attests preflight only; lane_valid also requires the final provenance footer.
    if row.get("provenance_scope") != "preflight":
        return False
    if row.get("record_type") != "sample" or row.get("latency_claim_allowed") is not False:
        return False
    if row.get("warm_sampling_error") is not None or row.get("warm_cpu_delta_error") is not None:
        return False
    proof = _native_public_proof(ready)
    if (
        proof is None
        or not isinstance(ready, dict)
        or not _evidence_link(ready.get("evidence_sha256"), ready.get("evidence_artifact"))
    ):
        return False
    for phase in ("before", "after"):
        if _native_public_proof(row.get(f"native_service_{phase}")) != proof:
            return False
        if not _evidence_link(
            row.get(f"native_service_{phase}_evidence_sha256"), row.get(f"native_service_{phase}_evidence_artifact")
        ):
            return False
    baseline, final = row.get("warm_baseline_resources"), row.get("warm_final_resources")
    if not _native_snapshot_valid(baseline, proof, final=False) or not _native_snapshot_valid(final, proof, final=True):
        return False
    assert isinstance(baseline, dict) and isinstance(final, dict)
    if final["samples"] <= baseline["samples"]:
        return False
    pairs = {
        "warm_peak_tree_rss_bytes": "peak_child_tree_rss_bytes",
        "warm_peak_worker_rss_bytes": "peak_worker_rss_bytes",
        "warm_peak_child_footprint_bytes": "peak_child_footprint_bytes",
        "warm_native_service_roots": "native_service_roots",
        "warm_unverified_processes": "unverified_processes",
        "warm_observed_child_processes": "observed_child_processes",
        "warm_observed_browser_during_native_run": "observed_browser_during_native_run",
        "warm_read_errors": "read_errors",
        "warm_fatal_read_errors": "fatal_read_errors",
        "warm_footprint_status": "footprint_status",
    }
    if any(type(row.get(key)) is not type(final[value]) or row[key] != final[value] for key, value in pairs.items()):
        return False
    delta = row.get("warm_task_tree_cpu_delta_seconds")
    worker_delta, worker_error = row.get("warm_worker_task_cpu_delta_seconds"), row.get("warm_worker_cpu_delta_error")
    tree_counters = (baseline.get("observed_tree_cpu_seconds"), final.get("observed_tree_cpu_seconds"))
    child_counters = (baseline["observed_child_cpu_seconds"], final["observed_child_cpu_seconds"])
    if all(
        _finite_counter(value) and value >= child for value, child in zip(tree_counters, child_counters, strict=True)
    ):
        assert isinstance(tree_counters[0], (int, float)) and isinstance(tree_counters[1], (int, float))
        expected_worker_delta = (float(tree_counters[1]) - float(child_counters[1])) - (
            float(tree_counters[0]) - float(child_counters[0])
        )
        if worker_error is not None or not _finite_counter(worker_delta) or worker_delta != expected_worker_delta:
            return False
    elif worker_delta is not None or worker_error != "invalid_counter":
        return False
    return (
        _finite_counter(delta)
        and final["observed_child_cpu_seconds"] >= baseline["observed_child_cpu_seconds"]
        and delta == float(final["observed_child_cpu_seconds"]) - float(baseline["observed_child_cpu_seconds"])
    )


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
    if terminal.get("native_root_pending"):
        return False
    if terminal.get("native_ready_failures"):
        return False
    if terminal.get("aborted", False) or terminal.get("setup_error") is not None:
        return False
    if terminal.get("fixture_errors"):
        return False
    if terminal.get("record_type") != "end" or terminal.get("method") != METHOD:
        return False
    if any(terminal.get(key) is not True for key in ("source_unchanged", "binding_unchanged")):
        return False
    for key, version in (
        ("sampler_method_version", SAMPLER_METHOD_VERSION),
        ("native_service_method_version", NATIVE_SERVICE_METHOD_VERSION),
    ):
        if type(terminal.get(key)) is not int or terminal[key] != version:
            return False
    warmups, cleanup = terminal.get("warmups"), terminal.get("cleanup")
    if not isinstance(warmups, dict) or not isinstance(cleanup, dict):
        return False
    ready_identities: set[tuple[int, int | float]] = set()
    ready_pids: set[int] = set()
    ready_scopes: set[str] = set()
    artifacts: set[str] = set()
    for warmup in warmups.values():
        if not isinstance(warmup, dict):
            return False
        ready = warmup.get("native_service")
        proof = _native_public_proof(ready)
        if (
            proof is None
            or not isinstance(ready, dict)
            or not _evidence_link(ready.get("evidence_sha256"), ready.get("evidence_artifact"))
        ):
            return False
        identity = (proof["pid"], proof["created_at"])
        if identity in ready_identities or proof["pid"] in ready_pids or proof["scope_digest"] in ready_scopes:
            return False
        name = ready["evidence_artifact"]
        if name in artifacts:
            return False
        artifacts.add(name)
        ready_identities.add(identity)
        ready_pids.add(proof["pid"])
        ready_scopes.add(proof["scope_digest"])
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
        for name, version in (
            ("sampler_method_version", SAMPLER_METHOD_VERSION),
            ("native_service_method_version", NATIVE_SERVICE_METHOD_VERSION),
        ):
            if type(row.get(name)) is not int or row[name] != version:
                return False
        warmup = warmups.get(arm)
        if not isinstance(warmup, dict) or not _warm_resource_row_valid(row, warmup.get("native_service")):
            return False
        initial_order = list(warmups)
        shift = sample % len(initial_order)
        rotated = initial_order[shift:] + initial_order[:shift]
        if row.get("arm_order") != rotated or type(row.get("order")) is not int or row["order"] != rotated.index(arm):
            return False
        for phase in ("before", "after"):
            name = row[f"native_service_{phase}_evidence_artifact"]
            if name in artifacts:
                return False
            artifacts.add(name)
        key = (task, arm, sample)
        if key in seen:
            return False
        seen.add(key)
    actual_arms = {arm for _, arm, _ in seen}
    if set(warmups) != actual_arms or set(cleanup) != actual_arms:
        return False
    if any(not isinstance(report, dict) or report.get("verified") is not True for report in cleanup.values()):
        return False
    plan_values = (arms, tasks, samples)
    if any(value is not None for value in plan_values):
        if arms is None or tasks is None or samples is None:
            return False
        if not isinstance(arms, (list, tuple)) or not isinstance(tasks, (list, tuple)) or not arms or not tasks:
            return False
        if not all(isinstance(value, str) and value for value in (*arms, *tasks)):
            return False
        if len(set(arms)) != len(arms) or len(set(tasks)) != len(tasks):
            return False
        if list(arms) != list(warmups):
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
