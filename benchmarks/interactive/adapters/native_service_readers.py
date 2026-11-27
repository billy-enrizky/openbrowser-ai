"""Compose exact native sources with live service proofs, without launching work.

No host reader runs at import. Callers supply the native handle identity or the
Playwright open identity. File sources read only their exact run-owned path.
Browser identity remains the separate matched-engine proof.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .native_service_sources import (
    MAX_PID_RECORD_BYTES,
    PidRecord,
    PidSource,
    ProcessCommand,
    SourceError,
    parse_native_pid_record,
    select_playwright_daemon,
)
from .service_roots import (
    ProcessFactory,
    ProcessView,
    ServiceFailure,
    ServiceIdentity,
    ServiceProofResult,
    ServiceReaderError,
    ServiceSource,
    capture_service_proof,
    verify_service_proof,
)


class NativeArm(StrEnum):
    OPENBROWSER_CLI = "openbrowser-fast"
    OPENBROWSER_MCP = "openbrowser-mcp-fast"
    PLAYWRIGHT_MCP = "playwright-mcp"
    PLAYWRIGHT_CLI = "playwright-cli"
    CHROME_DEVTOOLS = "chrome-devtools-mcp"
    AGENT_BROWSER = "agent-browser"
    BROWSER_USE = "browser-use"


class ReadFailure(StrEnum):
    MISSING = "missing"
    DENIED = "denied"
    MALFORMED = "malformed"
    AMBIGUOUS = "ambiguous"
    PROCESS_MISSING = "process_missing"
    IDENTITY_MISMATCH = "identity_mismatch"
    SCOPE_MISMATCH = "scope_mismatch"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NativeRunScope:
    root: Path
    home: Path
    config: Path
    profile: Path
    session: str


@dataclass(frozen=True)
class NativeServiceRequest:
    arm: NativeArm
    scope: NativeRunScope
    executable: str
    native_identity: ServiceIdentity | None = None
    daemon_entry: str | None = None


def _host_error(error: Exception, pid: int) -> ServiceReaderError:
    import psutil

    if isinstance(error, psutil.ZombieProcess):
        failure = ServiceFailure.ZOMBIE if type(error.pid) is int and error.pid == pid else ServiceFailure.UNKNOWN
    elif isinstance(error, psutil.NoSuchProcess):
        failure = ServiceFailure.GONE if type(error.pid) is int and error.pid == pid else ServiceFailure.UNKNOWN
    elif isinstance(error, (PermissionError, psutil.AccessDenied)):
        failure = ServiceFailure.DENIED
    elif isinstance(error, (FileNotFoundError, ProcessLookupError)):
        failure = ServiceFailure.GONE
    else:
        failure = ServiceFailure.UNKNOWN
    return ServiceReaderError(failure)


class FreshProcess:
    """Read each proof field through a new native object, avoiding psutil caches."""

    def __init__(self, pid: int, *, native_factory: Callable[[int], Any] | None = None) -> None:
        if type(pid) is not int or pid <= 0:
            raise ServiceReaderError(ServiceFailure.MALFORMED_PID)
        self.pid = pid
        self._pid = pid
        self._factory = native_factory

    def _read(self, field: str) -> Any:
        try:
            if self._factory is None:
                import psutil

                process = psutil.Process(self._pid)
            else:
                process = self._factory(self._pid)
            if type(process.pid) is not int or process.pid != self._pid:
                raise ServiceReaderError(ServiceFailure.PID_MISMATCH)
            return process.pid if field == "pid" else getattr(process, field)()
        except ServiceReaderError:
            raise
        except Exception as error:
            raise _host_error(error, self._pid) from error

    def create_time(self) -> object:
        return self._read("create_time")

    def exe(self) -> object:
        return self._read("exe")

    def is_running(self) -> object:
        return self._read("is_running")

    def status(self) -> object:
        return self._read("status")

    def cmdline(self) -> object:
        return self._read("cmdline")

    def environ(self) -> object:
        return self._read("environ")


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_bounded_pid_file(path: Path, limit: int) -> bytes:
    """Read one regular owned file with size, descriptor, and path fences."""
    if type(limit) is not int or not 0 < limit <= MAX_PID_RECORD_BYTES:
        raise SourceError("invalid_limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise SourceError("invalid_pid_file")
        if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise PermissionError("pid_file_owner")
        data = os.read(descriptor, limit + 1)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if len(data) != before.st_size or not (
            _file_identity(before) == _file_identity(after) == _file_identity(current)
        ):
            raise SourceError("pid_file_changed")
        return data
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class HostReaders:
    read_bytes: Callable[[Path, int], bytes] = read_bounded_pid_file
    process_factory: ProcessFactory = FreshProcess
    start_reader: Callable[[int], str | int | None] | None = None


@dataclass(frozen=True)
class SourceRead:
    request: NativeServiceRequest
    source: ServiceSource | None
    candidate: PidRecord | ProcessCommand | None
    record_bytes: bytes | None
    failure: ReadFailure | None

    def __post_init__(self) -> None:
        if (self.candidate is None) == (self.failure is None):
            raise ValueError("exactly one candidate or failure is required")


@dataclass(frozen=True)
class ServiceRead:
    before: SourceRead
    after: SourceRead | None
    proof: ServiceProofResult
    failure: ReadFailure | None
    initial_proof: ServiceProofResult | None = None
    closing_proof: ServiceProofResult | None = None

    @property
    def eligible(self) -> bool:
        if self.failure is not None or self.after is None or self.initial_proof is None or self.closing_proof is None:
            return False
        if self.before.failure is not None or self.after.failure is not None:
            return False
        if self.before.candidate is None or self.after.candidate is None:
            return False
        first, last = self.initial_proof, self.closing_proof
        if not first.eligible or not last.eligible or first.proof is None or last.proof is None:
            return False
        return (
            self.proof == last
            and first.proof == last.proof
            and self.before.request == self.after.request
            and self.before.source == self.after.source == last.proof.source
            and self.before.candidate.pid == self.after.candidate.pid == last.proof.identity.pid
            and _same_source(self.before, self.after)
        )


_FILE_SOURCES = {
    NativeArm.OPENBROWSER_CLI: PidSource.OPENBROWSER,
    NativeArm.AGENT_BROWSER: PidSource.AGENT_BROWSER,
    NativeArm.BROWSER_USE: PidSource.BROWSER_USE,
}


def _path(value: object) -> str | None:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value or "\x00" in value or not os.path.isabs(value):
        return None
    return os.path.normcase(os.path.normpath(value))


def _identity_valid(identity: object) -> bool:
    if not isinstance(identity, ServiceIdentity) or type(identity.pid) is not int or identity.pid <= 0:
        return False
    value = identity.created_at
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return value > 0 and math.isfinite(value)
    except OverflowError:
        return False


def _request_valid(target: NativeServiceRequest) -> bool:
    if not isinstance(target, NativeServiceRequest) or not isinstance(target.arm, NativeArm):
        return False
    if (
        not isinstance(target.scope, NativeRunScope)
        or _path(target.executable) is None
        or _path(target.executable) != target.executable
    ):
        return False
    s = target.scope
    paths = [_path(value) for value in (s.root, s.home, s.config, s.profile)]
    if not isinstance(s.session, str):
        return False
    if re.fullmatch(r"[A-Za-z0-9_-]+", s.session) is None:
        return False
    root = paths[0]
    if not isinstance(root, str):
        return False
    for value in paths[1:]:
        if not isinstance(value, str):
            return False
        try:
            if os.path.commonpath([root, value]) != root or value == root:
                return False
        except ValueError:
            return False
    if target.arm in _FILE_SOURCES:
        if target.native_identity is not None:
            return False
    elif not _identity_valid(target.native_identity):
        return False
    if target.arm is NativeArm.PLAYWRIGHT_CLI:
        return _path(target.daemon_entry) is not None and _path(target.daemon_entry) == target.daemon_entry
    return target.daemon_entry is None


def _pid_path(target: NativeServiceRequest) -> Path:
    s = target.scope
    if target.arm is NativeArm.OPENBROWSER_CLI:
        return s.home / ".openbrowser/daemon.pid"
    if target.arm is NativeArm.AGENT_BROWSER:
        return Path("/tmp") / ("abib-" + s.session) / (s.session + ".pid")
    return s.root / "bu-runtime/bu-default.pid"


def _failure(error: Exception) -> ReadFailure:
    if isinstance(error, PermissionError):
        return ReadFailure.DENIED
    if isinstance(error, FileNotFoundError):
        return ReadFailure.MISSING
    if isinstance(error, SourceError):
        return {
            "ambiguous_process": ReadFailure.AMBIGUOUS,
            "process_missing": ReadFailure.PROCESS_MISSING,
        }.get(str(error), ReadFailure.MALFORMED)
    if isinstance(error, ServiceReaderError):
        return {
            ServiceFailure.DENIED: ReadFailure.DENIED,
            ServiceFailure.GONE: ReadFailure.PROCESS_MISSING,
            ServiceFailure.ZOMBIE: ReadFailure.PROCESS_MISSING,
            ServiceFailure.SCOPE_MISMATCH: ReadFailure.SCOPE_MISMATCH,
            ServiceFailure.SCOPE_CHANGED: ReadFailure.SCOPE_MISMATCH,
            ServiceFailure.MALFORMED_SCOPE: ReadFailure.SCOPE_MISMATCH,
            ServiceFailure.PID_MISMATCH: ReadFailure.IDENTITY_MISMATCH,
            ServiceFailure.MALFORMED_PID: ReadFailure.MALFORMED,
            ServiceFailure.MALFORMED_CREATE_TIME: ReadFailure.MALFORMED,
            ServiceFailure.MALFORMED_EXECUTABLE: ReadFailure.MALFORMED,
            ServiceFailure.INVALID_SOURCE: ReadFailure.UNSUPPORTED,
            ServiceFailure.INVALID_CONTROLLER: ReadFailure.MALFORMED,
            ServiceFailure.INVALID_WORKER: ReadFailure.MALFORMED,
            ServiceFailure.INVALID_PROOF: ReadFailure.MALFORMED,
            ServiceFailure.SOURCE_MISMATCH: ReadFailure.IDENTITY_MISMATCH,
            ServiceFailure.EXCLUDED_CONTROLLER: ReadFailure.IDENTITY_MISMATCH,
            ServiceFailure.EXCLUDED_WORKER: ReadFailure.IDENTITY_MISMATCH,
            ServiceFailure.REUSED: ReadFailure.IDENTITY_MISMATCH,
            ServiceFailure.EXEC_CHANGED: ReadFailure.IDENTITY_MISMATCH,
            ServiceFailure.EXECUTABLE_MISMATCH: ReadFailure.IDENTITY_MISMATCH,
        }.get(error.failure, ReadFailure.UNKNOWN)
    return ReadFailure.UNKNOWN


def _method(process: ProcessView, name: str) -> Any:
    method = getattr(process, name, None)
    if not callable(method):
        raise ServiceReaderError(ServiceFailure.UNKNOWN)
    return method()


def read_service_source(target: NativeServiceRequest, host: HostReaders) -> SourceRead:
    """Read one exact candidate. This result carries no ownership eligibility."""
    if not _request_valid(target):
        return SourceRead(target, None, None, None, ReadFailure.MALFORMED)
    source = ServiceSource.PID_FILE if target.arm in _FILE_SOURCES else ServiceSource.POPEN
    retained_bytes: bytes | None = None
    try:
        if target.arm in _FILE_SOURCES:
            data = host.read_bytes(_pid_path(target), MAX_PID_RECORD_BYTES)
            if isinstance(data, bytes) and len(data) <= MAX_PID_RECORD_BYTES:
                retained_bytes = data
            candidate = parse_native_pid_record(data, source=_FILE_SOURCES[target.arm])
            return SourceRead(target, source, candidate, data, None)
        assert target.native_identity is not None
        if target.arm is NativeArm.PLAYWRIGHT_CLI:
            assert isinstance(target.daemon_entry, str)
            source = ServiceSource.NATIVE_STATUS
            process = host.process_factory(target.native_identity.pid)
            command = _process_command(process)
            candidate = select_playwright_daemon(
                [command],
                executable=target.executable,
                daemon_entry=target.daemon_entry,
                session=target.scope.session,
                profile=str(target.scope.profile),
                config=str(target.scope.config / "playwright-config.json"),
            )
            return SourceRead(target, source, candidate, None, None)
        return SourceRead(target, source, PidRecord(target.native_identity.pid, "decimal", None), None, None)
    except Exception as error:
        return SourceRead(target, source, None, retained_bytes, _failure(error))


def _argv(process: ProcessView) -> tuple[str, ...]:
    value = _method(process, "cmdline")
    if (
        not isinstance(value, (tuple, list))
        or not value
        or any(not isinstance(x, str) or not x or "\x00" in x for x in value)
    ):
        raise ServiceReaderError(ServiceFailure.SCOPE_MISMATCH)
    return tuple(value)


def _process_command(process: ProcessView) -> ProcessCommand:
    executable = process.exe()
    if not isinstance(executable, str):
        raise ServiceReaderError(ServiceFailure.MALFORMED_EXECUTABLE)
    return ProcessCommand(process.pid, executable, _argv(process))


def _flag(argv: tuple[str, ...], flag: str, expected: Path) -> None:
    values = []
    for index, token in enumerate(argv):
        if token == flag:
            values.append(argv[index + 1] if index + 1 < len(argv) else None)
        elif token.startswith(flag + "="):
            values.append(token[len(flag) + 1 :])
    if len(values) != 1 or _path(values[0]) != _path(expected):
        raise ServiceReaderError(ServiceFailure.SCOPE_MISMATCH)


def _scope_reader(target: NativeServiceRequest):
    s, arm = target.scope, target.arm
    expected_env: dict[str, Path] = {}
    if arm in {NativeArm.OPENBROWSER_CLI, NativeArm.OPENBROWSER_MCP}:
        expected_env = {"OPENBROWSER_CONFIG_DIR": s.config, "OPENBROWSER_SOCKET": s.home / ".openbrowser/daemon.sock"}
        expected_scope = s.config
    elif arm is NativeArm.AGENT_BROWSER:
        expected_scope = Path("/tmp") / ("abib-" + s.session)
        expected_env = {"AGENT_BROWSER_SOCKET_DIR": expected_scope}
    elif arm is NativeArm.BROWSER_USE:
        expected_scope = s.root / "bu-runtime"
        expected_env = {"BH_RUNTIME_DIR": expected_scope, "BH_TMP_DIR": s.root / "bu-tmp"}
    else:
        expected_scope = s.profile

    def read(process: ProcessView) -> object:
        if expected_env:
            environment = _method(process, "environ")
            if not isinstance(environment, dict) or any(
                _path(environment.get(k)) != _path(v) for k, v in expected_env.items()
            ):
                raise ServiceReaderError(ServiceFailure.SCOPE_MISMATCH)
        argv = _argv(process)
        if arm is NativeArm.OPENBROWSER_CLI and argv[1:] != ("-m", "openbrowser.daemon.server"):
            raise ServiceReaderError(ServiceFailure.SCOPE_MISMATCH)
        if arm is NativeArm.OPENBROWSER_MCP and argv.count("--mcp") != 1:
            raise ServiceReaderError(ServiceFailure.SCOPE_MISMATCH)
        if arm is NativeArm.BROWSER_USE and argv[1:] != ("-m", "browser_harness.daemon"):
            raise ServiceReaderError(ServiceFailure.SCOPE_MISMATCH)
        if arm is NativeArm.PLAYWRIGHT_MCP:
            _flag(argv, "--user-data-dir", s.profile)
            _flag(argv, "--config", s.config / "playwright-config.json")
        if arm is NativeArm.CHROME_DEVTOOLS:
            _flag(argv, "--userDataDir", s.profile)
        if arm is NativeArm.PLAYWRIGHT_CLI:
            assert isinstance(target.daemon_entry, str)
            select_playwright_daemon(
                [_process_command(process)],
                executable=target.executable,
                daemon_entry=target.daemon_entry,
                session=s.session,
                profile=str(s.profile),
                config=str(s.config / "playwright-config.json"),
            )
        return str(expected_scope)

    return str(expected_scope), read


def _rejected(
    before: SourceRead,
    failure: ReadFailure,
    *,
    after: SourceRead | None = None,
    initial_proof: ServiceProofResult | None = None,
) -> ServiceRead:
    reason = ServiceFailure.DENIED if failure is ReadFailure.DENIED else ServiceFailure.UNKNOWN
    return ServiceRead(before, after, ServiceProofResult(None, reason), failure, initial_proof=initial_proof)


def _same_source(before: SourceRead, after: SourceRead) -> bool:
    if before.candidate == after.candidate and before.record_bytes == after.record_bytes:
        return True
    left, right = before.candidate, after.candidate
    return (
        before.request.arm is NativeArm.BROWSER_USE
        and isinstance(left, PidRecord)
        and isinstance(right, PidRecord)
        and left.pid == right.pid
        and {left.record_format, right.record_format} == {"decimal", "json"}
    )


def _observe(
    target: NativeServiceRequest,
    host: HostReaders,
    controller: ServiceIdentity,
    worker: ServiceIdentity,
    previous: ServiceRead | None,
) -> ServiceRead:
    before = read_service_source(target, host)
    if before.failure is not None:
        return _rejected(before, before.failure)
    assert before.candidate is not None and before.source is not None
    if previous is not None and (not previous.eligible or previous.proof.proof is None):
        return _rejected(before, ReadFailure.IDENTITY_MISMATCH)
    candidate = before.candidate
    if isinstance(candidate, PidRecord) and candidate.started is not None:
        if host.start_reader is None:
            return _rejected(before, ReadFailure.UNSUPPORTED)
        try:
            observed = host.start_reader(candidate.pid)
            if type(observed) is not type(candidate.started) or observed != candidate.started:
                return _rejected(before, ReadFailure.IDENTITY_MISMATCH)
        except Exception as error:
            return _rejected(before, _failure(error))
    expected_scope, scope_reader = _scope_reader(target)
    kwargs = {
        "source": before.source,
        "expected_scope": expected_scope,
        "expected_executable": target.executable,
        "controller_identity": controller,
        "worker_identity": worker,
        "process_factory": host.process_factory,
        "scope_reader": scope_reader,
    }
    if previous is None:
        proof = capture_service_proof(candidate.pid, **kwargs)
    else:
        assert previous.proof.proof is not None
        if candidate.pid != previous.proof.proof.identity.pid:
            return _rejected(before, ReadFailure.IDENTITY_MISMATCH)
        proof = verify_service_proof(previous.proof.proof, **kwargs)
    if not proof.eligible:
        failure = _failure(ServiceReaderError(proof.failure or ServiceFailure.UNKNOWN))
        return ServiceRead(before, None, proof, failure, initial_proof=proof)
    assert proof.proof is not None
    if target.native_identity is not None and target.native_identity != proof.proof.identity:
        return _rejected(before, ReadFailure.IDENTITY_MISMATCH, initial_proof=proof)
    after = read_service_source(target, host)
    if after.failure is not None:
        return _rejected(before, after.failure, after=after, initial_proof=proof)
    if not _same_source(before, after):
        return _rejected(before, ReadFailure.IDENTITY_MISMATCH, after=after, initial_proof=proof)
    closing_candidate = after.candidate
    if isinstance(closing_candidate, PidRecord) and closing_candidate.started is not None:
        try:
            if host.start_reader is None:
                return _rejected(before, ReadFailure.UNSUPPORTED, after=after, initial_proof=proof)
            observed = host.start_reader(closing_candidate.pid)
            if type(observed) is not type(closing_candidate.started) or observed != closing_candidate.started:
                return _rejected(before, ReadFailure.IDENTITY_MISMATCH, after=after, initial_proof=proof)
        except Exception as error:
            return _rejected(before, _failure(error), after=after, initial_proof=proof)
    closing = verify_service_proof(proof.proof, **kwargs)
    failure = None if closing.eligible else _failure(ServiceReaderError(closing.failure or ServiceFailure.UNKNOWN))
    return ServiceRead(before, after, closing, failure, initial_proof=proof, closing_proof=closing)


def capture_native_service(
    target: NativeServiceRequest,
    host: HostReaders,
    *,
    controller: ServiceIdentity,
    worker: ServiceIdentity,
) -> ServiceRead:
    return _observe(target, host, controller, worker, None)


def verify_native_service(
    previous: ServiceRead,
    target: NativeServiceRequest,
    host: HostReaders,
    *,
    controller: ServiceIdentity,
    worker: ServiceIdentity,
) -> ServiceRead:
    return _observe(target, host, controller, worker, previous)
