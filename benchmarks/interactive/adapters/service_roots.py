"""Generic identity proofs for persistent native services.

The module is intentionally free of host readers and native PID parsing. A
caller supplies the exact PID, the exact expected executable and scope, and the
readers that expose process and scope state. This keeps the proof layer pure
and testable while native adapters decide where a PID came from.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ServiceSource(StrEnum):
    """Closed source categories for a native service PID."""

    POPEN = "popen"
    PID_FILE = "pid_file"
    OWNER_FILE = "owner_file"
    NATIVE_STATUS = "native_status"
    SESSION_FILE = "session_file"


class ServiceFailure(StrEnum):
    """Closed ineligibility reasons. A proof is either present or absent."""

    INVALID_SOURCE = "invalid_source"
    INVALID_CONTROLLER = "invalid_controller"
    INVALID_WORKER = "invalid_worker"
    MALFORMED_PID = "malformed_pid"
    MALFORMED_CREATE_TIME = "malformed_create_time"
    MALFORMED_EXECUTABLE = "malformed_executable"
    MALFORMED_SCOPE = "malformed_scope"
    INVALID_PROOF = "invalid_proof"
    SOURCE_MISMATCH = "source_mismatch"
    PID_MISMATCH = "pid_mismatch"
    EXECUTABLE_MISMATCH = "executable_mismatch"
    SCOPE_MISMATCH = "scope_mismatch"
    EXCLUDED_CONTROLLER = "excluded_controller"
    EXCLUDED_WORKER = "excluded_worker"
    DENIED = "denied"
    GONE = "gone"
    ZOMBIE = "zombie"
    UNKNOWN = "unknown"
    REUSED = "reused"
    EXEC_CHANGED = "exec_changed"
    SCOPE_CHANGED = "scope_changed"


class ServiceReaderError(Exception):
    """Reader failure with an explicit closed reason."""

    def __init__(self, failure: ServiceFailure) -> None:
        if not isinstance(failure, ServiceFailure):
            raise TypeError("failure must be a ServiceFailure")
        self.failure = failure
        super().__init__(failure.value)


@dataclass(frozen=True)
class ServiceIdentity:
    pid: int
    created_at: int | float


@dataclass(frozen=True)
class NativeServiceProof:
    identity: ServiceIdentity
    executable: str
    scope: str
    source: ServiceSource


@dataclass(frozen=True)
class ServiceProofResult:
    proof: NativeServiceProof | None
    failure: ServiceFailure | None

    def __post_init__(self) -> None:
        if (self.proof is None) == (self.failure is None):
            raise ValueError("exactly one of proof and failure is required")

    @property
    def eligible(self) -> bool:
        return self.proof is not None and self.failure is None


class ProcessView(Protocol):
    pid: int

    def create_time(self) -> object: ...

    def exe(self) -> object: ...

    def is_running(self) -> object: ...

    def status(self) -> object: ...


ProcessFactory = Callable[[int], ProcessView]
ScopeReader = Callable[[ProcessView], object]

_LIVE_STATUSES = frozenset(
    {
        "running",
        "sleeping",
        "disk-sleep",
        "idle",
        "waking",
        "parked",
        "locked",
        "waiting",
    }
)
_ZOMBIE_STATUSES = frozenset({"zombie", "dead"})


@dataclass(frozen=True)
class _ServiceState:
    identity: ServiceIdentity
    executable: str
    scope: str


def _canonical_path(value: object) -> str | None:
    """Normalize an absolute path without filesystem access."""
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    if not os.path.isabs(value):
        return None
    return os.path.normcase(os.path.normpath(value))


def _valid_time(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return value


def _valid_identity(value: object) -> bool:
    if not isinstance(value, ServiceIdentity):
        return False
    if type(value.pid) is not int or value.pid <= 0:
        return False
    return _valid_time(value.created_at) is not None


def _same_identity(left: ServiceIdentity, right: ServiceIdentity) -> bool:
    return left.pid == right.pid and left.created_at == right.created_at


def _failure_for_read(error: BaseException) -> ServiceFailure:
    if isinstance(error, ServiceReaderError):
        return error.failure
    if isinstance(error, PermissionError):
        return ServiceFailure.DENIED
    if isinstance(error, (ProcessLookupError, FileNotFoundError)):
        return ServiceFailure.GONE
    return ServiceFailure.UNKNOWN


def _read_identity(process: ProcessView, expected_pid: int) -> tuple[ServiceIdentity | None, ServiceFailure | None]:
    try:
        actual_pid = process.pid
    except Exception as error:
        return None, _failure_for_read(error)
    if type(actual_pid) is not int or actual_pid <= 0:
        return None, ServiceFailure.MALFORMED_PID
    if actual_pid != expected_pid:
        return None, ServiceFailure.PID_MISMATCH
    try:
        created = process.create_time()
    except Exception as error:
        return None, _failure_for_read(error)
    created_at = _valid_time(created)
    if created_at is None:
        return None, ServiceFailure.MALFORMED_CREATE_TIME
    try:
        running = process.is_running()
    except Exception as error:
        return None, _failure_for_read(error)
    if type(running) is not bool:
        return None, ServiceFailure.UNKNOWN
    if not running:
        return None, ServiceFailure.GONE
    try:
        status = process.status()
    except Exception as error:
        return None, _failure_for_read(error)
    if not isinstance(status, str) or not status:
        return None, ServiceFailure.UNKNOWN
    if status in _ZOMBIE_STATUSES:
        return None, ServiceFailure.ZOMBIE
    if status not in _LIVE_STATUSES:
        return None, ServiceFailure.UNKNOWN
    return ServiceIdentity(actual_pid, created_at), None


def _read_executable(process: ProcessView) -> tuple[str | None, ServiceFailure | None]:
    try:
        value = process.exe()
    except Exception as error:
        return None, _failure_for_read(error)
    executable = _canonical_path(value)
    if executable is None:
        return None, ServiceFailure.MALFORMED_EXECUTABLE
    return executable, None


def _read_scope(process: ProcessView, scope_reader: ScopeReader) -> tuple[str | None, ServiceFailure | None]:
    try:
        value = scope_reader(process)
    except Exception as error:
        return None, _failure_for_read(error)
    scope = _canonical_path(value)
    if scope is None:
        return None, ServiceFailure.MALFORMED_SCOPE
    return scope, None


def _read_state(
    process: ProcessView,
    expected_pid: int,
    scope_reader: ScopeReader,
) -> tuple[_ServiceState | None, ServiceFailure | None]:
    """Read the full state twice so identity, executable and scope are fenced."""
    first_identity, failure = _read_identity(process, expected_pid)
    if failure is not None:
        return None, failure
    first_executable, failure = _read_executable(process)
    if failure is not None:
        return None, failure
    first_scope, failure = _read_scope(process, scope_reader)
    if failure is not None:
        return None, failure

    second_identity, failure = _read_identity(process, expected_pid)
    if failure is not None:
        return None, failure
    assert first_identity is not None
    assert second_identity is not None
    if not _same_identity(first_identity, second_identity):
        return None, ServiceFailure.REUSED
    second_executable, failure = _read_executable(process)
    if failure is not None:
        return None, failure
    assert first_executable is not None
    assert second_executable is not None
    if first_executable != second_executable:
        return None, ServiceFailure.EXEC_CHANGED
    second_scope, failure = _read_scope(process, scope_reader)
    if failure is not None:
        return None, failure
    assert first_scope is not None
    assert second_scope is not None
    if first_scope != second_scope:
        return None, ServiceFailure.SCOPE_CHANGED
    return _ServiceState(second_identity, second_executable, second_scope), None


def _validate_common(
    *,
    source: object,
    expected_scope: object,
    expected_executable: object,
    controller_identity: object,
    worker_identity: object,
) -> tuple[str | None, str | None, ServiceFailure | None]:
    if not isinstance(source, ServiceSource):
        return None, None, ServiceFailure.INVALID_SOURCE
    if not _valid_identity(controller_identity):
        return None, None, ServiceFailure.INVALID_CONTROLLER
    if not _valid_identity(worker_identity):
        return None, None, ServiceFailure.INVALID_WORKER
    scope = _canonical_path(expected_scope)
    if scope is None:
        return None, None, ServiceFailure.MALFORMED_SCOPE
    executable = _canonical_path(expected_executable)
    if executable is None:
        return None, None, ServiceFailure.MALFORMED_EXECUTABLE
    return scope, executable, None


def capture_service_proof(
    pid: object,
    *,
    source: object,
    expected_scope: object,
    expected_executable: object,
    controller_identity: object,
    worker_identity: object,
    process_factory: ProcessFactory,
    scope_reader: ScopeReader,
) -> ServiceProofResult:
    """Capture a positive proof only when every identity and scope field matches."""
    expected_scope_value, expected_executable_value, failure = _validate_common(
        source=source,
        expected_scope=expected_scope,
        expected_executable=expected_executable,
        controller_identity=controller_identity,
        worker_identity=worker_identity,
    )
    if failure is not None:
        return ServiceProofResult(None, failure)
    if type(pid) is not int or pid <= 0:
        return ServiceProofResult(None, ServiceFailure.MALFORMED_PID)

    assert isinstance(source, ServiceSource)
    assert isinstance(controller_identity, ServiceIdentity)
    assert isinstance(worker_identity, ServiceIdentity)
    assert expected_scope_value is not None
    assert expected_executable_value is not None
    try:
        process = process_factory(pid)
    except Exception as error:
        return ServiceProofResult(None, _failure_for_read(error))
    state, failure = _read_state(process, pid, scope_reader)
    if failure is not None or state is None:
        return ServiceProofResult(None, failure or ServiceFailure.UNKNOWN)
    if _same_identity(state.identity, controller_identity):
        return ServiceProofResult(None, ServiceFailure.EXCLUDED_CONTROLLER)
    if _same_identity(state.identity, worker_identity):
        return ServiceProofResult(None, ServiceFailure.EXCLUDED_WORKER)
    if state.executable != expected_executable_value:
        return ServiceProofResult(None, ServiceFailure.EXECUTABLE_MISMATCH)
    if state.scope != expected_scope_value:
        return ServiceProofResult(None, ServiceFailure.SCOPE_MISMATCH)
    return ServiceProofResult(
        NativeServiceProof(
            identity=state.identity,
            executable=state.executable,
            scope=state.scope,
            source=source,
        ),
        None,
    )


def _valid_proof(proof: object) -> bool:
    if not isinstance(proof, NativeServiceProof):
        return False
    if not _valid_identity(proof.identity):
        return False
    if not isinstance(proof.source, ServiceSource):
        return False
    executable = _canonical_path(proof.executable)
    scope = _canonical_path(proof.scope)
    return executable is not None and executable == proof.executable and scope is not None and scope == proof.scope


def verify_service_proof(
    proof: object,
    *,
    source: object,
    expected_scope: object,
    expected_executable: object,
    controller_identity: object,
    worker_identity: object,
    process_factory: ProcessFactory,
    scope_reader: ScopeReader,
) -> ServiceProofResult:
    """Verify one previously captured proof against the current exact process."""
    if not _valid_proof(proof):
        return ServiceProofResult(None, ServiceFailure.INVALID_PROOF)
    expected_scope_value, expected_executable_value, failure = _validate_common(
        source=source,
        expected_scope=expected_scope,
        expected_executable=expected_executable,
        controller_identity=controller_identity,
        worker_identity=worker_identity,
    )
    if failure is not None:
        return ServiceProofResult(None, failure)
    assert isinstance(proof, NativeServiceProof)
    assert isinstance(source, ServiceSource)
    assert isinstance(controller_identity, ServiceIdentity)
    assert isinstance(worker_identity, ServiceIdentity)
    assert expected_scope_value is not None
    assert expected_executable_value is not None
    if proof.source is not source:
        return ServiceProofResult(None, ServiceFailure.SOURCE_MISMATCH)
    if _same_identity(proof.identity, controller_identity):
        return ServiceProofResult(None, ServiceFailure.EXCLUDED_CONTROLLER)
    if _same_identity(proof.identity, worker_identity):
        return ServiceProofResult(None, ServiceFailure.EXCLUDED_WORKER)
    if proof.executable != expected_executable_value:
        return ServiceProofResult(None, ServiceFailure.EXECUTABLE_MISMATCH)
    if proof.scope != expected_scope_value:
        return ServiceProofResult(None, ServiceFailure.SCOPE_MISMATCH)

    try:
        process = process_factory(proof.identity.pid)
    except Exception as error:
        return ServiceProofResult(None, _failure_for_read(error))
    state, failure = _read_state(process, proof.identity.pid, scope_reader)
    if failure is not None or state is None:
        return ServiceProofResult(None, failure or ServiceFailure.UNKNOWN)
    if not _same_identity(state.identity, proof.identity):
        return ServiceProofResult(None, ServiceFailure.REUSED)
    if state.executable != proof.executable:
        return ServiceProofResult(None, ServiceFailure.EXEC_CHANGED)
    if state.scope != proof.scope:
        return ServiceProofResult(None, ServiceFailure.SCOPE_CHANGED)
    return ServiceProofResult(proof, None)


__all__ = [
    "NativeServiceProof",
    "ProcessFactory",
    "ProcessView",
    "ScopeReader",
    "ServiceFailure",
    "ServiceIdentity",
    "ServiceProofResult",
    "ServiceReaderError",
    "ServiceSource",
    "capture_service_proof",
    "verify_service_proof",
]
