"""Browser-free contracts for generic native-service root proofs."""

from __future__ import annotations

import math
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Any

import psutil
import pytest

from benchmarks.interactive.adapters import service_roots


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("native I/O is forbidden in service-root fake tests")

    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


CONTROLLER = service_roots.ServiceIdentity(10, 100.0)
WORKER = service_roots.ServiceIdentity(20, 200.0)
SOFTWARE = "/usr/bin/python3"
SCOPE = "/tmp/openbrowser-owned"
LARGE_TIME = 2**54
LARGE_TIME_NEXT = LARGE_TIME + 1
HUGE_TIME = 10**1000


@dataclass
class FakeProcess:
    pid: int
    ctimes: list[Any] = field(default_factory=lambda: [300.0])
    exes: list[Any] = field(default_factory=lambda: [SOFTWARE])
    statuses: list[Any] = field(default_factory=lambda: ["running"])
    runnings: list[Any] = field(default_factory=lambda: [True])
    scopes: list[Any] = field(default_factory=lambda: [SCOPE])
    errors: dict[str, list[BaseException | None]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    positions: dict[str, int] = field(default_factory=dict)

    def _position(self, name: str) -> int:
        position = self.positions.get(name, 0)
        self.positions[name] = position + 1
        return position

    def _value(self, name: str, values: list[Any]) -> Any:
        position = self._position(name)
        if not values:
            return None
        return values[min(position, len(values) - 1)]

    def _error(self, name: str) -> BaseException | None:
        values = self.errors.get(name)
        if not values:
            return None
        position = self._position(f"{name}:error")
        return values[min(position, len(values) - 1)]

    def create_time(self) -> Any:
        self.calls.append("create_time")
        error = self._error("create_time")
        if error is not None:
            raise error
        return self._value("create_time", self.ctimes)

    def exe(self) -> Any:
        self.calls.append("exe")
        error = self._error("exe")
        if error is not None:
            raise error
        return self._value("exe", self.exes)

    def is_running(self) -> Any:
        self.calls.append("is_running")
        error = self._error("is_running")
        if error is not None:
            raise error
        return self._value("is_running", self.runnings)

    def status(self) -> Any:
        self.calls.append("status")
        error = self._error("status")
        if error is not None:
            raise error
        return self._value("status", self.statuses)

    def next_scope(self) -> Any:
        self.calls.append("scope")
        error = self._error("scope")
        if error is not None:
            raise error
        return self._value("scope", self.scopes)


def read_scope(process: Any) -> Any:
    return process.next_scope()


def make_factory(process: FakeProcess):
    return lambda pid: process


def capture(
    process: FakeProcess | None = None,
    *,
    pid: Any = 30,
    source: Any = service_roots.ServiceSource.PID_FILE,
    expected_scope: Any = SCOPE,
    expected_executable: Any = SOFTWARE,
    controller: Any = CONTROLLER,
    worker: Any = WORKER,
    factory: Any = None,
):
    return service_roots.capture_service_proof(
        pid,
        source=source,
        expected_scope=expected_scope,
        expected_executable=expected_executable,
        controller_identity=controller,
        worker_identity=worker,
        process_factory=factory or make_factory(process or FakeProcess(30)),
        scope_reader=read_scope,
    )


def verify(
    proof: Any,
    process: FakeProcess | None = None,
    *,
    source: Any = service_roots.ServiceSource.PID_FILE,
    expected_scope: Any = SCOPE,
    expected_executable: Any = SOFTWARE,
    controller: Any = CONTROLLER,
    worker: Any = WORKER,
    factory: Any = None,
):
    return service_roots.verify_service_proof(
        proof,
        source=source,
        expected_scope=expected_scope,
        expected_executable=expected_executable,
        controller_identity=controller,
        worker_identity=worker,
        process_factory=factory or make_factory(process or FakeProcess(30)),
        scope_reader=read_scope,
    )


def assert_failure(result, failure):
    assert result.eligible is False
    assert result.proof is None
    assert result.failure is failure


def test_fake_io_guard_cannot_be_swallowed_by_capture():
    def factory(pid):
        psutil.Process(pid)

    with pytest.raises(pytest.fail.Exception, match="native I/O is forbidden"):
        capture(factory=factory)


def test_capture_accepts_exact_identity_executable_and_scope():
    process = FakeProcess(30)
    result = capture(process)

    assert result.eligible is True
    assert result.failure is None
    assert result.proof == service_roots.NativeServiceProof(
        identity=service_roots.ServiceIdentity(30, 300.0),
        executable=SOFTWARE,
        scope=SCOPE,
        source=service_roots.ServiceSource.PID_FILE,
    )
    assert process.calls.count("create_time") == 2
    assert process.calls.count("exe") == 2
    assert process.calls.count("scope") == 2


def test_capture_requires_closed_source_enum():
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = capture(factory=factory, source="pid_file")
    assert_failure(result, service_roots.ServiceFailure.INVALID_SOURCE)
    assert called is False


@pytest.mark.parametrize("controller", [None, "controller", service_roots.ServiceIdentity(0, 100.0)])
def test_capture_rejects_unknown_controller_identity(controller):
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = capture(controller=controller, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.INVALID_CONTROLLER)
    assert called is False


@pytest.mark.parametrize("worker", [None, "worker", service_roots.ServiceIdentity(20, float("nan"))])
def test_capture_rejects_unknown_worker_identity(worker):
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = capture(worker=worker, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.INVALID_WORKER)
    assert called is False


@pytest.mark.parametrize("pid", [True, False, 0, -1, 1.0, "30", None])
def test_capture_rejects_malformed_pid(pid):
    called = False

    def factory(value):
        nonlocal called
        called = True
        return FakeProcess(value)

    result = capture(pid=pid, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.MALFORMED_PID)
    assert called is False


@pytest.mark.parametrize("scope", [None, "", "relative/scope", 3])
def test_capture_rejects_malformed_expected_scope(scope):
    result = capture(expected_scope=scope)
    assert_failure(result, service_roots.ServiceFailure.MALFORMED_SCOPE)


@pytest.mark.parametrize("executable", [None, "", "python3", 3])
def test_capture_rejects_malformed_expected_executable(executable):
    result = capture(expected_executable=executable)
    assert_failure(result, service_roots.ServiceFailure.MALFORMED_EXECUTABLE)


@pytest.mark.parametrize("created", [True, 0, -1, math.inf, math.nan, "300", None])
def test_capture_rejects_malformed_create_time(created):
    result = capture(FakeProcess(30, ctimes=[created]))
    assert_failure(result, service_roots.ServiceFailure.MALFORMED_CREATE_TIME)


def test_capture_preserves_large_integer_create_time_raw():
    result = capture(FakeProcess(30, ctimes=[LARGE_TIME]))
    assert result.proof is not None
    assert result.proof.identity.created_at == LARGE_TIME
    assert type(result.proof.identity.created_at) is int


def test_capture_rejects_large_integer_create_time_reuse():
    result = capture(FakeProcess(30, ctimes=[LARGE_TIME, LARGE_TIME_NEXT]))
    assert_failure(result, service_roots.ServiceFailure.REUSED)


@pytest.mark.parametrize("executable", ["", "python3", None, 3])
def test_capture_rejects_malformed_process_executable(executable):
    result = capture(FakeProcess(30, exes=[executable]))
    assert_failure(result, service_roots.ServiceFailure.MALFORMED_EXECUTABLE)


@pytest.mark.parametrize("scope", ["", "relative/scope", None, 3])
def test_capture_rejects_malformed_process_scope(scope):
    result = capture(FakeProcess(30, scopes=[scope]))
    assert_failure(result, service_roots.ServiceFailure.MALFORMED_SCOPE)


def test_capture_rejects_scope_mismatch():
    result = capture(FakeProcess(30, scopes=["/tmp/foreign-scope"]))
    assert_failure(result, service_roots.ServiceFailure.SCOPE_MISMATCH)


def test_capture_rejects_huge_controller_create_time_without_overflow():
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = capture(controller=service_roots.ServiceIdentity(10, HUGE_TIME), factory=factory)
    assert_failure(result, service_roots.ServiceFailure.INVALID_CONTROLLER)
    assert called is False


def test_capture_rejects_huge_worker_create_time_without_overflow():
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = capture(worker=service_roots.ServiceIdentity(20, HUGE_TIME), factory=factory)
    assert_failure(result, service_roots.ServiceFailure.INVALID_WORKER)
    assert called is False


def test_capture_rejects_huge_observed_create_time_without_overflow():
    result = capture(FakeProcess(30, ctimes=[HUGE_TIME]))
    assert_failure(result, service_roots.ServiceFailure.MALFORMED_CREATE_TIME)


def test_capture_rejects_executable_mismatch():
    result = capture(FakeProcess(30, exes=["/usr/bin/node"]))
    assert_failure(result, service_roots.ServiceFailure.EXECUTABLE_MISMATCH)


def test_capture_rejects_exact_controller_identity():
    result = capture(FakeProcess(10, ctimes=[100.0]), pid=10)
    assert_failure(result, service_roots.ServiceFailure.EXCLUDED_CONTROLLER)


def test_capture_rejects_exact_worker_identity():
    result = capture(FakeProcess(20, ctimes=[200.0]), pid=20)
    assert_failure(result, service_roots.ServiceFailure.EXCLUDED_WORKER)


def test_capture_allows_controller_pid_with_different_create_time():
    result = capture(FakeProcess(10, ctimes=[300.0]), pid=10)
    assert result.eligible is True
    assert result.proof is not None
    assert result.proof.identity == service_roots.ServiceIdentity(10, 300.0)


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (PermissionError("denied"), service_roots.ServiceFailure.DENIED),
        (ProcessLookupError("gone"), service_roots.ServiceFailure.GONE),
        (FileNotFoundError("gone"), service_roots.ServiceFailure.GONE),
        (OSError("unknown"), service_roots.ServiceFailure.UNKNOWN),
        (service_roots.ServiceReaderError(service_roots.ServiceFailure.ZOMBIE), service_roots.ServiceFailure.ZOMBIE),
    ],
)
def test_capture_maps_factory_failures_to_closed_reasons(error, failure):
    def factory(pid):
        raise error

    result = capture(factory=factory)
    assert_failure(result, failure)


@pytest.mark.parametrize("status", ["zombie", "dead"])
def test_capture_rejects_zombie_or_dead_status(status):
    result = capture(FakeProcess(30, statuses=[status]))
    assert_failure(result, service_roots.ServiceFailure.ZOMBIE)


@pytest.mark.parametrize("status", ["", "mystery", None, 3])
def test_capture_rejects_unknown_status(status):
    result = capture(FakeProcess(30, statuses=[status]))
    assert_failure(result, service_roots.ServiceFailure.UNKNOWN)


@pytest.mark.parametrize("running", [False, 0, 1, None, "yes"])
def test_capture_rejects_non_live_running_state(running):
    result = capture(FakeProcess(30, runnings=[running]))
    expected = service_roots.ServiceFailure.GONE if running is False else service_roots.ServiceFailure.UNKNOWN
    assert_failure(result, expected)


def test_capture_rejects_pid_mismatch_from_factory():
    result = capture(FakeProcess(31))
    assert_failure(result, service_roots.ServiceFailure.PID_MISMATCH)


def test_capture_rejects_create_time_change_during_capture():
    result = capture(FakeProcess(30, ctimes=[300.0, 301.0]))
    assert_failure(result, service_roots.ServiceFailure.REUSED)


def test_capture_rejects_executable_change_during_capture():
    result = capture(FakeProcess(30, exes=[SOFTWARE, "/usr/bin/node"]))
    assert_failure(result, service_roots.ServiceFailure.EXEC_CHANGED)


def test_capture_rejects_scope_change_during_capture():
    result = capture(FakeProcess(30, scopes=[SCOPE, "/tmp/other-scope"]))
    assert_failure(result, service_roots.ServiceFailure.SCOPE_CHANGED)


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (PermissionError("denied"), service_roots.ServiceFailure.DENIED),
        (ProcessLookupError("gone"), service_roots.ServiceFailure.GONE),
        (OSError("unknown"), service_roots.ServiceFailure.UNKNOWN),
    ],
)
def test_capture_maps_reader_failures_to_closed_reasons(error, failure):
    process = FakeProcess(30, errors={"scope": [error]})
    result = capture(process)
    assert_failure(result, failure)


def test_verify_accepts_unchanged_proof():
    proof = capture(FakeProcess(30)).proof
    assert proof is not None
    result = verify(proof, FakeProcess(30))
    assert result.eligible is True
    assert result.failure is None
    assert result.proof == proof


def test_verify_rejects_missing_proof_without_process_read():
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = verify(None, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.INVALID_PROOF)
    assert called is False


def test_verify_rejects_source_mismatch_without_process_read():
    proof = capture(FakeProcess(30)).proof
    assert proof is not None
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = verify(proof, source=service_roots.ServiceSource.POPEN, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.SOURCE_MISMATCH)
    assert called is False


def test_verify_rejects_large_integer_reuse():
    proof = service_roots.NativeServiceProof(
        identity=service_roots.ServiceIdentity(30, LARGE_TIME),
        executable=SOFTWARE,
        scope=SCOPE,
        source=service_roots.ServiceSource.PID_FILE,
    )
    result = verify(proof, FakeProcess(30, ctimes=[LARGE_TIME_NEXT]))
    assert_failure(result, service_roots.ServiceFailure.REUSED)


def test_verify_rejects_huge_proof_create_time_without_overflow():
    proof = service_roots.NativeServiceProof(
        identity=service_roots.ServiceIdentity(30, HUGE_TIME),
        executable=SOFTWARE,
        scope=SCOPE,
        source=service_roots.ServiceSource.PID_FILE,
    )
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = verify(proof, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.INVALID_PROOF)
    assert called is False


def test_verify_rejects_reused_pid():
    proof = capture(FakeProcess(30, ctimes=[300.0])).proof
    assert proof is not None
    result = verify(proof, FakeProcess(30, ctimes=[301.0]))
    assert_failure(result, service_roots.ServiceFailure.REUSED)


def test_verify_rejects_executable_change():
    proof = capture(FakeProcess(30)).proof
    assert proof is not None
    result = verify(proof, FakeProcess(30, exes=["/usr/bin/node"]))
    assert_failure(result, service_roots.ServiceFailure.EXEC_CHANGED)


def test_verify_rejects_scope_change():
    proof = capture(FakeProcess(30)).proof
    assert proof is not None
    result = verify(proof, FakeProcess(30, scopes=["/tmp/other-scope"]))
    assert_failure(result, service_roots.ServiceFailure.SCOPE_CHANGED)


def test_verify_rejects_controller_identity_without_process_read():
    proof = service_roots.NativeServiceProof(
        identity=CONTROLLER,
        executable=SOFTWARE,
        scope=SCOPE,
        source=service_roots.ServiceSource.PID_FILE,
    )
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = verify(proof, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.EXCLUDED_CONTROLLER)
    assert called is False


def test_verify_rejects_worker_identity_without_process_read():
    proof = service_roots.NativeServiceProof(
        identity=WORKER,
        executable=SOFTWARE,
        scope=SCOPE,
        source=service_roots.ServiceSource.PID_FILE,
    )
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = verify(proof, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.EXCLUDED_WORKER)
    assert called is False


def test_verify_rejects_malformed_proof_without_process_read():
    proof = service_roots.NativeServiceProof(
        identity=service_roots.ServiceIdentity(True, 300.0),
        executable=SOFTWARE,
        scope=SCOPE,
        source=service_roots.ServiceSource.PID_FILE,
    )
    called = False

    def factory(pid):
        nonlocal called
        called = True
        return FakeProcess(pid)

    result = verify(proof, factory=factory)
    assert_failure(result, service_roots.ServiceFailure.INVALID_PROOF)
    assert called is False


@pytest.mark.parametrize("status", ["zombie", "dead"])
def test_verify_rejects_zombie(status):
    proof = capture(FakeProcess(30)).proof
    assert proof is not None
    result = verify(proof, FakeProcess(30, statuses=[status]))
    assert_failure(result, service_roots.ServiceFailure.ZOMBIE)


def test_service_source_is_closed_to_declared_values():
    assert {item.value for item in service_roots.ServiceSource} == {
        "popen",
        "pid_file",
        "owner_file",
        "native_status",
        "session_file",
    }
