"""All-arm host-reader contracts with hard native-I/O guards."""

from __future__ import annotations

import os
import socket
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from benchmarks.interactive.adapters import native_service_readers as readers
from benchmarks.interactive.adapters.service_roots import ServiceFailure, ServiceIdentity, ServiceReaderError


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Host-reader tests require injected native observations")

    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "open", forbidden)


CONTROLLER = ServiceIdentity(11, 1.0)
WORKER = ServiceIdentity(12, 2.0)
SERVICE = ServiceIdentity(73, 20.25)


def scope():
    return readers.NativeRunScope(
        root=Path("/private/tmp/owned"),
        home=Path("/private/tmp/owned/home"),
        config=Path("/private/tmp/owned/config"),
        profile=Path("/private/tmp/owned/profile with spaces"),
        session="b123456789",
    )


def request(arm):
    direct = arm in {
        readers.NativeArm.OPENBROWSER_MCP,
        readers.NativeArm.PLAYWRIGHT_MCP,
        readers.NativeArm.PLAYWRIGHT_CLI,
        readers.NativeArm.CHROME_DEVTOOLS,
    }
    return readers.NativeServiceRequest(
        arm=arm,
        scope=scope(),
        executable="/opt/native/bin/service",
        native_identity=SERVICE if direct else None,
        daemon_entry="/opt/playwright/lib/entry/cliDaemon.js" if arm is readers.NativeArm.PLAYWRIGHT_CLI else None,
    )


class FakeProcess:
    def __init__(self, target):
        self.pid = SERVICE.pid
        self.created = SERVICE.created_at
        self.executable = target.executable
        self.running = True
        self.state = "sleeping"
        s = target.scope
        self.environment = {
            "OPENBROWSER_CONFIG_DIR": str(s.config),
            "OPENBROWSER_SOCKET": str(s.home / ".openbrowser/daemon.sock"),
            "AGENT_BROWSER_SOCKET_DIR": "/tmp/abib-" + s.session,
            "BH_RUNTIME_DIR": str(s.root / "bu-runtime"),
            "BH_TMP_DIR": str(s.root / "bu-tmp"),
        }
        self.argv = ["service"]
        match target.arm:
            case readers.NativeArm.OPENBROWSER_CLI:
                self.argv += ["-m", "openbrowser.daemon.server"]
            case readers.NativeArm.OPENBROWSER_MCP:
                self.argv += ["--mcp"]
            case readers.NativeArm.PLAYWRIGHT_MCP:
                self.argv += ["--user-data-dir", str(s.profile), "--config", str(s.config / "playwright-config.json")]
            case readers.NativeArm.PLAYWRIGHT_CLI:
                self.argv += [
                    target.daemon_entry,
                    s.session,
                    "--profile=" + str(s.profile),
                    "--config=" + str(s.config / "playwright-config.json"),
                ]
            case readers.NativeArm.CHROME_DEVTOOLS:
                self.argv += ["--userDataDir", str(s.profile)]
            case readers.NativeArm.BROWSER_USE:
                self.argv += ["-m", "browser_harness.daemon"]

    def create_time(self):
        return self.created

    def exe(self):
        return self.executable

    def is_running(self):
        return self.running

    def status(self):
        return self.state

    def cmdline(self):
        return list(self.argv)

    def environ(self):
        return dict(self.environment)


def fake_readers(target, *, payload=b"73", process=None):
    process = process or FakeProcess(target)
    seen = []

    def read_file(path, limit):
        seen.append((path, limit))
        return payload

    return readers.HostReaders(read_bytes=read_file, process_factory=lambda pid: process), process, seen


def capture(target, host):
    return readers.capture_native_service(target, host, controller=CONTROLLER, worker=WORKER)


@pytest.mark.parametrize("arm", list(readers.NativeArm))
def test_all_seven_sources_require_a_live_proof(arm):
    target = request(arm)
    host, process, _ = fake_readers(target)
    candidate = readers.read_service_source(target, host)
    assert candidate.failure is None
    assert candidate.candidate is not None
    assert candidate.candidate.pid == process.pid
    assert not hasattr(candidate, "eligible")
    result = capture(target, host)
    assert result.eligible is True
    assert result.proof.proof is not None
    assert result.proof.proof.identity == SERVICE
    assert result.before.candidate is not None
    assert result.after is not None and result.after.candidate is not None
    assert result.before.candidate.pid == result.after.candidate.pid == SERVICE.pid


@pytest.mark.parametrize(
    "arm", [readers.NativeArm.OPENBROWSER_CLI, readers.NativeArm.AGENT_BROWSER, readers.NativeArm.BROWSER_USE]
)
def test_only_exact_native_pid_file_is_read(arm):
    target = request(arm)
    host, _, seen = fake_readers(target)
    result = capture(target, host)
    assert result.eligible
    expected = {
        readers.NativeArm.OPENBROWSER_CLI: target.scope.home / ".openbrowser/daemon.pid",
        readers.NativeArm.AGENT_BROWSER: Path("/tmp/abib-b123456789/b123456789.pid"),
        readers.NativeArm.BROWSER_USE: target.scope.root / "bu-runtime/bu-default.pid",
    }[arm]
    assert seen == [(expected, 4096), (expected, 4096)]


@pytest.mark.parametrize("arm", list(readers.NativeArm))
def test_wrong_executable_never_becomes_eligible(arm):
    target = request(arm)
    host, process, _ = fake_readers(target)
    process.executable = "/other/native"
    assert capture(target, host).eligible is False


@pytest.mark.parametrize("arm", list(readers.NativeArm))
def test_denied_process_read_is_a_closed_failure(arm):
    target = request(arm)
    host, _, _ = fake_readers(target)

    def denied(pid):
        raise PermissionError("private diagnostics stay private")

    result = capture(target, replace(host, process_factory=denied))
    assert not result.eligible
    assert result.failure is readers.ReadFailure.DENIED
    assert "private diagnostics" not in str(result.failure)


@pytest.mark.parametrize("arm", list(readers.NativeArm))
def test_wrong_scope_never_becomes_eligible(arm):
    target = request(arm)
    host, process, _ = fake_readers(target)
    process.environment = {}
    process.argv = ["service", "--user-data-dir=/other", "--userDataDir=/other"]
    assert not capture(target, host).eligible


@pytest.mark.parametrize(
    "arm", [readers.NativeArm.OPENBROWSER_CLI, readers.NativeArm.AGENT_BROWSER, readers.NativeArm.BROWSER_USE]
)
@pytest.mark.parametrize(
    "error,expected", [(FileNotFoundError, "MISSING"), (PermissionError, "DENIED"), (OSError, "UNKNOWN")]
)
def test_native_file_read_failures_are_typed(arm, error, expected):
    target = request(arm)
    host, _, _ = fake_readers(target)

    def failed_file(path, limit):
        raise error()

    result = capture(target, replace(host, read_bytes=failed_file))
    assert not result.eligible
    assert result.failure is getattr(readers.ReadFailure, expected)


def test_browser_use_pending_json_requires_the_native_fingerprint():
    target = request(readers.NativeArm.BROWSER_USE)
    host, _, _ = fake_readers(target, payload=b'{"pid":73,"started":"native-start"}')
    assert not capture(target, host).eligible
    assert capture(target, replace(host, start_reader=lambda pid: "native-start")).eligible
    assert not capture(target, replace(host, start_reader=lambda pid: "wrong-start")).eligible


def test_changed_pid_record_after_proof_is_retained_and_rejected():
    target = request(readers.NativeArm.BROWSER_USE)
    host, _, _ = fake_readers(target)
    values = iter([b"73", b"74"])
    result = capture(target, replace(host, read_bytes=lambda path, limit: next(values)))
    assert not result.eligible
    assert result.before.candidate is not None
    assert result.after is not None and result.after.candidate is not None
    assert result.before.candidate.pid == 73
    assert result.after.candidate.pid == 74
    assert result.failure is readers.ReadFailure.IDENTITY_MISMATCH


@pytest.mark.parametrize("arm", list(readers.NativeArm))
def test_verify_uses_the_old_identity_and_never_recaches(arm):
    target = request(arm)
    host, process, _ = fake_readers(target)
    initial = capture(target, host)
    assert initial.eligible
    process.created += 1
    checked = readers.verify_native_service(initial, target, host, controller=CONTROLLER, worker=WORKER)
    assert not checked.eligible
    assert initial.proof.proof is not None
    assert initial.proof.proof.identity == SERVICE


def test_direct_source_identity_create_time_must_match():
    target = request(readers.NativeArm.OPENBROWSER_MCP)
    host, process, _ = fake_readers(target)
    process.created += 1
    assert not capture(target, host).eligible


@pytest.mark.parametrize("excluded", [CONTROLLER, WORKER])
def test_controller_and_worker_are_excluded(excluded):
    target = request(readers.NativeArm.BROWSER_USE)
    host, process, _ = fake_readers(target, payload=str(excluded.pid).encode())
    process.pid, process.created = excluded.pid, excluded.created_at
    result = capture(target, host)
    assert not result.eligible
    assert result.proof.failure in {ServiceFailure.EXCLUDED_CONTROLLER, ServiceFailure.EXCLUDED_WORKER}


def test_playwright_cli_uses_the_published_pid_and_exact_argv():
    target = request(readers.NativeArm.PLAYWRIGHT_CLI)
    host, process, seen = fake_readers(target)
    process.argv[-1] += "-foreign"
    result = capture(target, host)
    assert not result.eligible
    assert result.failure is readers.ReadFailure.PROCESS_MISSING
    assert seen == []


def test_fresh_facade_does_not_reuse_a_cached_native_process():
    target = request(readers.NativeArm.OPENBROWSER_MCP)
    calls = []

    def native_factory(pid):
        process = FakeProcess(target)
        calls.append(process)
        if len(calls) > 5:
            process.created += 1
        return process

    facade = readers.FreshProcess(SERVICE.pid, native_factory=native_factory)
    host = readers.HostReaders(read_bytes=lambda path, limit: b"73", process_factory=lambda pid: facade)
    assert not capture(target, host).eligible
    assert len(calls) > 5
    assert len({id(process) for process in calls}) == len(calls)


def test_default_file_reader_requires_a_native_window():
    with pytest.raises(pytest.fail.Exception):
        readers.read_bounded_pid_file(Path("/tmp/forbidden"), 4096)


@pytest.mark.parametrize("value", [None, True, 12, "relative"])
def test_invalid_expected_executable_fails_before_host_reads(value):
    target = replace(request(readers.NativeArm.BROWSER_USE), executable=value)

    def forbidden(*args):
        pytest.fail("An invalid request must not read native state")

    host = readers.HostReaders(read_bytes=forbidden, process_factory=forbidden)
    result = capture(target, host)
    assert not result.eligible
    assert result.failure is readers.ReadFailure.MALFORMED


def fake_file_read(monkeypatch, *, data=b"73", size=2, mode=stat.S_IFREG | 0o600, inode_changed=False):
    before = SimpleNamespace(st_dev=1, st_ino=2, st_mode=mode, st_uid=501, st_size=size, st_mtime_ns=3, st_ctime_ns=4)
    current = SimpleNamespace(**vars(before))
    if inode_changed:
        current.st_ino += 1
    seen = []
    with monkeypatch.context() as local:
        local.setattr(os, "open", lambda path, flags: seen.append((path, flags)) or 99)
        local.setattr(os, "fstat", lambda fd: before)
        local.setattr(os, "stat", lambda *args, **kwargs: current)
        local.setattr(os, "read", lambda fd, amount: data)
        local.setattr(os, "close", lambda fd: seen.append(("close", fd)))
        local.setattr(os, "geteuid", lambda: 501, raising=False)
        try:
            value = readers.read_bounded_pid_file(Path("/tmp/native.pid"), 4096)
        except Exception as error:
            value = error
    return value, seen


def test_regular_file_reader_preserves_exact_bytes(monkeypatch):
    value, seen = fake_file_read(monkeypatch)
    assert value == b"73"
    assert seen[-1] == ("close", 99)


def test_fifo_open_cannot_block_before_file_type_check(monkeypatch):
    value, seen = fake_file_read(monkeypatch, mode=stat.S_IFIFO | 0o600)
    assert isinstance(value, readers.SourceError)
    assert seen[0][1] & getattr(os, "O_NONBLOCK", 0)
    assert seen[-1] == ("close", 99)


def test_short_regular_file_read_is_not_a_different_pid(monkeypatch):
    value, _ = fake_file_read(monkeypatch, data=b"73", size=5)
    assert isinstance(value, readers.SourceError)


def test_changed_file_identity_is_not_a_native_record(monkeypatch):
    value, seen = fake_file_read(monkeypatch, inode_changed=True)
    assert isinstance(value, readers.SourceError)
    assert seen[-1] == ("close", 99)


@pytest.mark.parametrize("arm", list(readers.NativeArm))
def test_scope_read_failures_keep_their_specific_reason(arm):
    target = request(arm)
    host, process, _ = fake_readers(target)
    process.environment = {}
    process.argv = ["service", "--user-data-dir", "/foreign", "--userDataDir", "/foreign"]
    result = capture(target, host)
    expected = (
        readers.ReadFailure.PROCESS_MISSING
        if arm is readers.NativeArm.PLAYWRIGHT_CLI
        else readers.ReadFailure.SCOPE_MISMATCH
    )
    assert result.failure is expected


@pytest.mark.parametrize("arm", list(readers.NativeArm))
def test_missing_live_process_has_a_specific_failure(arm):
    target = request(arm)
    host, _, _ = fake_readers(target)

    def gone(pid):
        raise ServiceReaderError(ServiceFailure.GONE)

    result = capture(target, replace(host, process_factory=gone))
    assert not result.eligible
    assert result.failure is readers.ReadFailure.PROCESS_MISSING


@pytest.mark.parametrize(
    "error,expected",
    [
        (psutil.NoSuchProcess(73), ServiceFailure.GONE),
        (psutil.NoSuchProcess(74), ServiceFailure.UNKNOWN),
        (psutil.ZombieProcess(73), ServiceFailure.ZOMBIE),
        (psutil.AccessDenied(73), ServiceFailure.DENIED),
        (PermissionError(), ServiceFailure.DENIED),
        (OSError(), ServiceFailure.UNKNOWN),
    ],
)
def test_fresh_process_preserves_closed_native_errors(error, expected):
    def failed(pid):
        raise error

    with pytest.raises(ServiceReaderError) as caught:
        readers.FreshProcess(73, native_factory=failed).create_time()
    assert caught.value.failure is expected


def test_exception_pid_boolean_is_not_an_authoritative_exit():
    def failed(pid):
        raise psutil.NoSuchProcess(True)

    with pytest.raises(ServiceReaderError) as caught:
        readers.FreshProcess(1, native_factory=failed).create_time()
    assert caught.value.failure is ServiceFailure.UNKNOWN


@pytest.mark.parametrize("pid", [None, True, 0, -1, 1.5, "73"])
def test_fresh_process_rejects_invalid_pid_without_native_reads(pid):
    with pytest.raises(ServiceReaderError) as caught:
        readers.FreshProcess(pid)
    assert caught.value.failure is ServiceFailure.MALFORMED_PID


@pytest.mark.parametrize("created", [None, True, 0, -1, float("nan"), float("inf"), 10**500])
def test_invalid_published_identity_fails_before_host_reads(created):
    target = replace(request(readers.NativeArm.OPENBROWSER_MCP), native_identity=ServiceIdentity(73, created))

    def forbidden(*args):
        pytest.fail("An invalid native identity must not read the host")

    host = readers.HostReaders(read_bytes=forbidden, process_factory=forbidden)
    assert capture(target, host).failure is readers.ReadFailure.MALFORMED


@pytest.mark.parametrize("payload", [b"0", b"not-a-pid", b"\xff", b'{"pid":true}'])
def test_malformed_bounded_native_bytes_are_retained(payload):
    target = request(readers.NativeArm.BROWSER_USE)
    host, _, _ = fake_readers(target, payload=payload)
    result = capture(target, host)
    assert not result.eligible
    assert result.failure is readers.ReadFailure.MALFORMED
    assert result.before.record_bytes == payload


def test_malformed_closing_record_is_retained_separately():
    target = request(readers.NativeArm.BROWSER_USE)
    host, _, _ = fake_readers(target)
    values = iter([b"73", b"broken"])
    result = capture(target, replace(host, read_bytes=lambda path, limit: next(values)))
    assert not result.eligible
    assert result.before.record_bytes == b"73"
    assert result.after is not None
    assert result.after.record_bytes == b"broken"


@pytest.mark.parametrize("expected,observed", [(1, True), (1, 1.0), (2**54, float(2**54))])
@pytest.mark.parametrize("closing_only", [False, True])
def test_native_start_fingerprint_requires_exact_type(expected, observed, closing_only):
    import json

    target = request(readers.NativeArm.BROWSER_USE)
    payload = json.dumps({"pid": 73, "started": expected}).encode()
    host, _, _ = fake_readers(target, payload=payload)
    values = iter([expected, observed] if closing_only else [observed])
    result = capture(target, replace(host, start_reader=lambda pid: next(values)))
    assert not result.eligible
    assert result.failure is readers.ReadFailure.IDENTITY_MISMATCH


@pytest.mark.parametrize("reverse", [False, True])
def test_same_pid_browser_use_format_transition_keeps_one_verified_identity(reverse):
    target = request(readers.NativeArm.BROWSER_USE)
    host, _, _ = fake_readers(target)
    records = [b'{"pid":73,"started":"native-start"}', b"73"]
    values = iter(list(reversed(records)) if reverse else records)
    host = replace(host, read_bytes=lambda path, limit: next(values), start_reader=lambda pid: "native-start")
    result = capture(target, host)
    assert result.eligible
    assert result.after is not None
    assert result.before.record_bytes != result.after.record_bytes
    assert result.initial_proof is not None and result.closing_proof is not None
    assert result.initial_proof.proof is not None and result.closing_proof.proof is not None
    assert result.initial_proof.proof.identity == result.closing_proof.proof.identity == SERVICE


def test_closing_source_read_cannot_hide_service_exit():
    target = request(readers.NativeArm.BROWSER_USE)
    host, process, _ = fake_readers(target)
    reads = []

    def read_file(path, limit):
        reads.append(path)
        return b"73"

    def factory(pid):
        if len(reads) >= 2:
            raise ServiceReaderError(ServiceFailure.GONE)
        return process

    result = capture(target, replace(host, read_bytes=read_file, process_factory=factory))
    assert not result.eligible
    assert result.failure is readers.ReadFailure.PROCESS_MISSING
    assert result.initial_proof is not None and result.initial_proof.eligible
    assert result.closing_proof is not None and not result.closing_proof.eligible


@pytest.mark.parametrize(
    "failure,expected",
    [
        (ServiceFailure.PID_MISMATCH, readers.ReadFailure.IDENTITY_MISMATCH),
        (ServiceFailure.MALFORMED_PID, readers.ReadFailure.MALFORMED),
        (ServiceFailure.MALFORMED_CREATE_TIME, readers.ReadFailure.MALFORMED),
        (ServiceFailure.MALFORMED_EXECUTABLE, readers.ReadFailure.MALFORMED),
        (ServiceFailure.MALFORMED_SCOPE, readers.ReadFailure.SCOPE_MISMATCH),
        (ServiceFailure.SOURCE_MISMATCH, readers.ReadFailure.IDENTITY_MISMATCH),
        (ServiceFailure.EXCLUDED_CONTROLLER, readers.ReadFailure.IDENTITY_MISMATCH),
        (ServiceFailure.EXCLUDED_WORKER, readers.ReadFailure.IDENTITY_MISMATCH),
        (ServiceFailure.INVALID_SOURCE, readers.ReadFailure.UNSUPPORTED),
        (ServiceFailure.INVALID_CONTROLLER, readers.ReadFailure.MALFORMED),
        (ServiceFailure.INVALID_WORKER, readers.ReadFailure.MALFORMED),
        (ServiceFailure.INVALID_PROOF, readers.ReadFailure.MALFORMED),
    ],
)
def test_closed_proof_failures_are_not_collapsed(failure, expected):
    target = request(readers.NativeArm.OPENBROWSER_MCP)
    host, _, _ = fake_readers(target)

    def failed(pid):
        raise ServiceReaderError(failure)

    result = capture(target, replace(host, process_factory=failed))
    assert not result.eligible
    assert result.failure is expected


@pytest.mark.parametrize("field", ["after", "initial_proof", "closing_proof"])
def test_closing_contract_fields_are_required_for_eligibility(field):
    target = request(readers.NativeArm.BROWSER_USE)
    host, _, _ = fake_readers(target)
    result = capture(target, host)
    assert result.eligible
    assert not replace(result, **{field: None}).eligible


def test_a_foreign_closing_proof_cannot_complete_a_service_read():
    target = request(readers.NativeArm.BROWSER_USE)
    host, _, _ = fake_readers(target)
    result = capture(target, host)
    assert result.closing_proof is not None and result.closing_proof.proof is not None
    foreign = replace(result.closing_proof.proof, identity=ServiceIdentity(99, 99.0))
    closing = replace(result.closing_proof, proof=foreign)
    assert not replace(result, proof=closing, closing_proof=closing).eligible
