"""Browser-free tests for warm launch proofs and OpenBrowser adapter wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from benchmarks.interactive.adapters.base import RunContext


def proof_module():
    from benchmarks.interactive.adapters import launch_proof

    return launch_proof


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    import socket
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("native I/O is forbidden in fake tests")

    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(proof_module(), "_process_table", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


MARKER = "/tmp/launch-proof-test/config/profiles/daemon"


def context(task: str = "download") -> RunContext:
    root = Path("/tmp/launch-proof-test")
    return RunContext(
        task=task,
        arm="openbrowser-fast",
        sample=0,
        base_url="http://127.0.0.1:1",
        root=root,
        downloads=root / "downloads",
        profile=root / "profile",
        home=root / "home",
        tmp=root / "tmp",
        config=root / "config",
        session="bench",
    )


class FakeProcess:
    def __init__(self, created_at, *, running=True, state: str | None = psutil.STATUS_RUNNING):
        self.created_at = created_at
        self.running = running
        self.state = state

    def create_time(self):
        return self.created_at

    def is_running(self):
        return self.running

    def status(self):
        return self.state


class SequenceProcess(FakeProcess):
    def __init__(self, created_times):
        self.created_times = list(created_times)
        super().__init__(self.created_times[0])

    def create_time(self):
        if len(self.created_times) > 1:
            return self.created_times.pop(0)
        return self.created_times[0]


def process_factory(mapping):
    def make(pid):
        value = mapping.get(pid)
        if value is None:
            raise psutil.NoSuchProcess(pid)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, FakeProcess):
            return value
        return FakeProcess(value)

    return make


def command_table(rows):
    return lambda: list(rows)


def sequence_table(*tables):
    iterator = iter(tables)

    def next_table():
        try:
            return list(next(iterator))
        except StopIteration:
            return list(tables[-1])

    return next_table


def sample_proof(*, context_id: str | None = None):
    module = proof_module()
    return module.LaunchProof(
        server=module.ProcessIdentity(10, 1.0),
        browser=module.ProcessIdentity(20, 2.0),
        profile_marker=MARKER,
        context_id=context_id,
        keychain_verified=True,
    )


def test_capture_requires_positive_keychain_and_one_main_browser():
    module = proof_module()
    rows = [
        (10, "python3.12 -m openbrowser.daemon.server"),
        (20, "Google Chrome --user-data-dir=" + MARKER + " --use-mock-keychain"),
        (21, "Google Chrome Helper --type=renderer --user-data-dir=" + MARKER),
        (22, "chrome_crashpad_handler --user-data-dir=" + MARKER),
    ]
    factory = process_factory({10: 1.0, 20: 2.0, 21: 3.0, 22: 4.0})

    proof = module.capture_launch_proof(
        MARKER,
        10,
        keychain_verified=True,
        command_table=command_table(rows),
        process_factory=factory,
    )

    assert proof is not None
    assert proof.server == module.ProcessIdentity(10, 1.0)
    assert proof.browser == module.ProcessIdentity(20, 2.0)
    assert proof.profile_marker == MARKER
    assert proof.keychain_verified is True
    assert (
        module.capture_launch_proof(
            MARKER,
            10,
            keychain_verified=False,
            command_table=command_table(rows),
            process_factory=factory,
        )
        is None
    )
    assert (
        module.capture_launch_proof(
            MARKER,
            10,
            keychain_verified=None,
            command_table=command_table(rows),
            process_factory=factory,
        )
        is None
    )
    assert (
        module.capture_launch_proof(
            MARKER,
            10,
            keychain_verified=True,
            command_table=command_table(
                [*rows, (23, "Google Chrome --user-data-dir=" + MARKER + " --use-mock-keychain")]
            ),
            process_factory=process_factory({10: 1.0, 20: 2.0, 21: 3.0, 22: 4.0, 23: 5.0}),
        )
        is None
    )
    assert (
        module.capture_launch_proof(
            MARKER,
            10,
            keychain_verified=True,
            command_table=command_table([(10, "python3.12")]),
            process_factory=factory,
        )
        is None
    )


def test_proof_status_returns_closed_reasons():
    module = proof_module()
    proof = sample_proof(context_id="ctx")

    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: 1.0, 20: 2.0}),
        )
        == "valid"
    )
    assert (
        module.proof_status(
            proof,
            profile_marker="/tmp/other",
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: 1.0, 20: 2.0}),
        )
        == "profile_changed"
    )
    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="other",
            process_factory=process_factory({10: 1.0, 20: 2.0}),
        )
        == "context_changed"
    )
    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({20: 2.0}),
        )
        == "server_missing"
    )
    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: 9.0, 20: 2.0}),
        )
        == "server_replaced"
    )
    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: 1.0}),
        )
        == "browser_missing"
    )
    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: 1.0, 20: 9.0}),
        )
        == "browser_replaced"
    )
    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: psutil.AccessDenied(10), 20: 2.0}),
        )
        == "identity_unknown"
    )
    unproven = module.LaunchProof(
        server=proof.server,
        browser=proof.browser,
        profile_marker=proof.profile_marker,
        context_id=proof.context_id,
        keychain_verified=False,
    )
    assert (
        module.proof_status(
            unproven,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: 1.0, 20: 2.0}),
        )
        == "keychain_unproven"
    )


def test_read_pid_file_is_bounded(tmp_path):
    module = proof_module()
    path = tmp_path / "daemon.pid"

    assert module.read_pid_file(path) is None
    path.write_text("42\n")
    assert module.read_pid_file(path) == 42
    path.write_text("bad")
    assert module.read_pid_file(path) is None


def test_cli_warm_proof_hit_skips_full_keychain_scan(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser

    arm = openbrowser.OpenBrowserArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    proof = sample_proof()
    arm._launch_proof = proof
    calls = []

    def invoke(ctx, code, timeout):
        calls.append("invoke")
        arm._result.stdout += '{"ok": true}\n'

    monkeypatch.setattr(arm, "_invoke", invoke)
    monkeypatch.setattr(openbrowser, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(openbrowser, "read_pid_file", lambda path: 10)
    monkeypatch.setattr(openbrowser, "proof_status", lambda *args, **kwargs: "valid")
    monkeypatch.setattr(
        openbrowser,
        "verify_mock_keychain",
        lambda *args, **kwargs: pytest.fail("full keychain scan must not run on a proof hit"),
    )

    result = arm.run(ctx, cleanup=False)

    assert result.ok is True
    assert result.keychain_verified is True
    assert calls == ["invoke"]
    assert arm._launch_proof == proof


def test_cli_warm_proof_change_before_task_fails_without_running(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser

    arm = openbrowser.OpenBrowserArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    arm._launch_proof = sample_proof()
    calls = []

    monkeypatch.setattr(arm, "_invoke", lambda *args, **kwargs: calls.append("invoke"))
    monkeypatch.setattr(openbrowser, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(openbrowser, "read_pid_file", lambda path: 10)
    monkeypatch.setattr(openbrowser, "proof_status", lambda *args, **kwargs: "browser_missing")

    result = arm.run(ctx, cleanup=False)

    assert result.ok is False
    assert result.keychain_verified is None
    assert calls == []
    assert "launch proof invalid before task" in result.error
    assert arm._launch_proof is None


def test_cli_warm_proof_change_after_task_fails_and_never_recaches(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser

    arm = openbrowser.OpenBrowserArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    proof = sample_proof()
    arm._launch_proof = proof
    calls = []
    statuses = iter(["valid", "browser_missing"])

    def invoke(ctx, code, timeout):
        calls.append("invoke")
        arm._result.stdout += '{"ok": true}\n'

    monkeypatch.setattr(arm, "_invoke", invoke)
    monkeypatch.setattr(openbrowser, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(openbrowser, "read_pid_file", lambda path: 10)
    monkeypatch.setattr(openbrowser, "proof_status", lambda *args, **kwargs: next(statuses))

    result = arm.run(ctx, cleanup=False)

    assert result.ok is False
    assert result.keychain_verified is None
    assert calls == ["invoke"]
    assert "launch proof invalid after task" in result.error
    assert arm._launch_proof is None


def test_cli_cold_run_keeps_full_keychain_verification(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser

    arm = openbrowser.OpenBrowserArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    calls = []
    verified = []

    def invoke(ctx, code, timeout):
        calls.append("invoke")
        arm._result.stdout += '{"ok": true}\n'

    def verify(marker):
        verified.append(marker)
        return True

    monkeypatch.setattr(arm, "_invoke", invoke)
    monkeypatch.setattr(openbrowser, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(openbrowser, "verify_mock_keychain", verify)
    monkeypatch.setattr(
        openbrowser,
        "proof_status",
        lambda *args, **kwargs: pytest.fail("cold run must not use the warm proof"),
    )

    result = arm.run(ctx, cleanup=False)

    assert result.ok is True
    assert result.keychain_verified is True
    assert calls == ["invoke"]
    assert verified == [str(ctx.config / "profiles" / "daemon")]


def test_cli_warmup_captures_positive_proof(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser

    arm = openbrowser.OpenBrowserArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    proof = sample_proof()
    monkeypatch.setattr(openbrowser, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(openbrowser, "run_process", lambda *args, **kwargs: (0, '{"ok": true}', "", 0.0))
    monkeypatch.setattr(openbrowser, "verify_mock_keychain", lambda marker: True)
    monkeypatch.setattr(openbrowser, "read_pid_file", lambda path: 10)
    monkeypatch.setattr(openbrowser, "capture_launch_proof", lambda *args, **kwargs: proof)

    payload, _ = arm.warmup(ctx)

    assert payload["keychain_verified"] is True
    assert arm._launch_proof == proof


def test_cli_cleanup_and_reset_clear_proof(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser

    arm = openbrowser.OpenBrowserArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    arm._launch_proof = sample_proof()

    def run_process(command, **kwargs):
        if "reset" in command:
            return 0, '{"browser":{"phase":"stopped"}}', "", 0.0
        return 0, "", "", 0.0

    monkeypatch.setattr(openbrowser, "run_process", run_process)
    monkeypatch.setattr(openbrowser, "kill_pidfile", lambda path: None)
    monkeypatch.setattr(openbrowser, "terminate_matching", lambda marker: None)

    arm.cleanup(ctx)
    assert arm._launch_proof is None

    arm._launch_proof = sample_proof()
    arm._reset_browser(ctx)
    assert arm._launch_proof is None


def test_mcp_warm_proof_hit_skips_full_keychain_scan(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser_mcp

    arm = openbrowser_mcp.OpenBrowserMcpArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    proof = sample_proof(context_id="ctx")
    arm._launch_proof = proof
    arm._context_id = "ctx"
    client = SimpleNamespace(process=SimpleNamespace(pid=10))
    calls = []

    def execute(client, context_id, code, timeout):
        calls.append(context_id)
        return '{"ok": true}'

    monkeypatch.setattr(openbrowser_mcp, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(arm, "_start", lambda ctx: client)
    monkeypatch.setattr(arm, "_ensure_context", lambda client: "ctx")
    monkeypatch.setattr(arm, "_execute", execute)
    monkeypatch.setattr(openbrowser_mcp, "proof_status", lambda *args, **kwargs: "valid")
    monkeypatch.setattr(
        openbrowser_mcp,
        "verify_mock_keychain",
        lambda *args, **kwargs: pytest.fail("full keychain scan must not run on a proof hit"),
    )

    result = arm.run(ctx, cleanup=False)

    assert result.ok is True
    assert result.keychain_verified is True
    assert calls == ["ctx"]
    assert arm._launch_proof == proof


def test_mcp_context_change_fails_before_task(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser_mcp

    arm = openbrowser_mcp.OpenBrowserMcpArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    arm._launch_proof = sample_proof(context_id="ctx")
    arm._context_id = "other"
    client = SimpleNamespace(process=SimpleNamespace(pid=10))
    calls = []

    monkeypatch.setattr(openbrowser_mcp, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(arm, "_start", lambda ctx: client)
    monkeypatch.setattr(arm, "_ensure_context", lambda client: "other")
    monkeypatch.setattr(arm, "_execute", lambda *args, **kwargs: calls.append("execute"))
    monkeypatch.setattr(openbrowser_mcp, "proof_status", lambda *args, **kwargs: "context_changed")

    result = arm.run(ctx, cleanup=False)

    assert result.ok is False
    assert result.keychain_verified is None
    assert calls == []
    assert "launch proof invalid before task" in result.error
    assert arm._launch_proof is None


def test_mcp_warmup_captures_positive_proof(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser_mcp

    arm = openbrowser_mcp.OpenBrowserMcpArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    proof = sample_proof(context_id="ctx")
    client = SimpleNamespace(process=SimpleNamespace(pid=10))
    monkeypatch.setattr(openbrowser_mcp, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(arm, "_start", lambda ctx: client)
    monkeypatch.setattr(arm, "_create_context", lambda client: "ctx")
    monkeypatch.setattr(arm, "_execute", lambda *args, **kwargs: "pass")
    monkeypatch.setattr(openbrowser_mcp, "verify_mock_keychain", lambda marker: True)
    monkeypatch.setattr(openbrowser_mcp, "capture_launch_proof", lambda *args, **kwargs: proof)

    payload, _ = arm.warmup(ctx)

    assert payload["keychain_verified"] is True
    assert payload["context_id"] == "ctx"
    assert arm._launch_proof == proof


def test_identity_status_rejects_nonrunning_zombie_and_nonfinite():
    module = proof_module()

    assert module.identity_status(10, process_factory({10: FakeProcess(1.0, running=False)}))[1] == "missing"
    assert (
        module.identity_status(
            10,
            process_factory({10: FakeProcess(1.0, state=psutil.STATUS_ZOMBIE)}),
        )[1]
        == "missing"
    )
    assert (
        module.identity_status(
            10,
            process_factory({10: FakeProcess(1.0, state=psutil.STATUS_DEAD)}),
        )[1]
        == "missing"
    )
    assert module.identity_status(10, process_factory({10: FakeProcess(1.0, state=None)}))[1] == "unknown"
    assert module.identity_status(10, process_factory({10: FakeProcess(1.0, state="mystery")}))[1] == "unknown"
    for value in (float("nan"), float("inf"), 0.0, -1.0):
        assert module.identity_status(10, process_factory({10: FakeProcess(value)}))[1] == "unknown"


def test_main_browser_identity_requires_mock_flag_and_stable_table():
    module = proof_module()
    flagless = [(20, "Google Chrome --user-data-dir=" + MARKER)]
    flagged = [(20, "Google Chrome --user-data-dir=" + MARKER + " --use-mock-keychain")]
    replacement = [(21, "Google Chrome --user-data-dir=" + MARKER + " --use-mock-keychain")]

    assert (
        module.main_browser_identity(
            MARKER,
            command_table=command_table(flagless),
            process_factory=process_factory({20: 2.0}),
        )
        is None
    )
    assert (
        module.main_browser_identity(
            MARKER,
            command_table=sequence_table(flagged, replacement),
            process_factory=process_factory({20: 2.0, 21: 3.0}),
        )
        is None
    )
    assert (
        module.main_browser_identity(
            MARKER,
            command_table=command_table(flagged),
            process_factory=process_factory({20: SequenceProcess([2.0, 9.0])}),
        )
        is None
    )


def test_main_browser_identity_accepts_helper_in_profile_path():
    module = proof_module()
    marker = "/tmp/helper-run/profiles/daemon"
    rows = [(20, "Google Chrome --user-data-dir=" + marker + " --use-mock-keychain")]

    identity = module.main_browser_identity(
        marker,
        command_table=command_table(rows),
        process_factory=process_factory({20: 2.0}),
    )

    assert identity == module.ProcessIdentity(20, 2.0)


def test_capture_rejects_server_identity_change_after_browser_capture():
    module = proof_module()
    rows = [(20, "Google Chrome --user-data-dir=" + MARKER + " --use-mock-keychain")]
    factory = process_factory({10: SequenceProcess([1.0, 9.0]), 20: 2.0})

    assert (
        module.capture_launch_proof(
            MARKER,
            10,
            keychain_verified=True,
            command_table=command_table(rows),
            process_factory=factory,
        )
        is None
    )


def test_proof_status_treats_zombie_and_nonfinite_identity_as_missing_or_unknown():
    module = proof_module()
    proof = sample_proof(context_id="ctx")

    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: 1.0, 20: FakeProcess(2.0, state=psutil.STATUS_ZOMBIE)}),
        )
        == "browser_missing"
    )
    assert (
        module.proof_status(
            proof,
            profile_marker=MARKER,
            server_pid=10,
            context_id="ctx",
            process_factory=process_factory({10: FakeProcess(float("nan")), 20: 2.0}),
        )
        == "identity_unknown"
    )


def test_capture_rejects_browser_identity_change_after_capture():
    module = proof_module()
    rows = [(20, "Google Chrome --user-data-dir=" + MARKER + " --use-mock-keychain")]
    factory = process_factory({10: 1.0, 20: SequenceProcess([2.0, 2.0, 9.0])})

    assert (
        module.capture_launch_proof(
            MARKER,
            10,
            keychain_verified=True,
            command_table=command_table(rows),
            process_factory=factory,
        )
        is None
    )


def test_mcp_cleanup_run_clears_warm_proof_before_fresh_context(monkeypatch):
    from benchmarks.interactive.adapters import openbrowser_mcp

    arm = openbrowser_mcp.OpenBrowserMcpArm(binary="/fake/openbrowser", fast_selectors=True)
    ctx = context()
    arm._launch_proof = sample_proof(context_id="ctx")
    arm._context_id = "ctx"
    client = SimpleNamespace(process=SimpleNamespace(pid=10), close=lambda: None)
    calls = []

    monkeypatch.setattr(openbrowser_mcp, "_write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(arm, "_start", lambda ctx: client)
    monkeypatch.setattr(arm, "_create_context", lambda client: "fresh")
    monkeypatch.setattr(arm, "_execute", lambda *args, **kwargs: calls.append("execute") or '{"ok": true}')
    monkeypatch.setattr(arm, "_close_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(arm, "_mock_keychain_profile", lambda ctx: MARKER)
    monkeypatch.setattr(openbrowser_mcp, "verify_mock_keychain", lambda marker: True)
    monkeypatch.setattr(openbrowser_mcp, "terminate_matching", lambda marker: None)
    monkeypatch.setattr(
        openbrowser_mcp,
        "proof_status",
        lambda *args, **kwargs: pytest.fail("expected cleanup run must not use the warm proof"),
    )

    result = arm.run(ctx, cleanup=True)

    assert result.ok is True
    assert result.keychain_verified is True
    assert calls == ["execute"]
    assert arm._launch_proof is None


def test_fake_guard_blocks_host_process_lookup():
    with pytest.raises(AssertionError, match="native I/O"):
        psutil.Process(1)
    with pytest.raises(AssertionError, match="native I/O"):
        proof_module()._process_table()
