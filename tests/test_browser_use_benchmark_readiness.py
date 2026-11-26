"""Offline contracts for native browser-use readiness and warm preparation."""

from __future__ import annotations

import io
import json
import re
import socket
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.interactive.adapters import browser_use as adapter
from benchmarks.interactive.adapters.base import ArmResult, RunContext
from benchmarks.interactive.tasks import oracle_tabs


@pytest.fixture(autouse=True)
def no_live_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline test attempted a process, socket or browser binding")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(adapter, "run_process", forbidden)
    monkeypatch.setattr(adapter, "verify_mock_keychain", forbidden)
    monkeypatch.setattr(adapter.ChromeInstance, "start", forbidden)
    monkeypatch.setattr(adapter.ChromeInstance, "stop", forbidden)
    monkeypatch.setattr(adapter, "kill_pidfile", forbidden)
    monkeypatch.setattr(adapter, "terminate_matching", forbidden)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, duration):
        assert 0 < duration <= 0.1
        self.sleeps.append(duration)
        self.now += duration


class NativeTabs:
    """Model the native surface, not the private fixture endpoints."""

    def __init__(self, compare=None, approve=None):
        self.active = "old-tab"
        self.trace = []
        self.detail_open = False
        self.approved = False
        self.compare = list(compare or [[10, 20]])
        self.approve = list(approve or [[30, 40]])
        self.state = {"tab_events": []}

    def new_tab(self, url):
        self.trace.append(("new_tab", url))
        # Deliberately leave current_tab stale. The caller must bind the return ID.
        return "original"

    def switch_tab(self, target):
        self.active = target
        self.trace.append(("switch", target))

    def js(self, expression, target_id=None):
        target = target_id or self.active
        self.trace.append(("js", target, expression))
        if "getBoundingClientRect" in expression:
            selector = "#compare" if "#compare" in expression else "#approve"
            assert target == ("original" if selector == "#compare" else "detail")
            values = self.compare if selector == "#compare" else self.approve
            value = values.pop(0) if len(values) > 1 else values[0]
            if isinstance(value, Exception):
                raise value
            return value
        if "#approve" in expression and ".click()" in expression:
            assert target == "detail"
            self.approved = True
            self.state["tab_events"].extend([{"event": "approve"}, {"event": "status", "status": "approved:42"}])
            return None
        if "approved').textContent" in expression:
            assert target == "detail"
            return self.approved
        if "status').textContent ===" in expression:
            assert target == "original"
            return self.approved
        if expression == "document.title":
            assert target == "original"
            return "Tabs Bench"
        if expression == "document.getElementById('status').textContent":
            return "approved:42" if self.approved else "pending"
        raise AssertionError(f"unexpected native expression: {expression}")

    def click_at_xy(self, x, y):
        assert self.active == "original"
        assert (x, y) == (10, 20)
        self.trace.append(("click", x, y))
        self.detail_open = True
        self.state["tab_events"].append({"event": "detail_loaded", "opener": True})

    def list_tabs(self):
        return [{"targetId": "detail", "url": "https://fixture.invalid/detail"}] if self.detail_open else []

    def close_tab(self, target):
        self.trace.append(("close", target))
        if target == "detail":
            self.state["tab_events"].append({"event": "detail_closed"})

    def namespace(self):
        names = ("new_tab", "switch_tab", "js", "click_at_xy", "list_tabs", "close_tab")
        return {name: getattr(self, name) for name in names}


def execute_tabs(native, monkeypatch):
    clock = Clock()
    real_program = adapter._tabs("https://fixture.invalid/tabs")
    # The generated program imports time. Replace only that imported module in
    # its namespace, without changing the source under test or global time.
    namespace = native.namespace()
    namespace["current_tab"] = lambda: {"targetId": native.active}
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        return clock if name == "time" else real_import(name, *args, **kwargs)

    namespace["__builtins__"] = {**vars(__import__("builtins")), "__import__": fake_import}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(real_program, "<browser-use-tabs>", "exec"), namespace)
    payloads = [json.loads(line.removeprefix("BENCH_RESULT ")) for line in output.getvalue().splitlines()]
    return payloads[-1], clock, namespace


def test_tabs_resolves_returned_target_before_any_dom_access(monkeypatch):
    native = NativeTabs()
    payload, clock, _ = execute_tabs(native, monkeypatch)
    assert native.trace[1] == ("switch", "original")
    assert payload["status"] == "approved:42"
    assert oracle_tabs(native.state, payload).ok
    assert not clock.sleeps


def test_missing_geometry_retries_without_skipping_click(monkeypatch):
    native = NativeTabs(compare=[None, None, [10, 20]], approve=[None, [30, 40]])
    payload, clock, _ = execute_tabs(native, monkeypatch)
    assert oracle_tabs(native.state, payload).ok
    assert len(clock.sleeps) == 3
    assert sum(event[0] == "click" for event in native.trace) == 1
    assert payload["readiness_retries"] == 3


def test_readiness_retries_do_not_change_logical_task_call_count(monkeypatch):
    immediate, _, _ = execute_tabs(NativeTabs(), monkeypatch)
    delayed, _, _ = execute_tabs(NativeTabs(compare=[None, [10, 20]]), monkeypatch)
    assert delayed["tool_calls"] == immediate["tool_calls"]
    assert immediate["readiness_checks"] == 2  # Compare and approve both succeed on their first read.
    assert delayed["readiness_checks"] == 3
    assert immediate["readiness_retries"] == 0
    assert delayed["readiness_retries"] == 1
    assert delayed["readiness_retries"] == immediate["readiness_retries"] + 1
    assert delayed["readiness_checks"] == immediate["readiness_checks"] + 1


@pytest.mark.parametrize("selector", ["compare", "approve"])
def test_readiness_deadline_fails_without_success(selector, monkeypatch):
    native = NativeTabs(**{selector: [None]})
    with pytest.raises(RuntimeError, match=f"timeout.*#{selector}.*retries="):
        execute_tabs(native, monkeypatch)
    if selector == "compare":
        assert not native.detail_open
    assert not native.approved


@pytest.mark.parametrize("message", ["Execution context was destroyed.", "Cannot find context with specified id"])
def test_only_named_context_transitions_retry(message, monkeypatch):
    native = NativeTabs(compare=[RuntimeError(message), [10, 20]])
    payload, clock, _ = execute_tabs(native, monkeypatch)
    assert oracle_tabs(native.state, payload).ok
    assert payload["readiness_retries"] == 1
    assert payload["last_readiness_error"] == message
    assert len(clock.sleeps) == 1


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Target closed"),
        RuntimeError("Protocol error: disconnected"),
        RuntimeError("Permission denied"),
        RuntimeError("unrelated failure"),
        RuntimeError("JavaScript failed: Cannot find context with specified id in user code"),
        ValueError("Execution context was destroyed."),
    ],
)
def test_terminal_errors_propagate_without_retry(error, monkeypatch):
    native = NativeTabs(compare=[error])
    with pytest.raises(type(error), match=re.escape(str(error))):
        execute_tabs(native, monkeypatch)
    assert not native.detail_open


def test_single_deadline_survives_geometry_and_context_retries():
    clock = Clock()
    namespace: dict[str, Any] = {"js": lambda *args, **kwargs: None}
    exec(adapter.HEADER + adapter.TABS_READINESS, namespace)
    namespace["_time"] = clock
    with pytest.raises(RuntimeError, match="timeout"):
        namespace["ready_box"]("#compare", "original", timeout=0.25)
    assert clock.now == 0.25
    assert clock.sleeps == [0.1, 0.1, 0.04999999999999999]


def test_timeout_reports_last_predicate_and_final_context_error():
    namespace = {}
    clock = Clock()
    attempts = iter([None, RuntimeError("Execution context was destroyed.")])

    def js(*args, **kwargs):
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    exec(adapter.HEADER + adapter.TABS_READINESS, namespace)
    namespace.update(js=js, _time=clock)
    with pytest.raises(RuntimeError, match=r"retries=2.*last_predicate=None.*Execution context was destroyed"):
        namespace["ready_box"]("#compare", "original", timeout=0.2)


@pytest.mark.parametrize("code, retries", [(-32000, True), (-32601, False)])
def test_native_dict_context_errors_require_known_code(code, retries, monkeypatch):
    error = RuntimeError({"code": code, "message": "Cannot find context with specified id"})
    native = NativeTabs(compare=[error, [10, 20]])
    if retries:
        payload, _, _ = execute_tabs(native, monkeypatch)
        assert oracle_tabs(native.state, payload).ok
        assert payload["readiness_retries"] == 1
    else:
        with pytest.raises(RuntimeError, match="-32601"):
            execute_tabs(native, monkeypatch)


def test_geometry_return_after_deadline_does_not_click():
    namespace = {}
    clock = Clock()

    def js(*args, **kwargs):
        clock.now += 0.3
        return [10, 20]

    exec(adapter.HEADER + adapter.TABS_READINESS, namespace)
    namespace.update(js=js, _time=clock)
    with pytest.raises(RuntimeError, match=r"timeout.*last_predicate=\[10, 20\]"):
        namespace["ready_box"]("#compare", "original", timeout=0.2)


def test_atomic_geometry_checks_visibility_and_positive_dimensions():
    namespace: dict[str, Any] = {"js": lambda expression, **kwargs: expressions.append(expression)}
    expressions = []
    clock = Clock()
    exec(adapter.HEADER + adapter.TABS_READINESS, namespace)
    namespace["_time"] = clock
    with pytest.raises(RuntimeError, match="timeout"):
        namespace["ready_box"]("#compare", "original", timeout=0.1)
    expression = expressions[0]
    assert "if(!e)return null" in expression
    assert "getComputedStyle" in expression
    assert "checkVisibility" in expression
    assert "r.width>0" in expression and "r.height>0" in expression


def ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        task="wizard",
        arm="browser-use",
        sample=0,
        base_url="http://fixture.invalid",
        root=tmp_path,
        downloads=tmp_path / "downloads",
        profile=tmp_path / "profile",
        home=tmp_path / "home",
        tmp=tmp_path / "tmp",
        config=tmp_path / "config",
        session="fake",
    )


def fake_chrome(arm, events):
    events.append("chrome")
    arm._chrome = SimpleNamespace(cdp_url="http://127.0.0.1:1")
    arm._warm_keychain_verified = True


def test_warmup_exercises_native_interface_and_resets_counters(monkeypatch, tmp_path):
    arm = adapter.BrowserUseArm()
    events = []
    programs = []
    monkeypatch.setattr(arm, "_start_chrome", lambda context: fake_chrome(arm, events))

    def run(command, *, input_text, **kwargs):
        events.append("native")
        programs.append(input_text)
        namespace = {"list_tabs": lambda: [{"targetId": "blank"}]}
        output = io.StringIO()
        with redirect_stdout(output):
            exec(input_text, namespace)
        return 0, output.getvalue(), "", 0.0

    monkeypatch.setattr(adapter, "run_process", run)
    payload, _ = arm.warmup(ctx(tmp_path))
    assert events == ["chrome", "native"]
    assert payload["execution_ready"] is True
    assert payload["adapter_version"] == adapter.ADAPTER_VERSION
    assert "list_tabs()" in programs[0]
    assert all(token not in programs[0] for token in ("http", "new_tab", "wizard", "download", "#compare", "#approve"))
    assert arm._result == ArmResult(ok=False)


@pytest.mark.parametrize(
    "output",
    [
        "",
        "BENCH_RESULT {}",
        'BENCH_RESULT {"execution_ready": false}',
        'BENCH_RESULT {"execution_ready": true, "sentinel": "wrong"}',
    ],
)
def test_missing_or_wrong_warmup_sentinel_fails(output, monkeypatch, tmp_path):
    arm = adapter.BrowserUseArm()
    monkeypatch.setattr(arm, "_start_chrome", lambda context: fake_chrome(arm, []))
    monkeypatch.setattr(adapter, "run_process", lambda *args, **kwargs: (0, output, "", 0.0))
    cleanups = []
    monkeypatch.setattr(arm, "cleanup", lambda context: cleanups.append(context))
    with pytest.raises(RuntimeError, match=r"warmup.*sentinel"):
        arm.warmup(ctx(tmp_path))
    assert len(cleanups) == 1


def test_warmup_native_failure_propagates(monkeypatch, tmp_path):
    arm = adapter.BrowserUseArm()
    monkeypatch.setattr(arm, "_start_chrome", lambda context: fake_chrome(arm, []))
    monkeypatch.setattr(adapter, "run_process", lambda *args, **kwargs: (1, "", "native failed", 0.0))
    monkeypatch.setattr(arm, "cleanup", lambda context: None)
    with pytest.raises(RuntimeError, match="native failed"):
        arm.warmup(ctx(tmp_path))


def test_warmup_cleanup_error_does_not_hide_native_error(monkeypatch, tmp_path):
    arm = adapter.BrowserUseArm()
    monkeypatch.setattr(arm, "_start_chrome", lambda context: fake_chrome(arm, []))
    monkeypatch.setattr(adapter, "run_process", lambda *args, **kwargs: (1, "", "native failed", 0.0))

    def cleanup(context):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(arm, "cleanup", cleanup)
    with pytest.raises(RuntimeError, match="native failed"):
        arm.warmup(ctx(tmp_path))


@pytest.mark.parametrize("size", [2812, 100000])
def test_stderr_retains_final_exception_and_explicit_byte_metadata(size, monkeypatch, tmp_path):
    context = ctx(tmp_path)
    arm = adapter.BrowserUseArm()
    monkeypatch.setattr(arm, "_chrome", SimpleNamespace(cdp_url="http://127.0.0.1:1"))
    stderr = "x" * size + "\nFINAL EXCEPTION"
    monkeypatch.setattr(adapter, "run_process", lambda *args, **kwargs: (1, "", stderr, 0.0))
    with pytest.raises(RuntimeError, match="FINAL EXCEPTION") as error:
        arm._script(context, "pass", 1)
    assert arm._result.stderr.endswith("FINAL EXCEPTION")
    assert f"stderr_total_bytes={len(stderr.encode())}" in str(error.value)
    assert ("stderr_truncated=True" in str(error.value)) == (size > adapter.STDERR_TAIL_BYTES)
    assert len(arm._result.stderr.encode()) <= adapter.STDERR_TAIL_BYTES


def test_diagnostics_redact_paths_before_record_and_error(monkeypatch, tmp_path):
    context = ctx(tmp_path)
    arm = adapter.BrowserUseArm()
    monkeypatch.setattr(arm, "_chrome", SimpleNamespace(cdp_url="http://127.0.0.1:1"))
    private = f"{Path.home()}/Documents/repo/.worktrees/private/test.py {context.tmp}/secret {context.home}/secret"
    monkeypatch.setattr(adapter, "run_process", lambda *args, **kwargs: (1, "", private, 0.0))
    with pytest.raises(RuntimeError) as error:
        arm._script(context, "pass", 1)
    for value in (arm._result.stderr, str(error.value)):
        assert str(Path.home()) not in value
        assert str(context.tmp) not in value
        assert str(context.home) not in value
        assert ".worktrees" not in value


def test_adapter_method_version_is_in_success_payload(monkeypatch, tmp_path):
    arm = adapter.BrowserUseArm()
    monkeypatch.setattr(arm, "_chrome", SimpleNamespace(cdp_url="http://127.0.0.1:1"))
    monkeypatch.setattr(adapter, "run_process", lambda *args, **kwargs: (0, 'BENCH_RESULT {"tool_calls": 3}', "", 0.0))
    payload = arm._script(ctx(tmp_path), "pass", 1)
    assert payload["adapter_version"] == adapter.ADAPTER_VERSION
    assert arm._result.tool_calls == 3


def test_cumulative_stderr_byte_count_remains_raw_and_utf8_safe(monkeypatch, tmp_path):
    arm = adapter.BrowserUseArm()
    monkeypatch.setattr(arm, "_chrome", SimpleNamespace(cdp_url="http://127.0.0.1:1"))
    stderr = "界" * 20000 + "final exception"
    monkeypatch.setattr(adapter, "run_process", lambda *args, **kwargs: (0, "BENCH_RESULT {}", stderr, 0.0))
    first = arm._script(ctx(tmp_path), "pass", 1)
    second = arm._script(ctx(tmp_path), "pass", 1)
    assert first["stderr_total_bytes"] == len(stderr.encode())
    assert second["stderr_total_bytes"] == 2 * len(stderr.encode())
    assert second["stderr_truncated"] is True
    assert len(arm._result.stderr.encode()) <= adapter.STDERR_TAIL_BYTES
    assert "�" not in arm._result.stderr
    assert arm._result.stderr.endswith("final exception")


@pytest.mark.parametrize(
    "program,selector",
    [
        (adapter._wizard, "#name"),
        (adapter._modal, "#open-modal"),
        (adapter._download, "#download-report"),
        (adapter._login, "#username"),
        (adapter._read_account, "#secret"),
    ],
)
@pytest.mark.parametrize("delayed", [False, True])
def test_program_waits_for_first_control_on_returned_target(program, selector, delayed):
    class ActionReached(Exception):
        pass

    clock = Clock()
    outcomes = [RuntimeError("Execution context was destroyed."), None, [10, 20]] if delayed else [[10, 20]]
    checks = []
    action = []

    def first_action(*args, **kwargs):
        assert checks and checks[-1] == [10, 20], "action ran before the navigation control was ready"
        action.append(True)
        raise ActionReached

    def js(expression, target_id=None):
        if "getBoundingClientRect" not in expression:
            return first_action(expression)
        assert target_id == "new-target"
        assert json.dumps(selector) in expression
        value = outcomes.pop(0)
        checks.append(value)
        if isinstance(value, Exception):
            raise value
        return value

    real_import = __import__
    namespace = {
        "new_tab": lambda url: "new-target",
        "js": js,
        "fill_input": first_action,
        "__builtins__": {
            **vars(__import__("builtins")),
            "__import__": lambda name, *args, **kwargs: clock if name == "time" else real_import(name, *args, **kwargs),
        },
    }
    with pytest.raises(ActionReached):
        exec(compile(program("https://fixture.invalid/task"), "<native-readiness>", "exec"), namespace)
    assert action == [True]
    assert namespace["_calls"] == 3
    assert len(clock.sleeps) == (2 if delayed else 0)


def test_initial_readiness_waits_for_document_parse_with_visible_geometry():
    clock = Clock()
    checks = []
    states = ["loading", "interactive"]

    def js(expression, target_id=None):
        assert target_id == "new-target"
        checks.append(expression)
        state = states.pop(0)
        if "document.readyState==='loading'" in expression and state == "loading":
            return None
        return [10, 20]

    real_import = __import__
    namespace = {
        "new_tab": lambda url: "new-target",
        "js": js,
        "__builtins__": {
            **vars(__import__("builtins")),
            "__import__": lambda name, *args, **kwargs: clock if name == "time" else real_import(name, *args, **kwargs),
        },
    }
    exec(
        compile(adapter._ready_header("https://fixture.invalid/modal", "#open-modal"), "<parsed-document>", "exec"),
        namespace,
    )
    assert len(checks) == 2
    assert clock.sleeps == [0.1]
    assert all("document.readyState==='loading'" in expression for expression in checks)


def test_auth_keeps_method_and_stderr_provenance_after_restart(monkeypatch, tmp_path):
    from benchmarks.interactive.fixtures.server import ACCOUNT_SECRET

    arm = adapter.BrowserUseArm()
    monkeypatch.setattr(arm, "_chrome", SimpleNamespace(process=None, cdp_url="http://127.0.0.1:1"))
    monkeypatch.setattr(arm, "_warm_keychain_verified", True)
    monkeypatch.setattr(arm, "_stop_chrome", lambda: None)
    monkeypatch.setattr(arm, "_start_chrome", lambda context: None)
    calls = []

    def script(context, code, timeout):
        calls.append(code)
        arm._result.reported.update(
            adapter_version=adapter.ADAPTER_VERSION,
            stderr_total_bytes=42 + len(calls),
            stderr_truncated=False,
        )
        return {"secret": ACCOUNT_SECRET}

    monkeypatch.setattr(arm, "_script", script)
    context = ctx(tmp_path)
    context.task = "auth"
    result = arm.run(context, cleanup=False)
    assert result.ok is True
    assert len(calls) == 2
    assert result.reported["adapter_version"] == adapter.ADAPTER_VERSION
    assert result.reported["stderr_total_bytes"] == 44
    assert result.reported["stderr_truncated"] is False
    assert result.reported["secret"] == ACCOUNT_SECRET
    assert result.reported["phase1_secret"] == ACCOUNT_SECRET
