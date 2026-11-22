"""Explicit context registry state, isolation, capacity, and close semantics."""

import asyncio
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.mcp import server as module
from tests.support.config_lifecycle import startup_config


async def _event(event: asyncio.Event, label: str) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=5)
    except TimeoutError:
        pytest.fail(f"Timed out waiting for {label}", pytrace=False)


async def _settle(task: asyncio.Task[object], label: str):
    _done, pending = await asyncio.wait({task}, timeout=5)
    if pending:
        task.cancel()
        pytest.fail(f"Timed out waiting for {label}", pytrace=False)
    return task.result()


@pytest.fixture()
def registry(monkeypatch, tmp_path, isolated_config_path):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    monkeypatch.setattr(module, "_MCP_WORKSPACE_DIR", tmp_path / "workspace")
    server = module.OpenBrowserServer()
    return server, server._contexts


def test_close_result_rejects_invalid_combinations():
    with pytest.raises(ValueError):
        module.CloseResult(
            context_id="ctx", status="closed", error_code="bad", retryable=False
        )
    with pytest.raises(ValueError):
        module.CloseResult(
            context_id="ctx", status="close_failed", error_code=None, retryable=True
        )
    with pytest.raises(ValueError):
        module.CloseResult(context_id="ctx", status="unknown")


@pytest.mark.asyncio
async def test_capacity_counts_contexts_and_retry_keeps_failed_handle(registry):
    _, contexts = registry
    contexts.max_contexts = 1
    context_id = await contexts.create()
    with pytest.raises(module.ContextUnavailable) as limited:
        await contexts.create()
    assert limited.value.code == "context_limit_reached"

    entry = contexts._entries[context_id]
    attempts = 0

    async def close_owned():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("controlled cleanup failure")

    entry.owner._close_owned = close_owned
    first = await contexts.close(context_id)
    assert first == module.CloseResult(
        context_id=context_id,
        status="close_failed",
        error_code="cleanup_failed",
        retryable=True,
    )
    assert entry.phase is module.ContextPhase.CLOSE_FAILED
    assert context_id in contexts._entries

    second = await contexts.close(context_id)
    assert second == module.CloseResult(
        context_id=context_id, status="closed", error_code=None, retryable=False
    )
    assert context_id not in contexts._entries
    assert attempts == 2


@pytest.mark.asyncio
async def test_close_and_close_all_join_one_owned_cleanup(registry):
    _, contexts = registry
    context_id = await contexts.create()
    entry = contexts._entries[context_id]
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def close_owned():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()

    entry.owner._close_owned = close_owned
    close_task = asyncio.create_task(contexts.close(context_id))
    await _event(started, "owned cleanup start")
    close_all_task = asyncio.create_task(contexts.close_all())
    await asyncio.sleep(0)
    assert calls == 1
    assert contexts.admission is module.ServerAdmission.STOPPING

    release.set()
    close_result = await _settle(close_task, "close result")
    close_all_results = await _settle(close_all_task, "close_all result")
    assert close_result.status == "closed"
    assert [result.status for result in close_all_results] == ["closed"]
    assert calls == 1
    assert contexts.admission is module.ServerAdmission.STOPPED


@pytest.mark.asyncio
async def test_acquire_rejects_a_context_that_is_closing(registry):
    _, contexts = registry
    context_id = await contexts.create()
    entry = contexts._entries[context_id]
    started = asyncio.Event()
    release = asyncio.Event()

    async def close_owned():
        started.set()
        await release.wait()

    entry.owner._close_owned = close_owned
    close_task = asyncio.create_task(contexts.close(context_id))
    try:
        await _event(started, "owned cleanup start")
        with pytest.raises(module.ContextUnavailable) as closing:
            await contexts.acquire(context_id)
        assert closing.value.code == "context_closing"
    finally:
        release.set()
    assert (await _settle(close_task, "close result")).status == "closed"


@pytest.mark.asyncio
async def test_legacy_owner_uses_the_same_capacity_and_lease_path(registry):
    _, contexts = registry
    contexts.max_contexts = 1
    lease = await contexts.acquire_legacy()
    with pytest.raises(module.ContextUnavailable) as limited:
        await contexts.create()
    assert limited.value.code == "context_limit_reached"
    await lease.__aexit__(None, None, None)
    # The legacy owner stays open until shutdown, so it still owns the slot.
    with pytest.raises(module.ContextUnavailable) as still_limited:
        await contexts.create()
    assert still_limited.value.code == "context_limit_reached"
    await contexts.close_all()
    assert contexts._legacy_entry is None


@pytest.mark.asyncio
async def test_workspace_survives_close_and_owner_ids_do_not_reuse(registry, tmp_path):
    _, contexts = registry
    first_id = await contexts.create()
    first = contexts._entries[first_id].owner
    first._workspace_dir.mkdir(parents=True, exist_ok=True)
    retained = first._workspace_dir / "keep.txt"
    retained.write_text("keep", encoding="utf-8")

    assert (await contexts.close(first_id)).status == "closed"
    assert retained.read_text(encoding="utf-8") == "keep"

    second_id = await contexts.create()
    second = contexts._entries[second_id].owner
    assert second_id != first_id
    assert second.owner_id != first.owner_id
    assert second._workspace_dir != first._workspace_dir


@pytest.mark.asyncio
async def test_server_output_lock_serializes_execute_code_across_contexts(registry):
    _, contexts = registry
    first_id = await contexts.create()
    second_id = await contexts.create()
    first = contexts._entries[first_id].owner
    second = contexts._entries[second_id].owner
    active = 0
    maximum = 0

    async def fake_execute(self, code, *, deadline=None):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.02)
        active -= 1
        return code

    first._execute_code_locked = MethodType(fake_execute, first)
    second._execute_code_locked = MethodType(fake_execute, second)
    await asyncio.gather(first._execute_code("a"), second._execute_code("b"))
    assert maximum == 1


@pytest.mark.asyncio
async def test_shutdown_reports_failed_context_ids_and_marks_admission_failed(registry):
    server, contexts = registry
    context_id = await contexts.create()
    entry = contexts._entries[context_id]

    async def close_owned():
        raise RuntimeError("controlled shutdown failure")

    entry.owner._close_owned = close_owned
    with pytest.raises(RuntimeError, match=context_id):
        await server._shutdown()
    assert contexts.admission is module.ServerAdmission.FAILED


@pytest.mark.asyncio
async def test_close_drains_an_admitted_call_before_cleanup(registry):
    _, contexts = registry
    context_id = await contexts.create()
    entry = contexts._entries[context_id]
    cleanup_started = asyncio.Event()

    async def close_owned():
        cleanup_started.set()

    entry.owner._close_owned = close_owned
    lease = await contexts.acquire(context_id)
    close_task = asyncio.create_task(contexts.close(context_id))
    try:
        await _event(entry.drain_waiting, "close drain wait")
        assert not cleanup_started.is_set()
    finally:
        contexts.release(lease.entry)
    assert (await _settle(close_task, "close result")).status == "closed"
    assert cleanup_started.is_set()


@pytest.mark.asyncio
async def test_cancelled_lease_holder_releases_on_unwind(registry):
    _, contexts = registry
    context_id = await contexts.create()
    entry = contexts._entries[context_id]
    lease = await contexts.acquire(context_id)
    started = asyncio.Event()

    async def holder():
        async with lease:
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(holder())
    await _event(started, "lease holder start")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await _settle(task, "cancelled lease holder")
    assert entry.active_calls == 0
    assert entry.drained.is_set()


@pytest.mark.asyncio
async def test_stop_admission_rejects_create_and_acquire(registry):
    _, contexts = registry
    context_id = await contexts.create()
    await contexts.stop_admission()
    with pytest.raises(module.ContextUnavailable) as create_error:
        await contexts.create()
    assert create_error.value.code == "server_stopping"
    with pytest.raises(module.ContextUnavailable) as acquire_error:
        await contexts.acquire(context_id)
    assert acquire_error.value.code == "server_stopping"


@pytest.mark.asyncio
async def test_shutdown_waits_for_admitted_lease_and_rejects_new_acquire(registry):
    server, contexts = registry
    context_id = await contexts.create()
    entry = contexts._entries[context_id]
    cleanup_started = asyncio.Event()

    async def close_owned():
        cleanup_started.set()

    entry.owner._close_owned = close_owned
    lease = await contexts.acquire(context_id)
    shutdown_task = asyncio.create_task(server._shutdown())
    for _ in range(100):
        if contexts.admission is module.ServerAdmission.STOPPING:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("shutdown never fenced admission", pytrace=False)
    assert not cleanup_started.is_set()
    with pytest.raises(module.ContextUnavailable) as acquire_error:
        await contexts.acquire(context_id)
    assert acquire_error.value.code == "server_stopping"
    contexts.release(lease.entry)
    await _settle(shutdown_task, "server shutdown")
    assert cleanup_started.is_set()
    assert contexts.admission is module.ServerAdmission.STOPPED


@pytest.mark.asyncio
async def test_legacy_fail_once_cleanup_retry_retires_entry(registry):
    server, contexts = registry
    lease = await contexts.acquire_legacy()
    await lease.__aexit__(None, None, None)
    calls = 0

    async def shutdown_owner():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("controlled first cleanup failure")
        server.browser_session = None
        server._namespace = None

    server._shutdown_owner_locked = shutdown_owner
    server.browser_session = object()
    server._namespace = {"value": 1}
    await server._shutdown_locked()
    assert calls == 2
    assert contexts._legacy_entry is None
    assert contexts.admission is module.ServerAdmission.STOPPED


@pytest.mark.asyncio
async def test_context_profiles_and_downloads_are_isolated(registry, monkeypatch, tmp_path):
    _, contexts = registry
    monkeypatch.setattr(
        module,
        "CONFIG",
        SimpleNamespace(OPENBROWSER_PROFILES_DIR=tmp_path / "profiles"),
    )
    first_id = await contexts.create()
    second_id = await contexts.create()
    first = contexts._entries[first_id].owner
    second = contexts._entries[second_id].owner
    first_profile = first._get_managed_profile_dir()
    second_profile = second._get_managed_profile_dir()
    assert first_profile != second_profile
    assert first_profile.name == f"mcp-{first.owner_id}"
    assert second_profile.name == f"mcp-{second.owner_id}"
    assert first._downloads_dir != second._downloads_dir
    assert first._downloads_dir.name == first.owner_id
    assert second._downloads_dir.name == second.owner_id


@pytest.mark.asyncio
async def test_process_output_lock_is_shared_between_servers(registry):
    first_server, _ = registry
    second_server = module.OpenBrowserServer()

    async def fake_execute(self, code, *, deadline=None):
        return code

    first_server._execute_code_locked = MethodType(fake_execute, first_server)
    second_server._execute_code_locked = MethodType(fake_execute, second_server)
    await asyncio.gather(
        first_server._execute_code("a"), second_server._execute_code("b")
    )
    assert first_server._output_lock is second_server._output_lock


@pytest.mark.asyncio
async def test_execution_times_out_waiting_for_the_shared_output_lock(
    registry, monkeypatch
):
    _, contexts = registry
    context_id = await contexts.create()
    owner = contexts._entries[context_id].owner
    lock = module._process_output_lock()
    await lock.acquire()
    monkeypatch.setenv("OPENBROWSER_MCP_EXEC_TIMEOUT", "0.1")
    try:
        result = await asyncio.wait_for(owner._execute_code("x = 1"), timeout=2)
    finally:
        lock.release()
    assert "shared output lock" in result
    assert owner._execution_lock.locked() is False


@pytest.mark.asyncio
async def test_recovery_cancellation_reports_stopped_and_keeps_variables(registry):
    _, contexts = registry
    context_id = await contexts.create()
    owner = contexts._entries[context_id].owner
    cancellation = asyncio.CancelledError("caller cancelled")
    owner.browser_session = MagicMock()
    owner._namespace = {"answer": 42}

    async def kill(_session):
        return module._CleanupOutcome(True, cancellation)

    owner._kill_session_safely = kill
    with pytest.raises(asyncio.CancelledError) as raised:
        await owner._recover_browser_session_locked()
    assert raised.value is cancellation
    assert owner._browser_lifecycle.phase is LifecyclePhase.STOPPED
    assert owner._pending_variables == {"answer": 42}


@pytest.mark.asyncio
async def test_shutdown_retries_a_failed_modern_context(registry):
    server, contexts = registry
    context_id = await contexts.create()
    entry = contexts._entries[context_id]
    attempts = 0

    async def close_owned():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("controlled first cleanup failure")

    entry.owner._close_owned = close_owned
    await server._shutdown()
    assert attempts == 2
    assert context_id not in contexts._entries
    assert contexts.admission is module.ServerAdmission.STOPPED


@pytest.mark.asyncio
async def test_explicit_profile_path_is_shared_by_explicit_opt_in(
    monkeypatch, tmp_path
):
    shared = tmp_path / "shared-profile"
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    with startup_config({"browser_profile": {"user_data_dir": str(shared)}}):
        server = module.OpenBrowserServer()
        first_id = await server._contexts.create()
        second_id = await server._contexts.create()
        first = server._contexts._entries[first_id].owner
        second = server._contexts._entries[second_id].owner
        first_profile = first._build_browser_profile()
        second_profile = second._build_browser_profile()
        assert Path(first_profile.user_data_dir).resolve() == shared.resolve()
        assert Path(second_profile.user_data_dir).resolve() == shared.resolve()


@pytest.mark.asyncio
async def test_expired_deadline_does_not_start_user_code(registry):
    _, contexts = registry
    context_id = await contexts.create()
    owner = contexts._entries[context_id].owner
    result = await owner._execute_code_locked(
        "started = True", deadline=time.monotonic() - 1
    )
    assert "timed out" in result.lower()
    assert owner._namespace is None


@pytest.mark.asyncio
async def test_recovery_deadline_exhaustion_stops_after_old_cleanup(registry):
    _, contexts = registry
    context_id = await contexts.create()
    owner = contexts._entries[context_id].owner
    owner.browser_session = MagicMock()
    owner._namespace = {"answer": 42}
    owner._browser_generation = 3

    async def kill(_session):
        return module._CleanupOutcome(True)

    owner._kill_session_safely = kill
    with pytest.raises(TimeoutError):
        await owner._recover_browser_session_locked(deadline=time.monotonic() - 1)
    assert owner.browser_session is None
    assert owner._browser_lifecycle.phase is LifecyclePhase.STOPPED
    assert owner._pending_variables == {"answer": 42}
    assert owner._browser_generation == 4


@pytest.mark.asyncio
async def test_expired_health_budget_does_not_trigger_recovery(registry):
    _, contexts = registry
    context_id = await contexts.create()
    owner = contexts._entries[context_id].owner
    owner._namespace = {"__builtins__": __builtins__}
    owner.browser_session = MagicMock()

    async def slow_dead(*, deadline=None):
        await asyncio.sleep(0.02)
        return False

    async def forbid_recovery(**kwargs):
        pytest.fail("recovery ran after the execution budget expired", pytrace=False)

    owner._is_cdp_alive = slow_dead
    owner._recover_browser_session = forbid_recovery
    result = await owner._execute_code_locked(
        "x = 1", deadline=time.monotonic() + 0.01
    )
    assert "health check" in result
    assert "timed out" in result.lower()
