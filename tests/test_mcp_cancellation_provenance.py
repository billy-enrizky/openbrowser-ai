"""Cancellation provenance and ownership contracts at the MCP cleanup boundary."""

import asyncio
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest

from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.mcp import server as module

_EVENT_TIMEOUT_SECONDS = 10.0


async def _event(event: asyncio.Event, label: str) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=_EVENT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        pytest.fail(f"Timed out waiting for {label}", pytrace=False)


async def _settle(task: asyncio.Task[object], label: str):
    try:
        return await asyncio.wait_for(task, timeout=_EVENT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        pytest.fail(f"Timed out waiting for {label}", pytrace=False)


def test_cleanup_outcome_is_named_and_frozen():
    cancellation = asyncio.CancelledError("original")
    outcome = module._CleanupOutcome(succeeded=False, caller_cancellation=cancellation)
    assert outcome.succeeded is False
    assert outcome.caller_cancellation is cancellation
    with pytest.raises(FrozenInstanceError):
        outcome.succeeded = True


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
async def test_cleanup_retains_first_observed_cancellation_after_repeated_cancel(
    mcp_server, monkeypatch, fails
):
    started = asyncio.Event()
    release = asyncio.Event()
    observed = []
    seen = [asyncio.Event(), asyncio.Event()]
    shield = asyncio.shield

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            seen[min(len(observed), 2) - 1].set()
            raise

    async def kill():
        started.set()
        await _event(release, "worker release")
        if fails:
            raise RuntimeError("owned kill failed")

    monkeypatch.setattr(module.asyncio, "shield", capture)
    session = MagicMock()
    session.kill = kill
    task = asyncio.create_task(mcp_server._kill_session_safely(session))
    await _event(started, "worker start")
    task.cancel("first")
    await _event(seen[0], "first cancellation")
    task.cancel("second")
    await _event(seen[1], "second cancellation")
    release.set()
    outcome = await _settle(task, "cleanup outcome")
    assert outcome.succeeded is (not fails)
    assert outcome.caller_cancellation is observed[0]
    assert observed[0] is not observed[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["invoke", "fallback-dispatch", "fallback-await", "fallback-result"]
)
async def test_cleanup_child_invocation_and_fallback_cancellation_are_failures_not_caller_control_flow(
    mcp_server, boundary
):
    cancellation = asyncio.CancelledError("actual boundary cancellation")
    session = MagicMock()
    if boundary == "invoke":
        session.kill.side_effect = cancellation
    elif boundary == "fallback-dispatch":
        session.kill.return_value = None
        session.event_bus.dispatch.side_effect = cancellation
    elif boundary == "fallback-result":
        session.kill.return_value = None
        session.event_bus.dispatch.return_value.event_result = AsyncMock(
            side_effect=cancellation
        )
    else:
        session.kill.return_value = None

        async def stopped():
            raise cancellation

        session.event_bus.dispatch.return_value = stopped()
    outcome = await mcp_server._kill_session_safely(session)
    assert outcome.succeeded is False
    assert outcome.caller_cancellation is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize(
    "caller",
    ["reset", "idle", "recovery-old", "startup", "recovery-replacement", "shutdown"],
)
async def test_cleanup_callers_rethrow_actual_cancellation_after_ownership_updates(
    mcp_server, monkeypatch, caller, fails
):
    cancellation = asyncio.CancelledError("first observed cancellation")
    outcome = module._CleanupOutcome(
        succeeded=not fails, caller_cancellation=cancellation
    )
    session = MagicMock()
    session.start = AsyncMock(side_effect=RuntimeError("partial start failed"))
    mcp_server.browser_session = session
    mcp_server._namespace = {"user_value": 42}
    mcp_server._last_activity = 0
    mcp_server.session_timeout_minutes = 0
    monkeypatch.setattr(
        mcp_server, "_kill_session_safely", AsyncMock(return_value=outcome)
    )
    if caller in ("startup", "recovery-replacement"):
        mcp_server.browser_session = None
        mcp_server._namespace = None
        monkeypatch.setattr(mcp_server, "_build_browser_profile", lambda: None)
        monkeypatch.setattr(module, "BrowserSession", lambda **kwargs: session)
    operation = {
        "reset": mcp_server._reset_browser,
        "idle": mcp_server._cleanup_expired_session,
        "recovery-old": mcp_server._recover_browser_session_locked,
        "startup": mcp_server._ensure_namespace,
        "recovery-replacement": mcp_server._recover_browser_session_locked,
        "shutdown": mcp_server._shutdown_locked,
    }[caller]
    with pytest.raises(asyncio.CancelledError) as raised:
        await operation()
    assert raised.value is cancellation
    assert mcp_server.browser_session is (session if fails else None)
    assert not mcp_server._execution_lock.locked()
    assert not mcp_server._lifecycle_lock.locked()
    if fails:
        assert mcp_server._browser_lifecycle.phase is LifecyclePhase.FAILED
        assert mcp_server._browser_lifecycle.snapshot().last_error


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", [False, True], ids=["startup", "replacement"])
async def test_original_start_cancellation_wins_over_later_cleanup_cancellation(
    mcp_server, monkeypatch, recovery
):
    original = asyncio.CancelledError("original start cancellation")
    later = asyncio.CancelledError("later cleanup cancellation")
    session = MagicMock()
    session.start = AsyncMock(side_effect=original)
    monkeypatch.setattr(mcp_server, "_build_browser_profile", lambda: None)
    monkeypatch.setattr(module, "BrowserSession", lambda **kwargs: session)
    monkeypatch.setattr(
        mcp_server,
        "_kill_session_safely",
        AsyncMock(
            return_value=module._CleanupOutcome(
                succeeded=True, caller_cancellation=later
            )
        ),
    )
    operation = (
        mcp_server._recover_browser_session_locked
        if recovery
        else mcp_server._ensure_namespace
    )
    with pytest.raises(asyncio.CancelledError) as raised:
        await operation()
    assert raised.value is original
    assert mcp_server.browser_session is None


@pytest.mark.asyncio
async def test_shutdown_retains_first_outer_cancellation_when_cleanup_later_fails(
    mcp_server, monkeypatch
):
    started = asyncio.Event()
    release = asyncio.Event()
    observed = []
    seen = [asyncio.Event(), asyncio.Event()]
    shield = asyncio.shield
    session = MagicMock()
    mcp_server.browser_session = session

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            seen[min(len(observed), 2) - 1].set()
            raise

    async def shutdown():
        started.set()
        await _event(release, "worker release")
        raise RuntimeError("shutdown cleanup failed")

    monkeypatch.setattr(module.asyncio, "shield", capture)
    monkeypatch.setattr(mcp_server, "_shutdown_locked", shutdown)
    task = asyncio.create_task(mcp_server._shutdown())
    await _event(started, "worker start")
    task.cancel("first")
    await _event(seen[0], "first cancellation")
    task.cancel("second")
    await _event(seen[1], "second cancellation")
    release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await _settle(task, "cancelled task")
    assert raised.value is observed[0]
    assert mcp_server.browser_session is session
    assert mcp_server._browser_lifecycle.phase is LifecyclePhase.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("operation", ["reset", "shutdown"])
async def test_owned_cleanup_repeated_raw_cancellation_preserves_identity_and_truthful_owner(
    mcp_server, monkeypatch, fails, operation
):
    started = asyncio.Event()
    release = asyncio.Event()
    observed = []
    seen = [asyncio.Event(), asyncio.Event()]
    shield = asyncio.shield

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            seen[min(len(observed), 2) - 1].set()
            raise

    async def kill():
        started.set()
        await _event(release, "worker release")
        if fails:
            raise RuntimeError("owned kill failed")

    session = MagicMock()
    session.kill = kill
    mcp_server.browser_session = session
    mcp_server._namespace = {"answer": 42}
    monkeypatch.setattr(module.asyncio, "shield", capture)
    method = mcp_server._reset_browser if operation == "reset" else mcp_server._shutdown
    task = asyncio.create_task(method())
    await _event(started, "worker start")
    task.cancel("first")
    await _event(seen[0], "first cancellation")
    task.cancel("second")
    await _event(seen[1], "second cancellation")
    release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await _settle(task, "cancelled task")
    assert raised.value is observed[0]
    assert mcp_server.browser_session is (session if fails else None)
    if fails:
        assert mcp_server._namespace == {"answer": 42}
        assert mcp_server._browser_lifecycle.phase is LifecyclePhase.FAILED
    else:
        assert mcp_server._namespace is None
        if operation == "reset":
            assert mcp_server._pending_variables["answer"] == 42
    assert not mcp_server._execution_lock.locked()
    assert not mcp_server._lifecycle_lock.locked()


@pytest.mark.asyncio
async def test_cancelled_kill_task_result_is_cleanup_failure_not_caller_cancellation(
    mcp_server, monkeypatch
):
    observed = []
    shield = asyncio.shield

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            raise

    async def kill():
        raise asyncio.CancelledError("owned cleanup task cancelled itself")

    session = MagicMock()
    session.kill = kill
    monkeypatch.setattr(module.asyncio, "shield", capture)
    outcome = await mcp_server._kill_session_safely(session)
    assert not outcome.succeeded
    assert outcome.caller_cancellation is None
    assert observed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["raise", "self-request-return"])
async def test_shutdown_child_cancellation_is_failure_not_caller_control_flow(
    mcp_server, monkeypatch, termination
):
    session = MagicMock()
    mcp_server.browser_session = session

    async def shutdown():
        if termination == "self-request-return":
            asyncio.current_task().cancel("shutdown worker self cancellation")
            return
        raise asyncio.CancelledError("shutdown worker terminated")

    monkeypatch.setattr(mcp_server, "_shutdown_locked", shutdown)
    with pytest.raises(RuntimeError, match="Owned shutdown cleanup was cancelled"):
        await mcp_server._shutdown()
    assert mcp_server.browser_session is session
    assert mcp_server._browser_lifecycle.phase is LifecyclePhase.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["caller-first", "child-first"])
async def test_simultaneous_cleanup_and_caller_cancel_keep_only_actual_caller_exception(
    mcp_server, monkeypatch, order
):
    started = asyncio.Event()
    release = asyncio.Event()
    worker = None
    observed = []
    shield = asyncio.shield

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            raise

    async def kill():
        nonlocal worker
        worker = asyncio.current_task()
        started.set()
        await _event(release, "worker release")

    session = MagicMock()
    session.kill = kill
    monkeypatch.setattr(module.asyncio, "shield", capture)
    caller = asyncio.create_task(mcp_server._kill_session_safely(session))
    await _event(started, "worker start")
    if order == "caller-first":
        caller.cancel("genuine caller cancellation")
        worker.cancel("cleanup worker cancellation")
    else:
        worker.cancel("cleanup worker cancellation")
        caller.cancel("genuine caller cancellation")
    outcome = await _settle(caller, "cleanup caller")
    assert not outcome.succeeded
    assert len(observed) == 1
    assert outcome.caller_cancellation is observed[0]


@pytest.mark.asyncio
async def test_already_caught_outer_cancel_does_not_turn_child_failure_into_new_caller_cancel(
    mcp_server,
):
    prior = []
    session = MagicMock()
    session.kill = AsyncMock(side_effect=asyncio.CancelledError("worker stopped"))

    async def caller():
        asyncio.current_task().cancel("outer cancellation")
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as error:
            prior.append(error)
            assert asyncio.current_task().cancelling() > 0
            return await mcp_server._kill_session_safely(session)

    outcome = await asyncio.create_task(caller())
    assert not outcome.succeeded
    assert outcome.caller_cancellation is None
    assert len(prior) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("child_fails", [False, True])
async def test_caller_cancellation_pending_before_helper_entry_is_not_lost(
    mcp_server, monkeypatch, child_fails
):
    observed = []
    shield = asyncio.shield

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            raise

    async def kill():
        if child_fails:
            raise asyncio.CancelledError("worker stopped")

    session = MagicMock()
    session.kill = kill
    monkeypatch.setattr(module.asyncio, "shield", capture)

    async def caller():
        asyncio.current_task().cancel("already pending caller cancellation")
        assert asyncio.current_task().cancelling() > 0
        return await mcp_server._kill_session_safely(session)

    outcome = await asyncio.create_task(caller())
    assert outcome.succeeded is (not child_fails)
    assert outcome.caller_cancellation is observed[0]


@pytest.mark.asyncio
async def test_synchronous_cleanup_self_cancel_runs_in_owned_worker_not_request(
    mcp_server,
):
    request = asyncio.current_task()
    invoked = []
    session = MagicMock()
    session.event_bus.dispatch.return_value.event_result.return_value = None

    def kill():
        worker = asyncio.current_task()
        invoked.append(worker)
        worker.cancel("synchronous cleanup worker cancellation")

    session.kill = kill
    outcome = await mcp_server._kill_session_safely(session)
    assert invoked[0] is not request
    assert not outcome.succeeded
    assert outcome.caller_cancellation is None


@pytest.mark.asyncio
async def test_validated_tool_owner_is_cancelled_once_and_drained_before_telemetry(
    monkeypatch,
):
    from types import SimpleNamespace

    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    started = asyncio.Event()
    app_cancelled = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    observed = []
    repeated = asyncio.Event()
    child_cancellations = []
    telemetry = []
    shield = asyncio.shield

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            if len(observed) == 2:
                repeated.set()
            raise

    async def dispatch(name, arguments):
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError as error:
            child_cancellations.append(error)
            app_cancelled.set()
            try:
                await _event(release, "worker release")
            except asyncio.CancelledError as later:
                child_cancellations.append(later)
                raise
            completed.set()
            return "late result is discarded"

    def capture_telemetry(event):
        telemetry.append((event, completed.is_set()))

    server._handle_tool = dispatch
    server._telemetry = SimpleNamespace(capture=capture_telemetry)
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", True)
    monkeypatch.setattr(module.asyncio, "shield", capture)
    handler = server.server.get_request_handler("tools/call").handler
    params = module.types.CallToolRequestParams(name="get_status", arguments={})
    request = asyncio.create_task(handler(None, params))
    await _event(started, "worker start")
    request.cancel("first caller cancellation")
    await _event(app_cancelled, "application cancellation")
    request.cancel("second caller cancellation")
    await _event(repeated, "repeated cancellation")
    assert not request.done()
    assert telemetry == []
    release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await _settle(request, "cancelled request")
    assert raised.value is observed[0]
    assert len(child_cancellations) == 1
    assert len(telemetry) == 1 and telemetry[0][1]
    assert telemetry[0][0].error_message == "cancelled"


@pytest.mark.asyncio
async def test_tool_validation_precedes_application_task_creation(monkeypatch):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    handler = server.server.get_request_handler("tools/call").handler

    def unexpected_task(*args, **kwargs):
        pytest.fail("Invalid arguments created an application task")

    with monkeypatch.context() as patcher:
        patcher.setattr(module.asyncio, "create_task", unexpected_task)
        result = await handler(
            None,
            module.types.CallToolRequestParams(
                name="execute_code", arguments={"code": 5}
            ),
        )
    assert result.is_error


@pytest.mark.asyncio
async def test_invalid_arguments_never_reach_telemetry_values(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", True)
    server = module.OpenBrowserServer()
    events = []
    server._telemetry = SimpleNamespace(capture=events.append)
    handler = server.server.get_request_handler("tools/call").handler

    result = await handler(
        None,
        module.types.CallToolRequestParams(
            name="execute_code", arguments={"code": {"password": "hunter2"}}
        ),
    )

    assert result.is_error
    assert "hunter2" in result.content[0].text
    assert len(events) == 1
    assert events[0].error_message == "invalid_tool_arguments"
    assert "hunter2" not in str(events[0])


@pytest.mark.asyncio
async def test_recovery_cancellation_preserves_namespace_variables(mcp_server, monkeypatch):
    cancellation = asyncio.CancelledError("caller cancelled")
    session = MagicMock()
    mcp_server.browser_session = session
    mcp_server._namespace = {"answer": 42, "__private": 1}
    monkeypatch.setattr(
        mcp_server,
        "_kill_session_safely",
        AsyncMock(return_value=module._CleanupOutcome(True, cancellation)),
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await mcp_server._recover_browser_session_locked()

    assert raised.value is cancellation
    assert mcp_server.browser_session is None
    assert mcp_server._namespace is None
    assert mcp_server._pending_variables == {"answer": 42}


@pytest.mark.asyncio
@pytest.mark.parametrize("fatal", [SystemExit("stop"), KeyboardInterrupt("stop")])
async def test_owned_tool_fatal_base_exception_is_not_converted_to_tool_error(
    monkeypatch, fatal
):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()

    async def dispatch(name, arguments):
        raise fatal

    server._handle_tool = dispatch
    handler = server.server.get_request_handler("tools/call").handler
    with pytest.raises(type(fatal)) as raised:
        await handler(
            None, module.types.CallToolRequestParams(name="get_status", arguments={})
        )
    assert raised.value is fatal


@pytest.mark.asyncio
async def test_tool_request_cancellation_pending_before_handler_entry_keeps_identity(
    monkeypatch,
):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    observed = []
    shield = asyncio.shield

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            raise

    monkeypatch.setattr(module.asyncio, "shield", capture)
    handler = server.server.get_request_handler("tools/call").handler

    async def request():
        asyncio.current_task().cancel("pending request cancellation")
        return await handler(
            None, module.types.CallToolRequestParams(name="get_status", arguments={})
        )

    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.create_task(request())
    assert raised.value is observed[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["caller-first", "application-first"])
async def test_tool_caller_and_application_cancellation_race_keeps_caller_identity(
    monkeypatch, order
):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    started = asyncio.Event()
    worker = None
    observed = []
    shield = asyncio.shield

    async def capture(awaitable):
        try:
            return await shield(awaitable)
        except asyncio.CancelledError as error:
            observed.append(error)
            raise

    async def dispatch(name, arguments):
        nonlocal worker
        worker = asyncio.current_task()
        started.set()
        await asyncio.sleep(3600)

    server._handle_tool = dispatch
    monkeypatch.setattr(module.asyncio, "shield", capture)
    handler = server.server.get_request_handler("tools/call").handler
    caller = asyncio.create_task(
        handler(
            None, module.types.CallToolRequestParams(name="get_status", arguments={})
        )
    )
    await _event(started, "worker start")
    if order == "caller-first":
        caller.cancel("request cancellation")
        worker.cancel("application cancellation")
    else:
        worker.cancel("application cancellation")
        caller.cancel("request cancellation")
    with pytest.raises(asyncio.CancelledError) as raised:
        await _settle(caller, "cancelled caller")
    assert len(observed) == 1 and raised.value is observed[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["raise", "self-request-return"])
async def test_tool_child_termination_is_authoritative_owned_task_cancellation(
    monkeypatch, termination
):
    monkeypatch.setattr(module, "TELEMETRY_AVAILABLE", False)
    server = module.OpenBrowserServer()
    worker = None

    async def dispatch(name, arguments):
        nonlocal worker
        worker = asyncio.current_task()
        if termination == "self-request-return":
            worker.cancel("application stopped itself")
            return "discarded because its owned task is cancelled"
        raise asyncio.CancelledError("application terminated")

    server._handle_tool = dispatch
    handler = server.server.get_request_handler("tools/call").handler
    result = await handler(
        None, module.types.CallToolRequestParams(name="get_status", arguments={})
    )
    assert worker.cancelled()
    assert result.is_error
    assert result.content[0].text == "Error: Tool operation was cancelled"
