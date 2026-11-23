"""Behavior tests for CDP root health in BrowserSession."""

from __future__ import annotations

import asyncio
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_session_part2 import _make_session_with_mocks

from openbrowser.browser.cdp_connection import (
    CdpConnectionErrorCode,
    CdpConnectionPhase,
)
from openbrowser.browser.events import (
    BrowserConnectedEvent,
    BrowserStartEvent,
    BrowserStopEvent,
    TabCreatedEvent,
)
from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.browser.navigation import NavigationOperation
from openbrowser.browser.session import MAX_RETAINED_PROBES, BrowserSession
from openbrowser.browser.tabs import TabState


class _ObservedTask(asyncio.Task):
    """Test task that records whether its result or exception was retrieved."""

    def __init__(self, coro, **kwargs):
        super().__init__(coro, **kwargs)
        self.result_observed = False

    def __await__(self):
        self.result_observed = True
        return super().__await__()

    def result(self):
        self.result_observed = True
        return super().result()

    def exception(self):
        self.result_observed = True
        return super().exception()


def _unobserved_task_errors(tasks: set[_ObservedTask]) -> list[BaseException]:
    errors = []
    for task in tasks:
        if task.done() and not task.cancelled() and not task.result_observed:
            error = task.exception()
            if error is not None and type(error).__name__ != "QueueShutDown":
                errors.append(error)
    return errors


@pytest.fixture(autouse=True)
async def _fail_on_orphaned_test_tasks():
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    created: set[_ObservedTask] = set()

    def track_task(task_loop, coro, **kwargs):
        task = _ObservedTask(coro, loop=task_loop, **kwargs)
        created.add(task)
        return task

    loop.set_task_factory(track_task)
    try:
        yield
    finally:
        loop.set_task_factory(previous_factory)
        current = asyncio.current_task()
        if isinstance(current, _ObservedTask):
            created.discard(current)
        pending = {task for task in created if not task.done()}
        for task in pending:
            task.cancel()
        if pending:
            _done, still_pending = await asyncio.wait(pending, timeout=2.0)
            if still_pending:
                pytest.fail(f"test left {len(still_pending)} cancellation-resistant task(s) running")
        unobserved = _unobserved_task_errors(created)
        if unobserved:
            details = ", ".join(f"{type(error).__name__}: {error}" for error in unobserved)
            pytest.fail(f"test left unobserved task failure(s): {details}")


@pytest.mark.asyncio
async def test_orphan_task_guard_detects_completed_unobserved_failure_with_a_noop_callback():
    async def fail():
        raise RuntimeError("unobserved task failure")

    task = asyncio.create_task(fail())
    task.add_done_callback(lambda _task: None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert isinstance(task, _ObservedTask)
    assert task.result_observed is False
    errors = _unobserved_task_errors({task})
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "unobserved task failure" in str(errors[0])
    assert task.result_observed is True


async def _wait_until(predicate, *, limit_seconds=1.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_seconds
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not become true before the deadline")


def _ready_session():
    session, mocks = _make_session_with_mocks()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    return session, mocks


def _unconnected_session():
    session, mocks = _make_session_with_mocks()
    session._cdp_client_root = None
    session._session_manager = None
    session.agent_focus = None
    return session, mocks


def _page_target() -> dict[str, str]:
    return {
        "targetId": "target-page-1234",
        "type": "page",
        "url": "about:blank",
        "title": "",
    }


def _manager_for(mocks):
    manager = MagicMock()
    manager.start_monitoring = AsyncMock()
    manager.get_session_for_target = AsyncMock(return_value=mocks["cdp_session"])
    manager.get_live_session_for_target = AsyncMock(return_value=mocks["cdp_session"])
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    return manager


@pytest.mark.asyncio
async def test_probe_success_records_connected_health():
    session, mocks = _ready_session()
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(return_value={"product": "Chrome"})

    health = await session.probe_cdp_connection(timeout_seconds=0.5)

    assert mocks["cdp_root"].send.Browser.getVersion.await_count == 1
    assert health.phase is CdpConnectionPhase.CONNECTED
    assert health.client_present is True
    assert health.healthy is True
    assert health.probe_attempts == 1
    assert health.last_probe_at is not None
    assert health.last_latency_ms is not None


@pytest.mark.asyncio
async def test_probe_failure_records_degraded_health():
    session, mocks = _ready_session()
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=TimeoutError("getVersion timed out"))

    health = await session.probe_cdp_connection(timeout_seconds=0.01)

    assert health.phase is CdpConnectionPhase.DEGRADED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_probe_synchronous_error_records_degraded_health():
    session, mocks = _ready_session()
    mocks["cdp_root"].send.Browser.getVersion = MagicMock(side_effect=RuntimeError("closed"))

    health = await session.probe_cdp_connection(timeout_seconds=0.5)

    assert health.phase is CdpConnectionPhase.DEGRADED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.PROBE_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health_phase",
    [CdpConnectionPhase.RESOLVING, CdpConnectionPhase.CONNECTING, CdpConnectionPhase.FAILED],
)
async def test_probe_skips_health_phases_without_sending(health_phase):
    session, mocks = _ready_session()
    generation = session._cdp_health.begin_connect()
    if health_phase is CdpConnectionPhase.CONNECTING:
        session._cdp_health.mark_connecting(generation)
    elif health_phase is CdpConnectionPhase.FAILED:
        session._cdp_health.mark_connect_failed(generation, RuntimeError("connect failed"), client_present=False)
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(return_value={"product": "Chrome"})
    before = session.cdp_connection_health()

    health = await session.probe_cdp_connection()

    assert mocks["cdp_root"].send.Browser.getVersion.await_count == 0
    assert health == before


@pytest.mark.asyncio
async def test_probe_without_root_records_not_connected():
    session, mocks = _ready_session()
    root = mocks["cdp_root"]
    root.send.Browser.getVersion = AsyncMock(return_value={})
    session._cdp_client_root = None

    health = await session.probe_cdp_connection()

    assert root.send.Browser.getVersion.await_count == 0
    assert health.phase is CdpConnectionPhase.DISCONNECTED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.NOT_CONNECTED
    assert health.probe_attempts == 1
    assert health.last_probe_at is not None
    assert health.last_failure_at is not None
    assert health.last_failure_at >= health.last_probe_at


@pytest.mark.asyncio
async def test_probe_is_skipped_during_connect_setup():
    session, mocks = _ready_session()
    generation = session._cdp_health.begin_connect()
    session._cdp_health.mark_connecting(generation)
    session._lifecycle.transition(LifecyclePhase.STOPPED)
    session._lifecycle.transition(LifecyclePhase.STARTING)
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(return_value={})

    health = await session.probe_cdp_connection()

    assert mocks["cdp_root"].send.Browser.getVersion.await_count == 0
    assert health.phase is CdpConnectionPhase.CONNECTING
    assert health.probe_attempts == 0
    assert health.last_probe_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [LifecyclePhase.STARTING, LifecyclePhase.STOPPING, LifecyclePhase.STOPPED, LifecyclePhase.FAILED],
)
async def test_probe_skips_non_probeable_lifecycle_states(phase):
    session, mocks = _make_session_with_mocks()
    if phase is LifecyclePhase.STARTING:
        session._lifecycle.transition(LifecyclePhase.STARTING)
    elif phase is LifecyclePhase.STOPPING:
        session._lifecycle.transition(LifecyclePhase.STOPPING)
    elif phase is LifecyclePhase.FAILED:
        session._lifecycle.transition(LifecyclePhase.FAILED)
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(return_value={})
    before = session.cdp_connection_health()

    health = await session.probe_cdp_connection()

    assert mocks["cdp_root"].send.Browser.getVersion.await_count == 0
    assert health == before
    assert health.probe_attempts == 0
    assert health.last_probe_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [LifecyclePhase.DEGRADED, LifecyclePhase.RECOVERING])
async def test_probe_runs_in_degraded_and_recovering_states(phase):
    session, mocks = _make_session_with_mocks()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    session._lifecycle.transition(phase)
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(return_value={"product": "Chrome"})

    health = await session.probe_cdp_connection()

    assert mocks["cdp_root"].send.Browser.getVersion.await_count == 1
    assert health.phase is CdpConnectionPhase.CONNECTED
    assert health.healthy is True
    assert health.probe_attempts == 1


@pytest.mark.asyncio
async def test_probe_success_is_ignored_when_root_disappears():
    session, mocks = _ready_session()
    release = asyncio.Event()

    async def delayed_version():
        await asyncio.wait_for(release.wait(), timeout=5.0)
        return {}

    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=delayed_version)
    task = asyncio.create_task(session.probe_cdp_connection())
    try:
        await asyncio.sleep(0)
        session._cdp_client_root = None
        release.set()

        health = await asyncio.wait_for(task, timeout=5.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2.0)

    assert health.phase is CdpConnectionPhase.DISCONNECTED
    assert health.healthy is None
    assert health.probe_attempts == 1
    assert health.last_success_at is None


@pytest.mark.asyncio
async def test_probe_does_not_start_command_after_shutdown_transition():
    session, mocks = _ready_session()
    get_version = AsyncMock(return_value={})
    mocks["cdp_root"].send.Browser.getVersion = get_version
    gate_started = asyncio.Event()
    release_gate = asyncio.Event()
    real_create_task = asyncio.create_task

    async def gated_command(coro):
        gate_started.set()
        await asyncio.wait_for(release_gate.wait(), timeout=5.0)
        return await coro

    def create_task(coro):
        return real_create_task(gated_command(coro))

    probe_task = None
    try:
        with patch("openbrowser.browser.session.asyncio.create_task", side_effect=create_task):
            probe_task = real_create_task(session.probe_cdp_connection())
            await asyncio.wait_for(gate_started.wait(), timeout=1.0)
            session._lifecycle.transition(LifecyclePhase.STOPPING)
            release_gate.set()
            health = await asyncio.wait_for(probe_task, timeout=2.0)
    finally:
        release_gate.set()
        if probe_task is not None and not probe_task.done():
            probe_task.cancel()
            await asyncio.wait_for(asyncio.gather(probe_task, return_exceptions=True), timeout=2.0)

    get_version.assert_not_awaited()
    assert health.phase is CdpConnectionPhase.DISCONNECTED
    assert health.healthy is None


@pytest.mark.asyncio
async def test_queued_probe_rechecks_root_and_generation_before_command():
    session, mocks = _ready_session()
    old_root = mocks["cdp_root"]
    old_get_version = AsyncMock(return_value={})
    old_root.send.Browser.getVersion = old_get_version
    gate_started = asyncio.Event()
    release_gate = asyncio.Event()
    real_create_task = asyncio.create_task

    async def gated_command(coro):
        gate_started.set()
        await asyncio.wait_for(release_gate.wait(), timeout=5.0)
        return await coro

    def create_task(coro):
        return real_create_task(gated_command(coro))

    probe_task = None
    try:
        with patch("openbrowser.browser.session.asyncio.create_task", side_effect=create_task):
            probe_task = real_create_task(session.probe_cdp_connection())
            await asyncio.wait_for(gate_started.wait(), timeout=1.0)
            new_root = MagicMock()
            new_get_version = AsyncMock(return_value={})
            new_root.send.Browser.getVersion = new_get_version
            session._cdp_client_root = new_root
            session._cdp_health.begin_connect()
            release_gate.set()
            health = await asyncio.wait_for(probe_task, timeout=2.0)
    finally:
        release_gate.set()
        if probe_task is not None and not probe_task.done():
            probe_task.cancel()
            await asyncio.wait_for(asyncio.gather(probe_task, return_exceptions=True), timeout=2.0)

    old_get_version.assert_not_awaited()
    new_get_version.assert_not_awaited()
    assert health.client_present is True
    assert health.phase is CdpConnectionPhase.RESOLVING


@pytest.mark.asyncio
async def test_health_updates_do_not_change_lifecycle_phase_or_generation():
    session, mocks = _ready_session()
    before = session.lifecycle_snapshot()
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(return_value={})

    await session.probe_cdp_connection()

    after = session.lifecycle_snapshot()
    assert after["phase"] == before["phase"]
    assert after["generation"] == before["generation"]


@pytest.mark.asyncio
async def test_connect_success_records_connected_health():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": [_page_target()]})
    mocks["cdp_root"].start = AsyncMock()
    before = session.lifecycle_snapshot()

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]),
        patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
        patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
    ):
        result = await session.connect()

    assert result is session
    health = session._cdp_health.snapshot(client_present=True, connection_expected=True)
    assert health.phase is CdpConnectionPhase.CONNECTED
    assert health.healthy is True
    assert health.connect_attempts == 1
    assert health.probe_attempts == 0
    assert health.last_probe_at is None
    assert health.last_connected_at is not None
    after = session.lifecycle_snapshot()
    assert after["phase"] == before["phase"]
    assert after["generation"] == before["generation"]


@pytest.mark.asyncio
async def test_connect_does_not_publish_when_shutdown_fence_closes_during_setup():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    root = mocks["cdp_root"]
    root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": [_page_target()]})
    root.stop = AsyncMock()

    async def close_admission():
        session._connect_admission_closed = True

    root.start = AsyncMock(side_effect=close_admission)

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
        patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
        pytest.raises(RuntimeError, match="Failed to establish CDP connection"),
    ):
        await session.connect("ws://direct")

    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.cdp_connection_health().healthy is False
    root.stop.assert_awaited_once()
    manager.clear.assert_awaited_once()


@pytest.mark.asyncio
async def test_fresh_connect_does_not_publish_initial_tabs_after_owner_cancel():
    session, mocks = _unconnected_session()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    manager = _manager_for(mocks)
    root = mocks["cdp_root"]
    root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": [_page_target()]})
    root.start = AsyncMock()
    session.event_bus.dispatch = MagicMock()
    owner = asyncio.create_task(asyncio.sleep(60))
    session._start_task = owner
    owner.cancel()
    await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), timeout=2.0)

    try:
        with (
            patch("openbrowser.browser.session.CDPClient", return_value=root),
            patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
            patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
            pytest.raises(RuntimeError, match="no longer owns connection publication"),
        ):
            await session.connect(_start_owned=True)

        assert session._cdp_client_root is None
        assert session._session_manager is None
        dispatched = [call.args[0] for call in session.event_bus.dispatch.call_args_list]
        assert not any(isinstance(event, TabCreatedEvent) for event in dispatched)
    finally:
        session._start_task = None


@pytest.mark.asyncio
async def test_start_handler_does_not_publish_connection_or_tabs_after_owner_loss():
    session, mocks = _unconnected_session()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    manager = _manager_for(mocks)
    root = mocks["cdp_root"]
    root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": [_page_target()]})
    root.start = AsyncMock()
    session.event_bus.dispatch = MagicMock()
    identity_started = asyncio.Event()
    release_identity = asyncio.Event()

    async def blocked_identity(target_id):
        identity_started.set()
        await asyncio.wait_for(release_identity.wait(), timeout=5.0)
        return None

    session._read_document_identity = AsyncMock(side_effect=blocked_identity)
    owner = asyncio.create_task(asyncio.sleep(60))
    session._start_task = owner

    try:
        with (
            patch.object(BrowserSession, "attach_all_watchdogs", AsyncMock()),
            patch("openbrowser.browser.session.CDPClient", return_value=root),
            patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
            patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
        ):
            handler = asyncio.create_task(session.on_BrowserStartEvent(BrowserStartEvent()))
            await asyncio.wait_for(identity_started.wait(), timeout=5.0)
            owner.cancel()
            await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), timeout=2.0)
            release_identity.set()

            with pytest.raises(RuntimeError, match="no longer owns connection publication"):
                await asyncio.wait_for(handler, timeout=5.0)

        assert session._cdp_client_root is None
        assert session._session_manager is None
        assert session.tab_snapshot() == []
        dispatched = [call.args[0] for call in session.event_bus.dispatch.call_args_list]
        assert not any(isinstance(event, BrowserConnectedEvent) for event in dispatched)
        assert not any(isinstance(event, TabCreatedEvent) for event in dispatched)
    finally:
        session._start_task = None


@pytest.mark.asyncio
async def test_bounded_shutdown_retains_cancellation_resistant_task():
    session = BrowserSession()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    owner = asyncio.create_task(stubborn())
    await asyncio.wait_for(started.wait(), timeout=5.0)

    started_at = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        await session._await_task_bounded(owner, 0.05)
    assert time.monotonic() - started_at < 0.3

    assert session.lifecycle_snapshot()["phase"] == "failed"
    assert session._cdp_client_root is root
    assert session._session_manager is manager
    assert session.cdp_connection_health().cleanup_pending is True
    assert owner in session._detached_shutdown_tasks

    release.set()
    await _wait_until(lambda: not session._detached_shutdown_tasks)
    await session.kill()

    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_stop_returns_at_deadline_and_retains_cancellation_resistant_shutdown():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    session._cdp_client_root = MagicMock()
    manager = MagicMock()
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    session._session_manager = manager
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_cancel_start():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    session._cancel_owned_start = stubborn_cancel_start

    with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.05):
        started_at = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            await session.stop()
        assert time.monotonic() - started_at < 0.3
        with pytest.raises(TimeoutError, match="Prior browser shutdown cleanup"):
            await session.kill()

    assert started.is_set()
    assert session._detached_shutdown_tasks
    assert session.lifecycle_snapshot()["phase"] == "failed"

    release.set()
    await _wait_until(lambda: not session._detached_shutdown_tasks)
    with patch("openbrowser.browser.session._shutdown_timeout", return_value=1.0):
        await session.kill()

    assert session.lifecycle_snapshot()["phase"] == "stopped"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "kill"])
async def test_shutdown_rejects_retry_while_detached_cleanup_runs(operation):
    session = BrowserSession()
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    owner = asyncio.create_task(stubborn())
    await asyncio.wait_for(started.wait(), timeout=5.0)
    with pytest.raises(TimeoutError, match="timed out"):
        await session._await_task_bounded(owner, 0.05)

    assert owner in session._detached_shutdown_tasks
    with pytest.raises(TimeoutError, match="Prior browser shutdown cleanup is still running"):
        await getattr(session, operation)()

    release.set()
    await _wait_until(lambda: not session._detached_shutdown_tasks)
    await session.kill()


@pytest.mark.asyncio
async def test_direct_connect_does_not_publish_initial_tabs_after_shutdown_fence():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    root = mocks["cdp_root"]
    root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": [_page_target()]})
    root.start = AsyncMock()
    session.event_bus.dispatch = MagicMock()
    identity_started = asyncio.Event()
    release_identity = asyncio.Event()

    async def blocked_identity(target_id):
        identity_started.set()
        await asyncio.wait_for(release_identity.wait(), timeout=5.0)
        return None

    session._read_document_identity = AsyncMock(side_effect=blocked_identity)

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
        patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
    ):
        connect_task = asyncio.create_task(session.connect("ws://direct"))
        await asyncio.wait_for(identity_started.wait(), timeout=5.0)
        session._connect_admission_closed = True
        release_identity.set()

        with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
            await asyncio.wait_for(connect_task, timeout=5.0)

    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.tab_snapshot() == []
    dispatched = [call.args[0] for call in session.event_bus.dispatch.call_args_list]
    assert not any(isinstance(event, TabCreatedEvent) for event in dispatched)


@pytest.mark.asyncio
async def test_fresh_connect_does_not_publish_tabs_when_owner_is_lost_during_setup():
    session, mocks = _unconnected_session()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    manager = _manager_for(mocks)
    root = mocks["cdp_root"]
    root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": [_page_target()]})
    root.start = AsyncMock()
    session.event_bus.dispatch = MagicMock()
    identity_started = asyncio.Event()
    release_identity = asyncio.Event()

    async def blocked_identity(target_id):
        identity_started.set()
        await asyncio.wait_for(release_identity.wait(), timeout=5.0)
        return None

    session._read_document_identity = AsyncMock(side_effect=blocked_identity)
    owner = asyncio.create_task(asyncio.sleep(60))
    session._start_task = owner

    try:
        with (
            patch("openbrowser.browser.session.CDPClient", return_value=root),
            patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
            patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
        ):
            connect_task = asyncio.create_task(session.connect(_start_owned=True))
            await asyncio.wait_for(identity_started.wait(), timeout=5.0)
            owner.cancel()
            await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), timeout=2.0)
            release_identity.set()

            with pytest.raises(RuntimeError, match="no longer owns connection publication"):
                await asyncio.wait_for(connect_task, timeout=5.0)

        assert session._cdp_client_root is None
        assert session._session_manager is None
        assert session.tab_snapshot() == []
        dispatched = [call.args[0] for call in session.event_bus.dispatch.call_args_list]
        assert not any(isinstance(event, TabCreatedEvent) for event in dispatched)
    finally:
        session._start_task = None


@pytest.mark.asyncio
async def test_direct_connect_refuses_active_replacement():
    session, mocks = _unconnected_session()
    previous_root = mocks["cdp_root"]
    previous_root.stop = AsyncMock()
    previous_manager = _manager_for(mocks)
    previous_manager.clear = AsyncMock()
    session._cdp_client_root = previous_root
    session._session_manager = previous_manager

    with (
        patch("openbrowser.browser.session.CDPClient") as client_factory,
        pytest.raises(RuntimeError, match="connection already exists"),
    ):
        await session.connect()

    client_factory.assert_not_called()
    previous_root.stop.assert_not_awaited()
    previous_manager.clear.assert_not_awaited()
    assert session._cdp_client_root is previous_root
    assert session._session_manager is previous_manager


@pytest.mark.asyncio
@pytest.mark.parametrize("published_owner", ["root", "manager"])
async def test_direct_connect_refuses_each_published_owner_before_discovery(published_owner):
    session, mocks = _unconnected_session()
    root = mocks["cdp_root"]
    root.stop = AsyncMock()
    manager = _manager_for(mocks)
    manager.clear = AsyncMock()
    session._cdp_client_root = root if published_owner == "root" else None
    session._session_manager = manager if published_owner == "manager" else None
    session.browser_profile.cdp_url = "ws://existing.invalid/devtools/browser/fixture"
    before = session.cdp_connection_health()

    with (
        patch("httpx.AsyncClient") as http_client,
        patch("openbrowser.browser.session.CDPClient") as client_factory,
        pytest.raises(RuntimeError, match="connection already exists"),
    ):
        await session.connect("http://127.0.0.1:9333")

    http_client.assert_not_called()
    client_factory.assert_not_called()
    root.stop.assert_not_awaited()
    manager.clear.assert_not_awaited()
    assert session.browser_profile.cdp_url == "ws://existing.invalid/devtools/browser/fixture"
    assert session._cdp_client_root is (root if published_owner == "root" else None)
    assert session._session_manager is (manager if published_owner == "manager" else None)
    after = session.cdp_connection_health()
    assert after.connect_attempts == before.connect_attempts
    assert after.last_error_code is before.last_error_code


@pytest.mark.asyncio
async def test_direct_connect_refuses_active_replacement_before_http_discovery():
    session, mocks = _unconnected_session()
    previous_root = mocks["cdp_root"]
    previous_root.stop = AsyncMock()
    previous_manager = _manager_for(mocks)
    session._cdp_client_root = previous_root
    session._session_manager = previous_manager
    session.browser_profile.cdp_url = "ws://existing.invalid/devtools/browser/fixture"
    session._cdp_session_pool["live"] = MagicMock()
    session._tab_registry.register("target-page-1234", state=TabState.OPEN)
    before = session.cdp_connection_health()

    with (
        patch("httpx.AsyncClient") as http_client,
        pytest.raises(RuntimeError, match="connection already exists"),
    ):
        await session.connect("http://127.0.0.1:9333")

    http_client.assert_not_called()
    previous_root.stop.assert_not_awaited()
    previous_manager.clear.assert_not_awaited()
    assert session.browser_profile.cdp_url == "ws://existing.invalid/devtools/browser/fixture"
    assert session._cdp_client_root is previous_root
    assert session._session_manager is previous_manager
    assert "live" in session._cdp_session_pool
    assert session.tab_snapshot() != []
    after = session.cdp_connection_health()
    assert after.connect_attempts == before.connect_attempts
    assert after.last_error_code is before.last_error_code


@pytest.mark.asyncio
async def test_start_event_handler_owns_its_connection():
    session = BrowserSession(cdp_url="ws://127.0.0.1:9222")
    recorded = []

    async def fake_connect(cdp_url=None, *, _start_owned=False):
        recorded.append((cdp_url, _start_owned))
        session._cdp_client_root = MagicMock()
        return session

    try:
        with (
            patch.object(BrowserSession, "attach_all_watchdogs", AsyncMock()),
            patch.object(BrowserSession, "connect", side_effect=fake_connect),
        ):
            await session.start()

        assert recorded == [("ws://127.0.0.1:9222", True)]
        assert session.lifecycle_snapshot()["phase"] == "ready"
        assert session._cdp_client_root is not None
    finally:
        await session._retire_event_bus(session.event_bus, timeout=1.0)


@pytest.mark.asyncio
async def test_start_with_preconnected_root_dispatches_connection_events():
    session, _mocks = _make_session_with_mocks()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._cdp_get_all_pages = AsyncMock(
        return_value=[
            {
                "targetId": "target-page-1234",
                "type": "page",
                "url": "https://example.test/",
                "title": "Example",
            }
        ]
    )
    trace = []

    async def attach_watchdogs():
        trace.append("attach")

    def record_dispatch(event):
        trace.append(type(event).__name__)

    session.event_bus.dispatch = MagicMock(side_effect=record_dispatch)

    with patch.object(BrowserSession, "attach_all_watchdogs", side_effect=attach_watchdogs):
        result = await session.on_BrowserStartEvent(BrowserStartEvent())

    assert result == {"cdp_url": session.cdp_url}
    assert trace == ["attach", "BrowserConnectedEvent", "TabCreatedEvent"]


@pytest.mark.asyncio
async def test_preconnected_start_does_not_publish_tabs_after_owner_cancellation():
    session, _mocks = _make_session_with_mocks()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    pages_started = asyncio.Event()
    release_pages = asyncio.Event()

    async def blocked_pages():
        pages_started.set()
        await asyncio.wait_for(release_pages.wait(), timeout=5.0)
        return [{"targetId": "late-tab", "url": "https://late.test/"}]

    session._cdp_get_all_pages = AsyncMock(side_effect=blocked_pages)
    session.event_bus.dispatch = MagicMock()
    owner = asyncio.create_task(asyncio.sleep(60))
    session._start_task = owner
    owner.cancel()
    await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), timeout=2.0)

    try:
        with patch.object(BrowserSession, "attach_all_watchdogs", AsyncMock()):
            handler_task = asyncio.create_task(session.on_BrowserStartEvent(BrowserStartEvent()))
            await asyncio.wait_for(pages_started.wait(), timeout=5.0)
            release_pages.set()
            with pytest.raises(RuntimeError, match="no longer owns connection publication"):
                await asyncio.wait_for(handler_task, timeout=5.0)

        dispatched = [call.args[0] for call in session.event_bus.dispatch.call_args_list]
        assert not any(isinstance(event, TabCreatedEvent) for event in dispatched)
    finally:
        session._start_task = None


@pytest.mark.asyncio
async def test_preconnected_start_does_not_publish_tabs_after_ownership_loss():
    session, _mocks = _make_session_with_mocks()
    pages_started = asyncio.Event()
    release_pages = asyncio.Event()

    async def blocked_pages():
        pages_started.set()
        await asyncio.wait_for(release_pages.wait(), timeout=5.0)
        return [{"targetId": "late-tab", "url": "https://late.test/"}]

    session._cdp_get_all_pages = AsyncMock(side_effect=blocked_pages)
    session.event_bus.dispatch = MagicMock()

    with patch.object(BrowserSession, "attach_all_watchdogs", AsyncMock()):
        handler_task = asyncio.create_task(session.on_BrowserStartEvent(BrowserStartEvent()))
        await asyncio.wait_for(pages_started.wait(), timeout=5.0)
        session._lifecycle.transition(LifecyclePhase.STOPPING)
        session._lifecycle.seed_generation(session._lifecycle.generation + 1)
        session._cdp_client_root = None
        release_pages.set()

        with pytest.raises(RuntimeError, match="no longer owns connection publication"):
            await asyncio.wait_for(handler_task, timeout=5.0)

    dispatched = [call.args[0] for call in session.event_bus.dispatch.call_args_list]
    assert not any(isinstance(event, TabCreatedEvent) for event in dispatched)


@pytest.mark.asyncio
async def test_connect_missing_url_records_failure():
    session = BrowserSession()
    session.browser_profile.cdp_url = None

    with pytest.raises(RuntimeError, match="Cannot setup CDP connection without CDP URL"):
        await session.connect(cdp_url=None)

    health = session._cdp_health.snapshot(client_present=False, connection_expected=False)
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.healthy is False
    assert health.connect_attempts == 1
    assert health.last_error_code is CdpConnectionErrorCode.CONNECT_FAILED


@pytest.mark.asyncio
async def test_connect_discovery_timeout_records_failure():
    session, _mocks = _unconnected_session()
    session.browser_profile.cdp_url = "http://127.0.0.1:9222"
    client = AsyncMock()
    client.get = AsyncMock(side_effect=__import__("httpx").ConnectError("refused"))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    fake_loop = MagicMock()
    fake_loop.time = MagicMock(side_effect=[0.0, 31.0])

    with (
        patch("httpx.AsyncClient", return_value=client),
        patch("asyncio.get_event_loop", return_value=fake_loop),
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(TimeoutError, match="did not respond"),
    ):
        await session.connect()

    health = session._cdp_health.snapshot(client_present=False, connection_expected=False)
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.last_error_code is CdpConnectionErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_connect_discovery_without_a_websocket_url_records_failure():
    session, _mocks = _unconnected_session()
    session.browser_profile.cdp_url = "http://127.0.0.1:9222"
    response = MagicMock()
    response.json.return_value = {"webSocketDebuggerUrl": None}
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("httpx.AsyncClient", return_value=client),
        pytest.raises(RuntimeError, match="did not return a debugger URL"),
    ):
        await session.connect()

    health = session._cdp_health.snapshot(client_present=False, connection_expected=False)
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.CONNECT_FAILED


@pytest.mark.asyncio
async def test_connect_root_start_failure_stops_client_and_records_failure():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.send = MagicMock()
    root.start = AsyncMock(side_effect=RuntimeError("Connection refused"))
    root.stop = AsyncMock()

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        pytest.raises(RuntimeError, match="Failed to establish CDP connection"),
    ):
        await session.connect()

    assert root.stop.await_count == 1
    assert session._cdp_client_root is None
    health = session._cdp_health.snapshot(client_present=False, connection_expected=False)
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.cleanup_pending is False


@pytest.mark.asyncio
async def test_connect_failure_sanitizes_the_focus_error_detail():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.send = MagicMock()
    root.start = AsyncMock(side_effect=RuntimeError("Authorization: Bearer focus-secret"))
    root.stop = AsyncMock()

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        pytest.raises(RuntimeError, match="Failed to establish CDP connection"),
    ):
        await session.connect()

    focus_error = str(session._focus_state.last_error)
    assert "focus-secret" not in focus_error
    assert "authorization=<redacted>" in focus_error.lower()


@pytest.mark.asyncio
async def test_connect_failure_sanitizes_quoted_focus_assignments_with_spaces():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.send = MagicMock()
    root.start = AsyncMock(side_effect=RuntimeError("password='two word focus-secret'"))
    root.stop = AsyncMock()

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        pytest.raises(RuntimeError, match="Failed to establish CDP connection"),
    ):
        await session.connect()

    focus_error = str(session._focus_state.last_error)
    assert "two word focus-secret" not in focus_error
    assert "password=<redacted>" in focus_error.lower()


@pytest.mark.asyncio
async def test_start_failure_does_not_leak_raw_error_into_lifecycle_status():
    class FailingStartEvent:
        def __await__(self):
            async def done():
                return None

            return done().__await__()

        async def event_result(self, **_kwargs):
            raise RuntimeError("Authorization: Bearer lifecycle-secret")

    session = BrowserSession()
    session.event_bus.dispatch = MagicMock(return_value=FailingStartEvent())

    with pytest.raises(RuntimeError, match="lifecycle-secret"):
        await session.start()

    snapshot = session.lifecycle_snapshot()
    assert "lifecycle-secret" not in str(snapshot)
    assert "authorization=<redacted>" in (snapshot["last_error"] or "").lower()


def test_lifecycle_snapshot_sanitizes_terminal_error_text():
    session = BrowserSession()
    session._lifecycle.fail(RuntimeError("Authorization: Bearer lifecycle-status-secret"))

    snapshot = session.lifecycle_snapshot()

    assert "lifecycle-status-secret" not in str(snapshot)
    assert "authorization=<redacted>" in (snapshot["last_error"] or "").lower()


@pytest.mark.asyncio
async def test_worker_only_connect_cancellation_records_failure_without_cancelling_caller():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.send = MagicMock()
    root.start = AsyncMock(side_effect=asyncio.CancelledError())
    root.stop = AsyncMock()

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        pytest.raises(RuntimeError, match="worker was cancelled"),
    ):
        await session.connect()

    assert asyncio.current_task().cancelling() == 0
    assert root.stop.await_count == 1
    health = session._cdp_health.snapshot(client_present=False, connection_expected=False)
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.last_error_code is CdpConnectionErrorCode.CONNECT_FAILED
    assert health.cleanup_pending is False
    assert session.browser_health_snapshot()["phase"] == "failed"
    assert session.browser_health_snapshot()["reason"] == "connect_failed"


@pytest.mark.asyncio
async def test_caller_connect_cancellation_records_cancelled_health():
    session, _mocks = _unconnected_session()
    started = asyncio.Event()
    release = asyncio.Event()
    root = MagicMock()
    root.send = MagicMock()
    root.stop = AsyncMock()

    async def blocked_start():
        started.set()
        await asyncio.wait_for(release.wait(), timeout=5.0)

    root.start = AsyncMock(side_effect=blocked_start)
    connect_task = None
    try:
        with patch("openbrowser.browser.session.CDPClient", return_value=root):
            connect_task = asyncio.create_task(session.connect())
            await asyncio.wait_for(started.wait(), timeout=5.0)
            connect_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(connect_task, timeout=2.0)
    finally:
        release.set()
        if connect_task is not None and not connect_task.done():
            connect_task.cancel()
            await asyncio.wait_for(asyncio.gather(connect_task, return_exceptions=True), timeout=2.0)

    assert root.stop.await_count == 1
    health = session._cdp_health.snapshot(client_present=False, connection_expected=False)
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.last_error_code is CdpConnectionErrorCode.CANCELLED
    assert health.cleanup_pending is False
    assert session.browser_health_snapshot()["phase"] == "failed"
    assert session.browser_health_snapshot()["reason"] == "connect_failed"


@pytest.mark.asyncio
async def test_connect_later_setup_failure_clears_manager_and_root():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    mocks["cdp_root"].start = AsyncMock()
    mocks["cdp_root"].stop = AsyncMock()
    mocks["cdp_root"].send.Target.getTargets = AsyncMock(side_effect=RuntimeError("targets failed"))

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]),
        patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
        pytest.raises(RuntimeError, match="Failed to establish CDP connection"),
    ):
        await session.connect()

    assert manager.clear.await_count == 1
    assert mocks["cdp_root"].stop.await_count == 1
    assert session._session_manager is None
    assert session._cdp_client_root is None
    assert session._cdp_health.snapshot(client_present=False, connection_expected=False).phase is (
        CdpConnectionPhase.FAILED
    )


@pytest.mark.asyncio
async def test_connect_manager_cleanup_failure_retains_manager():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    manager.clear = AsyncMock(side_effect=RuntimeError("manager clear failed"))
    mocks["cdp_root"].start = AsyncMock()
    mocks["cdp_root"].stop = AsyncMock()
    mocks["cdp_root"].send.Target.getTargets = AsyncMock(side_effect=RuntimeError("targets failed"))

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]),
        patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
        pytest.raises(RuntimeError, match="Failed to establish CDP connection"),
    ):
        await session.connect()

    assert session._session_manager is manager
    assert session._cdp_client_root is mocks["cdp_root"]
    health = session._cdp_health.snapshot(client_present=False, connection_expected=False)
    assert health.cleanup_pending is True
    assert health.last_cleanup_error is not None
    assert session.browser_health_snapshot()["phase"] == "failed"
    assert session.browser_health_snapshot()["reason"] == "connect_failed"


@pytest.mark.asyncio
async def test_connect_cleanup_failure_retains_root_and_marks_pending():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    mocks["cdp_root"].start = AsyncMock()
    mocks["cdp_root"].stop = AsyncMock(side_effect=RuntimeError("root stop failed"))
    mocks["cdp_root"].send.Target.getTargets = AsyncMock(side_effect=RuntimeError("targets failed"))

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]),
        patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
        pytest.raises(RuntimeError, match="Failed to establish CDP connection"),
    ):
        await session.connect()

    health = session._cdp_health.snapshot(client_present=True, connection_expected=False)
    assert session._session_manager is manager
    assert session._cdp_client_root is mocks["cdp_root"]
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.cleanup_pending is True
    assert health.last_cleanup_error is not None
    assert session.browser_health_snapshot()["phase"] == "failed"
    assert session.browser_health_snapshot()["reason"] == "connect_failed"


@pytest.mark.asyncio
async def test_cancelled_start_retains_failed_connect_cleanup_ownership():
    session, _mocks = _unconnected_session()
    start_entered = asyncio.Event()

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    root = MagicMock()
    root.stop = AsyncMock(side_effect=RuntimeError("root stop failed"))
    manager = MagicMock()
    manager.clear = AsyncMock(side_effect=RuntimeError("manager clear failed"))
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager

    start_task = asyncio.create_task(session.start())
    await asyncio.wait_for(start_entered.wait(), timeout=5.0)
    start_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=5.0)

    assert session._cdp_client_root is root
    assert session._session_manager is manager
    assert session.cdp_connection_health().cleanup_pending is True


@pytest.mark.asyncio
async def test_cancelled_start_bounds_cancellation_resistant_watchdog_cleanup():
    session, _mocks = _unconnected_session()
    start_entered = asyncio.Event()
    kill_started = asyncio.Event()
    release = asyncio.Event()
    watchdog = MagicMock()
    watchdog._subprocess = MagicMock()
    watchdog._spawned_subprocess = None
    profile_lease = MagicMock()
    watchdog._profile_lease = profile_lease

    async def stubborn_kill(_event):
        kill_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)
            watchdog._subprocess = None
            watchdog._profile_lease = None

    watchdog.on_BrowserKillEvent = AsyncMock(side_effect=stubborn_kill)
    session._local_browser_watchdog = watchdog

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    try:
        with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.05):
            start_task = asyncio.create_task(session.start())
            await asyncio.wait_for(start_entered.wait(), timeout=5.0)
            started_at = time.monotonic()
            start_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(start_task, timeout=5.0)
            assert time.monotonic() - started_at < 0.3

        assert kill_started.is_set()
        assert session.lifecycle_snapshot()["phase"] == "failed"
        assert session.cdp_connection_health().cleanup_pending is True
        assert session._watchdog_has_live_resources(watchdog) is True
        assert watchdog._profile_lease is profile_lease
        assert session._detached_cleanup_tasks

        release.set()
        await _wait_until(lambda: not session._detached_cleanup_tasks)
        assert session._watchdog_has_live_resources(watchdog) is False
        await session.reset(wait_for_operations=False)
        assert session.lifecycle_snapshot()["phase"] == "stopped"
    finally:
        release.set()
        session._local_browser_watchdog = None


@pytest.mark.asyncio
async def test_public_shutdown_rejects_active_cancelled_start_cleanup(monkeypatch):
    from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

    session, _mocks = _unconnected_session()
    session.browser_profile.keep_alive = True
    kill_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    watchdog = LocalBrowserWatchdog(event_bus=session.event_bus, browser_session=session)
    watchdog._subprocess = MagicMock()
    watchdog._profile_lease = MagicMock()

    async def on_BrowserKillEvent(_self, _event):
        nonlocal calls
        calls += 1
        kill_started.set()
        await asyncio.wait_for(release.wait(), timeout=5.0)
        watchdog._subprocess = None
        watchdog._profile_lease = None

    monkeypatch.setattr(LocalBrowserWatchdog, "on_BrowserKillEvent", on_BrowserKillEvent)
    watchdog.attach_to_session()
    session._local_browser_watchdog = watchdog
    start_entered = asyncio.Event()

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    start_task = None
    try:
        with patch("openbrowser.browser.session._shutdown_timeout", return_value=5.0):
            start_task = asyncio.create_task(session.start())
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)
            start_task.cancel()
            await asyncio.wait_for(kill_started.wait(), timeout=1.0)
            with pytest.raises(TimeoutError, match="Prior browser cleanup is still running"):
                await session.stop()
            with pytest.raises(TimeoutError, match="Prior browser cleanup is still running"):
                await session.kill()
            assert calls == 1
            assert session._active_cleanup_tasks
            assert watchdog._profile_lease is not None
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(start_task, timeout=2.0)

        assert calls == 1
        assert not session._active_cleanup_tasks
        assert not session._detached_cleanup_tasks
        assert session._watchdog_has_live_resources(watchdog) is False
        await session.reset(wait_for_operations=False)
        assert session.lifecycle_snapshot()["phase"] == "stopped"
    finally:
        release.set()
        if start_task is not None and not start_task.done():
            start_task.cancel()
            await asyncio.wait_for(asyncio.gather(start_task, return_exceptions=True), timeout=2.0)
        await session._retire_event_bus(session.event_bus, timeout=1.0)
        session._local_browser_watchdog = None


@pytest.mark.asyncio
async def test_bounded_watchdog_timeout_keeps_one_cleanup_owner_and_blocks_retry(monkeypatch):
    from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

    session, _mocks = _unconnected_session()
    session.browser_profile.keep_alive = True
    kill_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    watchdog = LocalBrowserWatchdog(event_bus=session.event_bus, browser_session=session)
    process = MagicMock()
    lease = MagicMock()
    watchdog._subprocess = process
    watchdog._profile_lease = lease

    async def on_BrowserKillEvent(_self, _event):
        nonlocal calls
        calls += 1
        kill_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)
            watchdog._subprocess = None
            watchdog._profile_lease = None

    monkeypatch.setattr(LocalBrowserWatchdog, "on_BrowserKillEvent", on_BrowserKillEvent)
    watchdog.attach_to_session()
    session._local_browser_watchdog = watchdog
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    try:
        with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.05):
            await session._cleanup_cancelled_start()

        await asyncio.wait_for(kill_started.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        assert calls == 1
        assert watchdog._profile_lease is lease
        assert session._detached_cleanup_tasks
        assert session.cdp_connection_health().cleanup_pending is True

        with pytest.raises(TimeoutError, match="Prior browser cleanup is still running"):
            await session.stop()
        with pytest.raises(TimeoutError, match="Prior browser cleanup is still running"):
            await session.kill()
        with pytest.raises(TimeoutError, match="Prior browser cleanup is still running"):
            await session.reset(wait_for_operations=False)
        assert calls == 1
        assert watchdog._profile_lease is lease

        release.set()
        await _wait_until(lambda: not session._detached_cleanup_tasks)
        await session.reset(wait_for_operations=False)
        assert session.lifecycle_snapshot()["phase"] == "stopped"
        assert session.cdp_connection_health().cleanup_pending is False
    finally:
        release.set()
        await session._retire_event_bus(session.event_bus, timeout=1.0)
        session._local_browser_watchdog = None


@pytest.mark.asyncio
async def test_cancelled_start_watchdog_failure_is_retried_by_keep_alive_reset():
    session, _mocks = _unconnected_session()
    session.browser_profile.keep_alive = True
    start_entered = asyncio.Event()
    watchdog = MagicMock()
    watchdog._subprocess = MagicMock()
    watchdog._spawned_subprocess = None
    watchdog._profile_lease = MagicMock()

    kill_calls = 0

    async def release_owner(_event):
        nonlocal kill_calls
        kill_calls += 1
        if kill_calls == 1:
            raise RuntimeError("watchdog cleanup failed")
        watchdog._subprocess = None
        watchdog._profile_lease = None

    watchdog.on_BrowserKillEvent = AsyncMock(side_effect=release_owner)
    session._local_browser_watchdog = watchdog

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    try:
        start_task = asyncio.create_task(session.start())
        await asyncio.wait_for(start_entered.wait(), timeout=5.0)
        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(start_task, timeout=5.0)

        assert session._watchdog_cleanup_error is not None
        assert session.cdp_connection_health().cleanup_pending is True
        await session.reset(wait_for_operations=False)

        assert watchdog.on_BrowserKillEvent.await_count == 2
        assert kill_calls == 2
        assert session._watchdog_has_live_resources(watchdog) is False
        assert session.cdp_connection_health().cleanup_pending is False
        assert session.lifecycle_snapshot()["phase"] == "stopped"
    finally:
        session._local_browser_watchdog = None


@pytest.mark.asyncio
async def test_cancelled_start_keeps_cleanup_pending_when_watchdog_failure_is_immediate():
    session, _mocks = _unconnected_session()
    watchdog = MagicMock()
    watchdog._subprocess = MagicMock()
    watchdog._spawned_subprocess = None
    watchdog._profile_lease = MagicMock()
    watchdog.on_BrowserKillEvent = AsyncMock(side_effect=RuntimeError("watchdog cleanup failed"))
    session._local_browser_watchdog = watchdog

    try:
        await session._cleanup_cancelled_start()

        assert session._watchdog_has_live_resources(watchdog) is True
        assert session.cdp_connection_health().cleanup_pending is True
    finally:
        session._local_browser_watchdog = None


@pytest.mark.asyncio
async def test_cancelled_start_cleans_non_pending_manager_and_root():
    session, _mocks = _unconnected_session()
    start_entered = asyncio.Event()

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager

    start_task = asyncio.create_task(session.start())
    await asyncio.wait_for(start_entered.wait(), timeout=5.0)
    start_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=5.0)

    manager.clear.assert_awaited_once()
    root.stop.assert_awaited_once()
    assert session._session_manager is None
    assert session._cdp_client_root is None
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_cancelled_start_retains_current_root_when_its_stop_fails():
    session, _mocks = _unconnected_session()
    start_entered = asyncio.Event()

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    root = MagicMock()
    root.stop = AsyncMock(side_effect=RuntimeError("root stop failed"))
    manager = MagicMock()
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager

    start_task = asyncio.create_task(session.start())
    await asyncio.wait_for(start_entered.wait(), timeout=5.0)
    start_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=5.0)

    manager.clear.assert_awaited_once()
    assert session._session_manager is manager
    assert session._cdp_client_root is root
    assert session.cdp_connection_health().cleanup_pending is True


@pytest.mark.asyncio
async def test_direct_connect_refuses_while_cancelled_start_cleanup_is_running():
    session, _mocks = _unconnected_session()
    start_entered = asyncio.Event()

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    clear_started = asyncio.Event()
    release = asyncio.Event()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()

    async def blocked_clear():
        clear_started.set()
        await asyncio.wait_for(release.wait(), timeout=5.0)

    manager.clear = AsyncMock(side_effect=blocked_clear)
    manager.cancel_pending = MagicMock()
    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    session._cdp_client_root = root
    session._session_manager = manager

    start_task = None
    try:
        start_task = asyncio.create_task(session.start())
        await asyncio.wait_for(start_entered.wait(), timeout=5.0)
        start_task.cancel()
        await asyncio.wait_for(clear_started.wait(), timeout=5.0)

        with pytest.raises(RuntimeError, match="start cleanup is still in progress"):
            await session.connect()
    finally:
        release.set()
        if start_task is not None:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(start_task, timeout=5.0)

    root.stop.assert_awaited_once()
    assert session._cdp_client_root is None
    assert session._session_manager is None


@pytest.mark.asyncio
async def test_start_owned_connect_can_proceed_while_start_task_is_active():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    mocks["cdp_root"].start = AsyncMock()
    mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": [_page_target()]})
    session._lifecycle.transition(LifecyclePhase.STARTING)
    active_start = asyncio.create_task(asyncio.sleep(60))
    session._start_task = active_start

    try:
        with (
            patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]),
            patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
            patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
        ):
            result = await session.connect(_start_owned=True)
    finally:
        active_start.cancel()
        await asyncio.wait_for(asyncio.gather(active_start, return_exceptions=True), timeout=2.0)
        session._start_task = None

    assert result is session
    assert session._cdp_client_root is mocks["cdp_root"]
    assert session.cdp_connection_health().healthy is True


@pytest.mark.asyncio
async def test_start_waits_for_an_already_admitted_direct_connect():
    session = BrowserSession()
    direct_started = asyncio.Event()
    release_direct = asyncio.Event()
    start_calls = 0

    async def fake_connect(cdp_url=None, *, _start_owned=False):
        direct_started.set()
        await asyncio.wait_for(release_direct.wait(), timeout=5.0)
        return session

    async def fake_start():
        nonlocal start_calls
        start_calls += 1
        session._lifecycle.transition(LifecyclePhase.READY, reason="start after direct connect")

    session._connect_locked = AsyncMock(side_effect=fake_connect)
    session._start_locked = AsyncMock(side_effect=fake_start)
    direct = asyncio.create_task(session.connect("ws://direct"))
    await asyncio.wait_for(direct_started.wait(), timeout=5.0)
    start = asyncio.create_task(session.start())
    await _wait_until(lambda: start in session._start_admission_tasks)

    assert start_calls == 0
    with pytest.raises(RuntimeError, match="start cleanup is still in progress"):
        await session.connect("ws://late")

    release_direct.set()
    await asyncio.wait_for(direct, timeout=5.0)
    await asyncio.wait_for(start, timeout=5.0)

    assert start_calls == 1
    assert session.lifecycle_snapshot()["phase"] == "ready"


@pytest.mark.asyncio
async def test_overlapping_starts_keep_the_direct_connect_gate_closed():
    session = BrowserSession()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    calls = 0

    async def fake_start():
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await asyncio.wait_for(release_first.wait(), timeout=5.0)
            raise RuntimeError("first start failed")
        second_started.set()
        await asyncio.wait_for(release_second.wait(), timeout=5.0)
        session._lifecycle.transition(LifecyclePhase.READY, reason="second start")

    session._start_locked = AsyncMock(side_effect=fake_start)
    first = asyncio.create_task(session.start())
    await asyncio.wait_for(first_started.wait(), timeout=5.0)
    second = asyncio.create_task(session.start())
    await _wait_until(lambda: second in session._start_admission_tasks)

    release_first.set()
    await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), timeout=2.0)
    await asyncio.wait_for(second_started.wait(), timeout=5.0)

    with pytest.raises(RuntimeError, match="start cleanup is still in progress"):
        await session.connect("ws://late")

    release_second.set()
    await asyncio.wait_for(second, timeout=5.0)

    assert calls == 2
    assert session.lifecycle_snapshot()["phase"] == "ready"


@pytest.mark.asyncio
async def test_late_start_does_not_hold_lock_needed_by_shutdown_reset():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    session._lifecycle.transition(LifecyclePhase.STOPPING)
    stop_acquired = asyncio.Event()
    release_stop = asyncio.Event()

    async def shutdown_owner():
        async with session._connect_admission_lock:
            stop_acquired.set()
            await asyncio.wait_for(release_stop.wait(), timeout=5.0)
        session._lifecycle.transition(LifecyclePhase.STOPPED)

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="late start")

    session._stop_task = asyncio.create_task(shutdown_owner())
    with patch.object(session, "_start_locked", side_effect=fake_start_locked):
        start_task = asyncio.create_task(session.start())
        await _wait_until(
            lambda: (
                start_task in session._start_admission_tasks
                and start_task._fut_waiter is not None
                and not start_task._fut_waiter.done()
            )
        )
        await asyncio.wait_for(stop_acquired.wait(), timeout=1.0)
        release_stop.set()
        await asyncio.wait_for(start_task, timeout=1.0)

    assert session.lifecycle_snapshot()["phase"] == "ready"
    assert session._connect_admission_lock.locked() is False


@pytest.mark.asyncio
async def test_kill_cancels_start_waiting_for_direct_connect_admission():
    session = BrowserSession()
    direct_started = asyncio.Event()
    release_direct = asyncio.Event()

    async def fake_connect(cdp_url=None, *, _start_owned=False):
        direct_started.set()
        await asyncio.wait_for(release_direct.wait(), timeout=5.0)
        return session

    session._connect_locked = AsyncMock(side_effect=fake_connect)
    direct = asyncio.create_task(session.connect("ws://direct"))
    await asyncio.wait_for(direct_started.wait(), timeout=5.0)
    start = asyncio.create_task(session.start())
    await _wait_until(
        lambda: (
            start in session._start_admission_tasks
            and session._connect_admission_lock.locked()
            and start._fut_waiter is not None
        )
    )
    assert start.done() is False

    await session.kill()
    release_direct.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(direct, timeout=5.0)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start, timeout=5.0)

    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._start_admission_tasks == set()


@pytest.mark.asyncio
async def test_kill_cancels_an_admitted_direct_connect():
    session = BrowserSession()
    connect_started = asyncio.Event()
    hold_connect = asyncio.Event()

    async def hanging_connect(cdp_url=None, *, _start_owned=False):
        connect_started.set()
        await asyncio.wait_for(hold_connect.wait(), timeout=5.0)

    session._connect_locked = AsyncMock(side_effect=hanging_connect)
    connect_task = asyncio.create_task(session.connect("ws://direct"))
    await asyncio.wait_for(connect_started.wait(), timeout=5.0)

    await session.kill()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(connect_task, timeout=5.0)

    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._direct_connect_tasks == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "kill"])
async def test_shutdown_cancels_admitted_direct_connect_when_navigation_retirement_fails(operation):
    from openbrowser.browser.navigation import NavigationError

    session = BrowserSession()
    connect_started = asyncio.Event()
    release_connect = asyncio.Event()

    async def hanging_connect(cdp_url=None, *, _start_owned=False):
        connect_started.set()
        await asyncio.wait_for(release_connect.wait(), timeout=5.0)
        session._cdp_client_root = MagicMock()
        return session

    session._connect_locked = AsyncMock(side_effect=hanging_connect)
    session._cancel_navigation = AsyncMock(side_effect=NavigationError("policy authority has not retired"))
    connect_task = asyncio.create_task(session.connect("ws://direct"))
    await asyncio.wait_for(connect_started.wait(), timeout=5.0)

    with pytest.raises(NavigationError, match="policy authority"):
        await getattr(session, operation)()

    release_connect.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(connect_task, timeout=5.0)

    assert session._direct_connect_tasks == set()
    assert session._cdp_client_root is None


@pytest.mark.asyncio
async def test_cancel_owned_start_does_not_deadlock_on_shutdown_wait():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    stop_task = asyncio.create_task(asyncio.sleep(60))
    session._stop_task = stop_task
    start_task = asyncio.create_task(session.start())
    try:
        await _wait_until(
            lambda: (
                start_task in session._start_admission_tasks
                and start_task._fut_waiter is not None
                and not start_task._fut_waiter.done()
            )
        )
        await asyncio.wait_for(session._cancel_owned_start(), timeout=1.0)
        assert start_task.cancelled()
    finally:
        session._stop_task = None
        stop_task.cancel()
        await asyncio.wait_for(asyncio.gather(stop_task, start_task, return_exceptions=True), timeout=2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "kill"])
async def test_concurrent_start_and_shutdown_do_not_deadlock(operation):
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)

    shutdown_entered = asyncio.Event()
    allow_cancel = asyncio.Event()
    allow_return = asyncio.Event()
    real_cancel_owned_start = session._cancel_owned_start

    async def blocked_shutdown(*, force):
        shutdown_entered.set()
        await asyncio.wait_for(allow_cancel.wait(), timeout=5.0)
        await real_cancel_owned_start()
        await asyncio.wait_for(allow_return.wait(), timeout=5.0)

    session._shutdown_locked = blocked_shutdown
    shutdown_task = asyncio.create_task(getattr(session, operation)())
    await asyncio.wait_for(shutdown_entered.wait(), timeout=5.0)
    start_task = asyncio.create_task(session.start())
    await _wait_until(lambda: start_task in session._start_admission_tasks and start_task._fut_waiter is not None)
    assert session._connect_admission_lock.locked() is False
    allow_cancel.set()
    allow_return.set()

    await asyncio.wait_for(shutdown_task, timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=5.0)
    assert session._start_admission_tasks == set()
    assert not session._connect_admission_lock.locked()


@pytest.mark.asyncio
async def test_reset_drains_an_admitted_direct_connect_before_cleanup():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    root = mocks["cdp_root"]
    root.stop = AsyncMock()
    started = asyncio.Event()

    async def blocked_start():
        started.set()
        await asyncio.sleep(60)

    root.start = AsyncMock(side_effect=blocked_start)
    connect_task = None

    try:
        with (
            patch("openbrowser.browser.session.CDPClient", return_value=root),
            patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
            patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
        ):
            connect_task = asyncio.create_task(session.connect("ws://direct"))
            await asyncio.wait_for(started.wait(), timeout=5.0)
            assert session._direct_connect_tasks
            await asyncio.wait_for(session.reset(), timeout=1.0)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(connect_task, timeout=2.0)
    finally:
        if connect_task is not None and not connect_task.done():
            connect_task.cancel()
            await asyncio.wait_for(asyncio.gather(connect_task, return_exceptions=True), timeout=2.0)

    root.stop.assert_awaited_once()
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.tab_snapshot() == []
    assert session.lifecycle_snapshot()["phase"] == "stopped"


@pytest.mark.asyncio
async def test_reset_retains_an_admitted_connect_that_suppresses_cancellation():
    session, mocks = _unconnected_session()
    manager = _manager_for(mocks)
    root = mocks["cdp_root"]
    root.stop = AsyncMock()
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_start():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    root.start = AsyncMock(side_effect=stubborn_start)

    try:
        with (
            patch("openbrowser.browser.session.CDPClient", return_value=root),
            patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
            patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
            patch("openbrowser.browser.session._shutdown_timeout", return_value=0.05),
        ):
            connect_task = asyncio.create_task(session.connect("ws://direct"))
            await asyncio.wait_for(started.wait(), timeout=5.0)
            started_at = time.monotonic()
            with pytest.raises(TimeoutError, match="did not quiesce"):
                await session.reset()
            assert time.monotonic() - started_at < 0.3
            assert session.lifecycle_snapshot()["phase"] == "failed"
            assert session.cdp_connection_health().cleanup_pending is True
            assert session._cdp_client_root is root
            assert connect_task.done() is False
            release.set()
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(connect_task, timeout=5.0)
            assert session.tab_snapshot() == []
            dispatched = [call.args[0] for call in session.event_bus.dispatch.call_args_list]
            assert not any(isinstance(event, TabCreatedEvent) for event in dispatched)
    finally:
        release.set()
        if not connect_task.done():
            connect_task.cancel()
            await asyncio.wait_for(asyncio.gather(connect_task, return_exceptions=True), timeout=2.0)


@pytest.mark.asyncio
async def test_bus_retirement_failure_keeps_retained_owners_unstartable():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    session._watchdogs_attached = True
    session.event_bus.stop = AsyncMock(side_effect=RuntimeError("bus retirement failed"))

    with pytest.raises(RuntimeError, match="bus retirement failed"):
        await session.reset(stop_event_bus=True)

    assert session.lifecycle_snapshot()["phase"] == "failed"
    assert session.cdp_connection_health().cleanup_pending is True
    assert session._cdp_client_root is root
    assert session._session_manager is manager
    with pytest.raises(RuntimeError, match="cleanup is still pending"):
        await session.start()

    session.event_bus.stop = AsyncMock()
    await session.reset(wait_for_operations=False)
    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert root.stop.await_count == 1
    assert manager.clear.await_count == 1
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_queued_reset_does_not_reopen_admission_after_stop():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    navigation_started = asyncio.Event()
    release_navigation = asyncio.Event()
    navigation_calls = 0

    async def blocked_navigation():
        nonlocal navigation_calls
        navigation_calls += 1
        if navigation_calls == 1:
            navigation_started.set()
            await asyncio.wait_for(release_navigation.wait(), timeout=5.0)

    first_reset = None
    second_reset = None
    stop_task = None
    try:
        with patch.object(session, "_cancel_navigation", side_effect=blocked_navigation):
            first_reset = asyncio.create_task(session.reset(wait_for_operations=False))
            await asyncio.wait_for(navigation_started.wait(), timeout=1.0)
            second_reset = asyncio.create_task(session.reset(wait_for_operations=False))
            await asyncio.sleep(0)
            assert second_reset.done() is False

            before_stop_epoch = session._connect_admission_epoch
            stop_task = asyncio.create_task(session.stop())
            await _wait_until(lambda: session._connect_admission_epoch > before_stop_epoch)
            release_navigation.set()
            results = await asyncio.wait_for(
                asyncio.gather(first_reset, second_reset, stop_task, return_exceptions=True),
                timeout=3.0,
            )
            assert not any(isinstance(result, BaseException) for result in results)
    finally:
        release_navigation.set()
        pending = [task for task in (first_reset, second_reset, stop_task) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)

    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._connect_admission_closed is True
    with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
        await session.connect("ws://after-queued-reset-stop")


@pytest.mark.asyncio
async def test_successful_reset_reopens_direct_connection_admission():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)

    await session.reset(wait_for_operations=False)

    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._connect_admission_closed is False
    with (
        patch(
            "openbrowser.browser.session.CDPClient",
            side_effect=RuntimeError("reached cdp client creation"),
        ),
        pytest.raises(RuntimeError, match="reached cdp client creation"),
    ):
        await session.connect("ws://invalid.invalid/devtools/browser/test")


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_name", ["stop", "kill"])
async def test_noop_shutdown_after_reset_keeps_direct_connection_fenced(shutdown_name):
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)

    await session.reset(wait_for_operations=False)
    assert session._connect_admission_closed is False

    await getattr(session, shutdown_name)()

    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._connect_admission_closed is True
    with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
        await session.connect("ws://after-shutdown")

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="start after shutdown")

    with patch.object(session, "_start_locked", side_effect=fake_start_locked):
        await session.start()
    assert session._connect_admission_closed is False


@pytest.mark.asyncio
async def test_reset_rejects_a_connect_queued_before_its_admission_epoch():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    await session._connect_admission_lock.acquire()
    connect_task = None
    reset_task = None
    try:
        connect_task = asyncio.create_task(session.connect("ws://queued"))
        await _wait_until(lambda: connect_task in session._direct_connect_tasks)
        reset_task = asyncio.create_task(session.reset(wait_for_operations=False))
        await _wait_until(lambda: session._reset_task is reset_task and session._connect_admission_closed)
    finally:
        session._connect_admission_lock.release()
        pending = [task for task in (connect_task, reset_task) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)

    assert reset_task is not None and reset_task.exception() is None
    assert connect_task is not None
    with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
        await asyncio.wait_for(connect_task, timeout=5.0)
    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._connect_admission_closed is False

    with (
        patch(
            "openbrowser.browser.session.CDPClient",
            side_effect=RuntimeError("reached cdp client creation"),
        ),
        pytest.raises(RuntimeError, match="reached cdp client creation"),
    ):
        await session.connect("ws://fresh")


@pytest.mark.asyncio
async def test_start_waits_for_reset_transaction_before_acquiring_admission():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    retire_started = asyncio.Event()
    release_retire = asyncio.Event()

    async def blocked_retire(*_args, **_kwargs):
        retire_started.set()
        await asyncio.wait_for(release_retire.wait(), timeout=5.0)

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="start after reset")

    reset_task = None
    start_task = None
    try:
        with (
            patch.object(session, "_retire_event_bus", side_effect=blocked_retire),
            patch.object(session, "_start_locked", side_effect=fake_start_locked),
        ):
            reset_task = asyncio.create_task(session.reset())
            await asyncio.wait_for(retire_started.wait(), timeout=1.0)
            start_task = asyncio.create_task(session.start())
            await _wait_until(lambda: start_task._fut_waiter is not None and not start_task._fut_waiter.done())
            assert start_task.done() is False
            release_retire.set()
            await asyncio.wait_for(reset_task, timeout=1.0)
            await asyncio.wait_for(start_task, timeout=1.0)
    finally:
        release_retire.set()
        pending = [task for task in (reset_task, start_task) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)

    assert session.lifecycle_snapshot()["phase"] == "ready"


@pytest.mark.asyncio
async def test_cancelled_overlapping_reset_cannot_hide_active_reset():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    retire_started = asyncio.Event()
    release_retire = asyncio.Event()

    async def blocked_retire(*_args, **_kwargs):
        retire_started.set()
        await asyncio.wait_for(release_retire.wait(), timeout=5.0)

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="start after reset")

    first_reset = None
    start_task = None
    try:
        with (
            patch.object(session, "_retire_event_bus", side_effect=blocked_retire),
            patch.object(session, "_start_locked", side_effect=fake_start_locked),
        ):
            first_reset = asyncio.create_task(session.reset())
            await asyncio.wait_for(retire_started.wait(), timeout=1.0)
            second_reset = asyncio.create_task(session.reset())
            await asyncio.sleep(0)
            second_reset.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(second_reset, timeout=5.0)

            start_task = asyncio.create_task(session.start())
            await asyncio.sleep(0)
            assert start_task.done() is False
            assert session.lifecycle_snapshot()["phase"] == "ready"

            release_retire.set()
            await asyncio.wait_for(first_reset, timeout=1.0)
            await asyncio.wait_for(start_task, timeout=1.0)
    finally:
        release_retire.set()
        pending = [task for task in (first_reset, start_task) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=1.0)

    assert session.lifecycle_snapshot()["phase"] == "ready"


@pytest.mark.asyncio
async def test_public_reset_stop_start_overlap_completes_without_retained_cleanup():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    drain_started = asyncio.Event()
    release_drain = asyncio.Event()
    navigation_started = asyncio.Event()
    release_navigation = asyncio.Event()
    original_drain = session._drain_reset_work

    async def blocked_drain(budget):
        drain_started.set()
        await asyncio.wait_for(release_drain.wait(), timeout=5.0)
        await original_drain(budget)

    async def blocked_navigation():
        navigation_started.set()
        await asyncio.wait_for(release_navigation.wait(), timeout=5.0)

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="start after reset and stop")

    reset_task = None
    stop_task = None
    start_task = None
    try:
        with (
            patch.object(session, "_drain_reset_work", side_effect=blocked_drain),
            patch.object(session, "_cancel_navigation", side_effect=blocked_navigation),
            patch.object(session, "_start_locked", side_effect=fake_start_locked),
            patch("openbrowser.browser.session._shutdown_timeout", return_value=1.0),
        ):
            reset_task = asyncio.create_task(session.reset())
            await asyncio.wait_for(drain_started.wait(), timeout=1.0)
            stop_task = asyncio.create_task(session.stop())
            await _wait_until(lambda: session._stop_task is not None)
            await asyncio.wait_for(navigation_started.wait(), timeout=1.0)
            start_task = asyncio.create_task(session.start())
            await asyncio.sleep(0.05)
            release_drain.set()
            await asyncio.sleep(0.05)
            release_navigation.set()
            results = await asyncio.wait_for(
                asyncio.gather(reset_task, stop_task, start_task, return_exceptions=True),
                timeout=3.0,
            )

            assert not isinstance(results[0], BaseException)
            assert not isinstance(results[1], BaseException)
            assert isinstance(results[2], asyncio.CancelledError)
            assert not isinstance(results[2], TimeoutError)
            assert session.lifecycle_snapshot()["phase"] == "stopped"
            assert session.cdp_connection_health().cleanup_pending is False
            assert session._connect_admission_closed is True
            with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
                await session.connect("ws://after-reset-stop")

            await asyncio.wait_for(session.start(), timeout=1.0)
    finally:
        release_drain.set()
        release_navigation.set()
        pending = [task for task in (reset_task, stop_task, start_task) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)

    assert session.lifecycle_snapshot()["phase"] == "ready"


@pytest.mark.asyncio
async def test_cancelled_reset_retry_propagates_cancellation_and_keeps_owners():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    clear_started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_clear():
        clear_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    manager.clear = AsyncMock(side_effect=stubborn_clear)
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    session._cdp_health.mark_cleanup_failure(RuntimeError("prior cleanup pending"))
    reset_task = None
    try:
        reset_task = asyncio.create_task(session.reset(wait_for_operations=False))
        await asyncio.wait_for(clear_started.wait(), timeout=1.0)
        reset_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reset_task, timeout=5.0)
    finally:
        release.set()
        if reset_task is not None and not reset_task.done():
            reset_task.cancel()
            await asyncio.wait_for(asyncio.gather(reset_task, return_exceptions=True), timeout=2.0)

    await _wait_until(lambda: not session._detached_cleanup_tasks)
    assert session.lifecycle_snapshot()["phase"] == "failed"
    assert session._cdp_client_root is root
    assert session._session_manager is manager
    assert session.cdp_connection_health().cleanup_pending is True

    manager.clear = AsyncMock()
    await session.reset(wait_for_operations=False)
    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_cancelled_reset_before_cleanup_restores_connection_admission():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    await session._connect_admission_lock.acquire()
    reset_task = None

    try:
        reset_task = asyncio.create_task(session.reset(wait_for_operations=False))
        await _wait_until(lambda: session._connect_admission_closed and session._reset_task is not None)
        reset_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reset_task, timeout=5.0)
    finally:
        session._connect_admission_lock.release()
        if reset_task is not None and not reset_task.done():
            reset_task.cancel()
            await asyncio.wait_for(asyncio.gather(reset_task, return_exceptions=True), timeout=2.0)

    assert session.lifecycle_snapshot()["phase"] == "ready"
    assert session._connect_admission_closed is False
    await session.start()
    assert session.lifecycle_snapshot()["phase"] == "ready"


@pytest.mark.asyncio
async def test_cancelled_reset_after_cleanup_starts_fails_the_lifecycle():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    navigation_started = asyncio.Event()
    release = asyncio.Event()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    manager.clear = AsyncMock()
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager

    async def blocked_navigation():
        navigation_started.set()
        await asyncio.wait_for(release.wait(), timeout=5.0)

    reset_task = None
    try:
        with patch.object(session, "_cancel_navigation", side_effect=blocked_navigation):
            reset_task = asyncio.create_task(session.reset(wait_for_operations=False))
            await asyncio.wait_for(navigation_started.wait(), timeout=1.0)
            reset_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(reset_task, timeout=5.0)
    finally:
        release.set()
        if reset_task is not None and not reset_task.done():
            reset_task.cancel()
            await asyncio.wait_for(asyncio.gather(reset_task, return_exceptions=True), timeout=2.0)

    assert session.lifecycle_snapshot()["phase"] == "failed"
    assert session._connect_admission_closed is True
    assert session._cdp_client_root is root
    assert session._session_manager is manager
    assert session.cdp_connection_health().cleanup_pending is True

    await session.reset(wait_for_operations=False)
    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert root.stop.await_count == 1
    assert manager.clear.await_count == 1
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_stop_rechecks_cleanup_ownership_after_waiting_for_reset_lock():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    clear_started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_clear():
        clear_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    manager.clear = AsyncMock(side_effect=stubborn_clear)
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    dispatch = MagicMock()

    try:
        with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.05):
            reset_task = asyncio.create_task(session.reset(wait_for_operations=False))
            await asyncio.wait_for(clear_started.wait(), timeout=1.0)
            session.event_bus.dispatch = dispatch
            stop_task = asyncio.create_task(session.stop())
            await asyncio.sleep(0)
            with pytest.raises(TimeoutError, match="CDP session manager"):
                await asyncio.wait_for(reset_task, timeout=5.0)
            with pytest.raises(TimeoutError, match="Prior browser cleanup is still running"):
                await asyncio.wait_for(stop_task, timeout=5.0)

        dispatch.assert_not_called()
        assert session._stop_task is None
        assert session._detached_cleanup_tasks

        release.set()
        await _wait_until(lambda: not session._detached_cleanup_tasks)
        manager.clear = AsyncMock()
        with patch("openbrowser.browser.session._shutdown_timeout", return_value=1.0):
            await session.reset(wait_for_operations=False)
        assert session.lifecycle_snapshot()["phase"] == "stopped"
        assert session.cdp_connection_health().cleanup_pending is False
    finally:
        release.set()
        await session._retire_event_bus(session.event_bus, timeout=1.0)


@pytest.mark.asyncio
async def test_stop_handler_does_not_join_an_active_reset_inside_the_event_bus(monkeypatch):
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    active_reset = asyncio.create_task(asyncio.sleep(60))
    session._reset_task = active_reset
    reset = AsyncMock()
    monkeypatch.setattr(BrowserSession, "reset", reset)

    try:
        await asyncio.wait_for(session.on_BrowserStopEvent(BrowserStopEvent(force=True)), timeout=1.0)
        reset.assert_not_awaited()
    finally:
        active_reset.cancel()
        await asyncio.wait_for(asyncio.gather(active_reset, return_exceptions=True), timeout=2.0)
        await session._retire_event_bus(session.event_bus, timeout=1.0)


@pytest.mark.asyncio
async def test_ready_start_waits_for_in_flight_stop_task():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    release_stop = asyncio.Event()

    async def stop_owner():
        await asyncio.wait_for(release_stop.wait(), timeout=5.0)
        session._lifecycle.transition(LifecyclePhase.STOPPED)

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="start after stop")

    session._stop_task = asyncio.create_task(stop_owner())
    start_task = None
    try:
        with patch.object(session, "_start_locked", side_effect=fake_start_locked):
            start_task = asyncio.create_task(session.start())
            await asyncio.sleep(0)
            assert start_task.done() is False
            release_stop.set()
            await asyncio.wait_for(start_task, timeout=1.0)
    finally:
        release_stop.set()
        pending = [task for task in (session._stop_task, start_task) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)

    assert session.lifecycle_snapshot()["phase"] == "ready"


@pytest.mark.asyncio
async def test_reset_fences_direct_connect_for_the_full_transaction():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    retire_started = asyncio.Event()
    release_retire = asyncio.Event()

    async def blocked_retire(*_args, **_kwargs):
        retire_started.set()
        await asyncio.wait_for(release_retire.wait(), timeout=5.0)

    reset_task = None
    connect_task = None
    try:
        with patch.object(session, "_retire_event_bus", side_effect=blocked_retire):
            reset_task = asyncio.create_task(session.reset())
            await asyncio.wait_for(retire_started.wait(), timeout=1.0)
            connect_task = asyncio.create_task(session.connect("ws://late"))
            with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
                await asyncio.wait_for(connect_task, timeout=5.0)
            release_retire.set()
            await asyncio.wait_for(reset_task, timeout=5.0)
    finally:
        release_retire.set()
        pending = [task for task in (reset_task, connect_task) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=1.0)
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.lifecycle_snapshot()["phase"] == "stopped"


@pytest.mark.asyncio
async def test_reset_bounds_prior_stop_drain_and_retains_failed_state():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    session._lifecycle.transition(LifecyclePhase.STOPPING)
    stop_started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_stop():
        stop_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    stop_task = asyncio.create_task(stubborn_stop())
    await asyncio.wait_for(stop_started.wait(), timeout=5.0)
    session._stop_task = stop_task
    try:
        with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.05):
            started_at = time.monotonic()
            with pytest.raises(TimeoutError, match="prior stop"):
                await session.reset()
            assert time.monotonic() - started_at < 0.3

        assert session.lifecycle_snapshot()["phase"] == "failed"
        assert session.cdp_connection_health().cleanup_pending is True
        assert stop_task.done() is False
    finally:
        stop_task.cancel()
        release.set()
        await asyncio.wait_for(asyncio.gather(stop_task, return_exceptions=True), timeout=2.0)
        session._stop_task = None


@pytest.mark.asyncio
async def test_reset_from_stopping_restores_startable_state():
    from openbrowser.browser.navigation import NavigationError

    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    session._cancel_navigation = AsyncMock(side_effect=NavigationError("policy authority has not retired"))

    with pytest.raises(NavigationError, match="policy authority"):
        await session.stop()

    assert session.lifecycle_snapshot()["phase"] == "stopping"
    assert session._connect_admission_closed is True

    session._cancel_navigation = AsyncMock()
    await session.reset(wait_for_operations=False)

    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session._connect_admission_closed is False

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="test start")

    session._start_locked = fake_start_locked
    await session.start()

    assert session.lifecycle_snapshot()["phase"] == "ready"
    assert session._connect_admission_closed is False


@pytest.mark.asyncio
async def test_failed_shutdown_keeps_admission_closed_after_start_transition_rejection():
    from openbrowser.browser.lifecycle import LifecycleTransitionError
    from openbrowser.browser.navigation import NavigationError

    session = BrowserSession()
    real_connect_locked = session._connect_locked
    connect_started = asyncio.Event()
    release_connect = asyncio.Event()

    async def hanging_connect(cdp_url=None, *, _start_owned=False):
        connect_started.set()
        await asyncio.wait_for(release_connect.wait(), timeout=5.0)
        return session

    session._connect_locked = AsyncMock(side_effect=hanging_connect)
    connect_task = asyncio.create_task(session.connect("ws://direct"))
    await asyncio.wait_for(connect_started.wait(), timeout=5.0)
    session._cancel_navigation = AsyncMock(side_effect=NavigationError("policy authority has not retired"))

    with pytest.raises(NavigationError, match="policy authority"):
        await session.stop()

    release_connect.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(connect_task, timeout=5.0)

    with pytest.raises(LifecycleTransitionError):
        await session.start()

    session._connect_locked = real_connect_locked
    with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
        await session.connect("ws://after-failed-shutdown")


@pytest.mark.asyncio
async def test_cancelled_queued_connect_leaves_the_admission_set():
    session = BrowserSession()
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def fake_connect(cdp_url=None, *, _start_owned=False):
        first_started.set()
        await asyncio.wait_for(release_first.wait(), timeout=5.0)
        return session

    session._connect_locked = AsyncMock(side_effect=fake_connect)
    first = asyncio.create_task(session.connect("ws://first"))
    await asyncio.wait_for(first_started.wait(), timeout=5.0)
    queued = asyncio.create_task(session.connect("ws://queued"))
    await _wait_until(lambda: queued in session._direct_connect_tasks)
    queued.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(queued, timeout=5.0)

    release_first.set()
    await asyncio.wait_for(first, timeout=5.0)

    assert session._direct_connect_tasks == set()
    assert session._has_start_in_flight() is False


@pytest.mark.asyncio
async def test_failed_start_restores_a_previously_closed_admission_fence():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    await session.kill()
    assert session._connect_admission_closed is True

    session.event_bus.dispatch = MagicMock(side_effect=RuntimeError("start dispatch failed"))
    with pytest.raises(RuntimeError, match="start dispatch failed"):
        await session.start()

    assert session.lifecycle_snapshot()["phase"] == "failed"
    assert session._connect_admission_closed is True
    with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
        await session.connect("ws://after-failed-start")

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="start retry after failure")

    with patch.object(session, "_start_locked", side_effect=fake_start_locked):
        await session.start()
    assert session.lifecycle_snapshot()["phase"] == "ready"
    assert session._connect_admission_closed is False


@pytest.mark.asyncio
async def test_cancelled_start_restores_a_previously_closed_admission_fence():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    await session.kill()
    assert session._connect_admission_closed is True
    start_entered = asyncio.Event()

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    start_task = None
    try:
        start_task = asyncio.create_task(session.start())
        await asyncio.wait_for(start_entered.wait(), timeout=1.0)
        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(start_task, timeout=2.0)
    finally:
        if start_task is not None and not start_task.done():
            start_task.cancel()
            await asyncio.wait_for(asyncio.gather(start_task, return_exceptions=True), timeout=2.0)

    assert session.lifecycle_snapshot()["phase"] == "failed"
    assert session._connect_admission_closed is True
    with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
        await session.connect("ws://after-cancelled-start")

    async def fake_start_locked():
        session._lifecycle.transition(LifecyclePhase.READY, reason="start retry after cancellation")

    with patch.object(session, "_start_locked", side_effect=fake_start_locked):
        await session.start()
    assert session._connect_admission_closed is False


@pytest.mark.asyncio
async def test_cancelled_start_keeps_a_previously_open_admission_fence_open():
    session = BrowserSession()
    assert session._connect_admission_closed is False
    start_entered = asyncio.Event()

    async def hanging_start():
        start_entered.set()
        await asyncio.sleep(60)

    session.event_bus.dispatch = MagicMock(return_value=hanging_start())
    start_task = None
    try:
        start_task = asyncio.create_task(session.start())
        await asyncio.wait_for(start_entered.wait(), timeout=1.0)
        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(start_task, timeout=2.0)
    finally:
        if start_task is not None and not start_task.done():
            start_task.cancel()
            await asyncio.wait_for(asyncio.gather(start_task, return_exceptions=True), timeout=2.0)

    assert session.lifecycle_snapshot()["phase"] == "failed"
    assert session._connect_admission_closed is False
    with (
        patch(
            "openbrowser.browser.session.CDPClient",
            side_effect=RuntimeError("reached cdp client creation"),
        ),
        pytest.raises(RuntimeError, match="reached cdp client creation"),
    ):
        await session.connect("ws://after-cancelled-open-start")


@pytest.mark.asyncio
async def test_connect_after_shutdown_requires_start():
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    await session.kill()

    with pytest.raises(RuntimeError, match="shutdown closed direct connections"):
        await session.connect("ws://after-stop")


@pytest.mark.asyncio
async def test_start_refuses_while_cdp_cleanup_is_pending():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    manager = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._cdp_health.mark_cleanup_failure(RuntimeError("retained after failed connect"))
    before = session.lifecycle_snapshot()

    with pytest.raises(RuntimeError, match="CDP cleanup is still pending"):
        await session.start()

    after = session.lifecycle_snapshot()
    assert session._cdp_client_root is root
    assert session._session_manager is manager
    assert after["phase"] == before["phase"]
    assert after["generation"] == before["generation"]


@pytest.mark.asyncio
async def test_retry_start_rechecks_cleanup_pending_after_waiting_for_previous_start():
    session = BrowserSession()
    first_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_start():
        nonlocal calls
        calls += 1
        if calls == 1:
            session._lifecycle.fail("cancelled start")
            first_started.set()
            await asyncio.wait_for(release.wait(), timeout=5.0)
            session._cdp_health.mark_cleanup_failure(RuntimeError("retained cleanup"))
            raise asyncio.CancelledError()
        session._lifecycle.transition(LifecyclePhase.READY, reason="retry start")

    session._start_locked = AsyncMock(side_effect=fake_start)
    first = asyncio.create_task(session.start())
    await asyncio.wait_for(first_started.wait(), timeout=5.0)
    retry = asyncio.create_task(session.start())
    release.set()

    await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), timeout=2.0)
    with pytest.raises(RuntimeError, match="CDP cleanup is still pending"):
        await asyncio.wait_for(retry, timeout=5.0)

    assert calls == 1
    assert session.cdp_connection_health().cleanup_pending is True


@pytest.mark.asyncio
async def test_kill_retries_retained_cdp_root_stop_before_clearing_pointer():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock()
    session._cdp_client_root = root
    session._cdp_health.mark_cleanup_failure(RuntimeError("retained after failed connect"))

    with patch.object(session, "_shutdown_locked", AsyncMock()) as shutdown:
        await session.kill()

    root.stop.assert_awaited_once()
    assert session._cdp_client_root is None
    assert session.cdp_connection_health().cleanup_pending is False
    shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_kill_retries_both_retained_resources_after_partial_failure():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock(side_effect=[RuntimeError("root stop failed"), None])
    manager = MagicMock()
    manager.clear = AsyncMock(side_effect=[RuntimeError("manager clear failed"), None])
    session._cdp_client_root = root
    session._session_manager = manager
    session._cdp_health.mark_cleanup_failure(RuntimeError("retained after failed connect"))

    with pytest.raises(RuntimeError, match="Failed to release retained CDP resources"):
        await session.kill()

    assert session._cdp_client_root is root
    assert session._session_manager is manager
    assert session.cdp_connection_health().cleanup_pending is True

    with patch.object(session, "_shutdown_locked", AsyncMock()):
        await session.kill()

    assert root.stop.await_count == 2
    assert manager.clear.await_count == 2
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_reset_retries_retained_resources_after_partial_failure():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock(side_effect=[RuntimeError("root stop failed"), None])
    manager = MagicMock()
    manager.clear = AsyncMock(side_effect=[RuntimeError("manager clear failed"), None])
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._cdp_health.mark_cleanup_failure(RuntimeError("retained after failed connect"))

    with pytest.raises(RuntimeError, match="Retained CDP resources could not be cleaned up"):
        await session.reset(wait_for_operations=False)

    assert session._cdp_client_root is root
    assert session._session_manager is manager

    await session.reset(wait_for_operations=False)

    assert root.stop.await_count == 2
    assert manager.clear.await_count == 2
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_reset_cleanup_failure_retains_owners_and_blocks_start():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    manager.clear = AsyncMock(side_effect=RuntimeError("reset manager clear failed"))
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)

    with pytest.raises(RuntimeError, match="reset manager clear failed"):
        await session.reset(wait_for_operations=False)

    assert session.lifecycle_snapshot()["phase"] == "failed"
    assert session._cdp_client_root is root
    assert session._session_manager is manager
    assert session.cdp_connection_health().cleanup_pending is True
    assert session.browser_health_snapshot()["phase"] == "failed"
    assert session.browser_health_snapshot()["reason"] == "cleanup_failed"
    with pytest.raises(RuntimeError, match="CDP cleanup is still pending"):
        await session.start()

    manager.clear = AsyncMock()
    await session.reset(wait_for_operations=False)

    assert root.stop.await_count == 2
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.cdp_connection_health().cleanup_pending is False
    assert session.lifecycle_snapshot()["phase"] == "stopped"
    assert session.browser_health_snapshot()["phase"] == "stopped"
    assert session.browser_health_snapshot()["reason"] == "reset"


@pytest.mark.asyncio
async def test_reset_bounds_cancellation_resistant_manager_clear():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    root.stop = AsyncMock()
    manager = MagicMock()
    started = asyncio.Event()
    release = asyncio.Event()
    clear_calls = 0

    async def stubborn_clear():
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await asyncio.wait_for(release.wait(), timeout=5.0)

    manager.clear = AsyncMock(side_effect=stubborn_clear)
    manager.cancel_pending = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)

    with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.05):
        started_at = time.monotonic()
        with pytest.raises(TimeoutError, match="CDP session manager deadline expired"):
            await session.reset(wait_for_operations=False)
        assert time.monotonic() - started_at < 0.3
        assert session.lifecycle_snapshot()["phase"] == "failed"

    assert started.is_set()
    assert session._cdp_client_root is root
    assert session._session_manager is manager
    assert session.cdp_connection_health().cleanup_pending is True
    assert session._detached_cleanup_tasks

    release.set()
    await _wait_until(lambda: not session._detached_cleanup_tasks)
    with patch("openbrowser.browser.session._shutdown_timeout", return_value=1.0):
        await session.reset(wait_for_operations=False)

    root.stop.assert_awaited_once()
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_connect_refuses_to_overwrite_cleanup_pending_resources():
    session, _mocks = _unconnected_session()
    root = MagicMock()
    manager = MagicMock()
    session._cdp_client_root = root
    session._session_manager = manager
    session._cdp_health.mark_cleanup_failure(RuntimeError("retained after failed connect"))

    with (
        patch("openbrowser.browser.session.CDPClient") as client_factory,
        pytest.raises(RuntimeError, match="cleanup is still pending"),
    ):
        await session.connect()

    client_factory.assert_not_called()
    assert session._cdp_client_root is root
    assert session._session_manager is manager


@pytest.mark.asyncio
async def test_connect_cleanup_uses_one_shared_deadline_for_manager_and_root():
    session, _mocks = _unconnected_session()
    manager = MagicMock()

    async def slow_clear():
        await asyncio.sleep(0.2)

    manager.clear = AsyncMock(side_effect=slow_clear)
    root = MagicMock()
    root.stop = AsyncMock()

    with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.05):
        error = await session._cleanup_failed_cdp_connect(root, manager)

    assert isinstance(error, TimeoutError)
    root.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_deadline_holds_when_root_suppresses_cancellation():
    session, _mocks = _unconnected_session()
    release = asyncio.Event()

    async def stubborn_stop():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    root = MagicMock()
    root.stop = AsyncMock(side_effect=stubborn_stop)
    session._cdp_client_root = root

    with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.02):
        started = time.monotonic()
        error = await session._cleanup_current_cdp_ownership()
        elapsed = time.monotonic() - started

    assert isinstance(error, TimeoutError)
    assert elapsed < 0.2
    assert session._cdp_client_root is root
    assert session.cdp_connection_health().cleanup_pending is True

    release.set()
    await _wait_until(lambda: session._detached_cleanup_tasks == set())


@pytest.mark.asyncio
async def test_cleanup_retry_refuses_while_previous_child_is_running():
    session, _mocks = _unconnected_session()
    release = asyncio.Event()

    async def stubborn_stop():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    root = MagicMock()
    root.stop = AsyncMock(side_effect=stubborn_stop)
    session._cdp_client_root = root

    with patch("openbrowser.browser.session._shutdown_timeout", return_value=0.02):
        first = await session._cleanup_current_cdp_ownership()
        second = await session._cleanup_current_cdp_ownership()

    assert isinstance(first, TimeoutError)
    assert isinstance(second, TimeoutError)
    assert root.stop.await_count == 1
    assert session._cdp_client_root is root
    assert session.cdp_connection_health().cleanup_pending is True

    release.set()
    await _wait_until(lambda: session._detached_cleanup_tasks == set())
    root.stop = AsyncMock()
    with patch("openbrowser.browser.session._shutdown_timeout", return_value=1.0):
        third = await session._cleanup_current_cdp_ownership()

    assert third is None
    assert session._cdp_client_root is None
    assert session.cdp_connection_health().cleanup_pending is False


@pytest.mark.asyncio
async def test_cancelled_bounded_cleanup_retains_child_and_blocks_retry():
    session, _mocks = _unconnected_session()
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_stop():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)

    root = MagicMock()
    root.stop = AsyncMock(side_effect=stubborn_stop)
    operation = asyncio.create_task(
        session._await_bounded_operation(root.stop(), 60.0, "CDP root", retained_tasks=session._detached_cleanup_tasks)
    )
    await asyncio.wait_for(started.wait(), timeout=5.0)
    operation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=5.0)

    assert len(session._detached_cleanup_tasks) == 1
    assert any(not task.done() for task in session._detached_cleanup_tasks)
    with patch("openbrowser.browser.session._shutdown_timeout", return_value=1.0):
        retry_error = await session._cleanup_failed_cdp_connect(None, None)
    assert isinstance(retry_error, TimeoutError)

    release.set()
    await _wait_until(lambda: session._detached_cleanup_tasks == set())
    assert session._detached_cleanup_tasks == set()


@pytest.mark.asyncio
async def test_worker_only_probe_cancellation_records_failure_without_cancelling_caller():
    session, mocks = _ready_session()
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=asyncio.CancelledError())

    health = await session.probe_cdp_connection()

    assert asyncio.current_task().cancelling() == 0
    assert health.phase is CdpConnectionPhase.DEGRADED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.PROBE_FAILED
    assert health.probe_attempts == 1


@pytest.mark.asyncio
async def test_probe_cancellation_propagates_without_recording_failure():
    session, mocks = _ready_session()
    release = asyncio.Event()
    calls = 0

    async def version():
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.wait_for(release.wait(), timeout=5.0)
        return {"product": "Chrome"}

    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=version)
    probe_task = asyncio.create_task(session.probe_cdp_connection())
    try:
        await asyncio.sleep(0)
        probe_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(probe_task, timeout=2.0)
    finally:
        release.set()
        if not probe_task.done():
            probe_task.cancel()
            await asyncio.wait_for(asyncio.gather(probe_task, return_exceptions=True), timeout=2.0)

    after_cancel = session.cdp_connection_health()
    assert after_cancel.last_error_code is None
    assert after_cancel.last_failure_at is None

    renewed = await session.probe_cdp_connection()
    assert renewed.phase is CdpConnectionPhase.CONNECTED
    assert renewed.healthy is True
    assert renewed.probe_attempts == 2


@pytest.mark.asyncio
async def test_probe_caller_cancellation_retains_resistant_child():
    session, mocks = _ready_session()
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_version():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)
        return {"product": "Chrome"}

    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=stubborn_version)
    probe_task = asyncio.create_task(session.probe_cdp_connection())
    await asyncio.wait_for(started.wait(), timeout=5.0)
    probe_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(probe_task, timeout=5.0)

    assert len(session._probe_slots) == 1
    assert len(session._detached_probe_tasks) == 1
    before = session.cdp_connection_health()

    release.set()
    await _wait_until(lambda: not session._probe_slots and not session._detached_probe_tasks)
    assert session.cdp_connection_health() == before


@pytest.mark.asyncio
async def test_probe_deadline_holds_when_command_suppresses_cancellation():
    session, mocks = _ready_session()
    release = asyncio.Event()

    async def stubborn_version():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), timeout=5.0)
        return {"product": "Chrome"}

    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=stubborn_version)
    started = time.monotonic()
    health = await session.probe_cdp_connection(timeout_seconds=0.02)
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert health.phase is CdpConnectionPhase.DEGRADED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.TIMEOUT
    assert len(session._detached_probe_tasks) == 1
    assert any(not task.done() for task in session._detached_probe_tasks)

    release.set()
    await _wait_until(lambda: session._detached_probe_tasks == set())
    assert session._detached_probe_tasks == set()
    after = session.cdp_connection_health()
    assert after.phase is CdpConnectionPhase.DEGRADED
    assert after.healthy is False
    assert after.last_error_code is CdpConnectionErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_retained_probe_limit_skips_without_changing_health():
    session, mocks = _ready_session()
    releases = [asyncio.Event() for _ in range(MAX_RETAINED_PROBES)]
    calls = 0

    async def stubborn_version():
        nonlocal calls
        index = calls
        calls += 1
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(releases[index].wait(), timeout=5.0)
        return {"product": "Chrome"}

    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=stubborn_version)
    try:
        for _ in range(MAX_RETAINED_PROBES):
            health = await session.probe_cdp_connection(timeout_seconds=0.01)
            assert health.last_error_code is CdpConnectionErrorCode.TIMEOUT
        assert len(session._detached_probe_tasks) == MAX_RETAINED_PROBES
        before = session.cdp_connection_health()

        skipped = await session.probe_cdp_connection(timeout_seconds=0.01)

        after = session.cdp_connection_health()
        assert after == before
        assert skipped.probe_attempts == before.probe_attempts
        assert calls == MAX_RETAINED_PROBES
        assert len(session._detached_probe_tasks) == MAX_RETAINED_PROBES
    finally:
        for release in releases:
            release.set()
        await _wait_until(lambda: session._detached_probe_tasks == set())
    final = session.cdp_connection_health()
    assert final.phase is CdpConnectionPhase.DEGRADED
    assert final.healthy is False
    assert final.last_error_code is CdpConnectionErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_concurrent_probe_burst_respects_the_probe_cap():
    session, mocks = _ready_session()
    releases = [asyncio.Event() for _ in range(MAX_RETAINED_PROBES)]
    started = 0

    async def stubborn_version():
        nonlocal started
        index = started
        started += 1
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.wait_for(releases[index].wait(), timeout=5.0)
        return {"product": "Chrome"}

    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=stubborn_version)
    tasks = [
        asyncio.create_task(session.probe_cdp_connection(timeout_seconds=0.01)) for _ in range(MAX_RETAINED_PROBES + 4)
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=3.0)

        assert started == MAX_RETAINED_PROBES
        assert session.cdp_connection_health().probe_attempts == MAX_RETAINED_PROBES
        assert len(session._probe_slots) == MAX_RETAINED_PROBES
        assert len(session._detached_probe_tasks) == MAX_RETAINED_PROBES
    finally:
        for release in releases:
            release.set()
        pending = [task for task in tasks if not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)
        await _wait_until(lambda: not session._probe_slots and not session._detached_probe_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout_value",
    [
        0,
        -1.0,
        None,
        float("nan"),
        float("inf"),
        "not-a-number",
        pytest.param(10**10000, id="huge-int"),
    ],
)
async def test_probe_timeout_normalizes_invalid_values(timeout_value):
    session, mocks = _ready_session()
    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(return_value={"product": "Chrome"})
    effective_timeouts = []

    async def record_bounded(awaitable, remaining, label, *, retained_tasks=None):
        effective_timeouts.append(remaining)
        await awaitable

    with patch.object(session, "_await_bounded_operation", AsyncMock(side_effect=record_bounded)):
        health = await session.probe_cdp_connection(timeout_seconds=timeout_value)

    assert len(effective_timeouts) == 1
    assert math.isfinite(effective_timeouts[0])
    assert effective_timeouts[0] > 0
    assert health.phase is CdpConnectionPhase.CONNECTED
    assert health.healthy is True


@pytest.mark.asyncio
async def test_stale_probe_failure_after_root_replacement_is_ignored():
    session, mocks = _ready_session()
    generation = session._cdp_health.begin_connect()
    session._cdp_health.mark_connected(generation)
    release = asyncio.Event()

    async def delayed_failure():
        await asyncio.wait_for(release.wait(), timeout=5.0)
        raise TimeoutError("stale root timed out")

    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=delayed_failure)
    probe_task = asyncio.create_task(session.probe_cdp_connection())
    try:
        await asyncio.sleep(0)
        session._cdp_client_root = MagicMock()
        release.set()

        health = await asyncio.wait_for(probe_task, timeout=2.0)
    finally:
        release.set()
        if not probe_task.done():
            probe_task.cancel()
            await asyncio.wait_for(asyncio.gather(probe_task, return_exceptions=True), timeout=2.0)

    assert health.phase is CdpConnectionPhase.CONNECTED
    assert health.healthy is True
    assert health.last_error_code is None
    assert health.last_failure_at is None


@pytest.mark.asyncio
async def test_overlapping_session_probes_keep_newest_result():
    session, mocks = _ready_session()
    older_release = asyncio.Event()
    newer_release = asyncio.Event()
    newer_started = asyncio.Event()
    calls = 0

    async def version():
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.wait_for(older_release.wait(), timeout=5.0)
            raise TimeoutError("older probe timed out")
        newer_started.set()
        await asyncio.wait_for(newer_release.wait(), timeout=5.0)
        return {"product": "Chrome"}

    mocks["cdp_root"].send.Browser.getVersion = AsyncMock(side_effect=version)
    older = asyncio.create_task(session.probe_cdp_connection())
    newer = None
    try:
        await asyncio.sleep(0)
        newer = asyncio.create_task(session.probe_cdp_connection())
        await asyncio.wait_for(newer_started.wait(), timeout=5.0)

        newer_release.set()
        newer_health = await asyncio.wait_for(newer, timeout=2.0)
        older_release.set()
        older_health = await asyncio.wait_for(older, timeout=2.0)
    finally:
        older_release.set()
        newer_release.set()
        pending = [task for task in (older, newer) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)

    assert newer_health.phase is CdpConnectionPhase.CONNECTED
    assert newer_health.healthy is True
    assert older_health.phase is CdpConnectionPhase.CONNECTED
    assert older_health.healthy is True
    assert older_health.last_error_code is None
    assert older_health.last_failure_at is None


@pytest.mark.asyncio
async def test_connect_attempts_are_serialized():
    session = BrowserSession()
    first_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0
    calls = 0

    async def fake_connect(cdp_url=None, *, _start_owned=False):
        nonlocal active, max_active, calls
        calls += 1
        active += 1
        max_active = max(max_active, active)
        if calls == 1:
            first_started.set()
            await asyncio.wait_for(release.wait(), timeout=5.0)
        active -= 1
        return session

    session._connect_locked = AsyncMock(side_effect=fake_connect)
    first = asyncio.create_task(session.connect("ws://first"))
    second = None
    try:
        await asyncio.wait_for(first_started.wait(), timeout=5.0)
        second = asyncio.create_task(session.connect("ws://second"))
        await _wait_until(lambda: second in session._direct_connect_tasks)

        assert calls == 1
        assert second.done() is False

        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=3.0)
    finally:
        release.set()
        pending = [task for task in (first, second) if task is not None and not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2.0)
    assert calls == 2
    assert max_active == 1


@pytest.mark.asyncio
async def test_retry_start_waits_for_previous_cancelled_start_cleanup():
    session = BrowserSession()
    first_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_start():
        nonlocal calls
        calls += 1
        if calls == 1:
            session._lifecycle.fail("cancelled start")
            first_started.set()
            await asyncio.wait_for(release.wait(), timeout=5.0)
            raise asyncio.CancelledError()
        session._lifecycle.transition(LifecyclePhase.READY, reason="retry start")

    session._start_locked = AsyncMock(side_effect=fake_start)
    first = asyncio.create_task(session.start())
    await asyncio.wait_for(first_started.wait(), timeout=5.0)
    retry = asyncio.create_task(session.start())
    await _wait_until(lambda: retry in session._start_admission_tasks)

    assert calls == 1
    assert retry.done() is False
    assert session.lifecycle_snapshot()["phase"] == "failed"

    release.set()
    await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), timeout=2.0)
    await asyncio.wait_for(retry, timeout=5.0)

    assert calls == 2
    assert session.lifecycle_snapshot()["phase"] == "ready"


@pytest.mark.asyncio
async def test_no_root_probe_health_becomes_unknown_after_session_reset():
    session, _mocks = _ready_session()
    generation = session._cdp_health.begin_connect()
    session._cdp_health.mark_connected(generation)
    session._cdp_client_root = None

    probed = await session.probe_cdp_connection()
    assert probed.healthy is False
    assert probed.last_error_code is CdpConnectionErrorCode.NOT_CONNECTED
    probe_attempts = probed.probe_attempts
    last_probe_at = probed.last_probe_at
    last_failure_at = probed.last_failure_at

    await session.reset(wait_for_operations=False)

    health = session.lifecycle_snapshot()["cdp"]
    assert health["phase"] == "disconnected"
    assert health["healthy"] is None
    assert health["last_error_code"] == "not_connected"
    assert health["probe_attempts"] == probe_attempts
    assert health["last_probe_at"] == last_probe_at
    assert health["last_failure_at"] == last_failure_at


def test_cdp_health_tracker_is_isolated_between_browser_sessions():
    first = BrowserSession()
    second = BrowserSession()
    generation = first._cdp_health.begin_connect()
    first._cdp_health.mark_connect_failed(generation, RuntimeError("first failed"), client_present=False)

    first_health = first.cdp_connection_health()
    second_health = second.cdp_connection_health()

    assert first_health.connect_attempts == 1
    assert first_health.last_error_code is CdpConnectionErrorCode.CONNECT_FAILED
    assert second_health.connect_attempts == 0
    assert second_health.last_error_code is None


@pytest.mark.asyncio
async def test_presetup_connect_failure_clears_existing_session_state():
    session, _mocks = _make_session_with_mocks()
    navigation_root = MagicMock()
    session._cdp_client_root = None
    session._session_manager = None
    observer = MagicMock()
    session._navigation_monitor.bind(navigation_root, observer=observer)
    navigation_callback = navigation_root.register.Page.frameNavigated.call_args.args[0]
    navigation_callback({}, "stale-session")
    observer.assert_called_once()
    observer.reset_mock()
    active_navigation = NavigationOperation(
        navigation_id="stale-navigation",
        generation=session._lifecycle.generation,
        requested_url="https://example.test/",
        wait_until="load",
    )
    session._navigation_monitor.active = active_navigation
    session._cdp_session_pool["stale"] = MagicMock()
    session._tab_registry.register("target-page-1234", state=TabState.OPEN)
    session.browser_profile.cdp_url = None
    before = session.lifecycle_snapshot()

    with pytest.raises(RuntimeError, match="Cannot setup CDP connection without CDP URL"):
        await session.connect(cdp_url=None)

    after = session.lifecycle_snapshot()
    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.agent_focus is None
    assert session._navigation_monitor.active is None
    assert session._cdp_session_pool == {}
    assert session.tab_snapshot() == []
    navigation_callback({}, "stale-session")
    observer.assert_not_called()
    assert after["phase"] == before["phase"]
    assert after["generation"] == before["generation"]
    assert after["cdp"]["phase"] == "failed"


@pytest.mark.asyncio
async def test_presetup_discovery_failure_clears_existing_session_state():
    session, _mocks = _make_session_with_mocks()
    session._cdp_client_root = None
    session._session_manager = None
    session.browser_profile.cdp_url = "http://127.0.0.1:9222"
    session._cdp_session_pool["stale"] = MagicMock()
    session._tab_registry.register("target-page-1234", state=TabState.OPEN)
    client = AsyncMock()
    client.get = AsyncMock(side_effect=__import__("httpx").ConnectError("refused"))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    fake_loop = MagicMock()
    fake_loop.time = MagicMock(side_effect=[0.0, 31.0])

    with (
        patch("httpx.AsyncClient", return_value=client),
        patch("asyncio.get_event_loop", return_value=fake_loop),
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(TimeoutError, match="did not respond"),
    ):
        await session.connect()

    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.agent_focus is None
    assert session._cdp_session_pool == {}
    assert session.tab_snapshot() == []
    assert session.cdp_connection_health().last_error_code is CdpConnectionErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_presetup_connect_cancellation_clears_existing_session_state():
    session, _mocks = _make_session_with_mocks()
    session._cdp_client_root = None
    session._session_manager = None
    session.browser_profile.cdp_url = "http://127.0.0.1:9222"
    session._cdp_session_pool["stale"] = MagicMock()
    session._tab_registry.register("target-page-1234", state=TabState.OPEN)
    entered = asyncio.Event()
    client = AsyncMock()

    async def pending_get(*_args, **_kwargs):
        entered.set()
        await asyncio.sleep(60)

    client.get = AsyncMock(side_effect=pending_get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=client):
        connect_task = asyncio.create_task(session.connect())
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        connect_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(connect_task, timeout=5.0)

    assert session._cdp_client_root is None
    assert session._session_manager is None
    assert session.agent_focus is None
    assert session._cdp_session_pool == {}
    assert session.tab_snapshot() == []
    assert session.cdp_connection_health().last_error_code is CdpConnectionErrorCode.CANCELLED
