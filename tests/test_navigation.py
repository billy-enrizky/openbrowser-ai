import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser.navigation import (
    NavigationBusyError, NavigationError, NavigationMonitor, NavigationOperation, NavigationPhase,
)


def operation(wait_until='load'):
    op = NavigationOperation('request', 3, 'https://example.com/', wait_until)
    op.select_target('target', 'session')
    op.capture_document({'id': 'frame', 'loaderId': 'old', 'url': 'about:blank'})
    op.transition(NavigationPhase.NAVIGATING)
    return op


def test_illegal_transition_and_terminal_immutability():
    op = operation()
    with pytest.raises(NavigationError):
        op.transition(NavigationPhase.SUCCEEDED)
    op.finish(NavigationPhase.FAILED, 'failure')
    op.finish(NavigationPhase.SUCCEEDED)
    assert op.phase == NavigationPhase.FAILED
    assert op.last_error == 'failure'


@pytest.mark.parametrize('wait_until,milestone', [('commit', 'init'), ('domcontentloaded', 'DOMContentLoaded'), ('load', 'load'), ('networkidle', 'networkIdle')])
async def test_early_matching_milestone_is_replayed(wait_until, milestone):
    op = operation(wait_until)
    op.observe('lifecycle', {'frameId': 'frame', 'loaderId': 'next', 'name': milestone})
    op.accept_response({'frameId': 'frame', 'loaderId': 'next'})
    await asyncio.wait_for(op.wait_ready(), .1)
    assert op.phase == NavigationPhase.WAITING


async def test_wrong_frame_and_loader_do_not_complete():
    op = operation()
    op.accept_response({'frameId': 'frame', 'loaderId': 'next'})
    op.observe('lifecycle', {'frameId': 'child', 'loaderId': 'next', 'name': 'load'})
    op.observe('lifecycle', {'frameId': 'frame', 'loaderId': 'old', 'name': 'load'})
    task = asyncio.create_task(op.wait_ready())
    await asyncio.sleep(0)
    assert not task.done()
    op.observe('lifecycle', {'frameId': 'frame', 'loaderId': 'next', 'name': 'load'})
    await asyncio.wait_for(task, .1)


async def test_same_document_uses_current_loader_and_acknowledgment():
    op = operation()
    op.observe('lifecycle', {'frameId': 'frame', 'loaderId': 'old', 'name': 'load'})
    op.observe('within', {'frameId': 'frame', 'url': 'https://example.com/#new'})
    op.accept_response({'frameId': 'frame'})
    await asyncio.wait_for(op.wait_ready(), .1)
    assert op.observed_url == 'https://example.com/#new'


def test_same_document_load_waits_for_the_existing_document_load():
    op = operation()
    op.observe('lifecycle', {'frameId': 'frame', 'loaderId': 'old', 'name': 'DOMContentLoaded'})
    op.accept_response({'frameId': 'frame'})
    assert not op.ready
    op.observe('lifecycle', {'frameId': 'frame', 'loaderId': 'old', 'name': 'load'})
    assert op.ready


def test_explicit_document_loader_is_not_a_same_document_acknowledgment():
    op = operation('commit')
    op.accept_response({'frameId': 'frame', 'loaderId': 'old'})
    assert not op.ready


def test_download_overrides_error_text_but_protocol_error_fails():
    op = operation()
    op.accept_response({'frameId': 'frame', 'isDownload': True, 'errorText': 'net::ERR_ABORTED'})
    assert op.is_download
    assert op.phase == NavigationPhase.NOTIFYING
    with pytest.raises(NavigationError, match='ERR_NAME_NOT_RESOLVED'):
        operation().accept_response({'frameId': 'frame', 'errorText': 'net::ERR_NAME_NOT_RESOLVED'})


@pytest.mark.parametrize('response', [
    {'frameId': 12}, {'frameId': 'frame', 'loaderId': {}}, {'frameId': 'frame', 'loaderId': ''},
    {'frameId': 'frame', 'errorText': 3}, {'frameId': 'frame', 'isDownload': 'true'},
])
def test_malformed_command_response_fails_at_boundary(response):
    with pytest.raises(NavigationError, match='Page.navigate'):
        operation().accept_response(response)


def test_evidence_overflow_fails_instead_of_dropping_readiness():
    op = operation()
    for index in range(op.EVIDENCE_LIMIT + 1):
        op.observe('lifecycle', {'frameId': 'frame', 'loaderId': str(index), 'name': 'load'})
    assert op.phase == NavigationPhase.FAILED
    assert 'evidence' in op.last_error


def test_irrelevant_loader_evidence_is_ignored_after_response_identity_is_known():
    op = operation()
    op.accept_response({'frameId': 'frame', 'loaderId': 'next'})
    for _ in range(op.EVIDENCE_LIMIT + 1):
        op.observe('lifecycle', {'frameId': 'frame', 'loaderId': 'old', 'name': 'load'})
    assert op.phase == NavigationPhase.WAITING
    assert not op.ready


def test_monitor_ignores_wrong_session_old_root_and_terminal_callbacks():
    monitor = NavigationMonitor()
    root = SimpleNamespace(register=SimpleNamespace(Page=MagicMock(), Runtime=MagicMock()))
    monitor.bind(root)
    callback = root.register.Page.lifecycleEvent.call_args.args[0]
    monitor.bind(root)
    assert root.register.Page.lifecycleEvent.call_count == 1
    op = operation()
    monitor.reserve(op)
    callback({'frameId': 'frame', 'loaderId': 'next', 'name': 'load'}, 'other')
    op.accept_response({'frameId': 'frame', 'loaderId': 'next'})
    assert not op.ready
    monitor.invalidate()
    callback({'frameId': 'frame', 'loaderId': 'next', 'name': 'load'}, 'session')
    assert not op.ready


def test_monitor_refuses_overlap():
    monitor = NavigationMonitor()
    monitor.reserve(operation())
    with pytest.raises(NavigationBusyError):
        monitor.reserve(operation())


def test_forced_synchronous_reset_can_abort_a_reservation_without_a_running_loop():
    from openbrowser.browser import BrowserSession
    session = BrowserSession()
    op = NavigationOperation('sync-reset', session.lifecycle.generation, 'about:blank', 'load')
    session._navigation_monitor.reserve(op)
    session._force_reset_runtime_state(replace_event_bus=False)
    assert op.phase == NavigationPhase.CANCELLED
    assert session._navigation_monitor.active is None


def handler_fixture():
    from test_session_part1 import _make_browser_session, _setup_session_with_cdp
    from openbrowser.browser.session import BrowserSession
    session = _make_browser_session()
    client, focus, _ = _setup_session_with_cdp(session)
    client.register = SimpleNamespace(Page=MagicMock(), Runtime=MagicMock())
    frame = {'id': 'frame', 'loaderId': 'old', 'url': 'about:blank'}
    client.send.Page.getFrameTree = AsyncMock(side_effect=lambda **kw: {'frameTree': {'frame': dict(frame)}})
    client.send.Page.setLifecycleEventsEnabled = AsyncMock()
    async def navigate(**kwargs):
        frame.update(loaderId='next', url=kwargs['params']['url'])
        callback = client.register.Page.lifecycleEvent.call_args.args[0]
        callback({'frameId': 'frame', 'loaderId': 'next', 'name': 'load'}, focus.session_id)
        return {'frameId': 'frame', 'loaderId': 'next'}
    client.send.Page.navigate = AsyncMock(side_effect=navigate)
    patches = (
        patch.object(BrowserSession, '_cdp_get_all_pages', AsyncMock(return_value=[])),
        patch.object(BrowserSession, '_close_extension_options_pages', AsyncMock()),
    )
    return session, client, focus, frame, patches


async def test_handler_rejects_protocol_error_without_success_observation():
    from openbrowser.browser.events import NavigateToUrlEvent, NavigationCompleteEvent, NavigationFinishedEvent
    session, client, _, _, patches = handler_fixture()
    client.send.Page.navigate = AsyncMock(return_value={'frameId': 'frame', 'errorText': 'net::ERR_NAME_NOT_RESOLVED'})
    with patches[0], patches[1], pytest.raises(NavigationError, match='ERR_NAME_NOT_RESOLVED'):
        await session.on_NavigateToUrlEvent(NavigateToUrlEvent(url='https://example.com/'))
    emitted = session.event_bus.dispatch._dispatched_events
    assert not any(isinstance(e, NavigationCompleteEvent) and not e.error_message for e in emitted)
    assert [e.outcome for e in emitted if isinstance(e, NavigationFinishedEvent)] == ['failed']


async def test_handler_requires_focus_instead_of_false_success():
    from test_session_part1 import _make_browser_session
    from openbrowser.browser.events import NavigateToUrlEvent
    session = _make_browser_session()
    with pytest.raises(NavigationError, match='not connected'):
        await session.on_NavigateToUrlEvent(NavigateToUrlEvent(url='https://example.com/'))


async def test_handler_waits_for_actual_load_and_publishes_final_url():
    from openbrowser.browser.events import NavigateToUrlEvent, NavigationCompleteEvent
    session, client, focus, frame, patches = handler_fixture()
    sent = asyncio.Event()
    async def navigate(**kwargs):
        frame.update(loaderId='next', url='https://redirect.example/', urlFragment='#done')
        sent.set()
        return {'frameId': 'frame', 'loaderId': 'next'}
    client.send.Page.navigate.side_effect = navigate
    with patches[0], patches[1]:
        task = asyncio.create_task(session.on_NavigateToUrlEvent(NavigateToUrlEvent(url='https://example.com/')))
        await sent.wait()
        await asyncio.sleep(0)
        assert not task.done()
        callback = client.register.Page.lifecycleEvent.call_args.args[0]
        callback({'frameId': 'frame', 'loaderId': 'next', 'name': 'load'}, focus.session_id)
        await asyncio.wait_for(task, .2)
    emitted = session.event_bus.dispatch._dispatched_events
    complete = [e for e in emitted if isinstance(e, NavigationCompleteEvent)]
    assert complete[0].url == 'https://redirect.example/#done'


async def test_handler_timeout_is_terminal_and_releases_admission():
    from openbrowser.browser.events import NavigateToUrlEvent
    session, client, _, _, patches = handler_fixture()
    client.send.Page.navigate = AsyncMock(return_value={'frameId': 'frame', 'loaderId': 'next'})
    with patches[0], patches[1], pytest.raises(TimeoutError):
        await session.on_NavigateToUrlEvent(NavigateToUrlEvent(url='https://example.com/', timeout_ms=10))
    assert session.lifecycle_snapshot()['navigation']['phase'] == 'timed_out'
    assert not session._navigation_lock.locked()


async def test_handler_rejects_mutated_session_binding():
    from openbrowser.browser.events import NavigateToUrlEvent
    session, client, focus, _, patches = handler_fixture()
    original = client.send.Page.navigate.side_effect
    async def navigate(**kwargs):
        response = await original(**kwargs)
        focus.session_id = 'replacement'
        return response
    client.send.Page.navigate.side_effect = navigate
    with patches[0], patches[1], pytest.raises(NavigationError, match='binding'):
        await session.on_NavigateToUrlEvent(NavigateToUrlEvent(url='https://example.com/'))


async def test_public_overlap_and_cancellation_own_the_handler():
    from openbrowser.browser.session import BrowserSession
    from bubus import EventBus
    session, client, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered = asyncio.Event()
    async def navigate(**kwargs):
        entered.set()
        await asyncio.Event().wait()
    client.send.Page.navigate.side_effect = navigate
    with patches[0], patches[1]:
        task = asyncio.create_task(session.navigate_to('https://example.com/'))
        await entered.wait()
        with pytest.raises(NavigationBusyError):
            await session.navigate_to('https://other.example/')
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        owner = session._navigation_task
        if owner:
            await asyncio.wait({owner}, timeout=.2)
        assert session.lifecycle_snapshot()['navigation']['phase'] == 'cancelled'
    await session.event_bus.stop(clear=True, timeout=.2)


@pytest.mark.parametrize('child_kind', ['switch', 'focus'])
async def test_real_bus_stale_navigation_child_cannot_mutate_or_repair(child_kind):
    from bubus import EventBus
    from openbrowser.browser.events import AgentFocusChangedEvent, SwitchTabEvent
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _make_cdp_session
    session, client, original, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    op = NavigationOperation('stale', session.lifecycle.generation, 'https://example.com/', 'load')
    session._navigation_monitor.reserve(op)
    entered, release = asyncio.Event(), asyncio.Event()
    old = _make_cdp_session(client, target_id='old-target', session_id='old-session')
    replacement = _make_cdp_session(client, target_id='replacement-target', session_id='replacement-session')
    if child_kind == 'focus':
        session.agent_focus = old
        op.select_target(old.target_id, old.session_id)
        event = AgentFocusChangedEvent(target_id=old.target_id, url=op.requested_url,
            navigation_id=op.navigation_id, navigation_generation=op.generation, navigation_session_id=old.session_id)
    else:
        event = SwitchTabEvent(target_id=old.target_id, navigation_id=op.navigation_id, navigation_generation=op.generation)
    async def acquire(*args, **kwargs):
        entered.set()
        await release.wait()
        return old
    acquire_mock = AsyncMock(side_effect=acquire)
    with patches[0], patches[1], patch.object(BrowserSession, 'get_or_create_cdp_session', acquire_mock):
        dispatched = session.event_bus.dispatch(event)
        await asyncio.wait_for(entered.wait(), .5)
        op.finish(NavigationPhase.CANCELLED)
        session._force_reset_runtime_state(replace_event_bus=False)
        session.seed_generation(op.generation + 1)
        session.agent_focus = replacement
        session._cdp_client_root = client
        release.set()
        await asyncio.wait_for(dispatched, .5)
        assert session.agent_focus is replacement
        assert session.tab_snapshot() == []
        assert acquire_mock.await_count == 1
        assert acquire_mock.call_args.kwargs['focus'] is False
    await session.event_bus.stop(clear=True, timeout=.2)


async def test_shutdown_cancels_navigation_before_dispatching_storage_work():
    session, _, _, _, _ = handler_fixture()
    op = operation()
    session._navigation_monitor.reserve(op)
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def owner():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    task = asyncio.create_task(owner())
    await started.wait()
    session._navigation_task = task
    session._navigation_task_operation = op
    def dispatch(event):
        assert cancelled.is_set(), 'Shutdown dispatched work before cancelling navigation'
        from test_session_part1 import _DispatchResult
        return _DispatchResult()
    session.event_bus.dispatch = MagicMock(side_effect=dispatch)
    try:
        await session._shutdown_locked(force=False)
    finally:
        task.cancel()
        await asyncio.wait({task}, timeout=.1)


async def test_reset_retires_a_public_reservation_without_an_owner_task():
    session, _, _, _, _ = handler_fixture()
    op = operation()
    session._navigation_monitor.reserve(op)
    await session.reset(stop_event_bus=False)
    assert op.phase == NavigationPhase.CANCELLED
    assert session._navigation_monitor.active is None


async def test_slow_terminal_observer_does_not_delay_public_navigation():
    from bubus import EventBus
    from openbrowser.browser.events import NavigationFinishedEvent
    session, _, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def terminal_observer(event: NavigationFinishedEvent):
        entered.set()
        try:
            await release.wait()
        finally:
            finished.set()
    session.event_bus.on(NavigationFinishedEvent, terminal_observer)
    try:
        with patches[0], patches[1]:
            navigation = asyncio.create_task(session.navigate_to('https://example.com/'))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert not finished.is_set()
                await asyncio.wait_for(navigation, 5)
                assert not finished.is_set()
                assert session.lifecycle_snapshot()['navigation']['phase'] == 'succeeded'
            finally:
                release.set()
                if not navigation.done():
                    navigation.cancel()
                await asyncio.gather(navigation, return_exceptions=True)
    finally:
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_completion_observer_failure_is_failed_without_focus_repair_or_second_completion():
    from bubus import EventBus
    from openbrowser.browser.events import AgentFocusChangedEvent, NavigationCompleteEvent, NavigationFinishedEvent
    from openbrowser.browser.session import BrowserSession
    session, _, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    observed = []
    async def failing_observer(event: NavigationCompleteEvent):
        observed.append(event)
        raise ValueError('observer failed')
    terminal = []
    async def terminal_observer(event: NavigationFinishedEvent):
        terminal.append(event)
    session.event_bus.on(NavigationCompleteEvent, failing_observer)
    session.event_bus.on(NavigationFinishedEvent, terminal_observer)
    acquire = AsyncMock(side_effect=AssertionError('focus repair must not run'))
    try:
        with patches[0], patches[1], patch.object(BrowserSession, 'get_or_create_cdp_session', acquire):
            with pytest.raises(Exception, match='observer failed'):
                await session.navigate_to('https://example.com/')
            await session.event_bus.wait_until_idle()
            assert len(observed) == 1
            assert len(terminal) == 1
            assert terminal[0].outcome == 'failed'
            acquire.assert_not_awaited()
    finally:
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_public_cancel_drains_matching_owner_before_returning_when_cooperative():
    from bubus import EventBus
    session, client, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def navigate(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise
    client.send.Page.navigate.side_effect = navigate
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://example.com/'))
            await entered.wait()
            task.cancel()
            await asyncio.wait_for(cancelling.wait(), .5)
            await asyncio.sleep(0)
            assert not task.done(), 'Caller returned while cooperative owner was still cleaning up'
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert session._navigation_task is None
    finally:
        release.set()
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_real_public_cancellation_keeps_bus_alive_for_next_navigation():
    from bubus import EventBus
    session, client, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered = asyncio.Event()
    original = client.send.Page.navigate.side_effect
    async def blocked(**kwargs):
        entered.set()
        await asyncio.Event().wait()
    client.send.Page.navigate.side_effect = blocked
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://first.example/'))
            await entered.wait()
            owner = session._navigation_task
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert owner.done() and not owner.cancelled(), 'Owned cancellation must not escape into Bubus'
            client.send.Page.navigate.side_effect = original
            await asyncio.wait_for(session.navigate_to('https://next.example/'), .5)
            assert session.lifecycle_snapshot()['navigation']['phase'] == 'succeeded'
    finally:
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_reset_cancels_running_tab_observer_before_it_can_resume_stale_work():
    from bubus import EventBus
    from openbrowser.browser.events import TabCreatedEvent
    from openbrowser.browser.watchdog_base import BaseWatchdog
    session, _, _, _, _ = handler_fixture()
    session.event_bus = EventBus()
    op = NavigationOperation('tab-operation', session.lifecycle.generation, 'https://example.com/', 'load')
    session._navigation_monitor.reserve(op)
    entered, release = asyncio.Event(), asyncio.Event()
    mutations = []
    async def on_TabCreatedEvent(event: TabCreatedEvent):
        entered.set()
        await release.wait()
        mutations.append('stale')
    BaseWatchdog.attach_handler_to_session(session, TabCreatedEvent, on_TabCreatedEvent)
    event = TabCreatedEvent(target_id='old-target', url='about:blank', navigation_id=op.navigation_id,
        navigation_generation=op.generation)
    try:
        dispatched = session.event_bus.dispatch(event)
        await asyncio.wait_for(entered.wait(), .5)
        await session.reset(stop_event_bus=False)
        release.set()
        await asyncio.wait_for(dispatched, .5)
        assert mutations == []
    finally:
        release.set()
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_navigation_deadline_cancels_owned_observer_work():
    from bubus import EventBus
    from openbrowser.browser.events import NavigationStartedEvent
    from openbrowser.browser.watchdog_base import BaseWatchdog
    from openbrowser.browser.navigation import NavigationTimeoutError
    session, _, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, cleaned = asyncio.Event(), asyncio.Event()
    async def on_NavigationStartedEvent(event: NavigationStartedEvent):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    BaseWatchdog.attach_handler_to_session(session, NavigationStartedEvent, on_NavigationStartedEvent)
    task = None
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://example.com/', timeout_ms=1000))
            await asyncio.wait_for(entered.wait(), .5)
            with pytest.raises(NavigationTimeoutError):
                await asyncio.wait_for(task, 3)
        assert entered.is_set()
        assert cleaned.is_set(), 'Deadline left its navigation subscriber running'
        assert session._navigation_monitor.active is None
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_new_tab_observer_failure_is_not_creation_fallback():
    from openbrowser.browser.events import NavigateToUrlEvent, TabCreatedEvent
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _DispatchResult
    session, client, _, _, patches = handler_fixture()
    def dispatch(event):
        result = _DispatchResult()
        if isinstance(event, TabCreatedEvent):
            result.event_result = AsyncMock(side_effect=ValueError('tab observer failed'))
        return result
    session.event_bus.dispatch = MagicMock(side_effect=dispatch)
    with (patches[0], patches[1],
          patch.object(BrowserSession, 'get_current_page_url', AsyncMock(return_value='https://current.example/')),
          patch.object(BrowserSession, '_cdp_create_new_page', AsyncMock(return_value='created-target'))):
        with pytest.raises(NavigationError, match='tab observer failed'):
            await session.on_NavigateToUrlEvent(NavigateToUrlEvent(url='https://example.com/', new_tab=True))
    client.send.Page.navigate.assert_not_awaited()


async def test_queued_public_cancellation_never_sends_its_command():
    from bubus import BaseEvent, EventBus
    class Blocker(BaseEvent):
        pass
    session, client, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, release = asyncio.Event(), asyncio.Event()
    async def blocker(event: Blocker):
        entered.set()
        await release.wait()
    session.event_bus.on(Blocker, blocker)
    pending = session.event_bus.dispatch(Blocker())
    try:
        await entered.wait()
        task = asyncio.create_task(session.navigate_to('https://cancelled.example/'))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await pending
        with patches[0], patches[1]:
            await asyncio.wait_for(session.navigate_to('https://next.example/'), .5)
        assert [call.kwargs['params']['url'] for call in client.send.Page.navigate.await_args_list] == ['https://next.example/']
    finally:
        release.set()
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_public_deadline_includes_event_bus_queue_time():
    from bubus import BaseEvent, EventBus
    from openbrowser.browser.navigation import NavigationTimeoutError
    class DeadlineBlocker(BaseEvent):
        pass
    session, client, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, release = asyncio.Event(), asyncio.Event()
    async def blocker(event: DeadlineBlocker):
        entered.set()
        await release.wait()
    session.event_bus.on(DeadlineBlocker, blocker)
    pending = session.event_bus.dispatch(DeadlineBlocker())
    task = None
    try:
        await entered.wait()
        task = asyncio.create_task(session.navigate_to('https://expired.example/', timeout_ms=20))
        done, _ = await asyncio.wait({task}, timeout=.2)
        assert task in done, 'Public navigation deadline did not include event bus queue time'
        with pytest.raises(NavigationTimeoutError):
            await task
        assert session.lifecycle_snapshot()['navigation']['phase'] == 'timed_out'
        client.send.Page.navigate.assert_not_awaited()
        release.set()
        await pending
        with patches[0], patches[1]:
            await asyncio.wait_for(session.navigate_to('https://next.example/'), .5)
        assert [call.kwargs['params']['url'] for call in client.send.Page.navigate.await_args_list] == ['https://next.example/']
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        release.set()
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_transport_timeout_before_request_deadline_is_failure_not_deadline_expiry():
    from openbrowser.browser.events import NavigateToUrlEvent
    session, client, _, _, patches = handler_fixture()
    client.send.Page.navigate = AsyncMock(side_effect=TimeoutError('transport timeout'))
    with patches[0], patches[1], pytest.raises(NavigationError, match='transport timeout'):
        await session.on_NavigateToUrlEvent(NavigateToUrlEvent(url='https://example.com/', timeout_ms=30000))
    assert session.lifecycle_snapshot()['navigation']['phase'] == 'failed'


async def test_expired_adopted_deadline_does_not_start_target_work():
    from openbrowser.browser.events import NavigateToUrlEvent
    session, _, _, _, patches = handler_fixture()
    event = NavigateToUrlEvent(url='https://example.com/')
    op = event.get_navigation_operation(session.lifecycle.generation)
    op.deadline = asyncio.get_running_loop().time() - 1
    with patches[0] as pages, patches[1], pytest.raises(TimeoutError):
        await session.on_NavigateToUrlEvent(event)
    pages.assert_not_awaited()


async def test_navigation_cancel_does_not_cancel_unrelated_switch_pumped_by_bubus():
    from bubus import EventBus
    from openbrowser.browser.events import SwitchTabEvent
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _make_cdp_session
    session, client, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    preparing, continue_nav, switching, switch_cancelled, release_switch = [asyncio.Event() for _ in range(5)]
    target = _make_cdp_session(client, target_id='unrelated-target', session_id='unrelated-session')
    async def pages(**kwargs):
        preparing.set()
        await continue_nav.wait()
        return []
    async def acquire(*args, **kwargs):
        switching.set()
        try:
            await release_switch.wait()
        except asyncio.CancelledError:
            switch_cancelled.set()
            raise
        return target
    task = None
    try:
        with (patch.object(BrowserSession, '_cdp_get_all_pages', AsyncMock(side_effect=pages)), patches[1],
              patch.object(BrowserSession, 'get_or_create_cdp_session', AsyncMock(side_effect=acquire))):
            task = asyncio.create_task(session.navigate_to('https://first.example/'))
            await preparing.wait()
            independent = session.event_bus.dispatch(SwitchTabEvent(target_id=target.target_id))
            assert independent.event_parent_id is None
            continue_nav.set()
            await asyncio.wait_for(switching.wait(), .5)
            assert not session._navigation_monitor.active.child_tasks
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not switch_cancelled.is_set()
            release_switch.set()
            await asyncio.wait_for(independent, .5)
            await independent.event_result(raise_if_any=True, raise_if_none=False)
            assert session.agent_focus is target
    finally:
        release_switch.set()
        continue_nav.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_public_waiter_in_handler_context_does_not_cancel_unrelated_pumped_switch():
    from bubus import BaseEvent, EventBus
    from openbrowser.browser.events import SwitchTabEvent
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _make_cdp_session
    class PublicCaller(BaseEvent):
        pass
    session, client, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, continue_caller, switching, cancelled, release_switch = [asyncio.Event() for _ in range(5)]
    target = _make_cdp_session(client, target_id='independent', session_id='independent-session')
    waiters = []
    async def caller(event: PublicCaller):
        entered.set()
        await continue_caller.wait()
        task = asyncio.create_task(session.navigate_to('https://cancelled.example/'))
        waiters.append(task)
        try:
            await task
        except asyncio.CancelledError:
            pass
    async def acquire(*args, **kwargs):
        switching.set()
        try:
            await release_switch.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return target
    session.event_bus.on(PublicCaller, caller)
    try:
        with patches[0], patches[1], patch.object(BrowserSession, 'get_or_create_cdp_session', AsyncMock(side_effect=acquire)):
            pending = session.event_bus.dispatch(PublicCaller())
            await entered.wait()
            independent = session.event_bus.dispatch(SwitchTabEvent(target_id=target.target_id))
            continue_caller.set()
            await asyncio.wait_for(switching.wait(), .5)
            waiters[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiters[0]
            assert not cancelled.is_set()
            release_switch.set()
            await asyncio.wait_for(independent, .5)
            await independent.event_result(raise_if_any=True, raise_if_none=False)
            await asyncio.wait_for(pending, .5)
    finally:
        release_switch.set()
        continue_caller.set()
        for waiter in waiters:
            if not waiter.done():
                waiter.cancel()
        await session.event_bus.stop(clear=True, timeout=.2)


async def test_raw_navigation_deadline_cancels_owned_observer_without_cancelling_bus_pump():
    from bubus import EventBus
    from openbrowser.browser.events import NavigateToUrlEvent, NavigationStartedEvent
    from openbrowser.browser.watchdog_base import BaseWatchdog
    session, _, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def on_NavigationStartedEvent(event: NavigationStartedEvent):
        entered.set()
        try:
            await release.wait()
        finally:
            cleaned.set()
    BaseWatchdog.attach_handler_to_session(session, NavigationStartedEvent, on_NavigationStartedEvent)
    waiter = None
    try:
        with patches[0], patches[1]:
            event = session.event_bus.dispatch(NavigateToUrlEvent(url='https://example.com/', timeout_ms=1000))
            waiter = asyncio.ensure_future(event)
            await asyncio.wait_for(entered.wait(), 2)
            done, _ = await asyncio.wait({waiter}, timeout=2)
            assert done, 'Handler deadline did not cancel its owned observer work'
            await asyncio.wait_for(cleaned.wait(), 1)
            assert session._navigation_monitor.active is None
    finally:
        release.set()
        if waiter is not None and not waiter.done():
            waiter.cancel()
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)
        await session.event_bus.stop(clear=True, timeout=.2)


@pytest.mark.parametrize('forced', [False, True])
async def test_retiring_bus_cancels_its_retained_navigation_event_pumps(forced):
    from openbrowser.browser.events import NavigationStartedEvent
    session, _, _, _, _ = handler_fixture()
    entered, release = asyncio.Event(), asyncio.Event()
    class Pending:
        def __await__(self):
            async def wait():
                entered.set()
                await release.wait()
            return wait().__await__()
        async def event_result(self, **kwargs):
            pass
    old_bus = session.event_bus
    old_bus.dispatch = MagicMock(return_value=Pending())
    old_bus.stop = AsyncMock()
    owner = asyncio.create_task(session._await_navigation_event(old_bus,
        NavigationStartedEvent(target_id='target', url='about:blank')))
    try:
        await entered.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        pumps = set(session._navigation_event_pumps)
        assert len(pumps) == 1
        if forced:
            with pytest.raises(NavigationError, match='retir'):
                session._force_reset_runtime_state()
            assert session.event_bus is old_bus
        else:
            session._watchdogs_attached = True
            await session.reset(stop_event_bus=True)
        await asyncio.wait(pumps, timeout=.1)
        assert all(task.done() for task in pumps)
        assert not session._navigation_event_pumps
        if forced:
            session._force_reset_runtime_state()
            assert session.event_bus is not old_bus
    finally:
        release.set()
        await asyncio.gather(*session._navigation_event_pumps, return_exceptions=True)


async def test_second_cancellation_during_child_drain_cannot_strand_navigation_lock():
    from bubus import EventBus
    from openbrowser.browser.events import NavigationStartedEvent
    from openbrowser.browser.watchdog_base import BaseWatchdog
    session, _, _, _, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def on_NavigationStartedEvent(event: NavigationStartedEvent):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
    BaseWatchdog.attach_handler_to_session(session, NavigationStartedEvent, on_NavigationStartedEvent)
    task = None
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://example.com/'))
            await entered.wait()
            owner = session._navigation_task
            owner.cancel()
            await asyncio.wait_for(cleaning.wait(), .5)
            owner.cancel()
            await asyncio.wait({owner}, timeout=.2)
            assert not session._navigation_lock.locked(), 'Second cancellation skipped lock release'
            assert session._navigation_task is None
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            assert session._navigation_monitor.active is None
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await session.event_bus.stop(clear=True, timeout=.2)


@pytest.mark.parametrize('stop_failure', [asyncio.CancelledError, RuntimeError])
async def test_interrupted_bus_stop_retires_pumps_and_preserves_failed_runtime(stop_failure):
    from bubus import EventBus
    from openbrowser.browser.events import SwitchTabEvent
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _make_cdp_session
    session, client, focus, _, patches = handler_fixture()
    old_bus = EventBus()
    session.event_bus = old_bus
    session._core_handlers_attached = False
    session._attach_core_handlers()
    session._watchdogs_attached = True
    preparing, continue_nav, switching, release_switch, stopping, switch_cancelled = [asyncio.Event() for _ in range(6)]
    target = _make_cdp_session(client, target_id='independent-target', session_id='independent-session')
    async def pages(**kwargs):
        preparing.set()
        await continue_nav.wait()
        return []
    async def acquire(*args, **kwargs):
        switching.set()
        try:
            await release_switch.wait()
        except asyncio.CancelledError:
            switch_cancelled.set()
            raise
        return target
    original_stop = old_bus.stop
    async def observed_stop(**kwargs):
        stopping.set()
        if stop_failure is RuntimeError:
            raise RuntimeError('bus stop failed')
        await original_stop(**kwargs)
    old_bus.stop = observed_stop
    nav = reset = None
    try:
        with (patch.object(BrowserSession, '_cdp_get_all_pages', AsyncMock(side_effect=pages)), patches[1],
              patch.object(BrowserSession, 'get_or_create_cdp_session', AsyncMock(side_effect=acquire))):
            nav = asyncio.create_task(session.navigate_to('https://first.example/'))
            await preparing.wait()
            old_bus.dispatch(SwitchTabEvent(target_id=target.target_id))
            continue_nav.set()
            await asyncio.wait_for(switching.wait(), .5)
            nav.cancel()
            await asyncio.gather(nav, return_exceptions=True)
            reset = asyncio.create_task(session.reset(stop_event_bus=True))
            await stopping.wait()
            if stop_failure is asyncio.CancelledError:
                reset.cancel()
            result, = await asyncio.gather(reset, return_exceptions=True)
            assert isinstance(result, stop_failure), 'Reset must propagate unsuccessful bus retirement'
            assert session.event_bus is old_bus
            assert session._cdp_client_root is client
            assert session.lifecycle_snapshot()['phase'] == 'failed'
            assert not any(bus is old_bus and not task.done() for task, bus in session._navigation_event_pumps.items())
            assert switch_cancelled.is_set()
            release_switch.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert session.agent_focus is focus
            assert session._tab_registry.get(target.target_id) is None
    finally:
        release_switch.set()
        continue_nav.set()
        for task in (nav, reset):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await original_stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(old_bus)
        if session.event_bus is not old_bus:
            await session.event_bus.stop(clear=True, timeout=.1)


async def test_reset_preserves_runtime_when_pump_retirement_exceeds_bound():
    session, client, focus, _, _ = handler_fixture()
    old_bus = session.event_bus
    old_bus.stop = AsyncMock()
    session._watchdogs_attached = True
    entered, release = asyncio.Event(), asyncio.Event()
    async def stubborn_pump():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
    pump = asyncio.create_task(stubborn_pump())
    session._navigation_event_pumps[pump] = old_bus
    pump.add_done_callback(session._navigation_event_pump_done)
    try:
        await entered.wait()
        with pytest.raises(TimeoutError, match='pump'):
            await session.reset(stop_event_bus=True)
        assert session.event_bus is old_bus
        assert session._cdp_client_root is client
        assert session.agent_focus is focus
        assert session.lifecycle_snapshot()['phase'] == 'failed'
        assert not pump.done()
    finally:
        release.set()
        await asyncio.gather(pump, return_exceptions=True)


async def test_failed_shutdown_preserves_runtime_when_bus_and_pump_cannot_retire():
    from bubus import EventBus
    from openbrowser.browser.events import BrowserStopEvent, SwitchTabEvent
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _DispatchResult, _make_cdp_session
    session, client, focus, _, patches = handler_fixture()
    old_bus = EventBus()
    session.event_bus = old_bus
    session._core_handlers_attached = False
    session._attach_core_handlers()
    session._watchdogs_attached = True
    preparing, proceed, switching, release = [asyncio.Event() for _ in range(4)]
    target = _make_cdp_session(client, target_id='independent-old', session_id='independent-old-session')
    async def pages(**kwargs):
        preparing.set()
        await proceed.wait()
        return []
    async def acquire(*args, **kwargs):
        switching.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return target
    original_stop, original_dispatch = old_bus.stop, old_bus.dispatch
    def dispatch(event):
        result = _DispatchResult()
        if isinstance(event, BrowserStopEvent):
            result.event_result = AsyncMock(side_effect=RuntimeError('watchdog cleanup failed'))
        return result
    nav = None
    try:
        with (patch.object(BrowserSession, '_cdp_get_all_pages', AsyncMock(side_effect=pages)), patches[1],
              patch.object(BrowserSession, 'get_or_create_cdp_session', AsyncMock(side_effect=acquire))):
            nav = asyncio.create_task(session.navigate_to('https://first.example/'))
            await preparing.wait()
            original_dispatch(SwitchTabEvent(target_id=target.target_id))
            proceed.set()
            await asyncio.wait_for(switching.wait(), .5)
            nav.cancel()
            await asyncio.gather(nav, return_exceptions=True)
            old_bus.dispatch = MagicMock(side_effect=dispatch)
            old_bus.stop = AsyncMock(side_effect=RuntimeError('bus stop failed'))
            result, = await asyncio.gather(session.stop(), return_exceptions=True)
            assert isinstance(result, (RuntimeError, TimeoutError))
            assert session.event_bus is old_bus, 'Failed shutdown abandoned its unretired bus'
            assert session._cdp_client_root is client
            assert session.agent_focus is focus
            assert session.lifecycle_snapshot()['phase'] == 'failed'
    finally:
        release.set()
        proceed.set()
        old_bus.dispatch, old_bus.stop = original_dispatch, original_stop
        if nav is not None and not nav.done():
            nav.cancel()
            await asyncio.gather(nav, return_exceptions=True)
        await original_stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(old_bus)
        if session.event_bus is not old_bus:
            await session.event_bus.stop(clear=True, timeout=.1)


@pytest.mark.parametrize('entry', ['synchronous', 'forced_timeout', 'cancelled_start', 'retained_bus_reset'])
async def test_exceptional_cleanup_retains_runtime_until_stubborn_pump_is_quiescent(entry):
    session, client, focus, _, _ = handler_fixture()
    old_bus = session.event_bus
    old_bus.stop = AsyncMock()
    client.stop = AsyncMock()
    entered, release = asyncio.Event(), asyncio.Event()
    async def stubborn_pump():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
    pump = asyncio.create_task(stubborn_pump())
    session._navigation_event_pumps[pump] = old_bus
    pump.add_done_callback(session._navigation_event_pump_done)
    owner = None
    try:
        await entered.wait()
        if entry == 'synchronous':
            with pytest.raises(NavigationError, match='retir'):
                session._force_reset_runtime_state(replace_event_bus=False)
        elif entry == 'forced_timeout':
            owner = asyncio.create_task(asyncio.Event().wait())
            with pytest.raises(TimeoutError):
                await session._await_task_bounded(owner, timeout=.001)
        elif entry == 'cancelled_start':
            with pytest.raises((NavigationError, TimeoutError)):
                await session._cleanup_cancelled_start()
        else:
            with pytest.raises(TimeoutError):
                await session.reset(stop_event_bus=False)
        assert session.event_bus is old_bus
        assert session._cdp_client_root is client
        assert session.agent_focus is focus
        assert session.lifecycle_snapshot()['phase'] == 'failed'
        assert not pump.done()
    finally:
        release.set()
        if owner is not None:
            owner.cancel()
        pump.cancel()
        await asyncio.gather(pump, *([owner] if owner is not None else []), return_exceptions=True)


async def test_forced_timeout_retires_genuinely_quiescent_real_bus_without_long_wait():
    from bubus import BaseEvent, EventBus
    session, _, _, _, _ = handler_fixture()
    old_bus = EventBus()
    session.event_bus = old_bus
    await old_bus.dispatch(BaseEvent())
    owner = asyncio.create_task(asyncio.Event().wait())
    try:
        with pytest.raises(TimeoutError, match='Browser shutdown timed out'):
            await asyncio.wait_for(session._await_task_bounded(owner, timeout=.005), timeout=.5)
        assert session.event_bus is not old_bus
        assert session._cdp_client_root is None
        assert session.lifecycle_snapshot()['phase'] == 'stopped'
    finally:
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        await old_bus.stop(clear=True, timeout=0)
        await session.event_bus.stop(clear=True, timeout=0)


@pytest.mark.parametrize('synchronous', [False, True])
async def test_retirement_cannot_treat_its_current_bus_pump_as_quiescent(synchronous):
    session, client, focus, _, _ = handler_fixture()
    old_bus = session.event_bus
    async def current_pump():
        task = asyncio.current_task()
        session._navigation_event_pumps[task] = old_bus
        try:
            if synchronous:
                with pytest.raises(NavigationError, match='retir'):
                    session._force_reset_runtime_state(replace_event_bus=False)
            else:
                with pytest.raises(TimeoutError, match='pump'):
                    await session.reset(stop_event_bus=False)
            assert session.event_bus is old_bus
            assert session._cdp_client_root is client
            assert session.agent_focus is focus
            assert session.lifecycle_snapshot()['phase'] == 'failed'
        finally:
            session._navigation_event_pumps.pop(task, None)
    await asyncio.create_task(current_pump())


@pytest.mark.parametrize('entry', ['start', 'navigation'])
async def test_failed_runtime_cannot_resume_while_retained_bus_work_is_live(entry):
    from openbrowser.browser.lifecycle import LifecyclePhase
    from openbrowser.browser.session import BrowserSession
    session, client, focus, _, _ = handler_fixture()
    old_bus = session.event_bus
    session._lifecycle.fail('event bus retirement incomplete')
    release = asyncio.Event()
    pump = asyncio.create_task(release.wait())
    session._navigation_event_pumps[pump] = old_bus
    pump.add_done_callback(session._navigation_event_pump_done)
    async def successful_start():
        session._lifecycle.transition(LifecyclePhase.READY)
    start = AsyncMock(side_effect=successful_start)
    try:
        with patch.object(BrowserSession, '_start_locked', start):
            with pytest.raises(NavigationError, match='retir'):
                if entry == 'start':
                    await session.start()
                else:
                    await session.navigate_to('https://replacement.example/')
            start.assert_not_awaited()
            assert session.lifecycle_snapshot()['phase'] == 'failed'
            assert session._cdp_client_root is client
            assert session.agent_focus is focus
            assert session.event_bus is old_bus
            release.set()
            await pump
            await session.start()
            start.assert_awaited_once()
            assert session.lifecycle_snapshot()['phase'] == 'ready'
    finally:
        release.set()
        await asyncio.gather(pump, return_exceptions=True)


async def test_completed_cleanup_timeout_is_not_misclassified_as_shutdown_deadline():
    from openbrowser.browser.session import BrowserSession
    session, client, focus, _, _ = handler_fixture()
    async def retirement_failure():
        raise TimeoutError('event bus retirement blocked')
    owner = asyncio.create_task(retirement_failure())
    with patch.object(BrowserSession, '_force_stop_resources', MagicMock()) as force:
        with pytest.raises(TimeoutError, match='event bus retirement blocked'):
            await session._await_task_bounded(owner, timeout=20)
        force.assert_not_called()
    assert session._cdp_client_root is client
    assert session.agent_focus is focus


async def test_start_rechecks_retirement_after_acquiring_lifecycle_lock():
    from openbrowser.browser.session import BrowserSession
    session, _, _, _, _ = handler_fixture()
    release = asyncio.Event()
    pump = asyncio.create_task(release.wait())
    task = None
    start = AsyncMock()
    try:
        with patch.object(BrowserSession, '_start_locked', start):
            async with session._lifecycle_lock:
                task = asyncio.create_task(session.start())
                await asyncio.sleep(0)
                session._lifecycle.fail('event bus retirement incomplete')
                session._navigation_event_pumps[pump] = session.event_bus
                pump.add_done_callback(session._navigation_event_pump_done)
            with pytest.raises(NavigationError, match='retir'):
                await task
            start.assert_not_awaited()
            assert session.lifecycle_snapshot()['phase'] == 'failed'
    finally:
        release.set()
        await asyncio.gather(pump, *([task] if task is not None else []), return_exceptions=True)


async def test_shutdown_reset_failure_cannot_bypass_synchronous_retirement_gate():
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _DispatchResult
    session, client, focus, _, _ = handler_fixture()
    old_bus = session.event_bus
    old_bus.dispatch = MagicMock(side_effect=lambda event: _DispatchResult())
    old_bus.stop = AsyncMock()
    release = asyncio.Event()
    pumps = []
    async def reset_failure(**kwargs):
        pump = asyncio.create_task(release.wait())
        pumps.append(pump)
        session._navigation_event_pumps[pump] = old_bus
        pump.add_done_callback(session._navigation_event_pump_done)
        raise RuntimeError('reset failed with event work still live')
    try:
        with patch.object(BrowserSession, 'reset', AsyncMock(side_effect=reset_failure)):
            with pytest.raises(RuntimeError):
                await session.stop()
        assert session.event_bus is old_bus
        assert session._cdp_client_root is client
        assert session.agent_focus is focus
        assert session.lifecycle_snapshot()['phase'] == 'failed'
    finally:
        release.set()
        await asyncio.gather(*pumps, return_exceptions=True)


async def test_direct_connect_refuses_failed_live_transport_before_mutation_then_accepts_quiescence():
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _make_cdp_session, _make_mock_cdp_client
    session, client, focus, _, _ = handler_fixture()
    old_url = session.browser_profile.cdp_url
    old_manager = session._session_manager
    session._lifecycle.fail('event bus retirement incomplete')
    release = asyncio.Event()
    pump = asyncio.create_task(release.wait())
    session._navigation_event_pumps[pump] = session.event_bus
    pump.add_done_callback(session._navigation_event_pump_done)
    replacement_root = _make_mock_cdp_client()
    replacement_root.start = AsyncMock()
    replacement_root.send.Target.setAutoAttach = AsyncMock()
    target = {'targetId': 'replacement', 'type': 'page', 'url': 'https://replacement.example/'}
    replacement_root.send.Target.getTargets = AsyncMock(return_value={'targetInfos': [target]})
    replacement = _make_cdp_session(replacement_root, target_id=target['targetId'], session_id='replacement-session')
    manager = MagicMock()
    manager.start_monitoring = AsyncMock()
    manager.get_session_for_target = AsyncMock(return_value=replacement)
    try:
        with (patch('openbrowser.browser.session.CDPClient', return_value=replacement_root) as factory,
              patch('openbrowser.browser.session_manager.SessionManager', return_value=manager) as manager_factory,
              patch.object(BrowserSession, '_setup_proxy_auth', AsyncMock())):
            with pytest.raises(NavigationError, match='retir'):
                await session.connect(cdp_url='ws://replacement.invalid/devtools/browser/fixture')
            factory.assert_not_called()
            manager_factory.assert_not_called()
            assert session.browser_profile.cdp_url == old_url
            assert session._cdp_client_root is client
            assert session.agent_focus is focus
            assert session._session_manager is old_manager
            release.set()
            await pump
            result = await session.connect(cdp_url='ws://replacement.invalid/devtools/browser/fixture')
            assert result is session
            factory.assert_called_once_with('ws://replacement.invalid/devtools/browser/fixture')
            assert session._cdp_client_root is replacement_root
            assert session.agent_focus is replacement
    finally:
        release.set()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.parametrize('failure_boundary', ['response', 'context_exit'])
async def test_connect_rechecks_shared_retirement_guard_after_http_discovery(failure_boundary):
    from openbrowser.browser.session import BrowserSession
    from test_session_part1 import _make_cdp_session, _make_mock_cdp_client
    session, client, focus, _, _ = handler_fixture()
    old_manager = session._session_manager
    release = asyncio.Event()
    pumps = []
    http_url = 'http://replacement.invalid:9222'
    ws_url = 'ws://replacement.invalid/devtools/browser/fixture'
    def fail_runtime():
        session._lifecycle.fail('event bus retirement incomplete')
        pump = asyncio.create_task(release.wait())
        pumps.append(pump)
        session._navigation_event_pumps[pump] = session.event_bus
        pump.add_done_callback(session._navigation_event_pump_done)
    response = MagicMock()
    response.json.return_value = {'webSocketDebuggerUrl': ws_url}
    async def get(*args, **kwargs):
        if failure_boundary == 'response':
            fail_runtime()
        return response
    async def exit_context(*args):
        if failure_boundary == 'context_exit':
            fail_runtime()
        return False
    http = AsyncMock()
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(side_effect=exit_context)
    http.get = AsyncMock(side_effect=get)
    replacement_root = _make_mock_cdp_client()
    replacement_root.start = AsyncMock()
    replacement_root.send.Target.setAutoAttach = AsyncMock()
    target = {'targetId': 'replacement', 'type': 'page', 'url': 'https://replacement.example/'}
    replacement_root.send.Target.getTargets = AsyncMock(return_value={'targetInfos': [target]})
    replacement = _make_cdp_session(replacement_root, target_id=target['targetId'], session_id='replacement-session')
    manager = MagicMock()
    manager.start_monitoring = AsyncMock()
    manager.get_session_for_target = AsyncMock(return_value=replacement)
    try:
        with (patch('httpx.AsyncClient', return_value=http),
              patch('openbrowser.browser.session.CDPClient', return_value=replacement_root) as factory,
              patch('openbrowser.browser.session_manager.SessionManager', return_value=manager) as manager_factory,
              patch.object(BrowserSession, '_setup_proxy_auth', AsyncMock())):
            with pytest.raises(NavigationError, match='retir'):
                await session.connect(cdp_url=http_url)
            factory.assert_not_called()
            manager_factory.assert_not_called()
            assert session.browser_profile.cdp_url == (http_url if failure_boundary == 'response' else ws_url)
            assert session._cdp_client_root is client
            assert session.agent_focus is focus
            assert session._session_manager is old_manager
    finally:
        release.set()
        await asyncio.gather(*pumps, return_exceptions=True)


def test_policy_commit_is_separate_from_ack_and_survives_primary_terminalization():
    from openbrowser.browser.navigation import NavigationCommit
    op = operation()
    op.arm_policy()
    op.observe('frame', {'frame': {'id': 'frame', 'loaderId': 'pre-ack', 'url': 'https://blocked.test/'}})
    assert op.policy_commit == NavigationCommit('frame', 'pre-ack', 'https://blocked.test/')
    assert op.frame_id is None
    assert op.loader_id is None
    op.finish(NavigationPhase.CANCELLED)
    assert op.policy_commit.url == 'https://blocked.test/'
    assert not op.ready


def test_done_but_unconfirmed_containment_still_requires_retirement():
    from openbrowser.browser.navigation import ContainmentAttempt, ContainmentPhase, NavigationCommit
    attempt = ContainmentAttempt(NavigationCommit('frame', 'loader', 'https://blocked.test/'))
    attempt.phase = ContainmentPhase.UNCONFIRMED
    assert attempt.requires_retirement
    attempt.phase = ContainmentPhase.REVOKED
    assert not attempt.requires_retirement


def policy_handler_fixture(pre_ack=False):
    from bubus import EventBus
    session, client, focus, frame, patches = handler_fixture()
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    session.browser_profile.allowed_domains = ['allowed.test']
    client.register.Runtime = MagicMock()
    controls = SimpleNamespace(committed=asyncio.Event(), ack=asyncio.Event(), action=asyncio.Event(),
        release_action=asyncio.Event(), hold_action=False, action_error=None, on_action=None, acquired_loader=None,
        hold_world=False, world_entered=asyncio.Event(), release_world=asyncio.Event(), no_context=False)
    async def navigate(**kwargs):
        url = kwargs['params']['url']
        frame.update(loaderId='next', url='https://blocked.test/forbidden' if url.endswith('/start') else url)
        client.register.Page.frameNavigated.call_args.args[0]({'frame': dict(frame)}, focus.session_id)
        controls.committed.set()
        if not url.endswith('/start'):
            client.register.Page.lifecycleEvent.call_args.args[0]({'frameId': 'frame', 'loaderId': 'next', 'name': 'load'}, focus.session_id)
        if pre_ack:
            await controls.ack.wait()
        return {'frameId': 'frame', 'loaderId': 'next'}
    async def world(**kwargs):
        params = kwargs['params']
        assert params['grantUniveralAccess'] is False
        if not controls.no_context:
            client.register.Runtime.executionContextCreated.call_args.args[0]({'context': {
                'id': 7, 'uniqueId': 'unique-policy-context', 'name': params['worldName'], 'auxData': {'frameId': 'frame'},
            }}, focus.session_id)
        controls.world_entered.set()
        if controls.hold_world:
            await controls.release_world.wait()
        return {'executionContextId': 7}
    async def evaluate(**kwargs):
        params = kwargs['params']
        if params.get('expression') == 'document':
            assert params.get('uniqueContextId') == 'unique-policy-context'
            assert 'contextId' not in params
            controls.acquired_loader = frame['loaderId']
            return {'result': {'type': 'object', 'subtype': 'node', 'className': 'HTMLDocument', 'objectId': 'policy-document'}}
        return {'result': {'value': 2}}
    async def action(**kwargs):
        params = kwargs['params']
        assert params['objectId'] == 'policy-document'
        assert 'executionContextId' not in params and 'uniqueContextId' not in params
        controls.action.set()
        if controls.hold_action:
            await controls.release_action.wait()
        if controls.on_action:
            controls.on_action()
        if controls.action_error:
            raise controls.action_error
        expected_url = params['arguments'][0]['value']
        if frame['loaderId'] != controls.acquired_loader or frame['url'] != expected_url:
            return {'result': {'value': {'initiated': False}}}
        frame.update(loaderId='contained', url='about:blank')
        return {'result': {'value': {'initiated': True}}}
    client.send.Page.navigate = AsyncMock(side_effect=navigate)
    client.send.Page.createIsolatedWorld = AsyncMock(side_effect=world)
    client.send.Runtime.evaluate = AsyncMock(side_effect=evaluate)
    client.send.Runtime.callFunctionOn = AsyncMock(side_effect=action)
    client.send.Runtime.releaseObject = AsyncMock()
    return session, client, focus, frame, patches, controls


@pytest.mark.parametrize('pre_ack', [False, True])
async def test_real_bus_policy_contains_stalled_commit_independently_of_ack_and_load(pre_ack):
    from openbrowser.browser.navigation import NavigationBlockedError
    session, client, _, frame, patches, controls = policy_handler_fixture(pre_ack)
    try:
        with patches[0], patches[1], pytest.raises(NavigationBlockedError):
            await asyncio.wait_for(session.navigate_to('https://allowed.test/start', timeout_ms=1000), 3)
        assert frame['url'] == 'about:blank'
        assert [call.kwargs['params']['url'] for call in client.send.Page.navigate.await_args_list] == ['https://allowed.test/start']
        state = session.lifecycle_snapshot()['navigation']
        assert state['containment']['phase'] == 'contained'
        if pre_ack:
            assert state['frame_id'] is None and state['loader_id'] is None
    finally:
        controls.ack.set()
        controls.release_action.set()
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


@pytest.mark.parametrize('closed_record', [False, True])
@pytest.mark.parametrize('unrelated_focus', [False, True])
async def test_verified_containment_invalidates_only_its_current_focus_cache(unrelated_focus, closed_record):
    from openbrowser.browser.navigation import NavigationBlockedError
    from openbrowser.browser.tabs import TabState
    from test_session_part1 import _make_cdp_session
    session, client, focus, _, patches, controls = policy_handler_fixture()
    cached_node = object()
    session._cached_selector_map[5] = cached_node
    session._cached_browser_state_summary = 'old-summary'
    dom = MagicMock()
    session._dom_watchdog = dom
    state = TabState.CLOSED if closed_record else TabState.OPEN
    session._tab_registry.register(focus.target_id, url='https://old.example/', state=state)
    other = _make_cdp_session(client, target_id='other', session_id='other-session')
    if unrelated_focus:
        controls.on_action = lambda: setattr(session, 'agent_focus', other)
    try:
        with patches[0], patches[1], pytest.raises(NavigationBlockedError):
            await session.navigate_to('https://allowed.test/start')
        record = session._tab_registry.get(focus.target_id)
        assert record.state == state
        assert record.url == ('https://old.example/' if closed_record else 'about:blank')
        assert focus.url == 'about:blank'
        if unrelated_focus:
            assert session.agent_focus is other
            assert session._cached_selector_map[5] is cached_node
            assert session._cached_browser_state_summary == 'old-summary'
            dom.clear_cache.assert_not_called()
        else:
            assert session.agent_focus is focus
            assert not session._cached_selector_map
            assert session._cached_browser_state_summary is None
            dom.clear_cache.assert_called_once()
    finally:
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


async def wait_primary_terminal(operation):
    while not operation.terminal:
        operation._changed.clear()
        await asyncio.wait_for(operation._changed.wait(), 2)


@pytest.mark.parametrize('ending', ['timeout', 'cancel', 'double_cancel'])
@pytest.mark.parametrize('pre_ack', [False, True])
async def test_policy_sent_job_survives_primary_timeout_and_repeated_caller_cancellation(ending, pre_ack):
    from openbrowser.browser.navigation import NavigationTimeoutError
    session, _, _, frame, patches, controls = policy_handler_fixture(pre_ack)
    controls.hold_action = True
    task = None
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://allowed.test/start', timeout_ms=150 if ending == 'timeout' else 5000))
            await asyncio.wait_for(controls.action.wait(), 1)
            op = session._navigation_monitor.active
            if ending != 'timeout':
                task.cancel()
            await wait_primary_terminal(op)
            if ending == 'double_cancel':
                task.cancel()
                result, = await asyncio.gather(task, return_exceptions=True)
                assert isinstance(result, asyncio.CancelledError)
                assert not op.policy_task.done() and not op.policy_task.cancelled()
                with pytest.raises(NavigationError):
                    await session.navigate_to('https://allowed.test/too-early')
            controls.release_action.set()
            result, = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(result, NavigationTimeoutError if ending == 'timeout' else asyncio.CancelledError)
            await asyncio.wait_for(asyncio.shield(op.policy_task), 2)
            if session._navigation_task is not None:
                await asyncio.wait({session._navigation_task}, timeout=1)
            assert op.phase.value == ('timed_out' if ending == 'timeout' else 'cancelled')
            assert op.containment.phase.value == 'contained'
            assert frame['url'] == 'about:blank'
            if pre_ack:
                assert op.frame_id is None and op.loader_id is None
            controls.ack.set()
            await asyncio.wait_for(session.navigate_to('https://allowed.test/next'), 1)
    finally:
        controls.release_action.set()
        controls.ack.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


@pytest.mark.parametrize('replacement', ['root', 'generation', 'session'])
async def test_policy_preparation_cannot_act_on_replacement_binding(replacement):
    from openbrowser.browser.navigation import NavigationBlockedError
    from test_session_part1 import _make_mock_cdp_client
    session, client, focus, frame, patches, controls = policy_handler_fixture()
    controls.hold_world = True
    task = None
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://allowed.test/start'))
            await asyncio.wait_for(controls.world_entered.wait(), 1)
            if replacement == 'root':
                session._cdp_client_root = _make_mock_cdp_client()
            elif replacement == 'generation':
                session.seed_generation(session.lifecycle.generation + 1)
            else:
                focus.session_id = 'replacement-session'
            controls.release_world.set()
            with pytest.raises(NavigationError):
                await task
            client.send.Runtime.callFunctionOn.assert_not_awaited()
            assert frame['url'] == 'https://blocked.test/forbidden'
    finally:
        controls.release_world.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


@pytest.mark.parametrize('successor', ['same_url_new_loader', 'changed_hash'])
async def test_browser_conditional_policy_action_skips_changed_document(successor):
    from openbrowser.browser.navigation import NavigationBlockedError
    session, client, _, frame, patches, controls = policy_handler_fixture()
    controls.on_action = (lambda: frame.update(loaderId='successor')) if successor == 'same_url_new_loader' else (
        lambda: frame.update(url=frame['url'] + '#operator'))
    try:
        with patches[0], patches[1], pytest.raises(NavigationBlockedError):
            await session.navigate_to('https://allowed.test/start')
        assert frame['url'] != 'about:blank'
        assert session.lifecycle_snapshot()['navigation']['containment']['phase'] == 'skipped'
        assert client.send.Runtime.callFunctionOn.await_count == 1
        assert client.send.Page.navigate.await_count == 1
    finally:
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


async def test_completed_unconfirmed_policy_authority_keeps_remote_runtime_unusable():
    session, client, focus, _, patches, controls = policy_handler_fixture()
    controls.action_error = ConnectionError('sent action response lost')
    old_bus = session.event_bus
    old_url = session.browser_profile.cdp_url
    op = None
    try:
        with patches[0], patches[1], pytest.raises(NavigationError):
            await session.navigate_to('https://allowed.test/start')
        op = session._navigation_monitor.active
        assert op.policy_task.done()
        assert op.containment.phase.value == 'unconfirmed'
        assert op.containment.requires_retirement
        assert session.lifecycle_snapshot()['phase'] == 'failed'
        with pytest.raises(NavigationError, match='Policy|policy') as refused:
            await session.connect(cdp_url='ws://successor.invalid')
        assert 'new BrowserSession' in str(refused.value)
        with pytest.raises(NavigationError):
            await session.start()
        with pytest.raises(NavigationError):
            await session.navigate_to('https://allowed.test/next')
        with pytest.raises(NavigationError):
            await session.reset(stop_event_bus=False)
        with pytest.raises(NavigationError):
            session._force_reset_runtime_state()
        assert session._cdp_client_root is client and session.agent_focus is focus
        assert session.event_bus is old_bus and session.browser_profile.cdp_url == old_url
    finally:
        session._navigation_policies.clear()
        if op is not None:
            session._navigation_monitor.complete(op)
        await old_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(old_bus)


async def test_reset_does_not_cancel_sent_policy_action_or_replace_before_retirement():
    session, client, focus, _, patches, controls = policy_handler_fixture()
    controls.hold_action = True
    task = None
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://allowed.test/start'))
            await asyncio.wait_for(controls.action.wait(), 1)
            op = session._navigation_monitor.active
            with pytest.raises(NavigationError, match='retir'):
                await session.reset(stop_event_bus=False)
            assert not op.policy_task.done() and not op.policy_task.cancelled()
            assert session._cdp_client_root is client and session.agent_focus is focus
            assert session.lifecycle_snapshot()['phase'] == 'failed'
            controls.release_action.set()
            await asyncio.wait_for(asyncio.shield(op.policy_task), 2)
            await asyncio.gather(task, return_exceptions=True)
            assert op.containment.phase.value == 'contained'
            await session.reset(stop_event_bus=False)
            assert session._cdp_client_root is None
    finally:
        controls.release_action.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


@pytest.mark.parametrize('endpoint', ['owned', 'remote_same_pid', 'other_local_port', 'unowned_same_port',
    'other_ipv4_address', 'missing_peer', 'unknown_family'])
async def test_owned_browser_exit_revokes_unconfirmed_policy_authority_and_allows_retry(tmp_path, endpoint):
    import sys
    import socket
    import psutil
    from openbrowser.browser.lifecycle import LifecyclePhase
    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.watchdogs.profile_lease import ProfileLease
    session, client, _, _, patches, controls = policy_handler_fixture()
    controls.action_error = ConnectionError('sent response lost')
    profile = tmp_path / 'owned-profile'
    profile.mkdir()
    instance = 'policy-owned-fixture'
    marker = f'--openbrowser-instance-id={instance}'
    ready, close, closed = (tmp_path / name for name in ('port', 'close-listener', 'listener-closed'))
    code = ('import socket,sys,time\nfrom pathlib import Path\n'
        's=socket.socket(); s.bind(("127.0.0.1",0)); s.listen(); s.settimeout(.01)\n'
        'Path(sys.argv[1]).write_text(str(s.getsockname()[1]))\n'
        'connections=[]; end=time.monotonic()+30\n'
        'while time.monotonic()<end:\n'
        ' if s is not None and Path(sys.argv[2]).exists():\n'
        '  s.close(); s=None; Path(sys.argv[3]).touch()\n'
        ' if s is not None:\n'
        '  try: connections.append(s.accept()[0])\n'
        '  except TimeoutError: pass\n'
        ' time.sleep(.01)\n')
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', code, str(ready), str(close), str(closed),
        f'--user-data-dir={profile}', marker)
    for _ in range(200):
        if ready.exists():
            break
        await asyncio.sleep(.01)
    port = int(ready.read_text())
    owned = psutil.Process(process.pid)
    _reader, writer = await asyncio.open_connection('127.0.0.1', port)
    actual_socket = writer.transport.get_extra_info('socket')
    for _ in range(200):
        if any(connection.status == psutil.CONN_ESTABLISHED for connection in owned.net_connections(kind='tcp')):
            break
        await asyncio.sleep(.01)
    client.ws = SimpleNamespace(transport=writer.transport)
    record = {'pid': process.pid, 'start_time': owned.create_time(), 'profile_dir': str(profile),
        'instance_id': instance, 'ownership_marker': marker, 'cdp_port': port}
    watchdog = SimpleNamespace(_subprocess=owned, _spawned_subprocess=None, _browser_record=record,
        _profile_lease=SimpleNamespace(profile_dir=profile, instance_id=instance), _instance_id=instance,
        _owns_browser_resources=True)
    async def kill_owned(event):
        assert ProfileLease.process_matches_browser(owned, record)
        owned.kill()
        await asyncio.wait_for(process.wait(), 2)
        watchdog._subprocess = watchdog._profile_lease = watchdog._browser_record = None
    watchdog.on_BrowserKillEvent = kill_owned
    session._local_browser_watchdog = watchdog
    session.browser_profile.is_local = True
    session.browser_profile.user_data_dir = profile
    client.url = {'owned': f'ws://127.0.0.1:{port}/devtools/browser/owned-fixture',
        'remote_same_pid': f'ws://remote.invalid:{port}/devtools/browser/remote-fixture',
        'other_local_port': f'ws://127.0.0.1:{port - 1}/devtools/browser/remote-fixture',
        'unowned_same_port': f'ws://127.0.0.1:{port}/devtools/browser/remote-fixture',
        'other_ipv4_address': f'ws://127.0.0.2:{port}/devtools/browser/remote-fixture',
        'missing_peer': f'ws://127.0.0.1:{port}/devtools/browser/owned-fixture',
        'unknown_family': f'ws://127.0.0.1:{port}/devtools/browser/owned-fixture'}[endpoint]
    if endpoint in {'remote_same_pid', 'other_local_port', 'other_ipv4_address', 'unknown_family'}:
        peer = {'remote_same_pid': ('203.0.113.5', port), 'other_local_port': ('127.0.0.1', port - 1),
            'other_ipv4_address': ('127.0.0.2', port), 'unknown_family': ('127.0.0.1', port)}[endpoint]
        peer_socket = SimpleNamespace(family=socket.AF_UNIX if endpoint == 'unknown_family' else socket.AF_INET,
            getpeername=lambda: peer, getsockname=actual_socket.getsockname)
        client.ws = SimpleNamespace(transport=SimpleNamespace(get_extra_info=lambda name: peer_socket if name == 'socket' else None))
    elif endpoint == 'missing_peer':
        client.ws = None
    if endpoint == 'unowned_same_port':
        close.touch()
        for _ in range(200):
            if closed.exists():
                break
            await asyncio.sleep(.01)
        assert closed.exists()
    client.send.SystemInfo.getProcessInfo = AsyncMock(return_value={'processInfo': [{'type': 'browser', 'id': process.pid}]})
    op = None
    try:
        with patches[0], patches[1], pytest.raises(NavigationError):
            await session.navigate_to('https://allowed.test/start')
        op = session._navigation_monitor.active
        assert op.containment.phase.value == 'unconfirmed'
        if endpoint != 'owned':
            assert op.containment.owned_browser is None
            with pytest.raises(NavigationError, match='retir'):
                await asyncio.wait_for(session.kill(), 5)
            assert op.containment.phase.value == 'unconfirmed'
            assert session._cdp_client_root is client
            assert session.lifecycle_snapshot()['phase'] == 'failed'
            return
        await asyncio.wait_for(session.kill(), 5)
        assert process.returncode is not None
        assert op.containment.phase.value == 'revoked'
        assert session.lifecycle_snapshot()['phase'] == 'stopped'
        async def ready():
            session._lifecycle.transition(LifecyclePhase.READY)
        with patch.object(BrowserSession, '_start_locked', AsyncMock(side_effect=ready)):
            await session.start()
        assert session.lifecycle_snapshot()['phase'] == 'ready'
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        session._navigation_policies.clear()
        if op is not None:
            session._navigation_monitor.complete(op)
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


async def test_ready_runtime_cannot_connect_over_live_sent_policy_authority():
    from openbrowser.browser.lifecycle import LifecyclePhase
    session, client, focus, _, patches, controls = policy_handler_fixture()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    controls.hold_action = True
    task = None
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://allowed.test/start'))
            await asyncio.wait_for(controls.action.wait(), 1)
            assert session.lifecycle_snapshot()['phase'] == 'ready'
            old_url = session.browser_profile.cdp_url
            with patch('openbrowser.browser.session.CDPClient') as factory:
                with pytest.raises(NavigationError, match='retir'):
                    await session.connect(cdp_url='ws://successor.invalid')
                factory.assert_not_called()
            assert session._cdp_client_root is client and session.agent_focus is focus
            assert session.browser_profile.cdp_url == old_url
            assert session.lifecycle_snapshot()['phase'] == 'ready'
            controls.release_action.set()
            await asyncio.gather(task, return_exceptions=True)
    finally:
        controls.release_action.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


async def test_destroyed_old_document_command_is_skipped_without_successor_fallback():
    session, client, _, frame, patches, controls = policy_handler_fixture()
    controls.on_action = lambda: frame.update(loaderId='same-url-successor')
    controls.action_error = RuntimeError({'code': -32000, 'message': 'Cannot find context with specified id'})
    op = None
    try:
        with patches[0], patches[1], pytest.raises(NavigationError):
            await session.navigate_to('https://allowed.test/start')
        state = session.lifecycle_snapshot()['navigation']
        assert state['containment']['phase'] == 'skipped'
        assert frame['loaderId'] == 'same-url-successor'
        assert frame['url'] == 'https://blocked.test/forbidden'
        assert client.send.Runtime.callFunctionOn.await_count == 1
        assert client.send.Page.navigate.await_count == 1
    finally:
        op = session._navigation_monitor.active
        session._navigation_policies.clear()
        if op is not None:
            session._navigation_monitor.complete(op)
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


async def test_missing_unique_context_fails_before_action_without_false_containment():
    session, client, _, frame, patches, controls = policy_handler_fixture()
    controls.no_context = True
    try:
        with patches[0], patches[1], pytest.raises(NavigationError):
            await session.navigate_to('https://allowed.test/start', timeout_ms=5000)
        assert session.lifecycle_snapshot()['navigation']['containment']['phase'] == 'failed'
        client.send.Runtime.callFunctionOn.assert_not_awaited()
        assert frame['url'] == 'https://blocked.test/forbidden'
    finally:
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


async def test_policy_consumer_error_is_failure_not_disguised_readiness_timeout():
    session, _, _, _, patches, _ = policy_handler_fixture()
    def allowed(url):
        if url == 'https://allowed.test/start':
            return True
        raise RuntimeError('policy evaluator failed')
    session._security_watchdog = SimpleNamespace(_is_url_allowed=allowed)
    try:
        with patches[0], patches[1], pytest.raises(NavigationError, match='policy evaluator failed'):
            await session.navigate_to('https://allowed.test/start', timeout_ms=300)
        assert session.lifecycle_snapshot()['navigation']['phase'] == 'failed'
    finally:
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


async def test_expired_containment_grace_is_not_restarted_by_request_finalization():
    from openbrowser.browser.navigation import ContainmentAttempt, ContainmentPhase, NavigationCommit
    session, _, _, _, _ = handler_fixture()
    op = operation()
    op.containment = ContainmentAttempt(NavigationCommit('frame', 'next', 'https://blocked.test/'),
        phase=ContainmentPhase.ACTION_IN_FLIGHT, deadline=asyncio.get_running_loop().time() - 1)
    release = asyncio.Event()
    op.policy_task = asyncio.create_task(release.wait())
    try:
        with pytest.raises(NavigationError, match='retir'):
            await asyncio.wait_for(session._join_navigation_policy(op), timeout=.2)
        assert not op.policy_task.done() and not op.policy_task.cancelled()
        assert op.containment.phase is ContainmentPhase.UNCONFIRMED
    finally:
        release.set()
        await asyncio.gather(op.policy_task, return_exceptions=True)


async def test_real_wire_ipv4_policy_authority_cannot_be_revoked_by_same_port_ipv6_process_exit(tmp_path):
    import json
    import socket
    import sys
    import psutil
    import websockets
    from cdp_use import CDPClient
    from openbrowser.browser import BrowserSession
    from openbrowser.browser.session import CDPSession
    if not socket.has_ipv6:
        pytest.skip('IPv6 loopback is required for the distinct endpoint fixture')
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            probe.bind(('::1', 0))
    except OSError:
        pytest.skip('IPv6 loopback is unavailable for the distinct endpoint fixture')
    profile = tmp_path / 'profile'
    profile.mkdir()
    ready = tmp_path / 'ipv6-port'
    instance = 'policy-distinct-endpoints'
    marker = f'--openbrowser-instance-id={instance}'
    code = ('import socket,time,sys\nfrom pathlib import Path\n'
        's=socket.socket(socket.AF_INET6); s.setsockopt(socket.IPPROTO_IPV6,socket.IPV6_V6ONLY,1)\n'
        's.bind(("::1",0)); s.listen(); Path(sys.argv[1]).write_text(str(s.getsockname()[1]))\n'
        'time.sleep(30)\n')
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', code, str(ready), f'--user-data-dir={profile}', marker)
    session = root = operation = None
    actions = set()
    release, seen, executed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    frame = {'id': 'frame', 'loaderId': 'old', 'url': 'about:blank'}
    try:
        for _ in range(200):
            if ready.exists():
                break
            await asyncio.sleep(.01)
        port = int(ready.read_text())
        owned = psutil.Process(process.pid)
        record = {'pid': process.pid, 'start_time': owned.create_time(), 'profile_dir': str(profile),
            'instance_id': instance, 'ownership_marker': marker, 'cdp_port': port}
        async def delayed_action(ws, request):
            seen.set()
            await release.wait()
            frame.update(url='about:blank', loaderId='late-blank')
            executed.set()
            try:
                await ws.send(json.dumps({'id': request['id'], 'result': {'result': {'value': {'initiated': True}}}}))
            except websockets.exceptions.ConnectionClosed:
                pass
        async def endpoint(ws):
            async for message in ws:
                request = json.loads(message)
                method, sid = request['method'], request.get('sessionId')
                async def event(name, params):
                    await ws.send(json.dumps({'method': name, 'params': params, 'sessionId': sid}))
                result = {}
                if method == 'SystemInfo.getProcessInfo':
                    result = {'processInfo': [{'type': 'browser', 'id': process.pid}]}
                elif method == 'Page.getFrameTree':
                    result = {'frameTree': {'frame': dict(frame)}}
                elif method == 'Page.navigate':
                    frame.update(url='https://blocked.test/forbidden', loaderId='blocked')
                    await event('Page.frameNavigated', {'frame': dict(frame)})
                    result = {'frameId': 'frame', 'loaderId': 'blocked'}
                elif method == 'Page.createIsolatedWorld':
                    await event('Runtime.executionContextCreated', {'context': {'id': 7, 'uniqueId': 'wire-context',
                        'name': request['params']['worldName'], 'auxData': {'frameId': 'frame'}}})
                    result = {'executionContextId': 7}
                elif method == 'Runtime.evaluate':
                    result = {'result': {'type': 'object', 'subtype': 'node', 'className': 'HTMLDocument', 'objectId': 'wire-doc'}}
                elif method == 'Runtime.callFunctionOn':
                    action = asyncio.create_task(delayed_action(ws, request))
                    actions.add(action)
                    action.add_done_callback(actions.discard)
                    continue
                await ws.send(json.dumps({'id': request['id'], 'result': result}))
        async with websockets.serve(endpoint, '127.0.0.1', port):
            root = CDPClient(f'ws://127.0.0.1:{port}/devtools/browser/not-the-ipv6-owner')
            await root.start()
            try:
                session = BrowserSession(is_local=True, user_data_dir=str(profile), allowed_domains=['allowed.test'],
                    enable_default_extensions=False, keep_alive=False)
                session._cdp_client_root = root
                session.agent_focus = CDPSession.model_construct(cdp_client=root, target_id='target', session_id='s', url='about:blank', title='')
                watchdog = SimpleNamespace(_subprocess=owned, _spawned_subprocess=None, _browser_record=record,
                    _profile_lease=SimpleNamespace(profile_dir=profile, instance_id=instance), _instance_id=instance,
                    _owns_browser_resources=True)
                async def kill_owned(event):
                    process.kill()
                    await process.wait()
                    watchdog._subprocess = watchdog._profile_lease = watchdog._browser_record = None
                watchdog.on_BrowserKillEvent = kill_owned
                session._local_browser_watchdog = watchdog
                with (patch.object(BrowserSession, '_cdp_get_all_pages', AsyncMock(return_value=[])),
                      patch.object(BrowserSession, '_close_extension_options_pages', AsyncMock())):
                    with pytest.raises(NavigationError):
                        await session.navigate_to('https://allowed.test/start', timeout_ms=150)
                await asyncio.wait_for(seen.wait(), 2)
                operation = session._navigation_monitor.active
                outcome, = await asyncio.gather(session.kill(), return_exceptions=True)
                assert isinstance(outcome, NavigationError), 'Different address-family process death falsely retired the live endpoint'
                assert process.returncode is not None
                assert operation.containment.owned_browser is None
                assert operation.containment.phase.value == 'unconfirmed'
                assert session._cdp_client_root is root
                assert session.lifecycle_snapshot()['phase'] == 'failed'
                release.set()
                await asyncio.wait_for(executed.wait(), 2)
                await asyncio.wait_for(asyncio.shield(operation.policy_task), 2)
                assert operation.containment.phase.value == 'contained'
                assert session._cdp_client_root is root
            finally:
                release.set()
                if actions:
                    await asyncio.gather(*actions, return_exceptions=True)
                await root.stop()
    finally:
        release.set()
        if session is not None:
            if operation is not None and operation.policy_task is not None:
                await asyncio.wait({operation.policy_task}, timeout=1)
            session._navigation_policies.clear()
            await session.event_bus.stop(clear=True, timeout=.1)
            await session._drain_navigation_event_pumps(session.event_bus)
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.parametrize('peer_family,peer_ip,listen_family,listen_ip,accepted,expected', [
    ('v4', '127.0.0.1', 'v4', '127.0.0.1', 'exact', True),
    ('v6', '::1', 'v6', '::1', 'exact', True),
    ('v4', '127.0.0.1', 'v6', '::1', 'exact', False),
    ('v4', '127.0.0.1', 'v4', '127.0.0.2', 'exact', False),
    ('v4', '127.0.0.1', 'v4', '0.0.0.0', 'exact', True),
    ('v4', '127.0.0.1', 'v4', '0.0.0.0', 'missing', False),
    ('v6', '::1', 'v6', '::', 'exact', True),
    ('v4', '127.0.0.1', 'v6', '::', 'exact', False),
    ('v4', '127.0.0.1', 'v4', '127.0.0.1', 'other_client', False),
    ('v4', '127.0.0.1', 'v4', '127.0.0.1', 'peer_changed', False),
    ('v4', '127.0.0.1', 'v4', '127.0.0.1', 'permission_denied', False),
])
async def test_policy_peer_binding_requires_matching_family_address_and_connection(tmp_path, peer_family, peer_ip, listen_family, listen_ip, accepted, expected):
    import socket
    import psutil
    from openbrowser.browser.navigation import ContainmentAttempt, NavigationCommit
    from openbrowser.browser.watchdogs.profile_lease import ProfileLease
    session, client, _, _, _ = handler_fixture()
    family = socket.AF_INET if peer_family == 'v4' else socket.AF_INET6
    listener_family = socket.AF_INET if listen_family == 'v4' else socket.AF_INET6
    peer = [(peer_ip, 7777)]
    local = ('127.0.0.1' if peer_family == 'v4' else '::1', 5555)
    connection_socket = SimpleNamespace(family=family, getpeername=lambda: peer[0], getsockname=lambda: local)
    client.ws = SimpleNamespace(transport=SimpleNamespace(get_extra_info=lambda name: connection_socket if name == 'socket' else None))
    record = {'pid': 1234, 'start_time': 10.0, 'profile_dir': str(tmp_path.resolve()),
        'instance_id': 'fixture', 'ownership_marker': '--openbrowser-instance-id=fixture', 'cdp_port': 7777}
    sockets = [SimpleNamespace(family=listener_family, status=psutil.CONN_LISTEN, laddr=(listen_ip, 7777), raddr=())]
    if accepted != 'missing':
        sockets.append(SimpleNamespace(family=family, status=psutil.CONN_ESTABLISHED,
            laddr=peer[0], raddr=(local[0], 5556) if accepted == 'other_client' else local))
    process = SimpleNamespace(net_connections=MagicMock(return_value=sockets))
    if accepted == 'permission_denied':
        process.net_connections.side_effect = psutil.AccessDenied(1234)
    session._local_browser_watchdog = SimpleNamespace(_owns_browser_resources=True, _browser_record=record,
        _profile_lease=SimpleNamespace(profile_dir=tmp_path.resolve(), instance_id='fixture'), _instance_id='fixture',
        _subprocess=process)
    session.browser_profile.is_local = True
    session.browser_profile.user_data_dir = tmp_path
    async def info():
        if accepted == 'peer_changed':
            peer[0] = ('127.0.0.2', 7777)
        return {'processInfo': [{'type': 'browser', 'id': 1234}]}
    client.send.SystemInfo.getProcessInfo = AsyncMock(side_effect=info)
    op = operation()
    op.containment = ContainmentAttempt(NavigationCommit('frame', 'loader', 'https://blocked.test/'),
        deadline=asyncio.get_running_loop().time() + 1)
    with patch.object(ProfileLease, 'process_matches_browser', return_value=True):
        identity = await session._owned_policy_browser(op, client)
    assert (identity is not None) == expected


async def test_stale_tab_viewport_acquisition_does_not_command_or_register():
    from openbrowser.browser.events import TabCreatedEvent
    from openbrowser.browser.profile import ViewportSize
    from openbrowser.browser.session import BrowserSession
    session, client, focus, _, _ = handler_fixture()
    session.browser_profile.viewport = ViewportSize(width=800, height=600)
    session.browser_profile.no_viewport = False
    client.send.Emulation.setDeviceMetricsOverride = AsyncMock()
    op = NavigationOperation('viewport', session.lifecycle.generation, 'https://example.com/', 'load')
    session._navigation_monitor.reserve(op)
    entered, release = asyncio.Event(), asyncio.Event()
    async def acquire(**kwargs):
        entered.set()
        await release.wait()
        return focus
    with patch.object(BrowserSession, 'get_or_create_cdp_session', AsyncMock(side_effect=acquire)):
        task = asyncio.create_task(session.on_TabCreatedEvent(TabCreatedEvent(target_id=focus.target_id,
            url='about:blank', navigation_id=op.navigation_id, navigation_generation=op.generation)))
        await entered.wait()
        session._force_reset_runtime_state(replace_event_bus=False)
        session.seed_generation(op.generation + 1)
        release.set()
        with pytest.raises(NavigationError):
            await task
    client.send.Emulation.setDeviceMetricsOverride.assert_not_awaited()
    assert session.tab_snapshot() == []
