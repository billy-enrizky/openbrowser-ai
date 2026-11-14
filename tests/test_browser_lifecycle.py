"""Tests for BrowserSession lifecycle and tab integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.browser.events import CloseTabEvent, SwitchTabEvent
from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.browser.tabs import TabError, TabState


def _make_session() -> BrowserSession:
	return BrowserSession(browser_profile=BrowserProfile(headless=True))


def _ready_awaitable():
	future = asyncio.get_running_loop().create_future()
	future.set_result(None)
	return future


@pytest.mark.asyncio
async def test_start_twice_runs_one_start():
	session = _make_session()

	async def fake_start():
		session._lifecycle.transition(LifecyclePhase.READY, reason='test start')

	with patch.object(session, '_start_locked', side_effect=fake_start) as start_mock:
		await session.start()
		await session.start()

	assert start_mock.await_count == 1
	assert session.lifecycle_snapshot()['phase'] == 'ready'


@pytest.mark.asyncio
async def test_concurrent_start_waits_for_same_start():
	session = _make_session()
	release = asyncio.Event()
	calls = 0

	async def fake_start():
		nonlocal calls
		calls += 1
		await release.wait()
		session._lifecycle.transition(LifecyclePhase.READY, reason='test start')

	with patch.object(session, '_start_locked', side_effect=fake_start):
		tasks = [asyncio.create_task(session.start()) for _ in range(3)]
		await asyncio.sleep(0.05)
		assert calls == 1
		release.set()
		await asyncio.gather(*tasks)

	assert calls == 1
	assert session.lifecycle_snapshot()['phase'] == 'ready'


@pytest.mark.asyncio
async def test_start_timeout_cancels_the_start_task():
	session = _make_session()

	async def hanging_start():
		try:
			await asyncio.sleep(60)
		except BaseException as error:
			session._lifecycle.fail(error, reason='cancelled by timeout')
			raise

	with patch.object(session, '_start_locked', side_effect=hanging_start):
		with pytest.raises(asyncio.TimeoutError):
			await asyncio.wait_for(session.start(), timeout=0.05)

	assert session.lifecycle_snapshot()['phase'] == 'failed'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['stop', 'kill'])
async def test_stop_or_kill_during_start_cancels_the_owned_start(operation):
	session = _make_session()
	started = asyncio.Event()
	cancelled = asyncio.Event()

	async def hanging_start():
		started.set()
		try:
			await asyncio.sleep(60)
		except asyncio.CancelledError:
			cancelled.set()
			raise

	with patch.object(session, '_start_locked', side_effect=hanging_start):
		start_task = asyncio.create_task(session.start())
		await asyncio.wait_for(started.wait(), timeout=1)

		await getattr(session, operation)()
		with pytest.raises(asyncio.CancelledError):
			await start_task

	assert cancelled.is_set()
	assert session.lifecycle_snapshot()['phase'] == 'stopped'
	assert session._start_task is None


@pytest.mark.asyncio
async def test_kill_twice_runs_one_shutdown():
	session = _make_session()
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.READY)

	async def fake_shutdown(*, force):
		session._lifecycle.transition(LifecyclePhase.STOPPED, reason='test stop', increment_generation=True)

	with patch.object(session, '_shutdown_locked', side_effect=fake_shutdown) as shutdown_mock:
		await session.kill()
		await session.kill()

	assert shutdown_mock.await_count == 1
	assert session.lifecycle_snapshot()['phase'] == 'stopped'


@pytest.mark.asyncio
async def test_tab_registry_tracks_created_and_closed_events():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'

	await session.on_TabCreatedEvent(type('Event', (), {'target_id': target, 'url': 'https://example.com'})())
	record = session._tab_registry.get(target)
	assert record is not None
	assert record.state is TabState.OPENING

	session._tab_registry.update(target, state=TabState.OPEN)
	await session.on_TabClosedEvent(type('Event', (), {'target_id': target})())
	record = session._tab_registry.get(target)
	assert record is not None
	assert record.state is TabState.CLOSED


@pytest.mark.asyncio
async def test_ambiguous_tab_id_returns_candidates():
	session = _make_session()
	first = 'AAAABBBBCCCCDDDD1111222233337890'
	second = 'EEEEFFFF000011112222333344447890'
	session._tab_registry.register(first, state=TabState.OPEN)
	session._tab_registry.register(second, state=TabState.OPEN)

	with pytest.raises(TabError) as error:
		await session.get_target_id_from_tab_id('7890')

	assert set(error.value.candidates) == {first, second}


@pytest.mark.asyncio
async def test_full_target_id_resolves_without_truncation():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)

	assert await session.get_target_id_from_tab_id(target) == target


@pytest.mark.asyncio
async def test_chrome_error_target_is_not_marked_closed_by_tab_display():
	session = _make_session()
	session._cdp_client_root = MagicMock()
	target = {
		'type': 'page',
		'targetId': 'AAAABBBBCCCCDDDD1111222233334444',
		'url': 'chrome-error://chromewebdata/',
		'title': 'Error',
	}

	async def get_pages(**kwargs):
		return [target] if kwargs else []

	with patch.object(session, '_cdp_get_all_pages', side_effect=get_pages) as get_pages_mock:
		tabs = await session.get_tabs()

	assert get_pages_mock.await_count == 2
	assert any(
		call.kwargs
		== {
			'include_chrome': True,
			'include_chrome_extensions': True,
			'include_chrome_error': True,
		}
		for call in get_pages_mock.await_args_list
	)
	assert tabs == []
	record = session._tab_registry.get(target['targetId'])
	assert record is not None
	assert record.state is TabState.OPEN


@pytest.mark.asyncio
async def test_close_tab_reaches_a_terminal_state_when_the_target_disappears():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	cdp_session = MagicMock()
	cdp_session.cdp_client.send.Target.closeTarget = AsyncMock(side_effect=Exception('already closed'))
	dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())
	session.event_bus.dispatch = dispatch

	with patch.object(BrowserSession, 'get_or_create_cdp_session', new_callable=AsyncMock, return_value=cdp_session):
		await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	closed_event = dispatch.call_args.args[0]
	await session.on_TabClosedEvent(closed_event)
	record = session._tab_registry.get(target)
	assert record is not None
	assert record.state is TabState.CLOSED


@pytest.mark.asyncio
async def test_close_tab_ignores_an_already_closed_record():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.CLOSED)
	close_session = AsyncMock()
	dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())
	session.event_bus.dispatch = dispatch

	with patch.object(BrowserSession, 'get_or_create_cdp_session', close_session):
		await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	close_session.assert_not_awaited()
	dispatch.assert_not_called()
	assert session._tab_registry.get(target).state is TabState.CLOSED


@pytest.mark.asyncio
async def test_close_of_an_already_closed_focused_tab_repairs_focus():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.CLOSED)
	session.__dict__['agent_focus'] = MagicMock(target_id=target)
	dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())
	session.event_bus.dispatch = dispatch

	await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	dispatched = [call.args[0] for call in dispatch.call_args_list]
	assert any(isinstance(event, SwitchTabEvent) for event in dispatched)


@pytest.mark.asyncio
async def test_terminal_event_with_short_id_repairs_focus():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session.__dict__['agent_focus'] = MagicMock(target_id=target)
	dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())
	session.event_bus.dispatch = dispatch

	await session.on_TabClosedEvent(type('Event', (), {'target_id': '4444'})())

	dispatched = [call.args[0] for call in dispatch.call_args_list]
	assert any(isinstance(event, SwitchTabEvent) for event in dispatched)


@pytest.mark.asyncio
async def test_cancelled_close_restores_an_retryable_tab_state():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	session.event_bus.dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())

	async def cancelled_close(*args, **kwargs):
		raise asyncio.CancelledError

	with patch.object(BrowserSession, 'get_or_create_cdp_session', side_effect=cancelled_close):
		with pytest.raises(asyncio.CancelledError):
			await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	assert session._tab_registry.get(target).state is TabState.OPEN


@pytest.mark.asyncio
async def test_terminal_event_failure_still_leaves_the_tab_closed():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	session.event_bus.dispatch = MagicMock(side_effect=RuntimeError('event bus unavailable'))
	cdp_session = MagicMock()
	cdp_session.cdp_client.send.Target.closeTarget = AsyncMock()

	with patch.object(BrowserSession, 'get_or_create_cdp_session', new_callable=AsyncMock, return_value=cdp_session):
		await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	assert session._tab_registry.get(target).state is TabState.CLOSED


@pytest.mark.asyncio
async def test_kill_after_keep_alive_stop_forces_retained_browser_cleanup():
	session = _make_session()
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.STOPPING)
	session._lifecycle.transition(LifecyclePhase.STOPPED)
	watchdog = MagicMock()
	watchdog._subprocess = MagicMock()
	watchdog._spawned_subprocess = None
	watchdog._profile_lease = MagicMock()
	session._local_browser_watchdog = watchdog

	async def fake_shutdown(*, force):
		assert force is True
		session._lifecycle.transition(LifecyclePhase.STOPPING)
		session._lifecycle.transition(LifecyclePhase.STOPPED, increment_generation=True)

	with patch.object(session, '_shutdown_locked', side_effect=fake_shutdown) as shutdown:
		await session.kill()

	shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_stop_retained_browser_calls_the_watchdog_handler():
	session = _make_session()
	watchdog = MagicMock()
	watchdog._subprocess = MagicMock()
	watchdog._spawned_subprocess = None
	watchdog._profile_lease = None
	watchdog.on_BrowserKillEvent = AsyncMock()
	session._local_browser_watchdog = watchdog

	error = await session._force_stop_retained_browser()

	assert error is None
	watchdog.on_BrowserKillEvent.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_keeps_a_watchdog_that_still_owns_a_process():
	session = _make_session()
	watchdog = MagicMock()
	watchdog._subprocess = MagicMock()
	watchdog._spawned_subprocess = None
	watchdog._profile_lease = None
	session._local_browser_watchdog = watchdog

	await session.reset()

	assert session._local_browser_watchdog is watchdog


@pytest.mark.asyncio
async def test_reset_restores_core_handlers_on_a_fresh_bus():
	session = _make_session()
	session._watchdogs_attached = True

	await session.reset()

	assert session._core_handlers_attached is True
	assert session.event_bus.handlers.get('BrowserStartEvent')
	assert session.event_bus.handlers.get('BrowserStopEvent')


@pytest.mark.asyncio
async def test_bounded_cleanup_force_releases_a_stuck_task():
	session = _make_session()

	async def hang():
		await asyncio.sleep(60)

	task = asyncio.create_task(hang())
	with (
		patch.object(session, '_force_stop_resources') as force_stop,
		patch.object(session, '_force_reset_runtime_state') as force_reset,
	):
		with pytest.raises(TimeoutError, match='Browser shutdown timed out'):
			await session._await_task_bounded(task, 0.05)

	force_stop.assert_called_once()
	force_reset.assert_called_once()
