"""Tests for BrowserSession lifecycle and tab integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bubus import EventBus

from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.browser.events import (
	BrowserStartEvent,
	BrowserStopEvent,
	CloseTabEvent,
	TabClosedEvent,
	TabCreatedEvent,
)
from openbrowser.browser.focus import FocusTargetError
from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.browser.tabs import TabError, TabState


def _make_session() -> BrowserSession:
	return BrowserSession(browser_profile=BrowserProfile(headless=True))


def _ready_awaitable():
	future = asyncio.get_running_loop().create_future()
	future.set_result(None)
	return future


def _dispatch_result():
	class DispatchResult:
		def __await__(self):
			yield from asyncio.sleep(0).__await__()
			return self

		async def event_result(self, **_kwargs):
			return None

	return DispatchResult()


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
async def test_start_timeout_moves_the_real_lifecycle_to_failed():
	session = _make_session()

	async def hanging_start_event():
		await asyncio.sleep(60)

	session.event_bus.dispatch = MagicMock(return_value=hanging_start_event())

	with pytest.raises(asyncio.TimeoutError):
		await asyncio.wait_for(session.start(), timeout=0.05)

	assert session.lifecycle_snapshot()['phase'] == 'failed'
	assert session._start_task is None


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
		with pytest.raises(RuntimeError, match='cancelled by shutdown'):
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

	await session.on_TabCreatedEvent(TabCreatedEvent(target_id=target, url='https://example.com'))
	record = session._tab_registry.get(target)
	assert record is not None
	assert record.state is TabState.OPENING

	session._tab_registry.update(target, state=TabState.OPEN)
	await session.on_TabClosedEvent(TabClosedEvent(target_id=target))
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
	session._cdp_client_root = MagicMock()
	session._cdp_client_root.send.Target.closeTarget = AsyncMock(side_effect=Exception('already closed'))
	dispatch = MagicMock(side_effect=lambda event: _dispatch_result())
	session.event_bus.dispatch = dispatch

	await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	record = session._tab_registry.get(target)
	assert record is not None
	assert record.state is TabState.CLOSED
	closed_event = dispatch.call_args.args[0]
	assert isinstance(closed_event, TabClosedEvent)
	assert closed_event.target_id == target


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

	with patch.object(session, '_repair_focus_after_closed_tab', new_callable=AsyncMock) as repair:
		await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	repair.assert_awaited_once()
	assert repair.await_args.args[0] == target


@pytest.mark.asyncio
async def test_terminal_event_with_short_id_runs_focus_repair():
	session = _make_session()

	with patch.object(session, '_repair_focus_after_closed_tab', new_callable=AsyncMock) as repair:
		await session.on_TabClosedEvent(TabClosedEvent(target_id='4444'))

	repair.assert_awaited_once()
	assert repair.await_args.args[0] == '4444'


@pytest.mark.asyncio
async def test_cancelled_close_restores_an_retryable_tab_state():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	session.event_bus.dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())
	session._cdp_client_root = MagicMock()
	session._cdp_client_root.send.Target.closeTarget = AsyncMock(side_effect=asyncio.CancelledError)

	with patch.object(session, '_target_liveness', AsyncMock(return_value=True)):
		try:
			await session.on_CloseTabEvent(CloseTabEvent(target_id=target))
		except asyncio.CancelledError:
			pass
		else:
			raise AssertionError('Close cancellation did not propagate')
		await asyncio.wait_for(
			asyncio.gather(*list(session._focus_observer_tasks), return_exceptions=True),
			timeout=1.0,
		)

	assert session._tab_registry.get(target).state is TabState.OPEN


@pytest.mark.asyncio
async def test_cancelled_close_marks_a_dead_target_closed():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	session.event_bus.dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())
	session._cdp_client_root = MagicMock()
	session._cdp_client_root.send.Target.closeTarget = AsyncMock(side_effect=asyncio.CancelledError)

	with patch.object(session, '_target_liveness', AsyncMock(return_value=False)):
		try:
			await session.on_CloseTabEvent(CloseTabEvent(target_id=target))
		except asyncio.CancelledError:
			pass
		else:
			raise AssertionError('Close cancellation did not propagate')
		await asyncio.wait_for(
			asyncio.gather(*list(session._focus_observer_tasks), return_exceptions=True),
			timeout=1.0,
		)

	assert session._tab_registry.get(target).state is TabState.CLOSED


@pytest.mark.asyncio
async def test_timed_out_close_marks_a_dead_target_closed():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	session.event_bus.dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())
	session._cdp_client_root = MagicMock()
	session._cdp_client_root.send.Target.closeTarget = AsyncMock(side_effect=TimeoutError)

	with (
		patch.object(session, '_target_liveness', AsyncMock(return_value=False)),
		pytest.raises(FocusTargetError),
	):
		await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	assert session._tab_registry.get(target).state is TabState.CLOSED


@pytest.mark.asyncio
async def test_timed_out_close_keeps_a_live_target_open():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	session.event_bus.dispatch = MagicMock(side_effect=lambda event: _ready_awaitable())
	session._cdp_client_root = MagicMock()
	session._cdp_client_root.send.Target.closeTarget = AsyncMock(side_effect=TimeoutError)

	with (
		patch.object(session, '_target_liveness', AsyncMock(return_value=True)),
		pytest.raises(FocusTargetError),
	):
		await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	assert session._tab_registry.get(target).state is TabState.OPEN


@pytest.mark.asyncio
async def test_close_result_success_false_keeps_a_live_target_open():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	dispatch = MagicMock(side_effect=lambda event: _dispatch_result())
	session.event_bus.dispatch = dispatch
	session._cdp_client_root = MagicMock()
	session._cdp_client_root.send.Target.closeTarget = AsyncMock(return_value={'success': False})

	with (
		patch.object(session, '_target_liveness', AsyncMock(return_value=True)),
		pytest.raises(RuntimeError, match='success=false'),
	):
		await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	assert session._tab_registry.get(target).state is TabState.OPEN


@pytest.mark.asyncio
async def test_close_result_with_unknown_liveness_stays_closing():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	session.event_bus.dispatch = MagicMock(side_effect=lambda event: _dispatch_result())
	session._cdp_client_root = MagicMock()
	session._cdp_client_root.send.Target.closeTarget = AsyncMock(return_value={'success': False})

	with (
		patch.object(session, '_target_liveness', AsyncMock(return_value=None)),
		pytest.raises(RuntimeError, match='liveness is unknown'),
	):
		await session.on_CloseTabEvent(CloseTabEvent(target_id=target))

	assert session._tab_registry.get(target).state is TabState.CLOSING


@pytest.mark.asyncio
async def test_terminal_event_failure_propagates_and_leaves_the_tab_closed():
	session = _make_session()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	session._tab_registry.register(target, state=TabState.OPEN)
	session.event_bus.dispatch = MagicMock(side_effect=RuntimeError('event bus unavailable'))
	session._cdp_client_root = MagicMock()
	session._cdp_client_root.send.Target.closeTarget = AsyncMock()

	with pytest.raises(RuntimeError, match='event bus unavailable'):
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
async def test_kill_escalates_after_joining_a_graceful_shutdown():
	session = _make_session()
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.READY)
	watchdog = MagicMock()
	watchdog._subprocess = MagicMock()
	watchdog._spawned_subprocess = None
	watchdog._profile_lease = MagicMock()
	session._local_browser_watchdog = watchdog
	release = asyncio.Event()

	async def graceful_shutdown():
		await release.wait()
		session._lifecycle.transition(LifecyclePhase.STOPPING)
		session._lifecycle.transition(LifecyclePhase.STOPPED)

	session._stop_task = asyncio.create_task(graceful_shutdown())
	kill_task = asyncio.create_task(session.kill())
	await asyncio.sleep(0.05)
	with patch.object(session, '_force_stop_retained_browser', new_callable=AsyncMock, return_value=None) as force_stop:
		release.set()
		await kill_task

	force_stop.assert_awaited_once()


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
async def test_remote_cdp_url_is_not_a_live_resource_after_stop():
	session = BrowserSession(browser_profile=BrowserProfile(cdp_url='ws://127.0.0.1:9222', is_local=False))
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.STOPPED)

	await session.stop()
	await session.kill()

	assert session.lifecycle_snapshot()['phase'] == 'stopped'


@pytest.mark.asyncio
async def test_start_waits_for_an_in_flight_shutdown():
	session = _make_session()
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.STOPPING)
	release = asyncio.Event()

	async def shutdown():
		await release.wait()
		session._lifecycle.transition(LifecyclePhase.STOPPED)

	session._stop_task = asyncio.create_task(shutdown())

	async def fake_start():
		session._lifecycle.transition(LifecyclePhase.READY)

	with patch.object(session, '_start_locked', side_effect=fake_start):
		start_task = asyncio.create_task(session.start())
		await asyncio.sleep(0.05)
		assert session.lifecycle_snapshot()['phase'] == 'stopping'
		release.set()
		await start_task

	assert session.lifecycle_snapshot()['phase'] == 'ready'


def test_force_reset_clears_a_local_cdp_url_but_keeps_a_remote_one():
	local = BrowserSession(browser_profile=BrowserProfile(is_local=True, cdp_url='http://127.0.0.1:9222'))
	local._force_reset_runtime_state()
	assert local.cdp_url is None

	remote = BrowserSession(browser_profile=BrowserProfile(is_local=False, cdp_url='http://remote.example:9222'))
	remote._force_reset_runtime_state()
	assert remote.cdp_url == 'http://remote.example:9222'


@pytest.mark.asyncio
async def test_cancelled_start_releases_a_launched_browser_and_resets_state():
	session = _make_session()
	watchdog = MagicMock()
	watchdog._subprocess = MagicMock()
	watchdog._spawned_subprocess = None
	watchdog._profile_lease = MagicMock()
	watchdog.on_BrowserKillEvent = AsyncMock()

	async def hanging_start_event():
		await asyncio.sleep(60)

	session.event_bus.dispatch = MagicMock(return_value=hanging_start_event())
	start_task = asyncio.create_task(session.start())
	await asyncio.sleep(0.05)
	# Simulate resources acquired after the start operation began.
	session._local_browser_watchdog = watchdog
	session._cdp_client_root = MagicMock()
	start_task.cancel()

	with pytest.raises(asyncio.CancelledError):
		await start_task

	watchdog.on_BrowserKillEvent.assert_awaited_once()
	assert session._cdp_client_root is None
	assert session.lifecycle_snapshot()['phase'] == 'failed'


@pytest.mark.asyncio
async def test_stale_shutdown_does_not_reset_a_replacement_bus():
	session = _make_session()
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.READY)
	old_bus = session.event_bus
	replacement_bus = EventBus()

	def dispatch(event):
		if isinstance(event, BrowserStopEvent):
			session.event_bus = replacement_bus
			session._lifecycle.seed_generation(session._lifecycle.generation + 1)
		return _ready_awaitable()

	old_bus.dispatch = MagicMock(side_effect=dispatch)
	old_bus.stop = AsyncMock()

	with patch.object(session, '_cancel_owned_start', new_callable=AsyncMock):
		await session._shutdown_locked(force=False)

	old_bus.stop.assert_not_awaited()
	assert session.event_bus is replacement_bus


@pytest.mark.asyncio
async def test_stop_during_a_real_start_finishes_stopped_and_can_restart():
	session = _make_session()

	async def hanging_start_event():
		await asyncio.sleep(60)

	def dispatch(event):
		if isinstance(event, BrowserStartEvent):
			return hanging_start_event()
		return _ready_awaitable()

	session.event_bus.dispatch = MagicMock(side_effect=dispatch)
	start_task = asyncio.create_task(session.start())
	await asyncio.sleep(0.05)

	await session.stop()

	with pytest.raises(RuntimeError, match='cancelled by shutdown'):
		await start_task

	assert session.lifecycle_snapshot()['phase'] == 'stopped'
	assert session._stop_task is None

	async def fake_start():
		session._lifecycle.transition(LifecyclePhase.READY, reason='restart')

	with patch.object(session, '_start_locked', side_effect=fake_start):
		await session.start()

	assert session.lifecycle_snapshot()['phase'] == 'ready'


@pytest.mark.asyncio
async def test_failed_stop_retires_its_bus_and_allows_a_clean_restart():
	session = _make_session()
	session._cdp_client_root = MagicMock()
	save_event = _ready_awaitable()

	class StopEvent:
		async def _resolve(self):
			return None

		def __await__(self):
			return self._resolve().__await__()

	stop_event = StopEvent()
	stop_event.event_result = AsyncMock(side_effect=RuntimeError('watchdog cleanup failed'))
	old_bus = session.event_bus
	session.event_bus.dispatch = MagicMock(side_effect=[save_event, stop_event])
	old_bus.stop = AsyncMock()

	with pytest.raises(RuntimeError, match='watchdog cleanup failed'):
		await session.kill()

	assert session.event_bus is not old_bus
	assert session.lifecycle_snapshot()['phase'] == 'failed'

	async def fake_start():
		session._lifecycle.transition(LifecyclePhase.READY, reason='restart after failed stop')

	with patch.object(session, '_start_locked', side_effect=fake_start):
		await session.start()

	assert session.lifecycle_snapshot()['phase'] == 'ready'


@pytest.mark.asyncio
async def test_failed_force_stop_keeps_ownership_until_kill_retry_succeeds():
	session = _make_session()
	session._cdp_client_root = MagicMock()
	watchdog = MagicMock()
	watchdog._subprocess = MagicMock()
	watchdog._spawned_subprocess = None
	watchdog._profile_lease = MagicMock()
	watchdog.on_BrowserKillEvent = AsyncMock(side_effect=RuntimeError('kill failed'))
	session._local_browser_watchdog = watchdog

	class StopEvent:
		async def _resolve(self):
			return None

		def __await__(self):
			return self._resolve().__await__()

		def event_result(self, **kwargs):
			async def no_error():
				return None

			return no_error()

	old_bus = session.event_bus
	session.event_bus.dispatch = MagicMock(side_effect=[_ready_awaitable(), StopEvent()])
	old_bus.stop = AsyncMock()

	with pytest.raises(RuntimeError, match='kill failed'):
		await session.kill()

	assert session.event_bus is old_bus
	assert session._local_browser_watchdog is watchdog
	assert session.lifecycle_snapshot()['phase'] == 'failed'
	with pytest.raises(RuntimeError, match='resources are still live'):
		await session.start()
	with pytest.raises(RuntimeError, match='resources are still live'):
		await session.reset()

	async def release_on_retry(event):
		watchdog._subprocess = None
		watchdog._profile_lease = None

	watchdog.on_BrowserKillEvent = AsyncMock(side_effect=release_on_retry)
	session.event_bus.dispatch = MagicMock(side_effect=[_ready_awaitable(), StopEvent()])
	await session.kill()

	assert session.event_bus is not old_bus
	assert session._local_browser_watchdog is None
	assert session.lifecycle_snapshot()['phase'] == 'stopped'


@pytest.mark.asyncio
async def test_reset_cancels_and_waits_for_an_in_flight_start():
	session = _make_session()
	started = asyncio.Event()

	async def hanging_start():
		started.set()
		await asyncio.sleep(60)

	with patch.object(session, '_start_locked', side_effect=hanging_start):
		start_task = asyncio.create_task(session.start())
		await asyncio.wait_for(started.wait(), timeout=1)

		await session.reset()

		with pytest.raises(RuntimeError, match='cancelled by shutdown'):
			await start_task

	assert session.lifecycle_snapshot()['phase'] == 'stopped'


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


def test_lifecycle_snapshot_includes_storage_state_previews(tmp_path):
	state_path = tmp_path / 'state.json'
	state_path.write_text('{"cookies": [], "origins": []}')
	session = BrowserSession(browser_profile=BrowserProfile(headless=True, storage_state=str(state_path)))

	snapshot = session.lifecycle_snapshot()

	assert snapshot['storage_state']['phase'] == 'pending'
	assert snapshot['storage_state']['path'] == str(state_path.resolve())
	assert snapshot['storage_state_save']['phase'] == 'idle'
	assert snapshot['storage_state_save']['path'] == str(state_path.resolve())
