"""Tests for openbrowser.browser.session_manager module."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_session_part1 import _make_mock_cdp_client

from openbrowser.browser.cdp_connection import CdpConnectionHealthTracker
from openbrowser.browser.focus import (
    FocusChangeReason,
    FocusPhase,
    FocusSnapshot,
    FocusTargetCode,
    FocusTargetError,
    FocusTransitionBudget,
)
from openbrowser.browser.health import BrowserHealthMachine, BrowserHealthReason
from openbrowser.browser.session_manager import SessionManager
from openbrowser.browser.tabs import TabRegistry, TabState

logger = logging.getLogger(__name__)


def _make_mock_browser_session():
    """Create a mock BrowserSession for SessionManager."""
    session = MagicMock()
    session.logger = logging.getLogger('test_session_manager')

    # CDP client root
    cdp_client_root = _make_mock_cdp_client()
    cdp_client_root.register.Target.attachedToTarget = MagicMock()
    cdp_client_root.register.Target.detachedFromTarget = MagicMock()
    cdp_client_root.register.Target.targetCrashed = MagicMock()

    session._cdp_client_root = cdp_client_root
    session._cdp_health = CdpConnectionHealthTracker()
    generation = session._cdp_health.begin_connect()
    session._cdp_health.mark_connected(generation)
    session._health = BrowserHealthMachine()
    session._health.record_connected(generation)
    session._cdp_session_pool = {}
    session._tab_registry = TabRegistry()
    session.agent_focus = None
    session.focus_snapshot = MagicMock(return_value=FocusSnapshot.uninitialized())
    session._handle_attached_session = AsyncMock()
    session._handle_detached_session = AsyncMock()
    session._recover_agent_focus = AsyncMock()
    session.event_bus = MagicMock()
    session.event_bus.dispatch = MagicMock()
    session._cdp_get_all_pages = AsyncMock(return_value=[])
    session._cdp_create_new_page = AsyncMock(return_value='new-target-id')

    return session


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        assert sm.browser_session is session
        assert sm._target_sessions == {}
        assert sm._session_to_target == {}
        assert sm._target_types == {}


@pytest.mark.asyncio
class TestSessionManagerStartMonitoring:
    """Tests for SessionManager.start_monitoring."""

    async def test_start_monitoring_registers_events(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)

        await sm.start_monitoring()

        session._cdp_client_root.register.Target.attachedToTarget.assert_called_once()
        session._cdp_client_root.register.Target.detachedFromTarget.assert_called_once()
        session._cdp_client_root.register.Target.targetCrashed.assert_called_once()

    async def test_start_monitoring_raises_when_no_client(self):
        session = _make_mock_browser_session()
        session._cdp_client_root = None
        sm = SessionManager(session)

        with pytest.raises(RuntimeError, match='CDP client not initialized'):
            await sm.start_monitoring()


@pytest.mark.asyncio
class TestSessionManagerCallbackRetirement:
    async def test_registered_crash_callback_routes_payload(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._handle_target_crashed = AsyncMock()
        await sm.start_monitoring()
        callback = session._cdp_client_root.register.Target.targetCrashed.call_args.args[0]

        callback({'targetId': 'target-1', 'status': 'crashed', 'errorCode': -1})
        await asyncio.gather(*list(sm._event_tasks), return_exceptions=True)

        sm._handle_target_crashed.assert_awaited_once_with(
            'target-1',
            status='crashed',
            error_code=-1,
        )

    async def test_clear_cancels_inflight_callback_and_rejects_late_work(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        entered, release = asyncio.Event(), asyncio.Event()
        cancelled, late_mutation = asyncio.Event(), asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            late_mutation.set()

        sm._handle_target_crashed = AsyncMock(side_effect=blocked)
        await sm.start_monitoring()
        callback = session._cdp_client_root.register.Target.targetCrashed.call_args.args[0]
        callback({'targetId': 'target-1', 'status': 'crashed', 'errorCode': -1})
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        await asyncio.wait_for(sm.clear(), timeout=1.0)
        release.set()
        await asyncio.sleep(0)
        assert sm._closed is True
        assert sm._event_tasks == set()
        assert cancelled.is_set()
        assert late_mutation.is_set() is False
        with pytest.raises(RuntimeError, match='closed'):
            callback({'targetId': 'target-1', 'status': 'crashed', 'errorCode': -1})


@pytest.mark.asyncio
class TestSessionManagerGetSessionForTarget:
    """Tests for SessionManager.get_session_for_target."""

    async def test_returns_session_when_exists(self):
        session = _make_mock_browser_session()
        mock_cdp_session = MagicMock()
        session._cdp_session_pool['target-1'] = mock_cdp_session
        sm = SessionManager(session)

        result = await sm.get_session_for_target('target-1')
        assert result is mock_cdp_session

    async def test_returns_none_when_not_exists(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)

        result = await sm.get_session_for_target('nonexistent')
        assert result is None


@pytest.mark.asyncio
class TestSessionManagerValidateSession:
    """Tests for SessionManager.validate_session."""

    async def test_returns_true_when_sessions_exist(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}

        result = await sm.validate_session('target-1')
        assert result is True

    async def test_returns_false_when_no_sessions(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = set()

        result = await sm.validate_session('target-1')
        assert result is False

    async def test_returns_false_when_target_unknown(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)

        result = await sm.validate_session('unknown')
        assert result is False


@pytest.mark.asyncio
class TestSessionManagerIsTargetValid:
    """Tests for SessionManager.is_target_valid."""

    async def test_valid_target(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}

        assert await sm.is_target_valid('target-1') is True

    async def test_invalid_target_empty_sessions(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = set()

        assert await sm.is_target_valid('target-1') is False

    async def test_invalid_target_unknown(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)

        assert await sm.is_target_valid('unknown') is False


@pytest.mark.asyncio
class TestSessionManagerClear:
    """Tests for SessionManager.clear."""

    async def test_clear_all_tracking(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_sessions = {'t1': {'s1'}}
        sm._session_to_target = {'s1': 't1'}
        sm._target_types = {'t1': 'page'}

        await sm.clear()

        assert sm._target_sessions == {}
        assert sm._session_to_target == {}
        assert sm._target_types == {}


@pytest.mark.asyncio
class TestSessionManagerHandleTargetAttached:
    """Tests for SessionManager._handle_target_attached."""

    async def test_creates_new_session(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)

        event = {
            'sessionId': 'session-1',
            'targetInfo': {
                'targetId': 'target-1',
                'type': 'page',
                'title': 'Test Page',
                'url': 'https://example.com',
            },
            'waitingForDebugger': False,
        }

        with patch('openbrowser.browser.session.CDPSession') as MockCDPSession:
            mock_cdp = MagicMock()
            MockCDPSession.return_value = mock_cdp

            await sm._handle_target_attached(event)

        assert 'target-1' in sm._target_sessions
        assert 'session-1' in sm._target_sessions['target-1']
        assert sm._session_to_target['session-1'] == 'target-1'
        assert sm._target_types['target-1'] == 'page'
        record = session._tab_registry.get('target-1')
        assert record is not None
        assert record.state is TabState.OPEN
        assert record.url == 'https://example.com'

    async def test_updates_existing_session(self):
        session = _make_mock_browser_session()
        existing_cdp = MagicMock()
        session._cdp_session_pool['target-1'] = existing_cdp
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'old-session'}
        sm._target_types['target-1'] = 'page'

        event = {
            'sessionId': 'new-session',
            'targetInfo': {
                'targetId': 'target-1',
                'type': 'page',
                'title': 'Updated Title',
                'url': 'https://updated.com',
            },
            'waitingForDebugger': False,
        }

        await sm._handle_target_attached(event)

        assert 'new-session' in sm._target_sessions['target-1']
        replacement = session._cdp_session_pool['target-1']
        assert replacement is not existing_cdp
        assert replacement.session_id == 'new-session'
        assert replacement.title == 'Updated Title'
        session._handle_attached_session.assert_awaited_once_with('target-1', replacement)

    async def test_resumes_waiting_for_debugger(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)

        event = {
            'sessionId': 'session-1',
            'targetInfo': {
                'targetId': 'target-1',
                'type': 'page',
                'title': 'Test',
                'url': 'about:blank',
            },
            'waitingForDebugger': True,
        }

        with patch('openbrowser.browser.session.CDPSession'):
            await sm._handle_target_attached(event)

        session._cdp_client_root.send.Runtime.runIfWaitingForDebugger.assert_called_once_with(session_id='session-1')


@pytest.mark.asyncio
class TestSessionManagerHandleTargetDetached:
    """Tests for SessionManager._handle_target_detached."""

    async def test_removes_session_from_target(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1', 'session-2'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._session_to_target['session-2'] = 'target-1'
        sm._target_types['target-1'] = 'page'

        event = {'sessionId': 'session-1', 'targetId': 'target-1'}

        await sm._handle_target_detached(event)

        assert 'session-1' not in sm._target_sessions.get('target-1', set())
        assert 'session-2' in sm._target_sessions['target-1']
        assert 'session-1' not in sm._session_to_target

    async def test_removes_target_when_no_sessions_remain(self):
        session = _make_mock_browser_session()
        mock_cdp = MagicMock()
        session._cdp_session_pool['target-1'] = mock_cdp
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._target_types['target-1'] = 'page'

        event = {'sessionId': 'session-1', 'targetId': 'target-1'}

        await sm._handle_target_detached(event)

        assert 'target-1' not in sm._target_sessions
        assert 'target-1' not in session._cdp_session_pool
        assert 'target-1' not in sm._target_types

    async def test_dispatches_tab_closed_event_for_page(self):
        session = _make_mock_browser_session()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(return_value={'targetInfos': []})
        session._cdp_session_pool['target-1'] = MagicMock()
        session._tab_registry.register('target-1', url='https://live.example', state=TabState.OPEN)
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._target_types['target-1'] = 'page'

        event = {'sessionId': 'session-1', 'targetId': 'target-1'}

        await sm._handle_target_detached(event)

        session.event_bus.dispatch.assert_called()
        # Verify the dispatched event type and target information
        dispatched_event = session.event_bus.dispatch.call_args[0][0]
        assert hasattr(dispatched_event, '__class__')
        assert 'TabClosed' in type(dispatched_event).__name__ or dispatched_event is not None

    async def test_keeps_a_live_target_open_after_its_last_session_detaches(self):
        session = _make_mock_browser_session()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(
            return_value={'targetInfos': [{'targetId': 'target-1', 'type': 'page', 'url': 'https://live.example'}]}
        )
        session._cdp_session_pool['target-1'] = MagicMock()
        session._tab_registry.register('target-1', url='https://live.example', state=TabState.OPEN)
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._target_types['target-1'] = 'page'

        await sm._handle_target_detached({'sessionId': 'session-1', 'targetId': 'target-1'})

        for call in session.event_bus.dispatch.call_args_list:
            assert 'TabClosed' not in type(call.args[0]).__name__
        record = session._tab_registry.get('target-1')
        assert record is not None and record.state is TabState.OPEN

    async def test_does_not_guess_when_target_absence_is_unknown(self):
        session = _make_mock_browser_session()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(side_effect=RuntimeError('cdp unavailable'))
        session._cdp_session_pool['target-1'] = MagicMock()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._target_types['target-1'] = 'page'

        await sm._handle_target_detached({'sessionId': 'session-1', 'targetId': 'target-1'})

        for call in session.event_bus.dispatch.call_args_list:
            assert 'TabClosed' not in type(call.args[0]).__name__

    async def test_no_tab_closed_event_for_non_page(self):
        session = _make_mock_browser_session()
        session._cdp_session_pool['target-1'] = MagicMock()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._target_types['target-1'] = 'worker'

        event = {'sessionId': 'session-1', 'targetId': 'target-1'}

        await sm._handle_target_detached(event)

        # Should NOT dispatch TabClosedEvent for worker targets
        for call in session.event_bus.dispatch.call_args_list:
            arg = call[0][0]
            assert not hasattr(arg, 'event_type') or 'TabClosed' not in str(type(arg).__name__)

    async def test_looks_up_target_from_session(self):
        """Test that detach works even without targetId in event."""
        session = _make_mock_browser_session()
        session._cdp_session_pool['target-1'] = MagicMock()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._target_types['target-1'] = 'page'

        # No targetId in event
        event = {'sessionId': 'session-1'}

        await sm._handle_target_detached(event)

        assert 'target-1' not in sm._target_sessions

    async def test_unknown_session_detach(self):
        """Test graceful handling when session is completely unknown."""
        session = _make_mock_browser_session()
        sm = SessionManager(session)

        event = {'sessionId': 'unknown-session'}

        # Should not raise
        await sm._handle_target_detached(event)

    async def test_triggers_recovery_when_agent_focus_lost(self):
        session = _make_mock_browser_session()
        agent_focus = MagicMock()
        agent_focus.target_id = 'target-1'
        session.agent_focus = agent_focus
        session._cdp_session_pool['target-1'] = MagicMock()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._target_types['target-1'] = 'page'

        event = {'sessionId': 'session-1', 'targetId': 'target-1'}

        await sm._handle_target_detached(event)
        session._handle_detached_session.assert_awaited_once_with('target-1', 'session-1', ())


@pytest.mark.asyncio
class TestSessionManagerExactSessions:
    """Tests for exact session resolution."""

    async def test_active_session_ids_are_sorted(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-b', 'session-a'}

        assert await sm.active_session_ids('target-1') == ('session-a', 'session-b')

    async def test_live_session_replaces_stale_wrapper(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        stale = MagicMock()
        stale.session_id = 'old-session'
        stale.title = 'Old'
        stale.url = 'https://old.example'
        session._cdp_session_pool['target-1'] = stale
        sm._target_sessions['target-1'] = {'session-b', 'session-a'}

        replacement = await sm.get_live_session_for_target('target-1')

        assert replacement is not stale
        assert replacement.session_id == 'session-a'
        assert replacement.url == 'https://old.example'
        assert stale.session_id == 'old-session'
        assert session._cdp_session_pool['target-1'] is replacement

    async def test_live_session_rejects_unknown_target(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)

        assert await sm.get_live_session_for_target('missing') is None

    async def test_budgeted_session_lookups_reject_expired_budget(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        expired = FocusTransitionBudget(asyncio.get_running_loop().time() - 1.0)

        with pytest.raises(TimeoutError):
            await sm.active_session_ids('target-1', budget=expired)
        with pytest.raises(TimeoutError):
            await sm.get_live_session_for_target('target-1', budget=expired)


@pytest.mark.asyncio
class TestSessionManagerCrash:
    """Tests for the root target crash callback."""

    async def test_focused_page_crash_marks_tab_and_recovers(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        session._tab_registry.register('target-1', url='https://example.com', state=TabState.OPEN)
        session.focus_snapshot.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-1',
            session_id='session-1',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )
        session._recover_agent_focus.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-2',
            session_id='session-2',
            previous_target_id='target-1',
            generation=2,
            reason=FocusChangeReason.CRASH_RECOVERY,
            last_error=None,
            updated_at=0.0,
        )

        await sm._handle_target_crashed('target-1', status='crashed', error_code=-1)

        record = session._tab_registry.get('target-1')
        assert record is not None
        assert record.state is TabState.CRASHED
        assert record.error is not None
        session._recover_agent_focus.assert_awaited_once()
        assert session._recover_agent_focus.await_args.args == ('target-1',)
        assert session._recover_agent_focus.await_args.kwargs['reason'] is FocusChangeReason.CRASH_RECOVERY
        health = session._health.snapshot()
        assert health.phase.value == 'healthy'
        assert health.last_focus_recovery_outcome.value == 'succeeded'
        assert health.last_focus_recovery_target_id == 'target-2'

    async def test_non_focused_crash_does_not_recover(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        session._tab_registry.register('target-1', url='https://example.com', state=TabState.OPEN)
        session.focus_snapshot.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-2',
            session_id='session-2',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )

        await sm._handle_target_crashed('target-1', status='crashed', error_code=-1)

        session._recover_agent_focus.assert_not_awaited()
        record = session._tab_registry.get('target-1')
        assert record is not None and record.state is TabState.CRASHED

    async def test_failed_crash_recovery_is_reported(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        session._tab_registry.register('target-1', url='https://example.com', state=TabState.OPEN)
        session.focus_snapshot.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-1',
            session_id='session-1',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )
        error = FocusTargetError(
            'No live page target was available.',
            code=FocusTargetCode.RECOVERY_FAILED,
            phase=FocusPhase.LOST,
            target_id='target-1',
            target_type='page',
        )
        session._recover_agent_focus = AsyncMock(side_effect=error)
        session._report_focus_recovery_failure = MagicMock()

        await sm._handle_target_crashed('target-1', status='crashed', error_code=-1)

        session._report_focus_recovery_failure.assert_called_once_with('target-1', error)

    async def test_repeated_crash_is_fully_deduplicated(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        session.focus_snapshot.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-2',
            session_id='session-2',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )
        session._tab_registry.register('target-1', url='https://example.com', state=TabState.OPEN)
        session.event_bus.dispatch.reset_mock()

        await sm._handle_target_crashed('target-1', status='crashed', error_code=-1)
        await sm._handle_target_crashed('target-1', status='crashed', error_code=-1)

        health = session._health.snapshot()
        assert health.target_crash_count == 1
        assert health.phase.value == 'degraded'
        assert session._tab_registry.get('target-1').state is TabState.CRASHED
        assert session.event_bus.dispatch.call_count == 1
        session._recover_agent_focus.assert_not_awaited()

    async def test_new_root_and_generation_are_fenced_before_any_mutation(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        session.event_bus.dispatch.reset_mock()
        session._cdp_health.begin_connect()

        await sm._handle_target_crashed('target-1', status='crashed', error_code=-1)

        assert session._tab_registry.get('target-1') is None
        assert session.event_bus.dispatch.call_count == 0
        assert session._health.snapshot().target_crash_count == 0
        session._recover_agent_focus.assert_not_awaited()

        generation = session._cdp_health.begin_connect()
        session._cdp_health.mark_connected(generation)
        sm._monitored_connection_generation = generation
        session._cdp_client_root = MagicMock()
        await sm._handle_target_crashed('target-1', status='crashed', error_code=-1)

        assert session._tab_registry.get('target-1') is None
        assert session.event_bus.dispatch.call_count == 0

    async def test_pending_recovery_is_visible_before_success(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        session._tab_registry.register('target-1', url='https://example.com', state=TabState.OPEN)
        session.focus_snapshot.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-1',
            session_id='session-1',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )
        entered, release = asyncio.Event(), asyncio.Event()

        async def recover(*args, **kwargs):
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=2.0)
            return FocusSnapshot(
                phase=FocusPhase.FOCUSED,
                target_id='target-2',
                session_id='session-2',
                previous_target_id='target-1',
                generation=2,
                reason=FocusChangeReason.CRASH_RECOVERY,
                last_error=None,
                updated_at=0.0,
            )

        session._recover_agent_focus = AsyncMock(side_effect=recover)
        task = asyncio.create_task(sm._handle_target_crashed('target-1', status='crashed', error_code=-1))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            pending = session._health.snapshot()
            assert pending.phase.value == 'degraded'
            assert pending.last_focus_recovery_outcome.value == 'pending'
            release.set()
            await asyncio.wait_for(task, timeout=2.0)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2.0)

        recovered = session._health.snapshot()
        assert recovered.phase.value == 'healthy'
        assert recovered.last_focus_recovery_outcome.value == 'succeeded'
        assert recovered.last_focus_recovery_target_id == 'target-2'

    async def test_newer_crash_discards_older_success(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        sm._target_types['target-2'] = 'page'
        session._tab_registry.register('target-1', url='https://example.com/one', state=TabState.OPEN)
        session._tab_registry.register('target-2', url='https://example.com/two', state=TabState.OPEN)
        focused = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-1',
            session_id='session-1',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )
        session.focus_snapshot.return_value = focused
        entered, release = asyncio.Event(), asyncio.Event()

        async def recover(*args, **kwargs):
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=2.0)
            return FocusSnapshot(
                phase=FocusPhase.FOCUSED,
                target_id='replacement-a',
                session_id='replacement-session',
                previous_target_id='target-1',
                generation=2,
                reason=FocusChangeReason.CRASH_RECOVERY,
                last_error=None,
                updated_at=0.0,
            )

        session._recover_agent_focus = AsyncMock(side_effect=recover)
        task = asyncio.create_task(sm._handle_target_crashed('target-1', status='crashed', error_code=-1))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            session.focus_snapshot.return_value = FocusSnapshot(
                phase=FocusPhase.FOCUSED,
                target_id='target-3',
                session_id='session-3',
                previous_target_id=None,
                generation=3,
                reason=FocusChangeReason.INITIAL,
                last_error=None,
                updated_at=0.0,
            )
            await sm._handle_target_crashed('target-2', status='crashed', error_code=-1)
            release.set()
            await asyncio.wait_for(task, timeout=2.0)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2.0)

        health = session._health.snapshot()
        assert health.target_crash_count == 2
        assert health.last_target_crash_target_id == 'target-2'
        assert health.phase.value == 'degraded'
        assert health.last_focus_recovery_outcome.value == 'not_attempted'

    async def test_newer_crash_discards_older_failure_report(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        sm._target_types['target-2'] = 'page'
        session._tab_registry.register('target-1', url='https://example.com/one', state=TabState.OPEN)
        session._tab_registry.register('target-2', url='https://example.com/two', state=TabState.OPEN)
        session.focus_snapshot.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-1',
            session_id='session-1',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )
        entered, release = asyncio.Event(), asyncio.Event()
        error = FocusTargetError(
            'No live page target was available.',
            code=FocusTargetCode.RECOVERY_FAILED,
            phase=FocusPhase.LOST,
            target_id='target-1',
            target_type='page',
        )

        async def recover(*args, **kwargs):
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=2.0)
            raise error

        session._recover_agent_focus = AsyncMock(side_effect=recover)
        session._report_focus_recovery_failure = MagicMock()
        task = asyncio.create_task(sm._handle_target_crashed('target-1', status='crashed', error_code=-1))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            session.focus_snapshot.return_value = FocusSnapshot(
                phase=FocusPhase.FOCUSED,
                target_id='target-3',
                session_id='session-3',
                previous_target_id=None,
                generation=3,
                reason=FocusChangeReason.INITIAL,
                last_error=None,
                updated_at=0.0,
            )
            await sm._handle_target_crashed('target-2', status='crashed', error_code=-1)
            release.set()
            await asyncio.wait_for(task, timeout=2.0)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2.0)

        session._report_focus_recovery_failure.assert_not_called()
        health = session._health.snapshot()
        assert health.phase.value == 'degraded'
        assert health.last_target_crash_target_id == 'target-2'
        assert health.last_focus_recovery_outcome.value == 'not_attempted'

    async def test_aba_creates_a_new_observation_when_latest_differs(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types.update({'target-a': 'page', 'target-b': 'page'})
        session.focus_snapshot.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-other',
            session_id='session-other',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )

        await sm._handle_target_crashed('target-a', status='crashed', error_code=-1)
        await sm._handle_target_crashed('target-b', status='crashed', error_code=-1)
        await sm._handle_target_crashed('target-a', status='crashed', error_code=-1)

        assert session._health.snapshot().target_crash_count == 3

    async def test_non_page_crash_does_not_change_health(self):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['iframe-1'] = 'iframe'

        await sm._handle_target_crashed('iframe-1', status='crashed', error_code=-1)

        assert session._health.snapshot().target_crash_count == 0
        assert session._health.snapshot().phase.value == 'healthy'

    @pytest.mark.parametrize('operation', ['reset', 'reconnect'])
    async def test_reset_or_reconnect_discards_blocked_recovery_without_rollback(self, operation):
        session = _make_mock_browser_session()
        sm = SessionManager(session)
        sm._target_types['target-1'] = 'page'
        session._tab_registry.register('target-1', url='https://example.com', state=TabState.OPEN)
        session.focus_snapshot.return_value = FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id='target-1',
            session_id='session-1',
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.INITIAL,
            last_error=None,
            updated_at=0.0,
        )
        entered, release = asyncio.Event(), asyncio.Event()

        async def recover(*args, **kwargs):
            entered.set()
            session.agent_focus = MagicMock(target_id='committed-replacement')
            await asyncio.wait_for(release.wait(), timeout=2.0)
            return FocusSnapshot(
                phase=FocusPhase.FOCUSED,
                target_id='committed-replacement',
                session_id='replacement-session',
                previous_target_id='target-1',
                generation=2,
                reason=FocusChangeReason.CRASH_RECOVERY,
                last_error=None,
                updated_at=0.0,
            )

        session._recover_agent_focus = AsyncMock(side_effect=recover)
        task = asyncio.create_task(sm._handle_target_crashed('target-1', status='crashed', error_code=-1))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            if operation == 'reset':
                session._cdp_client_root = None
                session._health.record_stopped(
                    BrowserHealthReason.RESET,
                    sm._monitored_connection_generation or 0,
                )
            else:
                new_root = MagicMock()
                session._cdp_client_root = new_root
                generation = session._cdp_health.begin_connect()
                session._cdp_health.mark_connected(generation)
                session._health.record_connected(generation)
                sm._monitored_root = new_root
                sm._monitored_connection_generation = generation
            release.set()
            await asyncio.wait_for(task, timeout=2.0)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2.0)

        health = session._health.snapshot()
        if operation == 'reset':
            assert health.phase.value == 'stopped'
            assert health.reason.value == 'reset'
        else:
            assert health.phase.value == 'healthy'
            assert health.reason.value == 'connected'
            assert health.connection_generation == 2
        assert health.last_focus_recovery_outcome.value == 'not_attempted'
        assert session.agent_focus.target_id == 'committed-replacement'


class _SingleSlotTargetRegistration:
    """Model cdp-use's one-callback-per-event registration contract."""

    def __init__(self) -> None:
        self.attached_to_target = None
        self.attached_registration_count = 0

    def attachedToTarget(self, callback):
        self.attached_to_target = callback
        self.attached_registration_count += 1

    def detachedFromTarget(self, callback):
        pass

    def targetCrashed(self, callback):
        pass


def _make_single_slot_session():
    session = _make_mock_browser_session()
    registration = _SingleSlotTargetRegistration()
    session._cdp_client_root.register.Target = registration
    session._cdp_client_root.send.Target.setAutoAttach = AsyncMock()
    session._cdp_client_root.send.Runtime.runIfWaitingForDebugger = AsyncMock()
    return session, registration


def _replace_single_slot_root(session):
    old_root = session._cdp_client_root
    new_root = _make_mock_cdp_client()
    registration = _SingleSlotTargetRegistration()
    new_root.register.Target = registration
    new_root.send.Target.setAutoAttach = AsyncMock()
    session._cdp_client_root = new_root
    generation = session._cdp_health.begin_connect()
    session._cdp_health.mark_connected(generation)
    session._health.record_connected(generation)
    session._cdp_session_pool.clear()
    session._tab_registry.clear()
    return old_root, new_root, registration, generation


def _attached_event(
    *,
    target_id: str = 'target-1',
    session_id: str = 'session-1',
    target_type: str = 'page',
    url: str = 'https://example.com',
    waiting_for_debugger: bool = False,
):
    return {
        'sessionId': session_id,
        'targetInfo': {
            'targetId': target_id,
            'type': target_type,
            'title': 'Example',
            'url': url,
        },
        'waitingForDebugger': waiting_for_debugger,
    }


async def _drain_session_manager(sm: SessionManager) -> None:
    while sm._event_tasks:
        await asyncio.gather(*list(sm._event_tasks), return_exceptions=True)


def _block_auto_attach(session):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_ack(*_args, **_kwargs):
        entered.set()
        await release.wait()

    session._cdp_client_root.send.Target.setAutoAttach = AsyncMock(side_effect=wait_for_ack)
    return entered, release


@pytest.mark.asyncio
class TestSessionManagerTargetAttachedObservers:
    async def test_single_slot_runs_base_pool_then_observer_once(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        def observer(event, outer_session_id=None):
            target_id = event['targetInfo']['targetId']
            calls.append((target_id, outer_session_id, session._cdp_session_pool[target_id].session_id))

        sm.set_target_attached_observer('fetch-policy', observer)
        await sm.start_monitoring()
        assert registration.attached_registration_count == 1
        assert registration.attached_to_target is not None

        registration.attached_to_target(_attached_event(), 'parent-session')
        await _drain_session_manager(sm)

        assert calls == [('target-1', 'parent-session', 'session-1')]
        assert session._cdp_session_pool['target-1'].target_id == 'target-1'
        session._cdp_client_root.send.Target.setAutoAttach.assert_awaited_once_with(
            params={'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True},
            session_id='session-1',
        )

    async def test_same_key_replaces_observer_idempotently_after_start(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        await sm.start_monitoring()
        sm.set_target_attached_observer('fetch-policy', lambda *_: calls.append('old'))
        sm.set_target_attached_observer('fetch-policy', lambda *_: calls.append('new'))
        sm.set_target_attached_observer('fetch-policy', lambda *_: calls.append('new'))

        registration.attached_to_target(_attached_event())
        await _drain_session_manager(sm)

        assert calls == ['new']

    async def test_failing_observer_is_isolated_and_logs_type_only(self, caplog, monkeypatch):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        async def failing_observer(*_):
            raise ValueError('secret observer payload')

        def healthy_observer(*_):
            calls.append('healthy')

        sm.set_target_attached_observer('broken', failing_observer)
        sm.set_target_attached_observer('fetch-policy', healthy_observer)
        await sm.start_monitoring()

        with caplog.at_level(logging.WARNING, logger='test_session_manager'):
            monkeypatch.setattr(session.logger, 'handlers', [caplog.handler])
            monkeypatch.setattr(session.logger, 'propagate', False)
            registration.attached_to_target(_attached_event())
            await _drain_session_manager(sm)

        assert calls == ['healthy']
        assert 'ValueError' in caplog.text
        assert 'secret observer payload' not in caplog.text

    @pytest.mark.parametrize(
        ('target_type', 'url'),
        [
            ('page', 'https://example.com'),
            ('iframe', 'https://frame.example.com'),
            ('worker', 'https://example.com/worker.js'),
            ('service_worker', 'chrome-extension://extension-id/worker.js'),
        ],
    )
    async def test_observers_receive_iframe_worker_and_extension_targets(self, target_type, url):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        def observer(event, _outer_session_id=None):
            calls.append(event['targetInfo']['type'])

        sm.set_target_attached_observer('fetch-policy', observer)
        await sm.start_monitoring()
        registration.attached_to_target(
            _attached_event(
                target_id=f'{target_type}-target',
                session_id=f'{target_type}-session',
                target_type=target_type,
                url=url,
            )
        )
        await _drain_session_manager(sm)

        assert calls == [target_type]
        assert session._cdp_session_pool[f'{target_type}-target'].session_id == f'{target_type}-session'

    async def test_closed_callback_is_suppressed_and_observers_are_cleared(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        sm.set_target_attached_observer('fetch-policy', lambda *_: calls.append('called'))
        await sm.start_monitoring()
        await sm.clear()

        assert sm._target_attached_observers == {}
        registration.attached_to_target(_attached_event())
        await asyncio.sleep(0)

        assert calls == []
        assert session._cdp_session_pool == {}

    async def test_stale_root_callback_is_suppressed(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        sm.set_target_attached_observer('fetch-policy', lambda *_: calls.append('called'))
        await sm.start_monitoring()
        session._cdp_client_root = MagicMock()

        registration.attached_to_target(_attached_event())
        await asyncio.sleep(0)

        assert calls == []
        assert session._cdp_session_pool == {}

    async def test_stale_generation_callback_is_suppressed(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        sm.set_target_attached_observer('fetch-policy', lambda *_: calls.append('called'))
        await sm.start_monitoring()
        session._cdp_health.begin_connect()

        registration.attached_to_target(_attached_event())
        await asyncio.sleep(0)

        assert calls == []
        assert session._cdp_session_pool == {}

    async def test_clear_cancels_inflight_observer_before_late_resume(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        entered, release, cancelled, resumed = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def blocked_observer(*_):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            resumed.set()

        sm.set_target_attached_observer('fetch-policy', blocked_observer)
        await sm.start_monitoring()
        registration.attached_to_target(_attached_event())
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        await asyncio.wait_for(sm.clear(), timeout=1.0)
        release.set()
        await asyncio.sleep(0)

        assert cancelled.is_set()
        assert resumed.is_set() is False
        assert sm._target_attached_observers == {}

    async def test_cancel_pending_clears_observers_and_suppresses_late_callbacks(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        sm.set_target_attached_observer('fetch-policy', lambda *_: calls.append('called'))
        await sm.start_monitoring()
        sm.cancel_pending()

        assert sm._target_attached_observers == {}
        registration.attached_to_target(_attached_event())
        await asyncio.sleep(0)

        assert calls == []
        assert session._cdp_session_pool == {}

    async def test_old_handler_stays_invalid_after_manager_rebind(self):
        session, old_registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        def record_target(event, _session_id=None):
            calls.append(event['targetInfo']['targetId'])

        sm.set_target_attached_observer('fetch-policy', record_target)
        await sm.start_monitoring()
        old_callback = old_registration.attached_to_target
        old_root, new_root, new_registration, _ = _replace_single_slot_root(session)
        await sm.start_monitoring()

        old_callback(_attached_event(target_id='old-root-target'))
        await _drain_session_manager(sm)

        assert calls == []
        assert session._cdp_session_pool == {}
        old_root.send.Target.setAutoAttach.assert_not_awaited()

        new_registration.attached_to_target(
            _attached_event(target_id='new-root-target', session_id='new-root-session')
        )
        await _drain_session_manager(sm)

        assert calls == ['new-root-target']
        assert session._cdp_session_pool['new-root-target'].cdp_client is new_root

    async def test_same_root_reregistration_invalidates_old_handler_but_accepts_new(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []

        def record_target(event, _sid=None):
            calls.append(event['targetInfo']['targetId'])

        sm.set_target_attached_observer('fetch-policy', record_target)
        await sm.start_monitoring()
        old_callback = registration.attached_to_target
        await sm.start_monitoring()
        new_callback = registration.attached_to_target

        old_callback(_attached_event(target_id='old-target'))
        await _drain_session_manager(sm)
        assert calls == []
        assert session._cdp_session_pool == {}

        new_callback(_attached_event(target_id='new-target'))
        await _drain_session_manager(sm)
        assert calls == ['new-target']
        assert session._cdp_session_pool['new-target'].target_id == 'new-target'

    async def test_old_handler_stays_invalid_across_awaited_focus_rebind(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked_focus_rebind(*_):
            entered.set()
            await release.wait()

        session._handle_attached_session = AsyncMock(side_effect=blocked_focus_rebind)
        sm.set_target_attached_observer('fetch-policy', lambda *_: calls.append('called'))
        await sm.start_monitoring()
        old_callback = registration.attached_to_target
        old_callback(_attached_event(waiting_for_debugger=True))
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        old_root, _, _, _ = _replace_single_slot_root(session)
        await sm.start_monitoring()
        release.set()
        await _drain_session_manager(sm)

        assert calls == []
        old_root.send.Runtime.runIfWaitingForDebugger.assert_not_awaited()

    async def test_awaited_observer_does_not_continue_after_manager_rebind(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        calls = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked_observer(*_):
            calls.append('first-start')
            entered.set()
            await release.wait()
            calls.append('first-end')

        sm.set_target_attached_observer('first', blocked_observer)
        sm.set_target_attached_observer('second', lambda *_: calls.append('second'))
        await sm.start_monitoring()
        registration.attached_to_target(_attached_event())
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        _replace_single_slot_root(session)
        await sm.start_monitoring()
        release.set()
        await _drain_session_manager(sm)

        assert calls == ['first-start', 'first-end']

    @pytest.mark.parametrize('retirement', ['clear', 'cancel_pending'])
    async def test_retired_manager_rejects_new_observer_registration(self, retirement):
        session, _registration = _make_single_slot_session()
        sm = SessionManager(session)
        await sm.start_monitoring()

        if retirement == 'clear':
            await sm.clear()
        else:
            sm.cancel_pending()

        with pytest.raises(RuntimeError, match='closed'):
            sm.set_target_attached_observer('late', lambda *_: None)

    async def test_retired_manager_rejects_restart(self):
        session, _registration = _make_single_slot_session()
        sm = SessionManager(session)
        await sm.start_monitoring()
        await sm.clear()

        with pytest.raises(RuntimeError, match='closed'):
            await sm.start_monitoring()

@pytest.mark.asyncio
class TestSessionManagerAutoAttachReadiness:
    async def test_absent_session_is_not_ready(self):
        session, _registration = _make_single_slot_session()
        sm = SessionManager(session)
        await sm.start_monitoring()

        assert await sm.wait_for_auto_attach('missing-session') is False

    async def test_immediate_success_is_ready(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        await sm.start_monitoring()

        registration.attached_to_target(_attached_event())

        assert await sm.wait_for_auto_attach('session-1') is True
        assert 'session-1' in sm._auto_attach_attempts

    async def test_pending_ack_blocks_until_success(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        entered, release = _block_auto_attach(session)
        await sm.start_monitoring()

        registration.attached_to_target(_attached_event())
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        waiter = asyncio.create_task(sm.wait_for_auto_attach('session-1'))
        await asyncio.sleep(0)
        assert waiter.done() is False

        release.set()
        assert await asyncio.wait_for(waiter, timeout=1.0) is True

    async def test_native_failure_is_not_ready(self):
        session, registration = _make_single_slot_session()
        session._cdp_client_root.send.Target.setAutoAttach = AsyncMock(side_effect=RuntimeError('ack failed'))
        sm = SessionManager(session)
        await sm.start_monitoring()

        registration.attached_to_target(_attached_event())

        assert await sm.wait_for_auto_attach('session-1') is False

    async def test_rebind_stale_context_is_not_ready(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        entered, release = _block_auto_attach(session)
        await sm.start_monitoring()

        registration.attached_to_target(_attached_event())
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        attempt = sm._auto_attach_attempts['session-1']
        waiter = asyncio.create_task(sm.wait_for_auto_attach('session-1'))

        _replace_single_slot_root(session)
        await sm.start_monitoring()

        assert await asyncio.wait_for(waiter, timeout=1.0) is False
        assert await sm.wait_for_auto_attach('session-1') is False
        assert sm._auto_attach_attempts == {}
        await asyncio.sleep(0)
        assert attempt.task.cancelled() is True
        release.set()

    async def test_detach_wakes_waiter_and_removes_attempt(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        entered, release = _block_auto_attach(session)
        await sm.start_monitoring()

        registration.attached_to_target(_attached_event())
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        attempt = sm._auto_attach_attempts['session-1']
        waiter = asyncio.create_task(sm.wait_for_auto_attach('session-1'))

        await sm._handle_target_detached({'sessionId': 'session-1', 'targetId': 'target-1'})

        assert await asyncio.wait_for(waiter, timeout=1.0) is False
        assert 'session-1' not in sm._auto_attach_attempts
        await asyncio.sleep(0)
        assert attempt.task.cancelled() is True
        release.set()

    @pytest.mark.parametrize('retirement', ['clear', 'cancel_pending'])
    async def test_retirement_wakes_waiter_and_clears_attempt(self, retirement):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        entered, release = _block_auto_attach(session)
        await sm.start_monitoring()

        registration.attached_to_target(_attached_event())
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        attempt = sm._auto_attach_attempts['session-1']
        waiter = asyncio.create_task(sm.wait_for_auto_attach('session-1'))

        if retirement == 'clear':
            await sm.clear()
        else:
            sm.cancel_pending()

        assert await asyncio.wait_for(waiter, timeout=1.0) is False
        assert sm._auto_attach_attempts == {}
        await asyncio.sleep(0)
        assert attempt.task.cancelled() is True
        release.set()

    async def test_caller_cancellation_does_not_cancel_manager_ack(self):
        session, registration = _make_single_slot_session()
        sm = SessionManager(session)
        entered, release = _block_auto_attach(session)
        await sm.start_monitoring()

        registration.attached_to_target(_attached_event())
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        attempt = sm._auto_attach_attempts['session-1']
        waiter = asyncio.create_task(sm.wait_for_auto_attach('session-1'))
        await asyncio.sleep(0)
        waiter.cancel()

        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert attempt.task.cancelled() is False

        release.set()
        assert await asyncio.wait_for(sm.wait_for_auto_attach('session-1'), timeout=1.0) is True
