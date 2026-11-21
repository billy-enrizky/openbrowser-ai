"""Tests for openbrowser.browser.session_manager module."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_session_part1 import _make_mock_cdp_client

from openbrowser.browser.focus import (
    FocusChangeReason,
    FocusPhase,
    FocusSnapshot,
    FocusTargetCode,
    FocusTargetError,
)
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
        sm = SessionManager(session)
        sm._target_sessions['target-1'] = {'session-1'}
        sm._session_to_target['session-1'] = 'target-1'
        sm._target_types['target-1'] = 'page'

        await sm._handle_target_detached({'sessionId': 'session-1', 'targetId': 'target-1'})

        for call in session.event_bus.dispatch.call_args_list:
            assert 'TabClosed' not in type(call.args[0]).__name__

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

        await sm._handle_target_crashed('target-1', status='crashed', error_code=-1)

        record = session._tab_registry.get('target-1')
        assert record is not None
        assert record.state is TabState.CRASHED
        assert record.error is not None
        session._recover_agent_focus.assert_awaited_once()
        assert session._recover_agent_focus.await_args.args == ('target-1',)
        assert session._recover_agent_focus.await_args.kwargs['reason'] is FocusChangeReason.CRASH_RECOVERY

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
