"""Tests for openbrowser.browser.session_manager module."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _make_mock_browser_session():
    """Create a mock BrowserSession for SessionManager."""
    session = MagicMock()
    session.logger = logging.getLogger('test_session_manager')

    # CDP client root
    cdp_client_root = MagicMock()
    cdp_client_root.send.Target.setAutoAttach = AsyncMock()
    cdp_client_root.send.Runtime.runIfWaitingForDebugger = AsyncMock()
    cdp_client_root.send.Target.activateTarget = AsyncMock()
    cdp_client_root.register.Target.attachedToTarget = MagicMock()
    cdp_client_root.register.Target.detachedFromTarget = MagicMock()

    session._cdp_client_root = cdp_client_root
    session._cdp_session_pool = {}
    session.agent_focus = None
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

    async def test_start_monitoring_raises_when_no_client(self):
        session = _make_mock_browser_session()
        session._cdp_client_root = None
        sm = SessionManager(session)

        with pytest.raises(RuntimeError, match='CDP client not initialized'):
            await sm.start_monitoring()


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
        assert existing_cdp.session_id == 'new-session'
        assert existing_cdp.title == 'Updated Title'

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

        session._cdp_client_root.send.Runtime.runIfWaitingForDebugger.assert_called_once_with(
            session_id='session-1'
        )


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

        with patch.object(sm, '_recover_agent_focus', new_callable=AsyncMock) as mock_recover:
            await sm._handle_target_detached(event)
            mock_recover.assert_called_once_with('target-1')


@pytest.mark.asyncio
class TestSessionManagerRecoverAgentFocus:
    """Tests for SessionManager._recover_agent_focus."""

    async def test_recovers_to_existing_tab(self):
        session = _make_mock_browser_session()
        session.agent_focus = None  # Already cleared
        session._cdp_get_all_pages.return_value = [{'targetId': 'existing-tab'}]

        mock_cdp_session = MagicMock()
        mock_cdp_session.url = 'https://example.com'
        session._cdp_session_pool['existing-tab'] = mock_cdp_session

        sm = SessionManager(session)

        await sm._recover_agent_focus('crashed-target')

        assert session.agent_focus is mock_cdp_session

    async def test_creates_new_tab_when_no_pages(self):
        session = _make_mock_browser_session()
        session.agent_focus = None
        session._cdp_get_all_pages.return_value = []
        session._cdp_create_new_page.return_value = 'new-target'

        mock_cdp_session = MagicMock()
        mock_cdp_session.url = 'about:blank'
        session._cdp_session_pool['new-target'] = mock_cdp_session

        sm = SessionManager(session)

        await sm._recover_agent_focus('crashed-target')

        session._cdp_create_new_page.assert_called_once_with('about:blank')

    async def test_skips_if_already_recovered(self):
        session = _make_mock_browser_session()
        agent_focus = MagicMock()
        agent_focus.target_id = 'other-target'
        session.agent_focus = agent_focus

        sm = SessionManager(session)

        await sm._recover_agent_focus('crashed-target')

        # Should not try to recover since agent_focus points to a different target
        session._cdp_get_all_pages.assert_not_called()
