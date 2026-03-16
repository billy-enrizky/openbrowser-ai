"""Tests for openbrowser.browser.watchdogs.popups_watchdog module -- 100% coverage target."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_browser_session():
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test_popups_watchdog")
    session.event_bus = MagicMock()
    session._cdp_client_root = MagicMock()
    session._cdp_client_root.register = MagicMock()
    session._cdp_client_root.register.Page = MagicMock()
    session._cdp_client_root.register.Page.javascriptDialogOpening = MagicMock()
    session._cdp_client_root.send = MagicMock()
    session._cdp_client_root.send.Page = MagicMock()
    session._cdp_client_root.send.Page.enable = AsyncMock()
    session._cdp_client_root.send.Page.handleJavaScriptDialog = AsyncMock()
    session.agent_focus = MagicMock()
    session.agent_focus.session_id = "agent-sess-1234"
    session.get_or_create_cdp_session = AsyncMock()
    session.cdp_client = MagicMock()
    session._closed_popup_messages = []
    session.id = "test-popups-session"
    session.is_local = True
    return session


def _make_popups_watchdog(session=None, event_bus=None):
    from openbrowser.browser.watchdogs.popups_watchdog import PopupsWatchdog

    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = MagicMock()
    session.event_bus = event_bus
    return PopupsWatchdog.model_construct(
        event_bus=event_bus,
        browser_session=session,
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestPopupsWatchdogInit:
    def test_init_with_model_construct(self):
        watchdog = _make_popups_watchdog()
        assert hasattr(watchdog, "_dialog_listeners_registered")


# ---------------------------------------------------------------------------
# on_TabCreatedEvent
# ---------------------------------------------------------------------------

class TestOnTabCreatedEvent:
    @pytest.mark.asyncio
    async def test_skips_already_registered(self):
        watchdog = _make_popups_watchdog()
        target_id = "target-abc"
        watchdog._dialog_listeners_registered = {target_id}

        event = MagicMock()
        event.target_id = target_id
        await watchdog.on_TabCreatedEvent(event)
        # Should return early

    @pytest.mark.asyncio
    async def test_registers_dialog_handler(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-popup-1"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-new"

        await watchdog.on_TabCreatedEvent(event)

        assert "target-new" in watchdog._dialog_listeners_registered
        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening.assert_called_once()

    @pytest.mark.asyncio
    async def test_page_enable_fails_gracefully(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock(side_effect=RuntimeError("Page enable fail"))
        cdp_session_mock.session_id = "sess-popup-fail"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session._cdp_client_root.send.Page.enable = AsyncMock()

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-fail"

        # Should not raise
        await watchdog.on_TabCreatedEvent(event)
        assert "target-fail" in watchdog._dialog_listeners_registered

    @pytest.mark.asyncio
    async def test_root_page_enable_fails_gracefully(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-popup-root-fail"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session._cdp_client_root.send.Page.enable = AsyncMock(side_effect=RuntimeError("root enable fail"))

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-root-fail"

        await watchdog.on_TabCreatedEvent(event)
        assert "target-root-fail" in watchdog._dialog_listeners_registered

    @pytest.mark.asyncio
    async def test_no_root_cdp_client(self):
        session = _make_mock_browser_session()
        session._cdp_client_root = None
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-no-root"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-no-root"

        await watchdog.on_TabCreatedEvent(event)
        assert "target-no-root" in watchdog._dialog_listeners_registered

    @pytest.mark.asyncio
    async def test_root_register_fails_gracefully(self):
        session = _make_mock_browser_session()
        # Make root register raise
        session._cdp_client_root.register.Page.javascriptDialogOpening = MagicMock(
            side_effect=RuntimeError("register fail")
        )
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-reg-fail"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-reg-fail"

        await watchdog.on_TabCreatedEvent(event)
        assert "target-reg-fail" in watchdog._dialog_listeners_registered

    @pytest.mark.asyncio
    async def test_outer_exception(self):
        session = _make_mock_browser_session()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("CDP fail"))

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-outer-fail"

        # Should not raise
        await watchdog.on_TabCreatedEvent(event)
        assert "target-outer-fail" not in watchdog._dialog_listeners_registered


# ---------------------------------------------------------------------------
# handle_dialog (inner async function)
# ---------------------------------------------------------------------------

class TestHandleDialog:
    """Test the handle_dialog closure registered in on_TabCreatedEvent."""

    @pytest.mark.asyncio
    async def test_alert_dialog_accepted(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-dialog"

        # Capture the handle_dialog callback
        captured_handler = None

        def capture_handler(handler):
            nonlocal captured_handler
            captured_handler = handler

        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = capture_handler
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session._cdp_client_root.send.Page.handleJavaScriptDialog = AsyncMock()

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-dialog"

        await watchdog.on_TabCreatedEvent(event)
        assert captured_handler is not None

        # Simulate an alert dialog
        event_data = {"type": "alert", "message": "Hello World"}
        await captured_handler(event_data, "session-x")
        assert "[alert] Hello World" in session._closed_popup_messages

    @pytest.mark.asyncio
    async def test_prompt_dialog_dismissed(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-prompt"

        captured_handler = None

        def capture_handler(handler):
            nonlocal captured_handler
            captured_handler = handler

        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = capture_handler
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session._cdp_client_root.send.Page.handleJavaScriptDialog = AsyncMock()

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-prompt"

        await watchdog.on_TabCreatedEvent(event)

        # Simulate a prompt dialog (should be dismissed, not accepted)
        event_data = {"type": "prompt", "message": "Enter name"}
        await captured_handler(event_data, "session-y")

    @pytest.mark.asyncio
    async def test_dialog_approach1_fails_approach2_succeeds(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-fallback"

        captured_handler = None

        def capture_handler(handler):
            nonlocal captured_handler
            captured_handler = handler

        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = capture_handler
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        # Approach 1 fails
        call_count = 0

        async def mock_handle_dialog(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("approach 1 fail")
            # approach 2 succeeds

        session._cdp_client_root.send.Page.handleJavaScriptDialog = mock_handle_dialog

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-fallback"

        await watchdog.on_TabCreatedEvent(event)

        event_data = {"type": "confirm", "message": "Are you sure?"}
        await captured_handler(event_data, "session-z")

    @pytest.mark.asyncio
    async def test_dialog_both_approaches_fail(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-both-fail"

        captured_handler = None

        def capture_handler(handler):
            nonlocal captured_handler
            captured_handler = handler

        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = capture_handler
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        async def always_fail(**kwargs):
            raise RuntimeError("always fail")

        session._cdp_client_root.send.Page.handleJavaScriptDialog = always_fail

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-both-fail"

        await watchdog.on_TabCreatedEvent(event)

        event_data = {"type": "alert", "message": "Error!"}
        # Should not raise
        await captured_handler(event_data, "session-w")

    @pytest.mark.asyncio
    async def test_dialog_critical_error(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-crit"

        captured_handler = None

        def capture_handler(handler):
            nonlocal captured_handler
            captured_handler = handler

        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = capture_handler
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        # Make message access raise
        session._cdp_client_root = None

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-crit"

        await watchdog.on_TabCreatedEvent(event)

        event_data = {"type": "alert", "message": "test"}
        # handler catches the outer exception since _cdp_client_root is None
        await captured_handler(event_data, None)

    @pytest.mark.asyncio
    async def test_dialog_empty_message(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-empty"

        captured_handler = None

        def capture_handler(handler):
            nonlocal captured_handler
            captured_handler = handler

        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = capture_handler
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session._cdp_client_root.send.Page.handleJavaScriptDialog = AsyncMock()

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-empty"

        await watchdog.on_TabCreatedEvent(event)

        # Empty message should not be stored
        event_data = {"type": "alert", "message": ""}
        await captured_handler(event_data, "session-e")
        # Empty string = falsy, so should not be appended
        assert len([m for m in session._closed_popup_messages if "[alert]" in str(m)]) == 0

    @pytest.mark.asyncio
    async def test_dialog_no_session_id_no_root(self):
        """Test approach 1 skipped (no session_id), approach 2 with agent_focus."""
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-no-sid"

        captured_handler = None

        def capture_handler(handler):
            nonlocal captured_handler
            captured_handler = handler

        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = capture_handler
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session._cdp_client_root.send.Page.handleJavaScriptDialog = AsyncMock()

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-no-sid"

        await watchdog.on_TabCreatedEvent(event)

        # session_id=None: approach 1 skipped, approach 2 uses agent_focus
        event_data = {"type": "beforeunload", "message": "Leave page?"}
        await captured_handler(event_data, None)

    @pytest.mark.asyncio
    async def test_dialog_no_agent_focus(self):
        session = _make_mock_browser_session()
        session.agent_focus = None
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Page = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.enable = AsyncMock()
        cdp_session_mock.session_id = "sess-no-focus"

        captured_handler = None

        def capture_handler(handler):
            nonlocal captured_handler
            captured_handler = handler

        cdp_session_mock.cdp_client.register.Page.javascriptDialogOpening = capture_handler
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session._cdp_client_root = None

        watchdog = _make_popups_watchdog(session=session)
        event = MagicMock()
        event.target_id = "target-no-focus"

        await watchdog.on_TabCreatedEvent(event)

        event_data = {"type": "alert", "message": "test"}
        # Both approaches skip/fail since no root and no agent_focus
        await captured_handler(event_data, None)
