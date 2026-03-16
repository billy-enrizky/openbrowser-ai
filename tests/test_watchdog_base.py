"""Tests for openbrowser.browser.watchdog_base module."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bubus import BaseEvent, EventBus

from openbrowser.browser.watchdog_base import BaseWatchdog

logger = logging.getLogger(__name__)


# --- Test event classes (avoid Test prefix to prevent pytest collection) ---

class SampleEventA(BaseEvent):
    """Test event for handler registration."""
    pass


class SampleEventB(BaseEvent):
    """Another test event."""
    pass


# --- Mock browser session ---

def _make_mock_browser_session():
    """Create a mock BrowserSession."""
    session = MagicMock()
    session.logger = logging.getLogger('test_watchdog_base')
    session.event_bus = EventBus()
    session._cdp_client_root = MagicMock()
    session.agent_focus = None
    session.get_or_create_cdp_session = AsyncMock()
    return session


# --- Concrete watchdog for testing ---

def _make_concrete_watchdog(session=None, event_bus=None):
    """Create a concrete watchdog using model_construct to bypass Pydantic validation."""
    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = EventBus()

    class ConcreteWatchdog(BaseWatchdog):
        LISTENS_TO = []
        EMITS = []

        async def on_SampleEventA(self, event: SampleEventA) -> None:
            pass

    return ConcreteWatchdog.model_construct(event_bus=event_bus, browser_session=session)


def _make_watchdog_with_listens_to(session=None, event_bus=None):
    """Create a watchdog with LISTENS_TO declared."""
    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = EventBus()

    class WatchdogWithListensTo(BaseWatchdog):
        LISTENS_TO = [SampleEventA]
        EMITS = []

        async def on_SampleEventA(self, event: SampleEventA) -> None:
            pass

    return WatchdogWithListensTo.model_construct(event_bus=event_bus, browser_session=session)


def _make_watchdog_with_missing_handler(session=None, event_bus=None):
    """Create a watchdog that declares LISTENS_TO but missing a handler."""
    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = EventBus()

    class WatchdogMissingHandler(BaseWatchdog):
        LISTENS_TO = [SampleEventA, SampleEventB]
        EMITS = []

        async def on_SampleEventA(self, event: SampleEventA) -> None:
            pass
        # Missing on_SampleEventB

    return WatchdogMissingHandler.model_construct(event_bus=event_bus, browser_session=session)


class TestBaseWatchdogInit:
    """Tests for BaseWatchdog initialization."""

    def test_init(self):
        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = _make_concrete_watchdog(session, event_bus)
        assert watchdog.event_bus is event_bus
        assert watchdog.browser_session is session

    def test_logger_property(self):
        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = _make_concrete_watchdog(session, event_bus)
        assert watchdog.logger is session.logger


class TestAttachHandlerToSession:
    """Tests for BaseWatchdog.attach_handler_to_session static method."""

    def test_attach_valid_handler(self):
        session = _make_mock_browser_session()
        watchdog = _make_concrete_watchdog(session, session.event_bus)

        handler = watchdog.on_SampleEventA

        BaseWatchdog.attach_handler_to_session(session, SampleEventA, handler)

        # Verify handler was registered
        handlers = session.event_bus.handlers.get('SampleEventA', [])
        assert len(handlers) == 1

    def test_handler_name_must_start_with_on(self):
        session = _make_mock_browser_session()

        handler = MagicMock()
        handler.__name__ = 'handle_SampleEventA'

        with pytest.raises(AssertionError, match='must start with "on_"'):
            BaseWatchdog.attach_handler_to_session(session, SampleEventA, handler)

    def test_handler_name_must_end_with_event_type(self):
        session = _make_mock_browser_session()

        handler = MagicMock()
        handler.__name__ = 'on_WrongEventName'

        with pytest.raises(AssertionError, match='must end with event type'):
            BaseWatchdog.attach_handler_to_session(session, SampleEventA, handler)

    def test_duplicate_handler_raises(self):
        session = _make_mock_browser_session()
        watchdog = _make_concrete_watchdog(session, session.event_bus)

        handler = watchdog.on_SampleEventA
        BaseWatchdog.attach_handler_to_session(session, SampleEventA, handler)

        with pytest.raises(RuntimeError, match='Duplicate handler'):
            BaseWatchdog.attach_handler_to_session(session, SampleEventA, handler)


class TestAttachToSession:
    """Tests for BaseWatchdog.attach_to_session."""

    def test_attach_registers_handlers(self):
        session = _make_mock_browser_session()
        watchdog = _make_concrete_watchdog(session, session.event_bus)

        # Add SampleEventA to the events module temporarily
        import openbrowser.browser.events as events_module
        original = getattr(events_module, 'SampleEventA', None)
        setattr(events_module, 'SampleEventA', SampleEventA)
        try:
            watchdog.attach_to_session()
            handlers = session.event_bus.handlers.get('SampleEventA', [])
            assert len(handlers) == 1
        finally:
            if original is None:
                delattr(events_module, 'SampleEventA')
            else:
                setattr(events_module, 'SampleEventA', original)

    def test_attach_warns_missing_handlers(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog_with_missing_handler(session, session.event_bus)

        import openbrowser.browser.events as events_module
        setattr(events_module, 'SampleEventA', SampleEventA)
        setattr(events_module, 'SampleEventB', SampleEventB)

        try:
            # attach_to_session should log a warning about missing on_SampleEventB handler
            # but still register the handlers that exist
            watchdog.attach_to_session()
            handlers = session.event_bus.handlers.get('SampleEventA', [])
            assert len(handlers) == 1
        finally:
            delattr(events_module, 'SampleEventA')
            delattr(events_module, 'SampleEventB')


class TestBaseWatchdogDel:
    """Tests for BaseWatchdog.__del__."""

    def test_del_cancels_running_tasks(self):
        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = _make_concrete_watchdog(session, event_bus)

        # Simulate a running task attribute
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()

        object.__setattr__(watchdog, '_some_background_task', mock_task)

        watchdog.__del__()
        mock_task.cancel.assert_called_once()

    def test_del_cancels_tasks_collection(self):
        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = _make_concrete_watchdog(session, event_bus)

        mock_task1 = MagicMock()
        mock_task1.done.return_value = False
        mock_task1.cancel = MagicMock()

        mock_task2 = MagicMock()
        mock_task2.done.return_value = True  # Already done

        object.__setattr__(watchdog, '_background_tasks', [mock_task1, mock_task2])

        watchdog.__del__()
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_not_called()

    def test_del_handles_errors_gracefully(self):
        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = _make_concrete_watchdog(session, event_bus)

        mock_task = MagicMock()
        mock_task.done.side_effect = RuntimeError('boom')

        object.__setattr__(watchdog, '_broken_task', mock_task)

        # Should not raise
        watchdog.__del__()
