"""Comprehensive tests for openbrowser.browser.session module.

Covers CDPSession, BrowserSession init/config, CDP connection, page navigation,
DOM interaction, screenshot methods, tab management, lifecycle methods, and
error handling paths.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from cdp_use import CDPClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _setup_cdp_client_mock(client):
    """Set up all CDP domain methods on a mock CDPClient."""
    # Target domain
    client.send.Target.attachToTarget = AsyncMock(return_value={"sessionId": "sess-abc123"})
    client.send.Target.getTargetInfo = AsyncMock(
        return_value={"targetInfo": {"targetId": "target-001", "url": "https://example.com", "title": "Example", "type": "page"}}
    )
    client.send.Target.getTargets = AsyncMock(
        return_value={"targetInfos": [{"targetId": "target-001", "url": "https://example.com", "title": "Example", "type": "page"}]}
    )
    client.send.Target.setAutoAttach = AsyncMock()
    client.send.Target.activateTarget = AsyncMock()
    client.send.Target.createTarget = AsyncMock(return_value={"targetId": "target-new"})
    client.send.Target.closeTarget = AsyncMock()
    # Page domain
    client.send.Page.enable = AsyncMock()
    client.send.Page.navigate = AsyncMock()
    client.send.Page.captureScreenshot = AsyncMock(return_value={"data": "aGVsbG8="})
    client.send.Page.getLayoutMetrics = AsyncMock(return_value={})
    client.send.Page.getFrameTree = AsyncMock(return_value={"frameTree": {"frame": {"id": "frame-1", "url": "https://example.com"}}})
    client.send.Page.addScriptToEvaluateOnNewDocument = AsyncMock(return_value={"identifier": "script-1"})
    client.send.Page.removeScriptToEvaluateOnNewDocument = AsyncMock()
    # DOM domain
    client.send.DOM.enable = AsyncMock()
    client.send.DOM.getDocument = AsyncMock(return_value={"root": {"nodeId": 1}})
    client.send.DOM.querySelector = AsyncMock(return_value={"nodeId": 42})
    client.send.DOM.getBoxModel = AsyncMock(
        return_value={"model": {"content": [10, 20, 110, 20, 110, 70, 10, 70]}}
    )
    client.send.DOM.getContentQuads = AsyncMock(return_value={"quads": [[10, 20, 110, 20, 110, 70, 10, 70]]})
    client.send.DOM.resolveNode = AsyncMock(return_value={"object": {"objectId": "obj-1"}})
    client.send.DOM.getFrameOwner = AsyncMock(return_value={"backendNodeId": 99, "nodeId": 42})
    # DOMSnapshot domain
    client.send.DOMSnapshot.enable = AsyncMock()
    # Accessibility domain
    client.send.Accessibility.enable = AsyncMock()
    # Runtime domain
    client.send.Runtime.enable = AsyncMock()
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})
    client.send.Runtime.runIfWaitingForDebugger = AsyncMock()
    client.send.Runtime.callFunctionOn = AsyncMock(
        return_value={"result": {"value": {"x": 10, "y": 20, "width": 100, "height": 50}}}
    )
    # Inspector domain
    client.send.Inspector.enable = AsyncMock()
    # Debugger domain
    client.send.Debugger.setSkipAllPauses = AsyncMock()
    # Emulation domain
    client.send.Emulation.setDeviceMetricsOverride = AsyncMock()
    client.send.Emulation.setGeolocationOverride = AsyncMock()
    client.send.Emulation.clearGeolocationOverride = AsyncMock()
    # Network domain
    client.send.Network.getCookies = AsyncMock(return_value={"cookies": []})
    client.send.Network.clearBrowserCookies = AsyncMock()
    # Storage domain
    client.send.Storage.getCookies = AsyncMock(return_value={"cookies": []})
    client.send.Storage.setCookies = AsyncMock()
    client.send.Storage.clearCookies = AsyncMock()
    # DOMStorage domain
    client.send.DOMStorage.enable = AsyncMock()
    client.send.DOMStorage.disable = AsyncMock()
    client.send.DOMStorage.getDOMStorageItems = AsyncMock(return_value={"entries": []})
    # Fetch domain
    client.send.Fetch.enable = AsyncMock()
    client.send.Fetch.continueWithAuth = AsyncMock()
    client.send.Fetch.continueRequest = AsyncMock()
    # Register domain
    client.register.Target.attachedToTarget = MagicMock()
    client.register.Target.detachedFromTarget = MagicMock()
    client.register.Fetch.authRequired = MagicMock()
    client.register.Fetch.requestPaused = MagicMock()
    # start/stop
    client.start = AsyncMock()
    client.stop = AsyncMock()
    return client


def _make_mock_cdp_client():
    """Create a mock CDPClient with all needed CDP domain methods."""
    client = MagicMock()
    return _setup_cdp_client_mock(client)


def _make_mock_cdp_session(target_id="target-001", session_id="sess-abc123", url="https://example.com", title="Example"):
    """Create a mock CDPSession."""
    session = MagicMock()
    session.target_id = target_id
    session.session_id = session_id
    session.url = url
    session.title = title
    session.cdp_client = _make_mock_cdp_client()
    session.get_target_info = AsyncMock(return_value={"targetId": target_id, "url": url, "title": title})
    session.get_tab_info = AsyncMock()
    session.attach = AsyncMock(return_value=session)
    session.disconnect = AsyncMock()
    return session


def _make_mock_session_manager():
    """Create a mock SessionManager."""
    mgr = MagicMock()
    mgr.start_monitoring = AsyncMock()
    mgr.get_session_for_target = AsyncMock(return_value=_make_mock_cdp_session())
    mgr.validate_session = AsyncMock(return_value=True)
    mgr.clear = AsyncMock()
    return mgr


class _AwaitableEvent:
    """An object that can be directly awaited (like bubus Event objects)."""

    def __init__(self):
        self.event_result = AsyncMock(return_value=None)

    def __await__(self):
        return asyncio.sleep(0).__await__()


def _make_awaitable_event():
    """Create an awaitable mock event for patching EventBus.dispatch return values."""
    return _AwaitableEvent()


def _setattr(obj, name, value):
    """Bypass Pydantic's __setattr__ validation to set arbitrary attributes on a model."""
    object.__setattr__(obj, name, value)


def _make_browser_session(**kwargs):
    """Create a BrowserSession with event handler registration mocked out."""
    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.watchdog_base import BaseWatchdog

    defaults = {
        "cdp_url": "ws://localhost:9222/devtools/browser/abc",
        "headless": True,
    }
    defaults.update(kwargs)

    with patch.object(BaseWatchdog, "attach_handler_to_session"):
        session = BrowserSession(**defaults)

    return session


# ---------------------------------------------------------------------------
# CDPSession tests (lines 56-153)
# ---------------------------------------------------------------------------


class _FakeCDPClient(CDPClient):
    """Fake CDPClient that passes isinstance checks without connecting."""

    def __init__(self):
        # Do NOT call super().__init__ - avoid actual WebSocket setup
        self.send = MagicMock()
        self.register = MagicMock()


def _make_pydantic_cdp_client():
    """Create a CDPClient-compatible mock that passes Pydantic isinstance validation."""
    client = _FakeCDPClient()
    _setup_cdp_client_mock(client)
    return client


def _make_cdp_session_instance(target_id="t-1", session_id="s-1"):
    """Create a real CDPSession instance using Pydantic-compatible mock."""
    from openbrowser.browser.session import CDPSession

    client = _make_pydantic_cdp_client()
    sess = CDPSession(cdp_client=client, target_id=target_id, session_id=session_id)
    return sess


class TestCDPSession:
    """Tests for CDPSession class."""

    def test_cdp_session_model(self):
        sess = _make_cdp_session_instance()
        assert sess.target_id == "t-1"
        assert sess.session_id == "s-1"
        assert sess.title == "Unknown title"
        assert sess.url == "about:blank"

    @pytest.mark.asyncio
    async def test_for_target(self):
        """Test CDPSession.for_target class method -- lines 86-91."""
        from openbrowser.browser.session import CDPSession

        client = _make_pydantic_cdp_client()
        session = await CDPSession.for_target(client, "target-001")
        assert session.session_id == "sess-abc123"
        assert session.title == "Example"

    @pytest.mark.asyncio
    async def test_for_target_custom_domains(self):
        """Test CDPSession.for_target with custom domains."""
        from openbrowser.browser.session import CDPSession

        client = _make_pydantic_cdp_client()
        session = await CDPSession.for_target(client, "target-001", domains=["Page", "Runtime"])
        assert session.session_id == "sess-abc123"

    @pytest.mark.asyncio
    async def test_attach_enables_domains(self):
        """Test CDPSession.attach enables all CDP domains -- lines 94-135."""
        sess = _make_cdp_session_instance(target_id="t-1", session_id="connecting")
        result = await sess.attach()
        assert result.session_id == "sess-abc123"
        sess.cdp_client.send.Page.enable.assert_called()

    @pytest.mark.asyncio
    async def test_attach_domain_failure(self):
        """Test CDPSession.attach raises when domain enable fails -- line 119."""
        sess = _make_cdp_session_instance(target_id="t-1", session_id="connecting")
        sess.cdp_client.send.Page.enable = AsyncMock(side_effect=RuntimeError("domain error"))
        with pytest.raises(RuntimeError, match="Failed to enable"):
            await sess.attach()

    @pytest.mark.asyncio
    async def test_attach_debugger_skip_pauses_failure(self):
        """Test CDPSession.attach handles Debugger.setSkipAllPauses failure -- lines 123-130."""
        sess = _make_cdp_session_instance(target_id="t-1", session_id="connecting")
        sess.cdp_client.send.Debugger.setSkipAllPauses = AsyncMock(side_effect=Exception("no debugger"))
        result = await sess.attach()
        assert result.session_id == "sess-abc123"

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test CDPSession.disconnect -- line 141."""
        sess = _make_cdp_session_instance()
        await sess.disconnect()  # No-op, should not raise

    @pytest.mark.asyncio
    async def test_get_tab_info(self):
        """Test CDPSession.get_tab_info -- lines 143-149."""
        sess = _make_cdp_session_instance(target_id="target-001")
        tab = await sess.get_tab_info()
        assert tab.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_get_target_info(self):
        """Test CDPSession.get_target_info -- lines 151-153."""
        sess = _make_cdp_session_instance(target_id="target-001")
        info = await sess.get_target_info()
        assert info["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# BrowserSession __init__ and configuration (lines 156-425)
# ---------------------------------------------------------------------------


class TestBrowserSessionInit:
    """Tests for BrowserSession initialization and configuration."""

    def test_basic_init_with_cdp_url(self):
        from openbrowser.browser.session import BrowserSession

        with patch("openbrowser.browser.session.BrowserSession.model_post_init"):
            session = BrowserSession(cdp_url="ws://localhost:9222/devtools/browser/abc")
        assert session.cdp_url == "ws://localhost:9222/devtools/browser/abc"

    def test_init_with_browser_profile(self):
        from openbrowser.browser.profile import BrowserProfile
        from openbrowser.browser.session import BrowserSession

        profile = BrowserProfile(cdp_url="ws://localhost:1234", headless=True)
        with patch("openbrowser.browser.session.BrowserSession.model_post_init"):
            session = BrowserSession(browser_profile=profile)
        assert session.cdp_url == "ws://localhost:1234"

    def test_init_sets_is_local_when_no_cdp_url(self):
        from openbrowser.browser.session import BrowserSession

        with patch("openbrowser.browser.session.BrowserSession.model_post_init"):
            session = BrowserSession()
        assert session.is_local is True

    def test_init_executable_path_sets_is_local(self):
        from openbrowser.browser.session import BrowserSession

        with patch("openbrowser.browser.session.BrowserSession.model_post_init"):
            session = BrowserSession(executable_path="/usr/bin/chromium")
        assert session.is_local is True

    def test_cdp_url_property(self):
        session = _make_browser_session(cdp_url="ws://test:9222")
        assert session.cdp_url == "ws://test:9222"

    def test_is_local_property(self):
        session = _make_browser_session()
        assert isinstance(session.is_local, bool)


# ---------------------------------------------------------------------------
# BrowserSession properties and helpers (lines 395-425)
# ---------------------------------------------------------------------------


class TestBrowserSessionProperties:
    """Tests for BrowserSession properties."""

    def test_logger_property(self):
        session = _make_browser_session()
        log = session.logger
        assert log is not None

    def test_id_for_logs(self):
        session = _make_browser_session()
        log_id = session._id_for_logs
        assert isinstance(log_id, str)
        assert len(log_id) >= 1

    def test_tab_id_for_logs_no_focus(self):
        session = _make_browser_session()
        session.__dict__["agent_focus"] = None
        result = session._tab_id_for_logs
        assert "--" in result

    def test_tab_id_for_logs_with_focus(self):
        session = _make_browser_session()
        mock_focus = MagicMock()
        mock_focus.target_id = "abcd1234efgh5678"
        session.__dict__["agent_focus"] = mock_focus
        result = session._tab_id_for_logs
        assert result == "78"

    def test_repr(self):
        session = _make_browser_session()
        r = repr(session)
        assert "BrowserSession" in r

    def test_str(self):
        session = _make_browser_session()
        s = str(session)
        assert "BrowserSession" in s

    def test_current_target_id_none(self):
        session = _make_browser_session()
        assert session.current_target_id is None

    def test_current_target_id_with_focus(self):
        session = _make_browser_session()
        mock_focus = MagicMock()
        mock_focus.target_id = "target-xyz"
        session.__dict__["agent_focus"] = mock_focus
        assert session.current_target_id == "target-xyz"

    def test_current_session_id_none(self):
        session = _make_browser_session()
        assert session.current_session_id is None

    def test_current_session_id_with_focus(self):
        session = _make_browser_session()
        mock_focus = MagicMock()
        mock_focus.session_id = "sess-xyz"
        session.__dict__["agent_focus"] = mock_focus
        assert session.current_session_id == "sess-xyz"


# ---------------------------------------------------------------------------
# Reset and lifecycle (lines 426-534)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBrowserSessionLifecycle:
    """Tests for BrowserSession lifecycle methods."""

    async def test_reset(self):
        session = _make_browser_session()
        session._cdp_client_root = MagicMock()
        session._session_manager = MagicMock()
        session._session_manager.clear = AsyncMock()
        session.__dict__["agent_focus"] = MagicMock()
        session._cached_browser_state_summary = "cached"
        session._downloaded_files = ["file1"]

        await session.reset()

        assert session._cdp_client_root is None
        assert session.agent_focus is None
        assert session._cached_browser_state_summary is None
        assert len(session._downloaded_files) == 0

    async def test_kill(self):
        session = _make_browser_session()
        session._cdp_client_root = MagicMock()
        session._session_manager = MagicMock()
        session._session_manager.clear = AsyncMock()
        # dispatch() returns an event that is directly awaitable via __await__
        awaitable_event = _make_awaitable_event()
        with patch.object(session.event_bus, "dispatch", return_value=awaitable_event) as mock_dispatch, \
             patch.object(session.event_bus, "stop", new_callable=AsyncMock) as mock_stop:
            await session.kill()
            mock_stop.assert_called()

    async def test_stop(self):
        session = _make_browser_session()
        session._cdp_client_root = MagicMock()
        session._session_manager = MagicMock()
        session._session_manager.clear = AsyncMock()
        awaitable_event = _make_awaitable_event()
        with patch.object(session.event_bus, "dispatch", return_value=awaitable_event) as mock_dispatch, \
             patch.object(session.event_bus, "stop", new_callable=AsyncMock) as mock_stop:
            await session.stop()
            mock_stop.assert_called()

    async def test_cdp_client_property_raises_without_init(self):
        session = _make_browser_session()
        session._cdp_client_root = None
        with pytest.raises(AssertionError):
            _ = session.cdp_client

    async def test_cdp_client_property_returns_client(self):
        session = _make_browser_session()
        mock_client = _make_mock_cdp_client()
        session._cdp_client_root = mock_client
        assert session.cdp_client is mock_client


# ---------------------------------------------------------------------------
# on_BrowserStartEvent (lines 536-588)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnBrowserStartEvent:
    """Tests for BrowserSession.on_BrowserStartEvent."""

    async def test_start_with_cdp_url(self):
        session = _make_browser_session()
        session.__dict__["attach_all_watchdogs"] = AsyncMock()

        async def mock_connect(**kwargs):
            session._cdp_client_root = _make_mock_cdp_client()
            return session

        _setattr(session, "connect", AsyncMock(side_effect=mock_connect))
        session._cdp_client_root = None

        from openbrowser.browser.events import BrowserStartEvent

        with patch.object(session.event_bus, "dispatch", return_value=_make_awaitable_event()):
            result = await session.on_BrowserStartEvent(BrowserStartEvent())
        assert "cdp_url" in result
        session.connect.assert_called_once()

    async def test_start_already_connected(self):
        session = _make_browser_session()
        session.__dict__["attach_all_watchdogs"] = AsyncMock()
        session._cdp_client_root = _make_mock_cdp_client()

        from openbrowser.browser.events import BrowserStartEvent

        with patch.object(session.event_bus, "dispatch", return_value=_make_awaitable_event()):
            result = await session.on_BrowserStartEvent(BrowserStartEvent())
        assert "cdp_url" in result

    async def test_start_no_cdp_url_local(self):
        session = _make_browser_session()
        session.browser_profile.cdp_url = None
        session.browser_profile.is_local = True
        session.__dict__["attach_all_watchdogs"] = AsyncMock()

        from openbrowser.browser.events import BrowserLaunchResult

        mock_dispatched = _AwaitableEvent()
        mock_dispatched.event_result = AsyncMock(return_value=BrowserLaunchResult(cdp_url="ws://localhost:9333"))

        async def mock_connect(**kwargs):
            session._cdp_client_root = _make_mock_cdp_client()
            return session

        _setattr(session, "connect", AsyncMock(side_effect=mock_connect))
        session._cdp_client_root = None

        from openbrowser.browser.events import BrowserStartEvent

        with patch.object(session.event_bus, "dispatch", return_value=mock_dispatched):
            result = await session.on_BrowserStartEvent(BrowserStartEvent())
        assert result["cdp_url"] == "ws://localhost:9333"

    async def test_start_no_cdp_url_not_local_raises(self):
        session = _make_browser_session()
        session.browser_profile.cdp_url = None
        session.browser_profile.is_local = False
        session.__dict__["attach_all_watchdogs"] = AsyncMock()

        from openbrowser.browser.events import BrowserStartEvent

        with patch.object(session.event_bus, "dispatch", return_value=_make_awaitable_event()):
            with pytest.raises(ValueError, match="no cdp_url"):
                await session.on_BrowserStartEvent(BrowserStartEvent())

    async def test_start_exception_dispatches_error(self):
        session = _make_browser_session()
        session.__dict__["attach_all_watchdogs"] = AsyncMock()
        _setattr(session, "connect", AsyncMock(side_effect=RuntimeError("connection failed")))
        session._cdp_client_root = None

        from openbrowser.browser.events import BrowserStartEvent

        with patch.object(session.event_bus, "dispatch", return_value=_make_awaitable_event()):
            with pytest.raises(RuntimeError, match="connection failed"):
                await session.on_BrowserStartEvent(BrowserStartEvent())


# ---------------------------------------------------------------------------
# on_BrowserStopEvent (lines 884-912)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnBrowserStopEvent:
    """Tests for BrowserSession.on_BrowserStopEvent."""

    async def test_stop_with_keep_alive(self):
        session = _make_browser_session()
        session.browser_profile.keep_alive = True

        from openbrowser.browser.events import BrowserStopEvent

        with patch.object(session.event_bus, "dispatch", return_value=_make_awaitable_event()):
            await session.on_BrowserStopEvent(BrowserStopEvent(force=False))

    async def test_stop_forced(self):
        session = _make_browser_session()
        session.browser_profile.keep_alive = True
        _setattr(session, "reset", AsyncMock())

        from openbrowser.browser.events import BrowserStopEvent

        with patch.object(session.event_bus, "dispatch", return_value=_make_awaitable_event()):
            await session.on_BrowserStopEvent(BrowserStopEvent(force=True))
        session.reset.assert_called()

    async def test_stop_exception_dispatches_error(self):
        session = _make_browser_session()
        _setattr(session, "reset", AsyncMock(side_effect=RuntimeError("reset failed")))

        from openbrowser.browser.events import BrowserStopEvent

        with patch.object(session.event_bus, "dispatch", return_value=_make_awaitable_event()):
            await session.on_BrowserStopEvent(BrowserStopEvent(force=True))


# ---------------------------------------------------------------------------
# on_FileDownloadedEvent (lines 872-882)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnFileDownloadedEvent:
    """Tests for BrowserSession.on_FileDownloadedEvent."""

    async def test_new_download_tracked(self):
        session = _make_browser_session()
        from openbrowser.browser.events import FileDownloadedEvent

        event = FileDownloadedEvent(url="https://example.com/file.pdf", path="/tmp/file.pdf", file_name="file.pdf", file_size=1024)
        await session.on_FileDownloadedEvent(event)
        assert "/tmp/file.pdf" in session._downloaded_files

    async def test_duplicate_download_not_duplicated(self):
        session = _make_browser_session()
        session._downloaded_files = ["/tmp/file.pdf"]

        from openbrowser.browser.events import FileDownloadedEvent

        event = FileDownloadedEvent(url="https://example.com/file.pdf", path="/tmp/file.pdf", file_name="file.pdf", file_size=1024)
        await session.on_FileDownloadedEvent(event)
        assert session._downloaded_files.count("/tmp/file.pdf") == 1

    async def test_download_no_path(self):
        session = _make_browser_session()
        from openbrowser.browser.events import FileDownloadedEvent

        event = FileDownloadedEvent(url="https://example.com/file.pdf", path="", file_name="file.pdf", file_size=0)
        await session.on_FileDownloadedEvent(event)
        assert len(session._downloaded_files) == 0


# ---------------------------------------------------------------------------
# on_TabCreatedEvent / on_TabClosedEvent (lines 780-809)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTabEvents:
    """Tests for tab creation and closure events."""

    async def test_on_tab_created_no_viewport(self):
        session = _make_browser_session()
        session.browser_profile.viewport = None

        from openbrowser.browser.events import TabCreatedEvent

        await session.on_TabCreatedEvent(TabCreatedEvent(target_id="t-1", url="about:blank"))

    async def test_on_tab_created_with_viewport(self):
        from openbrowser.browser.profile import ViewportSize

        session = _make_browser_session()
        session.browser_profile.viewport = ViewportSize(width=1280, height=720)
        session.browser_profile.no_viewport = False
        session.browser_profile.device_scale_factor = 2.0
        session._cdp_set_viewport = AsyncMock()

        from openbrowser.browser.events import TabCreatedEvent

        await session.on_TabCreatedEvent(TabCreatedEvent(target_id="t-1", url="about:blank"))
        session._cdp_set_viewport.assert_called_once()

    async def test_on_tab_created_viewport_exception(self):
        from openbrowser.browser.profile import ViewportSize

        session = _make_browser_session()
        session.browser_profile.viewport = ViewportSize(width=1280, height=720)
        session.browser_profile.no_viewport = False
        session._cdp_set_viewport = AsyncMock(side_effect=Exception("viewport error"))

        from openbrowser.browser.events import TabCreatedEvent

        await session.on_TabCreatedEvent(TabCreatedEvent(target_id="t-1", url="about:blank"))

    async def test_on_tab_closed_no_focus(self):
        session = _make_browser_session()

        from openbrowser.browser.events import TabClosedEvent

        await session.on_TabClosedEvent(TabClosedEvent(target_id="t-1"))

    async def test_on_tab_closed_current_tab(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session(target_id="t-1")
        session.__dict__["agent_focus"] = mock_focus

        session.event_bus.dispatch = MagicMock(return_value=_make_awaitable_event())

        from openbrowser.browser.events import TabClosedEvent

        await session.on_TabClosedEvent(TabClosedEvent(target_id="t-1"))


# ---------------------------------------------------------------------------
# on_AgentFocusChangedEvent (lines 811-870)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnAgentFocusChangedEvent:
    """Tests for on_AgentFocusChangedEvent handler."""

    async def test_focus_changed_success(self):
        session = _make_browser_session()
        real_cdp_session = _make_cdp_session_instance(target_id="target-001", session_id="s-focus")
        session.__dict__["agent_focus"] = real_cdp_session
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        session._dom_watchdog = MagicMock()
        session._dom_watchdog.clear_cache = MagicMock()
        session._cdp_get_all_pages = AsyncMock(return_value=[{"targetId": "target-001", "url": "https://example.com", "type": "page"}])
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=real_cdp_session))

        from openbrowser.browser.events import AgentFocusChangedEvent

        session.event_bus.dispatch = MagicMock(return_value=_make_awaitable_event())

        await session.on_AgentFocusChangedEvent(
            AgentFocusChangedEvent(target_id="target-001", url="https://example.com")
        )

    async def test_focus_changed_no_target_id_raises(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_get_all_pages = AsyncMock(return_value=[])

        from openbrowser.browser.events import AgentFocusChangedEvent

        with pytest.raises(RuntimeError, match="no target_id"):
            await session.on_AgentFocusChangedEvent(
                AgentFocusChangedEvent(target_id="", url="")
            )

    async def test_focus_changed_unresponsive_tab(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        mock_focus.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=Exception("tab crashed"))
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        session._cdp_get_all_pages = AsyncMock(return_value=[{"targetId": "target-002", "url": "about:blank", "type": "page"}])
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        from openbrowser.browser.events import AgentFocusChangedEvent

        session.event_bus.dispatch = MagicMock(return_value=_make_awaitable_event())

        with pytest.raises(Exception):
            await session.on_AgentFocusChangedEvent(
                AgentFocusChangedEvent(target_id="target-001", url="https://example.com")
            )


# ---------------------------------------------------------------------------
# on_SwitchTabEvent (lines 727-763)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnSwitchTabEvent:
    """Tests for on_SwitchTabEvent handler."""

    async def test_switch_tab_no_focus_raises(self):
        session = _make_browser_session()

        from openbrowser.browser.events import SwitchTabEvent

        with pytest.raises(RuntimeError, match="not connected"):
            await session.on_SwitchTabEvent(SwitchTabEvent(target_id="t-1"))

    async def test_switch_tab_specific_target(self):
        session = _make_browser_session()
        real_new = _make_cdp_session_instance(target_id="target-001", session_id="s-new")
        session.__dict__["agent_focus"] = _make_cdp_session_instance(target_id="target-old", session_id="s-old")
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        session._cdp_get_all_pages = AsyncMock(return_value=[{"targetId": "target-001", "url": "https://example.com", "type": "page"}])
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=real_new))

        session.event_bus.dispatch = MagicMock(return_value=_make_awaitable_event())

        from openbrowser.browser.events import SwitchTabEvent

        result = await session.on_SwitchTabEvent(SwitchTabEvent(target_id="target-001"))
        assert result == "target-001"

    async def test_switch_tab_none_uses_last_page(self):
        session = _make_browser_session()
        real_cdp = _make_cdp_session_instance(target_id="target-001", session_id="s-1")
        session.__dict__["agent_focus"] = _make_cdp_session_instance(target_id="target-old", session_id="s-old")
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        session._cdp_get_all_pages = AsyncMock(
            return_value=[{"targetId": "target-001", "url": "https://example.com", "type": "page"}]
        )
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=real_cdp))

        session.event_bus.dispatch = MagicMock(return_value=_make_awaitable_event())

        from openbrowser.browser.events import SwitchTabEvent

        result = await session.on_SwitchTabEvent(SwitchTabEvent(target_id=None))
        assert result is not None

    async def test_switch_tab_none_no_pages_creates_new(self):
        session = _make_browser_session()
        session.__dict__["agent_focus"] = _make_cdp_session_instance(target_id="target-old", session_id="s-old")
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_get_all_pages = AsyncMock(return_value=[])

        from openbrowser.browser.events import SwitchTabEvent

        with patch.object(session.event_bus, "dispatch", return_value=_make_awaitable_event()):
            result = await session.on_SwitchTabEvent(SwitchTabEvent(target_id=None))
        assert result == "target-new"


# ---------------------------------------------------------------------------
# on_CloseTabEvent (lines 765-778)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnCloseTabEvent:
    """Tests for on_CloseTabEvent handler."""

    async def test_close_tab_success(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        session.event_bus.dispatch = MagicMock(return_value=_make_awaitable_event())

        from openbrowser.browser.events import CloseTabEvent

        await session.on_CloseTabEvent(CloseTabEvent(target_id="t-1"))

    async def test_close_tab_target_already_closed(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        mock_focus.cdp_client.send.Target.closeTarget = AsyncMock(side_effect=Exception("already closed"))
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        session.event_bus.dispatch = MagicMock(return_value=_make_awaitable_event())

        from openbrowser.browser.events import CloseTabEvent

        await session.on_CloseTabEvent(CloseTabEvent(target_id="t-1"))


# ---------------------------------------------------------------------------
# CDP page/tab methods (lines 921-996)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCDPPageMethods:
    """Tests for new_page, get_current_page, cookies, etc."""

    async def test_new_page(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()

        with patch("openbrowser.actor.page.Page") as MockPage:
            page = await session.new_page("https://example.com")

    async def test_get_current_page_no_focus(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        _setattr(session, "get_current_target_info", AsyncMock(return_value=None))
        result = await session.get_current_page()
        assert result is None

    async def test_get_current_page_with_target(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        _setattr(session, "get_current_target_info", AsyncMock(
            return_value={"targetId": "t-1", "url": "https://example.com", "type": "page"}
        ))
        with patch("openbrowser.actor.page.Page"):
            result = await session.get_current_page()
            assert result is not None

    async def test_must_get_current_page_raises(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        _setattr(session, "get_current_target_info", AsyncMock(return_value=None))
        with pytest.raises(RuntimeError, match="No current target"):
            await session.must_get_current_page()

    async def test_get_pages(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        with patch("openbrowser.actor.page.Page"):
            pages = await session.get_pages()
            assert len(pages) == 1

    async def test_close_page_by_target(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        # Pass string target_id to close_page
        await session.close_page("t-1")

    async def test_cookies(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        with patch("cdp_use.cdp.network.library.GetCookiesParameters", create=True):
            result = await session.cookies()
        assert result == []

    async def test_cookies_with_urls(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        with patch("cdp_use.cdp.network.library.GetCookiesParameters", create=True):
            result = await session.cookies(urls=["https://example.com"])
        assert result == []

    async def test_clear_cookies(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        await session.clear_cookies()


# ---------------------------------------------------------------------------
# export_storage_state (lines 998-1041)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExportStorageState:
    """Tests for export_storage_state."""

    async def test_export_without_path(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._session_manager = _make_mock_session_manager()
        session._cdp_get_cookies = AsyncMock(return_value=[
            {"name": "c1", "value": "v1", "domain": ".example.com", "path": "/"}
        ])

        result = await session.export_storage_state()
        assert "cookies" in result
        assert len(result["cookies"]) == 1

    async def test_export_with_path(self, tmp_path):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._session_manager = _make_mock_session_manager()
        session._cdp_get_cookies = AsyncMock(return_value=[])

        output = tmp_path / "state.json"
        result = await session.export_storage_state(output_path=str(output))
        assert output.exists()


# ---------------------------------------------------------------------------
# get_or_create_cdp_session (lines 1043-1117)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetOrCreateCdpSession:
    """Tests for get_or_create_cdp_session."""

    async def test_uses_current_focus_when_no_target_id(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session(target_id="t-focus")
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        result = await session.get_or_create_cdp_session()
        session._session_manager.get_session_for_target.assert_called()

    async def test_waits_for_target_to_appear(self):
        session = _make_browser_session()
        session.__dict__["agent_focus"] = _make_cdp_session_instance(target_id="t-focus", session_id="s-focus")
        session._cdp_client_root = _make_mock_cdp_client()

        mgr = MagicMock()
        call_count = 0

        async def delayed_get(target_id):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None
            return _make_cdp_session_instance(target_id=target_id, session_id="s-new")

        mgr.get_session_for_target = AsyncMock(side_effect=delayed_get)
        mgr.validate_session = AsyncMock(return_value=True)
        session._session_manager = mgr

        # Mock getTargets to return page type
        session._cdp_client_root.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": [{"targetId": "t-new", "type": "page"}]}
        )

        result = await session.get_or_create_cdp_session(target_id="t-new")
        assert result is not None

    async def test_timeout_waiting_for_target(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        mgr = MagicMock()
        mgr.get_session_for_target = AsyncMock(return_value=None)
        session._session_manager = mgr

        with pytest.raises(ValueError, match="not found"):
            await session.get_or_create_cdp_session(target_id="t-missing")

    async def test_invalid_session(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        mgr = _make_mock_session_manager()
        mgr.validate_session = AsyncMock(return_value=False)
        session._session_manager = mgr

        with pytest.raises(ValueError, match="detached"):
            await session.get_or_create_cdp_session(target_id="t-1")

    async def test_focus_change_to_page_target(self):
        session = _make_browser_session()
        session.__dict__["agent_focus"] = _make_cdp_session_instance(target_id="t-old", session_id="s-old")
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": [{"targetId": "t-new", "type": "page"}]}
        )
        new_session = _make_cdp_session_instance(target_id="t-new", session_id="s-new")
        mgr = _make_mock_session_manager()
        mgr.get_session_for_target = AsyncMock(return_value=new_session)
        session._session_manager = mgr

        result = await session.get_or_create_cdp_session(target_id="t-new", focus=True)
        assert session.agent_focus is new_session

    async def test_focus_ignored_for_non_page_target(self):
        session = _make_browser_session()
        old_focus = _make_mock_cdp_session(target_id="t-old")
        session.__dict__["agent_focus"] = old_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": [{"targetId": "t-iframe", "type": "iframe"}]}
        )
        iframe_session = _make_mock_cdp_session(target_id="t-iframe")
        mgr = _make_mock_session_manager()
        mgr.get_session_for_target = AsyncMock(return_value=iframe_session)
        session._session_manager = mgr

        result = await session.get_or_create_cdp_session(target_id="t-iframe", focus=True)
        # Should NOT change focus
        assert session.agent_focus is old_focus


# ---------------------------------------------------------------------------
# DOM helper methods (lines 1739-1889)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDOMHelperMethods:
    """Tests for DOM helper methods."""

    async def test_get_dom_element_by_index_cached(self):
        session = _make_browser_session()
        mock_node = MagicMock()
        session._cached_selector_map = {42: mock_node}
        result = await session.get_dom_element_by_index(42)
        assert result is mock_node

    async def test_get_dom_element_by_index_not_found(self):
        session = _make_browser_session()
        session._cached_selector_map = {}
        result = await session.get_dom_element_by_index(99)
        assert result is None

    def test_update_cached_selector_map(self):
        session = _make_browser_session()
        new_map = {1: MagicMock(), 2: MagicMock()}
        session.update_cached_selector_map(new_map)
        assert session._cached_selector_map == new_map

    async def test_get_element_by_index_alias(self):
        session = _make_browser_session()
        mock_node = MagicMock()
        session._cached_selector_map = {10: mock_node}
        result = await session.get_element_by_index(10)
        assert result is mock_node

    async def test_get_target_id_from_tab_id(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": [{"targetId": "abcdef12345678ABCD", "type": "page"}]}
        )
        session._cdp_session_pool = {}

        result = await session.get_target_id_from_tab_id("ABCD")
        assert result == "abcdef12345678ABCD"

    async def test_get_target_id_from_tab_id_not_found(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": []}
        )
        session._cdp_session_pool = {}

        with pytest.raises(ValueError, match="No TargetID found"):
            await session.get_target_id_from_tab_id("ZZZZ")

    async def test_get_target_id_from_tab_id_cached(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargetInfo = AsyncMock(return_value={"targetInfo": {"targetId": "xxx_ABCD"}})
        session._cdp_session_pool = {"xxx_ABCD": _make_mock_cdp_session(target_id="xxx_ABCD")}

        result = await session.get_target_id_from_tab_id("ABCD")
        assert result == "xxx_ABCD"

    async def test_is_target_valid_true(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        result = await session._is_target_valid("t-1")
        assert result is True

    async def test_is_target_valid_false(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargetInfo = AsyncMock(side_effect=Exception("not found"))
        result = await session._is_target_valid("t-invalid")
        assert result is False

    async def test_get_target_id_from_url(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        result = await session.get_target_id_from_url("https://example.com")
        assert result == "target-001"

    async def test_get_target_id_from_url_substring(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": [{"targetId": "t-1", "url": "https://example.com/page/123", "type": "page"}]}
        )
        result = await session.get_target_id_from_url("example.com/page")
        assert result == "t-1"

    async def test_get_target_id_from_url_not_found(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})

        with pytest.raises(ValueError, match="No TargetID found"):
            await session.get_target_id_from_url("https://notfound.com")

    async def test_get_most_recently_opened_target_id(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_get_all_pages = AsyncMock(
            return_value=[{"targetId": "t-old"}, {"targetId": "t-newest"}]
        )
        result = await session.get_most_recently_opened_target_id()
        assert result == "t-newest"

    def test_is_file_input_with_watchdog(self):
        session = _make_browser_session()
        session._dom_watchdog = MagicMock()
        session._dom_watchdog.is_file_input = MagicMock(return_value=True)
        result = session.is_file_input(MagicMock())
        assert result is True

    def test_is_file_input_fallback(self):
        session = _make_browser_session()
        session._dom_watchdog = None
        element = MagicMock()
        element.node_name = "INPUT"
        element.attributes = {"type": "file"}
        result = session.is_file_input(element)
        assert result is True

    def test_is_file_input_not_file(self):
        session = _make_browser_session()
        session._dom_watchdog = None
        element = MagicMock()
        element.node_name = "INPUT"
        element.attributes = {"type": "text"}
        result = session.is_file_input(element)
        assert result is False

    async def test_get_selector_map_cached(self):
        session = _make_browser_session()
        session._cached_selector_map = {1: MagicMock()}
        result = await session.get_selector_map()
        assert len(result) == 1

    async def test_get_selector_map_from_watchdog(self):
        session = _make_browser_session()
        session._cached_selector_map = {}
        session._dom_watchdog = MagicMock()
        session._dom_watchdog.selector_map = {2: MagicMock()}
        result = await session.get_selector_map()
        assert len(result) == 1

    async def test_get_selector_map_empty(self):
        session = _make_browser_session()
        session._cached_selector_map = {}
        session._dom_watchdog = None
        result = await session.get_selector_map()
        assert result == {}

    async def test_get_index_by_id(self):
        session = _make_browser_session()
        mock_node = MagicMock()
        mock_node.attributes = {"id": "search-box"}
        session._cached_selector_map = {5: mock_node}
        result = await session.get_index_by_id("search-box")
        assert result == 5

    async def test_get_index_by_id_not_found(self):
        session = _make_browser_session()
        session._cached_selector_map = {}
        result = await session.get_index_by_id("missing")
        assert result is None

    async def test_get_index_by_class(self):
        session = _make_browser_session()
        mock_node = MagicMock()
        mock_node.attributes = {"class": "btn primary"}
        session._cached_selector_map = {3: mock_node}
        result = await session.get_index_by_class("primary")
        assert result == 3

    async def test_get_index_by_class_not_found(self):
        session = _make_browser_session()
        session._cached_selector_map = {}
        result = await session.get_index_by_class("missing")
        assert result is None


# ---------------------------------------------------------------------------
# _is_valid_target (lines 2656-2713)
# ---------------------------------------------------------------------------


class TestIsValidTarget:
    """Tests for BrowserSession._is_valid_target static method."""

    def test_http_page(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "page", "url": "https://example.com"}
        assert BrowserSession._is_valid_target(target) is True

    def test_about_blank_page(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "page", "url": "about:blank"}
        assert BrowserSession._is_valid_target(target) is True

    def test_chrome_newtab(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "page", "url": "chrome://newtab/"}
        assert BrowserSession._is_valid_target(target, include_chrome=False) is True  # new tab pages always allowed

    def test_chrome_extension_excluded(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "page", "url": "chrome-extension://abc/options.html"}
        assert BrowserSession._is_valid_target(target, include_chrome_extensions=False) is False

    def test_chrome_extension_included(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "page", "url": "chrome-extension://abc/options.html"}
        assert BrowserSession._is_valid_target(target, include_chrome_extensions=True) is True

    def test_iframe_target(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "iframe", "url": "https://embed.example.com"}
        assert BrowserSession._is_valid_target(target, include_iframes=True) is True
        assert BrowserSession._is_valid_target(target, include_iframes=False) is False

    def test_worker_target(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "service_worker", "url": "https://example.com/sw.js"}
        assert BrowserSession._is_valid_target(target, include_workers=True) is True
        assert BrowserSession._is_valid_target(target, include_workers=False) is False

    def test_chrome_url(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "page", "url": "chrome://settings/"}
        assert BrowserSession._is_valid_target(target, include_chrome=True) is True
        assert BrowserSession._is_valid_target(target, include_chrome=False) is False

    def test_chrome_error(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "page", "url": "chrome-error://chromewebdata/"}
        assert BrowserSession._is_valid_target(target, include_chrome_error=True) is True
        assert BrowserSession._is_valid_target(target, include_chrome_error=False) is False

    def test_about_srcdoc_excluded(self):
        from openbrowser.browser.session import BrowserSession

        target = {"type": "page", "url": "about:srcdoc"}
        assert BrowserSession._is_valid_target(target) is False


# ---------------------------------------------------------------------------
# CDP helper methods (lines 2408-2630)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCDPHelperMethods:
    """Tests for internal CDP helper methods."""

    async def test_cdp_get_all_pages_no_client(self):
        session = _make_browser_session()
        session._cdp_client_root = None
        result = await session._cdp_get_all_pages()
        assert result == []

    async def test_cdp_get_all_pages(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        result = await session._cdp_get_all_pages()
        assert len(result) == 1

    async def test_cdp_create_new_page(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        result = await session._cdp_create_new_page()
        assert result == "target-new"

    async def test_cdp_create_new_page_no_root(self):
        session = _make_browser_session()
        session._cdp_client_root = None
        # Fallback path - uses cdp_client which requires _cdp_client_root
        # This will raise since cdp_client property asserts
        with pytest.raises(AssertionError):
            await session._cdp_create_new_page()

    async def test_cdp_close_page(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        await session._cdp_close_page("t-1")

    async def test_cdp_get_cookies(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        result = await session._cdp_get_cookies()
        assert isinstance(result, list)

    async def test_cdp_set_cookies(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        await session._cdp_set_cookies([{"name": "c1", "value": "v1"}])

    async def test_cdp_set_cookies_no_focus(self):
        session = _make_browser_session()
        await session._cdp_set_cookies([{"name": "c1"}])  # returns early

    async def test_cdp_set_cookies_empty(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        await session._cdp_set_cookies([])  # returns early

    async def test_cdp_clear_cookies(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        await session._cdp_clear_cookies()

    async def test_cdp_set_extra_headers_raises(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        with pytest.raises(NotImplementedError):
            await session._cdp_set_extra_headers({"X-Test": "1"})

    async def test_cdp_set_extra_headers_no_focus(self):
        session = _make_browser_session()
        await session._cdp_set_extra_headers({"X-Test": "1"})  # returns early

    async def test_cdp_grant_permissions_raises(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        with pytest.raises(NotImplementedError):
            await session._cdp_grant_permissions(["clipboard-read"])

    async def test_cdp_set_geolocation(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        await session._cdp_set_geolocation(40.0, -74.0)

    async def test_cdp_clear_geolocation(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        await session._cdp_clear_geolocation()

    async def test_cdp_add_init_script(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        result = await session._cdp_add_init_script("console.log('test')")
        assert result == "script-1"

    async def test_cdp_remove_init_script(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        await session._cdp_remove_init_script("script-1")

    async def test_cdp_set_viewport_with_target_id(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        await session._cdp_set_viewport(1280, 720, target_id="t-1")

    async def test_cdp_set_viewport_with_focus(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        await session._cdp_set_viewport(1280, 720)

    async def test_cdp_set_viewport_no_focus_no_target(self):
        session = _make_browser_session()
        await session._cdp_set_viewport(1280, 720)  # returns early with warning

    async def test_cdp_navigate(self):
        session = _make_browser_session()
        real_cdp = _make_cdp_session_instance(target_id="t-1", session_id="s-1")
        session.__dict__["agent_focus"] = real_cdp
        session._cdp_client_root = _make_mock_cdp_client()
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=real_cdp))
        await session._cdp_navigate("https://example.com")

    async def test_cdp_get_storage_state(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        session._cdp_get_cookies = AsyncMock(return_value=[])
        session._cdp_get_origins = AsyncMock(return_value=[])
        result = await session._cdp_get_storage_state()
        assert "cookies" in result
        assert "origins" in result


# ---------------------------------------------------------------------------
# get_tabs (lines 1629-1691)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetTabs:
    """Tests for get_tabs method."""

    async def test_get_tabs_no_client(self):
        session = _make_browser_session()
        session._cdp_client_root = None
        result = await session.get_tabs()
        assert result == []

    async def test_get_tabs_with_pages(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_get_all_pages = AsyncMock(
            return_value=[{"targetId": "t-1", "url": "https://example.com", "type": "page"}]
        )
        result = await session.get_tabs()
        assert len(result) == 1
        assert result[0].target_id == "t-1"

    async def test_get_tabs_new_tab_page(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_get_all_pages = AsyncMock(
            return_value=[{"targetId": "t-1", "url": "chrome://newtab/", "type": "page"}]
        )
        result = await session.get_tabs()
        assert len(result) == 1
        assert result[0].title == ""

    async def test_get_tabs_pdf_page(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargetInfo = AsyncMock(
            return_value={"targetInfo": {"title": "", "url": "https://example.com/doc.pdf"}}
        )
        session._cdp_get_all_pages = AsyncMock(
            return_value=[{"targetId": "t-1", "url": "https://example.com/doc.pdf", "type": "page"}]
        )
        result = await session.get_tabs()
        assert len(result) == 1
        assert "doc.pdf" in result[0].title

    async def test_get_tabs_target_info_exception(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargetInfo = AsyncMock(side_effect=Exception("CDP error"))
        session._cdp_get_all_pages = AsyncMock(
            return_value=[{"targetId": "t-1", "url": "https://example.com", "type": "page"}]
        )
        result = await session.get_tabs()
        assert len(result) == 1

    async def test_get_tabs_chrome_url_fallback(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargetInfo = AsyncMock(side_effect=Exception("fail"))
        session._cdp_get_all_pages = AsyncMock(
            return_value=[{"targetId": "t-1", "url": "chrome://settings", "type": "page"}]
        )
        result = await session.get_tabs()
        assert result[0].title == "chrome://settings"


# ---------------------------------------------------------------------------
# get_current_target_info / URL / title (lines 1696-1720)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCurrentTargetMethods:
    """Tests for current target info methods."""

    async def test_get_current_target_info_no_focus(self):
        session = _make_browser_session()
        result = await session.get_current_target_info()
        assert result is None

    async def test_get_current_target_info_found(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session(target_id="target-001")
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        result = await session.get_current_target_info()
        assert result is not None

    async def test_get_current_target_info_not_found(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session(target_id="t-missing")
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})
        result = await session.get_current_target_info()
        assert result is None

    async def test_get_current_page_url(self):
        session = _make_browser_session()
        _setattr(session, "get_current_target_info", AsyncMock(return_value={"url": "https://example.com"}))
        result = await session.get_current_page_url()
        assert result == "https://example.com"

    async def test_get_current_page_url_no_target(self):
        session = _make_browser_session()
        _setattr(session, "get_current_target_info", AsyncMock(return_value=None))
        result = await session.get_current_page_url()
        assert result == "about:blank"

    async def test_get_current_page_title(self):
        session = _make_browser_session()
        _setattr(session, "get_current_target_info", AsyncMock(return_value={"title": "Test Page"}))
        result = await session.get_current_page_title()
        assert result == "Test Page"

    async def test_get_current_page_title_no_target(self):
        session = _make_browser_session()
        _setattr(session, "get_current_target_info", AsyncMock(return_value=None))
        result = await session.get_current_page_title()
        assert result == "Unknown page title"


# ---------------------------------------------------------------------------
# downloaded_files property (line 2402)
# ---------------------------------------------------------------------------


class TestDownloadedFiles:
    """Tests for downloaded_files property."""

    def test_downloaded_files_returns_copy(self):
        session = _make_browser_session()
        session._downloaded_files = ["/tmp/file1.pdf", "/tmp/file2.pdf"]
        result = session.downloaded_files
        assert result == ["/tmp/file1.pdf", "/tmp/file2.pdf"]
        result.append("/tmp/extra.pdf")
        assert len(session._downloaded_files) == 2


# ---------------------------------------------------------------------------
# get_target_id_from_session_id (lines 2906-2920)
# ---------------------------------------------------------------------------


class TestGetTargetIdFromSessionId:
    """Tests for get_target_id_from_session_id."""

    def test_none_session_id(self):
        session = _make_browser_session()
        assert session.get_target_id_from_session_id(None) is None

    def test_found_in_pool(self):
        session = _make_browser_session()
        mock_cdp_sess = _make_mock_cdp_session(target_id="t-1", session_id="s-1")
        session._cdp_session_pool = {"t-1": mock_cdp_sess}
        result = session.get_target_id_from_session_id("s-1")
        assert result == "t-1"

    def test_not_found_in_pool(self):
        session = _make_browser_session()
        session._cdp_session_pool = {}
        result = session.get_target_id_from_session_id("s-missing")
        assert result is None


# ---------------------------------------------------------------------------
# find_frame_target (lines 2888-2901)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFindFrameTarget:
    """Tests for find_frame_target."""

    async def test_find_with_provided_frames(self):
        session = _make_browser_session()
        frames = {"frame-1": {"url": "https://example.com"}}
        result = await session.find_frame_target("frame-1", frames)
        assert result is not None

    async def test_find_not_found(self):
        session = _make_browser_session()
        frames = {"frame-1": {"url": "https://example.com"}}
        result = await session.find_frame_target("frame-missing", frames)
        assert result is None

    async def test_find_builds_frames_when_none(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session.__dict__["agent_focus"] = _make_mock_cdp_session()
        session._session_manager = _make_mock_session_manager()
        session._cdp_get_all_pages = AsyncMock(return_value=[])
        result = await session.find_frame_target("frame-missing")
        assert result is None


# ---------------------------------------------------------------------------
# cdp_client_for_target / cdp_client_for_frame (lines 2903-2958)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCdpClientForMethods:
    """Tests for cdp_client_for_target and cdp_client_for_frame."""

    async def test_cdp_client_for_target(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        result = await session.cdp_client_for_target("t-1")
        assert result is not None

    async def test_cdp_client_for_frame_no_cross_origin(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = False
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        result = await session.cdp_client_for_frame("frame-1")
        assert result is not None

    async def test_cdp_client_for_frame_not_found(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        session._cdp_get_all_pages = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="not found"):
            await session.cdp_client_for_frame("frame-missing")


# ---------------------------------------------------------------------------
# cdp_client_for_node (lines 2960-3008)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCdpClientForNode:
    """Tests for cdp_client_for_node."""

    async def test_node_with_session_id(self):
        session = _make_browser_session()
        mock_cdp_sess = _make_mock_cdp_session(session_id="s-abc")
        session._cdp_session_pool = {"t-1": mock_cdp_sess}
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus

        node = MagicMock()
        node.session_id = "s-abc"
        node.frame_id = None
        node.target_id = None
        node.backend_node_id = 42

        result = await session.cdp_client_for_node(node)
        assert result is mock_cdp_sess

    async def test_node_with_frame_id(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = False
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        node = MagicMock()
        node.session_id = None
        node.frame_id = "frame-1"
        node.target_id = None
        node.backend_node_id = 42

        result = await session.cdp_client_for_node(node)
        assert result is not None

    async def test_node_with_target_id(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        node = MagicMock()
        node.session_id = None
        node.frame_id = None
        node.target_id = "t-specific"
        node.backend_node_id = 42

        result = await session.cdp_client_for_node(node)
        assert result is not None

    async def test_node_fallback_to_agent_focus(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus

        node = MagicMock()
        node.session_id = None
        node.frame_id = None
        node.target_id = None
        node.backend_node_id = 42

        result = await session.cdp_client_for_node(node)
        assert result is mock_focus

    async def test_node_last_resort_main_session(self):
        session = _make_browser_session()
        session.__dict__["agent_focus"] = None
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        # Need agent_focus for get_or_create - set one temporarily
        mock_focus = _make_mock_cdp_session()

        async def set_focus_then_get(*args, **kwargs):
            session.__dict__["agent_focus"] = mock_focus
            return mock_focus

        _setattr(session, "get_or_create_cdp_session", AsyncMock(side_effect=set_focus_then_get))

        node = MagicMock()
        node.session_id = None
        node.frame_id = None
        node.target_id = None
        node.backend_node_id = 42

        result = await session.cdp_client_for_node(node)
        assert result is mock_focus


# ---------------------------------------------------------------------------
# take_screenshot (lines 3010-3067)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTakeScreenshot:
    """Tests for take_screenshot."""

    async def test_basic_screenshot(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        import base64

        expected_data = b"hello"
        mock_focus.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": base64.b64encode(expected_data).decode()}
        )

        result = await session.take_screenshot()
        assert result == expected_data

    async def test_screenshot_with_path(self, tmp_path):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        import base64

        data = b"screenshot-data"
        mock_focus.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": base64.b64encode(data).decode()}
        )

        path = str(tmp_path / "shot.png")
        result = await session.take_screenshot(path=path)
        assert Path(path).read_bytes() == data

    async def test_screenshot_with_clip(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        import base64

        mock_focus.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": base64.b64encode(b"clipped").decode()}
        )

        result = await session.take_screenshot(clip={"x": 0, "y": 0, "width": 100, "height": 100})
        assert result == b"clipped"

    async def test_screenshot_no_data_raises(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))
        mock_focus.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={})

        with pytest.raises(Exception, match="no data"):
            await session.take_screenshot()

    async def test_screenshot_jpeg_quality(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        import base64

        mock_focus.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": base64.b64encode(b"jpeg").decode()}
        )

        result = await session.take_screenshot(format="jpeg", quality=80)
        assert result == b"jpeg"


# ---------------------------------------------------------------------------
# screenshot_element / _get_element_bounds (lines 3069-3131)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScreenshotElement:
    """Tests for screenshot_element and _get_element_bounds."""

    async def test_screenshot_element_success(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        import base64

        mock_focus.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": base64.b64encode(b"element-shot").decode()}
        )

        result = await session.screenshot_element("#my-element")
        assert result == b"element-shot"

    async def test_screenshot_element_not_found(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        mock_focus.cdp_client.send.DOM.querySelector = AsyncMock(return_value={"nodeId": 0})
        session.__dict__["agent_focus"] = mock_focus
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        with pytest.raises(ValueError, match="not found"):
            await session.screenshot_element("#missing")

    async def test_get_element_bounds_no_box_model(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        mock_focus.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={})
        session.__dict__["agent_focus"] = mock_focus
        _setattr(session, "get_or_create_cdp_session", AsyncMock(return_value=mock_focus))

        result = await session._get_element_bounds("#test")
        assert result is None


# ---------------------------------------------------------------------------
# get_element_coordinates (lines 1931-2043)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetElementCoordinates:
    """Tests for get_element_coordinates."""

    async def test_method1_content_quads(self):
        session = _make_browser_session()
        cdp_sess = _make_mock_cdp_session()
        result = await session.get_element_coordinates(42, cdp_sess)
        assert result is not None
        assert result.width > 0

    async def test_method2_box_model(self):
        session = _make_browser_session()
        cdp_sess = _make_mock_cdp_session()
        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={"quads": []})
        result = await session.get_element_coordinates(42, cdp_sess)
        assert result is not None

    async def test_method3_js_fallback(self):
        session = _make_browser_session()
        cdp_sess = _make_mock_cdp_session()
        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={"quads": []})
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={})
        result = await session.get_element_coordinates(42, cdp_sess)
        assert result is not None

    async def test_all_methods_fail(self):
        session = _make_browser_session()
        cdp_sess = _make_mock_cdp_session()
        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(side_effect=Exception("fail"))
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(side_effect=Exception("fail"))
        cdp_sess.cdp_client.send.DOM.resolveNode = AsyncMock(side_effect=Exception("fail"))
        result = await session.get_element_coordinates(42, cdp_sess)
        assert result is None

    async def test_js_fallback_zero_size(self):
        session = _make_browser_session()
        cdp_sess = _make_mock_cdp_session()
        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={"quads": []})
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={})
        cdp_sess.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={"result": {"value": {"x": 0, "y": 0, "width": 0, "height": 0}}}
        )
        result = await session.get_element_coordinates(42, cdp_sess)
        assert result is None


# ---------------------------------------------------------------------------
# remove_highlights (lines 1886-1929)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoveHighlights:
    """Tests for remove_highlights."""

    async def test_remove_when_disabled(self):
        session = _make_browser_session()
        session.browser_profile.highlight_elements = False
        await session.remove_highlights()

    async def test_remove_success(self):
        session = _make_browser_session()
        session.browser_profile.highlight_elements = True
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        mock_focus.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": {"removed": 5}}}
        )

        await session.remove_highlights()

    async def test_remove_no_result_value(self):
        session = _make_browser_session()
        session.browser_profile.highlight_elements = True
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        mock_focus.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={})
        await session.remove_highlights()

    async def test_remove_exception(self):
        session = _make_browser_session()
        session.browser_profile.highlight_elements = True
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        _setattr(session, "get_or_create_cdp_session", AsyncMock(side_effect=Exception("no session")))

        await session.remove_highlights()


# ---------------------------------------------------------------------------
# highlight_interaction_element (lines 2045-2185)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHighlightInteractionElement:
    """Tests for highlight_interaction_element."""

    async def test_disabled(self):
        session = _make_browser_session()
        session.browser_profile.highlight_elements = False
        await session.highlight_interaction_element(MagicMock())

    async def test_no_coordinates(self):
        session = _make_browser_session()
        session.browser_profile.highlight_elements = True
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        _setattr(session, "get_element_coordinates", AsyncMock(return_value=None))

        node = MagicMock()
        node.backend_node_id = 42
        await session.highlight_interaction_element(node)

    async def test_success(self):
        session = _make_browser_session()
        session.browser_profile.highlight_elements = True
        session.browser_profile.interaction_highlight_color = "#FF0000"
        session.browser_profile.interaction_highlight_duration = 0.5
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        from openbrowser.dom.views import DOMRect

        _setattr(session, "get_element_coordinates", AsyncMock(return_value=DOMRect(x=10, y=20, width=100, height=50)))

        node = MagicMock()
        node.backend_node_id = 42
        await session.highlight_interaction_element(node)

    async def test_exception_caught(self):
        session = _make_browser_session()
        session.browser_profile.highlight_elements = True
        _setattr(session, "get_or_create_cdp_session", AsyncMock(side_effect=Exception("fail")))

        node = MagicMock()
        node.backend_node_id = 42
        await session.highlight_interaction_element(node)


# ---------------------------------------------------------------------------
# add_highlights (lines 2187-2370)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAddHighlights:
    """Tests for add_highlights."""

    async def test_disabled(self):
        session = _make_browser_session()
        session.browser_profile.dom_highlight_elements = False
        await session.add_highlights({})

    async def test_empty_selector_map(self):
        session = _make_browser_session()
        session.browser_profile.dom_highlight_elements = True
        await session.add_highlights({})

    async def test_no_valid_elements(self):
        session = _make_browser_session()
        session.browser_profile.dom_highlight_elements = True

        node = MagicMock()
        node.absolute_position = None
        await session.add_highlights({1: node})

    async def test_success(self):
        session = _make_browser_session()
        session.browser_profile.dom_highlight_elements = True
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()
        _setattr(session, "remove_highlights", AsyncMock())

        from openbrowser.dom.views import DOMRect

        node = MagicMock()
        node.absolute_position = DOMRect(x=10, y=20, width=100, height=50)
        node.node_name = "BUTTON"
        node.snapshot_node.is_clickable = True
        node.is_scrollable = False
        node.attributes = {}
        node.frame_id = None
        node.node_id = 1
        node.backend_node_id = 42
        node.xpath = "/html/body/button"
        node.get_all_children_text.return_value = "Click"
        node.node_value = "Click"

        mock_focus.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": {"added": 1}}}
        )

        await session.add_highlights({1: node})

    async def test_exception_caught(self):
        session = _make_browser_session()
        session.browser_profile.dom_highlight_elements = True
        _setattr(session, "remove_highlights", AsyncMock())

        from openbrowser.dom.views import DOMRect

        node = MagicMock()
        node.absolute_position = DOMRect(x=10, y=20, width=100, height=50)
        node.node_name = "BUTTON"
        node.snapshot_node.is_clickable = True
        node.is_scrollable = False
        node.attributes = {}
        node.frame_id = None
        node.node_id = 1
        node.backend_node_id = 42
        node.xpath = "/html/body/button"
        node.get_all_children_text.return_value = "Click"
        node.node_value = "Click"

        _setattr(session, "get_or_create_cdp_session", AsyncMock(side_effect=Exception("fail")))

        await session.add_highlights({1: node})


# ---------------------------------------------------------------------------
# _close_extension_options_pages (lines 2372-2393)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCloseExtensionOptionsPages:
    """Tests for _close_extension_options_pages."""

    async def test_no_extension_pages(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_get_all_pages = AsyncMock(
            return_value=[{"targetId": "t-1", "url": "https://example.com"}]
        )
        await session._close_extension_options_pages()

    async def test_closes_extension_pages(self):
        session = _make_browser_session()
        session._cdp_client_root = _make_mock_cdp_client()
        session._cdp_get_all_pages = AsyncMock(
            return_value=[
                {"targetId": "t-ext", "url": "chrome-extension://abc/options.html"},
                {"targetId": "t-1", "url": "https://example.com"},
            ]
        )
        session._cdp_close_page = AsyncMock()
        await session._close_extension_options_pages()
        session._cdp_close_page.assert_called_with("t-ext")

    async def test_exception_handled(self):
        session = _make_browser_session()
        session._cdp_get_all_pages = AsyncMock(side_effect=Exception("fail"))
        await session._close_extension_options_pages()


# ---------------------------------------------------------------------------
# _cdp_get_origins (lines 2556-2629)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCdpGetOrigins:
    """Tests for _cdp_get_origins."""

    async def test_get_origins_with_storage(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        mock_sess = _make_mock_cdp_session()
        mock_sess.cdp_client.send.Page.getFrameTree = AsyncMock(return_value={
            "frameTree": {
                "frame": {"id": "f1", "securityOrigin": "https://example.com"},
                "childFrames": []
            }
        })
        mock_sess.cdp_client.send.DOMStorage.getDOMStorageItems = AsyncMock(
            return_value={"entries": [["key1", "val1"]]}
        )
        session._session_manager.get_session_for_target = AsyncMock(return_value=mock_sess)

        result = await session._cdp_get_origins()
        assert isinstance(result, list)

    async def test_get_origins_exception(self):
        session = _make_browser_session()
        mock_focus = _make_mock_cdp_session()
        session.__dict__["agent_focus"] = mock_focus
        session._cdp_client_root = _make_mock_cdp_client()
        session._session_manager = _make_mock_session_manager()

        mock_sess = _make_mock_cdp_session()
        mock_sess.cdp_client.send.DOMStorage.enable = AsyncMock(side_effect=Exception("fail"))
        session._session_manager.get_session_for_target = AsyncMock(return_value=mock_sess)

        result = await session._cdp_get_origins()
        assert result == []


# ---------------------------------------------------------------------------
# navigate_to (lines 1722-1733)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNavigateTo:
    """Tests for navigate_to convenience method."""

    async def test_navigate_to(self):
        session = _make_browser_session()
        awaitable = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=awaitable)

        await session.navigate_to("https://example.com", new_tab=True)
        session.event_bus.dispatch.assert_called_once()
