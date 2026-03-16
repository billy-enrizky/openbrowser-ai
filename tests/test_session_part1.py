"""Tests for the first third of openbrowser.browser.session (lines 56-996).

Covers CDPSession and BrowserSession event handlers and public methods.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from bubus import EventBus
from cdp_use import CDPClient

from openbrowser.browser.events import (
    AgentFocusChangedEvent,
    BrowserConnectedEvent,
    BrowserErrorEvent,
    BrowserLaunchEvent,
    BrowserLaunchResult,
    BrowserStartEvent,
    BrowserStopEvent,
    BrowserStoppedEvent,
    CloseTabEvent,
    FileDownloadedEvent,
    NavigateToUrlEvent,
    NavigationCompleteEvent,
    NavigationStartedEvent,
    SwitchTabEvent,
    TabClosedEvent,
    TabCreatedEvent,
)
from openbrowser.browser.profile import BrowserProfile, ViewportSize
from openbrowser.browser.session import CDPSession, BrowserSession, _LOGGED_UNIQUE_SESSION_IDS
from openbrowser.browser.views import TabInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCDPClient(CDPClient):
    """Subclass CDPClient so isinstance checks pass for Pydantic validation."""

    def __init__(self):
        # Skip super().__init__ to avoid real websocket connections
        pass


def _make_mock_cdp_client():
    """Create a mock CDP client that passes isinstance(CDPClient) checks."""
    client = _MockCDPClient()
    client.send = MagicMock()

    # Target domain
    client.send.Target.attachToTarget = AsyncMock(return_value={"sessionId": "sess-1234"})
    client.send.Target.getTargetInfo = AsyncMock(
        return_value={
            "targetInfo": {
                "targetId": "ABCDEF1234567890ABCDEF1234567890",
                "url": "https://example.com",
                "title": "Example",
                "type": "page",
            }
        }
    )
    client.send.Target.activateTarget = AsyncMock()
    client.send.Target.createTarget = AsyncMock(return_value={"targetId": "new-target-id-0000"})
    client.send.Target.closeTarget = AsyncMock()
    client.send.Target.getTargets = AsyncMock(
        return_value={
            "targetInfos": [
                {"targetId": "target-1", "url": "https://a.com", "title": "A", "type": "page"},
                {"targetId": "target-2", "url": "https://b.com", "title": "B", "type": "page"},
            ]
        }
    )

    # CDP domains with enable methods
    for domain_name in ["Page", "DOM", "DOMSnapshot", "Accessibility", "Runtime", "Inspector", "Debugger"]:
        domain = getattr(client.send, domain_name)
        domain.enable = AsyncMock()

    # Debugger extras
    client.send.Debugger.setSkipAllPauses = AsyncMock()

    # Page navigate
    client.send.Page.navigate = AsyncMock()

    # Runtime evaluate
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})

    # Network
    client.send.Network.getCookies = AsyncMock(return_value={"cookies": []})
    client.send.Network.clearBrowserCookies = AsyncMock()

    return client


def _make_cdp_session(client=None, target_id="AAAA1111BBBB2222CCCC3333DDDD4444", session_id="session-focus",
                      url="https://example.com", title="Example"):
    """Create a CDPSession via model_construct to bypass Pydantic validation."""
    if client is None:
        client = _make_mock_cdp_client()
    return CDPSession.model_construct(
        cdp_client=client,
        target_id=target_id,
        session_id=session_id,
        url=url,
        title=title,
    )


class _AwaitableMock:
    """A simple awaitable that tracks dispatch calls.

    EventBus.dispatch() returns an event which is awaitable. Some code paths
    do `await event_bus.dispatch(...)` so the mock must return an awaitable.
    This also supports `event_result()` for the `start()` / `kill()` code paths.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, event):
        self.calls.append(event)
        return self

    def __await__(self):
        yield
        return self

    async def event_result(self, raise_if_any=False, raise_if_none=False):
        return None

    @property
    def call_args_list(self):
        """Compatibility with assertion patterns that inspect call_args_list."""
        from unittest.mock import call
        return [call(e) for e in self.calls]


class _DispatchResult:
    """Awaitable result from dispatch, also supports event_result()."""

    async def event_result(self, raise_if_any=False, raise_if_none=False):
        return None

    def __await__(self):
        yield
        return self


def _make_dispatch_mock(extra_side_effect=None):
    """Create a mock for EventBus.dispatch that returns awaitables.

    Args:
        extra_side_effect: Optional callable(event) for additional logic
            (e.g., updating agent_focus on SwitchTabEvent). Called before
            returning the awaitable result.
    """
    dispatched_events = []

    def dispatch_side_effect(event):
        dispatched_events.append(event)
        if extra_side_effect:
            extra_side_effect(event)
        return _DispatchResult()

    mock = MagicMock(side_effect=dispatch_side_effect)
    mock._dispatched_events = dispatched_events
    return mock


def _make_browser_session(**kwargs):
    """Create a BrowserSession with event_bus.dispatch replaced by an awaitable mock."""
    defaults = {
        "browser_profile": BrowserProfile(cdp_url="ws://localhost:9222", is_local=False),
    }
    defaults.update(kwargs)
    session = BrowserSession(**defaults)
    session.event_bus.dispatch = _make_dispatch_mock()
    return session


def _setup_session_with_cdp(session, cdp_client=None):
    """Wire up a mock CDP client + agent_focus + session_manager into a session."""
    client = cdp_client or _make_mock_cdp_client()
    session._cdp_client_root = client

    focus = _make_cdp_session(client=client)
    session.agent_focus = focus

    sm = MagicMock()
    sm.get_session_for_target = AsyncMock(return_value=focus)
    sm.validate_session = AsyncMock(return_value=True)
    sm.clear = AsyncMock()
    session._session_manager = sm

    return client, focus, sm


# ===========================================================================
# CDPSession tests
# ===========================================================================


class TestCDPSession:
    """Tests for CDPSession Pydantic model."""

    def test_basic_creation(self):
        client = _MockCDPClient()
        s = CDPSession(
            cdp_client=client,
            target_id="target-abc",
            session_id="session-xyz",
        )
        assert s.target_id == "target-abc"
        assert s.session_id == "session-xyz"
        assert s.title == "Unknown title"
        assert s.url == "about:blank"


@pytest.mark.asyncio
class TestCDPSessionForTarget:
    """Tests for CDPSession.for_target class method (lines 86-91)."""

    async def test_for_target_creates_and_attaches(self):
        client = _make_mock_cdp_client()
        session = await CDPSession.for_target(client, "target-abc")
        assert session.session_id == "sess-1234"
        assert session.target_id == "target-abc"
        assert session.url == "https://example.com"
        assert session.title == "Example"

    async def test_for_target_with_custom_domains(self):
        client = _make_mock_cdp_client()
        session = await CDPSession.for_target(client, "target-abc", domains=["Page", "Runtime"])
        client.send.Page.enable.assert_awaited_once()
        client.send.Runtime.enable.assert_awaited_once()


@pytest.mark.asyncio
class TestCDPSessionAttach:
    """Tests for CDPSession.attach (lines 93-135)."""

    async def test_attach_sets_session_id(self):
        client = _make_mock_cdp_client()
        s = CDPSession(cdp_client=client, target_id="t1", session_id="connecting")
        result = await s.attach()
        assert result.session_id == "sess-1234"
        assert result is s

    async def test_attach_enables_default_domains(self):
        client = _make_mock_cdp_client()
        s = CDPSession(cdp_client=client, target_id="t1", session_id="connecting")
        await s.attach()
        for domain in ["Page", "DOM", "DOMSnapshot", "Accessibility", "Runtime", "Inspector"]:
            getattr(client.send, domain).enable.assert_awaited()

    async def test_attach_raises_on_domain_enable_failure(self):
        client = _make_mock_cdp_client()
        client.send.Page.enable = AsyncMock(side_effect=RuntimeError("fail"))
        s = CDPSession(cdp_client=client, target_id="t1", session_id="connecting")
        with pytest.raises(RuntimeError, match="Failed to enable requested CDP domain"):
            await s.attach()

    async def test_attach_handles_debugger_skip_pauses_error(self):
        """Debugger.setSkipAllPauses failure should be silently ignored."""
        client = _make_mock_cdp_client()
        client.send.Debugger.setSkipAllPauses = AsyncMock(side_effect=Exception("no debugger"))
        s = CDPSession(cdp_client=client, target_id="t1", session_id="connecting")
        result = await s.attach()
        assert result.session_id == "sess-1234"

    async def test_attach_fetches_target_info(self):
        client = _make_mock_cdp_client()
        client.send.Target.getTargetInfo = AsyncMock(
            return_value={
                "targetInfo": {
                    "targetId": "t1",
                    "url": "https://test.dev",
                    "title": "Test Page",
                    "type": "page",
                }
            }
        )
        s = CDPSession(cdp_client=client, target_id="t1", session_id="connecting")
        await s.attach()
        assert s.title == "Test Page"
        assert s.url == "https://test.dev"

    async def test_attach_browser_domain_no_session_id(self):
        """Browser and Target domains should not pass session_id to enable."""
        client = _make_mock_cdp_client()
        client.send.Browser.enable = AsyncMock()
        client.send.Target.enable = AsyncMock()
        s = CDPSession(cdp_client=client, target_id="t1", session_id="connecting")
        await s.attach(domains=["Browser", "Target"])
        client.send.Browser.enable.assert_awaited_once_with()
        client.send.Target.enable.assert_awaited_once_with()


@pytest.mark.asyncio
class TestCDPSessionDisconnect:
    """Tests for CDPSession.disconnect (line 141)."""

    async def test_disconnect_is_noop(self):
        client = _MockCDPClient()
        s = CDPSession(cdp_client=client, target_id="t1", session_id="s1")
        await s.disconnect()


@pytest.mark.asyncio
class TestCDPSessionGetTabInfo:
    """Tests for CDPSession.get_tab_info (lines 143-149)."""

    async def test_get_tab_info_returns_tab_info(self):
        client = _make_mock_cdp_client()
        s = CDPSession(cdp_client=client, target_id="target-full-id", session_id="s1")
        info = await s.get_tab_info()
        assert isinstance(info, TabInfo)
        assert info.url == "https://example.com"
        assert info.title == "Example"


@pytest.mark.asyncio
class TestCDPSessionGetTargetInfo:
    """Tests for CDPSession.get_target_info (lines 151-153)."""

    async def test_get_target_info_calls_cdp(self):
        client = _make_mock_cdp_client()
        s = CDPSession(cdp_client=client, target_id="t-abc", session_id="s1")
        info = await s.get_target_info()
        client.send.Target.getTargetInfo.assert_awaited_once_with(params={"targetId": "t-abc"})
        assert info["url"] == "https://example.com"


# ===========================================================================
# BrowserSession construction tests
# ===========================================================================


class TestBrowserSessionInit:
    """Tests for BrowserSession.__init__ (lines 246-343)."""

    def test_basic_init_with_profile(self):
        profile = BrowserProfile(cdp_url="ws://localhost:9222", is_local=False)
        session = BrowserSession(browser_profile=profile)
        assert session.browser_profile.cdp_url == "ws://localhost:9222"

    def test_init_with_direct_params(self):
        session = BrowserSession(cdp_url="ws://localhost:9222", headless=True)
        assert session.browser_profile.cdp_url == "ws://localhost:9222"
        assert session.browser_profile.headless is True

    def test_init_sets_is_local_when_no_cdp_url(self):
        session = BrowserSession()
        assert session.browser_profile.is_local is True

    def test_init_sets_is_local_when_executable_path(self):
        session = BrowserSession(executable_path="/usr/bin/chromium", cdp_url="ws://localhost:9222")
        assert session.browser_profile.is_local is True

    def test_init_merges_profile_and_direct_kwargs(self):
        profile = BrowserProfile(cdp_url="ws://localhost:9222", is_local=False, headless=False)
        session = BrowserSession(browser_profile=profile, headless=True)
        assert session.browser_profile.headless is True

    def test_init_generates_unique_id(self):
        s1 = BrowserSession(cdp_url="ws://localhost:9222")
        s2 = BrowserSession(cdp_url="ws://localhost:9222")
        assert s1.id != s2.id

    def test_init_with_explicit_id(self):
        session = BrowserSession(id="my-session", cdp_url="ws://localhost:9222")
        assert session.id == "my-session"


class TestBrowserSessionProperties:
    """Tests for BrowserSession properties."""

    def test_cdp_url_property(self):
        session = _make_browser_session()
        assert session.cdp_url == "ws://localhost:9222"

    def test_is_local_property(self):
        # Must pass cdp_url as direct kwarg to prevent is_local being forced True
        session = BrowserSession(cdp_url="ws://localhost:9222")
        session.event_bus.dispatch = MagicMock()
        # When cdp_url is provided, is_local defaults to False
        assert session.is_local is False

    def test_id_for_logs_uses_port_when_unique(self):
        _LOGGED_UNIQUE_SESSION_IDS.clear()
        profile = BrowserProfile(cdp_url="ws://localhost:5678", is_local=False)
        session = BrowserSession(browser_profile=profile)
        session.event_bus.dispatch = MagicMock()
        assert session._id_for_logs == "5678"

    def test_id_for_logs_falls_back_to_uuid(self):
        _LOGGED_UNIQUE_SESSION_IDS.clear()
        profile = BrowserProfile(cdp_url="ws://localhost:9222", is_local=False)
        session = BrowserSession(browser_profile=profile)
        session.event_bus.dispatch = MagicMock()
        log_id = session._id_for_logs
        assert log_id == session.id[-4:]

    def test_id_for_logs_no_cdp(self):
        _LOGGED_UNIQUE_SESSION_IDS.clear()
        session = BrowserSession()
        session.event_bus.dispatch = MagicMock()
        log_id = session._id_for_logs
        assert log_id == session.id[-4:]

    def test_id_for_logs_duplicate_port_falls_back(self):
        """Second session with same port should fall back to UUID suffix."""
        _LOGGED_UNIQUE_SESSION_IDS.clear()
        profile1 = BrowserProfile(cdp_url="ws://localhost:4567", is_local=False)
        s1 = BrowserSession(browser_profile=profile1)
        s1.event_bus.dispatch = MagicMock()
        assert s1._id_for_logs == "4567"  # first one claims it

        profile2 = BrowserProfile(cdp_url="ws://localhost:4567", is_local=False)
        s2 = BrowserSession(browser_profile=profile2)
        s2.event_bus.dispatch = MagicMock()
        assert s2._id_for_logs == s2.id[-4:]  # duplicate, falls back

    def test_tab_id_for_logs_with_focus(self):
        session = _make_browser_session()
        focus = _make_cdp_session(target_id="AAAA1111BBBB2222")
        session.agent_focus = focus
        assert session._tab_id_for_logs == "22"

    def test_tab_id_for_logs_without_focus(self):
        session = _make_browser_session()
        session.agent_focus = None
        assert "--" in session._tab_id_for_logs

    def test_repr_and_str(self):
        session = _make_browser_session()
        assert "BrowserSession" in repr(session)
        assert "BrowserSession" in str(session)


class TestBrowserSessionModelPostInit:
    """Tests for model_post_init duplicate handler detection (lines 462-486)."""

    def test_duplicate_handler_raises(self):
        """Re-using an EventBus that already has handlers triggers duplicate detection."""
        from pydantic import BaseModel

        profile = BrowserProfile(cdp_url="ws://localhost:9222", is_local=False)
        s1 = BrowserSession(browser_profile=profile)
        eb = s1.event_bus  # has handlers from s1

        # Create a new BrowserSession instance sharing the same EventBus
        # by going through BaseModel.__init__ which triggers model_post_init
        s2 = BrowserSession.__new__(BrowserSession)
        with pytest.raises(RuntimeError, match="Duplicate handler registration"):
            BaseModel.__init__(s2, id="test-dup", browser_profile=profile, event_bus=eb)


# ===========================================================================
# BrowserSession.reset tests
# ===========================================================================


@pytest.mark.asyncio
class TestBrowserSessionReset:
    """Tests for BrowserSession.reset (lines 426-461)."""

    async def test_reset_clears_state(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        session._downloaded_files.append("/tmp/file.pdf")
        session._cached_selector_map[1] = MagicMock()
        session._cached_browser_state_summary = "cached"

        await session.reset()

        assert session._cdp_client_root is None
        assert session.agent_focus is None
        assert len(session._cdp_session_pool) == 0
        assert len(session._downloaded_files) == 0
        assert len(session._cached_selector_map) == 0
        assert session._cached_browser_state_summary is None
        sm.clear.assert_awaited_once()

    async def test_reset_clears_watchdogs(self):
        session = _make_browser_session()
        _setup_session_with_cdp(session)
        session._crash_watchdog = "wd"
        session._dom_watchdog = "wd"

        await session.reset()

        assert session._crash_watchdog is None
        assert session._dom_watchdog is None

    async def test_reset_clears_cdp_url_for_local(self):
        profile = BrowserProfile(cdp_url="ws://localhost:5555", is_local=True)
        session = BrowserSession(browser_profile=profile)
        session.event_bus.dispatch = MagicMock()
        _setup_session_with_cdp(session)

        await session.reset()
        assert session.browser_profile.cdp_url is None

    async def test_reset_without_session_manager(self):
        session = _make_browser_session()
        session._session_manager = None
        session._cdp_client_root = MagicMock()
        focus = _make_cdp_session()
        session.agent_focus = focus

        await session.reset()
        assert session._cdp_client_root is None


# ===========================================================================
# BrowserSession event handler tests
# ===========================================================================


@pytest.mark.asyncio
class TestOnBrowserStartEvent:
    """Tests for on_BrowserStartEvent (lines 536-588)."""

    async def test_start_with_existing_cdp_url_and_connected(self):
        """When already connected, skip reconnection (line 575)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch.object(BrowserSession, "attach_all_watchdogs", new_callable=AsyncMock):
            with patch.object(BrowserSession, "connect", new_callable=AsyncMock):
                result = await session.on_BrowserStartEvent(BrowserStartEvent())

        assert result == {"cdp_url": "ws://localhost:9222"}

    async def test_start_connects_when_not_connected(self):
        session = _make_browser_session()
        session._cdp_client_root = None
        session.agent_focus = None

        mock_client = _make_mock_cdp_client()

        async def fake_connect(cdp_url=None):
            session._cdp_client_root = mock_client
            return session

        with patch.object(BrowserSession, "attach_all_watchdogs", new_callable=AsyncMock):
            with patch.object(BrowserSession, "connect", side_effect=fake_connect):
                result = await session.on_BrowserStartEvent(BrowserStartEvent())

        assert result == {"cdp_url": "ws://localhost:9222"}

    async def test_start_raises_when_no_cdp_url_and_not_local(self):
        """Should raise ValueError when no cdp_url and not local (line 562)."""
        # Construct with cdp_url then clear it + set is_local=False to reach the error branch
        session = _make_browser_session()
        session.browser_profile.cdp_url = None
        session.browser_profile.is_local = False

        with patch.object(BrowserSession, "attach_all_watchdogs", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="no cdp_url was provided"):
                await session.on_BrowserStartEvent(BrowserStartEvent())

    async def test_start_dispatches_error_event_on_exception(self):
        """On exception, should dispatch BrowserErrorEvent (lines 580-588)."""
        session = _make_browser_session()
        session._cdp_client_root = None

        with patch.object(BrowserSession, "attach_all_watchdogs", new_callable=AsyncMock):
            with patch.object(BrowserSession, "connect", new_callable=AsyncMock, side_effect=ConnectionError("refused")):
                with pytest.raises(ConnectionError):
                    await session.on_BrowserStartEvent(BrowserStartEvent())

        dispatch_calls = session.event_bus.dispatch.call_args_list
        error_events = [c for c in dispatch_calls if isinstance(c[0][0], BrowserErrorEvent)]
        assert len(error_events) >= 1
        assert "BrowserStartEventError" in error_events[0][0][0].error_type


@pytest.mark.asyncio
class TestOnNavigateToUrlEvent:
    """Tests for on_NavigateToUrlEvent (lines 590-725)."""

    async def test_navigate_no_focus_returns_early(self):
        """If no agent_focus, return without navigating (lines 594-595)."""
        session = _make_browser_session()
        session.agent_focus = None
        event = NavigateToUrlEvent(url="https://example.com")
        await session.on_NavigateToUrlEvent(event)

    async def test_navigate_new_tab_true_already_on_new_tab(self):
        """new_tab=True but current page is new tab page => set new_tab=False (lines 601-609)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        event = NavigateToUrlEvent(url="https://example.com", new_tab=True)

        with patch.object(BrowserSession, "get_current_page_url", new_callable=AsyncMock, return_value="chrome://newtab/"):
            with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=[]):
                with patch.object(BrowserSession, "_close_extension_options_pages", new_callable=AsyncMock):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        await session.on_NavigateToUrlEvent(event)

        assert event.new_tab is False

    async def test_navigate_new_tab_check_url_exception(self):
        """If get_current_page_url raises, should continue (lines 608-609)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        blank_tid = "new-tab-id-00000000"
        event = NavigateToUrlEvent(url="https://example.com", new_tab=True)

        new_focus = _make_cdp_session(client=client, target_id=blank_tid, url="about:blank")

        def on_dispatch(evt):
            if isinstance(evt, SwitchTabEvent):
                session.agent_focus = new_focus

        session.event_bus.dispatch = _make_dispatch_mock(extra_side_effect=on_dispatch)

        with patch.object(BrowserSession, "get_current_page_url", new_callable=AsyncMock, side_effect=Exception("no page")):
            with patch.object(
                BrowserSession,
                "_cdp_get_all_pages",
                new_callable=AsyncMock,
                return_value=[{"targetId": blank_tid, "url": "about:blank"}],
            ):
                with patch.object(BrowserSession, "_close_extension_options_pages", new_callable=AsyncMock):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        await session.on_NavigateToUrlEvent(event)

    async def test_navigate_url_already_open_in_other_tab(self):
        """URL already open in another tab => switch to it (lines 615-616)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        target_url = "https://already-open.com"
        other_target_id = "OTHER-TARGET-12345678"

        pages = [
            {"targetId": focus.target_id, "url": "https://current.com"},
            {"targetId": other_target_id, "url": target_url},
        ]
        event = NavigateToUrlEvent(url=target_url, new_tab=False)

        other_focus = _make_cdp_session(client=client, target_id=other_target_id, url=target_url)

        def on_dispatch(evt):
            if isinstance(evt, SwitchTabEvent):
                session.agent_focus = other_focus

        session.event_bus.dispatch = _make_dispatch_mock(extra_side_effect=on_dispatch)

        with patch.object(BrowserSession, "get_current_page_url", new_callable=AsyncMock, return_value="https://current.com"):
            with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=pages):
                with patch.object(BrowserSession, "_close_extension_options_pages", new_callable=AsyncMock):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        await session.on_NavigateToUrlEvent(event)

    async def test_navigate_new_tab_creates_tab_when_no_blank(self):
        """new_tab=True with no about:blank tab creates new tab (lines 639-654)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        new_tid = "NEWTARGET-00001234"

        pages_no_blank = [{"targetId": focus.target_id, "url": "https://current.com"}]
        pages_after_create = pages_no_blank + [{"targetId": new_tid, "url": "about:blank"}]

        call_count = [0]

        async def mock_get_all_pages(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return pages_no_blank
            return pages_after_create

        event = NavigateToUrlEvent(url="https://example.com", new_tab=True)

        new_focus = _make_cdp_session(client=client, target_id=new_tid, url="about:blank")

        def on_dispatch(evt):
            if isinstance(evt, SwitchTabEvent):
                session.agent_focus = new_focus

        session.event_bus.dispatch = _make_dispatch_mock(extra_side_effect=on_dispatch)

        with patch.object(BrowserSession, "get_current_page_url", new_callable=AsyncMock, return_value="https://current.com"):
            with patch.object(BrowserSession, "_cdp_get_all_pages", side_effect=mock_get_all_pages):
                with patch.object(BrowserSession, "_cdp_create_new_page", new_callable=AsyncMock, return_value=new_tid):
                    with patch.object(BrowserSession, "_close_extension_options_pages", new_callable=AsyncMock):
                        with patch("asyncio.sleep", new_callable=AsyncMock):
                            await session.on_NavigateToUrlEvent(event)

    async def test_navigate_new_tab_create_failure_falls_back(self):
        """When creating new tab fails, fall back to current tab (lines 650-654)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        pages = [{"targetId": focus.target_id, "url": "https://current.com"}]
        event = NavigateToUrlEvent(url="https://example.com", new_tab=True)

        with patch.object(BrowserSession, "get_current_page_url", new_callable=AsyncMock, return_value="https://current.com"):
            with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=pages):
                with patch.object(BrowserSession, "_cdp_create_new_page", new_callable=AsyncMock, side_effect=Exception("creation failed")):
                    with patch.object(BrowserSession, "_close_extension_options_pages", new_callable=AsyncMock):
                        with patch("asyncio.sleep", new_callable=AsyncMock):
                            await session.on_NavigateToUrlEvent(event)

    async def test_navigate_reuses_about_blank_tab(self):
        """new_tab=True should reuse existing about:blank tab (lines 625-637)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        blank_tid = "BLANK-TARGET-0000001"
        pages = [
            {"targetId": focus.target_id, "url": "https://current.com"},
            {"targetId": blank_tid, "url": "about:blank"},
        ]
        event = NavigateToUrlEvent(url="https://example.com", new_tab=True)

        new_focus = _make_cdp_session(client=client, target_id=blank_tid, url="about:blank")

        def on_dispatch(evt):
            if isinstance(evt, SwitchTabEvent):
                session.agent_focus = new_focus

        session.event_bus.dispatch = _make_dispatch_mock(extra_side_effect=on_dispatch)

        with patch.object(BrowserSession, "get_current_page_url", new_callable=AsyncMock, return_value="https://current.com"):
            with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=pages):
                with patch.object(BrowserSession, "_close_extension_options_pages", new_callable=AsyncMock):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        await session.on_NavigateToUrlEvent(event)

    async def test_navigate_exception_dispatches_error(self):
        """Exception during navigation dispatches NavigationCompleteEvent with error (lines 714-725)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        event = NavigateToUrlEvent(url="https://example.com")

        focus.cdp_client.send.Page.navigate = AsyncMock(side_effect=RuntimeError("nav failed"))

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=[]):
            with pytest.raises(RuntimeError, match="nav failed"):
                await session.on_NavigateToUrlEvent(event)

        dispatch_calls = session.event_bus.dispatch.call_args_list
        nav_complete_calls = [c for c in dispatch_calls if isinstance(c[0][0], NavigationCompleteEvent)]
        assert any(e[0][0].error_message for e in nav_complete_calls if e[0][0].error_message)


@pytest.mark.asyncio
class TestOnSwitchTabEvent:
    """Tests for on_SwitchTabEvent (lines 727-763)."""

    async def test_switch_tab_no_focus_raises(self):
        session = _make_browser_session()
        session.agent_focus = None
        with pytest.raises(RuntimeError, match="Cannot switch tabs"):
            await session.on_SwitchTabEvent(SwitchTabEvent(target_id="t1"))

    async def test_switch_tab_with_target_id(self):
        """Switch to specific target (lines 748-763)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        new_tid = "NEW-TARGET-0000ABCD"

        new_focus = _make_cdp_session(client=client, target_id=new_tid, url="https://switched.com")

        pages = [{"targetId": new_tid, "url": "https://switched.com"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=pages):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=new_focus):
                result = await session.on_SwitchTabEvent(SwitchTabEvent(target_id=new_tid))

        assert result == new_tid

    async def test_switch_tab_none_uses_last_page(self):
        """target_id=None switches to most recently opened (lines 733-737)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        last_tid = "LAST-TARGET-00000000"

        new_focus = _make_cdp_session(client=client, target_id=last_tid, url="https://last.com")

        pages = [
            {"targetId": "first-target-00000000", "url": "https://first.com"},
            {"targetId": last_tid, "url": "https://last.com"},
        ]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=pages):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=new_focus):
                result = await session.on_SwitchTabEvent(SwitchTabEvent(target_id=None))

        assert result == last_tid

    async def test_switch_tab_none_no_pages_creates_new(self):
        """target_id=None with no pages creates new tab (lines 739-746)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        session._cdp_client_root = client
        new_tid = "CREATED-TARGET-0000"
        client.send.Target.createTarget = AsyncMock(return_value={"targetId": new_tid})

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=[]):
            result = await session.on_SwitchTabEvent(SwitchTabEvent(target_id=None))

        assert result == new_tid
        dispatch_calls = session.event_bus.dispatch.call_args_list
        tab_created = [c for c in dispatch_calls if isinstance(c[0][0], TabCreatedEvent)]
        focus_changed = [c for c in dispatch_calls if isinstance(c[0][0], AgentFocusChangedEvent)]
        assert len(tab_created) >= 1
        assert len(focus_changed) >= 1


@pytest.mark.asyncio
class TestOnCloseTabEvent:
    """Tests for on_CloseTabEvent (lines 765-778)."""

    async def test_close_tab_dispatches_tab_closed(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=focus):
            await session.on_CloseTabEvent(CloseTabEvent(target_id="tid-to-close"))

        dispatch_calls = session.event_bus.dispatch.call_args_list
        tab_closed = [c for c in dispatch_calls if isinstance(c[0][0], TabClosedEvent)]
        assert len(tab_closed) >= 1

    async def test_close_tab_handles_already_closed(self):
        """If target is already closed, should not raise (line 776)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        focus.cdp_client.send.Target.closeTarget = AsyncMock(side_effect=Exception("already closed"))

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=focus):
            await session.on_CloseTabEvent(CloseTabEvent(target_id="closed-target"))


@pytest.mark.asyncio
class TestOnTabCreatedEvent:
    """Tests for on_TabCreatedEvent (lines 780-797)."""

    async def test_tab_created_applies_viewport(self):
        profile = BrowserProfile(
            cdp_url="ws://localhost:9222",
            is_local=False,
            viewport=ViewportSize(width=1280, height=720),
            no_viewport=False,
            device_scale_factor=2.0,
        )
        session = BrowserSession(browser_profile=profile)
        session.event_bus.dispatch = MagicMock()

        with patch.object(BrowserSession, "_cdp_set_viewport", new_callable=AsyncMock) as mock_vp:
            await session.on_TabCreatedEvent(TabCreatedEvent(target_id="t1", url="about:blank"))
            mock_vp.assert_awaited_once_with(1280, 720, 2.0, target_id="t1")

    async def test_tab_created_no_viewport_skips(self):
        profile = BrowserProfile(
            cdp_url="ws://localhost:9222", is_local=False, headless=False, no_viewport=True,
        )
        session = BrowserSession(browser_profile=profile)
        session.event_bus.dispatch = _make_dispatch_mock()
        assert session.browser_profile.viewport is None
        with patch.object(BrowserSession, "_cdp_set_viewport", new_callable=AsyncMock) as mock_vp:
            await session.on_TabCreatedEvent(TabCreatedEvent(target_id="t1", url="about:blank"))
            mock_vp.assert_not_awaited()

    async def test_tab_created_viewport_error_logs_warning(self):
        """Viewport set failure logs warning but does not raise (line 797)."""
        profile = BrowserProfile(
            cdp_url="ws://localhost:9222",
            is_local=False,
            viewport=ViewportSize(width=800, height=600),
            no_viewport=False,
        )
        session = BrowserSession(browser_profile=profile)
        session.event_bus.dispatch = MagicMock()

        with patch.object(BrowserSession, "_cdp_set_viewport", new_callable=AsyncMock, side_effect=Exception("viewport fail")):
            await session.on_TabCreatedEvent(TabCreatedEvent(target_id="t1", url="about:blank"))


@pytest.mark.asyncio
class TestOnTabClosedEvent:
    """Tests for on_TabClosedEvent (lines 799-809)."""

    async def test_tab_closed_switches_if_was_current(self):
        session = _make_browser_session()
        _setup_session_with_cdp(session)
        current_tid = session.agent_focus.target_id

        await session.on_TabClosedEvent(TabClosedEvent(target_id=current_tid))

        dispatch_calls = session.event_bus.dispatch.call_args_list
        switch_events = [c for c in dispatch_calls if isinstance(c[0][0], SwitchTabEvent)]
        assert len(switch_events) >= 1
        assert switch_events[0][0][0].target_id is None

    async def test_tab_closed_other_tab_no_switch(self):
        session = _make_browser_session()
        _setup_session_with_cdp(session)

        await session.on_TabClosedEvent(TabClosedEvent(target_id="some-other-target"))

        dispatch_calls = session.event_bus.dispatch.call_args_list
        switch_events = [c for c in dispatch_calls if isinstance(c[0][0], SwitchTabEvent)]
        assert len(switch_events) == 0

    async def test_tab_closed_no_focus(self):
        session = _make_browser_session()
        session.agent_focus = None
        await session.on_TabClosedEvent(TabClosedEvent(target_id="any"))


@pytest.mark.asyncio
class TestOnAgentFocusChangedEvent:
    """Tests for on_AgentFocusChangedEvent (lines 811-868)."""

    async def test_focus_changed_updates_focus(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        new_tid = "NEWTARGET-1234ABCD"

        new_focus = _make_cdp_session(client=client, target_id=new_tid, url="https://new.com")

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=[]):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=new_focus):
                await session.on_AgentFocusChangedEvent(
                    AgentFocusChangedEvent(target_id=new_tid, url="https://new.com")
                )

        assert session.agent_focus is new_focus

    async def test_focus_changed_clears_cache(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        session._cached_browser_state_summary = "something"
        session._cached_selector_map = {1: MagicMock()}

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=[]):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=focus):
                await session.on_AgentFocusChangedEvent(
                    AgentFocusChangedEvent(target_id=focus.target_id, url="https://example.com")
                )

        assert session._cached_browser_state_summary is None
        assert len(session._cached_selector_map) == 0

    async def test_focus_changed_clears_dom_watchdog_cache(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        dom_wd = MagicMock()
        session._dom_watchdog = dom_wd

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=[]):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=focus):
                await session.on_AgentFocusChangedEvent(
                    AgentFocusChangedEvent(target_id=focus.target_id, url="https://example.com")
                )

        dom_wd.clear_cache.assert_called_once()

    async def test_focus_changed_no_target_id_raises(self):
        """AgentFocusChangedEvent with empty target_id should raise (line 833)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=[]):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=focus):
                with pytest.raises(RuntimeError, match="no target_id"):
                    await session.on_AgentFocusChangedEvent(
                        AgentFocusChangedEvent(target_id="", url="https://example.com")
                    )

    async def test_focus_changed_tab_unresponsive_falls_back(self):
        """Unresponsive tab should fall back to last page (lines 849-857)."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        tid = "RESPONSIVE-TARGET-00"

        responsive_focus = _make_cdp_session(client=client, target_id=tid, url="https://resp.com")

        # Runtime.evaluate will raise -> triggers fallback
        client.send.Runtime.evaluate = AsyncMock(side_effect=Exception("tab crashed"))

        fallback_focus = _make_cdp_session(client=client, target_id="FALLBACK-TARGET-0000", url="https://fallback.com")

        call_count = [0]

        async def mock_get_or_create(target_id=None, focus=True):
            call_count[0] += 1
            if call_count[0] == 1:
                return responsive_focus
            return fallback_focus

        fallback_pages = [{"targetId": "FALLBACK-TARGET-0000", "url": "https://fallback.com"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=fallback_pages):
            with patch.object(BrowserSession, "get_or_create_cdp_session", side_effect=mock_get_or_create):
                with pytest.raises(Exception, match="tab crashed"):
                    await session.on_AgentFocusChangedEvent(
                        AgentFocusChangedEvent(target_id=tid, url="https://resp.com")
                    )

    async def test_focus_changed_dispatches_navigation_complete(self):
        """Should dispatch NavigationCompleteEvent when target_id and url provided."""
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=[]):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=focus):
                await session.on_AgentFocusChangedEvent(
                    AgentFocusChangedEvent(target_id=focus.target_id, url="https://example.com")
                )

        dispatch_calls = session.event_bus.dispatch.call_args_list
        nav_complete = [c for c in dispatch_calls if isinstance(c[0][0], NavigationCompleteEvent)]
        assert len(nav_complete) >= 1


@pytest.mark.asyncio
class TestOnFileDownloadedEvent:
    """Tests for on_FileDownloadedEvent (lines 872-882)."""

    async def test_tracks_downloaded_file(self):
        session = _make_browser_session()
        event = FileDownloadedEvent(url="https://dl.com/f.pdf", path="/tmp/f.pdf", file_name="f.pdf", file_size=1234)
        await session.on_FileDownloadedEvent(event)
        assert "/tmp/f.pdf" in session._downloaded_files

    async def test_does_not_duplicate_tracked_file(self):
        session = _make_browser_session()
        session._downloaded_files.append("/tmp/f.pdf")
        event = FileDownloadedEvent(url="https://dl.com/f.pdf", path="/tmp/f.pdf", file_name="f.pdf", file_size=1234)
        await session.on_FileDownloadedEvent(event)
        assert session._downloaded_files.count("/tmp/f.pdf") == 1

    async def test_no_path_logs_warning(self):
        session = _make_browser_session()
        event = FileDownloadedEvent(url="https://dl.com/f.pdf", path="", file_name="f.pdf", file_size=1234)
        await session.on_FileDownloadedEvent(event)
        assert len(session._downloaded_files) == 0


@pytest.mark.asyncio
class TestOnBrowserStopEvent:
    """Tests for on_BrowserStopEvent (lines 884-912)."""

    async def test_stop_keep_alive_returns_early(self):
        """keep_alive=True and force=False should dispatch BrowserStoppedEvent and return."""
        profile = BrowserProfile(cdp_url="ws://localhost:9222", is_local=False, keep_alive=True)
        session = BrowserSession(browser_profile=profile)
        session.event_bus.dispatch = MagicMock()

        await session.on_BrowserStopEvent(BrowserStopEvent(force=False))

        dispatch_calls = session.event_bus.dispatch.call_args_list
        stopped = [c for c in dispatch_calls if isinstance(c[0][0], BrowserStoppedEvent)]
        assert len(stopped) >= 1
        assert "keep_alive" in stopped[0][0][0].reason.lower()

    async def test_stop_force_resets_state(self):
        session = _make_browser_session()
        _setup_session_with_cdp(session)

        with patch.object(BrowserSession, "reset", new_callable=AsyncMock) as mock_reset:
            await session.on_BrowserStopEvent(BrowserStopEvent(force=True))
            mock_reset.assert_awaited_once()

    async def test_stop_exception_dispatches_error(self):
        session = _make_browser_session()
        _setup_session_with_cdp(session)

        with patch.object(BrowserSession, "reset", new_callable=AsyncMock, side_effect=RuntimeError("reset fail")):
            await session.on_BrowserStopEvent(BrowserStopEvent(force=True))

        dispatch_calls = session.event_bus.dispatch.call_args_list
        error_events = [c for c in dispatch_calls if isinstance(c[0][0], BrowserErrorEvent)]
        assert len(error_events) >= 1
        assert "BrowserStopEventError" in error_events[0][0][0].error_type

    async def test_stop_resets_cdp_url_for_local(self):
        profile = BrowserProfile(cdp_url="ws://localhost:5555", is_local=True)
        session = BrowserSession(browser_profile=profile)
        session.event_bus.dispatch = MagicMock()
        _setup_session_with_cdp(session)

        with patch.object(BrowserSession, "reset", new_callable=AsyncMock):
            await session.on_BrowserStopEvent(BrowserStopEvent(force=True))

        assert session.browser_profile.cdp_url is None


# ===========================================================================
# BrowserSession public methods (CDP wrappers)
# ===========================================================================


@pytest.mark.asyncio
class TestNewPage:
    """Tests for new_page (lines 921-933)."""

    async def test_new_page_creates_target(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
            page = await session.new_page("https://example.com")
        assert page is not None
        client.send.Target.createTarget.assert_awaited_once()

    async def test_new_page_default_url(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
            page = await session.new_page()
        call_args = client.send.Target.createTarget.call_args
        assert call_args[0][0]["url"] == "about:blank"


@pytest.mark.asyncio
class TestGetCurrentPage:
    """Tests for get_current_page (lines 935-944)."""

    async def test_get_current_page_returns_page(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch.object(BrowserSession, "get_current_target_info", new_callable=AsyncMock, return_value={"targetId": "t1", "url": "https://a.com", "title": "A"}):
            page = await session.get_current_page()
        assert page is not None

    async def test_get_current_page_returns_none_when_no_target(self):
        session = _make_browser_session()
        with patch.object(BrowserSession, "get_current_target_info", new_callable=AsyncMock, return_value=None):
            page = await session.get_current_page()
        assert page is None


@pytest.mark.asyncio
class TestMustGetCurrentPage:
    """Tests for must_get_current_page (lines 946-952)."""

    async def test_must_get_raises_when_no_page(self):
        session = _make_browser_session()
        with patch.object(BrowserSession, "get_current_page", new_callable=AsyncMock, return_value=None):
            with pytest.raises(RuntimeError, match="No current target"):
                await session.must_get_current_page()

    async def test_must_get_returns_page(self):
        session = _make_browser_session()
        mock_page = MagicMock()
        with patch.object(BrowserSession, "get_current_page", new_callable=AsyncMock, return_value=mock_page):
            result = await session.must_get_current_page()
        assert result is mock_page


@pytest.mark.asyncio
class TestGetPages:
    """Tests for get_pages (lines 954-966)."""

    async def test_get_pages_returns_page_and_iframe_targets(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [
                    {"targetId": "t1", "url": "https://a.com", "title": "A", "type": "page"},
                    {"targetId": "t2", "url": "https://b.com", "title": "B", "type": "iframe"},
                    {"targetId": "t3", "url": "", "title": "", "type": "service_worker"},
                ]
            }
        )

        with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
            pages = await session.get_pages()
        assert len(pages) == 2

    async def test_get_pages_empty(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)
        client.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})

        with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
            pages = await session.get_pages()
        assert pages == []


@pytest.mark.asyncio
class TestClosePage:
    """Tests for close_page (lines 968-981)."""

    async def test_close_page_by_target_id_string(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
            await session.close_page("target-to-close")
        client.send.Target.closeTarget.assert_awaited_once()

    async def test_close_page_by_page_object(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        from openbrowser.actor.page import Page
        mock_page = MagicMock(spec=Page)
        mock_page._target_id = "page-target-id"

        with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
            await session.close_page(mock_page)
        call_args = client.send.Target.closeTarget.call_args
        assert call_args[0][0]["targetId"] == "page-target-id"


@pytest.mark.asyncio
class TestCookies:
    """Tests for cookies and clear_cookies (lines 983-996)."""

    async def test_cookies_no_filter(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        # Mock the import that fails in current cdp_use version
        import sys
        from unittest.mock import MagicMock as _MM
        fake_module = _MM()
        fake_module.GetCookiesParameters = dict
        with patch.dict(sys.modules, {"cdp_use.cdp.network.library": fake_module}):
            with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
                result = await session.cookies()
        assert result == []

    async def test_cookies_with_urls(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        import sys
        from unittest.mock import MagicMock as _MM
        fake_module = _MM()
        fake_module.GetCookiesParameters = dict
        with patch.dict(sys.modules, {"cdp_use.cdp.network.library": fake_module}):
            with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
                await session.cookies(urls=["https://example.com"])
        call_args = client.send.Network.getCookies.call_args
        assert call_args[0][0]["urls"] == ["https://example.com"]

    async def test_clear_cookies(self):
        session = _make_browser_session()
        client, focus, sm = _setup_session_with_cdp(session)

        with patch("openbrowser.browser.session.BrowserSession.cdp_client", new_callable=PropertyMock, return_value=client):
            await session.clear_cookies()
        client.send.Network.clearBrowserCookies.assert_awaited_once()


# ===========================================================================
# BrowserSession.cdp_client property
# ===========================================================================


class TestCDPClientProperty:
    """Tests for cdp_client property (lines 915-919)."""

    def test_cdp_client_raises_when_not_connected(self):
        session = _make_browser_session()
        session._cdp_client_root = None
        with pytest.raises(AssertionError, match="CDP client not initialized"):
            _ = session.cdp_client

    def test_cdp_client_returns_root(self):
        session = _make_browser_session()
        mock_client = _make_mock_cdp_client()
        session._cdp_client_root = mock_client
        assert session.cdp_client is mock_client


# ===========================================================================
# BrowserSession.start, kill, and stop
# ===========================================================================


@pytest.mark.asyncio
class TestStartKillStop:
    """Tests for start(), kill(), and stop() methods."""

    async def test_kill_dispatches_stop_and_resets(self):
        session = _make_browser_session()
        _setup_session_with_cdp(session)

        dispatch_mock = _make_dispatch_mock()
        session.event_bus.dispatch = dispatch_mock
        stop_mock = AsyncMock()
        session.event_bus.stop = stop_mock

        with patch.object(BrowserSession, "reset", new_callable=AsyncMock) as mock_reset:
            await session.kill()

        mock_reset.assert_awaited_once()
        stop_mock.assert_awaited_once()

    async def test_stop_dispatches_stop_event_non_force(self):
        session = _make_browser_session()
        _setup_session_with_cdp(session)

        dispatch_mock = _make_dispatch_mock()
        session.event_bus.dispatch = dispatch_mock
        session.event_bus.stop = AsyncMock()

        with patch.object(BrowserSession, "reset", new_callable=AsyncMock):
            await session.stop()

        dispatched = dispatch_mock._dispatched_events
        stop_events = [e for e in dispatched if isinstance(e, BrowserStopEvent)]
        assert any(not e.force for e in stop_events)
