"""Tests for BrowserSession middle third (lines ~1000-2050).

Covers: export_storage_state, get_or_create_cdp_session, current_target_id,
current_session_id, get_browser_state_summary, get_state_as_text,
attach_all_watchdogs, connect, _setup_proxy_auth, get_tabs,
get_current_target_info, get_current_page_url/title, navigate_to,
get_dom_element_by_index, update_cached_selector_map, get_element_by_index,
get_target_id_from_tab_id, _is_target_valid, get_target_id_from_url,
get_most_recently_opened_target_id, is_file_input, get_selector_map,
get_index_by_id, get_index_by_class, remove_highlights, get_element_coordinates.
"""

import asyncio
import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from openbrowser.browser.session import BrowserSession, CDPSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_with_mocks(**overrides):
    """Create a BrowserSession with mocked internals.

    Returns (session, mocks_dict) where mocks_dict has commonly needed mocks.
    """
    # Build a real BrowserSession (validation passes with defaults)
    session = BrowserSession(cdp_url="ws://127.0.0.1:9222")

    # Replace EventBus.dispatch so no real events fire
    session.event_bus.dispatch = MagicMock()

    # Build mock CDP root client
    cdp_root = MagicMock()
    cdp_root.send = MagicMock()
    cdp_root.start = AsyncMock()
    cdp_root.register = MagicMock()

    # Target methods
    cdp_root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})
    cdp_root.send.Target.getTargetInfo = AsyncMock(
        return_value={"targetInfo": {"targetId": "t1", "url": "https://example.com", "title": "Example", "type": "page"}}
    )
    cdp_root.send.Target.attachToTarget = AsyncMock(return_value={"sessionId": "s1"})
    cdp_root.send.Target.setAutoAttach = AsyncMock()
    cdp_root.send.Target.createTarget = AsyncMock(return_value={"targetId": "new-target"})
    cdp_root.send.Target.closeTarget = AsyncMock()
    cdp_root.send.Target.activateTarget = AsyncMock()

    # Fetch methods
    cdp_root.send.Fetch.enable = AsyncMock()
    cdp_root.send.Fetch.continueWithAuth = AsyncMock()
    cdp_root.send.Fetch.continueRequest = AsyncMock()

    # Network methods
    cdp_root.send.Network.getCookies = AsyncMock(return_value={"cookies": []})
    cdp_root.send.Network.clearBrowserCookies = AsyncMock()

    # Storage methods
    cdp_root.send.Storage.getCookies = AsyncMock(return_value={"cookies": []})

    # Runtime methods
    cdp_root.send.Runtime.runIfWaitingForDebugger = AsyncMock()
    cdp_root.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": {"removed": 0}}})
    cdp_root.send.Runtime.callFunctionOn = AsyncMock(
        return_value={"result": {"value": {"x": 10, "y": 20, "width": 100, "height": 50}}}
    )

    # DOM methods
    cdp_root.send.DOM.getContentQuads = AsyncMock(return_value={"quads": []})
    cdp_root.send.DOM.getBoxModel = AsyncMock(return_value={})
    cdp_root.send.DOM.resolveNode = AsyncMock(return_value={"object": {"objectId": "obj1"}})

    # Register handlers
    cdp_root.register.Fetch.authRequired = MagicMock()
    cdp_root.register.Fetch.requestPaused = MagicMock()
    cdp_root.register.Target.attachedToTarget = MagicMock()
    cdp_root.register.Target.detachedFromTarget = MagicMock()

    session._cdp_client_root = cdp_root

    # Build mock CDPSession for agent_focus
    mock_cdp_session = MagicMock(spec=CDPSession)
    mock_cdp_session.target_id = "abcdef1234567890abcdef1234567890"
    mock_cdp_session.session_id = "sess-1234"
    mock_cdp_session.cdp_client = cdp_root
    mock_cdp_session.title = "Example"
    mock_cdp_session.url = "https://example.com"
    session.agent_focus = mock_cdp_session

    # Session manager
    mock_sm = MagicMock()
    mock_sm.get_session_for_target = AsyncMock(return_value=mock_cdp_session)
    mock_sm.validate_session = AsyncMock(return_value=True)
    mock_sm.start_monitoring = AsyncMock()
    mock_sm.clear = AsyncMock()
    session._session_manager = mock_sm

    # Apply overrides
    for key, val in overrides.items():
        setattr(session, key, val)

    mocks = {
        "cdp_root": cdp_root,
        "cdp_session": mock_cdp_session,
        "session_manager": mock_sm,
    }
    return session, mocks


# ===========================================================================
# export_storage_state (lines 1010-1041)
# ===========================================================================


@pytest.mark.asyncio
class TestExportStorageState:
    """Tests for BrowserSession.export_storage_state."""

    async def test_export_returns_storage_dict(self):
        session, mocks = _make_session_with_mocks()
        # Mock _cdp_get_cookies to return sample cookies
        fake_cookies = [
            {
                "name": "sid",
                "value": "abc",
                "domain": ".example.com",
                "path": "/",
                "expires": 9999999,
                "httpOnly": True,
                "secure": True,
                "sameSite": "Strict",
            }
        ]
        session._cdp_get_cookies = AsyncMock(return_value=fake_cookies)

        result = await session.export_storage_state()
        assert "cookies" in result
        assert len(result["cookies"]) == 1
        assert result["cookies"][0]["name"] == "sid"
        assert result["cookies"][0]["sameSite"] == "Strict"
        assert result["origins"] == []

    async def test_export_writes_to_file(self, tmp_path):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_cookies = AsyncMock(return_value=[
            {"name": "a", "value": "1", "domain": ".d.com", "path": "/"}
        ])

        output_file = tmp_path / "state.json"
        result = await session.export_storage_state(output_path=str(output_file))

        assert output_file.exists()
        import json
        saved = json.loads(output_file.read_text())
        assert saved["cookies"][0]["name"] == "a"
        # Defaults for missing keys
        assert saved["cookies"][0]["expires"] == -1
        assert saved["cookies"][0]["httpOnly"] is False

    async def test_export_no_cookies(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_cookies = AsyncMock(return_value=[])

        result = await session.export_storage_state()
        assert result["cookies"] == []


# ===========================================================================
# get_or_create_cdp_session (lines 1043-1117)
# ===========================================================================


@pytest.mark.asyncio
class TestGetOrCreateCdpSession:
    """Tests for BrowserSession.get_or_create_cdp_session."""

    async def test_returns_session_for_current_focus(self):
        session, mocks = _make_session_with_mocks()
        result = await session.get_or_create_cdp_session()
        assert result is mocks["cdp_session"]

    async def test_returns_session_for_explicit_target(self):
        session, mocks = _make_session_with_mocks()
        specific_session = MagicMock(spec=CDPSession)
        specific_session.target_id = "specific-target-id-1234"
        specific_session.session_id = "specific-sess"
        specific_session.cdp_client = mocks["cdp_root"]
        mocks["session_manager"].get_session_for_target = AsyncMock(return_value=specific_session)

        result = await session.get_or_create_cdp_session(target_id="specific-target-id-1234", focus=False)
        assert result is specific_session

    async def test_waits_for_session_then_succeeds(self):
        """Session not available immediately, appears after retries."""
        session, mocks = _make_session_with_mocks()
        target_session = MagicMock(spec=CDPSession)
        target_session.target_id = "delayed-target-1234567890"
        target_session.session_id = "delayed-sess"
        target_session.cdp_client = mocks["cdp_root"]

        # First call returns None, second returns the session
        call_count = 0

        async def _get_session(tid):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return None
            return target_session

        mocks["session_manager"].get_session_for_target = AsyncMock(side_effect=_get_session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await session.get_or_create_cdp_session(target_id="delayed-target-1234567890", focus=False)
        assert result is target_session

    async def test_raises_when_target_never_appears(self):
        """Session never appears -- should raise ValueError."""
        session, mocks = _make_session_with_mocks()
        mocks["session_manager"].get_session_for_target = AsyncMock(return_value=None)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="not found"):
                await session.get_or_create_cdp_session(target_id="missing-target")

    async def test_raises_when_session_invalid(self):
        session, mocks = _make_session_with_mocks()
        mocks["session_manager"].validate_session = AsyncMock(return_value=False)

        with pytest.raises(ValueError, match="has detached"):
            await session.get_or_create_cdp_session()

    async def test_switches_focus_for_page_target(self):
        session, mocks = _make_session_with_mocks()
        new_session = MagicMock(spec=CDPSession)
        new_session.target_id = "new-page-target-1234567890"
        new_session.session_id = "new-sess"
        new_session.cdp_client = mocks["cdp_root"]

        mocks["session_manager"].get_session_for_target = AsyncMock(return_value=new_session)
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [{"targetId": "new-page-target-1234567890", "type": "page"}]
        })

        result = await session.get_or_create_cdp_session(target_id="new-page-target-1234567890", focus=True)
        assert session.agent_focus is new_session

    async def test_ignores_focus_for_iframe_target(self):
        session, mocks = _make_session_with_mocks()
        iframe_session = MagicMock(spec=CDPSession)
        iframe_session.target_id = "iframe-target-id-1234567890"
        iframe_session.session_id = "iframe-sess"
        iframe_session.cdp_client = mocks["cdp_root"]

        mocks["session_manager"].get_session_for_target = AsyncMock(return_value=iframe_session)
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [{"targetId": "iframe-target-id-1234567890", "type": "iframe"}]
        })

        original_focus = session.agent_focus
        await session.get_or_create_cdp_session(target_id="iframe-target-id-1234567890", focus=True)
        # Focus should NOT change for iframe
        assert session.agent_focus is original_focus

    async def test_runtime_run_if_waiting_exception_swallowed(self):
        session, mocks = _make_session_with_mocks()
        mocks["cdp_root"].send.Runtime.runIfWaitingForDebugger = AsyncMock(side_effect=Exception("not waiting"))

        # Should not raise
        result = await session.get_or_create_cdp_session()
        assert result is mocks["cdp_session"]


# ===========================================================================
# current_target_id / current_session_id (lines 1119-1125)
# ===========================================================================


class TestCurrentProperties:
    def test_current_target_id_with_focus(self):
        session, mocks = _make_session_with_mocks()
        assert session.current_target_id == mocks["cdp_session"].target_id

    def test_current_target_id_without_focus(self):
        session, _ = _make_session_with_mocks()
        session.agent_focus = None
        assert session.current_target_id is None

    def test_current_session_id_with_focus(self):
        session, mocks = _make_session_with_mocks()
        assert session.current_session_id == mocks["cdp_session"].session_id

    def test_current_session_id_without_focus(self):
        session, _ = _make_session_with_mocks()
        session.agent_focus = None
        assert session.current_session_id is None


# ===========================================================================
# get_browser_state_summary (lines 1130-1167)
# ===========================================================================


@pytest.mark.asyncio
class TestGetBrowserStateSummary:

    async def test_uses_cache_when_valid(self):
        session, mocks = _make_session_with_mocks()

        mock_dom_state = MagicMock()
        mock_dom_state.selector_map = {0: MagicMock()}

        cached = MagicMock()
        cached.dom_state = mock_dom_state
        cached.screenshot = "base64data"
        session._cached_browser_state_summary = cached

        result = await session.get_browser_state_summary(cached=True)
        assert result is cached

    async def test_skips_cache_when_no_screenshot_needed(self):
        session, mocks = _make_session_with_mocks()

        mock_dom_state = MagicMock()
        mock_dom_state.selector_map = {0: MagicMock()}

        cached = MagicMock()
        cached.dom_state = mock_dom_state
        cached.screenshot = None  # No screenshot in cache

        session._cached_browser_state_summary = cached

        # Set up a fresh result from event dispatch
        fresh_result = MagicMock()
        fresh_result.dom_state = MagicMock()

        mock_event = MagicMock()
        mock_event.event_result = AsyncMock(return_value=fresh_result)
        session.event_bus.dispatch = MagicMock(return_value=mock_event)

        result = await session.get_browser_state_summary(include_screenshot=True, cached=True)
        assert result is fresh_result

    async def test_skips_cache_when_empty_selector_map(self):
        session, mocks = _make_session_with_mocks()

        mock_dom_state = MagicMock()
        mock_dom_state.selector_map = {}  # Empty selector map

        cached = MagicMock()
        cached.dom_state = mock_dom_state
        cached.screenshot = "data"

        session._cached_browser_state_summary = cached

        fresh_result = MagicMock()
        fresh_result.dom_state = MagicMock()

        mock_event = MagicMock()
        mock_event.event_result = AsyncMock(return_value=fresh_result)
        session.event_bus.dispatch = MagicMock(return_value=mock_event)

        result = await session.get_browser_state_summary(cached=True)
        assert result is fresh_result


# ===========================================================================
# get_state_as_text (lines 1169-1174)
# ===========================================================================


@pytest.mark.asyncio
class TestGetStateAsText:

    async def test_returns_llm_representation(self):
        session, mocks = _make_session_with_mocks()

        mock_dom_state = MagicMock()
        mock_dom_state.llm_representation = MagicMock(return_value="[0] <button>Click me</button>")

        mock_summary = MagicMock()
        mock_summary.dom_state = mock_dom_state

        with patch.object(BrowserSession, "get_browser_state_summary", new_callable=AsyncMock, return_value=mock_summary):
            result = await session.get_state_as_text()

        assert result == "[0] <button>Click me</button>"


# ===========================================================================
# attach_all_watchdogs (lines 1176-1311)
# ===========================================================================


@pytest.mark.asyncio
class TestAttachAllWatchdogs:

    async def test_skips_duplicate_attachment(self):
        session, mocks = _make_session_with_mocks()
        session._watchdogs_attached = True

        # Should return early without importing watchdogs
        await session.attach_all_watchdogs()
        # Verify no watchdog was created
        assert session._downloads_watchdog is None

    async def test_attaches_watchdogs(self):
        session, mocks = _make_session_with_mocks()

        # Mock all the watchdog classes
        watchdog_patches = [
            "openbrowser.browser.watchdogs.aboutblank_watchdog.AboutBlankWatchdog",
            "openbrowser.browser.watchdogs.default_action_watchdog.DefaultActionWatchdog",
            "openbrowser.browser.watchdogs.dom_watchdog.DOMWatchdog",
            "openbrowser.browser.watchdogs.downloads_watchdog.DownloadsWatchdog",
            "openbrowser.browser.watchdogs.local_browser_watchdog.LocalBrowserWatchdog",
            "openbrowser.browser.watchdogs.permissions_watchdog.PermissionsWatchdog",
            "openbrowser.browser.watchdogs.popups_watchdog.PopupsWatchdog",
            "openbrowser.browser.watchdogs.recording_watchdog.RecordingWatchdog",
            "openbrowser.browser.watchdogs.screenshot_watchdog.ScreenshotWatchdog",
            "openbrowser.browser.watchdogs.security_watchdog.SecurityWatchdog",
            "openbrowser.browser.watchdogs.storage_state_watchdog.StorageStateWatchdog",
        ]

        patchers = []
        mock_classes = []
        for path in watchdog_patches:
            p = patch(path)
            mock_cls = p.start()
            mock_instance = MagicMock()
            mock_instance.attach_to_session = MagicMock()
            mock_cls.model_rebuild = MagicMock()
            mock_cls.return_value = mock_instance
            patchers.append(p)
            mock_classes.append(mock_cls)

        try:
            await session.attach_all_watchdogs()
            assert session._watchdogs_attached is True
            # Verify at least some watchdogs were attached
            assert session._downloads_watchdog is not None
            assert session._dom_watchdog is not None
        finally:
            for p in patchers:
                p.stop()

    async def test_storage_state_watchdog_disabled_without_config(self):
        session, mocks = _make_session_with_mocks()
        # Force both to None to trigger the disabled code path (line 1236)
        session.browser_profile.__dict__['storage_state'] = None
        session.browser_profile.__dict__['user_data_dir'] = None

        watchdog_patches = [
            "openbrowser.browser.watchdogs.aboutblank_watchdog.AboutBlankWatchdog",
            "openbrowser.browser.watchdogs.default_action_watchdog.DefaultActionWatchdog",
            "openbrowser.browser.watchdogs.dom_watchdog.DOMWatchdog",
            "openbrowser.browser.watchdogs.downloads_watchdog.DownloadsWatchdog",
            "openbrowser.browser.watchdogs.local_browser_watchdog.LocalBrowserWatchdog",
            "openbrowser.browser.watchdogs.permissions_watchdog.PermissionsWatchdog",
            "openbrowser.browser.watchdogs.popups_watchdog.PopupsWatchdog",
            "openbrowser.browser.watchdogs.recording_watchdog.RecordingWatchdog",
            "openbrowser.browser.watchdogs.screenshot_watchdog.ScreenshotWatchdog",
            "openbrowser.browser.watchdogs.security_watchdog.SecurityWatchdog",
            "openbrowser.browser.watchdogs.storage_state_watchdog.StorageStateWatchdog",
        ]
        patchers = []
        for path in watchdog_patches:
            p = patch(path)
            mock_cls = p.start()
            mock_instance = MagicMock()
            mock_instance.attach_to_session = MagicMock()
            mock_cls.model_rebuild = MagicMock()
            mock_cls.return_value = mock_instance
            patchers.append(p)

        try:
            await session.attach_all_watchdogs()
            # StorageStateWatchdog should NOT be created
            assert session._storage_state_watchdog is None
        finally:
            for p in patchers:
                p.stop()


# ===========================================================================
# connect (lines 1313-1481)
# ===========================================================================


@pytest.mark.asyncio
class TestConnect:

    async def test_connect_raises_without_cdp_url(self):
        session = BrowserSession()
        session.event_bus.dispatch = MagicMock()
        session.browser_profile.cdp_url = None

        with pytest.raises(RuntimeError, match="Cannot setup CDP connection without CDP URL"):
            await session.connect(cdp_url=None)

    async def test_connect_http_url_fetches_ws_url(self):
        """When given an HTTP URL, connect should query /json/version for the WS URL."""
        session, mocks = _make_session_with_mocks()

        mock_response = MagicMock()
        mock_response.json.return_value = {"webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/abc"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        session.browser_profile.cdp_url = "http://127.0.0.1:9222"

        page_target = {
            "targetId": "target-page-1234",
            "type": "page",
            "url": "about:blank",
            "title": "",
        }

        mocks["cdp_root"].start = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]), \
             patch("openbrowser.browser.session_manager.SessionManager") as MockSM, \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(BrowserSession, "_setup_proxy_auth", new_callable=AsyncMock):

            mock_sm_instance = MagicMock()
            mock_sm_instance.start_monitoring = AsyncMock()
            mock_sm_instance.get_session_for_target = AsyncMock(return_value=mocks["cdp_session"])
            MockSM.return_value = mock_sm_instance

            mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={
                "targetInfos": [page_target]
            })

            result = await session.connect()
            assert result is session
            mock_client.get.assert_called()

    async def test_connect_http_timeout(self):
        """HTTP endpoint that never responds should timeout."""
        session, mocks = _make_session_with_mocks()
        session.browser_profile.cdp_url = "http://127.0.0.1:9222"

        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Make time always exceed max_wait
        fake_loop = MagicMock()
        fake_loop.time = MagicMock(side_effect=[0.0, 31.0])

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("asyncio.get_event_loop", return_value=fake_loop), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError, match="did not respond"):
                await session.connect()

    async def test_connect_no_pages_creates_blank(self):
        """When no page targets exist, connect should create about:blank."""
        session, mocks = _make_session_with_mocks()

        with patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]), \
             patch("openbrowser.browser.session_manager.SessionManager") as MockSM, \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(BrowserSession, "_setup_proxy_auth", new_callable=AsyncMock), \
             patch.object(BrowserSession, "_is_valid_target", return_value=False):

            mock_sm_instance = MagicMock()
            mock_sm_instance.start_monitoring = AsyncMock()
            mock_sm_instance.get_session_for_target = AsyncMock(return_value=mocks["cdp_session"])
            MockSM.return_value = mock_sm_instance

            mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})
            mocks["cdp_root"].start = AsyncMock()
            mocks["cdp_root"].send.Target.createTarget = AsyncMock(return_value={"targetId": "new-blank"})

            result = await session.connect()
            assert result is session
            mocks["cdp_root"].send.Target.createTarget.assert_called()

    async def test_connect_fatal_error_cleans_up(self):
        """Fatal CDP error should clean up and raise RuntimeError."""
        session, mocks = _make_session_with_mocks()

        mock_cdp = MagicMock()
        mock_cdp.start = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("openbrowser.browser.session.CDPClient", return_value=mock_cdp):
            with pytest.raises(RuntimeError, match="Failed to establish CDP connection"):
                await session.connect()

        assert session._cdp_client_root is None
        assert session.agent_focus is None

    async def test_connect_attach_existing_target_failure_logged(self):
        """Failed attachment to existing target should be caught and logged."""
        session, mocks = _make_session_with_mocks()

        targets = [
            {"targetId": "fail-target", "type": "page", "url": "https://example.com"},
        ]

        with patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]), \
             patch("openbrowser.browser.session_manager.SessionManager") as MockSM, \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(BrowserSession, "_setup_proxy_auth", new_callable=AsyncMock):

            mock_sm_instance = MagicMock()
            mock_sm_instance.start_monitoring = AsyncMock()
            mock_sm_instance.get_session_for_target = AsyncMock(return_value=mocks["cdp_session"])
            MockSM.return_value = mock_sm_instance

            mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": targets})
            mocks["cdp_root"].send.Target.attachToTarget = AsyncMock(side_effect=Exception("attach failed"))
            mocks["cdp_root"].start = AsyncMock()

            result = await session.connect()
            assert result is session

    async def test_connect_session_wait_timeout_raises(self):
        """When session manager never returns a session for target, connect raises."""
        session, mocks = _make_session_with_mocks()

        page_target = {"targetId": "page-target-1", "type": "page", "url": "about:blank"}

        with patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]), \
             patch("openbrowser.browser.session_manager.SessionManager") as MockSM, \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(BrowserSession, "_setup_proxy_auth", new_callable=AsyncMock):

            mock_sm_instance = MagicMock()
            mock_sm_instance.start_monitoring = AsyncMock()
            mock_sm_instance.get_session_for_target = AsyncMock(return_value=None)
            MockSM.return_value = mock_sm_instance

            mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": [page_target]})
            mocks["cdp_root"].start = AsyncMock()

            # Reset agent_focus so the timeout check fires
            session.agent_focus = None

            with pytest.raises(RuntimeError, match="Failed to"):
                await session.connect()

    async def test_connect_unknown_title_warning(self):
        """Session with 'Unknown title' should log warning but not fail."""
        session, mocks = _make_session_with_mocks()

        page_target = {"targetId": "page-t", "type": "page", "url": "about:blank"}
        unknown_session = MagicMock(spec=CDPSession)
        unknown_session.target_id = "page-t"
        unknown_session.session_id = "s-unknown"
        unknown_session.cdp_client = mocks["cdp_root"]
        unknown_session.title = "Unknown title"

        with patch("openbrowser.browser.session.CDPClient", return_value=mocks["cdp_root"]), \
             patch("openbrowser.browser.session_manager.SessionManager") as MockSM, \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(BrowserSession, "_setup_proxy_auth", new_callable=AsyncMock):

            mock_sm_instance = MagicMock()
            mock_sm_instance.start_monitoring = AsyncMock()
            mock_sm_instance.get_session_for_target = AsyncMock(return_value=unknown_session)
            MockSM.return_value = mock_sm_instance

            mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": [page_target]})
            mocks["cdp_root"].start = AsyncMock()

            result = await session.connect()
            assert result is session


# ===========================================================================
# _setup_proxy_auth (lines 1483-1627)
# ===========================================================================


@pytest.mark.asyncio
class TestSetupProxyAuth:

    async def test_skips_when_no_credentials(self):
        session, mocks = _make_session_with_mocks()
        session.browser_profile.proxy = None
        # Should return without error
        await session._setup_proxy_auth()

    async def test_skips_when_proxy_no_username(self):
        session, mocks = _make_session_with_mocks()
        from openbrowser.browser.profile import ProxySettings
        session.browser_profile.proxy = ProxySettings(server="http://proxy:8080")
        await session._setup_proxy_auth()

    async def test_enables_fetch_with_credentials(self):
        session, mocks = _make_session_with_mocks()
        from openbrowser.browser.profile import ProxySettings
        session.browser_profile.proxy = ProxySettings(
            server="http://proxy:8080", username="user", password="pass"
        )

        await session._setup_proxy_auth()
        mocks["cdp_root"].send.Fetch.enable.assert_called()
        mocks["cdp_root"].register.Fetch.authRequired.assert_called()
        mocks["cdp_root"].register.Fetch.requestPaused.assert_called()

    async def test_fetch_enable_failure_swallowed(self):
        session, mocks = _make_session_with_mocks()
        from openbrowser.browser.profile import ProxySettings
        session.browser_profile.proxy = ProxySettings(
            server="http://proxy:8080", username="user", password="pass"
        )
        mocks["cdp_root"].send.Fetch.enable = AsyncMock(side_effect=Exception("Fetch not supported"))

        # Should not raise
        await session._setup_proxy_auth()

    async def test_outer_exception_swallowed(self):
        """Top-level exception in _setup_proxy_auth is caught."""
        session, mocks = _make_session_with_mocks()
        # Force the proxy property access to raise by patching browser_profile
        # The outer try/except (line 1626) should catch and log
        mock_profile = MagicMock()
        type(mock_profile).proxy = PropertyMock(side_effect=Exception("kaboom"))
        session.__dict__['browser_profile'] = mock_profile

        # Should not raise
        await session._setup_proxy_auth()


# ===========================================================================
# get_tabs (lines 1629-1691)
# ===========================================================================


@pytest.mark.asyncio
class TestGetTabs:

    async def test_returns_empty_when_not_connected(self):
        session, _ = _make_session_with_mocks()
        session._cdp_client_root = None
        tabs = await session.get_tabs()
        assert tabs == []

    async def test_returns_tabs_with_title(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t1", "url": "https://example.com", "type": "page"},
        ])
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(return_value={
            "targetInfo": {"targetId": "t1", "url": "https://example.com", "title": "Example"}
        })

        tabs = await session.get_tabs()
        assert len(tabs) == 1
        assert tabs[0].title == "Example"

    async def test_new_tab_page_gets_empty_title(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t2", "url": "chrome://newtab/", "type": "page"},
        ])
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(return_value={
            "targetInfo": {"targetId": "t2", "url": "chrome://newtab/", "title": "New Tab"}
        })

        tabs = await session.get_tabs()
        assert tabs[0].title == ""

    async def test_chrome_page_uses_url_as_title(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t3", "url": "chrome://settings", "type": "page"},
        ])
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(return_value={
            "targetInfo": {"targetId": "t3", "url": "chrome://settings", "title": ""}
        })

        tabs = await session.get_tabs()
        assert tabs[0].title == "chrome://settings"

    async def test_pdf_page_uses_filename(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t4", "url": "https://example.com/docs/report.pdf", "type": "page"},
        ])
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(return_value={
            "targetInfo": {"targetId": "t4", "url": "https://example.com/docs/report.pdf", "title": ""}
        })

        tabs = await session.get_tabs()
        assert tabs[0].title == "report.pdf"

    async def test_target_info_exception_fallback(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t5", "url": "https://example.com", "type": "page"},
        ])
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(side_effect=Exception("CDP error"))

        tabs = await session.get_tabs()
        assert len(tabs) == 1
        assert tabs[0].title == ""

    async def test_target_info_exception_newtab_fallback(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t6", "url": "about:blank", "type": "page"},
        ])
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(side_effect=Exception("CDP error"))

        tabs = await session.get_tabs()
        assert tabs[0].title == ""

    async def test_target_info_exception_chrome_fallback(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t7", "url": "chrome://extensions", "type": "page"},
        ])
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(side_effect=Exception("CDP error"))

        tabs = await session.get_tabs()
        assert tabs[0].title == "chrome://extensions"


# ===========================================================================
# get_current_target_info (lines 1696-1706)
# ===========================================================================


@pytest.mark.asyncio
class TestGetCurrentTargetInfo:

    async def test_returns_none_without_focus(self):
        session, _ = _make_session_with_mocks()
        session.agent_focus = None
        result = await session.get_current_target_info()
        assert result is None

    async def test_returns_matching_target(self):
        session, mocks = _make_session_with_mocks()
        tid = mocks["cdp_session"].target_id
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [
                {"targetId": tid, "url": "https://example.com", "type": "page"},
                {"targetId": "other", "url": "https://other.com", "type": "page"},
            ]
        })
        result = await session.get_current_target_info()
        assert result["targetId"] == tid

    async def test_returns_none_when_not_found(self):
        session, mocks = _make_session_with_mocks()
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})
        result = await session.get_current_target_info()
        assert result is None


# ===========================================================================
# get_current_page_url / get_current_page_title (lines 1708-1720)
# ===========================================================================


@pytest.mark.asyncio
class TestGetCurrentPageUrlTitle:

    async def test_url_returns_target_url(self):
        session, _ = _make_session_with_mocks()
        with patch.object(BrowserSession, "get_current_target_info", new_callable=AsyncMock,
                          return_value={"url": "https://test.com", "title": "Test"}):
            url = await session.get_current_page_url()
        assert url == "https://test.com"

    async def test_url_returns_about_blank_when_no_target(self):
        session, _ = _make_session_with_mocks()
        with patch.object(BrowserSession, "get_current_target_info", new_callable=AsyncMock, return_value=None):
            url = await session.get_current_page_url()
        assert url == "about:blank"

    async def test_title_returns_target_title(self):
        session, _ = _make_session_with_mocks()
        with patch.object(BrowserSession, "get_current_target_info", new_callable=AsyncMock,
                          return_value={"title": "My Page", "url": "https://test.com"}):
            title = await session.get_current_page_title()
        assert title == "My Page"

    async def test_title_returns_unknown_when_no_target(self):
        session, _ = _make_session_with_mocks()
        with patch.object(BrowserSession, "get_current_target_info", new_callable=AsyncMock, return_value=None):
            title = await session.get_current_page_title()
        assert title == "Unknown page title"


# ===========================================================================
# get_dom_element_by_index / update_cached_selector_map / get_element_by_index
# (lines 1739-1769)
# ===========================================================================


class TestUpdateCachedSelectorMap:

    def test_update_cached_selector_map(self):
        session, _ = _make_session_with_mocks()
        new_map = {0: MagicMock(), 1: MagicMock()}
        session.update_cached_selector_map(new_map)
        assert session._cached_selector_map is new_map


@pytest.mark.asyncio
class TestDomElementByIndex:

    async def test_returns_cached_element(self):
        session, _ = _make_session_with_mocks()
        mock_node = MagicMock()
        session._cached_selector_map = {0: mock_node, 1: MagicMock()}

        result = await session.get_dom_element_by_index(0)
        assert result is mock_node

    async def test_returns_none_when_not_found(self):
        session, _ = _make_session_with_mocks()
        session._cached_selector_map = {0: MagicMock()}

        result = await session.get_dom_element_by_index(99)
        assert result is None

    async def test_returns_none_when_map_empty(self):
        session, _ = _make_session_with_mocks()
        session._cached_selector_map = {}

        result = await session.get_dom_element_by_index(0)
        assert result is None

    async def test_get_element_by_index_alias(self):
        session, _ = _make_session_with_mocks()
        mock_node = MagicMock()
        session._cached_selector_map = {5: mock_node}

        result = await session.get_element_by_index(5)
        assert result is mock_node


# ===========================================================================
# get_target_id_from_tab_id (lines 1771-1789)
# ===========================================================================


@pytest.mark.asyncio
class TestGetTargetIdFromTabId:

    async def test_finds_target_in_session_pool(self):
        session, mocks = _make_session_with_mocks()
        full_id = "abcdef1234567890abcdef12345678aa"
        session._cdp_session_pool = {full_id: MagicMock()}

        with patch.object(BrowserSession, "_is_target_valid", new_callable=AsyncMock, return_value=True):
            result = await session.get_target_id_from_tab_id("78aa")
        assert result == full_id

    async def test_skips_stale_session_in_pool(self):
        session, mocks = _make_session_with_mocks()
        stale_id = "abcdef1234567890abcdef12345678bb"
        session._cdp_session_pool = {stale_id: MagicMock()}

        # Target is stale in pool, but found in live targets
        with patch.object(BrowserSession, "_is_target_valid", new_callable=AsyncMock, return_value=False):
            mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={
                "targetInfos": [{"targetId": "fresh1234567890abcdef12345678bb", "type": "page"}]
            })
            result = await session.get_target_id_from_tab_id("78bb")
        assert result == "fresh1234567890abcdef12345678bb"

    async def test_falls_back_to_cdp_targets(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_session_pool = {}
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [
                {"targetId": "target-xxxxx-abcd", "type": "page"},
            ]
        })

        result = await session.get_target_id_from_tab_id("abcd")
        assert result == "target-xxxxx-abcd"

    async def test_raises_when_not_found(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_session_pool = {}
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})

        with pytest.raises(ValueError, match="No TargetID found"):
            await session.get_target_id_from_tab_id("zzzz")


# ===========================================================================
# _is_target_valid (lines 1791-1797)
# ===========================================================================


@pytest.mark.asyncio
class TestIsTargetValid:

    async def test_valid_target(self):
        session, mocks = _make_session_with_mocks()
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(return_value={
            "targetInfo": {"targetId": "t1"}
        })
        result = await session._is_target_valid("t1")
        assert result is True

    async def test_invalid_target(self):
        session, mocks = _make_session_with_mocks()
        mocks["cdp_root"].send.Target.getTargetInfo = AsyncMock(side_effect=Exception("not found"))
        result = await session._is_target_valid("missing")
        assert result is False


# ===========================================================================
# get_target_id_from_url (lines 1799-1811)
# ===========================================================================


@pytest.mark.asyncio
class TestGetTargetIdFromUrl:

    async def test_exact_match(self):
        session, mocks = _make_session_with_mocks()
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [
                {"targetId": "t1", "url": "https://example.com", "type": "page"},
            ]
        })
        result = await session.get_target_id_from_url("https://example.com")
        assert result == "t1"

    async def test_substring_match(self):
        session, mocks = _make_session_with_mocks()
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [
                {"targetId": "t2", "url": "https://example.com/path/page?q=1", "type": "page"},
            ]
        })
        result = await session.get_target_id_from_url("example.com/path")
        assert result == "t2"

    async def test_raises_when_not_found(self):
        session, mocks = _make_session_with_mocks()
        mocks["cdp_root"].send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})

        with pytest.raises(ValueError, match="No TargetID found for url"):
            await session.get_target_id_from_url("https://notfound.com")


# ===========================================================================
# get_most_recently_opened_target_id (lines 1813-1816)
# ===========================================================================


@pytest.mark.asyncio
class TestGetMostRecentlyOpenedTargetId:

    async def test_returns_last_page(self):
        session, mocks = _make_session_with_mocks()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "first", "url": "https://a.com", "type": "page"},
            {"targetId": "last", "url": "https://b.com", "type": "page"},
        ])
        result = await session.get_most_recently_opened_target_id()
        assert result == "last"


# ===========================================================================
# is_file_input (lines 1818-1835)
# ===========================================================================


class TestIsFileInput:

    def test_delegates_to_dom_watchdog(self):
        session, _ = _make_session_with_mocks()
        mock_wd = MagicMock()
        mock_wd.is_file_input = MagicMock(return_value=True)
        session._dom_watchdog = mock_wd

        element = MagicMock()
        assert session.is_file_input(element) is True
        mock_wd.is_file_input.assert_called_once_with(element)

    def test_fallback_detects_file_input(self):
        session, _ = _make_session_with_mocks()
        session._dom_watchdog = None

        element = MagicMock()
        element.node_name = "INPUT"
        element.attributes = {"type": "file"}
        assert session.is_file_input(element) is True

    def test_fallback_non_file_input(self):
        session, _ = _make_session_with_mocks()
        session._dom_watchdog = None

        element = MagicMock()
        element.node_name = "INPUT"
        element.attributes = {"type": "text"}
        assert session.is_file_input(element) is False

    def test_fallback_non_input_element(self):
        session, _ = _make_session_with_mocks()
        session._dom_watchdog = None

        element = MagicMock()
        element.node_name = "DIV"
        element.attributes = {}
        assert session.is_file_input(element) is False


# ===========================================================================
# get_selector_map (lines 1837-1852)
# ===========================================================================


@pytest.mark.asyncio
class TestGetSelectorMap:

    async def test_returns_cached_map(self):
        session, _ = _make_session_with_mocks()
        mock_map = {0: MagicMock()}
        session._cached_selector_map = mock_map

        result = await session.get_selector_map()
        assert result is mock_map

    async def test_falls_back_to_dom_watchdog(self):
        session, _ = _make_session_with_mocks()
        session._cached_selector_map = {}

        wd_map = {1: MagicMock()}
        mock_wd = MagicMock()
        mock_wd.selector_map = wd_map
        session._dom_watchdog = mock_wd

        result = await session.get_selector_map()
        assert result is wd_map

    async def test_returns_empty_dict_when_nothing_available(self):
        session, _ = _make_session_with_mocks()
        session._cached_selector_map = {}
        session._dom_watchdog = None

        result = await session.get_selector_map()
        assert result == {}


# ===========================================================================
# get_index_by_id (lines 1854-1867)
# ===========================================================================


@pytest.mark.asyncio
class TestGetIndexById:

    async def test_finds_element_by_id(self):
        session, _ = _make_session_with_mocks()
        elem = MagicMock()
        elem.attributes = {"id": "submit-btn", "class": "primary"}
        session._cached_selector_map = {0: MagicMock(attributes={}), 1: elem}

        result = await session.get_index_by_id("submit-btn")
        assert result == 1

    async def test_returns_none_when_not_found(self):
        session, _ = _make_session_with_mocks()
        elem = MagicMock()
        elem.attributes = {"id": "other"}
        session._cached_selector_map = {0: elem}

        result = await session.get_index_by_id("nonexistent")
        assert result is None


# ===========================================================================
# get_index_by_class (lines 1869-1884)
# ===========================================================================


@pytest.mark.asyncio
class TestGetIndexByClass:

    async def test_finds_element_by_class(self):
        session, _ = _make_session_with_mocks()
        elem = MagicMock()
        elem.attributes = {"class": "btn primary large"}
        session._cached_selector_map = {0: elem}

        result = await session.get_index_by_class("primary")
        assert result == 0

    async def test_returns_none_when_not_found(self):
        session, _ = _make_session_with_mocks()
        elem = MagicMock()
        elem.attributes = {"class": "btn secondary"}
        session._cached_selector_map = {0: elem}

        result = await session.get_index_by_class("primary")
        assert result is None

    async def test_returns_none_for_no_attributes(self):
        session, _ = _make_session_with_mocks()
        elem = MagicMock()
        elem.attributes = None
        session._cached_selector_map = {0: elem}

        result = await session.get_index_by_class("primary")
        assert result is None


# ===========================================================================
# remove_highlights (lines 1886-1929)
# ===========================================================================


@pytest.mark.asyncio
class TestRemoveHighlights:

    async def test_skips_when_highlights_disabled(self):
        session, mocks = _make_session_with_mocks()
        session.browser_profile.highlight_elements = False

        await session.remove_highlights()
        # get_or_create_cdp_session should not be called
        mocks["session_manager"].get_session_for_target.assert_not_called()

    async def test_removes_highlights_successfully(self):
        session, mocks = _make_session_with_mocks()
        session.browser_profile.highlight_elements = True

        mocks["cdp_root"].send.Runtime.evaluate = AsyncMock(return_value={
            "result": {"value": {"removed": 3}}
        })

        await session.remove_highlights()

    async def test_handles_exception_gracefully(self):
        session, mocks = _make_session_with_mocks()
        session.browser_profile.highlight_elements = True

        # Make get_or_create_cdp_session raise
        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock,
                          side_effect=Exception("CDP error")):
            # Should not raise
            await session.remove_highlights()

    async def test_logs_when_no_result_value(self):
        session, mocks = _make_session_with_mocks()
        session.browser_profile.highlight_elements = True

        mocks["cdp_root"].send.Runtime.evaluate = AsyncMock(return_value={
            "result": {}
        })

        # Should not raise, hits the else branch
        await session.remove_highlights()


# ===========================================================================
# get_element_coordinates (lines 1931-2043)
# ===========================================================================


@pytest.mark.asyncio
class TestGetElementCoordinates:

    async def test_method1_content_quads(self):
        """Uses DOM.getContentQuads when quads are available."""
        session, mocks = _make_session_with_mocks()
        cdp_sess = mocks["cdp_session"]

        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={
            "quads": [[10, 20, 110, 20, 110, 70, 10, 70]]
        })

        result = await session.get_element_coordinates(123, cdp_sess)
        assert result is not None
        assert result.x == 10
        assert result.y == 20
        assert result.width == 100
        assert result.height == 50

    async def test_method2_box_model_fallback(self):
        """Falls back to DOM.getBoxModel when getContentQuads returns no quads."""
        session, mocks = _make_session_with_mocks()
        cdp_sess = mocks["cdp_session"]

        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={"quads": []})
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={
            "model": {"content": [0, 0, 200, 0, 200, 100, 0, 100]}
        })

        result = await session.get_element_coordinates(123, cdp_sess)
        assert result is not None
        assert result.width == 200
        assert result.height == 100

    async def test_method3_javascript_fallback(self):
        """Falls back to JS getBoundingClientRect when other methods fail."""
        session, mocks = _make_session_with_mocks()
        cdp_sess = mocks["cdp_session"]

        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(side_effect=Exception("failed"))
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(side_effect=Exception("failed"))
        cdp_sess.cdp_client.send.DOM.resolveNode = AsyncMock(return_value={
            "object": {"objectId": "obj-1"}
        })
        cdp_sess.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"x": 5, "y": 10, "width": 50, "height": 30}}
        })

        result = await session.get_element_coordinates(123, cdp_sess)
        assert result is not None
        assert result.x == 5
        assert result.width == 50

    async def test_returns_none_when_all_methods_fail(self):
        session, mocks = _make_session_with_mocks()
        cdp_sess = mocks["cdp_session"]

        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(side_effect=Exception("fail"))
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(side_effect=Exception("fail"))
        cdp_sess.cdp_client.send.DOM.resolveNode = AsyncMock(side_effect=Exception("fail"))

        result = await session.get_element_coordinates(123, cdp_sess)
        assert result is None

    async def test_returns_none_for_zero_size_element(self):
        """Element exists but has zero width/height."""
        session, mocks = _make_session_with_mocks()
        cdp_sess = mocks["cdp_session"]

        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={
            "quads": [[10, 20, 10, 20, 10, 20, 10, 20]]  # Zero-size
        })

        result = await session.get_element_coordinates(123, cdp_sess)
        assert result is None

    async def test_js_fallback_zero_size_returns_none(self):
        """JS fallback returns zero-size rect."""
        session, mocks = _make_session_with_mocks()
        cdp_sess = mocks["cdp_session"]

        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={"quads": []})
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={})
        cdp_sess.cdp_client.send.DOM.resolveNode = AsyncMock(return_value={
            "object": {"objectId": "obj-1"}
        })
        cdp_sess.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"x": 0, "y": 0, "width": 0, "height": 0}}
        })

        result = await session.get_element_coordinates(123, cdp_sess)
        assert result is None

    async def test_content_quads_empty_dict(self):
        """getContentQuads returns dict without quads key."""
        session, mocks = _make_session_with_mocks()
        cdp_sess = mocks["cdp_session"]

        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={})
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={})
        cdp_sess.cdp_client.send.DOM.resolveNode = AsyncMock(side_effect=Exception("no node"))

        result = await session.get_element_coordinates(123, cdp_sess)
        assert result is None

    async def test_box_model_incomplete_content(self):
        """getBoxModel returns content array with < 8 elements."""
        session, mocks = _make_session_with_mocks()
        cdp_sess = mocks["cdp_session"]

        cdp_sess.cdp_client.send.DOM.getContentQuads = AsyncMock(return_value={"quads": []})
        cdp_sess.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={
            "model": {"content": [0, 0, 50, 0]}  # Only 4 values
        })
        cdp_sess.cdp_client.send.DOM.resolveNode = AsyncMock(side_effect=Exception("no"))

        result = await session.get_element_coordinates(123, cdp_sess)
        assert result is None


# ===========================================================================
# navigate_to (lines 1722-1733)
# ===========================================================================


@pytest.mark.asyncio
class TestNavigateTo:

    async def test_dispatches_navigate_event(self):
        session, mocks = _make_session_with_mocks()

        # The event object must be directly awaitable (await event) and also
        # have an async event_result method.
        class _AwaitableEvent:
            def __init__(self):
                self.event_result = AsyncMock(return_value=None)

            def __await__(self):
                return asyncio.sleep(0).__await__()

        mock_event = _AwaitableEvent()
        session.event_bus.dispatch = MagicMock(return_value=mock_event)

        await session.navigate_to("https://example.com", new_tab=True)
        session.event_bus.dispatch.assert_called_once()
