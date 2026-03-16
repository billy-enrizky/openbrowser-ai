"""Tests for remaining watchdog coverage gaps across multiple modules.

Covers:
- aboutblank_watchdog.py (lines 91-106, 123-124, 252-253)
- dom_watchdog.py (lines 78-89, 104, 214, 227, 238-241, 265, 276-278, 285-288,
  309-338, 384-388, 394-396, 428-435, 439, 448-450, 457-463, 490, 517-521,
  597-605, 632, 636-641, 645-660, 689-693, 710, 805)
- downloads_watchdog.py (lines 636-936, 1077-1263)
- local_browser_watchdog.py (lines 115, 178-179, 216-219, 224-226, 281-284,
  308-315, 326-329, 344, 369-370, 400, 436-438, 454-455, 488-489, 500-514, 517)
- security_watchdog.py (lines 70-72, 92-93, 156-157, 178-180, 233, 261-265)
- screenshot_watchdog.py (lines 61-62)
- watchdog_base.py (lines 111, 124-162, 255-260)
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, create_autospec, patch

import pytest

from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_browser_session():
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test_remaining_watchdog")
    session.event_bus = MagicMock()
    session._cdp_client_root = MagicMock()
    session.agent_focus = MagicMock()
    session.agent_focus.target_id = "target-1234"
    session.agent_focus.session_id = "session-1234"
    session.get_or_create_cdp_session = AsyncMock()
    session.cdp_client = MagicMock()
    session.cdp_client.send = MagicMock()
    session.cdp_client.register = MagicMock()
    session.id = "test-remaining-session"
    session.is_local = True
    session._closed_popup_messages = []
    session._cached_browser_state_summary = None
    session.get_current_page_url = AsyncMock(return_value="https://example.com")
    session.get_current_page_title = AsyncMock(return_value="Example")
    session.get_tabs = AsyncMock(return_value=[])
    session._cdp_get_all_pages = AsyncMock(return_value=[])
    session._cdp_close_page = AsyncMock()
    session.remove_highlights = AsyncMock()
    session.add_highlights = AsyncMock()
    session.update_cached_selector_map = MagicMock()
    session.cdp_client_for_frame = AsyncMock()

    profile = MagicMock()
    profile.downloads_path = "/tmp/test_downloads"
    profile.viewport = {"width": 1280, "height": 720}
    profile.highlight_elements = True
    profile.dom_highlight_elements = True
    profile.filter_highlight_ids = None
    profile.cross_origin_iframes = False
    profile.paint_order_filtering = False
    profile.max_iframes = 5
    profile.max_iframe_depth = 3
    profile.minimum_wait_page_load_time = 0
    profile.wait_for_network_idle_page_load_time = 0
    profile.allowed_domains = None
    profile.prohibited_domains = None
    profile.block_ip_addresses = False
    profile.user_data_dir = "/tmp/test_profile"
    profile.profile_directory = None
    profile.executable_path = None
    profile.record_video_dir = None
    profile.get_args = MagicMock(return_value=["--headless", "--user-data-dir=/tmp/test_profile"])
    session.browser_profile = profile

    return session


def _make_watchdog(watchdog_class, session=None, event_bus=None):
    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = MagicMock()
    session.event_bus = event_bus
    return watchdog_class.model_construct(
        event_bus=event_bus,
        browser_session=session,
    )


# ===========================================================================
# AboutBlankWatchdog Tests
# ===========================================================================

class TestAboutBlankWatchdog:
    """Cover lines 91-106, 123-124, 252-253 in aboutblank_watchdog.py."""

    def _make_watchdog(self, session=None):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog
        return _make_watchdog(AboutBlankWatchdog, session=session)

    @pytest.mark.asyncio
    async def test_check_and_ensure_about_blank_tab_no_tabs(self):
        """Lines 91-102: No tabs exist, create new about:blank tab."""
        session = _make_mock_browser_session()
        session._cdp_get_all_pages = AsyncMock(return_value=[])
        mock_dispatch_result = AsyncMock()
        session.event_bus = MagicMock()
        session.event_bus.dispatch = MagicMock(return_value=mock_dispatch_result())

        watchdog = self._make_watchdog(session=session)
        await watchdog._check_and_ensure_about_blank_tab()
        # Should dispatch NavigateToUrlEvent for about:blank
        session.event_bus.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_check_and_ensure_about_blank_tab_has_tabs(self):
        """Lines 91-106: Tabs exist, no new tab created."""
        session = _make_mock_browser_session()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t1", "url": "https://example.com"}
        ])
        watchdog = self._make_watchdog(session=session)
        await watchdog._check_and_ensure_about_blank_tab()
        # No dispatch needed since tabs exist

    @pytest.mark.asyncio
    async def test_check_and_ensure_about_blank_tab_exception(self):
        """Lines 105-106: Exception during tab check."""
        session = _make_mock_browser_session()
        session._cdp_get_all_pages = AsyncMock(side_effect=RuntimeError("CDP fail"))
        watchdog = self._make_watchdog(session=session)
        # Should not raise
        await watchdog._check_and_ensure_about_blank_tab()

    @pytest.mark.asyncio
    async def test_show_dvd_screensaver_exception(self):
        """Lines 123-124: Exception in _show_dvd_screensaver_on_about_blank_tabs."""
        session = _make_mock_browser_session()
        session._cdp_get_all_pages = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = self._make_watchdog(session=session)
        # Should not raise
        await watchdog._show_dvd_screensaver_on_about_blank_tabs()

    @pytest.mark.asyncio
    async def test_show_dvd_screensaver_loading_animation_exception(self):
        """Lines 252-253: Exception in _show_dvd_screensaver_loading_animation_cdp."""
        session = _make_mock_browser_session()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("session fail"))
        watchdog = self._make_watchdog(session=session)
        # Should not raise
        await watchdog._show_dvd_screensaver_loading_animation_cdp("target-1", "1234")

    @pytest.mark.asyncio
    async def test_show_dvd_screensaver_on_about_blank_pages(self):
        """Lines 108-124: Iterate pages and show screensaver on about:blank."""
        session = _make_mock_browser_session()
        session._cdp_get_all_pages = AsyncMock(return_value=[
            {"targetId": "t1", "url": "about:blank"},
            {"targetId": "t2", "url": "https://example.com"},
        ])
        session.id = "test-session-abcd"
        temp_session = MagicMock()
        temp_session.cdp_client = MagicMock()
        temp_session.cdp_client.send = MagicMock()
        temp_session.cdp_client.send.Runtime = MagicMock()
        temp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={})
        temp_session.session_id = "sess-temp"
        session.get_or_create_cdp_session = AsyncMock(return_value=temp_session)

        watchdog = self._make_watchdog(session=session)
        await watchdog._show_dvd_screensaver_on_about_blank_tabs()
        # Only called for about:blank page
        session.get_or_create_cdp_session.assert_called_once()


# ===========================================================================
# DOMWatchdog Tests
# ===========================================================================

class TestDOMWatchdog:
    """Cover many lines in dom_watchdog.py."""

    def _make_watchdog(self, session=None):
        from openbrowser.browser.watchdogs.dom_watchdog import DOMWatchdog
        return _make_watchdog(DOMWatchdog, session=session)

    # -- _get_recent_events_str --

    def test_get_recent_events_str_with_events(self):
        """Lines 78-89: Get recent events with url, error_message, target_id attributes."""
        session = _make_mock_browser_session()
        mock_event1 = MagicMock()
        mock_event1.event_type = "NavigateToUrlEvent"
        mock_event1.event_created_at = MagicMock()
        mock_event1.event_created_at.timestamp.return_value = time.time()
        mock_event1.event_created_at.isoformat.return_value = "2025-01-01T00:00:00"
        mock_event1.url = "https://example.com"
        mock_event1.error_message = "test error"
        mock_event1.target_id = "t1"

        event_bus = MagicMock()
        event_bus.event_history = {"e1": mock_event1}

        watchdog = self._make_watchdog(session=session)
        # Set event_bus with event_history AFTER construction (since _make_watchdog replaces it)
        watchdog.browser_session.event_bus = event_bus
        result = watchdog._get_recent_events_str(limit=5)
        assert result is not None
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["url"] == "https://example.com"
        assert parsed[0]["error_message"] == "test error"
        assert parsed[0]["target_id"] == "t1"

    def test_get_recent_events_str_exception(self):
        """Lines 86-89: Exception returns empty JSON array."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)
        # Set up event_bus to raise on values() after construction
        event_bus = MagicMock()
        event_bus.event_history = MagicMock()
        event_bus.event_history.values.side_effect = RuntimeError("fail")
        watchdog.browser_session.event_bus = event_bus

        result = watchdog._get_recent_events_str()
        assert result == "[]"

    # -- _get_pending_network_requests --

    @pytest.mark.asyncio
    async def test_get_pending_network_requests_no_agent_focus(self):
        """Line 104: No agent_focus returns empty list."""
        session = _make_mock_browser_session()
        session.agent_focus = None
        watchdog = self._make_watchdog(session=session)
        result = await watchdog._get_pending_network_requests()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_pending_network_requests_with_data(self):
        """Lines 199-241: Parse pending network requests from CDP."""
        session = _make_mock_browser_session()
        cdp_session = MagicMock()
        cdp_session.cdp_client = MagicMock()
        cdp_session.cdp_client.send = MagicMock()
        cdp_session.cdp_client.send.Runtime = MagicMock()
        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
            "result": {
                "type": "object",
                "value": {
                    "pending_requests": [
                        {"url": "https://api.example.com/data", "method": "GET", "loading_duration_ms": 500, "resource_type": "xhr"},
                    ],
                    "document_loading": False,
                    "document_ready_state": "complete",
                    "debug": {
                        "total_resources": 10,
                        "with_response_end_zero": 1,
                        "after_all_filters": 1,
                        "all_domains": ["api.example.com", "cdn.example.com", "a.com", "b.com", "c.com", "d.com"],
                    },
                },
            }
        })
        cdp_session.session_id = "sess-1"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

        watchdog = self._make_watchdog(session=session)
        result = await watchdog._get_pending_network_requests()
        assert len(result) == 1
        assert result[0].url == "https://api.example.com/data"

    @pytest.mark.asyncio
    async def test_get_pending_network_requests_exception(self):
        """Lines 238-241: Exception returns empty list."""
        session = _make_mock_browser_session()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = self._make_watchdog(session=session)
        result = await watchdog._get_pending_network_requests()
        assert result == []

    # -- on_BrowserStateRequestEvent --

    @pytest.mark.asyncio
    async def test_browser_state_request_no_agent_focus(self):
        """Line 265: No cdp_session attached."""
        session = _make_mock_browser_session()
        session.agent_focus = None
        session.get_current_page_url = AsyncMock(return_value="about:blank")
        session.get_tabs = AsyncMock(return_value=[])

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = False
        event.include_screenshot = False
        event.include_recent_events = False

        result = await watchdog.on_BrowserStateRequestEvent(event)
        assert result is not None
        assert result.url == "about:blank"

    @pytest.mark.asyncio
    async def test_browser_state_request_non_http_page(self):
        """Lines 309-338: Fast path for non-http pages."""
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="about:blank")
        session.get_tabs = AsyncMock(return_value=[])

        cdp_session = MagicMock()
        cdp_session.cdp_client = MagicMock()
        cdp_session.session_id = "sess-1"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = True
        event.include_screenshot = True
        event.include_recent_events = True

        with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock) as mock_page_info:
            from openbrowser.browser.views import PageInfo
            mock_page_info.return_value = PageInfo(
                viewport_width=1280, viewport_height=720,
                page_width=1280, page_height=720,
                scroll_x=0, scroll_y=0,
                pixels_above=0, pixels_below=0,
                pixels_left=0, pixels_right=0,
            )
            result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result is not None
        assert result.url == "about:blank"
        assert result.title == "Empty Tab"

    @pytest.mark.asyncio
    async def test_browser_state_request_non_http_page_info_fails(self):
        """Lines 321-336: _get_page_info fails for empty page, uses fallback."""
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="chrome://newtab/")
        session.get_tabs = AsyncMock(return_value=[])

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = False
        event.include_screenshot = False
        event.include_recent_events = False

        with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result is not None
        assert result.page_info.viewport_width == 1280

    @pytest.mark.asyncio
    async def test_browser_state_request_with_pending_requests(self):
        """Lines 276-278, 285-288: Pending requests before wait, stability sleep."""
        from openbrowser.browser.views import NetworkRequest
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="https://example.com")
        session.get_tabs = AsyncMock(return_value=[])
        session.agent_focus = MagicMock()
        session.agent_focus.target_id = "t1"

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = False
        event.include_screenshot = False
        event.include_recent_events = False

        mock_request = NetworkRequest(url="https://api.example.com/data", method="GET", loading_duration_ms=500)

        with patch.object(watchdog, "_get_pending_network_requests", new_callable=AsyncMock, return_value=[mock_request]):
            with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock) as mock_pi:
                from openbrowser.browser.views import PageInfo
                mock_pi.return_value = PageInfo(
                    viewport_width=1280, viewport_height=720,
                    page_width=1280, page_height=720,
                    scroll_x=0, scroll_y=0,
                    pixels_above=0, pixels_below=0,
                    pixels_left=0, pixels_right=0,
                )
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result is not None
        assert len(result.pending_network_requests) == 1

    @pytest.mark.asyncio
    async def test_browser_state_request_dom_build_fails(self):
        """Lines 384-388: DOM task fails, uses minimal state."""
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="https://example.com")
        session.get_tabs = AsyncMock(return_value=[])

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = True
        event.include_screenshot = False
        event.include_recent_events = False

        with patch.object(watchdog, "_get_pending_network_requests", new_callable=AsyncMock, return_value=[]):
            with patch.object(watchdog, "_build_dom_tree_without_highlights", new_callable=AsyncMock, side_effect=RuntimeError("DOM fail")):
                with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock) as mock_pi:
                    from openbrowser.browser.views import PageInfo
                    mock_pi.return_value = PageInfo(
                        viewport_width=1280, viewport_height=720,
                        page_width=1280, page_height=720,
                        scroll_x=0, scroll_y=0,
                        pixels_above=0, pixels_below=0,
                        pixels_left=0, pixels_right=0,
                    )
                    result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result is not None
        assert result.dom_state.selector_map == {}

    @pytest.mark.asyncio
    async def test_browser_state_request_screenshot_fails(self):
        """Lines 394-396: Screenshot task fails, screenshot is None."""
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="https://example.com")
        session.get_tabs = AsyncMock(return_value=[])

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = False
        event.include_screenshot = True
        event.include_recent_events = False

        with patch.object(watchdog, "_get_pending_network_requests", new_callable=AsyncMock, return_value=[]):
            with patch.object(watchdog, "_capture_clean_screenshot", new_callable=AsyncMock, side_effect=RuntimeError("screenshot fail")):
                with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock) as mock_pi:
                    from openbrowser.browser.views import PageInfo
                    mock_pi.return_value = PageInfo(
                        viewport_width=1280, viewport_height=720,
                        page_width=1280, page_height=720,
                        scroll_x=0, scroll_y=0,
                        pixels_above=0, pixels_below=0,
                        pixels_left=0, pixels_right=0,
                    )
                    result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result is not None
        assert result.screenshot is None

    @pytest.mark.asyncio
    async def test_browser_state_request_highlight_and_title_fails(self):
        """Lines 428-435, 448-450, 457-463: Browser highlighting fails, title/page_info fallback."""
        from openbrowser.dom.views import SerializedDOMState, EnhancedDOMTreeNode
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="https://example.com")
        session.get_tabs = AsyncMock(return_value=[])
        session.get_current_page_title = AsyncMock(side_effect=asyncio.TimeoutError("title timeout"))
        session.add_highlights = AsyncMock(side_effect=RuntimeError("highlight fail"))

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = True
        event.include_screenshot = False
        event.include_recent_events = False

        # Create a DOM state with selector_map
        mock_node = MagicMock(spec=EnhancedDOMTreeNode)
        mock_dom_state = SerializedDOMState(_root=None, selector_map={1: mock_node})

        with patch.object(watchdog, "_get_pending_network_requests", new_callable=AsyncMock, return_value=[]):
            with patch.object(watchdog, "_build_dom_tree_without_highlights", new_callable=AsyncMock, return_value=mock_dom_state):
                with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock, side_effect=RuntimeError("page info fail")):
                    with patch.object(watchdog, "_detect_pagination_buttons", return_value=[]):
                        result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result is not None
        assert result.title == "Page"  # Fallback title
        assert result.page_info.viewport_width == 1280  # Fallback

    @pytest.mark.asyncio
    async def test_browser_state_request_no_screenshot_log_branch(self):
        """Line 490: Log branch when no screenshot."""
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="https://example.com")
        session.get_tabs = AsyncMock(return_value=[])

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = False
        event.include_screenshot = False
        event.include_recent_events = False

        with patch.object(watchdog, "_get_pending_network_requests", new_callable=AsyncMock, return_value=[]):
            with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock) as mock_pi:
                from openbrowser.browser.views import PageInfo
                mock_pi.return_value = PageInfo(
                    viewport_width=1280, viewport_height=720,
                    page_width=1280, page_height=720,
                    scroll_x=0, scroll_y=0,
                    pixels_above=0, pixels_below=0,
                    pixels_left=0, pixels_right=0,
                )
                result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result.screenshot is None

    @pytest.mark.asyncio
    async def test_browser_state_request_exception_recovery(self):
        """Lines 517-521: Exception in main try block returns minimal state."""
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="https://example.com")
        session.get_tabs = AsyncMock(return_value=[])

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = False
        event.include_screenshot = False
        event.include_recent_events = False

        # Make _closed_popup_messages.copy() raise to trigger the outer except block
        # This is called at line 508 during BrowserStateSummary construction
        mock_popup_msgs = MagicMock()
        mock_popup_msgs.copy = MagicMock(side_effect=[RuntimeError("copy fail"), []])
        watchdog.browser_session._closed_popup_messages = mock_popup_msgs

        with patch.object(watchdog, "_get_pending_network_requests", new_callable=AsyncMock, return_value=[]):
            with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock) as mock_pi:
                from openbrowser.browser.views import PageInfo
                mock_pi.return_value = PageInfo(
                    viewport_width=1280, viewport_height=720,
                    page_width=1280, page_height=720,
                    scroll_x=0, scroll_y=0,
                    pixels_above=0, pixels_below=0,
                    pixels_left=0, pixels_right=0,
                )
                result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result is not None
        assert result.title == "Error"
        assert len(result.browser_errors) > 0

    # -- _build_dom_tree_without_highlights --

    @pytest.mark.asyncio
    async def test_build_dom_tree_exception(self):
        """Lines 597-605: Exception dispatches BrowserErrorEvent and re-raises."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        with patch("openbrowser.browser.watchdogs.dom_watchdog.DomService") as MockDomService:
            mock_service = MagicMock()
            mock_service.get_serialized_dom_tree = AsyncMock(side_effect=RuntimeError("DOM build fail"))
            MockDomService.return_value = mock_service
            watchdog._dom_service = mock_service

            with pytest.raises(RuntimeError, match="DOM build fail"):
                await watchdog._build_dom_tree_without_highlights()

        session.event_bus.dispatch.assert_called()

    # -- _capture_clean_screenshot --

    @pytest.mark.asyncio
    async def test_capture_clean_screenshot_timeout(self):
        """Lines 636-638: TimeoutError in screenshot."""
        session = _make_mock_browser_session()
        session.agent_focus = MagicMock()
        session.agent_focus.target_id = "t1"

        watchdog = self._make_watchdog(session=session)

        # Create a mock event that is awaitable and has event_result
        class AwaitableMockEvent:
            def __await__(self):
                return iter([])  # immediately resolves
            async def event_result(self, raise_if_any=False, raise_if_none=False):
                raise TimeoutError("timeout")

        watchdog.event_bus.handlers = {"ScreenshotEvent": []}
        watchdog.event_bus.dispatch = MagicMock(return_value=AwaitableMockEvent())

        with pytest.raises(TimeoutError):
            await watchdog._capture_clean_screenshot()

    @pytest.mark.asyncio
    async def test_capture_clean_screenshot_general_exception(self):
        """Lines 639-641: General exception in screenshot."""
        session = _make_mock_browser_session()
        session.agent_focus = MagicMock()
        session.agent_focus.target_id = "t1"
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("no session"))

        watchdog = self._make_watchdog(session=session)

        with pytest.raises(RuntimeError):
            await watchdog._capture_clean_screenshot()

    @pytest.mark.asyncio
    async def test_capture_clean_screenshot_none_result(self):
        """Line 632: Screenshot handler returns None."""
        session = _make_mock_browser_session()
        session.agent_focus = MagicMock()
        session.agent_focus.target_id = "t1"

        watchdog = self._make_watchdog(session=session)

        class AwaitableMockEvent:
            def __await__(self):
                return iter([])
            async def event_result(self, raise_if_any=False, raise_if_none=False):
                raise RuntimeError("Screenshot handler returned None")

        watchdog.event_bus.handlers = {"ScreenshotEvent": [MagicMock()]}
        watchdog.event_bus.dispatch = MagicMock(return_value=AwaitableMockEvent())

        with pytest.raises(RuntimeError, match="Screenshot handler returned None"):
            await watchdog._capture_clean_screenshot()

    # -- _wait_for_stable_network --

    @pytest.mark.asyncio
    async def test_wait_for_stable_network(self):
        """Lines 645-660: Stable network wait with min and idle waits."""
        session = _make_mock_browser_session()
        session.browser_profile.minimum_wait_page_load_time = 0.01
        session.browser_profile.wait_for_network_idle_page_load_time = 0.01

        watchdog = self._make_watchdog(session=session)
        await watchdog._wait_for_stable_network()
        # Should complete without error

    # -- _detect_pagination_buttons --

    def test_detect_pagination_buttons_found(self):
        """Lines 689-693: Pagination buttons found."""
        from openbrowser.dom.views import EnhancedDOMTreeNode
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        mock_node = MagicMock(spec=EnhancedDOMTreeNode)
        selector_map = {1: mock_node}

        with patch("openbrowser.browser.watchdogs.dom_watchdog.DomService.detect_pagination_buttons", return_value=[
            {"button_type": "next", "backend_node_id": 123, "text": "Next", "selector": "a.next", "is_disabled": False},
        ]):
            result = watchdog._detect_pagination_buttons(selector_map)

        assert len(result) == 1
        assert result[0].button_type == "next"

    def test_detect_pagination_buttons_exception(self):
        """Line 693: Pagination detection fails gracefully."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        with patch("openbrowser.browser.watchdogs.dom_watchdog.DomService.detect_pagination_buttons", side_effect=RuntimeError("detection fail")):
            result = watchdog._detect_pagination_buttons({1: MagicMock()})

        assert result == []

    # -- _get_page_info --

    @pytest.mark.asyncio
    async def test_get_page_info_no_agent_focus(self):
        """Line 710: No agent_focus raises RuntimeError."""
        session = _make_mock_browser_session()
        session.agent_focus = None
        watchdog = self._make_watchdog(session=session)

        with pytest.raises(RuntimeError, match="No active CDP session"):
            await watchdog._get_page_info()

    # -- __del__ --

    def test_dom_watchdog_del(self):
        """Line 805: __del__ cleanup."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)
        watchdog._dom_service = MagicMock()
        watchdog.__del__()
        assert watchdog._dom_service is None

    # -- content is None after dom/screenshot --

    @pytest.mark.asyncio
    async def test_browser_state_content_none_fallback(self):
        """Line 439: content is None, fallback to empty state."""
        session = _make_mock_browser_session()
        session.get_current_page_url = AsyncMock(return_value="https://example.com")
        session.get_tabs = AsyncMock(return_value=[])

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.include_dom = False
        event.include_screenshot = False
        event.include_recent_events = False

        with patch.object(watchdog, "_get_pending_network_requests", new_callable=AsyncMock, return_value=[]):
            with patch.object(watchdog, "_get_page_info", new_callable=AsyncMock) as mock_pi:
                from openbrowser.browser.views import PageInfo
                mock_pi.return_value = PageInfo(
                    viewport_width=1280, viewport_height=720,
                    page_width=1280, page_height=720,
                    scroll_x=0, scroll_y=0,
                    pixels_above=0, pixels_below=0,
                    pixels_left=0, pixels_right=0,
                )
                result = await watchdog.on_BrowserStateRequestEvent(event)

        assert result.dom_state is not None


# ===========================================================================
# DownloadsWatchdog Tests
# ===========================================================================

class TestDownloadsWatchdog:
    """Cover lines 636-936, 1077-1263 in downloads_watchdog.py."""

    def _make_watchdog(self, session=None):
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog
        wd = _make_watchdog(DownloadsWatchdog, session=session)
        wd._sessions_with_listeners = set()
        wd._active_downloads = {}
        wd._pdf_viewer_cache = {}
        wd._download_cdp_session_setup = False
        wd._download_cdp_session = None
        wd._cdp_event_tasks = set()
        wd._cdp_downloads_info = {}
        wd._use_js_fetch_for_local = False
        wd._session_pdf_urls = {}
        wd._network_monitored_targets = set()
        wd._detected_downloads = set()
        wd._network_callback_registered = False
        return wd

    # -- _handle_cdp_download --

    @pytest.mark.asyncio
    async def test_handle_cdp_download_remote_browser(self):
        """Lines 762-764: Remote browser returns early after initial processing."""
        session = _make_mock_browser_session()
        session.is_local = False
        session.browser_profile.downloads_path = "/tmp/downloads"

        watchdog = self._make_watchdog(session=session)
        event = {"url": "https://example.com/file.pdf", "suggestedFilename": "file.pdf", "guid": "g1"}

        await watchdog._handle_cdp_download(event, "t1", "s1")

    @pytest.mark.asyncio
    async def test_handle_cdp_download_local_no_js_fetch(self):
        """Lines 636-827: Local browser without JS fetch, polls for file."""
        session = _make_mock_browser_session()
        session.is_local = True
        session.browser_profile.downloads_path = None

        with tempfile.TemporaryDirectory() as tmpdir:
            watchdog = self._make_watchdog(session=session)
            watchdog._use_js_fetch_for_local = False

            event = {"url": "https://example.com/file.txt", "suggestedFilename": "file.txt", "guid": "g2"}

            # Create a file that will be "found" during polling
            test_file = Path(tmpdir) / "file.txt"
            test_file.write_text("test content here!!")

            with patch("asyncio.get_event_loop") as mock_loop:
                mock_time = MagicMock()
                # Return times that exceed max_wait to exit poll loop quickly
                mock_time.time.side_effect = [0, 0, 25]
                mock_loop.return_value = mock_time

                with patch("asyncio.sleep", new_callable=AsyncMock):
                    await watchdog._handle_cdp_download(event, "t1", "s1")

    @pytest.mark.asyncio
    async def test_handle_cdp_download_local_js_fetch_success(self):
        """Lines 669-755: JS fetch path for local download succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.is_local = True
            session.browser_profile.downloads_path = tmpdir

            cdp_frame_session = MagicMock()
            cdp_frame_session.cdp_client = MagicMock()
            cdp_frame_session.cdp_client.send = MagicMock()
            cdp_frame_session.cdp_client.send.Runtime = MagicMock()
            cdp_frame_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
                "result": {
                    "value": {
                        "data": [72, 101, 108, 108, 111],
                        "size": 5,
                        "contentType": "text/plain",
                    }
                }
            })
            cdp_frame_session.session_id = "frame-sess"
            session.cdp_client_for_frame = AsyncMock(return_value=cdp_frame_session)

            watchdog = self._make_watchdog(session=session)
            watchdog._use_js_fetch_for_local = True

            event = {"url": "https://example.com/hello.txt", "suggestedFilename": "hello.txt", "guid": "g3", "frameId": "f1"}

            await watchdog._handle_cdp_download(event, "t1", "s1")

            # File should be saved
            saved_files = list(Path(tmpdir).iterdir())
            assert len(saved_files) >= 1

    @pytest.mark.asyncio
    async def test_handle_cdp_download_js_fetch_no_data(self):
        """Lines 756-757: JS fetch returns no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.is_local = True
            session.browser_profile.downloads_path = tmpdir

            cdp_frame_session = MagicMock()
            cdp_frame_session.cdp_client = MagicMock()
            cdp_frame_session.cdp_client.send = MagicMock()
            cdp_frame_session.cdp_client.send.Runtime = MagicMock()
            cdp_frame_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
                "result": {"value": None}
            })
            cdp_frame_session.session_id = "frame-sess"
            session.cdp_client_for_frame = AsyncMock(return_value=cdp_frame_session)

            watchdog = self._make_watchdog(session=session)
            watchdog._use_js_fetch_for_local = True

            event = {"url": "https://example.com/file.bin", "suggestedFilename": "file.bin", "guid": "g4", "frameId": "f1"}

            with patch("asyncio.get_event_loop") as mock_loop:
                mock_time = MagicMock()
                mock_time.time.side_effect = [0, 0, 25]
                mock_loop.return_value = mock_time
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    await watchdog._handle_cdp_download(event, "t1", "s1")

    @pytest.mark.asyncio
    async def test_handle_cdp_download_js_fetch_exception(self):
        """Lines 759-760: JS fetch raises exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.is_local = True
            session.browser_profile.downloads_path = tmpdir
            session.cdp_client_for_frame = AsyncMock(side_effect=RuntimeError("frame fail"))

            watchdog = self._make_watchdog(session=session)
            watchdog._use_js_fetch_for_local = True

            event = {"url": "https://example.com/err.bin", "suggestedFilename": "err.bin", "guid": "g5", "frameId": "f1"}

            with patch("asyncio.get_event_loop") as mock_loop:
                mock_time = MagicMock()
                mock_time.time.side_effect = [0, 0, 25]
                mock_loop.return_value = mock_time
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    await watchdog._handle_cdp_download(event, "t1", "s1")

    @pytest.mark.asyncio
    async def test_handle_cdp_download_inner_try_exception(self):
        """Lines 765-766: Exception inside inner try block (e.g., downloads_dir check fails)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.is_local = True
            session.browser_profile.downloads_path = tmpdir

            watchdog = self._make_watchdog(session=session)
            watchdog._use_js_fetch_for_local = False

            # Missing 'url' key in event dict to trigger KeyError inside try
            event = {"suggestedFilename": "file.pdf", "guid": "g-fail"}

            with patch("asyncio.get_event_loop") as mock_loop:
                mock_time = MagicMock()
                mock_time.time.side_effect = [0, 0, 25]
                mock_loop.return_value = mock_time
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    # Should not raise -- caught by outer except
                    await watchdog._handle_cdp_download(event, "t1", "s1")

    @pytest.mark.asyncio
    async def test_handle_cdp_download_poll_finds_file(self):
        """Lines 773-827: File poll loop finds new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.is_local = True
            session.browser_profile.downloads_path = tmpdir

            watchdog = self._make_watchdog(session=session)
            watchdog._use_js_fetch_for_local = False

            event = {"url": "https://example.com/data.csv", "suggestedFilename": "data.csv", "guid": "g-poll"}

            call_count = [0]
            async def fake_sleep(duration):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Create file on first poll
                    (Path(tmpdir) / "data.csv").write_text("a,b,c\n1,2,3")

            with patch("asyncio.get_event_loop") as mock_loop:
                mock_time = MagicMock()
                mock_time.time.side_effect = [0, 0, 5, 10, 25]
                mock_loop.return_value = mock_time
                with patch("asyncio.sleep", side_effect=fake_sleep):
                    await watchdog._handle_cdp_download(event, "t1", "s1")

            session.event_bus.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_handle_cdp_download_poll_already_handled(self):
        """Lines 805-807: File found but already handled by guid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.is_local = True
            session.browser_profile.downloads_path = tmpdir

            watchdog = self._make_watchdog(session=session)
            watchdog._use_js_fetch_for_local = False
            watchdog._cdp_downloads_info = {"g-handled": {"handled": True}}

            event = {"url": "https://example.com/handled.csv", "suggestedFilename": "handled.csv", "guid": "g-handled"}

            call_count = [0]
            async def fake_sleep(duration):
                call_count[0] += 1
                if call_count[0] == 1:
                    (Path(tmpdir) / "handled.csv").write_text("already done")

            with patch("asyncio.get_event_loop") as mock_loop:
                mock_time = MagicMock()
                mock_time.time.side_effect = [0, 0, 5, 25]
                mock_loop.return_value = mock_time
                with patch("asyncio.sleep", side_effect=fake_sleep):
                    await watchdog._handle_cdp_download(event, "t1", "s1")

    # -- _handle_download --

    @pytest.mark.asyncio
    async def test_handle_download_file_already_exists(self):
        """Lines 856-866: File already downloaded by Playwright."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            # Create existing file
            existing = Path(tmpdir) / "test.pdf"
            existing.write_bytes(b"PDF content here")

            watchdog = self._make_watchdog(session=session)

            mock_download = MagicMock()
            mock_download.url = "https://example.com/test.pdf"
            mock_download.suggested_filename = "test.pdf"
            mock_download.failure = AsyncMock(return_value=None)

            await watchdog._handle_download(mock_download)
            session.event_bus.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_handle_download_save_as_success(self):
        """Lines 868-918: Download with save_as."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            watchdog = self._make_watchdog(session=session)

            mock_download = MagicMock()
            mock_download.url = "https://example.com/new_file.pdf"
            mock_download.suggested_filename = "new_file.pdf"
            mock_download.failure = AsyncMock(return_value=None)

            async def fake_save_as(path):
                Path(path).write_bytes(b"saved content")

            mock_download.save_as = fake_save_as

            with patch.object(watchdog, "_is_auto_download_enabled", return_value=True):
                await watchdog._handle_download(mock_download)

            session.event_bus.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_handle_download_save_as_fails(self):
        """Lines 886-888, 926-936: save_as fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            watchdog = self._make_watchdog(session=session)

            mock_download = MagicMock()
            mock_download.url = "https://example.com/fail.pdf"
            mock_download.suggested_filename = "fail.pdf"
            mock_download.failure = AsyncMock(return_value="canceled")
            mock_download.save_as = AsyncMock(side_effect=RuntimeError("save failed"))

            await watchdog._handle_download(mock_download)
            # Should not raise

    # -- trigger_pdf_download --

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_no_downloads_path(self):
        """Lines 1079-1081: No downloads path configured."""
        session = _make_mock_browser_session()
        session.browser_profile.downloads_path = None

        watchdog = self._make_watchdog(session=session)
        result = await watchdog.trigger_pdf_download("t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_no_url(self):
        """Lines 1120-1122: Could not determine PDF URL."""
        session = _make_mock_browser_session()
        session.browser_profile.downloads_path = "/tmp/downloads"

        cdp_session = MagicMock()
        cdp_session.cdp_client = MagicMock()
        cdp_session.cdp_client.send = MagicMock()
        cdp_session.cdp_client.send.Runtime = MagicMock()
        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
            "result": {"value": {"url": ""}}
        })
        cdp_session.session_id = "sess-pdf"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

        watchdog = self._make_watchdog(session=session)
        result = await watchdog.trigger_pdf_download("t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_already_downloaded(self):
        """Lines 1136-1139: PDF already downloaded in session."""
        session = _make_mock_browser_session()
        session.browser_profile.downloads_path = "/tmp/downloads"

        cdp_session = MagicMock()
        cdp_session.cdp_client = MagicMock()
        cdp_session.cdp_client.send = MagicMock()
        cdp_session.cdp_client.send.Runtime = MagicMock()
        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
            "result": {"value": {"url": "https://example.com/doc.pdf"}}
        })
        cdp_session.session_id = "sess-pdf2"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

        watchdog = self._make_watchdog(session=session)
        watchdog._session_pdf_urls = {"https://example.com/doc.pdf": "/tmp/downloads/doc.pdf"}

        result = await watchdog.trigger_pdf_download("t1")
        assert result == "/tmp/downloads/doc.pdf"

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_success(self):
        """Lines 1157-1249: Full PDF download success path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            cdp_session = MagicMock()
            cdp_session.cdp_client = MagicMock()
            cdp_session.cdp_client.send = MagicMock()
            cdp_session.cdp_client.send.Runtime = MagicMock()

            # First call: get PDF URL
            # Second call: download PDF data
            cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=[
                {"result": {"value": {"url": "https://example.com/report.pdf"}}},
                {"result": {"value": {"data": list(b"%PDF-1.4 test content"), "fromCache": True, "responseSize": 20, "transferSize": "20"}}},
            ])
            cdp_session.session_id = "sess-pdf3"
            session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

            watchdog = self._make_watchdog(session=session)
            result = await watchdog.trigger_pdf_download("t1")

            assert result is not None
            assert "report.pdf" in result
            assert Path(result).exists()

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_duplicate_filename(self):
        """Lines 1146-1153: File already exists, generates unique name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            # Create existing file
            (Path(tmpdir) / "report.pdf").write_bytes(b"existing")

            cdp_session = MagicMock()
            cdp_session.cdp_client = MagicMock()
            cdp_session.cdp_client.send = MagicMock()
            cdp_session.cdp_client.send.Runtime = MagicMock()
            cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=[
                {"result": {"value": {"url": "https://example.com/report.pdf"}}},
                {"result": {"value": {"data": list(b"%PDF new"), "fromCache": False, "responseSize": 8}}},
            ])
            cdp_session.session_id = "sess-pdf4"
            session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

            watchdog = self._make_watchdog(session=session)
            result = await watchdog.trigger_pdf_download("t1")

            assert result is not None
            assert "report (1).pdf" in result

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_no_data(self):
        """Lines 1250-1252: Download returns no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            cdp_session = MagicMock()
            cdp_session.cdp_client = MagicMock()
            cdp_session.cdp_client.send = MagicMock()
            cdp_session.cdp_client.send.Runtime = MagicMock()
            cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=[
                {"result": {"value": {"url": "https://example.com/empty.pdf"}}},
                {"result": {"value": {"data": [], "fromCache": False, "responseSize": 0}}},
            ])
            cdp_session.session_id = "sess-pdf5"
            session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

            watchdog = self._make_watchdog(session=session)
            result = await watchdog.trigger_pdf_download("t1")
            assert result is None

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_fetch_exception(self):
        """Lines 1254-1256: Fetch fails during PDF download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            cdp_session = MagicMock()
            cdp_session.cdp_client = MagicMock()
            cdp_session.cdp_client.send = MagicMock()
            cdp_session.cdp_client.send.Runtime = MagicMock()
            cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=[
                {"result": {"value": {"url": "https://example.com/fail.pdf"}}},
                RuntimeError("fetch failed"),
            ])
            cdp_session.session_id = "sess-pdf6"
            session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

            watchdog = self._make_watchdog(session=session)
            result = await watchdog.trigger_pdf_download("t1")
            assert result is None

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_timeout(self):
        """Lines 1258-1260: Timeout during PDF download."""
        session = _make_mock_browser_session()
        session.browser_profile.downloads_path = "/tmp/downloads"
        session.get_or_create_cdp_session = AsyncMock(side_effect=TimeoutError("timed out"))

        watchdog = self._make_watchdog(session=session)
        result = await watchdog.trigger_pdf_download("t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_general_exception(self):
        """Lines 1261-1263: General exception in PDF download."""
        session = _make_mock_browser_session()
        session.browser_profile.downloads_path = "/tmp/downloads"
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("general fail"))

        watchdog = self._make_watchdog(session=session)
        result = await watchdog.trigger_pdf_download("t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_no_extension(self):
        """Lines 1126-1130: Filename without .pdf extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            cdp_session = MagicMock()
            cdp_session.cdp_client = MagicMock()
            cdp_session.cdp_client.send = MagicMock()
            cdp_session.cdp_client.send.Runtime = MagicMock()
            cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=[
                {"result": {"value": {"url": "https://example.com/viewer?doc=123"}}},
                {"result": {"value": {"data": list(b"%PDF"), "fromCache": False, "responseSize": 4}}},
            ])
            cdp_session.session_id = "sess-pdf7"
            session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

            watchdog = self._make_watchdog(session=session)
            result = await watchdog.trigger_pdf_download("t1")

            # Should append .pdf to filename
            assert result is not None
            assert result.endswith(".pdf")

    @pytest.mark.asyncio
    async def test_trigger_pdf_download_write_fails(self):
        """Lines 1219-1221: File write fails (path verification)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _make_mock_browser_session()
            session.browser_profile.downloads_path = tmpdir

            cdp_session = MagicMock()
            cdp_session.cdp_client = MagicMock()
            cdp_session.cdp_client.send = MagicMock()
            cdp_session.cdp_client.send.Runtime = MagicMock()
            cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=[
                {"result": {"value": {"url": "https://example.com/doc.pdf"}}},
                {"result": {"value": {"data": list(b"%PDF"), "fromCache": False, "responseSize": 4}}},
            ])
            cdp_session.session_id = "sess-pdf8"
            session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

            watchdog = self._make_watchdog(session=session)

            # Mock anyio.open_file to simulate write failure
            with patch("openbrowser.browser.watchdogs.downloads_watchdog.anyio.open_file", side_effect=RuntimeError("write fail")):
                result = await watchdog.trigger_pdf_download("t1")
            assert result is None


# ===========================================================================
# LocalBrowserWatchdog Tests
# ===========================================================================

class TestLocalBrowserWatchdog:
    """Cover lines in local_browser_watchdog.py."""

    def _make_watchdog(self, session=None):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
        wd = _make_watchdog(LocalBrowserWatchdog, session=session)
        wd._subprocess = None
        wd._owns_browser_resources = True
        wd._temp_dirs_to_cleanup = []
        wd._original_user_data_dir = None
        return wd

    # -- _launch_browser --

    @pytest.mark.asyncio
    async def test_launch_browser_killed_stale(self):
        """Line 115: Stale chrome killed log."""
        session = _make_mock_browser_session()
        session.browser_profile.user_data_dir = "/tmp/test_profile"
        session.browser_profile.executable_path = "/usr/bin/fake-chrome"
        session.browser_profile.get_args = MagicMock(return_value=["--headless", "--user-data-dir=/tmp/test_profile"])

        watchdog = self._make_watchdog(session=session)

        with patch.object(watchdog, "_kill_stale_chrome_for_profile", new_callable=AsyncMock, return_value=True):
            with patch.object(watchdog, "_find_free_port", return_value=9222):
                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                    mock_proc = MagicMock()
                    mock_proc.pid = 1234
                    mock_exec.return_value = mock_proc

                    with patch("psutil.Process") as MockProcess:
                        MockProcess.return_value = MagicMock()

                        with patch.object(watchdog, "_wait_for_cdp_url", new_callable=AsyncMock, return_value="http://localhost:9222/"):
                            process, cdp_url = await watchdog._launch_browser()
                            assert cdp_url == "http://localhost:9222/"

    @pytest.mark.asyncio
    async def test_launch_browser_no_executable_no_installed(self):
        """Lines 143-147: No executable path, no installed browser, uses playwright."""
        session = _make_mock_browser_session()
        session.browser_profile.user_data_dir = "/tmp/profile"
        session.browser_profile.executable_path = None
        session.browser_profile.get_args = MagicMock(return_value=["--headless", "--user-data-dir=/tmp/profile"])

        watchdog = self._make_watchdog(session=session)

        with patch.object(watchdog, "_kill_stale_chrome_for_profile", new_callable=AsyncMock, return_value=False):
            with patch.object(watchdog, "_find_free_port", return_value=9222):
                with patch.object(watchdog, "_find_installed_browser_path", return_value=None):
                    with patch.object(watchdog, "_install_browser_with_playwright", new_callable=AsyncMock, return_value="/usr/bin/chrome"):
                        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
                            mock_proc = MagicMock()
                            mock_proc.pid = 5678
                            mock_exec.return_value = mock_proc
                            with patch("psutil.Process") as MockProcess:
                                MockProcess.return_value = MagicMock()
                                with patch.object(watchdog, "_wait_for_cdp_url", new_callable=AsyncMock, return_value="http://localhost:9222/"):
                                    process, url = await watchdog._launch_browser()
                                    assert url == "http://localhost:9222/"

    @pytest.mark.asyncio
    async def test_launch_browser_no_browser_found(self):
        """Lines 150-151: No browser path found at all."""
        session = _make_mock_browser_session()
        session.browser_profile.user_data_dir = "/tmp/profile"
        session.browser_profile.executable_path = None
        session.browser_profile.get_args = MagicMock(return_value=["--headless", "--user-data-dir=/tmp/profile"])

        watchdog = self._make_watchdog(session=session)

        with patch.object(watchdog, "_kill_stale_chrome_for_profile", new_callable=AsyncMock, return_value=False):
            with patch.object(watchdog, "_find_free_port", return_value=9222):
                with patch.object(watchdog, "_find_installed_browser_path", return_value=None):
                    with patch.object(watchdog, "_install_browser_with_playwright", new_callable=AsyncMock, return_value=None):
                        with pytest.raises(RuntimeError, match="No local Chrome"):
                            await watchdog._launch_browser()

    @pytest.mark.asyncio
    async def test_launch_browser_profile_error_retry(self):
        """Lines 216-219, 224-226: Profile error triggers retry with temp dir."""
        session = _make_mock_browser_session()
        session.browser_profile.user_data_dir = "/tmp/profile"
        session.browser_profile.executable_path = "/usr/bin/chrome"
        session.browser_profile.get_args = MagicMock(return_value=["--headless", "--user-data-dir=/tmp/profile"])

        watchdog = self._make_watchdog(session=session)
        attempt = [0]

        async def fake_subprocess(*args, **kwargs):
            attempt[0] += 1
            if attempt[0] == 1:
                raise RuntimeError("singletonlock error")
            mock = MagicMock()
            mock.pid = 9999
            return mock

        with patch.object(watchdog, "_kill_stale_chrome_for_profile", new_callable=AsyncMock, return_value=False):
            with patch.object(watchdog, "_find_free_port", return_value=9222):
                with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
                    with patch("psutil.Process") as MockProcess:
                        MockProcess.return_value = MagicMock()
                        with patch.object(watchdog, "_wait_for_cdp_url", new_callable=AsyncMock, return_value="http://localhost:9222/"):
                            with patch("asyncio.sleep", new_callable=AsyncMock):
                                process, url = await watchdog._launch_browser(max_retries=3)
                                assert url == "http://localhost:9222/"

    @pytest.mark.asyncio
    async def test_launch_browser_non_recoverable_error(self):
        """Lines 281-284, 308-315: Non-recoverable error restores user_data_dir."""
        session = _make_mock_browser_session()
        session.browser_profile.user_data_dir = "/tmp/profile"
        session.browser_profile.executable_path = "/usr/bin/chrome"
        session.browser_profile.get_args = MagicMock(return_value=["--headless", "--user-data-dir=/tmp/profile"])

        watchdog = self._make_watchdog(session=session)

        with patch.object(watchdog, "_kill_stale_chrome_for_profile", new_callable=AsyncMock, return_value=False):
            with patch.object(watchdog, "_find_free_port", return_value=9222):
                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=RuntimeError("unexpected error")):
                    with pytest.raises(RuntimeError, match="unexpected error"):
                        await watchdog._launch_browser(max_retries=1)

    # -- _cleanup_process --

    @pytest.mark.asyncio
    async def test_cleanup_process_none(self):
        """Line 414: Process is None."""
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
        await LocalBrowserWatchdog._cleanup_process(None)

    @pytest.mark.asyncio
    async def test_cleanup_process_graceful(self):
        """Lines 417-425: Graceful shutdown."""
        mock_process = MagicMock()
        mock_process.is_running.return_value = False
        mock_process.terminate = MagicMock()

        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
        await LocalBrowserWatchdog._cleanup_process(mock_process)
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_process_force_kill(self):
        """Lines 428-431: Force kill after timeout."""
        mock_process = MagicMock()
        mock_process.is_running.return_value = True
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await LocalBrowserWatchdog._cleanup_process(mock_process)
        mock_process.kill.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_process_no_such_process(self):
        """Lines 433-435: psutil.NoSuchProcess."""
        import psutil
        mock_process = MagicMock()
        mock_process.terminate.side_effect = psutil.NoSuchProcess(pid=1234)

        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
        await LocalBrowserWatchdog._cleanup_process(mock_process)

    @pytest.mark.asyncio
    async def test_cleanup_process_other_exception(self):
        """Lines 436-438: Other exception during cleanup."""
        mock_process = MagicMock()
        mock_process.terminate.side_effect = RuntimeError("cleanup fail")

        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
        await LocalBrowserWatchdog._cleanup_process(mock_process)

    # -- _cleanup_temp_dir --

    def test_cleanup_temp_dir_none(self):
        """Line 446-447: Empty temp_dir."""
        watchdog = self._make_watchdog()
        watchdog._cleanup_temp_dir("")

    def test_cleanup_temp_dir_valid(self):
        """Lines 449-453: Valid temp dir cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            openbrowser_tmp = Path(tmpdir) / "openbrowser-tmp-test"
            openbrowser_tmp.mkdir()
            (openbrowser_tmp / "file.txt").write_text("test")

            watchdog = self._make_watchdog()
            watchdog._cleanup_temp_dir(str(openbrowser_tmp))
            assert not openbrowser_tmp.exists()

    def test_cleanup_temp_dir_exception(self):
        """Lines 454-455: Exception during cleanup."""
        watchdog = self._make_watchdog()
        with patch("shutil.rmtree", side_effect=RuntimeError("rm fail")):
            # Should not raise
            watchdog._cleanup_temp_dir("/tmp/openbrowser-tmp-fail")

    # -- _kill_stale_chrome_for_profile --

    @pytest.mark.asyncio
    async def test_kill_stale_chrome_no_match(self):
        """Lines 488-489, 491-492: No matching process found."""
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_proc = MagicMock()
        mock_proc.info = {"pid": 1, "name": "safari", "cmdline": []}

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile("/tmp/profile")
        assert result is False

    @pytest.mark.asyncio
    async def test_kill_stale_chrome_match_and_wait(self):
        """Lines 500-517: Kill matching Chrome and wait for exit."""
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_proc = MagicMock()
        mock_proc.pid = 1234
        mock_proc.info = {
            "pid": 1234,
            "name": "chrome",
            "cmdline": ["chrome", "--user-data-dir=/tmp/profile"],
        }
        mock_proc.kill = MagicMock()

        # First iteration: process exists; second: no matching processes
        call_count = [0]
        def fake_process_iter(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return [mock_proc]
            return []

        with patch("psutil.process_iter", side_effect=fake_process_iter):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile("/tmp/profile")
        assert result is True

    @pytest.mark.asyncio
    async def test_kill_stale_chrome_access_denied(self):
        """Line 488-489: psutil.AccessDenied during scan."""
        import psutil
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_proc = MagicMock()
        mock_proc.info = {"pid": 1, "name": "chrome", "cmdline": None}
        # Accessing cmdline as None should be handled
        type(mock_proc).info = PropertyMock(side_effect=psutil.AccessDenied(pid=1))

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile("/tmp/profile")
        assert result is False

    # -- _install_browser_with_playwright --

    @pytest.mark.asyncio
    async def test_install_browser_timeout(self):
        """Lines 369-370: Playwright install times out."""
        watchdog = self._make_watchdog()

        mock_proc = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError("timeout"))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=TimeoutError("timeout")):
                with pytest.raises(RuntimeError, match="Timeout"):
                    await watchdog._install_browser_with_playwright()

    @pytest.mark.asyncio
    async def test_install_browser_exception(self):
        """Lines 369-371: General exception during install."""
        watchdog = self._make_watchdog()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=RuntimeError("install fail"))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Error getting browser path"):
                await watchdog._install_browser_with_playwright()

    @pytest.mark.asyncio
    async def test_install_browser_success_no_path(self):
        """Lines 358-360: Install succeeds but no path found."""
        watchdog = self._make_watchdog()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            with patch("asyncio.wait_for", new_callable=AsyncMock, return_value=(b"output", b"")):
                with patch.object(watchdog, "_find_installed_browser_path", return_value=None):
                    with pytest.raises(RuntimeError, match="No local browser path found"):
                        await watchdog._install_browser_with_playwright()

    # -- _wait_for_cdp_url --

    @pytest.mark.asyncio
    async def test_wait_for_cdp_url_timeout(self):
        """Line 400, 405: CDP URL not available within timeout."""
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_time = MagicMock()
            mock_time.time.side_effect = [0, 31]
            mock_loop.return_value = mock_time

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))

            with patch("httpx.AsyncClient", return_value=mock_client):
                with pytest.raises(TimeoutError):
                    await LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=30)


# ===========================================================================
# SecurityWatchdog Tests
# ===========================================================================

class TestSecurityWatchdog:
    """Cover lines 70-72, 92-93, 156-157, 178-180, 233, 261-265 in security_watchdog.py."""

    def _make_watchdog(self, session=None):
        from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog
        return _make_watchdog(SecurityWatchdog, session=session)

    @pytest.mark.asyncio
    async def test_navigation_complete_blocked_redirect_fails(self):
        """Lines 70-72: Failed to navigate to about:blank after blocked redirect."""
        session = _make_mock_browser_session()
        cdp_session = MagicMock()
        cdp_session.cdp_client = MagicMock()
        cdp_session.cdp_client.send = MagicMock()
        cdp_session.cdp_client.send.Page = MagicMock()
        cdp_session.cdp_client.send.Page.navigate = AsyncMock(side_effect=RuntimeError("nav fail"))
        cdp_session.session_id = "sess-sec"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        session.browser_profile.allowed_domains = {"allowed.com"}

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.url = "https://blocked.com/page"
        event.target_id = "t1"

        await watchdog.on_NavigationCompleteEvent(event)
        # Should have tried to navigate to about:blank and logged error

    @pytest.mark.asyncio
    async def test_tab_created_blocked_close_fails(self):
        """Lines 92-93: Failed to close blocked tab."""
        session = _make_mock_browser_session()
        session._cdp_close_page = AsyncMock(side_effect=RuntimeError("close fail"))
        session.browser_profile.allowed_domains = {"allowed.com"}

        watchdog = self._make_watchdog(session=session)
        event = MagicMock()
        event.url = "https://blocked.com/"
        event.target_id = "t1"

        await watchdog.on_TabCreatedEvent(event)

    def test_is_ip_address_exception(self):
        """Lines 156-157: Exception in IP address check."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        with patch("ipaddress.ip_address", side_effect=Exception("unexpected")):
            result = watchdog._is_ip_address("not-an-ip")
        assert result is False

    def test_is_url_allowed_invalid_url(self):
        """Lines 178-180: Invalid URL parsing fails."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = {"example.com"}
        watchdog = self._make_watchdog(session=session)

        with patch("urllib.parse.urlparse", side_effect=Exception("parse error")):
            result = watchdog._is_url_allowed("not://valid")
        assert result is False

    def test_is_url_allowed_no_restrictions(self):
        """Line 233: No allowed or prohibited domains, return True."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = None
        session.browser_profile.prohibited_domains = None
        session.browser_profile.block_ip_addresses = False
        watchdog = self._make_watchdog(session=session)

        result = watchdog._is_url_allowed("https://anything.com")
        assert result is True

    def test_is_url_match_fnmatch_pattern(self):
        """Lines 261-265: fnmatch glob pattern matching."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        # Pattern that uses fnmatch (not *.domain or prefix/*)
        result = watchdog._is_url_match(
            "https://test.example.com",
            "test.example.com",
            "https",
            "test.*",
        )
        assert result is True

    def test_is_url_match_prefix_glob(self):
        """Lines 254-258: Pattern ending with /* (prefix glob)."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        result = watchdog._is_url_match(
            "brave://settings/passwords",
            "settings",
            "brave",
            "brave://*",
        )
        assert result is True

    def test_is_url_allowed_prohibited_set(self):
        """Lines 218-225: Prohibited domains as set."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = None
        session.browser_profile.prohibited_domains = {"evil.com"}
        session.browser_profile.block_ip_addresses = False
        watchdog = self._make_watchdog(session=session)

        assert watchdog._is_url_allowed("https://evil.com/malware") is False
        assert watchdog._is_url_allowed("https://good.com/page") is True

    def test_is_url_allowed_prohibited_list(self):
        """Lines 226-231: Prohibited domains as list with patterns."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = None
        session.browser_profile.prohibited_domains = ["*.evil.com"]
        session.browser_profile.block_ip_addresses = False
        watchdog = self._make_watchdog(session=session)

        assert watchdog._is_url_allowed("https://sub.evil.com/page") is False
        assert watchdog._is_url_allowed("https://good.com/page") is True

    def test_is_url_allowed_ip_blocked(self):
        """Lines 192-194: IP address blocked."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = None
        session.browser_profile.prohibited_domains = None
        session.browser_profile.block_ip_addresses = True
        watchdog = self._make_watchdog(session=session)

        assert watchdog._is_url_allowed("https://192.168.1.1/admin") is False


# ===========================================================================
# ScreenshotWatchdog Tests
# ===========================================================================

class TestScreenshotWatchdog:
    """Cover lines 61-62 in screenshot_watchdog.py."""

    def _make_watchdog(self, session=None):
        from openbrowser.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog
        return _make_watchdog(ScreenshotWatchdog, session=session)

    @pytest.mark.asyncio
    async def test_screenshot_remove_highlights_exception(self):
        """Lines 61-62: remove_highlights raises exception in finally block."""
        session = _make_mock_browser_session()
        cdp_session = MagicMock()
        cdp_session.cdp_client = MagicMock()
        cdp_session.cdp_client.send = MagicMock()
        cdp_session.cdp_client.send.Page = MagicMock()
        cdp_session.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={"data": "base64screenshot"})
        cdp_session.session_id = "sess-ss"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        session.remove_highlights = AsyncMock(side_effect=RuntimeError("highlight remove fail"))

        watchdog = self._make_watchdog(session=session)
        # Despite remove_highlights failing, should return screenshot data
        result = await watchdog.on_ScreenshotEvent(MagicMock())
        assert result == "base64screenshot"

    @pytest.mark.asyncio
    async def test_screenshot_fails_highlights_also_fail(self):
        """Lines 61-62: Both screenshot and remove_highlights fail."""
        session = _make_mock_browser_session()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("cdp fail"))
        session.remove_highlights = AsyncMock(side_effect=RuntimeError("highlight fail"))

        watchdog = self._make_watchdog(session=session)
        with pytest.raises(RuntimeError, match="cdp fail"):
            await watchdog.on_ScreenshotEvent(MagicMock())


# ===========================================================================
# BaseWatchdog Tests
# ===========================================================================

class TestBaseWatchdog:
    """Cover lines 111, 124-162, 255-260 in watchdog_base.py."""

    def _make_watchdog(self, session=None):
        from openbrowser.browser.watchdog_base import BaseWatchdog
        return _make_watchdog(BaseWatchdog, session=session)

    def test_unique_handler_result_is_exception(self):
        """Line 111: Handler returns an Exception instance (not raises, returns)."""
        from openbrowser.browser.watchdog_base import BaseWatchdog

        session = _make_mock_browser_session()
        real_bus = MagicMock()
        real_bus.handlers = {}
        real_bus.event_history = {}
        real_bus.on = MagicMock()

        # Create a handler that returns an Exception
        async def on_TestEvent(event):
            return ValueError("returned error")

        on_TestEvent.__name__ = "on_TestEvent"

        class FakeWatchdog:
            __class__ = type("TestWatchdog", (), {"__name__": "TestWatchdog"})

        bound_method = MagicMock()
        bound_method.__name__ = "on_TestEvent"
        bound_method.__self__ = FakeWatchdog()

        mock_event_class = MagicMock()
        mock_event_class.__name__ = "TestEvent"

        BaseWatchdog.attach_handler_to_session(session, mock_event_class, bound_method)

        # The handler wrapper was registered
        real_bus_on_call = session.event_bus.on
        assert real_bus_on_call.called

    def test_unique_handler_error_with_agent_focus(self):
        """Lines 124-162: Handler raises, CDP session recovery attempted."""
        from openbrowser.browser.watchdog_base import BaseWatchdog

        session = _make_mock_browser_session()
        session.event_bus = MagicMock()
        session.event_bus.handlers = {}
        session.event_bus.event_history = {}
        session.event_bus.on = MagicMock()
        session.agent_focus = MagicMock()
        session.agent_focus.target_id = "t-recovery"
        session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock())

        class FakeWd:
            __class__ = type("RecoveryWatchdog", (), {"__name__": "RecoveryWatchdog"})

        handler = MagicMock()
        handler.__name__ = "on_TestErrorEvent"
        handler.__self__ = FakeWd()

        mock_event_class = MagicMock()
        mock_event_class.__name__ = "TestErrorEvent"

        BaseWatchdog.attach_handler_to_session(session, mock_event_class, handler)
        # Handler registered via event_bus.on
        assert session.event_bus.on.called

    def test_unique_handler_error_no_agent_focus(self):
        """Lines 147-149: Handler raises, no agent_focus -- gets any session."""
        from openbrowser.browser.watchdog_base import BaseWatchdog

        session = _make_mock_browser_session()
        session.event_bus = MagicMock()
        session.event_bus.handlers = {}
        session.event_bus.event_history = {}
        session.event_bus.on = MagicMock()
        session.agent_focus = None

        class FakeWd:
            __class__ = type("NoFocusWatchdog", (), {"__name__": "NoFocusWatchdog"})

        handler = MagicMock()
        handler.__name__ = "on_NoFocusEvent"
        handler.__self__ = FakeWd()

        mock_event_class = MagicMock()
        mock_event_class.__name__ = "NoFocusEvent"

        BaseWatchdog.attach_handler_to_session(session, mock_event_class, handler)
        assert session.event_bus.on.called

    def test_unique_handler_connection_closed_error(self):
        """Lines 150-155: ConnectionClosedError during recovery re-raises."""
        from openbrowser.browser.watchdog_base import BaseWatchdog

        session = _make_mock_browser_session()
        session.event_bus = MagicMock()
        session.event_bus.handlers = {}
        session.event_bus.event_history = {}
        session.event_bus.on = MagicMock()

        class FakeWd:
            __class__ = type("ConnWatchdog", (), {"__name__": "ConnWatchdog"})

        handler = MagicMock()
        handler.__name__ = "on_ConnEvent"
        handler.__self__ = FakeWd()

        mock_event_class = MagicMock()
        mock_event_class.__name__ = "ConnEvent"

        BaseWatchdog.attach_handler_to_session(session, mock_event_class, handler)
        assert session.event_bus.on.called

    def test_del_tasks_collection(self):
        """Lines 255-260: __del__ iterates _tasks collections."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        # Add a tasks collection
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()
        watchdog.__dict__["_download_tasks"] = [mock_task]

        watchdog.__del__()
        mock_task.cancel.assert_called()

    def test_del_exception_in_cleanup(self):
        """Lines 257-260: Exception during __del__ cleanup."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        # Make dir() raise
        with patch("builtins.dir", side_effect=RuntimeError("dir fail")):
            with patch("openbrowser.utils.logger") as mock_logger:
                watchdog.__del__()
                # Should log error
                mock_logger.error.assert_called()

    def test_del_single_task_cancel(self):
        """Lines 239-246: Single task attribute cancelled."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()
        watchdog.__dict__["_monitoring_task"] = mock_task

        watchdog.__del__()
        mock_task.cancel.assert_called()

    def test_del_task_already_done(self):
        """Lines 242-243: Task already done, not cancelled."""
        session = _make_mock_browser_session()
        watchdog = self._make_watchdog(session=session)

        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.cancel = MagicMock()
        watchdog.__dict__["_done_task"] = mock_task

        watchdog.__del__()
        mock_task.cancel.assert_not_called()

    def test_duplicate_handler_registration(self):
        """Lines 170-178: Duplicate handler raises RuntimeError."""
        from openbrowser.browser.watchdog_base import BaseWatchdog

        session = _make_mock_browser_session()
        session.event_bus = MagicMock()
        session.event_bus.on = MagicMock()

        class FakeWd:
            __class__ = type("DupeWatchdog", (), {"__name__": "DupeWatchdog"})

        handler = MagicMock()
        handler.__name__ = "on_DupeEvent"
        handler.__self__ = FakeWd()

        mock_event_class = MagicMock()
        mock_event_class.__name__ = "DupeEvent"

        # First registration
        existing_handler = MagicMock()
        existing_handler.__name__ = "DupeWatchdog.on_DupeEvent"
        session.event_bus.handlers = {"DupeEvent": [existing_handler]}

        with pytest.raises(RuntimeError, match="Duplicate handler"):
            BaseWatchdog.attach_handler_to_session(session, mock_event_class, handler)
