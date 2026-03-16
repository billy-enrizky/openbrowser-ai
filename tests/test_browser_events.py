"""Comprehensive tests for openbrowser.browser.events module.

Covers: _get_timeout, all event classes, BrowserLaunchResult,
_check_event_names_dont_overlap.
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from openbrowser.browser.events import (
    AgentFocusChangedEvent,
    BrowserConnectedEvent,
    BrowserErrorEvent,
    BrowserKillEvent,
    BrowserLaunchEvent,
    BrowserLaunchResult,
    BrowserStartEvent,
    BrowserStateRequestEvent,
    BrowserStopEvent,
    BrowserStoppedEvent,
    ClickElementEvent,
    CloseTabEvent,
    DialogOpenedEvent,
    FileDownloadedEvent,
    GetDropdownOptionsEvent,
    GoBackEvent,
    GoForwardEvent,
    LoadStorageStateEvent,
    NavigateToUrlEvent,
    NavigationCompleteEvent,
    NavigationStartedEvent,
    RefreshEvent,
    SaveStorageStateEvent,
    ScreenshotEvent,
    ScrollEvent,
    ScrollToTextEvent,
    SelectDropdownOptionEvent,
    SendKeysEvent,
    StorageStateLoadedEvent,
    StorageStateSavedEvent,
    SwitchTabEvent,
    TabClosedEvent,
    TabCreatedEvent,
    TargetCrashedEvent,
    TypeTextEvent,
    UploadFileEvent,
    WaitEvent,
    AboutBlankDVDScreensaverShownEvent,
    _check_event_names_dont_overlap,
    _get_timeout,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _get_timeout
# ---------------------------------------------------------------------------


class TestGetTimeout:
    def test_default_value(self):
        # Without env var set, should return default
        result = _get_timeout("NONEXISTENT_TIMEOUT_VAR_12345", 15.0)
        assert result == 15.0

    def test_env_var_value(self):
        with patch.dict(os.environ, {"TEST_TIMEOUT_XYZ": "30.0"}):
            result = _get_timeout("TEST_TIMEOUT_XYZ", 15.0)
            assert result == 30.0

    def test_invalid_env_var_returns_default(self):
        with patch.dict(os.environ, {"TEST_TIMEOUT_XYZ": "not_a_number"}):
            result = _get_timeout("TEST_TIMEOUT_XYZ", 15.0)
            assert result == 15.0

    def test_negative_env_var_returns_default(self):
        with patch.dict(os.environ, {"TEST_TIMEOUT_XYZ": "-5.0"}):
            result = _get_timeout("TEST_TIMEOUT_XYZ", 15.0)
            assert result == 15.0

    def test_zero_env_var(self):
        with patch.dict(os.environ, {"TEST_TIMEOUT_XYZ": "0.0"}):
            result = _get_timeout("TEST_TIMEOUT_XYZ", 15.0)
            assert result == 0.0

    def test_integer_env_var(self):
        with patch.dict(os.environ, {"TEST_TIMEOUT_XYZ": "10"}):
            result = _get_timeout("TEST_TIMEOUT_XYZ", 15.0)
            assert result == 10.0


# ---------------------------------------------------------------------------
# Event instantiation tests
# ---------------------------------------------------------------------------


class TestNavigateToUrlEvent:
    def test_defaults(self):
        event = NavigateToUrlEvent(url="https://example.com")
        assert event.url == "https://example.com"
        assert event.wait_until == "load"
        assert event.new_tab is False
        assert event.timeout_ms is None

    def test_custom_values(self):
        event = NavigateToUrlEvent(
            url="https://example.com",
            wait_until="domcontentloaded",
            new_tab=True,
            timeout_ms=5000,
        )
        assert event.wait_until == "domcontentloaded"
        assert event.new_tab is True
        assert event.timeout_ms == 5000


class TestScreenshotEvent:
    def test_defaults(self):
        event = ScreenshotEvent()
        assert event.full_page is False
        assert event.clip is None

    def test_with_clip(self):
        event = ScreenshotEvent(clip={"x": 0, "y": 0, "width": 100, "height": 100})
        assert event.clip["width"] == 100


class TestBrowserStateRequestEvent:
    def test_defaults(self):
        event = BrowserStateRequestEvent()
        assert event.include_dom is True
        assert event.include_screenshot is True
        assert event.include_recent_events is False


class TestSwitchTabEvent:
    def test_defaults(self):
        event = SwitchTabEvent()
        assert event.target_id is None

    def test_custom_target(self):
        event = SwitchTabEvent(target_id="target123")
        assert event.target_id == "target123"


class TestCloseTabEvent:
    def test_creation(self):
        event = CloseTabEvent(target_id="target123")
        assert event.target_id == "target123"


class TestWaitEvent:
    def test_defaults(self):
        event = WaitEvent()
        assert event.seconds == 3.0
        assert event.max_seconds == 10.0


class TestSendKeysEvent:
    def test_creation(self):
        event = SendKeysEvent(keys="ctrl+a")
        assert event.keys == "ctrl+a"


class TestGoBackEvent:
    def test_creation(self):
        event = GoBackEvent()
        assert event.event_timeout is not None


class TestGoForwardEvent:
    def test_creation(self):
        event = GoForwardEvent()
        assert event.event_timeout is not None


class TestRefreshEvent:
    def test_creation(self):
        event = RefreshEvent()
        assert event.event_timeout is not None


class TestScrollToTextEvent:
    def test_defaults(self):
        event = ScrollToTextEvent(text="Find me")
        assert event.text == "Find me"
        assert event.direction == "down"


class TestBrowserStartEvent:
    def test_defaults(self):
        event = BrowserStartEvent()
        assert event.cdp_url is None
        assert event.launch_options == {}


class TestBrowserStopEvent:
    def test_defaults(self):
        event = BrowserStopEvent()
        assert event.force is False


class TestBrowserLaunchResult:
    def test_creation(self):
        result = BrowserLaunchResult(cdp_url="ws://localhost:9222")
        assert result.cdp_url == "ws://localhost:9222"


class TestBrowserLaunchEvent:
    def test_creation(self):
        event = BrowserLaunchEvent()
        assert event.event_timeout is not None


class TestBrowserKillEvent:
    def test_creation(self):
        event = BrowserKillEvent()
        assert event.event_timeout is not None


class TestBrowserConnectedEvent:
    def test_creation(self):
        event = BrowserConnectedEvent(cdp_url="ws://localhost:9222")
        assert event.cdp_url == "ws://localhost:9222"


class TestBrowserStoppedEvent:
    def test_defaults(self):
        event = BrowserStoppedEvent()
        assert event.reason is None

    def test_custom_reason(self):
        event = BrowserStoppedEvent(reason="User closed")
        assert event.reason == "User closed"


class TestTabCreatedEvent:
    def test_creation(self):
        event = TabCreatedEvent(target_id="t1", url="https://example.com")
        assert event.target_id == "t1"
        assert event.url == "https://example.com"


class TestTabClosedEvent:
    def test_creation(self):
        event = TabClosedEvent(target_id="t1")
        assert event.target_id == "t1"


class TestAgentFocusChangedEvent:
    def test_creation(self):
        event = AgentFocusChangedEvent(target_id="t1", url="https://example.com")
        assert event.url == "https://example.com"


class TestTargetCrashedEvent:
    def test_creation(self):
        event = TargetCrashedEvent(target_id="t1", error="OOM")
        assert event.error == "OOM"


class TestNavigationStartedEvent:
    def test_creation(self):
        event = NavigationStartedEvent(target_id="t1", url="https://example.com")
        assert event.url == "https://example.com"


class TestNavigationCompleteEvent:
    def test_defaults(self):
        event = NavigationCompleteEvent(target_id="t1", url="https://example.com")
        assert event.status is None
        assert event.error_message is None
        assert event.loading_status is None

    def test_with_error(self):
        event = NavigationCompleteEvent(
            target_id="t1",
            url="https://example.com",
            status=404,
            error_message="Not found",
        )
        assert event.status == 404


class TestBrowserErrorEvent:
    def test_creation(self):
        event = BrowserErrorEvent(
            error_type="NetworkError",
            message="Connection refused",
        )
        assert event.error_type == "NetworkError"
        assert event.details == {}


class TestSaveStorageStateEvent:
    def test_defaults(self):
        event = SaveStorageStateEvent()
        assert event.path is None


class TestStorageStateSavedEvent:
    def test_creation(self):
        event = StorageStateSavedEvent(path="/tmp/state.json", cookies_count=5, origins_count=3)
        assert event.cookies_count == 5


class TestLoadStorageStateEvent:
    def test_defaults(self):
        event = LoadStorageStateEvent()
        assert event.path is None


class TestStorageStateLoadedEvent:
    def test_creation(self):
        event = StorageStateLoadedEvent(path="/tmp/state.json", cookies_count=10, origins_count=2)
        assert event.origins_count == 2


class TestFileDownloadedEvent:
    def test_defaults(self):
        event = FileDownloadedEvent(
            url="https://example.com/file.pdf",
            path="/tmp/file.pdf",
            file_name="file.pdf",
            file_size=1024,
        )
        assert event.file_type is None
        assert event.mime_type is None
        assert event.from_cache is False
        assert event.auto_download is False


class TestAboutBlankDVDScreensaverShownEvent:
    def test_creation(self):
        event = AboutBlankDVDScreensaverShownEvent(target_id="t1")
        assert event.error is None


class TestDialogOpenedEvent:
    def test_creation(self):
        event = DialogOpenedEvent(
            dialog_type="alert",
            message="Hello!",
            url="https://example.com",
        )
        assert event.dialog_type == "alert"
        assert event.frame_id is None


# ---------------------------------------------------------------------------
# _check_event_names_dont_overlap
# ---------------------------------------------------------------------------


class TestCheckEventNamesDontOverlap:
    def test_no_overlapping_names(self):
        """This test verifies the module-level check passed during import."""
        # If we got here, it means the check passed at import time
        # Call it explicitly to verify
        _check_event_names_dont_overlap()


# ---------------------------------------------------------------------------
# Event timeout defaults
# ---------------------------------------------------------------------------


class TestEventTimeoutDefaults:
    """Verify all events have sensible default timeouts."""

    def test_navigate_timeout(self):
        assert NavigateToUrlEvent(url="x").event_timeout == 15.0

    def test_click_timeout(self):
        node = MagicMock()
        node.node_id = 1
        node.backend_node_id = 1
        node.session_id = "s"
        node.frame_id = "f"
        node.target_id = "t"
        node.node_type = 1
        node.node_name = "DIV"
        node.node_value = ""
        node.attributes = {}
        node.is_scrollable = False
        node.is_visible = True
        node.absolute_position = None
        # ClickElementEvent requires an EnhancedDOMTreeNode, use defaults
        event = ClickElementEvent(node=node)
        assert event.event_timeout == 15.0

    def test_wait_timeout(self):
        assert WaitEvent().event_timeout == 60.0

    def test_screenshot_timeout(self):
        assert ScreenshotEvent().event_timeout == 8.0

    def test_browser_state_request_timeout(self):
        assert BrowserStateRequestEvent().event_timeout == 30.0
