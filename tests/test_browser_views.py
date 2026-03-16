"""Tests for browser views data classes."""

import base64
import logging
from unittest.mock import MagicMock

from openbrowser.browser.views import (
    BrowserError,
    BrowserStateHistory,
    BrowserStateSummary,
    TabInfo,
    URLNotAllowedError,
)

logger = logging.getLogger(__name__)


class TestBrowserViews:
    """Tests for browser/views.py data classes."""

    def _make_tab_info(self, url, title, target_id, parent_target_id=None):
        """Create a TabInfo instance."""
        return TabInfo(
            url=url,
            title=title,
            target_id=target_id,
            parent_target_id=parent_target_id,
        )

    def test_tab_info_serialization_truncates_target_id(self):
        """TabInfo serializes target_id to last 4 chars."""
        tab = self._make_tab_info(
            url="https://example.com",
            title="Example",
            target_id="ABCDEFGH12345678ABCDEFGH12345678",
        )
        data = tab.model_dump(by_alias=True)
        assert data["tab_id"] == "5678"

    def test_tab_info_serialization_parent_target_id(self):
        """TabInfo serializes parent_target_id to last 4 chars."""
        tab = self._make_tab_info(
            url="https://popup.com",
            title="Popup",
            target_id="ABCDEFGH12345678ABCDEFGH12345678",
            parent_target_id="11112222333344445555666677778888",
        )
        data = tab.model_dump(by_alias=True)
        assert data["parent_tab_id"] == "8888"

    def test_tab_info_serialization_parent_none(self):
        """TabInfo serializes None parent_target_id as None."""
        tab = self._make_tab_info(
            url="https://example.com",
            title="Example",
            target_id="ABCDEFGH12345678ABCDEFGH12345678",
            parent_target_id=None,
        )
        data = tab.model_dump(by_alias=True)
        assert data["parent_tab_id"] is None

    def test_browser_state_summary_str_single_tab(self):
        """__str__ with a single tab shows URL and title, no tab list."""
        tab = self._make_tab_info(
            url="https://example.com",
            title="Example",
            target_id="ABCDEFGH12345678ABCDEFGH12345678",
        )
        mock_dom = MagicMock()
        mock_dom.selector_map = {}
        mock_dom.eval_representation = MagicMock(return_value="")

        state = BrowserStateSummary(
            dom_state=mock_dom,
            url="https://example.com",
            title="Example",
            tabs=[tab],
        )
        text = str(state)
        assert "URL: https://example.com" in text
        assert "Title: Example" in text
        # Single tab -- should NOT show tab listing
        assert "Tabs" not in text

    def test_browser_state_summary_str_multiple_tabs(self):
        """__str__ with multiple tabs shows tab listing with active marker."""
        tab1 = self._make_tab_info(
            url="https://example.com",
            title="Example",
            target_id="AAAA1111BBBB2222CCCC3333DDDD4444",
        )
        tab2 = self._make_tab_info(
            url="https://other.com",
            title="Other Page With A Very Long Title That Should Be Truncated After Sixty Characters In The Display",
            target_id="EEEE5555FFFF6666GGGG7777HHHH8888",
        )
        mock_dom = MagicMock()
        mock_dom.selector_map = {1: MagicMock(), 2: MagicMock()}
        mock_dom.eval_representation = MagicMock(return_value="<div>content</div>")

        state = BrowserStateSummary(
            dom_state=mock_dom,
            url="https://example.com",
            title="Example",
            tabs=[tab1, tab2],
        )
        text = str(state)
        assert "Tabs (2)" in text
        assert "(active)" in text
        assert "Interactive elements: 2" in text
        assert "<div>content</div>" in text

    def test_browser_state_summary_str_empty_dom(self):
        """__str__ when dom_state has no selector_map."""
        tab = self._make_tab_info(
            url="https://example.com",
            title="Title",
            target_id="AAAA1111BBBB2222CCCC3333DDDD4444",
        )
        mock_dom = MagicMock()
        mock_dom.selector_map = None
        mock_dom.eval_representation = MagicMock(return_value="")

        state = BrowserStateSummary(
            dom_state=mock_dom,
            url="https://example.com",
            title="Title",
            tabs=[tab],
        )
        text = str(state)
        assert "Interactive elements" not in text


class TestBrowserError:
    """Tests for BrowserError.__str__ formatting."""

    def test_browser_error_message_only(self):
        """BrowserError with only message shows just the message."""
        err = BrowserError("Something went wrong")
        assert str(err) == "Something went wrong"

    def test_browser_error_with_details(self):
        """BrowserError with details and event shows both."""
        err = BrowserError(
            "Click failed",
            details={"element": "button", "index": 5},
            event=MagicMock(__str__=lambda self: "ClickEvent(index=5)"),
        )
        text = str(err)
        assert "Click failed" in text
        assert "element" in text
        assert "button" in text

    def test_browser_error_with_event_no_details(self):
        """BrowserError with event but no details."""
        mock_event = MagicMock()
        mock_event.__str__ = MagicMock(return_value="NavigateEvent(url=...)")
        err = BrowserError("Nav failed", event=mock_event)
        text = str(err)
        assert "Nav failed" in text
        assert "while handling" in text

    def test_browser_error_memory_fields(self):
        """BrowserError stores short_term_memory and long_term_memory."""
        err = BrowserError(
            "Error",
            short_term_memory="Try clicking another element",
            long_term_memory="Element index 5 is not clickable",
        )
        assert err.short_term_memory == "Try clicking another element"
        assert err.long_term_memory == "Element index 5 is not clickable"


class TestBrowserStateHistory:
    """Tests for BrowserStateHistory serialization."""

    def test_to_dict(self):
        """to_dict returns correct structure."""
        tab = TabInfo(
            url="https://example.com",
            title="Test",
            target_id="AAAA1111BBBB2222CCCC3333DDDD4444",
        )
        history = BrowserStateHistory(
            url="https://example.com",
            title="Test",
            tabs=[tab],
            interacted_element=[None],
        )
        d = history.to_dict()
        assert d["url"] == "https://example.com"
        assert len(d["tabs"]) == 1
        assert d["interacted_element"] == [None]

    def test_get_screenshot_no_path(self):
        """get_screenshot returns None when no screenshot_path."""
        history = BrowserStateHistory(
            url="https://example.com",
            title="Test",
            tabs=[],
            interacted_element=[],
        )
        assert history.get_screenshot() is None

    def test_get_screenshot_missing_file(self, tmp_path):
        """get_screenshot returns None when file does not exist."""
        history = BrowserStateHistory(
            url="https://example.com",
            title="Test",
            tabs=[],
            interacted_element=[],
            screenshot_path=str(tmp_path / "nonexistent.png"),
        )
        assert history.get_screenshot() is None

    def test_get_screenshot_valid_file(self, tmp_path):
        """get_screenshot returns base64 encoded content."""
        img_data = b"PNG_DATA_HERE"
        img_path = tmp_path / "screenshot.png"
        img_path.write_bytes(img_data)

        history = BrowserStateHistory(
            url="https://example.com",
            title="Test",
            tabs=[],
            interacted_element=[],
            screenshot_path=str(img_path),
        )
        result = history.get_screenshot()
        assert result == base64.b64encode(img_data).decode("utf-8")


class TestURLNotAllowedError:
    """Tests for URLNotAllowedError subclass."""

    def test_inherits_browser_error(self):
        """URLNotAllowedError is a subclass of BrowserError."""
        err = URLNotAllowedError("https://evil.com is not allowed")
        assert isinstance(err, BrowserError)
        assert str(err) == "https://evil.com is not allowed"
