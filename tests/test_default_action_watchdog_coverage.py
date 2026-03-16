"""Tests for openbrowser.browser.watchdogs.default_action_watchdog module -- 100% coverage target."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch, PropertyMock

import pytest
from bubus import EventBus

from openbrowser.browser.session import BrowserSession
from openbrowser.browser.views import BrowserError, URLNotAllowedError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_element_node(**overrides):
    """Create a mock EnhancedDOMTreeNode."""
    node = MagicMock()
    node.backend_node_id = overrides.get("backend_node_id", 123)
    node.tag_name = overrides.get("tag_name", "button")
    node.node_name = overrides.get("node_name", "BUTTON")
    node.attributes = overrides.get("attributes", {})
    node.xpath = overrides.get("xpath", "/html/body/button")
    node.frame_id = overrides.get("frame_id", None)
    node.target_id = overrides.get("target_id", None)
    node.is_scrollable = overrides.get("is_scrollable", False)
    node.get_all_children_text = MagicMock(return_value="Click me")
    return node


def _make_mock_browser_session():
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test_default_action")
    session.event_bus = MagicMock()
    session._cdp_client_root = MagicMock()
    session.agent_focus = MagicMock()
    session.agent_focus.target_id = "AAAA1111BBBB2222CCCC3333DDDD4444"
    session.agent_focus.session_id = "sess-1234"
    session.agent_focus.cdp_client = MagicMock()
    session.get_or_create_cdp_session = AsyncMock()
    session.cdp_client = MagicMock()
    session.cdp_client_for_node = AsyncMock()
    session.cdp_client_for_frame = AsyncMock()
    session.get_element_coordinates = AsyncMock(return_value=None)
    session.is_file_input = MagicMock(return_value=False)
    session.get_current_page_url = AsyncMock(return_value="https://example.com")
    session.get_current_page_title = AsyncMock(return_value="Example")
    session.id = "test-session"
    session.is_local = True
    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = "/tmp/downloads"
    return session


def _make_cdp_session_mock():
    cdp_session = MagicMock()
    cdp_session.cdp_client = MagicMock()
    cdp_session.cdp_client.send = MagicMock()
    cdp_session.cdp_client.send.DOM = MagicMock()
    cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded = AsyncMock()
    cdp_session.cdp_client.send.DOM.resolveNode = AsyncMock(return_value={"object": {"objectId": "obj-1"}})
    cdp_session.cdp_client.send.DOM.focus = AsyncMock()
    cdp_session.cdp_client.send.DOM.getBoxModel = AsyncMock()
    cdp_session.cdp_client.send.DOM.enable = AsyncMock()
    cdp_session.cdp_client.send.DOM.getDocument = AsyncMock()
    cdp_session.cdp_client.send.DOM.performSearch = AsyncMock()
    cdp_session.cdp_client.send.DOM.getSearchResults = AsyncMock()
    cdp_session.cdp_client.send.DOM.discardSearchResults = AsyncMock()
    cdp_session.cdp_client.send.DOM.setFileInputFiles = AsyncMock()
    cdp_session.cdp_client.send.Input = MagicMock()
    cdp_session.cdp_client.send.Input.dispatchMouseEvent = AsyncMock()
    cdp_session.cdp_client.send.Input.dispatchKeyEvent = AsyncMock()
    cdp_session.cdp_client.send.Runtime = MagicMock()
    cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": True}})
    cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={"result": {"value": {"cleared": True, "method": "value", "finalText": ""}}})
    cdp_session.cdp_client.send.Runtime.runIfWaitingForDebugger = AsyncMock()
    cdp_session.cdp_client.send.Page = MagicMock()
    cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={
        "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
    })
    cdp_session.cdp_client.send.Page.getNavigationHistory = AsyncMock(return_value={
        "currentIndex": 1,
        "entries": [
            {"id": 1, "url": "https://first.com"},
            {"id": 2, "url": "https://current.com"},
            {"id": 3, "url": "https://next.com"},
        ]
    })
    cdp_session.cdp_client.send.Page.navigateToHistoryEntry = AsyncMock()
    cdp_session.cdp_client.send.Page.reload = AsyncMock()
    cdp_session.cdp_client.send.Page.printToPDF = AsyncMock()
    cdp_session.session_id = "cdp-sess-1"
    return cdp_session


def _make_watchdog(session=None, event_bus=None):
    from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = MagicMock()
    session.event_bus = event_bus
    return DefaultActionWatchdog.model_construct(
        event_bus=event_bus,
        browser_session=session,
    )


# ---------------------------------------------------------------------------
# _is_print_related_element
# ---------------------------------------------------------------------------

class TestIsPrintRelatedElement:
    def test_print_onclick(self):
        watchdog = _make_watchdog()
        node = _make_element_node(attributes={"onclick": "window.print()"})
        assert watchdog._is_print_related_element(node) is True

    def test_no_print(self):
        watchdog = _make_watchdog()
        node = _make_element_node(attributes={"onclick": "doSomething()"})
        assert watchdog._is_print_related_element(node) is False

    def test_no_attributes(self):
        watchdog = _make_watchdog()
        node = _make_element_node(attributes=None)
        assert watchdog._is_print_related_element(node) is False

    def test_empty_onclick(self):
        watchdog = _make_watchdog()
        node = _make_element_node(attributes={"onclick": ""})
        assert watchdog._is_print_related_element(node) is False


# ---------------------------------------------------------------------------
# _handle_print_button_click
# ---------------------------------------------------------------------------

class TestHandlePrintButtonClick:
    @pytest.mark.asyncio
    async def test_success(self):
        import base64
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        session.get_current_page_title = AsyncMock(return_value="Test Page")
        session.get_current_page_url = AsyncMock(return_value="https://example.com/page")
        session.browser_profile.downloads_path = "/tmp/test_downloads"

        pdf_data = base64.b64encode(b"fake pdf content").decode()
        cdp_session.cdp_client.send.Page.printToPDF = AsyncMock(return_value={"data": pdf_data})

        watchdog = _make_watchdog(session=session)
        node = _make_element_node()

        with patch("anyio.open_file") as mock_open:
            mock_file = AsyncMock()
            mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
            mock_open.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("pathlib.Path.exists", return_value=False):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 100
                    with patch("pathlib.Path.mkdir"):
                        result = await watchdog._handle_print_button_click(node)

        assert result is not None
        assert result.get("pdf_generated") is True

    @pytest.mark.asyncio
    async def test_no_pdf_data(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Page.printToPDF = AsyncMock(return_value={"data": None})
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        result = await watchdog._handle_print_button_click(_make_element_node())
        assert result is None

    @pytest.mark.asyncio
    async def test_no_downloads_path(self):
        session = _make_mock_browser_session()
        session.browser_profile.downloads_path = None
        cdp_session = _make_cdp_session_mock()
        import base64
        cdp_session.cdp_client.send.Page.printToPDF = AsyncMock(return_value={"data": base64.b64encode(b"pdf").decode()})
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        result = await watchdog._handle_print_button_click(_make_element_node())
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Page.printToPDF = AsyncMock(side_effect=TimeoutError())
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        result = await watchdog._handle_print_button_click(_make_element_node())
        assert result is None

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Page.printToPDF = AsyncMock(side_effect=RuntimeError("fail"))
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        result = await watchdog._handle_print_button_click(_make_element_node())
        assert result is None


# ---------------------------------------------------------------------------
# on_ClickElementEvent
# ---------------------------------------------------------------------------

class TestOnClickElementEvent:
    @pytest.mark.asyncio
    async def test_no_agent_focus(self):
        session = _make_mock_browser_session()
        session.agent_focus = None
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        with pytest.raises(BrowserError):
            await watchdog.on_ClickElementEvent(event)

    @pytest.mark.asyncio
    async def test_no_target_id(self):
        session = _make_mock_browser_session()
        session.agent_focus.target_id = None
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        with pytest.raises(BrowserError):
            await watchdog.on_ClickElementEvent(event)

    @pytest.mark.asyncio
    async def test_file_input_returns_validation_error(self):
        session = _make_mock_browser_session()
        session.is_file_input = MagicMock(return_value=True)
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        result = await watchdog.on_ClickElementEvent(event)
        assert isinstance(result, dict) and "validation_error" in result

    @pytest.mark.asyncio
    async def test_print_element_generates_pdf(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(attributes={"onclick": "window.print()"})
        event = MagicMock()
        event.node = node

        with patch.object(watchdog, "_handle_print_button_click", new_callable=AsyncMock, return_value={"pdf_generated": True, "path": "/tmp/test.pdf"}):
            result = await watchdog.on_ClickElementEvent(event)
            assert result.get("pdf_generated") is True

    @pytest.mark.asyncio
    async def test_print_element_fallback_to_click(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(attributes={"onclick": "window.print()"})
        event = MagicMock()
        event.node = node

        with patch.object(watchdog, "_handle_print_button_click", new_callable=AsyncMock, return_value=None):
            with patch.object(watchdog, "_click_element_node_impl", new_callable=AsyncMock, return_value=None):
                result = await watchdog.on_ClickElementEvent(event)

    @pytest.mark.asyncio
    async def test_successful_click(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()

        with patch.object(watchdog, "_click_element_node_impl", new_callable=AsyncMock, return_value={"click_x": 100, "click_y": 200}):
            result = await watchdog.on_ClickElementEvent(event)
            assert result == {"click_x": 100, "click_y": 200}

    @pytest.mark.asyncio
    async def test_click_validation_error_from_impl(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()

        with patch.object(watchdog, "_click_element_node_impl", new_callable=AsyncMock, return_value={"validation_error": "Cannot click"}):
            result = await watchdog.on_ClickElementEvent(event)
            assert "validation_error" in result

    @pytest.mark.asyncio
    async def test_click_raises_exception(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()

        with patch.object(watchdog, "_click_element_node_impl", new_callable=AsyncMock, side_effect=RuntimeError("click fail")):
            with pytest.raises(RuntimeError):
                await watchdog.on_ClickElementEvent(event)


# ---------------------------------------------------------------------------
# on_TypeTextEvent
# ---------------------------------------------------------------------------

class TestOnTypeTextEvent:
    @pytest.mark.asyncio
    async def test_type_to_page_when_no_backend_node(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(backend_node_id=0)
        event = MagicMock()
        event.node = node
        event.text = "hello"
        event.is_sensitive = False
        event.sensitive_key_name = None
        event.clear = True

        with patch.object(watchdog, "_type_to_page", new_callable=AsyncMock):
            result = await watchdog.on_TypeTextEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_type_to_element(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(backend_node_id=42)
        event = MagicMock()
        event.node = node
        event.text = "test input"
        event.is_sensitive = False
        event.sensitive_key_name = None
        event.clear = True

        with patch.object(watchdog, "_input_text_element_node_impl", new_callable=AsyncMock, return_value={"input_x": 100, "input_y": 50}):
            result = await watchdog.on_TypeTextEvent(event)
            assert result == {"input_x": 100, "input_y": 50}

    @pytest.mark.asyncio
    async def test_type_sensitive_with_key_name(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(backend_node_id=42)
        event = MagicMock()
        event.node = node
        event.text = "secret"
        event.is_sensitive = True
        event.sensitive_key_name = "password"
        event.clear = True

        with patch.object(watchdog, "_input_text_element_node_impl", new_callable=AsyncMock, return_value=None):
            result = await watchdog.on_TypeTextEvent(event)

    @pytest.mark.asyncio
    async def test_type_sensitive_no_key_name(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(backend_node_id=0)
        event = MagicMock()
        event.node = node
        event.text = "secret"
        event.is_sensitive = True
        event.sensitive_key_name = None
        event.clear = True

        with patch.object(watchdog, "_type_to_page", new_callable=AsyncMock):
            await watchdog.on_TypeTextEvent(event)

    @pytest.mark.asyncio
    async def test_type_fallback_to_page(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(backend_node_id=42)
        event = MagicMock()
        event.node = node
        event.text = "fallback text"
        event.is_sensitive = False
        event.sensitive_key_name = None
        event.clear = True

        with patch.object(watchdog, "_input_text_element_node_impl", new_callable=AsyncMock, side_effect=RuntimeError("no element")):
            with patch.object(watchdog, "_click_element_node_impl", new_callable=AsyncMock, return_value=None):
                with patch.object(watchdog, "_type_to_page", new_callable=AsyncMock):
                    result = await watchdog.on_TypeTextEvent(event)
                    assert result is None

    @pytest.mark.asyncio
    async def test_type_fallback_sensitive(self):
        session = _make_mock_browser_session()
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(backend_node_id=42)
        event = MagicMock()
        event.node = node
        event.text = "secret"
        event.is_sensitive = True
        event.sensitive_key_name = "api_key"
        event.clear = True

        with patch.object(watchdog, "_input_text_element_node_impl", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            with patch.object(watchdog, "_click_element_node_impl", new_callable=AsyncMock, side_effect=RuntimeError("click fail too")):
                with patch.object(watchdog, "_type_to_page", new_callable=AsyncMock):
                    await watchdog.on_TypeTextEvent(event)


# ---------------------------------------------------------------------------
# on_ScrollEvent
# ---------------------------------------------------------------------------

class TestOnScrollEvent:
    @pytest.mark.asyncio
    async def test_no_agent_focus(self):
        session = _make_mock_browser_session()
        session.agent_focus = None
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.direction = "down"
        event.amount = 300
        event.node = None
        with pytest.raises(BrowserError):
            await watchdog.on_ScrollEvent(event)

    @pytest.mark.asyncio
    async def test_scroll_down(self):
        watchdog = _make_watchdog()
        event = MagicMock()
        event.direction = "down"
        event.amount = 300
        event.node = None
        with patch.object(watchdog, "_scroll_with_cdp_gesture", new_callable=AsyncMock, return_value=True):
            result = await watchdog.on_ScrollEvent(event)
            assert result is None

    @pytest.mark.asyncio
    async def test_scroll_up(self):
        watchdog = _make_watchdog()
        event = MagicMock()
        event.direction = "up"
        event.amount = 200
        event.node = None
        with patch.object(watchdog, "_scroll_with_cdp_gesture", new_callable=AsyncMock, return_value=True):
            result = await watchdog.on_ScrollEvent(event)

    @pytest.mark.asyncio
    async def test_scroll_element(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="div")
        event = MagicMock()
        event.direction = "down"
        event.amount = 100
        event.node = node
        with patch.object(watchdog, "_scroll_element_container", new_callable=AsyncMock, return_value=True):
            result = await watchdog.on_ScrollEvent(event)

    @pytest.mark.asyncio
    async def test_scroll_iframe_element(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="IFRAME")
        event = MagicMock()
        event.direction = "down"
        event.amount = 150
        event.node = node
        with patch.object(watchdog, "_scroll_element_container", new_callable=AsyncMock, return_value=True):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await watchdog.on_ScrollEvent(event)

    @pytest.mark.asyncio
    async def test_scroll_element_fails_falls_through(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="div")
        event = MagicMock()
        event.direction = "down"
        event.amount = 100
        event.node = node
        with patch.object(watchdog, "_scroll_element_container", new_callable=AsyncMock, return_value=False):
            with patch.object(watchdog, "_scroll_with_cdp_gesture", new_callable=AsyncMock, return_value=True):
                result = await watchdog.on_ScrollEvent(event)


# ---------------------------------------------------------------------------
# _check_element_occlusion
# ---------------------------------------------------------------------------

class TestCheckElementOcclusion:
    @pytest.mark.asyncio
    async def test_element_clickable(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"isClickable": True, "targetInfo": {}}}
        })
        result = await watchdog._check_element_occlusion(123, 100.0, 50.0, cdp_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_element_occluded(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"isClickable": False, "targetInfo": {"tagName": "DIV"}, "elementAtPointInfo": {"tagName": "OVERLAY"}}}
        })
        result = await watchdog._check_element_occlusion(123, 100.0, 50.0, cdp_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_resolve_node_fails(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.DOM.resolveNode = AsyncMock(return_value={})
        result = await watchdog._check_element_occlusion(123, 100.0, 50.0, cdp_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_result_value(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={})
        result = await watchdog._check_element_occlusion(123, 100.0, 50.0, cdp_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.DOM.resolveNode = AsyncMock(side_effect=RuntimeError("fail"))
        result = await watchdog._check_element_occlusion(123, 100.0, 50.0, cdp_session)
        assert result is False


# ---------------------------------------------------------------------------
# _type_to_page
# ---------------------------------------------------------------------------

class TestTypeToPage:
    @pytest.mark.asyncio
    async def test_types_characters(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await watchdog._type_to_page("hi")

    @pytest.mark.asyncio
    async def test_types_newline(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await watchdog._type_to_page("a\nb")

    @pytest.mark.asyncio
    async def test_exception(self):
        session = _make_mock_browser_session()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_watchdog(session=session)

        with pytest.raises(Exception, match="Failed to type to page"):
            await watchdog._type_to_page("text")


# ---------------------------------------------------------------------------
# _get_char_modifiers_and_vk / _get_key_code_for_char
# ---------------------------------------------------------------------------

class TestCharHelpers:
    def test_shift_chars(self):
        watchdog = _make_watchdog()
        mods, vk, base = watchdog._get_char_modifiers_and_vk("!")
        assert mods == 8  # Shift
        assert base == "1"

    def test_uppercase(self):
        watchdog = _make_watchdog()
        mods, vk, base = watchdog._get_char_modifiers_and_vk("A")
        assert mods == 8
        assert base == "a"

    def test_lowercase(self):
        watchdog = _make_watchdog()
        mods, vk, base = watchdog._get_char_modifiers_and_vk("a")
        assert mods == 0

    def test_digit(self):
        watchdog = _make_watchdog()
        mods, vk, base = watchdog._get_char_modifiers_and_vk("5")
        assert mods == 0
        assert vk == ord("5")

    def test_no_shift_special(self):
        watchdog = _make_watchdog()
        mods, vk, base = watchdog._get_char_modifiers_and_vk(" ")
        assert mods == 0
        assert vk == 32

    def test_fallback_char(self):
        watchdog = _make_watchdog()
        # Unicode character
        mods, vk, base = watchdog._get_char_modifiers_and_vk("\u00e9")
        assert isinstance(mods, int)

    def test_key_code_digit(self):
        watchdog = _make_watchdog()
        assert watchdog._get_key_code_for_char("5") == "Digit5"

    def test_key_code_letter(self):
        watchdog = _make_watchdog()
        assert watchdog._get_key_code_for_char("a") == "KeyA"

    def test_key_code_special(self):
        watchdog = _make_watchdog()
        assert watchdog._get_key_code_for_char(" ") == "Space"
        assert watchdog._get_key_code_for_char(".") == "Period"

    def test_key_code_unknown(self):
        watchdog = _make_watchdog()
        result = watchdog._get_key_code_for_char("\u00e9")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _requires_direct_value_assignment
# ---------------------------------------------------------------------------

class TestRequiresDirectValueAssignment:
    def test_date_input(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "date"})
        assert watchdog._requires_direct_value_assignment(node) is True

    def test_text_input(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "text"})
        assert watchdog._requires_direct_value_assignment(node) is False

    def test_datepicker_class(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "text", "class": "form-control datepicker"})
        assert watchdog._requires_direct_value_assignment(node) is True

    def test_data_datepicker_attr(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "text", "data-datepicker": "true"})
        assert watchdog._requires_direct_value_assignment(node) is True

    def test_no_tag_name(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name=None, attributes={})
        assert watchdog._requires_direct_value_assignment(node) is False

    def test_no_attributes(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes=None)
        assert watchdog._requires_direct_value_assignment(node) is False

    def test_color_input(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "color"})
        assert watchdog._requires_direct_value_assignment(node) is True

    def test_range_input(self):
        watchdog = _make_watchdog()
        node = _make_element_node(tag_name="input", attributes={"type": "range"})
        assert watchdog._requires_direct_value_assignment(node) is True


# ---------------------------------------------------------------------------
# on_GoBackEvent / on_GoForwardEvent
# ---------------------------------------------------------------------------

class TestNavigationEvents:
    @pytest.mark.asyncio
    async def test_go_back(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)

        event = MagicMock()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await watchdog.on_GoBackEvent(event)
        cdp_session.cdp_client.send.Page.navigateToHistoryEntry.assert_awaited()

    @pytest.mark.asyncio
    async def test_go_back_no_history(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Page.getNavigationHistory = AsyncMock(return_value={
            "currentIndex": 0, "entries": [{"id": 1, "url": "https://only.com"}]
        })
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)

        event = MagicMock()
        await watchdog.on_GoBackEvent(event)
        cdp_session.cdp_client.send.Page.navigateToHistoryEntry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_go_forward(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)

        event = MagicMock()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await watchdog.on_GoForwardEvent(event)
        cdp_session.cdp_client.send.Page.navigateToHistoryEntry.assert_awaited()

    @pytest.mark.asyncio
    async def test_go_forward_no_next(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Page.getNavigationHistory = AsyncMock(return_value={
            "currentIndex": 0, "entries": [{"id": 1, "url": "https://only.com"}]
        })
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)

        event = MagicMock()
        await watchdog.on_GoForwardEvent(event)
        cdp_session.cdp_client.send.Page.navigateToHistoryEntry.assert_not_awaited()


# ---------------------------------------------------------------------------
# on_RefreshEvent
# ---------------------------------------------------------------------------

class TestOnRefreshEvent:
    @pytest.mark.asyncio
    async def test_refresh(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)

        event = MagicMock()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await watchdog.on_RefreshEvent(event)
        cdp_session.cdp_client.send.Page.reload.assert_awaited()


# ---------------------------------------------------------------------------
# on_WaitEvent
# ---------------------------------------------------------------------------

class TestOnWaitEvent:
    @pytest.mark.asyncio
    async def test_wait(self):
        watchdog = _make_watchdog()
        event = MagicMock()
        event.seconds = 1.0
        event.max_seconds = 10.0

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await watchdog.on_WaitEvent(event)
            mock_sleep.assert_awaited_once_with(1.0)

    @pytest.mark.asyncio
    async def test_wait_capped(self):
        watchdog = _make_watchdog()
        event = MagicMock()
        event.seconds = 30.0
        event.max_seconds = 10.0

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await watchdog.on_WaitEvent(event)
            mock_sleep.assert_awaited_once_with(10.0)

    @pytest.mark.asyncio
    async def test_wait_negative_clamped(self):
        watchdog = _make_watchdog()
        event = MagicMock()
        event.seconds = -5.0
        event.max_seconds = 10.0

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await watchdog.on_WaitEvent(event)
            mock_sleep.assert_awaited_once_with(0)


# ---------------------------------------------------------------------------
# on_SendKeysEvent
# ---------------------------------------------------------------------------

class TestOnSendKeysEvent:
    @pytest.mark.asyncio
    async def test_single_key_enter(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.keys = "Enter"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await watchdog.on_SendKeysEvent(event)

    @pytest.mark.asyncio
    async def test_key_combination(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.keys = "ctrl+a"

        await watchdog.on_SendKeysEvent(event)

    @pytest.mark.asyncio
    async def test_text_keys(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.keys = "abc"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await watchdog.on_SendKeysEvent(event)

    @pytest.mark.asyncio
    async def test_key_aliases(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.keys = "esc"

        await watchdog.on_SendKeysEvent(event)

    @pytest.mark.asyncio
    async def test_newline_in_text(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.keys = "\n"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await watchdog.on_SendKeysEvent(event)


# ---------------------------------------------------------------------------
# on_UploadFileEvent
# ---------------------------------------------------------------------------

class TestOnUploadFileEvent:
    @pytest.mark.asyncio
    async def test_not_file_input(self):
        session = _make_mock_browser_session()
        session.is_file_input = MagicMock(return_value=False)
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        event.file_path = "/tmp/test.txt"

        with pytest.raises(BrowserError):
            await watchdog.on_UploadFileEvent(event)

    @pytest.mark.asyncio
    async def test_upload_success(self):
        session = _make_mock_browser_session()
        session.is_file_input = MagicMock(return_value=True)
        session.cdp_client.send.DOM.setFileInputFiles = AsyncMock()
        watchdog = _make_watchdog(session=session)

        with patch.object(watchdog, "_get_session_id_for_element", new_callable=AsyncMock, return_value="sess-1"):
            event = MagicMock()
            event.node = _make_element_node()
            event.file_path = "/tmp/test.txt"
            await watchdog.on_UploadFileEvent(event)


# ---------------------------------------------------------------------------
# on_ScrollToTextEvent
# ---------------------------------------------------------------------------

class TestOnScrollToTextEvent:
    @pytest.mark.asyncio
    async def test_text_found_via_xpath(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client = cdp_session.cdp_client
        watchdog = _make_watchdog(session=session)

        cdp_session.cdp_client.send.DOM.performSearch = AsyncMock(return_value={"searchId": "s1", "resultCount": 1})
        cdp_session.cdp_client.send.DOM.getSearchResults = AsyncMock(return_value={"nodeIds": [42]})
        cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded = AsyncMock()
        cdp_session.cdp_client.send.DOM.discardSearchResults = AsyncMock()
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": {"nodeId": 1}})
        cdp_session.cdp_client.send.DOM.enable = AsyncMock()

        event = MagicMock()
        event.text = "Hello"
        result = await watchdog.on_ScrollToTextEvent(event)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_agent_focus(self):
        session = _make_mock_browser_session()
        session.agent_focus = None
        watchdog = _make_watchdog(session=session)

        event = MagicMock()
        event.text = "test"
        with pytest.raises(BrowserError):
            await watchdog.on_ScrollToTextEvent(event)

    @pytest.mark.asyncio
    async def test_text_found_via_js_fallback(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client = cdp_session.cdp_client

        # All XPath searches fail
        cdp_session.cdp_client.send.DOM.performSearch = AsyncMock(return_value={"searchId": "s1", "resultCount": 0})
        cdp_session.cdp_client.send.DOM.discardSearchResults = AsyncMock()
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": {"nodeId": 1}})
        cdp_session.cdp_client.send.DOM.enable = AsyncMock()

        # JS fallback succeeds
        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": True}})

        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.text = "FindMe"
        result = await watchdog.on_ScrollToTextEvent(event)
        assert result is None

    @pytest.mark.asyncio
    async def test_text_not_found(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client = cdp_session.cdp_client

        cdp_session.cdp_client.send.DOM.performSearch = AsyncMock(return_value={"searchId": "s1", "resultCount": 0})
        cdp_session.cdp_client.send.DOM.discardSearchResults = AsyncMock()
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": {"nodeId": 1}})
        cdp_session.cdp_client.send.DOM.enable = AsyncMock()
        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": False}})

        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.text = "NonExistent"

        with pytest.raises(BrowserError, match="Text not found"):
            await watchdog.on_ScrollToTextEvent(event)


# ---------------------------------------------------------------------------
# on_GetDropdownOptionsEvent / on_SelectDropdownOptionEvent
# ---------------------------------------------------------------------------

class TestDropdownEvents:
    @pytest.mark.asyncio
    async def test_get_dropdown_options_success(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {
                "type": "select",
                "options": [
                    {"text": "Option 1", "value": "1", "index": 0, "selected": True},
                    {"text": "Option 2", "value": "2", "index": 1, "selected": False},
                ],
                "id": "my-select",
                "name": "my-dropdown",
                "source": "target",
            }}
        })
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        result = await watchdog.on_GetDropdownOptionsEvent(event)
        assert result["type"] == "select"

    @pytest.mark.asyncio
    async def test_get_dropdown_options_error(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"error": "Not a dropdown"}}
        })
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        with pytest.raises(BrowserError):
            await watchdog.on_GetDropdownOptionsEvent(event)

    @pytest.mark.asyncio
    async def test_get_dropdown_no_options(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"type": "select", "options": [], "id": "", "name": "", "source": "target"}}
        })
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        result = await watchdog.on_GetDropdownOptionsEvent(event)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_dropdown_resolve_fails(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.DOM.resolveNode = AsyncMock(return_value={"object": {}})
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        with pytest.raises(BrowserError):
            await watchdog.on_GetDropdownOptionsEvent(event)

    @pytest.mark.asyncio
    async def test_get_dropdown_timeout(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=TimeoutError())
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        with pytest.raises(BrowserError):
            await watchdog.on_GetDropdownOptionsEvent(event)

    @pytest.mark.asyncio
    async def test_select_dropdown_success(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"success": True, "message": "Selected Option 1", "value": "1"}}
        })
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        event.text = "Option 1"
        result = await watchdog.on_SelectDropdownOptionEvent(event)
        assert result["success"] == "true"

    @pytest.mark.asyncio
    async def test_select_dropdown_not_found_with_options(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {
                "success": False,
                "error": "Option not found",
                "availableOptions": [{"text": "A", "value": "a"}, {"text": "B", "value": "b"}]
            }}
        })
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        event.text = "C"
        result = await watchdog.on_SelectDropdownOptionEvent(event)
        assert result["success"] == "false"
        assert "short_term_memory" in result

    @pytest.mark.asyncio
    async def test_select_dropdown_not_found_no_options(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"success": False, "error": "No dropdown found"}}
        })
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        event.text = "X"
        result = await watchdog.on_SelectDropdownOptionEvent(event)
        assert result["success"] == "false"

    @pytest.mark.asyncio
    async def test_select_dropdown_exception(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        event.text = "X"
        with pytest.raises(ValueError):
            await watchdog.on_SelectDropdownOptionEvent(event)

    @pytest.mark.asyncio
    async def test_get_dropdown_child_source(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {
                "type": "aria",
                "options": [{"text": "Item", "value": "item", "index": 0, "selected": False}],
                "id": "",
                "name": "",
                "source": "child-depth-1",
            }}
        })
        watchdog = _make_watchdog(session=session)
        event = MagicMock()
        event.node = _make_element_node()
        result = await watchdog.on_GetDropdownOptionsEvent(event)
        assert "child-depth-1" in result.get("source", "")


# ---------------------------------------------------------------------------
# _scroll_with_cdp_gesture
# ---------------------------------------------------------------------------

class TestScrollWithCdpGesture:
    @pytest.mark.asyncio
    async def test_scroll_success(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.agent_focus.cdp_client = cdp_session.cdp_client
        watchdog = _make_watchdog(session=session)
        result = await watchdog._scroll_with_cdp_gesture(300)
        assert result is True

    @pytest.mark.asyncio
    async def test_scroll_no_agent_focus(self):
        session = _make_mock_browser_session()
        session.agent_focus = None
        watchdog = _make_watchdog(session=session)
        # AssertionError is caught by the except Exception handler, returns False
        result = await watchdog._scroll_with_cdp_gesture(300)
        assert result is False

    @pytest.mark.asyncio
    async def test_scroll_exception(self):
        session = _make_mock_browser_session()
        session.agent_focus.cdp_client.send.Page.getLayoutMetrics = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_watchdog(session=session)
        result = await watchdog._scroll_with_cdp_gesture(300)
        assert result is False


# ---------------------------------------------------------------------------
# _scroll_element_container
# ---------------------------------------------------------------------------

class TestScrollElementContainer:
    @pytest.mark.asyncio
    async def test_scroll_iframe_content(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"success": True, "scrolled": 100}}
        })
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(tag_name="IFRAME")
        result = await watchdog._scroll_element_container(node, 100)
        assert result is True

    @pytest.mark.asyncio
    async def test_scroll_regular_element(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={
            "model": {"content": [0, 0, 100, 0, 100, 50, 0, 50]}
        })
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(tag_name="div")
        result = await watchdog._scroll_element_container(node, 200)
        assert result is True

    @pytest.mark.asyncio
    async def test_scroll_exception(self):
        session = _make_mock_browser_session()
        session.cdp_client_for_node = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_watchdog(session=session)
        node = _make_element_node()
        result = await watchdog._scroll_element_container(node, 100)
        assert result is False

    @pytest.mark.asyncio
    async def test_scroll_iframe_fails(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"success": False, "error": "cross-origin"}}
        })
        # Falls through to regular element scrolling
        cdp_session.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={
            "model": {"content": [0, 0, 100, 0, 100, 50, 0, 50]}
        })
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(tag_name="IFRAME")
        result = await watchdog._scroll_element_container(node, 100)
        assert result is True

    @pytest.mark.asyncio
    async def test_scroll_iframe_resolve_fails(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        cdp_session.cdp_client.send.DOM.resolveNode = AsyncMock(return_value={})
        # Falls through to regular element scrolling
        cdp_session.cdp_client.send.DOM.getBoxModel = AsyncMock(return_value={
            "model": {"content": [0, 0, 100, 0, 100, 50, 0, 50]}
        })
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(tag_name="IFRAME")
        result = await watchdog._scroll_element_container(node, 100)
        assert result is True


# ---------------------------------------------------------------------------
# _get_session_id_for_element
# ---------------------------------------------------------------------------

class TestGetSessionIdForElement:
    @pytest.mark.asyncio
    async def test_main_session(self):
        watchdog = _make_watchdog()
        node = _make_element_node(frame_id=None)
        result = await watchdog._get_session_id_for_element(node)
        assert result is not None

    @pytest.mark.asyncio
    async def test_iframe_found(self):
        session = _make_mock_browser_session()
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [{"type": "iframe", "targetId": "iframe-frame123"}]
        })
        temp_session = MagicMock()
        temp_session.session_id = "iframe-sess"
        session.get_or_create_cdp_session = AsyncMock(return_value=temp_session)
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(frame_id="frame123")
        result = await watchdog._get_session_id_for_element(node)
        assert result == "iframe-sess"

    @pytest.mark.asyncio
    async def test_iframe_not_found(self):
        session = _make_mock_browser_session()
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [{"type": "page", "targetId": "page-123"}]
        })
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(frame_id="unknown-frame")
        result = await watchdog._get_session_id_for_element(node)
        # Falls back to main session

    @pytest.mark.asyncio
    async def test_iframe_exception(self):
        session = _make_mock_browser_session()
        session.cdp_client.send.Target.getTargets = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(frame_id="some-frame")
        result = await watchdog._get_session_id_for_element(node)


# ---------------------------------------------------------------------------
# _dispatch_key_event
# ---------------------------------------------------------------------------

class TestDispatchKeyEvent:
    @pytest.mark.asyncio
    async def test_dispatch_key_with_modifiers(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        watchdog = _make_watchdog(session=session)

        await watchdog._dispatch_key_event(cdp_session, "keyDown", "a", modifiers=2)
        cdp_session.cdp_client.send.Input.dispatchKeyEvent.assert_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_key_no_modifiers(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        watchdog = _make_watchdog(session=session)

        await watchdog._dispatch_key_event(cdp_session, "keyUp", "Enter")
        cdp_session.cdp_client.send.Input.dispatchKeyEvent.assert_awaited()


# ---------------------------------------------------------------------------
# _clear_text_field
# ---------------------------------------------------------------------------

class TestClearTextField:
    @pytest.mark.asyncio
    async def test_cleared_successfully(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"cleared": True, "method": "value", "finalText": ""}}
        })
        result = await watchdog._clear_text_field("obj-1", cdp_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_cleared_partial_text_remaining(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"cleared": True, "method": "value", "finalText": "some leftover"}}
        })
        result = await watchdog._clear_text_field("obj-1", cdp_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_not_supported(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": {"cleared": False, "method": "none", "error": "Not supported"}}
        })
        result = await watchdog._clear_text_field("obj-1", cdp_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_exception(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=RuntimeError("fail"))
        result = await watchdog._clear_text_field("obj-1", cdp_session)
        assert result is False


# ---------------------------------------------------------------------------
# _focus_element_simple
# ---------------------------------------------------------------------------

class TestFocusElementSimple:
    @pytest.mark.asyncio
    async def test_cdp_focus_success(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        result = await watchdog._focus_element_simple(123, "obj-1", cdp_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_cdp_focus_fails_click_succeeds(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.DOM.focus = AsyncMock(side_effect=RuntimeError("focus fail"))
        coords = {"input_x": 100.0, "input_y": 50.0}
        result = await watchdog._focus_element_simple(123, "obj-1", cdp_session, input_coordinates=coords)
        assert result is True

    @pytest.mark.asyncio
    async def test_both_fail(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.DOM.focus = AsyncMock(side_effect=RuntimeError("fail"))
        result = await watchdog._focus_element_simple(123, "obj-1", cdp_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_click_focus_fails(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.DOM.focus = AsyncMock(side_effect=RuntimeError("fail"))
        cdp_session.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(side_effect=RuntimeError("click fail"))
        coords = {"input_x": 100.0, "input_y": 50.0}
        result = await watchdog._focus_element_simple(123, "obj-1", cdp_session, input_coordinates=coords)
        assert result is False


# ---------------------------------------------------------------------------
# _set_value_directly
# ---------------------------------------------------------------------------

class TestSetValueDirectly:
    @pytest.mark.asyncio
    async def test_success(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": "2024-01-01"}
        })
        node = _make_element_node(tag_name="input", attributes={"type": "date"})
        await watchdog._set_value_directly(node, "2024-01-01", "obj-1", cdp_session)

    @pytest.mark.asyncio
    async def test_no_verify(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={})
        node = _make_element_node(tag_name="input", attributes={"type": "date"})
        await watchdog._set_value_directly(node, "2024-01-01", "obj-1", cdp_session)

    @pytest.mark.asyncio
    async def test_exception(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=RuntimeError("fail"))
        node = _make_element_node(tag_name="input", attributes={"type": "date"})
        with pytest.raises(RuntimeError):
            await watchdog._set_value_directly(node, "val", "obj-1", cdp_session)


# ---------------------------------------------------------------------------
# _trigger_framework_events
# ---------------------------------------------------------------------------

class TestTriggerFrameworkEvents:
    @pytest.mark.asyncio
    async def test_success(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(return_value={
            "result": {"value": True}
        })
        await watchdog._trigger_framework_events("obj-1", cdp_session)

    @pytest.mark.asyncio
    async def test_exception(self):
        watchdog = _make_watchdog()
        cdp_session = _make_cdp_session_mock()
        cdp_session.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=RuntimeError("fail"))
        # Should not raise
        await watchdog._trigger_framework_events("obj-1", cdp_session)


# ---------------------------------------------------------------------------
# _click_element_node_impl edge cases
# ---------------------------------------------------------------------------

class TestClickElementNodeImpl:
    @pytest.mark.asyncio
    async def test_select_element_returns_validation_error(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(tag_name="select")
        result = await watchdog._click_element_node_impl(node)
        assert "validation_error" in result

    @pytest.mark.asyncio
    async def test_file_input_returns_validation_error(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        watchdog = _make_watchdog(session=session)
        node = _make_element_node(tag_name="input", attributes={"type": "file"})
        result = await watchdog._click_element_node_impl(node)
        assert "validation_error" in result

    @pytest.mark.asyncio
    async def test_no_quads_js_fallback(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        session.get_element_coordinates = AsyncMock(return_value=None)
        watchdog = _make_watchdog(session=session)
        node = _make_element_node()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await watchdog._click_element_node_impl(node)

    @pytest.mark.asyncio
    async def test_click_with_coordinates(self):
        session = _make_mock_browser_session()
        cdp_session = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp_session)
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

        rect = MagicMock()
        rect.x = 100
        rect.y = 50
        rect.width = 80
        rect.height = 30
        session.get_element_coordinates = AsyncMock(return_value=rect)

        watchdog = _make_watchdog(session=session)
        with patch.object(watchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("asyncio.wait_for", new_callable=AsyncMock, return_value=None):
                    node = _make_element_node()
                    result = await watchdog._click_element_node_impl(node)
