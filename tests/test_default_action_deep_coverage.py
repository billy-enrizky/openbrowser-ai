"""Deep coverage tests for openbrowser.browser.watchdogs.default_action_watchdog.

Targets uncovered lines: 103-104, 113-117, 204-205, 229, 249, 267, 274-275,
325-326, 465-466, 515-520, 528, 538, 556-557, 571-593, 626-627, 645-646,
653-680, 691, 695-713, 859, 1316-1510, 1811-1812, 1839-1840, 1857-1858,
1872-1873, 2066-2067, 2151-2153, 2192, 2440-2442, 2694-2697.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.session import BrowserSession
from openbrowser.browser.views import BrowserError, URLNotAllowedError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_element_node(**overrides):
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
    session.logger = logging.getLogger("test_daw_deep")
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
    session.get_current_page_title = AsyncMock(return_value="Example Page")
    session.id = "test-session"
    session.is_local = True
    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = "/tmp/downloads"
    return session


def _make_cdp_session_mock():
    cdp = MagicMock()
    cdp.cdp_client = MagicMock()
    cdp.cdp_client.send = MagicMock()
    cdp.cdp_client.send.DOM = MagicMock()
    cdp.cdp_client.send.DOM.scrollIntoViewIfNeeded = AsyncMock()
    cdp.cdp_client.send.DOM.resolveNode = AsyncMock(
        return_value={"object": {"objectId": "obj-1"}}
    )
    cdp.cdp_client.send.DOM.focus = AsyncMock()
    cdp.cdp_client.send.DOM.enable = AsyncMock()
    cdp.cdp_client.send.DOM.getDocument = AsyncMock()
    cdp.cdp_client.send.DOM.performSearch = AsyncMock()
    cdp.cdp_client.send.DOM.getSearchResults = AsyncMock()
    cdp.cdp_client.send.DOM.discardSearchResults = AsyncMock()
    cdp.cdp_client.send.DOM.setFileInputFiles = AsyncMock()
    cdp.cdp_client.send.Input = MagicMock()
    cdp.cdp_client.send.Input.dispatchMouseEvent = AsyncMock()
    cdp.cdp_client.send.Input.dispatchKeyEvent = AsyncMock()
    cdp.cdp_client.send.Runtime = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": True}}
    )
    cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
        return_value={
            "result": {"value": {"cleared": True, "method": "value", "finalText": ""}}
        }
    )
    cdp.cdp_client.send.Runtime.runIfWaitingForDebugger = AsyncMock()
    cdp.cdp_client.send.Page = MagicMock()
    cdp.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
        return_value={"layoutViewport": {"clientWidth": 1280, "clientHeight": 720}}
    )
    cdp.cdp_client.send.Page.getNavigationHistory = AsyncMock(
        return_value={
            "currentIndex": 1,
            "entries": [
                {"id": 1, "url": "https://first.com"},
                {"id": 2, "url": "https://current.com"},
                {"id": 3, "url": "https://next.com"},
            ],
        }
    )
    cdp.cdp_client.send.Page.navigateToHistoryEntry = AsyncMock()
    cdp.cdp_client.send.Page.reload = AsyncMock()
    cdp.cdp_client.send.Page.printToPDF = AsyncMock()
    cdp.session_id = "cdp-sess-1"
    return cdp


def _make_watchdog():
    from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

    session = _make_mock_browser_session()
    bus = EventBus()  # Real EventBus for Pydantic validation
    wd = DefaultActionWatchdog(event_bus=bus, browser_session=session)
    # Swap to MagicMock after construction to avoid unawaited coroutines
    wd.event_bus = MagicMock()
    return wd, session


# ---------------------------------------------------------------------------
# _handle_print_button_click  (lines 103-104, 113-117)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestHandlePrintButtonClick:
    async def test_filename_from_page_title_exception(self):
        """Lines 103-104: get_current_page_title raises -> filename = 'print.pdf'."""
        wd, session = _make_watchdog()
        session.get_current_page_title = AsyncMock(side_effect=Exception("timeout"))
        cdp = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
        import base64
        cdp.cdp_client.send.Page.printToPDF = AsyncMock(
            return_value={"data": base64.b64encode(b"fake-pdf").decode()}
        )
        node = _make_element_node(attributes={"onclick": "window.print()"})

        with patch("anyio.open_file") as mock_open, \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=100)), \
             patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.mkdir"):
            mock_file = AsyncMock()
            mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
            mock_open.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await wd._handle_print_button_click(node)

        assert result is not None
        assert result.get("pdf_generated") is True

    async def test_unique_filename_when_file_exists(self):
        """Lines 113-117: file exists -> counter incremented."""
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
        import base64
        cdp.cdp_client.send.Page.printToPDF = AsyncMock(
            return_value={"data": base64.b64encode(b"pdf-data").decode()}
        )
        node = _make_element_node()

        call_count = [0]
        def fake_exists(self_path=None):
            call_count[0] += 1
            return call_count[0] <= 1

        with patch("anyio.open_file") as mock_open, \
             patch("pathlib.Path.exists", side_effect=fake_exists), \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=200)), \
             patch("pathlib.Path.mkdir"):
            mock_file = AsyncMock()
            mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
            mock_open.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await wd._handle_print_button_click(node)

        assert result is not None


# ---------------------------------------------------------------------------
# on_ClickElementEvent  (lines 204-205)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestClickElementDownloadPath:
    async def test_click_returns_none_metadata(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        node = _make_element_node()
        event = MagicMock()
        event.node = node

        with patch.object(DefaultActionWatchdog, "_click_element_node_impl", new_callable=AsyncMock, return_value="not-a-dict"), \
             patch.object(DefaultActionWatchdog, "_is_print_related_element", return_value=False):
            result = await wd.on_ClickElementEvent(event)
        assert result is None


# ---------------------------------------------------------------------------
# on_TypeTextEvent  (lines 229, 249, 267, 274-275)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTypeTextEventDeep:
    async def test_type_to_page_with_sensitive_key_name(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        node = _make_element_node(backend_node_id=0)
        event = MagicMock()
        event.node = node
        event.text = "secret"
        event.is_sensitive = True
        event.sensitive_key_name = "PASSWORD"
        event.clear = False

        with patch.object(DefaultActionWatchdog, "_type_to_page", new_callable=AsyncMock):
            result = await wd.on_TypeTextEvent(event)
        assert result is None

    async def test_type_sensitive_without_key_name_to_element(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        node = _make_element_node(backend_node_id=42)
        event = MagicMock()
        event.node = node
        event.text = "secret"
        event.is_sensitive = True
        event.sensitive_key_name = None
        event.clear = True

        with patch.object(DefaultActionWatchdog, "_input_text_element_node_impl", new_callable=AsyncMock, return_value={"input_x": 100, "input_y": 200}):
            result = await wd.on_TypeTextEvent(event)
        assert result == {"input_x": 100, "input_y": 200}

    async def test_type_fallback_sensitive_without_key(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        node = _make_element_node(backend_node_id=42)
        event = MagicMock()
        event.node = node
        event.text = "pass"
        event.is_sensitive = True
        event.sensitive_key_name = None
        event.clear = False

        with patch.object(DefaultActionWatchdog, "_input_text_element_node_impl", new_callable=AsyncMock, side_effect=Exception("gone")), \
             patch.object(DefaultActionWatchdog, "_click_element_node_impl", new_callable=AsyncMock), \
             patch.object(DefaultActionWatchdog, "_type_to_page", new_callable=AsyncMock), \
             patch("asyncio.wait_for", new_callable=AsyncMock):
            result = await wd.on_TypeTextEvent(event)
        assert result is None

    async def test_type_outer_exception_reraise(self):
        wd, session = _make_watchdog()
        node = MagicMock()
        type(node).backend_node_id = property(lambda s: (_ for _ in ()).throw(RuntimeError("bad")))
        event = MagicMock()
        event.node = node
        with pytest.raises(RuntimeError, match="bad"):
            await wd.on_TypeTextEvent(event)


# ---------------------------------------------------------------------------
# on_ScrollEvent  (lines 325-326)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestScrollEventDeep:
    async def test_scroll_exception_reraise(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        event = MagicMock()
        event.amount = 300
        event.direction = "down"
        event.node = None

        with patch.object(DefaultActionWatchdog, "_scroll_with_cdp_gesture", new_callable=AsyncMock, side_effect=RuntimeError("scroll failed")):
            with pytest.raises(RuntimeError, match="scroll failed"):
                await wd.on_ScrollEvent(event)


# ---------------------------------------------------------------------------
# _click_element_node_impl  (lines 465-466, 515-520, 528, 538, 556-557,
#   571-593, 626-627, 645-646, 653-680, 691, 695-713)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestClickElementNodeImplDeep:
    async def test_scroll_into_view_fails(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        cdp.cdp_client.send.DOM.scrollIntoViewIfNeeded = AsyncMock(side_effect=Exception("detached"))
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        coord = MagicMock(x=100, y=100, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._click_element_node_impl(_make_element_node())
        assert result is not None

    async def test_js_fallback_no_node_found(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        session.get_element_coordinates = AsyncMock(return_value=None)
        cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=Exception("No node with given id found"))
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="Element with given id not found"):
                await wd._click_element_node_impl(_make_element_node())

    async def test_js_fallback_generic_error(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        session.get_element_coordinates = AsyncMock(return_value=None)
        cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=Exception("other"))
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="Failed to click element"):
                await wd._click_element_node_impl(_make_element_node())

    async def test_quad_outside_viewport_uses_first(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        coord = MagicMock(x=-500, y=-500, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._click_element_node_impl(_make_element_node())
        assert result is not None

    async def test_occluded_element_js_fallback(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        coord = MagicMock(x=100, y=100, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._click_element_node_impl(_make_element_node())
        assert result is None

    async def test_occluded_js_fallback_fails(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        coord = MagicMock(x=100, y=100, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)
        cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=Exception("js fail"))
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="Failed to click occluded element"):
                await wd._click_element_node_impl(_make_element_node())

    async def test_mouse_down_timeout(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        coord = MagicMock(x=100, y=100, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        original_wait_for = asyncio.wait_for

        async def patched_wait_for(coro, *, timeout=None):
            # We need to consume the coroutine to avoid warnings
            try:
                return await coro
            except:
                raise

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            # Simulate timeout on mousePressed by making dispatchMouseEvent raise on second call
            call_idx = [0]
            async def dispatch_side_effect(*args, **kwargs):
                call_idx[0] += 1
                if call_idx[0] == 2:  # mousePressed
                    raise TimeoutError("blocked")
            cdp.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(side_effect=dispatch_side_effect)
            result = await wd._click_element_node_impl(_make_element_node())
        assert result is not None

    async def test_cdp_click_exception_js_fallback(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        coord = MagicMock(x=100, y=100, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        cdp.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(side_effect=Exception("CDP fail"))
        # Reset callFunctionOn to succeed for JS fallback
        cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock()

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._click_element_node_impl(_make_element_node())
        assert result is None

    async def test_cdp_and_js_both_fail(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        coord = MagicMock(x=100, y=100, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

        cdp.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(side_effect=Exception("CDP error"))
        cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(side_effect=Exception("JS error"))

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="Failed to click element"):
                await wd._click_element_node_impl(_make_element_node())

    async def test_refocus_timeout(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        coord = MagicMock(x=100, y=100, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)

        async def fake_get_or_create(**kwargs):
            if kwargs.get("focus"):
                raise TimeoutError("refocus timeout")
            return cdp
        session.get_or_create_cdp_session = AsyncMock(side_effect=fake_get_or_create)

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._click_element_node_impl(_make_element_node())
        assert result is not None

    async def test_url_not_allowed_error_reraise(self):
        wd, session = _make_watchdog()
        session.cdp_client_for_node = AsyncMock(side_effect=URLNotAllowedError("blocked"))
        with pytest.raises(URLNotAllowedError):
            await wd._click_element_node_impl(_make_element_node())

    async def test_generic_exception_wraps_browser_error(self):
        wd, session = _make_watchdog()
        session.cdp_client_for_node = AsyncMock(side_effect=RuntimeError("unexpected"))
        with pytest.raises(BrowserError):
            await wd._click_element_node_impl(_make_element_node())


# ---------------------------------------------------------------------------
# _get_char_modifiers_and_vk  (line 859)
# ---------------------------------------------------------------------------

class TestGetCharModifiersAndVk:
    def test_fallback_for_unknown_char(self):
        wd, _ = _make_watchdog()
        modifiers, vk, base = wd._get_char_modifiers_and_vk("\t")
        assert modifiers == 0
        assert base == "\t"


# ---------------------------------------------------------------------------
# _input_text_element_node_impl  (lines 1316-1510)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestInputTextElementNodeImplDeep:
    async def test_basic_typing_flow(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        session.cdp_client = cdp.cdp_client
        coord = MagicMock(x=100, y=100, width=50, height=50)
        session.get_element_coordinates = AsyncMock(return_value=coord)

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False), \
             patch.object(DefaultActionWatchdog, "_focus_element_simple", new_callable=AsyncMock, return_value=True), \
             patch.object(DefaultActionWatchdog, "_requires_direct_value_assignment", return_value=False), \
             patch.object(DefaultActionWatchdog, "_trigger_framework_events", new_callable=AsyncMock), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._input_text_element_node_impl(_make_element_node(), "hi\n", clear=True)
        assert result is not None

    async def test_direct_value_assignment(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        session.cdp_client = cdp.cdp_client
        coord = MagicMock(x=50, y=50, width=30, height=30)
        session.get_element_coordinates = AsyncMock(return_value=coord)

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=False), \
             patch.object(DefaultActionWatchdog, "_focus_element_simple", new_callable=AsyncMock, return_value=True), \
             patch.object(DefaultActionWatchdog, "_requires_direct_value_assignment", return_value=True), \
             patch.object(DefaultActionWatchdog, "_set_value_directly", new_callable=AsyncMock), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._input_text_element_node_impl(_make_element_node(attributes={"type": "date"}), "2024-01-01")
        assert result is not None

    async def test_occluded_input_forces_fallback(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        session.cdp_client = cdp.cdp_client
        coord = MagicMock(x=50, y=50, width=30, height=30)
        session.get_element_coordinates = AsyncMock(return_value=coord)

        with patch.object(DefaultActionWatchdog, "_check_element_occlusion", new_callable=AsyncMock, return_value=True), \
             patch.object(DefaultActionWatchdog, "_focus_element_simple", new_callable=AsyncMock, return_value=True), \
             patch.object(DefaultActionWatchdog, "_requires_direct_value_assignment", return_value=False), \
             patch.object(DefaultActionWatchdog, "_trigger_framework_events", new_callable=AsyncMock), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._input_text_element_node_impl(_make_element_node(), "a", clear=False)
        assert result is None

    async def test_input_text_raises_browser_error(self):
        wd, session = _make_watchdog()
        session.cdp_client_for_node = AsyncMock(side_effect=RuntimeError("gone"))
        session.cdp_client = MagicMock()
        with pytest.raises(BrowserError, match="Failed to input text"):
            await wd._input_text_element_node_impl(_make_element_node(), "text")

    async def test_scroll_node_detached(self):
        from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        cdp.cdp_client.send.DOM.scrollIntoViewIfNeeded = AsyncMock(side_effect=Exception("Node is detached from document"))
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        session.cdp_client = cdp.cdp_client
        session.get_element_coordinates = AsyncMock(return_value=None)

        with patch.object(DefaultActionWatchdog, "_focus_element_simple", new_callable=AsyncMock, return_value=True), \
             patch.object(DefaultActionWatchdog, "_requires_direct_value_assignment", return_value=False), \
             patch.object(DefaultActionWatchdog, "_trigger_framework_events", new_callable=AsyncMock), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd._input_text_element_node_impl(_make_element_node(), "x", clear=False)
        assert result is None


# ---------------------------------------------------------------------------
# on_GoBackEvent / on_GoForwardEvent / on_RefreshEvent / on_WaitEvent /
# on_SendKeysEvent  (exception re-raise lines)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNavigationEventsDeep:
    async def test_go_back_exception(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        cdp.cdp_client.send.Page.getNavigationHistory = AsyncMock(side_effect=RuntimeError("nav"))
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
        with pytest.raises(RuntimeError, match="nav"):
            await wd.on_GoBackEvent(MagicMock())

    async def test_go_forward_exception(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        cdp.cdp_client.send.Page.getNavigationHistory = AsyncMock(side_effect=RuntimeError("fwd"))
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
        with pytest.raises(RuntimeError, match="fwd"):
            await wd.on_GoForwardEvent(MagicMock())

    async def test_refresh_exception(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        cdp.cdp_client.send.Page.reload = AsyncMock(side_effect=RuntimeError("reload"))
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
        with pytest.raises(RuntimeError, match="reload"):
            await wd.on_RefreshEvent(MagicMock())

    async def test_wait_exception(self):
        wd, _ = _make_watchdog()
        event = MagicMock()
        event.seconds = 0.01
        event.max_seconds = 10
        with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=RuntimeError("interrupted")):
            with pytest.raises(RuntimeError, match="interrupted"):
                await wd.on_WaitEvent(event)

    async def test_send_keys_exception(self):
        wd, session = _make_watchdog()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("dead"))
        event = MagicMock()
        event.keys = "Enter"
        with pytest.raises(RuntimeError, match="dead"):
            await wd.on_SendKeysEvent(event)


# ---------------------------------------------------------------------------
# on_ScrollToTextEvent  (lines 2151-2153, 2192)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestScrollToTextEventDeep:
    async def test_search_query_exception(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
        # The method uses session.cdp_client directly, not get_or_create_cdp_session
        session.cdp_client = cdp.cdp_client
        cdp.cdp_client.send.DOM.performSearch = AsyncMock(side_effect=Exception("search failed"))
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": False}})
        cdp.cdp_client.send.DOM.enable = AsyncMock()
        cdp.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": {"nodeId": 1}})

        event = MagicMock()
        event.text = "needle"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(BrowserError, match="Text not found"):
                await wd.on_ScrollToTextEvent(event)

    async def test_scroll_to_text_found(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
        # The method uses session.cdp_client directly
        session.cdp_client = cdp.cdp_client
        cdp.cdp_client.send.DOM.enable = AsyncMock()
        cdp.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": {"nodeId": 1}})
        cdp.cdp_client.send.DOM.performSearch = AsyncMock(return_value={"searchId": "s1", "resultCount": 1})
        cdp.cdp_client.send.DOM.getSearchResults = AsyncMock(return_value={"nodeIds": [42]})
        cdp.cdp_client.send.DOM.scrollIntoViewIfNeeded = AsyncMock()
        cdp.cdp_client.send.DOM.discardSearchResults = AsyncMock()

        event = MagicMock()
        event.text = "needle"
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await wd.on_ScrollToTextEvent(event)
        assert result is None


# ---------------------------------------------------------------------------
# on_SelectDropdownOptionEvent  (lines 2440-2442, 2694-2697)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSelectDropdownOptionDeep:
    async def test_resolve_node_fails(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        cdp.cdp_client.send.DOM.resolveNode = AsyncMock(return_value={"object": {}})

        event = MagicMock()
        event.node = _make_element_node()
        event.text = "Option A"
        with pytest.raises(ValueError, match="Could not get object ID"):
            await wd.on_SelectDropdownOptionEvent(event)

    async def test_selection_failed_with_string_options(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={"result": {"value": {
                "success": False,
                "error": "Option not found",
                "availableOptions": ["Apple", "Banana"],
            }}}
        )

        event = MagicMock()
        event.node = _make_element_node(tag_name="select")
        event.text = "Durian"
        result = await wd.on_SelectDropdownOptionEvent(event)
        assert result is not None
        assert result.get("success") == "false"

    async def test_selection_failed_with_dict_options(self):
        wd, session = _make_watchdog()
        cdp = _make_cdp_session_mock()
        session.cdp_client_for_node = AsyncMock(return_value=cdp)
        cdp.cdp_client.send.Runtime.callFunctionOn = AsyncMock(
            return_value={"result": {"value": {
                "success": False,
                "error": "Not found",
                "availableOptions": [
                    {"text": "", "value": "val1"},
                    {"text": "Label2", "value": "val2"},
                ],
            }}}
        )

        event = MagicMock()
        event.node = _make_element_node(tag_name="select")
        event.text = "Missing"
        result = await wd.on_SelectDropdownOptionEvent(event)
        assert "short_term_memory" in result
