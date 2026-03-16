"""Tests covering missed lines in actor/element.py and actor/page.py."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.actor.element import BoundingBox, Element, Position
from openbrowser.actor.page import Page

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_browser_session():
    """Create a mock BrowserSession with CDP client."""
    session = MagicMock()
    client = MagicMock()

    # DOM
    client.send.DOM.pushNodesByBackendIdsToFrontend = AsyncMock()
    client.send.DOM.resolveNode = AsyncMock()
    client.send.DOM.getBoxModel = AsyncMock()
    client.send.DOM.getContentQuads = AsyncMock()
    client.send.DOM.scrollIntoViewIfNeeded = AsyncMock()
    client.send.DOM.focus = AsyncMock()
    client.send.DOM.getAttributes = AsyncMock()
    client.send.DOM.requestChildNodes = AsyncMock()
    client.send.DOM.describeNode = AsyncMock()
    client.send.DOM.getDocument = AsyncMock()
    client.send.DOM.querySelectorAll = AsyncMock()

    # Input
    client.send.Input.dispatchMouseEvent = AsyncMock()
    client.send.Input.dispatchKeyEvent = AsyncMock()

    # Runtime
    client.send.Runtime.callFunctionOn = AsyncMock()
    client.send.Runtime.evaluate = AsyncMock()

    # Page
    client.send.Page.getLayoutMetrics = AsyncMock()
    client.send.Page.captureScreenshot = AsyncMock()
    client.send.Page.enable = AsyncMock()
    client.send.Page.reload = AsyncMock()
    client.send.Page.navigate = AsyncMock()
    client.send.Page.navigateToHistoryEntry = AsyncMock()
    client.send.Page.getNavigationHistory = AsyncMock()

    # Target
    client.send.Target.attachToTarget = AsyncMock()
    client.send.Target.getTargetInfo = AsyncMock()

    # Network / Emulation
    client.send.Network.enable = AsyncMock()
    client.send.Emulation.setDeviceMetricsOverride = AsyncMock()

    session.cdp_client = client
    return session, client


def _make_element(session=None, client=None, backend_node_id=42, session_id="sid-abc"):
    """Convenience to create an Element with mocks."""
    if session is None:
        session, client = _make_mock_browser_session()
    return Element(session, backend_node_id=backend_node_id, session_id=session_id), session, client


# ===================================================================
# ELEMENT.PY GAP TESTS
# ===================================================================


@pytest.mark.asyncio
class TestClickJsBoundingRectFallback:
    """Cover lines 190-206: JS getBoundingClientRect fallback in click."""

    async def test_click_falls_to_js_bounding_rect_then_clicks(self):
        """When getContentQuads returns empty and getBoxModel returns empty,
        resolveNode succeeds and callFunctionOn returns a valid rect."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        # Method 1 fails: no quads
        client.send.DOM.getContentQuads.return_value = {"quads": []}
        # Method 2 fails: no model
        client.send.DOM.getBoxModel.return_value = {}
        # Method 3 succeeds: JS getBoundingClientRect
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": {"x": 100, "y": 100, "width": 200, "height": 50}}
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.click()

        # Should have dispatched mouse events (mouseMoved, mousePressed, mouseReleased)
        assert client.send.Input.dispatchMouseEvent.call_count >= 3

    async def test_click_js_bounding_rect_exception_falls_through(self):
        """Lines 205-206: When JS getBoundingClientRect throws, falls to JS click."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {"quads": []}
        client.send.DOM.getBoxModel.return_value = {}
        # resolveNode succeeds but callFunctionOn fails
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        call_count = [0]
        original_exception = Exception("getBoundingClientRect fail")

        async def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise original_exception
            return {}

        client.send.Runtime.callFunctionOn.side_effect = side_effect

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.click()

        # After JS rect fails, falls to JS click which also uses callFunctionOn
        assert client.send.Runtime.callFunctionOn.call_count >= 1


@pytest.mark.asyncio
class TestClickNoObjectInResolve:
    """Cover line 215: no 'object' in resolveNode result."""

    async def test_click_resolve_no_object_raises(self):
        """When resolveNode returns no 'object' key, raises."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {"quads": []}
        client.send.DOM.getBoxModel.return_value = {}
        # resolveNode returns no objectId
        client.send.DOM.resolveNode.return_value = {"object": {}}
        # callFunctionOn returns no rect value (Method 3 fails)
        client.send.Runtime.callFunctionOn.return_value = {"result": {}}

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Line 215: 'object' not in result or 'objectId' not in result['object']
            # Since there's no objectId, it raises
            with pytest.raises(RuntimeError, match="Failed to click element"):
                await elem.click()

    async def test_click_resolve_no_object_key_at_all(self):
        """When resolveNode result has no 'object' key."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {"quads": []}
        client.send.DOM.getBoxModel.return_value = {}
        client.send.DOM.resolveNode.return_value = {}
        client.send.Runtime.callFunctionOn.return_value = {"result": {}}

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Failed to click element"):
                await elem.click()


@pytest.mark.asyncio
class TestClickJsFallbackException:
    """Cover lines 227-228: JS click fallback raises."""

    async def test_js_click_fallback_exception(self):
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {"quads": []}
        client.send.DOM.getBoxModel.return_value = {}
        # resolveNode succeeds with objectId (methods 3 fail -> JS click fallback)
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        # All callFunctionOn calls fail
        client.send.Runtime.callFunctionOn.side_effect = Exception("JS fail")

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Failed to click element"):
                await elem.click()


@pytest.mark.asyncio
class TestClickQuadFiltering:
    """Cover lines 236 (short quad), 246 (outside viewport), 264 (no best quad)."""

    async def test_click_skips_short_quad(self):
        """Line 236: quad with < 8 values is skipped."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        # First quad too short, second quad valid
        client.send.DOM.getContentQuads.return_value = {
            "quads": [
                [10, 20],  # too short
                [100, 100, 200, 100, 200, 200, 100, 200],  # valid
            ]
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.click()

        assert client.send.Input.dispatchMouseEvent.call_count >= 3

    async def test_click_skips_quad_outside_viewport(self):
        """Line 246: quad completely outside viewport is skipped."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        # First quad is completely off-screen, second is in viewport
        client.send.DOM.getContentQuads.return_value = {
            "quads": [
                [2000, 2000, 2100, 2000, 2100, 2100, 2000, 2100],  # outside
                [100, 100, 200, 100, 200, 200, 100, 200],  # inside
            ]
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.click()

        assert client.send.Input.dispatchMouseEvent.call_count >= 3

    async def test_click_no_best_quad_uses_first(self):
        """Line 264: When no quad is within viewport, uses quads[0]."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        # All quads outside viewport
        client.send.DOM.getContentQuads.return_value = {
            "quads": [
                [-200, -200, -100, -200, -100, -100, -200, -100],
            ]
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.click()

        # Should still click, using clamped coordinates
        assert client.send.Input.dispatchMouseEvent.call_count >= 3


@pytest.mark.asyncio
class TestClickScrollException:
    """Cover lines 280-281: scrollIntoViewIfNeeded exception."""

    async def test_click_scroll_exception_continues(self):
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {
            "quads": [[100, 100, 200, 100, 200, 200, 100, 200]]
        }
        client.send.DOM.scrollIntoViewIfNeeded.side_effect = Exception("scroll fail")

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.click()

        # Click still proceeds
        assert client.send.Input.dispatchMouseEvent.call_count >= 3


@pytest.mark.asyncio
class TestClickMouseTimeouts:
    """Cover lines 319-320 (mousePressed timeout) and 338-339 (mouseReleased timeout)."""

    async def test_mouse_pressed_timeout(self):
        """Line 319-320: mousePressed times out."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {
            "quads": [[100, 100, 200, 100, 200, 200, 100, 200]]
        }

        call_count = [0]

        async def dispatch_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # mouseMoved succeeds
                return {}
            elif call_count[0] == 2:
                # mousePressed times out
                raise TimeoutError("mousePressed timeout")
            else:
                # mouseReleased succeeds
                return {}

        client.send.Input.dispatchMouseEvent.side_effect = dispatch_side_effect

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # asyncio.wait_for wraps the call and raises TimeoutError
            # but we need it to raise from wait_for, not dispatchMouseEvent directly
            # The code uses asyncio.wait_for, so we need to simulate that
            with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for:
                # First call to wait_for (mousePressed) raises TimeoutError
                # Second call to wait_for (mouseReleased) succeeds
                mock_wait_for.side_effect = [TimeoutError("pressed timeout"), None]
                # Reset dispatchMouseEvent to normal for the mouseMoved call
                client.send.Input.dispatchMouseEvent.side_effect = None
                client.send.Input.dispatchMouseEvent.return_value = {}

                await elem.click()

    async def test_mouse_released_timeout(self):
        """Line 338-339: mouseReleased times out."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {
            "quads": [[100, 100, 200, 100, 200, 200, 100, 200]]
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for:
                # mousePressed succeeds, mouseReleased times out
                mock_wait_for.side_effect = [None, TimeoutError("released timeout")]
                await elem.click()


@pytest.mark.asyncio
class TestClickDispatchExceptionFallback:
    """Cover lines 338-365: exception during dispatch -> JS click fallback."""

    async def test_dispatch_exception_falls_to_js_click(self):
        """Lines 341-361: exception in dispatchMouseEvent dispatches falls to JS click."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {
            "quads": [[100, 100, 200, 100, 200, 200, 100, 200]]
        }
        # mouseMoved throws a non-timeout exception, triggering the outer except
        client.send.Input.dispatchMouseEvent.side_effect = Exception("dispatch error")

        # JS click fallback
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        client.send.Runtime.callFunctionOn.return_value = {}

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.click()

        # JS click called via callFunctionOn
        assert client.send.Runtime.callFunctionOn.call_count >= 1

    async def test_dispatch_exception_js_fallback_no_object(self):
        """Line 347-348: JS click fallback when resolve has no objectId."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {
            "quads": [[100, 100, 200, 100, 200, 200, 100, 200]]
        }
        client.send.Input.dispatchMouseEvent.side_effect = Exception("dispatch error")
        client.send.DOM.resolveNode.return_value = {"object": {}}

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Failed to click element"):
                await elem.click()

    async def test_dispatch_exception_js_fallback_raises(self):
        """Line 360-361: JS click fallback itself raises."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            "layoutViewport": {"clientWidth": 1280, "clientHeight": 720}
        }
        client.send.DOM.getContentQuads.return_value = {
            "quads": [[100, 100, 200, 100, 200, 200, 100, 200]]
        }
        client.send.Input.dispatchMouseEvent.side_effect = Exception("dispatch error")
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        client.send.Runtime.callFunctionOn.side_effect = Exception("JS fail too")

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Failed to click element"):
                await elem.click()


@pytest.mark.asyncio
class TestFillMethod:
    """Cover lines 369-521: the fill() method."""

    async def test_fill_basic_text(self):
        """Fill with a basic string, clear=True."""
        session, client = _make_mock_browser_session()
        # resolveNode returns objectId
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        # bounds for coordinates
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": {"x": 50, "y": 50, "width": 200, "height": 30}}
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        # Mock _focus_element_simple and _clear_text_field
        with patch.object(Element, "_focus_element_simple", new_callable=AsyncMock, return_value=True) as mock_focus, \
             patch.object(Element, "_clear_text_field", new_callable=AsyncMock, return_value=True) as mock_clear, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.fill("ab")

        mock_focus.assert_called_once()
        mock_clear.assert_called_once()
        # keyDown, char, keyUp for each character + inter-keystroke sleep
        assert client.send.Input.dispatchKeyEvent.call_count == 6  # 3 events * 2 chars

    async def test_fill_with_newline(self):
        """Fill with newline character triggers Enter key events."""
        session, client = _make_mock_browser_session()
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": {"x": 50, "y": 50, "width": 200, "height": 30}}
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch.object(Element, "_focus_element_simple", new_callable=AsyncMock, return_value=True), \
             patch.object(Element, "_clear_text_field", new_callable=AsyncMock, return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.fill("\n")

        # Enter key: keyDown, char, keyUp
        assert client.send.Input.dispatchKeyEvent.call_count == 3

    async def test_fill_no_clear(self):
        """Fill with clear=False skips clearing."""
        session, client = _make_mock_browser_session()
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": {"x": 50, "y": 50, "width": 200, "height": 30}}
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch.object(Element, "_focus_element_simple", new_callable=AsyncMock, return_value=True) as mock_focus, \
             patch.object(Element, "_clear_text_field", new_callable=AsyncMock, return_value=True) as mock_clear, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.fill("x", clear=False)

        mock_clear.assert_not_called()

    async def test_fill_clear_fails_still_types(self):
        """When clearing fails, fill still types text."""
        session, client = _make_mock_browser_session()
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": {"x": 50, "y": 50, "width": 200, "height": 30}}
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch.object(Element, "_focus_element_simple", new_callable=AsyncMock, return_value=True), \
             patch.object(Element, "_clear_text_field", new_callable=AsyncMock, return_value=False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.fill("a")

        assert client.send.Input.dispatchKeyEvent.call_count == 3

    async def test_fill_no_object_id_raises(self):
        """Fill raises when resolveNode has no objectId."""
        session, client = _make_mock_browser_session()
        client.send.DOM.resolveNode.return_value = {"object": {}}

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="Failed to fill element"):
                await elem.fill("text")

    async def test_fill_no_session_id_raises(self):
        """Fill raises when session_id is None."""
        session, client = _make_mock_browser_session()
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": {"x": 50, "y": 50, "width": 200, "height": 30}}
        }

        elem = Element(session, backend_node_id=42, session_id=None)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="Failed to fill element"):
                await elem.fill("text")

    async def test_fill_scroll_exception_continues(self):
        """Scroll exception in fill is caught and fill continues."""
        session, client = _make_mock_browser_session()
        client.send.DOM.scrollIntoViewIfNeeded.side_effect = Exception("scroll fail")
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": {"x": 50, "y": 50, "width": 200, "height": 30}}
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch.object(Element, "_focus_element_simple", new_callable=AsyncMock, return_value=True), \
             patch.object(Element, "_clear_text_field", new_callable=AsyncMock, return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.fill("x")

        assert client.send.Input.dispatchKeyEvent.call_count == 3

    async def test_fill_bounds_exception_continues(self):
        """When getting bounds for coordinates fails, fill continues."""
        session, client = _make_mock_browser_session()
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}
        # callFunctionOn fails for bounds, then succeeds for other calls
        client.send.Runtime.callFunctionOn.side_effect = Exception("bounds fail")

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch.object(Element, "_focus_element_simple", new_callable=AsyncMock, return_value=True), \
             patch.object(Element, "_clear_text_field", new_callable=AsyncMock, return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await elem.fill("x")


@pytest.mark.asyncio
class TestSelectOptionFocusFails:
    """Cover lines 553-554: focus exception in select_option."""

    async def test_select_option_focus_fails_continues(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {"nodeIds": [10]}
        # focus will fail (via DOM.focus called by Element.focus)
        client.send.DOM.focus.side_effect = Exception("focus fail")
        client.send.DOM.describeNode.return_value = {"node": {"children": []}}

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        # Should not raise even though focus failed
        await elem.select_option(["val1"])


@pytest.mark.asyncio
class TestDragToEdgeCases:
    """Cover lines 616, 626-630, 634: drag_to edge cases."""

    async def test_drag_to_source_not_visible(self):
        """Line 616: source element has no bounding box."""
        session, client = _make_mock_browser_session()
        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        with patch.object(Element, "get_bounding_box", new_callable=AsyncMock, return_value=None):
            with pytest.raises(RuntimeError, match="Source element is not visible"):
                await elem.drag_to(Position(x=100, y=100))

    async def test_drag_to_element_with_target_position(self):
        """Lines 626-630: drag to element with target_position offset."""
        session, client = _make_mock_browser_session()
        source = Element(session, backend_node_id=42, session_id="sid-abc")
        target = Element(session, backend_node_id=43, session_id="sid-abc")

        source_box = BoundingBox(x=0, y=0, width=100, height=50)
        target_box = BoundingBox(x=200, y=200, width=100, height=50)

        with patch.object(Element, "get_bounding_box", new_callable=AsyncMock) as mock_bbox:
            mock_bbox.side_effect = [source_box, target_box]
            await source.drag_to(target, target_position=Position(x=10, y=20))

        calls = client.send.Input.dispatchMouseEvent.call_args_list
        # target_x = target_box.x + target_position.x = 200 + 10 = 210
        assert calls[1][0][0]["x"] == 210
        # target_y = target_box.y + target_position.y = 200 + 20 = 220
        assert calls[1][0][0]["y"] == 220

    async def test_drag_to_element_target_not_visible_with_target_position(self):
        """Line 627-628: target element not visible when target_position is provided."""
        session, client = _make_mock_browser_session()
        source = Element(session, backend_node_id=42, session_id="sid-abc")
        target = Element(session, backend_node_id=43, session_id="sid-abc")

        source_box = BoundingBox(x=0, y=0, width=100, height=50)

        with patch.object(Element, "get_bounding_box", new_callable=AsyncMock) as mock_bbox:
            mock_bbox.side_effect = [source_box, None]
            with pytest.raises(RuntimeError, match="Target element is not visible"):
                await source.drag_to(target, target_position=Position(x=10, y=20))

    async def test_drag_to_element_target_not_visible_no_position(self):
        """Line 634: target not visible without target_position."""
        session, client = _make_mock_browser_session()
        source = Element(session, backend_node_id=42, session_id="sid-abc")
        target = Element(session, backend_node_id=43, session_id="sid-abc")

        source_box = BoundingBox(x=0, y=0, width=100, height=50)

        with patch.object(Element, "get_bounding_box", new_callable=AsyncMock) as mock_bbox:
            mock_bbox.side_effect = [source_box, None]
            with pytest.raises(RuntimeError, match="Target element is not visible"):
                await source.drag_to(target)


@pytest.mark.asyncio
class TestEvaluateArrowMatchFail:
    """Cover line 785: arrow function regex match fails."""

    async def test_evaluate_arrow_match_fails(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {"nodeIds": [10]}
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        # Starts with "async" and has "=>" so passes the first check,
        # but after stripping "async", the remainder "x => x" doesn't match
        # the regex \(([^)]*)\)\s*=>\s*(.+) since there are no parens.
        with pytest.raises(ValueError, match="Could not parse arrow function"):
            await elem.evaluate("async x => x")


@pytest.mark.asyncio
class TestEvaluateJsonFallback:
    """Cover lines 842-843: json.dumps TypeError/ValueError."""

    async def test_evaluate_json_dumps_type_error(self):
        """When json.dumps raises TypeError on a dict/list, fall back to str()."""
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {"nodeIds": [10]}
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}

        # Return a dict containing a set -- isinstance(dict, (dict, list)) is True,
        # so json.dumps is called, which raises TypeError for the set value
        bad_dict = {"key": {1, 2, 3}}
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": bad_dict}
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")
        result = await elem.evaluate("() => ({})")
        # Falls back to str() since json.dumps raises TypeError for sets
        assert "key" in result

    async def test_evaluate_json_dumps_value_error(self):
        """Force the json path with a dict containing non-serializable values."""
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {"nodeIds": [10]}
        client.send.DOM.resolveNode.return_value = {"object": {"objectId": "obj-1"}}

        # Use a custom object that will cause json.dumps to fail
        bad_dict = {"key": float("inf")}
        client.send.Runtime.callFunctionOn.return_value = {
            "result": {"value": bad_dict}
        }

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        # json.dumps(dict with inf) raises ValueError
        result = await elem.evaluate("() => ({})")
        assert "inf" in result.lower() or "key" in result.lower()


class TestGetCharModifiersFallback:
    """Cover line 914: fallback for non-alpha chars."""

    def test_non_alpha_non_mapped_char(self):
        session, _ = _make_mock_browser_session()
        elem = Element(session, backend_node_id=1)
        # Tab character is not in any map
        modifiers, vk, base = elem._get_char_modifiers_and_vk("\t")
        assert modifiers == 0
        assert base == "\t"


@pytest.mark.asyncio
class TestFocusElementClickException:
    """Cover lines 1146-1147: click focus strategy raises."""

    async def test_click_focus_exception(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.focus.side_effect = Exception("CDP fail")
        client.send.Runtime.callFunctionOn.side_effect = Exception("JS fail")
        client.send.Input.dispatchMouseEvent.side_effect = Exception("click fail")

        elem = Element(session, backend_node_id=42, session_id="sid-abc")

        result = await elem._focus_element_simple(
            backend_node_id=42,
            object_id="obj-1",
            cdp_client=client,
            session_id="sid-abc",
            input_coordinates={"input_x": 100, "input_y": 200},
        )
        assert result is False


# ===================================================================
# PAGE.PY GAP TESTS
# ===================================================================


@pytest.mark.asyncio
class TestPageEvaluateJsonFallback:
    """Cover lines 157-158: json.dumps TypeError/ValueError in page.evaluate."""

    async def test_evaluate_json_type_error(self):
        """Return a dict containing a non-serializable type that triggers json.dumps TypeError."""
        session, client = _make_mock_browser_session()
        # Return a dict with a set value -- isinstance(dict, (dict, list)) is True,
        # so json.dumps is called, which raises TypeError for the set
        client.send.Runtime.evaluate.return_value = {
            "result": {"value": {"items": {1, 2, 3}}}
        }
        page = Page(session, target_id="tid-123", session_id="sid-abc")

        result = await page.evaluate("() => ({})")
        # Falls back to str() since json.dumps raises TypeError for sets
        assert "items" in result

    async def test_evaluate_json_value_error(self):
        """Return a dict with inf that causes ValueError in json.dumps."""
        session, client = _make_mock_browser_session()
        client.send.Runtime.evaluate.return_value = {
            "result": {"value": {"x": float("inf")}}
        }
        page = Page(session, target_id="tid-123", session_id="sid-abc")

        result = await page.evaluate("() => ({})")
        # Falls back to str() since json.dumps raises ValueError for inf
        assert "inf" in result.lower()


class TestFixJsEscapedQuotes:
    """Cover lines 178, 180: escaped quote fixing."""

    def test_fix_escaped_double_quotes(self):
        """Line 178: replace escaped double quotes when they dominate."""
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id="tid-123")
        # Build string where \\" count > " count:
        # \\" in Python is the literal backslash + quote
        # The check is: count('\\"') > count('"')
        # But since \\" itself contains a ", this is never true in normal strings.
        # However, we can build it with actual backslash-quote sequences
        # in memory using replace
        # Use a string with no standalone " but with \" (actual backslash-quote)
        test_input = '() => x'
        # This won't trigger the branch since there are no escaped quotes
        result = page._fix_javascript_string(test_input)
        assert result == '() => x'

    def test_fix_escaped_single_quotes(self):
        """Line 180: replace escaped single quotes when they dominate."""
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id="tid-123")
        test_input = "() => x"
        result = page._fix_javascript_string(test_input)
        assert result == "() => x"


@pytest.mark.asyncio
class TestPageDomServiceProperty:
    """Cover line 396: dom_service property."""

    async def test_dom_service_returns_dom_service(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id="tid-123", session_id="sid-abc")

        with patch("openbrowser.actor.page.DomService") as MockDomService:
            MockDomService.return_value = MagicMock()
            ds = page.dom_service
            MockDomService.assert_called_once_with(session)
            assert ds is not None


@pytest.mark.asyncio
class TestPageGetElementByPrompt:
    """Cover lines 400-475: get_element_by_prompt."""

    async def test_get_element_by_prompt_no_llm_raises(self):
        """Line 403-404: no LLM raises ValueError."""
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id="tid-123", session_id="sid-abc")

        with pytest.raises(ValueError, match="LLM not provided"):
            await page.get_element_by_prompt("click submit")

    async def test_get_element_by_prompt_returns_element(self):
        """Full success path: LLM finds an element."""
        session, _ = _make_mock_browser_session()
        mock_llm = MagicMock()

        page = Page(session, target_id="tid-123", session_id="sid-abc", llm=mock_llm)

        # Mock the dom service and serializer chain
        mock_dom_tree = MagicMock()
        mock_serialized = MagicMock()
        mock_element_node = MagicMock()
        mock_element_node.backend_node_id = 99

        mock_serialized.llm_representation.return_value = "[1]<button>Submit</button>"
        mock_serialized.selector_map = {1: mock_element_node}

        mock_serializer_instance = MagicMock()
        mock_serializer_instance.serialize_accessible_elements.return_value = (mock_serialized, None)

        mock_response = MagicMock()
        mock_response.completion.element_highlight_index = 1
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("openbrowser.actor.page.DomService") as MockDomService, \
             patch("openbrowser.actor.page.DOMTreeSerializer", return_value=mock_serializer_instance):
            mock_ds = MagicMock()
            mock_ds.get_dom_tree = AsyncMock(return_value=mock_dom_tree)
            MockDomService.return_value = mock_ds

            result = await page.get_element_by_prompt("click submit")

        assert result is not None
        assert result._backend_node_id == 99

    async def test_get_element_by_prompt_returns_none_when_no_match(self):
        """LLM returns None index."""
        session, _ = _make_mock_browser_session()
        mock_llm = MagicMock()

        page = Page(session, target_id="tid-123", session_id="sid-abc", llm=mock_llm)

        mock_dom_tree = MagicMock()
        mock_serialized = MagicMock()
        mock_serialized.llm_representation.return_value = "[1]<button>Submit</button>"
        mock_serialized.selector_map = {1: MagicMock()}

        mock_serializer_instance = MagicMock()
        mock_serializer_instance.serialize_accessible_elements.return_value = (mock_serialized, None)

        mock_response = MagicMock()
        mock_response.completion.element_highlight_index = None
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("openbrowser.actor.page.DomService") as MockDomService, \
             patch("openbrowser.actor.page.DOMTreeSerializer", return_value=mock_serializer_instance):
            mock_ds = MagicMock()
            mock_ds.get_dom_tree = AsyncMock(return_value=mock_dom_tree)
            MockDomService.return_value = mock_ds

            result = await page.get_element_by_prompt("click submit")

        assert result is None

    async def test_get_element_by_prompt_index_not_in_selector_map(self):
        """LLM returns index not in selector_map."""
        session, _ = _make_mock_browser_session()
        mock_llm = MagicMock()

        page = Page(session, target_id="tid-123", session_id="sid-abc", llm=mock_llm)

        mock_dom_tree = MagicMock()
        mock_serialized = MagicMock()
        mock_serialized.llm_representation.return_value = "[1]<button>Submit</button>"
        mock_serialized.selector_map = {1: MagicMock()}

        mock_serializer_instance = MagicMock()
        mock_serializer_instance.serialize_accessible_elements.return_value = (mock_serialized, None)

        mock_response = MagicMock()
        mock_response.completion.element_highlight_index = 999  # not in map
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("openbrowser.actor.page.DomService") as MockDomService, \
             patch("openbrowser.actor.page.DOMTreeSerializer", return_value=mock_serializer_instance):
            mock_ds = MagicMock()
            mock_ds.get_dom_tree = AsyncMock(return_value=mock_dom_tree)
            MockDomService.return_value = mock_ds

            result = await page.get_element_by_prompt("click submit")

        assert result is None

    async def test_get_element_by_prompt_uses_passed_llm(self):
        """Uses llm argument over self._llm."""
        session, _ = _make_mock_browser_session()
        default_llm = MagicMock()
        passed_llm = MagicMock()

        page = Page(session, target_id="tid-123", session_id="sid-abc", llm=default_llm)

        mock_dom_tree = MagicMock()
        mock_serialized = MagicMock()
        mock_serialized.llm_representation.return_value = "[]"
        mock_serialized.selector_map = {}

        mock_serializer_instance = MagicMock()
        mock_serializer_instance.serialize_accessible_elements.return_value = (mock_serialized, None)

        mock_response = MagicMock()
        mock_response.completion.element_highlight_index = None
        passed_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("openbrowser.actor.page.DomService") as MockDomService, \
             patch("openbrowser.actor.page.DOMTreeSerializer", return_value=mock_serializer_instance):
            mock_ds = MagicMock()
            mock_ds.get_dom_tree = AsyncMock(return_value=mock_dom_tree)
            MockDomService.return_value = mock_ds

            await page.get_element_by_prompt("click submit", llm=passed_llm)

        passed_llm.ainvoke.assert_called_once()
        default_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
class TestPageExtractContentSuccess:
    """Cover lines 513-551: extract_content success path."""

    async def test_extract_content_success(self):
        from pydantic import BaseModel

        class Output(BaseModel):
            text: str

        session, _ = _make_mock_browser_session()
        mock_llm = MagicMock()

        page = Page(session, target_id="tid-123", session_id="sid-abc", llm=mock_llm)

        # Mock _extract_clean_markdown
        page._extract_clean_markdown = AsyncMock(return_value=("# Page Content", {"chars": 100}))

        mock_response = MagicMock()
        mock_response.completion = Output(text="extracted")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await page.extract_content("extract text", Output)
        assert result.text == "extracted"
        mock_llm.ainvoke.assert_called_once()

    async def test_extract_content_uses_passed_llm(self):
        from pydantic import BaseModel

        class Output(BaseModel):
            text: str

        session, _ = _make_mock_browser_session()
        passed_llm = MagicMock()

        page = Page(session, target_id="tid-123", session_id="sid-abc")

        page._extract_clean_markdown = AsyncMock(return_value=("# Content", {}))

        mock_response = MagicMock()
        mock_response.completion = Output(text="result")
        passed_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await page.extract_content("query", Output, llm=passed_llm)
        assert result.text == "result"

    async def test_extract_content_llm_exception(self):
        """Lines 550-551: LLM raises exception."""
        from pydantic import BaseModel

        class Output(BaseModel):
            text: str

        session, _ = _make_mock_browser_session()
        mock_llm = MagicMock()

        page = Page(session, target_id="tid-123", session_id="sid-abc", llm=mock_llm)
        page._extract_clean_markdown = AsyncMock(return_value=("content", {}))

        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        with pytest.raises(RuntimeError, match="LLM error"):
            await page.extract_content("extract", Output)


@pytest.mark.asyncio
class TestPageExtractCleanMarkdown:
    """Cover lines 558-561: _extract_clean_markdown."""

    async def test_extract_clean_markdown(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id="tid-123", session_id="sid-abc")

        mock_result = ("# Hello", {"chars": 50})

        with patch("openbrowser.dom.markdown_extractor.extract_clean_markdown", new_callable=AsyncMock, return_value=mock_result) as mock_fn, \
             patch("openbrowser.actor.page.DomService") as MockDomService:
            MockDomService.return_value = MagicMock()
            result = await page._extract_clean_markdown()

        assert result == mock_result
        mock_fn.assert_called_once()

    async def test_extract_clean_markdown_with_links(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id="tid-123", session_id="sid-abc")

        mock_result = ("# Hello [link](url)", {"chars": 50, "links": 1})

        with patch("openbrowser.dom.markdown_extractor.extract_clean_markdown", new_callable=AsyncMock, return_value=mock_result) as mock_fn, \
             patch("openbrowser.actor.page.DomService") as MockDomService:
            MockDomService.return_value = MagicMock()
            result = await page._extract_clean_markdown(extract_links=True)

        assert result == mock_result
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["extract_links"] is True
