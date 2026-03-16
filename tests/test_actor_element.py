"""Tests for openbrowser.actor.element module."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.actor.element import BoundingBox, Element, ElementInfo, Position, _MODIFIER_MAP

logger = logging.getLogger(__name__)


def _make_mock_browser_session():
    """Create a mock BrowserSession with CDP client."""
    session = MagicMock()
    client = MagicMock()

    # Set up async CDP methods for DOM, Input, Runtime, Page
    client.send.DOM.pushNodesByBackendIdsToFrontend = AsyncMock()
    client.send.DOM.resolveNode = AsyncMock()
    client.send.DOM.getBoxModel = AsyncMock()
    client.send.DOM.getContentQuads = AsyncMock()
    client.send.DOM.scrollIntoViewIfNeeded = AsyncMock()
    client.send.DOM.focus = AsyncMock()
    client.send.DOM.getAttributes = AsyncMock()
    client.send.DOM.requestChildNodes = AsyncMock()
    client.send.DOM.describeNode = AsyncMock()

    client.send.Input.dispatchMouseEvent = AsyncMock()
    client.send.Input.dispatchKeyEvent = AsyncMock()

    client.send.Runtime.callFunctionOn = AsyncMock()
    client.send.Runtime.evaluate = AsyncMock()

    client.send.Page.getLayoutMetrics = AsyncMock()
    client.send.Page.captureScreenshot = AsyncMock()

    session.cdp_client = client
    return session, client


class TestElementInit:
    """Tests for Element initialization."""

    def test_init_basic(self):
        session, client = _make_mock_browser_session()
        elem = Element(session, backend_node_id=42, session_id='sid-abc')
        assert elem._browser_session is session
        assert elem._client is client
        assert elem._backend_node_id == 42
        assert elem._session_id == 'sid-abc'

    def test_init_default_session_id(self):
        session, _ = _make_mock_browser_session()
        elem = Element(session, backend_node_id=1)
        assert elem._session_id is None

    def test_has_slots(self):
        assert '__slots__' in dir(Element)
        expected = ('_browser_session', '_client', '_backend_node_id', '_session_id')
        assert Element.__slots__ == expected


class TestModifierMap:
    """Tests for the module-level modifier map constant."""

    def test_modifier_map_values(self):
        assert _MODIFIER_MAP == {'Alt': 1, 'Control': 2, 'Meta': 4, 'Shift': 8}


class TestTypedDicts:
    """Tests for Position, BoundingBox, and ElementInfo TypedDicts."""

    def test_position_creation(self):
        pos = Position(x=10.0, y=20.0)
        assert pos['x'] == 10.0
        assert pos['y'] == 20.0

    def test_bounding_box_creation(self):
        box = BoundingBox(x=1.0, y=2.0, width=100.0, height=50.0)
        assert box['x'] == 1.0
        assert box['width'] == 100.0

    def test_element_info_creation(self):
        info = ElementInfo(
            backendNodeId=1,
            nodeId=2,
            nodeName='DIV',
            nodeType=1,
            nodeValue=None,
            attributes={'class': 'main'},
            boundingBox=None,
            error=None,
        )
        assert info['backendNodeId'] == 1
        assert info['nodeName'] == 'DIV'


@pytest.mark.asyncio
class TestElementGetNodeId:
    """Tests for Element._get_node_id."""

    async def test_get_node_id(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [99]}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        node_id = await elem._get_node_id()
        assert node_id == 99
        client.send.DOM.pushNodesByBackendIdsToFrontend.assert_called_once_with(
            {'backendNodeIds': [42]}, session_id='sid-abc'
        )


@pytest.mark.asyncio
class TestElementGetRemoteObjectId:
    """Tests for Element._get_remote_object_id."""

    async def test_returns_object_id(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-123'}}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        obj_id = await elem._get_remote_object_id()
        assert obj_id == 'obj-123'

    async def test_returns_none_when_no_object_id(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {}}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        obj_id = await elem._get_remote_object_id()
        assert obj_id is None


@pytest.mark.asyncio
class TestElementGetBoundingBox:
    """Tests for Element.get_bounding_box."""

    async def test_returns_bounding_box(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getBoxModel.return_value = {
            'model': {
                'content': [10, 20, 110, 20, 110, 70, 10, 70]
            }
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        box = await elem.get_bounding_box()
        assert box is not None
        assert box['x'] == 10
        assert box['y'] == 20
        assert box['width'] == 100
        assert box['height'] == 50

    async def test_returns_none_when_no_model(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getBoxModel.return_value = {}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        box = await elem.get_bounding_box()
        assert box is None

    async def test_returns_none_when_content_too_short(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getBoxModel.return_value = {'model': {'content': [10, 20, 30, 40]}}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        box = await elem.get_bounding_box()
        assert box is None

    async def test_returns_none_on_exception(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.side_effect = Exception('fail')
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        box = await elem.get_bounding_box()
        assert box is None


@pytest.mark.asyncio
class TestElementHover:
    """Tests for Element.hover."""

    async def test_hover_dispatches_mouse_moved(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getBoxModel.return_value = {
            'model': {'content': [0, 0, 100, 0, 100, 50, 0, 50]}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        await elem.hover()

        client.send.Input.dispatchMouseEvent.assert_called_once()
        call_args = client.send.Input.dispatchMouseEvent.call_args[0][0]
        assert call_args['type'] == 'mouseMoved'
        assert call_args['x'] == 50.0
        assert call_args['y'] == 25.0

    async def test_hover_raises_when_no_bounding_box(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.side_effect = Exception('fail')
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        with pytest.raises(RuntimeError, match='not visible'):
            await elem.hover()


@pytest.mark.asyncio
class TestElementFocus:
    """Tests for Element.focus."""

    async def test_focus_calls_dom_focus(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        await elem.focus()

        client.send.DOM.focus.assert_called_once_with({'nodeId': 10}, session_id='sid-abc')


@pytest.mark.asyncio
class TestElementGetAttribute:
    """Tests for Element.get_attribute."""

    async def test_get_existing_attribute(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getAttributes.return_value = {
            'attributes': ['class', 'btn', 'id', 'submit-btn', 'type', 'button']
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.get_attribute('class')
        assert result == 'btn'

        result = await elem.get_attribute('id')
        assert result == 'submit-btn'

    async def test_get_nonexistent_attribute(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getAttributes.return_value = {
            'attributes': ['class', 'btn']
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.get_attribute('data-test')
        assert result is None


@pytest.mark.asyncio
class TestElementScreenshot:
    """Tests for Element.screenshot."""

    async def test_screenshot_returns_base64(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getBoxModel.return_value = {
            'model': {'content': [0, 0, 100, 0, 100, 50, 0, 50]}
        }
        client.send.Page.captureScreenshot.return_value = {'data': 'base64imagedata'}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.screenshot()
        assert result == 'base64imagedata'

    async def test_screenshot_with_quality(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getBoxModel.return_value = {
            'model': {'content': [0, 0, 100, 0, 100, 50, 0, 50]}
        }
        client.send.Page.captureScreenshot.return_value = {'data': 'base64'}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        await elem.screenshot(format='jpeg', quality=80)

        call_args = client.send.Page.captureScreenshot.call_args[0][0]
        assert call_args['quality'] == 80
        assert call_args['format'] == 'jpeg'

    async def test_screenshot_no_quality_for_png(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getBoxModel.return_value = {
            'model': {'content': [0, 0, 100, 0, 100, 50, 0, 50]}
        }
        client.send.Page.captureScreenshot.return_value = {'data': 'base64'}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        await elem.screenshot(format='png', quality=80)

        call_args = client.send.Page.captureScreenshot.call_args[0][0]
        assert 'quality' not in call_args

    async def test_screenshot_raises_when_no_box(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.side_effect = Exception('fail')
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        with pytest.raises(RuntimeError, match='not visible'):
            await elem.screenshot()


@pytest.mark.asyncio
class TestElementClick:
    """Tests for Element.click method."""

    async def test_click_with_content_quads(self):
        """Test clicking when getContentQuads returns valid data."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            'layoutViewport': {'clientWidth': 1280, 'clientHeight': 720}
        }
        client.send.DOM.getContentQuads.return_value = {
            'quads': [[100, 100, 200, 100, 200, 200, 100, 200]]
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        await elem.click()

        # Should have dispatched mouseMoved, mousePressed, mouseReleased
        assert client.send.Input.dispatchMouseEvent.call_count >= 3

    async def test_click_falls_back_to_box_model(self):
        """Test fallback to box model when content quads fail."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            'layoutViewport': {'clientWidth': 1280, 'clientHeight': 720}
        }
        client.send.DOM.getContentQuads.side_effect = Exception('no quads')
        client.send.DOM.getBoxModel.return_value = {
            'model': {'content': [50, 50, 150, 50, 150, 100, 50, 100]}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        await elem.click()

        # Should still dispatch mouse events via box model fallback
        assert client.send.Input.dispatchMouseEvent.call_count >= 3

    async def test_click_falls_back_to_js_click(self):
        """Test fallback to JavaScript click when all geometry methods fail."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            'layoutViewport': {'clientWidth': 1280, 'clientHeight': 720}
        }
        client.send.DOM.getContentQuads.side_effect = Exception('no quads')
        client.send.DOM.getBoxModel.side_effect = Exception('no box')
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        client.send.Runtime.callFunctionOn.return_value = {}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        # The third fallback (JS getBoundingClientRect) also fails because
        # the callFunctionOn doesn't return proper value. Then the final JS click fallback fires.
        await elem.click()

        # Should have called callFunctionOn for JS click
        assert client.send.Runtime.callFunctionOn.call_count >= 1

    async def test_click_with_modifiers(self):
        """Test clicking with keyboard modifiers."""
        session, client = _make_mock_browser_session()
        client.send.Page.getLayoutMetrics.return_value = {
            'layoutViewport': {'clientWidth': 1280, 'clientHeight': 720}
        }
        client.send.DOM.getContentQuads.return_value = {
            'quads': [[100, 100, 200, 100, 200, 200, 100, 200]]
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        await elem.click(modifiers=['Shift', 'Control'])

        # Check that modifier bitmask was computed (Shift=8 | Control=2 = 10)
        # Verify dispatchMouseEvent was called with correct modifier bitmask
        # Shift=8, Control=2 => combined modifiers=10
        dispatch_calls = client.send.Input.dispatchMouseEvent.call_args_list
        assert len(dispatch_calls) >= 1
        # Check that mousePressed call has correct combined bitmask: Shift=8 | Control=2 = 10
        found_correct_modifiers = False
        for call in dispatch_calls:
            params = call[0][0] if call[0] else call[1].get('params', {})
            if params.get('type') == 'mousePressed' and params.get('modifiers', 0) == 10:
                found_correct_modifiers = True
                break
        assert found_correct_modifiers, (
            "Expected mousePressed call with modifiers=10 (Shift=8|Control=2)"
        )


@pytest.mark.asyncio
class TestElementCheck:
    """Tests for Element.check."""

    async def test_check_calls_click(self):
        session, client = _make_mock_browser_session()
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        # Mock click via patch since Element.click is read-only
        with patch.object(Element, 'click', new_callable=AsyncMock) as mock_click:
            await elem.check()
            mock_click.assert_called_once()


class TestGetCharModifiersAndVk:
    """Tests for Element._get_char_modifiers_and_vk."""

    def _make_element(self):
        session, _ = _make_mock_browser_session()
        return Element(session, backend_node_id=1)

    def test_shift_char(self):
        elem = self._make_element()
        modifiers, vk, base = elem._get_char_modifiers_and_vk('!')
        assert modifiers == 8  # Shift
        assert base == '1'
        assert vk == 49

    def test_uppercase_letter(self):
        elem = self._make_element()
        modifiers, vk, base = elem._get_char_modifiers_and_vk('A')
        assert modifiers == 8
        assert base == 'a'
        assert vk == 65

    def test_lowercase_letter(self):
        elem = self._make_element()
        modifiers, vk, base = elem._get_char_modifiers_and_vk('a')
        assert modifiers == 0
        assert vk == 65
        assert base == 'a'

    def test_digit(self):
        elem = self._make_element()
        modifiers, vk, base = elem._get_char_modifiers_and_vk('5')
        assert modifiers == 0
        assert vk == 53
        assert base == '5'

    def test_space(self):
        elem = self._make_element()
        modifiers, vk, base = elem._get_char_modifiers_and_vk(' ')
        assert modifiers == 0
        assert vk == 32
        assert base == ' '

    def test_special_no_shift_chars(self):
        elem = self._make_element()
        modifiers, vk, base = elem._get_char_modifiers_and_vk('-')
        assert modifiers == 0
        assert vk == 189

    def test_shift_chars_all(self):
        elem = self._make_element()
        shift_chars = '!@#$%^&*()_+{}|:"<>?~'
        for char in shift_chars:
            modifiers, vk, base = elem._get_char_modifiers_and_vk(char)
            assert modifiers == 8, f'Expected Shift for {char}'

    def test_fallback_for_unknown_char(self):
        elem = self._make_element()
        # Non-ASCII character
        modifiers, vk, base = elem._get_char_modifiers_and_vk('\u00e9')
        assert modifiers == 0


class TestGetKeyCodeForChar:
    """Tests for Element._get_key_code_for_char."""

    def _make_element(self):
        session, _ = _make_mock_browser_session()
        return Element(session, backend_node_id=1)

    def test_space(self):
        elem = self._make_element()
        assert elem._get_key_code_for_char(' ') == 'Space'

    def test_period(self):
        elem = self._make_element()
        assert elem._get_key_code_for_char('.') == 'Period'

    def test_alpha_char(self):
        elem = self._make_element()
        assert elem._get_key_code_for_char('a') == 'KeyA'
        assert elem._get_key_code_for_char('Z') == 'KeyZ'

    def test_digit_char(self):
        elem = self._make_element()
        assert elem._get_key_code_for_char('0') == 'Digit0'
        assert elem._get_key_code_for_char('9') == 'Digit9'

    def test_non_ascii_alpha_char(self):
        elem = self._make_element()
        # Non-ASCII alpha characters get KeyX format since isalpha() is True for them
        result = elem._get_key_code_for_char('\u00e9')
        assert result.startswith('Key')

    def test_non_alpha_non_digit_fallback(self):
        elem = self._make_element()
        # A character that is not alpha, not digit, and not in the mapping
        result = elem._get_key_code_for_char('\u2603')  # snowman symbol
        assert result == 'Unidentified'

    def test_all_mapped_chars(self):
        elem = self._make_element()
        mapped = {
            ' ': 'Space', '.': 'Period', ',': 'Comma', '-': 'Minus',
            '@': 'Digit2', '!': 'Digit1', '?': 'Slash', ':': 'Semicolon',
            ';': 'Semicolon', '/': 'Slash', '\\': 'Backslash',
        }
        for char, expected in mapped.items():
            assert elem._get_key_code_for_char(char) == expected, f'Failed for {char}'


@pytest.mark.asyncio
class TestElementFocusSimple:
    """Tests for Element._focus_element_simple."""

    async def test_cdp_focus_succeeds(self):
        session, client = _make_mock_browser_session()
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem._focus_element_simple(
            backend_node_id=42, object_id='obj-1',
            cdp_client=client, session_id='sid-abc'
        )
        assert result is True
        client.send.DOM.focus.assert_called_once()

    async def test_falls_back_to_js_focus(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.focus.side_effect = Exception('CDP focus failed')
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem._focus_element_simple(
            backend_node_id=42, object_id='obj-1',
            cdp_client=client, session_id='sid-abc'
        )
        assert result is True
        client.send.Runtime.callFunctionOn.assert_called_once()

    async def test_falls_back_to_click_focus(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.focus.side_effect = Exception('CDP focus failed')
        client.send.Runtime.callFunctionOn.side_effect = Exception('JS focus failed')
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem._focus_element_simple(
            backend_node_id=42, object_id='obj-1',
            cdp_client=client, session_id='sid-abc',
            input_coordinates={'input_x': 100, 'input_y': 200},
        )
        assert result is True
        # Should have dispatched mousePressed and mouseReleased
        assert client.send.Input.dispatchMouseEvent.call_count == 2

    async def test_all_strategies_fail_no_coords(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.focus.side_effect = Exception('fail')
        client.send.Runtime.callFunctionOn.side_effect = Exception('fail')
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem._focus_element_simple(
            backend_node_id=42, object_id='obj-1',
            cdp_client=client, session_id='sid-abc',
            input_coordinates=None,
        )
        assert result is False


@pytest.mark.asyncio
class TestElementClearTextField:
    """Tests for Element._clear_text_field."""

    async def test_clear_success(self):
        session, client = _make_mock_browser_session()
        # First call: clear the field. Second call: verify empty.
        client.send.Runtime.callFunctionOn.side_effect = [
            {'result': {'value': ''}},  # clear
            {'result': {'value': ''}},  # verify
        ]
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem._clear_text_field('obj-1', client, 'sid-abc')
        assert result is True

    async def test_clear_partial_failure_falls_back_to_triple_click(self):
        session, client = _make_mock_browser_session()
        # JS clear: field still has text. Triple-click fallback succeeds.
        client.send.Runtime.callFunctionOn.side_effect = [
            {'result': {'value': ''}},  # clear (seems to succeed)
            {'result': {'value': 'still here'}},  # verify (oops)
            {'result': {'value': {'x': 50, 'y': 50, 'width': 100, 'height': 30}}},  # bounds for triple-click
        ]
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem._clear_text_field('obj-1', client, 'sid-abc')
        assert result is True

    async def test_clear_all_strategies_fail(self):
        session, client = _make_mock_browser_session()
        client.send.Runtime.callFunctionOn.side_effect = Exception('fail')
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem._clear_text_field('obj-1', client, 'sid-abc')
        assert result is False


@pytest.mark.asyncio
class TestElementGetBasicInfo:
    """Tests for Element.get_basic_info."""

    async def test_get_basic_info_success(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.describeNode.return_value = {
            'node': {
                'nodeName': 'INPUT',
                'nodeType': 1,
                'nodeValue': None,
                'attributes': ['type', 'text', 'name', 'email'],
            }
        }
        client.send.DOM.getBoxModel.return_value = {
            'model': {'content': [0, 0, 100, 0, 100, 30, 0, 30]}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        info = await elem.get_basic_info()
        assert info['backendNodeId'] == 42
        assert info['nodeId'] == 10
        assert info['nodeName'] == 'INPUT'
        assert info['attributes'] == {'type': 'text', 'name': 'email'}
        assert info['boundingBox'] is not None
        assert info['error'] is None

    async def test_get_basic_info_on_error(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.side_effect = Exception('gone')
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        info = await elem.get_basic_info()
        assert info['backendNodeId'] == 42
        assert info['nodeId'] is None
        assert info['error'] is not None


@pytest.mark.asyncio
class TestElementEvaluate:
    """Tests for Element.evaluate."""

    async def test_evaluate_simple_arrow(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        client.send.Runtime.callFunctionOn.return_value = {
            'result': {'value': 'hello'}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.evaluate('() => this.textContent')
        assert result == 'hello'

    async def test_evaluate_with_args(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        client.send.Runtime.callFunctionOn.return_value = {
            'result': {'value': 'red'}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.evaluate('(color) => { this.style.color = color; return color; }', 'red')
        assert result == 'red'

    async def test_evaluate_returns_none_as_empty_string(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        client.send.Runtime.callFunctionOn.return_value = {
            'result': {'value': None}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.evaluate('() => null')
        assert result == ''

    async def test_evaluate_returns_dict_as_json(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        client.send.Runtime.callFunctionOn.return_value = {
            'result': {'value': {'key': 'val'}}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.evaluate('() => ({key: "val"})')
        assert '"key"' in result
        assert '"val"' in result

    async def test_evaluate_returns_number_as_string(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        client.send.Runtime.callFunctionOn.return_value = {
            'result': {'value': 42}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.evaluate('() => 42')
        assert result == '42'

    async def test_evaluate_raises_on_exception_details(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        client.send.Runtime.callFunctionOn.return_value = {
            'result': {},
            'exceptionDetails': 'ReferenceError: x is not defined'
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        with pytest.raises(RuntimeError, match='JavaScript evaluation failed'):
            await elem.evaluate('() => x')

    async def test_evaluate_rejects_non_arrow_function(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        with pytest.raises(ValueError, match='must start with'):
            await elem.evaluate('document.title')

    async def test_evaluate_no_remote_object_id(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {}}
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        with pytest.raises(RuntimeError, match='no remote object ID'):
            await elem.evaluate('() => this.id')

    async def test_evaluate_async_arrow(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.resolveNode.return_value = {'object': {'objectId': 'obj-1'}}
        client.send.Runtime.callFunctionOn.return_value = {
            'result': {'value': 'done'}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        result = await elem.evaluate('async () => { return "done"; }')
        assert result == 'done'

        # Verify the function declaration was converted to async function
        call_params = client.send.Runtime.callFunctionOn.call_args[0][0]
        assert 'async function' in call_params['functionDeclaration']


@pytest.mark.asyncio
class TestElementDragTo:
    """Tests for Element.drag_to."""

    async def test_drag_to_position(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.getBoxModel.return_value = {
            'model': {'content': [0, 0, 100, 0, 100, 50, 0, 50]}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        target_pos = Position(x=300, y=400)
        await elem.drag_to(target_pos)

        assert client.send.Input.dispatchMouseEvent.call_count == 3
        calls = client.send.Input.dispatchMouseEvent.call_args_list
        assert calls[0][0][0]['type'] == 'mousePressed'
        assert calls[1][0][0]['type'] == 'mouseMoved'
        assert calls[1][0][0]['x'] == 300
        assert calls[1][0][0]['y'] == 400
        assert calls[2][0][0]['type'] == 'mouseReleased'

    async def test_drag_to_with_source_position(self):
        session, client = _make_mock_browser_session()
        elem = Element(session, backend_node_id=42, session_id='sid-abc')

        source_pos = Position(x=10, y=20)
        target_pos = Position(x=300, y=400)
        await elem.drag_to(target_pos, source_position=source_pos)

        calls = client.send.Input.dispatchMouseEvent.call_args_list
        assert calls[0][0][0]['x'] == 10
        assert calls[0][0][0]['y'] == 20

    async def test_drag_to_element(self):
        session, client = _make_mock_browser_session()
        source = Element(session, backend_node_id=42, session_id='sid-abc')
        target = Element(session, backend_node_id=43, session_id='sid-abc')

        # Mock get_bounding_box via patch since Element uses __slots__
        source_box = BoundingBox(x=0, y=0, width=100, height=50)
        target_box = BoundingBox(x=200, y=200, width=100, height=50)

        with patch.object(Element, 'get_bounding_box', new_callable=AsyncMock) as mock_bbox:
            mock_bbox.side_effect = [source_box, target_box]
            await source.drag_to(target)

        assert client.send.Input.dispatchMouseEvent.call_count == 3


@pytest.mark.asyncio
class TestElementSelectOption:
    """Tests for Element.select_option."""

    async def test_select_option_with_string(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.describeNode.return_value = {
            'node': {
                'children': [
                    {
                        'nodeName': 'OPTION',
                        'attributes': ['value', 'opt1'],
                        'nodeValue': 'Option 1',
                        'nodeId': 20,
                    }
                ]
            }
        }
        # For the option describeNode call
        client.send.DOM.describeNode.side_effect = [
            # First call: describe the select element
            {
                'node': {
                    'children': [
                        {
                            'nodeName': 'OPTION',
                            'attributes': ['value', 'opt1'],
                            'nodeValue': 'Option 1',
                            'nodeId': 20,
                        }
                    ]
                }
            },
            # Second call: describe the option element
            {'node': {'backendNodeId': 100}},
        ]
        elem = Element(session, backend_node_id=42, session_id='sid-abc')
        # Mock the click method on option element to verify it gets called
        with patch.object(Element, 'click', new_callable=AsyncMock) as mock_click:
            await elem.select_option('opt1')
            mock_click.assert_called_once()

    async def test_select_option_converts_string_to_list(self):
        """Test that a single string value is converted to a list."""
        session, client = _make_mock_browser_session()
        client.send.DOM.pushNodesByBackendIdsToFrontend.return_value = {'nodeIds': [10]}
        client.send.DOM.describeNode.return_value = {
            'node': {'children': []}
        }
        elem = Element(session, backend_node_id=42, session_id='sid-abc')
        # Should not raise even with no matching children
        await elem.select_option('some_value')
