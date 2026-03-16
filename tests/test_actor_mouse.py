"""Tests for openbrowser.actor.mouse module."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.actor.mouse import Mouse

logger = logging.getLogger(__name__)


def _make_mock_browser_session():
    """Create a mock BrowserSession with CDP client."""
    session = MagicMock()
    client = MagicMock()

    # Set up async CDP methods
    client.send.Input.dispatchMouseEvent = AsyncMock()
    client.send.Input.synthesizeScrollGesture = AsyncMock()
    client.send.Page.getLayoutMetrics = AsyncMock()
    client.send.Runtime.evaluate = AsyncMock()

    session.cdp_client = client
    return session, client


class TestMouseInit:
    """Tests for Mouse initialization."""

    def test_init_with_all_params(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123', target_id='tid-456')
        assert mouse._browser_session is session
        assert mouse._client is client
        assert mouse._session_id == 'sid-123'
        assert mouse._target_id == 'tid-456'

    def test_init_with_defaults(self):
        session, _ = _make_mock_browser_session()
        mouse = Mouse(session)
        assert mouse._session_id is None
        assert mouse._target_id is None


@pytest.mark.asyncio
class TestMouseClick:
    """Tests for Mouse.click method."""

    async def test_click_dispatches_press_and_release(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        await mouse.click(100, 200)

        assert client.send.Input.dispatchMouseEvent.call_count == 2

        press_call = client.send.Input.dispatchMouseEvent.call_args_list[0]
        assert press_call[0][0]['type'] == 'mousePressed'
        assert press_call[0][0]['x'] == 100
        assert press_call[0][0]['y'] == 200
        assert press_call[0][0]['button'] == 'left'
        assert press_call[0][0]['clickCount'] == 1

        release_call = client.send.Input.dispatchMouseEvent.call_args_list[1]
        assert release_call[0][0]['type'] == 'mouseReleased'
        assert release_call[0][0]['x'] == 100
        assert release_call[0][0]['y'] == 200

    async def test_click_with_right_button(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        await mouse.click(50, 50, button='right', click_count=2)

        press_call = client.send.Input.dispatchMouseEvent.call_args_list[0]
        assert press_call[0][0]['button'] == 'right'
        assert press_call[0][0]['clickCount'] == 2


@pytest.mark.asyncio
class TestMouseDown:
    """Tests for Mouse.down method."""

    async def test_down_dispatches_press(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        await mouse.down()

        client.send.Input.dispatchMouseEvent.assert_called_once()
        call_args = client.send.Input.dispatchMouseEvent.call_args[0][0]
        assert call_args['type'] == 'mousePressed'
        assert call_args['button'] == 'left'

    async def test_down_with_custom_button(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        await mouse.down(button='middle', click_count=3)

        call_args = client.send.Input.dispatchMouseEvent.call_args[0][0]
        assert call_args['button'] == 'middle'
        assert call_args['clickCount'] == 3


@pytest.mark.asyncio
class TestMouseUp:
    """Tests for Mouse.up method."""

    async def test_up_dispatches_release(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        await mouse.up()

        client.send.Input.dispatchMouseEvent.assert_called_once()
        call_args = client.send.Input.dispatchMouseEvent.call_args[0][0]
        assert call_args['type'] == 'mouseReleased'
        assert call_args['button'] == 'left'


@pytest.mark.asyncio
class TestMouseMove:
    """Tests for Mouse.move method."""

    async def test_move_dispatches_mouse_moved(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        await mouse.move(300, 400)

        client.send.Input.dispatchMouseEvent.assert_called_once()
        call_args = client.send.Input.dispatchMouseEvent.call_args[0][0]
        assert call_args['type'] == 'mouseMoved'
        assert call_args['x'] == 300
        assert call_args['y'] == 400


@pytest.mark.asyncio
class TestMouseScroll:
    """Tests for Mouse.scroll method."""

    async def test_scroll_no_session_raises(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id=None)

        with pytest.raises(RuntimeError, match='Session ID is required'):
            await mouse.scroll(delta_y=100)

    async def test_scroll_with_mouse_wheel(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        client.send.Page.getLayoutMetrics.return_value = {
            'layoutViewport': {'clientWidth': 1280, 'clientHeight': 720}
        }

        await mouse.scroll(delta_y=200)

        # Should dispatch mouseWheel event
        client.send.Input.dispatchMouseEvent.assert_called_once()
        call_args = client.send.Input.dispatchMouseEvent.call_args[1]['params']
        assert call_args['type'] == 'mouseWheel'
        assert call_args['deltaY'] == 200
        assert call_args['deltaX'] == 0

    async def test_scroll_with_custom_coordinates(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        client.send.Page.getLayoutMetrics.return_value = {
            'layoutViewport': {'clientWidth': 1280, 'clientHeight': 720}
        }

        await mouse.scroll(x=100, y=200, delta_x=50, delta_y=100)

        call_args = client.send.Input.dispatchMouseEvent.call_args[1]['params']
        assert call_args['x'] == 100
        assert call_args['y'] == 200
        assert call_args['deltaX'] == 50
        assert call_args['deltaY'] == 100

    async def test_scroll_fallback_to_synthesize(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        # Make mouse wheel fail
        client.send.Page.getLayoutMetrics.side_effect = Exception('no metrics')

        await mouse.scroll(x=0, y=0, delta_y=100)

        # Should fall back to synthesizeScrollGesture
        client.send.Input.synthesizeScrollGesture.assert_called_once()

    async def test_scroll_fallback_to_javascript(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        # Make both mouse wheel and synthesize fail
        client.send.Page.getLayoutMetrics.side_effect = Exception('no metrics')
        client.send.Input.synthesizeScrollGesture.side_effect = Exception('no synthesize')

        await mouse.scroll(x=0, y=0, delta_y=100)

        # Should fall back to JavaScript
        client.send.Runtime.evaluate.assert_called_once()
        js_expr = client.send.Runtime.evaluate.call_args[1]['params']['expression']
        assert 'scrollBy' in js_expr

    async def test_scroll_uses_viewport_center_when_no_coords(self):
        session, client = _make_mock_browser_session()
        mouse = Mouse(session, session_id='sid-123')

        client.send.Page.getLayoutMetrics.return_value = {
            'layoutViewport': {'clientWidth': 1000, 'clientHeight': 500}
        }

        await mouse.scroll(delta_y=50)

        call_args = client.send.Input.dispatchMouseEvent.call_args[1]['params']
        assert call_args['x'] == 500  # center of 1000
        assert call_args['y'] == 250  # center of 500
