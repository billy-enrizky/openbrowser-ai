"""Focused tests for coordinate-based browser controls."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from openbrowser.actor.mouse import Mouse
from openbrowser.code_use.descriptions import EXECUTE_CODE_DESCRIPTION, EXECUTE_CODE_DESCRIPTION_COMPACT
from openbrowser.code_use.service import CodeAgent
from openbrowser.tools.service import CodeAgentTools, Tools
from openbrowser.tools.views import ClickXYAction, HoverXYAction, ScrollXYAction


PROMPT_PATHS = [
    Path('src/openbrowser/code_use/system_prompt.md'),
    Path('src/openbrowser/agent/system_prompt.md'),
    Path('src/openbrowser/agent/system_prompt_no_thinking.md'),
    Path('src/openbrowser/agent/system_prompt_flash.md'),
]


def _make_browser_session():
    session = MagicMock()
    cdp_session = MagicMock(session_id='session-123', target_id='target-456')
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
    session.cdp_client = MagicMock()
    session.cdp_client.send.Input.dispatchMouseEvent = AsyncMock()
    session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
        return_value={'layoutViewport': {'clientWidth': 1024, 'clientHeight': 768}}
    )
    return session


def _mouse_event_payload(call):
    """Normalize the CDP client's positional and keyword dispatch conventions."""
    return call.args[0] if call.args else call.kwargs['params']


def test_coordinate_actions_use_css_viewport_pixel_defaults():
    assert ClickXYAction(x=10, y=20).model_dump() == {
        'x': 10,
        'y': 20,
        'button': 'left',
        'click_count': 1,
    }
    assert HoverXYAction(x=10, y=20).model_dump() == {'x': 10, 'y': 20}
    assert ScrollXYAction(x=10, y=20, delta_y=120).model_dump() == {
        'x': 10,
        'y': 20,
        'delta_x': 0,
        'delta_y': 120,
    }


@pytest.mark.parametrize('model', [ClickXYAction, HoverXYAction, ScrollXYAction])
def test_coordinate_actions_preserve_fractional_css_pixels(model):
    action = model(x=12.5, y=34.25)

    assert action.x == 12.5
    assert action.y == 34.25


def test_coordinate_actions_dispatch_fractional_css_pixels():
    session = _make_browser_session()

    asyncio.run(Tools().registry.execute_action('click_xy', {'x': 12.5, 'y': 34.25}, browser_session=session))
    asyncio.run(Tools().registry.execute_action('hover_xy', {'x': 12.5, 'y': 34.25}, browser_session=session))

    events = [_mouse_event_payload(call) for call in session.cdp_client.send.Input.dispatchMouseEvent.await_args_list]
    assert events == [
        {'type': 'mousePressed', 'x': 12.5, 'y': 34.25, 'button': 'left', 'clickCount': 1},
        {'type': 'mouseReleased', 'x': 12.5, 'y': 34.25, 'button': 'left', 'clickCount': 1},
        {'type': 'mouseMoved', 'x': 12.5, 'y': 34.25},
    ]


@pytest.mark.parametrize(
    ('model', 'kwargs'),
    [
        (ClickXYAction, {'x': -1, 'y': 0}),
        (HoverXYAction, {'x': 0, 'y': -1}),
        (ScrollXYAction, {'x': -1, 'y': 0}),
        (ClickXYAction, {'x': 0, 'y': 0, 'button': 'side'}),
        (ClickXYAction, {'x': 0, 'y': 0, 'click_count': 0}),
        (ClickXYAction, {'x': 0, 'y': 0, 'click_count': 4}),
    ],
)
def test_coordinate_actions_reject_invalid_input(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_coordinate_actions_are_registered_for_tools_and_code_agent_tools():
    required = {'click_xy', 'hover_xy', 'scroll_xy'}

    for tools_type in (Tools, CodeAgentTools):
        assert required <= set(tools_type().registry.registry.actions)


def test_click_xy_dispatches_press_and_release_events():
    session = _make_browser_session()

    result = asyncio.run(
        Tools().registry.execute_action(
            'click_xy',
            {'x': 21, 'y': 34, 'click_count': 2},
            browser_session=session,
        )
    )

    calls = session.cdp_client.send.Input.dispatchMouseEvent.await_args_list
    assert [call.args[0]['type'] for call in calls] == ['mousePressed', 'mouseReleased']
    assert all(call.args[0]['x'] == 21 and call.args[0]['y'] == 34 for call in calls)
    assert all(call.args[0]['button'] == 'left' and call.args[0]['clickCount'] == 2 for call in calls)
    assert result.error is None
    assert result.extracted_content == 'Clicked at (21, 34) button=left count=2'


def test_hover_and_scroll_xy_dispatch_mouse_events():
    session = _make_browser_session()
    tools = Tools()

    hover_result = asyncio.run(
        tools.registry.execute_action('hover_xy', {'x': 31, 'y': 47}, browser_session=session)
    )
    scroll_result = asyncio.run(
        tools.registry.execute_action(
            'scroll_xy', {'x': 31, 'y': 47, 'delta_x': 12, 'delta_y': 240}, browser_session=session
        )
    )

    calls = session.cdp_client.send.Input.dispatchMouseEvent.await_args_list
    assert _mouse_event_payload(calls[0]) == {'type': 'mouseMoved', 'x': 31, 'y': 47}
    assert _mouse_event_payload(calls[1]) == {
        'type': 'mouseWheel', 'x': 31, 'y': 47, 'deltaX': 12, 'deltaY': 240,
    }
    assert hover_result.error is None
    assert hover_result.extracted_content == 'Hovered at (31, 47)'
    assert scroll_result.error is None
    assert scroll_result.extracted_content == 'Wheel at (31, 47) dx=12 dy=240'


def test_scroll_xy_preserves_zero_coordinate_anchor():
    session = _make_browser_session()

    asyncio.run(
        Tools().registry.execute_action(
            'scroll_xy', {'x': 0, 'y': 0, 'delta_y': 240}, browser_session=session
        )
    )

    event = _mouse_event_payload(session.cdp_client.send.Input.dispatchMouseEvent.await_args_list[0])
    assert event['x'] == 0
    assert event['y'] == 0
    assert event['deltaY'] == 240


@pytest.mark.parametrize(
    ('action_name', 'params', 'method_name'),
    [
        ('click_xy', {'x': 1, 'y': 2}, 'click'),
        ('hover_xy', {'x': 1, 'y': 2}, 'move'),
        ('scroll_xy', {'x': 1, 'y': 2, 'delta_y': 3}, 'scroll'),
    ],
)
def test_coordinate_actions_return_error_when_mouse_dispatch_fails(monkeypatch, action_name, params, method_name):
    session = _make_browser_session()
    monkeypatch.setattr(Mouse, method_name, AsyncMock(side_effect=RuntimeError('CDP unavailable')))

    result = asyncio.run(Tools().registry.execute_action(action_name, params, browser_session=session))

    assert result.error is not None
    assert 'CDP unavailable' in result.error


@pytest.mark.parametrize('path', PROMPT_PATHS)
def test_coordinate_prompt_guidance_requires_no_index_css_pixels_and_dpr(path):
    text = path.read_text()

    assert 'click_xy' in text
    assert 'CSS viewport pixels' in text
    assert 'devicePixelRatio' in text
    assert 'no `[index]`' in text or 'no-index' in text


def test_coordinate_actions_cover_a_canvas_style_surface_contract():
    """Models the exact no-index surface events needed by a local canvas fixture."""
    session = _make_browser_session()
    tools = Tools()

    asyncio.run(tools.registry.execute_action('hover_xy', {'x': 50, 'y': 60}, browser_session=session))
    asyncio.run(tools.registry.execute_action('click_xy', {'x': 50, 'y': 60}, browser_session=session))
    asyncio.run(
        tools.registry.execute_action('scroll_xy', {'x': 50, 'y': 60, 'delta_y': 180}, browser_session=session)
    )

    events = [_mouse_event_payload(call) for call in session.cdp_client.send.Input.dispatchMouseEvent.await_args_list]
    assert events == [
        {'type': 'mouseMoved', 'x': 50, 'y': 60},
        {'type': 'mousePressed', 'x': 50, 'y': 60, 'button': 'left', 'clickCount': 1},
        {'type': 'mouseReleased', 'x': 50, 'y': 60, 'button': 'left', 'clickCount': 1},
        {'type': 'mouseWheel', 'x': 50, 'y': 60, 'deltaX': 0, 'deltaY': 180},
    ]


def test_code_execution_descriptions_document_coordinate_actions():
    for description in (EXECUTE_CODE_DESCRIPTION_COMPACT, EXECUTE_CODE_DESCRIPTION):
        assert 'click_xy' in description
        assert 'hover_xy' in description
        assert 'scroll_xy' in description
        assert 'CSS viewport pixels' in description
        assert 'devicePixelRatio' in description


def test_code_agent_fallback_prompt_documents_coordinate_actions():
    agent = object.__new__(CodeAgent)

    with patch('builtins.open', side_effect=FileNotFoundError):
        prompt = agent._get_code_agent_system_prompt()

    assert 'click_xy' in prompt
    assert 'hover_xy' in prompt
    assert 'scroll_xy' in prompt
    assert 'CSS viewport pixels' in prompt
    assert 'devicePixelRatio' in prompt
    assert 'Prefer indexed actions' in prompt
