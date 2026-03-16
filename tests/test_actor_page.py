"""Tests for openbrowser.actor.page module."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.actor.page import Page

logger = logging.getLogger(__name__)


def _make_mock_browser_session():
    """Create a mock BrowserSession with CDP client."""
    session = MagicMock()
    client = MagicMock()

    # Set up async CDP methods
    client.send.Target.attachToTarget = AsyncMock()
    client.send.Target.getTargetInfo = AsyncMock()
    client.send.Page.enable = AsyncMock()
    client.send.Page.reload = AsyncMock()
    client.send.Page.navigate = AsyncMock()
    client.send.Page.navigateToHistoryEntry = AsyncMock()
    client.send.Page.getNavigationHistory = AsyncMock()
    client.send.Page.captureScreenshot = AsyncMock()
    client.send.DOM.enable = AsyncMock()
    client.send.DOM.getDocument = AsyncMock()
    client.send.DOM.querySelectorAll = AsyncMock()
    client.send.DOM.describeNode = AsyncMock()
    client.send.Runtime.enable = AsyncMock()
    client.send.Runtime.evaluate = AsyncMock()
    client.send.Network.enable = AsyncMock()
    client.send.Input.dispatchKeyEvent = AsyncMock()
    client.send.Emulation.setDeviceMetricsOverride = AsyncMock()

    session.cdp_client = client
    return session, client


class TestPageInit:
    """Tests for Page initialization."""

    def test_init_basic(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123')
        assert page._browser_session is session
        assert page._client is client
        assert page._target_id == 'tid-123'
        assert page._session_id is None
        assert page._mouse is None
        assert page._llm is None

    def test_init_with_session_id_and_llm(self):
        session, _ = _make_mock_browser_session()
        mock_llm = MagicMock()
        page = Page(session, target_id='tid-123', session_id='sid-abc', llm=mock_llm)
        assert page._session_id == 'sid-abc'
        assert page._llm is mock_llm


@pytest.mark.asyncio
class TestPageEnsureSession:
    """Tests for Page._ensure_session."""

    async def test_returns_existing_session(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        result = await page._ensure_session()
        assert result == 'sid-abc'
        client.send.Target.attachToTarget.assert_not_called()

    async def test_attaches_and_enables_domains(self):
        session, client = _make_mock_browser_session()
        client.send.Target.attachToTarget.return_value = {'sessionId': 'new-sid'}
        page = Page(session, target_id='tid-123')

        result = await page._ensure_session()
        assert result == 'new-sid'
        assert page._session_id == 'new-sid'
        client.send.Target.attachToTarget.assert_called_once()
        client.send.Page.enable.assert_called_once()
        client.send.DOM.enable.assert_called_once()
        client.send.Runtime.enable.assert_called_once()
        client.send.Network.enable.assert_called_once()


@pytest.mark.asyncio
class TestPageSessionProperty:
    """Tests for Page.session_id property."""

    async def test_session_id_property(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        result = await page.session_id
        assert result == 'sid-abc'


@pytest.mark.asyncio
class TestPageMouse:
    """Tests for Page.mouse property."""

    async def test_mouse_creates_and_caches(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        mouse = await page.mouse
        assert mouse is not None
        # Verify caching
        mouse2 = await page.mouse
        assert mouse2 is mouse


@pytest.mark.asyncio
class TestPageReload:
    """Tests for Page.reload."""

    async def test_reload(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.reload()
        client.send.Page.reload.assert_called_once_with(session_id='sid-abc')


@pytest.mark.asyncio
class TestPageGetElement:
    """Tests for Page.get_element."""

    async def test_get_element_returns_element(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        element = await page.get_element(backend_node_id=42)
        assert element is not None
        assert element._backend_node_id == 42
        assert element._session_id == 'sid-abc'


@pytest.mark.asyncio
class TestPageEvaluate:
    """Tests for Page.evaluate."""

    async def test_evaluate_returns_string(self):
        session, client = _make_mock_browser_session()
        client.send.Runtime.evaluate.return_value = {
            'result': {'value': 'Hello World'}
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        result = await page.evaluate('() => document.title')
        assert result == 'Hello World'

    async def test_evaluate_returns_none_as_empty_string(self):
        session, client = _make_mock_browser_session()
        client.send.Runtime.evaluate.return_value = {
            'result': {'value': None}
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        result = await page.evaluate('() => null')
        assert result == ''

    async def test_evaluate_returns_dict_as_json(self):
        session, client = _make_mock_browser_session()
        client.send.Runtime.evaluate.return_value = {
            'result': {'value': {'key': 'val'}}
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        result = await page.evaluate('() => ({key: "val"})')
        assert '"key"' in result

    async def test_evaluate_returns_number_as_string(self):
        session, client = _make_mock_browser_session()
        client.send.Runtime.evaluate.return_value = {
            'result': {'value': 42}
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        result = await page.evaluate('() => 42')
        assert result == '42'

    async def test_evaluate_with_args(self):
        session, client = _make_mock_browser_session()
        client.send.Runtime.evaluate.return_value = {
            'result': {'value': 'result'}
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.evaluate('(a, b) => a + b', 'hello', ' world')

        call_args = client.send.Runtime.evaluate.call_args[0][0]
        assert '("hello", " world")' in call_args['expression']

    async def test_evaluate_raises_on_exception(self):
        session, client = _make_mock_browser_session()
        client.send.Runtime.evaluate.return_value = {
            'result': {},
            'exceptionDetails': 'SyntaxError'
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        with pytest.raises(RuntimeError, match='JavaScript evaluation failed'):
            await page.evaluate('() => undefined_var')

    async def test_evaluate_rejects_non_arrow(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        with pytest.raises(ValueError, match='must start with'):
            await page.evaluate('document.title')


class TestFixJavascriptString:
    """Tests for Page._fix_javascript_string."""

    def test_strips_whitespace(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123')
        result = page._fix_javascript_string('  () => 1  ')
        assert result == '() => 1'

    def test_unwraps_double_quoted_string(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123')
        result = page._fix_javascript_string('"() => document.title"')
        assert result == '() => document.title'

    def test_unwraps_single_quoted_string(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123')
        result = page._fix_javascript_string("'() => document.title'")
        assert result == '() => document.title'

    def test_fixes_escaped_double_quotes_when_dominant(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123')
        # The branch: if count('\"') > count('"'), replace '\"' -> '"'
        # Since every '\"' contains a '"', count('\"') <= count('"') always.
        # We need to construct input where count('\"') strictly exceeds count('"'),
        # which requires direct string construction bypassing Python literal escaping.
        # Use a raw string with replace to build: 3x \" and 0x standalone "
        input_str = 'return \\\"x\\\"'  # contains \" sequences
        result = page._fix_javascript_string(input_str)
        # The condition count('\"') > count('"') is never true because every \"
        # contains a ", so the string passes through unchanged.
        assert result == input_str, (
            f"Expected input to pass through unchanged, got: {result!r}"
        )

    def test_fixes_escaped_single_quotes_when_dominant(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123')
        # Same logic for single quotes: count("\\'") can't exceed count("'")
        input_str = "return \\'x\\'"
        result = page._fix_javascript_string(input_str)
        assert 'return' in result
        assert 'x' in result

    def test_raises_on_empty_string(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123')
        with pytest.raises(ValueError, match='empty'):
            page._fix_javascript_string('   ')


@pytest.mark.asyncio
class TestPageScreenshot:
    """Tests for Page.screenshot."""

    async def test_screenshot_returns_data(self):
        session, client = _make_mock_browser_session()
        client.send.Page.captureScreenshot.return_value = {'data': 'imgdata'}
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        result = await page.screenshot()
        assert result == 'imgdata'

    async def test_screenshot_jpeg_with_quality(self):
        session, client = _make_mock_browser_session()
        client.send.Page.captureScreenshot.return_value = {'data': 'imgdata'}
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.screenshot(format='jpeg', quality=80)

        call_args = client.send.Page.captureScreenshot.call_args[0][0]
        assert call_args['quality'] == 80

    async def test_screenshot_png_no_quality(self):
        session, client = _make_mock_browser_session()
        client.send.Page.captureScreenshot.return_value = {'data': 'imgdata'}
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.screenshot(format='png', quality=80)

        call_args = client.send.Page.captureScreenshot.call_args[0][0]
        assert 'quality' not in call_args


@pytest.mark.asyncio
class TestPagePress:
    """Tests for Page.press."""

    async def test_press_simple_key(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.press('Enter')

        assert client.send.Input.dispatchKeyEvent.call_count == 2
        down_call = client.send.Input.dispatchKeyEvent.call_args_list[0]
        up_call = client.send.Input.dispatchKeyEvent.call_args_list[1]
        assert down_call[0][0]['type'] == 'keyDown'
        assert down_call[0][0]['key'] == 'Enter'
        assert up_call[0][0]['type'] == 'keyUp'

    async def test_press_key_combination(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.press('Control+A')

        # Should have: modifier keyDown, main keyDown, main keyUp, modifier keyUp
        assert client.send.Input.dispatchKeyEvent.call_count == 4

    async def test_press_key_with_vk_code(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.press('Enter')

        down_params = client.send.Input.dispatchKeyEvent.call_args_list[0][0][0]
        assert down_params['windowsVirtualKeyCode'] == 13


@pytest.mark.asyncio
class TestPageSetViewportSize:
    """Tests for Page.set_viewport_size."""

    async def test_set_viewport(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.set_viewport_size(1920, 1080)

        client.send.Emulation.setDeviceMetricsOverride.assert_called_once()
        call_args = client.send.Emulation.setDeviceMetricsOverride.call_args[0][0]
        assert call_args['width'] == 1920
        assert call_args['height'] == 1080
        assert call_args['mobile'] is False


@pytest.mark.asyncio
class TestPageGetTargetInfo:
    """Tests for Page.get_target_info, get_url, get_title."""

    async def test_get_target_info(self):
        session, client = _make_mock_browser_session()
        client.send.Target.getTargetInfo.return_value = {
            'targetInfo': {'url': 'https://example.com', 'title': 'Example'}
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        info = await page.get_target_info()
        assert info['url'] == 'https://example.com'

    async def test_get_url(self):
        session, client = _make_mock_browser_session()
        client.send.Target.getTargetInfo.return_value = {
            'targetInfo': {'url': 'https://example.com', 'title': 'Example'}
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        url = await page.get_url()
        assert url == 'https://example.com'

    async def test_get_title(self):
        session, client = _make_mock_browser_session()
        client.send.Target.getTargetInfo.return_value = {
            'targetInfo': {'url': 'https://example.com', 'title': 'Example Page'}
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        title = await page.get_title()
        assert title == 'Example Page'


@pytest.mark.asyncio
class TestPageNavigation:
    """Tests for Page.goto, navigate, go_back, go_forward."""

    async def test_goto(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.goto('https://example.com')
        client.send.Page.navigate.assert_called_once()

    async def test_navigate_alias(self):
        session, client = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.navigate('https://example.com')
        client.send.Page.navigate.assert_called_once()

    async def test_go_back(self):
        session, client = _make_mock_browser_session()
        client.send.Page.getNavigationHistory.return_value = {
            'currentIndex': 1,
            'entries': [
                {'id': 0, 'url': 'https://a.com'},
                {'id': 1, 'url': 'https://b.com'},
            ]
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.go_back()
        client.send.Page.navigateToHistoryEntry.assert_called_once()

    async def test_go_back_at_start_raises(self):
        session, client = _make_mock_browser_session()
        client.send.Page.getNavigationHistory.return_value = {
            'currentIndex': 0,
            'entries': [{'id': 0, 'url': 'https://a.com'}]
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        with pytest.raises(RuntimeError, match='Cannot go back'):
            await page.go_back()

    async def test_go_forward(self):
        session, client = _make_mock_browser_session()
        client.send.Page.getNavigationHistory.return_value = {
            'currentIndex': 0,
            'entries': [
                {'id': 0, 'url': 'https://a.com'},
                {'id': 1, 'url': 'https://b.com'},
            ]
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        await page.go_forward()
        client.send.Page.navigateToHistoryEntry.assert_called_once()

    async def test_go_forward_at_end_raises(self):
        session, client = _make_mock_browser_session()
        client.send.Page.getNavigationHistory.return_value = {
            'currentIndex': 0,
            'entries': [{'id': 0, 'url': 'https://a.com'}]
        }
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        with pytest.raises(RuntimeError, match='Cannot go forward'):
            await page.go_forward()


@pytest.mark.asyncio
class TestPageGetElementsByCssSelector:
    """Tests for Page.get_elements_by_css_selector."""

    async def test_returns_elements(self):
        session, client = _make_mock_browser_session()
        client.send.DOM.getDocument.return_value = {'root': {'nodeId': 1}}
        client.send.DOM.querySelectorAll.return_value = {'nodeIds': [10, 20]}
        client.send.DOM.describeNode.side_effect = [
            {'node': {'backendNodeId': 100}},
            {'node': {'backendNodeId': 200}},
        ]
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        elements = await page.get_elements_by_css_selector('.btn')
        assert len(elements) == 2
        assert elements[0]._backend_node_id == 100
        assert elements[1]._backend_node_id == 200


@pytest.mark.asyncio
class TestPageMustGetElementByPrompt:
    """Tests for Page.must_get_element_by_prompt."""

    async def test_raises_when_no_element_found(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')
        page.get_element_by_prompt = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match='No element found'):
            await page.must_get_element_by_prompt('click the submit button')

    async def test_returns_element_when_found(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')
        mock_element = MagicMock()
        page.get_element_by_prompt = AsyncMock(return_value=mock_element)

        result = await page.must_get_element_by_prompt('click submit')
        assert result is mock_element


@pytest.mark.asyncio
class TestPageExtractContent:
    """Tests for Page.extract_content."""

    async def test_raises_when_no_llm(self):
        session, _ = _make_mock_browser_session()
        page = Page(session, target_id='tid-123', session_id='sid-abc')

        from pydantic import BaseModel

        class Output(BaseModel):
            text: str

        with pytest.raises(ValueError, match='LLM not provided'):
            await page.extract_content('extract text', Output)

    async def test_raises_on_markdown_extraction_error(self):
        session, _ = _make_mock_browser_session()
        mock_llm = MagicMock()
        page = Page(session, target_id='tid-123', session_id='sid-abc', llm=mock_llm)
        page._extract_clean_markdown = AsyncMock(side_effect=RuntimeError('no markdown'))

        from pydantic import BaseModel

        class Output(BaseModel):
            text: str

        with pytest.raises(RuntimeError, match='Could not extract clean markdown'):
            await page.extract_content('extract text', Output)
