"""Tests for openbrowser.agent.gif and openbrowser.agent.graph modules.

Targets near-100% line coverage for both modules.
All external dependencies (PIL, langgraph, browser sessions, etc.) are mocked.
"""

import asyncio
import base64
import io
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal, TypedDict
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TINY_PNG_B64 = base64.b64encode(
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    b'\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02'
    b'\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx'
    b'\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
    b'\x00\x00\x00\x00IEND\xaeB`\x82'
).decode()

PLACEHOLDER_4PX = (
    'iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFElEQVR4nGP8//8/AwwwMSAB3BwAlm4DBfIlvvkAAAAASUVORK5CYII='
)


# ---------------------------------------------------------------------------
# Helpers -- all mocked images/fonts use real int values for arithmetic
# ---------------------------------------------------------------------------


def _make_pil_image_mock(width=800, height=600, mode='RGB'):
    """Create a mock PIL Image with real integer dimensions."""
    img = MagicMock()
    img.width = width
    img.height = height
    img.size = (width, height)
    img.mode = mode
    img.convert.return_value = img
    img.resize.return_value = img
    img.save = MagicMock()
    img.paste = MagicMock()
    return img


def _make_font_mock(size=40):
    """Create a mock ImageFont with real int size and proper getbbox."""
    font = MagicMock()
    font.size = size
    font.path = "/fake/font.ttf"
    font.getbbox.return_value = (0, 0, 100, 20)
    return font


def _make_draw_mock():
    """Create a mock ImageDraw.Draw with real tuple returns."""
    draw = MagicMock()
    draw.textbbox.return_value = (0, 0, 200, 30)
    draw.multiline_textbbox.return_value = (0, 0, 200, 60)
    return draw


def _make_browser_state_history(url="https://example.com", screenshot_b64=None):
    state = MagicMock()
    state.url = url
    state.title = "Example"
    state.tabs = []
    state.interacted_element = [None]
    state.screenshot_path = None
    state.get_screenshot.return_value = screenshot_b64
    return state


def _make_agent_history_item(screenshot_b64=None, url="https://example.com",
                             has_model_output=True, next_goal="click button"):
    item = MagicMock()
    item.state = _make_browser_state_history(url=url, screenshot_b64=screenshot_b64)
    if has_model_output:
        item.model_output = MagicMock()
        item.model_output.current_state = MagicMock()
        item.model_output.current_state.next_goal = next_goal
    else:
        item.model_output = None
    item.result = []
    return item


def _make_history_list(items=None, screenshots=None):
    history_list = MagicMock()
    history_list.history = items or []
    history_list.screenshots.return_value = screenshots or []
    return history_list


def _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw):
    """Return dict suitable for patch.dict('sys.modules', ...)."""
    pil_mod = MagicMock()
    pil_mod.Image = mock_Image
    pil_mod.ImageFont = mock_ImageFont
    pil_mod.ImageDraw = mock_ImageDraw
    return {
        'PIL': pil_mod,
        'PIL.Image': mock_Image,
        'PIL.ImageFont': mock_ImageFont,
        'PIL.ImageDraw': mock_ImageDraw,
    }


def _setup_pil_mocks():
    """Full PIL mock setup with real-int returning mocks."""
    mock_Image = MagicMock()
    mock_ImageFont = MagicMock()
    mock_ImageDraw = MagicMock()

    mock_img = _make_pil_image_mock()
    mock_Image.open.return_value = mock_img
    mock_Image.new.return_value = mock_img
    mock_Image.Resampling.LANCZOS = 1
    mock_Image.alpha_composite.return_value = mock_img

    mock_draw = _make_draw_mock()
    mock_ImageDraw.Draw.return_value = mock_draw

    return mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw


# ============================================================================
# gif.py -- decode_unicode_escapes_to_utf8
# ============================================================================


class TestDecodeUnicodeEscapesToUtf8:

    def test_no_escape_sequences(self):
        from openbrowser.agent.gif import decode_unicode_escapes_to_utf8
        assert decode_unicode_escapes_to_utf8("hello world") == "hello world"

    def test_with_unicode_escape(self):
        from openbrowser.agent.gif import decode_unicode_escapes_to_utf8
        result = decode_unicode_escapes_to_utf8(r"click \u0041 button")
        assert "A" in result

    def test_decode_failure_returns_original(self):
        from openbrowser.agent.gif import decode_unicode_escapes_to_utf8
        # Cyrillic char cannot encode to latin1 -> UnicodeEncodeError -> returns original
        text = r"\u0041" + "\u0410"
        result = decode_unicode_escapes_to_utf8(text)
        assert result == text, (
            f"On decode failure, expected original text {text!r} but got {result!r}"
        )

    def test_empty_string(self):
        from openbrowser.agent.gif import decode_unicode_escapes_to_utf8
        assert decode_unicode_escapes_to_utf8("") == ""


# ============================================================================
# gif.py -- _wrap_text
# ============================================================================


class TestWrapText:

    def test_short_text_no_wrap(self):
        from openbrowser.agent.gif import _wrap_text
        font = _make_font_mock()
        font.getbbox.return_value = (0, 0, 50, 20)
        assert _wrap_text("Hello", font, 500) == "Hello"

    def test_long_text_wraps(self):
        from openbrowser.agent.gif import _wrap_text
        font = _make_font_mock()

        def getbbox_side_effect(text):
            return (0, 0, len(text.split()) * 100, 20)

        font.getbbox.side_effect = getbbox_side_effect
        result = _wrap_text("one two three four", font, 150)
        assert len(result.split('\n')) > 1

    def test_single_long_word(self):
        from openbrowser.agent.gif import _wrap_text
        font = _make_font_mock()
        font.getbbox.return_value = (0, 0, 1000, 20)
        result = _wrap_text("superlongwordthatcannotfit", font, 200)
        assert "superlongwordthatcannotfit" in result

    def test_empty_text(self):
        from openbrowser.agent.gif import _wrap_text
        font = _make_font_mock()
        font.getbbox.return_value = (0, 0, 0, 0)
        assert _wrap_text("", font, 500) == ""

    def test_multiple_words_some_wrap(self):
        from openbrowser.agent.gif import _wrap_text
        font = _make_font_mock()

        def getbbox_side_effect(text):
            return (0, 0, len(text) * 10, 20)

        font.getbbox.side_effect = getbbox_side_effect
        result = _wrap_text("hello world foo bar", font, 120)
        assert len(result.split('\n')) >= 2


# ============================================================================
# gif.py -- create_history_gif
# ============================================================================


class TestCreateHistoryGif:

    def test_empty_history(self):
        from openbrowser.agent.gif import create_history_gif
        history = _make_history_list(items=[], screenshots=[])
        create_history_gif(task="test", history=history)

    def test_no_screenshots_after_pil_import(self):
        from openbrowser.agent.gif import create_history_gif
        mock_Image, mock_ImageFont, mock_ImageDraw, _, _ = _setup_pil_mocks()
        item = _make_agent_history_item(screenshot_b64=None)
        history = _make_history_list(items=[item], screenshots=[])
        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            create_history_gif(task="test", history=history)

    def test_all_placeholder_screenshots(self):
        from openbrowser.agent.gif import create_history_gif
        mock_Image, mock_ImageFont, mock_ImageDraw, _, _ = _setup_pil_mocks()
        item = _make_agent_history_item(screenshot_b64=PLACEHOLDER_4PX)
        history = _make_history_list(items=[item], screenshots=[PLACEHOLDER_4PX])
        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            create_history_gif(task="test", history=history)

    def _run_gif_test(self, *, task="test", items, screenshots,
                      show_task=False, show_goals=False, show_logo=False,
                      font_side_effect=None, is_new_tab=False,
                      extra_patches=None):
        """Reusable helper to run create_history_gif with full mocks."""
        from openbrowser.agent.gif import create_history_gif

        mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw = _setup_pil_mocks()

        if font_side_effect is not None:
            mock_ImageFont.truetype.side_effect = font_side_effect
        else:
            regular = _make_font_mock(40)
            title = _make_font_mock(56)
            goal = _make_font_mock(44)
            larger = _make_font_mock(56)
            fonts = [regular, title, goal]
            idx = [0]

            def truetype_fn(*a, **kw):
                if idx[0] < len(fonts):
                    f = fonts[idx[0]]
                    idx[0] += 1
                    return f
                return larger

            mock_ImageFont.truetype.side_effect = truetype_fn

        mock_ImageFont.load_default.return_value = _make_font_mock(10)

        history = _make_history_list(items=items, screenshots=screenshots)

        ctx_managers = [
            patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)),
            patch('openbrowser.utils.is_new_tab_page', return_value=is_new_tab),
        ]
        if extra_patches:
            ctx_managers.extend(extra_patches)

        # Enter all context managers
        entered = []
        for cm in ctx_managers:
            entered.append(cm.__enter__())
        try:
            create_history_gif(
                task=task,
                history=history,
                output_path="test.gif",
                show_goals=show_goals,
                show_task=show_task,
                show_logo=show_logo,
            )
        finally:
            for cm in reversed(ctx_managers):
                cm.__exit__(None, None, None)

        return mock_Image, mock_ImageFont, mock_ImageDraw

    def test_successful_gif_creation(self):
        """Full happy path with task frame and step frames."""
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64, url="https://example.com")
        item.state.get_screenshot.return_value = TINY_PNG_B64
        self._run_gif_test(
            task="Search for something",
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_task=True,
            show_goals=True,
        )

    def test_gif_no_task_no_goals(self):
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64, has_model_output=False)
        self._run_gif_test(
            task="Search",
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_task=False,
            show_goals=False,
        )

    def test_gif_skips_new_tab_pages(self):
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64, url="about:blank")
        self._run_gif_test(
            task="",
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_task=False,
            is_new_tab=True,
        )

    def test_gif_font_fallback_to_default(self):
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        self._run_gif_test(
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_task=False,
            show_goals=False,
            font_side_effect=OSError("No font found"),
        )

    def test_gif_with_logo(self):
        from openbrowser.agent.gif import create_history_gif

        mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw = _setup_pil_mocks()
        regular = _make_font_mock(40)
        title = _make_font_mock(56)
        goal = _make_font_mock(44)
        mock_ImageFont.truetype.side_effect = [regular, title, goal]

        mock_logo = _make_pil_image_mock(width=150, height=150, mode='RGBA')
        mock_Image.open.side_effect = [mock_logo, mock_img, mock_img, mock_img]

        mock_logo_path = MagicMock()
        mock_logo_path.is_file.return_value = True

        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        history = _make_history_list(items=[item], screenshots=[TINY_PNG_B64])

        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            with patch('importlib.resources.files') as mock_files:
                mock_static = MagicMock()
                mock_static.__truediv__ = MagicMock(return_value=mock_logo_path)
                mock_pkg = MagicMock()
                mock_pkg.__truediv__ = MagicMock(return_value=mock_static)
                mock_files.return_value = mock_pkg
                with patch('openbrowser.utils.is_new_tab_page', return_value=False):
                    create_history_gif(
                        task="test",
                        history=history,
                        show_logo=True,
                        show_task=False,
                        show_goals=True,
                    )

    def test_gif_logo_loading_fails(self):
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        self._run_gif_test(
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_logo=True,
            show_task=False,
            show_goals=False,
            extra_patches=[patch('importlib.resources.files', side_effect=Exception("cannot load"))],
        )

    def test_gif_none_screenshot_skipped(self):
        item1 = _make_agent_history_item(screenshot_b64=None)
        item2 = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        self._run_gif_test(
            items=[item1, item2],
            screenshots=[None, TINY_PNG_B64],
            show_task=False,
            show_goals=False,
        )

    def test_gif_task_frame_with_placeholder_state_screenshots(self):
        """show_task=True but state.get_screenshot returns placeholder -- skip task frame."""
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        item.state.get_screenshot.return_value = PLACEHOLDER_4PX
        self._run_gif_test(
            task="test task",
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_task=True,
            show_goals=False,
        )

    @patch('platform.system', return_value='Windows')
    def test_gif_windows_font_path(self, _mock_platform):
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        self._run_gif_test(
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_task=False,
            show_goals=False,
        )

    def test_gif_placeholder_in_screenshots_list(self):
        item1 = _make_agent_history_item(screenshot_b64=PLACEHOLDER_4PX)
        item2 = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        self._run_gif_test(
            items=[item1, item2],
            screenshots=[PLACEHOLDER_4PX, TINY_PNG_B64],
            show_task=False,
            show_goals=False,
        )

    def test_gif_logo_path_not_file(self):
        from openbrowser.agent.gif import create_history_gif

        mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw = _setup_pil_mocks()
        regular = _make_font_mock(40)
        title = _make_font_mock(56)
        goal = _make_font_mock(44)
        mock_ImageFont.truetype.side_effect = [regular, title, goal]

        mock_logo_path = MagicMock()
        mock_logo_path.is_file.return_value = False

        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        history = _make_history_list(items=[item], screenshots=[PLACEHOLDER_4PX])

        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            with patch('importlib.resources.files') as mock_files:
                mock_static = MagicMock()
                mock_static.__truediv__ = MagicMock(return_value=mock_logo_path)
                mock_pkg = MagicMock()
                mock_pkg.__truediv__ = MagicMock(return_value=mock_static)
                mock_files.return_value = mock_pkg
                create_history_gif(task="test", history=history, show_logo=True, show_task=False)

    def test_gif_logo_attribute_error(self):
        from openbrowser.agent.gif import create_history_gif

        mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw = _setup_pil_mocks()
        regular = _make_font_mock(40)
        title = _make_font_mock(56)
        goal = _make_font_mock(44)
        mock_ImageFont.truetype.side_effect = [regular, title, goal]

        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        history = _make_history_list(items=[item], screenshots=[PLACEHOLDER_4PX])

        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            with patch('importlib.resources.files', side_effect=AttributeError("no files")):
                create_history_gif(task="test", history=history, show_logo=True, show_task=False)

    def test_gif_logo_inner_type_error(self):
        """files() / 'static' raises TypeError -> caught by except (AttributeError, TypeError) at line 139-140."""
        from openbrowser.agent.gif import create_history_gif
        import importlib.resources

        mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw = _setup_pil_mocks()
        regular = _make_font_mock(40)
        title = _make_font_mock(56)
        goal = _make_font_mock(44)
        mock_ImageFont.truetype.side_effect = [regular, title, goal]

        # Need a non-placeholder screenshot so execution reaches logo loading code
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        history = _make_history_list(items=[item], screenshots=[TINY_PNG_B64])

        def fake_files(pkg_name):
            obj = MagicMock()
            obj.__truediv__ = MagicMock(side_effect=TypeError("bad path op"))
            return obj

        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            with patch.object(importlib.resources, 'files', side_effect=fake_files):
                with patch('openbrowser.utils.is_new_tab_page', return_value=False):
                    create_history_gif(task="test", history=history, show_logo=True, show_task=False)

    def test_gif_logo_path_not_file_debug_logged(self):
        """Logo path is_file returns False -> line 138 debug log."""
        from openbrowser.agent.gif import create_history_gif
        import importlib.resources

        mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw = _setup_pil_mocks()
        regular = _make_font_mock(40)
        title = _make_font_mock(56)
        goal = _make_font_mock(44)
        mock_ImageFont.truetype.side_effect = [regular, title, goal]

        mock_logo_path = MagicMock()
        mock_logo_path.is_file.return_value = False

        # Need a non-placeholder screenshot so execution reaches logo loading code
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64)
        history = _make_history_list(items=[item], screenshots=[TINY_PNG_B64])

        def fake_files(pkg_name):
            mock_static_dir = MagicMock()
            mock_static_dir.__truediv__ = MagicMock(return_value=mock_logo_path)
            mock_pkg = MagicMock()
            mock_pkg.__truediv__ = MagicMock(return_value=mock_static_dir)
            return mock_pkg

        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            with patch.object(importlib.resources, 'files', side_effect=fake_files):
                with patch('openbrowser.utils.is_new_tab_page', return_value=False):
                    create_history_gif(task="test", history=history, show_logo=True, show_task=False)

    def test_gif_with_task_and_real_screenshot_in_state(self):
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64, url="https://example.com")
        item.state.get_screenshot.return_value = TINY_PNG_B64
        self._run_gif_test(
            task="Test task",
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_task=True,
            show_goals=True,
        )

    def test_gif_no_images_produced(self):
        """All screenshots skipped (new tab) -> no images -> warning logged."""
        item = _make_agent_history_item(screenshot_b64=TINY_PNG_B64, url="about:blank")
        self._run_gif_test(
            task="",
            items=[item],
            screenshots=[TINY_PNG_B64],
            show_task=False,
            show_goals=False,
            is_new_tab=True,
        )


# ============================================================================
# gif.py -- _create_task_frame
# ============================================================================


class TestCreateTaskFrame:

    def _call(self, task="Search for flights", regular_font=None, title_font=None,
              logo=None, truetype_return=None, truetype_side_effect=None):
        from openbrowser.agent.gif import _create_task_frame

        mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw = _setup_pil_mocks()

        if regular_font is None:
            regular_font = _make_font_mock(40)
        if title_font is None:
            title_font = _make_font_mock(56)

        if truetype_side_effect is not None:
            mock_ImageFont.truetype.side_effect = truetype_side_effect
        elif truetype_return is not None:
            mock_ImageFont.truetype.return_value = truetype_return
        else:
            mock_ImageFont.truetype.return_value = _make_font_mock(56)

        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            result = _create_task_frame(
                task=task,
                first_screenshot=TINY_PNG_B64,
                title_font=title_font,
                regular_font=regular_font,
                logo=logo,
            )
        return result

    def test_basic_task_frame(self):
        assert self._call() is not None

    def test_task_frame_with_logo_rgba(self):
        logo = _make_pil_image_mock(width=100, height=50, mode='RGBA')
        assert self._call(logo=logo) is not None

    def test_task_frame_with_logo_non_rgba(self):
        logo = _make_pil_image_mock(width=100, height=50, mode='RGB')
        assert self._call(logo=logo) is not None

    def test_task_frame_long_text(self):
        assert self._call(task="x" * 250) is not None

    def test_task_frame_font_path_error(self):
        regular_font = _make_font_mock(40)
        del regular_font.path
        assert self._call(
            regular_font=regular_font,
            truetype_side_effect=AttributeError("no path"),
        ) is not None

    def test_task_frame_font_os_error(self):
        assert self._call(truetype_side_effect=OSError("font not found")) is not None


# ============================================================================
# gif.py -- _add_overlay_to_image
# ============================================================================


class TestAddOverlayToImage:

    def _call(self, *, step_number=1, goal_text="Click the button",
              margin=40, display_step=True, logo=None):
        from openbrowser.agent.gif import _add_overlay_to_image

        mock_Image, mock_ImageFont, mock_ImageDraw, mock_img, mock_draw = _setup_pil_mocks()
        title_font = _make_font_mock(56)
        regular_font = _make_font_mock(40)
        input_img = _make_pil_image_mock()

        with patch.dict('sys.modules', _pil_modules(mock_Image, mock_ImageFont, mock_ImageDraw)):
            result = _add_overlay_to_image(
                image=input_img,
                step_number=step_number,
                goal_text=goal_text,
                regular_font=regular_font,
                title_font=title_font,
                margin=margin,
                display_step=display_step,
                logo=logo,
            )
        return result

    def test_basic_overlay(self):
        assert self._call() is not None

    def test_overlay_no_step_display(self):
        """display_step=False should render goal text without step number."""
        result = self._call(display_step=False)
        assert result is not None

    def test_overlay_with_logo_rgba(self):
        logo = _make_pil_image_mock(width=100, height=50, mode='RGBA')
        assert self._call(logo=logo) is not None

    def test_overlay_with_logo_rgb(self):
        logo = _make_pil_image_mock(width=100, height=50, mode='RGB')
        assert self._call(logo=logo) is not None

    def test_overlay_with_unicode_goal(self):
        assert self._call(goal_text=r"Click \u0041 button") is not None

    def test_overlay_large_step_number(self):
        assert self._call(step_number=99) is not None


# ============================================================================
# graph.py helpers
# ============================================================================


def _make_mock_agent(has_downloads=False, max_failures=3,
                     final_response_after_failure=True):
    agent = MagicMock()
    agent.logger = MagicMock()
    agent.has_downloads_path = has_downloads

    settings = MagicMock()
    settings.max_failures = max_failures
    settings.final_response_after_failure = final_response_after_failure
    settings.use_vision = True
    settings.llm_timeout = 60
    agent.settings = settings

    state = MagicMock()
    state.last_model_output = None
    state.last_result = None
    state.consecutive_failures = 0
    state.n_steps = 0
    state.stopped = False
    state.paused = False
    agent.state = state

    agent.include_recent_events = True
    agent.sensitive_data = None
    agent.available_file_paths = None

    agent._check_stop_or_pause = AsyncMock()
    agent._check_and_update_downloads = AsyncMock()
    agent._update_action_models_for_page = AsyncMock()
    agent._force_done_after_last_step = AsyncMock()
    agent._force_done_after_failure = AsyncMock()
    agent._get_model_output_with_retry = AsyncMock()
    agent._handle_post_llm_processing = AsyncMock()
    agent._make_history_item = AsyncMock()
    agent.multi_act = AsyncMock()
    agent.save_file_system_state = MagicMock()
    agent.log_completion = AsyncMock()
    agent.register_done_callback = None

    agent.browser_session = MagicMock()
    agent.browser_session.get_browser_state_summary = AsyncMock()

    agent._message_manager = MagicMock()
    agent._message_manager.get_messages.return_value = []
    agent._message_manager.last_state_message_text = "test state"

    agent.tools = MagicMock()
    agent.tools.registry.get_prompt_description.return_value = "actions"

    history = MagicMock()
    history.is_done.return_value = False
    agent.history = history

    return agent


def _make_graph_builder(agent=None):
    """Create an AgentGraphBuilder with mocked StateGraph."""
    with patch('openbrowser.agent.graph.StateGraph') as mock_sg:
        mock_sg.return_value.compile.return_value = MagicMock()
        from openbrowser.agent.graph import AgentGraphBuilder
        return AgentGraphBuilder(agent or _make_mock_agent())


# ============================================================================
# graph.py -- GraphState
# ============================================================================


class TestGraphState:

    def test_graph_state_creation(self):
        from openbrowser.agent.graph import GraphState
        state: GraphState = {
            "step_number": 0, "max_steps": 100,
            "is_done": False, "consecutive_failures": 0,
        }
        assert state["step_number"] == 0
        assert state["max_steps"] == 100


# ============================================================================
# graph.py -- AgentGraphBuilder.__init__ and _build_graph
# ============================================================================


class TestAgentGraphBuilderInit:

    @patch('openbrowser.agent.graph.StateGraph')
    def test_init(self, mock_state_graph_cls):
        from openbrowser.agent.graph import AgentGraphBuilder
        mock_graph = MagicMock()
        mock_graph.compile.return_value = MagicMock()
        mock_state_graph_cls.return_value = mock_graph

        agent = _make_mock_agent()
        builder = AgentGraphBuilder(agent)

        assert builder.agent is agent
        assert builder._has_downloads == agent.has_downloads_path
        expected_max = agent.settings.max_failures + int(agent.settings.final_response_after_failure)
        assert builder._max_failures == expected_max
        mock_graph.add_node.assert_called_once_with("step", builder._step_node)
        mock_graph.add_edge.assert_called_once()
        mock_graph.add_conditional_edges.assert_called_once()
        mock_graph.compile.assert_called_once()


# ============================================================================
# graph.py -- _should_continue
# ============================================================================


class TestShouldContinue:

    def test_done_when_is_done(self):
        builder = _make_graph_builder()
        assert builder._should_continue({"is_done": True, "step_number": 1, "max_steps": 100}) == "done"

    def test_done_when_stopped(self):
        agent = _make_mock_agent()
        agent.state.stopped = True
        builder = _make_graph_builder(agent)
        assert builder._should_continue({"is_done": False, "step_number": 1, "max_steps": 100}) == "done"

    def test_done_when_paused(self):
        agent = _make_mock_agent()
        agent.state.paused = True
        builder = _make_graph_builder(agent)
        assert builder._should_continue({"is_done": False, "step_number": 1, "max_steps": 100}) == "done"

    def test_done_when_max_steps_reached(self):
        builder = _make_graph_builder()
        assert builder._should_continue({"is_done": False, "step_number": 100, "max_steps": 100}) == "done"

    def test_error_when_max_failures(self):
        agent = _make_mock_agent(max_failures=3, final_response_after_failure=True)
        builder = _make_graph_builder(agent)
        assert builder._should_continue({
            "is_done": False, "step_number": 1, "max_steps": 100, "consecutive_failures": 4
        }) == "error"

    def test_continue_normal(self):
        builder = _make_graph_builder()
        assert builder._should_continue({
            "is_done": False, "step_number": 1, "max_steps": 100, "consecutive_failures": 0
        }) == "continue"

    def test_continue_with_empty_state(self):
        builder = _make_graph_builder()
        assert builder._should_continue({}) == "continue"


# ============================================================================
# graph.py -- _step_node
# ============================================================================


class TestStepNode:

    @pytest.mark.asyncio
    async def test_success_with_done_action(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = [MagicMock()]
        agent._get_model_output_with_retry.return_value = mo

        r = MagicMock(); r.is_done = True; r.error = None; r.extracted_content = "done"
        agent.multi_act.return_value = [r]

        result = await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        assert result["step_number"] == 1
        assert result["is_done"] is True
        assert result["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_success_not_done_resets_failures(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = [MagicMock()]
        agent._get_model_output_with_retry.return_value = mo

        r = MagicMock(); r.is_done = False; r.error = None
        agent.multi_act.return_value = [r]

        result = await builder._step_node({"step_number": 5, "max_steps": 100, "is_done": False, "consecutive_failures": 2})
        assert result["step_number"] == 6
        assert result["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_single_action_error_increments_failures(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = [MagicMock()]
        agent._get_model_output_with_retry.return_value = mo

        r = MagicMock(); r.is_done = False; r.error = "oops"
        agent.multi_act.return_value = [r]

        result = await builder._step_node({"step_number": 3, "max_steps": 100, "is_done": False, "consecutive_failures": 1})
        assert result["consecutive_failures"] == 2

    @pytest.mark.asyncio
    async def test_empty_action_increments_failures(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = []
        agent._get_model_output_with_retry.return_value = mo

        result = await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        assert result["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_none_model_output_increments_failures(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs
        agent._get_model_output_with_retry.return_value = None

        result = await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        assert result["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_with_downloads_path(self):
        agent = _make_mock_agent(has_downloads=True)
        builder = _make_graph_builder(agent)

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = [MagicMock()]
        agent._get_model_output_with_retry.return_value = mo

        r = MagicMock(); r.is_done = False; r.error = None
        agent.multi_act.return_value = [r]

        await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        assert agent._check_and_update_downloads.call_count >= 2

    @pytest.mark.asyncio
    async def test_interrupted_error(self):
        builder = _make_graph_builder()
        builder.agent._check_stop_or_pause.side_effect = InterruptedError("stopped")

        result = await builder._step_node({"step_number": 2, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        assert result["is_done"] is True
        assert result["step_number"] == 3

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        async def wait_for_timeout(coro, **kwargs):
            """Await (and thus consume) the inner coroutine, then raise TimeoutError."""
            try:
                await coro
            except Exception:
                pass
            raise asyncio.TimeoutError()

        with patch('asyncio.wait_for', side_effect=wait_for_timeout):
            result = await builder._step_node({"step_number": 1, "max_steps": 100, "is_done": False, "consecutive_failures": 0})

        assert result["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        builder = _make_graph_builder()
        builder.agent._check_stop_or_pause.side_effect = RuntimeError("boom")

        result = await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 1})
        assert result["consecutive_failures"] == 2

    @pytest.mark.asyncio
    async def test_no_browser_state(self):
        builder = _make_graph_builder()
        agent = builder.agent
        agent.browser_session.get_browser_state_summary.return_value = None

        mo = MagicMock(); mo.action = [MagicMock()]
        agent._get_model_output_with_retry = AsyncMock(return_value=mo)

        r = MagicMock(); r.is_done = False; r.error = None
        agent.multi_act.return_value = [r]

        result = await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})

        assert result["step_number"] == 1
        agent._make_history_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_results_error_not_single(self):
        """len(results) > 1 with error should NOT increment failures."""
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = [MagicMock(), MagicMock()]
        agent._get_model_output_with_retry.return_value = mo

        r1 = MagicMock(); r1.is_done = False; r1.error = "err"
        r2 = MagicMock(); r2.is_done = False; r2.error = None
        agent.multi_act.return_value = [r1, r2]

        result = await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        assert result["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_history_item_and_save(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = [MagicMock()]
        agent._get_model_output_with_retry.return_value = mo

        r = MagicMock(); r.is_done = False; r.error = None
        agent.multi_act.return_value = [r]

        await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        agent._make_history_item.assert_called_once()
        agent.save_file_system_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_defaults(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = [MagicMock()]
        agent._get_model_output_with_retry.return_value = mo

        r = MagicMock(); r.is_done = False; r.error = None
        agent.multi_act.return_value = [r]

        result = await builder._step_node({})
        assert result["step_number"] == 1

    @pytest.mark.asyncio
    async def test_final_result_logged(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = [MagicMock()]
        agent._get_model_output_with_retry.return_value = mo

        r = MagicMock(); r.is_done = True; r.error = None; r.extracted_content = "Final answer"
        agent.multi_act.return_value = [r]

        await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        agent.logger.info.assert_any_call('\n Final Result:\nFinal answer\n\n')

    @pytest.mark.asyncio
    async def test_no_history_when_empty_results(self):
        builder = _make_graph_builder()
        agent = builder.agent

        bs = MagicMock(); bs.url = "https://example.com"
        agent.browser_session.get_browser_state_summary.return_value = bs

        mo = MagicMock(); mo.action = []
        agent._get_model_output_with_retry.return_value = mo

        await builder._step_node({"step_number": 0, "max_steps": 100, "is_done": False, "consecutive_failures": 0})
        agent._make_history_item.assert_not_called()


# ============================================================================
# graph.py -- run
# ============================================================================


class TestGraphBuilderRun:

    @pytest.mark.asyncio
    async def test_run_basic(self):
        with patch('openbrowser.agent.graph.StateGraph') as mock_sg:
            compiled = AsyncMock(); compiled.ainvoke = AsyncMock()
            mock_sg.return_value.compile.return_value = compiled
            agent = _make_mock_agent()
            from openbrowser.agent.graph import AgentGraphBuilder
            builder = AgentGraphBuilder(agent)
            result = await builder.run(max_steps=50)
            compiled.ainvoke.assert_called_once()
            assert result is agent.history

    @pytest.mark.asyncio
    async def test_run_done_sync_callback(self):
        with patch('openbrowser.agent.graph.StateGraph') as mock_sg:
            compiled = AsyncMock(); compiled.ainvoke = AsyncMock()
            mock_sg.return_value.compile.return_value = compiled
            agent = _make_mock_agent()
            agent.history.is_done.return_value = True
            sync_cb = MagicMock()
            agent.register_done_callback = sync_cb
            from openbrowser.agent.graph import AgentGraphBuilder
            builder = AgentGraphBuilder(agent)
            await builder.run(max_steps=100)
            agent.log_completion.assert_called_once()
            sync_cb.assert_called_once_with(agent.history)

    @pytest.mark.asyncio
    async def test_run_done_async_callback(self):
        with patch('openbrowser.agent.graph.StateGraph') as mock_sg:
            compiled = AsyncMock(); compiled.ainvoke = AsyncMock()
            mock_sg.return_value.compile.return_value = compiled
            agent = _make_mock_agent()
            agent.history.is_done.return_value = True
            async_cb = AsyncMock()
            agent.register_done_callback = async_cb
            from openbrowser.agent.graph import AgentGraphBuilder
            builder = AgentGraphBuilder(agent)
            await builder.run(max_steps=100)
            agent.log_completion.assert_called_once()
            async_cb.assert_called_once_with(agent.history)

    @pytest.mark.asyncio
    async def test_run_not_done(self):
        with patch('openbrowser.agent.graph.StateGraph') as mock_sg:
            compiled = AsyncMock(); compiled.ainvoke = AsyncMock()
            mock_sg.return_value.compile.return_value = compiled
            agent = _make_mock_agent()
            agent.history.is_done.return_value = False
            from openbrowser.agent.graph import AgentGraphBuilder
            builder = AgentGraphBuilder(agent)
            await builder.run(max_steps=100)
            agent.log_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_done_no_callback(self):
        with patch('openbrowser.agent.graph.StateGraph') as mock_sg:
            compiled = AsyncMock(); compiled.ainvoke = AsyncMock()
            mock_sg.return_value.compile.return_value = compiled
            agent = _make_mock_agent()
            agent.history.is_done.return_value = True
            agent.register_done_callback = None
            from openbrowser.agent.graph import AgentGraphBuilder
            builder = AgentGraphBuilder(agent)
            await builder.run(max_steps=100)
            agent.log_completion.assert_called_once()


# ============================================================================
# graph.py -- create_agent_graph
# ============================================================================


class TestCreateAgentGraph:

    @patch('openbrowser.agent.graph.StateGraph')
    def test_creates_builder(self, mock_sg):
        from openbrowser.agent.graph import AgentGraphBuilder, create_agent_graph
        mock_sg.return_value.compile.return_value = MagicMock()
        agent = _make_mock_agent()
        result = create_agent_graph(agent)
        assert isinstance(result, AgentGraphBuilder)
        assert result.agent is agent
