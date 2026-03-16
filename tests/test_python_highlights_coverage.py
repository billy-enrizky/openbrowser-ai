"""Comprehensive tests for openbrowser.browser.python_highlights module.

Covers ALL code paths in python_highlights.py (lines 7-548):
- get_cross_platform_font() - font loading with caching
- cleanup_font_cache() - cache cleanup
- get_element_color() - element color selection
- should_show_index_overlay() - index overlay display logic
- draw_enhanced_bounding_box_with_text() - enhanced box drawing
- draw_bounding_box_with_text() - simple box drawing
- process_element_highlight() - single element processing
- create_highlighted_screenshot() - main screenshot processing
- get_viewport_info_from_cdp() - CDP viewport info retrieval
- create_highlighted_screenshot_async() - async wrapper
"""

import asyncio
import base64
import io
import logging
import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper fixtures and mocks
# ---------------------------------------------------------------------------


@dataclass
class MockDOMRect:
    """Mock DOMRect for element positions."""

    x: float
    y: float
    width: float
    height: float


class MockEnhancedDOMTreeNode:
    """Mock EnhancedDOMTreeNode for testing."""

    def __init__(
        self,
        tag_name: str = "div",
        attributes: dict | None = None,
        absolute_position: MockDOMRect | None = None,
        backend_node_id: int | None = None,
    ):
        self.tag_name = tag_name
        self.attributes = attributes or {}
        self.absolute_position = absolute_position
        self.backend_node_id = backend_node_id

    def get_meaningful_text_for_llm(self) -> str:
        """Return meaningful text for the element."""
        return self.attributes.get("aria-label", "")


@pytest.fixture
def mock_image():
    """Create a mock PIL Image."""
    img = MagicMock()
    img.size = (1280, 720)
    img.convert.return_value = img
    return img


@pytest.fixture
def mock_draw():
    """Create a mock ImageDraw."""
    draw = MagicMock()
    draw.textbbox.return_value = (0, 0, 20, 12)
    draw.line = MagicMock()
    draw.rectangle = MagicMock()
    draw.text = MagicMock()
    return draw


# ---------------------------------------------------------------------------
# ELEMENT_COLORS and ELEMENT_TYPE_MAP (lines 73-89)
# ---------------------------------------------------------------------------


class TestElementConstants:
    """Test element color and type constants."""

    def test_element_colors_defined(self):
        """Test ELEMENT_COLORS has expected keys."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS

        assert "button" in ELEMENT_COLORS
        assert "input" in ELEMENT_COLORS
        assert "select" in ELEMENT_COLORS
        assert "a" in ELEMENT_COLORS
        assert "textarea" in ELEMENT_COLORS
        assert "default" in ELEMENT_COLORS

    def test_element_type_map_defined(self):
        """Test ELEMENT_TYPE_MAP has expected mappings."""
        from openbrowser.browser.python_highlights import ELEMENT_TYPE_MAP

        assert ELEMENT_TYPE_MAP["button"] == "button"
        assert ELEMENT_TYPE_MAP["input"] == "input"
        assert ELEMENT_TYPE_MAP["a"] == "a"


# ---------------------------------------------------------------------------
# get_cross_platform_font (lines 36-63)
# ---------------------------------------------------------------------------


class TestGetCrossPlatformFont:
    """Test get_cross_platform_font function."""

    def test_font_loading_with_caching(self):
        """Test that font loading uses cache."""
        from openbrowser.browser.python_highlights import _FONT_CACHE, get_cross_platform_font

        # Clear the cache first
        _FONT_CACHE.clear()

        mock_font = MagicMock()
        with patch("openbrowser.browser.python_highlights.ImageFont.truetype", return_value=mock_font):
            result = get_cross_platform_font(12)
            assert result == mock_font

            # Second call should use cache
            result2 = get_cross_platform_font(12)
            assert result2 == mock_font

        # Clean up
        _FONT_CACHE.clear()

    def test_font_loading_cache_hit(self):
        """Test cache hit returns cached font."""
        from openbrowser.browser.python_highlights import _FONT_CACHE, get_cross_platform_font

        mock_font = MagicMock()
        _FONT_CACHE[("system_font", 14)] = mock_font

        result = get_cross_platform_font(14)
        assert result == mock_font

        # Clean up
        _FONT_CACHE.clear()

    def test_font_loading_no_system_fonts(self):
        """Test font loading when no system fonts are available."""
        from openbrowser.browser.python_highlights import _FONT_CACHE, get_cross_platform_font

        _FONT_CACHE.clear()

        with patch("openbrowser.browser.python_highlights.ImageFont.truetype", side_effect=OSError("not found")):
            result = get_cross_platform_font(16)
            assert result is None

            # Verify None is cached
            assert ("system_font", 16) in _FONT_CACHE
            assert _FONT_CACHE[("system_font", 16)] is None

        _FONT_CACHE.clear()

    def test_font_loading_tries_multiple_paths(self):
        """Test that font loading tries multiple font paths."""
        from openbrowser.browser.python_highlights import _FONT_CACHE, _FONT_PATHS, get_cross_platform_font

        _FONT_CACHE.clear()

        call_count = 0
        mock_font = MagicMock()

        def side_effect(path, size):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("not found")
            return mock_font

        with patch("openbrowser.browser.python_highlights.ImageFont.truetype", side_effect=side_effect):
            result = get_cross_platform_font(10)
            assert result == mock_font
            assert call_count >= 3

        _FONT_CACHE.clear()

    def test_font_cache_none_prevents_repeated_loading(self):
        """Test that cached None prevents repeated font loading attempts."""
        from openbrowser.browser.python_highlights import _FONT_CACHE, get_cross_platform_font

        _FONT_CACHE[("system_font", 18)] = None

        with patch("openbrowser.browser.python_highlights.ImageFont.truetype") as mock_truetype:
            result = get_cross_platform_font(18)
            assert result is None
            mock_truetype.assert_not_called()

        _FONT_CACHE.clear()


# ---------------------------------------------------------------------------
# cleanup_font_cache (lines 67-69)
# ---------------------------------------------------------------------------


class TestCleanupFontCache:
    """Test cleanup_font_cache function."""

    def test_cleanup_clears_cache(self):
        """Test that cleanup clears the font cache."""
        from openbrowser.browser.python_highlights import _FONT_CACHE, cleanup_font_cache

        _FONT_CACHE[("system_font", 12)] = MagicMock()
        _FONT_CACHE[("system_font", 14)] = MagicMock()

        cleanup_font_cache()

        assert len(_FONT_CACHE) == 0

    def test_cleanup_empty_cache(self):
        """Test cleanup on already empty cache."""
        from openbrowser.browser.python_highlights import _FONT_CACHE, cleanup_font_cache

        _FONT_CACHE.clear()
        cleanup_font_cache()
        assert len(_FONT_CACHE) == 0


# ---------------------------------------------------------------------------
# get_element_color (lines 92-100)
# ---------------------------------------------------------------------------


class TestGetElementColor:
    """Test get_element_color function."""

    def test_button_color(self):
        """Test button element returns red color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("button") == ELEMENT_COLORS["button"]

    def test_input_color(self):
        """Test input element returns teal color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("input") == ELEMENT_COLORS["input"]

    def test_input_submit_type_returns_button_color(self):
        """Test input with type=submit returns button color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("input", "submit") == ELEMENT_COLORS["button"]

    def test_input_button_type_returns_button_color(self):
        """Test input with type=button returns button color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("input", "button") == ELEMENT_COLORS["button"]

    def test_input_text_type_returns_input_color(self):
        """Test input with type=text returns input color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("input", "text") == ELEMENT_COLORS["input"]

    def test_link_color(self):
        """Test anchor element returns green color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("a") == ELEMENT_COLORS["a"]

    def test_select_color(self):
        """Test select element returns blue color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("select") == ELEMENT_COLORS["select"]

    def test_textarea_color(self):
        """Test textarea element returns orange color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("textarea") == ELEMENT_COLORS["textarea"]

    def test_unknown_element_default_color(self):
        """Test unknown element returns default color."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("span") == ELEMENT_COLORS["default"]
        assert get_element_color("div") == ELEMENT_COLORS["default"]

    def test_case_insensitive_tag(self):
        """Test that tag names are case insensitive."""
        from openbrowser.browser.python_highlights import ELEMENT_COLORS, get_element_color

        assert get_element_color("BUTTON") == ELEMENT_COLORS["button"]
        assert get_element_color("Input") == ELEMENT_COLORS["input"]


# ---------------------------------------------------------------------------
# should_show_index_overlay (lines 103-105)
# ---------------------------------------------------------------------------


class TestShouldShowIndexOverlay:
    """Test should_show_index_overlay function."""

    def test_with_backend_node_id(self):
        """Test returns True when backend_node_id is not None."""
        from openbrowser.browser.python_highlights import should_show_index_overlay

        assert should_show_index_overlay(123) is True

    def test_without_backend_node_id(self):
        """Test returns False when backend_node_id is None."""
        from openbrowser.browser.python_highlights import should_show_index_overlay

        assert should_show_index_overlay(None) is False

    def test_with_zero_backend_node_id(self):
        """Test returns True when backend_node_id is 0 (not None)."""
        from openbrowser.browser.python_highlights import should_show_index_overlay

        assert should_show_index_overlay(0) is True


# ---------------------------------------------------------------------------
# draw_enhanced_bounding_box_with_text (lines 108-231)
# ---------------------------------------------------------------------------


class TestDrawEnhancedBoundingBoxWithText:
    """Test draw_enhanced_bounding_box_with_text function."""

    def test_basic_box_drawing(self, mock_draw):
        """Test basic bounding box drawing without text."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        draw_enhanced_bounding_box_with_text(
            mock_draw, (10, 20, 200, 100), "#FF0000"
        )

        # Should draw dashed lines
        assert mock_draw.line.call_count > 0

    def test_box_with_text_large_element(self, mock_draw):
        """Test box drawing with text on a large element."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        mock_font = MagicMock()

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=mock_font):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (10, 20, 300, 200),
                "#FF0000",
                text="42",
                font=mock_font,
                image_size=(1280, 720),
            )

        mock_draw.rectangle.assert_called()
        mock_draw.text.assert_called()

    def test_box_with_text_small_element(self, mock_draw):
        """Test box drawing with text on a small element (< 60px or < 30px)."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        mock_font = MagicMock()

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=mock_font):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (10, 20, 40, 35),  # Small element
                "#FF0000",
                text="5",
                font=mock_font,
                image_size=(1280, 720),
            )

        mock_draw.rectangle.assert_called()

    def test_box_with_text_no_font_found(self, mock_draw):
        """Test box drawing when no system font is found."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=None):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (10, 20, 200, 100),
                "#FF0000",
                text="10",
                font=None,
                image_size=(1280, 720),
            )

    def test_box_with_text_no_font_at_all(self, mock_draw):
        """Test box drawing when both system font and fallback are None."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=None):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (10, 20, 200, 100),
                "#FF0000",
                text="10",
                font=None,
                image_size=(1280, 720),
            )

    def test_box_bounds_correction_left(self, mock_draw):
        """Test bounds correction when box goes off left edge."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        mock_font = MagicMock()
        # Make text very wide so bg_x1 goes negative
        mock_draw.textbbox.return_value = (0, 0, 500, 12)

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=mock_font):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (0, 50, 50, 100),  # Small element at left edge
                "#FF0000",
                text="999",
                font=mock_font,
                image_size=(1280, 720),
            )

    def test_box_bounds_correction_top(self, mock_draw):
        """Test bounds correction when box goes off top edge."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        mock_font = MagicMock()

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=mock_font):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (100, 0, 130, 15),  # Small element near top
                "#FF0000",
                text="1",
                font=mock_font,
                image_size=(1280, 720),
            )

    def test_box_bounds_correction_right(self, mock_draw):
        """Test bounds correction when box goes off right edge."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        mock_font = MagicMock()
        mock_draw.textbbox.return_value = (0, 0, 100, 12)

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=mock_font):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (1200, 50, 1280, 100),  # Element at right edge
                "#FF0000",
                text="999",
                font=mock_font,
                image_size=(1280, 720),
            )

    def test_box_bounds_correction_bottom(self, mock_draw):
        """Test bounds correction when box goes off bottom edge."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        mock_font = MagicMock()

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=mock_font):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (100, 700, 200, 720),  # Element at bottom edge
                "#FF0000",
                text="1",
                font=mock_font,
                image_size=(1280, 720),
            )

    def test_box_text_exception_handled(self, mock_draw):
        """Test that exceptions in text drawing are handled."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        mock_draw.textbbox.side_effect = Exception("PIL error")

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=None):
            # Should not raise
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (10, 20, 200, 100),
                "#FF0000",
                text="10",
                font=None,
                image_size=(1280, 720),
            )

    def test_box_with_device_pixel_ratio(self, mock_draw):
        """Test box drawing with non-default device pixel ratio."""
        from openbrowser.browser.python_highlights import draw_enhanced_bounding_box_with_text

        mock_font = MagicMock()

        with patch("openbrowser.browser.python_highlights.get_cross_platform_font", return_value=mock_font):
            draw_enhanced_bounding_box_with_text(
                mock_draw,
                (10, 20, 200, 100),
                "#FF0000",
                text="42",
                font=mock_font,
                image_size=(2560, 1440),
                device_pixel_ratio=2.0,
            )


# ---------------------------------------------------------------------------
# draw_bounding_box_with_text (lines 234-337)
# ---------------------------------------------------------------------------


class TestDrawBoundingBoxWithText:
    """Test draw_bounding_box_with_text function."""

    def test_basic_dashed_box(self, mock_draw):
        """Test basic dashed bounding box drawing."""
        from openbrowser.browser.python_highlights import draw_bounding_box_with_text

        draw_bounding_box_with_text(mock_draw, (10, 20, 200, 100), "#FF0000")

        assert mock_draw.line.call_count > 0

    def test_box_with_text_small_element(self, mock_draw):
        """Test box with text on small element (size_ratio < 4)."""
        from openbrowser.browser.python_highlights import draw_bounding_box_with_text

        mock_font = MagicMock()
        mock_draw.textbbox.return_value = (0, 0, 20, 12)

        draw_bounding_box_with_text(
            mock_draw, (10, 20, 30, 30), "#FF0000", text="1", font=mock_font
        )

        mock_draw.rectangle.assert_called()
        mock_draw.text.assert_called()

    def test_box_with_text_medium_element(self, mock_draw):
        """Test box with text on medium element (size_ratio 4-16)."""
        from openbrowser.browser.python_highlights import draw_bounding_box_with_text

        mock_font = MagicMock()
        mock_draw.textbbox.return_value = (0, 0, 20, 12)

        draw_bounding_box_with_text(
            mock_draw, (10, 20, 100, 80), "#FF0000", text="5", font=mock_font
        )

        mock_draw.rectangle.assert_called()

    def test_box_with_text_large_element(self, mock_draw):
        """Test box with text on large element (size_ratio >= 16)."""
        from openbrowser.browser.python_highlights import draw_bounding_box_with_text

        mock_font = MagicMock()
        mock_draw.textbbox.return_value = (0, 0, 10, 8)

        draw_bounding_box_with_text(
            mock_draw, (10, 20, 500, 400), "#FF0000", text="3", font=mock_font
        )

        mock_draw.rectangle.assert_called()

    def test_box_with_text_no_font(self, mock_draw):
        """Test box with text but no font (uses default)."""
        from openbrowser.browser.python_highlights import draw_bounding_box_with_text

        mock_draw.textbbox.return_value = (0, 0, 20, 12)

        draw_bounding_box_with_text(
            mock_draw, (10, 20, 200, 100), "#FF0000", text="7", font=None
        )

        mock_draw.text.assert_called()

    def test_box_text_exception_handled(self, mock_draw):
        """Test that text drawing exceptions are caught."""
        from openbrowser.browser.python_highlights import draw_bounding_box_with_text

        mock_draw.textbbox.side_effect = Exception("PIL error")

        # Should not raise
        draw_bounding_box_with_text(
            mock_draw, (10, 20, 200, 100), "#FF0000", text="1", font=None
        )

    def test_box_no_text(self, mock_draw):
        """Test box drawing without text."""
        from openbrowser.browser.python_highlights import draw_bounding_box_with_text

        draw_bounding_box_with_text(mock_draw, (10, 20, 200, 100), "#FF0000")

        # rectangle and text should NOT be called (no text overlay)
        mock_draw.rectangle.assert_not_called()
        mock_draw.text.assert_not_called()


# ---------------------------------------------------------------------------
# process_element_highlight (lines 340-404)
# ---------------------------------------------------------------------------


class TestProcessElementHighlight:
    """Test process_element_highlight function."""

    def test_element_with_position(self, mock_draw):
        """Test processing element with valid position."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            tag_name="button",
            absolute_position=MockDOMRect(10, 20, 100, 50),
            backend_node_id=42,
            attributes={"aria-label": "AB"},
        )

        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )

    def test_element_without_position(self, mock_draw):
        """Test processing element without position (early return)."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            absolute_position=None,
            backend_node_id=1,
        )

        # Should return early without drawing
        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )
        mock_draw.line.assert_not_called()

    def test_element_too_small(self, mock_draw):
        """Test processing element that is too small (< 2px)."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            absolute_position=MockDOMRect(10, 20, 0.5, 0.5),
            backend_node_id=1,
        )

        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )

    def test_element_with_device_pixel_ratio(self, mock_draw):
        """Test coordinate scaling with device pixel ratio."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            tag_name="a",
            absolute_position=MockDOMRect(10, 20, 100, 50),
            backend_node_id=5,
            attributes={"aria-label": "Link"},
        )

        process_element_highlight(
            1, element, mock_draw, 2.0, None, True, (2560, 1440)
        )

    def test_element_filter_highlight_ids_short_text(self, mock_draw):
        """Test filter_highlight_ids with short meaningful text (< 3 chars)."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            tag_name="button",
            absolute_position=MockDOMRect(10, 20, 100, 50),
            backend_node_id=42,
            attributes={"aria-label": "AB"},  # 2 chars < 3
        )

        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )

    def test_element_filter_highlight_ids_long_text(self, mock_draw):
        """Test filter_highlight_ids with long meaningful text (>= 3 chars)."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            tag_name="button",
            absolute_position=MockDOMRect(10, 20, 100, 50),
            backend_node_id=42,
            attributes={"aria-label": "Click Me"},  # > 3 chars
        )

        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )

    def test_element_filter_disabled(self, mock_draw):
        """Test with filter_highlight_ids disabled."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            tag_name="button",
            absolute_position=MockDOMRect(10, 20, 100, 50),
            backend_node_id=42,
            attributes={"aria-label": "Long text here"},
        )

        process_element_highlight(
            1, element, mock_draw, 1.0, None, False, (1280, 720)
        )

    def test_element_no_backend_node_id(self, mock_draw):
        """Test element without backend_node_id."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            tag_name="div",
            absolute_position=MockDOMRect(10, 20, 100, 50),
            backend_node_id=None,
        )

        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )

    def test_element_input_with_type_attribute(self, mock_draw):
        """Test input element with type attribute."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            tag_name="input",
            absolute_position=MockDOMRect(10, 20, 200, 30),
            backend_node_id=15,
            attributes={"type": "submit", "aria-label": "Go"},
        )

        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )

    def test_element_bounds_clamping(self, mock_draw):
        """Test that coordinates are clamped to image bounds."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MockEnhancedDOMTreeNode(
            tag_name="div",
            absolute_position=MockDOMRect(-10, -5, 2000, 1500),
            backend_node_id=1,
            attributes={"aria-label": "XY"},
        )

        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )

    def test_element_exception_handled(self, mock_draw):
        """Test that exceptions in element processing are caught."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MagicMock()
        element.absolute_position = MagicMock()
        element.absolute_position.x = "not_a_number"  # Will cause TypeError

        # Should not raise
        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )

    def test_element_no_tag_name_attribute(self, mock_draw):
        """Test element without tag_name attribute uses default."""
        from openbrowser.browser.python_highlights import process_element_highlight

        element = MagicMock(spec=[])
        element.absolute_position = MockDOMRect(10, 20, 100, 50)
        element.backend_node_id = 1

        # hasattr(element, 'tag_name') will be False
        process_element_highlight(
            1, element, mock_draw, 1.0, None, True, (1280, 720)
        )


# ---------------------------------------------------------------------------
# create_highlighted_screenshot (lines 407-467)
# ---------------------------------------------------------------------------


class TestCreateHighlightedScreenshot:
    """Test create_highlighted_screenshot async function."""

    @pytest.mark.asyncio
    async def test_basic_screenshot_highlighting(self):
        """Test basic screenshot highlighting with elements."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot

        # Create a small test image
        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        element = MockEnhancedDOMTreeNode(
            tag_name="button",
            absolute_position=MockDOMRect(10, 10, 30, 20),
            backend_node_id=1,
            attributes={"aria-label": "OK"},
        )

        selector_map = {1: element}

        result = await create_highlighted_screenshot(screenshot_b64, selector_map)
        assert isinstance(result, str)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_empty_selector_map(self):
        """Test with empty selector map."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot

        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = await create_highlighted_screenshot(screenshot_b64, {})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_invalid_screenshot_returns_original(self):
        """Test that invalid screenshot data returns original."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot

        invalid_b64 = base64.b64encode(b"not a valid image").decode("utf-8")

        result = await create_highlighted_screenshot(invalid_b64, {})
        assert result == invalid_b64

    @pytest.mark.asyncio
    async def test_with_device_pixel_ratio(self):
        """Test highlighting with custom device pixel ratio."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot

        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (200, 200), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = await create_highlighted_screenshot(
            screenshot_b64, {}, device_pixel_ratio=2.0
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# get_viewport_info_from_cdp (lines 470-498)
# ---------------------------------------------------------------------------


class TestGetViewportInfoFromCDP:
    """Test get_viewport_info_from_cdp function."""

    @pytest.mark.asyncio
    async def test_successful_viewport_info(self):
        """Test successful viewport info retrieval."""
        from openbrowser.browser.python_highlights import get_viewport_info_from_cdp

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={
            "visualViewport": {"clientWidth": 2560},
            "cssVisualViewport": {"clientWidth": 1280, "pageX": 0, "pageY": 100},
            "cssLayoutViewport": {"clientWidth": 1280},
        })

        ratio, x, y = await get_viewport_info_from_cdp(mock_cdp)

        assert ratio == 2.0
        assert x == 0
        assert y == 100

    @pytest.mark.asyncio
    async def test_viewport_info_exception(self):
        """Test viewport info retrieval when exception occurs."""
        from openbrowser.browser.python_highlights import get_viewport_info_from_cdp

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            side_effect=Exception("CDP error")
        )

        ratio, x, y = await get_viewport_info_from_cdp(mock_cdp)

        assert ratio == 1.0
        assert x == 0
        assert y == 0

    @pytest.mark.asyncio
    async def test_viewport_zero_css_width(self):
        """Test viewport info with zero CSS width (avoid division by zero)."""
        from openbrowser.browser.python_highlights import get_viewport_info_from_cdp

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={
            "visualViewport": {"clientWidth": 0},
            "cssVisualViewport": {"clientWidth": 0},
            "cssLayoutViewport": {"clientWidth": 0},
        })

        ratio, x, y = await get_viewport_info_from_cdp(mock_cdp)

        assert ratio == 1.0  # Fallback for zero width

    @pytest.mark.asyncio
    async def test_viewport_missing_keys(self):
        """Test viewport info with missing keys in metrics."""
        from openbrowser.browser.python_highlights import get_viewport_info_from_cdp

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={})

        ratio, x, y = await get_viewport_info_from_cdp(mock_cdp)

        assert ratio == 1.0
        assert x == 0
        assert y == 0


# ---------------------------------------------------------------------------
# create_highlighted_screenshot_async (lines 501-544)
# ---------------------------------------------------------------------------


class TestCreateHighlightedScreenshotAsync:
    """Test create_highlighted_screenshot_async function."""

    @pytest.mark.asyncio
    async def test_async_without_cdp_session(self):
        """Test async wrapper without CDP session."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot_async

        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = await create_highlighted_screenshot_async(screenshot_b64, {})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_async_with_cdp_session(self):
        """Test async wrapper with CDP session."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot_async

        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={
            "visualViewport": {"clientWidth": 1280},
            "cssVisualViewport": {"clientWidth": 1280, "pageX": 0, "pageY": 0},
            "cssLayoutViewport": {"clientWidth": 1280},
        })

        result = await create_highlighted_screenshot_async(
            screenshot_b64, {}, cdp_session=mock_cdp
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_async_cdp_exception_handled(self):
        """Test async wrapper when CDP session raises exception."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot_async

        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            side_effect=Exception("CDP error")
        )

        result = await create_highlighted_screenshot_async(
            screenshot_b64, {}, cdp_session=mock_cdp
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_async_saves_screenshot_to_file(self, tmp_path):
        """Test that screenshot is saved to file when env var is set."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot_async

        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        output_file = str(tmp_path / "screenshot.png")

        with patch.dict("os.environ", {"OPENBROWSER_SCREENSHOT_FILE": output_file}):
            result = await create_highlighted_screenshot_async(screenshot_b64, {})

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_async_save_screenshot_error(self, tmp_path):
        """Test screenshot save with error."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot_async

        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Use an invalid path to trigger write error
        output_file = "/nonexistent/dir/screenshot.png"

        with patch.dict("os.environ", {"OPENBROWSER_SCREENSHOT_FILE": output_file}):
            result = await create_highlighted_screenshot_async(screenshot_b64, {})

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_async_no_screenshot_file_env(self):
        """Test that no file is saved when env var is not set."""
        from openbrowser.browser.python_highlights import create_highlighted_screenshot_async

        from PIL import Image as RealImage

        img = RealImage.new("RGBA", (100, 100), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        screenshot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        with patch.dict("os.environ", {}, clear=False):
            # Ensure the env var is not set
            if "OPENBROWSER_SCREENSHOT_FILE" in os.environ:
                del os.environ["OPENBROWSER_SCREENSHOT_FILE"]

            result = await create_highlighted_screenshot_async(screenshot_b64, {})

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# __all__ exports (line 548)
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Test module __all__ exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from openbrowser.browser.python_highlights import __all__

        assert "create_highlighted_screenshot" in __all__
        assert "create_highlighted_screenshot_async" in __all__
        assert "cleanup_font_cache" in __all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
