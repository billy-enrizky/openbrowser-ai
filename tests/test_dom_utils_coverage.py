"""Tests for DOM utils and enhanced_snapshot modules - comprehensive coverage."""

import logging

import pytest

from openbrowser.dom.utils import cap_text_length, generate_css_selector_for_element
from openbrowser.dom.enhanced_snapshot import (
    REQUIRED_COMPUTED_STYLES,
    _parse_computed_styles,
    _parse_rare_boolean_data,
    build_snapshot_lookup,
)
from openbrowser.dom.views import (
    DOMRect,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_node(tag="div", attributes=None):
    """Minimal EnhancedDOMTreeNode for CSS selector testing."""
    _id_counter = getattr(_make_node, "_counter", 0) + 1
    _make_node._counter = _id_counter

    return EnhancedDOMTreeNode(
        node_id=_id_counter,
        backend_node_id=_id_counter + 70000,
        node_type=NodeType.ELEMENT_NODE,
        node_name=tag,
        node_value="",
        attributes=attributes or {},
        is_scrollable=None,
        is_visible=True,
        absolute_position=None,
        target_id="target-1",
        frame_id=None,
        session_id=None,
        content_document=None,
        shadow_root_type=None,
        shadow_roots=None,
        parent_node=None,
        children_nodes=None,
        ax_node=None,
        snapshot_node=None,
    )


# ────────────────────────────────────────────────────────────────────
# cap_text_length (utils.py line 1-5)
# ────────────────────────────────────────────────────────────────────

class TestCapTextLength:
    """Test cap_text_length function."""

    def test_short_text_unchanged(self):
        assert cap_text_length("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert cap_text_length("hello", 5) == "hello"

    def test_long_text_capped(self):
        result = cap_text_length("hello world", 5)
        assert result == "hello..."

    def test_empty_text(self):
        assert cap_text_length("", 10) == ""


# ────────────────────────────────────────────────────────────────────
# generate_css_selector_for_element (utils.py lines 8-129)
# ────────────────────────────────────────────────────────────────────

class TestGenerateCssSelector:
    """Test generate_css_selector_for_element."""

    def test_none_input(self):
        result = generate_css_selector_for_element(None)
        assert result is None

    def test_no_tag_name(self):
        node = _make_node(tag="")
        result = generate_css_selector_for_element(node)
        assert result is None

    def test_invalid_tag_name(self):
        node = _make_node(tag="123invalid")
        result = generate_css_selector_for_element(node)
        assert result is None

    def test_simple_tag(self):
        node = _make_node(tag="div")
        result = generate_css_selector_for_element(node)
        assert result == "div"

    def test_with_valid_id(self):
        node = _make_node(tag="div", attributes={"id": "main"})
        result = generate_css_selector_for_element(node)
        assert result == "#main"

    def test_with_special_char_id(self):
        node = _make_node(tag="div", attributes={"id": "my$id.test"})
        result = generate_css_selector_for_element(node)
        assert 'id="my$id.test"' in result

    def test_with_empty_id(self):
        node = _make_node(tag="div", attributes={"id": "  "})
        result = generate_css_selector_for_element(node)
        # Empty ID after strip should fall through
        # Result will be just "div" since ID is whitespace
        assert result is not None

    def test_with_classes(self):
        node = _make_node(tag="div", attributes={"class": "main container"})
        result = generate_css_selector_for_element(node)
        assert ".main" in result
        assert ".container" in result

    def test_with_invalid_class_skipped(self):
        node = _make_node(tag="div", attributes={"class": "valid 123invalid"})
        result = generate_css_selector_for_element(node)
        assert ".valid" in result
        assert "123invalid" not in result

    def test_with_empty_class(self):
        node = _make_node(tag="div", attributes={"class": "  "})
        result = generate_css_selector_for_element(node)
        assert result == "div"

    def test_safe_attribute_name(self):
        node = _make_node(tag="input", attributes={"name": "email"})
        result = generate_css_selector_for_element(node)
        assert '[name="email"]' in result

    def test_empty_attribute_value(self):
        node = _make_node(tag="input", attributes={"required": ""})
        result = generate_css_selector_for_element(node)
        assert "[required]" in result

    def test_attribute_with_special_chars(self):
        node = _make_node(tag="input", attributes={"placeholder": 'Enter "name"'})
        result = generate_css_selector_for_element(node)
        assert "placeholder" in result

    def test_attribute_with_newline(self):
        node = _make_node(tag="div", attributes={"title": "line1\nline2"})
        result = generate_css_selector_for_element(node)
        assert "title" in result

    def test_non_safe_attribute_skipped(self):
        node = _make_node(tag="div", attributes={"data-random": "value"})
        result = generate_css_selector_for_element(node)
        assert "data-random" not in result

    def test_safe_dynamic_attributes(self):
        node = _make_node(tag="div", attributes={"data-testid": "submit-btn"})
        result = generate_css_selector_for_element(node)
        assert 'data-testid="submit-btn"' in result

    def test_colon_in_attribute(self):
        node = _make_node(tag="div", attributes={"aria-label": "Close dialog"})
        result = generate_css_selector_for_element(node)
        assert "aria-label" in result

    def test_href_attribute(self):
        node = _make_node(tag="a", attributes={"href": "https://example.com"})
        result = generate_css_selector_for_element(node)
        assert 'href="https://example.com"' in result

    def test_data_cy_attribute(self):
        node = _make_node(tag="button", attributes={"data-cy": "submit"})
        result = generate_css_selector_for_element(node)
        assert 'data-cy="submit"' in result

    def test_data_qa_attribute(self):
        node = _make_node(tag="input", attributes={"data-qa": "email-field"})
        result = generate_css_selector_for_element(node)
        assert 'data-qa="email-field"' in result

    def test_empty_attribute_name_skipped(self):
        node = _make_node(tag="div", attributes={"": "value", "name": "test"})
        result = generate_css_selector_for_element(node)
        assert '[name="test"]' in result

    def test_class_attr_skipped_in_safe_attrs(self):
        """class attribute should be handled separately, not in safe attributes."""
        node = _make_node(tag="div", attributes={"class": "my-class", "name": "test"})
        result = generate_css_selector_for_element(node)
        assert ".my-class" in result

    def test_multiple_attributes(self):
        node = _make_node(
            tag="input",
            attributes={"type": "text", "name": "email", "placeholder": "Enter email"},
        )
        result = generate_css_selector_for_element(node)
        assert 'type="text"' in result
        assert 'name="email"' in result

    def test_whitespace_in_value(self):
        node = _make_node(tag="div", attributes={"title": "hello   world\ttab"})
        result = generate_css_selector_for_element(node)
        assert "title" in result

    def test_result_with_no_problematic_chars(self):
        node = _make_node(tag="div", attributes={"name": "normal"})
        result = generate_css_selector_for_element(node)
        assert "\n" not in result
        assert "\r" not in result
        assert "\t" not in result


# ────────────────────────────────────────────────────────────────────
# _parse_rare_boolean_data (enhanced_snapshot.py line 33-35)
# ────────────────────────────────────────────────────────────────────

class TestParseRareBooleanData:
    """Test _parse_rare_boolean_data."""

    def test_index_present(self):
        rare_data = {"index": [0, 2, 5]}
        assert _parse_rare_boolean_data(rare_data, 0) is True
        assert _parse_rare_boolean_data(rare_data, 2) is True

    def test_index_not_present(self):
        rare_data = {"index": [0, 2, 5]}
        assert _parse_rare_boolean_data(rare_data, 1) is False
        assert _parse_rare_boolean_data(rare_data, 3) is False

    def test_empty_index(self):
        rare_data = {"index": []}
        assert _parse_rare_boolean_data(rare_data, 0) is False


# ────────────────────────────────────────────────────────────────────
# _parse_computed_styles (enhanced_snapshot.py line 38-44)
# ────────────────────────────────────────────────────────────────────

class TestParseComputedStyles:
    """Test _parse_computed_styles."""

    def test_basic_styles(self):
        strings = ["block", "visible", "1", "visible", "auto", "auto", "pointer", "auto", "relative", "rgba(0,0,0,0)"]
        style_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        result = _parse_computed_styles(strings, style_indices)
        assert result["display"] == "block"
        assert result["visibility"] == "visible"
        assert result["cursor"] == "pointer"

    def test_out_of_bounds_index(self):
        strings = ["block", "visible"]
        style_indices = [0, 1, 999]  # 999 is out of bounds
        result = _parse_computed_styles(strings, style_indices)
        assert "display" in result
        assert "visibility" in result

    def test_empty_styles(self):
        result = _parse_computed_styles([], [])
        assert result == {}


# ────────────────────────────────────────────────────────────────────
# build_snapshot_lookup (enhanced_snapshot.py lines 47-161)
# ────────────────────────────────────────────────────────────────────

class TestBuildSnapshotLookup:
    """Test build_snapshot_lookup."""

    def test_empty_documents(self):
        snapshot = {"documents": [], "strings": []}
        result = build_snapshot_lookup(snapshot)
        assert result == {}

    def test_basic_snapshot(self):
        snapshot = {
            "documents": [{
                "nodes": {
                    "backendNodeId": [100, 101],
                    "isClickable": {"index": [0]},
                },
                "layout": {
                    "nodeIndex": [0, 1],
                    "bounds": [[10, 20, 100, 50], [30, 40, 200, 80]],
                    "styles": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                    "paintOrders": [1, 2],
                    "clientRects": [[0, 0, 100, 50], [0, 0, 200, 80]],
                    "scrollRects": [[0, 0, 100, 50], [0, 0, 200, 80]],
                    "stackingContexts": {"index": [0, 1]},
                },
            }],
            "strings": ["block", "visible", "1", "visible", "auto", "auto", "pointer", "auto", "relative", "rgba(0,0,0,0)"],
        }
        result = build_snapshot_lookup(snapshot)
        assert 100 in result
        assert 101 in result
        assert result[100].is_clickable is True
        assert result[101].is_clickable is False
        assert result[100].bounds is not None
        assert result[100].bounds.x == 10.0
        assert result[100].paint_order == 1

    def test_device_pixel_ratio(self):
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0],
                    "bounds": [[20, 40, 200, 100]],
                    "styles": [],
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot, device_pixel_ratio=2.0)
        assert result[100].bounds.x == 10.0  # 20/2
        assert result[100].bounds.y == 20.0  # 40/2
        assert result[100].bounds.width == 100.0  # 200/2

    def test_no_layout_data(self):
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {"nodeIndex": [], "bounds": []},
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert 100 in result
        assert result[100].bounds is None

    def test_no_is_clickable(self):
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {"nodeIndex": [], "bounds": []},
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result[100].is_clickable is None

    def test_incomplete_bounds(self):
        """Bounds with fewer than 4 values should be handled gracefully."""
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0],
                    "bounds": [[10, 20]],  # Only 2 values
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result[100].bounds is None  # Not enough values for DOMRect

    def test_no_backend_node_id(self):
        snapshot = {
            "documents": [{
                "nodes": {},
                "layout": {"nodeIndex": [], "bounds": []},
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result == {}

    def test_duplicate_layout_indices_first_wins(self):
        """When duplicate nodeIndex entries exist, first occurrence wins."""
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0, 0],  # Duplicate index
                    "bounds": [[10, 20, 100, 50], [30, 40, 200, 80]],
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        # First occurrence should win - bounds should be from first entry
        assert result[100].bounds.x == 10.0

    def test_client_rects_incomplete(self):
        """ClientRects with fewer than 4 values should not create DOMRect."""
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0],
                    "bounds": [[0, 0, 100, 50]],
                    "clientRects": [[10]],  # Incomplete
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result[100].clientRects is None

    def test_scroll_rects_incomplete(self):
        """ScrollRects with fewer than 4 values should not create DOMRect."""
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0],
                    "bounds": [[0, 0, 100, 50]],
                    "scrollRects": [[10, 20]],  # Incomplete
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result[100].scrollRects is None

    def test_empty_client_rects(self):
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0],
                    "bounds": [[0, 0, 100, 50]],
                    "clientRects": [None],
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result[100].clientRects is None

    def test_empty_scroll_rects(self):
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0],
                    "bounds": [[0, 0, 100, 50]],
                    "scrollRects": [None],
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result[100].scrollRects is None

    def test_multiple_documents(self):
        snapshot = {
            "documents": [
                {
                    "nodes": {"backendNodeId": [100]},
                    "layout": {
                        "nodeIndex": [0],
                        "bounds": [[0, 0, 100, 50]],
                    },
                },
                {
                    "nodes": {"backendNodeId": [200]},
                    "layout": {
                        "nodeIndex": [0],
                        "bounds": [[10, 10, 200, 100]],
                    },
                },
            ],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert 100 in result
        assert 200 in result

    def test_stacking_contexts(self):
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0],
                    "bounds": [[0, 0, 100, 50]],
                    "stackingContexts": {"index": [42]},
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result[100].stacking_contexts == 42

    def test_no_paint_orders(self):
        snapshot = {
            "documents": [{
                "nodes": {"backendNodeId": [100]},
                "layout": {
                    "nodeIndex": [0],
                    "bounds": [[0, 0, 100, 50]],
                },
            }],
            "strings": [],
        }
        result = build_snapshot_lookup(snapshot)
        assert result[100].paint_order is None
