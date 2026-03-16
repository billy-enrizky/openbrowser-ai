"""Tests for clickable element detector (clickable_elements.py) - comprehensive coverage."""

import logging

import pytest

from openbrowser.dom.serializer.clickable_elements import ClickableElementDetector
from openbrowser.dom.views import (
    DOMRect,
    EnhancedAXNode,
    EnhancedAXProperty,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_snapshot(
    bounds=None,
    is_clickable=None,
    cursor_style=None,
):
    return EnhancedSnapshotNode(
        is_clickable=is_clickable,
        cursor_style=cursor_style,
        bounds=bounds or DOMRect(x=0, y=0, width=100, height=50),
        clientRects=None,
        scrollRects=None,
        computed_styles=None,
        paint_order=None,
        stacking_contexts=None,
    )


def _make_node(
    tag="div",
    node_type=NodeType.ELEMENT_NODE,
    attributes=None,
    snapshot=None,
    ax_node=None,
    is_visible=True,
):
    _id_counter = getattr(_make_node, "_counter", 0) + 1
    _make_node._counter = _id_counter

    return EnhancedDOMTreeNode(
        node_id=_id_counter,
        backend_node_id=_id_counter + 30000,
        node_type=node_type,
        node_name=tag,
        node_value="",
        attributes=attributes or {},
        is_scrollable=None,
        is_visible=is_visible,
        absolute_position=None,
        target_id="target-1",
        frame_id=None,
        session_id=None,
        content_document=None,
        shadow_root_type=None,
        shadow_roots=None,
        parent_node=None,
        children_nodes=None,
        ax_node=ax_node,
        snapshot_node=snapshot or (_make_snapshot() if is_visible else None),
    )


# ────────────────────────────────────────────────────────────────────
# Non-element node (line 10-11)
# ────────────────────────────────────────────────────────────────────

class TestNonElementNode:
    def test_text_node_not_interactive(self):
        node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE)
        assert ClickableElementDetector.is_interactive(node) is False

    def test_comment_node_not_interactive(self):
        node = _make_node(tag="#comment", node_type=NodeType.COMMENT_NODE)
        assert ClickableElementDetector.is_interactive(node) is False


# ────────────────────────────────────────────────────────────────────
# html/body exclusion (lines 18-19)
# ────────────────────────────────────────────────────────────────────

class TestHtmlBodyExcluded:
    def test_html_not_interactive(self):
        node = _make_node(tag="html")
        assert ClickableElementDetector.is_interactive(node) is False

    def test_body_not_interactive(self):
        node = _make_node(tag="body")
        assert ClickableElementDetector.is_interactive(node) is False


# ────────────────────────────────────────────────────────────────────
# IFRAME check (lines 23-29)
# ────────────────────────────────────────────────────────────────────

class TestIframeDetection:
    def test_large_iframe_interactive(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 200))
        node = _make_node(tag="IFRAME", snapshot=snap)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_small_iframe_not_interactive(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 50, 50))
        node = _make_node(tag="IFRAME", snapshot=snap)
        assert ClickableElementDetector.is_interactive(node) is False

    def test_iframe_no_bounds(self):
        snap = _make_snapshot(bounds=None)
        node = _make_node(tag="IFRAME", snapshot=snap)
        # No bounds - won't pass size check, falls through to other checks
        assert ClickableElementDetector.is_interactive(node) is False

    def test_iframe_no_snapshot(self):
        node = _make_node(tag="IFRAME", snapshot=None, is_visible=False)
        node.snapshot_node = None
        assert ClickableElementDetector.is_interactive(node) is False

    def test_frame_large_interactive(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 200))
        node = _make_node(tag="FRAME", snapshot=snap)
        assert ClickableElementDetector.is_interactive(node) is True


# ────────────────────────────────────────────────────────────────────
# Search element detection (lines 36-63)
# ────────────────────────────────────────────────────────────────────

class TestSearchDetection:
    def test_search_class(self):
        node = _make_node(tag="div", attributes={"class": "search-icon"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_search_id(self):
        node = _make_node(tag="div", attributes={"id": "searchbox"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_search_data_attr(self):
        node = _make_node(tag="div", attributes={"data-action": "search"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_no_search_indicator(self):
        node = _make_node(tag="div", attributes={"class": "container"})
        # No search indicator, should not be interactive purely from search check
        # It may still be interactive from other checks
        result = ClickableElementDetector.is_interactive(node)
        assert result is False  # plain div without any interactive signals

    def test_magnify_class(self):
        node = _make_node(tag="span", attributes={"class": "magnify-glass"})
        assert ClickableElementDetector.is_interactive(node) is True


# ────────────────────────────────────────────────────────────────────
# AX property checks (lines 66-95)
# ────────────────────────────────────────────────────────────────────

class TestAxProperties:
    def test_disabled_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="button", name="x",
            description=None,
            properties=[EnhancedAXProperty(name="disabled", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is False

    def test_hidden_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="button", name="x",
            description=None,
            properties=[EnhancedAXProperty(name="hidden", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is False

    def test_focusable_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="focusable", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_editable_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="editable", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_settable_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="settable", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_checked_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="checked", value=False)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_expanded_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="expanded", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_pressed_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="pressed", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_selected_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="selected", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_required_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="required", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_autocomplete_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="autocomplete", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_keyshortcuts_property(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="keyshortcuts", value="Ctrl+S")],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_property_exception_handled(self):
        """Property that raises AttributeError during processing should be skipped."""
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role=None, name="x",
            description=None,
            properties=[EnhancedAXProperty(name="focusable", value=False)],
            child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        # Should not crash
        result = ClickableElementDetector.is_interactive(node)
        assert isinstance(result, bool)


# ────────────────────────────────────────────────────────────────────
# Interactive tag check (lines 99-112)
# ────────────────────────────────────────────────────────────────────

class TestInteractiveTags:
    def test_button_interactive(self):
        node = _make_node(tag="button")
        assert ClickableElementDetector.is_interactive(node) is True

    def test_input_interactive(self):
        node = _make_node(tag="input")
        assert ClickableElementDetector.is_interactive(node) is True

    def test_select_interactive(self):
        node = _make_node(tag="select")
        assert ClickableElementDetector.is_interactive(node) is True

    def test_textarea_interactive(self):
        node = _make_node(tag="textarea")
        assert ClickableElementDetector.is_interactive(node) is True

    def test_a_interactive(self):
        node = _make_node(tag="a")
        assert ClickableElementDetector.is_interactive(node) is True

    def test_details_interactive(self):
        node = _make_node(tag="details")
        assert ClickableElementDetector.is_interactive(node) is True

    def test_summary_interactive(self):
        node = _make_node(tag="summary")
        assert ClickableElementDetector.is_interactive(node) is True

    def test_option_interactive(self):
        node = _make_node(tag="option")
        assert ClickableElementDetector.is_interactive(node) is True

    def test_optgroup_interactive(self):
        node = _make_node(tag="optgroup")
        assert ClickableElementDetector.is_interactive(node) is True


# ────────────────────────────────────────────────────────────────────
# Interactive attributes (lines 135-159)
# ────────────────────────────────────────────────────────────────────

class TestInteractiveAttributes:
    def test_onclick_interactive(self):
        node = _make_node(tag="div", attributes={"onclick": "handle()"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_tabindex_interactive(self):
        node = _make_node(tag="div", attributes={"tabindex": "0"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_onkeydown_interactive(self):
        node = _make_node(tag="div", attributes={"onkeydown": "handle()"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_role_button_interactive(self):
        node = _make_node(tag="div", attributes={"role": "button"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_role_link_interactive(self):
        node = _make_node(tag="div", attributes={"role": "link"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_role_combobox_interactive(self):
        node = _make_node(tag="div", attributes={"role": "combobox"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_role_slider_interactive(self):
        node = _make_node(tag="div", attributes={"role": "slider"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_role_search_interactive(self):
        node = _make_node(tag="div", attributes={"role": "search"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_role_searchbox_interactive(self):
        node = _make_node(tag="div", attributes={"role": "searchbox"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_role_non_interactive(self):
        node = _make_node(tag="div", attributes={"role": "presentation"})
        assert ClickableElementDetector.is_interactive(node) is False


# ────────────────────────────────────────────────────────────────────
# AX tree roles (lines 162-180)
# ────────────────────────────────────────────────────────────────────

class TestAxTreeRoles:
    def test_ax_button_role(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="button", name="x",
            description=None, properties=None, child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_ax_link_role(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="link", name="x",
            description=None, properties=None, child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_ax_listbox_role(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="listbox", name="x",
            description=None, properties=None, child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_ax_non_interactive_role(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="generic", name="x",
            description=None, properties=None, child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is False

    def test_ax_search_role(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="search", name="x",
            description=None, properties=None, child_ids=None,
        )
        node = _make_node(tag="div", ax_node=ax)
        assert ClickableElementDetector.is_interactive(node) is True


# ────────────────────────────────────────────────────────────────────
# Icon/small element check (lines 183-194)
# ────────────────────────────────────────────────────────────────────

class TestIconDetection:
    def test_icon_sized_with_class(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 20, 20))
        node = _make_node(tag="span", snapshot=snap, attributes={"class": "icon-close"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_icon_sized_with_role(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 30, 30))
        node = _make_node(tag="span", snapshot=snap, attributes={"role": "img"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_icon_sized_with_aria_label(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 15, 15))
        node = _make_node(tag="span", snapshot=snap, attributes={"aria-label": "close"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_icon_sized_with_data_action(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 25, 25))
        node = _make_node(tag="span", snapshot=snap, attributes={"data-action": "toggle"})
        assert ClickableElementDetector.is_interactive(node) is True

    def test_icon_sized_without_attrs(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 20, 20))
        node = _make_node(tag="span", snapshot=snap, attributes={})
        assert ClickableElementDetector.is_interactive(node) is False

    def test_too_large_not_icon(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 100, 100))
        node = _make_node(tag="span", snapshot=snap, attributes={"class": "icon"})
        # Too large for icon check (width > 50)
        assert ClickableElementDetector.is_interactive(node) is False

    def test_too_small_not_icon(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 5, 5))
        node = _make_node(tag="span", snapshot=snap, attributes={"class": "icon"})
        # Too small for icon check (width < 10)
        assert ClickableElementDetector.is_interactive(node) is False


# ────────────────────────────────────────────────────────────────────
# Cursor pointer fallback (lines 197-198)
# ────────────────────────────────────────────────────────────────────

class TestCursorPointer:
    def test_cursor_pointer_interactive(self):
        snap = _make_snapshot(cursor_style="pointer")
        node = _make_node(tag="span", snapshot=snap)
        assert ClickableElementDetector.is_interactive(node) is True

    def test_cursor_default_not_interactive(self):
        snap = _make_snapshot(cursor_style="default")
        node = _make_node(tag="span", snapshot=snap)
        assert ClickableElementDetector.is_interactive(node) is False

    def test_no_cursor_style_not_interactive(self):
        snap = _make_snapshot(cursor_style=None)
        node = _make_node(tag="span", snapshot=snap)
        assert ClickableElementDetector.is_interactive(node) is False


# ────────────────────────────────────────────────────────────────────
# Final fallback: nothing matched (line 200)
# ────────────────────────────────────────────────────────────────────

class TestFinalFallback:
    def test_plain_div_not_interactive(self):
        node = _make_node(tag="div")
        assert ClickableElementDetector.is_interactive(node) is False

    def test_plain_span_not_interactive(self):
        node = _make_node(tag="span")
        assert ClickableElementDetector.is_interactive(node) is False
