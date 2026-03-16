"""Tests for DOM views (views.py) - comprehensive coverage for remaining gaps."""

import logging
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from openbrowser.dom.views import (
    DEFAULT_INCLUDE_ATTRIBUTES,
    DOMInteractedElement,
    DOMRect,
    EnhancedAXNode,
    EnhancedAXProperty,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    PropagatingBounds,
    SerializedDOMState,
    SimplifiedNode,
    TargetAllTrees,
    CurrentPageTargets,
    STATIC_ATTRIBUTES,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_snapshot(bounds=None, computed_styles=None, scroll_rects=None, client_rects=None, cursor_style=None):
    return EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=cursor_style,
        bounds=bounds or DOMRect(x=0, y=0, width=100, height=50),
        clientRects=client_rects,
        scrollRects=scroll_rects,
        computed_styles=computed_styles,
        paint_order=None,
        stacking_contexts=None,
    )


def _make_node(
    tag="div",
    node_type=NodeType.ELEMENT_NODE,
    node_value="",
    attributes=None,
    children=None,
    is_visible=True,
    snapshot=None,
    ax_node=None,
    is_scrollable=None,
    backend_node_id=None,
    node_id=None,
    content_document=None,
    shadow_roots=None,
    shadow_root_type=None,
    parent_node=None,
    frame_id=None,
):
    _id_counter = getattr(_make_node, "_counter", 0) + 1
    _make_node._counter = _id_counter

    nid = node_id if node_id is not None else _id_counter
    bnid = backend_node_id if backend_node_id is not None else _id_counter + 60000

    node = EnhancedDOMTreeNode(
        node_id=nid,
        backend_node_id=bnid,
        node_type=node_type,
        node_name=tag,
        node_value=node_value,
        attributes=attributes or {},
        is_scrollable=is_scrollable,
        is_visible=is_visible,
        absolute_position=None,
        target_id="target-1",
        frame_id=frame_id,
        session_id=None,
        content_document=content_document,
        shadow_root_type=shadow_root_type,
        shadow_roots=shadow_roots,
        parent_node=parent_node,
        children_nodes=children,
        ax_node=ax_node,
        snapshot_node=snapshot or (_make_snapshot() if is_visible else None),
    )
    if children:
        for c in children:
            c.parent_node = node
    if shadow_roots:
        for sr in shadow_roots:
            sr.parent_node = node
    return node


def _make_text_node(text="Hello World", is_visible=True, snapshot=None):
    return _make_node(
        tag="#text",
        node_type=NodeType.TEXT_NODE,
        node_value=text,
        is_visible=is_visible,
        snapshot=snapshot or (_make_snapshot() if is_visible else None),
    )


# ────────────────────────────────────────────────────────────────────
# SimplifiedNode
# ────────────────────────────────────────────────────────────────────

class TestSimplifiedNode:
    """Test SimplifiedNode (lines 164-204)."""

    def test_clean_original_node_json(self):
        node = _make_node()
        simp = SimplifiedNode(original_node=node, children=[])
        json_data = {"children_nodes": [], "shadow_roots": [], "other": "data"}
        result = simp._clean_original_node_json(json_data)
        assert "children_nodes" not in result
        assert "shadow_roots" not in result
        assert result["other"] == "data"

    def test_clean_original_node_json_with_content_document(self):
        node = _make_node()
        simp = SimplifiedNode(original_node=node, children=[])
        json_data = {
            "children_nodes": [],
            "content_document": {"children_nodes": [], "shadow_roots": [], "inner": True},
        }
        result = simp._clean_original_node_json(json_data)
        assert "children_nodes" not in result
        assert "children_nodes" not in result["content_document"]
        assert result["content_document"]["inner"] is True

    def test_json_serialization(self):
        node = _make_node()
        child_node = _make_node(tag="span")
        child_simp = SimplifiedNode(original_node=child_node, children=[])
        simp = SimplifiedNode(
            original_node=node,
            children=[child_simp],
            is_interactive=True,
            excluded_by_parent=True,
            ignored_by_paint_order=True,
        )
        result = simp.__json__()
        assert result["is_interactive"] is True
        assert result["excluded_by_parent"] is True
        assert result["ignored_by_paint_order"] is True
        assert "children" in result
        assert len(result["children"]) == 1
        assert "original_node" in result

    def test_default_values(self):
        node = _make_node()
        simp = SimplifiedNode(original_node=node, children=[])
        assert simp.should_display is True
        assert simp.is_interactive is False
        assert simp.is_new is False
        assert simp.ignored_by_paint_order is False
        assert simp.excluded_by_parent is False
        assert simp.is_shadow_host is False
        assert simp.is_compound_component is False


# ────────────────────────────────────────────────────────────────────
# EnhancedDOMTreeNode - properties
# ────────────────────────────────────────────────────────────────────

class TestEnhancedDOMTreeNodeProperties:
    """Test properties (lines 394-464)."""

    def test_parent_property(self):
        child = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child])
        assert child.parent is parent

    def test_parent_none(self):
        node = _make_node()
        assert node.parent is None

    def test_children_property(self):
        child = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child])
        assert len(parent.children) == 1

    def test_children_none(self):
        node = _make_node()
        node.children_nodes = None
        assert node.children == []

    def test_children_and_shadow_roots(self):
        child = _make_node(tag="span")
        frag = _make_node(
            tag="#document-fragment",
            node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
        )
        parent = _make_node(tag="div", children=[child], shadow_roots=[frag])
        result = parent.children_and_shadow_roots
        assert len(result) == 2

    def test_children_and_shadow_roots_no_shadow(self):
        child = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child])
        result = parent.children_and_shadow_roots
        assert len(result) == 1

    def test_children_and_shadow_roots_no_children(self):
        node = _make_node(tag="div")
        node.children_nodes = None
        result = node.children_and_shadow_roots
        assert result == []

    def test_tag_name(self):
        node = _make_node(tag="DIV")
        assert node.tag_name == "div"

    def test_xpath_simple(self):
        node = _make_node(tag="div")
        assert node.xpath == "div"

    def test_xpath_with_parent(self):
        child = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child])
        xpath = child.xpath
        assert "span" in xpath
        assert "div" in xpath

    def test_xpath_with_siblings(self):
        child1 = _make_node(tag="span")
        child2 = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child1, child2])
        xpath1 = child1.xpath
        xpath2 = child2.xpath
        assert "[1]" in xpath1
        assert "[2]" in xpath2

    def test_xpath_stops_at_iframe(self):
        child = _make_node(tag="div")
        iframe = _make_node(tag="iframe", children=[child])
        # xpath should stop at iframe boundary
        assert "iframe" not in child.xpath

    def test_xpath_through_shadow_root(self):
        child = _make_node(tag="span")
        frag = _make_node(
            tag="#document-fragment",
            node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
            children=[child],
        )
        host = _make_node(tag="div", shadow_roots=[frag])
        frag.parent_node = host
        child.parent_node = frag
        xpath = child.xpath
        assert "span" in xpath

    def test_get_element_position_no_parent(self):
        node = _make_node(tag="div")
        assert node._get_element_position(node) == 0

    def test_get_element_position_single(self):
        child = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child])
        assert child._get_element_position(child) == 0

    def test_get_element_position_multiple(self):
        child1 = _make_node(tag="span")
        child2 = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child1, child2])
        assert child1._get_element_position(child1) == 1
        assert child2._get_element_position(child2) == 2


# ────────────────────────────────────────────────────────────────────
# EnhancedDOMTreeNode - __json__
# ────────────────────────────────────────────────────────────────────

class TestEnhancedDOMTreeNodeJson:
    """Test __json__ (lines 466-487)."""

    def test_basic_json(self):
        node = _make_node(tag="div", attributes={"id": "test"})
        j = node.__json__()
        assert j["node_name"] == "div"
        assert j["attributes"]["id"] == "test"
        assert j["node_type"] == "ELEMENT_NODE"

    def test_json_with_children(self):
        child = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child])
        j = parent.__json__()
        assert len(j["children_nodes"]) == 1

    def test_json_with_content_document(self):
        content_doc = _make_node(tag="html")
        node = _make_node(tag="iframe", content_document=content_doc)
        node.content_document = content_doc
        j = node.__json__()
        assert j["content_document"] is not None

    def test_json_with_shadow_roots(self):
        frag = _make_node(
            tag="#document-fragment",
            node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
        )
        host = _make_node(tag="div", shadow_roots=[frag])
        j = host.__json__()
        assert len(j["shadow_roots"]) == 1

    def test_json_with_ax_node(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="button",
            name="Click", description=None, properties=None, child_ids=None,
        )
        node = _make_node(tag="button", ax_node=ax)
        j = node.__json__()
        assert j["ax_node"]["role"] == "button"

    def test_json_with_snapshot(self):
        snap = _make_snapshot(bounds=DOMRect(10, 20, 100, 50))
        node = _make_node(tag="div", snapshot=snap)
        j = node.__json__()
        assert j["snapshot_node"]["bounds"]["x"] == 10

    def test_json_no_optional_fields(self):
        node = _make_node(tag="div")
        node.content_document = None
        node.shadow_roots = None
        node.ax_node = None
        node.snapshot_node = None
        node.children_nodes = None
        j = node.__json__()
        assert j["content_document"] is None
        assert j["ax_node"] is None
        assert j["snapshot_node"] is None
        assert j["shadow_roots"] == []
        assert j["children_nodes"] == []


# ────────────────────────────────────────────────────────────────────
# get_all_children_text
# ────────────────────────────────────────────────────────────────────

class TestGetAllChildrenText:
    """Test get_all_children_text (lines 489-509)."""

    def test_simple_text(self):
        text = _make_text_node("Hello")
        parent = _make_node(tag="div", children=[text])
        assert "Hello" in parent.get_all_children_text()

    def test_nested_text(self):
        text = _make_text_node("Deep")
        span = _make_node(tag="span", children=[text])
        div = _make_node(tag="div", children=[span])
        assert "Deep" in div.get_all_children_text()

    def test_max_depth(self):
        text = _make_text_node("Deep")
        span = _make_node(tag="span", children=[text])
        div = _make_node(tag="div", children=[span])
        # max_depth=0 should only look at immediate children
        result = div.get_all_children_text(max_depth=0)
        assert result == ""  # Text is 2 levels deep

    def test_no_children(self):
        node = _make_node(tag="div")
        assert node.get_all_children_text() == ""


# ────────────────────────────────────────────────────────────────────
# __repr__ and __str__
# ────────────────────────────────────────────────────────────────────

class TestReprAndStr:
    """Test __repr__ (lines 511-521) and __str__ (lines 756-757)."""

    def test_repr(self):
        node = _make_node(tag="button", attributes={"id": "submit"}, is_scrollable=True)
        r = repr(node)
        assert "button" in r
        assert "id=submit" in r
        assert "is_scrollable=True" in r

    def test_str(self):
        node = _make_node(tag="div", frame_id="ABCDEF1234")
        s = str(node)
        assert "div" in s
        assert "1234" in s  # Last 4 chars of frame_id

    def test_str_no_frame_id(self):
        node = _make_node(tag="div", frame_id=None)
        s = str(node)
        assert "?" in s


# ────────────────────────────────────────────────────────────────────
# llm_representation and get_meaningful_text_for_llm
# ────────────────────────────────────────────────────────────────────

class TestLlmRepresentation:
    """Test llm_representation (lines 523-528) and get_meaningful_text_for_llm (lines 530-547)."""

    def test_llm_representation(self):
        text = _make_text_node("Hello World")
        node = _make_node(tag="button", children=[text])
        result = node.llm_representation()
        assert "<button>" in result
        assert "Hello World" in result

    def test_llm_representation_empty(self):
        node = _make_node(tag="div")
        result = node.llm_representation()
        assert "<div>" in result

    def test_get_meaningful_text_value(self):
        node = _make_node(tag="input", attributes={"value": "test@email.com"})
        result = node.get_meaningful_text_for_llm()
        assert result == "test@email.com"

    def test_get_meaningful_text_aria_label(self):
        node = _make_node(tag="button", attributes={"aria-label": "Submit form"})
        result = node.get_meaningful_text_for_llm()
        assert result == "Submit form"

    def test_get_meaningful_text_title(self):
        node = _make_node(tag="a", attributes={"title": "Homepage"})
        result = node.get_meaningful_text_for_llm()
        assert result == "Homepage"

    def test_get_meaningful_text_placeholder(self):
        node = _make_node(tag="input", attributes={"placeholder": "Enter name"})
        result = node.get_meaningful_text_for_llm()
        assert result == "Enter name"

    def test_get_meaningful_text_alt(self):
        node = _make_node(tag="img", attributes={"alt": "Logo image"})
        result = node.get_meaningful_text_for_llm()
        assert result == "Logo image"

    def test_get_meaningful_text_fallback_to_children(self):
        text = _make_text_node("Click Me")
        node = _make_node(tag="button", children=[text])
        result = node.get_meaningful_text_for_llm()
        assert "Click Me" in result

    def test_get_meaningful_text_no_attrs_no_children(self):
        node = _make_node(tag="div")
        result = node.get_meaningful_text_for_llm()
        assert result == ""


# ────────────────────────────────────────────────────────────────────
# is_actually_scrollable
# ────────────────────────────────────────────────────────────────────

class TestIsActuallyScrollable:
    """Test is_actually_scrollable (lines 549-598)."""

    def test_cdp_scrollable(self):
        node = _make_node(tag="div", is_scrollable=True)
        assert node.is_actually_scrollable is True

    def test_no_snapshot(self):
        node = _make_node(tag="div", is_scrollable=False, snapshot=None, is_visible=False)
        node.snapshot_node = None
        assert node.is_actually_scrollable is False

    def test_scroll_larger_than_client(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is True

    def test_scroll_not_larger_than_client(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 100),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is False

    def test_overflow_visible_not_scrollable(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "visible"},
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is False

    def test_overflow_scroll_scrollable(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "scroll"},
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is True

    def test_overflow_overlay_scrollable(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "overlay"},
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is True

    def test_overflow_x_auto(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 500, 100),
            computed_styles={"overflow-x": "auto"},
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is True

    def test_overflow_y_auto(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow-y": "auto"},
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is True

    def test_no_computed_styles_div_scrollable(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles=None,
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is True  # div is in scrollable_tags

    def test_no_computed_styles_unknown_tag_not_scrollable(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles=None,
        )
        node = _make_node(tag="span", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is False

    def test_no_scroll_rects(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=None,
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is False

    def test_no_client_rects(self):
        snap = _make_snapshot(
            client_rects=None,
            scroll_rects=DOMRect(0, 0, 100, 500),
        )
        node = _make_node(tag="div", snapshot=snap, is_scrollable=False)
        assert node.is_actually_scrollable is False


# ────────────────────────────────────────────────────────────────────
# should_show_scroll_info
# ────────────────────────────────────────────────────────────────────

class TestShouldShowScrollInfo:
    """Test should_show_scroll_info (lines 600-626)."""

    def test_iframe_always_shows(self):
        node = _make_node(tag="iframe")
        assert node.should_show_scroll_info is True

    def test_not_scrollable(self):
        node = _make_node(tag="div", is_scrollable=False)
        assert node.should_show_scroll_info is False

    def test_body_always_shows(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="body", is_scrollable=True, snapshot=snap)
        assert node.should_show_scroll_info is True

    def test_html_always_shows(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="html", is_scrollable=True, snapshot=snap)
        assert node.should_show_scroll_info is True

    def test_parent_scrollable_hides(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        child = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        parent = _make_node(tag="div", is_scrollable=True, children=[child], snapshot=snap)
        assert child.should_show_scroll_info is False

    def test_parent_not_scrollable_shows(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        child = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        parent = _make_node(tag="div", is_scrollable=False, children=[child])
        assert child.should_show_scroll_info is True


# ────────────────────────────────────────────────────────────────────
# _find_html_in_content_document
# ────────────────────────────────────────────────────────────────────

class TestFindHtmlInContentDocument:
    """Test _find_html_in_content_document (lines 628-643)."""

    def test_no_content_document(self):
        node = _make_node(tag="iframe")
        assert node._find_html_in_content_document() is None

    def test_content_document_is_html(self):
        html = _make_node(tag="html")
        iframe = _make_node(tag="iframe", content_document=html)
        iframe.content_document = html
        assert iframe._find_html_in_content_document() is html

    def test_content_document_child_is_html(self):
        html = _make_node(tag="html")
        doc = _make_node(
            tag="#document", node_type=NodeType.DOCUMENT_NODE,
            children=[html],
        )
        iframe = _make_node(tag="iframe", content_document=doc)
        iframe.content_document = doc
        result = iframe._find_html_in_content_document()
        assert result is html

    def test_no_html_in_content_document(self):
        div = _make_node(tag="div")
        doc = _make_node(
            tag="#document", node_type=NodeType.DOCUMENT_NODE,
            children=[div],
        )
        iframe = _make_node(tag="iframe", content_document=doc)
        iframe.content_document = doc
        result = iframe._find_html_in_content_document()
        assert result is None


# ────────────────────────────────────────────────────────────────────
# scroll_info
# ────────────────────────────────────────────────────────────────────

class TestScrollInfo:
    """Test scroll_info property (lines 645-714)."""

    def test_not_scrollable(self):
        node = _make_node(tag="div")
        assert node.scroll_info is None

    def test_scrollable_with_info(self):
        snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100),
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 50, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        info = node.scroll_info
        assert info is not None
        assert info["scroll_top"] == 50
        assert info["scrollable_height"] == 500
        assert info["visible_height"] == 100
        assert info["can_scroll_down"] is True

    def test_no_scroll_rects(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=None,
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        assert node.scroll_info is None

    def test_horizontal_scroll(self):
        snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100),
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(50, 0, 500, 100),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        info = node.scroll_info
        assert info is not None
        assert info["scroll_left"] == 50
        assert info["can_scroll_right"] is True

    def test_scroll_percentages(self):
        snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100),
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 200, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        info = node.scroll_info
        assert info is not None
        assert info["vertical_scroll_percentage"] == 50.0  # 200/(500-100) * 100


# ────────────────────────────────────────────────────────────────────
# get_scroll_info_text
# ────────────────────────────────────────────────────────────────────

class TestGetScrollInfoText:
    """Test get_scroll_info_text (lines 716-750)."""

    def test_iframe_with_content_scroll(self):
        html_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 800, 600),
            client_rects=DOMRect(0, 0, 800, 600),
            scroll_rects=DOMRect(0, 100, 800, 2000),
            computed_styles={"overflow": "auto"},
        )
        html = _make_node(tag="html", snapshot=html_snap, is_scrollable=True)
        doc = _make_node(
            tag="#document", node_type=NodeType.DOCUMENT_NODE,
            children=[html],
        )
        iframe = _make_node(tag="iframe", content_document=doc)
        iframe.content_document = doc
        result = iframe.get_scroll_info_text()
        assert "scroll" in result.lower()

    def test_iframe_without_content(self):
        iframe = _make_node(tag="iframe")
        result = iframe.get_scroll_info_text()
        assert result == "scroll"

    def test_non_scrollable_empty(self):
        node = _make_node(tag="div")
        result = node.get_scroll_info_text()
        assert result == ""

    def test_scrollable_shows_pages(self):
        snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100),
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 50, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        result = node.get_scroll_info_text()
        assert "pages" in result

    def test_horizontal_scroll_info(self):
        snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100),
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(50, 0, 500, 100),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        result = node.get_scroll_info_text()
        assert "horizontal" in result

    def test_iframe_no_html_child(self):
        div = _make_node(tag="div")
        doc = _make_node(
            tag="#document", node_type=NodeType.DOCUMENT_NODE,
            children=[div],
        )
        iframe = _make_node(tag="iframe", content_document=doc)
        iframe.content_document = doc
        result = iframe.get_scroll_info_text()
        assert result == "scroll"


# ────────────────────────────────────────────────────────────────────
# __hash__ and parent_branch_hash
# ────────────────────────────────────────────────────────────────────

class TestHashing:
    """Test __hash__ (lines 759-779) and parent_branch_hash (lines 781-789)."""

    def test_hash_deterministic(self):
        node = _make_node(tag="button", attributes={"id": "submit"})
        h1 = hash(node)
        h2 = hash(node)
        assert h1 == h2

    def test_hash_different_attributes(self):
        node1 = _make_node(tag="button", attributes={"id": "a"})
        node2 = _make_node(tag="button", attributes={"id": "b"})
        assert hash(node1) != hash(node2)

    def test_element_hash_property(self):
        node = _make_node(tag="div")
        assert node.element_hash == hash(node)

    def test_parent_branch_hash(self):
        child = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child])
        h = child.parent_branch_hash()
        assert isinstance(h, int)

    def test_parent_branch_path(self):
        child = _make_node(tag="span")
        parent = _make_node(tag="div", children=[child])
        path = child._get_parent_branch_path()
        assert "div" in path
        assert "span" in path


# ────────────────────────────────────────────────────────────────────
# DOMInteractedElement.load_from_enhanced_dom_tree
# ────────────────────────────────────────────────────────────────────

class TestDOMInteractedElementLoadFrom:
    """Test load_from_enhanced_dom_tree (lines 892-905)."""

    def test_load_basic(self):
        snap = _make_snapshot(bounds=DOMRect(10, 20, 100, 50))
        node = _make_node(
            tag="button", snapshot=snap,
            attributes={"id": "btn"}, frame_id="f-1",
        )
        elem = DOMInteractedElement.load_from_enhanced_dom_tree(node)
        assert elem.node_name == "button"
        assert elem.attributes == {"id": "btn"}
        assert elem.bounds is not None
        assert elem.bounds.x == 10

    def test_load_no_snapshot(self):
        node = _make_node(tag="div", snapshot=None, is_visible=False)
        node.snapshot_node = None
        elem = DOMInteractedElement.load_from_enhanced_dom_tree(node)
        assert elem.bounds is None


# ────────────────────────────────────────────────────────────────────
# DOMRect
# ────────────────────────────────────────────────────────────────────

class TestDOMRect:
    """Test DOMRect to_dict and __json__."""

    def test_to_dict(self):
        r = DOMRect(x=1.0, y=2.0, width=3.0, height=4.0)
        d = r.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}

    def test_json(self):
        r = DOMRect(x=5.0, y=6.0, width=7.0, height=8.0)
        j = r.__json__()
        assert j == {"x": 5.0, "y": 6.0, "width": 7.0, "height": 8.0}


# ────────────────────────────────────────────────────────────────────
# SerializedDOMState llm_representation and eval_representation
# ────────────────────────────────────────────────────────────────────

class TestSerializedDOMStateRepresentations:
    """Test llm_representation and eval_representation (lines 815-851)."""

    def test_llm_representation_empty(self):
        state = SerializedDOMState(_root=None, selector_map={})
        result = state.llm_representation()
        assert "Empty DOM tree" in result

    def test_eval_representation_empty(self):
        state = SerializedDOMState(_root=None, selector_map={})
        result = state.eval_representation()
        assert "Empty DOM tree" in result

    def test_llm_representation_with_root(self):
        text = _make_text_node("Hello")
        node = _make_node(tag="div", children=[text])
        simp = SimplifiedNode(original_node=node, children=[
            SimplifiedNode(original_node=text, children=[])
        ])
        state = SerializedDOMState(_root=simp, selector_map={})
        result = state.llm_representation()
        assert "Hello" in result

    def test_eval_representation_with_root(self):
        text = _make_text_node("World")
        node = _make_node(tag="div", children=[text])
        simp = SimplifiedNode(original_node=node, children=[
            SimplifiedNode(original_node=text, children=[])
        ])
        state = SerializedDOMState(_root=simp, selector_map={})
        result = state.eval_representation()
        assert "div" in result.lower()

    def test_llm_representation_custom_attributes(self):
        node = _make_node(tag="input", attributes={"type": "text"})
        simp = SimplifiedNode(original_node=node, children=[])
        state = SerializedDOMState(_root=simp, selector_map={})
        result = state.llm_representation(include_attributes=["type"])
        assert isinstance(result, str)


# ────────────────────────────────────────────────────────────────────
# PropagatingBounds
# ────────────────────────────────────────────────────────────────────

class TestPropagatingBounds:
    """Test PropagatingBounds dataclass."""

    def test_creation(self):
        rect = DOMRect(x=0, y=0, width=100, height=50)
        bounds = PropagatingBounds(tag="a", bounds=rect, node_id=1, depth=0)
        assert bounds.tag == "a"
        assert bounds.bounds is rect
        assert bounds.node_id == 1


# ────────────────────────────────────────────────────────────────────
# TargetAllTrees, CurrentPageTargets
# ────────────────────────────────────────────────────────────────────

class TestDataClasses:
    """Test remaining dataclasses."""

    def test_target_all_trees(self):
        t = TargetAllTrees(
            snapshot={}, dom_tree={}, ax_tree={},
            device_pixel_ratio=1.0, cdp_timing={"total": 0.5},
        )
        assert t.device_pixel_ratio == 1.0

    def test_current_page_targets(self):
        t = CurrentPageTargets(
            page_session={"targetId": "t1"},
            iframe_sessions=[],
        )
        assert t.page_session["targetId"] == "t1"
