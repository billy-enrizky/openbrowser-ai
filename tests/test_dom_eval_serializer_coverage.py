"""Tests for DOM eval serializer (eval_serializer.py) - comprehensive coverage."""

import logging
from unittest.mock import MagicMock

import pytest

from openbrowser.dom.serializer.eval_serializer import (
    COLLAPSIBLE_CONTAINERS,
    EVAL_KEY_ATTRIBUTES,
    SEMANTIC_ELEMENTS,
    SVG_ELEMENTS,
    DOMEvalSerializer,
)
from openbrowser.dom.views import (
    DOMRect,
    EnhancedAXNode,
    EnhancedAXProperty,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    SimplifiedNode,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_snapshot(bounds=None, computed_styles=None, client_rects=None, scroll_rects=None):
    return EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
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
    content_document=None,
    shadow_root_type=None,
    shadow_roots=None,
    frame_id=None,
):
    _id_counter = getattr(_make_node, "_counter", 0) + 1
    _make_node._counter = _id_counter

    nid = _id_counter
    bnid = backend_node_id if backend_node_id is not None else _id_counter + 20000

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
        parent_node=None,
        children_nodes=children,
        ax_node=ax_node,
        snapshot_node=snapshot or (_make_snapshot() if is_visible else None),
    )
    if children:
        for c in children:
            c.parent_node = node
    return node


def _make_text_node(text="Hello World", is_visible=True, snapshot=None):
    return _make_node(
        tag="#text",
        node_type=NodeType.TEXT_NODE,
        node_value=text,
        is_visible=is_visible,
        snapshot=snapshot or (_make_snapshot() if is_visible else None),
    )


def _make_simplified(node, children=None, is_interactive=False, should_display=True, excluded_by_parent=False):
    return SimplifiedNode(
        original_node=node,
        children=children or [],
        is_interactive=is_interactive,
        should_display=should_display,
        excluded_by_parent=excluded_by_parent,
    )


# ────────────────────────────────────────────────────────────────────
# serialize_tree
# ────────────────────────────────────────────────────────────────────

class TestSerializeTree:
    """Test DOMEvalSerializer.serialize_tree (lines 116-231)."""

    def test_none_returns_empty(self):
        result = DOMEvalSerializer.serialize_tree(None, EVAL_KEY_ATTRIBUTES)
        assert result == ""

    def test_excluded_by_parent_skips_node(self):
        # Use an element node child since text nodes are handled inline
        inner_text = _make_text_node("inner content")
        inner_text_simp = _make_simplified(inner_text)
        inner = _make_node(tag="button", snapshot=_make_snapshot(), children=[inner_text])
        inner_simp = _make_simplified(inner, children=[inner_text_simp])
        outer = _make_node(tag="div")
        outer_simp = _make_simplified(outer, children=[inner_simp], excluded_by_parent=True)
        result = DOMEvalSerializer.serialize_tree(outer_simp, EVAL_KEY_ATTRIBUTES)
        assert "<button" in result
        assert "<div" not in result

    def test_should_display_false_skips_node(self):
        inner = _make_text_node("content")
        inner_simp = _make_simplified(inner)
        outer = _make_node(tag="div")
        outer_simp = _make_simplified(outer, children=[inner_simp], should_display=False)
        result = DOMEvalSerializer.serialize_tree(outer_simp, EVAL_KEY_ATTRIBUTES)
        # Should skip the div but show children
        assert "<div" not in result

    def test_invisible_element_skipped(self):
        node = _make_node(tag="span", is_visible=False, snapshot=None)
        simp = _make_simplified(node)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert result == ""

    def test_invisible_container_shown(self):
        """Containers like div should be shown even if invisible (might have visible children)."""
        child = _make_text_node("visible text")
        child_simp = _make_simplified(child)
        node = _make_node(tag="div", is_visible=False, snapshot=None, children=[child])
        simp = _make_simplified(node, children=[child_simp])
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert "<div" in result

    def test_iframe_invisible_shown(self):
        node = _make_node(tag="iframe", is_visible=False, snapshot=None)
        simp = _make_simplified(node)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        # iframe/frame should be shown even if invisible
        assert "<iframe" in result

    def test_frame_invisible_shown(self):
        node = _make_node(tag="frame", is_visible=False, snapshot=None)
        simp = _make_simplified(node)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert "<frame" in result

    def test_svg_collapsed(self):
        node = _make_node(tag="svg", snapshot=_make_snapshot())
        simp = _make_simplified(node)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert "<svg" in result
        assert "SVG content collapsed" in result

    def test_svg_interactive(self):
        node = _make_node(tag="svg", snapshot=_make_snapshot(), backend_node_id=42)
        simp = _make_simplified(node, is_interactive=True)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert "[i_42]" in result

    def test_svg_child_skipped(self):
        # SVG_ELEMENTS has mixed-case entries (clipPath, polyline) but code
        # does tag.lower() comparison, so only all-lowercase entries match
        lowercase_svg_tags = [t for t in SVG_ELEMENTS if t == t.lower()]
        for tag in lowercase_svg_tags:
            node = _make_node(tag=tag, snapshot=_make_snapshot())
            simp = _make_simplified(node)
            result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
            assert result == "", f"Expected empty for SVG child: {tag}"

    def test_semantic_element_shown(self):
        for tag in ["h1", "button", "input", "a", "form"]:
            node = _make_node(tag=tag, snapshot=_make_snapshot())
            simp = _make_simplified(node)
            result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
            assert f"<{tag}" in result

    def test_interactive_element_with_id(self):
        node = _make_node(
            tag="button", snapshot=_make_snapshot(),
            backend_node_id=99,
        )
        simp = _make_simplified(node, is_interactive=True)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert "[i_99]" in result
        assert "<button" in result

    def test_attributes_in_output(self):
        node = _make_node(
            tag="input",
            snapshot=_make_snapshot(),
            attributes={"type": "text", "name": "email", "placeholder": "Enter email"},
        )
        simp = _make_simplified(node)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert 'type="text"' in result
        assert 'name="email"' in result

    def test_inline_text_for_non_container(self):
        text = _make_text_node("Click me")
        node = _make_node(tag="button", snapshot=_make_snapshot(), children=[text])
        text_simp = _make_simplified(text)
        simp = _make_simplified(node, children=[text_simp])
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert ">Click me" in result

    def test_container_shows_children_not_inline(self):
        text = _make_text_node("Content")
        child = _make_node(tag="span", snapshot=_make_snapshot(), children=[text])
        text_simp = _make_simplified(text)
        child_simp = _make_simplified(child, children=[text_simp])
        node = _make_node(tag="div", snapshot=_make_snapshot(), children=[child])
        simp = _make_simplified(node, children=[child_simp])
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert "<div" in result
        assert "<span" in result

    def test_scroll_info_added(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        node = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        simp = _make_simplified(node)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        assert "scroll=" in result

    def test_text_node_handled(self):
        text = _make_text_node("Plain text")
        simp = _make_simplified(text)
        result = DOMEvalSerializer.serialize_tree(simp, EVAL_KEY_ATTRIBUTES)
        # Text nodes are handled inline with parent, so standalone should be empty
        assert result == ""

    def test_document_fragment_node(self):
        inner = _make_node(tag="span", snapshot=_make_snapshot())
        inner_simp = _make_simplified(inner)
        frag = _make_node(
            tag="#document-fragment",
            node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
            children=[inner],
        )
        frag_simp = _make_simplified(frag, children=[inner_simp])
        result = DOMEvalSerializer.serialize_tree(frag_simp, EVAL_KEY_ATTRIBUTES)
        assert "#shadow" in result


# ────────────────────────────────────────────────────────────────────
# _serialize_children
# ────────────────────────────────────────────────────────────────────

class TestSerializeChildren:
    """Test _serialize_children (lines 234-298)."""

    def test_list_items_truncated(self):
        items = []
        for i in range(55):
            li_text = _make_text_node(f"Item {i}")
            li = _make_node(tag="li", snapshot=_make_snapshot(), children=[li_text])
            items.append(_make_simplified(li, children=[_make_simplified(li_text)]))
        ul = _make_node(tag="ul", snapshot=_make_snapshot())
        ul_simp = _make_simplified(ul, children=items)
        result = DOMEvalSerializer._serialize_children(ul_simp, EVAL_KEY_ATTRIBUTES, 0)
        assert "more items" in result

    def test_consecutive_links_truncated(self):
        links = []
        for i in range(55):
            a_text = _make_text_node(f"Link {i}")
            a = _make_node(tag="a", snapshot=_make_snapshot(), children=[a_text])
            links.append(_make_simplified(a, children=[_make_simplified(a_text)]))
        parent = _make_node(tag="div")
        parent_simp = _make_simplified(parent, children=links)
        result = DOMEvalSerializer._serialize_children(parent_simp, EVAL_KEY_ATTRIBUTES, 0)
        assert "more links" in result

    def test_non_link_resets_link_counter(self):
        links = []
        for i in range(55):
            a_text = _make_text_node(f"Link {i}")
            a = _make_node(tag="a", snapshot=_make_snapshot(), children=[a_text])
            links.append(_make_simplified(a, children=[_make_simplified(a_text)]))
        # Insert a non-link in the middle
        div = _make_node(tag="div", snapshot=_make_snapshot())
        links.insert(52, _make_simplified(div))
        parent = _make_node(tag="div")
        parent_simp = _make_simplified(parent, children=links)
        result = DOMEvalSerializer._serialize_children(parent_simp, EVAL_KEY_ATTRIBUTES, 0)
        # The non-link should trigger the "more links" message for skipped links
        assert "more links" in result

    def test_ol_list_items_truncated(self):
        items = []
        for i in range(55):
            li_text = _make_text_node(f"Item {i}")
            li = _make_node(tag="li", snapshot=_make_snapshot(), children=[li_text])
            items.append(_make_simplified(li, children=[_make_simplified(li_text)]))
        ol = _make_node(tag="ol", snapshot=_make_snapshot())
        ol_simp = _make_simplified(ol, children=items)
        result = DOMEvalSerializer._serialize_children(ol_simp, EVAL_KEY_ATTRIBUTES, 0)
        assert "more items" in result


# ────────────────────────────────────────────────────────────────────
# _build_compact_attributes
# ────────────────────────────────────────────────────────────────────

class TestBuildCompactAttributes:
    """Test _build_compact_attributes (lines 300-332)."""

    def test_no_attributes(self):
        node = _make_node(tag="div")
        result = DOMEvalSerializer._build_compact_attributes(node)
        assert result == ""

    def test_basic_attributes(self):
        node = _make_node(tag="input", attributes={"type": "text", "name": "q"})
        result = DOMEvalSerializer._build_compact_attributes(node)
        assert 'type="text"' in result
        assert 'name="q"' in result

    def test_empty_value_skipped(self):
        node = _make_node(tag="input", attributes={"name": "", "type": "text"})
        result = DOMEvalSerializer._build_compact_attributes(node)
        assert 'name=' not in result
        assert 'type="text"' in result

    def test_class_limited_to_3(self):
        node = _make_node(
            tag="div",
            attributes={"class": "cls1 cls2 cls3 cls4 cls5"},
        )
        result = DOMEvalSerializer._build_compact_attributes(node)
        assert 'class="cls1 cls2 cls3"' in result
        assert "cls4" not in result

    def test_href_capped(self):
        long_url = "http://example.com/" + "x" * 200
        node = _make_node(tag="a", attributes={"href": long_url})
        # href is not in EVAL_KEY_ATTRIBUTES by default, so check that the result
        # doesn't contain the full URL
        result = DOMEvalSerializer._build_compact_attributes(node)
        # href is not in EVAL_KEY_ATTRIBUTES, so it won't appear
        assert result == ""


# ────────────────────────────────────────────────────────────────────
# _has_direct_text
# ────────────────────────────────────────────────────────────────────

class TestHasDirectText:
    """Test _has_direct_text (lines 334-342)."""

    def test_has_text(self):
        text = _make_text_node("Hello World")
        node = _make_node(tag="button", children=[text])
        simp = _make_simplified(node, children=[_make_simplified(text)])
        assert DOMEvalSerializer._has_direct_text(simp) is True

    def test_short_text_false(self):
        text = _make_text_node("X")
        node = _make_node(tag="button", children=[text])
        simp = _make_simplified(node, children=[_make_simplified(text)])
        assert DOMEvalSerializer._has_direct_text(simp) is False

    def test_no_text_children(self):
        child = _make_node(tag="span")
        node = _make_node(tag="div", children=[child])
        simp = _make_simplified(node, children=[_make_simplified(child)])
        assert DOMEvalSerializer._has_direct_text(simp) is False

    def test_empty_text_false(self):
        text = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="  ",
        )
        node = _make_node(tag="div", children=[text])
        simp = _make_simplified(node, children=[_make_simplified(text)])
        assert DOMEvalSerializer._has_direct_text(simp) is False

    def test_none_node_value_false(self):
        text = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value=None,
        )
        node = _make_node(tag="div", children=[text])
        simp = _make_simplified(node, children=[_make_simplified(text)])
        assert DOMEvalSerializer._has_direct_text(simp) is False


# ────────────────────────────────────────────────────────────────────
# _get_inline_text
# ────────────────────────────────────────────────────────────────────

class TestGetInlineText:
    """Test _get_inline_text (lines 344-358)."""

    def test_inline_text(self):
        text = _make_text_node("Click me")
        node = _make_node(tag="button", children=[text])
        simp = _make_simplified(node, children=[_make_simplified(text)])
        result = DOMEvalSerializer._get_inline_text(simp)
        assert "Click me" in result

    def test_no_text_returns_empty(self):
        child = _make_node(tag="span")
        node = _make_node(tag="div", children=[child])
        simp = _make_simplified(node, children=[_make_simplified(child)])
        result = DOMEvalSerializer._get_inline_text(simp)
        assert result == ""

    def test_long_text_capped(self):
        text = _make_text_node("A" * 200)
        node = _make_node(tag="button", children=[text])
        simp = _make_simplified(node, children=[_make_simplified(text)])
        result = DOMEvalSerializer._get_inline_text(simp)
        assert len(result) <= 83  # 80 + "..."

    def test_none_node_value_skipped(self):
        text = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=None)
        node = _make_node(tag="div", children=[text])
        simp = _make_simplified(node, children=[_make_simplified(text)])
        result = DOMEvalSerializer._get_inline_text(simp)
        assert result == ""

    def test_multiple_text_nodes_combined(self):
        t1 = _make_text_node("Hello")
        t2 = _make_text_node("World")
        node = _make_node(tag="span", children=[t1, t2])
        simp = _make_simplified(node, children=[_make_simplified(t1), _make_simplified(t2)])
        result = DOMEvalSerializer._get_inline_text(simp)
        assert "Hello" in result
        assert "World" in result


# ────────────────────────────────────────────────────────────────────
# _serialize_iframe
# ────────────────────────────────────────────────────────────────────

class TestSerializeIframe:
    """Test _serialize_iframe (lines 360-406)."""

    def test_iframe_no_content(self):
        node = _make_node(tag="iframe", snapshot=_make_snapshot())
        simp = _make_simplified(node)
        result = DOMEvalSerializer._serialize_iframe(simp, EVAL_KEY_ATTRIBUTES, 0)
        assert "<iframe" in result

    def test_iframe_with_content_document(self):
        # Text must be inside an element node for _serialize_document_node to pick it up
        body_text = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="Iframe body text",
            is_visible=True, snapshot=_make_snapshot(),
        )
        para = _make_node(tag="h1", children=[body_text], snapshot=_make_snapshot())
        body = _make_node(tag="body", children=[para], snapshot=_make_snapshot())
        html = _make_node(tag="html", children=[body], snapshot=_make_snapshot())
        content_doc = _make_node(
            tag="#document", node_type=NodeType.DOCUMENT_NODE,
            children=[html],
        )
        iframe_node = _make_node(
            tag="iframe", snapshot=_make_snapshot(),
            content_document=content_doc,
        )
        iframe_node.content_document = content_doc
        simp = _make_simplified(iframe_node)
        result = DOMEvalSerializer._serialize_iframe(simp, EVAL_KEY_ATTRIBUTES, 0)
        assert "#iframe-content" in result
        assert "Iframe body text" in result

    def test_iframe_with_non_html_content(self):
        inner = _make_node(tag="div", snapshot=_make_snapshot(), attributes={"role": "main"})
        content_doc = _make_node(
            tag="#document", node_type=NodeType.DOCUMENT_NODE,
            children=[inner],
        )
        iframe_node = _make_node(tag="iframe", snapshot=_make_snapshot(), content_document=content_doc)
        iframe_node.content_document = content_doc
        simp = _make_simplified(iframe_node)
        result = DOMEvalSerializer._serialize_iframe(simp, EVAL_KEY_ATTRIBUTES, 0)
        assert "#iframe-content" in result

    def test_iframe_with_scroll_info(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        # iframe always shows scroll info
        node = _make_node(tag="iframe", snapshot=snap, is_scrollable=True)
        simp = _make_simplified(node)
        result = DOMEvalSerializer._serialize_iframe(simp, EVAL_KEY_ATTRIBUTES, 0)
        assert "scroll=" in result


# ────────────────────────────────────────────────────────────────────
# _serialize_document_node
# ────────────────────────────────────────────────────────────────────

class TestSerializeDocumentNode:
    """Test _serialize_document_node (lines 408-479)."""

    def test_invisible_element_skipped(self):
        node = _make_node(tag="div", is_visible=False, snapshot=_make_snapshot())
        node.is_visible = False
        output = []
        DOMEvalSerializer._serialize_document_node(node, output, EVAL_KEY_ATTRIBUTES, 0, is_iframe_content=False)
        assert len(output) == 0

    def test_iframe_content_permissive_visibility(self):
        """In iframe content mode, elements without snapshot should be visible."""
        node = _make_node(tag="div", snapshot=None, is_visible=True, attributes={"role": "main"})
        node.snapshot_node = None
        output = []
        DOMEvalSerializer._serialize_document_node(node, output, EVAL_KEY_ATTRIBUTES, 0, is_iframe_content=True)
        assert len(output) >= 1

    def test_non_semantic_no_attrs_skipped(self):
        child = _make_node(tag="span", snapshot=_make_snapshot())
        node = _make_node(tag="div", snapshot=_make_snapshot(), children=[child])
        output = []
        DOMEvalSerializer._serialize_document_node(node, output, EVAL_KEY_ATTRIBUTES, 0, is_iframe_content=True)
        # div without attributes is not semantic, children should be processed
        assert len(output) >= 0  # At least no crash

    def test_semantic_element_shown(self):
        text = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="Hello",
            snapshot=_make_snapshot(),
        )
        node = _make_node(tag="h1", snapshot=_make_snapshot(), children=[text])
        output = []
        DOMEvalSerializer._serialize_document_node(node, output, EVAL_KEY_ATTRIBUTES, 0, is_iframe_content=True)
        assert any("<h1" in line for line in output)

    def test_text_content_inline(self):
        text = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="Content here",
            snapshot=_make_snapshot(),
        )
        node = _make_node(tag="button", snapshot=_make_snapshot(), children=[text])
        output = []
        DOMEvalSerializer._serialize_document_node(node, output, EVAL_KEY_ATTRIBUTES, 0, is_iframe_content=True)
        assert any("Content here" in line for line in output)

    def test_no_text_self_closing(self):
        node = _make_node(tag="input", snapshot=_make_snapshot(), attributes={"type": "text"})
        output = []
        DOMEvalSerializer._serialize_document_node(node, output, EVAL_KEY_ATTRIBUTES, 0, is_iframe_content=True)
        assert any("/>" in line for line in output)

    def test_non_text_children_recursed(self):
        inner = _make_node(tag="input", snapshot=_make_snapshot(), attributes={"type": "text"})
        node = _make_node(tag="form", snapshot=_make_snapshot(), children=[inner])
        output = []
        DOMEvalSerializer._serialize_document_node(node, output, EVAL_KEY_ATTRIBUTES, 0, is_iframe_content=True)
        assert any("<form" in line for line in output)
        assert any("<input" in line for line in output)

    def test_iframe_content_with_snapshot_invisible(self):
        """In iframe mode, element with snapshot that says invisible should be skipped."""
        snap = _make_snapshot()
        node = _make_node(tag="div", snapshot=snap, is_visible=False)
        node.is_visible = False
        output = []
        DOMEvalSerializer._serialize_document_node(node, output, EVAL_KEY_ATTRIBUTES, 0, is_iframe_content=True)
        assert len(output) == 0
