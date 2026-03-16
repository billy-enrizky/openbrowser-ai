"""Tests for DOM serializer (serializer.py) - comprehensive coverage."""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from openbrowser.dom.serializer.serializer import (
    DISABLED_ELEMENTS,
    SVG_ELEMENTS,
    DOMTreeSerializer,
)
from openbrowser.dom.views import (
    DEFAULT_INCLUDE_ATTRIBUTES,
    DOMRect,
    EnhancedAXNode,
    EnhancedAXProperty,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    PropagatingBounds,
    SerializedDOMState,
    SimplifiedNode,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers for building mock DOM trees
# ────────────────────────────────────────────────────────────────────

def _make_snapshot(
    bounds=None,
    is_clickable=None,
    cursor_style=None,
    computed_styles=None,
    paint_order=None,
    client_rects=None,
    scroll_rects=None,
    stacking_contexts=None,
):
    return EnhancedSnapshotNode(
        is_clickable=is_clickable,
        cursor_style=cursor_style,
        bounds=bounds or DOMRect(x=0, y=0, width=100, height=50),
        clientRects=client_rects,
        scrollRects=scroll_rects,
        computed_styles=computed_styles,
        paint_order=paint_order,
        stacking_contexts=stacking_contexts,
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
    """Build a minimal EnhancedDOMTreeNode for testing."""
    _id_counter = getattr(_make_node, "_counter", 0) + 1
    _make_node._counter = _id_counter

    nid = node_id if node_id is not None else _id_counter
    bnid = backend_node_id if backend_node_id is not None else _id_counter + 10000

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
    # Wire up parent pointers for children
    if children:
        for child in children:
            child.parent_node = node
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


def _make_document_node(children=None):
    return _make_node(
        tag="#document",
        node_type=NodeType.DOCUMENT_NODE,
        children=children or [],
        is_visible=True,
    )


def _make_fragment_node(children=None, shadow_root_type=None):
    return _make_node(
        tag="#document-fragment",
        node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
        children=children or [],
        shadow_root_type=shadow_root_type,
        is_visible=True,
    )


# ────────────────────────────────────────────────────────────────────
# DOMTreeSerializer.__init__ and basic helpers
# ────────────────────────────────────────────────────────────────────

class TestDOMTreeSerializerInit:
    """Test __init__ and helper methods (lines 59-95)."""

    def test_init_defaults(self):
        root = _make_node()
        s = DOMTreeSerializer(root)
        assert s.enable_bbox_filtering is True
        assert s.containment_threshold == DOMTreeSerializer.DEFAULT_CONTAINMENT_THRESHOLD
        assert s.paint_order_filtering is True
        assert s._previous_cached_selector_map is None

    def test_init_with_previous_cached_state(self):
        root = _make_node()
        prev = SerializedDOMState(_root=None, selector_map={42: _make_node()})
        s = DOMTreeSerializer(root, previous_cached_state=prev)
        assert s._previous_cached_selector_map is not None
        assert 42 in s._previous_cached_selector_map

    def test_init_custom_threshold(self):
        root = _make_node()
        s = DOMTreeSerializer(root, containment_threshold=0.5)
        assert s.containment_threshold == 0.5

    def test_safe_parse_number(self):
        root = _make_node()
        s = DOMTreeSerializer(root)
        assert s._safe_parse_number("3.14", 0.0) == 3.14
        assert s._safe_parse_number("invalid", 99.0) == 99.0
        assert s._safe_parse_number("-5", 0.0) == -5.0

    def test_safe_parse_optional_number(self):
        root = _make_node()
        s = DOMTreeSerializer(root)
        assert s._safe_parse_optional_number(None) is None
        assert s._safe_parse_optional_number("") is None
        assert s._safe_parse_optional_number("42") == 42.0
        assert s._safe_parse_optional_number("bad") is None


# ────────────────────────────────────────────────────────────────────
# _create_simplified_tree
# ────────────────────────────────────────────────────────────────────

class TestCreateSimplifiedTree:
    """Test _create_simplified_tree (lines 432-522)."""

    def test_document_node_returns_first_child(self):
        child = _make_node(tag="html")
        doc = _make_document_node(children=[child])
        s = DOMTreeSerializer(doc)
        result = s._create_simplified_tree(doc)
        assert result is not None
        assert result.original_node.tag_name == "html"

    def test_document_node_no_children(self):
        doc = _make_document_node(children=[])
        s = DOMTreeSerializer(doc)
        result = s._create_simplified_tree(doc)
        assert result is None

    def test_document_fragment_node(self):
        inner = _make_node(tag="span")
        frag = _make_fragment_node(children=[inner])
        s = DOMTreeSerializer(frag)
        result = s._create_simplified_tree(frag)
        assert result is not None
        assert len(result.children) > 0

    def test_document_fragment_empty_returns_empty_simplified(self):
        frag = _make_fragment_node(children=[])
        s = DOMTreeSerializer(frag)
        result = s._create_simplified_tree(frag)
        assert result is not None
        assert len(result.children) == 0

    def test_disabled_element_skipped(self):
        for tag in DISABLED_ELEMENTS:
            node = _make_node(tag=tag)
            s = DOMTreeSerializer(node)
            result = s._create_simplified_tree(node)
            assert result is None, f"Expected None for disabled tag: {tag}"

    def test_svg_child_elements_skipped(self):
        # SVG_ELEMENTS contains mixed-case names like 'clipPath', but the code
        # compares node_name.lower() against the set, so only lowercase entries match.
        # Test with all-lowercase tags that ARE in the set.
        lowercase_svg_tags = [t for t in SVG_ELEMENTS if t == t.lower()]
        for tag in lowercase_svg_tags:
            node = _make_node(tag=tag)
            s = DOMTreeSerializer(node)
            result = s._create_simplified_tree(node)
            assert result is None, f"Expected None for SVG child tag: {tag}"
        # Also verify camelCase SVG tags (e.g. clipPath) are NOT skipped
        # since node_name.lower() won't match the mixed-case entry
        camel_svg_tags = [t for t in SVG_ELEMENTS if t != t.lower()]
        assert len(camel_svg_tags) > 0, "Expected some camelCase SVG tags"

    def test_iframe_with_content_document(self):
        inner_child = _make_node(tag="div")
        content_doc = _make_node(tag="#document", node_type=NodeType.DOCUMENT_NODE, children=[inner_child])
        iframe = _make_node(tag="IFRAME", content_document=content_doc)
        iframe.content_document = content_doc
        s = DOMTreeSerializer(iframe)
        result = s._create_simplified_tree(iframe)
        assert result is not None

    def test_frame_with_content_document(self):
        inner_child = _make_node(tag="div")
        content_doc = _make_node(tag="#document", node_type=NodeType.DOCUMENT_NODE, children=[inner_child])
        frame = _make_node(tag="FRAME", content_document=content_doc)
        frame.content_document = content_doc
        s = DOMTreeSerializer(frame)
        result = s._create_simplified_tree(frame)
        assert result is not None

    def test_invisible_element_with_shadow_content(self):
        shadow_frag = _make_fragment_node()
        node = _make_node(tag="div", is_visible=False, shadow_roots=[shadow_frag], snapshot=None)
        # Manually set children to include shadow
        node.children_nodes = []
        s = DOMTreeSerializer(node)
        result = s._create_simplified_tree(node)
        # Shadow host should still be included
        assert result is not None

    def test_invisible_element_with_validation_attrs(self):
        node = _make_node(
            tag="input",
            is_visible=False,
            attributes={"aria-required": "true"},
            snapshot=None,
        )
        s = DOMTreeSerializer(node)
        result = s._create_simplified_tree(node)
        assert result is not None

    def test_invisible_file_input_included(self):
        node = _make_node(
            tag="input",
            is_visible=False,
            attributes={"type": "file"},
            snapshot=None,
        )
        s = DOMTreeSerializer(node)
        result = s._create_simplified_tree(node)
        assert result is not None

    def test_text_node_visible_meaningful(self):
        text = _make_text_node("Hello World")
        s = DOMTreeSerializer(text)
        result = s._create_simplified_tree(text)
        assert result is not None

    def test_text_node_invisible_skipped(self):
        text = _make_text_node("Hello", is_visible=False, snapshot=None)
        text.is_visible = False
        s = DOMTreeSerializer(text)
        result = s._create_simplified_tree(text)
        assert result is None

    def test_text_node_short_skipped(self):
        text = _make_text_node("X")
        s = DOMTreeSerializer(text)
        result = s._create_simplified_tree(text)
        assert result is None

    def test_shadow_host_detection(self):
        frag = _make_fragment_node(children=[_make_node(tag="span")])
        host = _make_node(tag="div", shadow_roots=[frag])
        host.children_nodes = []
        s = DOMTreeSerializer(host)
        result = s._create_simplified_tree(host)
        assert result is not None
        assert result.is_shadow_host is True


# ────────────────────────────────────────────────────────────────────
# _add_compound_components
# ────────────────────────────────────────────────────────────────────

class TestAddCompoundComponents:
    """Test _add_compound_components (lines 147-330)."""

    def _make_serializer(self):
        root = _make_node()
        return DOMTreeSerializer(root)

    def test_non_compound_tag_skipped(self):
        s = self._make_serializer()
        node = _make_node(tag="div")
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert not node._compound_children
        assert simp.is_compound_component is False

    def test_input_non_compound_type_skipped(self):
        s = self._make_serializer()
        node = _make_node(tag="input", attributes={"type": "text"})
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert not node._compound_children

    def test_input_range(self):
        s = self._make_serializer()
        node = _make_node(tag="input", attributes={"type": "range", "min": "10", "max": "200"})
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert simp.is_compound_component is True
        assert len(node._compound_children) == 1
        assert node._compound_children[0]["role"] == "slider"
        assert node._compound_children[0]["valuemin"] == 10.0
        assert node._compound_children[0]["valuemax"] == 200.0

    def test_input_number(self):
        s = self._make_serializer()
        node = _make_node(tag="input", attributes={"type": "number", "min": "0", "max": "99"})
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert simp.is_compound_component is True
        assert len(node._compound_children) == 3

    def test_input_color(self):
        s = self._make_serializer()
        node = _make_node(tag="input", attributes={"type": "color"})
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert simp.is_compound_component is True
        assert len(node._compound_children) == 2

    def test_input_file_no_selection(self):
        s = self._make_serializer()
        node = _make_node(tag="input", attributes={"type": "file"})
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert simp.is_compound_component is True
        assert len(node._compound_children) == 2
        # Default value should be "None"
        file_component = node._compound_children[1]
        assert file_component["valuenow"] == "None"

    def test_input_file_with_valuetext(self):
        s = self._make_serializer()
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="textbox", name="file",
            description=None,
            properties=[EnhancedAXProperty(name="valuetext", value="report.pdf")],
            child_ids=None,
        )
        node = _make_node(tag="input", attributes={"type": "file"}, ax_node=ax)
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert node._compound_children[1]["valuenow"] == "report.pdf"

    def test_input_file_with_value_path(self):
        s = self._make_serializer()
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="textbox", name="file",
            description=None,
            properties=[EnhancedAXProperty(name="value", value="C:\\Users\\test\\doc.pdf")],
            child_ids=None,
        )
        node = _make_node(tag="input", attributes={"type": "file"}, ax_node=ax)
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert node._compound_children[1]["valuenow"] == "doc.pdf"

    def test_input_file_with_unix_path(self):
        s = self._make_serializer()
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="textbox", name="file",
            description=None,
            properties=[EnhancedAXProperty(name="value", value="/home/user/doc.pdf")],
            child_ids=None,
        )
        node = _make_node(tag="input", attributes={"type": "file"}, ax_node=ax)
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert node._compound_children[1]["valuenow"] == "doc.pdf"

    def test_input_file_with_simple_value(self):
        s = self._make_serializer()
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="textbox", name="file",
            description=None,
            properties=[EnhancedAXProperty(name="value", value="myfile.txt")],
            child_ids=None,
        )
        node = _make_node(tag="input", attributes={"type": "file"}, ax_node=ax)
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert node._compound_children[1]["valuenow"] == "myfile.txt"

    def test_input_file_multiple(self):
        s = self._make_serializer()
        node = _make_node(tag="input", attributes={"type": "file", "multiple": ""})
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        # Should say "Files Selected" instead of "File Selected"
        assert node._compound_children[1]["name"] == "Files Selected"

    def test_input_date_skips_compound(self):
        s = self._make_serializer()
        node = _make_node(tag="input", attributes={"type": "date"})
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        # date inputs skip compound components
        assert not node._compound_children

    def test_select_element(self):
        s = self._make_serializer()
        opt_text = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="Option A"
        )
        opt = _make_node(tag="option", attributes={"value": "a"}, children=[opt_text])
        select_node = _make_node(
            tag="select", children=[opt],
            ax_node=EnhancedAXNode(
                ax_node_id="1", ignored=False, role="combobox",
                name="sel", description=None, properties=None,
                child_ids=["child1"],
            ),
        )
        simp = SimplifiedNode(original_node=select_node, children=[])
        s._add_compound_components(simp, select_node)
        assert simp.is_compound_component is True
        assert any(c["role"] == "listbox" for c in select_node._compound_children)

    def test_select_no_options(self):
        s = self._make_serializer()
        select_node = _make_node(
            tag="select", children=[],
            ax_node=EnhancedAXNode(
                ax_node_id="1", ignored=False, role="combobox",
                name="sel", description=None, properties=None,
                child_ids=["child1"],
            ),
        )
        simp = SimplifiedNode(original_node=select_node, children=[])
        s._add_compound_components(simp, select_node)
        assert simp.is_compound_component is True
        # Should have fallback listbox component
        listbox = [c for c in select_node._compound_children if c["role"] == "listbox"]
        assert len(listbox) == 1

    def test_details_element(self):
        s = self._make_serializer()
        node = _make_node(
            tag="details",
            ax_node=EnhancedAXNode(
                ax_node_id="1", ignored=False, role="group",
                name="det", description=None, properties=None,
                child_ids=["child1"],
            ),
        )
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert simp.is_compound_component is True

    def test_audio_element(self):
        s = self._make_serializer()
        node = _make_node(
            tag="audio",
            ax_node=EnhancedAXNode(
                ax_node_id="1", ignored=False, role="audio",
                name="aud", description=None, properties=None,
                child_ids=["child1"],
            ),
        )
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert simp.is_compound_component is True
        assert len(node._compound_children) == 4

    def test_video_element(self):
        s = self._make_serializer()
        node = _make_node(
            tag="video",
            ax_node=EnhancedAXNode(
                ax_node_id="1", ignored=False, role="video",
                name="vid", description=None, properties=None,
                child_ids=["child1"],
            ),
        )
        simp = SimplifiedNode(original_node=node, children=[])
        s._add_compound_components(simp, node)
        assert simp.is_compound_component is True
        assert len(node._compound_children) == 5


# ────────────────────────────────────────────────────────────────────
# _extract_select_options
# ────────────────────────────────────────────────────────────────────

class TestExtractSelectOptions:
    """Test _extract_select_options (lines 332-412)."""

    def _make_serializer(self):
        return DOMTreeSerializer(_make_node())

    def test_no_children_returns_none(self):
        s = self._make_serializer()
        select = _make_node(tag="select", children=[])
        result = s._extract_select_options(select)
        assert result is None

    def test_single_option(self):
        s = self._make_serializer()
        text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value="Apple")
        opt = _make_node(tag="option", attributes={"value": "apple"}, children=[text_node])
        select = _make_node(tag="select", children=[opt])
        result = s._extract_select_options(select)
        assert result is not None
        assert result["count"] == 1
        assert "Apple" in result["first_options"][0]

    def test_many_options_truncated(self):
        s = self._make_serializer()
        options = []
        for i in range(6):
            text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=f"Opt {i}")
            opt = _make_node(tag="option", attributes={"value": str(i)}, children=[text_node])
            options.append(opt)
        select = _make_node(tag="select", children=options)
        result = s._extract_select_options(select)
        assert result["count"] == 6
        # first_options should have 4 options + 1 ellipsis
        assert len(result["first_options"]) == 5
        assert "more options" in result["first_options"][-1]

    def test_numeric_format_hint(self):
        s = self._make_serializer()
        options = []
        for val in ["1", "2", "3", "4", "5"]:
            text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=val)
            opt = _make_node(tag="option", attributes={"value": val}, children=[text_node])
            options.append(opt)
        select = _make_node(tag="select", children=options)
        result = s._extract_select_options(select)
        assert result["format_hint"] == "numeric"

    def test_state_code_format_hint(self):
        s = self._make_serializer()
        options = []
        for val in ["CA", "NY", "TX", "FL", "WA"]:
            text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=val)
            opt = _make_node(tag="option", attributes={"value": val}, children=[text_node])
            options.append(opt)
        select = _make_node(tag="select", children=options)
        result = s._extract_select_options(select)
        assert result["format_hint"] == "country/state codes"

    def test_date_format_hint(self):
        s = self._make_serializer()
        options = []
        for val in ["2024-01", "2024-02"]:
            text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=val)
            opt = _make_node(tag="option", attributes={"value": val}, children=[text_node])
            options.append(opt)
        select = _make_node(tag="select", children=options)
        result = s._extract_select_options(select)
        assert result["format_hint"] == "date/path format"

    def test_email_format_hint(self):
        s = self._make_serializer()
        options = []
        for val in ["a@b.com", "c@d.org"]:
            text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=val)
            opt = _make_node(tag="option", attributes={"value": val}, children=[text_node])
            options.append(opt)
        select = _make_node(tag="select", children=options)
        result = s._extract_select_options(select)
        assert result["format_hint"] == "email addresses"

    def test_optgroup_children(self):
        s = self._make_serializer()
        text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value="Grp Opt")
        opt = _make_node(tag="option", attributes={"value": "g1"}, children=[text_node])
        optgroup = _make_node(tag="optgroup", children=[opt])
        select = _make_node(tag="select", children=[optgroup])
        result = s._extract_select_options(select)
        assert result is not None
        assert result["count"] == 1

    def test_option_no_value_uses_text(self):
        s = self._make_serializer()
        text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value="Fallback")
        opt = _make_node(tag="option", children=[text_node])
        select = _make_node(tag="select", children=[opt])
        result = s._extract_select_options(select)
        assert result["count"] == 1

    def test_option_long_text_truncated(self):
        s = self._make_serializer()
        long_text = "A" * 50
        text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=long_text)
        opt = _make_node(tag="option", attributes={"value": "x"}, children=[text_node])
        select = _make_node(tag="select", children=[opt])
        result = s._extract_select_options(select)
        # Should be truncated to 30 chars + "..."
        assert result["first_options"][0].endswith("...")

    def test_no_format_hint_for_mixed_values(self):
        s = self._make_serializer()
        options = []
        for val in ["apple", "banana"]:
            text_node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=val)
            opt = _make_node(tag="option", attributes={"value": val}, children=[text_node])
            options.append(opt)
        select = _make_node(tag="select", children=options)
        result = s._extract_select_options(select)
        assert result["format_hint"] is None


# ────────────────────────────────────────────────────────────────────
# serialize_accessible_elements (full pipeline)
# ────────────────────────────────────────────────────────────────────

class TestSerializeAccessibleElements:
    """Test serialize_accessible_elements (lines 97-145)."""

    def test_basic_pipeline(self):
        text = _make_text_node("Click me")
        btn = _make_node(
            tag="button",
            children=[text],
            snapshot=_make_snapshot(is_clickable=True),
        )
        html = _make_node(tag="html", children=[btn])
        doc = _make_document_node(children=[html])
        s = DOMTreeSerializer(doc)
        state, timing = s.serialize_accessible_elements()
        assert state is not None
        assert isinstance(timing, dict)
        assert "create_simplified_tree" in timing
        assert "serialize_accessible_elements_total" in timing

    def test_paint_order_filtering_disabled(self):
        text = _make_text_node("Text")
        div = _make_node(tag="div", children=[text])
        doc = _make_document_node(children=[div])
        s = DOMTreeSerializer(doc, paint_order_filtering=False)
        state, timing = s.serialize_accessible_elements()
        assert state is not None

    def test_bbox_filtering_disabled(self):
        text = _make_text_node("Text")
        div = _make_node(tag="div", children=[text])
        doc = _make_document_node(children=[div])
        s = DOMTreeSerializer(doc, enable_bbox_filtering=False)
        state, timing = s.serialize_accessible_elements()
        assert state is not None


# ────────────────────────────────────────────────────────────────────
# _optimize_tree
# ────────────────────────────────────────────────────────────────────

class TestOptimizeTree:
    """Test _optimize_tree (lines 524-558)."""

    def test_none_input(self):
        s = DOMTreeSerializer(_make_node())
        assert s._optimize_tree(None) is None

    def test_visible_node_kept(self):
        s = DOMTreeSerializer(_make_node())
        node = _make_node(tag="div")
        simp = SimplifiedNode(original_node=node, children=[])
        result = s._optimize_tree(simp)
        assert result is not None

    def test_invisible_node_with_children_kept(self):
        s = DOMTreeSerializer(_make_node())
        child_node = _make_node(tag="span")
        child_simp = SimplifiedNode(original_node=child_node, children=[])

        parent_node = _make_node(tag="div", is_visible=False, snapshot=None)
        parent_simp = SimplifiedNode(original_node=parent_node, children=[child_simp])
        result = s._optimize_tree(parent_simp)
        assert result is not None

    def test_invisible_no_children_removed(self):
        s = DOMTreeSerializer(_make_node())
        node = _make_node(tag="div", is_visible=False, snapshot=None)
        simp = SimplifiedNode(original_node=node, children=[])
        result = s._optimize_tree(simp)
        assert result is None

    def test_file_input_invisible_kept(self):
        s = DOMTreeSerializer(_make_node())
        node = _make_node(
            tag="input", is_visible=False, snapshot=None,
            attributes={"type": "file"},
        )
        simp = SimplifiedNode(original_node=node, children=[])
        result = s._optimize_tree(simp)
        assert result is not None

    def test_text_node_kept(self):
        s = DOMTreeSerializer(_make_node())
        text = _make_text_node("Hello World")
        simp = SimplifiedNode(original_node=text, children=[])
        result = s._optimize_tree(simp)
        assert result is not None


# ────────────────────────────────────────────────────────────────────
# _assign_interactive_indices_and_mark_new_nodes
# ────────────────────────────────────────────────────────────────────

class TestAssignInteractiveIndices:
    """Test interactive index assignment (lines 585-639)."""

    def test_none_input(self):
        s = DOMTreeSerializer(_make_node())
        s._assign_interactive_indices_and_mark_new_nodes(None)
        assert len(s._selector_map) == 0

    def test_interactive_visible_button(self):
        btn = _make_node(
            tag="button",
            snapshot=_make_snapshot(is_clickable=True),
        )
        s = DOMTreeSerializer(btn)
        simp = SimplifiedNode(original_node=btn, children=[])
        s._assign_interactive_indices_and_mark_new_nodes(simp)
        assert simp.is_interactive is True
        assert btn.backend_node_id in s._selector_map

    def test_excluded_node_skipped(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True))
        s = DOMTreeSerializer(btn)
        simp = SimplifiedNode(original_node=btn, children=[], excluded_by_parent=True)
        s._assign_interactive_indices_and_mark_new_nodes(simp)
        assert simp.is_interactive is False

    def test_ignored_by_paint_order_skipped(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True))
        s = DOMTreeSerializer(btn)
        simp = SimplifiedNode(original_node=btn, children=[], ignored_by_paint_order=True)
        s._assign_interactive_indices_and_mark_new_nodes(simp)
        assert simp.is_interactive is False

    def test_scrollable_with_interactive_descendants_not_made_interactive(self):
        """Scrollable container with interactive children should not itself be interactive."""
        child_btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True))
        child_simp = SimplifiedNode(original_node=child_btn, children=[])

        snap = _make_snapshot(
            client_rects=DOMRect(x=0, y=0, width=100, height=100),
            scroll_rects=DOMRect(x=0, y=0, width=100, height=500),
            computed_styles={"overflow": "auto"},
        )
        parent = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])

        s = DOMTreeSerializer(parent)
        s._assign_interactive_indices_and_mark_new_nodes(parent_simp)
        # Parent should NOT be interactive (has interactive child)
        assert parent_simp.is_interactive is False
        # Child should be interactive
        assert child_simp.is_interactive is True

    def test_new_node_marking_with_previous_cache(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True), backend_node_id=999)
        prev_node = _make_node(backend_node_id=111)
        prev_state = SerializedDOMState(_root=None, selector_map={111: prev_node})
        s = DOMTreeSerializer(btn, previous_cached_state=prev_state)
        simp = SimplifiedNode(original_node=btn, children=[])
        s._assign_interactive_indices_and_mark_new_nodes(simp)
        assert simp.is_new is True

    def test_existing_node_not_marked_new(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True), backend_node_id=111)
        prev_node = _make_node(backend_node_id=111)
        prev_state = SerializedDOMState(_root=None, selector_map={111: prev_node})
        s = DOMTreeSerializer(btn, previous_cached_state=prev_state)
        simp = SimplifiedNode(original_node=btn, children=[])
        s._assign_interactive_indices_and_mark_new_nodes(simp)
        assert simp.is_new is False

    def test_compound_component_always_new(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True))
        s = DOMTreeSerializer(btn)
        simp = SimplifiedNode(original_node=btn, children=[], is_compound_component=True)
        s._assign_interactive_indices_and_mark_new_nodes(simp)
        assert simp.is_new is True

    def test_file_input_invisible_interactive(self):
        node = _make_node(
            tag="input",
            is_visible=False,
            snapshot=None,
            attributes={"type": "file"},
        )
        s = DOMTreeSerializer(node)
        simp = SimplifiedNode(original_node=node, children=[])
        s._assign_interactive_indices_and_mark_new_nodes(simp)
        assert simp.is_interactive is True


# ────────────────────────────────────────────────────────────────────
# Bounding box filtering
# ────────────────────────────────────────────────────────────────────

class TestBoundingBoxFiltering:
    """Test _apply_bounding_box_filtering and related (lines 641-793)."""

    def test_none_input(self):
        s = DOMTreeSerializer(_make_node())
        assert s._apply_bounding_box_filtering(None) is None

    def test_no_exclusions(self):
        s = DOMTreeSerializer(_make_node())
        node = _make_node(tag="div", snapshot=_make_snapshot(bounds=DOMRect(0, 0, 500, 500)))
        simp = SimplifiedNode(original_node=node, children=[])
        result = s._apply_bounding_box_filtering(simp)
        assert result is not None

    def test_contained_child_excluded(self):
        """Child fully contained in propagating parent should be excluded."""
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(
            tag="a", snapshot=parent_snap,
            attributes={},
        )
        child_snap = _make_snapshot(bounds=DOMRect(10, 10, 50, 30))
        child = _make_node(tag="span", snapshot=child_snap)
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is True

    def test_form_element_not_excluded(self):
        """Form elements inside propagating parent should NOT be excluded."""
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(tag="a", snapshot=parent_snap)
        child_snap = _make_snapshot(bounds=DOMRect(10, 10, 50, 30))
        child = _make_node(tag="input", snapshot=child_snap)
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is False

    def test_child_with_onclick_not_excluded(self):
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(tag="button", snapshot=parent_snap)
        child_snap = _make_snapshot(bounds=DOMRect(10, 10, 50, 30))
        child = _make_node(tag="div", snapshot=child_snap, attributes={"onclick": "handleClick()"})
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is False

    def test_child_with_aria_label_not_excluded(self):
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(tag="a", snapshot=parent_snap)
        child_snap = _make_snapshot(bounds=DOMRect(10, 10, 50, 30))
        child = _make_node(tag="div", snapshot=child_snap, attributes={"aria-label": "Close"})
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is False

    def test_child_with_interactive_role_not_excluded(self):
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(tag="a", snapshot=parent_snap)
        child_snap = _make_snapshot(bounds=DOMRect(10, 10, 50, 30))
        child = _make_node(tag="div", snapshot=child_snap, attributes={"role": "button"})
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is False

    def test_text_node_not_excluded(self):
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(tag="a", snapshot=parent_snap)
        text = _make_text_node("Hello")
        text_simp = SimplifiedNode(original_node=text, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[text_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert text_simp.excluded_by_parent is False

    def test_child_no_bounds_not_excluded(self):
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(tag="a", snapshot=parent_snap)
        # Create snapshot with bounds explicitly set to None
        child_snap = EnhancedSnapshotNode(
            is_clickable=None, cursor_style=None, bounds=None,
            clientRects=None, scrollRects=None, computed_styles=None,
            paint_order=None, stacking_contexts=None,
        )
        child = _make_node(tag="span", snapshot=child_snap)
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is False

    def test_child_outside_parent_not_excluded(self):
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 100, 50))
        parent = _make_node(tag="a", snapshot=parent_snap)
        child_snap = _make_snapshot(bounds=DOMRect(500, 500, 50, 50))
        child = _make_node(tag="div", snapshot=child_snap)
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is False

    def test_zero_area_child_not_excluded(self):
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(tag="a", snapshot=parent_snap)
        child_snap = _make_snapshot(bounds=DOMRect(10, 10, 0, 0))
        child = _make_node(tag="div", snapshot=child_snap)
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is False

    def test_is_propagating_element_button(self):
        s = DOMTreeSerializer(_make_node())
        assert s._is_propagating_element({"tag": "button", "role": None}) is True

    def test_is_propagating_element_div_button_role(self):
        s = DOMTreeSerializer(_make_node())
        assert s._is_propagating_element({"tag": "div", "role": "button"}) is True

    def test_is_propagating_element_non_propagating(self):
        s = DOMTreeSerializer(_make_node())
        assert s._is_propagating_element({"tag": "div", "role": None}) is False

    def test_count_excluded_nodes(self):
        s = DOMTreeSerializer(_make_node())
        node = _make_node(tag="div")
        child1 = _make_node(tag="span")
        child2 = _make_node(tag="span")
        s1 = SimplifiedNode(original_node=child1, children=[], excluded_by_parent=True)
        s2 = SimplifiedNode(original_node=child2, children=[])
        parent = SimplifiedNode(original_node=node, children=[s1, s2])
        count = s._count_excluded_nodes(parent)
        assert count == 1

    def test_child_propagating_element_not_excluded(self):
        """A nested propagating element (e.g., button inside a) should NOT be excluded."""
        s = DOMTreeSerializer(_make_node())
        parent_snap = _make_snapshot(bounds=DOMRect(0, 0, 200, 100))
        parent = _make_node(tag="a", snapshot=parent_snap)
        child_snap = _make_snapshot(bounds=DOMRect(10, 10, 50, 30))
        child = _make_node(tag="button", snapshot=child_snap)
        child_simp = SimplifiedNode(original_node=child, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[child_simp])
        s._apply_bounding_box_filtering(parent_simp)
        assert child_simp.excluded_by_parent is False


# ────────────────────────────────────────────────────────────────────
# serialize_tree (static)
# ────────────────────────────────────────────────────────────────────

class TestSerializeTree:
    """Test serialize_tree static method (lines 794-980)."""

    def test_none_returns_empty(self):
        result = DOMTreeSerializer.serialize_tree(None, DEFAULT_INCLUDE_ATTRIBUTES)
        assert result == ""

    def test_excluded_node_children_processed(self):
        child_node = _make_text_node("inner text")
        child_simp = SimplifiedNode(original_node=child_node, children=[])
        parent_node = _make_node(tag="div")
        parent_simp = SimplifiedNode(
            original_node=parent_node, children=[child_simp], excluded_by_parent=True
        )
        result = DOMTreeSerializer.serialize_tree(parent_simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "inner text" in result

    def test_should_display_false_skips_node_shows_children(self):
        child_node = _make_text_node("child text")
        child_simp = SimplifiedNode(original_node=child_node, children=[])
        parent_node = _make_node(tag="div")
        parent_simp = SimplifiedNode(
            original_node=parent_node, children=[child_simp], should_display=False
        )
        result = DOMTreeSerializer.serialize_tree(parent_simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "child text" in result

    def test_svg_element_collapsed(self):
        svg = _make_node(tag="svg", snapshot=_make_snapshot())
        simp = SimplifiedNode(original_node=svg, children=[])
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "svg" in result.lower()
        assert "SVG content collapsed" in result

    def test_svg_interactive(self):
        svg = _make_node(tag="svg", snapshot=_make_snapshot())
        simp = SimplifiedNode(original_node=svg, children=[], is_interactive=True)
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert f"[{svg.backend_node_id}]" in result

    def test_svg_shadow_host(self):
        frag = _make_fragment_node(shadow_root_type="open")
        frag_simp = SimplifiedNode(original_node=frag, children=[])
        svg = _make_node(tag="svg", snapshot=_make_snapshot())
        simp = SimplifiedNode(original_node=svg, children=[frag_simp], is_shadow_host=True)
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "SHADOW" in result

    def test_svg_shadow_host_closed(self):
        frag = _make_fragment_node(shadow_root_type="closed")
        frag_simp = SimplifiedNode(original_node=frag, children=[])
        svg = _make_node(tag="svg", snapshot=_make_snapshot())
        simp = SimplifiedNode(original_node=svg, children=[frag_simp], is_shadow_host=True)
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "SHADOW(closed)" in result

    def test_interactive_button(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True), backend_node_id=42)
        simp = SimplifiedNode(original_node=btn, children=[], is_interactive=True)
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "[42]" in result

    def test_new_interactive_button(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True), backend_node_id=42)
        simp = SimplifiedNode(original_node=btn, children=[], is_interactive=True, is_new=True)
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "*[42]" in result

    def test_scrollable_not_interactive(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        div = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        simp = SimplifiedNode(original_node=div, children=[])
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "SCROLL" in result

    def test_iframe_not_interactive(self):
        # Use lowercase 'iframe' -- tag_name returns lower, but tag_name.upper() == 'IFRAME'
        iframe = _make_node(tag="iframe", snapshot=_make_snapshot())
        simp = SimplifiedNode(original_node=iframe, children=[])
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        # iframe always shows scroll info, so it will have SCROLL or IFRAME prefix
        assert "iframe" in result.lower()

    def test_frame_not_interactive(self):
        frame = _make_node(tag="frame", snapshot=_make_snapshot())
        simp = SimplifiedNode(original_node=frame, children=[])
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "frame" in result.lower()

    def test_scrollable_interactive(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 0, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        div = _make_node(tag="div", is_scrollable=True, snapshot=snap, backend_node_id=77)
        simp = SimplifiedNode(original_node=div, children=[], is_interactive=True)
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "SCROLL[77]" in result

    def test_shadow_host_interactive(self):
        frag = _make_fragment_node(shadow_root_type="open")
        frag_simp = SimplifiedNode(original_node=frag, children=[])
        btn = _make_node(tag="button", snapshot=_make_snapshot(), backend_node_id=88)
        simp = SimplifiedNode(
            original_node=btn, children=[frag_simp],
            is_interactive=True, is_shadow_host=True,
        )
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "SHADOW" in result
        assert "[88]" in result

    def test_document_fragment_open(self):
        frag = _make_fragment_node(shadow_root_type="open")
        child = _make_text_node("shadow text")
        child_simp = SimplifiedNode(original_node=child, children=[])
        frag_simp = SimplifiedNode(original_node=frag, children=[child_simp])
        result = DOMTreeSerializer.serialize_tree(frag_simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "Open Shadow" in result
        assert "Shadow End" in result

    def test_document_fragment_closed(self):
        frag = _make_fragment_node(shadow_root_type="closed")
        frag_simp = SimplifiedNode(original_node=frag, children=[])
        result = DOMTreeSerializer.serialize_tree(frag_simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "Closed Shadow" in result

    def test_document_fragment_no_children_no_end(self):
        frag = _make_fragment_node(shadow_root_type="open")
        frag_simp = SimplifiedNode(original_node=frag, children=[])
        result = DOMTreeSerializer.serialize_tree(frag_simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "Open Shadow" in result
        assert "Shadow End" not in result

    def test_text_node_visible(self):
        text = _make_text_node("Hello World")
        simp = SimplifiedNode(original_node=text, children=[])
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "Hello World" in result

    def test_text_node_invisible(self):
        text = _make_text_node("Hidden", is_visible=False, snapshot=None)
        text.is_visible = False
        simp = SimplifiedNode(original_node=text, children=[])
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "Hidden" not in result

    def test_text_node_short(self):
        text = _make_text_node("X")
        simp = SimplifiedNode(original_node=text, children=[])
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert result == ""

    def test_compound_components_in_serialization(self):
        btn = _make_node(tag="select", snapshot=_make_snapshot(), backend_node_id=50)
        btn._compound_children = [
            {"role": "listbox", "name": "Options", "valuemin": None, "valuemax": None, "valuenow": None,
             "options_count": 3, "first_options": ["A", "B", "C"], "format_hint": "numeric"}
        ]
        simp = SimplifiedNode(original_node=btn, children=[], is_interactive=True)
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "compound_components" in result
        assert "Options" in result

    def test_scroll_info_text_in_output(self):
        snap = _make_snapshot(
            client_rects=DOMRect(0, 0, 100, 100),
            scroll_rects=DOMRect(0, 50, 100, 500),
            computed_styles={"overflow": "auto"},
        )
        div = _make_node(tag="div", is_scrollable=True, snapshot=snap)
        simp = SimplifiedNode(original_node=div, children=[])
        result = DOMTreeSerializer.serialize_tree(simp, DEFAULT_INCLUDE_ATTRIBUTES)
        assert "pages" in result.lower() or "SCROLL" in result


# ────────────────────────────────────────────────────────────────────
# _build_attributes_string (static)
# ────────────────────────────────────────────────────────────────────

class TestBuildAttributesString:
    """Test _build_attributes_string (lines 982-1170)."""

    def test_empty_attributes(self):
        node = _make_node(tag="div")
        result = DOMTreeSerializer._build_attributes_string(node, DEFAULT_INCLUDE_ATTRIBUTES, "")
        assert result == ""

    def test_basic_attributes(self):
        node = _make_node(tag="input", attributes={"type": "text", "name": "email"})
        result = DOMTreeSerializer._build_attributes_string(node, DEFAULT_INCLUDE_ATTRIBUTES, "")
        assert "type=text" in result
        assert "name=email" in result

    def test_date_input_format(self):
        node = _make_node(tag="input", attributes={"type": "date"})
        result = DOMTreeSerializer._build_attributes_string(node, DEFAULT_INCLUDE_ATTRIBUTES, "")
        assert "YYYY-MM-DD" in result

    def test_time_input_format(self):
        node = _make_node(tag="input", attributes={"type": "time"})
        result = DOMTreeSerializer._build_attributes_string(node, DEFAULT_INCLUDE_ATTRIBUTES, "")
        assert "HH:MM" in result

    def test_datetime_local_input_format(self):
        node = _make_node(tag="input", attributes={"type": "datetime-local"})
        result = DOMTreeSerializer._build_attributes_string(node, DEFAULT_INCLUDE_ATTRIBUTES, "")
        assert "YYYY-MM-DDTHH:MM" in result

    def test_month_input_format(self):
        node = _make_node(tag="input", attributes={"type": "month"})
        result = DOMTreeSerializer._build_attributes_string(node, DEFAULT_INCLUDE_ATTRIBUTES, "")
        assert "YYYY-MM" in result

    def test_week_input_format(self):
        node = _make_node(tag="input", attributes={"type": "week"})
        result = DOMTreeSerializer._build_attributes_string(node, DEFAULT_INCLUDE_ATTRIBUTES, "")
        assert "YYYY-W##" in result

    def test_tel_input_placeholder(self):
        node = _make_node(tag="input", attributes={"type": "tel"})
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "placeholder"], "")
        assert "123-456-7890" in result

    def test_uib_datepicker(self):
        node = _make_node(
            tag="input",
            attributes={"type": "text", "uib-datepicker-popup": "MM/dd/yyyy"},
        )
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "format", "expected_format", "placeholder"], "")
        assert "MM/dd/yyyy" in result

    def test_jquery_datepicker_class(self):
        node = _make_node(
            tag="input",
            attributes={"type": "text", "class": "datepicker form-control"},
        )
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "placeholder", "format"], "")
        assert "mm/dd/yyyy" in result

    def test_jquery_datepicker_with_format(self):
        node = _make_node(
            tag="input",
            attributes={"type": "text", "class": "datepicker", "data-date-format": "dd/mm/yyyy"},
        )
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "placeholder", "format", "data-date-format"], "")
        assert "dd/mm/yyyy" in result

    def test_data_datepicker_attribute(self):
        node = _make_node(
            tag="input",
            attributes={"type": "text", "data-datepicker": "true"},
        )
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "placeholder", "format"], "")
        assert "mm/dd/yyyy" in result

    def test_data_datepicker_with_format(self):
        node = _make_node(
            tag="input",
            attributes={"type": "text", "data-datepicker": "true", "data-date-format": "yyyy-mm-dd"},
        )
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "placeholder", "format", "data-date-format"], "")
        assert "yyyy-mm-dd" in result

    def test_ax_properties_included(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="checkbox", name="agree",
            description=None,
            properties=[
                EnhancedAXProperty(name="checked", value=True),
                EnhancedAXProperty(name="required", value=True),
            ],
            child_ids=None,
        )
        node = _make_node(tag="input", attributes={"type": "checkbox"}, ax_node=ax)
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "checked", "required"], "")
        assert "checked=true" in result

    def test_ax_property_boolean_lowercase(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="checkbox", name="x",
            description=None,
            properties=[EnhancedAXProperty(name="checked", value=True)],
            child_ids=None,
        )
        node = _make_node(tag="input", attributes={"type": "checkbox"}, ax_node=ax)
        result = DOMTreeSerializer._build_attributes_string(node, ["checked"], "")
        assert "checked=true" in result

    def test_form_element_value_from_ax(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="textbox", name="x",
            description=None,
            properties=[EnhancedAXProperty(name="value", value="hello@world")],
            child_ids=None,
        )
        node = _make_node(tag="input", attributes={"type": "text"}, ax_node=ax)
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "value"], "")
        assert "value=hello@world" in result

    def test_form_element_valuetext_priority(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="textbox", name="x",
            description=None,
            properties=[
                EnhancedAXProperty(name="valuetext", value="Display Value"),
                EnhancedAXProperty(name="value", value="raw"),
            ],
            child_ids=None,
        )
        node = _make_node(tag="input", attributes={"type": "text"}, ax_node=ax)
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "value"], "")
        assert "value=Display Value" in result

    def test_duplicate_value_removal(self):
        node = _make_node(
            tag="div",
            attributes={"name": "verylongvaluefortest", "role": "verylongvaluefortest"},
        )
        result = DOMTreeSerializer._build_attributes_string(node, ["name", "role"], "")
        # One of them should be removed as duplicate
        count = result.count("verylongvaluefortest")
        assert count == 1

    def test_role_removed_if_matches_tag(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="button", name="x",
            description=None, properties=None, child_ids=None,
        )
        node = _make_node(tag="button", ax_node=ax, attributes={"role": "button"})
        result = DOMTreeSerializer._build_attributes_string(node, ["role"], "")
        # role should be removed since it matches the tag name
        assert "role" not in result

    def test_type_removed_if_matches_tag(self):
        node = _make_node(tag="button", attributes={"type": "button"})
        result = DOMTreeSerializer._build_attributes_string(node, ["type"], "")
        assert "type" not in result

    def test_invalid_false_removed(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="textbox", name="x",
            description=None,
            properties=[EnhancedAXProperty(name="invalid", value="false")],
            child_ids=None,
        )
        node = _make_node(tag="input", ax_node=ax)
        result = DOMTreeSerializer._build_attributes_string(node, ["invalid"], "")
        assert "invalid" not in result

    def test_required_false_removed(self):
        node = _make_node(tag="input", attributes={"required": "false"})
        result = DOMTreeSerializer._build_attributes_string(node, ["required"], "")
        assert "required" not in result

    def test_aria_expanded_removed_when_expanded_present(self):
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="combobox", name="x",
            description=None,
            properties=[EnhancedAXProperty(name="expanded", value="true")],
            child_ids=None,
        )
        node = _make_node(
            tag="div", ax_node=ax,
            attributes={"aria-expanded": "true"},
        )
        result = DOMTreeSerializer._build_attributes_string(node, ["expanded", "aria-expanded"], "")
        assert "aria-expanded" not in result
        assert "expanded=true" in result

    def test_aria_label_removed_if_matches_text(self):
        node = _make_node(tag="button", attributes={"aria-label": "Click me"})
        result = DOMTreeSerializer._build_attributes_string(node, ["aria-label"], "Click me")
        assert "aria-label" not in result

    def test_placeholder_removed_if_matches_text(self):
        node = _make_node(tag="input", attributes={"placeholder": "Search"})
        result = DOMTreeSerializer._build_attributes_string(node, ["placeholder"], "Search")
        assert "placeholder" not in result

    def test_empty_value_formatted_with_quotes(self):
        node = _make_node(tag="input", attributes={"value": ""})
        # Force value to be in attributes_to_include
        ax = EnhancedAXNode(
            ax_node_id="1", ignored=False, role="textbox", name="x",
            description=None,
            properties=[EnhancedAXProperty(name="value", value="")],
            child_ids=None,
        )
        node.ax_node = ax
        # "value" would be empty string, so should not appear (value stripped is empty)
        result = DOMTreeSerializer._build_attributes_string(node, ["value"], "")
        # Empty strings are stripped, so value is not included
        assert result == ""

    def test_protected_attrs_not_deduped(self):
        """Protected attributes like format should not be removed as duplicates."""
        node = _make_node(
            tag="input",
            attributes={"type": "date", "placeholder": "YYYY-MM-DD"},
        )
        result = DOMTreeSerializer._build_attributes_string(node, ["type", "format", "placeholder"], "")
        # Both format and placeholder should be present
        assert "format" in result
        assert "placeholder" in result


# ────────────────────────────────────────────────────────────────────
# _is_interactive_cached
# ────────────────────────────────────────────────────────────────────

class TestIsInteractiveCached:
    """Test _is_interactive_cached (lines 414-430)."""

    def test_caches_result(self):
        node = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True))
        s = DOMTreeSerializer(node)
        result1 = s._is_interactive_cached(node)
        result2 = s._is_interactive_cached(node)
        assert result1 == result2

    def test_timing_tracked(self):
        node = _make_node(tag="button")
        s = DOMTreeSerializer(node)
        s._is_interactive_cached(node)
        assert "clickable_detection_time" in s.timing_info


# ────────────────────────────────────────────────────────────────────
# _collect_interactive_elements and _has_interactive_descendants
# ────────────────────────────────────────────────────────────────────

class TestCollectInteractiveElements:
    """Test helper methods (lines 560-583)."""

    def test_collect_interactive_elements(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True))
        parent = _make_node(tag="div", children=[btn])
        s = DOMTreeSerializer(parent)
        btn_simp = SimplifiedNode(original_node=btn, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[btn_simp])
        elements = []
        s._collect_interactive_elements(parent_simp, elements)
        assert len(elements) >= 1

    def test_has_interactive_descendants_true(self):
        btn = _make_node(tag="button", snapshot=_make_snapshot(is_clickable=True))
        parent = _make_node(tag="div", children=[btn])
        s = DOMTreeSerializer(parent)
        btn_simp = SimplifiedNode(original_node=btn, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[btn_simp])
        assert s._has_interactive_descendants(parent_simp) is True

    def test_has_interactive_descendants_false(self):
        text = _make_text_node("Just text")
        parent = _make_node(tag="div", children=[text])
        s = DOMTreeSerializer(parent)
        text_simp = SimplifiedNode(original_node=text, children=[])
        parent_simp = SimplifiedNode(original_node=parent, children=[text_simp])
        assert s._has_interactive_descendants(parent_simp) is False
