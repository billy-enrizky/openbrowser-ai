"""Tests for DOMEvalSerializer.

This module provides test coverage for the DOMEvalSerializer class,
which serializes DOM trees for evaluation and analysis purposes.
It validates:

    - Basic DOM tree serialization with proper HTML structure
    - List truncation at 50 items with truncation messages
    - SVG content collapsing for compact output
    - Compact attribute output for form elements
    - Iframe content serialization
    - Inline text limiting to 80 characters
    - Handling of empty or None nodes

The DOMEvalSerializer produces output optimized for LLM evaluation
of page content and structure, with appropriate truncation and
collapsing for large or complex elements.
"""

import pytest

from src.openbrowser.browser.dom.serializer.eval_serializer import DOMEvalSerializer
from src.openbrowser.browser.dom.views import (
    DEFAULT_INCLUDE_ATTRIBUTES,
    EnhancedDOMTreeNode,
    NodeType,
    SimplifiedNode,
)


@pytest.fixture
def sample_element_node():
    """Create a sample element node for testing.

    Returns:
        EnhancedDOMTreeNode: A div element with id and class attributes.
    """
    return EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={"id": "test", "class": "container"},
    )


@pytest.fixture
def sample_simplified_node(sample_element_node):
    """Create a sample simplified node for serialization tests.

    Args:
        sample_element_node: The element node fixture to wrap.

    Returns:
        SimplifiedNode: A simplified node wrapping the sample element.
    """
    return SimplifiedNode(original_node=sample_element_node, children=[])


def test_serialize_tree_basic(sample_simplified_node):
    """Test basic serialization."""
    result = DOMEvalSerializer.serialize_tree(sample_simplified_node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<div" in result
    assert 'id="test"' in result


def test_list_truncation():
    """Test that lists are truncated at 50 items."""
    ul_node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="ul",
        node_value="",
        attributes={},
    )
    
    # Create 60 list items
    children = []
    for i in range(60):
        li_node = EnhancedDOMTreeNode(
            node_id=i + 2,
            backend_node_id=100 + i + 1,
            node_type=NodeType.ELEMENT_NODE,
            node_name="li",
            node_value="",
            attributes={},
        )
        children.append(SimplifiedNode(original_node=li_node, children=[]))
    
    node = SimplifiedNode(original_node=ul_node, children=children)
    result = DOMEvalSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    
    # Should contain truncation message
    assert "truncated" in result.lower() or "more items" in result.lower()


def test_svg_collapsing():
    """Test that SVG elements are collapsed."""
    svg_node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="svg",
        node_value="",
        attributes={},
    )
    
    # Add SVG child elements
    path_node = EnhancedDOMTreeNode(
        node_id=2,
        backend_node_id=101,
        node_type=NodeType.ELEMENT_NODE,
        node_name="path",
        node_value="",
        attributes={},
    )
    children = [SimplifiedNode(original_node=path_node, children=[])]
    node = SimplifiedNode(original_node=svg_node, children=children)
    
    result = DOMEvalSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<svg" in result
    assert "SVG content collapsed" in result
    # SVG child elements should not appear
    assert "<path" not in result


def test_compact_attributes():
    """Test that attributes are compact."""
    node_with_attrs = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="input",
        node_value="",
        attributes={
            "id": "test",
            "type": "text",
            "placeholder": "Enter text",
            "required": "true",
        },
    )
    node = SimplifiedNode(original_node=node_with_attrs, children=[])
    result = DOMEvalSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert 'id="test"' in result
    assert 'type="text"' in result
    assert 'placeholder="Enter text"' in result


def test_iframe_content():
    """Test iframe content serialization."""
    iframe_node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="iframe",
        node_value="",
        attributes={"src": "https://example.com"},
    )
    node = SimplifiedNode(original_node=iframe_node, children=[])
    result = DOMEvalSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<iframe" in result


def test_inline_text_limit():
    """Test that inline text is limited to 80 chars."""
    text_node = EnhancedDOMTreeNode(
        node_id=2,
        backend_node_id=101,
        node_type=NodeType.TEXT_NODE,
        node_name="#text",
        node_value="A" * 100,  # 100 characters
        attributes={},
    )
    text_simplified = SimplifiedNode(original_node=text_node, children=[])
    div_node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
    )
    div_node.children_nodes = [text_node]
    node = SimplifiedNode(original_node=div_node, children=[text_simplified])
    
    result = DOMEvalSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    # Text should be truncated
    assert len(result.split(">")[1].split("<")[0]) <= 83  # 80 chars + "..." = 83


def test_empty_node():
    """Test serialization of None/empty node."""
    result = DOMEvalSerializer.serialize_tree(None, DEFAULT_INCLUDE_ATTRIBUTES)
    assert result == ""

