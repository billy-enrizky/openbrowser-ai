"""Tests for DOMCodeAgentSerializer.

This module provides test coverage for the DOMCodeAgentSerializer class,
which serializes DOM trees into a compact HTML-like format optimized for
code-based browser automation agents. It validates:

    - Basic DOM tree serialization with proper HTML structure
    - Attribute handling with minimal class output (top 2 classes only)
    - Inline text truncation to 40 characters for readability
    - Iframe element handling with source attributes
    - Semantic structure preservation for headings and landmarks
    - Interactive element visibility (buttons, links, inputs)
    - Handling of empty or None nodes

The DOMCodeAgentSerializer produces output suitable for LLM consumption
when executing code-based browser automation tasks.
"""

import pytest

from src.openbrowser.browser.dom.serializer.code_use_serializer import DOMCodeAgentSerializer
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
        attributes={"id": "test", "class": "container wrapper"},
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
    result = DOMCodeAgentSerializer.serialize_tree(sample_simplified_node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<div" in result
    assert 'id="test"' in result


def test_minimal_attributes(sample_element_node):
    """Test that only top 2 classes are included."""
    sample_element_node.attributes = {"class": "class1 class2 class3 class4"}
    node = SimplifiedNode(original_node=sample_element_node, children=[])
    result = DOMCodeAgentSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "class1 class2" in result
    assert "class3" not in result
    assert "class4" not in result


def test_inline_text_truncation(sample_element_node):
    """Test that inline text is truncated to 40 chars."""
    text_node = EnhancedDOMTreeNode(
        node_id=2,
        backend_node_id=101,
        node_type=NodeType.TEXT_NODE,
        node_name="#text",
        node_value="This is a very long text that should be truncated at 40 characters",
        attributes={},
    )
    text_simplified = SimplifiedNode(original_node=text_node, children=[])
    sample_element_node.children_nodes = [text_node]
    node = SimplifiedNode(original_node=sample_element_node, children=[text_simplified])
    
    result = DOMCodeAgentSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    # Check that text is truncated
    assert len(result.split(">")[1].split("<")[0]) <= 43  # 40 chars + "..." = 43


def test_iframe_handling():
    """Test iframe serialization."""
    iframe_node = EnhancedDOMTreeNode(
        node_id=3,
        backend_node_id=102,
        node_type=NodeType.ELEMENT_NODE,
        node_name="iframe",
        node_value="",
        attributes={"src": "https://example.com"},
    )
    node = SimplifiedNode(original_node=iframe_node, children=[])
    result = DOMCodeAgentSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<iframe" in result
    assert 'src="https://example.com"' in result


def test_semantic_structure_preservation():
    """Test that semantic structure elements are preserved."""
    h1_node = EnhancedDOMTreeNode(
        node_id=4,
        backend_node_id=103,
        node_type=NodeType.ELEMENT_NODE,
        node_name="h1",
        node_value="",
        attributes={},
    )
    node = SimplifiedNode(original_node=h1_node, children=[])
    result = DOMCodeAgentSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<h1" in result


def test_empty_node():
    """Test serialization of None/empty node."""
    result = DOMCodeAgentSerializer.serialize_tree(None, DEFAULT_INCLUDE_ATTRIBUTES)
    assert result == ""


def test_interactive_elements():
    """Test that interactive elements are shown."""
    button_node = EnhancedDOMTreeNode(
        node_id=5,
        backend_node_id=104,
        node_type=NodeType.ELEMENT_NODE,
        node_name="button",
        node_value="",
        attributes={"aria-label": "Click me"},
    )
    node = SimplifiedNode(original_node=button_node, children=[])
    result = DOMCodeAgentSerializer.serialize_tree(node, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<button" in result
    assert 'aria-label="Click me"' in result

