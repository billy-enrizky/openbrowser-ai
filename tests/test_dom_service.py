"""Tests for the DOM service module.

This module provides comprehensive test coverage for the DOM service
subsystem, which handles parsing, processing, and representing DOM
structures from browser pages. It validates:

    - DomNode: Basic DOM node representation with attributes and text
    - DomState: DOM state container for element trees and selector maps
    - DomService: Core service for parsing and processing DOM trees
    - EnhancedDOMTreeNode: Extended node representation with parent/child links
    - Interactive element detection for buttons, links, inputs, and ARIA roles
    - Text extraction from DOM subtrees
    - Attribute parsing from raw CDP format

The DOM service is fundamental to the browser automation agent's ability
to understand and interact with web page content.
"""

import pytest

from src.openbrowser.browser.dom import (
    DomNode,
    DomState,
    DomService,
    NodeType,
    EnhancedDOMTreeNode,
)


class TestDomNode:
    """Tests for the DomNode class.

    Validates basic DOM node creation with tag names, attributes,
    text content, and node identifiers.
    """

    def test_dom_node_creation(self):
        """Test DomNode creation."""
        node = DomNode(
            tag_name="button",
            attributes={"id": "submit"},
            text="Click me",
            backend_node_id=123,
            distinct_id=1,
        )
        assert node.tag_name == "button"
        assert node.attributes["id"] == "submit"
        assert node.text == "Click me"
        assert node.backend_node_id == 123
        assert node.distinct_id == 1

    def test_dom_node_defaults(self):
        """Test DomNode default values."""
        node = DomNode(backend_node_id=1, distinct_id=1)
        assert node.tag_name is None
        assert node.attributes == {}
        assert node.text == ""


class TestDomState:
    """Tests for the DomState class.

    Validates DOM state creation including the serialized element tree
    and the selector map for element-to-backend-node-id mapping.
    """

    def test_dom_state_creation(self):
        """Test DomState creation."""
        state = DomState(
            element_tree="[1] <button>Click me</button>",
            selector_map={1: 123},
        )
        assert "[1] <button>" in state.element_tree
        assert state.selector_map[1] == 123


class TestDomService:
    """Tests for the DomService class.

    Validates the core DOM service functionality including attribute
    parsing, interactive element detection, and text extraction.
    """

    def test_parse_attributes_empty(self):
        """Test parsing empty attributes."""
        result = DomService._parse_attributes(None)
        assert result == {}

        result = DomService._parse_attributes([])
        assert result == {}

    def test_parse_attributes(self):
        """Test parsing attributes array."""
        attrs = ["id", "submit", "class", "btn btn-primary"]
        result = DomService._parse_attributes(attrs)
        assert result == {"id": "submit", "class": "btn btn-primary"}

    def test_is_interactive_button(self):
        """Test interactive detection for button."""
        node = {"nodeType": 1, "nodeName": "BUTTON", "attributes": []}
        assert DomService._is_interactive(node) is True

    def test_is_interactive_link(self):
        """Test interactive detection for link."""
        node = {"nodeType": 1, "nodeName": "A", "attributes": []}
        assert DomService._is_interactive(node) is True

    def test_is_interactive_input(self):
        """Test interactive detection for input."""
        node = {"nodeType": 1, "nodeName": "INPUT", "attributes": []}
        assert DomService._is_interactive(node) is True

    def test_is_not_interactive_div(self):
        """Test non-interactive detection for div."""
        node = {"nodeType": 1, "nodeName": "DIV", "attributes": []}
        assert DomService._is_interactive(node) is False

    def test_is_interactive_onclick(self):
        """Test interactive detection for onclick."""
        node = {"nodeType": 1, "nodeName": "DIV", "attributes": ["onclick", "doSomething()"]}
        assert DomService._is_interactive(node) is True

    def test_is_interactive_role_button(self):
        """Test interactive detection for role=button."""
        node = {"nodeType": 1, "nodeName": "DIV", "attributes": ["role", "button"]}
        assert DomService._is_interactive(node) is True

    def test_extract_text(self):
        """Test text extraction."""
        node = {
            "nodeType": 1,
            "nodeName": "DIV",
            "children": [
                {"nodeType": 3, "nodeValue": "Hello "},
                {"nodeType": 3, "nodeValue": "World"},
            ],
        }
        result = DomService._extract_text(node)
        assert result == "Hello World"


class TestEnhancedDOMTreeNode:
    """Tests for the EnhancedDOMTreeNode class.

    Validates the enhanced node representation with parent/child
    relationships and text extraction from descendant nodes.
    """

    def test_enhanced_node_creation(self):
        """Test EnhancedDOMTreeNode creation."""
        node = EnhancedDOMTreeNode(
            node_id=1,
            backend_node_id=123,
            node_type=NodeType.ELEMENT_NODE,
            node_name="BUTTON",
            node_value="",
            attributes={"id": "submit"},
        )
        assert node.node_id == 1
        assert node.backend_node_id == 123
        assert node.tag_name == "button"
        assert node.attributes["id"] == "submit"

    def test_enhanced_node_get_all_children_text(self):
        """Test getting all children text."""
        child = EnhancedDOMTreeNode(
            node_id=2,
            backend_node_id=124,
            node_type=NodeType.TEXT_NODE,
            node_name="#text",
            node_value="Hello World",
            attributes={},
        )
        parent = EnhancedDOMTreeNode(
            node_id=1,
            backend_node_id=123,
            node_type=NodeType.ELEMENT_NODE,
            node_name="BUTTON",
            node_value="",
            attributes={},
            children_nodes=[child],
        )
        child.parent_node = parent

        text = parent.get_all_children_text()
        assert text == "Hello World"

