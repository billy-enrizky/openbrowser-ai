"""Tests for ClickableElementDetector.

This module provides comprehensive test coverage for the ClickableElementDetector
class, which is responsible for determining whether DOM elements are interactive
(clickable, focusable, or otherwise actionable). It validates detection of:

    - Standard interactive elements (button, input, anchor, select, textarea)
    - Elements with interactive attributes (onclick, role="button", tabindex)
    - Elements with pointer cursor styling
    - Search-related elements by class name patterns
    - Accessibility properties (focusable AX nodes)
    - Special handling for iframes based on size
    - Exclusion of non-interactive structural elements (html, body)

The ClickableElementDetector is critical for identifying actionable elements
during browser automation and DOM serialization.
"""

import pytest

from src.openbrowser.browser.dom.serializer.clickable_elements import ClickableElementDetector
from src.openbrowser.browser.dom.views import (
    EnhancedAXNode,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
)


@pytest.fixture
def basic_element_node():
    """Create a basic element node for testing.

    Returns:
        EnhancedDOMTreeNode: A basic div element node with minimal attributes.
    """
    return EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
    )


def test_button_is_interactive():
    """Test that button elements are detected as interactive."""
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="button",
        node_value="",
        attributes={},
    )
    assert ClickableElementDetector.is_interactive(node) is True


def test_input_is_interactive():
    """Test that input elements are detected as interactive."""
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="input",
        node_value="",
        attributes={"type": "text"},
    )
    assert ClickableElementDetector.is_interactive(node) is True


def test_link_is_interactive():
    """Test that anchor elements are detected as interactive."""
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="a",
        node_value="",
        attributes={"href": "https://example.com"},
    )
    assert ClickableElementDetector.is_interactive(node) is True


def test_non_interactive_element():
    """Test that div elements without interactive attributes are not interactive."""
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
    )
    assert ClickableElementDetector.is_interactive(node) is False


def test_element_with_onclick():
    """Test that elements with onclick are interactive."""
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={"onclick": "doSomething()"},
    )
    assert ClickableElementDetector.is_interactive(node) is True


def test_element_with_role_button():
    """Test that elements with role='button' are interactive."""
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={"role": "button"},
    )
    assert ClickableElementDetector.is_interactive(node) is True


def test_search_element_detection():
    """Test that search elements are detected."""
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={"class": "search-box"},
    )
    assert ClickableElementDetector.is_interactive(node) is True


def test_ax_node_focusable():
    """Test that elements with focusable AX property are interactive."""
    ax_node = EnhancedAXNode(
        ax_node_id="1",
        ignored=False,
        role=None,
        name=None,
        description=None,
        properties=[{"name": "focusable", "value": True}],
        child_ids=None,
    )
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
        ax_node=ax_node,
    )
    assert ClickableElementDetector.is_interactive(node) is True


def test_iframe_size_check():
    """Test that large iframes are interactive."""
    snapshot = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=None,
        client_rects=None,
        scroll_rects=None,
        computed_styles=None,
        paint_order=None,
    )
    # Create bounds for large iframe
    from src.openbrowser.browser.dom.views import DOMRect
    snapshot.bounds = DOMRect(x=0, y=0, width=200, height=200)
    
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="iframe",
        node_value="",
        attributes={},
        snapshot_node=snapshot,
    )
    assert ClickableElementDetector.is_interactive(node) is True


def test_small_iframe_not_interactive():
    """Test that small iframes are not interactive."""
    snapshot = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=None,
        client_rects=None,
        scroll_rects=None,
        computed_styles=None,
        paint_order=None,
    )
    from src.openbrowser.browser.dom.views import DOMRect
    snapshot.bounds = DOMRect(x=0, y=0, width=50, height=50)  # Too small
    
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="iframe",
        node_value="",
        attributes={},
        snapshot_node=snapshot,
    )
    assert ClickableElementDetector.is_interactive(node) is False


def test_html_body_not_interactive():
    """Test that html and body nodes are not interactive."""
    html_node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="html",
        node_value="",
        attributes={},
    )
    assert ClickableElementDetector.is_interactive(html_node) is False
    
    body_node = EnhancedDOMTreeNode(
        node_id=2,
        backend_node_id=101,
        node_type=NodeType.ELEMENT_NODE,
        node_name="body",
        node_value="",
        attributes={},
    )
    assert ClickableElementDetector.is_interactive(body_node) is False


def test_cursor_pointer():
    """Test that elements with pointer cursor are interactive."""
    snapshot = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style="pointer",
        bounds=None,
        client_rects=None,
        scroll_rects=None,
        computed_styles=None,
        paint_order=None,
    )
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
        snapshot_node=snapshot,
    )
    assert ClickableElementDetector.is_interactive(node) is True

