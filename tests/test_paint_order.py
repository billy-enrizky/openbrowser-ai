"""Tests for PaintOrderRemover.

This module provides test coverage for the PaintOrderRemover class
and related utilities for paint order-based visibility filtering.
It validates:

    - Rect: Rectangle geometry calculations (area, intersection, containment)
    - RectUnionPure: Union of rectangles for coverage tracking
    - PaintOrderRemover: DOM element visibility based on paint order
    - Transparent element handling (skipped from coverage calculations)

The paint order system identifies elements that are visually obscured
by elements painted on top of them, allowing serializers to skip
non-visible content for more efficient DOM representation.
"""

import pytest

from src.openbrowser.browser.dom.serializer.paint_order import PaintOrderRemover, Rect, RectUnionPure
from src.openbrowser.browser.dom.views import (
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    SimplifiedNode,
)


@pytest.fixture
def dom_rect():
    """Create a DOMRect class reference for testing.

    Returns:
        type: The DOMRect class from browser.dom.views.
    """
    from src.openbrowser.browser.dom.views import DOMRect
    return DOMRect


def test_rect_area():
    """Test Rect area calculation."""
    rect = Rect(x1=0, y1=0, x2=10, y2=20)
    assert rect.area() == 200


def test_rect_intersects():
    """Test Rect intersection detection."""
    rect1 = Rect(x1=0, y1=0, x2=10, y2=10)
    rect2 = Rect(x1=5, y1=5, x2=15, y2=15)
    assert rect1.intersects(rect2) is True
    
    rect3 = Rect(x1=20, y1=20, x2=30, y2=30)
    assert rect1.intersects(rect3) is False


def test_rect_contains():
    """Test Rect containment detection."""
    outer = Rect(x1=0, y1=0, x2=20, y2=20)
    inner = Rect(x1=5, y1=5, x2=15, y2=15)
    assert outer.contains(inner) is True
    assert inner.contains(outer) is False


def test_rect_union_add():
    """Test RectUnionPure add method."""
    union = RectUnionPure()
    rect1 = Rect(x1=0, y1=0, x2=10, y2=10)
    assert union.add(rect1) is True
    assert union.add(rect1) is False  # Already covered


def test_rect_union_contains():
    """Test RectUnionPure contains method."""
    union = RectUnionPure()
    rect1 = Rect(x1=0, y1=0, x2=10, y2=10)
    union.add(rect1)
    
    assert union.contains(rect1) is True
    assert union.contains(Rect(x1=5, y1=5, x2=8, y2=8)) is True
    assert union.contains(Rect(x1=20, y1=20, x2=30, y2=30)) is False


def test_paint_order_remover():
    """Test PaintOrderRemover with sample nodes."""
    # Create nodes with paint order
    node1 = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
    )
    snapshot1 = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=None,
        client_rects=None,
        scroll_rects=None,
        computed_styles=None,
        paint_order=1,
    )
    from src.openbrowser.browser.dom.views import DOMRect
    snapshot1.bounds = DOMRect(x=0, y=0, width=100, height=100)
    snapshot1.computed_styles = {"opacity": "1.0", "background-color": "rgba(255, 0, 0, 1)"}
    node1.snapshot_node = snapshot1
    
    # Node 2 covers node 1 (higher paint order, same area)
    node2 = EnhancedDOMTreeNode(
        node_id=2,
        backend_node_id=101,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
    )
    snapshot2 = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=None,
        client_rects=None,
        scroll_rects=None,
        computed_styles=None,
        paint_order=2,
    )
    snapshot2.bounds = DOMRect(x=0, y=0, width=100, height=100)
    snapshot2.computed_styles = {"opacity": "1.0", "background-color": "rgba(0, 255, 0, 1)"}
    node2.snapshot_node = snapshot2
    
    simplified1 = SimplifiedNode(original_node=node1, children=[])
    simplified2 = SimplifiedNode(original_node=node2, children=[])
    
    # Create root with both children
    root_node = EnhancedDOMTreeNode(
        node_id=0,
        backend_node_id=99,
        node_type=NodeType.ELEMENT_NODE,
        node_name="body",
        node_value="",
        attributes={},
    )
    root = SimplifiedNode(original_node=root_node, children=[simplified1, simplified2])
    
    # Run paint order remover
    remover = PaintOrderRemover(root)
    remover.calculate_paint_order()
    
    # Node 1 should be marked as ignored (covered by node 2)
    assert simplified1.ignored_by_paint_order is True
    assert simplified2.ignored_by_paint_order is False


def test_transparent_elements_skipped():
    """Test that transparent elements are skipped."""
    node = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
    )
    snapshot = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=None,
        client_rects=None,
        scroll_rects=None,
        computed_styles=None,
        paint_order=1,
    )
    from src.openbrowser.browser.dom.views import DOMRect
    snapshot.bounds = DOMRect(x=0, y=0, width=100, height=100)
    snapshot.computed_styles = {"opacity": "0.5", "background-color": "rgba(255, 0, 0, 0.5)"}  # Low opacity
    node.snapshot_node = snapshot
    
    simplified = SimplifiedNode(original_node=node, children=[])
    root = SimplifiedNode(original_node=node, children=[simplified])
    
    remover = PaintOrderRemover(root)
    remover.calculate_paint_order()
    
    # Transparent element should not mark others as ignored
    assert simplified.ignored_by_paint_order is False

