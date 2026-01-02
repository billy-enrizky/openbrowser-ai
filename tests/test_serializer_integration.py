"""Integration tests for serializer components.

This module provides integration test coverage for the DOM serialization
subsystem, validating that all serializer components work together
correctly. It tests:

    - ClickableElementDetector integration with DOM tree processing
    - PaintOrderRemover integration with snapshot data
    - DOMTreeSerializer with paint order filtering enabled
    - DOMCodeAgentSerializer for code-use output format
    - DOMEvalSerializer for evaluation output format
    - Different serializer modes and their output characteristics

These integration tests ensure the serialization pipeline produces
correct output when all components are combined, validating
end-to-end behavior rather than isolated unit functionality.
"""

import pytest

from src.openbrowser.browser.dom.serializer.service import DOMTreeSerializer
from src.openbrowser.browser.dom.serializer.clickable_elements import ClickableElementDetector
from src.openbrowser.browser.dom.serializer.paint_order import PaintOrderRemover
from src.openbrowser.browser.dom.serializer.code_use_serializer import DOMCodeAgentSerializer
from src.openbrowser.browser.dom.serializer.eval_serializer import DOMEvalSerializer
from src.openbrowser.browser.dom.views import (
    DEFAULT_INCLUDE_ATTRIBUTES,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    SimplifiedNode,
)


@pytest.fixture
def sample_tree():
    """Create a sample DOM tree for integration testing.

    Returns:
        SimplifiedNode: A simplified DOM tree with body, button, and div nodes
        configured with parent-child relationships for testing serialization.
    """
    root = EnhancedDOMTreeNode(
        node_id=0,
        backend_node_id=99,
        node_type=NodeType.ELEMENT_NODE,
        node_name="body",
        node_value="",
        attributes={},
    )
    
    button = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="button",
        node_value="",
        attributes={"id": "click-me", "aria-label": "Click button"},
    )
    
    div = EnhancedDOMTreeNode(
        node_id=2,
        backend_node_id=101,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={"class": "container"},
    )
    
    root.children_nodes = [button, div]
    button.parent_node = root
    div.parent_node = root
    
    simplified_button = SimplifiedNode(original_node=button, children=[])
    simplified_div = SimplifiedNode(original_node=div, children=[])
    simplified_root = SimplifiedNode(original_node=root, children=[simplified_button, simplified_div])
    
    return simplified_root


def test_clickable_detector_integration():
    """Test ClickableElementDetector integration."""
    button = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="button",
        node_value="",
        attributes={},
    )
    assert ClickableElementDetector.is_interactive(button) is True
    
    div = EnhancedDOMTreeNode(
        node_id=2,
        backend_node_id=101,
        node_type=NodeType.ELEMENT_NODE,
        node_name="div",
        node_value="",
        attributes={},
    )
    assert ClickableElementDetector.is_interactive(div) is False


def test_paint_order_integration(sample_tree):
    """Test PaintOrderRemover integration."""
    # Add paint order to nodes
    from src.openbrowser.browser.dom.views import DOMRect
    
    snapshot1 = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=DOMRect(x=0, y=0, width=100, height=100),
        client_rects=None,
        scroll_rects=None,
        computed_styles={"opacity": "1.0", "background-color": "rgba(255, 0, 0, 1)"},
        paint_order=1,
    )
    sample_tree.children[0].original_node.snapshot_node = snapshot1
    
    snapshot2 = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=DOMRect(x=0, y=0, width=100, height=100),
        client_rects=None,
        scroll_rects=None,
        computed_styles={"opacity": "1.0", "background-color": "rgba(0, 255, 0, 1)"},
        paint_order=2,
    )
    sample_tree.children[1].original_node.snapshot_node = snapshot2
    
    # Run paint order remover
    remover = PaintOrderRemover(sample_tree)
    remover.calculate_paint_order()
    
    # First node should be marked as ignored (covered by second)
    assert sample_tree.children[0].ignored_by_paint_order is True


def test_serializer_with_paint_order(sample_tree):
    """Test serializer with paint order filtering."""
    # Add paint order
    from src.openbrowser.browser.dom.views import DOMRect
    
    snapshot = EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=DOMRect(x=0, y=0, width=100, height=100),
        client_rects=None,
        scroll_rects=None,
        computed_styles={"opacity": "1.0", "background-color": "rgba(255, 0, 0, 1)"},
        paint_order=1,
    )
    sample_tree.children[0].original_node.snapshot_node = snapshot
    
    # Serialize with paint order filtering
    result = DOMTreeSerializer.serialize_tree(sample_tree, DEFAULT_INCLUDE_ATTRIBUTES, paint_order_filtering=True)
    assert result  # Should produce output


def test_code_use_serializer_integration(sample_tree):
    """Test code-use serializer integration."""
    result = DOMCodeAgentSerializer.serialize_tree(sample_tree, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<button" in result
    assert 'id="click-me"' in result


def test_eval_serializer_integration(sample_tree):
    """Test eval serializer integration."""
    result = DOMEvalSerializer.serialize_tree(sample_tree, DEFAULT_INCLUDE_ATTRIBUTES)
    assert "<button" in result
    assert "<div" in result


def test_serializer_modes(sample_tree):
    """Test different serializer modes."""
    # Default serializer
    default_result = DOMTreeSerializer.serialize_tree(sample_tree, DEFAULT_INCLUDE_ATTRIBUTES)
    assert default_result
    
    # Code-use serializer
    code_use_result = DOMTreeSerializer.serialize_for_code_use(sample_tree, DEFAULT_INCLUDE_ATTRIBUTES)
    assert code_use_result
    
    # Eval serializer
    eval_result = DOMTreeSerializer.serialize_for_eval(sample_tree, DEFAULT_INCLUDE_ATTRIBUTES)
    assert eval_result
    
    # Results should be different (code-use is more compact)
    assert len(code_use_result) <= len(default_result) or len(code_use_result) <= len(eval_result)

