"""DOM processing module."""

from .views import (
    DomNode,
    DomState,
    DOMRect,
    DOMInteractedElement,
    DOMSelectorMap,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    EnhancedAXNode,
    NodeType,
    SimplifiedNode,
    SerializedDOMState,
    DEFAULT_INCLUDE_ATTRIBUTES,
)
from .service import DomService

__all__ = [
    "DomNode",
    "DomState",
    "DomService",
    "DOMRect",
    "DOMInteractedElement",
    "DOMSelectorMap",
    "EnhancedDOMTreeNode",
    "EnhancedSnapshotNode",
    "EnhancedAXNode",
    "NodeType",
    "SimplifiedNode",
    "SerializedDOMState",
    "DEFAULT_INCLUDE_ATTRIBUTES",
]
