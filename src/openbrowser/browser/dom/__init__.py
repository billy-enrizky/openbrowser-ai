"""DOM processing module for browser automation.

This module provides enhanced DOM tree extraction, processing, and serialization
for LLM-driven browser automation. It follows the browser-use pattern for
DOM handling with support for accessibility trees, shadow DOM, and iframes.

Key Components:
    DomService: Main service for DOM extraction and processing.
    DomNode: Simplified DOM node for interactive elements.
    DomState: Container for element tree and selector mapping.
    EnhancedDOMTreeNode: Full DOM tree node with AX and snapshot data.
    SimplifiedNode: Reduced node for serialization.
    DOMTreeSerializer: Converts DOM trees to LLM-friendly formats.

Submodules:
    views: Data structures for DOM representation.
    service: DOM extraction and processing service.
    serializer: Serialization utilities for different agent modes.

Example:
    >>> from openbrowser.browser.dom import DomService, DomState
    >>> service = DomService(browser_session)
    >>> state = await service.get_dom_state(cdp_session)
    >>> print(state.element_tree)
"""

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
