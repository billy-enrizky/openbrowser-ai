"""Enhanced DOM tree views following browser-use pattern.

This module defines data structures for representing DOM elements with varying
levels of detail, from full enhanced nodes to simplified serializable forms.

Constants:
    DEFAULT_INCLUDE_ATTRIBUTES: Attributes included in LLM representations.
    STATIC_ATTRIBUTES: Attributes used for element hashing.

Classes:
    NodeType: Enum of DOM node types (ELEMENT_NODE, TEXT_NODE, etc.).
    DOMRect: Rectangle representing element bounds.
    EnhancedSnapshotNode: Visibility and layout data from DOMSnapshot.
    EnhancedAXNode: Accessibility node data.
    EnhancedDOMTreeNode: Full DOM node with AX, snapshot, and hierarchy.
    SimplifiedNode: Reduced node for serialization.
    DOMInteractedElement: Record of an interacted element.
    SerializedDOMState: Final serialized output for agents.
    DomNode: Legacy simplified DOM node.
    DomState: Legacy DOM state container.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# Default attributes to include in LLM representation
DEFAULT_INCLUDE_ATTRIBUTES = [
    "title",
    "type",
    "checked",
    "id",
    "name",
    "role",
    "value",
    "placeholder",
    "alt",
    "aria-label",
    "aria-expanded",
    "data-state",
    "aria-checked",
    "aria-valuemin",
    "aria-valuemax",
    "aria-valuenow",
    "pattern",
    "min",
    "max",
    "minlength",
    "maxlength",
    "required",
    "disabled",
    "readonly",
    "href",
]

# Static attributes used for element hashing
STATIC_ATTRIBUTES = {
    "class",
    "id",
    "name",
    "type",
    "placeholder",
    "aria-label",
    "title",
    "role",
    "data-testid",
    "for",
    "required",
    "disabled",
    "checked",
    "href",
}


class NodeType(IntEnum):
    """DOM node types based on the DOM specification.

    These constants match the W3C DOM Node.nodeType values:
    https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeType

    Commonly used:
        ELEMENT_NODE (1): Regular HTML elements like <div>, <button>.
        TEXT_NODE (3): Text content within elements.
        DOCUMENT_NODE (9): The document root.
        DOCUMENT_FRAGMENT_NODE (11): Shadow DOM roots.
    """

    ELEMENT_NODE = 1
    ATTRIBUTE_NODE = 2
    TEXT_NODE = 3
    CDATA_SECTION_NODE = 4
    ENTITY_REFERENCE_NODE = 5
    ENTITY_NODE = 6
    PROCESSING_INSTRUCTION_NODE = 7
    COMMENT_NODE = 8
    DOCUMENT_NODE = 9
    DOCUMENT_TYPE_NODE = 10
    DOCUMENT_FRAGMENT_NODE = 11
    NOTATION_NODE = 12


@dataclass(slots=True)
class DOMRect:
    """Rectangle representing element bounds in viewport coordinates.

    Used for element positioning, visibility checking, and click targeting.
    Coordinates are relative to the viewport (not the document).

    Attributes:
        x: Left edge position in pixels.
        y: Top edge position in pixels.
        width: Element width in pixels.
        height: Element height in pixels.

    Example:
        >>> rect = DOMRect(x=100, y=200, width=150, height=50)
        >>> rect.to_dict()
        {'x': 100, 'y': 200, 'width': 150, 'height': 50}
    """

    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation.

        Returns:
            Dict with x, y, width, height keys.
        """
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(slots=True)
class EnhancedSnapshotNode:
    """Snapshot data for enhanced DOM functionality.

    Contains visibility, clickability, cursor styles, and layout information
    parsed from CDP DOMSnapshot.captureSnapshot data.

    Attributes:
        is_clickable: Whether element responds to click events.
        cursor_style: CSS cursor value (e.g., 'pointer', 'text').
        bounds: Element bounding rectangle.
        client_rects: Client rects from getClientRects().
        scroll_rects: Scrollable area rectangles.
        computed_styles: CSS computed style values.
        paint_order: Paint order index for visibility filtering.
        stacking_contexts: Stacking context level.

    Note:
        All fields are optional as snapshot data may be incomplete.
    """

    is_clickable: bool | None
    cursor_style: str | None
    bounds: DOMRect | None
    client_rects: DOMRect | None
    scroll_rects: DOMRect | None
    computed_styles: dict[str, str] | None
    paint_order: int | None
    stacking_contexts: int | None = None


@dataclass(slots=True)
class EnhancedAXNode:
    """Enhanced accessibility node from the AX tree.

    Contains accessibility information extracted from CDP Accessibility tree,
    useful for understanding element semantics and screen reader presentation.

    Attributes:
        ax_node_id: Unique ID in the accessibility tree.
        ignored: Whether this node is ignored by accessibility tools.
        role: ARIA role (e.g., 'button', 'link', 'textbox').
        name: Accessible name (visible or aria-label text).
        description: Accessible description (aria-describedby).
        properties: Additional accessibility properties.
        child_ids: IDs of child nodes in the AX tree.
    """

    ax_node_id: str
    ignored: bool
    role: str | None
    name: str | None
    description: str | None
    properties: list[dict[str, Any]] | None
    child_ids: list[str] | None


@dataclass
class EnhancedDOMTreeNode:
    """Enhanced DOM tree node with AX, DOM, and Snapshot data.

    Core data structure for DOM representation, combining information from
    multiple CDP domains (DOM, Accessibility, DOMSnapshot) into a unified
    tree structure. Follows the browser-use pattern.

    Attributes:
        node_id: CDP DOM node ID (may change between commands).
        backend_node_id: Stable CDP backend node ID.
        node_type: DOM node type (element, text, etc.).
        node_name: HTML tag name or node type name.
        node_value: Text content for text nodes.
        attributes: Element attributes as key-value pairs.
        is_scrollable: Whether element has scrollable overflow.
        is_visible: Whether element is visible in viewport.
        absolute_position: Element bounds in absolute coordinates.
        target_id: CDP target ID for the frame.
        frame_id: CDP frame ID.
        session_id: CDP session ID.
        content_document: Iframe's content document (if applicable).
        shadow_root_type: 'open' or 'closed' shadow root type.
        shadow_roots: Shadow DOM root nodes.
        parent_node: Parent in the DOM tree.
        children_nodes: Child nodes.
        ax_node: Associated accessibility node.
        snapshot_node: Associated snapshot data.

    Example:
        >>> node = DomService._build_enhanced_node(cdp_node)
        >>> print(node.tag_name, node.xpath)
        'button' 'html/body/div[1]/button[2]'
    """

    node_id: int
    backend_node_id: int
    node_type: NodeType
    node_name: str
    node_value: str
    attributes: dict[str, str]
    is_scrollable: bool | None = None
    is_visible: bool | None = None
    absolute_position: DOMRect | None = None

    # Frame info
    target_id: str | None = None
    frame_id: str | None = None
    session_id: str | None = None
    content_document: EnhancedDOMTreeNode | None = None

    # Shadow DOM
    shadow_root_type: str | None = None
    shadow_roots: list[EnhancedDOMTreeNode] | None = None

    # Navigation
    parent_node: EnhancedDOMTreeNode | None = None
    children_nodes: list[EnhancedDOMTreeNode] | None = None

    # Enhanced data
    ax_node: EnhancedAXNode | None = None
    snapshot_node: EnhancedSnapshotNode | None = None

    @property
    def tag_name(self) -> str:
        return self.node_name.lower()

    @property
    def children(self) -> list[EnhancedDOMTreeNode]:
        return self.children_nodes or []

    @property
    def children_and_shadow_roots(self) -> list[EnhancedDOMTreeNode]:
        """Get all children including shadow roots.

        Returns shadow roots first, then regular children, matching
        browser rendering order for shadow DOM.

        Returns:
            Combined list of shadow root nodes and child nodes.
        """
        result = []
        if self.shadow_roots:
            result.extend(self.shadow_roots)
        if self.children_nodes:
            result.extend(self.children_nodes)
        return result

    def get_all_children_text(self, max_depth: int = -1) -> str:
        """Get all text content from children.

        Recursively collects text node values from the subtree.

        Args:
            max_depth: Maximum depth to traverse (-1 for unlimited).

        Returns:
            Concatenated text content, newline-separated.
        """
        text_parts = []

        def collect_text(node: EnhancedDOMTreeNode, current_depth: int) -> None:
            if max_depth != -1 and current_depth > max_depth:
                return
            if node.node_type == NodeType.TEXT_NODE:
                text_parts.append(node.node_value)
            elif node.node_type == NodeType.ELEMENT_NODE:
                for child in node.children:
                    collect_text(child, current_depth + 1)

        collect_text(self, 0)
        return "\n".join(text_parts).strip()

    def __hash__(self) -> int:
        """Hash based on parent branch path and static attributes.

        Creates a stable hash from the element's position in the DOM tree
        and its static attributes. Used for element identification across
        page states.

        Returns:
            Integer hash derived from SHA-256 of path and attributes.
        """
        parent_branch_path = self._get_parent_branch_path()
        parent_branch_path_string = "/".join(parent_branch_path)
        attributes_string = "".join(
            f"{k}={v}"
            for k, v in sorted((k, v) for k, v in self.attributes.items() if k in STATIC_ATTRIBUTES)
        )
        combined_string = f"{parent_branch_path_string}|{attributes_string}"
        element_hash = hashlib.sha256(combined_string.encode()).hexdigest()
        return int(element_hash[:16], 16)

    def _get_parent_branch_path(self) -> list[str]:
        """Get parent branch path as list of tag names.

        Traverses up the DOM tree collecting element tag names.

        Returns:
            List of tag names from root to this element.
        """
        parents: list[EnhancedDOMTreeNode] = []
        current: EnhancedDOMTreeNode | None = self
        while current is not None:
            if current.node_type == NodeType.ELEMENT_NODE:
                parents.append(current)
            current = current.parent_node
        parents.reverse()
        return [p.tag_name for p in parents]

    @property
    def xpath(self) -> str:
        """Generate XPath for this DOM node.

        Builds an XPath expression from the root to this element,
        using positional indexing for disambiguation among siblings.
        Stops at iframe boundaries.

        Returns:
            XPath string like 'html/body/div[1]/button[2]'.
        """
        segments = []
        current = self
        while current and current.node_type in (NodeType.ELEMENT_NODE, NodeType.DOCUMENT_FRAGMENT_NODE):
            if current.node_type == NodeType.DOCUMENT_FRAGMENT_NODE:
                current = current.parent_node
                continue
            if current.parent_node and current.parent_node.node_name.lower() == "iframe":
                break
            position = self._get_element_position(current)
            tag_name = current.node_name.lower()
            xpath_index = f"[{position}]" if position > 0 else ""
            segments.insert(0, f"{tag_name}{xpath_index}")
            current = current.parent_node
        return "/".join(segments)

    def _get_element_position(self, element: EnhancedDOMTreeNode) -> int:
        """Get position among siblings with same tag name.

        Args:
            element: Element to find position for.

        Returns:
            1-based position among same-tag siblings, or 0 if unique.
        """
        if not element.parent_node or not element.parent_node.children_nodes:
            return 0
        same_tag_siblings = [
            child
            for child in element.parent_node.children_nodes
            if child.node_type == NodeType.ELEMENT_NODE and child.node_name.lower() == element.node_name.lower()
        ]
        if len(same_tag_siblings) <= 1:
            return 0
        try:
            return same_tag_siblings.index(element) + 1
        except ValueError:
            return 0


@dataclass(slots=True)
class SimplifiedNode:
    """Simplified tree node for serialization.

    Wraps an EnhancedDOMTreeNode with serialization metadata, used
    during DOM tree conversion to LLM-readable formats.

    Attributes:
        original_node: The full EnhancedDOMTreeNode.
        children: Child SimplifiedNodes.
        should_display: Whether to include in serialized output.
        is_interactive: Whether element is clickable/interactable.
        ignored_by_paint_order: Filtered out by paint order.
        excluded_by_parent: Excluded due to parent exclusion.
    """

    original_node: EnhancedDOMTreeNode
    children: list[SimplifiedNode]
    should_display: bool = True
    is_interactive: bool = False
    ignored_by_paint_order: bool = False
    excluded_by_parent: bool = False


DOMSelectorMap = dict[int, EnhancedDOMTreeNode]


@dataclass(slots=True)
class DOMInteractedElement:
    """Represents a DOM element that has been interacted with.

    Records element state at interaction time for agent history tracking.
    Used to track which elements were clicked/typed during execution.

    Attributes:
        node_id: CDP DOM node ID at interaction time.
        backend_node_id: Stable CDP backend node ID.
        frame_id: Frame containing the element.
        node_type: DOM node type.
        node_value: Text content if text node.
        node_name: HTML tag name.
        attributes: Element attributes.
        bounds: Bounding rectangle at interaction time.
        x_path: XPath to the element.
        element_hash: Stable hash for element identification.

    Example:
        >>> interacted = DOMInteractedElement.load_from_enhanced_dom_tree(node)
        >>> print(interacted.x_path)
        'html/body/button[1]'
    """

    node_id: int
    backend_node_id: int
    frame_id: str | None

    node_type: NodeType
    node_value: str
    node_name: str
    attributes: dict[str, str] | None

    bounds: DOMRect | None

    x_path: str

    element_hash: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dict with all attributes, bounds as nested dict.
        """
        return {
            'node_id': self.node_id,
            'backend_node_id': self.backend_node_id,
            'frame_id': self.frame_id,
            'node_type': self.node_type.value,
            'node_value': self.node_value,
            'node_name': self.node_name,
            'attributes': self.attributes,
            'x_path': self.x_path,
            'element_hash': self.element_hash,
            'bounds': self.bounds.to_dict() if self.bounds else None,
        }

    @classmethod
    def load_from_enhanced_dom_tree(cls, enhanced_dom_tree: EnhancedDOMTreeNode) -> 'DOMInteractedElement':
        """Create an interacted element from an EnhancedDOMTreeNode.

        Args:
            enhanced_dom_tree: Source node with full DOM data.

        Returns:
            DOMInteractedElement with copied data.
        """
        return cls(
            node_id=enhanced_dom_tree.node_id,
            backend_node_id=enhanced_dom_tree.backend_node_id,
            frame_id=enhanced_dom_tree.frame_id,
            node_type=enhanced_dom_tree.node_type,
            node_value=enhanced_dom_tree.node_value,
            node_name=enhanced_dom_tree.node_name,
            attributes=enhanced_dom_tree.attributes,
            bounds=enhanced_dom_tree.snapshot_node.bounds if enhanced_dom_tree.snapshot_node else None,
            x_path=enhanced_dom_tree.xpath,
            element_hash=hash(enhanced_dom_tree),
        )


class SerializedDOMState(BaseModel):
    """Serialized DOM state for agent consumption.

    Final output format containing the LLM-readable element tree
    and mapping for resolving element references.

    Attributes:
        element_tree: Formatted string of interactive elements.
        selector_map: Maps element_index to backend_node_id.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    element_tree: str
    selector_map: dict[int, int]  # element_index -> backend_node_id


# Legacy compatibility
class DomNode(BaseModel):
    """Simplified DOM node (legacy compatibility).

    Lightweight representation for basic DOM extraction.
    Prefer EnhancedDOMTreeNode for full functionality.

    Attributes:
        tag_name: Lowercase HTML tag name.
        attributes: Element attributes.
        text: Combined text content.
        backend_node_id: CDP backend node ID.
        distinct_id: Assigned ID for LLM reference.
    """

    tag_name: Optional[str] = None
    attributes: dict[str, str] = {}
    text: str = ""
    backend_node_id: int
    distinct_id: int


class DomState(BaseModel):
    """DOM state for LLM consumption (legacy compatibility).

    Container for element tree and selector mapping.
    Prefer SerializedDOMState for new implementations.

    Attributes:
        element_tree: Formatted string of interactive elements.
        selector_map: Maps distinct_id to backend_node_id.
    """

    element_tree: str
    selector_map: dict[int, int]

