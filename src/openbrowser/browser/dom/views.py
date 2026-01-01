"""Enhanced DOM tree views following browser-use pattern."""

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
    """DOM node types based on the DOM specification."""

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
    """Rectangle representing element bounds."""

    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(slots=True)
class EnhancedSnapshotNode:
    """Snapshot data for enhanced DOM functionality.
    
    Contains visibility, clickability, cursor styles, and layout information
    parsed from CDP DOMSnapshot data.
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
    """Enhanced accessibility node."""

    ax_node_id: str
    ignored: bool
    role: str | None
    name: str | None
    description: str | None
    properties: list[dict[str, Any]] | None
    child_ids: list[str] | None


@dataclass
class EnhancedDOMTreeNode:
    """
    Enhanced DOM tree node with information from AX, DOM, and Snapshot trees.
    Following browser-use pattern.
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
        """Get all children including shadow roots."""
        result = []
        if self.shadow_roots:
            result.extend(self.shadow_roots)
        if self.children_nodes:
            result.extend(self.children_nodes)
        return result

    def get_all_children_text(self, max_depth: int = -1) -> str:
        """Get all text content from children."""
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
        """Hash based on parent branch path and static attributes."""
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
        """Get parent branch path as list of tag names."""
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
        """Generate XPath for this DOM node."""
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
        """Get position among siblings with same tag name."""
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
    """Simplified tree node for serialization."""

    original_node: EnhancedDOMTreeNode
    children: list[SimplifiedNode]
    should_display: bool = True
    is_interactive: bool = False
    ignored_by_paint_order: bool = False
    excluded_by_parent: bool = False


DOMSelectorMap = dict[int, EnhancedDOMTreeNode]


@dataclass(slots=True)
class DOMInteractedElement:
    """
    Represents a DOM element that has been interacted with.
    Used to track which elements were clicked/typed during agent execution.
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
    """Serialized DOM state for agent consumption."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    element_tree: str
    selector_map: dict[int, int]  # element_index -> backend_node_id


# Legacy compatibility
class DomNode(BaseModel):
    """Simplified DOM node (legacy)."""

    tag_name: Optional[str] = None
    attributes: dict[str, str] = {}
    text: str = ""
    backend_node_id: int
    distinct_id: int


class DomState(BaseModel):
    """DOM state for LLM consumption (legacy)."""

    element_tree: str
    selector_map: dict[int, int]

