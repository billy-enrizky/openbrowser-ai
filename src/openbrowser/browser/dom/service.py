"""Enhanced DOM service following browser-use pattern."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from cdp_use.client import CDPClient

from src.openbrowser.browser.dom.views import (
    DomNode,
    DomState,
    DOMRect,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    SimplifiedNode,
    SerializedDOMState,
)
from src.openbrowser.browser.dom.serializer.service import DOMTreeSerializer

if TYPE_CHECKING:
    from src.openbrowser.browser.session import BrowserSession, CDPSession

logger = logging.getLogger(__name__)

# Interactive HTML tag names
INTERACTIVE_TAGS = {"a", "button", "input", "textarea", "select", "option", "label", "summary"}


class DomService:
    """
    Service for getting the DOM tree and other DOM-related information.
    Enhanced version following browser-use pattern.
    """

    def __init__(
        self,
        browser_session: Optional[BrowserSession] = None,
        cross_origin_iframes: bool = False,
        paint_order_filtering: bool = True,
        max_iframes: int = 100,
        max_iframe_depth: int = 5,
    ):
        self.browser_session = browser_session
        self.cross_origin_iframes = cross_origin_iframes
        self.paint_order_filtering = paint_order_filtering
        self.max_iframes = max_iframes
        self.max_iframe_depth = max_iframe_depth

    async def get_dom_state(self, cdp_session: CDPSession) -> DomState:
        """Get DOM state using the CDPSession."""
        return await self.get_clickable_elements(
            client=cdp_session.cdp_client,
            session_id=cdp_session.session_id,
        )

    @staticmethod
    def _parse_attributes(attributes_list: Optional[list[str]]) -> dict[str, str]:
        """Parse CDP attributes array into a dictionary."""
        if not attributes_list:
            return {}
        attrs = {}
        for i in range(0, len(attributes_list), 2):
            if i + 1 < len(attributes_list):
                attrs[attributes_list[i]] = attributes_list[i + 1]
        return attrs

    @staticmethod
    def _is_interactive(node: dict) -> bool:
        """Check if a node is interactive."""
        if node.get("nodeType") != NodeType.ELEMENT_NODE:
            return False

        node_name = node.get("nodeName", "").lower()
        if node_name in INTERACTIVE_TAGS:
            return True

        # Check for event handler attributes
        attributes = DomService._parse_attributes(node.get("attributes"))
        event_handlers = {"onclick", "onmousedown", "onmouseup", "onkeydown", "onkeyup", "ontouchstart"}
        if any(attr.lower() in event_handlers for attr in attributes.keys()):
            return True

        # Check for role attributes that indicate interactivity
        role = attributes.get("role", "").lower()
        if role in {"button", "link", "checkbox", "radio", "tab", "menuitem", "option"}:
            return True

        # Check for tabindex (makes element focusable/interactive)
        if "tabindex" in attributes:
            tabindex = attributes.get("tabindex", "")
            if tabindex != "-1":  # tabindex=-1 means not keyboard accessible
                return True

        return False

    @staticmethod
    def _extract_text(node: dict) -> str:
        """Extract text content from a node and its children recursively."""
        text_parts = []

        if node.get("nodeType") == NodeType.TEXT_NODE:
            node_value = node.get("nodeValue", "")
            if node_value:
                text_parts.append(node_value.strip())

        children = node.get("children", [])
        for child in children:
            child_text = DomService._extract_text(child)
            if child_text:
                text_parts.append(child_text)

        return " ".join(text_parts).strip()

    @staticmethod
    def _build_enhanced_node(
        node: dict,
        parent: Optional[EnhancedDOMTreeNode] = None,
        target_id: str | None = None,
        frame_id: str | None = None,
        session_id: str | None = None,
    ) -> EnhancedDOMTreeNode:
        """Build an enhanced DOM tree node from a raw CDP node."""
        node_type_value = node.get("nodeType", 1)
        try:
            node_type = NodeType(node_type_value)
        except ValueError:
            node_type = NodeType.ELEMENT_NODE

        attributes = DomService._parse_attributes(node.get("attributes"))

        enhanced_node = EnhancedDOMTreeNode(
            node_id=node.get("nodeId", 0),
            backend_node_id=node.get("backendNodeId", 0),
            node_type=node_type,
            node_name=node.get("nodeName", ""),
            node_value=node.get("nodeValue", "") or "",
            attributes=attributes,
            parent_node=parent,
            target_id=target_id,
            frame_id=frame_id or node.get("frameId"),
            session_id=session_id,
            children_nodes=[],
            shadow_roots=[],
        )

        # Process children
        children = node.get("children", [])
        for child in children:
            child_node = DomService._build_enhanced_node(
                child,
                parent=enhanced_node,
                target_id=target_id,
                frame_id=frame_id,
                session_id=session_id,
            )
            enhanced_node.children_nodes.append(child_node)

        # Process content document (iframes)
        content_document = node.get("contentDocument")
        if content_document:
            enhanced_node.content_document = DomService._build_enhanced_node(
                content_document,
                parent=enhanced_node,
                target_id=target_id,
                frame_id=content_document.get("frameId"),
                session_id=session_id,
            )

        # Process shadow roots
        shadow_roots = node.get("shadowRoots", [])
        for shadow_root in shadow_roots:
            shadow_node = DomService._build_enhanced_node(
                shadow_root,
                parent=enhanced_node,
                target_id=target_id,
                frame_id=frame_id,
                session_id=session_id,
            )
            enhanced_node.shadow_roots.append(shadow_node)

        return enhanced_node

    @staticmethod
    def _traverse_and_filter(
        node: dict, interactive_elements: list[DomNode], distinct_id_counter: int
    ) -> int:
        """Recursively traverse DOM tree and collect interactive elements."""
        if DomService._is_interactive(node):
            attributes = DomService._parse_attributes(node.get("attributes"))
            text = DomService._extract_text(node)
            backend_node_id = node.get("backendNodeId")

            if backend_node_id is not None:
                dom_node = DomNode(
                    tag_name=node.get("nodeName", "").lower(),
                    attributes=attributes,
                    text=text,
                    backend_node_id=backend_node_id,
                    distinct_id=distinct_id_counter,
                )
                interactive_elements.append(dom_node)
                distinct_id_counter += 1

        children = node.get("children", [])
        for child in children:
            distinct_id_counter = DomService._traverse_and_filter(
                child, interactive_elements, distinct_id_counter
            )

        content_document = node.get("contentDocument")
        if content_document:
            distinct_id_counter = DomService._traverse_and_filter(
                content_document, interactive_elements, distinct_id_counter
            )

        shadow_roots = node.get("shadowRoots", [])
        for shadow_root in shadow_roots:
            distinct_id_counter = DomService._traverse_and_filter(
                shadow_root, interactive_elements, distinct_id_counter
            )

        return distinct_id_counter

    @staticmethod
    def _format_element_for_llm(node: DomNode) -> str:
        """Format a DOM node as a string for LLM consumption."""
        tag = node.tag_name or "unknown"
        text = node.text if node.text else ""

        attrs_str = ""
        if node.attributes:
            key_attrs = ["id", "name", "type", "class", "aria-label", "role", "placeholder", "href"]
            attr_parts = []
            for key in key_attrs:
                if key in node.attributes:
                    value = node.attributes[key]
                    if len(value) > 50:
                        value = value[:47] + "..."
                    attr_parts.append(f'{key}="{value}"')

            if attr_parts:
                attrs_str = " " + " ".join(attr_parts)

        if text:
            if len(text) > 100:
                text = text[:97] + "..."
            return f"[{node.distinct_id}] <{tag}{attrs_str}>{text}</{tag}>"
        else:
            return f"[{node.distinct_id}] <{tag}{attrs_str} />"

    @staticmethod
    async def get_clickable_elements(client: CDPClient, session_id: str) -> DomState:
        """Get all clickable/interactive elements from the current page."""
        logger.info("Enabling DOM domain")
        try:
            await client.send.DOM.enable(session_id=session_id)
        except Exception as e:
            logger.warning(f"DOM domain may already be enabled: {e}")

        logger.info("Fetching DOM document with depth=-1")
        dom_result = await client.send.DOM.getDocument(
            params={"depth": -1, "pierce": True}, session_id=session_id
        )

        root_node = dom_result["root"]
        logger.info(f"DOM document fetched, root nodeId: {root_node.get('nodeId')}")

        interactive_elements: list[DomNode] = []
        distinct_id_counter = 1

        logger.info("Traversing DOM tree and filtering interactive elements")
        DomService._traverse_and_filter(root_node, interactive_elements, distinct_id_counter)

        logger.info(f"Found {len(interactive_elements)} interactive elements")

        element_tree_lines = []
        selector_map: dict[int, int] = {}

        for node in interactive_elements:
            element_tree_lines.append(DomService._format_element_for_llm(node))
            selector_map[node.distinct_id] = node.backend_node_id

        element_tree = "\n".join(element_tree_lines)

        logger.info(f"Generated element tree with {len(selector_map)} mapped elements")

        return DomState(element_tree=element_tree, selector_map=selector_map)

    async def get_serialized_dom_state(
        self,
        cdp_session: CDPSession,
        include_attributes: list[str] | None = None,
        serializer_mode: str = "default",
    ) -> SerializedDOMState:
        """Get serialized DOM state with enhanced processing.
        
        Args:
            cdp_session: CDP session to use
            include_attributes: List of attributes to include in serialization
            serializer_mode: Serializer mode - 'default', 'code_use', or 'eval'
        """
        from src.openbrowser.browser.dom.serializer.service import (
            DOMTreeSerializer,
        )
        
        dom_state = await self.get_dom_state(cdp_session)

        # For now, return the existing format
        # TODO: Integrate with SimplifiedNode tree creation and use serializer modes
        # This requires implementing the full serializer pipeline like browser-use
        return SerializedDOMState(
            element_tree=dom_state.element_tree,
            selector_map=dom_state.selector_map,
        )

