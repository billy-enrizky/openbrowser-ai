"""DOM processing service for extracting and filtering interactive elements."""

import logging
from typing import Optional

from cdp_use.client import CDPClient
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Node type constants (from DOM spec)
ELEMENT_NODE = 1
TEXT_NODE = 3

# Interactive HTML tag names
INTERACTIVE_TAGS = {"a", "button", "input", "textarea", "select"}


class DomNode(BaseModel):
    """Represents a simplified DOM node."""

    tag_name: Optional[str] = None
    attributes: dict[str, str] = {}
    text: str = ""
    backend_node_id: int
    distinct_id: int


class DomState(BaseModel):
    """Contains the DOM state for LLM consumption."""

    element_tree: str
    selector_map: dict[int, int]  # distinct_id -> backend_node_id


class DomService:
    """Service for processing DOM and extracting interactive elements."""

    @staticmethod
    def _parse_attributes(attributes_list: Optional[list[str]]) -> dict[str, str]:
        """Parse CDP attributes array into a dictionary.
        
        CDP returns attributes as a flat array: [name1, value1, name2, value2, ...]
        """
        if not attributes_list:
            return {}
        
        attrs = {}
        for i in range(0, len(attributes_list), 2):
            if i + 1 < len(attributes_list):
                attrs[attributes_list[i]] = attributes_list[i + 1]
        return attrs

    @staticmethod
    def _is_interactive(node: dict) -> bool:
        """Check if a node is interactive.
        
        A node is interactive if:
        1. It's an element node (nodeType == 1)
        2. It has an interactive tag name (a, button, input, textarea, select)
        3. OR it has an onclick attribute (or other event handlers)
        """
        # Must be an element node
        if node.get("nodeType") != ELEMENT_NODE:
            return False
        
        node_name = node.get("nodeName", "").lower()
        
        # Check for interactive tag names
        if node_name in INTERACTIVE_TAGS:
            return True
        
        # Check for event handler attributes
        attributes = DomService._parse_attributes(node.get("attributes"))
        event_handlers = {
            "onclick",
            "onmousedown",
            "onmouseup",
            "onkeydown",
            "onkeyup",
            "ontouchstart",
        }
        
        if any(attr.lower() in event_handlers for attr in attributes.keys()):
            return True
        
        return False

    @staticmethod
    def _extract_text(node: dict) -> str:
        """Extract text content from a node and its children recursively."""
        text_parts = []
        
        # Add node's own text value if it's a text node
        if node.get("nodeType") == TEXT_NODE:
            node_value = node.get("nodeValue", "")
            if node_value:
                text_parts.append(node_value.strip())
        
        # Recursively extract text from children
        children = node.get("children", [])
        for child in children:
            child_text = DomService._extract_text(child)
            if child_text:
                text_parts.append(child_text)
        
        return " ".join(text_parts).strip()

    @staticmethod
    def _traverse_and_filter(
        node: dict, interactive_elements: list[DomNode], distinct_id_counter: int
    ) -> int:
        """Recursively traverse DOM tree and collect interactive elements.
        
        Args:
            node: The current DOM node to process
            interactive_elements: List to append interactive elements to
            distinct_id_counter: Current counter for assigning distinct IDs
            
        Returns:
            Updated distinct_id_counter after processing this node and its children
        """
        # Check if current node is interactive
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
        
        # Recursively process children
        children = node.get("children", [])
        for child in children:
            distinct_id_counter = DomService._traverse_and_filter(
                child, interactive_elements, distinct_id_counter
            )
        
        # Process contentDocument (for iframes)
        content_document = node.get("contentDocument")
        if content_document:
            distinct_id_counter = DomService._traverse_and_filter(
                content_document, interactive_elements, distinct_id_counter
            )
        
        # Process shadow roots
        shadow_roots = node.get("shadowRoots", [])
        for shadow_root in shadow_roots:
            distinct_id_counter = DomService._traverse_and_filter(
                shadow_root, interactive_elements, distinct_id_counter
            )
        
        return distinct_id_counter

    @staticmethod
    def _format_element_for_llm(node: DomNode) -> str:
        """Format a DOM node as a string for LLM consumption.
        
        Format: [ID] <tag>text</tag>
        """
        tag = node.tag_name or "unknown"
        text = node.text if node.text else ""
        
        # Build attribute string if there are meaningful attributes
        attrs_str = ""
        if node.attributes:
            # Include key attributes that might be useful
            key_attrs = ["id", "name", "type", "class", "aria-label", "role"]
            attr_parts = []
            for key in key_attrs:
                if key in node.attributes:
                    value = node.attributes[key]
                    attr_parts.append(f'{key}="{value}"')
            
            if attr_parts:
                attrs_str = " " + " ".join(attr_parts)
        
        if text:
            return f"[{node.distinct_id}] <{tag}{attrs_str}>{text}</{tag}>"
        else:
            return f"[{node.distinct_id}] <{tag}{attrs_str} />"

    @staticmethod
    async def get_clickable_elements(
        client: CDPClient, session_id: str
    ) -> DomState:
        """Get all clickable/interactive elements from the current page.
        
        Args:
            client: CDP client instance
            session_id: CDP session ID
            
        Returns:
            DomState containing element_tree string and selector_map
        """
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
        
        # Traverse and collect interactive elements
        interactive_elements: list[DomNode] = []
        distinct_id_counter = 1
        
        logger.info("Traversing DOM tree and filtering interactive elements")
        DomService._traverse_and_filter(root_node, interactive_elements, distinct_id_counter)
        
        logger.info(f"Found {len(interactive_elements)} interactive elements")
        
        # Build element tree string for LLM
        element_tree_lines = []
        selector_map: dict[int, int] = {}
        
        for node in interactive_elements:
            element_tree_lines.append(DomService._format_element_for_llm(node))
            selector_map[node.distinct_id] = node.backend_node_id
        
        element_tree = "\n".join(element_tree_lines)
        
        logger.info(f"Generated element tree with {len(selector_map)} mapped elements")
        
        return DomState(element_tree=element_tree, selector_map=selector_map)

