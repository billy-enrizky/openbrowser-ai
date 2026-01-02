"""DOM processing service for extracting and filtering interactive elements.

This module provides a lightweight DOM extraction service that identifies
interactive elements (links, buttons, inputs, etc.) from a page via CDP.
It produces a simplified element tree suitable for LLM consumption.

Constants:
    ELEMENT_NODE: DOM element node type (1).
    TEXT_NODE: DOM text node type (3).
    INTERACTIVE_TAGS: Set of interactive HTML tag names.

Classes:
    DomNode: Simplified representation of an interactive DOM element.
    DomState: Container for element tree string and selector mapping.
    DomService: Service for DOM traversal and interactive element extraction.

Example:
    >>> dom_state = await DomService.get_clickable_elements(client, session_id)
    >>> print(dom_state.element_tree)  # Human-readable element list
    >>> backend_id = dom_state.selector_map[1]  # Get backend node ID
"""

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
    """Represents a simplified DOM node for interactive elements.

    This is a lightweight representation of an interactive DOM element,
    containing only the information needed for LLM-based interaction.

    Attributes:
        tag_name: The lowercase HTML tag name (e.g., 'button', 'input').
        attributes: Dictionary of element attributes.
        text: Combined text content from the element and its children.
        backend_node_id: CDP backend node ID for targeting the element.
        distinct_id: Unique identifier assigned during DOM traversal.

    Example:
        >>> node = DomNode(
        ...     tag_name='button',
        ...     attributes={'class': 'submit-btn'},
        ...     text='Submit',
        ...     backend_node_id=123,
        ...     distinct_id=1
        ... )
    """

    tag_name: Optional[str] = None
    attributes: dict[str, str] = {}
    text: str = ""
    backend_node_id: int
    distinct_id: int


class DomState(BaseModel):
    """Contains the DOM state for LLM consumption.

    Encapsulates both the human-readable element tree string and
    the mapping from distinct IDs to CDP backend node IDs.

    Attributes:
        element_tree: Formatted string listing interactive elements
            with IDs, suitable for LLM consumption.
        selector_map: Mapping from distinct_id to backend_node_id,
            used to resolve LLM element references for CDP commands.

    Example:
        >>> state = DomState(
        ...     element_tree='[1] <button>Submit</button>',
        ...     selector_map={1: 123}
        ... )
        >>> backend_id = state.selector_map[1]  # 123
    """

    element_tree: str
    selector_map: dict[int, int]  # distinct_id -> backend_node_id


class DomService:
    """Service for processing DOM and extracting interactive elements.

    Provides static methods for traversing the CDP DOM tree, identifying
    interactive elements, and formatting them for LLM consumption.

    The main entry point is `get_clickable_elements()`, which:
    1. Enables the DOM domain via CDP
    2. Fetches the full DOM tree with shadow DOM piercing
    3. Traverses and filters for interactive elements
    4. Formats elements into an LLM-readable string
    5. Returns a mapping for resolving element references

    Interactive elements include:
        - Links (<a>)
        - Buttons (<button>)
        - Form controls (<input>, <textarea>, <select>)
        - Elements with event handlers (onclick, etc.)

    Example:
        >>> dom_state = await DomService.get_clickable_elements(client, session_id)
        >>> print(dom_state.element_tree)
        [1] <a href="/home">Home</a>
        [2] <button type="submit">Submit</button>
    """

    @staticmethod
    def _parse_attributes(attributes_list: Optional[list[str]]) -> dict[str, str]:
        """Parse CDP attributes array into a dictionary.

        CDP returns attributes as a flat array: [name1, value1, name2, value2, ...]
        This method converts it to a standard dictionary.

        Args:
            attributes_list: Flat list of alternating attribute names and values,
                or None if no attributes present.

        Returns:
            Dictionary mapping attribute names to values.

        Example:
            >>> DomService._parse_attributes(['id', 'submit-btn', 'class', 'primary'])
            {'id': 'submit-btn', 'class': 'primary'}
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

        A node is considered interactive if all conditions are met:
        1. It's an element node (nodeType == 1)
        2. It has an interactive tag name OR has event handler attributes

        Interactive tags: a, button, input, textarea, select.
        Event handlers: onclick, onmousedown, onmouseup, onkeydown, onkeyup, ontouchstart.

        Args:
            node: CDP DOM node dictionary with nodeType, nodeName, attributes.

        Returns:
            True if the node is interactive, False otherwise.

        Example:
            >>> DomService._is_interactive({'nodeType': 1, 'nodeName': 'BUTTON'})
            True
            >>> DomService._is_interactive({'nodeType': 3, 'nodeName': '#text'})
            False
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
        """Extract text content from a node and its children recursively.

        Collects all text node values from the subtree and joins them
        with spaces. Text is stripped of leading/trailing whitespace.

        Args:
            node: CDP DOM node dictionary with children and nodeValue.

        Returns:
            Combined text content, space-separated and stripped.

        Example:
            >>> # Node tree: <span>Hello <b>World</b></span>
            >>> DomService._extract_text(span_node)
            'Hello World'
        """
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

        Performs depth-first traversal of the DOM tree, including:
        - Regular child nodes
        - Iframe content documents (pierce: True)
        - Shadow DOM roots

        Interactive elements are appended to the provided list with
        sequentially assigned distinct IDs.

        Args:
            node: Current CDP DOM node to process.
            interactive_elements: List to append found interactive elements.
            distinct_id_counter: Current ID counter for assigning distinct IDs.

        Returns:
            Updated distinct_id_counter after processing this subtree.

        Note:
            This method modifies interactive_elements in place.
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

        Produces a human-readable representation with:
        - Distinct ID in brackets: [ID]
        - Tag name with key attributes: <tag attr="value">
        - Text content if present: >text</tag>

        Key attributes included: id, name, type, class, aria-label, role.

        Args:
            node: DomNode to format.

        Returns:
            Formatted string like '[1] <button class="primary">Submit</button>'.

        Example:
            >>> node = DomNode(distinct_id=1, tag_name='button', text='Submit', ...)
            >>> DomService._format_element_for_llm(node)
            '[1] <button>Submit</button>'
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

        Main entry point for DOM extraction. Enables the DOM domain,
        fetches the complete DOM tree (with shadow DOM piercing), and
        extracts interactive elements.

        Args:
            client: CDP client instance for sending commands.
            session_id: CDP session ID for the target page.

        Returns:
            DomState containing:
                - element_tree: Formatted string of interactive elements.
                - selector_map: Dict mapping distinct_id to backend_node_id.

        Example:
            >>> state = await DomService.get_clickable_elements(client, session_id)
            >>> print(state.element_tree)
            [1] <a href="/">Home</a>
            [2] <button>Login</button>
            >>> backend_id = state.selector_map[2]  # Get backend ID for button
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

