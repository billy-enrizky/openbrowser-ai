"""DOM tree serializer following browser-use pattern.

This module provides the main DOMTreeSerializer class for converting
SimplifiedNode trees into LLM-readable string formats. Supports
multiple serialization modes and paint order visibility filtering.

Classes:
    DOMTreeSerializer: Main serializer with multiple output formats.
    ClickableElementsSerializer: Focused serializer for interactive elements only.

Example:
    >>> serializer = DOMTreeSerializer(paint_order_filtering=True)
    >>> html = DOMTreeSerializer.serialize_tree(simplified_root)
    >>> code_html = DOMTreeSerializer.serialize_for_code_use(simplified_root)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.openbrowser.browser.dom.views import SimplifiedNode, EnhancedDOMTreeNode

from src.openbrowser.browser.dom.views import (
    DEFAULT_INCLUDE_ATTRIBUTES,
    NodeType,
)

from .utils import cap_text_length
from .clickable_elements import ClickableElementDetector
from .paint_order import PaintOrderRemover
from .code_use_serializer import DOMCodeAgentSerializer
from .eval_serializer import DOMEvalSerializer

logger = logging.getLogger(__name__)


class DOMTreeSerializer:
    """Serializes DOM tree for LLM consumption.

    Converts SimplifiedNode trees to human-readable string formats suitable
    for LLM processing. Supports multiple serialization modes:

    - Default: Balanced format with structure and attributes
    - Code-use: Token-efficient format for code generation
    - Eval: Verbose format for evaluation/testing

    Features:
        - Paint order filtering to hide overlapping elements
        - Interactive element detection and indexing
        - Attribute filtering and truncation
        - Shadow DOM and iframe handling

    Class Attributes:
        INTERACTIVE_TAGS: Tags considered interactive.
        IGNORED_TAGS: Tags skipped during serialization.

    Example:
        >>> html = DOMTreeSerializer.serialize_tree(
        ...     root=simplified_root,
        ...     include_attributes=['id', 'class', 'type'],
        ...     paint_order_filtering=True
        ... )
    """

    # Tags that are typically interactive
    INTERACTIVE_TAGS = {
        "a",
        "button",
        "input",
        "textarea",
        "select",
        "option",
        "label",
        "summary",
        "details",
    }

    # Tags that should be ignored for display
    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "meta",
        "link",
        "head",
        "title",
        "svg",
        "path",
    }

    def __init__(self, paint_order_filtering: bool = True):
        """Initialize serializer with optional paint order filtering.

        Args:
            paint_order_filtering: If True, filters elements hidden by
                overlapping elements with higher paint order.
        """
        self.paint_order_filtering = paint_order_filtering
        self._clickable_cache: dict[int, bool] = {}

    def _is_interactive_cached(self, node: EnhancedDOMTreeNode) -> bool:
        """Cached version of clickable element detection.

        Avoids redundant interactivity checks by caching results
        by node_id.

        Args:
            node: DOM tree node to check.

        Returns:
            True if node is interactive.
        """
        if node.node_id not in self._clickable_cache:
            result = ClickableElementDetector.is_interactive(node)
            self._clickable_cache[node.node_id] = result
        return self._clickable_cache[node.node_id]

    @classmethod
    def serialize_tree(
        cls,
        root: SimplifiedNode,
        include_attributes: list[str] | None = None,
        paint_order_filtering: bool = True,
    ) -> str:
        """Serialize the simplified DOM tree to a string for LLM.

        Default serialization mode with balanced detail and structure.

        Args:
            root: Root SimplifiedNode of the tree.
            include_attributes: Attributes to include in output.
                Defaults to DEFAULT_INCLUDE_ATTRIBUTES.
            paint_order_filtering: If True, apply paint order filtering
                to hide overlapping elements.

        Returns:
            Multi-line string representation of the DOM tree.
        """
        if include_attributes is None:
            include_attributes = DEFAULT_INCLUDE_ATTRIBUTES

        # Apply paint order filtering if enabled
        if paint_order_filtering and root:
            PaintOrderRemover(root).calculate_paint_order()

        lines: list[str] = []
        cls._serialize_node(root, lines, include_attributes, 0)
        return "\n".join(lines)

    @classmethod
    def serialize_for_code_use(
        cls,
        root: SimplifiedNode,
        include_attributes: list[str] | None = None,
        paint_order_filtering: bool = True,
    ) -> str:
        """Serialize DOM tree using code-use optimized serializer.

        Token-efficient format optimized for code generation agents.
        Minimizes output while preserving essential interactive context.

        Args:
            root: Root SimplifiedNode of the tree.
            include_attributes: Attributes to include in output.
            paint_order_filtering: If True, apply paint order filtering.

        Returns:
            Compact string representation optimized for code agents.
        """
        if include_attributes is None:
            include_attributes = DEFAULT_INCLUDE_ATTRIBUTES

        # Apply paint order filtering if enabled
        if paint_order_filtering and root:
            PaintOrderRemover(root).calculate_paint_order()

        return DOMCodeAgentSerializer.serialize_tree(root, include_attributes, 0)

    @classmethod
    def serialize_for_eval(
        cls,
        root: SimplifiedNode,
        include_attributes: list[str] | None = None,
        paint_order_filtering: bool = True,
    ) -> str:
        """Serialize DOM tree using eval optimized serializer.

        Verbose format with complete DOM structure for evaluation
        and testing scenarios.

        Args:
            root: Root SimplifiedNode of the tree.
            include_attributes: Attributes to include in output.
            paint_order_filtering: If True, apply paint order filtering.

        Returns:
            Detailed string representation for evaluation.
        """
        if include_attributes is None:
            include_attributes = DEFAULT_INCLUDE_ATTRIBUTES

        # Apply paint order filtering if enabled
        if paint_order_filtering and root:
            PaintOrderRemover(root).calculate_paint_order()

        return DOMEvalSerializer.serialize_tree(root, include_attributes, 0)

    @classmethod
    def _serialize_node(
        cls,
        node: SimplifiedNode,
        lines: list[str],
        include_attributes: list[str],
        depth: int,
    ) -> None:
        """Recursively serialize a node and its children.

        Internal method that builds the serialized output by appending
        lines to the provided list.

        Args:
            node: Current SimplifiedNode to serialize.
            lines: List to append output lines to.
            include_attributes: Attributes to include.
            depth: Current indentation depth.
        """
        original = node.original_node

        # Skip ignored tags
        if original.tag_name in cls.IGNORED_TAGS:
            return

        # Skip non-displayable nodes
        if not node.should_display:
            # Still process children
            for child in node.children:
                cls._serialize_node(child, lines, include_attributes, depth)
            return

        # Skip text nodes with only whitespace
        if original.node_type == NodeType.TEXT_NODE:
            text = original.node_value.strip()
            if text:
                # Indent text content
                indent = "  " * depth
                lines.append(f"{indent}{cap_text_length(text)}")
            return

        # Skip document fragments but process children
        if original.node_type == NodeType.DOCUMENT_FRAGMENT_NODE:
            for child in node.children:
                cls._serialize_node(child, lines, include_attributes, depth)
            return

        # Only process element nodes
        if original.node_type != NodeType.ELEMENT_NODE:
            return

        # Build the element line
        indent = "  " * depth
        tag = original.tag_name

        # Get element index if interactive
        element_index = None
        if node.is_interactive:
            # Find the element index from the parent context
            # This should be set during tree building
            element_index = getattr(node, "element_index", None)

        # Build attribute string
        attr_parts = []
        for attr_name in include_attributes:
            if attr_name in original.attributes:
                value = original.attributes[attr_name]
                if value:
                    # Truncate long attribute values
                    if len(value) > 50:
                        value = value[:47] + "..."
                    attr_parts.append(f'{attr_name}="{value}"')

        attrs_str = " ".join(attr_parts)
        if attrs_str:
            attrs_str = " " + attrs_str

        # Get text content
        text_content = ""
        if original.node_type == NodeType.ELEMENT_NODE:
            text_content = cls._get_element_text(original)

        # Format the element
        if element_index is not None:
            if text_content:
                line = f"{indent}[{element_index}] <{tag}{attrs_str}>{text_content}</{tag}>"
            else:
                line = f"{indent}[{element_index}] <{tag}{attrs_str} />"
        else:
            if text_content:
                line = f"{indent}<{tag}{attrs_str}>{text_content}</{tag}>"
            else:
                line = f"{indent}<{tag}{attrs_str} />"

        lines.append(line)

        # Process children
        for child in node.children:
            cls._serialize_node(child, lines, include_attributes, depth + 1)

    @classmethod
    def _get_element_text(cls, node: EnhancedDOMTreeNode, max_length: int = 100) -> str:
        """Get meaningful text content from an element.

        Checks meaningful attributes (value, aria-label, title, placeholder,
        alt) first, then falls back to child text content.

        Args:
            node: DOM tree node to extract text from.
            max_length: Maximum text length (default: 100).

        Returns:
            Truncated text content or empty string.
        """
        # Check meaningful attributes first
        for attr in ["value", "aria-label", "title", "placeholder", "alt"]:
            if attr in node.attributes and node.attributes[attr]:
                return cap_text_length(node.attributes[attr], max_length)

        # Fall back to text content
        text = node.get_all_children_text()
        return cap_text_length(text, max_length)


class ClickableElementsSerializer:
    """Serializer focused on clickable/interactive elements.

    Outputs only interactive elements, ignoring non-interactive
    structure. Useful for action-focused agent prompts.

    Class Attributes:
        INTERACTIVE_TAGS: Tags considered interactive.

    Example:
        >>> html = ClickableElementsSerializer.serialize(simplified_root)
        >>> # Output contains only buttons, links, inputs, etc.
    """

    INTERACTIVE_TAGS = {
        "a",
        "button",
        "input",
        "textarea",
        "select",
        "option",
        "label",
        "summary",
    }

    @classmethod
    def serialize(
        cls,
        root: SimplifiedNode,
        include_attributes: list[str] | None = None,
    ) -> str:
        """Serialize only interactive elements.

        Args:
            root: Root SimplifiedNode of the tree.
            include_attributes: Attributes to include in output.

        Returns:
            String containing only interactive elements.
        """
        if include_attributes is None:
            include_attributes = DEFAULT_INCLUDE_ATTRIBUTES

        lines: list[str] = []
        cls._collect_interactive(root, lines, include_attributes)
        return "\n".join(lines)

    @classmethod
    def _collect_interactive(
        cls,
        node: SimplifiedNode,
        lines: list[str],
        include_attributes: list[str],
    ) -> None:
        """Collect interactive elements from the tree.

        Recursively traverses the tree and appends formatted lines
        for each interactive element.

        Args:
            node: Current SimplifiedNode to process.
            lines: List to append output lines to.
            include_attributes: Attributes to include.
        """
        original = node.original_node

        if node.is_interactive and original.node_type == NodeType.ELEMENT_NODE:
            element_index = getattr(node, "element_index", None)
            tag = original.tag_name

            # Build attribute string
            attr_parts = []
            for attr_name in include_attributes:
                if attr_name in original.attributes:
                    value = original.attributes[attr_name]
                    if value:
                        if len(value) > 50:
                            value = value[:47] + "..."
                        attr_parts.append(f'{attr_name}="{value}"')

            attrs_str = " ".join(attr_parts)
            if attrs_str:
                attrs_str = " " + attrs_str

            text = cap_text_length(original.get_all_children_text(), 100)

            if element_index is not None:
                if text:
                    lines.append(f"[{element_index}] <{tag}{attrs_str}>{text}</{tag}>")
                else:
                    lines.append(f"[{element_index}] <{tag}{attrs_str} />")

        # Process children
        for child in node.children:
            cls._collect_interactive(child, lines, include_attributes)

