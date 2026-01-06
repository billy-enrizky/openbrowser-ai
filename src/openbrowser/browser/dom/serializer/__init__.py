"""DOM serialization module for LLM consumption.

This module provides various serializers for converting enhanced DOM trees
into formats optimized for different LLM agent modes.

Serializers:
    DOMTreeSerializer: Standard serializer with paint order filtering.
    DOMCodeAgentSerializer: Token-efficient format for code generation.
    DOMEvalSerializer: Verbose format for evaluation/testing.

Utilities:
    ClickableElementDetector: Enhanced interactive element detection.
    PaintOrderRemover: Filters hidden/overlapping elements.
    Rect: Axis-aligned rectangle for geometry operations.
    RectUnionPure: Disjoint rectangle union for visibility tracking.

Example:
    >>> from openbrowser.browser.dom.serializer import DOMTreeSerializer
    >>> html = DOMTreeSerializer.serialize_tree(simplified_root)
"""

from .service import DOMTreeSerializer
from .code_use_serializer import DOMCodeAgentSerializer
from .eval_serializer import DOMEvalSerializer
from .clickable_elements import ClickableElementDetector
from .paint_order import PaintOrderRemover, Rect, RectUnionPure

__all__ = [
    "DOMTreeSerializer",
    "DOMCodeAgentSerializer",
    "DOMEvalSerializer",
    "ClickableElementDetector",
    "PaintOrderRemover",
    "Rect",
    "RectUnionPure",
]

