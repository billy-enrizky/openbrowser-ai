"""Shared utilities for DOM serializers.

This module provides common utility functions used across different
serializer implementations.

Functions:
    cap_text_length: Truncate text with ellipsis for LLM consumption.
"""


def cap_text_length(text: str, max_length: int = 100) -> str:
    """Cap text length for LLM consumption.

    Normalizes whitespace (collapses multiple spaces) and truncates
    long text with ellipsis to fit within token budgets.

    Args:
        text: Input text to process.
        max_length: Maximum length before truncation (default: 100).

    Returns:
        Normalized text, truncated with '...' if exceeds max_length.

    Example:
        >>> cap_text_length('Hello   World', 50)
        'Hello World'
        >>> cap_text_length('A' * 150, 100)
        'AAA...AAA...'  # 97 chars + '...'
    """
    if not text:
        return ""
    text = " ".join(text.split())  # Normalize whitespace
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."

