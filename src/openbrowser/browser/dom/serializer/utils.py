"""Shared utilities for DOM serializers."""


def cap_text_length(text: str, max_length: int = 100) -> str:
    """Cap text length for LLM consumption."""
    if not text:
        return ""
    text = " ".join(text.split())  # Normalize whitespace
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."

