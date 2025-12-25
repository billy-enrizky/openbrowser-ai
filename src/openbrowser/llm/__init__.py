"""LLM integrations for OpenBrowser."""

from src.openbrowser.llm.google import ChatGoogle
from src.openbrowser.llm.openai import ChatOpenAI

__all__ = ["ChatGoogle", "ChatOpenAI"]

