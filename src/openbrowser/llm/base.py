"""Base LLM chat model following browser-use pattern."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Type, Optional

from langchain_core.language_models import BaseChatModel as LangChainBaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseChatModel(ABC):
    """
    Abstract base class for chat models.
    Provides a consistent interface for all LLM providers.
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        """Async invoke the model with messages."""
        pass

    @abstractmethod
    def bind_tools(self, tools: list[Any]) -> BaseChatModel:
        """Bind tools to the model for function calling."""
        pass

    @abstractmethod
    def with_structured_output(
        self,
        schema: Type[T],
        **kwargs: Any,
    ) -> BaseChatModel:
        """Configure the model to return structured output."""
        pass


class LangChainChatModelWrapper(BaseChatModel):
    """
    Wrapper for LangChain chat models to provide a consistent interface.
    """

    def __init__(self, llm: LangChainBaseChatModel):
        self._llm = llm

    @property
    def provider(self) -> str:
        return getattr(self._llm, "provider", "unknown")

    @property
    def model(self) -> str:
        return getattr(self._llm, "model_name", getattr(self._llm, "model", "unknown"))

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        return await self._llm.ainvoke(messages, **kwargs)

    def bind_tools(self, tools: list[Any]) -> LangChainChatModelWrapper:
        bound = self._llm.bind_tools(tools)
        return LangChainChatModelWrapper(bound)

    def with_structured_output(
        self,
        schema: Type[T],
        **kwargs: Any,
    ) -> LangChainChatModelWrapper:
        structured = self._llm.with_structured_output(schema, **kwargs)
        return LangChainChatModelWrapper(structured)

