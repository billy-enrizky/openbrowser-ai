"""Base LLM chat model following browser-use pattern.

This module defines the abstract base class for all chat models in OpenBrowser,
as well as a wrapper for integrating LangChain chat models. All provider-specific
implementations inherit from these base classes to ensure a consistent interface.

The base classes provide:
- Unified async/sync invocation methods
- Tool binding for function calling
- Structured output support
- Provider and model identification

Example:
    >>> class CustomChatModel(BaseChatModel):
    ...     @property
    ...     def provider(self) -> str:
    ...         return "custom"
    ...     @property  
    ...     def model(self) -> str:
    ...         return "custom-model"
    ...     async def ainvoke(self, messages, **kwargs):
    ...         # Implementation
    ...         pass
"""

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
    """Abstract base class for chat models.
    
    This class defines the interface that all LLM provider implementations must follow.
    It provides a consistent API for invoking models, binding tools for function calling,
    and configuring structured output.
    
    All methods that interact with the LLM should be implemented as async methods
    to support non-blocking I/O when making API calls.
    
    Subclasses must implement:
        - provider: Property returning the provider name (e.g., 'openai', 'anthropic')
        - model: Property returning the model identifier
        - ainvoke: Async method for invoking the model with messages
        - bind_tools: Method for binding tools/functions to the model
        - with_structured_output: Method for configuring structured JSON output
    
    Example:
        >>> class MyChatModel(BaseChatModel):
        ...     @property
        ...     def provider(self) -> str:
        ...         return "my_provider"
        ...     # ... implement other abstract methods
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name.
        
        Returns:
            str: The name of the LLM provider (e.g., 'openai', 'anthropic', 'google').
                This is used for logging, debugging, and provider-specific handling.
        """
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model name.
        
        Returns:
            str: The model identifier (e.g., 'gpt-4o', 'claude-3-sonnet').
                This should match the model name used in API requests.
        """
        pass

    @abstractmethod
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        """Asynchronously invoke the model with messages.
        
        This is the primary method for generating responses from the LLM.
        It should handle message formatting, API calls, and response parsing.
        
        Args:
            messages: A list of LangChain BaseMessage objects representing
                the conversation history. Can include SystemMessage, HumanMessage,
                and AIMessage instances.
            **kwargs: Additional provider-specific parameters such as:
                - stop: List of stop sequences
                - temperature: Override default temperature
                - max_tokens: Override default max tokens
        
        Returns:
            BaseMessage: The model's response, typically an AIMessage.
                May include tool_calls if tools are bound.
        
        Raises:
            ModelProviderError: If the API call fails.
            ModelRateLimitError: If rate limits are exceeded.
            ModelAuthenticationError: If authentication fails.
        """
        pass

    @abstractmethod
    def bind_tools(self, tools: list[Any]) -> BaseChatModel:
        """Bind tools to the model for function calling.
        
        Creates a new model instance with the specified tools bound, enabling
        the model to generate tool/function calls in its responses.
        
        Args:
            tools: A list of tools to bind. Can be LangChain Tool objects,
                Pydantic models, or dictionaries with the following structure:
                {
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "description": "Tool description",
                        "parameters": {...}  # JSON Schema
                    }
                }
        
        Returns:
            BaseChatModel: A new model instance with tools bound. The original
                instance is not modified.
        
        Example:
            >>> from langchain_core.tools import tool
            >>> @tool
            ... def search(query: str) -> str:
            ...     '''Search the web.'''
            ...     return "results"
            >>> bound_model = model.bind_tools([search])
        """
        pass

    @abstractmethod
    def with_structured_output(
        self,
        schema: Type[T],
        **kwargs: Any,
    ) -> BaseChatModel:
        """Configure the model to return structured output.
        
        Creates a new model instance that will generate responses conforming
        to the specified Pydantic schema. This is typically implemented using
        tool/function calling or JSON mode.
        
        Args:
            schema: A Pydantic BaseModel subclass defining the expected output
                structure. The model will generate JSON matching this schema.
            **kwargs: Additional provider-specific parameters for structured output.
        
        Returns:
            BaseChatModel: A new model instance configured for structured output.
                When invoked, responses will be validated against the schema.
        
        Example:
            >>> from pydantic import BaseModel
            >>> class Answer(BaseModel):
            ...     reasoning: str
            ...     answer: str
            >>> structured_model = model.with_structured_output(Answer)
            >>> result = await structured_model.ainvoke(messages)
        """
        pass


class LangChainChatModelWrapper(BaseChatModel):
    """Wrapper for LangChain chat models to provide a consistent interface.
    
    This class wraps any LangChain-compatible chat model and adapts it to
    the OpenBrowser BaseChatModel interface. Use this when you want to integrate
    an existing LangChain model with OpenBrowser.
    
    The wrapper delegates all operations to the underlying LangChain model
    while providing the standard BaseChatModel interface.
    
    Args:
        llm: A LangChain BaseChatModel instance to wrap.
    
    Attributes:
        _llm: The wrapped LangChain chat model instance.
    
    Example:
        >>> from langchain_openai import ChatOpenAI as LCChatOpenAI
        >>> lc_model = LCChatOpenAI(model="gpt-4o")
        >>> wrapped = LangChainChatModelWrapper(lc_model)
        >>> response = await wrapped.ainvoke(messages)
    """

    def __init__(self, llm: LangChainBaseChatModel):
        """Initialize the wrapper with a LangChain chat model.
        
        Args:
            llm: The LangChain chat model instance to wrap. Should be an instance
                of langchain_core.language_models.BaseChatModel or a compatible subclass.
        """
        self._llm = llm

    @property
    def provider(self) -> str:
        """Return the provider name from the wrapped model.
        
        Returns:
            str: The provider name if available on the wrapped model,
                otherwise 'unknown'.
        """
        return getattr(self._llm, "provider", "unknown")

    @property
    def model(self) -> str:
        """Return the model name from the wrapped model.
        
        Returns:
            str: The model name, checking 'model_name' first, then 'model',
                otherwise 'unknown'.
        """
        return getattr(self._llm, "model_name", getattr(self._llm, "model", "unknown"))

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        """Asynchronously invoke the wrapped model.
        
        Delegates to the wrapped LangChain model's ainvoke method.
        
        Args:
            messages: List of messages to send to the model.
            **kwargs: Additional arguments passed to the wrapped model.
        
        Returns:
            BaseMessage: The model's response.
        """
        return await self._llm.ainvoke(messages, **kwargs)

    def bind_tools(self, tools: list[Any]) -> LangChainChatModelWrapper:
        """Bind tools to the wrapped model.
        
        Args:
            tools: List of tools to bind to the model.
        
        Returns:
            LangChainChatModelWrapper: A new wrapper around the tool-bound model.
        """
        bound = self._llm.bind_tools(tools)
        return LangChainChatModelWrapper(bound)

    def with_structured_output(
        self,
        schema: Type[T],
        **kwargs: Any,
    ) -> LangChainChatModelWrapper:
        """Configure the wrapped model for structured output.
        
        Args:
            schema: Pydantic model class defining the output structure.
            **kwargs: Additional arguments for structured output configuration.
        
        Returns:
            LangChainChatModelWrapper: A new wrapper around the configured model.
        """
        structured = self._llm.with_structured_output(schema, **kwargs)
        return LangChainChatModelWrapper(structured)

