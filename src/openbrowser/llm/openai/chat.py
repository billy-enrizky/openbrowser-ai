"""OpenAI LLM integration compatible with LangChain."""

import logging
import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI as LangChainChatOpenAI
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class ChatOpenAI(BaseChatModel):
    """
    LangChain-compatible wrapper for OpenAI chat models.

    This class wraps LangChain's ChatOpenAI and provides a consistent interface
    with other LLM providers in OpenBrowser.

    Args:
        model: The OpenAI model to use (e.g., "gpt-4o", "gpt-4-turbo")
        temperature: Temperature for response generation
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters passed to LangChain's ChatOpenAI
    """

    _model: str = PrivateAttr()
    _temperature: float = PrivateAttr()
    _api_key: str | None = PrivateAttr()
    _max_tokens: int | None = PrivateAttr()
    _llm: LangChainChatOpenAI = PrivateAttr()

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0,
        api_key: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is not set and no api_key provided")

        self._llm = LangChainChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=self._api_key,
            max_tokens=max_tokens,
            **kwargs,
        )

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def _llm_type(self) -> str:
        return "openai"

    def bind_tools(self, tools: list[Any]) -> "ChatOpenAI":
        """Bind tools to this LLM instance for function calling."""
        bound_llm = self._llm.bind_tools(tools)
        wrapped = ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            api_key=self._api_key,
            max_tokens=self._max_tokens,
        )
        wrapped._llm = bound_llm
        return wrapped

    def with_structured_output(self, output_schema: type[Any]) -> Any:
        """Get structured output from the model."""
        return self._llm.with_structured_output(output_schema)

    async def agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a chat response from OpenAI."""
        return await self._llm.agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def ainvoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the model with the given messages."""
        return await self._llm.ainvoke(messages, config=config, **kwargs)

    def invoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronously invoke the model with the given messages."""
        return self._llm.invoke(messages, config=config, **kwargs)

    def generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronously generate a chat response from OpenAI."""
        return self._llm.generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Internal generate method required by BaseChatModel."""
        return self._llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

