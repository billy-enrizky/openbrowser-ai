"""Cerebras LLM integration for fast inference."""

import json
import logging
import os
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import PrivateAttr

from src.openbrowser.llm.exceptions import ModelProviderError

logger = logging.getLogger(__name__)


class ChatCerebras(BaseChatModel):
    """
    Chat model for Cerebras fast inference.

    Cerebras provides extremely fast inference for LLMs.
    Uses OpenAI-compatible API format.

    Args:
        model: The Cerebras model to use (e.g., "llama3.1-70b", "llama3.1-8b")
        temperature: Temperature for response generation
        api_key: Cerebras API key (defaults to CEREBRAS_API_KEY env var)
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters
    """

    _model: str = PrivateAttr()
    _temperature: float = PrivateAttr()
    _api_key: str = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _client: httpx.AsyncClient = PrivateAttr()
    _base_url: str = PrivateAttr()

    def __init__(
        self,
        model: str = "llama3.1-70b",
        temperature: float = 0,
        api_key: str | None = None,
        max_tokens: int = 4096,
        base_url: str = "https://api.cerebras.ai/v1",
        **kwargs: Any,
    ):
        super().__init__()
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or os.getenv("CEREBRAS_API_KEY", "")
        self._max_tokens = max_tokens
        self._base_url = base_url
        self._client = httpx.AsyncClient(timeout=120.0)

        if not self._api_key:
            raise ValueError("CEREBRAS_API_KEY is not set and no api_key provided")

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def _llm_type(self) -> str:
        return "cerebras"

    def bind_tools(self, tools: list[Any]) -> "ChatCerebras":
        """Bind tools to this LLM instance."""
        logger.warning("Cerebras tool binding support may be limited")
        return self

    def with_structured_output(self, output_schema: type[Any], **kwargs: Any) -> Any:
        """Get structured output from the model."""
        return self

    async def ainvoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the model with the given messages."""
        from langchain_core.messages import AIMessage

        # Convert messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'type'):
                role = 'system' if msg.type == 'system' else ('assistant' if msg.type == 'ai' else 'user')
                formatted_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)

        request_body = {
            "model": self._model,
            "messages": formatted_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=request_body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return AIMessage(content=content)

        except httpx.HTTPStatusError as e:
            raise ModelProviderError(
                message=f"Cerebras API error: {e.response.text}",
                status_code=e.response.status_code,
                model=self._model,
            )

    def invoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronously invoke the model."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(messages, config, **kwargs)
        )

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Internal generate method required by BaseChatModel."""
        from langchain_core.outputs import ChatGeneration, ChatResult
        result = self.invoke(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=result)])

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

