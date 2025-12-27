"""DeepSeek LLM integration with reasoning support."""

import json
import logging
import os
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import PrivateAttr

from src.openbrowser.llm.exceptions import ModelProviderError

logger = logging.getLogger(__name__)


class ChatDeepSeek(BaseChatModel):
    """
    Chat model for DeepSeek with reasoning support.

    DeepSeek provides models with strong reasoning capabilities.
    Uses OpenAI-compatible API format.

    Args:
        model: The DeepSeek model to use (e.g., "deepseek-chat", "deepseek-reasoner")
        temperature: Temperature for response generation
        api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters
    """

    _model: str = PrivateAttr()
    _temperature: float = PrivateAttr()
    _api_key: str = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _client: httpx.AsyncClient = PrivateAttr()
    _base_url: str = PrivateAttr()
    _reasoning_enabled: bool = PrivateAttr()

    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0,
        api_key: str | None = None,
        max_tokens: int = 4096,
        base_url: str = "https://api.deepseek.com/v1",
        reasoning_enabled: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._max_tokens = max_tokens
        self._base_url = base_url
        self._reasoning_enabled = reasoning_enabled
        self._client = httpx.AsyncClient(timeout=120.0)

        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set and no api_key provided")

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def bind_tools(self, tools: list[Any]) -> "ChatDeepSeek":
        """Bind tools to this LLM instance."""
        # DeepSeek supports function calling
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

        # Add reasoning mode if enabled
        if self._reasoning_enabled:
            request_body["response_format"] = {"type": "text"}

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

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # Extract reasoning if present
            reasoning_content = message.get("reasoning_content", "")
            if reasoning_content:
                logger.debug(f"DeepSeek reasoning: {reasoning_content[:200]}...")

            return AIMessage(content=content)

        except httpx.HTTPStatusError as e:
            raise ModelProviderError(
                message=f"DeepSeek API error: {e.response.text}",
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

