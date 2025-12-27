"""Browser-use cloud LLM integration.

This module provides integration with browser-use's hosted LLM endpoint,
allowing users to leverage browser-use's cloud infrastructure for LLM calls.
"""

import json
import logging
import os
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import PrivateAttr

from src.openbrowser.llm.exceptions import ModelProviderError

logger = logging.getLogger(__name__)


class ChatBrowserUse(BaseChatModel):
    """
    Chat model for browser-use's hosted LLM endpoint.

    This provides access to browser-use's cloud LLM service, which handles
    model selection and optimization for browser automation tasks.

    Args:
        api_key: Browser-use API key (defaults to BROWSER_USE_API_KEY env var)
        base_url: Browser-use API base URL
        model: Model preference (optional, service may auto-select)
        temperature: Temperature for response generation
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters
    """

    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr()
    _model: str = PrivateAttr()
    _temperature: float = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.browser-use.com",
        model: str = "auto",
        temperature: float = 0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__()
        self._api_key = api_key or os.getenv("BROWSER_USE_API_KEY", "")
        self._base_url = base_url.rstrip('/')
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=120.0)

        if not self._api_key:
            raise ValueError("BROWSER_USE_API_KEY is not set and no api_key provided")

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def _llm_type(self) -> str:
        return "browser-use"

    def bind_tools(self, tools: list[Any]) -> "ChatBrowserUse":
        """Bind tools to this LLM instance."""
        # Browser-use cloud handles tool binding internally
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

        # Convert messages to OpenAI-compatible format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'type'):
                role = 'system' if msg.type == 'system' else ('assistant' if msg.type == 'ai' else 'user')
                content = msg.content
                
                # Handle multimodal content
                if isinstance(content, list):
                    formatted_content = []
                    for part in content:
                        if hasattr(part, 'type'):
                            if part.type == 'text':
                                formatted_content.append({"type": "text", "text": part.text})
                            elif part.type == 'image_url':
                                formatted_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": part.image_url.url}
                                })
                    content = formatted_content
                
                formatted_messages.append({"role": role, "content": content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)

        request_body = {
            "model": self._model,
            "messages": formatted_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        # Add any structured output schema if provided
        output_format = kwargs.get("output_format")
        if output_format:
            schema = output_format.model_json_schema()
            request_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_format.__name__,
                    "schema": schema,
                }
            }

        try:
            response = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=request_body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "X-Client": "openbrowser",
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Handle structured output
            if output_format and content:
                try:
                    parsed = json.loads(content)
                    return AIMessage(content=content, additional_kwargs={"parsed": parsed})
                except json.JSONDecodeError:
                    pass

            return AIMessage(content=content)

        except httpx.HTTPStatusError as e:
            raise ModelProviderError(
                message=f"Browser-use API error: {e.response.text}",
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

    # Browser-use specific methods

    async def get_usage(self) -> dict[str, Any]:
        """Get API usage statistics."""
        try:
            response = await self._client.get(
                f"{self._base_url}/v1/usage",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get usage: {e}")
            return {}

    async def get_available_models(self) -> list[str]:
        """Get list of available models."""
        try:
            response = await self._client.get(
                f"{self._base_url}/v1/models",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get models: {e}")
            return []

