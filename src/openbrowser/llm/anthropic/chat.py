"""Anthropic Claude chat model wrapper."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Type, TypeVar

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ChatAnthropic(BaseChatModel):
    """
    LangChain-compatible wrapper for Anthropic Claude models.
    Requires: pip install anthropic
    """

    model_name: str = Field(default="claude-sonnet-4-20250514", alias="model")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)
    api_key: Optional[str] = Field(default=None)
    timeout: float = Field(default=60.0)
    max_retries: int = Field(default=3)

    _client: Any = PrivateAttr(default=None)
    _bound_tools: List[Any] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._bound_tools = []
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @property
    def _llm_type(self) -> str:
        return "anthropic"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation - raises NotImplementedError."""
        raise NotImplementedError("Use ainvoke for async generation")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation with Anthropic API."""
        anthropic_messages = self._convert_messages(messages)
        system_message = None
        filtered_messages = []

        for msg in anthropic_messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)

        request_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": filtered_messages,
        }

        if system_message:
            request_params["system"] = system_message

        if self._bound_tools:
            request_params["tools"] = self._bound_tools

        if stop:
            request_params["stop_sequences"] = stop

        response = await self._client.messages.create(**request_params)

        # Convert response to LangChain format
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                })

        message = AIMessage(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to Anthropic format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    # Handle multimodal content
                    anthropic_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                anthropic_content.append({"type": "text", "text": item["text"]})
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:image"):
                                    # Extract base64 data
                                    media_type = image_url.split(";")[0].split(":")[1]
                                    base64_data = image_url.split(",")[1]
                                    anthropic_content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": base64_data,
                                        }
                                    })
                        else:
                            anthropic_content.append({"type": "text", "text": str(item)})
                    result.append({"role": "user", "content": anthropic_content})
                else:
                    result.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> ChatAnthropic:
        """Bind tools for function calling."""
        new_instance = ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )
        new_instance._bound_tools = self._convert_tools(tools)
        return new_instance

    def _convert_tools(self, tools: List[Any]) -> List[dict]:
        """Convert tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                schema = {}
                if hasattr(tool, "args_schema"):
                    schema = tool.args_schema.model_json_schema()
                anthropic_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": schema,
                })
            elif isinstance(tool, dict):
                anthropic_tools.append(tool)
        return anthropic_tools

    def with_structured_output(
        self,
        schema: Type[T],
        **kwargs: Any,
    ) -> ChatAnthropic:
        """Configure for structured output using tool use."""
        tool = {
            "name": schema.__name__,
            "description": f"Output structured data as {schema.__name__}",
            "input_schema": schema.model_json_schema(),
        }
        return self.bind_tools([tool])

