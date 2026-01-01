"""Ollama chat model wrapper for local models."""

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


class ChatOllama(BaseChatModel):
    """
    LangChain-compatible wrapper for Ollama local models.
    Requires: pip install ollama
    """

    model_name: str = Field(default="llama3.2", alias="model")
    temperature: float = Field(default=0.0)
    base_url: str = Field(default="http://localhost:11434")
    timeout: float = Field(default=120.0)

    _client: Any = PrivateAttr(default=None)
    _bound_tools: List[Any] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._bound_tools = []
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Ollama client."""
        try:
            from ollama import AsyncClient
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")

        self._client = AsyncClient(host=self.base_url, timeout=self.timeout)

    @property
    def _llm_type(self) -> str:
        return "ollama"

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
        raise NotImplementedError("Use ainvoke for async generation")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation with Ollama API."""
        ollama_messages = self._convert_messages(messages)

        options = {"temperature": self.temperature}
        if stop:
            options["stop"] = stop

        response = await self._client.chat(
            model=self.model_name,
            messages=ollama_messages,
            options=options,
            tools=self._bound_tools if self._bound_tools else None,
        )

        content = response.get("message", {}).get("content", "")
        tool_calls = []

        if response.get("message", {}).get("tool_calls"):
            for tc in response["message"]["tool_calls"]:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "args": tc.get("function", {}).get("arguments", {}),
                })

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,  # Always pass list, never None (Pydantic validation)
        )

        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to Ollama format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    # Handle multimodal - Ollama expects images separately
                    text_parts = []
                    images = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item["text"])
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:image"):
                                    base64_data = image_url.split(",")[1]
                                    images.append(base64_data)
                        else:
                            text_parts.append(str(item))
                    msg_dict = {"role": "user", "content": " ".join(text_parts)}
                    if images:
                        msg_dict["images"] = images
                    result.append(msg_dict)
                else:
                    result.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> ChatOllama:
        """Bind tools for function calling."""
        new_instance = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url,
        )
        new_instance._bound_tools = self._convert_tools(tools)
        return new_instance

    def _convert_tools(self, tools: List[Any]) -> List[dict]:
        """Convert tools to Ollama format."""
        ollama_tools = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                schema = {}
                if hasattr(tool, "args_schema"):
                    schema = tool.args_schema.model_json_schema()
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                })
            elif isinstance(tool, dict):
                ollama_tools.append(tool)
        return ollama_tools

    def with_structured_output(
        self,
        schema: Type[T],
        **kwargs: Any,
    ) -> ChatOllama:
        """Configure for structured output."""
        tool = {
            "type": "function",
            "function": {
                "name": schema.__name__,
                "description": f"Output structured data as {schema.__name__}",
                "parameters": schema.model_json_schema(),
            },
        }
        return self.bind_tools([tool])

