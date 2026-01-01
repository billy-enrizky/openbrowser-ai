"""OpenRouter chat model wrapper for multi-provider gateway."""

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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ChatOpenRouter(BaseChatModel):
    """
    LangChain-compatible wrapper for OpenRouter API.
    OpenRouter provides access to multiple LLM providers through a single API.
    Uses OpenAI-compatible format.
    """

    model_name: str = Field(default="anthropic/claude-3.5-sonnet", alias="model")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)
    api_key: Optional[str] = Field(default=None)
    timeout: float = Field(default=120.0)
    site_url: Optional[str] = Field(default=None)
    site_name: Optional[str] = Field(default=None)

    _client: Any = PrivateAttr(default=None)
    _bound_tools: List[Any] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._bound_tools = []
        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenRouter client using OpenAI SDK."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=self.timeout,
        )

    @property
    def _llm_type(self) -> str:
        return "openrouter"

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
        """Async generation with OpenRouter API."""
        openai_messages = self._convert_messages(messages)

        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        request_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": openai_messages,
        }

        if self._bound_tools:
            request_params["tools"] = self._bound_tools

        if stop:
            request_params["stop"] = stop

        if extra_headers:
            request_params["extra_headers"] = extra_headers

        response = await self._client.chat.completions.create(**request_params)

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": tc.function.arguments,
                })

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,  # Always pass list, never None (Pydantic validation)
        )

        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    openai_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                openai_content.append({"type": "text", "text": item["text"]})
                            elif item.get("type") == "image_url":
                                openai_content.append(item)
                        else:
                            openai_content.append({"type": "text", "text": str(item)})
                    result.append({"role": "user", "content": openai_content})
                else:
                    result.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> ChatOpenRouter:
        """Bind tools for function calling."""
        new_instance = ChatOpenRouter(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            site_url=self.site_url,
            site_name=self.site_name,
        )
        new_instance._bound_tools = self._convert_tools(tools)
        return new_instance

    def _convert_tools(self, tools: List[Any]) -> List[dict]:
        """Convert tools to OpenAI format."""
        openai_tools = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                schema = {}
                if hasattr(tool, "args_schema"):
                    schema = tool.args_schema.model_json_schema()
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                })
            elif isinstance(tool, dict):
                openai_tools.append(tool)
        return openai_tools

    def with_structured_output(
        self,
        schema: Type[T],
        **kwargs: Any,
    ) -> ChatOpenRouter:
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

