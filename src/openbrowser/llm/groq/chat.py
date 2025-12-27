"""Groq chat model wrapper for fast inference."""

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


class ChatGroq(BaseChatModel):
    """
    LangChain-compatible wrapper for Groq API.
    Requires: pip install groq
    """

    model_name: str = Field(default="llama-3.3-70b-versatile", alias="model")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)
    api_key: Optional[str] = Field(default=None)
    timeout: float = Field(default=60.0)

    _client: Any = PrivateAttr(default=None)
    _bound_tools: List[Any] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._bound_tools = []
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Groq client."""
        try:
            from groq import AsyncGroq
        except ImportError:
            raise ImportError("Please install groq: pip install groq")

        api_key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        self._client = AsyncGroq(api_key=api_key, timeout=self.timeout)

    @property
    def _llm_type(self) -> str:
        return "groq"

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
        """Async generation with Groq API."""
        openai_messages = self._convert_messages(messages)

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
            tool_calls=tool_calls if tool_calls else None,
        )

        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> ChatGroq:
        """Bind tools for function calling."""
        new_instance = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
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
    ) -> ChatGroq:
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

