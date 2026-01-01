"""Groq chat model wrapper for fast inference."""

from __future__ import annotations

import json
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


class StructuredOutputWrapper:
    """Wrapper that parses tool call responses into Pydantic models."""
    
    def __init__(self, llm: "ChatGroq", schema: Type[BaseModel]):
        self.llm = llm
        self.schema = schema
    
    async def ainvoke(self, messages: List[BaseMessage], **kwargs: Any) -> BaseModel:
        """Invoke the model and parse the response into the schema."""
        result = await self.llm.ainvoke(messages, **kwargs)
        
        # Check if we got tool calls
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for tool_call in result.tool_calls:
                try:
                    # Get the args from the tool call
                    args = tool_call.get('args', {}) if isinstance(tool_call, dict) else tool_call.args
                    # Handle string args (JSON) by parsing them
                    if isinstance(args, str):
                        args = json.loads(args)
                    return self.schema.model_validate(args)
                except Exception as e:
                    logger.warning("Failed to parse tool call: %s", e)
                    continue
        
        # If no valid tool calls, try to parse content as JSON
        if hasattr(result, 'content') and result.content:
            try:
                data = json.loads(result.content)
                return self.schema.model_validate(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Failed to parse content as JSON: %s", e)
        
        raise ValueError(f"Failed to get structured output from Groq. Response: {result}")


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
                # Parse args from JSON string to dict (Groq returns args as string)
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                    logger.warning("Failed to parse tool call arguments: %s", tc.function.arguments)
                
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,  # Always pass list, never None (Pydantic validation)
        )

        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to OpenAI/Groq format.
        
        Groq supports both string content and multi-part content (for vision models).
        For non-vision models, we extract text from multi-part content.
        """
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System message content should always be a string
                content = msg.content
                if isinstance(content, list):
                    # Extract text from multi-part content
                    content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                        if isinstance(part, (str, dict))
                    )
                result.append({"role": "system", "content": content})
            elif isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    # Check if model supports vision
                    # Vision models: llama-4-scout, llama-4-maverick, or legacy llama-3.2-*-vision
                    model_lower = self.model_name.lower()
                    is_vision_model = (
                        "vision" in model_lower or
                        "llama-4-scout" in model_lower or
                        "llama-4-maverick" in model_lower
                    )
                    if is_vision_model:
                        # Convert to OpenAI-style multi-part content
                        parts = []
                        for part in content:
                            if isinstance(part, str):
                                parts.append({"type": "text", "text": part})
                            elif isinstance(part, dict):
                                if part.get("type") == "text":
                                    parts.append({"type": "text", "text": part.get("text", "")})
                                elif part.get("type") == "image_url":
                                    parts.append({
                                        "type": "image_url",
                                        "image_url": part.get("image_url", {})
                                    })
                        content = parts
                    else:
                        # For non-vision models, extract text only
                        text_parts = []
                        for part in content:
                            if isinstance(part, str):
                                text_parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        content = " ".join(text_parts)
                result.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                # AI message content should be a string
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                        if isinstance(part, (str, dict))
                    )
                result.append({"role": "assistant", "content": content})
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
    ) -> StructuredOutputWrapper:
        """Configure for structured output.
        
        Returns a wrapper that invokes the model and parses responses into the schema.
        """
        tool = {
            "type": "function",
            "function": {
                "name": schema.__name__,
                "description": f"Output structured data as {schema.__name__}",
                "parameters": schema.model_json_schema(),
            },
        }
        bound_llm = self.bind_tools([tool])
        return StructuredOutputWrapper(bound_llm, schema)

