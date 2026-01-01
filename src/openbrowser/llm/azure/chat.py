"""Azure OpenAI chat model wrapper."""

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
    
    def __init__(self, llm: ChatAzureOpenAI, schema: Type[BaseModel]):
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
                    # Parse JSON if args is a string
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
        
        raise ValueError(f"Failed to get structured output from Azure OpenAI. Response: {result}")


class ChatAzureOpenAI(BaseChatModel):
    """
    LangChain-compatible wrapper for Azure OpenAI.
    Uses OpenAI SDK with Azure endpoint.
    """

    model_name: str = Field(default="gpt-4o", alias="model")
    deployment_name: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)
    api_key: Optional[str] = Field(default=None)
    azure_endpoint: Optional[str] = Field(default=None)
    api_version: str = Field(default="2024-02-01")
    timeout: float = Field(default=60.0)

    _client: Any = PrivateAttr(default=None)
    _bound_tools: List[Any] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._bound_tools = []
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Azure OpenAI client."""
        try:
            from openai import AsyncAzureOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        api_key = self.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = self.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not set")
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not set")

        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=self.api_version,
            timeout=self.timeout,
        )

    @property
    def _llm_type(self) -> str:
        return "azure_openai"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "deployment_name": self.deployment_name}

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
        """Async generation with Azure OpenAI API."""
        openai_messages = self._convert_messages(messages)

        request_params = {
            "model": self.deployment_name,
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
            tool_calls=tool_calls,  # Always pass list (empty or with items) - Pydantic requires list type
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

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> ChatAzureOpenAI:
        """Bind tools for function calling."""
        new_instance = ChatAzureOpenAI(
            model=self.model_name,
            deployment_name=self.deployment_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
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
        
        Returns a wrapper that invokes the model and parses the response
        into the specified Pydantic schema.
        """
        # Create a new instance with the tool bound
        new_instance = ChatAzureOpenAI(
            model=self.model_name,
            deployment_name=self.deployment_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        
        tool = {
            "type": "function",
            "function": {
                "name": schema.__name__,
                "description": f"Output structured data as {schema.__name__}. You MUST use this tool to respond.",
                "parameters": schema.model_json_schema(),
            },
        }
        new_instance._bound_tools = [tool]
        
        return StructuredOutputWrapper(new_instance, schema)

