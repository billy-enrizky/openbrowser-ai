"""Cerebras LLM integration for fast inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, List, Optional, Type, TypeVar

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field, PrivateAttr

from openbrowser.llm.exceptions import ModelProviderError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0  # Base delay in seconds (exponential backoff)


class StructuredOutputWrapper:
    """Wrapper that parses tool call responses into Pydantic models."""
    
    def __init__(self, llm: "ChatCerebras", schema: Type[BaseModel]):
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
        
        raise ValueError(f"Failed to get structured output from Cerebras. Response: {result}")


class ChatCerebras(BaseChatModel):
    """
    Chat model for Cerebras fast inference.

    Cerebras provides extremely fast inference for LLMs.
    Uses OpenAI-compatible API format.

    Note: Cerebras does NOT support vision/image content. When BrowserAgent sends
    screenshots, this provider automatically extracts text-only content from
    multi-part messages.

    Args:
        model: The Cerebras model to use (e.g., "llama-3.3-70b", "llama3.1-8b")
        temperature: Temperature for response generation
        api_key: Cerebras API key (defaults to CEREBRAS_API_KEY env var)
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters
    """

    model_name: str = Field(default="llama-3.3-70b", alias="model")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)
    api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://api.cerebras.ai/v1")
    timeout: float = Field(default=120.0)

    _client: httpx.AsyncClient = PrivateAttr(default=None)
    _bound_tools: List[Any] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs: Any):
        # Handle 'model' alias - extract before Pydantic processing
        model_value = kwargs.get('model') or kwargs.get('model_name')
        super().__init__(**kwargs)
        # Set model_name after init to ensure it's properly set
        if model_value:
            object.__setattr__(self, 'model_name', model_value)
        self._bound_tools = []
        self._init_client()

    def _init_client(self) -> None:
        """Initialize HTTP client."""
        resolved_api_key = self.api_key or os.getenv("CEREBRAS_API_KEY", "")
        if not resolved_api_key:
            raise ValueError("CEREBRAS_API_KEY is not set and no api_key provided")
        self._client = httpx.AsyncClient(timeout=self.timeout)

    @property
    def model(self) -> str:
        """Get the model name."""
        return self.model_name

    @property
    def _llm_type(self) -> str:
        return "cerebras"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to OpenAI/Cerebras format.
        
        Cerebras does NOT support vision/image content. For multi-part messages
        (e.g., from BrowserAgent with screenshots), we extract text only.
        """
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = msg.content
                if isinstance(content, list):
                    # Extract text from multi-part content
                    content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                        if isinstance(part, (str, dict)) and (isinstance(part, str) or part.get("type") == "text")
                    )
                result.append({"role": "system", "content": content})
            elif isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    # Cerebras doesn't support vision - extract text only
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        # Skip image_url parts silently
                    content = " ".join(text_parts)
                result.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                        if isinstance(part, (str, dict))
                    )
                result.append({"role": "assistant", "content": content})
        return result

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

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "ChatCerebras":
        """Bind tools for function calling."""
        new_instance = ChatCerebras(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        new_instance._bound_tools = self._convert_tools(tools)
        return new_instance

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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generate - not implemented, use async."""
        raise NotImplementedError("Use ainvoke for async generation")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation with Cerebras API with retry logic."""
        resolved_api_key = self.api_key or os.getenv("CEREBRAS_API_KEY", "")
        openai_messages = self._convert_messages(messages)

        request_body = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self._bound_tools:
            request_body["tools"] = self._bound_tools

        if stop:
            request_body["stop"] = stop

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=request_body,
                    headers={
                        "Authorization": f"Bearer {resolved_api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout,
                )
                
                # Check for retryable errors (503, 429, 502, 504)
                if response.status_code in (502, 503, 504, 429):
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(
                        "Cerebras API returned %d, retrying in %.1fs (attempt %d/%d)",
                        response.status_code, delay, attempt + 1, MAX_RETRIES
                    )
                    last_error = ModelProviderError(
                        message=f"Cerebras API error: {response.text}",
                        status_code=response.status_code,
                        model=self.model_name,
                    )
                    await asyncio.sleep(delay)
                    continue
                
                response.raise_for_status()
                data = response.json()

                choice = data.get("choices", [{}])[0]
                message_data = choice.get("message", {})
                content = message_data.get("content", "") or ""
                tool_calls = []

                # Parse tool calls if present
                if message_data.get("tool_calls"):
                    for tc in message_data["tool_calls"]:
                        try:
                            args = json.loads(tc["function"]["arguments"]) if tc["function"].get("arguments") else {}
                        except json.JSONDecodeError:
                            args = {}
                            logger.warning("Failed to parse tool call arguments: %s", tc["function"].get("arguments"))
                        
                        tool_calls.append({
                            "id": tc.get("id", ""),
                            "name": tc["function"]["name"],
                            "args": args,
                        })

                message = AIMessage(
                    content=content,
                    tool_calls=tool_calls,  # Always pass list, never None (Pydantic validation)
                )

                return ChatResult(generations=[ChatGeneration(message=message)])

            except httpx.TimeoutException as e:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    "Cerebras API timeout, retrying in %.1fs (attempt %d/%d): %s",
                    delay, attempt + 1, MAX_RETRIES, str(e)
                )
                last_error = ModelProviderError(
                    message=f"Cerebras API timeout: {str(e)}",
                    status_code=408,
                    model=self.model_name,
                )
                await asyncio.sleep(delay)
                continue
                
            except httpx.HTTPStatusError as e:
                raise ModelProviderError(
                    message=f"Cerebras API error: {e.response.text}",
                    status_code=e.response.status_code,
                    model=self.model_name,
                )
        
        # All retries exhausted
        if last_error:
            raise last_error
        raise ModelProviderError(
            message="Cerebras API failed after all retries",
            status_code=500,
            model=self.model_name,
        )

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the model with the given messages."""
        result = await self._agenerate(messages, **kwargs)
        return result.generations[0].message

    def invoke(
        self,
        messages: List[BaseMessage],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Synchronously invoke the model."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(messages, config, **kwargs)
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()

