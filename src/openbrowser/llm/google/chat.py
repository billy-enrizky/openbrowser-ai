"""Google Gemini LLM integration compatible with LangChain."""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Literal, TypeVar, overload

from google import genai
from google.auth.credentials import Credentials
from google.genai import types
from google.genai.types import MediaModality
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


VerifiedGeminiModels = Literal[
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite-preview-02-05",
    "Gemini-2.0-exp",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",
    "gemma-3-27b-it",
    "gemma-3-4b",
    "gemma-3-12b",
    "gemma-3n-e2b",
    "gemma-3n-e4b",
]


class ModelProviderError(Exception):
    """Exception raised when a model provider returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        model: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.model = model


def _convert_langchain_to_google_messages(
    messages: list[BaseMessage],
) -> tuple[types.ContentListUnion, str | None]:
    """Convert LangChain messages to Google Gemini format."""
    formatted_messages: types.ContentListUnion = []
    system_instruction: str | None = None

    for message in messages:
        if isinstance(message, SystemMessage):
            if isinstance(message.content, str):
                system_instruction = message.content
            continue

        role = "user" if isinstance(message, HumanMessage) else "model"
        parts: list[types.Part] = []

        if isinstance(message, ToolMessage):
            role = "user"
            parts.append(types.Part.from_text(text=f"Tool result: {message.content}"))
        elif isinstance(message, AIMessage):
            if message.content:
                if isinstance(message.content, str):
                    parts.append(types.Part.from_text(text=message.content))
                elif isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append(types.Part.from_text(text=part["text"]))
                            elif part.get("type") == "image_url":
                                url = part["image_url"]["url"]
                                if url.startswith("data:"):
                                    header, data = url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    image_bytes = base64.b64decode(data)
                                    parts.append(
                                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                                    )
        else:
            if isinstance(message.content, str):
                parts.append(types.Part.from_text(text=message.content))
            elif isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(types.Part.from_text(text=part["text"]))
                        elif part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                header, data = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                image_bytes = base64.b64decode(data)
                                parts.append(
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                                )

        if parts:
            formatted_messages.append(types.Content(role=role, parts=parts))

    return formatted_messages, system_instruction


def _convert_google_to_langchain_message(
    response: types.GenerateContentResponse,
) -> AIMessage:
    """Convert Google Gemini response to LangChain AIMessage."""
    if not response.candidates:
        return AIMessage(content="")

    candidate = response.candidates[0]
    content_parts = []

    if candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                content_parts.append({"type": "text", "text": part.text})

    content = content_parts[0]["text"] if content_parts else ""

    tool_calls = []
    if candidate.content and candidate.content.parts:
        for idx, part in enumerate(candidate.content.parts):
            if hasattr(part, "function_call") and part.function_call:
                func_call = part.function_call
                args_dict = {}
                if hasattr(func_call, "args") and func_call.args:
                    if isinstance(func_call.args, dict):
                        args_dict = func_call.args
                    elif isinstance(func_call.args, str):
                        try:
                            args_dict = json.loads(func_call.args)
                        except json.JSONDecodeError:
                            args_dict = {}
                
                tool_calls.append(
                    {
                        "id": f"call_{idx}",
                        "name": func_call.name,
                        "args": args_dict,
                    }
                )

    return AIMessage(content=content, tool_calls=tool_calls if tool_calls else [])


class ChatGoogle(BaseChatModel):
    """
    LangChain-compatible wrapper for Google's Gemini chat model.

    This class wraps Google's genai client and provides a LangChain-compatible interface
    that works with bind_tools() and other LangChain features.

    Args:
        model: The Gemini model to use
        temperature: Temperature for response generation
        api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        vertexai: Whether to use Vertex AI (defaults to False)
        credentials: Google credentials object for Vertex AI
        project: Google Cloud project ID for Vertex AI
        location: Google Cloud location for Vertex AI
        max_output_tokens: Maximum output tokens
        max_retries: Number of retries for retryable errors
        retryable_status_codes: List of HTTP status codes to retry on
        retry_delay: Delay in seconds between retries
    """

    model: VerifiedGeminiModels | str = Field(default="gemini-flash-latest")
    temperature: float | None = Field(default=0.5)
    top_p: float | None = Field(default=None)
    seed: int | None = Field(default=None)
    thinking_budget: int | None = Field(default=None)
    max_output_tokens: int | None = Field(default=8096)

    api_key: str | None = Field(default=None)
    vertexai: bool | None = Field(default=None)
    credentials: Credentials | None = Field(default=None)
    project: str | None = Field(default=None)
    location: str | None = Field(default=None)
    http_options: types.HttpOptions | types.HttpOptionsDict | None = Field(default=None)

    max_retries: int = Field(default=3)
    retryable_status_codes: list[int] = Field(default_factory=lambda: [403, 503])
    retry_delay: float = Field(default=0.01)

    _client: genai.Client | None = PrivateAttr(default=None)
    _bound_tools: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "google-gemini"

    def bind_tools(self, tools: list[Any], tool_choice: Any = None, **kwargs: Any) -> "ChatGoogle":
        """Bind tools to this LLM instance for function calling.
        
        Args:
            tools: List of tools to bind
            tool_choice: Tool choice parameter (ignored, kept for compatibility with with_structured_output)
            **kwargs: Additional keyword arguments (ignored)
        """
        bound = ChatGoogle(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            thinking_budget=self.thinking_budget,
            max_output_tokens=self.max_output_tokens,
            api_key=self.api_key,
            vertexai=self.vertexai,
            credentials=self.credentials,
            project=self.project,
            location=self.location,
            http_options=self.http_options,
            max_retries=self.max_retries,
            retryable_status_codes=self.retryable_status_codes,
            retry_delay=self.retry_delay,
        )
        bound._client = self._client
        bound._bound_tools = tools
        return bound

    def with_structured_output(self, output_schema: type[T], **kwargs: Any) -> Any:
        """Get structured output from the model using Gemini's response schema.
        
        Args:
            output_schema: Pydantic model class for structured output
            **kwargs: Additional keyword arguments (ignored)
        
        Returns:
            A callable that can be invoked with messages to get structured output
        """
        class StructuredOutputWrapper:
            def __init__(self, llm: "ChatGoogle", schema: type[T]):
                self.llm = llm
                self.schema = schema
            
            async def ainvoke(self, input: Any, config: dict[str, Any] | None = None) -> T:
                """Invoke with structured output."""
                # Convert input to messages if it's a string
                if isinstance(input, str):
                    messages = [HumanMessage(content=input)]
                elif isinstance(input, list):
                    messages = input
                else:
                    messages = [HumanMessage(content=str(input))]
                
                # Get the schema JSON
                schema_json = self.schema.model_json_schema()
                
                # Add schema instruction to system message or first message
                system_instruction = (
                    f"You must respond with valid JSON matching this schema:\n"
                    f"{json.dumps(schema_json, indent=2)}\n\n"
                    f"Return only the JSON object, no other text."
                )
                
                # Convert messages and add system instruction
                contents, existing_system = _convert_langchain_to_google_messages(messages)
                
                # Prepare config with response schema
                config_dict: types.GenerateContentConfigDict = {}
                if self.llm.temperature is not None:
                    config_dict["temperature"] = self.llm.temperature
                if self.llm.max_output_tokens is not None:
                    config_dict["max_output_tokens"] = self.llm.max_output_tokens
                
                # Use Gemini's response schema feature
                config_dict["response_mime_type"] = "application/json"
                config_dict["response_schema"] = schema_json
                
                # Combine system instructions
                if existing_system:
                    config_dict["system_instruction"] = f"{existing_system}\n\n{system_instruction}"
                else:
                    config_dict["system_instruction"] = system_instruction
                
                # Make API call
                client = self.llm._get_client()
                response = await client.aio.models.generate_content(
                    model=self.llm.model,
                    contents=contents,
                    config=config_dict,
                )
                
                # Parse response
                if not response.candidates or not response.candidates[0].content:
                    raise ModelProviderError(
                        message="No response from model",
                        status_code=500,
                        model=self.llm.model,
                    )
                
                candidate = response.candidates[0]
                text_content = ""
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text_content += part.text
                
                if not text_content:
                    raise ModelProviderError(
                        message="Empty response from model",
                        status_code=500,
                        model=self.llm.model,
                    )
                
                # Parse JSON and validate against schema
                try:
                    parsed_json = json.loads(text_content)
                    return self.schema.model_validate(parsed_json)
                except json.JSONDecodeError as e:
                    raise ModelProviderError(
                        message=f"Failed to parse JSON response: {e}",
                        status_code=500,
                        model=self.llm.model,
                    ) from e
                except Exception as e:
                    raise ModelProviderError(
                        message=f"Failed to validate response against schema: {e}",
                        status_code=500,
                        model=self.llm.model,
                    ) from e
            
            def invoke(self, input: Any, config: dict[str, Any] | None = None) -> T:
                """Synchronous invoke (not supported)."""
                raise NotImplementedError("ChatGoogle structured output only supports async. Use ainvoke().")
        
        return StructuredOutputWrapper(self, output_schema)

    def _get_client(self) -> genai.Client:
        """Get or create Google genai client."""
        if self._client is not None:
            return self._client

        client_params: dict[str, Any] = {}

        if self.api_key:
            client_params["api_key"] = self.api_key
        elif not self.vertexai:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                client_params["api_key"] = api_key

        if self.vertexai is not None:
            client_params["vertexai"] = self.vertexai

        if self.credentials:
            client_params["credentials"] = self.credentials

        if self.project:
            client_params["project"] = self.project

        if self.location:
            client_params["location"] = self.location

        if self.http_options:
            client_params["http_options"] = self.http_options

        self._client = genai.Client(**client_params)
        return self._client

    def _get_usage(self, response: types.GenerateContentResponse) -> dict[str, Any] | None:
        """Extract usage information from Google response."""
        if not response.usage_metadata:
            return None

        image_tokens = 0
        if response.usage_metadata.prompt_tokens_details:
            image_tokens = sum(
                detail.token_count or 0
                for detail in response.usage_metadata.prompt_tokens_details
                if detail.modality == MediaModality.IMAGE
            )

        return {
            "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
            "completion_tokens": (response.usage_metadata.candidates_token_count or 0)
            + (response.usage_metadata.thoughts_token_count or 0),
            "total_tokens": response.usage_metadata.total_token_count or 0,
            "prompt_image_tokens": image_tokens,
        }

    def _convert_tools_to_google_format(self, tools: Any) -> list[types.FunctionDeclaration] | None:
        """Convert LangChain tools to Google Gemini function declarations."""
        if not tools:
            return None

        google_tools = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "args_schema"):
                schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
                google_tools.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=schema,
                    )
                )
            elif hasattr(tool, "name") and hasattr(tool, "description"):
                google_tools.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description or "",
                        parameters={"type": "object", "properties": {}},
                    )
                )

        return google_tools if google_tools else None

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from Google Gemini."""
        start_time = time.time()
        logger.debug(f"Starting API call to {self.model}")

        contents, system_instruction = _convert_langchain_to_google_messages(messages)

        config: types.GenerateContentConfigDict = {}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.top_p is not None:
            config["top_p"] = self.top_p
        if self.seed is not None:
            config["seed"] = self.seed
        if self.max_output_tokens is not None:
            config["max_output_tokens"] = self.max_output_tokens

        if system_instruction:
            config["system_instruction"] = system_instruction

        if self.thinking_budget is None and (
            "gemini-2.5-flash" in self.model or "gemini-flash" in self.model
        ):
            self.thinking_budget = 0

        if self.thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": self.thinking_budget}

        if stop:
            config["stop_sequences"] = stop

        tools = kwargs.get("tools") or getattr(self, "_bound_tools", None)
        if tools:
            google_tools = self._convert_tools_to_google_format(tools)
            if google_tools:
                config["tools"] = [types.Tool(function_declarations=google_tools)]

        async def _make_api_call():
            try:
                client = self._get_client()
                response = await client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                elapsed = time.time() - start_time
                logger.debug(f"Got response in {elapsed:.2f}s")

                message = _convert_google_to_langchain_message(response)
                usage = self._get_usage(response)

                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation], llm_output={"usage": usage})

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"API call failed after {elapsed:.2f}s: {type(e).__name__}: {e}")

                error_message = str(e)
                status_code: int | None = None

                if hasattr(e, "response"):
                    response_obj = getattr(e, "response", None)
                    if response_obj and hasattr(response_obj, "status_code"):
                        status_code = getattr(response_obj, "status_code", None)

                if "timeout" in error_message.lower() or "cancelled" in error_message.lower():
                    status_code = 504 if "CancelledError" in str(type(e)) else 408
                elif any(indicator in error_message.lower() for indicator in ["forbidden", "403"]):
                    status_code = 403
                elif any(
                    indicator in error_message.lower()
                    for indicator in ["rate limit", "resource exhausted", "quota exceeded", "429"]
                ):
                    status_code = 429
                elif any(
                    indicator in error_message.lower()
                    for indicator in ["service unavailable", "internal server error", "503", "502", "500"]
                ):
                    status_code = 503

                raise ModelProviderError(
                    message=error_message,
                    status_code=status_code or 502,
                    model=self.model,
                ) from e

        for attempt in range(self.max_retries):
            try:
                return await _make_api_call()
            except ModelProviderError as e:
                if e.status_code in self.retryable_status_codes and attempt < self.max_retries - 1:
                    logger.warning(
                        f"Got {e.status_code} error, retrying... (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise
            except Exception as e:
                raise ModelProviderError(
                    message=str(e),
                    status_code=502,
                    model=self.model,
                ) from e

        raise RuntimeError("Retry loop completed without return or exception")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generate (not supported, raises error)."""
        raise NotImplementedError("ChatGoogle only supports async generation. Use agenerate() or ainvoke().")

