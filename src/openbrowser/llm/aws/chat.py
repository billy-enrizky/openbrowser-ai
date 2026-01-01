"""AWS Bedrock chat model wrapper for Claude via AWS."""

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
    
    def __init__(self, llm: ChatAWSBedrock, schema: Type[BaseModel]):
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
        
        raise ValueError(f"Failed to get structured output from Bedrock. Response: {result}")


class ChatAWSBedrock(BaseChatModel):
    """
    LangChain-compatible wrapper for AWS Bedrock (Claude models).
    Requires: pip install boto3
    """

    model_name: str = Field(default="anthropic.claude-3-5-sonnet-20241022-v2:0", alias="model")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)
    region_name: str = Field(default="us-east-1")
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    aws_session_token: Optional[str] = Field(default=None)

    _client: Any = PrivateAttr(default=None)
    _bound_tools: List[Any] = PrivateAttr(default_factory=list)
    _output_schema: Optional[Type[BaseModel]] = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._bound_tools = []
        self._output_schema = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize AWS Bedrock client."""
        try:
            import boto3
        except ImportError:
            raise ImportError("Please install boto3: pip install boto3")

        session_kwargs = {"region_name": self.region_name}

        if self.aws_access_key_id:
            session_kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            session_kwargs["aws_session_token"] = self.aws_session_token

        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime")

    @property
    def _llm_type(self) -> str:
        return "aws_bedrock"

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
        """Async generation with AWS Bedrock API."""
        import asyncio

        # Convert messages to Anthropic format
        anthropic_messages, system = self._convert_messages(messages)

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": anthropic_messages,
        }

        if system:
            request_body["system"] = system

        if self._bound_tools:
            request_body["tools"] = self._bound_tools
            # Force tool use when structured output is expected
            if self._output_schema is not None:
                request_body["tool_choice"] = {"type": "any"}

        if stop:
            request_body["stop_sequences"] = stop

        # Run sync boto3 call in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(request_body),
            ),
        )

        response_body = json.loads(response["body"].read())

        content = ""
        tool_calls = []

        for block in response_body.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "args": block.get("input", {}),
                })

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,  # Always pass list (empty or with items)
        )

        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages(self, messages: List[BaseMessage]) -> tuple[List[dict], str]:
        """Convert LangChain messages to Anthropic format for Bedrock."""
        result = []
        system = ""

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system = msg.content
            elif isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    anthropic_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                anthropic_content.append({"type": "text", "text": item["text"]})
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:image"):
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

        return result, system

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> ChatAWSBedrock:
        """Bind tools for function calling."""
        new_instance = ChatAWSBedrock(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )
        new_instance._bound_tools = self._convert_tools(tools)
        new_instance._output_schema = self._output_schema
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
    ) -> StructuredOutputWrapper:
        """Configure for structured output.
        
        Returns a wrapper that invokes the model and parses the response
        into the specified Pydantic schema.
        """
        # Create a new instance with the tool bound
        new_instance = ChatAWSBedrock(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )
        
        tool = {
            "name": schema.__name__,
            "description": f"Output structured data as {schema.__name__}. You MUST use this tool to respond.",
            "input_schema": schema.model_json_schema(),
        }
        new_instance._bound_tools = [tool]
        new_instance._output_schema = schema
        
        return StructuredOutputWrapper(new_instance, schema)

