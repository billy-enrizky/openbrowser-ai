"""Base serializer for LLM message handling."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from openbrowser.agent.views import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    SystemMessage,
    ToolCall,
    UserMessage,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class BaseMessageSerializer(ABC):
    """Base class for LLM message serializers.
    
    Each LLM provider has different message formats. This base class defines
    the interface for serializing messages to provider-specific formats.
    """

    @abstractmethod
    def serialize(self, message: BaseMessage) -> dict[str, Any]:
        """Serialize a single message to provider-specific format.
        
        Args:
            message: Message to serialize
            
        Returns:
            Provider-specific message dict
        """
        pass

    def serialize_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Serialize a list of messages.
        
        Args:
            messages: Messages to serialize
            
        Returns:
            List of provider-specific message dicts
        """
        return [self.serialize(m) for m in messages]

    def serialize_tool_schema(self, output_format: type[T]) -> dict[str, Any]:
        """Serialize a Pydantic model to a tool/function schema.
        
        Args:
            output_format: Pydantic model class
            
        Returns:
            Tool schema dict
        """
        schema = output_format.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": output_format.__name__,
                "description": output_format.__doc__ or f"Generate {output_format.__name__}",
                "parameters": schema,
            }
        }

    @staticmethod
    def serialize_content_part_text(part: ContentPartTextParam) -> dict[str, Any]:
        """Serialize a text content part."""
        return {"type": "text", "text": part.text}

    @staticmethod
    def serialize_content_part_image(part: ContentPartImageParam) -> dict[str, Any]:
        """Serialize an image content part."""
        return {
            "type": "image_url",
            "image_url": {
                "url": part.image_url.url,
                "detail": part.image_url.detail,
            }
        }

    @staticmethod
    def serialize_content_part_refusal(part: ContentPartRefusalParam) -> dict[str, Any]:
        """Serialize a refusal content part."""
        return {"type": "refusal", "refusal": part.refusal}

    def serialize_user_content(
        self,
        content: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> str | list[dict[str, Any]]:
        """Serialize content for user messages."""
        if isinstance(content, str):
            return content

        parts = []
        for part in content:
            if part.type == 'text':
                parts.append(self.serialize_content_part_text(part))
            elif part.type == 'image_url':
                parts.append(self.serialize_content_part_image(part))
        return parts

    def serialize_system_content(
        self,
        content: str | list[ContentPartTextParam],
    ) -> str | list[dict[str, Any]]:
        """Serialize content for system messages."""
        if isinstance(content, str):
            return content

        parts = []
        for part in content:
            if part.type == 'text':
                parts.append(self.serialize_content_part_text(part))
        return parts

    def serialize_assistant_content(
        self,
        content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None,
    ) -> str | list[dict[str, Any]] | None:
        """Serialize content for assistant messages."""
        if content is None:
            return None
        if isinstance(content, str):
            return content

        parts = []
        for part in content:
            if part.type == 'text':
                parts.append(self.serialize_content_part_text(part))
            elif part.type == 'refusal':
                parts.append(self.serialize_content_part_refusal(part))
        return parts

    def serialize_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Serialize a tool call."""
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
        }

    def parse_tool_call_response(
        self,
        response: dict[str, Any],
        output_format: type[T],
    ) -> T:
        """Parse a tool call response into a Pydantic model.
        
        Args:
            response: Raw response from the LLM
            output_format: Expected Pydantic model class
            
        Returns:
            Parsed Pydantic model instance
        """
        # This is a base implementation that can be overridden
        # by provider-specific serializers
        raise NotImplementedError("Subclasses must implement parse_tool_call_response")

