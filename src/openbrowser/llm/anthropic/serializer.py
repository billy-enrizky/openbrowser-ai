"""Anthropic message serializer."""

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from openbrowser.agent.views import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    SystemMessage,
    UserMessage,
)
from openbrowser.llm.serializer import BaseMessageSerializer

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class AnthropicMessageSerializer(BaseMessageSerializer):
    """Serializer for converting messages to Anthropic format.
    
    Anthropic has a different message format where system messages are
    passed separately from the messages array.
    """

    def serialize(self, message: BaseMessage) -> dict[str, Any]:
        """Serialize a message to Anthropic format."""
        if isinstance(message, UserMessage):
            return {
                'role': 'user',
                'content': self._serialize_user_content(message.content),
            }

        elif isinstance(message, SystemMessage):
            # System messages are handled separately in Anthropic
            # Return as user message with system prefix for compatibility
            return {
                'role': 'user',
                'content': f"[SYSTEM]: {self._get_text_content(message.content)}",
            }

        elif isinstance(message, AssistantMessage):
            content = []
            if message.content is not None:
                content = self._serialize_assistant_content(message.content)
            
            if message.tool_calls:
                for tc in message.tool_calls:
                    content.append({
                        'type': 'tool_use',
                        'id': tc.id,
                        'name': tc.function.name,
                        'input': json.loads(tc.function.arguments) if tc.function.arguments else {},
                    })
            
            return {'role': 'assistant', 'content': content if content else ''}

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    def _serialize_user_content(
        self,
        content: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> list[dict[str, Any]]:
        """Serialize user content to Anthropic format."""
        if isinstance(content, str):
            return [{'type': 'text', 'text': content}]

        parts = []
        for part in content:
            if part.type == 'text':
                parts.append({'type': 'text', 'text': part.text})
            elif part.type == 'image_url':
                # Anthropic uses different image format
                parts.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64' if part.image_url.url.startswith('data:') else 'url',
                        'media_type': self._get_media_type(part.image_url.url),
                        'data': self._extract_base64(part.image_url.url) if part.image_url.url.startswith('data:') else part.image_url.url,
                    }
                })
        return parts

    def _serialize_assistant_content(
        self,
        content: str | list,
    ) -> list[dict[str, Any]]:
        """Serialize assistant content to Anthropic format."""
        if isinstance(content, str):
            return [{'type': 'text', 'text': content}]

        parts = []
        for part in content:
            if hasattr(part, 'type') and part.type == 'text':
                parts.append({'type': 'text', 'text': part.text})
        return parts

    def _get_text_content(self, content: str | list) -> str:
        """Extract text from content."""
        if isinstance(content, str):
            return content
        
        texts = []
        for part in content:
            if hasattr(part, 'type') and part.type == 'text':
                texts.append(part.text)
        return '\n'.join(texts)

    def _get_media_type(self, url: str) -> str:
        """Get media type from data URL or file extension."""
        if url.startswith('data:'):
            # Extract media type from data URL
            parts = url.split(';')[0].split(':')
            if len(parts) > 1:
                return parts[1]
        elif url.lower().endswith('.png'):
            return 'image/png'
        elif url.lower().endswith(('.jpg', '.jpeg')):
            return 'image/jpeg'
        elif url.lower().endswith('.gif'):
            return 'image/gif'
        elif url.lower().endswith('.webp'):
            return 'image/webp'
        return 'image/jpeg'  # Default

    def _extract_base64(self, data_url: str) -> str:
        """Extract base64 data from data URL."""
        if ',' in data_url:
            return data_url.split(',', 1)[1]
        return data_url

    def extract_system_message(
        self,
        messages: list[BaseMessage],
    ) -> tuple[str | None, list[BaseMessage]]:
        """Extract system message from message list.
        
        Anthropic requires system message to be passed separately.
        
        Returns:
            Tuple of (system_content, remaining_messages)
        """
        system_content = None
        remaining = []

        for msg in messages:
            if isinstance(msg, SystemMessage) and system_content is None:
                system_content = self._get_text_content(msg.content)
            else:
                remaining.append(msg)

        return system_content, remaining

    def serialize_tool_schema(self, output_format: type[T]) -> dict[str, Any]:
        """Serialize a Pydantic model to Anthropic tool format."""
        schema = output_format.model_json_schema()
        
        # Remove $defs and resolve references
        if '$defs' in schema:
            defs = schema.pop('$defs')
            schema = self._resolve_refs(schema, defs)
        
        return {
            "name": output_format.__name__,
            "description": output_format.__doc__ or f"Generate {output_format.__name__}",
            "input_schema": schema,
        }

    def _resolve_refs(self, obj: Any, defs: dict) -> Any:
        """Recursively resolve $ref references in schema."""
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref_path = obj['$ref'].split('/')[-1]
                if ref_path in defs:
                    return self._resolve_refs(defs[ref_path], defs)
            return {k: self._resolve_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_refs(item, defs) for item in obj]
        return obj

    def parse_tool_call_response(
        self,
        response: dict[str, Any],
        output_format: type[T],
    ) -> T:
        """Parse an Anthropic tool use response into a Pydantic model."""
        content = response.get('content', [])
        
        for block in content:
            if block.get('type') == 'tool_use':
                data = block.get('input', {})
                return output_format.model_validate(data)

        raise ValueError("No tool use block in response")

    def parse_content_response(self, response: dict[str, Any]) -> str:
        """Parse plain text content from Anthropic response."""
        content = response.get('content', [])
        texts = []
        
        for block in content:
            if block.get('type') == 'text':
                texts.append(block.get('text', ''))
        
        return '\n'.join(texts)


# Singleton instance
anthropic_serializer = AnthropicMessageSerializer()

