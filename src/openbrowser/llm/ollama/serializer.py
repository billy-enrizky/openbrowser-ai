"""Ollama message serializer.

Ollama uses OpenAI-compatible API format.
"""

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


class OllamaMessageSerializer(BaseMessageSerializer):
    """Serializer for converting messages to Ollama format.
    
    Ollama supports both its native format and OpenAI-compatible format.
    This serializer uses the native Ollama format for better compatibility.
    """

    def serialize(self, message: BaseMessage) -> dict[str, Any]:
        """Serialize a message to Ollama format."""
        if isinstance(message, UserMessage):
            result = {
                'role': 'user',
                'content': self._get_text_content(message.content),
            }
            # Add images if present
            images = self._extract_images(message.content)
            if images:
                result['images'] = images
            return result

        elif isinstance(message, SystemMessage):
            return {
                'role': 'system',
                'content': self._get_text_content(message.content),
            }

        elif isinstance(message, AssistantMessage):
            result = {'role': 'assistant'}
            
            if message.content is not None:
                result['content'] = self._get_text_content(message.content)
            
            if message.tool_calls:
                result['tool_calls'] = [
                    {
                        'function': {
                            'name': tc.function.name,
                            'arguments': json.loads(tc.function.arguments) if tc.function.arguments else {},
                        }
                    }
                    for tc in message.tool_calls
                ]
            
            return result

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    def _get_text_content(
        self,
        content: str | list[ContentPartTextParam | ContentPartImageParam] | None,
    ) -> str:
        """Extract text from content."""
        if content is None:
            return ''
        if isinstance(content, str):
            return content
        
        texts = []
        for part in content:
            if hasattr(part, 'type') and part.type == 'text':
                texts.append(part.text)
        return '\n'.join(texts)

    def _extract_images(
        self,
        content: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> list[str]:
        """Extract base64 images from content."""
        if isinstance(content, str):
            return []
        
        images = []
        for part in content:
            if hasattr(part, 'type') and part.type == 'image_url':
                url = part.image_url.url
                # Ollama expects raw base64 without data URL prefix
                if url.startswith('data:'):
                    if ',' in url:
                        images.append(url.split(',', 1)[1])
                else:
                    images.append(url)
        return images

    def serialize_tool_schema(self, output_format: type[T]) -> dict[str, Any]:
        """Serialize a Pydantic model to Ollama tool format."""
        schema = output_format.model_json_schema()
        
        # Remove $defs and resolve references
        if '$defs' in schema:
            defs = schema.pop('$defs')
            schema = self._resolve_refs(schema, defs)
        
        return {
            "type": "function",
            "function": {
                "name": output_format.__name__,
                "description": output_format.__doc__ or f"Generate {output_format.__name__}",
                "parameters": schema,
            }
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

    def get_format_schema(self, output_format: type[T]) -> dict[str, Any]:
        """Get the format parameter for Ollama structured output.
        
        Ollama supports a 'format' parameter for JSON output.
        """
        schema = output_format.model_json_schema()
        
        if '$defs' in schema:
            defs = schema.pop('$defs')
            schema = self._resolve_refs(schema, defs)
        
        return schema

    def parse_tool_call_response(
        self,
        response: dict[str, Any],
        output_format: type[T],
    ) -> T:
        """Parse an Ollama response into a Pydantic model."""
        message = response.get('message', {})
        tool_calls = message.get('tool_calls', [])

        if tool_calls:
            args = tool_calls[0].get('function', {}).get('arguments', {})
            return output_format.model_validate(args)

        # Try to parse content as JSON
        content = message.get('content', '')
        if content:
            try:
                data = json.loads(content)
                return output_format.model_validate(data)
            except json.JSONDecodeError:
                pass

        raise ValueError("No tool calls or valid JSON in response")

    def parse_content_response(self, response: dict[str, Any]) -> str:
        """Parse plain text content from Ollama response."""
        message = response.get('message', {})
        return message.get('content', '')


# Singleton instance
ollama_serializer = OllamaMessageSerializer()

