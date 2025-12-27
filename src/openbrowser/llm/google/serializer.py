"""Google (Gemini) message serializer."""

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from src.openbrowser.agent.views import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    SystemMessage,
    UserMessage,
)
from src.openbrowser.llm.serializer import BaseMessageSerializer

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class GoogleMessageSerializer(BaseMessageSerializer):
    """Serializer for converting messages to Google Gemini format."""

    def serialize(self, message: BaseMessage) -> dict[str, Any]:
        """Serialize a message to Google Gemini format."""
        if isinstance(message, UserMessage):
            return {
                'role': 'user',
                'parts': self._serialize_user_parts(message.content),
            }

        elif isinstance(message, SystemMessage):
            # Google handles system messages as system_instruction
            # Return as user message for compatibility in messages array
            return {
                'role': 'user',
                'parts': [{'text': self._get_text_content(message.content)}],
            }

        elif isinstance(message, AssistantMessage):
            parts = []
            if message.content is not None:
                parts = self._serialize_assistant_parts(message.content)
            
            if message.tool_calls:
                for tc in message.tool_calls:
                    parts.append({
                        'functionCall': {
                            'name': tc.function.name,
                            'args': json.loads(tc.function.arguments) if tc.function.arguments else {},
                        }
                    })
            
            return {'role': 'model', 'parts': parts}

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    def _serialize_user_parts(
        self,
        content: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> list[dict[str, Any]]:
        """Serialize user content to Google parts format."""
        if isinstance(content, str):
            return [{'text': content}]

        parts = []
        for part in content:
            if part.type == 'text':
                parts.append({'text': part.text})
            elif part.type == 'image_url':
                parts.append({
                    'inlineData': {
                        'mimeType': self._get_mime_type(part.image_url.url),
                        'data': self._extract_base64(part.image_url.url),
                    }
                })
        return parts

    def _serialize_assistant_parts(
        self,
        content: str | list,
    ) -> list[dict[str, Any]]:
        """Serialize assistant content to Google parts format."""
        if isinstance(content, str):
            return [{'text': content}]

        parts = []
        for part in content:
            if hasattr(part, 'type') and part.type == 'text':
                parts.append({'text': part.text})
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

    def _get_mime_type(self, url: str) -> str:
        """Get MIME type from URL."""
        if url.startswith('data:'):
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
        return 'image/jpeg'

    def _extract_base64(self, data_url: str) -> str:
        """Extract base64 data from data URL."""
        if ',' in data_url:
            return data_url.split(',', 1)[1]
        return data_url

    def extract_system_instruction(
        self,
        messages: list[BaseMessage],
    ) -> tuple[str | None, list[BaseMessage]]:
        """Extract system instruction from message list.
        
        Google uses system_instruction parameter.
        
        Returns:
            Tuple of (system_instruction, remaining_messages)
        """
        system_instruction = None
        remaining = []

        for msg in messages:
            if isinstance(msg, SystemMessage) and system_instruction is None:
                system_instruction = self._get_text_content(msg.content)
            else:
                remaining.append(msg)

        return system_instruction, remaining

    def serialize_tool_schema(self, output_format: type[T]) -> dict[str, Any]:
        """Serialize a Pydantic model to Google function declaration format."""
        schema = output_format.model_json_schema()
        
        # Remove $defs and resolve references
        if '$defs' in schema:
            defs = schema.pop('$defs')
            schema = self._resolve_refs(schema, defs)
        
        # Google uses different schema format
        return {
            "name": output_format.__name__,
            "description": output_format.__doc__ or f"Generate {output_format.__name__}",
            "parameters": self._convert_to_google_schema(schema),
        }

    def _convert_to_google_schema(self, schema: dict) -> dict:
        """Convert JSON Schema to Google's schema format."""
        result = {}
        
        if 'type' in schema:
            result['type'] = schema['type'].upper()
        
        if 'properties' in schema:
            result['properties'] = {
                k: self._convert_to_google_schema(v)
                for k, v in schema['properties'].items()
            }
        
        if 'required' in schema:
            result['required'] = schema['required']
        
        if 'items' in schema:
            result['items'] = self._convert_to_google_schema(schema['items'])
        
        if 'description' in schema:
            result['description'] = schema['description']
        
        if 'enum' in schema:
            result['enum'] = schema['enum']
        
        return result

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
        """Parse a Google function call response into a Pydantic model."""
        candidates = response.get('candidates', [])
        if not candidates:
            raise ValueError("No candidates in response")

        content = candidates[0].get('content', {})
        parts = content.get('parts', [])

        for part in parts:
            if 'functionCall' in part:
                data = part['functionCall'].get('args', {})
                return output_format.model_validate(data)

        raise ValueError("No function call in response")

    def parse_content_response(self, response: dict[str, Any]) -> str:
        """Parse plain text content from Google response."""
        candidates = response.get('candidates', [])
        if not candidates:
            return ""
        
        content = candidates[0].get('content', {})
        parts = content.get('parts', [])
        texts = []
        
        for part in parts:
            if 'text' in part:
                texts.append(part['text'])
        
        return '\n'.join(texts)


# Singleton instance
google_serializer = GoogleMessageSerializer()

