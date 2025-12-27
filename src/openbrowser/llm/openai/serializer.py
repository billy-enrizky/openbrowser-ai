"""OpenAI message serializer."""

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from src.openbrowser.agent.views import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    UserMessage,
)
from src.openbrowser.llm.serializer import BaseMessageSerializer

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class OpenAIMessageSerializer(BaseMessageSerializer):
    """Serializer for converting messages to OpenAI format."""

    def serialize(self, message: BaseMessage) -> dict[str, Any]:
        """Serialize a message to OpenAI format."""
        if isinstance(message, UserMessage):
            result = {
                'role': 'user',
                'content': self.serialize_user_content(message.content),
            }
            if message.name is not None:
                result['name'] = message.name
            return result

        elif isinstance(message, SystemMessage):
            result = {
                'role': 'system',
                'content': self.serialize_system_content(message.content),
            }
            if message.name is not None:
                result['name'] = message.name
            return result

        elif isinstance(message, AssistantMessage):
            content = None
            if message.content is not None:
                content = self.serialize_assistant_content(message.content)

            result = {'role': 'assistant'}

            if content is not None:
                result['content'] = content
            if message.name is not None:
                result['name'] = message.name
            if message.refusal is not None:
                result['refusal'] = message.refusal
            if message.tool_calls:
                result['tool_calls'] = [
                    self.serialize_tool_call(tc) for tc in message.tool_calls
                ]
            return result

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    def serialize_tool_schema(self, output_format: type[T]) -> dict[str, Any]:
        """Serialize a Pydantic model to OpenAI function calling format."""
        schema = output_format.model_json_schema()
        
        # Remove definitions key if present and inline them
        if '$defs' in schema:
            defs = schema.pop('$defs')
            schema = self._resolve_refs(schema, defs)
        
        return {
            "type": "function",
            "function": {
                "name": output_format.__name__,
                "description": output_format.__doc__ or f"Generate {output_format.__name__}",
                "strict": True,
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

    def get_response_format(self, output_format: type[T]) -> dict[str, Any]:
        """Get response_format parameter for structured output.
        
        OpenAI supports json_schema response format for guaranteed JSON output.
        """
        schema = output_format.model_json_schema()
        
        # Remove definitions and resolve refs
        if '$defs' in schema:
            defs = schema.pop('$defs')
            schema = self._resolve_refs(schema, defs)
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": output_format.__name__,
                "strict": True,
                "schema": schema,
            }
        }

    def parse_tool_call_response(
        self,
        response: dict[str, Any],
        output_format: type[T],
    ) -> T:
        """Parse an OpenAI tool call response into a Pydantic model."""
        choices = response.get('choices', [])
        if not choices:
            raise ValueError("No choices in response")

        message = choices[0].get('message', {})
        tool_calls = message.get('tool_calls', [])

        if tool_calls:
            # Parse from tool call arguments
            arguments = tool_calls[0].get('function', {}).get('arguments', '{}')
            data = json.loads(arguments)
            return output_format.model_validate(data)

        # Parse from content if using response_format
        content = message.get('content', '')
        if content:
            data = json.loads(content)
            return output_format.model_validate(data)

        raise ValueError("No tool calls or content in response")

    def parse_content_response(self, response: dict[str, Any]) -> str:
        """Parse plain text content from OpenAI response."""
        choices = response.get('choices', [])
        if not choices:
            return ""
        
        message = choices[0].get('message', {})
        return message.get('content', '') or ""


# Singleton instance for convenience
openai_serializer = OpenAIMessageSerializer()

