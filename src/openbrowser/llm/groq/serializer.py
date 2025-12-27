"""Groq message serializer.

Groq uses OpenAI-compatible API format, so we inherit from OpenAI serializer.
"""

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from src.openbrowser.llm.openai.serializer import OpenAIMessageSerializer

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class GroqMessageSerializer(OpenAIMessageSerializer):
    """Serializer for converting messages to Groq format.
    
    Groq uses OpenAI-compatible format with some limitations:
    - Some models may not support function calling
    - Image support varies by model
    """

    def serialize_tool_schema(self, output_format: type[T]) -> dict[str, Any]:
        """Serialize a Pydantic model to Groq function calling format.
        
        Note: Not all Groq models support function calling.
        """
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


# Singleton instance
groq_serializer = GroqMessageSerializer()

