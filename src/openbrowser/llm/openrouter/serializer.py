"""OpenRouter message serializer.

OpenRouter uses OpenAI-compatible API format.
"""

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from src.openbrowser.llm.openai.serializer import OpenAIMessageSerializer

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class OpenRouterMessageSerializer(OpenAIMessageSerializer):
    """Serializer for converting messages to OpenRouter format.
    
    OpenRouter uses OpenAI-compatible format and routes to various models.
    Some models may have different capabilities for function calling and images.
    """

    def serialize_tool_schema(self, output_format: type[T]) -> dict[str, Any]:
        """Serialize a Pydantic model to OpenRouter function calling format.
        
        Note: Function calling support varies by model on OpenRouter.
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

    def get_provider_preferences(self) -> dict[str, Any]:
        """Get OpenRouter-specific provider preferences.
        
        Can be used to specify provider routing preferences.
        """
        return {
            "allow_fallbacks": True,
            "require_parameters": False,
        }


# Singleton instance
openrouter_serializer = OpenRouterMessageSerializer()

