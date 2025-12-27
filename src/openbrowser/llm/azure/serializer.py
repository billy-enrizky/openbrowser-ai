"""Azure OpenAI message serializer.

Azure OpenAI uses the same format as OpenAI, so we inherit from OpenAI serializer.
"""

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from src.openbrowser.llm.openai.serializer import OpenAIMessageSerializer

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class AzureOpenAIMessageSerializer(OpenAIMessageSerializer):
    """Serializer for converting messages to Azure OpenAI format.
    
    Azure OpenAI uses the same format as OpenAI with identical message structure.
    """
    pass


# Singleton instance
azure_openai_serializer = AzureOpenAIMessageSerializer()

