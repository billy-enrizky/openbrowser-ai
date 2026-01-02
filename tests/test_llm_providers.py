"""Tests for LLM provider modules.

This module provides test coverage for the LLM provider subsystem,
which provides unified interfaces to multiple language model providers.
It validates:

    - Import availability for all supported providers (OpenAI, Google,
      Anthropic, Groq, Ollama, OpenRouter, AWS Bedrock, Azure OpenAI)
    - Factory function get_llm_by_name for provider instantiation
    - BaseChatModel abstract class and LangChainChatModelWrapper
    - Error handling for unknown providers

The LLM provider module enables flexible integration with various
language model services while maintaining a consistent interface.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock


class TestLLMImports:
    """Tests for LLM provider imports.

    Validates that all supported LLM provider classes can be imported
    without errors, ensuring proper module structure and dependencies.
    """

    def test_import_openai(self):
        """Test OpenAI import."""
        from src.openbrowser.llm.openai import ChatOpenAI
        assert ChatOpenAI is not None

    def test_import_google(self):
        """Test Google import."""
        from src.openbrowser.llm.google import ChatGoogle
        assert ChatGoogle is not None

    def test_import_anthropic(self):
        """Test Anthropic import."""
        from src.openbrowser.llm.anthropic import ChatAnthropic
        assert ChatAnthropic is not None

    def test_import_groq(self):
        """Test Groq import."""
        from src.openbrowser.llm.groq import ChatGroq
        assert ChatGroq is not None

    def test_import_ollama(self):
        """Test Ollama import."""
        from src.openbrowser.llm.ollama import ChatOllama
        assert ChatOllama is not None

    def test_import_openrouter(self):
        """Test OpenRouter import."""
        from src.openbrowser.llm.openrouter import ChatOpenRouter
        assert ChatOpenRouter is not None

    def test_import_aws(self):
        """Test AWS Bedrock import."""
        from src.openbrowser.llm.aws import ChatAWSBedrock
        assert ChatAWSBedrock is not None

    def test_import_azure(self):
        """Test Azure OpenAI import."""
        from src.openbrowser.llm.azure import ChatAzureOpenAI
        assert ChatAzureOpenAI is not None


class TestGetLLMByName:
    """Tests for the get_llm_by_name factory function.

    Validates provider instantiation through the factory function
    including successful creation and error handling for unknown providers.
    """

    def test_get_llm_openai(self):
        """Test getting OpenAI LLM."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from src.openbrowser.llm import get_llm_by_name
            llm = get_llm_by_name("openai", model="gpt-4o")
            assert llm is not None

    def test_get_llm_unknown_provider(self):
        """Test getting unknown provider raises error."""
        from src.openbrowser.llm import get_llm_by_name
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_by_name("unknown_provider")


class TestBaseChatModel:
    """Tests for the BaseChatModel abstract class.

    Validates the base chat model interface and the LangChain wrapper
    class that adapts LangChain models to the openbrowser interface.
    """

    def test_base_chat_model_import(self):
        """Test BaseChatModel import."""
        from src.openbrowser.llm.base import BaseChatModel
        assert BaseChatModel is not None

    def test_langchain_wrapper_import(self):
        """Test LangChainChatModelWrapper import."""
        from src.openbrowser.llm.base import LangChainChatModelWrapper
        assert LangChainChatModelWrapper is not None

