"""Tests for LLM provider modules.

This module provides test coverage for the LLM provider subsystem,
which provides unified interfaces to multiple language model providers.
"""

import pytest
import os
from unittest.mock import patch


class TestLLMImports:
    """Tests for LLM provider imports."""

    def test_import_openai(self):
        """Test OpenAI import."""
        from openbrowser.llm import ChatOpenAI
        assert ChatOpenAI is not None

    def test_import_google(self):
        """Test Google import."""
        from openbrowser.llm import ChatGoogle
        assert ChatGoogle is not None

    def test_import_anthropic(self):
        """Test Anthropic import."""
        from openbrowser.llm import ChatAnthropic
        assert ChatAnthropic is not None

    def test_import_groq(self):
        """Test Groq import."""
        from openbrowser.llm import ChatGroq
        assert ChatGroq is not None

    def test_import_ollama(self):
        """Test Ollama import."""
        from openbrowser.llm import ChatOllama
        assert ChatOllama is not None

    def test_import_aws(self):
        """Test AWS Bedrock import."""
        from openbrowser.llm import ChatAWSBedrock
        assert ChatAWSBedrock is not None

    def test_import_azure(self):
        """Test Azure OpenAI import."""
        from openbrowser.llm import ChatAzureOpenAI
        assert ChatAzureOpenAI is not None

    def test_import_browser_use(self):
        """Test ChatBrowserUse import."""
        from openbrowser.llm import ChatBrowserUse
        assert ChatBrowserUse is not None


class TestGetLLMByName:
    """Tests for the get_llm_by_name factory function."""

    def test_get_llm_models_import(self):
        """Test models module import."""
        from openbrowser.llm import models
        assert models is not None


class TestBaseChatModel:
    """Tests for the BaseChatModel abstract class."""

    def test_base_chat_model_import(self):
        """Test BaseChatModel import."""
        from openbrowser.llm.base import BaseChatModel
        assert BaseChatModel is not None
