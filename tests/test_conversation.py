"""Tests for conversation saving and loading.

This module provides test coverage for the conversation persistence system,
which allows agent conversations to be saved to and loaded from files.
It validates:

    - Saving conversations with messages and metadata to JSON files
    - Loading conversations and reconstructing message objects
    - Handling of non-existent files with appropriate errors
    - Complete metadata preservation (task, steps, success, LLM info)
    - Text conversion for human-readable conversation exports
    - Truncation of long message content for display
    - ConversationMetadata model with default values

The conversation system enables session persistence, debugging, and
replay functionality for browser automation tasks.
"""

import pytest
import tempfile
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.openbrowser.agent.conversation import (
    save_conversation,
    load_conversation,
    conversation_to_text,
    ConversationMetadata,
)


class TestConversationSaveLoad:
    """Tests for saving and loading conversations.

    Validates the complete round-trip of conversation data through
    file-based persistence, including message content and metadata.
    """

    def test_save_and_load_conversation(self):
        """Test saving and loading a conversation."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there! How can I help you?"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "conversation.json"
            
            # Save
            save_conversation(
                messages=messages,
                file_path=file_path,
                task="Test task",
                total_steps=3,
            )
            
            assert file_path.exists()
            
            # Load
            loaded_messages, metadata = load_conversation(file_path)
            
            assert len(loaded_messages) == 3
            assert metadata.task == "Test task"
            assert metadata.total_steps == 3
            assert metadata.total_messages == 3

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_conversation("/nonexistent/path.json")

    def test_save_with_all_metadata(self):
        """Test saving with all metadata."""
        messages = [HumanMessage(content="Test")]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "conversation.json"
            
            save_conversation(
                messages=messages,
                file_path=file_path,
                task="Complete task",
                total_steps=10,
                is_successful=True,
                final_result="Task completed",
                llm_provider="openai",
                llm_model="gpt-4o",
            )
            
            _, metadata = load_conversation(file_path)
            
            assert metadata.is_successful is True
            assert metadata.final_result == "Task completed"
            assert metadata.llm_provider == "openai"
            assert metadata.llm_model == "gpt-4o"


class TestConversationToText:
    """Tests for the conversation_to_text function.

    Validates text conversion of conversation messages including role
    labeling, content formatting, and truncation of long content.
    """

    def test_conversation_to_text(self):
        """Test converting conversation to text."""
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User message"),
            AIMessage(content="Assistant response"),
        ]

        text = conversation_to_text(messages)
        
        assert "[System]" in text
        assert "System prompt" in text
        assert "[Human]" in text
        assert "User message" in text
        assert "[AI]" in text
        assert "Assistant response" in text

    def test_conversation_to_text_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = "x" * 1000
        messages = [HumanMessage(content=long_content)]

        text = conversation_to_text(messages)
        
        assert "..." in text
        assert len(text) < len(long_content) + 50


class TestConversationMetadata:
    """Tests for the ConversationMetadata class.

    Validates default values and optional field handling for
    conversation metadata including success status and LLM info.
    """

    def test_metadata_defaults(self):
        """Test ConversationMetadata defaults."""
        metadata = ConversationMetadata(
            task="Test",
            created_at="2024-01-01",
            total_messages=0,
            total_steps=0,
        )
        assert metadata.is_successful is None
        assert metadata.final_result is None
        assert metadata.llm_provider == "unknown"
        assert metadata.llm_model == "unknown"

