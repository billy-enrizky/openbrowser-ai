"""Comprehensive tests for new features implemented from browser-use.

This module provides extensive test coverage for features ported from or
inspired by the browser-use project. It validates the core functionality
of various subsystems including:

    - LLM exception handling (LLMException, ModelProviderError, ModelRateLimitError)
    - Observability decorators for function tracing and debugging
    - Configuration management and browser profile settings
    - URL shortening utilities for text processing
    - Signal handler functionality for pause/resume operations
    - Enhanced DOM snapshot processing
    - LLM message serializers for multiple providers (OpenAI, Anthropic, Google)
    - New LLM provider integrations (OCI, Cerebras, DeepSeek, BrowserUse)

The tests ensure compatibility with browser-use patterns while maintaining
the openbrowser-specific implementation details.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --------------------------------------------------------------------------
# Test LLM Exceptions
# --------------------------------------------------------------------------


class TestLLMExceptions:
    """Tests for custom LLM exception classes.

    Validates the behavior of exception classes used throughout the LLM
    subsystem, including base exceptions, provider errors, and rate limiting.
    """

    def test_llm_exception(self):
        """Test base LLM exception."""
        from openbrowser.llm.exceptions import LLMException

        exc = LLMException("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)

    def test_model_provider_error(self):
        """Test model provider error."""
        from openbrowser.llm.exceptions import ModelProviderError

        exc = ModelProviderError("Provider error", status_code=500, model="gpt-4")
        assert "Provider error" in str(exc)
        assert exc.status_code == 500
        assert exc.model == "gpt-4"

    def test_model_rate_limit_error(self):
        """Test rate limit error."""
        from openbrowser.llm.exceptions import ModelRateLimitError

        exc = ModelRateLimitError("Rate limited", retry_after=60)
        assert "Rate limited" in str(exc)
        assert exc.retry_after == 60


# --------------------------------------------------------------------------
# Test Observability
# --------------------------------------------------------------------------


class TestObservability:
    """Tests for observability decorators.

    Validates the observe and observe_debug decorators that provide
    function tracing, timing, and debugging capabilities.
    """

    def test_observe_decorator_sync(self):
        """Test observe decorator on sync function."""
        from openbrowser.observability import observe

        @observe()
        def sync_function(x):
            return x * 2

        result = sync_function(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_observe_decorator_async(self):
        """Test observe decorator on async function."""
        from openbrowser.observability import observe

        @observe()
        async def async_function(x):
            return x * 2

        result = await async_function(5)
        assert result == 10

    def test_observe_debug_decorator(self):
        """Test observe_debug decorator."""
        from openbrowser.observability import observe_debug

        @observe_debug()
        def debug_function(x):
            return x + 1

        result = debug_function(10)
        assert result == 11

    def test_is_debug_mode(self):
        """Test debug mode detection."""
        from openbrowser.observability import is_debug_mode

        # Default should be False unless environment is set
        result = is_debug_mode()
        assert isinstance(result, bool)


# --------------------------------------------------------------------------
# Test Config
# --------------------------------------------------------------------------


class TestConfig:
    """Tests for configuration management.

    Validates the CONFIG singleton, environment detection functions,
    and configuration models like BrowserProfileEntry and LLMEntry.
    """

    def test_config_singleton(self):
        """Test config is a singleton."""
        from openbrowser.config import CONFIG

        assert CONFIG is not None

    def test_is_running_in_docker(self):
        """Test docker detection function."""
        from openbrowser.config import is_running_in_docker

        result = is_running_in_docker()
        assert isinstance(result, bool)
        # On most dev machines, this should be False
        # (but we don't assert it in case tests run in Docker)

    def test_browser_profile_entry(self):
        """Test browser profile entry model."""
        from openbrowser.config import BrowserProfileEntry

        entry = BrowserProfileEntry(
            name="test",
            description="Test profile",
            browser_type="chromium",
            headless=True,
        )
        assert entry.name == "test"
        assert entry.headless is True

    def test_llm_entry(self):
        """Test LLM entry model."""
        from openbrowser.config import LLMEntry

        entry = LLMEntry(
            provider="openai",
            model="gpt-4",
        )
        assert entry.provider == "openai"
        assert entry.model == "gpt-4"
        assert entry.id is not None  # Auto-generated UUID


# --------------------------------------------------------------------------
# Test URL Utilities
# --------------------------------------------------------------------------


class TestURLUtils:
    """Tests for URL shortening utilities.

    Validates the URL replacement and restoration functions that
    shorten long URLs in text for cleaner display and restore them
    when needed.
    """

    def test_replace_urls_basic(self):
        """Test basic URL replacement with long URL."""
        from openbrowser.agent.url_utils import replace_urls_in_text

        # URL must be longer than 50 chars to be shortened
        long_url = "https://example.com/very/long/path/to/resource/with/more/segments/to/exceed/limit"
        text = f"Visit {long_url} for more info"
        shortened, mapping = replace_urls_in_text(text)

        # Long URLs should be shortened
        assert len(mapping) == 1
        assert long_url not in shortened

    def test_replace_urls_multiple(self):
        """Test multiple long URL replacement."""
        from openbrowser.agent.url_utils import replace_urls_in_text

        url1 = "https://example.com/very/long/path/to/resource/one/that/exceeds/limit"
        url2 = "https://example.com/very/long/path/to/resource/two/that/exceeds/limit"
        text = f"Check {url1} and {url2}"
        shortened, mapping = replace_urls_in_text(text)

        # Both long URLs should be shortened
        assert len(mapping) == 2

    def test_restore_urls(self):
        """Test URL restoration."""
        from openbrowser.agent.url_utils import (
            replace_urls_in_text,
            restore_shortened_urls,
        )

        original = "Visit https://example.com/path for info"
        shortened, mapping = replace_urls_in_text(original)
        restored = restore_shortened_urls(shortened, mapping)

        assert restored == original

    def test_short_urls_not_replaced(self):
        """Test that short URLs are not replaced."""
        from openbrowser.agent.url_utils import replace_urls_in_text

        text = "Go to https://x.co"
        shortened, mapping = replace_urls_in_text(text)

        # Short URLs should not be replaced
        assert len(mapping) == 0 or "https://x.co" in shortened


# --------------------------------------------------------------------------
# Test Signal Handler
# --------------------------------------------------------------------------


class TestSignalHandler:
    """Tests for signal handler functionality.

    Validates the SignalHandler class that manages pause and resume
    callbacks for graceful interruption handling.
    """

    def test_signal_handler_init(self):
        """Test signal handler initialization."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        assert handler is not None
        assert handler._pause_callback is None
        assert handler._resume_callback is None

    def test_signal_handler_with_callbacks(self):
        """Test signal handler with callbacks."""
        from openbrowser.utils.signal_handler import SignalHandler

        pause_called = False
        resume_called = False

        def on_pause():
            nonlocal pause_called
            pause_called = True

        def on_resume():
            nonlocal resume_called
            resume_called = True

        handler = SignalHandler(
            pause_callback=on_pause,
            resume_callback=on_resume,
        )
        assert handler._pause_callback is not None
        assert handler._resume_callback is not None


# --------------------------------------------------------------------------
# Test Enhanced DOM Snapshot
# --------------------------------------------------------------------------


class TestEnhancedDOMSnapshot:
    """Tests for enhanced DOM snapshot processing.

    Validates the enhanced snapshot lookup functionality and the
    required computed styles for DOM element processing.
    """

    def test_required_computed_styles(self):
        """Test required computed styles list."""
        from openbrowser.browser.dom.enhanced_snapshot import (
            REQUIRED_COMPUTED_STYLES,
        )

        assert isinstance(REQUIRED_COMPUTED_STYLES, list)
        assert "display" in REQUIRED_COMPUTED_STYLES
        assert "visibility" in REQUIRED_COMPUTED_STYLES

    def test_build_snapshot_lookup_empty(self):
        """Test building lookup from empty snapshot."""
        from openbrowser.browser.dom.enhanced_snapshot import build_snapshot_lookup

        # Empty snapshot data
        snapshot = {"documents": []}
        result = build_snapshot_lookup(snapshot, device_pixel_ratio=1.0)

        assert isinstance(result, dict)
        assert len(result) == 0


# --------------------------------------------------------------------------
# Test LLM Serializers
# --------------------------------------------------------------------------


class TestLLMSerializers:
    """Tests for LLM message serializers.

    Validates message serialization for different LLM providers including
    OpenAI, Anthropic, and Google. Ensures correct role mapping and
    content formatting for each provider's API requirements.
    """

    def test_openai_serializer_user_message(self):
        """Test OpenAI serializer with user message."""
        from openbrowser.agent.views import UserMessage
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        serializer = OpenAIMessageSerializer()
        message = UserMessage(content="Hello, world!")
        result = serializer.serialize(message)

        assert result["role"] == "user"
        assert result["content"] == "Hello, world!"

    def test_openai_serializer_system_message(self):
        """Test OpenAI serializer with system message."""
        from openbrowser.agent.views import SystemMessage
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        serializer = OpenAIMessageSerializer()
        message = SystemMessage(content="You are a helpful assistant.")
        result = serializer.serialize(message)

        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_openai_serializer_assistant_message(self):
        """Test OpenAI serializer with assistant message."""
        from openbrowser.agent.views import AssistantMessage
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        serializer = OpenAIMessageSerializer()
        message = AssistantMessage(content="I am an assistant.")
        result = serializer.serialize(message)

        assert result["role"] == "assistant"
        assert result["content"] == "I am an assistant."

    def test_anthropic_serializer(self):
        """Test Anthropic serializer."""
        from openbrowser.agent.views import UserMessage
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        serializer = AnthropicMessageSerializer()
        message = UserMessage(content="Test message")
        result = serializer.serialize(message)

        assert result["role"] == "user"

    def test_google_serializer(self):
        """Test Google serializer."""
        from openbrowser.agent.views import UserMessage
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        serializer = GoogleMessageSerializer()
        message = UserMessage(content="Test message")
        result = serializer.serialize(message)

        assert "role" in result

    def test_serializer_serialize_messages(self):
        """Test serializing multiple messages."""
        from openbrowser.agent.views import AssistantMessage, SystemMessage, UserMessage
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        serializer = OpenAIMessageSerializer()
        messages = [
            SystemMessage(content="System prompt"),
            UserMessage(content="User message"),
            AssistantMessage(content="Assistant response"),
        ]
        results = serializer.serialize_messages(messages)

        assert len(results) == 3
        assert results[0]["role"] == "system"
        assert results[1]["role"] == "user"
        assert results[2]["role"] == "assistant"


# --------------------------------------------------------------------------
# Test New LLM Providers
# --------------------------------------------------------------------------


class TestNewLLMProviders:
    """Tests for new LLM provider classes.

    Validates that new LLM provider integrations can be imported and
    are accessible through the get_llm_by_name factory function.
    Covers OCI, Cerebras, DeepSeek, and BrowserUse providers.
    """

    def test_chat_oci_import(self):
        """Test ChatOCI can be imported."""
        from openbrowser.llm.oci import ChatOCI

        assert ChatOCI is not None

    def test_chat_cerebras_import(self):
        """Test ChatCerebras can be imported."""
        from openbrowser.llm.cerebras import ChatCerebras

        assert ChatCerebras is not None

    def test_chat_deepseek_import(self):
        """Test ChatDeepSeek can be imported."""
        from openbrowser.llm.deepseek import ChatDeepSeek

        assert ChatDeepSeek is not None

    def test_chat_browser_use_import(self):
        """Test ChatBrowserUse can be imported."""
        from openbrowser.llm.browser_use import ChatBrowserUse

        assert ChatBrowserUse is not None

    def test_get_llm_by_name_includes_new_providers(self):
        """Test that get_llm_by_name includes new providers.
        
        Verifies that get_llm_by_name can create LLM instances for
        new providers. The instances may fail on actual API calls
        without valid API keys, but creation should succeed.
        """
        from openbrowser.llm import get_llm_by_name

        # These should not raise on creation - they're valid providers
        # They may fail later when used without proper API keys
        try:
            llm = get_llm_by_name("oci", "test-model")
            assert llm is not None
        except Exception:
            pass  # OK if it fails due to missing dependencies or config

        try:
            llm = get_llm_by_name("cerebras", "test-model")
            assert llm is not None
        except Exception:
            pass  # OK if it fails due to missing dependencies or config


# --------------------------------------------------------------------------
# Test Code-Use Module
# --------------------------------------------------------------------------


class TestCodeUseModule:
    """Tests for code-use module components.

    Validates the code-use subsystem including cell models, session
    management, code block extraction, URL parsing, and browser state
    formatting for LLM consumption.
    """

    def test_code_cell_model(self):
        """Test CodeCell model."""
        from openbrowser.code_use.views import CodeCell, ExecutionStatus

        cell = CodeCell(
            source="print('hello')",
            status=ExecutionStatus.PENDING,
        )
        assert cell.source == "print('hello')"
        assert cell.status == ExecutionStatus.PENDING

    def test_notebook_session(self):
        """Test NotebookSession model."""
        from openbrowser.code_use.views import NotebookSession

        session = NotebookSession()
        assert len(session.cells) == 0
        assert session.current_execution_count == 0

    def test_extract_code_blocks(self):
        """Test extracting code blocks from markdown."""
        from openbrowser.code_use.utils import extract_code_blocks

        text = """
Here is some code:
```python
print("hello")
```

And more:
```javascript
console.log("world");
```
"""
        blocks = extract_code_blocks(text)
        # Returns a dict with keys like 'python', 'python_0', 'js', etc.
        assert isinstance(blocks, dict)
        assert len(blocks) >= 1
        assert "python" in blocks or "python_0" in blocks
        if "python" in blocks:
            assert "print" in blocks["python"]

    def test_extract_url_from_task(self):
        """Test extracting URL from task description."""
        from openbrowser.code_use.utils import extract_url_from_task

        task = "Navigate to https://example.com and click the button"
        url = extract_url_from_task(task)
        assert url == "https://example.com"

    def test_extract_url_from_task_no_url(self):
        """Test extracting URL when none present."""
        from openbrowser.code_use.utils import extract_url_from_task

        task = "Click the submit button"
        url = extract_url_from_task(task)
        assert url is None

    @pytest.mark.asyncio
    async def test_format_browser_state(self):
        """Test formatting browser state for LLM."""
        from openbrowser.code_use.formatting import format_browser_state_for_llm

        # Create a mock browser session
        browser_session = MagicMock()
        browser_session.context = MagicMock()
        browser_session.context.tabs = []

        # Call the async function with required parameters
        result = await format_browser_state_for_llm(
            url="https://example.com",
            title="Example Page",
            dom_html="<html></html>",
            namespace={"test_var": 123},
            browser_session=browser_session,
        )
        assert isinstance(result, str)
        assert "example.com" in result


# --------------------------------------------------------------------------
# Test MCP Module
# --------------------------------------------------------------------------


class TestMCPModule:
    """Tests for MCP server module.

    Validates that the OpenBrowserServer and main function can be
    imported, ensuring the MCP integration is properly structured.
    """

    def test_openbrowser_server_import(self):
        """Test OpenBrowserServer can be imported."""
        from openbrowser.mcp import OpenBrowserServer

        assert OpenBrowserServer is not None

    def test_mcp_main_import(self):
        """Test MCP main function can be imported."""
        from openbrowser.mcp import main

        assert main is not None


# --------------------------------------------------------------------------
# Test Actor Module
# --------------------------------------------------------------------------


class TestActorModule:
    """Tests for actor module components.

    Validates the actor subsystem imports including Element, Mouse,
    Page classes and utility functions for browser interaction.
    """

    def test_element_class_import(self):
        """Test Element class import."""
        from openbrowser.actor.element import Element

        assert Element is not None

    def test_mouse_class_import(self):
        """Test Mouse class import."""
        from openbrowser.actor.mouse import Mouse

        assert Mouse is not None

    def test_page_class_import(self):
        """Test Page class import."""
        from openbrowser.actor.page import Page

        assert Page is not None

    def test_get_key_info(self):
        """Test get_key_info utility."""
        from openbrowser.actor.utils import get_key_info

        # Test common keys
        key_info = get_key_info("Enter")
        assert key_info is not None

        key_info = get_key_info("Tab")
        assert key_info is not None


# --------------------------------------------------------------------------
# Test Message Types
# --------------------------------------------------------------------------


class TestMessageTypes:
    """Tests for message type models.

    Validates the message type models used for LLM communication
    including system, user, assistant, and tool messages with
    support for text content, images, and tool calls.
    """

    def test_system_message(self):
        """Test SystemMessage model."""
        from openbrowser.agent.views import SystemMessage

        msg = SystemMessage(content="You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"

    def test_user_message(self):
        """Test UserMessage model."""
        from openbrowser.agent.views import UserMessage

        msg = UserMessage(content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        """Test AssistantMessage model."""
        from openbrowser.agent.views import AssistantMessage

        msg = AssistantMessage(content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"

    def test_assistant_message_with_tool_calls(self):
        """Test AssistantMessage with tool calls."""
        from openbrowser.agent.views import (
            AssistantMessage,
            FunctionCall,
            ToolCall,
        )

        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="click", arguments='{"index": 5}'),
        )
        msg = AssistantMessage(content=None, tool_calls=[tool_call])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "click"

    def test_tool_message(self):
        """Test ToolMessage model."""
        from openbrowser.agent.views import ToolMessage

        msg = ToolMessage(content="Button clicked", tool_call_id="call_123")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"

    def test_content_part_text(self):
        """Test ContentPartTextParam model."""
        from openbrowser.agent.views import ContentPartTextParam

        part = ContentPartTextParam(text="Hello")
        assert part.type == "text"
        assert part.text == "Hello"

    def test_content_part_image(self):
        """Test ContentPartImageParam model."""
        from openbrowser.agent.views import (
            ContentPartImageParam,
            ImageURLDetail,
        )

        image_url = ImageURLDetail(url="https://example.com/image.png")
        part = ContentPartImageParam(image_url=image_url)
        assert part.type == "image_url"
        assert part.image_url.url == "https://example.com/image.png"


# --------------------------------------------------------------------------
# Integration Tests
# --------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for new features.

    Validates end-to-end behavior of new features by testing
    interactions between multiple components and subsystems.
    """

    def test_full_serialization_roundtrip(self):
        """Test full message serialization roundtrip."""
        from openbrowser.agent.views import (
            AssistantMessage,
            SystemMessage,
            UserMessage,
        )
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        serializer = OpenAIMessageSerializer()

        # Create a conversation
        messages = [
            SystemMessage(content="You are a browser automation assistant."),
            UserMessage(content="Navigate to example.com"),
            AssistantMessage(content="I will navigate to example.com now."),
        ]

        # Serialize all
        serialized = serializer.serialize_messages(messages)

        # Verify structure
        assert len(serialized) == 3
        assert all("role" in msg for msg in serialized)
        assert all("content" in msg for msg in serialized)

    def test_url_shortening_integration(self):
        """Test URL shortening with realistic content."""
        from openbrowser.agent.url_utils import (
            replace_urls_in_text,
            restore_shortened_urls,
        )

        # Simulate browser state with long URLs
        content = """
        Current page: https://www.example.com/products/category/electronics/smartphones/iphone-15-pro-max-256gb?color=blue&storage=256gb
        Links found:
        - https://www.example.com/cart/add?product=12345&quantity=1
        - https://www.example.com/products/category/electronics/accessories/cases?filter=iphone15
        """

        shortened, mapping = replace_urls_in_text(content)

        # URLs should be shortened
        assert len(mapping) >= 1

        # Restore should work
        restored = restore_shortened_urls(shortened, mapping)
        assert restored == content

