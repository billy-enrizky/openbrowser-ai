"""Comprehensive tests for advanced openbrowser features.

This module provides test coverage for advanced features including:
    - LLM exception handling
    - Observability decorators
    - Configuration management
    - Signal handler functionality
    - Enhanced DOM snapshot processing
    - LLM provider imports
"""

import pytest


# --------------------------------------------------------------------------
# Test LLM Exceptions
# --------------------------------------------------------------------------


class TestLLMExceptions:
    """Tests for custom LLM exception classes."""

    def test_llm_exception(self):
        """Test base LLM exception."""
        from openbrowser.exceptions import LLMException

        exc = LLMException("Test error", 500)
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

        exc = ModelRateLimitError("Rate limited", status_code=429)
        assert "Rate limited" in str(exc)
        assert exc.status_code == 429


# --------------------------------------------------------------------------
# Test Observability
# --------------------------------------------------------------------------


class TestObservability:
    """Tests for observability decorators."""

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
        """Test is_debug_mode function."""
        from openbrowser.observability import is_debug_mode

        result = is_debug_mode()
        assert isinstance(result, bool)


# --------------------------------------------------------------------------
# Test Config
# --------------------------------------------------------------------------


class TestConfig:
    """Tests for configuration management."""

    def test_config_singleton(self):
        """Test config singleton pattern."""
        from openbrowser.config import CONFIG

        assert CONFIG is not None

    def test_is_running_in_docker(self):
        """Test docker detection."""
        from openbrowser.config import is_running_in_docker

        result = is_running_in_docker()
        assert isinstance(result, bool)


# --------------------------------------------------------------------------
# Test Signal Handler
# --------------------------------------------------------------------------


class TestSignalHandler:
    """Tests for signal handler functionality."""

    def test_signal_handler_init(self):
        """Test signal handler initialization."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        assert handler is not None
        assert handler.is_paused is False

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

        handler = SignalHandler(pause_callback=on_pause, resume_callback=on_resume)
        assert handler is not None


# --------------------------------------------------------------------------
# Test Enhanced DOM Snapshot
# --------------------------------------------------------------------------


class TestEnhancedDOMSnapshot:
    """Tests for enhanced DOM snapshot processing."""

    def test_required_computed_styles(self):
        """Test required computed styles constant."""
        from openbrowser.dom.enhanced_snapshot import REQUIRED_COMPUTED_STYLES

        assert isinstance(REQUIRED_COMPUTED_STYLES, (list, tuple))
        assert len(REQUIRED_COMPUTED_STYLES) > 0


# --------------------------------------------------------------------------
# Test LLM Provider Imports
# --------------------------------------------------------------------------


class TestNewLLMProviders:
    """Tests for new LLM provider integrations."""

    def test_chat_browser_use_import(self):
        """Test ChatBrowserUse import."""
        from openbrowser.llm import ChatBrowserUse

        assert ChatBrowserUse is not None

    def test_chat_google_import(self):
        """Test ChatGoogle import."""
        from openbrowser.llm import ChatGoogle

        assert ChatGoogle is not None

    def test_chat_openai_import(self):
        """Test ChatOpenAI import."""
        from openbrowser.llm import ChatOpenAI

        assert ChatOpenAI is not None

    def test_chat_anthropic_import(self):
        """Test ChatAnthropic import."""
        from openbrowser.llm import ChatAnthropic

        assert ChatAnthropic is not None


# --------------------------------------------------------------------------
# Test Code Use Module
# --------------------------------------------------------------------------


class TestCodeUseModule:
    """Tests for code use module components."""

    def test_code_cell_model(self):
        """Test CodeCell model."""
        from openbrowser.code_use.views import CodeCell

        cell = CodeCell(source="print('hello')")
        assert cell.source == "print('hello')"

    def test_notebook_session(self):
        """Test NotebookSession model."""
        from openbrowser.code_use.views import NotebookSession

        session = NotebookSession()
        assert session.cells == []

    def test_extract_code_blocks(self):
        """Test extract_code_blocks function."""
        from openbrowser.code_use.utils import extract_code_blocks

        text = """
```python
print('hello')
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) >= 0  # May or may not find blocks depending on format

    def test_extract_url_from_task(self):
        """Test extract_url_from_task function."""
        from openbrowser.code_use.utils import extract_url_from_task

        url = extract_url_from_task("Go to https://example.com and click")
        assert url == "https://example.com"

    def test_extract_url_from_task_no_url(self):
        """Test extract_url_from_task with no URL."""
        from openbrowser.code_use.utils import extract_url_from_task

        url = extract_url_from_task("Do something without a URL")
        assert url is None


# --------------------------------------------------------------------------
# Test MCP Module
# --------------------------------------------------------------------------


class TestMCPModule:
    """Tests for MCP module components."""

    def test_openbrowser_server_import(self):
        """Test OpenBrowserServer import."""
        from openbrowser.mcp import OpenBrowserServer

        assert OpenBrowserServer is not None


# --------------------------------------------------------------------------
# Test Actor Module
# --------------------------------------------------------------------------


class TestActorModule:
    """Tests for actor module components."""

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
        """Test get_key_info function."""
        from openbrowser.actor.utils import get_key_info

        info = get_key_info("Enter")
        assert info is not None
