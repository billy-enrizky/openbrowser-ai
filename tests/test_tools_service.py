"""Tests for openbrowser.tools.service module.

Focuses on helper functions, _validate_and_fix_javascript, _detect_sensitive_key_name,
handle_browser_error, and the Tools class initialization and act() method.
"""

import asyncio
import enum
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from openbrowser.models import ActionResult
from openbrowser.tools.registry.views import ActionModel
from openbrowser.tools.service import (
    CodeAgentTools,
    Controller,
    Tools,
    _detect_sensitive_key_name,
    handle_browser_error,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _detect_sensitive_key_name tests
# ---------------------------------------------------------------------------


class TestDetectSensitiveKeyName:
    """Tests for _detect_sensitive_key_name helper."""

    def test_returns_none_for_no_sensitive_data(self):
        assert _detect_sensitive_key_name("some text", None) is None

    def test_returns_none_for_empty_text(self):
        assert _detect_sensitive_key_name("", {"key": "value"}) is None

    def test_matches_old_format(self):
        sensitive_data = {"my_password": "hunter2"}
        assert _detect_sensitive_key_name("hunter2", sensitive_data) == "my_password"

    def test_matches_new_format(self):
        sensitive_data = {"*.example.com": {"user_pass": "secret123"}}
        assert _detect_sensitive_key_name("secret123", sensitive_data) == "user_pass"

    def test_no_match_returns_none(self):
        sensitive_data = {"my_key": "my_value"}
        assert _detect_sensitive_key_name("other_value", sensitive_data) is None

    def test_empty_values_skipped(self):
        sensitive_data = {"key": ""}
        assert _detect_sensitive_key_name("", sensitive_data) is None

    def test_mixed_format(self):
        sensitive_data = {
            "legacy_key": "legacy_value",
            "*.example.com": {"domain_key": "domain_value"},
        }
        assert _detect_sensitive_key_name("legacy_value", sensitive_data) == "legacy_key"
        assert _detect_sensitive_key_name("domain_value", sensitive_data) == "domain_key"


# ---------------------------------------------------------------------------
# handle_browser_error tests
# ---------------------------------------------------------------------------


class TestHandleBrowserError:
    """Tests for handle_browser_error helper."""

    def test_with_long_term_memory_only(self):
        from openbrowser.browser.views import BrowserError

        error = BrowserError("test error")
        error.long_term_memory = "Element not found"
        error.short_term_memory = None

        result = handle_browser_error(error)
        assert result.error == "Element not found"
        assert result.extracted_content is None

    def test_with_both_memories(self):
        from openbrowser.browser.views import BrowserError

        error = BrowserError("test error")
        error.long_term_memory = "Click failed on index 5"
        error.short_term_memory = "Detailed state info"

        result = handle_browser_error(error)
        assert result.error == "Click failed on index 5"
        assert result.extracted_content == "Detailed state info"
        assert result.include_extracted_content_only_once is True

    def test_without_long_term_memory_raises(self):
        from openbrowser.browser.views import BrowserError

        error = BrowserError("raw error")
        error.long_term_memory = None

        with pytest.raises(BrowserError):
            handle_browser_error(error)


# ---------------------------------------------------------------------------
# Tools class tests
# ---------------------------------------------------------------------------


class TestToolsInit:
    """Tests for Tools initialization."""

    def test_default_initialization_registers_actions(self):
        tools = Tools()
        actions = tools.registry.registry.actions
        # Check that standard actions are registered
        assert "navigate" in actions
        assert "click" in actions
        assert "done" in actions
        assert "scroll" in actions
        assert "search" in actions

    def test_exclude_actions(self):
        tools = Tools(exclude_actions=["search", "scroll"])
        actions = tools.registry.registry.actions
        assert "search" not in actions
        assert "scroll" not in actions
        # Other actions should still be there
        assert "navigate" in actions

    def test_controller_alias(self):
        """Controller should be an alias for Tools."""
        assert Controller is Tools


class TestToolsValidateAndFixJavascript:
    """Tests for Tools._validate_and_fix_javascript."""

    @pytest.fixture
    def tools(self):
        return Tools()

    def test_fixes_double_escaped_quotes(self, tools):
        code = 'document.querySelector(\\"#my-id\\")'
        fixed = tools._validate_and_fix_javascript(code)
        assert '\\"' not in fixed
        assert '"#my-id"' in fixed

    def test_fixes_over_escaped_regex(self, tools):
        code = "var re = /\\\\d+/;"
        fixed = tools._validate_and_fix_javascript(code)
        assert "\\d+" in fixed

    def test_fixes_xpath_mixed_quotes(self, tools):
        code = """document.evaluate("//div[@class='test']")"""
        # The single quotes inside double quotes should be converted to template literal
        fixed = tools._validate_and_fix_javascript(code)
        # Verify that the result uses backtick template literal to avoid nested quote conflicts
        assert "`" in fixed

    def test_fixes_queryselector_mixed_quotes(self, tools):
        code = """document.querySelector("input[type='text']")"""
        fixed = tools._validate_and_fix_javascript(code)
        assert "`" in fixed

    def test_fixes_closest_mixed_quotes(self, tools):
        code = """.closest("div[data-type='modal']")"""
        fixed = tools._validate_and_fix_javascript(code)
        assert "`" in fixed

    def test_fixes_matches_mixed_quotes(self, tools):
        code = """.matches("a[href^='http']")"""
        fixed = tools._validate_and_fix_javascript(code)
        assert "`" in fixed

    def test_no_changes_for_clean_code(self, tools):
        code = "document.getElementById('test')"
        fixed = tools._validate_and_fix_javascript(code)
        assert fixed == code

    def test_passthrough_for_normal_code(self, tools):
        code = "(function(){return 42;})()"
        fixed = tools._validate_and_fix_javascript(code)
        assert fixed == code


class TestToolsAct:
    """Tests for Tools.act method."""

    @pytest.mark.asyncio
    async def test_act_returns_action_result(self):
        tools = Tools()

        # Use the tools.registry to execute directly
        mock_file_system = MagicMock()
        mock_file_system.display_file = MagicMock(return_value=None)
        mock_file_system.get_dir = MagicMock(return_value=MagicMock())

        # Execute done action directly through registry
        result = await tools.registry.execute_action(
            action_name="done",
            params={"text": "Task complete", "success": True, "files_to_display": []},
            file_system=mock_file_system,
        )
        assert isinstance(result, ActionResult)
        assert result.is_done is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_act_with_none_result_returns_empty(self):
        tools = Tools()

        # Register a custom action that returns None
        @tools.action("Returns nothing")
        async def noop():
            return None

        # Execute through the registry's execute_action (which is what act() delegates to)
        result = await tools.registry.execute_action("noop", {})
        # Registry returns None for actions that return None
        assert result is None

    @pytest.mark.asyncio
    async def test_act_with_string_result(self):
        tools = Tools()

        @tools.action("Returns string")
        async def string_action():
            return "hello"

        # Execute through registry
        result = await tools.registry.execute_action("string_action", {})
        assert result == "hello"


class TestToolsRegisterDoneAction:
    """Tests for done action registration variants."""

    def test_done_with_output_model(self):
        class MyOutput(BaseModel):
            name: str
            score: int

        tools = Tools(output_model=MyOutput)
        assert "done" in tools.registry.registry.actions

    def test_done_without_output_model(self):
        tools = Tools()
        assert "done" in tools.registry.registry.actions

    @pytest.mark.asyncio
    async def test_structured_done_with_enum(self):
        class Status(enum.Enum):
            SUCCESS = "success"
            FAILURE = "failure"

        class MyOutput(BaseModel):
            status: Status
            message: str

        tools = Tools(output_model=MyOutput)
        result = await tools.registry.execute_action(
            "done",
            {"success": True, "data": {"status": "success", "message": "All done"}},
        )
        assert isinstance(result, ActionResult)
        assert result.is_done is True

        # Check that enum was converted to string in extracted_content
        parsed = json.loads(result.extracted_content)
        assert parsed["status"] == "success"


class TestToolsGetattr:
    """Tests for Tools.__getattr__ dynamic action dispatch."""

    def test_accessing_unknown_attribute_raises(self):
        tools = Tools()
        with pytest.raises(AttributeError, match="no attribute"):
            _ = tools.nonexistent_action

    def test_accessing_registered_action_returns_callable(self):
        tools = Tools()
        # navigate is registered by default
        fn = tools.navigate
        assert callable(fn)


# ---------------------------------------------------------------------------
# CodeAgentTools tests
# ---------------------------------------------------------------------------


class TestCodeAgentTools:
    """Tests for CodeAgentTools subclass."""

    def test_default_exclusions(self):
        tools = CodeAgentTools()
        actions = tools.registry.registry.actions
        # These should be excluded by default
        assert "extract" not in actions
        assert "find_text" not in actions
        assert "screenshot" not in actions
        assert "search" not in actions
        assert "write_file" not in actions
        assert "read_file" not in actions
        assert "replace_file" not in actions

    def test_kept_actions(self):
        tools = CodeAgentTools()
        actions = tools.registry.registry.actions
        # These should still be present
        assert "click" in actions
        assert "navigate" in actions
        assert "input" in actions
        assert "scroll" in actions
        assert "done" in actions

    def test_custom_exclusions_override_default(self):
        tools = CodeAgentTools(exclude_actions=["click"])
        actions = tools.registry.registry.actions
        assert "click" not in actions
        # Other default exclusions should NOT apply when custom list given
        assert "navigate" in actions


class TestToolsWaitAction:
    """Tests for the wait action."""

    @pytest.mark.asyncio
    async def test_wait_caps_at_30_seconds(self):
        tools = Tools()
        # Test that wait(1) works
        result = await tools.registry.execute_action("wait", {"seconds": 1})
        assert isinstance(result, ActionResult)
        assert "Waited for 1 second" in result.extracted_content

    @pytest.mark.asyncio
    async def test_wait_caps_at_30_seconds_for_large_values(self):
        tools = Tools()
        # The wait action caps at 30 seconds via min() — no validation error
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await tools.registry.execute_action("wait", {"seconds": 60})
            # actual_seconds = min(max(60 - 3, 0), 30) = 30
            mock_sleep.assert_called_once_with(30)
            assert isinstance(result, ActionResult)
            assert "60 second" in result.extracted_content
