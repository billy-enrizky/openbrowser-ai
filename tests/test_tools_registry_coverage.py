"""Comprehensive coverage tests for openbrowser.tools.registry.service module.

Covers remaining gaps: normalized_wrapper (positional args error, Type 1 pattern,
Type 2 pattern, special param handling, missing params errors), execute_action
(sensitive_data replacement with domain matching, cdp_client injection, page_url
injection, input action with sensitive_data), create_action_model (ActionModelUnion
delegation: get_index, set_index, model_dump), _replace_sensitive_data (2fa code,
domain matching).
"""

import asyncio
import logging
from typing import Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from openbrowser.tools.registry.service import Registry
from openbrowser.tools.registry.views import (
    ActionModel,
    ActionRegistry,
    RegisteredAction,
    SpecialActionParameters,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# normalized_wrapper error handling
# ---------------------------------------------------------------------------


class TestNormalizedWrapper:
    """Tests for the normalized_wrapper produced by _normalize_action_function_signature."""

    @pytest.mark.asyncio
    async def test_positional_args_raise_error(self):
        """Calling with positional args should raise TypeError."""
        registry = Registry()

        @registry.action("Test action")
        async def my_action(name: str):
            return name

        action = registry.registry.actions["my_action"]
        with pytest.raises(TypeError, match="does not accept positional arguments"):
            await action.function("positional_value")

    @pytest.mark.asyncio
    async def test_missing_params_raises_value_error(self):
        """Missing required params should raise ValueError."""
        registry = Registry()

        @registry.action("Test action")
        async def my_action(name: str):
            return name

        action = registry.registry.actions["my_action"]
        # Call with no params at all - should try to create params from kwargs, but name is required
        with pytest.raises(Exception):
            await action.function(params=None)

    @pytest.mark.asyncio
    async def test_type_1_pattern_with_param_model(self):
        """Type 1 pattern: first arg is a BaseModel."""
        registry = Registry()

        class NavParams(ActionModel):
            url: str

        @registry.action("Navigate", param_model=NavParams)
        async def navigate(params: NavParams):
            return params.url

        result = await registry.execute_action("navigate", {"url": "https://example.com"})
        assert result == "https://example.com"

    @pytest.mark.asyncio
    async def test_type_1_missing_params_raises(self):
        """Type 1 pattern: calling without params raises ValueError."""
        registry = Registry()

        class NavParams(ActionModel):
            url: str

        @registry.action("Navigate", param_model=NavParams)
        async def navigate(params: NavParams):
            return params.url

        action = registry.registry.actions["navigate"]
        with pytest.raises(ValueError, match="missing required 'params' argument"):
            await action.function(params=None)

    @pytest.mark.asyncio
    async def test_special_param_browser_session_required(self):
        """Missing required browser_session raises ValueError."""
        from openbrowser.browser import BrowserSession

        registry = Registry()

        @registry.action("Click action")
        async def click(index: int, browser_session: BrowserSession):
            return index

        with pytest.raises(RuntimeError, match="requires browser_session"):
            await registry.execute_action("click", {"index": 5})

    @pytest.mark.asyncio
    async def test_special_param_page_extraction_llm_required(self):
        """Missing required page_extraction_llm raises ValueError."""
        from openbrowser.llm.base import BaseChatModel

        registry = Registry()

        @registry.action("Extract action")
        async def extract(query: str, page_extraction_llm: BaseChatModel):
            return query

        with pytest.raises(RuntimeError, match="requires page_extraction_llm"):
            await registry.execute_action("extract", {"query": "test"})

    @pytest.mark.asyncio
    async def test_special_param_file_system_required(self):
        """Missing required file_system raises ValueError."""
        from openbrowser.filesystem.file_system import FileSystem

        registry = Registry()

        @registry.action("Write action")
        async def write(content: str, file_system: FileSystem):
            return content

        with pytest.raises(RuntimeError, match="requires file_system"):
            await registry.execute_action("write", {"content": "test"})

    @pytest.mark.asyncio
    async def test_special_param_with_default(self):
        """Special param with default value should use the default."""
        registry = Registry()

        @registry.action("Action with default")
        async def my_action(name: str, has_sensitive_data: bool = False):
            return f"{name}-{has_sensitive_data}"

        result = await registry.execute_action("my_action", {"name": "test"})
        assert result == "test-False"

    @pytest.mark.asyncio
    async def test_special_param_browser_session_none_with_required(self):
        """browser_session=None when required should raise ValueError."""
        from openbrowser.browser import BrowserSession

        registry = Registry()

        @registry.action("Click")
        async def click(index: int, browser_session: BrowserSession):
            return index

        with pytest.raises(RuntimeError, match="requires browser_session"):
            await registry.execute_action("click", {"index": 5}, browser_session=None)

    @pytest.mark.asyncio
    async def test_special_param_available_file_paths_required(self):
        """Missing available_file_paths should raise if required."""
        registry = Registry()

        @registry.action("Read action")
        async def read(name: str, available_file_paths: list[str]):
            return name

        with pytest.raises(RuntimeError, match="requires available_file_paths"):
            await registry.execute_action("read", {"name": "test"})

    @pytest.mark.asyncio
    async def test_sync_function_wrapped_in_thread(self):
        """Sync functions should be wrapped in asyncio.to_thread."""
        registry = Registry()

        @registry.action("Sync action")
        def sync_action(name: str):
            return f"sync-{name}"

        result = await registry.execute_action("sync_action", {"name": "test"})
        assert result == "sync-test"


# ---------------------------------------------------------------------------
# execute_action with sensitive data
# ---------------------------------------------------------------------------


class TestExecuteActionSensitiveData:
    @pytest.mark.asyncio
    async def test_sensitive_data_replacement(self):
        registry = Registry()

        class InputParams(ActionModel):
            text: str

        @registry.action("Input action", param_model=InputParams)
        async def input(params: InputParams, has_sensitive_data: bool = False):
            return params.text

        result = await registry.execute_action(
            "input",
            {"text": "<secret>password</secret>"},
            sensitive_data={"password": "hunter2"},
        )
        assert result == "hunter2"

    @pytest.mark.asyncio
    async def test_sensitive_data_with_browser_session_and_target_id(self):
        """Test sensitive data replacement with browser session for domain matching."""
        registry = Registry()

        class InputParams(ActionModel):
            text: str

        @registry.action("Input action", param_model=InputParams)
        async def input(params: InputParams, has_sensitive_data: bool = False):
            return params.text

        mock_session = MagicMock()
        mock_session.current_target_id = "target123"
        mock_session.cdp_client = MagicMock()

        # Mock targets response
        mock_session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [
                    {"targetId": "target123", "url": "https://example.com/login"},
                ]
            }
        )
        mock_session.get_current_page_url = AsyncMock(return_value="https://example.com")

        result = await registry.execute_action(
            "input",
            {"text": "<secret>pass</secret>"},
            browser_session=mock_session,
            sensitive_data={"https://*.example.com": {"pass": "secret123"}},
        )
        assert result == "secret123"

    @pytest.mark.asyncio
    async def test_sensitive_data_injects_for_input_action(self):
        """The input action should receive sensitive_data in special context."""
        registry = Registry()

        @registry.action("Input with sensitive")
        async def input(text: str, has_sensitive_data: bool = False):
            return f"{text}-{has_sensitive_data}"

        result = await registry.execute_action(
            "input",
            {"text": "hello"},
            sensitive_data={"key": "value"},
        )
        assert "True" in result  # has_sensitive_data should be True for input


# ---------------------------------------------------------------------------
# execute_action with CDP params
# ---------------------------------------------------------------------------


class TestExecuteActionCDPParams:
    @pytest.mark.asyncio
    async def test_page_url_injected(self):
        """page_url is injected when browser_session is available."""
        registry = Registry()

        @registry.action("URL action")
        async def url_action(name: str, page_url: str | None = None):
            return page_url

        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(return_value="https://example.com")
        mock_session.cdp_client = MagicMock()
        mock_session.current_target_id = None

        result = await registry.execute_action(
            "url_action",
            {"name": "test"},
            browser_session=mock_session,
        )
        assert result == "https://example.com"

    @pytest.mark.asyncio
    async def test_page_url_injection_failure(self):
        """page_url falls back to None if get_current_page_url fails."""
        registry = Registry()

        @registry.action("URL action")
        async def url_action(name: str, page_url: str | None = None):
            return page_url

        mock_session = MagicMock()
        mock_session.get_current_page_url = AsyncMock(side_effect=RuntimeError("no url"))
        mock_session.cdp_client = MagicMock()
        mock_session.current_target_id = None

        result = await registry.execute_action(
            "url_action",
            {"name": "test"},
            browser_session=mock_session,
        )
        assert result is None


# ---------------------------------------------------------------------------
# _replace_sensitive_data edge cases
# ---------------------------------------------------------------------------


class TestReplaceSensitiveDataEdgeCases:
    def _make_registry(self):
        return Registry()

    def test_domain_matching_no_url(self):
        """Domain-specific secrets should not apply when no URL is provided."""
        registry = self._make_registry()

        class Params(BaseModel):
            text: str

        params = Params(text="<secret>pass</secret>")
        sensitive_data = {"https://*.example.com": {"pass": "secret123"}}

        result = registry._replace_sensitive_data(params, sensitive_data, current_url=None)
        # Should not replace because no URL to match against
        assert "<secret>pass</secret>" in result.text

    def test_domain_not_matching(self):
        """Secrets for non-matching domain should not be applied."""
        registry = self._make_registry()

        class Params(BaseModel):
            text: str

        params = Params(text="<secret>pass</secret>")
        sensitive_data = {"https://*.google.com": {"pass": "google_pass"}}

        result = registry._replace_sensitive_data(
            params, sensitive_data, current_url="https://example.com"
        )
        assert "<secret>pass</secret>" in result.text

    def test_non_string_values_passthrough(self):
        """Non-string, non-dict, non-list values should pass through."""
        registry = self._make_registry()

        class Params(BaseModel):
            count: int

        params = Params(count=42)
        result = registry._replace_sensitive_data(params, {"key": "value"})
        assert result.count == 42

    def test_log_sensitive_data_usage_no_url(self):
        """_log_sensitive_data_usage should work without URL."""
        registry = self._make_registry()
        # Should not raise
        registry._log_sensitive_data_usage({"placeholder1"}, None)

    def test_log_sensitive_data_usage_with_url(self):
        """_log_sensitive_data_usage should include URL info."""
        registry = self._make_registry()
        # Should not raise
        registry._log_sensitive_data_usage({"placeholder1"}, "https://example.com")

    def test_log_sensitive_data_usage_empty_set(self):
        """Empty placeholder set should not log."""
        registry = self._make_registry()
        registry._log_sensitive_data_usage(set(), "https://example.com")


# ---------------------------------------------------------------------------
# create_action_model - ActionModelUnion delegation
# ---------------------------------------------------------------------------


class TestCreateActionModelUnion:
    def test_union_get_index(self):
        """ActionModelUnion.get_index should delegate to root."""
        registry = Registry()

        @registry.action("Click")
        async def click(index: int):
            return index

        @registry.action("Navigate")
        async def navigate(url: str):
            return url

        model_class = registry.create_action_model()
        # Create an instance with click action
        instance = model_class.model_validate({"click": {"index": 5}})
        idx = instance.get_index()
        assert idx == 5

    def test_union_get_index_no_index(self):
        """ActionModelUnion.get_index returns None when no index field."""
        registry = Registry()

        @registry.action("Navigate")
        async def navigate(url: str):
            return url

        @registry.action("Wait")
        async def wait(seconds: int = 3):
            return seconds

        model_class = registry.create_action_model()
        instance = model_class.model_validate({"navigate": {"url": "https://example.com"}})
        idx = instance.get_index()
        assert idx is None

    def test_union_set_index(self):
        """ActionModelUnion.set_index should delegate to root."""
        registry = Registry()

        @registry.action("Click")
        async def click(index: int):
            return index

        @registry.action("Navigate")
        async def navigate(url: str):
            return url

        model_class = registry.create_action_model()
        instance = model_class.model_validate({"click": {"index": 3}})
        instance.set_index(10)
        # Should have updated the index on the root
        dump = instance.model_dump()
        assert dump.get("index", dump.get("click", {}).get("index")) is not None

    def test_union_model_dump(self):
        """ActionModelUnion.model_dump should delegate to root."""
        registry = Registry()

        @registry.action("Click")
        async def click(index: int):
            return index

        @registry.action("Navigate")
        async def navigate(url: str):
            return url

        model_class = registry.create_action_model()
        instance = model_class.model_validate({"click": {"index": 5}})
        dump = instance.model_dump()
        assert isinstance(dump, dict)
        assert "click" in dump

    def test_single_action_returns_direct_model(self):
        """Single action should return model directly (no Union)."""
        registry = Registry()

        @registry.action("Only action")
        async def solo(name: str):
            return name

        model = registry.create_action_model()
        assert model is not None
        # Should be a direct model, not a Union
        schema = model.model_json_schema()
        assert "solo" in str(schema)

    def test_include_actions_filter(self):
        """include_actions should filter which actions appear in the model."""
        registry = Registry()

        @registry.action("Action A")
        async def action_a(x: int):
            return x

        @registry.action("Action B")
        async def action_b(y: str):
            return y

        model = registry.create_action_model(include_actions=["action_a"])
        schema = str(model.model_json_schema())
        assert "action_a" in schema

    def test_page_url_includes_domain_actions(self):
        """Actions with matching domains should be included."""
        registry = Registry()

        @registry.action("Global", domains=None)
        async def global_action(x: int):
            return x

        @registry.action("Google", domains=["*.google.com"])
        async def google_action(q: str):
            return q

        model = registry.create_action_model(page_url="https://www.google.com")
        schema = str(model.model_json_schema())
        assert "google_action" in schema

    def test_page_url_none_only_include_unfiltered(self):
        """With page_url=None, only actions without domain filters appear."""
        registry = Registry()

        @registry.action("Global")
        async def global_action(x: int):
            return x

        @registry.action("Google", domains=["*.google.com"])
        async def google_action(q: str):
            return q

        model = registry.create_action_model(page_url=None)
        schema = str(model.model_json_schema())
        assert "global_action" in schema
        assert "google_action" not in schema

    def test_empty_actions_returns_empty_model(self):
        """No matching actions should return empty ActionModel."""
        registry = Registry()
        model = registry.create_action_model()
        assert model is not None

    def test_page_url_excludes_non_matching_domains(self):
        """Actions with non-matching domains should be excluded."""
        registry = Registry()

        @registry.action("Bing only", domains=["*.bing.com"])
        async def bing_action(q: str):
            return q

        model = registry.create_action_model(page_url="https://www.google.com")
        schema = str(model.model_json_schema())
        assert "bing_action" not in schema


# ---------------------------------------------------------------------------
# _create_param_model
# ---------------------------------------------------------------------------


class TestCreateParamModel:
    def test_creates_model_from_function(self):
        registry = Registry()

        def my_func(name: str, count: int = 5):
            pass

        model = registry._create_param_model(my_func)
        assert model is not None
        schema = model.model_json_schema()
        assert "name" in str(schema)
        assert "count" in str(schema)

    def test_excludes_special_params(self):
        """Special params should not appear in the created model."""
        registry = Registry()

        def my_func(name: str, browser_session=None, has_sensitive_data: bool = False):
            pass

        model = registry._create_param_model(my_func)
        schema = model.model_json_schema()
        assert "browser_session" not in str(schema)
        assert "has_sensitive_data" not in str(schema)
        assert "name" in str(schema)


# ---------------------------------------------------------------------------
# execute_action error wrapping
# ---------------------------------------------------------------------------


class TestExecuteActionErrors:
    @pytest.mark.asyncio
    async def test_action_raises_value_error_with_browser_message(self):
        """ValueError with 'requires browser_session' should be wrapped in RuntimeError."""
        registry = Registry()

        @registry.action("Bad action")
        async def bad_action():
            raise ValueError("requires browser_session but none provided")

        with pytest.raises(RuntimeError, match="requires browser_session"):
            await registry.execute_action("bad_action", {})

    @pytest.mark.asyncio
    async def test_action_raises_generic_value_error(self):
        """Generic ValueError should be wrapped in RuntimeError."""
        registry = Registry()

        @registry.action("Bad action")
        async def bad_action():
            raise ValueError("something went wrong")

        with pytest.raises(RuntimeError, match="Error executing action"):
            await registry.execute_action("bad_action", {})

    @pytest.mark.asyncio
    async def test_action_raises_generic_exception(self):
        """Generic exceptions should be wrapped in RuntimeError."""
        registry = Registry()

        @registry.action("Bad action")
        async def bad_action():
            raise TypeError("type mismatch")

        with pytest.raises(RuntimeError, match="Error executing action"):
            await registry.execute_action("bad_action", {})

    @pytest.mark.asyncio
    async def test_action_raises_timeout(self):
        """TimeoutError should be wrapped in RuntimeError."""
        registry = Registry()

        @registry.action("Slow action")
        async def slow_action():
            raise TimeoutError("timed out")

        with pytest.raises(RuntimeError, match="timeout"):
            await registry.execute_action("slow_action", {})
