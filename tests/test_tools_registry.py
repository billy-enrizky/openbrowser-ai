"""Tests for openbrowser.tools.registry (service.py and views.py)."""

import logging
from typing import Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from openbrowser.tools.registry.service import Registry
from openbrowser.tools.registry.views import (
    ActionModel,
    ActionRegistry,
    RegisteredAction,
    SpecialActionParameters,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# views.py tests
# ---------------------------------------------------------------------------


class TestRegisteredAction:
    """Tests for RegisteredAction model."""

    def test_prompt_description_with_params(self):
        """prompt_description should include param names and types."""

        class MyParams(ActionModel):
            url: str
            index: int

        action = RegisteredAction(
            name="navigate",
            description="Go to URL",
            function=lambda: None,
            param_model=MyParams,
        )
        desc = action.prompt_description()
        assert "navigate:" in desc
        assert "Go to URL" in desc
        assert "url" in desc
        assert "index" in desc

    def test_prompt_description_no_params(self):
        """prompt_description without params should just show name and description."""

        class EmptyParams(ActionModel):
            pass

        action = RegisteredAction(
            name="go_back",
            description="Navigate back",
            function=lambda: None,
            param_model=EmptyParams,
        )
        desc = action.prompt_description()
        assert "go_back: Navigate back" == desc

    def test_prompt_description_with_param_descriptions(self):
        """param descriptions should be shown in parentheses."""
        from pydantic import Field

        class DescParams(ActionModel):
            query: str = Field(description="Search query text")

        action = RegisteredAction(
            name="search",
            description="Search the web",
            function=lambda: None,
            param_model=DescParams,
        )
        desc = action.prompt_description()
        assert "Search query text" in desc


class TestActionModel:
    """Tests for ActionModel base class."""

    def test_get_index_returns_none_when_empty(self):
        model = ActionModel()
        assert model.get_index() is None

    def test_get_index_extracts_index_from_params(self):
        from pydantic import create_model

        ClickModel = create_model(
            "ClickModel",
            __base__=ActionModel,
            click=(dict, {"index": 5}),
        )
        m = ClickModel(click={"index": 5})
        assert m.get_index() == 5

    def test_get_index_returns_none_when_no_index_in_params(self):
        from pydantic import create_model

        NavModel = create_model(
            "NavModel",
            __base__=ActionModel,
            navigate=(dict, {"url": "http://example.com"}),
        )
        m = NavModel(navigate={"url": "http://example.com"})
        assert m.get_index() is None

    def test_set_index_updates_index(self):
        """set_index should update the index on the nested action params."""

        class ClickParams(BaseModel):
            index: int

        from pydantic import create_model

        ClickActionModel = create_model(
            "ClickActionModel",
            __base__=ActionModel,
            click=(Optional[ClickParams], None),
        )

        m = ClickActionModel(click=ClickParams(index=3))
        m.set_index(10)
        assert m.click.index == 10


class TestActionRegistry:
    """Tests for ActionRegistry."""

    def test_match_domains_none_returns_true(self):
        assert ActionRegistry._match_domains(None, "https://example.com") is True

    def test_match_domains_empty_url_returns_true(self):
        assert ActionRegistry._match_domains(["*.example.com"], "") is True

    def test_match_domains_matching(self):
        result = ActionRegistry._match_domains(
            ["*.example.com"], "https://sub.example.com/page"
        )
        assert result is True

    def test_match_domains_no_match(self):
        result = ActionRegistry._match_domains(
            ["*.google.com"], "https://example.com"
        )
        assert result is False

    def test_get_prompt_description_no_url_filters_unfiltered(self):
        """Without page_url, only actions with no domain filter should be included."""
        reg = ActionRegistry()

        class P(ActionModel):
            pass

        reg.actions["global_action"] = RegisteredAction(
            name="global_action",
            description="Available everywhere",
            function=lambda: None,
            param_model=P,
            domains=None,
        )
        reg.actions["domain_action"] = RegisteredAction(
            name="domain_action",
            description="Only on google",
            function=lambda: None,
            param_model=P,
            domains=["*.google.com"],
        )

        desc = reg.get_prompt_description(page_url=None)
        assert "global_action" in desc
        assert "domain_action" not in desc

    def test_get_prompt_description_with_url_filters_by_domain(self):
        """With page_url, only domain-filtered actions matching URL are included."""
        reg = ActionRegistry()

        class P(ActionModel):
            pass

        reg.actions["global_action"] = RegisteredAction(
            name="global_action",
            description="Available everywhere",
            function=lambda: None,
            param_model=P,
            domains=None,
        )
        reg.actions["google_action"] = RegisteredAction(
            name="google_action",
            description="Google only",
            function=lambda: None,
            param_model=P,
            domains=["*.google.com"],
        )
        reg.actions["bing_action"] = RegisteredAction(
            name="bing_action",
            description="Bing only",
            function=lambda: None,
            param_model=P,
            domains=["*.bing.com"],
        )

        desc = reg.get_prompt_description(page_url="https://www.google.com/search?q=test")
        # global_action should NOT appear (no domains filter means it's in system prompt)
        assert "global_action" not in desc
        # google_action should appear
        assert "google_action" in desc
        # bing_action should NOT appear
        assert "bing_action" not in desc


class TestSpecialActionParameters:
    """Tests for SpecialActionParameters."""

    def test_get_browser_requiring_params(self):
        params = SpecialActionParameters.get_browser_requiring_params()
        assert "browser_session" in params
        assert "cdp_client" in params
        assert "page_url" in params

    def test_default_values(self):
        sap = SpecialActionParameters()
        assert sap.context is None
        assert sap.browser_session is None
        assert sap.page_url is None
        assert sap.cdp_client is None
        assert sap.page_extraction_llm is None
        assert sap.file_system is None
        assert sap.available_file_paths is None
        assert sap.has_sensitive_data is False


# ---------------------------------------------------------------------------
# service.py tests
# ---------------------------------------------------------------------------


class TestRegistryAction:
    """Tests for Registry.action decorator."""

    def test_register_basic_async_action(self):
        registry = Registry()

        @registry.action("Test action")
        async def my_action(name: str):
            return name

        assert "my_action" in registry.registry.actions
        assert registry.registry.actions["my_action"].description == "Test action"

    def test_register_action_with_param_model(self):
        registry = Registry()

        class MyParams(ActionModel):
            url: str

        @registry.action("Navigate action", param_model=MyParams)
        async def navigate(params: MyParams):
            return params.url

        assert "navigate" in registry.registry.actions
        assert registry.registry.actions["navigate"].param_model is MyParams

    def test_register_action_with_domains(self):
        registry = Registry()

        @registry.action("Google only action", domains=["*.google.com"])
        async def google_action(query: str):
            return query

        assert registry.registry.actions["google_action"].domains == ["*.google.com"]

    def test_register_action_with_allowed_domains_alias(self):
        registry = Registry()

        @registry.action("Bing only action", allowed_domains=["*.bing.com"])
        async def bing_action(query: str):
            return query

        assert registry.registry.actions["bing_action"].domains == ["*.bing.com"]

    def test_domains_and_allowed_domains_conflict_raises(self):
        registry = Registry()
        with pytest.raises(ValueError, match="Cannot specify both"):
            @registry.action(
                "Conflicting",
                domains=["*.google.com"],
                allowed_domains=["*.bing.com"],
            )
            async def bad_action(query: str):
                return query

    def test_exclude_actions_skips_registration(self):
        registry = Registry(exclude_actions=["skip_me"])

        @registry.action("Should be skipped")
        async def skip_me(x: int):
            return x

        assert "skip_me" not in registry.registry.actions

    def test_kwargs_not_allowed_raises(self):
        registry = Registry()
        with pytest.raises(ValueError, match="not allowed"):

            @registry.action("Bad kwargs action")
            async def bad_action(**kwargs):
                pass

    def test_wrong_special_param_type_raises(self):
        registry = Registry()
        with pytest.raises(ValueError, match="conflicts with special argument"):

            @registry.action("Wrong type for browser_session")
            async def bad_types(index: int, browser_session: int = 0):
                pass


class TestRegistryExecuteAction:
    """Tests for Registry.execute_action."""

    @pytest.fixture
    def registry_with_action(self):
        registry = Registry()

        @registry.action("Simple action")
        async def simple(name: str):
            return f"Hello {name}"

        return registry

    @pytest.mark.asyncio
    async def test_execute_unknown_action_raises(self, registry_with_action):
        with pytest.raises(ValueError, match="not found"):
            await registry_with_action.execute_action("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_with_invalid_params_raises(self, registry_with_action):
        with pytest.raises(RuntimeError, match="Invalid parameters"):
            await registry_with_action.execute_action("simple", {"wrong_param": 123})

    @pytest.mark.asyncio
    async def test_execute_action_success(self, registry_with_action):
        result = await registry_with_action.execute_action("simple", {"name": "World"})
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_execute_action_timeout_raises(self):
        registry = Registry()

        @registry.action("Timeout action")
        async def slow():
            import asyncio
            await asyncio.sleep(100)

        # Patch time_execution_async to not interfere
        with pytest.raises(RuntimeError, match="timeout"):
            # We simulate timeout by making the function raise TimeoutError
            registry.registry.actions["slow"].function = AsyncMock(
                side_effect=TimeoutError("timed out")
            )
            await registry.execute_action("slow", {})


class TestRegistryReplaceSensitiveData:
    """Tests for Registry._replace_sensitive_data."""

    def _make_registry(self):
        return Registry()

    def test_replace_old_format(self):
        registry = self._make_registry()

        class Params(BaseModel):
            text: str

        params = Params(text="Login with <secret>my_password</secret>")
        sensitive_data = {"my_password": "hunter2"}

        result = registry._replace_sensitive_data(params, sensitive_data)
        assert result.text == "Login with hunter2"

    def test_replace_new_format_matching_url(self):
        registry = self._make_registry()

        class Params(BaseModel):
            text: str

        params = Params(text="<secret>user_pass</secret>")
        sensitive_data = {"https://*.example.com": {"user_pass": "secret123"}}

        result = registry._replace_sensitive_data(
            params, sensitive_data, current_url="https://www.example.com/login"
        )
        assert result.text == "secret123"

    def test_missing_placeholder_kept_as_is(self):
        registry = self._make_registry()

        class Params(BaseModel):
            text: str

        params = Params(text="<secret>unknown_key</secret>")
        sensitive_data = {"known_key": "value"}

        result = registry._replace_sensitive_data(params, sensitive_data)
        assert "<secret>unknown_key</secret>" in result.text

    def test_replace_nested_dict(self):
        registry = self._make_registry()

        class Inner(BaseModel):
            value: str

        class Params(BaseModel):
            inner: Inner

        params = Params(inner=Inner(value="<secret>api_key</secret>"))
        sensitive_data = {"api_key": "sk-12345"}

        result = registry._replace_sensitive_data(params, sensitive_data)
        assert result.inner.value == "sk-12345"

    def test_replace_list_values(self):
        registry = self._make_registry()

        class Params(BaseModel):
            items: list[str]

        params = Params(items=["<secret>a</secret>", "plain"])
        sensitive_data = {"a": "replaced"}

        result = registry._replace_sensitive_data(params, sensitive_data)
        assert result.items[0] == "replaced"
        assert result.items[1] == "plain"

    def test_empty_sensitive_values_filtered(self):
        registry = self._make_registry()

        class Params(BaseModel):
            text: str

        params = Params(text="<secret>empty_key</secret>")
        sensitive_data = {"empty_key": ""}

        result = registry._replace_sensitive_data(params, sensitive_data)
        # Empty value means the key won't be in applicable_secrets
        assert "<secret>empty_key</secret>" in result.text


class TestRegistryCreateActionModel:
    """Tests for Registry.create_action_model."""

    def test_empty_registry_returns_empty_model(self):
        registry = Registry()
        model = registry.create_action_model()
        assert model is not None

    def test_single_action_no_union(self):
        registry = Registry()

        @registry.action("Only action")
        async def solo(name: str):
            return name

        model = registry.create_action_model()
        assert model is not None

    def test_multiple_actions_create_union(self):
        registry = Registry()

        @registry.action("Action A")
        async def action_a(x: int):
            return x

        @registry.action("Action B")
        async def action_b(y: str):
            return y

        model = registry.create_action_model()
        assert model is not None

    def test_include_actions_filter(self):
        registry = Registry()

        @registry.action("Action A")
        async def action_a(x: int):
            return x

        @registry.action("Action B")
        async def action_b(y: str):
            return y

        model = registry.create_action_model(include_actions=["action_a"])
        # The model should only contain action_a
        schema = model.model_json_schema()
        assert "action_a" in str(schema)

    def test_page_url_filters_domain_actions(self):
        registry = Registry()

        @registry.action("Global")
        async def global_action(x: int):
            return x

        @registry.action("Google", domains=["*.google.com"])
        async def google_action(q: str):
            return q

        # With google URL, both global (no domains) and google should appear
        model = registry.create_action_model(page_url="https://www.google.com")
        schema = str(model.model_json_schema())
        assert "google_action" in schema

    def test_page_url_none_excludes_domain_actions(self):
        registry = Registry()

        @registry.action("Global")
        async def global_action(x: int):
            return x

        @registry.action("Google", domains=["*.google.com"])
        async def google_action(q: str):
            return q

        model = registry.create_action_model(page_url=None)
        schema = str(model.model_json_schema())
        assert "google_action" not in schema


class TestRegistryGetPromptDescription:
    """Tests for Registry.get_prompt_description."""

    def test_delegates_to_action_registry(self):
        registry = Registry()

        @registry.action("Test action")
        async def test_action(x: int):
            return x

        desc = registry.get_prompt_description()
        assert "test_action" in desc

    def test_with_page_url(self):
        registry = Registry()

        @registry.action("Domain action", domains=["*.example.com"])
        async def domain_action(x: int):
            return x

        desc = registry.get_prompt_description(page_url="https://www.example.com")
        assert "domain_action" in desc
