"""Tests for the tools registry module.

This module provides comprehensive test coverage for the tools registry
system, which manages registration and discovery of browser actions.
It validates:

    - Registry: Action registration, exclusion, and prompt generation
    - ActionModel: Base model for action parameters with validation
    - Action decorator for registering async action handlers
    - Exclusion of specific actions from registration
    - Prompt description generation for LLM consumption

The tools registry enables dynamic registration of browser actions
and automatic generation of tool descriptions for LLM agents.
"""

import pytest
from pydantic import Field

from src.openbrowser.tools.registry import Registry, ActionModel


class SampleActionParams(ActionModel):
    """Sample action parameters for testing.

    A test model demonstrating action parameter definition with
    required and optional fields.

    Attributes:
        text: Required test text parameter.
        number: Optional test number parameter with default value.
    """
    text: str = Field(description="Test text")
    number: int = Field(default=0, description="Test number")


class TestRegistry:
    """Tests for the Registry class.

    Validates action registration, exclusion, retrieval, and prompt
    description generation for the tools registry system.
    """

    def test_registry_init(self):
        """Test registry initialization."""
        registry = Registry()
        assert registry.registry.actions == {}
        assert registry.exclude_actions == []

    def test_registry_init_with_exclusions(self):
        """Test registry initialization with exclusions."""
        registry = Registry(exclude_actions=["action1", "action2"])
        assert registry.exclude_actions == ["action1", "action2"]

    def test_action_decorator(self):
        """Test action decorator registration."""
        registry = Registry()

        @registry.action("Test action", param_model=SampleActionParams)
        async def test_action(params: SampleActionParams):
            return f"Result: {params.text}"

        assert "test_action" in registry.registry.actions
        assert registry.registry.actions["test_action"].description == "Test action"

    def test_get_action_registered(self):
        """Test that action is registered."""
        registry = Registry()

        @registry.action("Test action", param_model=SampleActionParams)
        async def test_action(params: SampleActionParams):
            return f"Result: {params.text}"

        assert "test_action" in registry.registry.actions
        action = registry.registry.actions["test_action"]
        assert action.name == "test_action"

    def test_registry_empty_initially(self):
        """Test registry is empty initially."""
        registry = Registry()
        assert len(registry.registry.actions) == 0

    def test_excluded_action_not_registered(self):
        """Test that excluded actions are not registered."""
        registry = Registry(exclude_actions=["excluded_action"])

        @registry.action("Excluded action", param_model=SampleActionParams)
        async def excluded_action(params: SampleActionParams):
            return "Should not be registered"

        assert "excluded_action" not in registry.registry.actions

    def test_get_prompt_description(self):
        """Test getting prompt description for actions."""
        registry = Registry()

        @registry.action("Test action for prompt", param_model=SampleActionParams)
        async def test_action(params: SampleActionParams):
            return "Result"

        description = registry.get_prompt_description()
        assert "test_action" in description
        assert "Test action for prompt" in description


class TestActionModel:
    """Tests for the ActionModel base class.

    Validates action model creation with required and optional fields,
    demonstrating the base class behavior for action parameters.
    """

    def test_action_model_creation(self):
        """Test ActionModel creation."""
        # ActionModel is a base class, action_name is added by registry
        model = SampleActionParams(text="hello")
        assert model.text == "hello"
        assert model.number == 0

    def test_action_model_with_number(self):
        """Test ActionModel with custom number."""
        model = SampleActionParams(text="hello", number=42)
        assert model.text == "hello"
        assert model.number == 42

