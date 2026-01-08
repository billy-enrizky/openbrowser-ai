"""Tests for agent views module.

This module provides comprehensive test coverage for the agent views
subsystem, which defines the core data structures used by the browser
automation agent. It validates:

    - AgentOutput: Model output containing thinking, memory, goals, and actions
    - AgentSettings: Configuration options for agent behavior
    - ActionResult: Results from executing browser actions
    - AgentHistory: History tracking for individual agent steps
    - AgentHistoryList: Collection of agent history entries
    - AgentStepInfo: Step-by-step execution tracking
    - BrowserStateHistory: Browser state snapshots for history

These data structures form the foundation of the agent's decision-making
and action execution pipeline.
"""

import pytest

from src.openbrowser.agent.views import (
    AgentOutput,
    AgentSettings,
    ActionResult,
    AgentHistory,
    AgentHistoryList,
    AgentStepInfo,
    BrowserStateHistory,
)


class TestAgentOutput:
    """Tests for the AgentOutput class.

    Validates the creation and properties of AgentOutput instances,
    including thinking, memory, goals, and action fields.
    """

    def test_agent_output_with_action(self):
        """Test AgentOutput with required action field."""
        output = AgentOutput(
            action=[{"type": "navigate", "url": "https://example.com"}]
        )
        assert output.thinking is None
        assert output.evaluation_previous_goal is None
        assert output.memory is None
        assert output.next_goal is None
        assert len(output.action) == 1

    def test_agent_output_with_values(self):
        """Test AgentOutput with values."""
        output = AgentOutput(
            thinking="Analyzing the page...",
            memory="Found a search button",
            next_goal="Click the search button",
            action=[{"type": "click", "index": 1}],
        )
        assert output.thinking == "Analyzing the page..."
        assert output.memory == "Found a search button"
        assert output.next_goal == "Click the search button"
        assert len(output.action) == 1

    def test_agent_output_current_state(self):
        """Test AgentOutput current_state property."""
        output = AgentOutput(
            thinking="Test thinking",
            evaluation_previous_goal="Previous was good",
            memory="Remembered something",
            next_goal="Do something next",
            action=[{"type": "done"}],
        )
        brain = output.current_state
        assert brain.thinking == "Test thinking"
        assert brain.evaluation_previous_goal == "Previous was good"
        assert brain.memory == "Remembered something"
        assert brain.next_goal == "Do something next"


class TestAgentSettings:
    """Tests for the AgentSettings class.

    Validates default values and custom configuration of agent settings
    including vision mode, failure limits, and flash mode options.
    """

    def test_agent_settings_defaults(self):
        """Test AgentSettings defaults."""
        settings = AgentSettings()
        assert settings.use_vision == "auto"
        assert settings.max_failures == 3
        assert settings.max_actions_per_step == 10  # Default value
        assert settings.use_thinking is True
        assert settings.flash_mode is False

    def test_agent_settings_custom(self):
        """Test AgentSettings with custom values."""
        settings = AgentSettings(
            use_vision=True,
            max_failures=5,
            max_actions_per_step=10,
            flash_mode=True,
        )
        assert settings.use_vision is True
        assert settings.max_failures == 5
        assert settings.max_actions_per_step == 10
        assert settings.flash_mode is True


class TestActionResult:
    """Tests for the ActionResult class.

    Validates action result creation including success/failure states,
    error handling, and the relationship between is_done and success flags.
    """

    def test_action_result_basic(self):
        """Test basic ActionResult."""
        result = ActionResult(
            extracted_content="Page title: Google",
        )
        assert result.is_done is False
        assert result.success is None
        assert result.error is None
        assert result.extracted_content == "Page title: Google"

    def test_action_result_error(self):
        """Test ActionResult with error."""
        result = ActionResult(
            error="Element not found",
        )
        assert result.error == "Element not found"

    def test_action_result_done_success(self):
        """Test ActionResult with is_done and success."""
        # success=True requires is_done=True
        result = ActionResult(
            is_done=True,
            success=True,
            extracted_content="Task completed",
        )
        assert result.is_done is True
        assert result.success is True

    def test_action_result_done_failure(self):
        """Test ActionResult with is_done but failure."""
        result = ActionResult(
            is_done=True,
            success=False,
            error="Could not complete task",
        )
        assert result.is_done is True
        assert result.success is False

    def test_action_result_success_requires_done(self):
        """Test that success=True requires is_done=True."""
        with pytest.raises(ValueError, match="success=True can only be set when is_done=True"):
            ActionResult(
                is_done=False,
                success=True,
            )


class TestAgentHistory:
    """Tests for the AgentHistory class.

    Validates agent history creation, serialization, and the relationship
    between model output, action results, and browser state.
    """

    def test_agent_history_creation(self):
        """Test AgentHistory creation."""
        output = AgentOutput(
            thinking="Analysis",
            memory="Memory",
            next_goal="Goal",
            action=[{"type": "click", "index": 1}],
        )
        result = ActionResult(extracted_content="Clicked element")
        state = BrowserStateHistory(url="https://example.com", title="Example")
        
        history = AgentHistory(
            model_output=output,
            result=[result],
            state=state,
        )
        
        assert history.model_output == output
        assert len(history.result) == 1
        assert history.state.url == "https://example.com"

    def test_agent_history_model_dump(self):
        """Test AgentHistory serialization."""
        output = AgentOutput(
            thinking="Analysis",
            memory="Memory",
            next_goal="Goal",
            action=[{"type": "done"}],
        )
        result = ActionResult(is_done=True, success=True)
        state = BrowserStateHistory(url="https://example.com", title="Example")
        
        history = AgentHistory(
            model_output=output,
            result=[result],
            state=state,
        )
        
        dump = history.model_dump()
        assert "model_output" in dump
        assert "result" in dump
        assert "state" in dump


class TestAgentHistoryList:
    """Tests for the AgentHistoryList class.

    Validates the history list collection, item addition, and the is_done
    detection for determining when the agent has completed its task.
    """

    def test_agent_history_list_empty(self):
        """Test empty AgentHistoryList."""
        history_list = AgentHistoryList()
        assert history_list.history == []
        assert len(history_list) == 0

    def test_agent_history_list_add_item(self):
        """Test adding items to AgentHistoryList."""
        history_list = AgentHistoryList()
        
        output = AgentOutput(action=[{"type": "navigate"}])
        result = ActionResult(extracted_content="Navigated")
        state = BrowserStateHistory(url="https://google.com")
        
        item = AgentHistory(
            model_output=output,
            result=[result],
            state=state,
        )
        
        history_list.add_item(item)
        assert len(history_list) == 1
        assert history_list.history[0].state.url == "https://google.com"

    def test_agent_history_list_is_done(self):
        """Test is_done method."""
        history_list = AgentHistoryList()
        
        # Not done initially
        assert history_list.is_done() is False
        
        # Add a done item
        output = AgentOutput(action=[{"type": "done"}])
        result = ActionResult(is_done=True, success=True)
        state = BrowserStateHistory()
        
        item = AgentHistory(
            model_output=output,
            result=[result],
            state=state,
        )
        
        history_list.add_item(item)
        assert history_list.is_done() is True


class TestAgentStepInfo:
    """Tests for the AgentStepInfo class.

    Validates step tracking functionality including step number, max steps,
    and is_last_step detection for execution flow control.
    """

    def test_agent_step_info(self):
        """Test AgentStepInfo."""
        step_info = AgentStepInfo(step_number=5, max_steps=10)
        assert step_info.step_number == 5
        assert step_info.max_steps == 10

    def test_is_last_step(self):
        """Test is_last_step method."""
        step_info = AgentStepInfo(step_number=9, max_steps=10)
        assert step_info.is_last_step() is True
        
        step_info2 = AgentStepInfo(step_number=5, max_steps=10)
        assert step_info2.is_last_step() is False


class TestBrowserStateHistory:
    """Tests for the BrowserStateHistory class.

    Validates browser state snapshot creation and serialization including
    URL, title, and optional screenshot data.
    """

    def test_browser_state_history_creation(self):
        """Test BrowserStateHistory creation."""
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example Domain",
        )
        assert state.url == "https://example.com"
        assert state.title == "Example Domain"
        assert state.screenshot is None

    def test_browser_state_history_to_dict(self):
        """Test to_dict method."""
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
        )
        d = state.to_dict()
        assert d["url"] == "https://example.com"
        assert d["title"] == "Example"
        assert "screenshot" not in d  # None values are excluded
