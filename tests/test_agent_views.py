"""Tests for openbrowser.agent.views module."""

import json
import logging
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model

from openbrowser.agent.views import (
    AgentBrain,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentSettings,
    AgentState,
    AgentStepInfo,
    BrowserStateHistory,
    StepMetadata,
)
from openbrowser.models import ActionResult
from openbrowser.tools.registry.views import ActionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentSettings tests
# ---------------------------------------------------------------------------


class TestAgentSettings:
    def test_default_values(self):
        settings = AgentSettings()
        assert settings.use_vision == "auto"
        assert settings.max_failures == 3
        assert settings.max_actions_per_step == 4
        assert settings.use_thinking is True
        assert settings.flash_mode is False
        assert settings.calculate_cost is False
        assert settings.llm_timeout == 60

    def test_custom_values(self):
        settings = AgentSettings(
            use_vision=True,
            max_failures=5,
            max_actions_per_step=8,
        )
        assert settings.use_vision is True
        assert settings.max_failures == 5
        assert settings.max_actions_per_step == 8


# ---------------------------------------------------------------------------
# AgentState tests
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_default_values(self):
        state = AgentState()
        assert state.n_steps == 1
        assert state.consecutive_failures == 0
        assert state.last_result is None
        assert state.paused is False
        assert state.stopped is False
        assert state.session_initialized is False

    def test_custom_values(self):
        state = AgentState(n_steps=5, consecutive_failures=2)
        assert state.n_steps == 5
        assert state.consecutive_failures == 2


# ---------------------------------------------------------------------------
# AgentStepInfo tests
# ---------------------------------------------------------------------------


class TestAgentStepInfo:
    def test_is_last_step_true(self):
        info = AgentStepInfo(step_number=9, max_steps=10)
        assert info.is_last_step() is True

    def test_is_last_step_false(self):
        info = AgentStepInfo(step_number=5, max_steps=10)
        assert info.is_last_step() is False

    def test_is_last_step_at_boundary(self):
        info = AgentStepInfo(step_number=10, max_steps=10)
        assert info.is_last_step() is True

    def test_first_step_is_not_last(self):
        info = AgentStepInfo(step_number=0, max_steps=10)
        assert info.is_last_step() is False


# ---------------------------------------------------------------------------
# StepMetadata tests
# ---------------------------------------------------------------------------


class TestStepMetadata:
    def test_duration_seconds(self):
        meta = StepMetadata(step_start_time=100.0, step_end_time=105.5, step_number=1)
        assert meta.duration_seconds == 5.5

    def test_zero_duration(self):
        meta = StepMetadata(step_start_time=100.0, step_end_time=100.0, step_number=0)
        assert meta.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# AgentOutput tests
# ---------------------------------------------------------------------------


class TestAgentOutput:
    def _make_action_model(self):
        return create_model(
            "TestAction",
            __base__=ActionModel,
            done=(Optional[dict], None),
        )

    def test_current_state_property(self):
        ActionType = self._make_action_model()
        output = AgentOutput(
            thinking="I am thinking",
            evaluation_previous_goal="Goal met",
            memory="Remembered state",
            next_goal="Navigate to page",
            action=[ActionType(done={"text": "done"})],
        )
        brain = output.current_state
        assert isinstance(brain, AgentBrain)
        assert brain.thinking == "I am thinking"
        assert brain.memory == "Remembered state"

    def test_model_json_schema_has_required_fields(self):
        schema = AgentOutput.model_json_schema()
        assert "required" in schema
        assert "evaluation_previous_goal" in schema["required"]
        assert "memory" in schema["required"]
        assert "next_goal" in schema["required"]
        assert "action" in schema["required"]

    def test_type_with_custom_actions(self):
        CustomAction = create_model(
            "CustomAction",
            __base__=ActionModel,
            custom_field=(Optional[str], None),
        )
        ModelClass = AgentOutput.type_with_custom_actions(CustomAction)
        assert ModelClass is not None

    def test_type_with_custom_actions_no_thinking(self):
        CustomAction = create_model(
            "CustomAction",
            __base__=ActionModel,
            custom_field=(Optional[str], None),
        )
        ModelClass = AgentOutput.type_with_custom_actions_no_thinking(CustomAction)
        schema = ModelClass.model_json_schema()
        assert "thinking" not in schema.get("properties", {})

    def test_type_with_custom_actions_flash_mode(self):
        CustomAction = create_model(
            "CustomAction",
            __base__=ActionModel,
            custom_field=(Optional[str], None),
        )
        ModelClass = AgentOutput.type_with_custom_actions_flash_mode(CustomAction)
        schema = ModelClass.model_json_schema()
        assert "thinking" not in schema.get("properties", {})
        assert "evaluation_previous_goal" not in schema.get("properties", {})
        assert "next_goal" not in schema.get("properties", {})
        assert "memory" in schema["required"]
        assert "action" in schema["required"]


# ---------------------------------------------------------------------------
# AgentError tests
# ---------------------------------------------------------------------------


class TestAgentError:
    def test_format_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            AgentSettings(max_failures="not_a_number")
        result = AgentError.format_error(exc_info.value)
        assert AgentError.VALIDATION_ERROR in result

    def test_format_rate_limit_error(self):
        from unittest.mock import MagicMock

        error = MagicMock(spec=Exception)
        error.__class__ = type("RateLimitError", (Exception,), {})

        # Use the actual RateLimitError import
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit"}}
        mock_response.headers = {}

        try:
            raise RateLimitError(
                message="Rate limit",
                response=mock_response,
                body={"error": {"message": "Rate limit"}},
            )
        except RateLimitError as e:
            result = AgentError.format_error(e)
            assert result == AgentError.RATE_LIMIT_ERROR

    def test_format_generic_error(self):
        result = AgentError.format_error(RuntimeError("Something went wrong"))
        assert "Something went wrong" in result

    def test_format_error_with_trace(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = AgentError.format_error(e, include_trace=True)
            assert "test error" in result
            assert "Stacktrace:" in result

    def test_format_llm_response_error(self):
        error = RuntimeError("LLM response missing required fields: action")
        result = AgentError.format_error(error)
        assert "invalid output structure" in result


# ---------------------------------------------------------------------------
# AgentHistory tests
# ---------------------------------------------------------------------------


class TestAgentHistory:
    def _make_history(self, model_output=None, results=None, sensitive_data=None):
        if results is None:
            results = [ActionResult(extracted_content="test result")]

        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )

        return AgentHistory(
            model_output=model_output,
            result=results,
            state=state,
        )

    def test_filter_sensitive_data_from_string(self):
        h = self._make_history()
        filtered = h._filter_sensitive_data_from_string(
            "My password is hunter2",
            {"password": "hunter2"},
        )
        assert "hunter2" not in filtered
        assert "<secret>password</secret>" in filtered

    def test_filter_sensitive_data_empty(self):
        h = self._make_history()
        result = h._filter_sensitive_data_from_string("Hello world", None)
        assert result == "Hello world"

    def test_filter_sensitive_data_new_format(self):
        h = self._make_history()
        sensitive = {"*.example.com": {"api_key": "sk-12345"}}
        filtered = h._filter_sensitive_data_from_string("Key: sk-12345", sensitive)
        assert "sk-12345" not in filtered

    def test_filter_sensitive_data_from_dict(self):
        h = self._make_history()
        data = {"text": "password is hunter2", "nested": {"value": "hunter2 is here"}}
        sensitive = {"password": "hunter2"}
        filtered = h._filter_sensitive_data_from_dict(data, sensitive)
        assert "hunter2" not in str(filtered)

    def test_model_dump_without_model_output(self):
        h = self._make_history(model_output=None)
        dump = h.model_dump()
        assert dump["model_output"] is None
        assert len(dump["result"]) == 1


# ---------------------------------------------------------------------------
# AgentHistoryList tests
# ---------------------------------------------------------------------------


class TestAgentHistoryList:
    def _make_list(self, n=3, is_done_last=False):
        histories = []
        for i in range(n):
            is_last = i == n - 1
            results = [
                ActionResult(
                    extracted_content=f"Result {i}",
                    is_done=True if (is_last and is_done_last) else False,
                    success=True if (is_last and is_done_last) else None,
                )
            ]
            state = BrowserStateHistory(
                url=f"https://example.com/page{i}",
                title=f"Page {i}",
                tabs=[],
                interacted_element=[],
                screenshot_path=None,
            )
            histories.append(
                AgentHistory(
                    model_output=None,
                    result=results,
                    state=state,
                    metadata=StepMetadata(
                        step_start_time=float(i),
                        step_end_time=float(i) + 1.0,
                        step_number=i,
                    ),
                )
            )
        return AgentHistoryList(history=histories)

    def test_len(self):
        hl = self._make_list(n=5)
        assert len(hl) == 5

    def test_total_duration(self):
        hl = self._make_list(n=3)
        assert hl.total_duration_seconds() == 3.0

    def test_add_item(self):
        hl = self._make_list(n=1)
        assert len(hl) == 1
        state = BrowserStateHistory(
            url="https://new.com",
            title="New",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        hl.add_item(
            AgentHistory(
                model_output=None,
                result=[ActionResult()],
                state=state,
            )
        )
        assert len(hl) == 2

    def test_final_result_none_when_no_content(self):
        hl = self._make_list(n=1)
        # Result has extracted_content but not in the "final result" sense
        # final_result checks last history item's last result's extracted_content
        result = hl.final_result()
        assert result == "Result 0"

    def test_is_done_true(self):
        hl = self._make_list(n=2, is_done_last=True)
        assert hl.is_done() is True

    def test_is_done_false(self):
        hl = self._make_list(n=2, is_done_last=False)
        assert hl.is_done() is False

    def test_is_successful_true(self):
        hl = self._make_list(n=2, is_done_last=True)
        assert hl.is_successful() is True

    def test_is_successful_none_when_not_done(self):
        hl = self._make_list(n=2, is_done_last=False)
        assert hl.is_successful() is None

    def test_has_errors_false(self):
        hl = self._make_list(n=2)
        assert hl.has_errors() is False

    def test_errors_list(self):
        hl = self._make_list(n=2)
        errors = hl.errors()
        assert len(errors) == 2
        assert all(e is None for e in errors)

    def test_urls(self):
        hl = self._make_list(n=3)
        urls = hl.urls()
        assert urls == [
            "https://example.com/page0",
            "https://example.com/page1",
            "https://example.com/page2",
        ]

    def test_number_of_steps(self):
        hl = self._make_list(n=5)
        assert hl.number_of_steps() == 5

    def test_action_results(self):
        hl = self._make_list(n=3)
        results = hl.action_results()
        assert len(results) == 3

    def test_extracted_content(self):
        hl = self._make_list(n=3)
        content = hl.extracted_content()
        assert len(content) == 3
        assert content[0] == "Result 0"

    def test_str_representation(self):
        hl = self._make_list(n=1)
        s = str(hl)
        assert "AgentHistoryList" in s

    def test_repr_same_as_str(self):
        hl = self._make_list(n=1)
        assert repr(hl) == str(hl)

    def test_last_action_none_when_no_output(self):
        hl = self._make_list(n=1)
        assert hl.last_action() is None

    def test_screenshot_paths_empty_n(self):
        hl = self._make_list(n=3)
        assert hl.screenshot_paths(n_last=0) == []

    def test_screenshots_empty_n(self):
        hl = self._make_list(n=3)
        assert hl.screenshots(n_last=0) == []

    def test_save_to_file(self, tmp_path):
        hl = self._make_list(n=2)
        filepath = tmp_path / "history.json"
        hl.save_to_file(str(filepath))
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert "history" in data

    def test_model_dump(self):
        hl = self._make_list(n=2)
        dump = hl.model_dump()
        assert "history" in dump
        assert len(dump["history"]) == 2

    def test_model_actions_filtered(self):
        hl = self._make_list(n=2)
        # No model outputs means no actions
        result = hl.model_actions_filtered(include=["click"])
        assert result == []

    def test_structured_output_none_when_no_schema(self):
        hl = self._make_list(n=1)
        assert hl.structured_output is None
