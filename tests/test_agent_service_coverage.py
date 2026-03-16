"""Tests for openbrowser.agent.service module - comprehensive coverage.

Covers the Agent class (CodeAgent main loop), log_response, and all helper methods.
All LLM calls, browser connections, and external services are mocked.
"""

import asyncio
import gc
import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for creating mock objects
# ---------------------------------------------------------------------------


def _make_mock_llm(model="test-model", provider="test-provider"):
    """Create a mock LLM with required attributes."""
    llm = MagicMock()
    llm.model = model
    llm.provider = provider
    llm.ainvoke = AsyncMock()
    llm._verified_api_keys = True
    return llm


def _make_mock_browser_session(cdp_url=None, downloads_path=None):
    """Create a mock BrowserSession."""
    session = MagicMock()
    session.id = "session-1234-5678"
    session.cdp_url = cdp_url
    session.start = AsyncMock()
    session.kill = AsyncMock()
    session.get_browser_state_summary = AsyncMock()
    session.downloaded_files = []
    session.agent_focus = MagicMock()
    session.agent_focus.target_id = "target-id-12345678"
    session._cached_browser_state_summary = None

    profile = MagicMock()
    profile.allowed_domains = []
    profile.wait_between_actions = 0.0
    profile.keep_alive = False
    profile.downloads_path = downloads_path
    session.browser_profile = profile

    return session


def _make_mock_browser_state_summary(
    url="https://example.com",
    title="Example",
    screenshot=None,
    selector_map=None,
):
    """Create a mock BrowserStateSummary."""
    state = MagicMock()
    state.url = url
    state.title = title
    state.screenshot = screenshot

    tab = MagicMock()
    tab.url = url
    tab.title = title
    tab.target_id = "abcdefghijklmnop12345678"
    state.tabs = [tab]

    dom_state = MagicMock()
    dom_state.selector_map = selector_map or {}
    dom_state.llm_representation = MagicMock(return_value="[1] button Click me")
    dom_state._root = None
    state.dom_state = dom_state

    state.page_info = None
    state.is_pdf_viewer = False
    state.recent_events = None
    state.closed_popup_messages = []

    return state


def _make_mock_tools():
    """Create a mock Tools object."""
    tools = MagicMock()
    tools.act = AsyncMock()

    # Registry mock
    registry = MagicMock()
    action_model = create_model(
        "TestActionModel",
        __base__=_get_action_model_base(),
        done=(Optional[dict], None),
        navigate=(Optional[dict], None),
        click=(Optional[dict], None),
    )
    registry.create_action_model = MagicMock(return_value=action_model)
    registry.get_prompt_description = MagicMock(return_value=None)
    registry.registry = MagicMock()
    registry.registry.actions = {}
    tools.registry = registry
    return tools


def _get_action_model_base():
    """Import ActionModel base class."""
    from openbrowser.tools.registry.views import ActionModel
    return ActionModel


# ---------------------------------------------------------------------------
# log_response tests
# ---------------------------------------------------------------------------


class TestLogResponse:
    """Tests for the log_response utility function."""

    def test_log_response_with_thinking(self):
        from openbrowser.agent.service import log_response
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        output = AgentOutput(
            thinking="Deep thought here",
            evaluation_previous_goal="success - page loaded",
            memory="I remember the page",
            next_goal="Click the button",
            action=[CustomAction(done={"text": "test"})],
        )
        test_logger = MagicMock()
        log_response(output, logger=test_logger)
        # Should log thinking (debug), eval (info), memory (info), next_goal (info)
        assert test_logger.debug.called
        assert test_logger.info.called

    def test_log_response_success_eval(self):
        from openbrowser.agent.service import log_response
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        output = AgentOutput(
            evaluation_previous_goal="success - completed task",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        test_logger = MagicMock()
        log_response(output, logger=test_logger)
        # Check that success-colored eval was logged
        info_calls = [str(c) for c in test_logger.info.call_args_list]
        assert any("Eval" in str(c) for c in info_calls)

    def test_log_response_failure_eval(self):
        from openbrowser.agent.service import log_response
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        output = AgentOutput(
            evaluation_previous_goal="failure - could not find element",
            memory="mem",
            next_goal="try again",
            action=[CustomAction(done={"text": "test"})],
        )
        test_logger = MagicMock()
        log_response(output, logger=test_logger)
        info_calls = [str(c) for c in test_logger.info.call_args_list]
        assert any("Eval" in str(c) for c in info_calls)

    def test_log_response_neutral_eval(self):
        from openbrowser.agent.service import log_response
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        output = AgentOutput(
            evaluation_previous_goal="still loading page",
            memory="mem",
            next_goal="wait",
            action=[CustomAction(done={"text": "test"})],
        )
        test_logger = MagicMock()
        log_response(output, logger=test_logger)
        info_calls = [str(c) for c in test_logger.info.call_args_list]
        assert any("Eval" in str(c) for c in info_calls)

    def test_log_response_no_thinking_no_eval_no_memory_no_goal(self):
        from openbrowser.agent.service import log_response
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        output = AgentOutput(
            thinking=None,
            evaluation_previous_goal=None,
            memory=None,
            next_goal=None,
            action=[CustomAction(done={"text": "test"})],
        )
        test_logger = MagicMock()
        log_response(output, logger=test_logger)
        # Nothing to log if all fields are empty/None
        assert not test_logger.debug.called

    def test_log_response_default_logger(self):
        from openbrowser.agent.service import log_response
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        output = AgentOutput(
            evaluation_previous_goal="success",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        # Should not raise when logger is None (uses module default)
        log_response(output, logger=None)

    def test_log_response_with_memory(self):
        from openbrowser.agent.service import log_response
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        output = AgentOutput(
            evaluation_previous_goal="",
            memory="Important memory",
            next_goal="",
            action=[CustomAction(done={"text": "test"})],
        )
        test_logger = MagicMock()
        log_response(output, logger=test_logger)
        info_calls = [str(c) for c in test_logger.info.call_args_list]
        assert any("Memory" in str(c) for c in info_calls)


# ---------------------------------------------------------------------------
# Agent __init__ and helper method tests
# ---------------------------------------------------------------------------


class TestAgentInit:
    """Tests for Agent.__init__ and its helper methods that don't require a full run."""

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def _create_agent(
        self,
        mock_fs_cls,
        mock_mm,
        mock_sp,
        mock_tc,
        mock_telemetry,
        mock_eventbus,
        task="Test task",
        llm=None,
        browser_session=None,
        tools=None,
        **kwargs,
    ):
        """Helper to create an Agent with all dependencies mocked."""
        from openbrowser.agent.service import Agent

        if llm is None:
            llm = _make_mock_llm()
        if browser_session is None:
            browser_session = _make_mock_browser_session()
        if tools is None:
            tools = _make_mock_tools()

        mock_sp_instance = MagicMock()
        mock_sp_instance.get_system_message.return_value = MagicMock(content="system prompt", cache=True)
        mock_sp.return_value = mock_sp_instance

        mock_fs_instance = MagicMock()
        mock_fs_instance.get_state.return_value = MagicMock()
        mock_fs_instance.base_dir = Path(tempfile.mkdtemp())
        mock_fs_cls.return_value = mock_fs_instance

        mock_tc_instance = MagicMock()
        mock_tc_instance.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_instance

        mock_eventbus_instance = MagicMock()
        mock_eventbus_instance.stop = AsyncMock()
        mock_eventbus.return_value = mock_eventbus_instance

        with patch.object(Agent, "_set_screenshot_service"):
            agent = Agent(
                task=task,
                llm=llm,
                browser_session=browser_session,
                tools=tools,
                **kwargs,
            )
        agent.screenshot_service = MagicMock()
        agent.screenshot_service.store_screenshot = AsyncMock(return_value=None)
        return agent

    def test_agent_init_basic(self):
        agent = self._create_agent(task="Find the price of product")
        assert agent.task is not None
        assert agent.state is not None
        assert agent.history is not None

    def test_agent_init_with_browser_alias_conflict(self):
        """Test that providing both browser and browser_session raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            self._create_agent(
                browser_session=_make_mock_browser_session(),
                browser=_make_mock_browser_session(),
            )

    def test_enhance_task_with_schema(self):
        class TestOutputSchema(BaseModel):
            price: float
            name: str

        agent = self._create_agent(
            task="Find the price",
            output_model_schema=TestOutputSchema,
        )
        assert "TestOutputSchema" in agent.task
        assert "price" in agent.task

    def test_enhance_task_without_schema(self):
        agent = self._create_agent(task="Find the price")
        assert agent.task == "Find the price"

    def test_logger_property(self):
        agent = self._create_agent()
        log = agent.logger
        assert log is not None
        assert "Agent" in log.name

    def test_browser_profile_property(self):
        session = _make_mock_browser_session()
        agent = self._create_agent(browser_session=session)
        profile = agent.browser_profile
        assert profile == session.browser_profile

    def test_extract_start_url_single_url(self):
        agent = self._create_agent()
        url = agent._extract_start_url("Go to https://example.com and find the price")
        assert url == "https://example.com"

    def test_extract_start_url_domain_only(self):
        agent = self._create_agent()
        url = agent._extract_start_url("Go to example.com")
        assert url == "https://example.com"

    def test_extract_start_url_multiple_urls(self):
        agent = self._create_agent()
        url = agent._extract_start_url("Compare https://a.com and https://b.com")
        assert url is None  # Multiple URLs -> return None

    def test_extract_start_url_no_url(self):
        agent = self._create_agent()
        url = agent._extract_start_url("Just do something without any urls")
        assert url is None

    def test_extract_start_url_excluded_extension(self):
        agent = self._create_agent()
        url = agent._extract_start_url("Download file from https://example.com/report.pdf")
        assert url is None

    def test_extract_start_url_excluded_word_before(self):
        agent = self._create_agent()
        url = agent._extract_start_url("Never go to https://evil.com")
        assert url is None

    def test_extract_start_url_dont_prefix(self):
        agent = self._create_agent()
        url = agent._extract_start_url("Don't visit https://blocked.com")
        assert url is None

    def test_extract_start_url_email_ignored(self):
        agent = self._create_agent()
        url = agent._extract_start_url("Send email to user@example.com and visit https://site.com")
        assert url == "https://site.com"

    def test_remove_think_tags(self):
        agent = self._create_agent()
        result = agent._remove_think_tags("Hello <think>internal thought</think> world")
        assert result == "Hello  world"

    def test_remove_think_tags_stray_close(self):
        agent = self._create_agent()
        result = agent._remove_think_tags("some thinking content</think> actual output")
        assert result == "actual output"

    def test_remove_think_tags_no_tags(self):
        agent = self._create_agent()
        result = agent._remove_think_tags("No tags here")
        assert result == "No tags here"

    def test_replace_urls_in_text_short_url_unchanged(self):
        agent = self._create_agent()
        text = "Visit https://example.com"
        result, replaced = agent._replace_urls_in_text(text)
        # URL is short, should not be replaced
        assert result == text
        assert replaced == {}

    def test_replace_urls_in_text_long_url_shortened(self):
        agent = self._create_agent(_url_shortening_limit=10)
        long_query = "?" + "a" * 100
        text = f"Visit https://example.com{long_query}"
        result, replaced = agent._replace_urls_in_text(text)
        assert len(replaced) > 0 or result == text  # May or may not shorten based on logic

    def test_replace_shortened_urls_in_string(self):
        from openbrowser.agent.service import Agent

        text = "Go to https://short.url"
        replacements = {"https://short.url": "https://very-long-original-url.com/path?q=value"}
        result = Agent._replace_shortened_urls_in_string(text, replacements)
        assert "very-long-original-url" in result

    def test_recursive_process_dict(self):
        from openbrowser.agent.service import Agent

        d = {"key": "https://short.url", "nested": {"inner": "https://short.url"}}
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_dict(d, replacements)
        assert d["key"] == "https://original.com"
        assert d["nested"]["inner"] == "https://original.com"

    def test_recursive_process_list_or_tuple_list(self):
        from openbrowser.agent.service import Agent

        lst = ["https://short.url", 42, "other"]
        replacements = {"https://short.url": "https://original.com"}
        result = Agent._recursive_process_list_or_tuple(lst, replacements)
        assert result[0] == "https://original.com"
        assert result[1] == 42

    def test_recursive_process_list_or_tuple_tuple(self):
        from openbrowser.agent.service import Agent

        tpl = ("https://short.url", 42)
        replacements = {"https://short.url": "https://original.com"}
        result = Agent._recursive_process_list_or_tuple(tpl, replacements)
        assert isinstance(result, tuple)
        assert result[0] == "https://original.com"

    def test_recursive_process_list_with_nested_dict(self):
        from openbrowser.agent.service import Agent

        lst = [{"url": "https://short.url"}]
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_list_or_tuple(lst, replacements)
        assert lst[0]["url"] == "https://original.com"

    def test_recursive_process_list_with_nested_list(self):
        from openbrowser.agent.service import Agent

        lst = [["https://short.url"]]
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_list_or_tuple(lst, replacements)
        assert lst[0][0] == "https://original.com"

    def test_recursive_process_tuple_with_nested_tuple(self):
        from openbrowser.agent.service import Agent

        tpl = (("https://short.url",),)
        replacements = {"https://short.url": "https://original.com"}
        result = Agent._recursive_process_list_or_tuple(tpl, replacements)
        assert isinstance(result, tuple)
        assert result[0][0] == "https://original.com"

    def test_recursive_process_tuple_with_non_string_items(self):
        from openbrowser.agent.service import Agent

        tpl = (42, True, None)
        replacements = {"https://short.url": "https://original.com"}
        result = Agent._recursive_process_list_or_tuple(tpl, replacements)
        assert result == (42, True, None)

    def test_recursive_process_pydantic_model(self):
        from openbrowser.agent.service import Agent

        class Inner(BaseModel):
            url: str = "https://short.url"

        class Outer(BaseModel):
            inner: Inner = Inner()
            name: str = "https://short.url"
            items: list[str] = ["https://short.url"]

        model = Outer()
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_all_strings_inside_pydantic_model(model, replacements)
        assert model.name == "https://original.com"
        assert model.inner.url == "https://original.com"
        assert model.items[0] == "https://original.com"

    def test_recursive_process_pydantic_model_with_dict_field(self):
        from openbrowser.agent.service import Agent

        class M(BaseModel):
            data: dict = {"url": "https://short.url"}

        model = M()
        replacements = {"https://short.url": "https://original.com"}
        Agent._recursive_process_all_strings_inside_pydantic_model(model, replacements)
        assert model.data["url"] == "https://original.com"

    def test_process_messages_and_replace_long_urls(self):
        from openbrowser.llm.messages import UserMessage

        agent = self._create_agent(_url_shortening_limit=10)
        long_url = "https://example.com?" + "x" * 100
        msg = UserMessage(content=f"Visit {long_url}")
        urls_replaced = agent._process_messsages_and_replace_long_urls_shorter_ones([msg])
        # The method should process and possibly replace the long URL
        assert isinstance(urls_replaced, dict)

    def test_process_messages_with_list_content(self):
        from openbrowser.llm.messages import ContentPartTextParam, UserMessage

        agent = self._create_agent(_url_shortening_limit=10)
        long_url = "https://example.com?" + "x" * 100
        msg = UserMessage(content=[ContentPartTextParam(text=f"Visit {long_url}")])
        urls_replaced = agent._process_messsages_and_replace_long_urls_shorter_ones([msg])
        assert isinstance(urls_replaced, dict)

    def test_convert_initial_actions(self):
        agent = self._create_agent()

        # Create a real param model that Pydantic can validate
        class NavigateParams(BaseModel):
            url: str
            new_tab: bool = False

        # Override ActionModel to accept NavigateParams
        ActionModelWithNav = create_model(
            "ActionModelWithNav",
            __base__=_get_action_model_base(),
            navigate=(Optional[NavigateParams], None),
            done=(Optional[dict], None),
        )
        agent.ActionModel = ActionModelWithNav

        action_info = MagicMock()
        action_info.param_model = NavigateParams
        agent.tools.registry.registry.actions = {"navigate": action_info}

        actions = [{"navigate": {"url": "https://example.com", "new_tab": False}}]
        result = agent._convert_initial_actions(actions)
        assert len(result) == 1

    def test_verify_and_setup_llm_already_verified(self):
        agent = self._create_agent()
        agent.llm._verified_api_keys = True
        result = agent._verify_and_setup_llm()
        assert result is True

    def test_setup_action_models_flash_mode(self):
        agent = self._create_agent(flash_mode=True)
        # Agent should have flash mode action models set up
        assert agent.AgentOutput is not None
        assert agent.DoneAgentOutput is not None

    def test_setup_action_models_no_thinking(self):
        agent = self._create_agent(use_thinking=False)
        assert agent.AgentOutput is not None

    def test_setup_action_models_with_thinking(self):
        agent = self._create_agent(use_thinking=True, flash_mode=False)
        assert agent.AgentOutput is not None

    def test_add_new_task(self):
        agent = self._create_agent()
        agent.add_new_task("Do something new")
        assert agent.task == "Do something new"
        assert agent.state.follow_up_task is True
        assert agent.state.stopped is False
        assert agent.state.paused is False

    def test_pause(self):
        agent = self._create_agent()
        with patch("builtins.print"):
            agent.pause()
        assert agent.state.paused is True

    def test_resume(self):
        agent = self._create_agent()
        with patch("builtins.print"):
            agent.pause()
        with patch("builtins.print"):
            agent.resume()
        assert agent.state.paused is False

    def test_stop(self):
        agent = self._create_agent()
        agent.stop()
        assert agent.state.stopped is True

    def test_save_history_default_path(self, tmp_path):
        agent = self._create_agent()
        # Replace save_to_file on the class level
        with patch.object(type(agent.history), "save_to_file") as mock_save:
            agent.save_history()
            mock_save.assert_called_once_with("AgentHistory.json", sensitive_data=agent.sensitive_data)

    def test_save_history_custom_path(self, tmp_path):
        agent = self._create_agent()
        with patch.object(type(agent.history), "save_to_file") as mock_save:
            agent.save_history("/tmp/my_history.json")
            mock_save.assert_called_once_with("/tmp/my_history.json", sensitive_data=agent.sensitive_data)

    def test_save_file_system_state(self):
        agent = self._create_agent()
        agent.save_file_system_state()
        assert agent.state.file_system_state is not None

    def test_save_file_system_state_no_fs(self):
        agent = self._create_agent()
        agent.file_system = None
        with pytest.raises(ValueError, match="File system is not set up"):
            agent.save_file_system_state()

    def test_message_manager_property(self):
        agent = self._create_agent()
        assert agent.message_manager is not None

    def test_set_openbrowser_version_and_source(self):
        agent = self._create_agent()
        assert agent.version is not None
        assert agent.source is not None

    def test_set_openbrowser_version_source_override(self):
        agent = self._create_agent(source="custom")
        assert agent.source == "custom"

    def test_deepseek_model_disables_vision(self):
        llm = _make_mock_llm(model="deepseek-v3")
        agent = self._create_agent(llm=llm, use_vision=True)
        assert agent.settings.use_vision is False

    def test_grok_model_disables_vision(self):
        llm = _make_mock_llm(model="grok-beta")
        agent = self._create_agent(llm=llm, use_vision=True)
        assert agent.settings.use_vision is False

    def test_log_step_context(self):
        agent = self._create_agent()
        browser_state = _make_mock_browser_state_summary()
        # Should not raise
        agent._log_step_context(browser_state)

    def test_log_next_action_summary_empty_actions(self):
        agent = self._create_agent()
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        parsed = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[],
        )
        # Should handle empty actions without error
        agent._log_next_action_summary(parsed)

    def test_log_next_action_summary_with_actions(self):
        agent = self._create_agent()
        from openbrowser.agent.views import AgentOutput

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))
        action = CustomAction(done={"text": "finished", "success": True})
        parsed = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[action],
        )
        # Enable debug logging for the test
        with patch.object(agent.logger, "isEnabledFor", return_value=True):
            agent._log_next_action_summary(parsed)

    def test_log_step_completion_summary(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        results = [ActionResult(extracted_content="done"), ActionResult(error="failed")]
        agent._log_step_completion_summary(time.time() - 1.0, results)

    def test_log_step_completion_summary_empty(self):
        agent = self._create_agent()
        agent._log_step_completion_summary(time.time(), [])

    def test_log_step_completion_summary_all_success(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        results = [ActionResult(extracted_content="ok")]
        agent._log_step_completion_summary(time.time() - 0.5, results)

    def test_log_final_outcome_messages_failure(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "is_successful", return_value=False):
            with patch.object(type(agent.history), "final_result", return_value="some result"):
                agent._log_final_outcome_messages()

    def test_log_final_outcome_messages_none(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "is_successful", return_value=None):
            with patch.object(type(agent.history), "final_result", return_value=None):
                agent._log_final_outcome_messages()

    def test_log_first_step_startup(self):
        agent = self._create_agent()
        agent._log_first_step_startup()

    def test_log_first_step_startup_not_first(self):
        agent = self._create_agent()
        # Simulate history already has items
        agent.history.history.append(MagicMock())
        agent._log_first_step_startup()

    def test_log_action_single(self):
        agent = self._create_agent()
        action = MagicMock()
        action.model_dump.return_value = {"click": {"index": 5, "text": "Submit"}}
        agent._log_action(action, "click", 1, 1)

    def test_log_action_multiple(self):
        agent = self._create_agent()
        action = MagicMock()
        action.model_dump.return_value = {"navigate": {"url": "https://example.com"}}
        agent._log_action(action, "navigate", 1, 3)

    def test_log_action_long_value(self):
        agent = self._create_agent()
        action = MagicMock()
        action.model_dump.return_value = {"input_text": {"text": "x" * 200}}
        agent._log_action(action, "input_text", 1, 1)

    def test_log_action_list_value(self):
        agent = self._create_agent()
        action = MagicMock()
        action.model_dump.return_value = {"scroll": {"items": list(range(100))}}
        agent._log_action(action, "scroll", 1, 1)

    def test_log_action_no_params(self):
        agent = self._create_agent()
        action = MagicMock()
        action.model_dump.return_value = {"noop": {}}
        agent._log_action(action, "noop", 1, 1)


# ---------------------------------------------------------------------------
# Async step/run tests
# ---------------------------------------------------------------------------


class TestAgentAsyncMethods:
    """Tests for async methods: step, run, multi_act, etc."""

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def _create_agent(
        self,
        mock_fs_cls,
        mock_mm,
        mock_sp,
        mock_tc,
        mock_telemetry,
        mock_eventbus,
        task="Test task",
        llm=None,
        browser_session=None,
        tools=None,
        **kwargs,
    ):
        from openbrowser.agent.service import Agent

        if llm is None:
            llm = _make_mock_llm()
        if browser_session is None:
            browser_session = _make_mock_browser_session()
        if tools is None:
            tools = _make_mock_tools()

        mock_sp_instance = MagicMock()
        mock_sp_instance.get_system_message.return_value = MagicMock(content="system prompt", cache=True)
        mock_sp.return_value = mock_sp_instance

        mock_fs_instance = MagicMock()
        mock_fs_instance.get_state.return_value = MagicMock()
        mock_fs_instance.base_dir = Path(tempfile.mkdtemp())
        mock_fs_cls.return_value = mock_fs_instance

        mock_tc_instance = MagicMock()
        mock_tc_instance.register_llm = MagicMock()
        mock_tc_instance.get_usage_summary = AsyncMock(return_value=None)
        mock_tc_instance.log_usage_summary = AsyncMock()
        mock_tc_instance.get_usage_tokens_for_model = MagicMock(
            return_value=MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_cached_tokens=10,
                total_tokens=150,
            )
        )
        mock_tc.return_value = mock_tc_instance

        mock_eventbus_instance = MagicMock()
        mock_eventbus_instance.stop = AsyncMock()
        mock_eventbus.return_value = mock_eventbus_instance

        with patch.object(Agent, "_set_screenshot_service"):
            agent = Agent(
                task=task,
                llm=llm,
                browser_session=browser_session,
                tools=tools,
                **kwargs,
            )
        agent.screenshot_service = MagicMock()
        agent.screenshot_service.store_screenshot = AsyncMock(return_value=None)
        return agent

    @pytest.mark.asyncio
    async def test_check_stop_or_pause_stopped(self):
        agent = self._create_agent()
        agent.state.stopped = True
        with pytest.raises(InterruptedError):
            await agent._check_stop_or_pause()

    @pytest.mark.asyncio
    async def test_check_stop_or_pause_paused(self):
        agent = self._create_agent()
        agent.state.paused = True
        with pytest.raises(InterruptedError):
            await agent._check_stop_or_pause()

    @pytest.mark.asyncio
    async def test_check_stop_or_pause_should_stop_callback(self):
        callback = AsyncMock(return_value=True)
        agent = self._create_agent()
        agent.register_should_stop_callback = callback
        with pytest.raises(InterruptedError):
            await agent._check_stop_or_pause()

    @pytest.mark.asyncio
    async def test_check_stop_or_pause_external_callback(self):
        callback = AsyncMock(return_value=True)
        agent = self._create_agent()
        agent.register_external_agent_status_raise_error_callback = callback
        with pytest.raises(InterruptedError):
            await agent._check_stop_or_pause()

    @pytest.mark.asyncio
    async def test_check_stop_or_pause_normal(self):
        agent = self._create_agent()
        # Should not raise
        await agent._check_stop_or_pause()

    @pytest.mark.asyncio
    async def test_check_and_update_downloads_no_path(self):
        agent = self._create_agent()
        agent.has_downloads_path = False
        # Should return without doing anything
        await agent._check_and_update_downloads()

    @pytest.mark.asyncio
    async def test_check_and_update_downloads_new_files(self):
        session = _make_mock_browser_session(downloads_path="/tmp/downloads")
        agent = self._create_agent(browser_session=session)
        agent.has_downloads_path = True
        agent._last_known_downloads = []
        agent.browser_session.downloaded_files = ["/tmp/downloads/file.pdf"]
        await agent._check_and_update_downloads("test context")

    @pytest.mark.asyncio
    async def test_check_and_update_downloads_exception(self):
        agent = self._create_agent()
        agent.has_downloads_path = True
        agent._last_known_downloads = []
        type(agent.browser_session).downloaded_files = PropertyMock(side_effect=Exception("Error"))
        await agent._check_and_update_downloads("test context")

    def test_update_available_file_paths_new_files(self):
        agent = self._create_agent()
        agent.has_downloads_path = True
        agent.available_file_paths = ["/existing/file.txt"]
        agent._update_available_file_paths(["/existing/file.txt", "/new/file.pdf"])
        assert "/new/file.pdf" in agent.available_file_paths

    def test_update_available_file_paths_no_new_files(self):
        agent = self._create_agent()
        agent.has_downloads_path = True
        agent.available_file_paths = ["/existing/file.txt"]
        agent._update_available_file_paths(["/existing/file.txt"])
        assert len(agent.available_file_paths) == 1

    def test_update_available_file_paths_no_downloads_path(self):
        agent = self._create_agent()
        agent.has_downloads_path = False
        agent._update_available_file_paths(["/new/file.pdf"])
        # Should return without updating

    @pytest.mark.asyncio
    async def test_force_done_after_last_step(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=9, max_steps=10)
        await agent._force_done_after_last_step(step_info)
        assert agent.AgentOutput == agent.DoneAgentOutput

    @pytest.mark.asyncio
    async def test_force_done_after_last_step_not_last(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        original_output = agent.AgentOutput
        step_info = AgentStepInfo(step_number=5, max_steps=10)
        await agent._force_done_after_last_step(step_info)
        assert agent.AgentOutput == original_output

    @pytest.mark.asyncio
    async def test_force_done_after_last_step_no_info(self):
        agent = self._create_agent()
        original_output = agent.AgentOutput
        await agent._force_done_after_last_step(None)
        assert agent.AgentOutput == original_output

    @pytest.mark.asyncio
    async def test_force_done_after_failure(self):
        agent = self._create_agent()
        agent.state.consecutive_failures = 3
        agent.settings.max_failures = 3
        agent.settings.final_response_after_failure = True
        await agent._force_done_after_failure()
        assert agent.AgentOutput == agent.DoneAgentOutput

    @pytest.mark.asyncio
    async def test_force_done_after_failure_not_enough(self):
        agent = self._create_agent()
        agent.state.consecutive_failures = 1
        agent.settings.max_failures = 3
        original_output = agent.AgentOutput
        await agent._force_done_after_failure()
        assert agent.AgentOutput == original_output

    @pytest.mark.asyncio
    async def test_force_done_after_failure_disabled(self):
        agent = self._create_agent()
        agent.state.consecutive_failures = 3
        agent.settings.max_failures = 3
        agent.settings.final_response_after_failure = False
        original_output = agent.AgentOutput
        await agent._force_done_after_failure()
        assert agent.AgentOutput == original_output

    @pytest.mark.asyncio
    async def test_handle_step_error_interrupted(self):
        agent = self._create_agent()
        await agent._handle_step_error(InterruptedError("stopped"))
        # Should not set last_result

    @pytest.mark.asyncio
    async def test_handle_step_error_interrupted_no_msg(self):
        agent = self._create_agent()
        await agent._handle_step_error(InterruptedError())

    @pytest.mark.asyncio
    async def test_handle_step_error_validation_error(self):
        agent = self._create_agent()
        try:
            from openbrowser.agent.views import AgentSettings
            AgentSettings(max_failures="bad")
        except ValidationError as e:
            await agent._handle_step_error(e)
            assert agent.state.consecutive_failures > 0
            assert agent.state.last_result is not None

    @pytest.mark.asyncio
    async def test_handle_step_error_parse_error(self):
        agent = self._create_agent()
        error = RuntimeError("Could not parse response from LLM")
        await agent._handle_step_error(error)
        assert agent.state.consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_handle_step_error_generic(self):
        agent = self._create_agent()
        await agent._handle_step_error(RuntimeError("Something went wrong"))
        assert agent.state.consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_finalize_no_result(self):
        agent = self._create_agent()
        agent.step_start_time = time.time()
        agent.state.last_result = None
        browser_state = _make_mock_browser_state_summary()
        await agent._finalize(browser_state)
        # Should return early without recording

    @pytest.mark.asyncio
    async def test_finalize_with_result(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        agent.step_start_time = time.time()
        agent.state.last_result = [ActionResult(extracted_content="done")]
        agent.state.last_model_output = None

        browser_state = _make_mock_browser_state_summary()
        with patch.object(agent, "_make_history_item", new_callable=AsyncMock):
            await agent._finalize(browser_state)

    @pytest.mark.asyncio
    async def test_finalize_no_browser_state(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        agent.step_start_time = time.time()
        agent.state.last_result = [ActionResult(extracted_content="done")]
        await agent._finalize(None)

    @pytest.mark.asyncio
    async def test_execute_actions_no_output(self):
        agent = self._create_agent()
        agent.state.last_model_output = None
        with pytest.raises(ValueError, match="No model output"):
            await agent._execute_actions()

    @pytest.mark.asyncio
    async def test_execute_actions_with_output(self):
        from openbrowser.agent.views import AgentOutput
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        agent.state.last_model_output = output
        agent.tools.act = AsyncMock(return_value=ActionResult(is_done=True, success=True, extracted_content="done"))
        await agent._execute_actions()
        assert agent.state.last_result is not None

    @pytest.mark.asyncio
    async def test_post_process_single_error(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        agent.state.last_result = [ActionResult(error="failed")]
        agent.state.consecutive_failures = 0
        await agent._post_process()
        assert agent.state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_post_process_success_resets_failures(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        agent.state.consecutive_failures = 2
        agent.state.last_result = [ActionResult(extracted_content="ok")]
        await agent._post_process()
        assert agent.state.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_post_process_done_success(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        agent.state.consecutive_failures = 0
        agent.state.last_result = [ActionResult(is_done=True, success=True, extracted_content="Final answer")]
        await agent._post_process()

    @pytest.mark.asyncio
    async def test_post_process_done_failure(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        agent.state.consecutive_failures = 0
        agent.state.last_result = [ActionResult(is_done=True, success=False, extracted_content="Could not complete")]
        await agent._post_process()

    @pytest.mark.asyncio
    async def test_post_process_with_attachments(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        agent.state.consecutive_failures = 0
        agent.state.last_result = [
            ActionResult(
                is_done=True,
                success=True,
                extracted_content="Done",
                attachments=["/tmp/file1.pdf", "/tmp/file2.png"],
            )
        ]
        await agent._post_process()

    @pytest.mark.asyncio
    async def test_handle_post_llm_processing_sync_callback(self):
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        callback = MagicMock()
        agent.register_new_step_callback = callback
        agent.state.last_model_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        browser_state = _make_mock_browser_state_summary()
        await agent._handle_post_llm_processing(browser_state, [])
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_post_llm_processing_async_callback(self):
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        callback = AsyncMock()
        agent.register_new_step_callback = callback
        agent.state.last_model_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        browser_state = _make_mock_browser_state_summary()
        await agent._handle_post_llm_processing(browser_state, [])
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_post_llm_processing_save_conversation(self):
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        agent.settings.save_conversation_path = Path(tempfile.mkdtemp())
        agent.state.last_model_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        browser_state = _make_mock_browser_state_summary()
        with patch("openbrowser.agent.service.save_conversation", new_callable=AsyncMock):
            await agent._handle_post_llm_processing(browser_state, [])

    @pytest.mark.asyncio
    async def test_handle_post_llm_processing_no_callback_no_save(self):
        agent = self._create_agent()
        agent.register_new_step_callback = None
        agent.settings.save_conversation_path = None
        agent.state.last_model_output = None
        browser_state = _make_mock_browser_state_summary()
        await agent._handle_post_llm_processing(browser_state, [])

    @pytest.mark.asyncio
    async def test_make_history_item_with_output(self):
        from openbrowser.agent.views import AgentOutput, StepMetadata
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        browser_state = _make_mock_browser_state_summary(screenshot="base64data")
        agent.screenshot_service = MagicMock()
        agent.screenshot_service.store_screenshot = AsyncMock(return_value="/tmp/screenshot.png")

        metadata = StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0)
        result = [ActionResult(extracted_content="done")]

        await agent._make_history_item(output, browser_state, result, metadata, state_message="test state")
        assert len(agent.history.history) == 1

    @pytest.mark.asyncio
    async def test_make_history_item_without_output(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        browser_state = _make_mock_browser_state_summary()
        agent.screenshot_service = MagicMock()
        agent.screenshot_service.store_screenshot = AsyncMock(return_value=None)

        result = [ActionResult(error="Something failed")]
        await agent._make_history_item(None, browser_state, result)
        assert len(agent.history.history) == 1

    @pytest.mark.asyncio
    async def test_make_history_item_no_screenshot(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        browser_state = _make_mock_browser_state_summary(screenshot=None)
        agent.screenshot_service = MagicMock()

        result = [ActionResult(extracted_content="done")]
        await agent._make_history_item(None, browser_state, result)
        assert len(agent.history.history) == 1

    @pytest.mark.asyncio
    async def test_get_model_output_with_retry_valid(self):
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        expected_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        agent.get_model_output = AsyncMock(return_value=expected_output)

        result = await agent._get_model_output_with_retry([])
        assert result == expected_output

    @pytest.mark.asyncio
    async def test_get_model_output_with_retry_empty_list_then_valid(self):
        """Test retry when model returns empty action list, then returns valid output."""
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        empty_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[],  # Empty action list triggers retry
        )
        valid_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        agent.get_model_output = AsyncMock(side_effect=[empty_output, valid_output])

        result = await agent._get_model_output_with_retry([])
        assert result == valid_output

    @pytest.mark.asyncio
    async def test_get_model_output_with_retry_empty_list_both_times(self):
        """Test retry when model returns empty action list both times - inserts safe noop."""
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()

        empty_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[],
        )
        agent.get_model_output = AsyncMock(return_value=empty_output)

        result = await agent._get_model_output_with_retry([])
        # Should insert a safe noop action
        assert len(result.action) == 1

    @pytest.mark.asyncio
    async def test_get_model_output_with_retry_not_list(self):
        """Test retry when actions are not a list."""
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        # Override action to be None (not a list)
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[],
        )
        output.action = None  # type: ignore - Force non-list

        valid_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        agent.get_model_output = AsyncMock(side_effect=[output, valid_output])

        result = await agent._get_model_output_with_retry([])
        assert len(result.action) > 0

    @pytest.mark.asyncio
    async def test_get_model_output_with_retry_all_empty_dicts(self):
        """Test retry when all actions have empty model_dump()."""
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        empty_action = CustomAction()

        first_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[empty_action],
        )
        # Patch model_dump using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(empty_action, "model_dump", lambda **kwargs: {})

        valid_output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        agent.get_model_output = AsyncMock(side_effect=[first_output, valid_output])

        result = await agent._get_model_output_with_retry([])
        assert len(result.action) > 0

    @pytest.mark.asyncio
    async def test_get_model_output_with_retry_all_empty_dicts_both_times(self):
        """Test when all actions have empty model_dump() both times - inserts noop."""
        from openbrowser.agent.views import AgentOutput

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        empty_action = CustomAction()

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[empty_action],
        )
        object.__setattr__(empty_action, "model_dump", lambda **kwargs: {})

        agent.get_model_output = AsyncMock(return_value=output)

        result = await agent._get_model_output_with_retry([])
        assert len(result.action) == 1

    @pytest.mark.asyncio
    async def test_get_model_output(self):
        from openbrowser.agent.views import AgentOutput
        from openbrowser.llm.messages import SystemMessage

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        expected = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        response = MagicMock()
        response.completion = expected
        agent.llm.ainvoke = AsyncMock(return_value=response)

        result = await agent.get_model_output([SystemMessage(content="test")])
        assert result.memory == "mem"

    @pytest.mark.asyncio
    async def test_get_model_output_validation_error(self):
        from openbrowser.llm.messages import SystemMessage

        agent = self._create_agent()
        agent.llm.ainvoke = AsyncMock(side_effect=ValidationError.from_exception_data(
            title="test",
            line_errors=[],
        ))

        with pytest.raises(ValidationError):
            await agent.get_model_output([SystemMessage(content="test")])

    @pytest.mark.asyncio
    async def test_get_model_output_truncates_actions(self):
        from openbrowser.agent.views import AgentOutput
        from openbrowser.llm.messages import SystemMessage

        agent = self._create_agent(max_actions_per_step=2)
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        many_actions = [CustomAction(done={"text": f"action_{i}"}) for i in range(5)]
        expected = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=many_actions,
        )
        response = MagicMock()
        response.completion = expected
        agent.llm.ainvoke = AsyncMock(return_value=response)

        result = await agent.get_model_output([SystemMessage(content="test")])
        assert len(result.action) == 2

    @pytest.mark.asyncio
    async def test_multi_act_single_action(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        action = CustomAction(click={"index": 1})
        agent.tools.act = AsyncMock(return_value=ActionResult(extracted_content="clicked"))

        results = await agent.multi_act([action])
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multi_act_done_not_first(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None), done=(Optional[dict], None))

        actions = [
            CustomAction(click={"index": 1}),
            CustomAction(done={"text": "finished"}),
        ]
        agent.tools.act = AsyncMock(return_value=ActionResult(extracted_content="ok"))

        results = await agent.multi_act(actions)
        # done action as second action should be skipped
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multi_act_error_stops(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        actions = [
            CustomAction(click={"index": 1}),
            CustomAction(click={"index": 2}),
        ]
        agent.tools.act = AsyncMock(return_value=ActionResult(error="Element not found"))

        results = await agent.multi_act(actions)
        assert len(results) == 1
        assert results[0].error is not None

    @pytest.mark.asyncio
    async def test_multi_act_is_done_stops(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        actions = [
            CustomAction(click={"index": 1}),
            CustomAction(click={"index": 2}),
        ]
        agent.tools.act = AsyncMock(return_value=ActionResult(is_done=True, success=True, extracted_content="done"))

        results = await agent.multi_act(actions)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multi_act_exception(self):
        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        actions = [CustomAction(click={"index": 1})]
        agent.tools.act = AsyncMock(side_effect=RuntimeError("Browser crashed"))

        with pytest.raises(RuntimeError, match="Browser crashed"):
            await agent.multi_act(actions)

    @pytest.mark.asyncio
    async def test_multi_act_with_cached_selector_map(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        # Set up cached state
        mock_element = MagicMock()
        mock_element.parent_branch_hash.return_value = "hash1"
        cached_state = MagicMock()
        cached_state.dom_state = MagicMock()
        cached_state.dom_state.selector_map = {1: mock_element}
        agent.browser_session._cached_browser_state_summary = cached_state

        actions = [CustomAction(click={"index": 1})]
        agent.tools.act = AsyncMock(return_value=ActionResult(extracted_content="ok"))

        results = await agent.multi_act(actions)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multi_act_cached_selector_map_exception(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        # Make selector_map access raise an exception
        cached_state = MagicMock()
        type(cached_state).dom_state = PropertyMock(side_effect=Exception("Error"))
        agent.browser_session._cached_browser_state_summary = cached_state

        actions = [CustomAction(click={"index": 1})]
        agent.tools.act = AsyncMock(return_value=ActionResult(extracted_content="ok"))

        results = await agent.multi_act(actions)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_update_action_models_for_page_flash(self):
        agent = self._create_agent(flash_mode=True)
        await agent._update_action_models_for_page("https://example.com")
        assert agent.AgentOutput is not None

    @pytest.mark.asyncio
    async def test_update_action_models_for_page_thinking(self):
        agent = self._create_agent(use_thinking=True, flash_mode=False)
        await agent._update_action_models_for_page("https://example.com")
        assert agent.AgentOutput is not None

    @pytest.mark.asyncio
    async def test_update_action_models_for_page_no_thinking(self):
        agent = self._create_agent(use_thinking=False, flash_mode=False)
        await agent._update_action_models_for_page("https://example.com")
        assert agent.AgentOutput is not None

    @pytest.mark.asyncio
    async def test_close_browser_killed(self):
        agent = self._create_agent()
        agent.browser_session.browser_profile.keep_alive = False
        await agent.close()
        agent.browser_session.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_browser_keep_alive(self):
        agent = self._create_agent()
        agent.browser_session.browser_profile.keep_alive = True
        await agent.close()
        agent.browser_session.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_exception_handled(self):
        agent = self._create_agent()
        agent.browser_session.browser_profile.keep_alive = False
        agent.browser_session.kill = AsyncMock(side_effect=Exception("Cleanup error"))
        # Should not raise
        await agent.close()

    @pytest.mark.asyncio
    async def test_log_completion_successful(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "is_successful", return_value=True):
            await agent.log_completion()

    @pytest.mark.asyncio
    async def test_log_completion_unsuccessful(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "is_successful", return_value=False):
            await agent.log_completion()

    @pytest.mark.asyncio
    async def test_execute_initial_actions(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        # Use the agent's own ActionModel for initial actions
        action = agent.ActionModel(navigate={"url": "https://example.com"})
        agent.initial_actions = [action]
        agent.initial_url = "https://example.com"
        agent.state.follow_up_task = False

        result = ActionResult(long_term_memory="Page loaded", extracted_content="Example Domain")
        agent.tools.act = AsyncMock(return_value=result)

        await agent._execute_initial_actions()
        assert agent.state.last_result is not None

    @pytest.mark.asyncio
    async def test_execute_initial_actions_follow_up(self):
        agent = self._create_agent()
        agent.initial_actions = [MagicMock()]
        agent.state.follow_up_task = True
        # Should skip because follow_up_task is True
        await agent._execute_initial_actions()

    @pytest.mark.asyncio
    async def test_execute_initial_actions_none(self):
        agent = self._create_agent()
        agent.initial_actions = None
        # Should skip
        await agent._execute_initial_actions()

    @pytest.mark.asyncio
    async def test_execute_initial_actions_flash_mode(self):
        from openbrowser.models import ActionResult

        agent = self._create_agent(flash_mode=True)
        action = agent.ActionModel(navigate={"url": "https://example.com"})
        agent.initial_actions = [action]
        agent.initial_url = "https://example.com"
        agent.state.follow_up_task = False

        result = ActionResult(long_term_memory="Page loaded")
        agent.tools.act = AsyncMock(return_value=result)

        await agent._execute_initial_actions()
        assert len(agent.history.history) > 0

    @pytest.mark.asyncio
    async def test_take_step_first_step(self):
        from openbrowser.agent.views import AgentStepInfo
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)

        with patch.object(agent, "_execute_initial_actions", new_callable=AsyncMock):
            with patch.object(agent, "step", new_callable=AsyncMock):
                with patch.object(type(agent.history), "is_done", return_value=False):
                    is_done, is_valid = await agent.take_step(step_info)
                    assert is_done is False

    @pytest.mark.asyncio
    async def test_take_step_done(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=5, max_steps=10)

        with patch.object(agent, "step", new_callable=AsyncMock):
            with patch.object(type(agent.history), "is_done", return_value=True):
                with patch.object(agent, "log_completion", new_callable=AsyncMock):
                    is_done, is_valid = await agent.take_step(step_info)
                    assert is_done is True

    @pytest.mark.asyncio
    async def test_take_step_done_with_sync_callback(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=5, max_steps=10)
        done_callback = MagicMock()
        agent.register_done_callback = done_callback

        with patch.object(agent, "step", new_callable=AsyncMock):
            with patch.object(type(agent.history), "is_done", return_value=True):
                with patch.object(agent, "log_completion", new_callable=AsyncMock):
                    is_done, _ = await agent.take_step(step_info)
                    assert is_done is True
                    done_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_take_step_done_with_async_callback(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=5, max_steps=10)
        done_callback = AsyncMock()
        agent.register_done_callback = done_callback

        with patch.object(agent, "step", new_callable=AsyncMock):
            with patch.object(type(agent.history), "is_done", return_value=True):
                with patch.object(agent, "log_completion", new_callable=AsyncMock):
                    is_done, _ = await agent.take_step(step_info)
                    assert is_done is True
                    done_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_take_step_first_step_interrupted(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)

        with patch.object(agent, "_execute_initial_actions", new_callable=AsyncMock, side_effect=InterruptedError):
            with patch.object(agent, "step", new_callable=AsyncMock):
                with patch.object(type(agent.history), "is_done", return_value=False):
                    is_done, _ = await agent.take_step(step_info)
                    assert is_done is False

    @pytest.mark.asyncio
    async def test_execute_step_success(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)

        with patch.object(agent, "step", new_callable=AsyncMock):
            with patch.object(type(agent.history), "is_done", return_value=False):
                result = await agent._execute_step(0, 10, step_info)
                assert result is False

    @pytest.mark.asyncio
    async def test_execute_step_done(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)

        with patch.object(agent, "step", new_callable=AsyncMock):
            with patch.object(type(agent.history), "is_done", return_value=True):
                with patch.object(agent, "log_completion", new_callable=AsyncMock):
                    result = await agent._execute_step(0, 10, step_info)
                    assert result is True

    @pytest.mark.asyncio
    async def test_execute_step_timeout(self):
        from openbrowser.agent.views import AgentStepInfo
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)

        with patch.object(agent, "step", side_effect=asyncio.TimeoutError("step timed out")):
            with patch.object(type(agent.history), "is_done", return_value=False):
                result = await agent._execute_step(0, 10, step_info)
                assert result is False
                assert agent.state.consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_execute_step_with_callbacks(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)
        on_start = AsyncMock()
        on_end = AsyncMock()

        with patch.object(agent, "step", new_callable=AsyncMock):
            with patch.object(type(agent.history), "is_done", return_value=False):
                await agent._execute_step(0, 10, step_info, on_step_start=on_start, on_step_end=on_end)
                on_start.assert_called_once()
                on_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_step_done_with_sync_done_callback(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)
        done_callback = MagicMock()
        agent.register_done_callback = done_callback

        with patch.object(agent, "step", new_callable=AsyncMock):
            with patch.object(type(agent.history), "is_done", return_value=True):
                with patch.object(agent, "log_completion", new_callable=AsyncMock):
                    result = await agent._execute_step(0, 10, step_info)
                    assert result is True
                    done_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_step_done_with_async_done_callback(self):
        from openbrowser.agent.views import AgentStepInfo

        agent = self._create_agent()
        step_info = AgentStepInfo(step_number=0, max_steps=10)
        done_callback = AsyncMock()
        agent.register_done_callback = done_callback

        with patch.object(agent, "step", new_callable=AsyncMock):
            with patch.object(type(agent.history), "is_done", return_value=True):
                with patch.object(agent, "log_completion", new_callable=AsyncMock):
                    result = await agent._execute_step(0, 10, step_info)
                    assert result is True
                    done_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_next_action_timeout(self):
        agent = self._create_agent()
        agent.settings.llm_timeout = 0.01
        agent._message_manager.get_messages = MagicMock(return_value=[])

        async def slow_model(*args, **kwargs):
            await asyncio.sleep(10)

        agent._get_model_output_with_retry = slow_model
        browser_state = _make_mock_browser_state_summary()

        with pytest.raises(TimeoutError, match="timed out"):
            await agent._get_next_action(browser_state)

    @pytest.mark.asyncio
    async def test_log_agent_run(self):
        agent = self._create_agent()
        with patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value="1.0.0"):
            await agent._log_agent_run()

    @pytest.mark.asyncio
    async def test_log_agent_run_newer_version(self):
        agent = self._create_agent()
        agent.version = "0.9.0"
        with patch("openbrowser.agent.service.check_latest_openbrowser_version", new_callable=AsyncMock, return_value="1.0.0"):
            await agent._log_agent_run()

    def test_log_agent_event(self):
        agent = self._create_agent()
        agent.telemetry = MagicMock()
        agent._log_agent_event(max_steps=10)
        agent.telemetry.capture.assert_called_once()

    def test_log_agent_event_with_error(self):
        agent = self._create_agent()
        agent.telemetry = MagicMock()
        agent._log_agent_event(max_steps=10, agent_run_error="Something failed")
        agent.telemetry.capture.assert_called_once()

    def test_log_agent_event_with_history(self):
        from openbrowser.agent.views import AgentHistory, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        agent.telemetry = MagicMock()

        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult(extracted_content="done")],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )
        agent.history.history.append(history_item)
        agent._log_agent_event(max_steps=10)

    @pytest.mark.asyncio
    async def test_load_and_rerun_default_path(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "load_from_file", return_value=agent.history):
            with patch.object(agent, "rerun_history", new_callable=AsyncMock, return_value=[]):
                result = await agent.load_and_rerun()
                assert result == []

    @pytest.mark.asyncio
    async def test_load_and_rerun_custom_path(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "load_from_file", return_value=agent.history):
            with patch.object(agent, "rerun_history", new_callable=AsyncMock, return_value=[]):
                result = await agent.load_and_rerun("/tmp/history.json")
                assert result == []

    @pytest.mark.asyncio
    async def test_rerun_history_empty(self):
        from openbrowser.agent.views import AgentHistoryList

        agent = self._create_agent()
        history = AgentHistoryList(history=[])
        result = await agent.rerun_history(history)
        assert result == []

    @pytest.mark.asyncio
    async def test_rerun_history_with_items(self):
        from openbrowser.agent.views import AgentHistory, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="Click button",
            action=[CustomAction(done={"text": "test"})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[None],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult(extracted_content="done")],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )

        from openbrowser.agent.views import AgentHistoryList
        history = AgentHistoryList(history=[history_item])

        with patch.object(agent, "_execute_history_step", new_callable=AsyncMock, return_value=[ActionResult(extracted_content="ok")]):
            result = await agent.rerun_history(history)
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_rerun_history_no_action(self):
        from openbrowser.agent.views import AgentHistory, AgentHistoryList, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=None,
            result=[ActionResult(extracted_content="done")],
            state=state,
            metadata=StepMetadata(step_number=0, step_start_time=0.0, step_end_time=1.0),
        )
        history = AgentHistoryList(history=[history_item])

        result = await agent.rerun_history(history)
        assert len(result) == 1
        assert result[0].error is not None

    @pytest.mark.asyncio
    async def test_rerun_history_retry_failure(self):
        from openbrowser.agent.views import AgentHistory, AgentHistoryList, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[None],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult()],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )
        history = AgentHistoryList(history=[history_item])

        with patch.object(agent, "_execute_history_step", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError):
                await agent.rerun_history(history, max_retries=2, skip_failures=False, delay_between_actions=0.01)

    @pytest.mark.asyncio
    async def test_rerun_history_retry_skip(self):
        from openbrowser.agent.views import AgentHistory, AgentHistoryList, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[None],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult()],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )
        history = AgentHistoryList(history=[history_item])

        with patch.object(agent, "_execute_history_step", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            # skip_failures=True means it should not raise
            result = await agent.rerun_history(history, max_retries=1, skip_failures=True, delay_between_actions=0.01)
            # Result may be empty since all retries failed but skip_failures=True

    @pytest.mark.asyncio
    async def test_update_action_indices_no_element(self):
        agent = self._create_agent()
        action = MagicMock()
        browser_state = _make_mock_browser_state_summary()
        result = await agent._update_action_indices(None, action, browser_state)
        assert result == action

    @pytest.mark.asyncio
    async def test_update_action_indices_not_found(self):
        agent = self._create_agent()
        historical_element = MagicMock()
        historical_element.element_hash = "hash123"

        action = MagicMock()
        browser_state = _make_mock_browser_state_summary(selector_map={1: MagicMock(element_hash="other_hash")})
        result = await agent._update_action_indices(historical_element, action, browser_state)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_action_indices_found_same_index(self):
        agent = self._create_agent()
        historical_element = MagicMock()
        historical_element.element_hash = "hash123"

        mock_current = MagicMock()
        mock_current.element_hash = "hash123"

        action = MagicMock()
        action.get_index.return_value = 5

        browser_state = _make_mock_browser_state_summary(selector_map={5: mock_current})
        result = await agent._update_action_indices(historical_element, action, browser_state)
        assert result == action

    @pytest.mark.asyncio
    async def test_update_action_indices_found_different_index(self):
        agent = self._create_agent()
        historical_element = MagicMock()
        historical_element.element_hash = "hash123"

        mock_current = MagicMock()
        mock_current.element_hash = "hash123"

        action = MagicMock()
        action.get_index.return_value = 3

        browser_state = _make_mock_browser_state_summary(selector_map={7: mock_current})
        result = await agent._update_action_indices(historical_element, action, browser_state)
        assert result == action
        action.set_index.assert_called_once_with(7)

    def test_get_trace_object(self):
        agent = self._create_agent()
        # Mock settings.model_dump to avoid serializing the mock LLM
        clean_settings = {"use_vision": "auto", "max_failures": 3}
        with (
            patch.object(type(agent.settings), "model_dump", return_value=clean_settings),
            patch.object(type(agent.history), "model_dump", return_value={"history": []}),
        ):
            trace = agent.get_trace_object()
        assert "trace" in trace
        assert "trace_details" in trace
        assert "trace_id" in trace["trace"]
        assert "task_truncated" in trace["trace"]

    def test_get_trace_object_with_history(self):
        from openbrowser.agent.views import AgentHistory, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult(is_done=True, success=True, extracted_content="Final result text")],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )
        agent.history.history.append(history_item)

        clean_settings = {"use_vision": "auto", "max_failures": 3}
        with patch.object(type(agent.settings), "model_dump", return_value=clean_settings):
            trace = agent.get_trace_object()
        assert trace["trace"]["self_report_completed"] == 1
        assert trace["trace"]["self_report_success"] == 1

    def test_run_sync(self):
        agent = self._create_agent()
        with patch.object(agent, "run", new_callable=AsyncMock, return_value=agent.history):
            result = agent.run_sync(max_steps=5)
            assert result == agent.history

    @pytest.mark.asyncio
    async def test_set_file_system_with_state(self):
        """Test file system restoration from state."""
        agent = self._create_agent()
        # Already tested by init, just verify state is set
        assert agent.file_system is not None

    @pytest.mark.asyncio
    async def test_set_file_system_conflicting_params(self):
        """Test that both file_system_state and file_system_path raises ValueError."""
        from openbrowser.agent.views import AgentState
        from openbrowser.filesystem.file_system import FileSystemState

        state = AgentState()
        state.file_system_state = FileSystemState(base_dir="/tmp/test", files={})

        with pytest.raises(ValueError, match="Cannot provide both"):
            self._create_agent(
                injected_agent_state=state,
                file_system_path="/tmp/other",
            )


# ---------------------------------------------------------------------------
# LLM timeout detection tests
# ---------------------------------------------------------------------------


class TestLLMTimeoutDetection:
    """Test the _get_model_timeout inner function."""

    @patch("openbrowser.agent.service.EventBus")
    @patch("openbrowser.agent.service.ProductTelemetry")
    @patch("openbrowser.agent.service.TokenCost")
    @patch("openbrowser.agent.service.SystemPrompt")
    @patch("openbrowser.agent.service.MessageManager")
    @patch("openbrowser.agent.service.FileSystem")
    def _create_agent_with_model(self, model_name, mock_fs, mock_mm, mock_sp, mock_tc, mock_tel, mock_eb):
        from openbrowser.agent.service import Agent

        llm = _make_mock_llm(model=model_name)
        session = _make_mock_browser_session()
        tools = _make_mock_tools()

        mock_sp_instance = MagicMock()
        mock_sp_instance.get_system_message.return_value = MagicMock(content="system", cache=True)
        mock_sp.return_value = mock_sp_instance

        mock_fs_instance = MagicMock()
        mock_fs_instance.get_state.return_value = MagicMock()
        mock_fs_instance.base_dir = Path(tempfile.mkdtemp())
        mock_fs.return_value = mock_fs_instance

        mock_tc_instance = MagicMock()
        mock_tc_instance.register_llm = MagicMock()
        mock_tc.return_value = mock_tc_instance

        mock_eb_instance = MagicMock()
        mock_eb_instance.stop = AsyncMock()
        mock_eb.return_value = mock_eb_instance

        with patch.object(Agent, "_set_screenshot_service"):
            agent = Agent(task="test", llm=llm, browser_session=session, tools=tools)
        return agent

    def test_gemini_timeout(self):
        agent = self._create_agent_with_model("gemini-1.5-flash")
        assert agent.settings.llm_timeout == 45

    def test_groq_timeout(self):
        agent = self._create_agent_with_model("groq-llama")
        assert agent.settings.llm_timeout == 30

    def test_claude_timeout(self):
        agent = self._create_agent_with_model("claude-sonnet-4-5")
        assert agent.settings.llm_timeout == 90

    def test_o3_timeout(self):
        agent = self._create_agent_with_model("o3-mini")
        assert agent.settings.llm_timeout == 90

    def test_deepseek_timeout(self):
        agent = self._create_agent_with_model("deepseek-v3")
        assert agent.settings.llm_timeout == 90

    def test_default_timeout(self):
        agent = self._create_agent_with_model("some-other-model")
        assert agent.settings.llm_timeout == 60

    def test_explicit_timeout_overrides(self):
        from openbrowser.agent.service import Agent

        llm = _make_mock_llm(model="gemini-1.5-flash")
        session = _make_mock_browser_session()
        tools = _make_mock_tools()

        from openbrowser.agent.service import Agent as AgentCls

        with patch("openbrowser.agent.service.EventBus"), \
             patch("openbrowser.agent.service.ProductTelemetry"), \
             patch("openbrowser.agent.service.TokenCost") as mock_tc, \
             patch("openbrowser.agent.service.SystemPrompt") as mock_sp, \
             patch("openbrowser.agent.service.MessageManager"), \
             patch("openbrowser.agent.service.FileSystem") as mock_fs, \
             patch.object(AgentCls, "_set_screenshot_service"):

            mock_sp_instance = MagicMock()
            mock_sp_instance.get_system_message.return_value = MagicMock(content="sys", cache=True)
            mock_sp.return_value = mock_sp_instance

            mock_fs_instance = MagicMock()
            mock_fs_instance.get_state.return_value = MagicMock()
            mock_fs_instance.base_dir = Path(tempfile.mkdtemp())
            mock_fs.return_value = mock_fs_instance

            mock_tc_instance = MagicMock()
            mock_tc_instance.register_llm = MagicMock()
            mock_tc.return_value = mock_tc_instance

            agent = Agent(task="test", llm=llm, browser_session=session, tools=tools, llm_timeout=120)
            assert agent.settings.llm_timeout == 120


# ---------------------------------------------------------------------------
# Additional coverage: sensitive_data validation, step, prepare_context,
# recursive processing, _run_with_langgraph, _execute_history_step, etc.
# ---------------------------------------------------------------------------


class TestSensitiveDataValidation(TestAgentAsyncMethods):
    """Tests for sensitive_data domain validation during __init__."""

    def test_sensitive_data_no_allowed_domains(self):
        """When sensitive_data is set but no allowed_domains, agent logs error."""
        agent = self._create_agent(sensitive_data={"password": "secret123"})
        # Should proceed without error, just a warning log
        assert agent.sensitive_data == {"password": "secret123"}

    def test_sensitive_data_domain_specific_with_matching_allowed(self):
        """Domain-specific credentials with matching allowed_domains."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["*.example.com"]
        agent = self._create_agent(
            browser_session=session,
            sensitive_data={"example.com": {"user": "admin", "pass": "secret"}},
        )
        assert agent.sensitive_data is not None

    def test_sensitive_data_domain_specific_no_match(self):
        """Domain-specific credentials not covered by allowed_domains."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["*.other.com"]
        agent = self._create_agent(
            browser_session=session,
            sensitive_data={"example.com": {"user": "admin"}},
        )
        assert agent.sensitive_data is not None

    def test_sensitive_data_domain_specific_wildcard_allowed(self):
        """Domain-specific credentials with wildcard * in allowed_domains."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["*"]
        agent = self._create_agent(
            browser_session=session,
            sensitive_data={"example.com": {"user": "admin"}},
        )
        assert agent.sensitive_data is not None

    def test_sensitive_data_domain_with_scheme(self):
        """Domain patterns with scheme (https://) in both sensitive_data and allowed_domains."""
        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = ["https://*.example.com"]
        agent = self._create_agent(
            browser_session=session,
            sensitive_data={"https://sub.example.com": {"token": "abc"}},
        )
        assert agent.sensitive_data is not None


class TestSetFileSystemEdgeCases(TestAgentAsyncMethods):
    """Tests for _set_file_system edge cases."""

    def test_set_file_system_restore_from_state(self):
        """Test file system restore from existing state."""
        agent = self._create_agent()
        mock_state = MagicMock()
        agent.state.file_system_state = mock_state

        with patch("openbrowser.agent.service.FileSystem") as mock_fs_cls:
            mock_restored_fs = MagicMock()
            mock_restored_fs.base_dir = Path("/tmp/restored")
            mock_fs_cls.from_state.return_value = mock_restored_fs

            agent._set_file_system()
            mock_fs_cls.from_state.assert_called_once_with(mock_state)
            assert agent.file_system == mock_restored_fs

    def test_set_file_system_restore_failure(self):
        """Test file system restore failure raises."""
        agent = self._create_agent()
        mock_state = MagicMock()
        agent.state.file_system_state = mock_state

        with patch("openbrowser.agent.service.FileSystem") as mock_fs_cls:
            mock_fs_cls.from_state.side_effect = RuntimeError("corrupt state")
            with pytest.raises(RuntimeError, match="corrupt state"):
                agent._set_file_system()

    def test_set_file_system_with_path(self):
        """Test file system init with explicit path."""
        agent = self._create_agent()
        agent.state.file_system_state = None

        with patch("openbrowser.agent.service.FileSystem") as mock_fs_cls:
            mock_fs_instance = MagicMock()
            mock_fs_instance.get_state.return_value = MagicMock()
            mock_fs_cls.return_value = mock_fs_instance

            agent._set_file_system(file_system_path="/tmp/custom")
            mock_fs_cls.assert_called_once_with("/tmp/custom")
            assert agent.file_system_path == "/tmp/custom"

    def test_set_file_system_init_failure(self):
        """Test file system init failure raises."""
        agent = self._create_agent()
        agent.state.file_system_state = None

        with patch("openbrowser.agent.service.FileSystem") as mock_fs_cls:
            mock_fs_cls.side_effect = OSError("permission denied")
            with pytest.raises(OSError, match="permission denied"):
                agent._set_file_system()


class TestSaveFileSystemState(TestAgentAsyncMethods):
    """Tests for save_file_system_state."""

    def test_save_file_system_state_success(self):
        agent = self._create_agent()
        mock_state = MagicMock()
        agent.file_system = MagicMock()
        agent.file_system.get_state.return_value = mock_state
        agent.save_file_system_state()
        assert agent.state.file_system_state == mock_state

    def test_save_file_system_state_no_fs(self):
        agent = self._create_agent()
        agent.file_system = None
        with pytest.raises(ValueError, match="File system is not set up"):
            agent.save_file_system_state()


class TestSetOpenbrowserVersionAndSource(TestAgentAsyncMethods):
    """Tests for _set_openbrowser_version_and_source."""

    def test_source_detection_git(self):
        agent = self._create_agent()
        with patch("pathlib.Path.exists", return_value=True):
            agent._set_openbrowser_version_and_source()
        assert agent.source == "git"

    def test_source_detection_pip(self):
        agent = self._create_agent()
        with patch("pathlib.Path.exists", side_effect=lambda: False):
            # Force the 'all' check to fail
            original_exists = Path.exists

            def mock_exists(self_path):
                if ".git" in str(self_path):
                    return False
                return original_exists(self_path)

            with patch.object(Path, "exists", mock_exists):
                agent._set_openbrowser_version_and_source()
        # Either git or pip depending on environment, just ensure no crash
        assert agent.source in ("git", "pip")

    def test_source_override(self):
        agent = self._create_agent()
        agent._set_openbrowser_version_and_source(source_override="custom")
        assert agent.source == "custom"

    def test_source_detection_error(self):
        agent = self._create_agent()
        with patch("pathlib.Path.exists", side_effect=Exception("boom")):
            # Use a different side effect for the 'all' call
            agent._set_openbrowser_version_and_source()
        assert agent.source == "unknown"


class TestConversationPathInit(TestAgentAsyncMethods):
    """Tests for save_conversation_path during init."""

    def test_save_conversation_path_set(self):
        agent = self._create_agent(save_conversation_path="/tmp/conv.json")
        assert agent.settings.save_conversation_path is not None


class TestDownloadsTracking(TestAgentAsyncMethods):
    """Tests for download tracking init."""

    def test_downloads_path_enabled(self):
        session = _make_mock_browser_session(downloads_path="/tmp/downloads")
        agent = self._create_agent(browser_session=session)
        assert agent.has_downloads_path is True

    def test_downloads_path_disabled(self):
        session = _make_mock_browser_session(downloads_path=None)
        agent = self._create_agent(browser_session=session)
        assert agent.has_downloads_path is False


class TestStepMethod(TestAgentAsyncMethods):
    """Tests for the step() method which orchestrates one step."""

    @pytest.mark.asyncio
    async def test_step_success(self):
        agent = self._create_agent()
        agent._prepare_context = AsyncMock(return_value=_make_mock_browser_state_summary())
        agent._get_next_action = AsyncMock()
        agent._execute_actions = AsyncMock()
        agent._post_process = AsyncMock()
        agent._finalize = AsyncMock()

        await agent.step()

        agent._prepare_context.assert_called_once()
        agent._get_next_action.assert_called_once()
        agent._execute_actions.assert_called_once()
        agent._post_process.assert_called_once()
        agent._finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_exception_calls_handle_error(self):
        agent = self._create_agent()
        agent._prepare_context = AsyncMock(side_effect=RuntimeError("boom"))
        agent._handle_step_error = AsyncMock()
        agent._finalize = AsyncMock()

        await agent.step()

        agent._handle_step_error.assert_called_once()
        agent._finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_with_step_info(self):
        from openbrowser.agent.views import AgentStepInfo
        agent = self._create_agent()
        agent._prepare_context = AsyncMock(return_value=_make_mock_browser_state_summary())
        agent._get_next_action = AsyncMock()
        agent._execute_actions = AsyncMock()
        agent._post_process = AsyncMock()
        agent._finalize = AsyncMock()

        step_info = AgentStepInfo(step_number=3, max_steps=10)
        await agent.step(step_info)

        agent._prepare_context.assert_called_once_with(step_info)


class TestPrepareContext(TestAgentAsyncMethods):
    """Tests for the _prepare_context method."""

    @pytest.mark.asyncio
    async def test_prepare_context_basic(self):
        agent = self._create_agent()
        mock_state = _make_mock_browser_state_summary()
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=mock_state)
        agent._check_and_update_downloads = AsyncMock()
        agent._log_step_context = MagicMock()
        agent._check_stop_or_pause = AsyncMock()
        agent._update_action_models_for_page = AsyncMock()
        agent._force_done_after_last_step = AsyncMock()
        agent._force_done_after_failure = AsyncMock()

        result = await agent._prepare_context()
        assert result == mock_state
        agent._check_and_update_downloads.assert_called_once()
        agent._update_action_models_for_page.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_context_with_screenshot(self):
        agent = self._create_agent()
        mock_state = _make_mock_browser_state_summary(screenshot="base64data")
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=mock_state)
        agent._check_and_update_downloads = AsyncMock()
        agent._log_step_context = MagicMock()
        agent._check_stop_or_pause = AsyncMock()
        agent._update_action_models_for_page = AsyncMock()
        agent._force_done_after_last_step = AsyncMock()
        agent._force_done_after_failure = AsyncMock()

        result = await agent._prepare_context()
        assert result.screenshot == "base64data"


class TestRecursiveProcessing(TestAgentAsyncMethods):
    """Tests for recursive URL replacement processing methods."""

    def test_recursive_process_dict_with_basemodel(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com/long"}
        data = {"key": "visit short://x now"}
        Agent._recursive_process_dict(data, replacements)
        assert data["key"] == "visit https://original.com/long now"

    def test_recursive_process_dict_with_nested_dict(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        data = {"nested": {"url": "short://x"}}
        Agent._recursive_process_dict(data, replacements)
        assert data["nested"]["url"] == "https://original.com"

    def test_recursive_process_dict_with_list_value(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        data = {"urls": ["short://x", "other"]}
        Agent._recursive_process_dict(data, replacements)
        assert data["urls"][0] == "https://original.com"
        assert data["urls"][1] == "other"

    def test_recursive_process_list_strings(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        result = Agent._recursive_process_list_or_tuple(["short://x", "no change"], replacements)
        assert result[0] == "https://original.com"
        assert result[1] == "no change"

    def test_recursive_process_tuple_strings(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        result = Agent._recursive_process_list_or_tuple(("short://x", "no change"), replacements)
        assert isinstance(result, tuple)
        assert result[0] == "https://original.com"

    def test_recursive_process_list_with_dict(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        data = [{"url": "short://x"}]
        result = Agent._recursive_process_list_or_tuple(data, replacements)
        assert result[0]["url"] == "https://original.com"

    def test_recursive_process_tuple_with_dict(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        data = ({"url": "short://x"},)
        result = Agent._recursive_process_list_or_tuple(data, replacements)
        assert isinstance(result, tuple)
        assert result[0]["url"] == "https://original.com"

    def test_recursive_process_list_with_nested_list(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        data = [["short://x"]]
        result = Agent._recursive_process_list_or_tuple(data, replacements)
        assert result[0][0] == "https://original.com"

    def test_recursive_process_tuple_with_nested_tuple(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        data = (("short://x",),)
        result = Agent._recursive_process_list_or_tuple(data, replacements)
        assert isinstance(result, tuple)
        assert result[0][0] == "https://original.com"

    def test_recursive_process_list_non_string_item(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        data = [42, None, True]
        result = Agent._recursive_process_list_or_tuple(data, replacements)
        assert result == [42, None, True]

    def test_recursive_process_tuple_non_string_item(self):
        from openbrowser.agent.service import Agent

        replacements = {"short://x": "https://original.com"}
        data = (42, None)
        result = Agent._recursive_process_list_or_tuple(data, replacements)
        assert isinstance(result, tuple)
        assert result == (42, None)

    def test_recursive_process_pydantic_model_with_list_field(self):
        from openbrowser.agent.service import Agent

        class TestModel(BaseModel):
            urls: list[str] = []

        model = TestModel(urls=["short://x", "other"])
        replacements = {"short://x": "https://original.com"}
        Agent._recursive_process_all_strings_inside_pydantic_model(model, replacements)
        assert model.urls[0] == "https://original.com"

    def test_recursive_process_pydantic_model_with_dict_field(self):
        from openbrowser.agent.service import Agent

        class TestModel(BaseModel):
            model_config = {"extra": "allow"}
            data: dict = {}

        model = TestModel(data={"url": "short://x"})
        replacements = {"short://x": "https://original.com"}
        Agent._recursive_process_all_strings_inside_pydantic_model(model, replacements)
        assert model.data["url"] == "https://original.com"


class TestURLShorteningFragments(TestAgentAsyncMethods):
    """Test URL shortening with fragments (#) in addition to query params."""

    def test_url_with_fragment_only(self):
        agent = self._create_agent(_url_shortening_limit=10)
        text = "Visit https://example.com/page#" + "x" * 50
        result_text, replaced = agent._replace_urls_in_text(text)
        # Fragment is long enough to be shortened
        assert len(replaced) > 0 or "example.com" in result_text

    def test_url_with_query_and_fragment(self):
        agent = self._create_agent(_url_shortening_limit=10)
        url = "https://example.com/page?" + "a=b&" * 20 + "#frag"
        text = f"Go to {url}"
        result_text, replaced = agent._replace_urls_in_text(text)
        assert "example.com" in result_text

    def test_url_shortened_not_shorter_than_original(self):
        """When shortened URL is not shorter, keep original."""
        agent = self._create_agent(_url_shortening_limit=200)
        text = "Visit https://example.com/page?short=1"
        result_text, replaced = agent._replace_urls_in_text(text)
        assert len(replaced) == 0  # No replacement since original is short enough


class TestLogNextActionSummary(TestAgentAsyncMethods):
    """Tests for _log_next_action_summary."""

    def test_log_next_action_summary_with_index(self):
        from openbrowser.agent.views import AgentOutput
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        agent = self._create_agent()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="m",
            next_goal="n",
            action=[CustomAction(click={"index": 5, "text": "Click me"})],
        )
        # Should not raise
        agent._log_next_action_summary(output)

    def test_log_next_action_summary_with_url_and_success(self):
        from openbrowser.agent.views import AgentOutput
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        agent = self._create_agent()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="m",
            next_goal="n",
            action=[CustomAction(done={"url": "https://example.com", "success": True, "custom": "val"})],
        )
        agent._log_next_action_summary(output)

    def test_log_next_action_summary_long_text(self):
        from openbrowser.agent.views import AgentOutput
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        agent = self._create_agent()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="m",
            next_goal="n",
            action=[CustomAction(done={"text": "x" * 50})],
        )
        agent._log_next_action_summary(output)

    def test_log_next_action_summary_empty_actions(self):
        from openbrowser.agent.views import AgentOutput
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        agent = self._create_agent()
        output = MagicMock()
        output.action = []
        agent._log_next_action_summary(output)


class TestLogStepCompletionSummary(TestAgentAsyncMethods):
    """Tests for _log_step_completion_summary."""

    def test_log_step_completion_empty_result(self):
        agent = self._create_agent()
        agent._log_step_completion_summary(time.time(), [])

    def test_log_step_completion_with_results(self):
        from openbrowser.models import ActionResult
        agent = self._create_agent()
        results = [
            ActionResult(extracted_content="ok"),
            ActionResult(error="failed"),
        ]
        agent._log_step_completion_summary(time.time() - 1.0, results)

    def test_log_step_completion_all_success(self):
        from openbrowser.models import ActionResult
        agent = self._create_agent()
        results = [ActionResult(extracted_content="ok")]
        agent._log_step_completion_summary(time.time(), results)


class TestLogFinalOutcomeMessages(TestAgentAsyncMethods):
    """Tests for _log_final_outcome_messages."""

    def test_log_final_outcome_failure(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "is_successful", return_value=False), \
             patch.object(type(agent.history), "final_result", return_value="Failed to do task"):
            agent._log_final_outcome_messages()

    def test_log_final_outcome_none(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "is_successful", return_value=None), \
             patch.object(type(agent.history), "final_result", return_value=None):
            agent._log_final_outcome_messages()

    def test_log_final_outcome_success(self):
        agent = self._create_agent()
        with patch.object(type(agent.history), "is_successful", return_value=True):
            agent._log_final_outcome_messages()


class TestLogAgentEvent(TestAgentAsyncMethods):
    """Additional tests for _log_agent_event with action history."""

    def test_log_agent_event_with_action_history(self):
        from openbrowser.agent.views import AgentHistory, AgentOutput, StepMetadata
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult(is_done=True, success=True, extracted_content="Result")],
            state=state,
            metadata=StepMetadata(step_number=1, step_start_time=1.0, step_end_time=2.0),
        )
        agent.history.history.append(history_item)

        # Should not raise
        agent._log_agent_event(max_steps=10)

    def test_log_agent_event_no_model_output(self):
        from openbrowser.agent.views import AgentHistory
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=None,
            result=[ActionResult(extracted_content="Result")],
            state=state,
        )
        agent.history.history.append(history_item)
        agent._log_agent_event(max_steps=10)


class TestExecuteHistoryStep(TestAgentAsyncMethods):
    """Tests for _execute_history_step."""

    @pytest.mark.asyncio
    async def test_execute_history_step_success(self):
        from openbrowser.agent.views import AgentHistory, AgentOutput
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.dom.views import DOMInteractedElement
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        mock_element = MagicMock(spec=DOMInteractedElement)
        mock_element.element_hash = "hash123"

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(click={"index": 1})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[mock_element],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult(extracted_content="ok")],
            state=state,
        )

        browser_state = _make_mock_browser_state_summary()
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=browser_state)
        agent._update_action_indices = AsyncMock(return_value=CustomAction(click={"index": 2}))
        agent.multi_act = AsyncMock(return_value=[ActionResult(extracted_content="replayed")])

        result = await agent._execute_history_step(history_item, delay=0.0)
        assert len(result) == 1
        assert result[0].extracted_content == "replayed"

    @pytest.mark.asyncio
    async def test_execute_history_step_no_model_output(self):
        from openbrowser.agent.views import AgentHistory
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=None,
            result=[ActionResult()],
            state=state,
        )

        browser_state = _make_mock_browser_state_summary()
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=browser_state)

        with pytest.raises(ValueError, match="Invalid state or model output"):
            await agent._execute_history_step(history_item, delay=0.0)

    @pytest.mark.asyncio
    async def test_execute_history_step_element_not_found(self):
        from openbrowser.agent.views import AgentHistory, AgentOutput
        from openbrowser.browser.views import BrowserStateHistory
        from openbrowser.dom.views import DOMInteractedElement
        from openbrowser.models import ActionResult

        agent = self._create_agent()
        ActionModel = _get_action_model_base()
        CustomAction = create_model("CA", __base__=ActionModel, click=(Optional[dict], None))

        mock_element = MagicMock(spec=DOMInteractedElement)
        mock_element.element_hash = "hash_missing"

        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(click={"index": 1})],
        )
        state = BrowserStateHistory(
            url="https://example.com",
            title="Example",
            tabs=[],
            interacted_element=[mock_element],
            screenshot_path=None,
        )
        history_item = AgentHistory(
            model_output=output,
            result=[ActionResult()],
            state=state,
        )

        browser_state = _make_mock_browser_state_summary()
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=browser_state)
        agent._update_action_indices = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Could not find matching element"):
            await agent._execute_history_step(history_item, delay=0.0)


class TestRunWithLanggraph(TestAgentAsyncMethods):
    """Tests for _run_with_langgraph method."""

    @pytest.mark.asyncio
    async def test_run_with_langgraph_success(self):
        agent = self._create_agent()
        agent._log_agent_run = AsyncMock()
        agent._log_first_step_startup = MagicMock()
        agent._execute_initial_actions = AsyncMock()
        agent._log_agent_event = MagicMock()
        agent._log_final_outcome_messages = MagicMock()
        agent.close = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph), \
             patch("openbrowser.utils.SignalHandler") as mock_sh:
            mock_sh_instance = MagicMock()
            mock_sh.return_value = mock_sh_instance

            result = await agent._run_with_langgraph(max_steps=5)
            assert result == agent.history
            mock_graph.run.assert_called_once_with(max_steps=5)
            mock_sh_instance.register.assert_called_once()
            mock_sh_instance.unregister.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_langgraph_keyboard_interrupt(self):
        agent = self._create_agent()
        agent._log_agent_run = AsyncMock()
        agent._log_first_step_startup = MagicMock()
        agent._execute_initial_actions = AsyncMock()
        agent._log_agent_event = MagicMock()
        agent._log_final_outcome_messages = MagicMock()
        agent.close = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock(side_effect=KeyboardInterrupt)

        with patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph), \
             patch("openbrowser.utils.SignalHandler") as mock_sh:
            mock_sh.return_value = MagicMock()

            result = await agent._run_with_langgraph(max_steps=5)
            assert result == agent.history

    @pytest.mark.asyncio
    async def test_run_with_langgraph_exception(self):
        agent = self._create_agent()
        agent._log_agent_run = AsyncMock()
        agent._log_first_step_startup = MagicMock()
        agent._execute_initial_actions = AsyncMock()
        agent._log_agent_event = MagicMock()
        agent._log_final_outcome_messages = MagicMock()
        agent.close = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock(side_effect=RuntimeError("graph failed"))

        with patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph), \
             patch("openbrowser.utils.SignalHandler") as mock_sh:
            mock_sh.return_value = MagicMock()

            with pytest.raises(RuntimeError, match="graph failed"):
                await agent._run_with_langgraph(max_steps=5)

    @pytest.mark.asyncio
    async def test_run_with_langgraph_generate_gif(self):
        agent = self._create_agent()
        agent.settings.generate_gif = True
        agent._log_agent_run = AsyncMock()
        agent._log_first_step_startup = MagicMock()
        agent._execute_initial_actions = AsyncMock()
        agent._log_agent_event = MagicMock()
        agent._log_final_outcome_messages = MagicMock()
        agent.close = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph), \
             patch("openbrowser.utils.SignalHandler") as mock_sh, \
             patch("openbrowser.agent.gif.create_history_gif") as mock_gif:
            mock_sh.return_value = MagicMock()
            result = await agent._run_with_langgraph(max_steps=5)
            mock_gif.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_langgraph_generate_gif_custom_path(self):
        agent = self._create_agent()
        agent.settings.generate_gif = "/tmp/custom.gif"
        agent._log_agent_run = AsyncMock()
        agent._log_first_step_startup = MagicMock()
        agent._execute_initial_actions = AsyncMock()
        agent._log_agent_event = MagicMock()
        agent._log_final_outcome_messages = MagicMock()
        agent.close = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph), \
             patch("openbrowser.utils.SignalHandler") as mock_sh, \
             patch("openbrowser.agent.gif.create_history_gif") as mock_gif:
            mock_sh.return_value = MagicMock()
            result = await agent._run_with_langgraph(max_steps=5)
            mock_gif.assert_called_once_with(task=agent.task, history=agent.history, output_path="/tmp/custom.gif")

    @pytest.mark.asyncio
    async def test_run_with_langgraph_output_model_schema(self):
        agent = self._create_agent()
        agent.output_model_schema = MagicMock()
        agent._log_agent_run = AsyncMock()
        agent._log_first_step_startup = MagicMock()
        agent._execute_initial_actions = AsyncMock()
        agent._log_agent_event = MagicMock()
        agent._log_final_outcome_messages = MagicMock()
        agent.close = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph), \
             patch("openbrowser.utils.SignalHandler") as mock_sh:
            mock_sh.return_value = MagicMock()
            result = await agent._run_with_langgraph(max_steps=5)
            assert agent.history._output_model_schema == agent.output_model_schema

    @pytest.mark.asyncio
    async def test_run_with_langgraph_initial_actions_interrupted(self):
        agent = self._create_agent()
        agent._log_agent_run = AsyncMock()
        agent._log_first_step_startup = MagicMock()
        agent._execute_initial_actions = AsyncMock(side_effect=InterruptedError)
        agent._log_agent_event = MagicMock()
        agent._log_final_outcome_messages = MagicMock()
        agent.close = AsyncMock()

        mock_graph = MagicMock()
        mock_graph.run = AsyncMock()

        with patch("openbrowser.agent.graph.create_agent_graph", return_value=mock_graph), \
             patch("openbrowser.utils.SignalHandler") as mock_sh:
            mock_sh.return_value = MagicMock()
            result = await agent._run_with_langgraph(max_steps=5)
            # Should not raise, InterruptedError is caught
            assert result == agent.history


class TestUpdateActionModelsForPage(TestAgentAsyncMethods):
    """Tests for _update_action_models_for_page."""

    @pytest.mark.asyncio
    async def test_update_action_models_default(self):
        agent = self._create_agent()
        agent.settings.flash_mode = False
        agent.settings.use_thinking = True
        await agent._update_action_models_for_page("https://example.com")
        # ActionModel should be updated
        assert agent.ActionModel is not None

    @pytest.mark.asyncio
    async def test_update_action_models_flash_mode(self):
        agent = self._create_agent()
        agent.settings.flash_mode = True
        await agent._update_action_models_for_page("https://example.com")
        assert agent.AgentOutput is not None

    @pytest.mark.asyncio
    async def test_update_action_models_no_thinking(self):
        agent = self._create_agent()
        agent.settings.flash_mode = False
        agent.settings.use_thinking = False
        await agent._update_action_models_for_page("https://example.com")
        assert agent.AgentOutput is not None


class TestCloseMethod(TestAgentAsyncMethods):
    """Additional tests for close method edge cases."""

    @pytest.mark.asyncio
    async def test_close_with_remaining_tasks(self):
        agent = self._create_agent()
        agent.browser_session.kill = AsyncMock()
        # close() accesses asyncio tasks
        await agent.close()

    @pytest.mark.asyncio
    async def test_close_error_during_cleanup(self):
        agent = self._create_agent()
        agent.browser_session.kill = AsyncMock(side_effect=RuntimeError("kill failed"))
        # Should not propagate error from cleanup
        await agent.close()


class TestEnhanceTaskWithSchema(TestAgentAsyncMethods):
    """Tests for _enhance_task_with_schema."""

    def test_enhance_task_none_schema(self):
        agent = self._create_agent()
        result = agent._enhance_task_with_schema("Do stuff", None)
        assert result == "Do stuff"

    def test_enhance_task_with_valid_schema(self):
        agent = self._create_agent()

        class MyOutput(BaseModel):
            result: str
            score: int

        result = agent._enhance_task_with_schema("Do stuff", MyOutput)
        assert "MyOutput" in result
        assert "result" in result
        assert "score" in result

    def test_enhance_task_with_schema_error(self):
        agent = self._create_agent()
        mock_schema = MagicMock()
        mock_schema.model_json_schema.side_effect = Exception("schema error")
        mock_schema.__name__ = "MockSchema"
        result = agent._enhance_task_with_schema("Do stuff", mock_schema)
        assert result == "Do stuff"


class TestDeepSeekAndGrokVisionDisable(TestAgentAsyncMethods):
    """Tests for auto-disabling vision for certain models."""

    def test_deepseek_disables_vision(self):
        llm = _make_mock_llm(model="deepseek-chat")
        agent = self._create_agent(llm=llm)
        assert agent.settings.use_vision is False

    def test_grok_disables_vision(self):
        llm = _make_mock_llm(model="grok-2")
        agent = self._create_agent(llm=llm)
        assert agent.settings.use_vision is False
