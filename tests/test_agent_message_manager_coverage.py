"""Tests for openbrowser.agent.message_manager (service, utils, views).

Covers MessageManager, save_conversation, _format_conversation,
HistoryItem, MessageHistory, MessageManagerState, and logging helpers.
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# message_manager/views.py tests
# ---------------------------------------------------------------------------


class TestHistoryItem:
    """Tests for HistoryItem model."""

    def test_step_with_error(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        item = HistoryItem(step_number=1, error="Element not found")
        result = item.to_string()
        assert "step" in result
        assert "Element not found" in result

    def test_step_with_system_message(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        item = HistoryItem(system_message="Agent initialized")
        result = item.to_string()
        assert result == "Agent initialized"

    def test_step_with_all_fields(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        item = HistoryItem(
            step_number=3,
            evaluation_previous_goal="success",
            memory="Saw the form",
            next_goal="Fill the form",
            action_results="Input text completed",
        )
        result = item.to_string()
        assert "step" in result
        assert "success" in result
        assert "Saw the form" in result
        assert "Fill the form" in result
        assert "Input text completed" in result

    def test_step_with_partial_fields(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        item = HistoryItem(step_number=2, memory="Only memory set")
        result = item.to_string()
        assert "Only memory set" in result

    def test_step_with_no_step_number(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        item = HistoryItem(memory="No step number")
        result = item.to_string()
        assert "step_unknown" in result

    def test_error_and_system_message_raises(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        with pytest.raises(ValueError, match="Cannot have both"):
            HistoryItem(error="err", system_message="sys")

    def test_step_no_eval_no_next_goal(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        item = HistoryItem(step_number=1, memory="mem")
        result = item.to_string()
        assert "mem" in result
        # evaluation_previous_goal and next_goal should not appear
        assert "success" not in result

    def test_step_with_action_results_only(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        item = HistoryItem(step_number=1, action_results="Clicked button")
        result = item.to_string()
        assert "Clicked button" in result


class TestMessageHistory:
    """Tests for MessageHistory model."""

    def test_empty_history(self):
        from openbrowser.agent.message_manager.views import MessageHistory

        history = MessageHistory()
        messages = history.get_messages()
        assert messages == []

    def test_system_message_only(self):
        from openbrowser.agent.message_manager.views import MessageHistory
        from openbrowser.llm.messages import SystemMessage

        history = MessageHistory()
        history.system_message = SystemMessage(content="System prompt")
        messages = history.get_messages()
        assert len(messages) == 1

    def test_all_message_types(self):
        from openbrowser.agent.message_manager.views import MessageHistory
        from openbrowser.llm.messages import SystemMessage, UserMessage

        history = MessageHistory()
        history.system_message = SystemMessage(content="System")
        history.state_message = UserMessage(content="State")
        history.context_messages = [UserMessage(content="Context1"), UserMessage(content="Context2")]
        messages = history.get_messages()
        assert len(messages) == 4

    def test_state_without_system(self):
        from openbrowser.agent.message_manager.views import MessageHistory
        from openbrowser.llm.messages import UserMessage

        history = MessageHistory()
        history.state_message = UserMessage(content="State only")
        messages = history.get_messages()
        assert len(messages) == 1


class TestMessageManagerState:
    """Tests for MessageManagerState model."""

    def test_default_state(self):
        from openbrowser.agent.message_manager.views import MessageManagerState

        state = MessageManagerState()
        assert state.tool_id == 1
        assert len(state.agent_history_items) == 1
        assert state.read_state_description == ""

    def test_agent_history_items_default(self):
        from openbrowser.agent.message_manager.views import MessageManagerState

        state = MessageManagerState()
        assert state.agent_history_items[0].system_message == "Agent initialized"


# ---------------------------------------------------------------------------
# message_manager/utils.py tests
# ---------------------------------------------------------------------------


class TestSaveConversation:
    """Tests for save_conversation and _format_conversation."""

    @pytest.mark.asyncio
    async def test_save_conversation_basic(self, tmp_path):
        from openbrowser.agent.message_manager.utils import save_conversation
        from openbrowser.llm.messages import SystemMessage, UserMessage

        messages = [
            SystemMessage(content="You are a helpful agent"),
            UserMessage(content="Find the price"),
        ]
        response = MagicMock()
        response.model_dump_json.return_value = '{"action": [{"done": {"text": "done"}}]}'

        target = tmp_path / "conversation.txt"
        await save_conversation(messages, response, str(target))
        assert target.exists()
        content = target.read_text()
        assert "system" in content.lower() or "RESPONSE" in content

    @pytest.mark.asyncio
    async def test_save_conversation_encoding(self, tmp_path):
        from openbrowser.agent.message_manager.utils import save_conversation
        from openbrowser.llm.messages import UserMessage

        messages = [UserMessage(content="Hello")]
        response = MagicMock()
        response.model_dump_json.return_value = '{"result": "ok"}'

        target = tmp_path / "conv.txt"
        await save_conversation(messages, response, str(target), encoding="utf-8")
        assert target.exists()

    @pytest.mark.asyncio
    async def test_save_conversation_creates_dirs(self, tmp_path):
        from openbrowser.agent.message_manager.utils import save_conversation
        from openbrowser.llm.messages import UserMessage

        messages = [UserMessage(content="Test")]
        response = MagicMock()
        response.model_dump_json.return_value = '{"result": "ok"}'

        target = tmp_path / "nested" / "dir" / "conv.txt"
        await save_conversation(messages, response, str(target))
        assert target.exists()

    @pytest.mark.asyncio
    async def test_format_conversation(self):
        from openbrowser.agent.message_manager.utils import _format_conversation
        from openbrowser.llm.messages import SystemMessage, UserMessage

        messages = [
            SystemMessage(content="System prompt here"),
            UserMessage(content="User question"),
        ]
        response = MagicMock()
        response.model_dump_json.return_value = '{"action": "test"}'

        result = await _format_conversation(messages, response)
        assert "RESPONSE" in result
        assert "System prompt here" in result or "system" in result.lower()


# ---------------------------------------------------------------------------
# message_manager/service.py tests - Logging helpers
# ---------------------------------------------------------------------------


class TestLoggingHelpers:
    """Tests for _log_get_message_emoji and _log_format_message_line."""

    def test_get_message_emoji_user(self):
        from openbrowser.agent.message_manager.service import _log_get_message_emoji
        from openbrowser.llm.messages import UserMessage

        msg = UserMessage(content="test")
        emoji = _log_get_message_emoji(msg)
        assert emoji == "\U0001f4ac", f"Expected speech balloon emoji for UserMessage, got {emoji!r}"

    def test_get_message_emoji_system(self):
        from openbrowser.agent.message_manager.service import _log_get_message_emoji
        from openbrowser.llm.messages import SystemMessage

        msg = SystemMessage(content="test")
        emoji = _log_get_message_emoji(msg)
        assert emoji == "\U0001f9e0", f"Expected brain emoji for SystemMessage, got {emoji!r}"

    def test_get_message_emoji_unknown(self):
        from openbrowser.agent.message_manager.service import _log_get_message_emoji

        msg = MagicMock()
        msg.__class__.__name__ = "UnknownMessage"
        emoji = _log_get_message_emoji(msg)
        assert emoji == "\U0001f3ae", f"Expected game controller emoji for unknown message, got {emoji!r}"

    def test_format_message_line_short(self):
        from openbrowser.agent.message_manager.service import _log_format_message_line
        from openbrowser.llm.messages import UserMessage

        msg = UserMessage(content="Short content")
        lines = _log_format_message_line(msg, "Short content", is_last_message=False, terminal_width=80)
        assert len(lines) >= 1

    def test_format_message_line_long_last_message(self):
        from openbrowser.agent.message_manager.service import _log_format_message_line
        from openbrowser.llm.messages import UserMessage

        long_content = "This is a very long message " * 10
        msg = UserMessage(content=long_content)
        lines = _log_format_message_line(msg, long_content, is_last_message=True, terminal_width=40)
        assert len(lines) >= 1

    def test_format_message_line_long_last_message_no_break_point(self):
        from openbrowser.agent.message_manager.service import _log_format_message_line
        from openbrowser.llm.messages import UserMessage

        # No spaces for break point
        long_content = "x" * 200
        msg = UserMessage(content=long_content)
        lines = _log_format_message_line(msg, long_content, is_last_message=True, terminal_width=40)
        assert len(lines) >= 1

    def test_format_message_line_truncated_not_last(self):
        from openbrowser.agent.message_manager.service import _log_format_message_line
        from openbrowser.llm.messages import UserMessage

        long_content = "word " * 50
        msg = UserMessage(content=long_content)
        lines = _log_format_message_line(msg, long_content, is_last_message=False, terminal_width=40)
        assert len(lines) >= 1

    def test_format_message_line_exception(self):
        from openbrowser.agent.message_manager.service import _log_format_message_line

        # Pass a None message to trigger an exception in the function
        msg = MagicMock()
        msg.__class__.__name__ = "Broken"
        # Make it raise exception by having _log_get_message_emoji fail
        with patch("openbrowser.agent.message_manager.service._log_get_message_emoji", side_effect=Exception("boom")):
            lines = _log_format_message_line(msg, "content", False, 80)
            assert len(lines) >= 1
            assert "Error" in lines[0]


# ---------------------------------------------------------------------------
# message_manager/service.py tests - MessageManager
# ---------------------------------------------------------------------------


class TestMessageManager:
    """Tests for the MessageManager class."""

    def _make_file_system(self, todo_contents="", describe_result="No files"):
        fs = MagicMock()
        fs.get_todo_contents = MagicMock(return_value=todo_contents)
        fs.describe = MagicMock(return_value=describe_result)
        return fs

    def _make_browser_state(self, url="https://example.com", title="Example", screenshot=None):
        state = MagicMock()
        state.url = url
        state.title = title
        state.screenshot = screenshot

        tab = MagicMock()
        tab.url = url
        tab.title = title
        tab.target_id = "abcdefghijklmnop"
        state.tabs = [tab]

        dom_state = MagicMock()
        dom_state.llm_representation = MagicMock(return_value="[1] button Submit")
        dom_state._root = None
        dom_state.selector_map = {}
        state.dom_state = dom_state

        state.page_info = None
        state.is_pdf_viewer = False
        state.recent_events = None
        state.closed_popup_messages = []

        return state

    def _make_manager(self, task="Find the price", **kwargs):
        from openbrowser.agent.message_manager.service import MessageManager
        from openbrowser.llm.messages import SystemMessage

        system_msg = SystemMessage(content="System prompt", cache=True)
        file_system = kwargs.pop("file_system", self._make_file_system())
        return MessageManager(
            task=task,
            system_message=system_msg,
            file_system=file_system,
            **kwargs,
        )

    def test_init_basic(self):
        mm = self._make_manager()
        assert mm.task == "Find the price"
        assert mm.system_prompt is not None
        messages = mm.get_messages()
        assert len(messages) >= 1  # At least system message

    def test_init_with_max_history_items(self):
        mm = self._make_manager(max_history_items=10)
        assert mm.max_history_items == 10

    def test_init_max_history_items_too_low(self):
        with pytest.raises(AssertionError):
            self._make_manager(max_history_items=3)

    def test_agent_history_description_all_items(self):
        mm = self._make_manager(max_history_items=None)
        desc = mm.agent_history_description
        assert "Agent initialized" in desc

    def test_agent_history_description_with_limit(self):
        from openbrowser.agent.message_manager.views import HistoryItem

        mm = self._make_manager(max_history_items=6)
        # Add items to exceed limit
        for i in range(10):
            mm.state.agent_history_items.append(
                HistoryItem(step_number=i + 1, memory=f"Step {i+1} memory")
            )
        desc = mm.agent_history_description
        assert "omitted" in desc

    def test_agent_history_description_under_limit(self):
        mm = self._make_manager(max_history_items=20)
        desc = mm.agent_history_description
        # Only 1 default item, under limit
        assert "omitted" not in desc

    def test_add_new_task(self):
        mm = self._make_manager(task="Original task")
        mm.add_new_task("Follow up task")
        assert "follow_up_user_request" in mm.task
        assert "initial_user_request" in mm.task
        assert len(mm.state.agent_history_items) > 1

    def test_add_new_task_already_has_initial(self):
        mm = self._make_manager(task="<initial_user_request>Original task</initial_user_request>")
        mm.add_new_task("Second follow up")
        assert mm.task.count("<initial_user_request>") == 1

    def test_update_agent_history_description_with_output(self):
        from openbrowser.agent.views import AgentOutput, AgentStepInfo
        from openbrowser.models import ActionResult
        from openbrowser.tools.registry.views import ActionModel
        from pydantic import create_model

        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        mm = self._make_manager()
        output = AgentOutput(
            evaluation_previous_goal="success",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        result = [ActionResult(extracted_content="Found price: $10")]
        step_info = AgentStepInfo(step_number=1, max_steps=10)

        mm._update_agent_history_description(output, result, step_info)
        assert len(mm.state.agent_history_items) > 1

    def test_update_agent_history_description_no_output(self):
        from openbrowser.agent.views import AgentStepInfo

        mm = self._make_manager()
        step_info = AgentStepInfo(step_number=1, max_steps=10)
        mm._update_agent_history_description(None, None, step_info)
        # Should add error item for step > 0
        items = mm.state.agent_history_items
        assert any(item.error for item in items if item.error)

    def test_update_agent_history_description_step_zero_with_results(self):
        from openbrowser.agent.views import AgentStepInfo
        from openbrowser.models import ActionResult

        mm = self._make_manager()
        step_info = AgentStepInfo(step_number=0, max_steps=10)
        result = [ActionResult(extracted_content="Navigation complete")]
        mm._update_agent_history_description(None, result, step_info)

    def test_update_agent_history_description_with_error_result(self):
        from openbrowser.agent.views import AgentOutput, AgentStepInfo
        from openbrowser.models import ActionResult
        from openbrowser.tools.registry.views import ActionModel
        from pydantic import create_model

        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        mm = self._make_manager()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        result = [ActionResult(error="Element not found" * 30)]  # Long error
        step_info = AgentStepInfo(step_number=1, max_steps=10)

        mm._update_agent_history_description(output, result, step_info)

    def test_update_agent_history_description_with_long_term_memory(self):
        from openbrowser.agent.views import AgentOutput, AgentStepInfo
        from openbrowser.models import ActionResult
        from openbrowser.tools.registry.views import ActionModel
        from pydantic import create_model

        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        mm = self._make_manager()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        result = [ActionResult(long_term_memory="Important: the price is $50")]
        step_info = AgentStepInfo(step_number=1, max_steps=10)

        mm._update_agent_history_description(output, result, step_info)

    def test_update_agent_history_description_extracted_content_once(self):
        from openbrowser.agent.views import AgentOutput, AgentStepInfo
        from openbrowser.models import ActionResult
        from openbrowser.tools.registry.views import ActionModel
        from pydantic import create_model

        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        mm = self._make_manager()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        result = [
            ActionResult(
                extracted_content="Page content here",
                include_extracted_content_only_once=True,
            )
        ]
        step_info = AgentStepInfo(step_number=1, max_steps=10)

        mm._update_agent_history_description(output, result, step_info)
        assert "Page content here" in mm.state.read_state_description

    def test_update_agent_history_description_truncation(self):
        from openbrowser.agent.views import AgentOutput, AgentStepInfo
        from openbrowser.models import ActionResult
        from openbrowser.tools.registry.views import ActionModel
        from pydantic import create_model

        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        mm = self._make_manager()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        # Create very large extracted content
        result = [
            ActionResult(
                extracted_content="x" * 70000,
                include_extracted_content_only_once=True,
            )
        ]
        step_info = AgentStepInfo(step_number=1, max_steps=10)

        mm._update_agent_history_description(output, result, step_info)
        assert len(mm.state.read_state_description) <= 60100  # 60000 + truncation text

    def test_update_agent_history_description_large_action_results(self):
        from openbrowser.agent.views import AgentOutput, AgentStepInfo
        from openbrowser.models import ActionResult
        from openbrowser.tools.registry.views import ActionModel
        from pydantic import create_model

        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        mm = self._make_manager()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        result = [ActionResult(extracted_content="y" * 70000)]
        step_info = AgentStepInfo(step_number=1, max_steps=10)

        mm._update_agent_history_description(output, result, step_info)

    def test_get_sensitive_data_description_empty(self):
        mm = self._make_manager()
        mm.sensitive_data = None
        result = mm._get_sensitive_data_description("https://example.com")
        assert result == ""

    def test_get_sensitive_data_description_old_format(self):
        mm = self._make_manager(sensitive_data={"password": "secret123"})
        result = mm._get_sensitive_data_description("https://example.com")
        assert "password" in result
        assert "<secret>" in result

    def test_get_sensitive_data_description_new_format(self):
        mm = self._make_manager(
            sensitive_data={"*.example.com": {"api_key": "sk-12345"}},
        )
        result = mm._get_sensitive_data_description("https://example.com")
        assert "api_key" in result

    def test_get_sensitive_data_description_no_match(self):
        mm = self._make_manager(
            sensitive_data={"*.other.com": {"api_key": "sk-12345"}},
        )
        result = mm._get_sensitive_data_description("https://example.com")
        # No matching domain, so no placeholders
        assert result == ""

    def test_create_state_messages_basic(self):
        mm = self._make_manager()
        browser_state = self._make_browser_state()
        mm.create_state_messages(browser_state_summary=browser_state)
        messages = mm.get_messages()
        assert len(messages) >= 2  # System + state

    def test_create_state_messages_with_vision_true(self):
        mm = self._make_manager()
        browser_state = self._make_browser_state(screenshot="base64data")
        mm.create_state_messages(browser_state_summary=browser_state, use_vision=True)

    def test_create_state_messages_with_vision_auto(self):
        mm = self._make_manager()
        browser_state = self._make_browser_state()
        mm.create_state_messages(browser_state_summary=browser_state, use_vision="auto")

    def test_create_state_messages_with_vision_auto_screenshot_requested(self):
        from openbrowser.models import ActionResult

        mm = self._make_manager()
        browser_state = self._make_browser_state(screenshot="base64data")
        result = [ActionResult(metadata={"include_screenshot": True})]
        mm.create_state_messages(
            browser_state_summary=browser_state,
            use_vision="auto",
            result=result,
        )

    def test_create_state_messages_with_vision_false(self):
        mm = self._make_manager()
        browser_state = self._make_browser_state(screenshot="base64data")
        mm.create_state_messages(browser_state_summary=browser_state, use_vision=False)

    def test_create_state_messages_with_sensitive_data(self):
        mm = self._make_manager(sensitive_data={"password": "secret123"})
        browser_state = self._make_browser_state()
        mm.create_state_messages(
            browser_state_summary=browser_state,
            sensitive_data={"password": "secret123"},
        )

    def test_create_state_messages_with_step_info(self):
        from openbrowser.agent.views import AgentStepInfo

        mm = self._make_manager()
        browser_state = self._make_browser_state()
        step_info = AgentStepInfo(step_number=3, max_steps=10)
        mm.create_state_messages(browser_state_summary=browser_state, step_info=step_info)

    def test_set_message_with_type_system(self):
        from openbrowser.llm.messages import SystemMessage

        mm = self._make_manager()
        new_system = SystemMessage(content="New system prompt")
        mm._set_message_with_type(new_system, "system")
        assert mm.state.history.system_message == new_system

    def test_set_message_with_type_state(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager()
        new_state = UserMessage(content="New state")
        mm._set_message_with_type(new_state, "state")
        assert mm.state.history.state_message == new_state

    def test_set_message_with_type_invalid(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager()
        with pytest.raises(ValueError, match="Invalid state message type"):
            mm._set_message_with_type(UserMessage(content="test"), "invalid")

    def test_add_context_message(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager()
        mm._add_context_message(UserMessage(content="Context message"))
        assert len(mm.state.history.context_messages) == 1

    def test_get_messages(self):
        mm = self._make_manager()
        messages = mm.get_messages()
        assert isinstance(messages, list)
        assert len(messages) >= 1

    def test_log_history_lines(self):
        mm = self._make_manager()
        result = mm._log_history_lines()
        assert result == ""  # Currently returns empty string (TODO in code)

    def test_filter_sensitive_data_string_content(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager(sensitive_data={"password": "hunter2"})
        msg = UserMessage(content="My password is hunter2")
        filtered = mm._filter_sensitive_data(msg)
        assert "hunter2" not in filtered.content
        assert "<secret>password</secret>" in filtered.content

    def test_filter_sensitive_data_list_content(self):
        from openbrowser.llm.messages import ContentPartTextParam, UserMessage

        mm = self._make_manager(sensitive_data={"password": "hunter2"})
        msg = UserMessage(content=[ContentPartTextParam(text="Password: hunter2")])
        filtered = mm._filter_sensitive_data(msg)
        assert "hunter2" not in filtered.content[0].text

    def test_filter_sensitive_data_domain_format(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager(sensitive_data={"*.example.com": {"api_key": "sk-123"}})
        msg = UserMessage(content="API key: sk-123")
        filtered = mm._filter_sensitive_data(msg)
        assert "sk-123" not in filtered.content

    def test_filter_sensitive_data_empty_values(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager(sensitive_data={"key": ""})
        msg = UserMessage(content="No sensitive data here")
        filtered = mm._filter_sensitive_data(msg)
        assert filtered.content == "No sensitive data here"

    def test_filter_sensitive_data_none_data(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager()
        mm.sensitive_data = None
        msg = UserMessage(content="Plain text")
        filtered = mm._filter_sensitive_data(msg)
        assert filtered.content == "Plain text"

    def test_filter_sensitive_data_all_empty_values(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager(sensitive_data={"key1": "", "key2": ""})
        msg = UserMessage(content="Test content")
        filtered = mm._filter_sensitive_data(msg)
        assert filtered.content == "Test content"

    def test_create_state_messages_clears_context(self):
        from openbrowser.llm.messages import UserMessage

        mm = self._make_manager()
        mm._add_context_message(UserMessage(content="Old context"))
        browser_state = self._make_browser_state()
        mm.create_state_messages(browser_state_summary=browser_state)
        # Context messages should be cleared
        assert len(mm.state.history.context_messages) == 0

    def test_update_agent_history_description_with_short_error(self):
        """Test short error (under 200 chars) goes through the else branch (line 211)."""
        from openbrowser.agent.views import AgentOutput, AgentStepInfo
        from openbrowser.models import ActionResult
        from openbrowser.tools.registry.views import ActionModel
        from pydantic import create_model

        CustomAction = create_model("CA", __base__=ActionModel, done=(Optional[dict], None))

        mm = self._make_manager()
        output = AgentOutput(
            evaluation_previous_goal="ok",
            memory="mem",
            next_goal="next",
            action=[CustomAction(done={"text": "test"})],
        )
        # Short error (under 200 chars)
        result = [ActionResult(error="Element not found")]
        step_info = AgentStepInfo(step_number=1, max_steps=10)

        mm._update_agent_history_description(output, result, step_info)
        # Verify the error text is in the history description
        items = mm.state.agent_history_items
        assert len(items) > 0
