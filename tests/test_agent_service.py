"""Tests for openbrowser.agent.service module.

Focuses on testable helper methods and initialization logic.
Most of Agent.run() requires a full browser session and LLM,
so we focus on unit-testable pieces with mocked dependencies.
"""

import logging
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from pydantic import BaseModel, create_model

from openbrowser.agent.prompts import AgentMessagePrompt, SystemPrompt
from openbrowser.agent.views import (
    AgentOutput,
    AgentSettings,
    AgentState,
    AgentStepInfo,
    BrowserStateHistory,
)
from openbrowser.models import ActionResult
from openbrowser.tools.registry.views import ActionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SystemPrompt tests
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """Tests for SystemPrompt initialization and message generation."""

    def test_default_system_prompt(self):
        sp = SystemPrompt()
        msg = sp.get_system_message()
        assert msg is not None
        assert msg.content  # Should have content
        assert msg.cache is True

    def test_override_system_message(self):
        custom = "You are a custom agent."
        sp = SystemPrompt(override_system_message=custom)
        msg = sp.get_system_message()
        assert msg.content == custom

    def test_extend_system_message(self):
        extension = "Also do this extra thing."
        sp = SystemPrompt(extend_system_message=extension)
        msg = sp.get_system_message()
        assert extension in msg.content

    def test_override_with_extension(self):
        custom = "Base prompt."
        extension = "Extra instructions."
        sp = SystemPrompt(
            override_system_message=custom,
            extend_system_message=extension,
        )
        msg = sp.get_system_message()
        assert "Base prompt." in msg.content
        assert "Extra instructions." in msg.content

    def test_flash_mode_template(self):
        sp = SystemPrompt(flash_mode=True)
        msg = sp.get_system_message()
        assert msg.content  # Should load flash template

    def test_no_thinking_template(self):
        sp = SystemPrompt(use_thinking=False)
        msg = sp.get_system_message()
        assert msg.content  # Should load no-thinking template

    def test_max_actions_in_prompt(self):
        sp = SystemPrompt(max_actions_per_step=7)
        msg = sp.get_system_message()
        assert "7" in msg.content


# ---------------------------------------------------------------------------
# AgentMessagePrompt tests
# ---------------------------------------------------------------------------


class TestAgentMessagePrompt:
    """Tests for AgentMessagePrompt state description and user message."""

    def _make_browser_state(
        self,
        url="https://example.com",
        title="Example",
        tabs=None,
        page_info=None,
        dom_state=None,
        is_pdf_viewer=False,
        recent_events=None,
        closed_popup_messages=None,
    ):
        state = MagicMock()
        state.url = url
        state.title = title

        if tabs is None:
            tab = MagicMock()
            tab.url = url
            tab.title = title
            tab.target_id = "abcdefgh12345678"
            tabs = [tab]

        state.tabs = tabs
        state.page_info = page_info
        state.is_pdf_viewer = is_pdf_viewer
        state.recent_events = recent_events
        state.closed_popup_messages = closed_popup_messages

        if dom_state is None:
            dom_state = MagicMock()
            dom_state.llm_representation = MagicMock(return_value="[1] button Click me")
            dom_state._root = None
        state.dom_state = dom_state

        return state

    def _make_file_system(self, todo_contents="", describe_result="No files"):
        fs = MagicMock()
        fs.get_todo_contents = MagicMock(return_value=todo_contents)
        fs.describe = MagicMock(return_value=describe_result)
        return fs

    def test_get_user_message_text_only(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Find the price",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert msg.content  # Should have text content
        assert "Find the price" in msg.content

    def test_get_user_message_with_vision(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Find the price",
            screenshots=["base64_screenshot_data"],
        )
        msg = prompt.get_user_message(use_vision=True)
        # Should have content parts (list)
        assert isinstance(msg.content, list)

    def test_new_tab_page_disables_vision(self):
        browser_state = self._make_browser_state(url="about:blank")
        file_system = self._make_file_system()
        step_info = AgentStepInfo(step_number=0, max_steps=10)

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Do something",
            step_info=step_info,
            screenshots=["base64_data"],
        )
        msg = prompt.get_user_message(use_vision=True)
        # For about:blank on step 0 with single tab, vision should be disabled
        # So content should be a string, not a list
        assert isinstance(msg.content, str)

    def test_agent_state_description_includes_task(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Navigate to Google",
        )
        desc = prompt._get_agent_state_description()
        assert "Navigate to Google" in desc

    def test_agent_state_description_includes_step_info(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()
        step_info = AgentStepInfo(step_number=3, max_steps=10)

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
            step_info=step_info,
        )
        desc = prompt._get_agent_state_description()
        assert "Step4" in desc  # step_number + 1
        assert "maximum:10" in desc

    def test_agent_state_description_with_sensitive_data(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Login",
            sensitive_data="password: <secret>my_pass</secret>",
        )
        desc = prompt._get_agent_state_description()
        assert "<sensitive_data>" in desc

    def test_agent_state_description_with_available_file_paths(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Upload file",
            available_file_paths=["/tmp/file.pdf", "/tmp/image.png"],
        )
        desc = prompt._get_agent_state_description()
        assert "/tmp/file.pdf" in desc
        assert "/tmp/image.png" in desc

    def test_browser_state_description_empty_page(self):
        dom_state = MagicMock()
        dom_state.llm_representation = MagicMock(return_value="")
        dom_state._root = None

        browser_state = self._make_browser_state(dom_state=dom_state)
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
        )
        desc = prompt._get_browser_state_description()
        assert "empty page" in desc

    def test_browser_state_description_with_pdf_viewer(self):
        browser_state = self._make_browser_state(is_pdf_viewer=True)
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Read PDF",
        )
        desc = prompt._get_browser_state_description()
        assert "PDF viewer" in desc

    def test_browser_state_description_with_closed_popups(self):
        browser_state = self._make_browser_state(
            closed_popup_messages=["Alert: Cookie consent"]
        )
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
        )
        desc = prompt._get_browser_state_description()
        assert "Cookie consent" in desc

    def test_browser_state_description_with_recent_events(self):
        browser_state = self._make_browser_state(recent_events="Download started")
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
            include_recent_events=True,
        )
        desc = prompt._get_browser_state_description()
        assert "Download started" in desc

    def test_page_filtered_actions_in_message(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
            page_filtered_actions="custom_action: Do something special",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert "page_specific_actions" in msg.content
        assert "custom_action" in msg.content

    def test_read_state_in_message(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
            read_state_description="Previously extracted content here",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert "read_state" in msg.content
        assert "Previously extracted content here" in msg.content

    def test_empty_read_state_not_included(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
            read_state_description="   ",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert "read_state" not in msg.content

    def test_todo_empty_shows_placeholder(self):
        browser_state = self._make_browser_state()
        file_system = self._make_file_system(todo_contents="")

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
        )
        desc = prompt._get_agent_state_description()
        assert "empty todo.md" in desc

    def test_extract_page_statistics_no_root(self):
        dom_state = MagicMock()
        dom_state._root = None
        dom_state.llm_representation = MagicMock(return_value="content")

        browser_state = self._make_browser_state(dom_state=dom_state)
        file_system = self._make_file_system()

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=file_system,
            task="Task",
        )
        stats = prompt._extract_page_statistics()
        assert stats["total_elements"] == 0
        assert stats["links"] == 0
