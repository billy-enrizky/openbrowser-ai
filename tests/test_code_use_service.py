"""Tests for openbrowser.code_use.service module.

Focuses on testable initialization logic and utility functions.
Full run() testing requires browser + LLM so we focus on unit-testable pieces.
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CodeAgent initialization tests
# ---------------------------------------------------------------------------


class TestCodeAgentInit:
    """Tests for CodeAgent.__init__ parameter handling."""

    def _make_mock_llm(self):
        llm = MagicMock()
        llm.__class__.__name__ = "MockLLM"
        llm.model = "test-model"
        llm.provider = "test-provider"
        llm.ainvoke = AsyncMock()
        return llm

    def _make_mock_session(self):
        session = MagicMock()
        session.is_local = True
        return session

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_basic_init_with_llm(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = self._make_mock_llm()
        agent = CodeAgent(task="Test task", llm=llm)
        assert agent.task == "Test task"
        assert agent.llm is llm
        assert agent.max_steps == 100
        assert agent.max_failures == 8

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_browser_and_browser_session_conflict(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = self._make_mock_llm()
        session1 = self._make_mock_session()
        session2 = self._make_mock_session()

        with pytest.raises(ValueError, match='Cannot specify both "browser" and "browser_session"'):
            CodeAgent(task="Test", llm=llm, browser=session1, browser_session=session2)

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_controller_and_tools_conflict(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent
        from openbrowser.tools.service import Tools

        llm = self._make_mock_llm()
        tools1 = Tools()
        tools2 = Tools()

        with pytest.raises(ValueError, match='Cannot specify both "controller" and "tools"'):
            CodeAgent(task="Test", llm=llm, controller=tools1, tools=tools2)

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_browser_alias_works(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = self._make_mock_llm()
        session = self._make_mock_session()

        agent = CodeAgent(task="Test", llm=llm, browser=session)
        assert agent.browser_session is session

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_custom_settings(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = self._make_mock_llm()
        agent = CodeAgent(
            task="Custom task",
            llm=llm,
            max_steps=50,
            max_failures=3,
            use_vision=False,
        )
        assert agent.max_steps == 50
        assert agent.max_failures == 3
        assert agent.use_vision is False

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_sensitive_data_stored(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = self._make_mock_llm()
        sensitive = {"password": "hunter2"}
        agent = CodeAgent(task="Login", llm=llm, sensitive_data=sensitive)
        assert agent.sensitive_data == sensitive

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_calculate_cost_passed_to_token_service(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = self._make_mock_llm()
        agent = CodeAgent(task="Test", llm=llm, calculate_cost=True)
        assert agent.token_cost_service.include_cost is True

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_unknown_kwargs_ignored(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = self._make_mock_llm()
        # Should not raise
        agent = CodeAgent(task="Test", llm=llm, unknown_param="value")
        assert agent.task == "Test"

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_browser_use_llm_detection(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = MagicMock()
        llm.__class__.__name__ = "ChatBrowserUse"
        llm.model = "browser-use"
        llm.provider = "browser-use"
        llm.ainvoke = AsyncMock()

        agent = CodeAgent(task="Test", llm=llm)
        assert agent._is_browser_use_llm is True

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_non_browser_use_llm_detection(self, mock_version, mock_screenshot, mock_telemetry):
        from openbrowser.code_use.service import CodeAgent

        llm = self._make_mock_llm()
        agent = CodeAgent(task="Test", llm=llm)
        assert agent._is_browser_use_llm is False


# ---------------------------------------------------------------------------
# CodeAgent utilities tests
# ---------------------------------------------------------------------------


class TestCodeAgentUtilities:
    """Tests for code_use utility functions."""

    def test_extract_code_blocks(self):
        from openbrowser.code_use.utils import extract_code_blocks

        text = """Here is some code:
```python
await navigate("https://example.com")
```
And more code:
```python
result = await click(5)
```"""
        blocks = extract_code_blocks(text)
        assert isinstance(blocks, dict)
        assert len(blocks) >= 2
        # Check that content is extracted
        all_content = " ".join(blocks.values())
        assert "navigate" in all_content
        assert "click" in all_content

    def test_extract_code_blocks_empty(self):
        from openbrowser.code_use.utils import extract_code_blocks

        blocks = extract_code_blocks("No code blocks here")
        assert blocks == {}

    def test_extract_url_from_task(self):
        from openbrowser.code_use.utils import extract_url_from_task

        url = extract_url_from_task("Go to https://example.com and find the price")
        assert url == "https://example.com"

    def test_extract_url_from_task_no_url(self):
        from openbrowser.code_use.utils import extract_url_from_task

        url = extract_url_from_task("Find the best laptop on amazon")
        assert url is None

    def test_detect_token_limit_issue_max_tokens(self):
        from openbrowser.code_use.utils import detect_token_limit_issue

        # stop_reason max_tokens should be detected
        is_problem, msg = detect_token_limit_issue(
            completion="some output",
            completion_tokens=100,
            max_tokens=100,
            stop_reason="max_tokens",
        )
        assert is_problem is True
        assert msg is not None

    def test_detect_token_limit_issue_fine(self):
        from openbrowser.code_use.utils import detect_token_limit_issue

        is_problem, msg = detect_token_limit_issue(
            completion="Everything is fine",
            completion_tokens=50,
            max_tokens=4096,
            stop_reason="end_turn",
        )
        assert is_problem is False

    def test_truncate_message_content(self):
        from openbrowser.code_use.utils import truncate_message_content

        long_content = "A" * 20000
        truncated = truncate_message_content(long_content, max_length=1000)
        assert len(truncated) <= 1100  # 1000 + truncation marker
        assert "truncated" in truncated

    def test_truncate_message_content_short(self):
        from openbrowser.code_use.utils import truncate_message_content

        content = "Short content"
        result = truncate_message_content(content, max_length=1000)
        assert result == content


# ---------------------------------------------------------------------------
# CodeAgent views tests
# ---------------------------------------------------------------------------


class TestCodeAgentViews:
    """Tests for code_use.views models."""

    def test_notebook_session(self):
        from openbrowser.code_use.views import NotebookSession

        session = NotebookSession()
        assert session.cells == []
        assert session.current_execution_count == 0

    def test_notebook_session_add_cell(self):
        from openbrowser.code_use.views import NotebookSession

        session = NotebookSession()
        cell = session.add_cell(source="print('hello')")
        assert len(session.cells) == 1
        assert cell.source == "print('hello')"

    def test_notebook_session_increment_counter(self):
        from openbrowser.code_use.views import NotebookSession

        session = NotebookSession()
        c1 = session.increment_execution_count()
        c2 = session.increment_execution_count()
        assert c1 == 1
        assert c2 == 2

    def test_execution_status_enum(self):
        from openbrowser.code_use.views import ExecutionStatus

        assert ExecutionStatus.SUCCESS == "success"
        assert ExecutionStatus.ERROR == "error"
        assert ExecutionStatus.PENDING == "pending"

    def test_code_agent_result(self):
        from openbrowser.code_use.views import CodeAgentResult

        result = CodeAgentResult(
            is_done=True,
            success=True,
            extracted_content="Final answer",
        )
        assert result.is_done is True
        assert result.success is True

    def test_code_agent_state(self):
        from openbrowser.code_use.views import CodeAgentState

        state = CodeAgentState()
        assert state.url is None
        assert state.title is None
        assert state.screenshot_path is None
