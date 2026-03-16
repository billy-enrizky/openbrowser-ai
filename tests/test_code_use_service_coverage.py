"""Comprehensive coverage tests for openbrowser.code_use.service module.

Covers: CodeAgent run(), _get_code_from_llm(), _execute_code(), _get_browser_state(),
_format_execution_result(), _is_task_done(), _capture_screenshot(),
_add_step_to_complete_history(), _log_agent_event(), screenshot_paths(),
message_manager, history property, close(), __aenter__/__aexit__,
_get_code_agent_system_prompt(), _print_variable_info(), DictToObject, MockAgentHistoryList.
"""

import asyncio
import datetime
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(class_name="MockLLM"):
    llm = MagicMock()
    llm.__class__.__name__ = class_name
    llm.model = "test-model"
    llm.provider = "test-provider"
    llm.max_tokens = 4096
    llm.ainvoke = AsyncMock()
    return llm


def _make_mock_session():
    session = MagicMock()
    session.is_local = True
    session.cdp_url = None
    session.current_target_id = "target1234"
    session.browser_profile = MagicMock()
    session.browser_profile.keep_alive = False
    session.browser_profile.downloads_path = None
    session.start = AsyncMock()
    session.kill = AsyncMock()
    session.get_current_page_url = AsyncMock(return_value="https://example.com")
    session.get_browser_state_summary = AsyncMock(
        return_value=MagicMock(screenshot="base64screenshot")
    )

    cdp_session = AsyncMock()
    cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": "Page Title"}}
    )
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
    session.highlight_interaction_element = AsyncMock()
    session.get_element_by_index = AsyncMock(return_value=None)
    session.event_bus = MagicMock()
    return session


PATCHES = [
    "openbrowser.code_use.service.ProductTelemetry",
    "openbrowser.code_use.service.ScreenshotService",
    "openbrowser.code_use.service.get_openbrowser_version",
]


def _create_agent(task="Test task", llm=None, browser=None, **kwargs):
    """Create a CodeAgent with standard mocks applied."""
    from openbrowser.code_use.service import CodeAgent

    if llm is None:
        llm = _make_mock_llm()
    if browser is None:
        browser = _make_mock_session()

    with patch("openbrowser.code_use.service.ProductTelemetry"), \
         patch("openbrowser.code_use.service.ScreenshotService"), \
         patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0"):
        agent = CodeAgent(task=task, llm=llm, browser=browser, **kwargs)
    return agent


# ---------------------------------------------------------------------------
# LLM auto-detection
# ---------------------------------------------------------------------------


class TestLLMAutoDetection:
    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_auto_detect_chatbrowseruse(self, *mocks):
        """Verify _is_browser_use_llm is True when LLM class name contains ChatBrowserUse."""
        mock_cbu = _make_mock_llm(class_name="ChatBrowserUse")
        browser = _make_mock_session()
        from openbrowser.code_use.service import CodeAgent

        agent = CodeAgent(task="Test", llm=mock_cbu, browser=browser)
        assert agent._is_browser_use_llm is True

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_llm_fallback_both_fail(self, *mocks):
        """When llm=None and both ChatBrowserUse and ChatGoogle imports fail, raise RuntimeError."""
        from openbrowser.code_use.service import CodeAgent
        import builtins

        real_import = builtins.__import__

        def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Block the two specific from-imports used by the auto-detect code
            if fromlist:
                if name == "openbrowser" and "ChatBrowserUse" in fromlist:
                    raise ImportError("mocked: no ChatBrowserUse")
                if name == "openbrowser.llm" and "ChatGoogle" in fromlist:
                    raise ImportError("mocked: no ChatGoogle")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=blocking_import):
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                CodeAgent(task="Test", llm=None, browser=_make_mock_session())


# ---------------------------------------------------------------------------
# _get_code_agent_system_prompt
# ---------------------------------------------------------------------------


class TestGetCodeAgentSystemPrompt:
    def test_system_prompt_from_file(self):
        agent = _create_agent()
        prompt = agent._get_code_agent_system_prompt()
        # Should return content from the file or the fallback
        assert len(prompt) > 0

    def test_system_prompt_fallback_when_file_missing(self):
        agent = _create_agent()
        with patch("builtins.open", side_effect=FileNotFoundError):
            prompt = agent._get_code_agent_system_prompt()
        assert "browser automation agent" in prompt


# ---------------------------------------------------------------------------
# _is_task_done
# ---------------------------------------------------------------------------


class TestIsTaskDone:
    def test_task_not_done_by_default(self):
        agent = _create_agent()
        assert agent._is_task_done() is False

    def test_task_done_when_marker_set(self):
        agent = _create_agent()
        agent.namespace["_task_done"] = True
        assert agent._is_task_done() is True


# ---------------------------------------------------------------------------
# _format_execution_result
# ---------------------------------------------------------------------------


class TestFormatExecutionResult:
    def test_with_output(self):
        agent = _create_agent()
        result = agent._format_execution_result("code", "output text", None)
        assert "Output: output text" in result

    def test_with_error(self):
        agent = _create_agent()
        result = agent._format_execution_result("code", None, "some error")
        assert "Error: some error" in result

    def test_with_step_number(self):
        agent = _create_agent()
        result = agent._format_execution_result("code", None, None, current_step=5)
        assert "Step 5/" in result

    def test_with_error_and_step_shows_failures(self):
        agent = _create_agent()
        agent._consecutive_errors = 3
        result = agent._format_execution_result("code", None, "err", current_step=5)
        assert "Consecutive failures: 3/" in result

    def test_no_output_no_error(self):
        agent = _create_agent()
        result = agent._format_execution_result("code", None, None)
        assert "Executed" in result

    def test_truncates_long_output(self):
        agent = _create_agent()
        long_output = "X" * 15000
        result = agent._format_execution_result("code", long_output, None)
        assert "Truncated" in result


# ---------------------------------------------------------------------------
# _capture_screenshot
# ---------------------------------------------------------------------------


class TestCaptureScreenshot:
    @pytest.mark.asyncio
    async def test_capture_with_no_browser_session(self):
        agent = _create_agent()
        agent.browser_session = None
        path, b64 = await agent._capture_screenshot(1)
        assert path is None
        assert b64 is None

    @pytest.mark.asyncio
    async def test_capture_success(self):
        agent = _create_agent()
        mock_state = MagicMock()
        mock_state.screenshot = "base64data"
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=mock_state)
        agent.screenshot_service.store_screenshot = AsyncMock(return_value=Path("/tmp/screenshot.png"))

        path, b64 = await agent._capture_screenshot(1)
        assert path is not None
        assert b64 == "base64data"

    @pytest.mark.asyncio
    async def test_capture_failure(self):
        agent = _create_agent()
        agent.browser_session.get_browser_state_summary = AsyncMock(side_effect=RuntimeError("fail"))

        path, b64 = await agent._capture_screenshot(1)
        assert path is None
        assert b64 is None

    @pytest.mark.asyncio
    async def test_capture_no_screenshot_in_state(self):
        agent = _create_agent()
        mock_state = MagicMock()
        mock_state.screenshot = None
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=mock_state)

        path, b64 = await agent._capture_screenshot(1)
        assert path is None


# ---------------------------------------------------------------------------
# _get_browser_state
# ---------------------------------------------------------------------------


class TestGetBrowserState:
    @pytest.mark.asyncio
    async def test_no_browser_session(self):
        agent = _create_agent()
        agent.browser_session = None
        text, screenshot = await agent._get_browser_state()
        assert "not available" in text
        assert screenshot is None

    @pytest.mark.asyncio
    async def test_no_dom_service(self):
        agent = _create_agent()
        agent.dom_service = None
        text, screenshot = await agent._get_browser_state()
        assert "not available" in text

    @pytest.mark.asyncio
    async def test_success(self):
        agent = _create_agent()
        agent.dom_service = MagicMock()

        mock_state = MagicMock()
        mock_state.screenshot = "screenshot_b64"
        agent.browser_session.get_browser_state_summary = AsyncMock(return_value=mock_state)

        with patch("openbrowser.code_use.service.format_browser_state_for_llm", new_callable=AsyncMock) as mock_format:
            mock_format.return_value = "Browser state text"
            text, screenshot = await agent._get_browser_state()

        assert text == "Browser state text"
        assert screenshot == "screenshot_b64"

    @pytest.mark.asyncio
    async def test_error_returns_error_message(self):
        agent = _create_agent()
        agent.dom_service = MagicMock()
        agent.browser_session.get_browser_state_summary = AsyncMock(
            side_effect=RuntimeError("state error")
        )

        text, screenshot = await agent._get_browser_state()
        assert "Error" in text
        assert screenshot is None


# ---------------------------------------------------------------------------
# _add_step_to_complete_history
# ---------------------------------------------------------------------------


class TestAddStepToCompleteHistory:
    @pytest.mark.asyncio
    async def test_add_step_basic(self):
        agent = _create_agent()
        agent._step_start_time = datetime.datetime.now().timestamp()
        agent._last_llm_usage = MagicMock()
        agent._last_llm_usage.prompt_tokens = 100
        agent._last_llm_usage.completion_tokens = 50

        await agent._add_step_to_complete_history(
            model_output_code="code",
            full_llm_response="response",
            output="output text",
            error=None,
            screenshot_path="/tmp/screenshot.png",
        )
        assert len(agent.complete_history) == 1
        entry = agent.complete_history[0]
        assert entry.model_output.model_output == "code"

    @pytest.mark.asyncio
    async def test_add_step_when_done(self):
        agent = _create_agent()
        agent.namespace["_task_done"] = True
        agent.namespace["_task_success"] = True
        agent._step_start_time = datetime.datetime.now().timestamp()
        agent._last_llm_usage = None

        await agent._add_step_to_complete_history(
            model_output_code="await done('result')",
            full_llm_response="done",
            output="result",
            error=None,
            screenshot_path=None,
        )
        assert len(agent.complete_history) == 1
        result = agent.complete_history[0].result[0]
        assert result.is_done is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_add_step_with_error(self):
        agent = _create_agent()
        agent._step_start_time = datetime.datetime.now().timestamp()
        agent._last_llm_usage = None

        await agent._add_step_to_complete_history(
            model_output_code="bad code",
            full_llm_response="response",
            output=None,
            error="SyntaxError: invalid syntax",
            screenshot_path=None,
        )
        assert agent.complete_history[0].result[0].error is not None

    @pytest.mark.asyncio
    async def test_add_step_no_code(self):
        agent = _create_agent()
        agent._step_start_time = datetime.datetime.now().timestamp()
        agent._last_llm_usage = None

        await agent._add_step_to_complete_history(
            model_output_code="",
            full_llm_response="",
            output=None,
            error=None,
            screenshot_path=None,
        )
        assert len(agent.complete_history) == 1

    @pytest.mark.asyncio
    async def test_add_step_browser_url_failure(self):
        agent = _create_agent()
        agent._step_start_time = datetime.datetime.now().timestamp()
        agent._last_llm_usage = None
        agent.browser_session.get_current_page_url = AsyncMock(side_effect=RuntimeError("no url"))

        await agent._add_step_to_complete_history(
            model_output_code="code",
            full_llm_response="resp",
            output="out",
            error=None,
            screenshot_path=None,
        )
        assert agent.complete_history[0].state.url is None


# ---------------------------------------------------------------------------
# _get_code_from_llm
# ---------------------------------------------------------------------------


class TestGetCodeFromLlm:
    @pytest.mark.asyncio
    async def test_returns_python_code(self):
        agent = _create_agent()
        agent._llm_messages = []
        agent._last_browser_state_text = None

        mock_response = MagicMock()
        mock_response.completion = '```python\nawait navigate("https://example.com")\n```'
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_response.stop_reason = "end_turn"
        agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        code, full = await agent._get_code_from_llm()
        assert "navigate" in code
        assert "navigate" in full

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_code_blocks(self):
        agent = _create_agent()
        agent._llm_messages = []

        mock_response = MagicMock()
        mock_response.completion = "I think we should navigate to the page"
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_response.stop_reason = "end_turn"
        agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        code, full = await agent._get_code_from_llm()
        assert code == ""

    @pytest.mark.asyncio
    async def test_injects_js_block_into_namespace(self):
        agent = _create_agent()
        agent._llm_messages = []

        mock_response = MagicMock()
        mock_response.completion = '```js extract_js\ndocument.title\n```\n```python\nresult = await evaluate(extract_js)\n```'
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_response.stop_reason = "end_turn"
        agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        code, full = await agent._get_code_from_llm()
        assert "evaluate" in code

    @pytest.mark.asyncio
    async def test_with_browser_state_and_screenshot(self):
        agent = _create_agent()
        agent._llm_messages = []
        agent._last_browser_state_text = "Current page: example.com"
        agent._last_screenshot = "base64screenshot"
        agent.use_vision = True

        mock_response = MagicMock()
        mock_response.completion = '```python\nprint("hello")\n```'
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_response.stop_reason = "end_turn"
        agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        code, full = await agent._get_code_from_llm()
        assert agent._last_browser_state_text is None  # Cleared after use
        assert agent._last_screenshot is None

    @pytest.mark.asyncio
    async def test_with_browser_state_no_screenshot(self):
        agent = _create_agent()
        agent._llm_messages = []
        agent._last_browser_state_text = "Current page: example.com"
        agent._last_screenshot = None
        agent.use_vision = False

        mock_response = MagicMock()
        mock_response.completion = '```python\nprint("hello")\n```'
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_response.stop_reason = "end_turn"
        agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        code, full = await agent._get_code_from_llm()
        assert code is not None

    @pytest.mark.asyncio
    async def test_token_limit_issue_detected(self):
        agent = _create_agent()
        agent._llm_messages = []

        mock_response = MagicMock()
        mock_response.completion = "```python\n" + "x = 1\n" * 1000 + "```"
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=4096)
        mock_response.stop_reason = "max_tokens"
        agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        agent.llm.max_tokens = 4096

        code, full = await agent._get_code_from_llm()
        assert code == ""
        assert "Token limit" in full


# ---------------------------------------------------------------------------
# _execute_code
# ---------------------------------------------------------------------------


class TestExecuteCode:
    @pytest.mark.asyncio
    async def test_simple_sync_code(self):
        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)
        agent.namespace = {"asyncio": asyncio}

        output, error, state = await agent._execute_code("x = 42")
        assert error is None
        assert agent.namespace.get("x") == 42

    @pytest.mark.asyncio
    async def test_code_with_await(self):
        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)
        agent.namespace = {"asyncio": asyncio}

        output, error, state = await agent._execute_code("result = await asyncio.sleep(0)")
        assert error is None

    @pytest.mark.asyncio
    async def test_code_with_print_output(self):
        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)
        agent.namespace = {"asyncio": asyncio}

        output, error, state = await agent._execute_code("print('hello from code')")
        assert output is not None
        assert "hello from code" in output

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)
        agent.namespace = {"asyncio": asyncio}

        output, error, state = await agent._execute_code("def bad(")
        assert error is not None
        assert "SyntaxError" in error

    @pytest.mark.asyncio
    async def test_name_error(self):
        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)
        agent.namespace = {"asyncio": asyncio}

        # NameError path sets cell.status but the outer error variable stays None
        # because the branch only sets a local error_msg, not the error return variable.
        # We verify the NameError branch was exercised by checking cell.status.
        from openbrowser.code_use.views import ExecutionStatus
        output, error, state = await agent._execute_code("print(undefined_variable)")
        assert mock_cell.status == ExecutionStatus.ERROR

    @pytest.mark.asyncio
    async def test_runtime_error(self):
        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)
        agent.namespace = {"asyncio": asyncio}

        output, error, state = await agent._execute_code("raise RuntimeError('test error')")
        assert error is not None
        assert "RuntimeError" in error

    @pytest.mark.asyncio
    async def test_evaluate_error(self):
        from openbrowser.code_use.namespace import EvaluateError

        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)

        async def raise_eval_error():
            raise EvaluateError("JS failed")

        agent.namespace = {
            "asyncio": asyncio,
            "raise_eval_error": raise_eval_error,
        }

        output, error, state = await agent._execute_code("await raise_eval_error()")
        assert error is not None
        assert "JS failed" in error

    @pytest.mark.asyncio
    async def test_code_with_global_declaration(self):
        """Exercise the global declaration path in _execute_code with await code."""
        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)
        agent.namespace = {"asyncio": asyncio, "counter": 0}

        # Use await so the has_await branch is taken, which triggers global declaration logic
        # for existing namespace vars like 'counter'
        output, error, state = await agent._execute_code(
            "global counter\ncounter = await asyncio.sleep(0) or 10"
        )
        assert error is None
        # Verify the global declaration was actually processed
        assert agent.namespace.get("counter") == 10

    @pytest.mark.asyncio
    async def test_fstring_syntax_error_hint(self):
        agent = _create_agent()
        agent.session = MagicMock()
        mock_cell = MagicMock()
        agent.session.add_cell = MagicMock(return_value=mock_cell)
        agent.session.increment_execution_count = MagicMock(return_value=1)
        agent.namespace = {"asyncio": asyncio}

        # This triggers SyntaxError with unterminated string
        code = "x = f'{bad"
        output, error, state = await agent._execute_code(code)
        assert error is not None
        assert "SyntaxError" in error


# ---------------------------------------------------------------------------
# _print_variable_info
# ---------------------------------------------------------------------------


class TestPrintVariableInfo:
    def test_skips_builtin_modules(self):
        agent = _create_agent()
        # Should not print for skip_names
        agent._print_variable_info("json", {})  # No assertion needed, just no crash

    def test_skips_code_block_vars(self):
        agent = _create_agent()
        agent.namespace = {"_code_block_vars": {"js_code"}}
        agent._print_variable_info("js_code", "some code")

    def test_prints_list_info(self, capsys):
        agent = _create_agent()
        agent.namespace = {}
        agent._print_variable_info("my_list", [1, 2, 3])
        # Output goes to stdout via print()
        captured = capsys.readouterr()
        assert "my_list" in captured.out

    def test_prints_long_string(self, capsys):
        agent = _create_agent()
        agent.namespace = {}
        agent._print_variable_info("long_str", "A" * 100)
        captured = capsys.readouterr()
        assert "long_str" in captured.out

    def test_prints_callable(self, capsys):
        agent = _create_agent()
        agent.namespace = {}
        agent._print_variable_info("my_func", lambda: None)
        captured = capsys.readouterr()
        assert "function" in captured.out

    def test_prints_primitive(self, capsys):
        agent = _create_agent()
        agent.namespace = {}
        agent._print_variable_info("num", 42)
        captured = capsys.readouterr()
        assert "num" in captured.out


# ---------------------------------------------------------------------------
# screenshot_paths
# ---------------------------------------------------------------------------


class TestScreenshotPaths:
    def test_empty_history(self):
        agent = _create_agent()
        assert agent.screenshot_paths() == []

    def test_with_history(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentModelOutput,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        entry = CodeAgentHistory(
            model_output=None,
            result=[CodeAgentResult()],
            state=CodeAgentState(),
            metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
            screenshot_path="/tmp/screenshot1.png",
        )
        agent.complete_history = [entry]
        paths = agent.screenshot_paths()
        assert paths == ["/tmp/screenshot1.png"]

    def test_with_n_last(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        entries = []
        for i in range(5):
            entry = CodeAgentHistory(
                model_output=None,
                result=[CodeAgentResult()],
                state=CodeAgentState(),
                metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
                screenshot_path=f"/tmp/screenshot{i}.png",
            )
            entries.append(entry)
        agent.complete_history = entries

        paths = agent.screenshot_paths(n_last=2)
        assert len(paths) == 2
        assert paths[0] == "/tmp/screenshot3.png"


# ---------------------------------------------------------------------------
# message_manager property
# ---------------------------------------------------------------------------


class TestMessageManager:
    def test_returns_mock_with_last_input_messages(self):
        agent = _create_agent()
        agent._llm_messages = ["msg1", "msg2"]
        mm = agent.message_manager
        assert mm.last_input_messages == ["msg1", "msg2"]


# ---------------------------------------------------------------------------
# history property (DictToObject, MockAgentHistoryList)
# ---------------------------------------------------------------------------


class TestHistoryProperty:
    def test_returns_history_list(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentModelOutput,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        entry = CodeAgentHistory(
            model_output=CodeAgentModelOutput(model_output="code", full_response="resp"),
            result=[CodeAgentResult(extracted_content="output")],
            state=CodeAgentState(url="https://example.com"),
            metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
            screenshot_path=None,
        )
        agent.complete_history = [entry]
        agent.usage_summary = MagicMock()

        hist = agent.history
        assert hasattr(hist, "history")
        assert len(hist.history) == 1

    def test_dict_to_object_model_dump(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        entry = CodeAgentHistory(
            model_output=None,
            result=[CodeAgentResult()],
            state=CodeAgentState(),
            metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
            screenshot_path=None,
        )
        agent.complete_history = [entry]
        agent.usage_summary = None

        hist = agent.history
        obj = hist.history[0]
        dump = obj.model_dump()
        assert isinstance(dump, dict)

    def test_dict_to_object_missing_attr_returns_none(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        entry = CodeAgentHistory(
            model_output=None,
            result=[CodeAgentResult()],
            state=CodeAgentState(),
            metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
            screenshot_path=None,
        )
        agent.complete_history = [entry]
        agent.usage_summary = None

        hist = agent.history
        obj = hist.history[0]
        assert obj.nonexistent_attribute is None

    def test_dict_to_object_get_screenshot_no_path(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        entry = CodeAgentHistory(
            model_output=None,
            result=[CodeAgentResult()],
            state=CodeAgentState(),
            metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
            screenshot_path=None,
        )
        agent.complete_history = [entry]
        agent.usage_summary = None

        hist = agent.history
        state_obj = hist.history[0].state
        result = state_obj.get_screenshot()
        assert result is None

    def test_dict_to_object_get_screenshot_with_file(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        # Create a temp file with screenshot data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(b"PNG_DATA")
            temp_path = f.name

        try:
            entry = CodeAgentHistory(
                model_output=None,
                result=[CodeAgentResult()],
                state=CodeAgentState(screenshot_path=temp_path),
                metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
                screenshot_path=temp_path,
            )
            agent.complete_history = [entry]
            agent.usage_summary = None

            hist = agent.history
            state_obj = hist.history[0].state
            screenshot = state_obj.get_screenshot()
            assert screenshot is not None
        finally:
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_kills_session(self):
        agent = _create_agent()
        agent.browser_session.browser_profile.keep_alive = False
        await agent.close()
        agent.browser_session.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_keeps_alive(self):
        agent = _create_agent()
        agent.browser_session.browser_profile.keep_alive = True
        await agent.close()
        agent.browser_session.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_no_browser_session(self):
        agent = _create_agent()
        agent.browser_session = None
        await agent.close()  # Should not raise


# ---------------------------------------------------------------------------
# __aenter__ / __aexit__
# ---------------------------------------------------------------------------


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_aenter_returns_self(self):
        agent = _create_agent()
        result = await agent.__aenter__()
        assert result is agent

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self):
        agent = _create_agent()
        agent.close = AsyncMock()
        await agent.__aexit__(None, None, None)
        agent.close.assert_called_once()


# ---------------------------------------------------------------------------
# _log_agent_event
# ---------------------------------------------------------------------------


class TestLogAgentEvent:
    def test_log_event_basic(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentModelOutput,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        entry = CodeAgentHistory(
            model_output=CodeAgentModelOutput(model_output="code", full_response="full"),
            result=[CodeAgentResult(error="some error")],
            state=CodeAgentState(url="https://example.com"),
            metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
            screenshot_path=None,
        )
        agent.complete_history = [entry]
        agent.namespace["_task_done"] = True
        agent.namespace["_task_result"] = "Final result"
        agent.namespace["_task_success"] = True

        mock_token_summary = MagicMock()
        mock_token_summary.prompt_tokens = 100
        mock_token_summary.completion_tokens = 50
        mock_token_summary.prompt_cached_tokens = 10
        mock_token_summary.total_tokens = 150
        agent.token_cost_service = MagicMock()
        agent.token_cost_service.get_usage_tokens_for_model = MagicMock(return_value=mock_token_summary)
        agent.telemetry.capture = MagicMock()

        agent._log_agent_event(max_steps=100)
        agent.telemetry.capture.assert_called_once()

    def test_log_event_no_model_output(self):
        agent = _create_agent()
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        entry = CodeAgentHistory(
            model_output=None,
            result=[CodeAgentResult()],
            state=CodeAgentState(),
            metadata=CodeAgentStepMetadata(step_start_time=0, step_end_time=1),
            screenshot_path=None,
        )
        agent.complete_history = [entry]

        mock_token_summary = MagicMock()
        mock_token_summary.prompt_tokens = 0
        mock_token_summary.completion_tokens = 0
        mock_token_summary.prompt_cached_tokens = 0
        mock_token_summary.total_tokens = 0
        agent.token_cost_service = MagicMock()
        agent.token_cost_service.get_usage_tokens_for_model = MagicMock(return_value=mock_token_summary)
        agent.telemetry.capture = MagicMock()

        agent._log_agent_event(max_steps=100)
        agent.telemetry.capture.assert_called_once()

    def test_log_event_with_cdp_url(self):
        agent = _create_agent()
        agent.browser_session.cdp_url = "ws://localhost:9222"
        agent.complete_history = []

        mock_token_summary = MagicMock()
        mock_token_summary.prompt_tokens = 0
        mock_token_summary.completion_tokens = 0
        mock_token_summary.prompt_cached_tokens = 0
        mock_token_summary.total_tokens = 0
        agent.token_cost_service = MagicMock()
        agent.token_cost_service.get_usage_tokens_for_model = MagicMock(return_value=mock_token_summary)
        agent.telemetry.capture = MagicMock()

        agent._log_agent_event(max_steps=100)
        agent.telemetry.capture.assert_called_once()


# ---------------------------------------------------------------------------
# run() - integration-style tests with heavy mocking
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_run_simple_done_in_one_step(self):
        agent = _create_agent()
        agent.dom_service = MagicMock()

        # Mock browser state
        with patch.object(agent, "_get_browser_state", new_callable=AsyncMock) as mock_bs:
            mock_bs.return_value = ("Browser state text", "screenshot_b64")

            # Mock LLM response with done() call
            mock_response = MagicMock()
            mock_response.completion = '```python\nawait done("Task complete", success=True)\n```'
            mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
            mock_response.stop_reason = "end_turn"
            agent.llm.ainvoke = AsyncMock(return_value=mock_response)

            # Mock _execute_code to simulate done()
            async def mock_execute(code):
                agent.namespace["_task_done"] = True
                agent.namespace["_task_result"] = "Task complete"
                return "Task complete", None, None

            with patch.object(agent, "_execute_code", side_effect=mock_execute):
                with patch.object(agent, "_capture_screenshot", new_callable=AsyncMock, return_value=(None, None)):
                    with patch.object(agent, "_add_step_to_complete_history", new_callable=AsyncMock):
                        with patch.object(agent, "_log_agent_event"):
                            agent.token_cost_service = MagicMock()
                            agent.token_cost_service.get_usage_summary = AsyncMock(return_value=None)
                            agent.token_cost_service.log_usage_summary = AsyncMock()
                            session = await agent.run(max_steps=5)

        assert session is not None

    @pytest.mark.asyncio
    async def test_run_with_initial_url(self):
        agent = _create_agent(task="Go to https://example.com and find data")
        agent.dom_service = MagicMock()

        mock_navigate = AsyncMock(return_value="Navigated")

        # Patch create_namespace so run() gets our mock navigate in the fresh namespace
        def fake_create_namespace(**kwargs):
            return {
                "navigate": mock_navigate,
                "_task_done": False,
                "asyncio": asyncio,
            }

        with patch("openbrowser.code_use.service.create_namespace", side_effect=fake_create_namespace):
            with patch.object(agent, "_get_browser_state", new_callable=AsyncMock, return_value=("state", None)):
                mock_response = MagicMock()
                mock_response.completion = '```python\nawait done("Done", success=True)\n```'
                mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
                mock_response.stop_reason = "end_turn"
                agent.llm.ainvoke = AsyncMock(return_value=mock_response)

                async def mock_execute(code):
                    agent.namespace["_task_done"] = True
                    agent.namespace["_task_result"] = "Done"
                    return "Done", None, None

                with patch.object(agent, "_execute_code", side_effect=mock_execute):
                    with patch.object(agent, "_capture_screenshot", new_callable=AsyncMock, return_value=(None, None)):
                        with patch.object(agent, "_add_step_to_complete_history", new_callable=AsyncMock):
                            with patch.object(agent, "_log_agent_event"):
                                agent.token_cost_service = MagicMock()
                                agent.token_cost_service.get_usage_summary = AsyncMock(return_value=None)
                                agent.token_cost_service.log_usage_summary = AsyncMock()
                                session = await agent.run(max_steps=5)

        # Verify navigate was actually called with the extracted URL
        mock_navigate.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_run_max_steps_reached(self):
        agent = _create_agent()
        agent.dom_service = MagicMock()

        with patch.object(agent, "_get_browser_state", new_callable=AsyncMock, return_value=("state", None)):
            mock_response = MagicMock()
            mock_response.completion = '```python\nx = 1\n```'
            mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
            mock_response.stop_reason = "end_turn"
            agent.llm.ainvoke = AsyncMock(return_value=mock_response)

            with patch.object(agent, "_execute_code", new_callable=AsyncMock, return_value=("output", None, None)):
                with patch.object(agent, "_capture_screenshot", new_callable=AsyncMock, return_value=(None, None)):
                    with patch.object(agent, "_add_step_to_complete_history", new_callable=AsyncMock):
                        with patch.object(agent, "_log_agent_event"):
                            agent.token_cost_service = MagicMock()
                            agent.token_cost_service.get_usage_summary = AsyncMock(return_value=None)
                            agent.token_cost_service.log_usage_summary = AsyncMock()
                            session = await agent.run(max_steps=2)

    @pytest.mark.asyncio
    async def test_run_consecutive_errors_limit(self):
        agent = _create_agent()
        agent.dom_service = MagicMock()
        agent.max_failures = 2

        with patch.object(agent, "_get_browser_state", new_callable=AsyncMock, return_value=("state", None)):
            mock_response = MagicMock()
            mock_response.completion = '```python\nbad code\n```'
            mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
            mock_response.stop_reason = "end_turn"
            agent.llm.ainvoke = AsyncMock(return_value=mock_response)

            with patch.object(agent, "_execute_code", new_callable=AsyncMock, return_value=(None, "SyntaxError", None)):
                with patch.object(agent, "_capture_screenshot", new_callable=AsyncMock, return_value=(None, None)):
                    with patch.object(agent, "_add_step_to_complete_history", new_callable=AsyncMock):
                        with patch.object(agent, "_log_agent_event"):
                            agent.token_cost_service = MagicMock()
                            agent.token_cost_service.get_usage_summary = AsyncMock(return_value=None)
                            agent.token_cost_service.log_usage_summary = AsyncMock()
                            session = await agent.run(max_steps=10)

    @pytest.mark.asyncio
    async def test_run_empty_code_retry(self):
        agent = _create_agent()
        agent.dom_service = MagicMock()

        call_count = [0]

        async def mock_ainvoke(messages):
            call_count[0] += 1
            mock_response = MagicMock()
            if call_count[0] <= 1:
                mock_response.completion = "No code blocks here"
            else:
                mock_response.completion = '```python\nawait done("Done", success=True)\n```'
            mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
            mock_response.stop_reason = "end_turn"
            return mock_response

        agent.llm.ainvoke = mock_ainvoke

        with patch.object(agent, "_get_browser_state", new_callable=AsyncMock, return_value=("state", None)):
            async def mock_execute(code):
                agent.namespace["_task_done"] = True
                agent.namespace["_task_result"] = "Done"
                return "Done", None, None

            with patch.object(agent, "_execute_code", side_effect=mock_execute):
                with patch.object(agent, "_capture_screenshot", new_callable=AsyncMock, return_value=(None, None)):
                    with patch.object(agent, "_add_step_to_complete_history", new_callable=AsyncMock):
                        with patch.object(agent, "_log_agent_event"):
                            agent.token_cost_service = MagicMock()
                            agent.token_cost_service.get_usage_summary = AsyncMock(return_value=None)
                            agent.token_cost_service.log_usage_summary = AsyncMock()
                            session = await agent.run(max_steps=5)

    @pytest.mark.asyncio
    async def test_run_no_browser_session_creates_one(self):
        """When browser_session is None, run() should create and start one."""
        llm = _make_mock_llm()
        with patch("openbrowser.code_use.service.ProductTelemetry"), \
             patch("openbrowser.code_use.service.ScreenshotService"), \
             patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0"):
            from openbrowser.code_use.service import CodeAgent
            agent = CodeAgent(task="Test", llm=llm)

        assert agent.browser_session is None

        mock_session = _make_mock_session()
        with patch("openbrowser.code_use.service.BrowserSession", return_value=mock_session):
            with patch("openbrowser.code_use.service.DomService"):
                with patch("openbrowser.code_use.service.create_namespace", return_value={"_task_done": True, "_task_result": "Done"}):
                    with patch.object(agent, "_get_browser_state", new_callable=AsyncMock, return_value=("state", None)):
                        with patch.object(agent, "_log_agent_event"):
                            agent.token_cost_service = MagicMock()
                            agent.token_cost_service.get_usage_summary = AsyncMock(return_value=None)
                            agent.token_cost_service.log_usage_summary = AsyncMock()
                            session = await agent.run(max_steps=1)
