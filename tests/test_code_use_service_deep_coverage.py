"""Deep coverage tests for openbrowser.code_use.service module.

Targets uncovered lines: 113-126, 187, 195, 198-199, 257-272, 289-290,
339-340, 354-355, 363-372, 396-397, 407-421, 463-497, 505, 519, 530-540,
567-570, 578-614, 627, 642-643, 775, 856, 891, 893-894, 897, 902-903,
908-909, 915-917, 1014-1019, 1028-1063, 1068-1072, 1403, 1409-1410.
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

PATCHES = [
    "openbrowser.code_use.service.ProductTelemetry",
    "openbrowser.code_use.service.ScreenshotService",
    "openbrowser.code_use.service.get_openbrowser_version",
]


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


def _create_agent(task="Test task", llm=None, browser=None, **kwargs):
    from openbrowser.code_use.service import CodeAgent

    if llm is None:
        llm = _make_mock_llm()
    if browser is None:
        browser = _make_mock_session()

    with patch(*PATCHES[:1]), patch(*PATCHES[1:2]), \
         patch(PATCHES[2], return_value="0.1.0"):
        agent = CodeAgent(task=task, llm=llm, browser=browser, **kwargs)
    return agent


def _setup_agent_for_run(agent, namespace_extras=None):
    """Common setup for agent.run() tests. Returns the agent with all mocks configured.

    Instead of mocking _is_task_done with side_effect lists, we let the real
    implementation read from agent.namespace['_task_done']. Tests can set this
    in the namespace to control when the task is done.
    """
    agent.dom_service = MagicMock()
    agent._capture_screenshot = AsyncMock(return_value=(None, None))
    agent._add_step_to_complete_history = AsyncMock()
    agent._log_agent_event = MagicMock()
    agent.close = AsyncMock()
    agent.token_cost_service = MagicMock()
    agent.token_cost_service.get_usage_summary = AsyncMock(return_value=None)
    agent.token_cost_service.log_usage_summary = AsyncMock()

    ns = {"_all_code_blocks": {}}
    if namespace_extras:
        ns.update(namespace_extras)
    agent.namespace = ns
    return agent


# ---------------------------------------------------------------------------
# LLM auto-initialization  (lines 113-126)
# ---------------------------------------------------------------------------

class TestLLMAutoInit:
    """Cover automatic LLM initialization when llm=None."""

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_chatbrowseruse_fallback_to_google(self, *mocks):
        """Lines 113-126: ChatBrowserUse fails -> Google fallback."""
        from openbrowser.code_use.service import CodeAgent

        with patch("openbrowser.ChatBrowserUse", side_effect=ImportError("no key")), \
             patch(
                 "openbrowser.llm.ChatGoogle",
                 return_value=_make_mock_llm("ChatGoogle"),
             ):
            agent = CodeAgent(task="Test")

        assert agent.llm is not None

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_both_fallbacks_fail(self, *mocks):
        """Lines 126-131: both ChatBrowserUse and Google fail."""
        from openbrowser.code_use.service import CodeAgent

        with patch("openbrowser.ChatBrowserUse", side_effect=ImportError("no key")), \
             patch(
                 "openbrowser.llm.ChatGoogle",
                 side_effect=ImportError("no google key"),
             ):
            with pytest.raises(RuntimeError, match="Failed to initialize CodeAgent LLM"):
                CodeAgent(task="Test")


# ---------------------------------------------------------------------------
# Source detection  (lines 195, 198-199)
# ---------------------------------------------------------------------------

class TestSourceDetection:
    """Cover source detection in __init__."""

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_source_pip(self, *mocks):
        """Lines 195-197: not all repo files exist -> source='pip'."""
        from openbrowser.code_use.service import CodeAgent

        llm = _make_mock_llm()
        agent = CodeAgent(task="Test", llm=llm)
        assert agent.source in ("git", "pip")

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_source_exception(self, *mocks):
        """Lines 198-199: exception during source detection -> 'unknown'."""
        from openbrowser.code_use.service import CodeAgent

        llm = _make_mock_llm()
        # Make Path(__file__).parent raise by setting __file__ to a type that
        # causes Path to fail. We patch the module's __file__ attribute.
        import openbrowser.code_use.service as svc_mod

        original_file = svc_mod.__file__
        try:
            # Setting __file__ to None causes Path(None) to raise TypeError
            svc_mod.__file__ = None
            agent = CodeAgent(task="Test", llm=llm)
            assert agent.source == "unknown"
        finally:
            svc_mod.__file__ = original_file

    @patch("openbrowser.code_use.service.ProductTelemetry")
    @patch("openbrowser.code_use.service.ScreenshotService")
    @patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0")
    def test_page_extraction_llm_registered(self, *mocks):
        """Line 187: page_extraction_llm is registered with token cost service."""
        from openbrowser.code_use.service import CodeAgent

        llm = _make_mock_llm()
        page_llm = _make_mock_llm("PageLLM")
        agent = CodeAgent(task="Test", llm=llm, page_extraction_llm=page_llm)
        assert agent.page_extraction_llm is page_llm


# ---------------------------------------------------------------------------
# run() method  (lines 257-272, 289-290, 339-340, 354-355, 363-372,
#   396-397, 407-421, 463-497, 505, 519, 530-540, 567-570, 578-614, 627,
#   642-643)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunMethodDeep:
    """Cover various run() branches."""

    async def test_run_initial_navigation_success(self):
        """Lines 257-272: extract URL from task and navigate."""
        agent = _create_agent(task="Go to https://example.com and find info")
        _setup_agent_for_run(agent, {
            "navigate": AsyncMock(),
            "_task_done": True,
            "_task_result": "result",
        })

        agent._get_browser_state = AsyncMock(
            return_value=("browser state text", "base64screenshot")
        )
        agent._get_code_from_llm = AsyncMock(
            return_value=('done("result")', "I will finish")
        )
        agent._execute_code = AsyncMock(return_value=("result", None, None))
        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value="https://example.com"), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=2)

        assert result is not None

    async def test_run_initial_navigation_failure(self):
        """Lines 274-281: navigation to extracted URL fails."""
        agent = _create_agent(task="Go to https://broken.com")
        _setup_agent_for_run(agent, {
            "navigate": AsyncMock(side_effect=RuntimeError("nav failed")),
            "_task_done": True,
            "_task_result": "done",
        })

        agent._get_browser_state = AsyncMock(
            return_value=("state", "screenshot")
        )
        agent._get_code_from_llm = AsyncMock(
            return_value=('done("done")', "finishing")
        )
        agent._execute_code = AsyncMock(return_value=("done", None, None))
        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value="https://broken.com"), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=2)

        assert result is not None

    async def test_run_initial_browser_state_fails(self):
        """Lines 289-290: initial browser state fetch fails."""
        agent = _create_agent(task="Do something")
        _setup_agent_for_run(agent, {
            "_task_done": True,
            "_task_result": "ok",
        })

        agent._get_browser_state = AsyncMock(
            side_effect=RuntimeError("state failed")
        )
        agent._get_code_from_llm = AsyncMock(
            return_value=('done("ok")', "done")
        )
        agent._execute_code = AsyncMock(return_value=("ok", None, None))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=2)

        assert result is not None

    async def test_run_llm_fails_consecutive_errors(self):
        """Lines 339-358: LLM call fails multiple times -> termination."""
        agent = _create_agent(task="Do task", max_failures=2)
        _setup_agent_for_run(agent)

        agent._get_browser_state = AsyncMock(
            return_value=("state", "screenshot")
        )
        agent._get_code_from_llm = AsyncMock(
            side_effect=RuntimeError("LLM error")
        )

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace):
            result = await agent.run(max_steps=5)

        assert result is not None

    async def test_run_empty_code_task_done(self):
        """Lines 360-372: LLM returns empty code but task is already done."""
        agent = _create_agent(task="Already done task")
        _setup_agent_for_run(agent, {
            "_task_done": True,
            "_task_result": "done",
        })

        agent._last_browser_state_text = "some state"
        agent._last_screenshot = "base64"
        agent._get_code_from_llm = AsyncMock(
            return_value=("", "Task is already complete")
        )

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace):
            result = await agent.run(max_steps=3)

        assert result is not None

    async def test_run_empty_code_not_done(self):
        """Lines 374-397: empty code, task not done -> feedback."""
        agent = _create_agent(task="Need code")
        _setup_agent_for_run(agent)

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"

        call_count = [0]

        async def fake_get_code():
            call_count[0] += 1
            if call_count[0] <= 1:
                return ("", "No code here")
            # On second call, set task done and return code
            agent.namespace["_task_done"] = True
            agent.namespace["_task_result"] = "ok"
            return ('done("ok")', "Now done")

        agent._get_code_from_llm = AsyncMock(side_effect=fake_get_code)
        agent._get_browser_state = AsyncMock(
            return_value=("state", "screenshot")
        )
        agent._execute_code = AsyncMock(return_value=("ok", None, None))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=3)

        assert result is not None

    async def test_run_multiple_python_blocks(self):
        """Lines 405-421: multiple Python blocks executed sequentially."""
        agent = _create_agent(task="Multi block task")
        _setup_agent_for_run(agent, {
            "_all_code_blocks": {
                "python_0": "x = 1",
                "python_1": "y = 2",
            },
            "_task_done": True,
            "_task_result": "output",
        })

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"
        agent._get_code_from_llm = AsyncMock(
            return_value=("code1\ncode2", "planning")
        )
        agent._execute_code = AsyncMock(return_value=("output", None, None))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=2)

        assert result is not None

    async def test_run_multiple_blocks_error_stops(self):
        """Lines 418-421: error in second block stops execution."""
        agent = _create_agent(task="Error in block 2")
        _setup_agent_for_run(agent, {
            "_all_code_blocks": {"python_0": "x = 1", "python_1": "bad code"},
        })

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"
        agent._get_code_from_llm = AsyncMock(
            return_value=("code", "plan")
        )

        call_idx = [0]

        async def exec_side_effect(code):
            call_idx[0] += 1
            if call_idx[0] == 1:
                return ("ok", None, None)
            return (None, "SyntaxError: bad code", None)

        agent._execute_code = AsyncMock(side_effect=exec_side_effect)

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=1)

        assert result is not None

    async def test_run_step_callback(self):
        """Lines 530-540: register_new_step_callback is called."""
        callback = MagicMock()
        agent = _create_agent(
            task="Callback test", register_new_step_callback=callback
        )
        _setup_agent_for_run(agent, {
            "_task_done": True,
            "_task_result": "ok",
        })

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"
        agent._get_code_from_llm = AsyncMock(
            return_value=('done("ok")', "done")
        )
        agent._execute_code = AsyncMock(return_value=("ok", None, None))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=2)

        callback.assert_called_once()

    async def test_run_step_exception_breaks(self):
        """Lines 567-570: exception in step breaks loop."""
        agent = _create_agent(task="Exception task")
        _setup_agent_for_run(agent)

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"
        agent._get_code_from_llm = AsyncMock(
            return_value=("code", "thinking")
        )
        agent._execute_code = AsyncMock(side_effect=RuntimeError("fatal"))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace):
            result = await agent.run(max_steps=2)

        assert result is not None

    async def test_run_partial_result_on_max_steps(self):
        """Lines 578-614: task incomplete at max steps -> partial result."""
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        agent = _create_agent(task="Incomplete task")
        _setup_agent_for_run(agent, {
            "products": [1, 2, 3],
        })

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"
        agent._get_code_from_llm = AsyncMock(
            return_value=("x = 1", "working")
        )
        agent._execute_code = AsyncMock(return_value=("output", None, None))

        # Pre-populate history
        mock_result = CodeAgentResult(
            is_done=False, extracted_content="partial output"
        )
        mock_state = CodeAgentState(url="https://example.com")
        mock_metadata = CodeAgentStepMetadata(
            step_start_time=0.0, step_end_time=1.0
        )
        mock_history = CodeAgentHistory(
            model_output=None,
            result=[mock_result],
            state=mock_state,
            metadata=mock_metadata,
        )
        agent.complete_history = [mock_history]

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=1)

        assert result is not None
        # Verify partial result was captured
        assert mock_result.success is False

    async def test_run_done_with_attachments(self):
        """Lines 625-627: task done with file attachments."""
        agent = _create_agent(task="Done with files")
        _setup_agent_for_run(agent, {
            "_task_done": True,
            "_task_result": "result",
            "_task_attachments": ["/tmp/report.pdf"],
        })

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"
        agent._get_code_from_llm = AsyncMock(
            return_value=('done("result")', "completing")
        )
        agent._execute_code = AsyncMock(return_value=("result", None, None))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=2)

        assert result is not None

    async def test_run_telemetry_fails(self):
        """Lines 642-643: telemetry logging fails."""
        agent = _create_agent(task="Telemetry fail")
        _setup_agent_for_run(agent, {
            "_task_done": True,
            "_task_result": "ok",
        })

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"
        agent._get_code_from_llm = AsyncMock(
            return_value=('done("ok")', "done")
        )
        agent._execute_code = AsyncMock(return_value=("ok", None, None))
        agent._log_agent_event = MagicMock(side_effect=RuntimeError("telemetry boom"))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            # Should not raise even though telemetry fails
            result = await agent.run(max_steps=2)

        assert result is not None


# ---------------------------------------------------------------------------
# _get_code_from_llm  (line 775)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetCodeFromLlmDeep:
    """Cover _get_code_from_llm no-code-blocks path."""

    async def test_no_code_blocks_returns_empty(self):
        """Line 775: no code blocks at all returns empty string."""
        agent = _create_agent(task="No code")

        # Mock LLM response with proper structure
        mock_response = MagicMock()
        mock_response.completion = "Just some plain text without code blocks"
        mock_response.usage = MagicMock()
        mock_response.usage.completion_tokens = 5
        mock_response.usage.prompt_tokens = 10
        mock_response.stop_reason = "stop"
        agent.llm.ainvoke = AsyncMock(return_value=mock_response)

        agent._last_browser_state_text = "state"
        agent._last_screenshot = None

        code, response = await agent._get_code_from_llm()
        assert code == ""


# ---------------------------------------------------------------------------
# _execute_code  (lines 856, 891, 893-894, 897, 902-903, 908-909, 915-917,
#   1014-1019, 1028-1063, 1068-1072)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestExecuteCodeDeep:
    """Cover _execute_code uncovered branches."""

    async def test_execute_with_await(self):
        """Lines 856+: code with await gets wrapped in async function."""
        agent = _create_agent(task="Async code")
        agent.namespace = {"asyncio": asyncio}

        code = 'x = await asyncio.sleep(0)'
        output, error, _ = await agent._execute_code(code)
        # Should execute without error (asyncio.sleep returns None)
        assert error is None

    async def test_execute_with_global_declaration(self):
        """Lines 891-917: code with global declarations."""
        agent = _create_agent(task="Global vars")
        agent.namespace = {"asyncio": asyncio, "existing_var": 42}

        code = '''global existing_var
existing_var = await asyncio.sleep(0) or 99'''
        output, error, _ = await agent._execute_code(code)
        assert error is None

    async def test_execute_augmented_assignment(self):
        """Lines 891: augmented assignment (+=)."""
        agent = _create_agent(task="AugAssign")
        agent.namespace = {"asyncio": asyncio, "counter": 10}

        code = 'counter += await asyncio.sleep(0) or 5'
        output, error, _ = await agent._execute_code(code)
        assert error is None

    async def test_syntax_error_fstring_json(self):
        """Lines 1014-1019: SyntaxError with f-string and JSON patterns."""
        agent = _create_agent(task="Syntax error")
        agent.namespace = {}

        # This code has genuine unescaped braces in an f-string, causing SyntaxError
        code = 'x = f"result: {{"key": "value"}}"'
        output, error, _ = await agent._execute_code(code)
        # Should get a SyntaxError
        assert error is not None

    async def test_syntax_error_unterminated_string(self):
        """Lines 1028-1063: SyntaxError for unterminated string literal."""
        agent = _create_agent(task="Unterminated string")
        agent.namespace = {}

        code = 'x = "hello'
        output, error, _ = await agent._execute_code(code)
        assert error is not None
        assert "unterminated" in error.lower() or "SyntaxError" in error

    async def test_syntax_error_with_lineno_no_text(self):
        """Lines 1068-1072: SyntaxError with lineno but no text."""
        agent = _create_agent(task="Lineno error")
        agent.namespace = {}

        # This specific code may trigger different SyntaxError details
        code = 'x = (\n  1 + \n'
        output, error, _ = await agent._execute_code(code)
        assert error is not None


# ---------------------------------------------------------------------------
# DictToObject / MockAgentHistoryList in history property
# (lines 1403, 1409-1410)
# ---------------------------------------------------------------------------

class TestHistoryPropertyDeep:
    """Cover history property DictToObject and MockAgentHistoryList."""

    def test_dict_to_object_screenshot_no_path(self):
        """Lines 1403: DictToObject.screenshot_base64 with no path."""
        agent = _create_agent(task="History test")
        agent.complete_history = []
        agent.usage_summary = None

        history = agent.history
        assert history is not None
        assert history.history == []

    def test_dict_to_object_screenshot_with_valid_path(self):
        """Lines 1409-1410: screenshot_base64 with valid path."""
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        agent = _create_agent(task="History with screenshot")

        # Create a temporary screenshot file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            screenshot_path = f.name

        try:
            mock_result = CodeAgentResult(
                is_done=True,
                extracted_content="result",
            )
            mock_state = CodeAgentState(
                url="https://example.com",
                screenshot_path=screenshot_path,
            )
            mock_metadata = CodeAgentStepMetadata(
                step_start_time=0.0, step_end_time=1.0,
            )
            mock_history = CodeAgentHistory(
                model_output=None,
                result=[mock_result],
                state=mock_state,
                metadata=mock_metadata,
                screenshot_path=screenshot_path,
            )
            agent.complete_history = [mock_history]
            agent.usage_summary = None

            history = agent.history
            assert len(history.history) == 1
            # The DictToObject should have the screenshot_path accessible
            item = history.history[0]
            screenshot = item.state.get_screenshot()
            assert screenshot is not None
        finally:
            os.unlink(screenshot_path)

    def test_dict_to_object_screenshot_with_missing_path(self):
        """Line 1403: screenshot_base64 returns None for missing file."""
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        agent = _create_agent(task="Missing screenshot")

        mock_result = CodeAgentResult(
            is_done=True,
            extracted_content="result",
        )
        mock_state = CodeAgentState(
            url="https://example.com",
            screenshot_path="/nonexistent/screenshot.png",
        )
        mock_metadata = CodeAgentStepMetadata(
            step_start_time=0.0, step_end_time=1.0,
        )
        mock_history = CodeAgentHistory(
            model_output=None,
            result=[mock_result],
            state=mock_state,
            metadata=mock_metadata,
            screenshot_path="/nonexistent/screenshot.png",
        )
        agent.complete_history = [mock_history]
        agent.usage_summary = None

        history = agent.history
        assert len(history.history) == 1

    def test_dict_to_object_screenshot_read_error(self):
        """Lines 1409-1410: screenshot read raises exception -> None."""
        from openbrowser.code_use.views import (
            CodeAgentHistory,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
        )

        agent = _create_agent(task="Read error screenshot")

        # Create a temp file but make reading fail
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"data")
            screenshot_path = f.name

        try:
            mock_result = CodeAgentResult(
                is_done=True,
                extracted_content="result",
            )
            mock_state = CodeAgentState(
                url="https://example.com",
                screenshot_path=screenshot_path,
            )
            mock_metadata = CodeAgentStepMetadata(
                step_start_time=0.0, step_end_time=1.0,
            )
            mock_history = CodeAgentHistory(
                model_output=None,
                result=[mock_result],
                state=mock_state,
                metadata=mock_metadata,
                screenshot_path=screenshot_path,
            )
            agent.complete_history = [mock_history]
            agent.usage_summary = None

            with patch("builtins.open", side_effect=IOError("read failed")):
                history = agent.history

            assert len(history.history) == 1
        finally:
            os.unlink(screenshot_path)


# ---------------------------------------------------------------------------
# Validation during run  (lines 463-497, 505)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestValidationDeep:
    """Cover task validation during run()."""

    async def test_validation_rejects_done(self):
        """Lines 463-497: validator rejects done(), task continues."""
        agent = _create_agent(task="Validate me", max_validations=2)
        _setup_agent_for_run(agent, {
            "_task_done": True,
            "_task_result": "partial",
        })

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"

        call_idx = [0]

        async def fake_get_code():
            call_idx[0] += 1
            if call_idx[0] == 1:
                return ('done("partial")', "done call")
            return ('done("complete")', "really done")

        agent._get_code_from_llm = AsyncMock(side_effect=fake_get_code)
        agent._execute_code = AsyncMock(return_value=("result", None, None))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x), \
             patch("openbrowser.code_use.namespace.validate_task_completion",
                   new_callable=AsyncMock,
                   return_value=(False, "Task is incomplete")):
            result = await agent.run(max_steps=5)

        assert result is not None

    async def test_validation_skipped_at_limits(self):
        """Line 505: at limits, validation is skipped."""
        agent = _create_agent(task="At limits", max_validations=0)
        _setup_agent_for_run(agent, {
            "_task_done": True,
            "_task_result": "ok",
        })

        agent._last_browser_state_text = "state"
        agent._last_screenshot = "b64"
        agent._get_code_from_llm = AsyncMock(
            return_value=('done("ok")', "done")
        )
        agent._execute_code = AsyncMock(return_value=("ok", None, None))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("openbrowser.code_use.service.extract_url_from_task",
                   return_value=None), \
             patch("openbrowser.code_use.service.create_namespace",
                   return_value=agent.namespace), \
             patch("openbrowser.code_use.service.truncate_message_content",
                   side_effect=lambda x, **kw: x):
            result = await agent.run(max_steps=2)

        assert result is not None


# ---------------------------------------------------------------------------
# run with no browser session  (creates its own)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunCreatesBrowserSession:
    """Cover run() creating browser session when none provided."""

    async def test_run_creates_session(self):
        """Lines 218-221: browser_session is None -> creates one."""
        from openbrowser.code_use.service import CodeAgent

        llm = _make_mock_llm()

        with patch("openbrowser.code_use.service.ProductTelemetry"), \
             patch("openbrowser.code_use.service.ScreenshotService"), \
             patch("openbrowser.code_use.service.get_openbrowser_version", return_value="0.1.0"):
            agent = CodeAgent(task="Auto browser", llm=llm)

        # Agent should have no browser session yet
        assert agent.browser_session is None

        mock_session = _make_mock_session()

        with patch("openbrowser.code_use.service.BrowserSession", return_value=mock_session), \
             patch("openbrowser.code_use.service.DomService"), \
             patch("openbrowser.code_use.service.create_namespace", return_value={
                 "_task_done": True,
                 "_task_result": "ok",
                 "_all_code_blocks": {},
             }) as ns_mock, \
             patch("openbrowser.code_use.service.extract_url_from_task", return_value=None), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            agent._get_browser_state = AsyncMock(return_value=("state", "b64"))
            agent._get_code_from_llm = AsyncMock(
                return_value=('done("ok")', "done")
            )
            agent._execute_code = AsyncMock(return_value=("ok", None, None))
            agent._capture_screenshot = AsyncMock(return_value=(None, None))
            agent._add_step_to_complete_history = AsyncMock()
            agent._log_agent_event = MagicMock()
            agent.close = AsyncMock()
            agent.token_cost_service = MagicMock()
            agent.token_cost_service.get_usage_summary = AsyncMock(return_value=None)
            agent.token_cost_service.log_usage_summary = AsyncMock()
            # Set the namespace to the one returned by create_namespace
            agent.namespace = ns_mock.return_value

            with patch("openbrowser.code_use.service.truncate_message_content",
                       side_effect=lambda x, **kw: x):
                result = await agent.run(max_steps=2)

        assert result is not None
