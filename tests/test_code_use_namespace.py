"""Comprehensive tests for openbrowser.code_use.namespace module.

Covers: _strip_js_comments, EvaluateError, evaluate(), validate_task_completion(),
create_namespace(), get_namespace_documentation(), download_file, list_downloads.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.code_use.namespace import (
    EvaluateError,
    _strip_js_comments,
    create_namespace,
    evaluate,
    get_namespace_documentation,
    validate_task_completion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _strip_js_comments (augmenting existing tests with more edge cases)
# ---------------------------------------------------------------------------


class TestStripJsCommentsExtended:
    """Extended edge-case tests for _strip_js_comments."""

    def test_multiple_multiline_comments(self):
        code = "a /* c1 */ b /* c2 */ c"
        result = _strip_js_comments(code)
        assert "c1" not in result
        assert "c2" not in result
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_multiline_comment_spanning_lines(self):
        code = "start\n/* multi\nline\ncomment */\nend"
        result = _strip_js_comments(code)
        assert "multi" not in result
        assert "start" in result
        assert "end" in result

    def test_only_comments(self):
        code = "// single\n/* block */"
        result = _strip_js_comments(code)
        assert result.strip() == ""

    def test_xpath_double_slash_preserved(self):
        code = 'document.evaluate("//div[@class=\'test\']", document);'
        result = _strip_js_comments(code)
        assert "//div" in result

    def test_regex_double_slash_preserved(self):
        code = 'var re = /\\/\\/pattern/;'
        result = _strip_js_comments(code)
        assert "pattern" in result


# ---------------------------------------------------------------------------
# EvaluateError
# ---------------------------------------------------------------------------


class TestEvaluateError:
    def test_is_exception(self):
        err = EvaluateError("boom")
        assert isinstance(err, Exception)
        assert str(err) == "boom"


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_returns_value(self):
        """evaluate() returns the value from CDP Runtime.evaluate."""
        mock_session = MagicMock()
        mock_cdp_session = AsyncMock()
        mock_cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": 42}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)

        result = await evaluate("1+1", mock_session)
        assert result == 42

    @pytest.mark.asyncio
    async def test_evaluate_returns_dict(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": {"key": "val"}}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        result = await evaluate("{}", mock_session)
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_evaluate_returns_list(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": [1, 2, 3]}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        result = await evaluate("[]", mock_session)
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_evaluate_returns_none_when_value_key_present(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": None}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        result = await evaluate("null", mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_evaluate_returns_undefined_when_no_value_key(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        result = await evaluate("void 0", mock_session)
        assert result == "undefined"

    @pytest.mark.asyncio
    async def test_evaluate_raises_on_exception_details(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "exceptionDetails": {
                    "text": "ReferenceError",
                    "exception": {"description": "x is not defined"},
                }
            }
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with pytest.raises(EvaluateError, match="ReferenceError"):
            await evaluate("x", mock_session)

    @pytest.mark.asyncio
    async def test_evaluate_raises_on_exception_with_value(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "exceptionDetails": {
                    "text": "Error",
                    "exception": {"value": "some error value"},
                }
            }
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with pytest.raises(EvaluateError, match="some error value"):
            await evaluate("throw 'err'", mock_session)

    @pytest.mark.asyncio
    async def test_evaluate_propagates_connection_errors(self):
        """Non-JavaScript evaluation errors are not wrapped in EvaluateError."""
        mock_session = MagicMock()
        mock_session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("connection lost"))

        with pytest.raises(RuntimeError, match="connection lost"):
            await evaluate("code", mock_session)

    @pytest.mark.asyncio
    async def test_evaluate_strips_comments_before_execution(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": "ok"}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        await evaluate("// comment\n1+1", mock_session)
        # Verify the comment was stripped
        call_args = mock_cdp.cdp_client.send.Runtime.evaluate.call_args
        sent_code = call_args.kwargs["params"]["expression"]
        assert "// comment" not in sent_code

    @pytest.mark.asyncio
    async def test_evaluate_returns_string_primitive(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": "hello"}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        result = await evaluate("'hello'", mock_session)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_evaluate_returns_boolean_primitive(self):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": True}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        result = await evaluate("true", mock_session)
        assert result is True


# ---------------------------------------------------------------------------
# validate_task_completion()
# ---------------------------------------------------------------------------


class TestValidateTaskCompletion:
    @pytest.mark.asyncio
    async def test_returns_true_when_verdict_yes(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Reasoning: looks done\nVerdict: YES")
        )

        is_complete, reasoning = await validate_task_completion("task", "output", mock_llm)
        assert is_complete is True
        assert "looks done" in reasoning

    @pytest.mark.asyncio
    async def test_returns_false_when_verdict_no(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Reasoning: not done\nVerdict: NO")
        )

        is_complete, reasoning = await validate_task_completion("task", "output", mock_llm)
        assert is_complete is False

    @pytest.mark.asyncio
    async def test_returns_true_on_exception(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))

        is_complete, reasoning = await validate_task_completion("task", "output", mock_llm)
        assert is_complete is True
        assert "Validation failed" in reasoning

    @pytest.mark.asyncio
    async def test_no_output_provided(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Reasoning: no output\nVerdict: YES")
        )

        is_complete, reasoning = await validate_task_completion("task", None, mock_llm)
        assert is_complete is True

    @pytest.mark.asyncio
    async def test_no_reasoning_line_uses_full_response(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Verdict: YES")
        )

        is_complete, reasoning = await validate_task_completion("task", "output", mock_llm)
        assert is_complete is True
        assert "Verdict: YES" in reasoning

    @pytest.mark.asyncio
    async def test_defaults_to_no_if_no_verdict(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Some random text with no verdict line")
        )

        is_complete, reasoning = await validate_task_completion("task", "output", mock_llm)
        assert is_complete is False


# ---------------------------------------------------------------------------
# create_namespace()
# ---------------------------------------------------------------------------


class TestCreateNamespace:
    def _make_mock_browser_session(self):
        mock = MagicMock()
        mock.browser_profile = MagicMock()
        mock.browser_profile.downloads_path = None
        return mock

    def _make_mock_tools(self):
        mock_tools = MagicMock()
        mock_tools.registry.registry.actions = {}
        return mock_tools

    def test_namespace_contains_standard_modules(self):
        bs = self._make_mock_browser_session()
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        assert "json" in ns
        assert "asyncio" in ns
        assert "Path" in ns
        assert "csv" in ns
        assert "re" in ns
        assert "datetime" in ns
        assert "requests" in ns

    def test_namespace_contains_browser_session(self):
        bs = self._make_mock_browser_session()
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        assert ns["browser"] is bs

    def test_namespace_contains_evaluate(self):
        bs = self._make_mock_browser_session()
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        assert "evaluate" in ns
        assert asyncio.iscoroutinefunction(ns["evaluate"])

    def test_namespace_contains_download_file(self):
        bs = self._make_mock_browser_session()
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        assert "download_file" in ns
        assert asyncio.iscoroutinefunction(ns["download_file"])

    def test_namespace_contains_list_downloads(self):
        bs = self._make_mock_browser_session()
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        assert "list_downloads" in ns
        assert callable(ns["list_downloads"])

    def test_namespace_contains_get_selector_from_index(self):
        bs = self._make_mock_browser_session()
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        assert "get_selector_from_index" in ns
        assert asyncio.iscoroutinefunction(ns["get_selector_from_index"])

    def test_namespace_contains_evaluate_failures_tracker(self):
        bs = self._make_mock_browser_session()
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        assert "_evaluate_failures" in ns
        assert isinstance(ns["_evaluate_failures"], list)

    def test_namespace_file_system_passed_through(self):
        bs = self._make_mock_browser_session()
        tools = self._make_mock_tools()
        mock_fs = MagicMock()
        ns = create_namespace(bs, tools=tools, file_system=mock_fs)

        assert ns["file_system"] is mock_fs

    def test_namespace_renames_input_to_input_text(self):
        """The 'input' action should be renamed to 'input_text' in namespace."""
        bs = self._make_mock_browser_session()

        mock_action = MagicMock()
        mock_action.param_model = MagicMock()
        mock_action.param_model.model_fields = {"text": MagicMock()}
        mock_action.function = AsyncMock()

        mock_tools = MagicMock()
        mock_tools.registry.registry.actions = {"input": mock_action}

        ns = create_namespace(bs, tools=mock_tools)
        assert "input_text" in ns
        assert "input" not in ns

    def test_namespace_skips_evaluate_from_tools(self):
        """evaluate in tools registry should be skipped (custom impl used instead)."""
        bs = self._make_mock_browser_session()

        mock_action = MagicMock()
        mock_action.param_model = MagicMock()
        mock_action.function = AsyncMock()

        mock_tools = MagicMock()
        mock_tools.registry.registry.actions = {"evaluate": mock_action}

        ns = create_namespace(bs, tools=mock_tools)
        # evaluate should be the custom one, not the tools one
        assert asyncio.iscoroutinefunction(ns["evaluate"])

    def test_list_downloads_returns_empty_with_no_path(self):
        bs = self._make_mock_browser_session()
        bs.browser_profile.downloads_path = None
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        result = ns["list_downloads"]()
        assert result == []

    def test_list_downloads_returns_files(self, tmp_path):
        bs = self._make_mock_browser_session()
        bs.browser_profile.downloads_path = str(tmp_path)
        tools = self._make_mock_tools()

        # Create some files
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.pdf").write_text("b")

        ns = create_namespace(bs, tools=tools)
        result = ns["list_downloads"]()
        assert len(result) == 2
        assert any("file1.txt" in r for r in result)
        assert any("file2.pdf" in r for r in result)

    def test_list_downloads_returns_empty_for_nonexistent_dir(self):
        bs = self._make_mock_browser_session()
        bs.browser_profile.downloads_path = "/nonexistent/path/12345"
        tools = self._make_mock_tools()
        ns = create_namespace(bs, tools=tools)

        result = ns["list_downloads"]()
        assert result == []


# ---------------------------------------------------------------------------
# evaluate_wrapper inside create_namespace
# ---------------------------------------------------------------------------


class TestEvaluateWrapper:
    def _make_ns(self, eval_return_value="ok"):
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": eval_return_value}}
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        mock_tools = MagicMock()
        mock_tools.registry.registry.actions = {}

        ns = create_namespace(mock_session, tools=mock_tools)
        return ns, mock_cdp

    def _get_sent_expression(self, mock_cdp):
        """Extract the JavaScript expression that was sent to CDP."""
        call_args = mock_cdp.cdp_client.send.Runtime.evaluate.call_args
        return call_args.kwargs["params"]["expression"]

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_no_code_raises(self):
        ns, _ = self._make_ns()
        with pytest.raises(ValueError, match="No JavaScript code"):
            await ns["evaluate"]()

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_auto_wraps_single_expression(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("document.title")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "document.title" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_auto_wraps_multi_statement(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("var x = 1;\nreturn x;")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "var x = 1" in sent
        assert "return x" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_does_not_double_wrap_iife(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("(function(){return 1})()")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        # Should not be double-wrapped -- the IIFE should appear directly
        assert "(function(){return 1})()" in sent
        # Verify no double-wrapping: the expression should NOT contain a nested
        # (async function(){ ... (function(){return 1})() ... })() wrapper
        assert sent.count("(function()") == 1, (
            f"IIFE was double-wrapped: {sent}"
        )

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_variables(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("return params.x", variables={"x": 10})
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "params" in sent
        assert "10" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_variables_and_iife(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("(function(){return params.x})()", variables={"x": 5})
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "params.x" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_variables_and_arrow_iife(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("(() => { return params.x })()", variables={"x": 5})
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "params.x" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_parameterized_function(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("(function(params){ return params.x })", variables={"x": 5})
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "params" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_keyword_arg_code(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"](code="document.title")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "document.title" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_keyword_js_code(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"](js_code="document.title")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "document.title" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_keyword_expression(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"](expression="document.title")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "document.title" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_tracks_failures(self):
        """The evaluate wrapper tracks failures in _evaluate_failures list."""
        mock_session = MagicMock()
        mock_session.get_or_create_cdp_session = AsyncMock(
            side_effect=RuntimeError("fail")
        )
        mock_tools = MagicMock()
        mock_tools.registry.registry.actions = {}
        ns = create_namespace(mock_session, tools=mock_tools)

        with pytest.raises(RuntimeError, match="fail"):
            await ns["evaluate"]("code")
        assert len(ns["_evaluate_failures"]) == 1
        assert ns["_evaluate_failures"][0]["error"] == "fail"

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_does_not_wrap_statement_prefixes(self):
        """Multi-statement code starting with var/let/const should not get 'return' prepended."""
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("var x = 1")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "var x = 1" in sent
        # Should NOT have 'return var x = 1' which would be a syntax error
        assert "return var" not in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_async_iife(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("(async function(){return 1})()")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "async function" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_async_arrow_iife(self):
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("(async () => { return 1 })()")
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "async" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_variables_with_arrow_expression_fallback(self):
        """Arrow function with expression body falls back to outer wrapper."""
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("(() => 42)()", variables={"x": 1})
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "42" in sent

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_variables_with_non_wrapped_code(self):
        """Non-IIFE code with variables gets wrapped with params."""
        ns, mock_cdp = self._make_ns()
        result = await ns["evaluate"]("return params.x", variables={"x": 10})
        assert result == "ok"
        sent = self._get_sent_expression(mock_cdp)
        assert "params" in sent


# ---------------------------------------------------------------------------
# get_namespace_documentation()
# ---------------------------------------------------------------------------


class TestGetNamespaceDocumentation:
    def test_generates_documentation_for_callable(self):
        def my_func():
            """Does something useful."""
            pass

        ns = {"my_func": my_func, "_private": lambda: None, "non_callable": 42}
        doc = get_namespace_documentation(ns)
        assert "my_func" in doc
        assert "Does something useful" in doc
        assert "_private" not in doc
        assert "non_callable" not in doc

    def test_empty_namespace(self):
        doc = get_namespace_documentation({})
        assert "Available Functions" in doc

    def test_skips_callable_without_docstring(self):
        ns = {"no_doc": lambda: None}
        doc = get_namespace_documentation(ns)
        assert "no_doc" not in doc or "##" not in doc.split("no_doc")[0] if "no_doc" in doc else True
