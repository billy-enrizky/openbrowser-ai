"""Comprehensive tests for openbrowser.code_use.namespace module.

Covers: _strip_js_comments, EvaluateError, evaluate(), validate_task_completion(),
create_namespace(), get_namespace_documentation(), download_file, list_downloads.
"""

import asyncio
import logging
import time
from types import SimpleNamespace
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

        is_complete, _reasoning = await validate_task_completion("task", "output", mock_llm)
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

        is_complete, _reasoning = await validate_task_completion("task", None, mock_llm)
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

        is_complete, _reasoning = await validate_task_completion("task", "output", mock_llm)
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



# ---------------------------------------------------------------------------
# Interactive element helpers
# ---------------------------------------------------------------------------


class TestInteractiveHelpers:
    def _make_session(self, summaries):
        session = MagicMock()
        session.browser_profile = MagicMock()
        session.browser_profile.downloads_path = None
        session.get_browser_state_summary = AsyncMock(side_effect=summaries)
        return session

    def _make_tools(self):
        tools = MagicMock()
        tools.registry.registry.actions = {}
        return tools

    def _node(self, *, tag='div', element_id=None, role=None, name=None):
        node = MagicMock()
        node.tag_name = tag
        node.attributes = {'id': element_id} if element_id else {}
        if role is None and name is None:
            node.ax_node = None
        else:
            node.ax_node = SimpleNamespace(role=role, name=name)
        return node

    def _summary(self, nodes):
        summary = MagicMock()
        summary.dom_state.selector_map = nodes
        return summary

    def _namespace(self, session):
        return create_namespace(session, tools=self._make_tools())

    @pytest.mark.asyncio
    async def test_find_element_by_role_name(self):
        node = self._node(tag='input', role='textbox', name='Email')
        session = self._make_session([self._summary({7: node})])
        ns = self._namespace(session)

        index = await ns['find_element'](role='textbox', name='Email')

        assert index == 7
        session.get_browser_state_summary.assert_awaited_once_with(include_screenshot=False)

    @pytest.mark.asyncio
    async def test_find_element_by_id_and_tag(self):
        node = self._node(tag='INPUT', element_id='email', role='textbox', name='Email')
        summary = self._summary({4: node})
        session = self._make_session([summary, summary])
        ns = self._namespace(session)

        assert await ns['find_element'](element_id='email') == 4
        assert await ns['find_element'](tag='input', name='Email') == 4

    @pytest.mark.asyncio
    async def test_find_element_requires_a_filter(self):
        session = self._make_session([self._summary({})])
        ns = self._namespace(session)

        with pytest.raises(ValueError, match='needs role, name, element_id, or tag'):
            await ns['find_element']()

    @pytest.mark.asyncio
    async def test_find_element_reports_a_missing_element(self):
        session = self._make_session([self._summary({})])
        ns = self._namespace(session)

        with pytest.raises(RuntimeError, match="role='button'"):
            await ns['find_element'](role='button', name='Next')

    @pytest.mark.asyncio
    async def test_find_element_polls_until_the_element_appears(self):
        node = self._node(tag='button', role='button', name='Next')
        session = self._make_session([self._summary({}), self._summary({2: node})])
        ns = self._namespace(session)

        index = await ns['find_element'](role='button', name='Next', timeout=0.2, poll_interval=0.01)

        assert index == 2
        assert session.get_browser_state_summary.await_count == 2

    @pytest.mark.asyncio
    async def test_find_element_requires_every_supplied_filter_to_match(self):
        wrong = self._node(tag='input', element_id='shipping-email', role='textbox', name='Email')
        right = self._node(tag='input', element_id='billing-email', role='textbox', name='Email')
        session = self._make_session([self._summary({1: wrong, 2: right})])
        ns = self._namespace(session)

        index = await ns['find_element'](
            element_id='billing-email',
            role='textbox',
            name='Email',
        )

        assert index == 2

    @pytest.mark.asyncio
    async def test_find_element_caps_sleep_at_the_remaining_timeout(self):
        session = self._make_session([self._summary({})])
        ns = self._namespace(session)

        started = time.monotonic()
        with pytest.raises(RuntimeError, match='Element not found'):
            await ns['find_element'](role='button', timeout=0.05, poll_interval=10)

        assert time.monotonic() - started < 1.0

    @pytest.mark.asyncio
    async def test_find_element_does_not_return_after_the_deadline(self):
        node = self._node(tag='button', role='button', name='Next')

        async def slow_summary(*args, **kwargs):
            await asyncio.sleep(0.05)
            return self._summary({2: node})

        session = self._make_session([])
        session.get_browser_state_summary = AsyncMock(side_effect=slow_summary)
        ns = self._namespace(session)

        with pytest.raises(RuntimeError, match='Element not found'):
            await ns['find_element'](role='button', name='Next', timeout=0.01)

    @pytest.mark.asyncio
    async def test_click_element_resolves_and_clicks(self):
        node = self._node(tag='button', role='button', name='Next')
        session = self._make_session([self._summary({3: node})])
        ns = self._namespace(session)
        ns['click'] = AsyncMock(return_value={'clicked': True})

        result = await ns['click_element'](role='button', name='Next')

        assert result == {'clicked': True}
        ns['click'].assert_awaited_once_with(index=3)

    @pytest.mark.asyncio
    async def test_fill_element_resolves_and_types(self):
        node = self._node(tag='input', element_id='seats', role='textbox', name='Seats')
        session = self._make_session([self._summary({5: node})])
        ns = self._namespace(session)
        ns['input_text'] = AsyncMock(return_value={'typed': True})

        result = await ns['fill_element']('5', element_id='seats')

        assert result == {'typed': True}
        ns['input_text'].assert_awaited_once_with(index=5, text='5')

    @pytest.mark.asyncio
    async def test_get_tabs_uses_the_cdp_tab_list(self):
        session = self._make_session([self._summary({})])
        session.current_target_id = 'TARGET-TWO'
        session.get_tabs = AsyncMock(
            return_value=[
                SimpleNamespace(
                    target_id='TARGET-ONE',
                    url='https://one.example',
                    title='One',
                    parent_target_id=None,
                ),
                SimpleNamespace(
                    target_id='TARGET-TWO',
                    url='https://two.example',
                    title='Two',
                    parent_target_id='TARGET-ONE',
                ),
            ]
        )
        ns = self._namespace(session)

        tabs = await ns['get_tabs']()

        assert tabs == [
            {
                'target_id': 'TARGET-ONE',
                'url': 'https://one.example',
                'title': 'One',
                'parent_target_id': None,
                'active': False,
            },
            {
                'target_id': 'TARGET-TWO',
                'url': 'https://two.example',
                'title': 'Two',
                'parent_target_id': 'TARGET-ONE',
                'active': True,
            },
        ]
        session.get_tabs.assert_awaited_once_with()
        session.get_browser_state_summary.assert_not_awaited()

    def test_descriptions_and_namespace_docs_name_the_helpers(self):
        from openbrowser.code_use.descriptions import (
            EXECUTE_CODE_DESCRIPTION,
            EXECUTE_CODE_DESCRIPTION_COMPACT,
        )

        for description in (EXECUTE_CODE_DESCRIPTION_COMPACT, EXECUTE_CODE_DESCRIPTION):
            assert 'find_element' in description
            assert 'click_element' in description
            assert 'fill_element' in description
            assert 'get_tabs' in description

        session = self._make_session([self._summary({})])
        docs = get_namespace_documentation(self._namespace(session))
        assert '## find_element' in docs
        assert '## click_element' in docs
        assert '## fill_element' in docs
        assert '## get_tabs' in docs

# ---------------------------------------------------------------------------
# scrape_page()
# ---------------------------------------------------------------------------


class TestScrapeHttp:
    def _response(self, *, text='', status=200, content_type='text/html', payload=None, url='https://example.com'):
        response = MagicMock()
        response.status_code = status
        response.headers = {'content-type': content_type}
        response.text = text
        response.url = url
        response.json = MagicMock(return_value=payload)
        return response

    def _client(self, response):
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(return_value=response)
        return client

    @pytest.mark.asyncio
    async def test_http_markdown_returns_compact_content(self):
        from openbrowser.code_use.namespace import scrape_http

        html = '<html><head><title>T</title></head><body><h1>H</h1><script>x</script><p>Body</p></body></html>'
        with patch('httpx.AsyncClient', return_value=self._client(self._response(text=html))):
            result = await scrape_http(
                'https://example.com',
                format='markdown',
                selector=None,
                extract_links=False,
                timeout_ms=None,
                engine='http',
            )
        assert result is not None
        assert result['engine_used'] == 'http'
        assert '# H' in result['content']
        assert 'Body' in result['content']
        assert 'x' not in result['content']

    @pytest.mark.asyncio
    async def test_auto_uses_http_for_substantial_text(self):
        from openbrowser.code_use.namespace import scrape_http

        paragraph = 'Substantial server rendered content. ' * 5
        html = f'<html><head><title>T</title></head><body><p>{paragraph}</p><script>x</script></body></html>'
        with patch('httpx.AsyncClient', return_value=self._client(self._response(text=html))):
            result = await scrape_http(
                'https://example.com',
                format='text',
                selector=None,
                extract_links=False,
                timeout_ms=None,
                engine='auto',
            )
        assert result is not None
        assert 'Substantial server rendered content' in result['content']

    def test_static_response_reuses_one_parsed_html_tree(self, monkeypatch):
        from openbrowser.code_use.namespace import _http_result_from_response
        from openbrowser.dom import fast_html

        html = (
            '<html><head><title>Static page</title></head><body>'
            '<h1>Static heading</h1>'
            '<p>Substantial visible text keeps this page on the HTTP extraction path.</p>'
            '</body></html>'
        )
        parse = fast_html._parse
        parse_calls = 0

        def count_parse(value):
            nonlocal parse_calls
            parse_calls += 1
            return parse(value)

        monkeypatch.setattr(fast_html, '_parse', count_parse)
        result = _http_result_from_response(
            self._response(text=html),
            format='markdown',
            selector=None,
            engine='auto',
        )

        assert result is not None
        assert result['title'] == 'Static page'
        assert result['content'] == (
            '# Static heading\n\n'
            'Substantial visible text keeps this page on the HTTP extraction path.'
        )
        assert parse_calls == 1

    @pytest.mark.asyncio
    async def test_auto_falls_back_for_thin_shell(self):
        from openbrowser.code_use.namespace import scrape_http

        html = '<html><body><div id="root"></div><script>mount()</script></body></html>'
        with patch('httpx.AsyncClient', return_value=self._client(self._response(text=html))):
            result = await scrape_http(
                'https://example.com',
                format='markdown',
                selector=None,
                extract_links=False,
                timeout_ms=None,
                engine='auto',
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_http_engine_does_not_fall_back(self):
        from openbrowser.code_use.namespace import scrape_http

        html = '<html><body><div id="root"></div><p>Short</p></body></html>'
        with patch('httpx.AsyncClient', return_value=self._client(self._response(text=html))):
            result = await scrape_http(
                'https://example.com',
                format='text',
                selector=None,
                extract_links=False,
                timeout_ms=None,
                engine='http',
            )
        assert result is not None
        assert result['content'] == 'Short'

    @pytest.mark.asyncio
    async def test_auto_falls_back_for_an_empty_application_root_with_static_text(self):
        from openbrowser.code_use.namespace import scrape_http

        html = (
            '<html><body><div id="root"></div>'
            "<nav>Static navigation text that is long enough to pass the length probe.</nav></body></html>"
        )
        with patch('httpx.AsyncClient', return_value=self._client(self._response(text=html))):
            result = await scrape_http(
                'https://example.com',
                format='markdown',
                selector=None,
                extract_links=False,
                timeout_ms=None,
                engine='auto',
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_http_client_does_not_persist_response_cookies(self):
        import httpx

        from openbrowser.code_use.namespace import _NoCookieJar, scrape_http

        seen_cookies: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_cookies.append(request.headers.get('cookie'))
            return httpx.Response(
                200,
                headers={'content-type': 'text/html', 'set-cookie': 'session=abc; Path=/'},
                text='<html><body><p>Body</p></body></html>',
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), cookies=_NoCookieJar())
        try:
            for _ in range(2):
                result = await scrape_http(
                    'https://example.com',
                    format='text',
                    selector=None,
                    extract_links=False,
                    timeout_ms=None,
                    engine='http',
                    client=client,
                )
                assert result is not None
        finally:
            await client.aclose()

        assert seen_cookies == [None, None]
        assert len(client.cookies) == 0

    @pytest.mark.asyncio
    async def test_http_client_owner_closes_once(self):
        from openbrowser.code_use.namespace import _HttpClientOwner

        owner = _HttpClientOwner()
        client = owner.get(1.0)
        assert not client.is_closed

        await owner.close()
        assert client.is_closed
        await owner.close()

    @pytest.mark.asyncio
    async def test_json_content_type_returns_payload(self):
        from openbrowser.code_use.namespace import scrape_http

        with patch(
            'httpx.AsyncClient',
            return_value=self._client(
                self._response(content_type='application/json', payload={'ok': True})
            ),
        ):
            result = await scrape_http(
                'https://example.com/api',
                format='json',
                selector=None,
                extract_links=False,
                timeout_ms=None,
                engine='auto',
            )
        assert result is not None
        assert result['content'] == {'ok': True}


class TestScrapeHttpClientLifecycle:
    @pytest.mark.asyncio
    async def test_namespace_exposes_http_client_cleanup(self):
        session = MagicMock()
        session.browser_profile = MagicMock()
        session.browser_profile.downloads_path = None
        tools = MagicMock()
        tools.registry.registry.actions = {}

        ns = create_namespace(session, tools=tools)

        assert callable(ns['_close_http_client'])
        await ns['_close_http_client']()


class TestScrapePage:
    @pytest.fixture(autouse=True)
    def _disable_http_fast_path(self):
        with patch(
            'openbrowser.code_use.namespace.scrape_http',
            new_callable=AsyncMock,
            return_value=None,
        ):
            yield

    def _make_namespace(self, values):
        session = MagicMock()
        session.browser_profile = MagicMock()
        session.browser_profile.downloads_path = None
        session.navigate_to = AsyncMock()
        session.get_current_page_url = AsyncMock(return_value="https://example.com/final")
        cdp = AsyncMock()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=[{"result": {"value": value}} for value in values]
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
        tools = MagicMock()
        tools.registry.registry.actions = {}
        return create_namespace(session, tools=tools), session, cdp

    @pytest.mark.asyncio
    async def test_text_format_navigates_and_returns_text(self):
        ns, session, _ = self._make_namespace(["Example", "Hello world"])
        result = await ns["scrape_page"]("https://example.com", format="text")
        session.navigate_to.assert_awaited_once()
        assert result["title"] == "Example"
        assert result["url"] == "https://example.com/final"
        assert result["content"] == "Hello world"
        assert result["format"] == "text"

    @pytest.mark.asyncio
    async def test_wait_for_polls_until_selector_appears(self):
        ns, _, _ = self._make_namespace([False, True, "Example", "Ready"])
        result = await ns["scrape_page"](format="text", wait_for=".ready")
        assert result["content"] == "Ready"

    @pytest.mark.asyncio
    async def test_links_format_returns_link_objects(self):
        links = [{"text": "Docs", "href": "https://example.com/docs", "title": None}]
        ns, _, _ = self._make_namespace(["Example", links])
        result = await ns["scrape_page"](format="links")
        assert result["content"] == links
        assert result["content_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_markdown_format_uses_shared_extractor(self):
        ns, _, _ = self._make_namespace(["Example"])
        with patch(
            "openbrowser.dom.markdown_extractor.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=("# Title", {"final_filtered_chars": 7}),
        ):
            result = await ns["scrape_page"](format="markdown", extract_links=True)
        assert result["content"] == "# Title"
        assert result["stats"]["final_filtered_chars"] == 7

    @pytest.mark.asyncio
    async def test_json_selector_returns_matches(self):
        matches = [{"text": "x", "html": "<p>x</p>", "attributes": {}}]
        ns, _, _ = self._make_namespace(["Example", matches])
        result = await ns["scrape_page"](format="json", selector="p")
        assert result["content"] == matches
        assert result["selector"] == "p"

    @pytest.mark.asyncio
    async def test_invalid_format_raises(self):
        ns, _, _ = self._make_namespace([])
        with pytest.raises(ValueError, match="format must be one of"):
            await ns["scrape_page"](format="xml")

    @pytest.mark.asyncio
    async def test_max_chars_truncates_text(self):
        ns, _, _ = self._make_namespace(["Example", "abcdef"])
        result = await ns["scrape_page"](format="text", max_chars=3)
        assert result["content"] == "abc"
        assert result["truncated"] is True
        assert result["total_chars"] == 6

    @pytest.mark.asyncio
    async def test_scrape_alias_points_to_scrape_page(self):
        ns, _, _ = self._make_namespace([])
        assert ns["scrape"] is ns["scrape_page"]


# ---------------------------------------------------------------------------
# scrape_pages()
# ---------------------------------------------------------------------------


class TestScrapePages:
    def _make_namespace(self):
        session = MagicMock()
        session.browser_profile = MagicMock()
        session.browser_profile.downloads_path = None
        session.start = AsyncMock()
        session.kill = AsyncMock()
        tools = MagicMock()
        tools.registry.registry.actions = {}
        return create_namespace(session, tools=tools), session

    @pytest.mark.asyncio
    async def test_http_batch_runs_concurrently_and_keeps_order(self):
        ns, session = self._make_namespace()
        urls = [f"https://example.com/{index}" for index in range(4)]
        active = 0
        max_active = 0
        gate = asyncio.Event()

        async def fake_http(url, **_kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            if active >= 2:
                gate.set()
            try:
                await asyncio.wait_for(gate.wait(), timeout=1)
            finally:
                active -= 1
            return {
                "url": url,
                "title": url.rsplit("/", 1)[-1],
                "format": "text",
                "content_type": "text/text",
                "content": url,
                "stats": {"engine_used": "http"},
                "engine_used": "http",
            }

        with patch("openbrowser.code_use.namespace.scrape_http", new=fake_http):
            results = await ns["scrape_pages"](urls, concurrency=2, format="text", engine="http")

        assert max_active == 2
        assert [result["url"] for result in results] == urls
        assert all(result["ok"] for result in results)
        assert all(result["engine_used"] == "http" for result in results)
        assert all(result["content"] == result["url"] for result in results)
        assert all(result["duration_ms"] >= 0 for result in results)
        assert all("total_ms" in result["timing"] for result in results)
        session.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_http_batch_isolates_failures(self):
        ns, session = self._make_namespace()
        urls = ["https://example.com/ok", "https://example.com/empty", "https://example.com/error"]

        async def fake_http(url, **_kwargs):
            if url.endswith("/ok"):
                return {
                    "url": url,
                    "title": "OK",
                    "format": "text",
                    "content_type": "text/text",
                    "content": "ok",
                    "stats": {"engine_used": "http"},
                    "engine_used": "http",
                }
            if url.endswith("/empty"):
                return None
            raise RuntimeError("network down")

        with patch("openbrowser.code_use.namespace.scrape_http", new=fake_http):
            results = await ns["scrape_pages"](urls, format="text", engine="http")

        assert len(results) == 3
        assert results[0]["ok"] is True
        assert results[1]["ok"] is False
        assert "HTTP scrape failed" in results[1]["error"]
        assert results[2]["ok"] is False
        assert "RuntimeError: network down" in results[2]["error"]
        assert all(result["content"] is None for result in results[1:])
        session.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_browser_fallbacks_run_one_at_a_time_after_http(self):
        ns, session = self._make_namespace()
        urls = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
        browser_calls = []
        browser_active = 0
        browser_max_active = 0
        http_finished = 0

        async def fake_http(_url, **_kwargs):
            nonlocal http_finished
            await asyncio.sleep(0)
            http_finished += 1
            return None

        async def fake_browser(url=None, **_kwargs):
            nonlocal browser_active, browser_max_active
            assert http_finished == len(urls)
            browser_calls.append(url)
            browser_active += 1
            browser_max_active = max(browser_max_active, browser_active)
            try:
                await asyncio.sleep(0.01)
            finally:
                browser_active -= 1
            return {
                "url": url,
                "title": "Browser",
                "format": "text",
                "content_type": "text/text",
                "content": f"browser:{url}",
                "stats": {"engine_used": "browser"},
                "engine_used": "browser",
            }

        ns["scrape_page"] = fake_browser
        with patch("openbrowser.code_use.namespace.scrape_http", new=fake_http):
            results = await ns["scrape_pages"](urls, concurrency=3, format="text", engine="auto")

        assert http_finished == 3
        assert browser_calls == urls
        assert browser_max_active == 1
        assert [result["url"] for result in results] == urls
        assert all(result["ok"] for result in results)
        assert all(result["engine_used"] == "browser" for result in results)
        assert all(result["timing"]["browser_ms"] >= 0 for result in results)
        session.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_browser_failure_does_not_abort_batch(self):
        ns, session = self._make_namespace()
        urls = ["https://example.com/ok", "https://example.com/fail", "https://example.com/next"]

        async def fake_http(_url, **_kwargs):
            return None

        async def fake_browser(url=None, **_kwargs):
            if url.endswith("/fail"):
                raise RuntimeError("browser down")
            return {
                "url": url,
                "title": "Browser",
                "format": "text",
                "content_type": "text/text",
                "content": f"browser:{url}",
                "stats": {"engine_used": "browser"},
                "engine_used": "browser",
            }

        ns["scrape_page"] = fake_browser
        with patch("openbrowser.code_use.namespace.scrape_http", new=fake_http):
            results = await ns["scrape_pages"](urls, format="text", engine="auto")

        assert len(results) == 3
        assert results[0]["ok"] is True
        assert results[1]["ok"] is False
        assert "RuntimeError: browser down" in results[1]["error"]
        assert results[2]["ok"] is True
        assert [result["url"] for result in results] == urls
        session.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_http_max_chars_applies_per_result(self):
        ns, _ = self._make_namespace()

        async def fake_http(url, **_kwargs):
            return {
                "url": url,
                "title": "Long",
                "format": "text",
                "content_type": "text/text",
                "content": "abcdef",
                "stats": {"engine_used": "http"},
                "engine_used": "http",
            }

        with patch("openbrowser.code_use.namespace.scrape_http", new=fake_http):
            results = await ns["scrape_pages"](["https://example.com"], format="text", max_chars=3, engine="http")

        assert results[0]["content"] == "abc"
        assert results[0]["truncated"] is True
        assert results[0]["total_chars"] == 6

    @pytest.mark.asyncio
    async def test_invalid_concurrency_raises(self):
        ns, _ = self._make_namespace()
        with pytest.raises(ValueError, match="concurrency must be a positive integer"):
            await ns["scrape_pages"](["https://example.com"], concurrency=0)

    @pytest.mark.asyncio
    async def test_string_input_is_treated_as_one_url(self):
        ns, _ = self._make_namespace()

        async def fake_http(url, **_kwargs):
            return {
                "url": url,
                "title": "One",
                "format": "text",
                "content_type": "text/text",
                "content": "one",
                "stats": {"engine_used": "http"},
                "engine_used": "http",
            }

        with patch("openbrowser.code_use.namespace.scrape_http", new=fake_http):
            results = await ns["scrape_pages"]("https://example.com", format="text", engine="http")

        assert [result["url"] for result in results] == ["https://example.com"]


def test_scraping_descriptions_cover_single_and_batch():
    from openbrowser.code_use.descriptions import EXECUTE_CODE_DESCRIPTION, EXECUTE_CODE_DESCRIPTION_COMPACT

    for description in (EXECUTE_CODE_DESCRIPTION_COMPACT, EXECUTE_CODE_DESCRIPTION):
        assert "scrape_page" in description
        assert "scrape_pages" in description
        assert "`scrape` is an alias" in description
        assert "auto" in description
        assert "HTTP first" in description
        assert "browser" in description
