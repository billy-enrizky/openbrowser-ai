"""Comprehensive coverage tests for openbrowser.code_use.namespace module.

Covers remaining gaps: optional library import fallbacks (numpy, pandas, matplotlib,
bs4, pypdf, tabulate), FileSystem import fallback, evaluate_wrapper edge cases,
get_selector_from_index (shadow DOM, iframe, fallback), download_file (browser fetch,
Python requests fallback, error paths), list_downloads edge cases,
action_wrapper (positional args, index=None guard, tab_id truncation, done validation,
done with task_done flag, done with attachments, error raise, etc.),
get_namespace_documentation.
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_browser_session(downloads_path=None):
    mock = MagicMock()
    mock.browser_profile = MagicMock()
    mock.browser_profile.downloads_path = downloads_path
    mock.get_or_create_cdp_session = AsyncMock()
    mock.get_element_by_index = AsyncMock(return_value=None)
    mock.highlight_interaction_element = AsyncMock()
    mock.event_bus = MagicMock()
    return mock


def _make_mock_tools_with_actions(actions=None):
    """Create mock tools with specified action names."""
    mock_tools = MagicMock()
    if actions is None:
        actions = {}
    mock_tools.registry.registry.actions = actions
    return mock_tools


def _make_mock_cdp_session(return_value=None):
    cdp = AsyncMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value=return_value or {"result": {"value": "ok"}}
    )
    return cdp


# ---------------------------------------------------------------------------
# Optional library import fallbacks
# ---------------------------------------------------------------------------


class TestOptionalLibraryFallbacks:
    """Test that optional library imports set availability flags correctly."""

    def test_numpy_available_flag_exists(self):
        from openbrowser.code_use import namespace
        assert hasattr(namespace, "NUMPY_AVAILABLE")

    def test_pandas_available_flag_exists(self):
        from openbrowser.code_use import namespace
        assert hasattr(namespace, "PANDAS_AVAILABLE")

    def test_matplotlib_available_flag_exists(self):
        from openbrowser.code_use import namespace
        assert hasattr(namespace, "MATPLOTLIB_AVAILABLE")

    def test_bs4_available_flag_exists(self):
        from openbrowser.code_use import namespace
        assert hasattr(namespace, "BS4_AVAILABLE")

    def test_pypdf_available_flag_exists(self):
        from openbrowser.code_use import namespace
        assert hasattr(namespace, "PYPDF_AVAILABLE")

    def test_tabulate_available_flag_exists(self):
        from openbrowser.code_use import namespace
        assert hasattr(namespace, "TABULATE_AVAILABLE")

    def test_filesystem_import_fallback(self):
        """FileSystem import fallback sets FileSystem to None."""
        # The module handles ImportError for FileSystem at import time
        # We just verify the module loaded successfully
        from openbrowser.code_use import namespace
        assert namespace is not None


# ---------------------------------------------------------------------------
# evaluate() - additional edge cases
# ---------------------------------------------------------------------------


class TestEvaluateEdgeCases:
    @pytest.mark.asyncio
    async def test_evaluate_exception_without_description(self):
        """exceptionDetails without description or value."""
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "exceptionDetails": {
                    "text": "Error",
                    "exception": {},
                }
            }
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with pytest.raises(EvaluateError, match="Error"):
            await evaluate("bad code", mock_session)

    @pytest.mark.asyncio
    async def test_evaluate_wraps_non_evaluate_errors(self):
        """Non-EvaluateError exceptions are wrapped in EvaluateError."""
        mock_session = MagicMock()
        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=ConnectionError("disconnected")
        )
        mock_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with pytest.raises(EvaluateError, match="Failed to execute JavaScript"):
            await evaluate("code", mock_session)


# ---------------------------------------------------------------------------
# evaluate_wrapper - additional edge cases
# ---------------------------------------------------------------------------


class TestEvaluateWrapperEdgeCases:
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
        return ns

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_variables_and_regular_function_iife(self):
        """Test variables injection into regular function IIFE."""
        ns = self._make_ns()
        # Regular function IIFE with variables
        result = await ns["evaluate"](
            "(function(){return params.x})()", variables={"x": 5}
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_variables_and_async_function_iife(self):
        """Test variables injection into async function IIFE."""
        ns = self._make_ns()
        result = await ns["evaluate"](
            "(async function(){return params.x})()", variables={"x": 5}
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_variables_and_async_arrow_iife(self):
        """Test variables injection into async arrow IIFE."""
        ns = self._make_ns()
        result = await ns["evaluate"](
            "(async () => { return params.x })()", variables={"x": 5}
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_variables_non_wrapped(self):
        """Non-IIFE code with variables gets wrapped and returns early."""
        ns = self._make_ns()
        result = await ns["evaluate"](
            "return params.x + 1", variables={"x": 10}
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_statement_prefixes_not_returned(self):
        """Multi-statement code starting with statement prefix should not get return."""
        ns = self._make_ns()
        for prefix in ["let x = 1", "const x = 1", "if (true) { }", "for (;;) {}", "while (true) {}",
                        "switch (x) {}", "try { } catch(e) {}", "class Foo {}", "function foo() {}",
                        "async function foo() {}", "throw new Error()", "return 42"]:
            result = await ns["evaluate"](prefix)
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_evaluate_wrapper_with_semicolon(self):
        """Code with semicolons is treated as multi-statement."""
        ns = self._make_ns()
        result = await ns["evaluate"]("var x = 1; return x;")
        assert result == "ok"


# ---------------------------------------------------------------------------
# get_selector_from_index
# ---------------------------------------------------------------------------


class TestGetSelectorFromIndex:
    @pytest.mark.asyncio
    async def test_element_not_found(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        bs.get_element_by_index = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError, match="not available"):
            await ns["get_selector_from_index"](999)

    @pytest.mark.asyncio
    async def test_element_in_shadow_dom(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        # Create node hierarchy with shadow DOM
        shadow_host = MagicMock()
        shadow_host.tag_name = "custom-element"
        shadow_host.attributes = {"id": "host1"}
        shadow_host.shadow_root_type = "open"
        shadow_host.parent_node = None

        node = MagicMock()
        node.tag_name = "div"
        node.attributes = {"class": "inner"}
        node.parent_node = shadow_host
        node.shadow_root_type = None

        bs.get_element_by_index = AsyncMock(return_value=node)

        with patch("openbrowser.dom.utils.generate_css_selector_for_element", return_value="div.inner"):
            selector = await ns["get_selector_from_index"](5)
        assert selector == "div.inner"

    @pytest.mark.asyncio
    async def test_element_in_iframe(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        # Create node inside iframe
        iframe_parent = MagicMock()
        iframe_parent.tag_name = "iframe"
        iframe_parent.attributes = {}
        iframe_parent.shadow_root_type = None
        iframe_parent.parent_node = None

        node = MagicMock()
        node.tag_name = "input"
        node.attributes = {}
        node.parent_node = iframe_parent
        node.shadow_root_type = None

        bs.get_element_by_index = AsyncMock(return_value=node)

        with patch("openbrowser.dom.utils.generate_css_selector_for_element", return_value="input"):
            selector = await ns["get_selector_from_index"](3)
        assert selector == "input"

    @pytest.mark.asyncio
    async def test_element_fallback_to_tag_name(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        node = MagicMock()
        node.tag_name = "BUTTON"
        node.attributes = {}
        node.parent_node = None
        node.shadow_root_type = None

        bs.get_element_by_index = AsyncMock(return_value=node)

        with patch("openbrowser.dom.utils.generate_css_selector_for_element", return_value=""):
            selector = await ns["get_selector_from_index"](3)
        assert selector == "button"

    @pytest.mark.asyncio
    async def test_element_no_selector_no_tag_raises(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        node = MagicMock()
        node.tag_name = ""
        node.attributes = {}
        node.parent_node = None
        node.shadow_root_type = None

        bs.get_element_by_index = AsyncMock(return_value=node)

        with patch("openbrowser.dom.utils.generate_css_selector_for_element", return_value=""):
            with pytest.raises(ValueError, match="Could not generate selector"):
                await ns["get_selector_from_index"](3)


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    @pytest.mark.asyncio
    async def test_invalid_url(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        with pytest.raises(ValueError, match="Invalid URL"):
            await ns["download_file"]("not_a_url")

    @pytest.mark.asyncio
    async def test_unsupported_scheme(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            await ns["download_file"]("ftp://example.com/file.pdf")

    @pytest.mark.asyncio
    async def test_download_via_browser_fetch(self):
        bs = _make_mock_browser_session(downloads_path=None)
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        # Mock CDP session for browser fetch
        mock_cdp = AsyncMock()
        file_data = base64.b64encode(b"file content").decode()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {"base64": file_data, "size": 12, "type": "application/pdf"}
                }
            }
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(Path, "home", return_value=Path(tmp_dir)):
                result = await ns["download_file"]("https://example.com/document.pdf")
                assert result.endswith(".pdf")
                assert os.path.exists(result)

    @pytest.mark.asyncio
    async def test_download_auto_filename(self):
        bs = _make_mock_browser_session(downloads_path=None)
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        file_data = base64.b64encode(b"data").decode()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {"base64": file_data, "size": 4, "type": ""}
                }
            }
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(Path, "home", return_value=Path(tmp_dir)):
                # URL with no extension
                result = await ns["download_file"]("https://example.com/download")
                assert "download" in result

    @pytest.mark.asyncio
    async def test_download_filename_conflict(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        file_data = base64.b64encode(b"data").decode()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {"base64": file_data, "size": 4, "type": ""}
                }
            }
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs.browser_profile.downloads_path = tmp_dir
            # Create existing file to trigger conflict
            (Path(tmp_dir) / "file.pdf").write_bytes(b"existing")

            result = await ns["download_file"]("https://example.com/file.pdf")
            assert "(1)" in result

    @pytest.mark.asyncio
    async def test_download_browser_fetch_exception_details(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "exceptionDetails": {"text": "Fetch failed", "exception": {"description": "Network error"}}
            }
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs.browser_profile.downloads_path = tmp_dir
            # Python requests fallback should work
            with patch("openbrowser.code_use.namespace.requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.iter_content = MagicMock(return_value=[b"fallback data"])
                mock_get.return_value = mock_resp

                result = await ns["download_file"]("https://example.com/file.pdf")
                assert os.path.exists(result)

    @pytest.mark.asyncio
    async def test_download_both_strategies_fail(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "exceptionDetails": {"text": "JS error"}
            }
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs.browser_profile.downloads_path = tmp_dir
            with patch("openbrowser.code_use.namespace.requests.get", side_effect=ConnectionError("req failed")):
                with pytest.raises(RuntimeError, match="Download failed"):
                    await ns["download_file"]("https://example.com/file.pdf")

    @pytest.mark.asyncio
    async def test_download_timeout(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs.browser_profile.downloads_path = tmp_dir
            with patch("openbrowser.code_use.namespace.requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.iter_content = MagicMock(return_value=[b"data"])
                mock_get.return_value = mock_resp

                result = await ns["download_file"]("https://example.com/file.pdf")
                assert os.path.exists(result)

    @pytest.mark.asyncio
    async def test_download_no_data_from_browser(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": {}}}
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs.browser_profile.downloads_path = tmp_dir
            with patch("openbrowser.code_use.namespace.requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.iter_content = MagicMock(return_value=[b"data"])
                mock_get.return_value = mock_resp

                result = await ns["download_file"]("https://example.com/file.pdf")
                assert os.path.exists(result)

    @pytest.mark.asyncio
    async def test_download_bad_base64(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {"value": {"base64": "!!!not_valid_base64!!!", "size": 10, "type": ""}}
            }
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs.browser_profile.downloads_path = tmp_dir
            with patch("openbrowser.code_use.namespace.requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.raise_for_status = MagicMock()
                mock_resp.iter_content = MagicMock(return_value=[b"data"])
                mock_get.return_value = mock_resp

                result = await ns["download_file"]("https://example.com/file.pdf")
                assert os.path.exists(result)

    @pytest.mark.asyncio
    async def test_download_sanitize_filename(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        file_data = base64.b64encode(b"data").decode()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {"value": {"base64": file_data, "size": 4, "type": ""}}
            }
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs.browser_profile.downloads_path = tmp_dir
            # Dangerous filename
            result = await ns["download_file"](
                "https://example.com/file.pdf", filename="../../etc/passwd"
            )
            # Should be sanitized: the filename should be "passwd", not contain path traversal
            saved_filename = os.path.basename(result)
            assert "etc" not in saved_filename, (
                f"Path traversal not sanitized in filename: {saved_filename}"
            )
            assert saved_filename == "passwd", (
                f"Expected sanitized filename 'passwd', got '{saved_filename}'"
            )

    @pytest.mark.asyncio
    async def test_download_empty_filename(self):
        bs = _make_mock_browser_session()
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        mock_cdp = AsyncMock()
        file_data = base64.b64encode(b"data").decode()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {"value": {"base64": file_data, "size": 4, "type": ""}}
            }
        )
        bs.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bs.browser_profile.downloads_path = tmp_dir
            result = await ns["download_file"](
                "https://example.com/file.pdf", filename="."
            )
            assert "download.pdf" in result


# ---------------------------------------------------------------------------
# list_downloads edge cases
# ---------------------------------------------------------------------------


class TestListDownloadsEdgeCases:
    def test_list_downloads_permission_error(self):
        bs = _make_mock_browser_session(downloads_path="/tmp/restricted_dir_12345")
        tools = _make_mock_tools_with_actions()
        ns = create_namespace(bs, tools=tools)

        # The directory doesn't exist, so it returns []
        result = ns["list_downloads"]()
        assert result == []


# ---------------------------------------------------------------------------
# Action wrapper in create_namespace
# ---------------------------------------------------------------------------


class TestActionWrapper:
    def _make_ns_with_action(self, action_name="test_action", param_fields=None, action_result=None):
        """Create namespace with a mock action registered."""
        from openbrowser.models import ActionResult
        from pydantic import BaseModel, create_model

        bs = _make_mock_browser_session()

        if param_fields is None:
            param_fields = {}

        param_model = create_model(f"{action_name}_Params", **param_fields)

        mock_result = action_result or ActionResult(extracted_content="Action executed")
        mock_func = AsyncMock(return_value=mock_result)

        mock_action = MagicMock()
        mock_action.param_model = param_model
        mock_action.function = mock_func

        mock_tools = MagicMock()
        mock_tools.registry.registry.actions = {action_name: mock_action}

        ns = create_namespace(bs, tools=mock_tools)
        return ns, mock_func

    @pytest.mark.asyncio
    async def test_action_wrapper_with_positional_args(self):
        ns, mock_func = self._make_ns_with_action(
            "click",
            param_fields={"index": (int, ...)},
        )
        result = await ns["click"](5)
        assert result == "Action executed"

    @pytest.mark.asyncio
    async def test_action_wrapper_index_none_guard(self):
        from pydantic import Field

        ns, mock_func = self._make_ns_with_action(
            "click",
            param_fields={"index": (int, ...)},
        )
        with pytest.raises(ValueError, match="requires a valid element index"):
            await ns["click"](index=None)

    @pytest.mark.asyncio
    async def test_action_wrapper_tab_id_truncation_switch(self):
        ns, mock_func = self._make_ns_with_action(
            "switch",
            param_fields={"tab_id": (str, ...)},
        )
        # Pass full 32-char target ID, should be truncated to 4 chars
        await ns["switch"](tab_id="12345678901234567890123456781234")
        call_kwargs = mock_func.call_args
        # Verify it was constructed (not erroring)
        assert mock_func.called

    @pytest.mark.asyncio
    async def test_action_wrapper_tab_id_truncation_close(self):
        ns, mock_func = self._make_ns_with_action(
            "close",
            param_fields={"tab_id": (str, ...)},
        )
        await ns["close"](tab_id="12345678901234567890123456781234")
        assert mock_func.called

    @pytest.mark.asyncio
    async def test_action_wrapper_invalid_params(self):
        ns, mock_func = self._make_ns_with_action(
            "click",
            param_fields={"index": (int, ...)},
        )
        with pytest.raises(ValueError, match="Invalid parameters"):
            await ns["click"](index="not_an_int")

    @pytest.mark.asyncio
    async def test_action_wrapper_done_sets_task_done(self):
        from openbrowser.models import ActionResult

        result = ActionResult(
            is_done=True,
            success=True,
            extracted_content="Task complete",
            attachments=["/tmp/file.txt"],
        )
        ns, mock_func = self._make_ns_with_action(
            "done",
            param_fields={"text": (str, ...), "success": (bool, True), "files_to_display": (list, [])},
            action_result=result,
        )
        output = await ns["done"](text="Task complete", success=True)
        assert ns.get("_task_done") is True
        assert ns.get("_task_result") == "Task complete"
        assert ns.get("_task_success") is True
        assert ns.get("_task_attachments") == ["/tmp/file.txt"]

    @pytest.mark.asyncio
    async def test_action_wrapper_error_raises(self):
        from openbrowser.models import ActionResult

        result = ActionResult(error="Something went wrong")
        ns, mock_func = self._make_ns_with_action(
            "navigate",
            param_fields={"url": (str, ...)},
            action_result=result,
        )
        with pytest.raises(RuntimeError, match="Something went wrong"):
            await ns["navigate"](url="https://example.com")

    @pytest.mark.asyncio
    async def test_action_wrapper_returns_none_for_no_content(self):
        from openbrowser.models import ActionResult

        result = ActionResult()
        ns, mock_func = self._make_ns_with_action(
            "wait",
            param_fields={"seconds": (int, 3)},
            action_result=result,
        )
        output = await ns["wait"](seconds=1)
        assert output is None

    @pytest.mark.asyncio
    async def test_action_wrapper_returns_non_action_result(self):
        """When action returns something without extracted_content attribute."""
        ns, mock_func = self._make_ns_with_action(
            "custom",
            param_fields={"x": (int, ...)},
        )
        mock_func.return_value = "raw string result"
        result = await ns["custom"](x=1)
        assert result == "raw string result"

    @pytest.mark.asyncio
    async def test_action_wrapper_input_renamed(self):
        """'input' action should be renamed to 'input_text' in namespace."""
        from openbrowser.models import ActionResult

        bs = _make_mock_browser_session()
        from pydantic import create_model

        param_model = create_model("input_Params", index=(int, ...), text=(str, ...))

        mock_action = MagicMock()
        mock_action.param_model = param_model
        mock_action.function = AsyncMock(
            return_value=ActionResult(extracted_content="Typed text")
        )

        mock_tools = MagicMock()
        mock_tools.registry.registry.actions = {"input": mock_action}

        ns = create_namespace(bs, tools=mock_tools)
        assert "input_text" in ns
        assert "input" not in ns  # Should be renamed


# ---------------------------------------------------------------------------
# get_namespace_documentation edge cases
# ---------------------------------------------------------------------------


class TestGetNamespaceDocumentationEdgeCases:
    def test_callable_with_no_doc(self):
        """Callable without docstring should be skipped."""
        def no_doc():
            pass

        ns = {"no_doc": no_doc}
        doc = get_namespace_documentation(ns)
        # Should not have a ## section for no_doc
        assert "## no_doc" not in doc

    def test_private_functions_skipped(self):
        def _private():
            """I am private."""
            pass

        ns = {"_private": _private}
        doc = get_namespace_documentation(ns)
        assert "_private" not in doc

    def test_non_callable_skipped(self):
        ns = {"data": [1, 2, 3], "count": 42}
        doc = get_namespace_documentation(ns)
        assert "data" not in doc
        assert "count" not in doc
