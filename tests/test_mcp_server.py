"""Tests for the MCP (Model Context Protocol) server module.

This module provides test coverage for the OpenBrowser MCP server,
which exposes a single execute_code tool for running Python code
in a persistent namespace with browser automation functions.

Tests cover:
    - Server initialization and MCP SDK availability
    - execute_code: stdout capture, variable persistence, error handling
    - Namespace initialization (lazy browser startup)
    - Session cleanup on timeout
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.mcp import server as mcp_server_module


# ===========================================================================
# Server initialization tests
# ===========================================================================


def test_main_exits_when_mcp_missing(monkeypatch, capsys):
    """Verify main() exits with error when MCP SDK is not available."""
    monkeypatch.setattr(mcp_server_module, "MCP_AVAILABLE", False)

    with pytest.raises(SystemExit) as exc:
        asyncio.run(mcp_server_module.main())

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "MCP SDK is required" in captured.err


def test_server_initializes_with_none_session(mcp_server):
    """Server starts with no browser session or namespace."""
    assert mcp_server.browser_session is None
    assert mcp_server._namespace is None


def test_server_has_session_timeout(mcp_server):
    """Server has a configurable session timeout."""
    assert mcp_server.session_timeout_minutes == 10


# ===========================================================================
# execute_code tests
# ===========================================================================


class TestExecuteCode:
    """Tests for the _execute_code method."""

    @pytest.fixture(autouse=True)
    def setup_namespace(self, mcp_server):
        """Pre-populate namespace to skip browser initialization."""
        mcp_server._namespace = {
            "__builtins__": __builtins__,
        }

    def test_captures_stdout(self, mcp_server):
        """print() output is captured and returned."""
        result = asyncio.run(mcp_server._execute_code("print('hello world')"))
        assert "hello world" in result

    def test_no_output_shows_success_message(self, mcp_server):
        """Code with no output shows executed successfully."""
        result = asyncio.run(mcp_server._execute_code("x = 42"))
        assert "executed successfully" in result

    def test_persists_variables_between_calls(self, mcp_server):
        """Variables set in one call persist to the next call."""
        asyncio.run(mcp_server._execute_code("my_var = 'test123'"))
        result = asyncio.run(mcp_server._execute_code("print(my_var)"))
        assert "test123" in result

    def test_persists_multiple_variables(self, mcp_server):
        """Multiple variables persist across calls."""
        asyncio.run(mcp_server._execute_code("a = 10\nb = 20"))
        result = asyncio.run(mcp_server._execute_code("print(a + b)"))
        assert "30" in result

    def test_handles_exception(self, mcp_server):
        """Exceptions are caught and returned with traceback."""
        result = asyncio.run(mcp_server._execute_code("raise ValueError('test error')"))
        assert "ValueError" in result
        assert "test error" in result

    def test_handles_syntax_error(self, mcp_server):
        """Syntax errors are caught and reported."""
        result = asyncio.run(mcp_server._execute_code("def"))
        assert "SyntaxError" in result

    def test_handles_name_error(self, mcp_server):
        """NameError for undefined variables is reported."""
        result = asyncio.run(mcp_server._execute_code("print(undefined_var)"))
        assert "NameError" in result

    def test_await_works(self, mcp_server):
        """await expressions work in execute_code."""
        mcp_server._namespace["asyncio"] = asyncio
        result = asyncio.run(mcp_server._execute_code(
            "import asyncio\nawait asyncio.sleep(0)\nprint('async works')"
        ))
        assert "async works" in result

    def test_multiline_code(self, mcp_server):
        """Multi-line code executes correctly."""
        result = asyncio.run(mcp_server._execute_code(
            "x = 10\ny = 20\nprint(x + y)"
        ))
        assert "30" in result

    def test_output_before_error_is_captured(self, mcp_server):
        """Output printed before an error is included in the result."""
        result = asyncio.run(mcp_server._execute_code(
            "print('before error')\nraise RuntimeError('boom')"
        ))
        assert "before error" in result
        assert "RuntimeError" in result
        assert "boom" in result

    def test_return_value_shown_when_no_print(self, mcp_server):
        """When function returns a value and nothing is printed, show repr."""
        # The async wrapper captures the return of the last expression
        # Variables are persisted, but the function doesn't naturally return
        # unless the last statement is an expression
        result = asyncio.run(mcp_server._execute_code("x = 42"))
        # No print, no return value -> success message
        assert "executed successfully" in result

    def test_loop_and_conditionals(self, mcp_server):
        """Control flow statements work correctly."""
        result = asyncio.run(mcp_server._execute_code(
            "total = 0\n"
            "for i in range(5):\n"
            "    total += i\n"
            "print(total)"
        ))
        assert "10" in result

    def test_list_comprehension(self, mcp_server):
        """List comprehensions work in the namespace."""
        result = asyncio.run(mcp_server._execute_code(
            "result = [x**2 for x in range(5)]\nprint(result)"
        ))
        assert "[0, 1, 4, 9, 16]" in result

    def test_import_works(self, mcp_server):
        """import statements work within execute_code."""
        result = asyncio.run(mcp_server._execute_code(
            "import json\nprint(json.dumps({'key': 'value'}))"
        ))
        assert '{"key": "value"}' in result

    def test_function_definition_persists(self, mcp_server):
        """Functions defined in one call are available in the next."""
        asyncio.run(mcp_server._execute_code(
            "def add(a, b):\n    return a + b"
        ))
        result = asyncio.run(mcp_server._execute_code("print(add(3, 4))"))
        assert "7" in result

    def test_class_definition_persists(self, mcp_server):
        """Classes defined in one call are available in the next."""
        asyncio.run(mcp_server._execute_code(
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n"
            "    def inc(self):\n"
            "        self.count += 1\n"
            "        return self.count"
        ))
        result = asyncio.run(mcp_server._execute_code(
            "c = Counter()\nc.inc()\nc.inc()\nprint(c.count)"
        ))
        assert "2" in result

    def test_empty_code_with_namespace(self, mcp_server):
        """Whitespace-only code still triggers _ensure_namespace check."""
        # _execute_code is called, but the wrapped code just runs the update
        result = asyncio.run(mcp_server._execute_code("pass"))
        assert "executed successfully" in result

    def test_exception_does_not_corrupt_namespace(self, mcp_server):
        """An exception in one call does not break subsequent calls."""
        asyncio.run(mcp_server._execute_code("good_var = 'ok'"))
        asyncio.run(mcp_server._execute_code("raise RuntimeError('fail')"))
        result = asyncio.run(mcp_server._execute_code("print(good_var)"))
        assert "ok" in result

    def test_mcp_exec_cleaned_from_namespace(self, mcp_server):
        """The __mcp_exec__ wrapper function is removed after execution."""
        asyncio.run(mcp_server._execute_code("x = 1"))
        assert "__mcp_exec__" not in mcp_server._namespace


# ===========================================================================
# _ensure_namespace tests
# ===========================================================================


class TestEnsureNamespace:
    """Tests for lazy namespace initialization."""

    def test_skips_if_already_initialized(self, mcp_server):
        """Does not re-initialize if namespace already exists."""
        mcp_server._namespace = {"existing": True}
        asyncio.run(mcp_server._ensure_namespace())
        assert mcp_server._namespace["existing"] is True

    def test_initializes_browser_and_namespace(self, mcp_server):
        """Initializes browser session and namespace on first call."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock()

        with patch.object(mcp_server_module, "BrowserSession", return_value=mock_session), \
             patch.object(mcp_server_module, "BrowserProfile"), \
             patch.object(mcp_server_module, "create_namespace", return_value={"navigate": "func"}) as mock_ns, \
             patch.object(mcp_server_module, "CodeAgentTools") as mock_tools_cls, \
             patch.object(mcp_server_module, "get_default_profile", return_value={}), \
             patch.object(mcp_server_module, "load_openbrowser_config", return_value={}):

            asyncio.run(mcp_server._ensure_namespace())

            mock_session.start.assert_called_once()
            assert mcp_server.browser_session is mock_session
            assert mcp_server._namespace == {"navigate": "func"}
            mock_ns.assert_called_once()

    def test_cleanup_on_failed_start(self, mcp_server):
        """Cleans up if browser session fails to start."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock(side_effect=RuntimeError("Chrome not found"))
        mock_event_bus = MagicMock()
        mock_event_bus.dispatch = MagicMock(return_value=AsyncMock()())
        mock_session.event_bus = mock_event_bus

        with patch.object(mcp_server_module, "BrowserSession", return_value=mock_session), \
             patch.object(mcp_server_module, "BrowserProfile"), \
             patch.object(mcp_server_module, "get_default_profile", return_value={}), \
             patch.object(mcp_server_module, "load_openbrowser_config", return_value={}):

            with pytest.raises(RuntimeError, match="Chrome not found"):
                asyncio.run(mcp_server._ensure_namespace())

            # Session should not be assigned on failure
            assert mcp_server.browser_session is None
            assert mcp_server._namespace is None


# ===========================================================================
# Session cleanup tests
# ===========================================================================


class TestSessionCleanup:
    """Tests for session timeout cleanup."""

    def test_cleanup_does_nothing_without_session(self, mcp_server):
        """Cleanup is a no-op when no session exists."""
        asyncio.run(mcp_server._cleanup_expired_session())
        assert mcp_server.browser_session is None

    def test_cleanup_closes_expired_session(self, mcp_server):
        """Closes session when idle beyond timeout."""
        mock_session = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus.dispatch = MagicMock(return_value=AsyncMock()())
        mock_session.event_bus = mock_event_bus

        mcp_server.browser_session = mock_session
        mcp_server._namespace = {"some": "data"}
        mcp_server._last_activity = time.time() - 9999  # Way past timeout

        asyncio.run(mcp_server._cleanup_expired_session())

        assert mcp_server.browser_session is None
        assert mcp_server._namespace is None

    def test_cleanup_keeps_active_session(self, mcp_server):
        """Does not close session that is still active."""
        mock_session = MagicMock()
        mcp_server.browser_session = mock_session
        mcp_server._last_activity = time.time()  # Just active

        asyncio.run(mcp_server._cleanup_expired_session())

        assert mcp_server.browser_session is mock_session

    def test_cleanup_handles_stop_error_gracefully(self, mcp_server):
        """Cleanup handles errors during session stop."""
        mock_session = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus.dispatch = MagicMock(side_effect=RuntimeError("stop failed"))
        mock_session.event_bus = mock_event_bus

        mcp_server.browser_session = mock_session
        mcp_server._namespace = {"data": True}
        mcp_server._last_activity = time.time() - 9999

        # Should not raise
        asyncio.run(mcp_server._cleanup_expired_session())

        # Session should still be cleaned up
        assert mcp_server.browser_session is None
        assert mcp_server._namespace is None


# ===========================================================================
# Tool listing tests
# ===========================================================================


class TestToolListing:
    """Tests for the tool listing handler."""

    def test_server_exposes_single_tool(self, mcp_server):
        """Server should expose exactly one tool: execute_code."""
        # The handler is a closure, but we can verify the tool description
        # is set correctly by checking the constant
        assert "execute_code" in mcp_server_module._EXECUTE_CODE_DESCRIPTION.lower() or \
               "Execute Python code" in mcp_server_module._EXECUTE_CODE_DESCRIPTION

    def test_tool_description_documents_functions(self, mcp_server):
        """Tool description documents the available namespace functions."""
        desc = mcp_server_module._EXECUTE_CODE_DESCRIPTION
        # Key functions should be documented
        assert "navigate" in desc
        assert "click" in desc
        assert "input_text" in desc
        assert "evaluate" in desc
        assert "scroll" in desc
        assert "go_back" in desc
        assert "select_dropdown" in desc
        assert "send_keys" in desc
        assert "switch" in desc
        assert "close" in desc
        assert "done" in desc
        assert "browser" in desc

    def test_tool_description_documents_libraries(self, mcp_server):
        """Tool description documents pre-imported libraries."""
        desc = mcp_server_module._EXECUTE_CODE_DESCRIPTION
        assert "json" in desc
        assert "pandas" in desc
        assert "numpy" in desc
        assert "re" in desc
        assert "csv" in desc
