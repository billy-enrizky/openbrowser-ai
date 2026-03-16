"""Tests for MCP server helpers and error detection."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from openbrowser.code_use.executor import ExecutionResult

logger = logging.getLogger(__name__)


class TestMCPServerHelpers:
    """Tests for mcp/server.py helper functions."""

    def test_get_parent_process_cmdline_with_psutil(self):
        """get_parent_process_cmdline returns a string when psutil available."""
        from openbrowser.mcp.server import get_parent_process_cmdline

        mock_parent = MagicMock()
        mock_parent.cmdline.return_value = ["/usr/bin/python", "test.py"]
        mock_parent.parent.return_value = None  # stop recursion

        mock_process = MagicMock()
        mock_process.parent.return_value = mock_parent

        with patch("openbrowser.mcp.server.PSUTIL_AVAILABLE", True), \
             patch("openbrowser.mcp.server.psutil.Process", return_value=mock_process):
            result = get_parent_process_cmdline()
            assert isinstance(result, str)
            assert "python" in result

    def test_get_parent_process_cmdline_no_psutil(self):
        """get_parent_process_cmdline returns None when psutil unavailable."""
        from openbrowser.mcp.server import get_parent_process_cmdline

        with patch("openbrowser.mcp.server.PSUTIL_AVAILABLE", False):
            result = get_parent_process_cmdline()
            assert result is None

    def test_get_parent_process_cmdline_exception(self):
        """get_parent_process_cmdline returns None on exception."""
        from openbrowser.mcp.server import get_parent_process_cmdline

        with patch("openbrowser.mcp.server.PSUTIL_AVAILABLE", True), \
             patch("openbrowser.mcp.server.psutil.Process", side_effect=Exception("fail")):
            result = get_parent_process_cmdline()
            assert result is None


class TestMCPServerConnectionError:
    """Tests for OpenBrowserServer._is_connection_error."""

    @pytest.fixture
    def server_instance(self, mcp_server):
        """Use the shared mcp_server fixture from conftest.py."""
        return mcp_server

    def test_is_connection_error_with_string(self, server_instance):
        """_is_connection_error detects CDP keywords in string."""
        assert server_instance._is_connection_error("ConnectionClosedError: x") is True
        assert server_instance._is_connection_error("no close frame") is True
        assert server_instance._is_connection_error("WebSocket failed") is True
        assert server_instance._is_connection_error("connection closed") is True

    def test_is_connection_error_with_exception_object(self, server_instance):
        """_is_connection_error works with exception objects."""
        err = ConnectionError("WebSocket connection lost")
        assert server_instance._is_connection_error(err) is True

    def test_is_connection_error_normal_error(self, server_instance):
        """_is_connection_error returns False for non-CDP errors."""
        assert server_instance._is_connection_error("ValueError: bad") is False
        assert server_instance._is_connection_error("") is False
        err = ValueError("something")
        assert server_instance._is_connection_error(err) is False


class TestExecutionResult:
    """Tests for the ExecutionResult dataclass."""

    def test_execution_result_defaults(self):
        """ExecutionResult default error is None."""
        result = ExecutionResult(success=True, output="ok")
        assert result.error is None

    def test_execution_result_with_error(self):
        """ExecutionResult stores error."""
        result = ExecutionResult(success=False, output="Error: boom", error="boom")
        assert result.error == "boom"
