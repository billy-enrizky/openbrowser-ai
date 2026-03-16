"""Tests for namespace helpers and daemon initialization."""

import logging
import os
import platform
from unittest.mock import patch

from openbrowser.code_use.namespace import _strip_js_comments
from openbrowser.daemon import IS_WINDOWS, get_pid_path, get_socket_path

logger = logging.getLogger(__name__)


class TestStripJsComments:
    """Tests for _strip_js_comments helper function."""

    def test_strip_single_line_comment(self):
        """Removes lines that start with //."""
        code = "// This is a comment\nvar x = 1;"
        result = _strip_js_comments(code)
        assert "This is a comment" not in result
        assert "var x = 1;" in result

    def test_strip_multi_line_comment(self):
        """Removes /* ... */ block comments."""
        code = "var x = 1; /* this is\na comment */ var y = 2;"
        result = _strip_js_comments(code)
        assert "this is" not in result
        assert "var x = 1;" in result
        assert "var y = 2;" in result

    def test_preserves_url_with_double_slash(self):
        """Does not strip // inside URLs or string values."""
        code = 'var url = "https://example.com";'
        result = _strip_js_comments(code)
        assert "https://example.com" in result

    def test_preserves_inline_comment_not_at_start(self):
        """Does not strip // that appears mid-line (not at start)."""
        code = 'var path = "a//b";'
        result = _strip_js_comments(code)
        assert "a//b" in result

    def test_strip_indented_single_line_comment(self):
        """Strips // comments that are indented with whitespace."""
        code = "  // indented comment\nvar x = 1;"
        result = _strip_js_comments(code)
        assert "indented comment" not in result
        assert "var x = 1;" in result

    def test_no_comments(self):
        """No change when there are no comments."""
        code = "var x = 1;\nvar y = 2;"
        result = _strip_js_comments(code)
        assert "var x = 1;" in result
        assert "var y = 2;" in result

    def test_nested_multiline_in_single_line(self):
        """Handles /* */ within a single line."""
        code = "var x = /* inline */ 5;"
        result = _strip_js_comments(code)
        assert "inline" not in result
        assert "var x =" in result
        assert "5;" in result

    def test_mixed_comments(self):
        """Handles mix of single-line and multi-line comments."""
        code = """// top comment
var x = 1;
/* block
   comment */
   // another line comment
var y = 2;"""
        result = _strip_js_comments(code)
        assert "top comment" not in result
        assert "block" not in result
        assert "another line comment" not in result
        assert "var x = 1;" in result
        assert "var y = 2;" in result

    def test_empty_string(self):
        """Empty input returns empty output."""
        assert _strip_js_comments("") == ""


class TestDaemonInit:
    """Tests for openbrowser.daemon.__init__ helpers."""

    def test_get_socket_path_default(self):
        """get_socket_path returns default when env var unset."""
        from openbrowser.daemon import SOCKET_PATH

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENBROWSER_SOCKET", None)
            result = get_socket_path()
            assert result == SOCKET_PATH

    def test_get_socket_path_custom_env(self, tmp_path):
        """get_socket_path respects OPENBROWSER_SOCKET env var."""
        custom = str(tmp_path / "custom.sock")
        with patch.dict(os.environ, {"OPENBROWSER_SOCKET": custom}):
            result = get_socket_path()
            assert str(result) == custom

    def test_get_pid_path_default(self):
        """get_pid_path returns default PID_PATH when no env var."""
        from openbrowser.daemon import PID_PATH

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENBROWSER_SOCKET", None)
            result = get_pid_path()
            assert result == PID_PATH

    def test_get_pid_path_custom_env(self, tmp_path):
        """get_pid_path derives PID path from custom socket path."""
        custom_socket = str(tmp_path / "my_daemon.sock")
        with patch.dict(os.environ, {"OPENBROWSER_SOCKET": custom_socket}):
            result = get_pid_path()
            assert result == tmp_path / "my_daemon.pid"

    def test_is_windows_constant(self):
        """IS_WINDOWS is a boolean reflecting current platform."""
        assert IS_WINDOWS == (platform.system() == "Windows")
