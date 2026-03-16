"""Tests for daemon server helpers and request handling."""

import json
import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.daemon.server import (
    DaemonServer,
    _cleanup_pid,
    _read_pid,
    _write_pid,
)

logger = logging.getLogger(__name__)


class TestDaemonServerHelpers:
    """Tests for daemon/server.py helper functions."""

    def test_read_pid_missing_file(self, tmp_path):
        """_read_pid returns None when PID file does not exist."""
        with patch("openbrowser.daemon.server.get_pid_path", return_value=tmp_path / "nope.pid"):
            assert _read_pid() is None

    def test_read_pid_stale_process(self, tmp_path):
        """_read_pid returns None and cleans up when process is dead."""
        pid_file = tmp_path / "daemon.pid"
        # Use a PID that almost certainly does not exist
        pid_file.write_text("999999999")
        with patch("openbrowser.daemon.server.get_pid_path", return_value=pid_file):
            result = _read_pid()
            assert result is None
            # File should be cleaned up
            assert not pid_file.exists()

    def test_read_pid_invalid_content(self, tmp_path):
        """_read_pid handles non-integer PID file gracefully."""
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("not_a_number")
        with patch("openbrowser.daemon.server.get_pid_path", return_value=pid_file):
            result = _read_pid()
            assert result is None

    def test_read_pid_alive_process(self, tmp_path):
        """_read_pid returns PID when process is alive."""
        pid_file = tmp_path / "daemon.pid"
        current_pid = os.getpid()
        pid_file.write_text(str(current_pid))
        with patch("openbrowser.daemon.server.get_pid_path", return_value=pid_file):
            result = _read_pid()
            assert result == current_pid

    def test_write_pid(self, tmp_path):
        """_write_pid creates PID file with correct content and permissions."""
        pid_file = tmp_path / "subdir" / "daemon.pid"
        with patch("openbrowser.daemon.server.get_pid_path", return_value=pid_file):
            _write_pid()
            assert pid_file.exists()
            assert pid_file.read_text() == str(os.getpid())
            # Check permission bits (owner read/write only)
            mode = pid_file.stat().st_mode & 0o777
            assert mode == 0o600

    def test_cleanup_pid(self, tmp_path):
        """_cleanup_pid removes PID and socket files."""
        pid_file = tmp_path / "daemon.pid"
        sock_file = tmp_path / "daemon.sock"
        pid_file.write_text("12345")
        sock_file.write_text("socket")

        with patch("openbrowser.daemon.server.get_pid_path", return_value=pid_file), \
             patch("openbrowser.daemon.server.get_socket_path", return_value=sock_file):
            _cleanup_pid()
            assert not pid_file.exists()
            assert not sock_file.exists()

    def test_cleanup_pid_missing_files(self, tmp_path):
        """_cleanup_pid does not raise when files already gone."""
        with patch("openbrowser.daemon.server.get_pid_path", return_value=tmp_path / "nope.pid"), \
             patch("openbrowser.daemon.server.get_socket_path", return_value=tmp_path / "nope.sock"):
            _cleanup_pid()  # Should not raise


class TestDaemonServerHandleRequest:
    """Tests for DaemonServer._handle_request for various actions."""

    @pytest.fixture
    def daemon_server(self):
        server = DaemonServer(idle_timeout=60, exec_timeout=30)
        return server

    @pytest.mark.asyncio
    async def test_handle_status(self, daemon_server):
        """Status action returns success with PID and init state."""
        result = await daemon_server._handle_request({"action": "status", "id": 5})
        assert result["success"] is True
        assert result["id"] == 5
        output = json.loads(result["output"])
        assert "pid" in output
        assert output["initialized"] is False

    @pytest.mark.asyncio
    async def test_handle_stop(self, daemon_server):
        """Stop action sets _running to False and signals stop event."""
        daemon_server._running = True
        result = await daemon_server._handle_request({"action": "stop", "id": 2})
        assert result["success"] is True
        assert "stopping" in result["output"].lower()
        assert daemon_server._running is False
        assert daemon_server._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_handle_reset_no_session(self, daemon_server):
        """Reset action clears executor and session when no active session."""
        daemon_server._executor = MagicMock()
        daemon_server._session = None
        result = await daemon_server._handle_request({"action": "reset", "id": 3})
        assert result["success"] is True
        assert "reset" in result["output"].lower()
        assert daemon_server._executor is None
        assert daemon_server._session is None

    @pytest.mark.asyncio
    async def test_handle_reset_with_session(self, daemon_server):
        """Reset action kills the session, then clears state."""
        mock_session = AsyncMock()
        daemon_server._session = mock_session
        daemon_server._executor = MagicMock()
        result = await daemon_server._handle_request({"action": "reset", "id": 4})
        assert result["success"] is True
        mock_session.kill.assert_awaited_once()
        assert daemon_server._executor is None
        assert daemon_server._session is None

    @pytest.mark.asyncio
    async def test_handle_reset_session_kill_error(self, daemon_server):
        """Reset still clears state even if session.kill() raises."""
        mock_session = AsyncMock()
        mock_session.kill.side_effect = RuntimeError("boom")
        daemon_server._session = mock_session
        daemon_server._executor = MagicMock()
        result = await daemon_server._handle_request({"action": "reset", "id": 6})
        assert result["success"] is True
        assert daemon_server._executor is None

    @pytest.mark.asyncio
    async def test_handle_unknown_action(self, daemon_server):
        """Unknown action returns error response."""
        result = await daemon_server._handle_request({"action": "foobar", "id": 99})
        assert result["success"] is False
        assert "Unknown action" in result["error"]
        assert "foobar" in result["error"]
        assert result["id"] == 99

    @pytest.mark.asyncio
    async def test_handle_empty_action(self, daemon_server):
        """Missing/empty action returns unknown action error."""
        result = await daemon_server._handle_request({"id": 7})
        assert result["success"] is False
        assert "Unknown action" in result["error"]

    def test_is_connection_error_true(self, daemon_server):
        """_is_connection_error detects CDP error keywords."""
        assert daemon_server._is_connection_error("ConnectionClosedError: foo") is True
        assert daemon_server._is_connection_error("no close frame received") is True
        assert daemon_server._is_connection_error("WebSocket connection failed") is True
        assert daemon_server._is_connection_error("connection closed unexpectedly") is True

    def test_is_connection_error_false(self, daemon_server):
        """_is_connection_error returns False for normal errors."""
        assert daemon_server._is_connection_error("ValueError: bad input") is False
        assert daemon_server._is_connection_error("timeout occurred") is False
        assert daemon_server._is_connection_error("") is False
