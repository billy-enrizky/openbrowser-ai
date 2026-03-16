"""Comprehensive tests for src/openbrowser/daemon/server.py to cover remaining gaps.

Missing lines: 71-86, 92-126, 137-162, 173, 180-181, 191-198, 241, 244,
248-256, 261-262, 266-267, 273-277, 290-291, 303, 331-334, 339-340, 344
"""

import asyncio
import json
import logging
import os
import signal
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def daemon_server():
    """Create a DaemonServer instance."""
    from openbrowser.daemon.server import DaemonServer

    return DaemonServer(idle_timeout=5, exec_timeout=2)


class TestPidFunctions:
    """Test PID file management functions."""

    def test_read_pid_no_file(self):
        """Test _read_pid with no PID file."""
        from openbrowser.daemon.server import _read_pid

        with patch("openbrowser.daemon.server.get_pid_path") as mock_path:
            mock_path.return_value = MagicMock(exists=MagicMock(return_value=False))
            assert _read_pid() is None

    def test_read_pid_valid(self):
        """Test _read_pid with valid PID."""
        from openbrowser.daemon.server import _read_pid

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False) as f:
            f.write(str(os.getpid()))
            f.flush()

            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.read_text.return_value = str(os.getpid())

            with patch("openbrowser.daemon.server.get_pid_path", return_value=mock_path):
                result = _read_pid()
                assert result == os.getpid()

    def test_read_pid_stale(self):
        """Test _read_pid with stale PID."""
        from openbrowser.daemon.server import _read_pid

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "99999999"

        with patch("openbrowser.daemon.server.get_pid_path", return_value=mock_path):
            with patch("os.kill", side_effect=OSError("No such process")):
                result = _read_pid()
                assert result is None

    def test_write_pid(self):
        """Test _write_pid."""
        from openbrowser.daemon.server import _write_pid

        mock_path = MagicMock()
        mock_path.parent = MagicMock()

        with patch("openbrowser.daemon.server.get_pid_path", return_value=mock_path):
            _write_pid()
            mock_path.write_text.assert_called_once()
            mock_path.chmod.assert_called_once_with(0o600)

    def test_cleanup_pid(self):
        """Test _cleanup_pid."""
        from openbrowser.daemon.server import _cleanup_pid

        mock_pid = MagicMock()
        mock_sock = MagicMock()

        with patch("openbrowser.daemon.server.get_pid_path", return_value=mock_pid):
            with patch("openbrowser.daemon.server.get_socket_path", return_value=mock_sock):
                _cleanup_pid()
                mock_pid.unlink.assert_called_once_with(missing_ok=True)
                mock_sock.unlink.assert_called_once_with(missing_ok=True)


class TestDaemonServerBuildProfile:
    """Test _build_browser_profile."""

    def test_build_browser_profile(self, daemon_server):
        """Lines 71-86: build browser profile from config."""
        mock_profile = MagicMock()
        mock_profile_class = MagicMock(return_value=mock_profile)

        with patch("openbrowser.browser.BrowserProfile", mock_profile_class):
            with patch(
                "openbrowser.config.load_openbrowser_config", return_value={}
            ):
                with patch(
                    "openbrowser.config.get_default_profile", return_value={}
                ):
                    result = daemon_server._build_browser_profile()
                    assert result is mock_profile
                    mock_profile_class.assert_called_once()


class TestDaemonServerIsConnectionError:
    """Test _is_connection_error."""

    def test_connection_error_keywords(self, daemon_server):
        """Test connection error detection."""
        assert daemon_server._is_connection_error("ConnectionClosedError") is True
        assert daemon_server._is_connection_error("no close frame received") is True
        assert daemon_server._is_connection_error("websocket error") is True
        assert daemon_server._is_connection_error("connection closed unexpectedly") is True
        assert daemon_server._is_connection_error("regular error") is False


class TestDaemonServerHandleRequest:
    """Test _handle_request."""

    @pytest.mark.asyncio
    async def test_execute_empty_code(self, daemon_server):
        """Line 173: execute with empty code."""
        result = await daemon_server._handle_request(
            {"action": "execute", "code": "", "id": 1}
        )
        assert result["success"] is False
        assert "No code" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_timeout(self, daemon_server):
        """Lines 180-181: execute timeout."""
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(side_effect=asyncio.TimeoutError())
        daemon_server._executor = mock_executor

        result = await daemon_server._handle_request(
            {"action": "execute", "code": "x = 1", "id": 1}
        )
        assert result["success"] is False
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_success(self, daemon_server):
        """Test successful execution."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "42"
        mock_result.error = None

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_result)
        daemon_server._executor = mock_executor

        result = await daemon_server._handle_request(
            {"action": "execute", "code": "x = 1", "id": 1}
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_connection_error_recovery(self, daemon_server):
        """Lines 191-198: execute with connection error triggers recovery."""
        mock_result_fail = MagicMock()
        mock_result_fail.success = False
        mock_result_fail.output = "ConnectionClosedError"
        mock_result_fail.error = "ConnectionClosedError"

        mock_result_ok = MagicMock()
        mock_result_ok.success = True
        mock_result_ok.output = "ok"
        mock_result_ok.error = None

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(side_effect=[mock_result_fail, mock_result_ok])
        daemon_server._executor = mock_executor

        with patch.object(
            daemon_server, "_recover_browser_session", new_callable=AsyncMock
        ):
            result = await daemon_server._handle_request(
                {"action": "execute", "code": "x = 1", "id": 1}
            )
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_status(self, daemon_server):
        """Test status action."""
        result = await daemon_server._handle_request({"action": "status", "id": 1})
        assert result["success"] is True
        assert "pid" in result["output"]

    @pytest.mark.asyncio
    async def test_stop(self, daemon_server):
        """Test stop action."""
        result = await daemon_server._handle_request({"action": "stop", "id": 1})
        assert result["success"] is True
        assert daemon_server._running is False

    @pytest.mark.asyncio
    async def test_reset(self, daemon_server):
        """Test reset action."""
        daemon_server._session = MagicMock()
        daemon_server._session.kill = AsyncMock()
        daemon_server._executor = MagicMock()

        result = await daemon_server._handle_request({"action": "reset", "id": 1})
        assert result["success"] is True
        assert daemon_server._executor is None
        assert daemon_server._session is None

    @pytest.mark.asyncio
    async def test_reset_with_kill_error(self, daemon_server):
        """Test reset with session kill error."""
        daemon_server._session = MagicMock()
        daemon_server._session.kill = AsyncMock(side_effect=Exception("kill error"))
        daemon_server._executor = MagicMock()

        result = await daemon_server._handle_request({"action": "reset", "id": 1})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_unknown_action(self, daemon_server):
        """Test unknown action."""
        result = await daemon_server._handle_request({"action": "unknown", "id": 1})
        assert result["success"] is False
        assert "Unknown action" in result["error"]


class TestDaemonServerHandleClient:
    """Test _handle_client."""

    @pytest.mark.asyncio
    async def test_handle_client_success(self, daemon_server):
        """Lines 241-248: handle client successfully."""
        reader = AsyncMock()
        writer = AsyncMock()

        request = {"action": "status", "id": 1}
        reader.readline = AsyncMock(return_value=json.dumps(request).encode() + b"\n")
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon_server._handle_client(reader, writer)
        writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_client_empty_data(self, daemon_server):
        """Line 241: handle client with empty data."""
        reader = AsyncMock()
        writer = AsyncMock()

        reader.readline = AsyncMock(return_value=b"")
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon_server._handle_client(reader, writer)

    @pytest.mark.asyncio
    async def test_handle_client_timeout(self, daemon_server):
        """Lines 248-249: handle client timeout."""
        reader = AsyncMock()
        writer = AsyncMock()

        reader.readline = AsyncMock(side_effect=asyncio.TimeoutError())
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon_server._handle_client(reader, writer)

    @pytest.mark.asyncio
    async def test_handle_client_invalid_json(self, daemon_server):
        """Lines 248-256: handle client with invalid JSON."""
        reader = AsyncMock()
        writer = AsyncMock()

        reader.readline = AsyncMock(return_value=b"not json\n")
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon_server._handle_client(reader, writer)
        # Should have written an error response
        writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_client_non_dict_json(self, daemon_server):
        """Line 244: handle client with non-dict JSON."""
        reader = AsyncMock()
        writer = AsyncMock()

        reader.readline = AsyncMock(return_value=b'"just a string"\n')
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon_server._handle_client(reader, writer)
        writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_client_write_error(self, daemon_server):
        """Lines 261-262: handle client with write error in error handler."""
        reader = AsyncMock()
        writer = AsyncMock()

        reader.readline = AsyncMock(return_value=b"not json\n")
        writer.drain = AsyncMock(side_effect=Exception("write error"))
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon_server._handle_client(reader, writer)


class TestDaemonServerSignalShutdown:
    """Test _signal_shutdown."""

    def test_signal_shutdown(self, daemon_server):
        """Lines 266-267: signal handler marks daemon for shutdown."""
        daemon_server._running = True
        daemon_server._signal_shutdown()
        assert daemon_server._running is False
        assert daemon_server._stop_event.is_set()


class TestDaemonServerIdleCheck:
    """Test _idle_check_loop."""

    @pytest.mark.asyncio
    async def test_idle_timeout(self, daemon_server):
        """Lines 273-277: idle check shuts down after timeout."""
        daemon_server._running = True
        daemon_server._idle_timeout = 0  # Immediate timeout
        daemon_server._last_activity = time.time() - 100

        # Run idle check - it should trigger shutdown
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await daemon_server._idle_check_loop()
            assert daemon_server._running is False


class TestDaemonServerRun:
    """Test run method."""

    @pytest.mark.asyncio
    async def test_run_another_daemon(self, daemon_server):
        """Lines 290-291: another daemon already running."""
        mock_sock = MagicMock()
        mock_sock.parent = MagicMock()
        mock_sock.unlink = MagicMock()

        with patch("openbrowser.daemon.server.get_socket_path", return_value=mock_sock):
            with patch("openbrowser.daemon.server._read_pid", return_value=99999):
                with patch("os.getpid", return_value=1):
                    await daemon_server.run()
                    # Should return without starting

    @pytest.mark.asyncio
    async def test_run_unix_server(self, daemon_server):
        """Lines 303-334: run Unix server."""
        mock_sock = MagicMock()
        mock_sock.parent = MagicMock()
        mock_sock.unlink = MagicMock()

        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=False)

        daemon_server._session = None

        async def stop_after_start():
            await asyncio.sleep(0.01)
            daemon_server._stop_event.set()

        with patch("openbrowser.daemon.server.get_socket_path", return_value=mock_sock):
            with patch("openbrowser.daemon.server._read_pid", return_value=None):
                with patch("openbrowser.daemon.server._write_pid"):
                    with patch("openbrowser.daemon.server._cleanup_pid"):
                        with patch("openbrowser.daemon.server.IS_WINDOWS", False):
                            with patch(
                                "asyncio.start_unix_server",
                                new_callable=AsyncMock,
                                return_value=mock_server,
                            ):
                                with patch("os.chmod"):
                                    asyncio.create_task(stop_after_start())
                                    await daemon_server.run()

    @pytest.mark.asyncio
    async def test_run_cleanup_session(self, daemon_server):
        """Lines 331-334: run cleans up session on exit."""
        mock_sock = MagicMock()
        mock_sock.parent = MagicMock()
        mock_sock.unlink = MagicMock()

        daemon_server._session = MagicMock()
        daemon_server._session.kill = AsyncMock()

        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=False)

        async def stop_quickly():
            await asyncio.sleep(0.01)
            daemon_server._stop_event.set()

        with patch("openbrowser.daemon.server.get_socket_path", return_value=mock_sock):
            with patch("openbrowser.daemon.server._read_pid", return_value=None):
                with patch("openbrowser.daemon.server._write_pid"):
                    with patch("openbrowser.daemon.server._cleanup_pid"):
                        with patch("openbrowser.daemon.server.IS_WINDOWS", False):
                            with patch(
                                "asyncio.start_unix_server",
                                new_callable=AsyncMock,
                                return_value=mock_server,
                            ):
                                with patch("os.chmod"):
                                    asyncio.create_task(stop_quickly())
                                    await daemon_server.run()
                                    daemon_server._session.kill.assert_called_once()


class TestDaemonMain:
    """Test module-level _main function."""

    @pytest.mark.asyncio
    async def test_main(self):
        """Lines 339-340: _main function."""
        from openbrowser.daemon.server import _main

        with patch("openbrowser.daemon.server.DaemonServer") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock()
            mock_cls.return_value = mock_instance
            await _main()
            mock_instance.run.assert_called_once()
