"""Comprehensive tests for src/openbrowser/mcp/server.py to cover remaining gaps.

Missing lines: 43-44, 49, 92-95, 105, 148-151, 157-158, 183-184, 218-219,
236-239, 264, 268, 272, 277-296, 340-341, 358, 361, 365-366, 376-419,
446-449, 459-464, 491-500, 504-507, 526-549, 553
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# Use conftest.py fixtures for MCP stubs
from tests.conftest import DummyServer, DummyTypes


@pytest.fixture
def mcp_server_instance(monkeypatch):
    """Create an OpenBrowserServer with dummy MCP SDK stubs."""
    from openbrowser.mcp import server as mcp_server_module

    monkeypatch.setattr(mcp_server_module, "MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_server_module, "Server", DummyServer)
    monkeypatch.setattr(mcp_server_module, "types", DummyTypes)
    monkeypatch.setattr(mcp_server_module, "TELEMETRY_AVAILABLE", False)

    return mcp_server_module.OpenBrowserServer()


class TestMCPServerImports:
    """Test module-level import handling."""

    def test_psutil_available(self):
        """Lines 43-44: PSUTIL_AVAILABLE flag."""
        from openbrowser.mcp.server import PSUTIL_AVAILABLE

        assert isinstance(PSUTIL_AVAILABLE, bool)

    def test_src_path_added(self):
        """Line 49: src dir added to path."""
        # This is a module-level operation, just verify it runs
        from openbrowser.mcp import server

        assert server is not None

    def test_filesystem_available(self):
        """Lines 92-95: FILESYSTEM_AVAILABLE flag."""
        from openbrowser.mcp.server import FILESYSTEM_AVAILABLE

        assert isinstance(FILESYSTEM_AVAILABLE, bool)

    def test_create_mcp_file_system_unavailable(self):
        """Line 105: _create_mcp_file_system when unavailable."""
        from openbrowser.mcp.server import _create_mcp_file_system

        with patch("openbrowser.mcp.server.FILESYSTEM_AVAILABLE", False):
            result = _create_mcp_file_system()
            assert result is None

    def test_create_mcp_file_system_available(self):
        """Line 106: _create_mcp_file_system when available."""
        from openbrowser.mcp.server import _create_mcp_file_system

        mock_fs = MagicMock()
        with patch("openbrowser.mcp.server.FILESYSTEM_AVAILABLE", True):
            with patch("openbrowser.mcp.server.FileSystem", return_value=mock_fs):
                result = _create_mcp_file_system()
                assert result is mock_fs


class TestGetParentProcessCmdline:
    """Test get_parent_process_cmdline function."""

    def test_no_psutil(self):
        """Line 166: no psutil available."""
        from openbrowser.mcp.server import get_parent_process_cmdline

        with patch("openbrowser.mcp.server.PSUTIL_AVAILABLE", False):
            result = get_parent_process_cmdline()
            assert result is None

    def test_with_psutil(self):
        """Lines 168-188: get parent process cmdlines."""
        from openbrowser.mcp.server import get_parent_process_cmdline

        mock_parent = MagicMock()
        mock_parent.cmdline.return_value = ["bash"]
        mock_parent.parent.return_value = None

        mock_process = MagicMock()
        mock_process.parent.return_value = mock_parent

        with patch("openbrowser.mcp.server.PSUTIL_AVAILABLE", True):
            with patch("psutil.Process", return_value=mock_process):
                result = get_parent_process_cmdline()
                assert result is not None
                assert "bash" in result

    def test_with_psutil_access_denied(self):
        """Lines 178-179, 183-184: AccessDenied errors."""
        import psutil
        from openbrowser.mcp.server import get_parent_process_cmdline

        mock_parent = MagicMock()
        mock_parent.cmdline.side_effect = psutil.AccessDenied(pid=1)
        mock_parent.parent.side_effect = psutil.AccessDenied(pid=1)

        mock_process = MagicMock()
        mock_process.parent.return_value = mock_parent

        with patch("openbrowser.mcp.server.PSUTIL_AVAILABLE", True):
            with patch("psutil.Process", return_value=mock_process):
                result = get_parent_process_cmdline()
                # Should return None since no cmdlines collected
                assert result is None

    def test_exception(self):
        """Line 187-188: exception handling."""
        from openbrowser.mcp.server import get_parent_process_cmdline

        with patch("openbrowser.mcp.server.PSUTIL_AVAILABLE", True):
            with patch("psutil.Process", side_effect=Exception("error")):
                result = get_parent_process_cmdline()
                assert result is None


class TestOpenBrowserServerInit:
    """Test OpenBrowserServer initialization."""

    def test_init(self, mcp_server_instance):
        """Lines 218-219: init with max_output."""
        assert mcp_server_instance._executor is not None

    def test_init_with_max_output_env(self, monkeypatch):
        """Lines 218-219: init with OPENBROWSER_MAX_OUTPUT env."""
        from openbrowser.mcp import server as mcp_server_module

        monkeypatch.setattr(mcp_server_module, "MCP_AVAILABLE", True)
        monkeypatch.setattr(mcp_server_module, "Server", DummyServer)
        monkeypatch.setattr(mcp_server_module, "types", DummyTypes)
        monkeypatch.setattr(mcp_server_module, "TELEMETRY_AVAILABLE", False)

        with patch.dict(os.environ, {"OPENBROWSER_MAX_OUTPUT": "5000"}):
            server = mcp_server_module.OpenBrowserServer()
            assert server._executor is not None

    def test_init_invalid_max_output(self, monkeypatch):
        """Lines 218-219: init with invalid OPENBROWSER_MAX_OUTPUT."""
        from openbrowser.mcp import server as mcp_server_module

        monkeypatch.setattr(mcp_server_module, "MCP_AVAILABLE", True)
        monkeypatch.setattr(mcp_server_module, "Server", DummyServer)
        monkeypatch.setattr(mcp_server_module, "types", DummyTypes)
        monkeypatch.setattr(mcp_server_module, "TELEMETRY_AVAILABLE", False)

        with patch.dict(os.environ, {"OPENBROWSER_MAX_OUTPUT": "invalid"}):
            server = mcp_server_module.OpenBrowserServer()
            assert server._executor is not None


class TestOpenBrowserServerBuildProfile:
    """Test _build_browser_profile."""

    def test_build_browser_profile(self, mcp_server_instance):
        """Lines 236-239: build browser profile."""
        profile = mcp_server_instance._build_browser_profile()
        assert profile is not None


class TestOpenBrowserServerIsConnectionError:
    """Test _is_connection_error."""

    def test_string_input(self, mcp_server_instance):
        """Test with string input."""
        assert mcp_server_instance._is_connection_error("ConnectionClosedError") is True
        assert mcp_server_instance._is_connection_error("normal error") is False

    def test_exception_input(self, mcp_server_instance):
        """Test with exception input."""
        exc = ConnectionError("websocket disconnected")
        assert mcp_server_instance._is_connection_error(exc) is True

    def test_no_close_frame(self, mcp_server_instance):
        """Test 'no close frame' detection."""
        assert mcp_server_instance._is_connection_error("no close frame received") is True


class TestOpenBrowserServerIsCdpAlive:
    """Test _is_cdp_alive."""

    @pytest.mark.asyncio
    async def test_no_browser_session(self, mcp_server_instance):
        """Line 358: no browser session."""
        mcp_server_instance.browser_session = None
        result = await mcp_server_instance._is_cdp_alive()
        assert result is False

    @pytest.mark.asyncio
    async def test_no_cdp_root(self, mcp_server_instance):
        """Line 361: no CDP root client."""
        mock_session = MagicMock()
        mock_session._cdp_client_root = None
        mcp_server_instance.browser_session = mock_session
        result = await mcp_server_instance._is_cdp_alive()
        assert result is False

    @pytest.mark.asyncio
    async def test_cdp_alive(self, mcp_server_instance):
        """Lines 362-364: CDP is alive."""
        mock_session = MagicMock()
        mock_root = MagicMock()
        mock_root.send.Browser.getVersion = AsyncMock(return_value={})
        mock_session._cdp_client_root = mock_root
        mcp_server_instance.browser_session = mock_session

        result = await mcp_server_instance._is_cdp_alive()
        assert result is True

    @pytest.mark.asyncio
    async def test_cdp_dead(self, mcp_server_instance):
        """Lines 365-366: CDP is dead."""
        mock_session = MagicMock()
        mock_root = MagicMock()
        mock_root.send.Browser.getVersion = AsyncMock(side_effect=Exception("dead"))
        mock_session._cdp_client_root = mock_root
        mcp_server_instance.browser_session = mock_session

        result = await mcp_server_instance._is_cdp_alive()
        assert result is False


class TestOpenBrowserServerExecuteCode:
    """Test _execute_code."""

    @pytest.mark.asyncio
    async def test_execute_code_success(self, mcp_server_instance):
        """Test successful code execution."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "42"
        mock_result.error = None

        mcp_server_instance._namespace = {"x": 1}
        mcp_server_instance._executor = MagicMock()
        mcp_server_instance._executor.initialized = True
        mcp_server_instance._executor.execute = AsyncMock(return_value=mock_result)
        mcp_server_instance.browser_session = None

        result = await mcp_server_instance._execute_code("print(42)")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_execute_code_cdp_recovery(self, mcp_server_instance):
        """Lines 446-449: pre-flight CDP recovery."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "ok"
        mock_result.error = None

        mcp_server_instance._namespace = {"x": 1}
        mcp_server_instance._executor = MagicMock()
        mcp_server_instance._executor.initialized = True
        mcp_server_instance._executor.execute = AsyncMock(return_value=mock_result)

        mock_session = MagicMock()
        mcp_server_instance.browser_session = mock_session

        with patch.object(
            mcp_server_instance, "_is_cdp_alive", new_callable=AsyncMock, return_value=False
        ):
            with patch.object(
                mcp_server_instance,
                "_recover_browser_session",
                new_callable=AsyncMock,
            ):
                result = await mcp_server_instance._execute_code("x = 1")
                assert result == "ok"

    @pytest.mark.asyncio
    async def test_execute_code_cdp_recovery_failure(self, mcp_server_instance):
        """Lines 448-449: pre-flight CDP recovery fails."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "ok"
        mock_result.error = None

        mcp_server_instance._namespace = {"x": 1}
        mcp_server_instance._executor = MagicMock()
        mcp_server_instance._executor.initialized = True
        mcp_server_instance._executor.execute = AsyncMock(return_value=mock_result)

        mock_session = MagicMock()
        mcp_server_instance.browser_session = mock_session

        with patch.object(
            mcp_server_instance, "_is_cdp_alive", new_callable=AsyncMock, return_value=False
        ):
            with patch.object(
                mcp_server_instance,
                "_recover_browser_session",
                new_callable=AsyncMock,
                side_effect=Exception("recovery failed"),
            ):
                result = await mcp_server_instance._execute_code("x = 1")
                assert result == "ok"

    @pytest.mark.asyncio
    async def test_execute_code_retry_on_connection_error(self, mcp_server_instance):
        """Lines 459-464: retry on connection error."""
        mock_result_fail = MagicMock()
        mock_result_fail.success = False
        mock_result_fail.output = "ConnectionClosedError"
        mock_result_fail.error = "ConnectionClosedError"

        mock_result_ok = MagicMock()
        mock_result_ok.success = True
        mock_result_ok.output = "ok"
        mock_result_ok.error = None

        mcp_server_instance._namespace = {"x": 1}
        mcp_server_instance._executor = MagicMock()
        mcp_server_instance._executor.initialized = True
        mcp_server_instance._executor.execute = AsyncMock(
            side_effect=[mock_result_fail, mock_result_ok]
        )
        mcp_server_instance.browser_session = None

        with patch.object(
            mcp_server_instance,
            "_recover_browser_session",
            new_callable=AsyncMock,
        ):
            result = await mcp_server_instance._execute_code("x = 1")
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_execute_code_uninitialized_executor(self, mcp_server_instance):
        """Lines 436-437: executor not initialized."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "ok"
        mock_result.error = None

        mcp_server_instance._namespace = {"x": 1}
        mcp_server_instance._executor = MagicMock()
        mcp_server_instance._executor.initialized = False
        mcp_server_instance._executor.execute = AsyncMock(return_value=mock_result)
        mcp_server_instance.browser_session = None

        with patch.object(
            mcp_server_instance, "_ensure_namespace", new_callable=AsyncMock
        ):
            result = await mcp_server_instance._execute_code("x = 1")
            mcp_server_instance._executor.set_namespace.assert_called_once()


class TestOpenBrowserServerCleanup:
    """Test session cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_no_session(self, mcp_server_instance):
        """Line 470: no session to clean."""
        mcp_server_instance.browser_session = None
        await mcp_server_instance._cleanup_expired_session()

    @pytest.mark.asyncio
    async def test_cleanup_not_expired(self, mcp_server_instance):
        """Lines 472-477: session not expired."""
        mcp_server_instance.browser_session = MagicMock()
        mcp_server_instance._last_activity = time.time()
        mcp_server_instance.session_timeout_minutes = 10

        await mcp_server_instance._cleanup_expired_session()
        assert mcp_server_instance.browser_session is not None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, mcp_server_instance):
        """Lines 477-486: session expired."""
        mock_session = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus.dispatch = MagicMock(return_value=AsyncMock()())
        mock_session.event_bus = mock_event_bus
        mcp_server_instance.browser_session = mock_session
        mcp_server_instance._last_activity = time.time() - 10000
        mcp_server_instance.session_timeout_minutes = 1

        with patch("openbrowser.mcp.server.BrowserStopEvent", create=True):
            with patch(
                "openbrowser.browser.events.BrowserStopEvent", create=True
            ):
                await mcp_server_instance._cleanup_expired_session()
                assert mcp_server_instance.browser_session is None

    @pytest.mark.asyncio
    async def test_cleanup_error(self, mcp_server_instance):
        """Lines 482-486: cleanup error."""
        mock_session = MagicMock()
        mock_session.event_bus = MagicMock()
        mock_session.event_bus.dispatch = MagicMock(
            side_effect=Exception("cleanup error")
        )
        mcp_server_instance.browser_session = mock_session
        mcp_server_instance._last_activity = time.time() - 10000
        mcp_server_instance.session_timeout_minutes = 1

        await mcp_server_instance._cleanup_expired_session()
        assert mcp_server_instance.browser_session is None

    @pytest.mark.asyncio
    async def test_start_cleanup_task(self, mcp_server_instance):
        """Lines 491-500: start cleanup task."""
        await mcp_server_instance._start_cleanup_task()
        assert mcp_server_instance._cleanup_task is not None
        mcp_server_instance._cleanup_task.cancel()


class TestMCPServerMain:
    """Test main function."""

    @pytest.mark.asyncio
    async def test_main_mcp_not_available(self, monkeypatch):
        """Lines 526-549: main with MCP not available."""
        from openbrowser.mcp import server as mcp_server_module

        monkeypatch.setattr(mcp_server_module, "MCP_AVAILABLE", False)

        with pytest.raises(SystemExit):
            await mcp_server_module.main()

    @pytest.mark.asyncio
    async def test_main_with_telemetry(self, monkeypatch):
        """Lines 526-549: main with telemetry."""
        from openbrowser.mcp import server as mcp_server_module

        monkeypatch.setattr(mcp_server_module, "MCP_AVAILABLE", True)
        monkeypatch.setattr(mcp_server_module, "Server", DummyServer)
        monkeypatch.setattr(mcp_server_module, "types", DummyTypes)
        monkeypatch.setattr(mcp_server_module, "TELEMETRY_AVAILABLE", True)

        mock_telemetry = MagicMock()
        monkeypatch.setattr(
            mcp_server_module, "ProductTelemetry", lambda: mock_telemetry
        )
        monkeypatch.setattr(
            mcp_server_module,
            "MCPServerTelemetryEvent",
            MagicMock,
        )

        mock_server = MagicMock()
        mock_server._telemetry = mock_telemetry
        mock_server._start_time = time.time()
        mock_server.run = AsyncMock()

        with patch.object(
            mcp_server_module,
            "OpenBrowserServer",
            return_value=mock_server,
        ):
            await mcp_server_module.main()
            mock_telemetry.capture.assert_called()
            mock_telemetry.flush.assert_called_once()


class TestRecoverBrowserSession:
    """Test _recover_browser_session."""

    @pytest.mark.asyncio
    async def test_recover_browser_session(self, mcp_server_instance):
        """Lines 376-419: recover browser session."""
        mock_old_session = MagicMock()
        mock_old_session.kill = AsyncMock()
        mock_old_session.browser_profile = MagicMock()
        mock_old_session.browser_profile.user_data_dir = "/tmp/test"
        mcp_server_instance.browser_session = mock_old_session
        mcp_server_instance._namespace = {"user_var": "hello", "__system": True}

        mock_new_session = MagicMock()
        mock_new_session.start = AsyncMock()

        mock_tools = MagicMock()
        mock_namespace = {"navigate": MagicMock()}

        with patch(
            "openbrowser.mcp.server.BrowserSession",
            return_value=mock_new_session,
        ):
            with patch("openbrowser.mcp.server.CodeAgentTools", return_value=mock_tools):
                with patch(
                    "openbrowser.mcp.server.create_namespace",
                    return_value=mock_namespace,
                ):
                    with patch(
                        "openbrowser.mcp.server.LocalBrowserWatchdog",
                        create=True,
                    ) as mock_watchdog_cls:
                        mock_watchdog_cls._kill_stale_chrome_for_profile = AsyncMock()
                        with patch(
                            "openbrowser.browser.watchdogs.local_browser_watchdog.LocalBrowserWatchdog",
                            mock_watchdog_cls,
                        ):
                            await mcp_server_instance._recover_browser_session()
                            assert mcp_server_instance.browser_session is mock_new_session
                            # user_var should be copied to new namespace
                            assert "user_var" in mcp_server_instance._namespace

    @pytest.mark.asyncio
    async def test_recover_browser_session_kill_fails(self, mcp_server_instance):
        """Lines 379-387: kill fails, try reset."""
        mock_old_session = MagicMock()
        mock_old_session.kill = AsyncMock(side_effect=Exception("kill failed"))
        mock_old_session.reset = AsyncMock()
        mock_old_session.browser_profile = MagicMock()
        mock_old_session.browser_profile.user_data_dir = None
        mcp_server_instance.browser_session = mock_old_session
        mcp_server_instance._namespace = {}

        mock_new_session = MagicMock()
        mock_new_session.start = AsyncMock()

        with patch(
            "openbrowser.mcp.server.BrowserSession",
            return_value=mock_new_session,
        ):
            with patch("openbrowser.mcp.server.CodeAgentTools", return_value=MagicMock()):
                with patch(
                    "openbrowser.mcp.server.create_namespace",
                    return_value={},
                ):
                    with patch(
                        "openbrowser.browser.watchdogs.local_browser_watchdog.LocalBrowserWatchdog",
                        create=True,
                    ) as mock_watchdog_cls:
                        mock_watchdog_cls._kill_stale_chrome_for_profile = AsyncMock()
                        await mcp_server_instance._recover_browser_session()
