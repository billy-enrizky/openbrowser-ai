"""Comprehensive tests for src/openbrowser/daemon/client.py to cover remaining gaps.

Missing lines: 39, 50-73, 89-90
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


class TestDaemonResponse:
    """Test DaemonResponse dataclass."""

    def test_dataclass_fields(self):
        """Test DaemonResponse has correct fields."""
        from openbrowser.daemon.client import DaemonResponse

        resp = DaemonResponse(success=True, output="hello", error=None)
        assert resp.success is True
        assert resp.output == "hello"
        assert resp.error is None


class TestDaemonClientConnect:
    """Test _connect method."""

    @pytest.mark.asyncio
    async def test_connect_unix(self):
        """Test _connect on Unix."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        async def fake_wait_for(coro, **kwargs):
            """Await the inner coroutine and return the mock result."""
            await coro
            return (mock_reader, mock_writer)

        with patch("openbrowser.daemon.client.IS_WINDOWS", False):
            with patch("openbrowser.daemon.client.get_socket_path", return_value=Path("/tmp/test.sock")):
                with patch(
                    "asyncio.open_unix_connection",
                    new_callable=AsyncMock,
                    return_value=(mock_reader, mock_writer),
                ):
                    with patch("asyncio.wait_for", side_effect=fake_wait_for):
                        reader, writer = await client._connect()
                        assert reader is mock_reader

    @pytest.mark.asyncio
    async def test_connect_windows(self):
        """Line 39: _connect on Windows."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        async def fake_wait_for(coro, **kwargs):
            """Await the inner coroutine and return the mock result."""
            await coro
            return (mock_reader, mock_writer)

        with patch("openbrowser.daemon.client.IS_WINDOWS", True):
            with patch("asyncio.open_connection", new_callable=AsyncMock, return_value=(mock_reader, mock_writer)):
                with patch("asyncio.wait_for", side_effect=fake_wait_for):
                    reader, writer = await client._connect()
                    assert reader is mock_reader


class TestDaemonClientStartDaemon:
    """Test _start_daemon method."""

    @pytest.mark.asyncio
    async def test_start_daemon_success(self):
        """Lines 50-73: _start_daemon spawns and waits."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_sock_path = MagicMock()
        mock_sock_path.parent = MagicMock()
        mock_sock_path.exists.return_value = True

        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("openbrowser.daemon.client.get_socket_path", return_value=mock_sock_path):
            with patch("openbrowser.daemon.client.DAEMON_DIR", MagicMock()):
                with patch("os.open", return_value=3):
                    with patch("builtins.open", MagicMock()):
                        with patch("subprocess.Popen"):
                            with patch.object(
                                client,
                                "_connect",
                                new_callable=AsyncMock,
                                return_value=(AsyncMock(), mock_writer),
                            ):
                                with patch("openbrowser.daemon.client.IS_WINDOWS", False):
                                    await client._start_daemon()

    @pytest.mark.asyncio
    async def test_start_daemon_timeout(self):
        """Line 73: _start_daemon timeout."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_sock_path = MagicMock()
        mock_sock_path.parent = MagicMock()
        mock_sock_path.exists.return_value = False

        with patch("openbrowser.daemon.client.get_socket_path", return_value=mock_sock_path):
            with patch("openbrowser.daemon.client.DAEMON_DIR", MagicMock()):
                with patch("openbrowser.daemon.client.DAEMON_START_TIMEOUT", 0.1):
                    with patch("os.open", return_value=3):
                        with patch("builtins.open", MagicMock()):
                            with patch("subprocess.Popen"):
                                with patch("openbrowser.daemon.client.IS_WINDOWS", False):
                                    with pytest.raises(TimeoutError):
                                        await client._start_daemon()


class TestDaemonClientSend:
    """Test _send method."""

    @pytest.mark.asyncio
    async def test_send_success(self):
        """Test _send returns response."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        response = {"id": 1, "success": True, "output": "ok", "error": None}

        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(
            return_value=json.dumps(response).encode() + b"\n"
        )
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        mock_writer.drain = AsyncMock()

        with patch.object(
            client,
            "_connect",
            new_callable=AsyncMock,
            return_value=(mock_reader, mock_writer),
        ):
            result = await client._send({"action": "status"})
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_send_empty_response(self):
        """Test _send with empty response."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()

        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(return_value=b"")
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        mock_writer.drain = AsyncMock()

        with patch.object(
            client,
            "_connect",
            new_callable=AsyncMock,
            return_value=(mock_reader, mock_writer),
        ):
            with pytest.raises(ConnectionResetError):
                await client._send({"action": "status"})

    @pytest.mark.asyncio
    async def test_send_writer_wait_closed_error(self):
        """Lines 89-90: _send writer.wait_closed raises."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        response = {"id": 1, "success": True, "output": "ok", "error": None}

        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(
            return_value=json.dumps(response).encode() + b"\n"
        )
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock(side_effect=Exception("close error"))
        mock_writer.drain = AsyncMock()

        with patch.object(
            client,
            "_connect",
            new_callable=AsyncMock,
            return_value=(mock_reader, mock_writer),
        ):
            result = await client._send({"action": "status"})
            assert result["success"] is True


class TestDaemonClientExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test execute success."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        response = {"id": 1, "success": True, "output": "42", "error": None}

        with patch.object(
            client, "_send", new_callable=AsyncMock, return_value=response
        ):
            result = await client.execute("print(42)")
            assert result.success is True
            assert result.output == "42"

    @pytest.mark.asyncio
    async def test_execute_auto_start(self):
        """Test execute auto-starts daemon on connection error."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        response = {"id": 1, "success": True, "output": "42", "error": None}

        call_count = [0]

        async def mock_send(req):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionRefusedError("not running")
            return response

        with patch.object(client, "_send", side_effect=mock_send):
            with patch.object(client, "_start_daemon", new_callable=AsyncMock):
                result = await client.execute("print(42)")
                assert result.success is True


class TestDaemonClientStatus:
    """Test status method."""

    @pytest.mark.asyncio
    async def test_status_success(self):
        """Test status success."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        response = {"id": 1, "success": True, "output": "running", "error": None}

        with patch.object(
            client, "_send", new_callable=AsyncMock, return_value=response
        ):
            result = await client.status()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_status_not_running(self):
        """Test status when daemon not running."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()

        with patch.object(
            client,
            "_send",
            new_callable=AsyncMock,
            side_effect=ConnectionRefusedError(),
        ):
            result = await client.status()
            assert result.success is False
            assert "not running" in result.error


class TestDaemonClientStop:
    """Test stop method."""

    @pytest.mark.asyncio
    async def test_stop_success(self):
        """Test stop success."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        response = {"id": 1, "success": True, "output": "Daemon stopped", "error": None}

        with patch.object(
            client, "_send", new_callable=AsyncMock, return_value=response
        ):
            result = await client.stop()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Test stop when daemon not running."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()

        with patch.object(
            client,
            "_send",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError(),
        ):
            result = await client.stop()
            assert result.success is False


class TestDaemonClientReset:
    """Test reset method."""

    @pytest.mark.asyncio
    async def test_reset_success(self):
        """Test reset success."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        response = {"id": 1, "success": True, "output": "Session reset", "error": None}

        with patch.object(
            client, "_send", new_callable=AsyncMock, return_value=response
        ):
            result = await client.reset()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_reset_not_running(self):
        """Test reset when daemon not running."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()

        with patch.object(
            client,
            "_send",
            new_callable=AsyncMock,
            side_effect=OSError(),
        ):
            result = await client.reset()
            assert result.success is False


class TestExecuteCodeViaDaemon:
    """Test convenience function."""

    @pytest.mark.asyncio
    async def test_execute_code_via_daemon(self):
        """Test execute_code_via_daemon."""
        from openbrowser.daemon.client import execute_code_via_daemon, DaemonResponse

        with patch(
            "openbrowser.daemon.client.DaemonClient.execute",
            new_callable=AsyncMock,
            return_value=DaemonResponse(success=True, output="42", error=None),
        ):
            result = await execute_code_via_daemon("print(42)")
            assert result.success is True
            assert result.output == "42"
