"""Tests for daemon client methods and responses."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.daemon.client import (
    DaemonClient,
    DaemonResponse,
    execute_code_via_daemon,
)

logger = logging.getLogger(__name__)


class TestDaemonClientMethods:
    """Tests for DaemonClient status, stop, and reset methods."""

    def _make_mock_transport(self, response_data):
        """Create mock reader/writer that return response_data."""
        response_bytes = json.dumps(response_data).encode() + b"\n"
        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(return_value=response_bytes)
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        return mock_reader, mock_writer

    @pytest.mark.asyncio
    async def test_status_success(self):
        """status() returns parsed response on success."""
        reader, writer = self._make_mock_transport(
            {"id": 1, "success": True, "output": '{"pid":123}', "error": None}
        )
        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            client = DaemonClient()
            result = await client.status()
            assert result.success is True
            assert "123" in result.output

    @pytest.mark.asyncio
    async def test_status_daemon_not_running(self):
        """status() returns error when daemon is not running."""
        with patch("asyncio.open_unix_connection", side_effect=ConnectionRefusedError("nope")):
            client = DaemonClient()
            result = await client.status()
            assert result.success is False
            assert "not running" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stop_success(self):
        """stop() returns parsed response on success."""
        reader, writer = self._make_mock_transport(
            {"id": 1, "success": True, "output": "Daemon stopping", "error": None}
        )
        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            client = DaemonClient()
            result = await client.stop()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_stop_daemon_not_running(self):
        """stop() returns error when daemon is not running."""
        with patch("asyncio.open_unix_connection", side_effect=FileNotFoundError("nope")):
            client = DaemonClient()
            result = await client.stop()
            assert result.success is False
            assert "not running" in result.error.lower()

    @pytest.mark.asyncio
    async def test_reset_success(self):
        """reset() returns parsed response on success."""
        reader, writer = self._make_mock_transport(
            {"id": 1, "success": True, "output": "Session reset", "error": None}
        )
        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            client = DaemonClient()
            result = await client.reset()
            assert result.success is True
            assert "reset" in result.output.lower()

    @pytest.mark.asyncio
    async def test_reset_daemon_not_running(self):
        """reset() returns error when daemon is not running."""
        with patch("asyncio.open_unix_connection", side_effect=OSError("nope")):
            client = DaemonClient()
            result = await client.reset()
            assert result.success is False
            assert "not running" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_connection_reset(self):
        """_send raises ConnectionResetError when daemon closes without responding."""
        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(return_value=b"")
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_unix_connection", return_value=(mock_reader, mock_writer)):
            client = DaemonClient()
            with pytest.raises(ConnectionResetError):
                await client._send({"action": "status"})

    @pytest.mark.asyncio
    async def test_execute_code_via_daemon(self):
        """execute_code_via_daemon convenience function works."""
        reader, writer = self._make_mock_transport(
            {"id": 1, "success": True, "output": "42", "error": None}
        )
        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            result = await execute_code_via_daemon("print(42)")
            assert result.success is True
            assert result.output == "42"


class TestDaemonResponse:
    """Tests for the DaemonResponse dataclass."""

    def test_daemon_response_fields(self):
        """DaemonResponse stores fields correctly."""
        resp = DaemonResponse(success=True, output="hello", error=None)
        assert resp.success is True
        assert resp.output == "hello"
        assert resp.error is None

    def test_daemon_response_with_error(self):
        """DaemonResponse with error field."""
        resp = DaemonResponse(success=False, output="", error="boom")
        assert resp.success is False
        assert resp.error == "boom"
