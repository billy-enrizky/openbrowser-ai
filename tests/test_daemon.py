# tests/test_daemon.py
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestDaemonClient:
    """Tests for the daemon client."""

    @pytest.mark.asyncio
    async def test_send_execute_request(self):
        """Test sending an execute request over a mock socket."""
        from openbrowser.daemon.client import DaemonClient, DaemonResponse

        response_data = {'id': 1, 'success': True, 'output': 'hello', 'error': None}
        response_bytes = json.dumps(response_data).encode() + b'\n'

        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(return_value=response_bytes)
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch('asyncio.open_unix_connection', return_value=(mock_reader, mock_writer)):
            client = DaemonClient()
            result = await client.execute('print("hello")')
            assert result.success is True
            assert result.output == 'hello'
            # Verify the correct request was written to the socket
            written = mock_writer.write.call_args[0][0]
            request = json.loads(written.rstrip(b'\n').decode())
            assert request['action'] == 'execute'
            assert request['code'] == 'print("hello")'

    @pytest.mark.asyncio
    async def test_auto_start_on_connection_refused(self):
        """Test that client starts daemon when connection fails."""
        from openbrowser.daemon.client import DaemonClient

        call_count = 0

        async def mock_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionRefusedError('No daemon')
            # Second call succeeds
            response_data = {'id': 1, 'success': True, 'output': 'ok', 'error': None}
            mock_reader = AsyncMock()
            mock_reader.readline = AsyncMock(return_value=json.dumps(response_data).encode() + b'\n')
            mock_writer = MagicMock()
            mock_writer.write = MagicMock()
            mock_writer.drain = AsyncMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            return mock_reader, mock_writer

        with patch('asyncio.open_unix_connection', side_effect=mock_connect), \
             patch('openbrowser.daemon.client.DaemonClient._start_daemon', new_callable=AsyncMock) as mock_start:
            client = DaemonClient()
            result = await client.execute('pass')
            assert result.success is True
            mock_start.assert_awaited_once()
