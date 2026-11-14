"""Comprehensive tests for src/openbrowser/daemon/client.py to cover remaining gaps.

Missing lines: 39, 50-73, 89-90
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _noop_start_lock():
    yield


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
    async def test_start_daemon_success(self, tmp_path):
        """Lines 50-73: _start_daemon spawns and waits."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_sock_path = MagicMock()
        mock_sock_path.parent = MagicMock()
        mock_sock_path.exists.return_value = True

        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("openbrowser.daemon.client.get_socket_path", return_value=mock_sock_path), patch(
            "openbrowser.daemon.client.get_pid_path", return_value=tmp_path / "daemon.pid"
        ):
            with patch("openbrowser.daemon.client.DAEMON_DIR", MagicMock()):
                with patch("os.open", return_value=3):
                    with patch("builtins.open", MagicMock()):
                        process = MagicMock()
                        process.poll.return_value = None
                        with patch.object(client, "_start_lock", _noop_start_lock), patch(
                            "subprocess.Popen", return_value=process
                        ):
                            with patch.object(
                                client,
                                "_probe",
                                new_callable=AsyncMock,
                                return_value=True,
                            ):
                                with patch("openbrowser.daemon.client.IS_WINDOWS", False):
                                    await client._start_daemon()

    @pytest.mark.asyncio
    async def test_start_daemon_removes_a_stale_socket(self, tmp_path):
        """A socket with no listener and no owner is cleaned before spawning."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_sock_path = MagicMock()
        mock_sock_path.parent = MagicMock()
        mock_sock_path.exists.return_value = True

        with patch("openbrowser.daemon.client.get_socket_path", return_value=mock_sock_path), patch(
            "openbrowser.daemon.client.get_pid_path", return_value=tmp_path / "daemon.pid"
        ):
            with patch("openbrowser.daemon.client.DAEMON_DIR", MagicMock()):
                with patch("os.open", return_value=3):
                    with patch("builtins.open", MagicMock()):
                        process = MagicMock()
                        process.poll.return_value = None
                        with patch.object(client, "_start_lock", _noop_start_lock), patch(
                            "subprocess.Popen", return_value=process
                        ):
                            with patch.object(
                                client,
                                "_socket_accepts_connection",
                                new_callable=AsyncMock,
                                return_value=False,
                            ), patch.object(client, "_daemon_owner_for_socket", return_value=None):
                                with patch.object(
                                    client,
                                    "_probe",
                                    new_callable=AsyncMock,
                                    side_effect=[False, True],
                                ):
                                    with patch("openbrowser.daemon.client.IS_WINDOWS", False):
                                        await client._start_daemon()

        mock_sock_path.unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    async def test_start_daemon_refuses_a_busy_listener(self, tmp_path):
        """A socket that accepts connections is busy, not stale, and is not killed."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_sock_path = MagicMock()
        mock_sock_path.exists.return_value = True

        with (
            patch("openbrowser.daemon.client.get_socket_path", return_value=mock_sock_path),
            patch("openbrowser.daemon.client.get_pid_path", return_value=tmp_path / "daemon.pid"),
            patch.object(client, "_start_lock", _noop_start_lock),
            patch.object(client, "_probe", new_callable=AsyncMock, return_value=False),
            patch.object(client, "_socket_accepts_connection", new_callable=AsyncMock, return_value=True),
            patch("subprocess.Popen") as popen,
        ):
            with pytest.raises(OSError, match="did not answer status"):
                await client._start_daemon()

        popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_daemon_refuses_a_verified_owner_without_a_socket(self, tmp_path):
        """A live owner with no socket is an orphan, not permission to start another daemon."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_sock_path = MagicMock()
        mock_sock_path.exists.return_value = False

        with (
            patch("openbrowser.daemon.client.get_socket_path", return_value=mock_sock_path),
            patch("openbrowser.daemon.client.get_pid_path", return_value=tmp_path / "daemon.pid"),
            patch.object(client, "_start_lock", _noop_start_lock),
            patch.object(client, "_daemon_owner_for_socket", return_value={"pid": 123}),
            patch("subprocess.Popen") as popen,
        ):
            with pytest.raises(OSError, match="daemon reap"):
                await client._start_daemon()

        popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_daemon_timeout(self, tmp_path):
        """Line 73: _start_daemon timeout."""
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        mock_sock_path = MagicMock()
        mock_sock_path.parent = MagicMock()
        mock_sock_path.exists.return_value = False

        with patch("openbrowser.daemon.client.get_socket_path", return_value=mock_sock_path), patch(
            "openbrowser.daemon.client.get_pid_path", return_value=tmp_path / "daemon.pid"
        ), patch.object(client, "_daemon_owner_for_socket", return_value=None):
            with patch("openbrowser.daemon.client.DAEMON_DIR", MagicMock()):
                with patch("openbrowser.daemon.client.DAEMON_START_TIMEOUT", 0.1):
                    with patch("os.open", return_value=3):
                        with patch("builtins.open", MagicMock()):
                            process = MagicMock()
                            process.poll.return_value = None
                            with patch.object(client, "_start_lock", _noop_start_lock), patch(
                                "subprocess.Popen", return_value=process
                            ):
                                with patch.object(
                                    client,
                                    "_probe",
                                    new_callable=AsyncMock,
                                    return_value=False,
                                ):
                                    with patch("openbrowser.daemon.client.IS_WINDOWS", False):
                                        with pytest.raises(TimeoutError):
                                            await client._start_daemon()

    @pytest.mark.asyncio
    async def test_start_daemon_serializes_concurrent_callers(self, tmp_path):
        """Concurrent auto-start callers share one daemon start operation."""
        from openbrowser.daemon.client import DaemonClient

        first_client = DaemonClient()
        second_client = DaemonClient()
        socket_path = tmp_path / "daemon.sock"
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0
        active = 0
        max_active = 0

        async def fake_start_unlocked():
            nonlocal calls, active, max_active
            calls += 1
            active += 1
            max_active = max(max_active, active)
            entered.set()
            await release.wait()
            active -= 1

        with (
            patch("openbrowser.daemon.client.get_socket_path", return_value=socket_path),
            patch.object(first_client, "_start_daemon_unlocked", side_effect=fake_start_unlocked),
            patch.object(second_client, "_start_daemon_unlocked", side_effect=fake_start_unlocked),
        ):
            first = asyncio.create_task(first_client._start_daemon())
            await entered.wait()
            second = asyncio.create_task(second_client._start_daemon())
            await asyncio.sleep(0.05)
            assert calls == 1
            release.set()
            await asyncio.gather(first, second)

        assert calls == 2
        assert max_active == 1


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
            assert result.success is True
            assert result.output == "ok"

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
            result = await client._send({"action": "status"})
            assert result.success is False
            assert result.error_code == "daemon_connection_lost"

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
            assert result.success is True


class TestDaemonClientExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test execute success."""
        from openbrowser.daemon.client import DaemonClient, DaemonResponse

        client = DaemonClient()

        with patch.object(
            client,
            "_execute_once",
            new_callable=AsyncMock,
            return_value=DaemonResponse(success=True, output="42", error=None),
        ):
            result = await client.execute("print(42)")
            assert result.success is True
            assert result.output == "42"

    @pytest.mark.asyncio
    async def test_execute_auto_start(self):
        """Test execute auto-starts daemon on connection error."""
        from openbrowser.daemon.client import DaemonClient, DaemonResponse

        client = DaemonClient()
        with patch.object(
            client,
            "_execute_once",
            new_callable=AsyncMock,
            side_effect=[None, DaemonResponse(success=True, output="42", error=None)],
        ):
            with patch.object(client, "_start_daemon", new_callable=AsyncMock):
                result = await client.execute("print(42)")
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_does_not_retry_after_ambiguous_failure(self):
        """A lost response must not replay code that may have already run."""
        from openbrowser.daemon.client import DaemonClient, DaemonResponse

        client = DaemonClient()
        failure = DaemonResponse(
            success=False,
            output='',
            error='connection lost',
            error_code='daemon_connection_lost',
        )
        with patch.object(client, '_execute_once', new_callable=AsyncMock, return_value=failure):
            with patch.object(client, '_start_daemon', new_callable=AsyncMock) as start:
                result = await client.execute("click(index=1)")

        assert result is failure
        start.assert_not_awaited()


class TestDaemonClientStatus:
    """Test status method."""

    @pytest.mark.asyncio
    async def test_status_success(self):
        """Test status success."""
        from openbrowser.daemon.client import DaemonClient, DaemonResponse

        client = DaemonClient()

        with patch.object(
            client,
            "_send",
            new_callable=AsyncMock,
            return_value=DaemonResponse(success=True, output="running", error=None),
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
        from openbrowser.daemon.client import DaemonClient, DaemonResponse

        client = DaemonClient()

        with patch.object(
            client,
            "_send",
            new_callable=AsyncMock,
            return_value=DaemonResponse(success=True, output="Daemon stopped", error=None),
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
        from openbrowser.daemon.client import DaemonClient, DaemonResponse

        client = DaemonClient()

        with patch.object(
            client,
            "_send",
            new_callable=AsyncMock,
            return_value=DaemonResponse(success=True, output="Session reset", error=None),
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
