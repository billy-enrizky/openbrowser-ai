"""Comprehensive tests for src/openbrowser/daemon/client.py to cover remaining gaps.

Missing lines: 39, 50-73, 89-90
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.daemon.ownership import SocketEntryIdentity

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
                        ), patch("openbrowser.daemon.client.DaemonOwnerLock.acquire", return_value=True), patch(
                            "openbrowser.daemon.client.socket_entry_identity", return_value=None
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
                        ), patch("openbrowser.daemon.client.DaemonOwnerLock.acquire", return_value=True):
                            with patch.object(
                                client,
                                "_socket_accepts_connection",
                                new_callable=AsyncMock,
                                return_value=False,
                            ), patch.object(client, "_daemon_owner_for_socket", return_value=None), patch.object(
                                client, "_remove_stale_socket"
                            ) as remove_stale, patch.object(
                                client, "_socket_state", new_callable=AsyncMock, return_value="stale"
                            ), patch(
                                "openbrowser.daemon.client.socket_entry_identity",
                                side_effect=[
                                    SocketEntryIdentity(1, 1, True),
                                    SocketEntryIdentity(1, 1, True),
                                ],
                            ):
                                with patch.object(
                                    client,
                                    "_probe",
                                    new_callable=AsyncMock,
                                    side_effect=[False, True],
                                ):
                                    with patch("openbrowser.daemon.client.IS_WINDOWS", False):
                                        await client._start_daemon()

        remove_stale.assert_called_once_with(mock_sock_path)

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
            patch("openbrowser.daemon.client.get_log_path", return_value=tmp_path / "daemon.log"),
            patch("openbrowser.daemon.client.DAEMON_DIR", tmp_path),
            patch.object(client, "_start_lock", _noop_start_lock),
            patch.object(client, "_probe", new_callable=AsyncMock, return_value=False),
            patch.object(client, "_socket_state", new_callable=AsyncMock, return_value="live"),
            patch(
                "openbrowser.daemon.client.socket_entry_identity",
                return_value=SocketEntryIdentity(1, 1, True),
            ),
            patch("subprocess.Popen") as popen,
        ):
            with pytest.raises(OSError, match="did not answer status"):
                await client._start_daemon()

        popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_daemon_refuses_the_lifetime_owner_lock(self, tmp_path):
        """A second starter cannot inspect or remove files while another owns startup."""
        from openbrowser.daemon.client import DaemonClient
        from openbrowser.daemon.ownership import (
            DaemonOwnerLock,
            owner_lock_path_for_pid_file,
        )

        client = DaemonClient()
        pid_path = tmp_path / 'daemon.pid'
        socket_path = tmp_path / 'daemon.sock'
        pid_path.write_text('4321')
        socket_path.write_text('owner sentinel')
        lock = DaemonOwnerLock(owner_lock_path_for_pid_file(pid_path))
        assert lock.acquire()
        try:
            with (
                patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
                patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
                patch('openbrowser.daemon.client.get_log_path', return_value=tmp_path / 'daemon.log'),
                patch('openbrowser.daemon.client.DAEMON_DIR', tmp_path),
                patch.object(client, '_start_lock', _noop_start_lock),
                patch.object(client, '_probe', new_callable=AsyncMock, return_value=False),
                patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='stale'),
                patch(
                    'openbrowser.daemon.client.socket_entry_identity',
                    return_value=SocketEntryIdentity(1, 1, True),
                ),
                patch('openbrowser.daemon.client._start_timeout', return_value=0.1),
                patch('openbrowser.daemon.client.IS_WINDOWS', False),
                patch('subprocess.Popen') as popen,
            ):
                with pytest.raises(OSError, match='startup is already in progress'):
                    await client._start_daemon()
        finally:
            lock.release()

        popen.assert_not_called()
        assert pid_path.read_text() == '4321'
        assert socket_path.read_text() == 'owner sentinel'

    @pytest.mark.asyncio
    async def test_start_daemon_adopts_a_competing_owner(self, tmp_path):
        """A direct server launch that wins the owner lock is adopted, not reported as a failure."""
        from openbrowser.daemon.client import DaemonClient
        from openbrowser.daemon.ownership import (
            DaemonOwnerLock,
            owner_lock_path_for_pid_file,
        )

        client = DaemonClient()
        pid_path = tmp_path / 'daemon.pid'
        socket_path = tmp_path / 'daemon.sock'
        lock = DaemonOwnerLock(owner_lock_path_for_pid_file(pid_path))
        assert lock.acquire()
        try:
            with (
                patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
                patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
                patch('openbrowser.daemon.client.get_log_path', return_value=tmp_path / 'daemon.log'),
                patch('openbrowser.daemon.client.DAEMON_DIR', tmp_path),
                patch.object(client, '_start_lock', _noop_start_lock),
                patch.object(client, '_probe', new_callable=AsyncMock, return_value=True),
                patch('openbrowser.daemon.client._start_timeout', return_value=5.0),
                patch('openbrowser.daemon.client.IS_WINDOWS', False),
                patch('subprocess.Popen') as popen,
            ):
                await client._start_daemon()
        finally:
            lock.release()

        popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_daemon_adopts_an_owner_that_won_after_spawn(self, tmp_path):
        """A child that exits on lock contention yields to the competing daemon."""
        from openbrowser.daemon import EXIT_OWNER_BUSY
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        pid_path = tmp_path / 'daemon.pid'
        socket_path = tmp_path / 'daemon.sock'
        process = MagicMock()
        process.poll.return_value = EXIT_OWNER_BUSY
        process.returncode = EXIT_OWNER_BUSY

        with (
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
            patch('openbrowser.daemon.client.get_log_path', return_value=tmp_path / 'daemon.log'),
            patch('openbrowser.daemon.client.DAEMON_DIR', tmp_path),
            patch.object(client, '_start_lock', _noop_start_lock),
            patch.object(client, '_daemon_owner_for_socket', return_value=None),
            patch.object(client, '_verified_daemon_pid', return_value=None),
            patch.object(
                client,
                '_probe',
                new_callable=AsyncMock,
                side_effect=[False, True],
            ),
            patch('openbrowser.daemon.client.DaemonOwnerLock.acquire', return_value=True),
            patch('openbrowser.daemon.client._start_timeout', return_value=5.0),
            patch('openbrowser.daemon.client.IS_WINDOWS', False),
            patch('subprocess.Popen', return_value=process) as popen,
        ):
            await client._start_daemon()

        popen.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_daemon_retries_once_after_a_transient_owner(self):
        """A cleanup-held lock releases without a daemon, so startup tries once more."""
        from openbrowser.daemon.client import DaemonClient, _StartRetry

        client = DaemonClient()
        calls: list[int] = []

        async def attempt():
            calls.append(1)
            if len(calls) == 1:
                raise _StartRetry()

        with (
            patch.object(client, '_start_lock', _noop_start_lock),
            patch.object(client, '_start_daemon_unlocked', side_effect=attempt),
        ):
            await client._start_daemon()

        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_start_daemon_converts_a_repeated_transient_owner_to_a_clear_error(self):
        """The internal retry signal never escapes to callers."""
        from openbrowser.daemon.client import DaemonClient, _StartRetry

        client = DaemonClient()

        with (
            patch.object(client, '_start_lock', _noop_start_lock),
            patch.object(
                client,
                '_start_daemon_unlocked',
                new_callable=AsyncMock,
                side_effect=_StartRetry(),
            ) as attempt,
            pytest.raises(OSError, match='startup is already in progress'),
        ):
            await client._start_daemon()

        assert attempt.await_count == 2

    @pytest.mark.asyncio
    async def test_start_daemon_refuses_a_socket_that_changed_after_the_probe(self, tmp_path):
        """A path replaced between the probe and cleanup is never unlinked."""
        from openbrowser.daemon.client import DaemonClient
        from openbrowser.daemon.ownership import SocketEntryIdentity

        client = DaemonClient()
        socket_path = tmp_path / 'daemon.sock'
        socket_path.write_text('sentinel')
        process = MagicMock()
        process.poll.return_value = None

        with (
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch('openbrowser.daemon.client.get_pid_path', return_value=tmp_path / 'daemon.pid'),
            patch('openbrowser.daemon.client.get_log_path', return_value=tmp_path / 'daemon.log'),
            patch('openbrowser.daemon.client.DAEMON_DIR', tmp_path),
            patch.object(client, '_start_lock', _noop_start_lock),
            patch.object(client, '_probe', new_callable=AsyncMock, return_value=False),
            patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='stale'),
            patch.object(client, '_daemon_owner_for_socket', return_value=None),
            patch.object(client, '_verified_daemon_pid', return_value=None),
            patch(
                'openbrowser.daemon.client.socket_entry_identity',
                side_effect=[
                    SocketEntryIdentity(1, 1, True),
                    SocketEntryIdentity(1, 2, True),
                ],
            ),
            patch('openbrowser.daemon.client.DaemonOwnerLock.acquire', return_value=True),
            patch('openbrowser.daemon.client.IS_WINDOWS', False),
            patch('subprocess.Popen', return_value=process) as popen,
            pytest.raises(OSError, match='changed during startup'),
        ):
            await client._start_daemon()

        popen.assert_not_called()
        assert socket_path.read_text() == 'sentinel'

    @pytest.mark.asyncio
    async def test_start_daemon_reaps_a_child_that_never_becomes_ready(self, tmp_path):
        """A child that outlives the readiness budget is terminated and killed."""
        import subprocess

        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired('daemon', 5)

        with (
            patch('openbrowser.daemon.client.get_socket_path', return_value=tmp_path / 'daemon.sock'),
            patch('openbrowser.daemon.client.get_pid_path', return_value=tmp_path / 'daemon.pid'),
            patch('openbrowser.daemon.client.get_log_path', return_value=tmp_path / 'daemon.log'),
            patch('openbrowser.daemon.client.DAEMON_DIR', tmp_path),
            patch.object(client, '_start_lock', _noop_start_lock),
            patch.object(client, '_daemon_owner_for_socket', return_value=None),
            patch.object(client, '_verified_daemon_pid', return_value=None),
            patch.object(client, '_probe', new_callable=AsyncMock, return_value=False),
            patch('openbrowser.daemon.client.socket_entry_identity', return_value=None),
            patch('openbrowser.daemon.client.DaemonOwnerLock.acquire', return_value=True),
            patch('openbrowser.daemon.client._start_timeout', return_value=0.05),
            patch('openbrowser.daemon.client.IS_WINDOWS', False),
            patch('subprocess.Popen', return_value=process),
            pytest.raises(TimeoutError, match='did not become ready'),
        ):
            await client._start_daemon()

        process.terminate.assert_called_once()
        process.kill.assert_called_once()

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
            patch("openbrowser.daemon.client.get_log_path", return_value=tmp_path / "daemon.log"),
            patch("openbrowser.daemon.client.DAEMON_DIR", tmp_path),
            patch.object(client, "_start_lock", _noop_start_lock),
            patch.object(client, "_daemon_owner_for_socket", return_value={"pid": 123}),
            patch("openbrowser.daemon.client.socket_entry_identity", return_value=None),
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
        ), patch.object(client, "_daemon_owner_for_socket", return_value=None), patch(
            "openbrowser.daemon.client.socket_entry_identity", return_value=None
        ):
            with patch("openbrowser.daemon.client.DAEMON_DIR", MagicMock()):
                with patch("openbrowser.daemon.client.DAEMON_START_TIMEOUT", 0.1):
                    with patch("os.open", return_value=3):
                        with patch("builtins.open", MagicMock()):
                            process = MagicMock()
                            process.poll.return_value = None
                            with patch.object(client, "_start_lock", _noop_start_lock), patch(
                                "subprocess.Popen", return_value=process
                            ), patch("openbrowser.daemon.client.DaemonOwnerLock.acquire", return_value=True):
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
        from openbrowser.daemon.client import DaemonResponse, execute_code_via_daemon

        with patch(
            "openbrowser.daemon.client.DaemonClient.execute",
            new_callable=AsyncMock,
            return_value=DaemonResponse(success=True, output="42", error=None),
        ):
            result = await execute_code_via_daemon("print(42)")
            assert result.success is True
            assert result.output == "42"
