"""Tests for daemon client methods and responses."""

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
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
    async def test_readiness_probe_uses_lightweight_ping(self):
        """The startup probe must not build the full status payload."""
        client = DaemonClient()
        reader, writer = self._make_mock_transport({"id": 1, "success": True})
        with (
            patch.object(client, "_connect", AsyncMock(return_value=(reader, writer))),
            patch.object(
                client,
                "_exchange",
                AsyncMock(return_value={"success": True}),
            ) as exchange,
        ):
            assert await client._probe() is True
        assert exchange.await_args.args[2] == {"id": 1, "action": "ping"}

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
        """_send returns a structured error when the daemon closes without responding."""
        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(return_value=b"")
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch("asyncio.open_unix_connection", return_value=(mock_reader, mock_writer)):
            client = DaemonClient()
            result = await client._send({"action": "status"})
            assert result.success is False
            assert result.error_code == "daemon_connection_lost"

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


class TestDaemonClientRecoveryDiscovery:
    """Tests for finding browser leases without a responsive daemon socket."""

    def _write_metadata(self, profile_dir: Path, *, owner_pid: int, owner_start_time: float) -> None:
        profile_dir.mkdir(parents=True)
        (profile_dir / '.openbrowser-profile.json').write_text(
            json.dumps(
                {
                    'schema_version': 1,
                    'profile_dir': str(profile_dir),
                    'instance_id': 'lease-id',
                    'owner_pid': owner_pid,
                    'owner_start_time': owner_start_time,
                    'browser': None,
                }
            )
        )

    def test_profile_dirs_for_daemon_matches_pid_and_start_time(self, tmp_path):
        profiles_root = tmp_path / 'profiles'
        profile_dir = profiles_root / 'daemon'
        self._write_metadata(profile_dir, owner_pid=123, owner_start_time=456.0)
        process = MagicMock(pid=123)
        process.create_time.return_value = 456.0

        with patch('openbrowser.config.CONFIG', new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=profiles_root)):
            matches = DaemonClient()._profile_dirs_for_daemon(process, 123)

        assert matches == [str(profile_dir.resolve())]

    def test_profile_dirs_for_daemon_rejects_a_reused_pid(self, tmp_path):
        profiles_root = tmp_path / 'profiles'
        profile_dir = profiles_root / 'daemon'
        self._write_metadata(profile_dir, owner_pid=123, owner_start_time=456.0)
        process = MagicMock(pid=123)
        process.create_time.return_value = 999.0

        with patch('openbrowser.config.CONFIG', new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=profiles_root)):
            matches = DaemonClient()._profile_dirs_for_daemon(process, 123)

        assert matches == []

    @pytest.mark.asyncio
    async def test_force_stop_discovers_browser_lease_without_status(self, tmp_path):
        client = DaemonClient()
        client.status = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client.stop = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client._read_pid = MagicMock(return_value=123)

        process = MagicMock()
        process.cmdline.return_value = ['python', '-m', 'openbrowser.daemon.server']
        pid_path = tmp_path / 'daemon.pid'
        socket_path = tmp_path / 'daemon.sock'
        pid_path.write_text('123')

        with (
            patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch('psutil.Process', return_value=process),
            patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='stale'),
            patch.object(client, '_profile_dirs_for_daemon', return_value=['/tmp/profile']) as discover,
            patch.object(
                client,
                '_kill_recorded_browsers',
                new_callable=AsyncMock,
                return_value=(True, None),
            ) as cleanup,
        ):
            result = await client.force_stop(timeout=0.01)

        assert result.success is True
        discover.assert_called_once_with(process, 123)
        cleanup.assert_awaited_once_with(['/tmp/profile'], 0.01, expected_owner_pid=123)

    @pytest.mark.skipif(os.name == 'nt', reason='Unix sockets are not used on Windows')
    def test_remove_stale_socket_deletes_a_socket_file(self):
        short_dir = Path(tempfile.mkdtemp(prefix='ob_sock_', dir='/tmp'))
        path = short_dir / 'd.sock'
        try:
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path))
            listener.close()
            assert path.exists()

            DaemonClient._remove_stale_socket(path)

            assert not path.exists()
        finally:
            shutil.rmtree(short_dir, ignore_errors=True)


    def test_remove_stale_socket_refuses_a_regular_file(self, tmp_path):
        path = tmp_path / 'not-a-socket'
        path.write_text('important data')

        with pytest.raises(OSError, match='Refusing to remove non-socket'):
            DaemonClient._remove_stale_socket(path)

        assert path.read_text() == 'important data'


    def test_windows_owner_is_the_daemon_listening_on_the_loopback_port(self):
        with (
            patch('openbrowser.daemon.client.IS_WINDOWS', True),
            patch(
                'openbrowser.daemon.diagnostics.daemon_processes',
                return_value=[
                    {'pid': 1, 'socket': None, 'socket_live': False},
                    {'pid': 2, 'socket': None, 'socket_live': True},
                ],
            ),
        ):
            owner = DaemonClient()._daemon_owner_for_socket()

        assert owner == {'pid': 2, 'socket': None, 'socket_live': True}


    @pytest.mark.asyncio
    async def test_kill_recorded_browser_terminates_only_the_verified_process(self, tmp_path):
        import asyncio

        import psutil

        from openbrowser.browser.watchdogs.profile_lease import ProfileLease

        profile_dir = tmp_path / 'profile'
        profile_dir.mkdir()
        instance_id = 'test-instance'
        marker = f'--openbrowser-instance-id={instance_id}'
        process = subprocess.Popen(
            [
                sys.executable,
                '-c',
                'import time; print("OPENBROWSER_TEST_CHILD_READY", flush=True); time.sleep(60)',
                f'--user-data-dir={profile_dir}',
                marker,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            assert process.stdout is not None
            ready = await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), timeout=5)
            assert ready == 'OPENBROWSER_TEST_CHILD_READY\n'
            observed = psutil.Process(process.pid)
            browser_record = {
                'pid': process.pid,
                'start_time': observed.create_time(),
                'profile_dir': str(profile_dir.resolve()),
                'instance_id': instance_id,
                'ownership_marker': marker,
                'executable': sys.executable,
                'cdp_port': 1,
            }
            (profile_dir / '.openbrowser-profile.json').write_text(
                json.dumps(
                    {
                        'schema_version': 1,
                        'profile_dir': str(profile_dir.resolve()),
                        'instance_id': instance_id,
                        'owner_pid': 999999,
                        'owner_start_time': 1.0,
                        'browser': browser_record,
                    }
                )
            )

            identity_check = psutil.Process(process.pid)
            assert ProfileLease.process_matches_browser(identity_check, browser_record), {
                'expected_pid': browser_record['pid'],
                'observed_pid': identity_check.pid,
                'expected_start_time': browser_record['start_time'],
                'observed_start_time': identity_check.create_time(),
                'cmdline': identity_check.cmdline(),
            }
            ok, error = await DaemonClient()._kill_recorded_browser(str(profile_dir), timeout=5)

            assert ok is True
            assert error is None
            process.wait(timeout=3)
            assert process.returncode is not None
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            if process.stdout is not None:
                process.stdout.close()


    @pytest.mark.asyncio
    async def test_force_stop_runs_browser_cleanup_before_a_filesystem_error(self, tmp_path):
        """A refused file cleanup must not hide a browser that still needs killing."""
        client = DaemonClient()
        client.status = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client.stop = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client._read_pid = MagicMock(return_value=123)
        process = MagicMock()
        process.cmdline.return_value = ['python', '-m', 'openbrowser.daemon.server']
        pid_path = tmp_path / 'daemon.pid'
        socket_path = tmp_path / 'daemon.sock'
        pid_path.write_text('123')

        with (
            patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch('psutil.Process', return_value=process),
            patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='stale'),
            patch.object(
                client,
                '_remove_stale_daemon_files',
                new_callable=AsyncMock,
                return_value=DaemonResponse(
                    success=False,
                    output='',
                    error='foreign socket',
                    error_code='daemon_socket_foreign',
                ),
            ),
            patch.object(client, '_profile_dirs_for_daemon', return_value=['/tmp/profile']),
            patch.object(
                client,
                '_kill_recorded_browsers',
                new_callable=AsyncMock,
                return_value=(True, None),
            ) as cleanup,
        ):
            result = await client.force_stop(timeout=0.01)

        assert result.error_code == 'daemon_socket_foreign'
        assert result.error == 'foreign socket'
        cleanup.assert_awaited_once_with(['/tmp/profile'], 0.01, expected_owner_pid=123)

    @pytest.mark.asyncio
    async def test_force_stop_does_not_kill_browsers_when_another_owner_is_live(self, tmp_path):
        """An ambiguous cleanup state must not signal a replacement owner's browser."""
        client = DaemonClient()
        client.status = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client.stop = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client._read_pid = MagicMock(return_value=123)
        process = MagicMock()
        process.cmdline.return_value = ['python', '-m', 'openbrowser.daemon.server']
        pid_path = tmp_path / 'daemon.pid'
        pid_path.write_text('123')

        with (
            patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
            patch('openbrowser.daemon.client.get_socket_path', return_value=tmp_path / 'daemon.sock'),
            patch('psutil.Process', return_value=process),
            patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='stale'),
            patch.object(
                client,
                '_remove_stale_daemon_files',
                new_callable=AsyncMock,
                return_value=DaemonResponse(
                    success=False,
                    output='',
                    error='another owner',
                    error_code='daemon_busy',
                ),
            ),
            patch.object(client, '_profile_dirs_for_daemon', return_value=['/tmp/profile']),
            patch.object(
                client,
                '_kill_recorded_browsers',
                new_callable=AsyncMock,
                return_value=(True, None),
            ) as cleanup,
        ):
            result = await client.force_stop(timeout=0.01)

        assert result.error_code == 'daemon_busy'
        cleanup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kill_recorded_browser_skips_a_replacement_lease(self, tmp_path):
        """A lease rewritten by a new owner is never signalled."""
        client = DaemonClient()

        with (
            patch(
                'openbrowser.browser.watchdogs.profile_lease.ProfileLease.read_metadata',
                return_value={'owner_pid': 456, 'browser': {'pid': 321}},
            ),
            patch('psutil.Process') as process,
        ):
            result = await client._kill_recorded_browser(
                str(tmp_path),
                1.0,
                expected_owner_pid=123,
            )

        assert result == (True, None)
        process.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_cleanup_refuses_while_another_owner_holds_the_lock(self, tmp_path):
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
                patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
                patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            ):
                result = await client._remove_stale_daemon_files()
        finally:
            lock.release()

        assert result is not None
        assert result.error_code == 'daemon_busy'
        assert pid_path.read_text() == '4321'
        assert socket_path.read_text() == 'owner sentinel'

    @pytest.mark.asyncio
    async def test_stale_cleanup_refuses_a_socket_that_changed(self, tmp_path):
        from openbrowser.daemon.ownership import SocketEntryIdentity

        client = DaemonClient()
        pid_path = tmp_path / 'daemon.pid'
        socket_path = tmp_path / 'daemon.sock'
        pid_path.write_text('4321')
        socket_path.write_text('replacement')

        with (
            patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='stale'),
            patch.object(client, '_daemon_owner_for_socket', return_value=None),
            patch(
                'openbrowser.daemon.client.socket_entry_identity',
                side_effect=[None, SocketEntryIdentity(1, 1, True)],
            ),
        ):
            result = await client._remove_stale_daemon_files()

        assert result is not None
        assert result.error_code == 'daemon_socket_changed'
        assert socket_path.read_text() == 'replacement'
        assert pid_path.exists()

    @pytest.mark.asyncio
    async def test_stale_cleanup_preserves_a_foreign_path(self, tmp_path):
        client = DaemonClient()
        pid_path = tmp_path / 'daemon.pid'
        socket_path = tmp_path / 'daemon.sock'
        pid_path.write_text('4321')
        socket_path.write_text('not a socket')

        with (
            patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='stale'),
        ):
            result = await client._remove_stale_daemon_files()

        assert result is not None
        assert result.error_code == 'daemon_socket_foreign'
        assert socket_path.read_text() == 'not a socket'
        assert pid_path.exists()

    @pytest.mark.asyncio
    async def test_stale_cleanup_refuses_an_unverifiable_socket_state(self, tmp_path):
        client = DaemonClient()
        pid_path = tmp_path / 'daemon.pid'
        socket_path = tmp_path / 'daemon.sock'
        pid_path.write_text('4321')
        socket_path.write_text('unreadable socket')

        with (
            patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='unknown'),
        ):
            result = await client._remove_stale_daemon_files()

        assert result is not None
        assert result.error_code == 'daemon_socket_unverified'
        assert socket_path.read_text() == 'unreadable socket'
        assert pid_path.exists()

    @pytest.mark.asyncio
    async def test_force_stop_without_pid_does_not_hide_a_live_listener(self, tmp_path):
        client = DaemonClient()
        client.status = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client.stop = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client._read_pid = MagicMock(return_value=None)
        socket_path = tmp_path / 'daemon.sock'
        socket_path.write_text('')

        with (
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch('openbrowser.daemon.client.get_pid_path', return_value=tmp_path / 'daemon.pid'),
            patch.object(client, '_socket_accepts_connection', new_callable=AsyncMock, return_value=True),
        ):
            result = await client.force_stop(timeout=0.01)

        assert result.success is False
        assert result.error_code == 'daemon_stop_timeout'
        assert socket_path.exists()

    @pytest.mark.asyncio
    async def test_force_stop_with_stale_pid_does_not_hide_a_live_listener(self, tmp_path):
        client = DaemonClient()
        client.status = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client.stop = AsyncMock(return_value=DaemonResponse(success=False, output='', error='unresponsive'))
        client._read_pid = MagicMock(return_value=999)
        socket_path = tmp_path / 'daemon.sock'
        pid_path = tmp_path / 'daemon.pid'
        socket_path.write_text('')
        pid_path.write_text('999')

        with (
            patch('openbrowser.daemon.client.get_socket_path', return_value=socket_path),
            patch('openbrowser.daemon.client.get_pid_path', return_value=pid_path),
            patch('psutil.Process', side_effect=__import__('psutil').NoSuchProcess(999)),
            patch.object(client, '_socket_accepts_connection', new_callable=AsyncMock, return_value=True),
            patch.object(client, '_socket_state', new_callable=AsyncMock, return_value='live'),
        ):
            result = await client.force_stop(timeout=0.01)

        assert result.success is False
        assert result.error_code == 'daemon_stop_timeout'
        assert socket_path.exists()
        assert pid_path.exists()
