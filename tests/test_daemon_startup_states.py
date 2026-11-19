"""Behavior tests for daemon startup ownership and terminal states."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser.lifecycle import LifecycleMachine, LifecyclePhase
from openbrowser.browser.storage_state import StorageStatePhase, StorageStateSnapshot
from openbrowser.daemon.server import (
    DaemonServer,
    DaemonStartupPhase,
    PidInspection,
    PidOwnerState,
    SocketProbeState,
    _cleanup_pid,
    _daemon_pid_exists,
    _inspect_pid,
    _is_openbrowser_daemon_process,
    _probe_socket,
)


@pytest.fixture
def daemon_paths():
    shortest_directory = None if os.name == 'nt' else '/tmp'
    directory = Path(tempfile.mkdtemp(prefix='ob-daemon-states-', dir=shortest_directory))
    yield directory / 'daemon.pid', directory / 'daemon.sock', directory
    for path in sorted(directory.rglob('*'), reverse=True):
        path.unlink(missing_ok=True)
    directory.rmdir()


class _FakeServer:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _wait_for_phase(server: DaemonServer, phase: DaemonStartupPhase, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while server._startup_phase is not phase:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f'startup stayed in {server._startup_phase.value}, wanted {phase.value}')
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_startup_reaches_listening_then_stopped(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    server = DaemonServer(idle_timeout=5, exec_timeout=2)
    fake_server = _FakeServer()

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server._inspect_pid', return_value=PidInspection(state=PidOwnerState.ABSENT)),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
        patch('asyncio.start_unix_server', new_callable=AsyncMock, return_value=fake_server),
        patch('os.chmod'),
    ):
        run_task = asyncio.create_task(server.serve_daemon())
        await _wait_for_phase(server, DaemonStartupPhase.LISTENING)

        assert server._startup_history == [
            DaemonStartupPhase.CHECKING_OWNERSHIP,
            DaemonStartupPhase.ACQUIRING_OWNERSHIP,
            DaemonStartupPhase.BINDING_SOCKET,
            DaemonStartupPhase.LISTENING,
        ]
        assert pid_path.read_text() == str(os.getpid())
        server._profile_summary = {
            'user_data_dir': '/tmp/profile',
            'headless': True,
            'storage_state': None,
        }
        status = server._status_payload()
        assert status['startup']['phase'] == 'listening'
        assert status['startup']['owner_state'] == 'active'
        assert status['storage_state']['phase'] == 'not_configured'

        server._stop_event.set()
        await asyncio.wait_for(run_task, timeout=2)

    assert server._startup_phase is DaemonStartupPhase.STOPPED
    assert server._lifecycle.phase is LifecyclePhase.STOPPED
    assert not pid_path.exists()


@pytest.mark.asyncio
async def test_owner_lock_refuses_a_second_starter(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    first = DaemonServer(idle_timeout=5, exec_timeout=2)
    second = DaemonServer(idle_timeout=5, exec_timeout=2)

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server._inspect_pid', return_value=PidInspection(state=PidOwnerState.ABSENT)),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
        patch('asyncio.start_unix_server', new_callable=AsyncMock, return_value=_FakeServer()),
        patch('os.chmod'),
    ):
        first_task = asyncio.create_task(first.serve_daemon())
        await _wait_for_phase(first, DaemonStartupPhase.LISTENING)
        first_pid = pid_path.read_text()

        await asyncio.wait_for(second.serve_daemon(), timeout=2)

        assert second._startup_phase is DaemonStartupPhase.FAILED
        assert 'startup lock' in (second._startup_detail or '')
        assert second._startup_owner_pid == int(first_pid)
        assert pid_path.read_text() == first_pid

        first._stop_event.set()
        await asyncio.wait_for(first_task, timeout=2)

        third = DaemonServer(idle_timeout=5, exec_timeout=2)
        third_task = asyncio.create_task(third.serve_daemon())
        await _wait_for_phase(third, DaemonStartupPhase.LISTENING)
        third._stop_event.set()
        await asyncio.wait_for(third_task, timeout=2)


@pytest.mark.asyncio
async def test_stale_pid_is_removed_and_reported(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    pid_path.write_text('99999999')
    server = DaemonServer(idle_timeout=5, exec_timeout=2)

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('os.kill', side_effect=ProcessLookupError),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
        patch('asyncio.start_unix_server', new_callable=AsyncMock, return_value=_FakeServer()),
        patch('os.chmod'),
    ):
        run_task = asyncio.create_task(server.serve_daemon())
        await _wait_for_phase(server, DaemonStartupPhase.LISTENING)

        assert DaemonStartupPhase.CLEANING_STALE_LOCK in server._startup_history
        assert server._startup_stale_pid_removed is True
        server._stop_event.set()
        await asyncio.wait_for(run_task, timeout=2)


def test_live_daemon_with_unresponsive_socket_is_not_stolen(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    pid_path.write_text(str(os.getpid()))

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
        patch('openbrowser.daemon.server._probe_socket', return_value=SocketProbeState.ABSENT),
        patch('openbrowser.daemon.server._is_openbrowser_daemon_process', return_value=True),
    ):
        inspection = _inspect_pid()

    assert inspection.state is PidOwnerState.UNRESPONSIVE
    assert inspection.pid == os.getpid()
    assert pid_path.exists()


def test_reused_non_daemon_pid_is_cleaned_without_touching_socket(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    pid_path.write_text(str(os.getpid()))
    socket_path.write_text('not a socket')

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
        patch('openbrowser.daemon.server._probe_socket', return_value=SocketProbeState.ABSENT),
        patch('openbrowser.daemon.server._is_openbrowser_daemon_process', return_value=False),
    ):
        inspection = _inspect_pid()

    assert inspection.state is PidOwnerState.STALE
    assert inspection.stale_pid_removed is True
    assert not pid_path.exists()
    assert socket_path.exists()


def test_unknown_socket_state_is_not_treated_as_stale(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    pid_path.write_text(str(os.getpid()))
    socket_path.write_text('unreadable socket identity')

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
        patch('openbrowser.daemon.server._probe_socket', return_value=SocketProbeState.UNKNOWN),
        patch('openbrowser.daemon.server._is_openbrowser_daemon_process', return_value=False),
    ):
        inspection = _inspect_pid()

    assert inspection.state is PidOwnerState.UNRESPONSIVE
    assert pid_path.exists()
    assert socket_path.exists()


def test_incomplete_pid_file_is_not_removed_while_it_may_be_written(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    pid_path.write_text('')

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
    ):
        inspection = _inspect_pid()
        assert inspection.state is PidOwnerState.UNRESPONSIVE
        assert pid_path.exists()

        old = time.time() - 5
        os.utime(pid_path, (old, old))
        inspection = _inspect_pid()

    assert inspection.state is PidOwnerState.STALE
    assert not pid_path.exists()


def test_daemon_identity_includes_the_owned_socket(daemon_paths):
    _, socket_path, _ = daemon_paths
    process = MagicMock()
    process.cmdline.return_value = ['python', '-m', 'openbrowser.daemon.server']

    with (
        patch('psutil.Process', return_value=process),
        patch('openbrowser.daemon.diagnostics.is_daemon_cmdline', return_value=True),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
    ):
        process.environ.return_value = {'OPENBROWSER_SOCKET': str(socket_path)}
        assert _is_openbrowser_daemon_process(123) is True

        process.environ.return_value = {'OPENBROWSER_SOCKET': str(socket_path) + '.other'}
        assert _is_openbrowser_daemon_process(123) is False


def test_liveness_probe_never_sends_a_signal():
    with (
        patch('psutil.pid_exists', return_value=True),
        patch('os.kill') as kill,
    ):
        assert _daemon_pid_exists(os.getpid()) is True

    kill.assert_not_called()


@pytest.mark.asyncio
async def test_unresponsive_live_owner_fails_without_deleting_files(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    pid_path.write_text('4242')
    socket_path.write_text('keep me')
    server = DaemonServer(idle_timeout=5, exec_timeout=2)

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch(
            'openbrowser.daemon.server._inspect_pid',
            return_value=PidInspection(
                state=PidOwnerState.UNRESPONSIVE,
                pid=4242,
                detail='owner is not listening',
            ),
        ),
    ):
        await server.serve_daemon()

    assert server._startup_phase is DaemonStartupPhase.FAILED
    assert server._startup_owner_pid == 4242
    assert server._lifecycle.phase is LifecyclePhase.FAILED
    assert pid_path.read_text() == '4242'
    assert socket_path.read_text() == 'keep me'


@pytest.mark.asyncio
async def test_pid_acquisition_race_does_not_bind_or_delete_owner_files(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    pid_path.write_text('4242')
    socket_path.write_text('keep me')
    server = DaemonServer(idle_timeout=5, exec_timeout=2)
    start_server = AsyncMock()

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server._inspect_pid', return_value=PidInspection(state=PidOwnerState.ABSENT)),
        patch('openbrowser.daemon.server._acquire_pid_file', return_value=False),
        patch('openbrowser.daemon.server._peek_pid', return_value=4242),
        patch('asyncio.start_unix_server', start_server),
    ):
        await server.serve_daemon()

    start_server.assert_not_awaited()
    assert server._startup_phase is DaemonStartupPhase.FAILED
    assert server._startup_owner_pid == 4242
    assert pid_path.read_text() == '4242'
    assert socket_path.read_text() == 'keep me'


@pytest.mark.skipif(os.name == 'nt', reason='Unix socket ownership semantics')
@pytest.mark.asyncio
async def test_busy_socket_without_pid_owner_is_refused_and_preserved(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    server = DaemonServer(idle_timeout=5, exec_timeout=2)
    start_server = AsyncMock()
    try:
        with (
            patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
            patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
            patch('openbrowser.daemon.server._inspect_pid', return_value=PidInspection(state=PidOwnerState.ABSENT)),
            patch('openbrowser.daemon.server.IS_WINDOWS', False),
            patch('asyncio.start_unix_server', start_server),
        ):
            await server.serve_daemon()
    finally:
        listener.close()

    start_server.assert_not_awaited()
    assert server._startup_phase is DaemonStartupPhase.FAILED
    assert not pid_path.exists()
    assert socket_path.exists()
    socket_path.unlink()


@pytest.mark.asyncio
async def test_bind_failure_remains_failed_after_cleanup(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    server = DaemonServer(idle_timeout=5, exec_timeout=2)

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server._inspect_pid', return_value=PidInspection(state=PidOwnerState.ABSENT)),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
        patch('asyncio.start_unix_server', new_callable=AsyncMock, side_effect=OSError('bind failed')),
        pytest.raises(OSError, match='bind failed'),
    ):
        await server.serve_daemon()

    assert server._startup_phase is DaemonStartupPhase.FAILED
    assert server._lifecycle.phase is LifecyclePhase.FAILED
    assert not pid_path.exists()


@pytest.mark.asyncio
async def test_chmod_failure_removes_the_bound_socket(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    server = DaemonServer(idle_timeout=5, exec_timeout=2)
    cleanup = MagicMock()

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server._inspect_pid', return_value=PidInspection(state=PidOwnerState.ABSENT)),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
        patch('openbrowser.daemon.server._cleanup_pid', cleanup),
        patch('asyncio.start_unix_server', new_callable=AsyncMock, return_value=_FakeServer()),
        patch('os.chmod', side_effect=OSError('chmod failed')),
        pytest.raises(OSError, match='chmod failed'),
    ):
        await server.serve_daemon()

    cleanup.assert_called_once()
    assert cleanup.call_args.kwargs['remove_socket'] is True
    assert server._startup_phase is DaemonStartupPhase.FAILED


def test_cleanup_does_not_remove_another_owners_files(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    pid_path.write_text('456')
    socket_path.write_text('keep me')

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
    ):
        _cleanup_pid(expected_pid=123)
        assert pid_path.read_text() == '456'
        assert socket_path.read_text() == 'keep me'

        _cleanup_pid(expected_pid=456)
        assert not pid_path.exists()
        assert not socket_path.exists()


def test_regular_file_is_not_mistaken_for_a_live_socket(daemon_paths):
    _, socket_path, _ = daemon_paths
    socket_path.write_text('ordinary file')

    with patch('openbrowser.daemon.server.IS_WINDOWS', False):
        assert _probe_socket(socket_path) is SocketProbeState.FOREIGN


@pytest.mark.asyncio
async def test_foreign_socket_path_is_refused_and_preserved(daemon_paths):
    pid_path, socket_path, _ = daemon_paths
    socket_path.write_text('ordinary file')
    server = DaemonServer(idle_timeout=5, exec_timeout=2)

    with (
        patch('openbrowser.daemon.server.get_pid_path', return_value=pid_path),
        patch('openbrowser.daemon.server.get_socket_path', return_value=socket_path),
        patch('openbrowser.daemon.server._inspect_pid', return_value=PidInspection(state=PidOwnerState.ABSENT)),
        patch('openbrowser.daemon.server.IS_WINDOWS', False),
    ):
        await server.serve_daemon()

    assert server._startup_phase is DaemonStartupPhase.FAILED
    assert not pid_path.exists()
    assert socket_path.read_text() == 'ordinary file'


def test_status_reads_storage_state_from_the_watchdog():
    server = DaemonServer(idle_timeout=5, exec_timeout=2)
    server._profile_summary = {
        'user_data_dir': '/tmp/profile',
        'headless': True,
        'storage_state': '/tmp/state.json',
    }
    watchdog = MagicMock()
    watchdog.storage_state_snapshot.return_value = StorageStateSnapshot(
        phase=StorageStatePhase.LOADED,
        path='/tmp/state.json',
        bytes=45_000_000,
        load_ms=123.5,
        cookies_count=2,
        origins_count=3,
        script_count=4,
    )
    server._session = MagicMock()
    server._session.lifecycle_snapshot.return_value = {'phase': 'ready'}
    server._session._storage_state_watchdog = watchdog

    status = server._status_payload()

    assert status['storage_state']['phase'] == 'loaded'
    assert status['storage_state']['bytes'] == 45_000_000
    assert status['storage_state']['script_count'] == 4


def test_status_reports_configured_storage_state_before_browser_start(tmp_path):
    state_path = tmp_path / 'state.json'
    state_path.write_text('{"cookies": [], "origins": []}')
    server = DaemonServer(idle_timeout=5, exec_timeout=2)
    profile = MagicMock()
    profile.user_data_dir = tmp_path / 'profile'
    profile.headless = True
    profile.storage_state = str(state_path)

    with (
        patch('openbrowser.browser.BrowserProfile', return_value=profile),
        patch('openbrowser.config.load_openbrowser_config', return_value={}),
        patch(
            'openbrowser.config.get_default_profile',
            return_value={'storage_state': str(state_path)},
        ),
        patch(
            'openbrowser.config.apply_managed_browser_profile_defaults',
            side_effect=lambda profile_data, default_user_data_dir=None: dict(profile_data),
        ),
    ):
        status = server._status_payload()

    assert server._session is None
    assert status['storage_state']['phase'] == 'pending'
    assert status['storage_state']['path'] == str(state_path)
    assert status['storage_state']['bytes'] == state_path.stat().st_size


@pytest.mark.asyncio
async def test_main_exits_nonzero_when_startup_fails():
    from openbrowser.daemon.server import _main

    instance = MagicMock()
    instance.serve_daemon = AsyncMock()
    instance._startup_phase = DaemonStartupPhase.FAILED
    instance._lifecycle = LifecycleMachine('daemon', phase=LifecyclePhase.FAILED)

    with (
        patch('openbrowser.daemon.server.DaemonServer', return_value=instance),
        pytest.raises(SystemExit) as exc_info,
    ):
        await _main()

    assert exc_info.value.code == 1


@pytest.mark.skipif(os.name == 'nt', reason='Unix socket subprocess lifecycle')
@pytest.mark.asyncio
async def test_real_daemon_process_reports_status_and_cleans_up(daemon_paths, monkeypatch):
    pid_path, socket_path, _ = daemon_paths
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment['OPENBROWSER_SOCKET'] = str(socket_path)
    environment['PYTHONPATH'] = str(repository_root / 'src')
    monkeypatch.setenv('OPENBROWSER_SOCKET', str(socket_path))

    process = subprocess.Popen(
        [sys.executable, '-m', 'openbrowser.daemon.server'],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        from openbrowser.daemon.client import DaemonClient

        client = DaemonClient()
        deadline = asyncio.get_running_loop().time() + 10
        status = None
        while asyncio.get_running_loop().time() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ''
                pytest.fail(f'daemon subprocess exited with {process.returncode}: {stderr}')
            status = await client.status()
            if status.success:
                break
            await asyncio.sleep(0.1)

        assert status is not None and status.success
        payload = json.loads(status.output)
        assert payload['startup']['phase'] == 'listening'
        assert payload['startup']['owner_state'] == 'active'
        assert payload['storage_state']['phase'] == 'not_configured'
        assert pid_path.read_text() == str(process.pid)

        stopped = await client.stop()
        assert stopped.success
        return_code = await asyncio.to_thread(process.wait, timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, timeout=5)

    assert return_code == 0
    assert not pid_path.exists()
    assert not socket_path.exists()
