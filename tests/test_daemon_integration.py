# tests/test_daemon_integration.py
"""Integration tests for daemon server + client over a real Unix socket.

These tests spin up a DaemonServer on a temp socket, send requests via
DaemonClient, and verify end-to-end behavior. The browser/namespace is
mocked so no real Chrome is needed.
"""

import asyncio
import json
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def daemon_env():
    """Provide a short temp socket path to stay under macOS 104-byte limit."""
    # Use /tmp directly to keep path short (macOS tmp_path is too long for Unix sockets)
    test_id = uuid.uuid4().hex
    tmp_dir = Path(f'/tmp/ob_test_{test_id}')
    tmp_dir.mkdir(parents=True, exist_ok=False)
    sock = tmp_dir / 'd.sock'
    pid_path = tmp_dir / 'd.pid'
    assert not sock.exists()
    assert not pid_path.exists()

    with (
        patch.dict(os.environ, {'OPENBROWSER_SOCKET': str(sock)}),
        patch('openbrowser.daemon.server.get_socket_path', return_value=sock),
        patch('openbrowser.daemon.server.get_pid_path', return_value=tmp_dir / 'd.pid'),
        patch('openbrowser.daemon.client.get_socket_path', return_value=sock),
        patch('openbrowser.daemon.client.DAEMON_DIR', tmp_dir),
    ):
        yield sock

    # Cleanup -- PID path is derived from socket path (d.sock -> d.pid)
    sock.unlink(missing_ok=True)
    (tmp_dir / 'd.pid').unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


@pytest.fixture
def mock_namespace():
    """A minimal namespace dict that mimics create_namespace() output."""
    session = MagicMock()
    session.start = AsyncMock()
    session.kill = AsyncMock()
    return {
        'browser': session,
        'file_system': None,
        'json': __import__('json'),
        'asyncio': __import__('asyncio'),
    }


async def _start_daemon_with_mock(daemon_env, mock_namespace, idle_timeout=600):
    """Start a DaemonServer that uses a pre-built mock namespace."""
    from openbrowser.code_use.executor import CodeExecutor
    from openbrowser.daemon.server import DaemonServer

    server = DaemonServer(idle_timeout=idle_timeout)
    executor = CodeExecutor()
    executor.set_namespace(mock_namespace)
    server._executor = executor

    pid_path = daemon_env.parent / f'{daemon_env.stem}.pid'
    task = asyncio.create_task(server.serve_daemon())
    deadline = asyncio.get_running_loop().time() + 10
    while not daemon_env.exists():
        if task.done():
            await task
            pytest.fail(f'Daemon task exited before creating its socket: {server._lifecycle.snapshot().to_dict()}')
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            state = server._lifecycle.snapshot().to_dict()
            pid_file_exists = pid_path.exists()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            pytest.fail(f'Daemon socket did not appear within 10 seconds: {state}, pid_file_exists={pid_file_exists}')
        await asyncio.sleep(min(0.05, remaining))
    if task.done():
        await task
        pytest.fail('Daemon task exited immediately after creating its socket')
    return server, task


async def test_start_daemon_helper_surfaces_server_startup_failure(daemon_env, mock_namespace, monkeypatch):
    from openbrowser.daemon.server import DaemonServer

    async def fail_to_start(self):
        raise OSError('synthetic socket bind failure')

    monkeypatch.setattr(DaemonServer, 'serve_daemon', fail_to_start)
    with pytest.raises(OSError, match='synthetic socket bind failure'):
        await _start_daemon_with_mock(daemon_env, mock_namespace)


async def _cleanup(server, task):
    """Gracefully stop a daemon server and cancel its task."""
    server._running = False
    server._stop_event.set()
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


class TestDaemonIntegration:
    """End-to-end tests for daemon server + client communication."""

    @pytest.mark.asyncio
    async def test_execute_simple_code(self, daemon_env, mock_namespace):
        """Execute a simple print statement through the daemon."""
        server, task = await _start_daemon_with_mock(daemon_env, mock_namespace)

        try:
            from openbrowser.daemon.client import DaemonClient
            client = DaemonClient()
            result = await client.execute('print("hello from daemon")')
            assert result.success is True
            assert 'hello from daemon' in result.output
        finally:
            await _cleanup(server, task)

    @pytest.mark.asyncio
    async def test_variable_persistence(self, daemon_env, mock_namespace):
        """Variables set in one call should be available in the next."""
        server, task = await _start_daemon_with_mock(daemon_env, mock_namespace)

        try:
            from openbrowser.daemon.client import DaemonClient
            client = DaemonClient()

            r1 = await client.execute('my_var = "persisted_value"')
            assert r1.success is True

            r2 = await client.execute('print(my_var)')
            assert r2.success is True
            assert 'persisted_value' in r2.output
        finally:
            await _cleanup(server, task)

    @pytest.mark.asyncio
    async def test_daemon_status(self, daemon_env, mock_namespace):
        """Status action returns daemon info as JSON."""
        server, task = await _start_daemon_with_mock(daemon_env, mock_namespace)

        try:
            from openbrowser.daemon.client import DaemonClient
            client = DaemonClient()
            result = await client.status()
            assert result.success is True

            info = json.loads(result.output)
            assert 'pid' in info
            assert info['initialized'] is True
        finally:
            await _cleanup(server, task)

    @pytest.mark.asyncio
    async def test_error_returns_traceback(self, daemon_env, mock_namespace):
        """Errors should include traceback in the response."""
        server, task = await _start_daemon_with_mock(daemon_env, mock_namespace)

        try:
            from openbrowser.daemon.client import DaemonClient
            client = DaemonClient()
            result = await client.execute('raise ValueError("test boom")')
            assert result.success is False
            assert 'ValueError' in result.output
            assert 'test boom' in result.output
        finally:
            await _cleanup(server, task)

    @pytest.mark.asyncio
    async def test_daemon_stop(self, daemon_env, mock_namespace):
        """Stop action should shut down the daemon."""
        server, task = await _start_daemon_with_mock(daemon_env, mock_namespace)

        try:
            from openbrowser.daemon.client import DaemonClient
            client = DaemonClient()
            result = await client.stop()
            assert result.success is True

            await asyncio.sleep(0.5)
            assert server._running is False
        finally:
            await _cleanup(server, task)

    @pytest.mark.asyncio
    async def test_daemon_reset(self, daemon_env, mock_namespace):
        """Reset action should clear the executor."""
        server, task = await _start_daemon_with_mock(daemon_env, mock_namespace)

        try:
            from openbrowser.daemon.client import DaemonClient
            client = DaemonClient()
            result = await client.reset()
            assert result.success is True
            assert 'reset' in result.output.lower()
            assert server._executor is None
        finally:
            await _cleanup(server, task)

    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(self, daemon_env, mock_namespace):
        """Multiple requests should all succeed on sequential connections."""
        server, task = await _start_daemon_with_mock(daemon_env, mock_namespace)

        try:
            from openbrowser.daemon.client import DaemonClient
            client = DaemonClient()

            for i in range(5):
                result = await client.execute(f'print("iteration {i}")')
                assert result.success is True
                assert f'iteration {i}' in result.output
        finally:
            await _cleanup(server, task)

    @pytest.mark.asyncio
    async def test_status_when_no_daemon(self, daemon_env):
        """Status should report error when no daemon is running."""
        from openbrowser.daemon.client import DaemonClient
        client = DaemonClient()
        result = await client.status()
        assert result.success is False
        assert result.error is not None
