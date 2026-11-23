# tests/test_cli_c_help.py
"""Tests for `openbrowser-ai -c` (no argument) self-documenting behaviour."""

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the CLI entry-point as a subprocess."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, '-m', 'openbrowser.cli', *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=merged,
    )


@pytest.fixture()
def short_socket_env():
    root = '/tmp' if os.name != 'nt' else None
    with tempfile.TemporaryDirectory(prefix='ob_', dir=root) as tmp:
        env = {'OPENBROWSER_SOCKET': str(Path(tmp) / 'd.sock')}
        try:
            yield env
        finally:
            _run_cli('daemon', 'force-stop', env=env)


class TestCliCHelp:
    """Tests for the -c flag with no code argument."""

    def test_c_no_arg_exits_zero(self):
        """openbrowser-ai -c (no argument) should exit 0."""
        result = _run_cli('-c')
        assert result.returncode == 0, f'stderr: {result.stderr}'

    def test_c_no_arg_prints_to_stdout(self):
        """Output should go to stdout, not stderr."""
        result = _run_cli('-c')
        assert len(result.stdout.strip()) > 0, 'Expected output on stdout'

    def test_c_no_arg_contains_navigate(self):
        """Output should contain the navigate function."""
        result = _run_cli('-c')
        assert 'navigate' in result.stdout

    def test_c_no_arg_contains_click(self):
        """Output should contain the click function."""
        result = _run_cli('-c')
        assert 'click' in result.stdout

    def test_c_no_arg_contains_evaluate(self):
        """Output should contain the evaluate function."""
        result = _run_cli('-c')
        assert 'evaluate' in result.stdout

    def test_c_no_arg_verbose_when_daemon_not_running(self, short_socket_env):
        """When daemon is not running, output should be the verbose description with section headers."""
        env = short_socket_env
        _run_cli('daemon', 'stop', env=env)
        result = _run_cli('-c', env=env)
        assert result.returncode == 0
        # Verbose description has the '## Navigation' section header
        assert '## Navigation' in result.stdout

    def test_c_no_arg_compact_when_daemon_running(self, short_socket_env):
        """When daemon is already running, output should be the compact description."""
        env = short_socket_env
        try:
            started = _run_cli('daemon', 'start', env=env)
            assert started.returncode == 0, started.stderr
            result = _run_cli('-c', env=env)
            assert result.returncode == 0
            # Compact description has '## Core Functions' but NOT '## Navigation'
            assert '## Core Functions' in result.stdout
            assert 'get_status' in result.stdout
            assert 'openbrowser-ai daemon status' in result.stdout
            assert '`health` is not page readiness' in result.stdout
        finally:
            _run_cli('daemon', 'stop', env=env)

    def test_c_with_code_still_works(self):
        """openbrowser-ai -c 'print(1+1)' should execute code via a mock daemon."""
        from openbrowser.code_use.executor import CodeExecutor
        from openbrowser.daemon.server import DaemonServer

        # Set up a unique temp socket for isolation
        test_id = uuid.uuid4().hex[:8]
        tmp_dir = Path(f'/tmp/ob_test_{test_id}')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        sock = tmp_dir / 'd.sock'
        sock_str = str(sock)

        # Build a daemon with pre-initialized executor (no browser needed)
        server = DaemonServer(idle_timeout=60)
        executor = CodeExecutor()
        executor.set_namespace({
            'json': __import__('json'),
            'asyncio': __import__('asyncio'),
        })
        server._executor = executor

        loop = asyncio.new_event_loop()

        def run_daemon():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve_daemon())

        with (
            patch('openbrowser.daemon.server.get_socket_path', return_value=sock),
            patch('openbrowser.daemon.server.get_pid_path', return_value=tmp_dir / 'd.pid'),
        ):
            t = threading.Thread(target=run_daemon, daemon=True)
            t.start()

            # Wait for socket to appear
            for _ in range(200):
                if sock.exists():
                    break
                time.sleep(0.05)
            else:
                server._running = False
                server._stop_event.set()
                t.join(timeout=5)
                pytest.fail('Mock daemon socket never appeared')

            try:
                env = {**os.environ, 'OPENBROWSER_SOCKET': sock_str}
                result = subprocess.run(
                    [sys.executable, '-m', 'openbrowser.cli', '-c', 'print(1+1)'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                )
                assert result.returncode == 0, f'stderr: {result.stderr}'
                assert '2' in result.stdout
            finally:
                server._running = False
                server._stop_event.set()
                t.join(timeout=5)
                sock.unlink(missing_ok=True)
                (tmp_dir / 'd.pid').unlink(missing_ok=True)
                try:
                    tmp_dir.rmdir()
                except OSError:
                    pass
