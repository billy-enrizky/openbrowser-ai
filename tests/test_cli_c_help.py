# tests/test_cli_c_help.py
"""Tests for `openbrowser-ai -c` (no argument) self-documenting behaviour."""

import asyncio
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the CLI entry-point as a subprocess."""
    return subprocess.run(
        [sys.executable, '-m', 'openbrowser.cli', *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


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

    def test_c_no_arg_verbose_when_daemon_not_running(self):
        """When daemon is not running, output should be the verbose description with section headers."""
        # Stop daemon first to ensure cold-start path
        subprocess.run(
            [sys.executable, '-m', 'openbrowser.cli', 'daemon', 'stop'],
            capture_output=True,
            timeout=10,
        )
        result = _run_cli('-c')
        assert result.returncode == 0
        # Verbose description has the '## Navigation' section header
        assert '## Navigation' in result.stdout

    def test_c_no_arg_compact_when_daemon_running(self):
        """When daemon is already running, output should be the compact description."""
        # Ensure daemon is started
        subprocess.run(
            [sys.executable, '-m', 'openbrowser.cli', 'daemon', 'start'],
            capture_output=True,
            timeout=20,
        )
        result = _run_cli('-c')
        assert result.returncode == 0
        # Compact description has '## Core Functions' but NOT '## Navigation'
        assert '## Core Functions' in result.stdout
        # Clean up: stop the daemon
        subprocess.run(
            [sys.executable, '-m', 'openbrowser.cli', 'daemon', 'stop'],
            capture_output=True,
            timeout=10,
        )

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
            os.environ['OPENBROWSER_SOCKET'] = sock_str
            try:
                loop.run_until_complete(server.run())
            finally:
                os.environ.pop('OPENBROWSER_SOCKET', None)

        t = threading.Thread(target=run_daemon, daemon=True)
        t.start()

        # Wait for socket to appear
        for _ in range(50):
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
