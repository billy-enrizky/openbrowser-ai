# tests/test_cli_c_help.py
"""Tests for `openbrowser-ai -c` (no argument) self-documenting behaviour."""

import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the CLI entry-point as a subprocess."""
    return subprocess.run(
        [sys.executable, '-m', 'openbrowser.cli', *args],
        capture_output=True,
        text=True,
        timeout=30,
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
        """openbrowser-ai -c 'print(1+1)' should still execute code."""
        result = _run_cli('-c', 'print(1+1)')
        assert result.returncode == 0
        assert '2' in result.stdout
