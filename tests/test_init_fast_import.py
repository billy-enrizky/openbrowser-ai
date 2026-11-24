"""Package import must stay cheap for explicit CLI fast paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _run(source: str, *, argv: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    env.pop("OPENBROWSER_SETUP_LOGGING", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_fast_cli_argv_skips_logging_configuration():
    result = _run(
        "import sys\n"
        "sys.argv = ['openbrowser-ai', '-c', 'print(1)']\n"
        "import openbrowser\n"
        "print('config', 'openbrowser.config' in sys.modules)\n"
        "print('asyncio', 'asyncio' in sys.modules)\n"
        "print('logger', openbrowser.logger.name)\n",
        argv=[],
    )
    assert result.returncode == 0, result.stderr
    assert "config False" in result.stdout
    assert "asyncio False" in result.stdout
    assert "logger openbrowser" in result.stdout


def test_ordinary_script_argv_keeps_logging_configuration():
    result = _run(
        "import sys\n"
        "sys.argv = ['my-script.py', '-c', 'config.json']\n"
        "import openbrowser\n"
        "print('has_setup', hasattr(openbrowser.logger, 'handlers'))\n",
        argv=[],
    )
    assert result.returncode == 0, result.stderr
    assert "has_setup True" in result.stdout


def test_should_setup_logging_matrix():
    from openbrowser import _should_setup_logging

    assert _should_setup_logging(["openbrowser-ai", "-c", "print(1)"], {}) is False
    assert _should_setup_logging(["openbrowser-ai", "--mcp"], {}) is False
    assert _should_setup_logging(["openbrowser-ai", "daemon", "status"], {}) is False
    assert _should_setup_logging(["openbrowser-ai", "-p", "-c"], {}) is True
    assert _should_setup_logging(["openbrowser-ai", "scrape", "https://example.com"], {}) is True
    assert _should_setup_logging(["my-script.py", "-c", "config.json"], {}) is True
    assert _should_setup_logging(["openbrowser-ai", "-c", "print(1)"], {"OPENBROWSER_SETUP_LOGGING": "false"}) is False
