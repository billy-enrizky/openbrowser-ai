"""Import safety and explicit CLI mode selection."""

import os
import re
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from openbrowser.cli import CliMode, main, select_cli_mode

ROOT = Path(__file__).parents[1]


def _run_python(source: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    return subprocess.run(
        [sys.executable, '-c', source],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.parametrize(
    ('args', 'expected'),
    [
        (['--mcp'], CliMode.MCP),
        (['-c', 'print(1)'], CliMode.EXECUTE_CODE),
        (['daemon', 'status'], CliMode.DAEMON),
        (['install'], CliMode.INSTALL),
        (['init'], CliMode.INIT),
        (['--template', 'default'], CliMode.TEMPLATE),
        (['-p', 'init'], None),
        (['--prompt', '--template'], None),
        (['--version'], None),
        ([], None),
    ],
)
def test_select_cli_mode_is_explicit(args, expected):
    assert select_cli_mode(args) is expected


def test_import_with_mcp_argv_is_side_effect_free():
    result = _run_python(
        "import sys, types\n"
        "fake = types.ModuleType('openbrowser.mcp.server')\n"
        "def fail(): raise RuntimeError('mcp fast path executed')\n"
        "fake.main = fail\n"
        "sys.modules['openbrowser.mcp.server'] = fake\n"
        "sys.argv = ['openbrowser-ai', '--mcp']\n"
        "import openbrowser.cli\n"
        "print('safe')\n"
    )
    assert result.returncode == 0, result.stderr
    assert 'safe' in result.stdout


def test_import_with_execute_argv_does_not_run_code():
    result = _run_python(
        "import sys, types\n"
        "fake = types.ModuleType('openbrowser.daemon.client')\n"
        "def fail(_code): raise RuntimeError('execute fast path executed')\n"
        "fake.execute_code_via_daemon = fail\n"
        "sys.modules['openbrowser.daemon.client'] = fake\n"
        "sys.argv = ['openbrowser-ai', '-c', 'print(999)']\n"
        "import openbrowser.cli\n"
        "print('safe')\n"
    )
    assert result.returncode == 0, result.stderr
    assert 'safe' in result.stdout
    assert '999' not in result.stdout


def test_import_with_init_argv_does_not_run_or_mutate():
    result = _run_python(
        "import sys, types\n"
        "fake = types.ModuleType('openbrowser.init_cmd')\n"
        "fake.INIT_TEMPLATES = {}\n"
        "def fail(): raise RuntimeError('init fast path executed')\n"
        "fake.main = fail\n"
        "sys.modules['openbrowser.init_cmd'] = fake\n"
        "sys.argv = ['openbrowser-ai', 'init']\n"
        "import openbrowser.cli\n"
        "print('argv=' + repr(sys.argv[1:]))\n"
        "print('safe')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "argv=['init']" in result.stdout
    assert 'safe' in result.stdout


@pytest.mark.parametrize('module', ['openbrowser.cli', 'openbrowser.cli_entry'])
def test_version_paths_work_for_module_and_console_wrapper(module):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    result = subprocess.run(
        [sys.executable, '-m', module, '--version'],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert re.search(r'\d+\.\d+', result.stdout), result.stdout


@pytest.mark.parametrize('module', ['openbrowser.cli', 'openbrowser.cli_entry'])
def test_help_paths_work_for_module_and_console_wrapper(module):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    result = subprocess.run(
        [sys.executable, '-m', module, '--help'],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'Usage:' in result.stdout or 'Usage:' in result.stderr


@pytest.mark.parametrize('module', ['openbrowser.cli', 'openbrowser.cli_entry'])
def test_template_generation_creates_the_bundled_default(module, tmp_path):
    output = tmp_path / 'generated.py'
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            module,
            '--template',
            'default',
            '--output',
            str(output),
            '--force',
        ],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert 'Agent' in output.read_text(encoding='utf-8')


def test_installed_console_script_maps_to_cli_entry():
    matches = [
        entry
        for entry in entry_points(group='console_scripts')
        if entry.name == 'openbrowser-ai'
    ]
    assert matches
    assert matches[0].value == 'openbrowser.cli_entry:cli'


def test_prompt_value_init_keeps_prompt_mode(monkeypatch):
    monkeypatch.setattr('openbrowser.cli.Agent', object())
    prompt = AsyncMock()
    with patch('openbrowser.cli.run_prompt_mode', prompt):
        result = CliRunner().invoke(main, ['-p', 'init'])
    assert result.exit_code == 0, result.output
    prompt.assert_awaited_once()
    assert prompt.await_args.args[0] == 'init'
