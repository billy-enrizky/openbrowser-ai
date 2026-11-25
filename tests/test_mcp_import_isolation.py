"""MCP startup must not pay for the code-use stack before it is needed."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _run(source: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    env.pop("OPENBROWSER_SETUP_LOGGING", None)
    return subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_mcp_server_import_defers_the_code_use_namespace():
    result = _run(
        "import sys\n"
        "sys.argv = ['openbrowser-ai', '--mcp']\n"
        "import openbrowser.mcp.server as server\n"
        "print('namespace', 'openbrowser.code_use.namespace' in sys.modules)\n"
        "print('has_create', hasattr(server, 'create_namespace'))\n"
    )
    assert result.returncode == 0, result.stderr
    assert "namespace False" in result.stdout, result.stdout


def test_mcp_server_still_exposes_the_server_class():
    result = _run(
        "import sys\n"
        "sys.argv = ['openbrowser-ai', '--mcp']\n"
        "import openbrowser.mcp.server as server\n"
        "print('has_server', hasattr(server, 'OpenBrowserServer'))\n"
    )
    assert result.returncode == 0, result.stderr
    assert "has_server True" in result.stdout


def test_lean_import_does_not_load_the_schema_validator():
    result = _run(
        "import sys\n"
        "import openbrowser.mcp.lean\n"
        "print('schema', 'openbrowser.mcp.schema' in sys.modules)\n"
        "print('jsonschema', 'jsonschema' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert "schema False" in result.stdout, result.stdout
    assert "jsonschema False" in result.stdout, result.stdout


def test_stateless_validation_loads_the_schema_on_demand():
    result = _run(
        "import sys\n"
        "from openbrowser.mcp.stateless import validate_stateless_arguments\n"
        "print('before', 'openbrowser.mcp.schema' in sys.modules)\n"
        "validate_stateless_arguments('scrape_web', {'url': 'https://example.com/'})\n"
        "print('after', 'openbrowser.mcp.schema' in sys.modules)\n"
        "print('jsonschema', 'jsonschema' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert "before False" in result.stdout, result.stdout
    assert "after True" in result.stdout, result.stdout
    assert "jsonschema False" in result.stdout, result.stdout
