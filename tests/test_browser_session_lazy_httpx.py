"""Local browser model import keeps HTTP client setup deferred to its consumer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_browser_session_import_defers_httpx():
    root = Path(__file__).resolve().parents[1]
    program = r"""
import importlib.abc
import socket
import sys
sys.path.insert(0, sys.argv[1])
class RejectHttpx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'httpx' or fullname.startswith('httpx.'):
            raise AssertionError('BrowserSession eagerly imported HTTPX')
sys.meta_path.insert(0, RejectHttpx())
def guard(event, arguments):
    if event in {'socket.connect', 'socket.bind', 'subprocess.Popen'}:
        raise AssertionError('Import attempted native I/O')
sys.addaudithook(guard)
import openbrowser.browser.session
assert 'httpx' not in sys.modules
"""
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBROWSER_SETUP_LOGGING": "false",
        "OPENBROWSER_DISABLE_TELEMETRY": "true",
        "ANONYMIZED_TELEMETRY": "false",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(root / "src")], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
