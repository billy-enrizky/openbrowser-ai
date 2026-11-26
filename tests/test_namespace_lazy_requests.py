"""Tests for the lazy requests binding in the code-use namespace."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.code_use import namespace

ROOT = Path(__file__).resolve().parents[1]


def _mock_tools() -> MagicMock:
    tools = MagicMock()
    tools.registry.registry.actions = {}
    tools.registry.exclude_actions = []
    return tools


def _mock_browser() -> MagicMock:
    browser = MagicMock()
    browser.browser_profile.downloads_path = None
    return browser


def test_namespace_requests_binding_is_lazy_and_shared() -> None:
    assert isinstance(namespace.requests, namespace._LazyOptionalBinding)
    assert namespace.requests._module_name == "requests"

    created = namespace.create_namespace(_mock_browser(), tools=_mock_tools())
    assert created["requests"] is namespace.requests


def test_namespace_import_does_not_import_requests(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBROWSER_SETUP_LOGGING"] = "false"
    env["OPENBROWSER_DISABLE_TELEMETRY"] = "true"

    script = """
import sys
blocked_events = []
def guard(event, arguments):
    if event in {
        'socket.__new__', 'socket.connect', 'socket.bind', 'socket.listen',
        'socket.getaddrinfo', 'subprocess.Popen',
    }:
        blocked_events.append(event)
        raise AssertionError('namespace import attempted native I/O: ' + event)
sys.addaudithook(guard)
import openbrowser.code_use.namespace as namespace
assert 'requests' not in sys.modules, 'requests imported during namespace import'
assert not blocked_events, blocked_events
getattr(namespace.requests, 'get')
assert 'requests' in sys.modules, 'requests was not resolved on first use'
sys.stdout.write('lazy-ok\\n')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy-ok"


def test_lazy_binding_resolves_once_and_preserves_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    requested_error = ConnectionError("network down")
    fake_module = SimpleNamespace(get=MagicMock(side_effect=requested_error))

    def fake_import(module_name: str):
        calls.append(module_name)
        return fake_module

    monkeypatch.setattr(namespace.importlib, "import_module", fake_import)
    binding = namespace._LazyModuleBinding("requests")

    with pytest.raises(ConnectionError, match="network down"):
        binding.get("https://example.com")
    with pytest.raises(ConnectionError, match="network down"):
        binding.get("https://example.com")

    assert calls == ["requests"]
    assert binding._resolved is fake_module


def test_lazy_binding_attribute_patch_uses_resolved_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(get=lambda: "original")
    monkeypatch.setattr(namespace.importlib, "import_module", lambda _name: fake_module)
    binding = namespace._LazyModuleBinding("requests")

    with patch.object(binding, "get", return_value="patched"):
        assert binding.get() == "patched"

    assert binding.get() == "original"


def test_lazy_binding_attribute_writes_and_deletes_reach_the_real_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real_module = SimpleNamespace()

    def load(name):
        calls.append(name)
        return real_module

    monkeypatch.setattr(namespace.importlib, "import_module", load)
    binding = namespace._LazyModuleBinding("requests")
    binding.temporary = "value"
    assert real_module.temporary == "value"
    del binding.temporary
    assert not hasattr(real_module, "temporary")
    assert calls == ["requests"]


def test_lazy_binding_can_resolve_after_an_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real_module = SimpleNamespace(get=lambda: "resolved")

    def load(name):
        calls.append(name)
        if len(calls) == 1:
            raise ImportError("requests unavailable")
        return real_module

    monkeypatch.setattr(namespace.importlib, "import_module", load)
    binding = namespace._LazyModuleBinding("requests")
    with pytest.raises(ImportError, match="requests unavailable"):
        binding.get()
    assert binding._resolved is None
    assert binding.get() == "resolved"
    assert calls == ["requests", "requests"]


@pytest.mark.asyncio
async def test_download_fallback_uses_lazy_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    browser = _mock_browser()
    browser.browser_profile.downloads_path = tmp_path
    browser.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("browser fetch failed"))
    tools = _mock_tools()
    created = namespace.create_namespace(browser, tools=tools)

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.iter_content = MagicMock(return_value=[b"fallback data"])
    fake_module = SimpleNamespace(get=MagicMock(return_value=response))
    monkeypatch.setattr(namespace.requests, "_resolved", fake_module)

    path = await created["download_file"]("https://example.com/file.pdf")

    assert await asyncio.to_thread(Path(path).read_bytes) == b"fallback data"
    fake_module.get.assert_called_once()
