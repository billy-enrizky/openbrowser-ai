"""Cold MCP startup overlaps namespace work with Chrome after shared imports."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openbrowser.mcp import server as module


@pytest.mark.asyncio
async def test_namespace_prewarm_follows_shared_browser_import(monkeypatch, isolated_config_path):
    events = []
    server = module.OpenBrowserServer(install_sdk=False)
    session = MagicMock()
    session.start = AsyncMock(side_effect=lambda: events.append("browser-start"))
    monkeypatch.setattr(server, "_build_browser_profile", lambda: events.append("profile") or MagicMock())
    monkeypatch.setattr(module, "_ensure_browser_stack", lambda: events.append("shared-stack"))
    monkeypatch.setattr(module, "BrowserSession", lambda **kwargs: session)
    monkeypatch.setattr(module, "CodeAgentTools", MagicMock)
    monkeypatch.setattr(module, "start_namespace_import", lambda: events.append("namespace-start") or object())
    monkeypatch.setattr(module, "join_namespace_import", lambda thread: events.append("namespace-join"))
    monkeypatch.setattr(module, "create_namespace", lambda **kwargs: {})
    monkeypatch.setattr(server, "_create_owner_file_system", lambda: None)
    await server._ensure_namespace()
    assert events.index("shared-stack") < events.index("namespace-start") < events.index("browser-start")
    assert events.index("browser-start") < events.index("namespace-join")
