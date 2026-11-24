from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openbrowser.mcp import server as mcp_module
from openbrowser.search import SearchError, SearchResult


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol_version", "legacy"),
    [("2026-07-28", False), ("2024-11-05", True)],
)
async def test_search_web_dispatch_is_stateless_for_modern_and_legacy(
    monkeypatch,
    protocol_version,
    legacy,
):
    monkeypatch.setattr(mcp_module, "TELEMETRY_AVAILABLE", False)
    server = mcp_module.OpenBrowserServer()
    calls = []

    async def fake_asearch(query, **kwargs):
        calls.append((query, kwargs))
        return [SearchResult("Docs", "https://docs.example/", "Preview")]

    monkeypatch.setattr("openbrowser.search.asearch", fake_asearch)
    denied_acquire = AsyncMock(side_effect=AssertionError("stateless search acquired a context"))
    if legacy:
        monkeypatch.setattr(server._contexts, "acquire_legacy", denied_acquire)
    else:
        monkeypatch.setattr(server._contexts, "acquire", denied_acquire)

    handler = server.server._request_handlers["tools/call"].handler
    context = SimpleNamespace(protocol_version=protocol_version)
    params = mcp_module.types.CallToolRequestParams(
        name="search_web",
        arguments={"query": "browser docs", "max_results": 2},
    )
    result = await handler(context, params)

    assert result.is_error is False
    assert json.loads(result.content[0].text) == [
        {"title": "Docs", "url": "https://docs.example/", "snippet": "Preview"}
    ]
    assert calls == [
        (
            "browser docs",
            {
                "max_results": 2,
                "backend": "auto",
                "region": "us-en",
                "safe_search": "moderate",
                "timelimit": None,
            },
        )
    ]
    denied_acquire.assert_not_awaited()
    assert server.browser_session is None


@pytest.mark.asyncio
async def test_search_web_returns_actionable_json_error_without_context(monkeypatch):
    monkeypatch.setattr(mcp_module, "TELEMETRY_AVAILABLE", False)
    server = mcp_module.OpenBrowserServer()

    async def fail_search(*_args, **_kwargs):
        raise SearchError("Use browser search as fallback.", attempts=3)

    monkeypatch.setattr("openbrowser.search.asearch", fail_search)
    monkeypatch.setattr(
        server._contexts,
        "acquire",
        AsyncMock(side_effect=AssertionError("stateless search acquired a context")),
    )
    handler = server.server._request_handlers["tools/call"].handler
    result = await handler(
        SimpleNamespace(protocol_version="2026-07-28"),
        mcp_module.types.CallToolRequestParams(name="search_web", arguments={"query": "query"}),
    )

    assert result.is_error is True
    assert json.loads(result.content[0].text) == {
        "error": "search_failed",
        "message": "Use browser search as fallback.",
    }
    assert server.browser_session is None


def test_packaged_mcp_manifest_lists_stateless_search_tool():
    manifest_path = Path(__file__).parents[1] / "src/openbrowser/mcp/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    search_tool = next(tool for tool in manifest["tools"] if tool["name"] == "search_web")
    assert "without starting a browser" in search_tool["description"]
