from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openbrowser.mcp import server as mcp_module
from openbrowser.web_search import WebSearchError, WebSearchResult


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["web_search", "search_web"])
@pytest.mark.parametrize(
    ("protocol_version", "legacy"),
    [("2026-07-28", False), ("2024-11-05", True)],
)
async def test_web_search_dispatch_is_stateless_for_modern_and_legacy(
    monkeypatch,
    protocol_version,
    legacy,
    tool_name,
):
    monkeypatch.setattr(mcp_module, "TELEMETRY_AVAILABLE", False)
    server = mcp_module.OpenBrowserServer()
    calls = []

    async def fake_asearch(query, **kwargs):
        calls.append((query, kwargs))
        return [WebSearchResult("Docs", "https://docs.example/", "Preview")]

    monkeypatch.setattr("openbrowser.web_search.aweb_search", fake_asearch)
    denied_acquire = AsyncMock(side_effect=AssertionError("stateless search acquired a context"))
    if legacy:
        monkeypatch.setattr(server._contexts, "acquire_legacy", denied_acquire)
    else:
        monkeypatch.setattr(server._contexts, "acquire", denied_acquire)

    handler = server.server._request_handlers["tools/call"].handler
    context = SimpleNamespace(protocol_version=protocol_version)
    params = mcp_module.types.CallToolRequestParams(
        name=tool_name,
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
async def test_web_search_returns_actionable_json_error_without_context(monkeypatch):
    monkeypatch.setattr(mcp_module, "TELEMETRY_AVAILABLE", False)
    server = mcp_module.OpenBrowserServer()

    async def fail_search(*_args, **_kwargs):
        raise WebSearchError("Use browser search as fallback.", attempts=3)

    monkeypatch.setattr("openbrowser.web_search.aweb_search", fail_search)
    monkeypatch.setattr(
        server._contexts,
        "acquire",
        AsyncMock(side_effect=AssertionError("stateless search acquired a context")),
    )
    handler = server.server._request_handlers["tools/call"].handler
    result = await handler(
        SimpleNamespace(protocol_version="2026-07-28"),
        mcp_module.types.CallToolRequestParams(name="web_search", arguments={"query": "query"}),
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

    search_tool = next(tool for tool in manifest["tools"] if tool["name"] == "web_search")
    assert "without starting a browser" in search_tool["description"]
    scrape_tool = next(tool for tool in manifest["tools"] if tool["name"] == "scrape_web")
    assert "no-Blink" in scrape_tool["description"]
    assert "no-V8" in scrape_tool["description"]


@pytest.mark.asyncio
async def test_scrape_web_dispatch_is_stateless(monkeypatch):
    monkeypatch.setattr(mcp_module, "TELEMETRY_AVAILABLE", False)
    server = mcp_module.OpenBrowserServer()
    monkeypatch.setattr(
        "openbrowser.mcp.stateless.execute_stateless_tool",
        AsyncMock(return_value=('{"ok": true, "engine_used": "http"}', False)),
    )
    monkeypatch.setattr(
        server._contexts,
        "acquire",
        AsyncMock(side_effect=AssertionError("stateless scrape acquired a context")),
    )
    handler = server.server._request_handlers["tools/call"].handler
    result = await handler(
        SimpleNamespace(protocol_version="2026-07-28"),
        mcp_module.types.CallToolRequestParams(
            name="scrape_web",
            arguments={"url": "https://example.com/", "format": "text"},
        ),
    )
    assert result.is_error is False
    assert json.loads(result.content[0].text)["engine_used"] == "http"
    assert server.browser_session is None
