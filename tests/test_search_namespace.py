from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import openbrowser.code_use.namespace as namespace_module
from openbrowser.code_use.descriptions import EXECUTE_CODE_DESCRIPTION, EXECUTE_CODE_DESCRIPTION_COMPACT
from openbrowser.code_use.namespace import create_namespace
from openbrowser.web_search import WebSearchResult


def _make_namespace():
    session = MagicMock()
    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = None
    session.navigate_to = AsyncMock()
    tools = MagicMock()
    tools.registry.registry.actions = {}
    return create_namespace(session, tools=tools), session


def test_code_agent_descriptions_explain_web_search_and_snippet_limits():
    for description in (EXECUTE_CODE_DESCRIPTION_COMPACT, EXECUTE_CODE_DESCRIPTION):
        assert "web_search" in description
        assert "web_search_and_scrape" in description
        assert "Snippets are discovery previews" in description
        assert "Read the source page before you cite it" in description
        assert "scrape_pages" in description


@pytest.mark.asyncio
async def test_namespace_search_web_returns_plain_result_dicts(monkeypatch):
    expected = [WebSearchResult("Docs", "https://docs.example/", "Preview")]
    called = {}

    async def fake_asearch(query, **kwargs):
        called["query"] = query
        called.update(kwargs)
        return expected

    monkeypatch.setattr("openbrowser.web_search.aweb_search", fake_asearch)
    namespace, _ = _make_namespace()

    results = await namespace["web_search"](
        "browser docs",
        max_results=3,
        backend="duckduckgo",
        region="ca-en",
        safe_search="on",
        timelimit="w",
    )

    assert results == [{"title": "Docs", "url": "https://docs.example/", "snippet": "Preview"}]
    assert called == {
        "query": "browser docs",
        "max_results": 3,
        "backend": "duckduckgo",
        "region": "ca-en",
        "safe_search": "on",
        "timelimit": "w",
    }
    await namespace["_close_http_client"]()


@pytest.mark.asyncio
async def test_search_and_scrape_feeds_ordered_urls_to_existing_http_scraper(monkeypatch):
    search_results = [
        WebSearchResult("First", "https://first.example/", "First preview"),
        WebSearchResult("Second", "https://second.example/", "Second preview"),
    ]
    scraped_urls: list[str] = []

    async def fake_asearch(query, **kwargs):
        assert query == " browser docs "
        assert kwargs["max_results"] == 5
        return search_results

    async def fake_scrape_http(url, **kwargs):
        scraped_urls.append(url)
        return {
            "url": url,
            "title": url.split("//", 1)[-1],
            "format": kwargs["format"],
            "content_type": "text/text",
            "content": f"Content from {url}",
            "stats": {"engine_used": "http"},
            "engine_used": "http",
        }

    monkeypatch.setattr("openbrowser.web_search.aweb_search", fake_asearch)
    monkeypatch.setattr(namespace_module, "scrape_http", fake_scrape_http)
    namespace, session = _make_namespace()

    try:
        result = await namespace["web_search_and_scrape"](
            " browser docs ",
            concurrency=2,
            format="text",
            engine="http",
        )
    finally:
        await namespace["_close_http_client"]()

    assert result["query"] == "browser docs"
    assert result["web_search_results"] == [item.as_dict() for item in search_results]
    assert [item["url"] for item in result["page_results"]] == [
        "https://first.example/",
        "https://second.example/",
    ]
    assert [item["content"] for item in result["page_results"]] == [
        "Content from https://first.example/",
        "Content from https://second.example/",
    ]
    assert scraped_urls == ["https://first.example/", "https://second.example/"]
    session.navigate_to.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_search_and_scrape_returns_only_search_results(monkeypatch):
    search_results = [WebSearchResult("First", "https://first.example/", "First preview")]

    async def fake_asearch(query, **kwargs):
        return search_results

    async def fake_scrape_http(url, **kwargs):
        return {
            "url": url,
            "title": url.split("//", 1)[-1],
            "format": kwargs["format"],
            "content_type": "text/text",
            "content": f"Content from {url}",
            "stats": {"engine_used": "http"},
            "engine_used": "http",
        }

    monkeypatch.setattr("openbrowser.web_search.aweb_search", fake_asearch)
    monkeypatch.setattr(namespace_module, "scrape_http", fake_scrape_http)
    namespace, _ = _make_namespace()

    try:
        result = await namespace["search_and_scrape"]("browser docs", engine="http")
    finally:
        await namespace["_close_http_client"]()

    assert set(result) == {"query", "search_results", "page_results"}
    assert result["search_results"] == [item.as_dict() for item in search_results]


@pytest.mark.asyncio
async def test_namespace_keeps_legacy_search_aliases():
    namespace, _ = _make_namespace()
    assert namespace["search_web"] is namespace["web_search"]
    assert namespace["search_and_scrape"] is not namespace["web_search_and_scrape"]
    await namespace["_close_http_client"]()
