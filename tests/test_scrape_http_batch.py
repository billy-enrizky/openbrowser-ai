"""HTTP-only batch scraping without a browser session."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_scrape_http_batch_preserves_input_order_and_reports_failures(monkeypatch):
    from openbrowser.scrape import http as http_module

    async def fake_scrape_http(url: str, **_kwargs):
        if url.endswith("/bad"):
            raise RuntimeError("boom")
        return {
            "url": url,
            "title": "",
            "format": "text",
            "content_type": "text/text",
            "content": f"content for {url}",
            "stats": {"engine_used": "http"},
            "engine_used": "http",
        }

    monkeypatch.setattr(http_module, "scrape_http", fake_scrape_http)
    results = await http_module.scrape_http_batch(
        ["https://a.example/one", "https://a.example/bad", "https://a.example/three"],
        concurrency=2,
        format="text",
        engine="http",
    )

    assert [item["url"] for item in results] == [
        "https://a.example/one",
        "https://a.example/bad",
        "https://a.example/three",
    ]
    assert [item["ok"] for item in results] == [True, False, True]
    assert results[0]["content"] == "content for https://a.example/one"
    assert "boom" in results[1]["error"]
    assert results[0]["timing"]["http_ms"] >= 0


@pytest.mark.asyncio
async def test_scrape_http_batch_rejects_invalid_concurrency():
    from openbrowser.scrape.http import scrape_http_batch

    with pytest.raises(ValueError, match="positive integer"):
        await scrape_http_batch(["https://example.com"], concurrency=0)


@pytest.mark.asyncio
async def test_scrape_http_batch_rejects_browser_engine():
    from openbrowser.scrape.http import scrape_http_batch

    with pytest.raises(ValueError, match="auto"):
        await scrape_http_batch(["https://example.com"], engine="browser")
