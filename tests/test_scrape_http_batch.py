"""HTTP-only batch scraping without a browser session."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


class _OwnedClientHandler(BaseHTTPRequestHandler):
    """Local fixture for the shared owned-client HTTP path."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        return

    def _send(self, status: int, body: bytes, *, content_type: str = "text/html", extra=()):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/redirect":
            self._send(302, b"", extra=[("location", "/article")])
            return
        if self.path.startswith("/cookie"):
            cookie_seen = self.headers.get("cookie")
            body = f"<html><head><title>Cookie</title></head><body><p>cookie={cookie_seen}</p></body></html>"
            self._send(
                200,
                body.encode(),
                extra=[("set-cookie", "session=abc; Path=/")],
            )
            return
        if self.path.startswith("/article"):
            body = (
                "<html><head><title>Article</title></head>"
                "<body><p>Static page one with enough text for the static probe.</p></body></html>"
            )
            self._send(200, body.encode())
            return
        if self.path.startswith("/page/"):
            page = self.path.rsplit("/", 1)[-1]
            body = (
                f"<html><head><title>Page {page}</title></head>"
                f"<body><p>Static page {page} with enough text for the static probe.</p></body></html>"
            )
            self._send(200, body.encode())
            return
        self._send(404, b"missing")


@pytest.fixture()
def owned_client_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OwnedClientHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_owned_client_result_matches_one_shot_client(owned_client_server):
    from openbrowser.scrape.http import _HttpClientOwner, scrape_http

    url = f"{owned_client_server}/article"
    one_shot = await scrape_http(
        url,
        format="markdown",
        selector=None,
        extract_links=False,
        timeout_ms=5000,
        engine="http",
    )
    owner = _HttpClientOwner()
    try:
        owned = await scrape_http(
            url,
            format="markdown",
            selector=None,
            extract_links=False,
            timeout_ms=5000,
            engine="http",
            client=owner.get(5.0),
        )
    finally:
        await owner.close()

    assert one_shot is not None
    assert owned is not None
    assert owned == one_shot


@pytest.mark.asyncio
async def test_owned_client_survives_concurrent_requests(owned_client_server):
    from openbrowser.scrape.http import scrape_http_batch

    urls = [f"{owned_client_server}/page/{index}" for index in range(25)]
    results = await scrape_http_batch(
        urls,
        concurrency=5,
        format="text",
        engine="http",
        timeout_ms=5000,
    )

    assert [item["url"] for item in results] == urls
    assert all(item["ok"] for item in results), [item["error"] for item in results if not item["ok"]]
    assert [item["content"].strip() for item in results] == [
        f"Static page {index} with enough text for the static probe." for index in range(25)
    ]


@pytest.mark.asyncio
async def test_owned_client_follows_redirects(owned_client_server):
    from openbrowser.scrape.http import _HttpClientOwner, scrape_http

    owner = _HttpClientOwner()
    try:
        result = await scrape_http(
            f"{owned_client_server}/redirect",
            format="text",
            selector=None,
            extract_links=False,
            timeout_ms=5000,
            engine="http",
            client=owner.get(5.0),
        )
    finally:
        await owner.close()

    assert result is not None
    assert result["url"].endswith("/article")
    assert "Static page" in result["content"]


@pytest.mark.asyncio
async def test_owned_client_does_not_send_or_store_cookies(owned_client_server):
    from openbrowser.scrape.http import _HttpClientOwner, scrape_http

    owner = _HttpClientOwner()
    try:
        client = owner.get(5.0)
        contents = []
        for _ in range(3):
            result = await scrape_http(
                f"{owned_client_server}/cookie",
                format="text",
                selector=None,
                extract_links=False,
                timeout_ms=5000,
                engine="http",
                client=client,
            )
            assert result is not None
            contents.append(result["content"])
    finally:
        await owner.close()

    assert contents == ["cookie=None", "cookie=None", "cookie=None"]


@pytest.mark.asyncio
async def test_owned_client_closes_once_and_reopens_after_close():
    from openbrowser.scrape.http import _HttpClientOwner

    owner = _HttpClientOwner()
    first = owner.get(1.0)
    assert first.is_closed is False
    await owner.close()
    assert first.is_closed is True

    await owner.close()
    assert first.is_closed is True

    second = owner.get(1.0)
    assert second is not first
    assert second.is_closed is False
    await owner.close()
    assert second.is_closed is True
