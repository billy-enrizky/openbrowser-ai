"""URL discovery and bounded crawling over HTTP."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _SiteHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        return

    def _send(self, status: int, body: bytes, content_type: str = "text/html"):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = (
                "<html><head><title>Home</title></head><body>"
                '<a href="/a">A</a> <a href="/a#frag">A again</a> '
                '<a href="/b?x=1">B</a> <a href="https://elsewhere.example/">off site</a> '
                '<a href="mailto:x@example.com">mail</a> '
                "<p>Home page content that is long enough for a static probe.</p>"
                "</body></html>"
            )
            self._send(200, body.encode())
            return
        if self.path == "/a":
            body = (
                "<html><head><title>Page A</title></head><body>"
                '<a href="/c">C</a> <a href="/">home</a>'
                "<p>Page A content.</p></body></html>"
            )
            self._send(200, body.encode())
            return
        if self.path == "/b?x=1":
            body = (
                '<html><head><title>Page B</title></head><body><a href="/d">D</a><p>Page B content.</p></body></html>'
            )
            self._send(200, body.encode())
            return
        if self.path == "/c":
            self._send(200, b"<html><head><title>Page C</title></head><body><p>Page C content.</p></body></html>")
            return
        if self.path == "/d":
            self._send(200, b"<html><head><title>Page D</title></head><body><p>Page D content.</p></body></html>")
            return
        if self.path == "/sitemap.xml":
            body = "<urlset><url><loc>/e</loc></url><url><loc>/a</loc></url></urlset>"
            self._send(200, body.encode(), "application/xml")
            return
        if self.path == "/e":
            self._send(200, b"<html><head><title>Page E</title></head><body><p>Page E content.</p></body></html>")
            return
        if self.path == "/asset.pdf":
            self._send(200, b"%PDF-1.4", "application/pdf")
            return
        self._send(404, b"missing")


@pytest.fixture()
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SiteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_map_urls_returns_start_then_page_links_then_sitemap(site):
    from openbrowser.scrape.crawl import map_urls

    result = await map_urls(f"{site}/", timeout_ms=5000)

    urls = [item["url"] for item in result["urls"]]
    assert urls[0] == f"{site}/"
    assert f"{site}/a" in urls
    assert f"{site}/b?x=1" in urls
    assert f"{site}/e" in urls
    assert urls.count(f"{site}/a") == 1
    assert all("elsewhere.example" not in url for url in urls)
    assert all(not url.startswith("mailto:") for url in urls)
    assert result["stats"]["page_added"] == 2
    assert result["stats"]["sitemap_added"] == 1
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_map_urls_respects_max_urls_include_and_exclude(site):
    from openbrowser.scrape.crawl import map_urls

    limited = await map_urls(f"{site}/", max_urls=2, timeout_ms=5000)
    assert len(limited["urls"]) == 2
    assert limited["truncated"] is True

    filtered = await map_urls(f"{site}/", include=r"/[ab]", exclude=r"x=1", timeout_ms=5000)
    urls = [item["url"] for item in filtered["urls"]]
    assert f"{site}/a" in urls
    assert f"{site}/b?x=1" not in urls


@pytest.mark.asyncio
async def test_map_urls_can_skip_the_sitemap(site):
    from openbrowser.scrape.crawl import map_urls

    result = await map_urls(f"{site}/", use_sitemap=False, timeout_ms=5000)
    urls = [item["url"] for item in result["urls"]]
    assert f"{site}/e" not in urls
    assert result["stats"]["sitemap_added"] == 0


@pytest.mark.asyncio
async def test_crawl_site_follows_same_origin_links_with_depth_limit(site):
    from openbrowser.scrape.crawl import crawl_site

    result = await crawl_site(f"{site}/", max_pages=10, max_depth=1, format="text", timeout_ms=5000)

    by_url = {page["url"]: page for page in result["pages"]}
    assert by_url[f"{site}/"]["ok"] is True
    assert "Home page content" in by_url[f"{site}/"]["content"]
    assert by_url[f"{site}/a"]["ok"] is True
    assert by_url[f"{site}/b?x=1"]["ok"] is True
    assert f"{site}/c" not in by_url
    assert result["total_pages"] == 3


@pytest.mark.asyncio
async def test_crawl_site_respects_max_pages_and_reports_truncation(site):
    from openbrowser.scrape.crawl import crawl_site

    result = await crawl_site(f"{site}/", max_pages=2, max_depth=3, format="text", timeout_ms=5000)
    assert result["attempted"] == 2
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_crawl_site_reports_non_html_as_skipped(site):
    from openbrowser.scrape.crawl import crawl_site

    result = await crawl_site(f"{site}/asset.pdf", max_pages=1, format="text", timeout_ms=5000)
    page = result["pages"][0]
    assert page["ok"] is False
    assert page["skipped"] is True
    assert "content type" in page["error"]


@pytest.mark.asyncio
async def test_map_urls_normalizes_fragments_and_case():
    from openbrowser.scrape.crawl import normalize_url

    assert normalize_url("HTTP://Example.COM/a#x") == "http://example.com/a"
    assert normalize_url("https://example.com") == "https://example.com/"
