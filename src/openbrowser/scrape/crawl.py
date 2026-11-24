"""URL discovery and bounded site crawling over HTTP.

`map_urls` answers "which pages exist on this site" from one page plus its
sitemap. `crawl_site` follows same-origin links breadth first and returns page
content. Both use the shared owned HTTP client, so a static site costs one
request per page and never starts a browser.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from openbrowser.scrape.http import _HttpClientOwner

DEFAULT_MAX_URLS = 100
DEFAULT_MAX_PAGES = 10
DEFAULT_MAX_DEPTH = 2
_SITEMAP_PATHS = ("/sitemap.xml", "/sitemap_index.xml")
_MAX_CHILD_SITEMAPS = 5
_LOC_PATTERN = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
_SITEMAP_INDEX_PATTERN = re.compile(r"<sitemapindex\b", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """Lowercase the scheme and host, drop the fragment, and keep the path."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = parts.netloc.lower()
    path = parts.path or "/"
    return urlunsplit((scheme, host, path, parts.query, ""))


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _is_http(url: str) -> bool:
    return urlsplit(url).scheme.lower() in ("http", "https")


def _matches(url: str, include: str | None, exclude: str | None) -> bool:
    if include is not None and not re.search(include, url):
        return False
    return not (exclude is not None and re.search(exclude, url))


def _links_from_document(document: Any, page_url: str, *, same_origin: bool, origin: str) -> list[str]:
    found: list[str] = []
    for item in document.links():
        href = item.get("href")
        if not isinstance(href, str) or not href.strip():
            continue
        if href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        absolute = urljoin(page_url, href.strip())
        if not _is_http(absolute):
            continue
        normalized = normalize_url(absolute)
        if same_origin and origin_of(normalized) != origin:
            continue
        found.append(normalized)
    return found


async def _get(client: Any, url: str, timeout: float) -> Any | None:  # noqa: ASYNC109 - transport owns the deadline
    """Fetch a URL and return the response, or None on any transport error."""
    try:
        return await client.get(url, timeout=timeout)
    except Exception:
        return None


async def _sitemap_urls(client: Any, origin: str, timeout: float) -> list[str]:  # noqa: ASYNC109 - transport owns the deadline
    """Read sitemap entries for one origin, following one index level."""
    urls: list[str] = []
    for path in _SITEMAP_PATHS:
        response = await _get(client, origin + path, timeout)
        if response is None or response.status_code >= 400:
            continue
        text = response.text
        locs = _LOC_PATTERN.findall(text)
        if not locs:
            continue
        if _SITEMAP_INDEX_PATTERN.search(text):
            for child in locs[:_MAX_CHILD_SITEMAPS]:
                child_response = await _get(client, child, timeout)
                if child_response is None or child_response.status_code >= 400:
                    continue
                urls.extend(_LOC_PATTERN.findall(child_response.text))
        else:
            urls.extend(locs)
        if urls:
            break
    return urls


async def map_urls(
    url: str,
    *,
    max_urls: int = DEFAULT_MAX_URLS,
    same_origin: bool = True,
    include: str | None = None,
    exclude: str | None = None,
    timeout_ms: int | None = None,
    use_sitemap: bool = True,
) -> dict[str, Any]:
    """Discover URLs from one page and, optionally, its sitemap.

    Returns the start URL first, then links in page order, then sitemap
    entries. Every URL is normalized, deduplicated, and filtered by origin and
    by the optional include and exclude patterns.
    """
    if max_urls < 1:
        raise ValueError("max_urls must be positive")
    timeout = (timeout_ms / 1000) if timeout_ms else 30.0
    start = normalize_url(url)
    origin = origin_of(start)
    ordered: list[tuple[str, str]] = []
    seen: set[str] = {start}
    ordered.append((start, "start"))

    warnings: list[str] = []
    owner = _HttpClientOwner()
    try:
        client = owner.get(timeout)
        response = await _get(client, start, timeout)
        if response is None:
            raise RuntimeError(f"Could not fetch {start}")
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} for {start}")
        content_type = response.headers.get("content-type", "")
        if "html" in content_type or content_type == "":
            from openbrowser.dom.fast_html import parse_document

            document = parse_document(response.text)
            for candidate in _links_from_document(document, str(response.url), same_origin=same_origin, origin=origin):
                if candidate in seen:
                    continue
                seen.add(candidate)
                ordered.append((candidate, "page"))
        else:
            warnings.append(f"start URL is {content_type}, not HTML; only the start URL is mapped")

        if use_sitemap and len(ordered) < max_urls:
            for candidate in await _sitemap_urls(client, origin, timeout):
                resolved = candidate if _is_http(candidate) else urljoin(origin, candidate)
                if not _is_http(resolved):
                    continue
                normalized = normalize_url(resolved)
                if same_origin and origin_of(normalized) != origin:
                    continue
                if normalized in seen:
                    continue
                seen.add(normalized)
                ordered.append((normalized, "sitemap"))
    finally:
        await owner.close()

    filtered = [
        (candidate, source)
        for candidate, source in ordered
        if source == "start" or _matches(candidate, include, exclude)
    ]
    truncated = len(filtered) > max_urls
    selected = filtered[:max_urls]
    return {
        "start_url": start,
        "urls": [{"url": candidate, "source": source} for candidate, source in selected],
        "total": len(selected),
        "truncated": truncated,
        "stats": {
            "page_added": sum(1 for _, source in ordered if source == "page"),
            "sitemap_added": sum(1 for _, source in ordered if source == "sitemap"),
            "max_urls": max_urls,
            "same_origin": same_origin,
        },
        "warnings": warnings,
    }


async def crawl_site(
    url: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    same_origin: bool = True,
    format: str = "markdown",
    selector: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    timeout_ms: int | None = None,
    concurrency: int = 4,
) -> dict[str, Any]:
    """Crawl same-origin links breadth first and return page content.

    Each page is fetched once and parsed once. The crawl stops at
    ``max_pages`` total pages or at ``max_depth`` link hops from the start URL.
    Non-HTML responses are reported as skipped, not as failures.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be zero or positive")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    allowed = {"markdown", "text", "html", "links", "json"}
    if format not in allowed:
        raise ValueError(f"format must be one of {sorted(allowed)}, got {format!r}")

    timeout = (timeout_ms / 1000) if timeout_ms else 30.0
    start = normalize_url(url)
    origin = origin_of(start)
    visited: set[str] = {start}
    pages: list[dict[str, Any]] = []
    frontier: list[tuple[str, int]] = [(start, 0)]
    truncated = False
    owner = _HttpClientOwner()

    async def fetch_page(page_url: str, depth: int) -> tuple[dict[str, Any], list[str]]:
        from openbrowser.dom.fast_html import parse_document

        client = owner.get(timeout)
        response = await _get(client, page_url, timeout)
        if response is None:
            return (
                {"url": page_url, "depth": depth, "ok": False, "error": "request failed"},
                [],
            )
        if response.status_code >= 400:
            return (
                {
                    "url": page_url,
                    "depth": depth,
                    "ok": False,
                    "error": f"HTTP {response.status_code}",
                },
                [],
            )
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type and "html" not in content_type and content_type != "text/plain":
            return (
                {
                    "url": page_url,
                    "depth": depth,
                    "ok": False,
                    "skipped": True,
                    "error": f"unsupported content type {content_type}",
                },
                [],
            )
        html = response.text
        document = parse_document(html)
        links = _links_from_document(document, str(response.url), same_origin=same_origin, origin=origin)
        page = {
            "url": page_url,
            "final_url": str(response.url),
            "depth": depth,
            "ok": True,
            "title": document.title(),
            "content": document.extract(format, selector),
            "links_found": len(links),
        }
        return page, links

    try:
        while frontier and len(pages) < max_pages:
            level = frontier
            frontier = []
            results = await asyncio.gather(*(fetch_page(page_url, depth) for page_url, depth in level))
            for page, links in results:
                if len(pages) >= max_pages:
                    truncated = True
                    break
                pages.append(page)
                if not page["ok"] or page["depth"] >= max_depth:
                    continue
                for candidate in links:
                    if candidate in visited:
                        continue
                    if not _matches(candidate, include, exclude):
                        continue
                    visited.add(candidate)
                    frontier.append((candidate, page["depth"] + 1))
            if not frontier and len(pages) < max_pages:
                break
            if frontier and len(pages) >= max_pages:
                truncated = True
    finally:
        await owner.close()

    return {
        "start_url": start,
        "pages": pages,
        "total_pages": sum(1 for page in pages if page.get("ok")),
        "attempted": len(pages),
        "max_pages": max_pages,
        "max_depth": max_depth,
        "truncated": truncated or bool(frontier),
        "stats": {"same_origin": same_origin, "format": format},
    }
