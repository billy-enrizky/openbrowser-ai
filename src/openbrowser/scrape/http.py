"""HTTP scraping primitives that do not require a browser session.

The browser path keeps the enhanced DOM and markdownify pipeline. Static
scraping uses the same Lexbor parser as the namespace HTTP path, but these
functions own their client and can run without importing the browser stack.
"""

from __future__ import annotations

import asyncio
import time
from http.cookiejar import CookieJar
from typing import Any


class _NoCookieJar(CookieJar):
    """A cookie jar that neither sends nor stores cookies."""

    def extract_cookies(self, response: Any, request: Any) -> None:
        return None

    def add_cookie_header(self, request: Any) -> None:
        return None


class _HttpClientOwner:
    """Own one reusable HTTP client for a namespace or a scrape batch."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def get(self, timeout: float):
        """Return the owned client, creating it on first use."""
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                cookies=_NoCookieJar(),
            )
        return self._client

    async def close(self) -> None:
        """Close the owned client once."""
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()


def _new_http_client(timeout: float):
    """Create a one-shot client that cannot read or write host cookies."""
    import httpx

    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        cookies=_NoCookieJar(),
    )


def _http_result_from_response(
    response: Any,
    *,
    format: str,
    selector: str | None,
    engine: str,
) -> dict[str, Any] | None:
    """Convert an HTTP response into a scrape result, or None to use the browser."""
    from openbrowser.dom.fast_html import parse_document

    if response.status_code >= 400:
        return None

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if "json" in content_type:
        try:
            payload = response.json()
        except Exception:
            return None
        return {
            "url": str(response.url),
            "title": "",
            "format": format,
            "content_type": "application/json",
            "content": payload,
            "stats": {"engine_used": "http", "status_code": response.status_code},
            "engine_used": "http",
            "status_code": response.status_code,
        }

    if "html" not in content_type and content_type not in ("", "text/plain"):
        return None

    html = response.text
    document = parse_document(html)
    if engine == "auto":
        # Use HTTP only when the server-rendered text is substantial. Thin
        # shells usually need JavaScript, so fall back to the browser.
        text_probe, empty_application_root = document.probe_static()
        if len(text_probe) < 40 or empty_application_root:
            return None

    title = document.title()
    content = document.extract(format, selector)

    return {
        "url": str(response.url),
        "title": title,
        "format": format,
        "content_type": "application/json" if format in {"links", "json"} else f"text/{format}",
        "content": content,
        "stats": {
            "engine_used": "http",
            "status_code": response.status_code,
            "html_chars": len(html),
        },
        "engine_used": "http",
        "status_code": response.status_code,
    }


async def scrape_http(
    url: str,
    *,
    format: str,
    selector: str | None,
    extract_links: bool,
    timeout_ms: int | None,
    engine: str,
    client: Any | None = None,
) -> dict[str, Any] | None:
    """Fetch and parse a URL without a browser.

    Returns None when engine=auto finds page JavaScript and the browser path
    should handle the page. Pass a client only when the caller owns its
    lifecycle and will close it.
    """
    timeout = (timeout_ms / 1000) if timeout_ms else 30.0
    if client is None:
        async with _new_http_client(timeout) as owned_client:
            response = await owned_client.get(url, timeout=timeout)
            return _http_result_from_response(
                response,
                format=format,
                selector=selector,
                engine=engine,
            )
    response = await client.get(url, timeout=timeout)
    return _http_result_from_response(
        response,
        format=format,
        selector=selector,
        engine=engine,
    )


def _truncate_scrape_result(result: dict[str, Any], max_chars: int | None) -> dict[str, Any]:
    """Apply the optional output cap to a scrape result."""
    if max_chars is not None:
        content = result.get("content")
        if isinstance(content, str) and len(content) > max_chars:
            result["content"] = content[:max_chars]
            result["truncated"] = True
            result["total_chars"] = len(content)
    return result


async def scrape_http_batch(
    urls: list[str],
    *,
    concurrency: int = 4,
    format: str = "markdown",
    selector: str | None = None,
    extract_links: bool = False,
    timeout_ms: int | None = None,
    max_chars: int | None = None,
    engine: str = "http",
) -> list[dict[str, Any]]:
    """Scrape URLs over HTTP concurrently without starting a browser.

    When ``engine`` is ``http``, a URL that cannot be fetched or parsed is
    reported as a failed batch item. When it is ``auto``, a page that looks
    like a JavaScript shell is also reported as a failed item because this
    function has no browser fallback. Use ``scrape_pages()`` when a fallback is
    required.
    """
    allowed_formats = {"markdown", "text", "html", "links", "json"}
    if format not in allowed_formats:
        raise ValueError(f"format must be one of {sorted(allowed_formats)}, got {format!r}")
    if engine not in {"auto", "http"}:
        raise ValueError(f"engine must be one of ['auto', 'http'], got {engine!r}")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if isinstance(urls, str):
        url_list = [urls]
    else:
        try:
            url_list = list(urls)
        except TypeError as error:
            raise ValueError("urls must be a sequence of URL strings") from error
    url_list = [str(url).strip() for url in url_list]
    if not url_list:
        return []

    owner = _HttpClientOwner()
    semaphore = asyncio.Semaphore(concurrency)
    started_at = [time.perf_counter() for _ in url_list]

    async def _fetch(url: str) -> tuple[dict[str, Any] | None, str | None, float]:
        async with semaphore:
            http_started = time.perf_counter()
            try:
                result = await scrape_http(
                    url,
                    format=format,
                    selector=selector,
                    extract_links=extract_links,
                    timeout_ms=timeout_ms,
                    engine=engine,
                    client=owner.get((timeout_ms / 1000) if timeout_ms else 30.0),
                )
                return result, None, (time.perf_counter() - http_started) * 1000.0
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}", (time.perf_counter() - http_started) * 1000.0

    try:
        attempts = await asyncio.gather(*(_fetch(url) for url in url_list))
    finally:
        await owner.close()

    results: list[dict[str, Any]] = []
    for index, url in enumerate(url_list):
        result, error, http_ms = attempts[index]
        if result is not None:
            _truncate_scrape_result(result, max_chars)
            item = dict(result)
            final_url = item.pop("url", None)
            item["url"] = url
            if final_url and final_url != url:
                item["final_url"] = final_url
            item["ok"] = True
            item["engine_used"] = str(result.get("engine_used") or "http")
            item["error"] = None
        else:
            item = {
                "url": url,
                "ok": False,
                "engine_used": "http",
                "content": None,
                "error": error or f"HTTP scrape failed for {url}",
            }
        elapsed_ms = (time.perf_counter() - started_at[index]) * 1000.0
        duration_ms = http_ms if result is not None else elapsed_ms
        item["duration_ms"] = round(duration_ms, 3)
        item["timing"] = {"total_ms": item["duration_ms"], "http_ms": round(http_ms, 3)}
        results.append(item)
    return results


def run_http_scrape_sync(
    url_list: list[str],
    *,
    batch_mode: bool,
    output_format: str,
    selector: str | None,
    extract_links: bool,
    timeout_ms: int | None,
    max_chars: int | None,
    concurrency: int,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run a single or batch HTTP scrape through one event loop."""
    if batch_mode:
        return asyncio.run(
            scrape_http_batch(
                url_list,
                concurrency=concurrency,
                format=output_format,
                selector=selector,
                extract_links=extract_links,
                timeout_ms=timeout_ms,
                max_chars=max_chars,
                engine="http",
            )
        )
    result = asyncio.run(
        scrape_http(
            url_list[0],
            format=output_format,
            selector=selector,
            extract_links=extract_links,
            timeout_ms=timeout_ms,
            engine="http",
        )
    )
    if result is None:
        raise RuntimeError(f"HTTP scrape failed for {url_list[0]}")
    return _truncate_scrape_result(result, max_chars)
