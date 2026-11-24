"""Search the web with a typed, provider-backed API."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast
from urllib.parse import urlsplit, urlunsplit

from ddgs import DDGS
from ddgs.exceptions import DDGSException

__all__ = ["SafeSearch", "SearchError", "SearchResult", "asearch", "search"]

SafeSearch = Literal["on", "moderate", "off"]
MAX_SEARCH_RESULTS = 20
MAX_QUERY_LENGTH = 2048
_MAX_OPTION_LENGTH = 256
_RETRY_BACKOFF_SECONDS = (0.2, 0.4)
_NO_RESULTS_MESSAGE = "No results found."


@dataclass(frozen=True, slots=True)
class _SearchRequest:
    query: str
    max_results: int
    backend: str
    region: str
    safe_search: SafeSearch
    timelimit: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A search result whose snippet is an unverified provider preview."""

    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        """Return the stable mapping used by the CLI, CodeAgent, and MCP."""
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchError(RuntimeError):
    """A provider failure that callers can handle with browser search."""

    def __init__(self, message: str, *, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(message)


def search(
    query: str,
    max_results: int = 5,
    *,
    backend: str = "auto",
    region: str = "us-en",
    safe_search: SafeSearch = "moderate",
    timelimit: str | None = None,
) -> list[SearchResult]:
    """Search synchronously and return normalized results in provider order."""
    request = _make_request(query, max_results, backend, region, safe_search, timelimit)
    return _search_sync(request)


async def asearch(
    query: str,
    max_results: int = 5,
    *,
    backend: str = "auto",
    region: str = "us-en",
    safe_search: SafeSearch = "moderate",
    timelimit: str | None = None,
) -> list[SearchResult]:
    """Search asynchronously without blocking the event loop."""
    request = _make_request(query, max_results, backend, region, safe_search, timelimit)
    return await asyncio.to_thread(_search_sync, request)


def _make_request(
    query: str,
    max_results: int,
    backend: str,
    region: str,
    safe_search: str,
    timelimit: str | None,
) -> _SearchRequest:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"query must not exceed {MAX_QUERY_LENGTH} characters")

    if not isinstance(max_results, int) or isinstance(max_results, bool):
        raise TypeError("max_results must be an integer")
    if max_results < 1:
        raise ValueError("max_results must be a positive integer")
    max_results = min(max_results, MAX_SEARCH_RESULTS)

    backend = _text_option("backend", backend, _MAX_OPTION_LENGTH)
    backend_names = [name.strip() for name in backend.split(",")]
    if any(not name for name in backend_names):
        raise ValueError("backend must contain one or more backend names")
    backend = ",".join(backend_names)
    region = _text_option("region", region, 32)
    safe_search = _text_option("safe_search", safe_search, 16).lower()
    if safe_search not in {"on", "moderate", "off"}:
        raise ValueError("safe_search must be 'on', 'moderate', or 'off'")
    if timelimit is not None:
        timelimit = _text_option("timelimit", timelimit, 64)

    return _SearchRequest(
        query=query,
        max_results=max_results,
        backend=backend,
        region=region,
        safe_search=cast(SafeSearch, safe_search),
        timelimit=timelimit,
    )


def _text_option(name: str, value: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > max_length:
        raise ValueError(f"{name} must not exceed {max_length} characters")
    return value


def _search_sync(request: _SearchRequest) -> list[SearchResult]:
    last_error: Exception | None = None
    previous_attempt_was_no_results = False
    for attempt in range(len(_RETRY_BACKOFF_SECONDS) + 1):
        try:
            with DDGS() as provider:
                rows = provider.text(
                    request.query,
                    max_results=request.max_results,
                    backend=request.backend,
                    region=request.region,
                    safesearch=request.safe_search,
                    timelimit=request.timelimit,
                )
        except DDGSException as error:
            is_no_results = type(error) is DDGSException and str(error).strip() == _NO_RESULTS_MESSAGE
            if is_no_results and previous_attempt_was_no_results:
                return []
            previous_attempt_was_no_results = is_no_results
            last_error = error
        except Exception as error:
            previous_attempt_was_no_results = False
            last_error = error
        else:
            return _normalize_results(rows, request.max_results)

        if attempt < len(_RETRY_BACKOFF_SECONDS):
            time.sleep(_RETRY_BACKOFF_SECONDS[attempt])

    attempts = len(_RETRY_BACKOFF_SECONDS) + 1
    error_type = type(last_error).__name__ if last_error is not None else "UnknownError"
    raise SearchError(
        f"DDGS search failed after {attempts} attempts ({error_type}). "
        "Check network access or the selected backend, then retry or use the browser search fallback.",
        attempts=attempts,
    ) from last_error


def _normalize_results(rows: object, max_results: int) -> list[SearchResult]:
    if not isinstance(rows, list):
        raise SearchError(
            "DDGS returned an invalid result list. Try another backend or use browser search.",
            attempts=1,
        )
    if not rows:
        return []

    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        url = _normalize_url(row.get("href")) or _normalize_url(row.get("url"))
        if url is None or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(
            SearchResult(
                title=_clean_result_text(row.get("title")),
                url=url,
                snippet=_clean_result_text(row.get("body") or row.get("snippet") or row.get("description")),
            )
        )
        if len(results) == max_results:
            break

    if not results:
        raise SearchError(
            "DDGS returned no absolute HTTP or HTTPS URLs. Try another backend or use browser search.",
            attempts=1,
        )
    return results


def _clean_result_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _normalize_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or any(character.isspace() for character in candidate):
        return None
    try:
        parts = urlsplit(candidate)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.netloc or not hostname:
        return None
    if parts.username is not None or parts.password is not None or port == 0:
        return None

    if ":" in hostname:
        host = f"[{hostname.lower()}]"
    else:
        try:
            host = hostname.encode("idna").decode("ascii").lower().rstrip(".")
        except UnicodeError:
            return None
    if not host:
        return None
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    return urlunsplit((scheme, host, parts.path or "/", parts.query, ""))
