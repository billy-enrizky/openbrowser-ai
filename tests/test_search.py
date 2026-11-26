from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest
from ddgs.exceptions import DDGSException

import openbrowser.web_search as search_module
from openbrowser.web_search import WebSearchError, WebSearchResult, aweb_search, web_search


def _install_provider(monkeypatch, responder: Callable[[int, str, dict[str, Any]], object]):
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def text(self, query: str, **kwargs: Any):
            calls.append((query, kwargs))
            return responder(len(calls), query, kwargs)

    monkeypatch.setattr(search_module, "DDGS", FakeDDGS)
    return calls


def test_search_normalizes_urls_deduplicates_and_preserves_provider_order(monkeypatch):
    rows = [
        {
            "title": " First result ",
            "href": "HTTPS://EXAMPLE.COM:443#section",
            "body": "  first   preview\ntext ",
        },
        {"title": "Duplicate", "url": "https://example.com/", "snippet": "ignored"},
        {"title": "Relative", "href": "/relative"},
        {"title": "Other scheme", "href": "ftp://example.net/file"},
        {
            "title": "Second result",
            "url": "http://Example.org:80?topic=search#top",
            "description": "second preview",
        },
    ]
    calls = _install_provider(monkeypatch, lambda *_: rows)

    results = web_search(
        "  browser search  ",
        5,
        backend="duckduckgo",
        region="ca-en",
        safe_search="on",
        timelimit="w",
    )

    assert results == [
        WebSearchResult(title="First result", url="https://example.com/", snippet="first preview text"),
        WebSearchResult(title="Second result", url="http://example.org/?topic=search", snippet="second preview"),
    ]
    assert calls == [
        (
            "browser search",
            {
                "max_results": 5,
                "backend": "duckduckgo",
                "region": "ca-en",
                "safesearch": "on",
                "timelimit": "w",
            },
        )
    ]


def test_search_caps_results_at_safe_limit(monkeypatch):
    rows = [{"title": f"Result {index}", "href": f"https://example-{index}.com"} for index in range(30)]
    calls = _install_provider(monkeypatch, lambda *_: rows)

    results = web_search("query", max_results=100)

    assert len(results) == 20
    assert calls[0][1]["max_results"] == 20


@pytest.mark.parametrize(
    ("query", "kwargs", "error_type"),
    [
        ("  ", {}, ValueError),
        ("query", {"max_results": 0}, ValueError),
        ("query", {"max_results": True}, TypeError),
        ("query", {"safe_search": "strict"}, ValueError),
        ("query", {"backend": "duckduckgo, "}, ValueError),
        ("query", {"region": " "}, ValueError),
        ("query", {"timelimit": "x" * 65}, ValueError),
        ("x" * 2049, {}, ValueError),
    ],
)
def test_search_rejects_invalid_options_before_provider_call(monkeypatch, query, kwargs, error_type):
    calls = _install_provider(monkeypatch, lambda *_: pytest.fail("provider must not run"))

    with pytest.raises(error_type):
        web_search(query, **kwargs)

    assert calls == []


def test_empty_provider_result_returns_empty_list(monkeypatch):
    calls = _install_provider(monkeypatch, lambda *_: [])

    assert web_search("query") == []
    assert len(calls) == 1


def test_provider_no_results_error_retries_then_returns_empty(monkeypatch):
    delays: list[float] = []
    calls = _install_provider(monkeypatch, lambda *_: (_ for _ in ()).throw(DDGSException("No results found.")))
    monkeypatch.setattr(search_module.time, "sleep", delays.append)

    assert web_search("query") == []
    assert len(calls) == 2
    assert delays == [0.2]


def test_provider_no_results_retry_can_recover(monkeypatch):
    delays: list[float] = []

    def respond(attempt: int, query: str, kwargs: dict[str, Any]):
        if attempt == 1:
            raise DDGSException("No results found.")
        return [{"title": "Recovered", "href": "https://example.com"}]

    calls = _install_provider(monkeypatch, respond)
    monkeypatch.setattr(search_module.time, "sleep", delays.append)

    assert web_search("query") == [WebSearchResult("Recovered", "https://example.com/", "")]
    assert len(calls) == 2
    assert delays == [0.2]


def test_no_results_after_another_provider_failure_does_not_become_empty(monkeypatch):
    delays: list[float] = []

    def respond(attempt: int, query: str, kwargs: dict[str, Any]):
        if attempt == 1:
            raise TimeoutError("provider timeout")
        raise DDGSException("No results found.")

    calls = _install_provider(monkeypatch, respond)
    monkeypatch.setattr(search_module.time, "sleep", delays.append)

    with pytest.raises(WebSearchError) as error:
        web_search("query")

    assert error.value.attempts == 3
    assert len(calls) == 3
    assert delays == [0.2, 0.4]


def test_nonempty_invalid_provider_rows_raise_instead_of_becoming_empty(monkeypatch):
    calls = _install_provider(monkeypatch, lambda *_: [{"title": "No usable URL", "href": "file:///tmp/a"}])

    with pytest.raises(WebSearchError, match="absolute HTTP or HTTPS URLs") as error:
        web_search("query")

    assert error.value.attempts == 1
    assert len(calls) == 1


def test_provider_failures_retry_with_bounded_backoff_and_actionable_error(monkeypatch):
    delays: list[float] = []
    calls = _install_provider(monkeypatch, lambda *_: (_ for _ in ()).throw(TimeoutError("provider timeout")))
    monkeypatch.setattr(search_module.time, "sleep", delays.append)

    with pytest.raises(WebSearchError, match="browser search fallback") as error:
        web_search("query")

    assert error.value.attempts == 3
    assert len(calls) == 3
    assert delays == [0.2, 0.4]


def test_provider_retry_can_recover(monkeypatch):
    delays: list[float] = []

    def respond(attempt: int, query: str, kwargs: dict[str, Any]):
        if attempt == 1:
            raise TimeoutError("temporary timeout")
        return [{"title": "Ready", "href": "https://example.com"}]

    calls = _install_provider(monkeypatch, respond)
    monkeypatch.setattr(search_module.time, "sleep", delays.append)

    assert web_search("query") == [WebSearchResult("Ready", "https://example.com/", "")]
    assert len(calls) == 2
    assert delays == [0.2]


@pytest.mark.asyncio
async def test_async_search_runs_blocking_provider_in_worker_thread(monkeypatch):
    event_loop_thread = threading.get_ident()
    provider_thread: list[int] = []

    def respond(*_):
        provider_thread.append(threading.get_ident())
        return [{"title": "Async result", "href": "https://example.com"}]

    _install_provider(monkeypatch, respond)

    results = await aweb_search("query")

    assert results == [WebSearchResult("Async result", "https://example.com/", "")]
    assert provider_thread and provider_thread[0] != event_loop_thread


def test_legacy_openbrowser_search_imports_still_alias_web_search():
    import openbrowser.search as legacy
    from openbrowser.web_search import MAX_QUERY_LENGTH, MAX_WEB_SEARCH_RESULTS

    assert legacy.search is web_search
    assert legacy.asearch is aweb_search
    assert legacy.SearchResult is WebSearchResult
    assert legacy.SearchError is WebSearchError
    assert legacy.MAX_SEARCH_RESULTS == MAX_WEB_SEARCH_RESULTS
    assert legacy.MAX_QUERY_LENGTH == MAX_QUERY_LENGTH
