from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from openbrowser.cli import main
from openbrowser.search import SearchError, SearchResult


def _results():
    return [
        SearchResult("OpenBrowser Docs", "https://docs.example/", "Search preview only."),
        SearchResult("Python Docs", "https://python.example/", "Another preview."),
    ]


def test_search_prints_compact_table_and_does_not_start_browser(monkeypatch):
    monkeypatch.setattr("openbrowser.search.search", lambda *_args, **_kwargs: _results())
    with patch("openbrowser.browser.BrowserSession.start", new_callable=AsyncMock) as start:
        result = CliRunner().invoke(main, ["search", "browser automation"])

    assert result.exit_code == 0
    assert "Title" in result.output
    assert "URL" in result.output
    assert "Snippet" in result.output
    assert "OpenBrowser Docs" in result.output
    assert "https://docs.example/" in result.output
    start.assert_not_awaited()


def test_search_json_emits_result_array(monkeypatch):
    monkeypatch.setattr("openbrowser.search.search", lambda *_args, **_kwargs: _results())

    result = CliRunner().invoke(main, ["search", "browser automation", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == [item.as_dict() for item in _results()]


def test_search_urls_only_emits_one_url_per_line(monkeypatch):
    monkeypatch.setattr("openbrowser.search.search", lambda *_args, **_kwargs: _results())

    result = CliRunner().invoke(main, ["search", "browser automation", "--urls-only"])

    assert result.exit_code == 0
    assert result.output.splitlines() == ["https://docs.example/", "https://python.example/"]


def test_search_urls_only_can_write_scrape_input_file(monkeypatch, tmp_path):
    destination = tmp_path / "search-urls.txt"
    monkeypatch.setattr("openbrowser.search.search", lambda *_args, **_kwargs: _results())

    result = CliRunner().invoke(
        main,
        ["search", "browser automation", "--urls-only", "--output", str(destination)],
    )

    assert result.exit_code == 0
    assert destination.read_text(encoding="utf-8") == "https://docs.example/\nhttps://python.example/\n"
    assert "Wrote search output" in result.output


def test_search_rejects_conflicting_output_modes_before_provider_call(monkeypatch):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        return _results()

    monkeypatch.setattr("openbrowser.search.search", fail_if_called)

    result = CliRunner().invoke(main, ["search", "query", "--json", "--urls-only"])

    assert result.exit_code == 2
    assert "cannot be used together" in result.output
    assert called is False


def test_search_passes_options_to_package_api(monkeypatch):
    received = {}

    def fake_search(query, **kwargs):
        received["query"] = query
        received.update(kwargs)
        return _results()

    monkeypatch.setattr("openbrowser.search.search", fake_search)

    result = CliRunner().invoke(
        main,
        [
            "search",
            " browser automation ",
            "--max-results",
            "3",
            "--backend",
            "duckduckgo",
            "--region",
            "ca-en",
            "--safe-search",
            "on",
            "--timelimit",
            "w",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert received == {
        "query": " browser automation ",
        "max_results": 3,
        "backend": "duckduckgo",
        "region": "ca-en",
        "safe_search": "on",
        "timelimit": "w",
    }


def test_search_provider_error_has_actionable_cli_message(monkeypatch):
    def fail(*_args, **_kwargs):
        raise SearchError("Use browser search as fallback.", attempts=3)

    monkeypatch.setattr("openbrowser.search.search", fail)

    result = CliRunner().invoke(main, ["search", "query"])

    assert result.exit_code == 1
    assert "Use browser search as fallback." in result.output


def test_search_empty_results_keep_json_and_urls_output_well_formed(monkeypatch):
    monkeypatch.setattr("openbrowser.search.search", lambda *_args, **_kwargs: [])

    json_result = CliRunner().invoke(main, ["search", "query", "--json"])
    urls_result = CliRunner().invoke(main, ["search", "query", "--urls-only"])

    assert json_result.exit_code == 0
    assert json.loads(json_result.output) == []
    assert urls_result.exit_code == 0
    assert urls_result.output == ""
