"""Exact-oracle tests for the no-browser MCP memory lane."""

from __future__ import annotations

import json

from benchmarks.scraping.mcp_http_resource import EXPECTED_TEXT, _result_passes


def _payload(**overrides):
    value = {
        "ok": True,
        "engine_used": "http",
        "content": EXPECTED_TEXT[:5000],
        "truncated": True,
        "total_chars": len(EXPECTED_TEXT),
    }
    value.update(overrides)
    return json.dumps(value)


def test_accepts_the_exact_truncated_fixture_result():
    assert _result_passes(_payload()) is True


def test_rejects_a_partial_content_match():
    assert _result_passes(_payload(content="OpenBrowser static content")) is False


def test_rejects_a_result_that_is_not_truncated():
    assert _result_passes(_payload(truncated=False)) is False


def test_rejects_a_wrong_total_length():
    assert _result_passes(_payload(total_chars=len(EXPECTED_TEXT) - 1)) is False


def test_rejects_a_browser_engine():
    assert _result_passes(_payload(engine_used="browser")) is False


def test_rejects_an_error_payload():
    assert _result_passes(json.dumps({"error": "http_fetch_failed"})) is False
