"""Exact-oracle tests for the no-browser MCP memory lane."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.scraping import mcp_http_resource as lane
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


@pytest.fixture
def fake_lane(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("native I/O is forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(lane.psutil, "Process", forbidden)
    server = SimpleNamespace(
        server_port=12345, serve_forever=lambda: None, shutdown=lambda: None, server_close=lambda: None
    )
    monkeypatch.setattr(lane, "ThreadingHTTPServer", lambda *args: server)
    monkeypatch.setattr(
        lane.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda **kwargs: None)
    )
    monkeypatch.setattr(lane, "_binary", lambda: "/fake/openbrowser-ai")
    calls = []

    def client(binary, url, count):
        calls.append(count)
        return {"calls": count, "ok": count, "peak_rss_mib": 50.0, "browser_processes_observed": False}

    monkeypatch.setattr(lane, "_one_client", client)
    path = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["mcp_http_resource", "--output", str(path)])
    return path, calls


def test_default_http_resource_lane_runs_five_independent_observations_per_series(fake_lane):
    path, calls = fake_lane
    assert lane.main() == 0
    assert calls == [1] * 5 + [25] * 5
    report = json.loads(path.read_text())
    assert report["valid"] is True
    assert len(report["sequential"]) == len(report["concurrent"]) == 5
    assert sum(row["ok"] for row in report["concurrent"]) == 125


@pytest.mark.parametrize("flag", ["--samples", "--concurrency", "--concurrency-samples"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_http_resource_lane_rejects_empty_or_negative_plan(monkeypatch, fake_lane, flag, value):
    path, calls = fake_lane
    monkeypatch.setattr(sys, "argv", ["mcp_http_resource", "--output", str(path), flag, value])
    with pytest.raises(SystemExit) as error:
        lane.main()
    assert error.value.code == 2
    assert calls == []
    assert not path.exists()
