"""The scraping runner selects the official native Playwright bridge."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.scraping import scrape_compare as scrape


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scraping integration fake attempted native I/O")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.mark.parametrize("name", ["playwright-mcp", "playwright-cli"])
def test_scraping_factory_selects_native_bridge(monkeypatch, name):
    from benchmarks.interactive.adapters import playwright_scrape_bridge

    calls = []
    native = SimpleNamespace(name=name)
    monkeypatch.setattr(playwright_scrape_bridge, "build_scrape_arm", lambda selected: calls.append(selected) or native)
    assert scrape.make_scrape_arm(name) is native
    assert calls == [name]


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
@pytest.mark.parametrize("interleave", [False, True])
def test_scraping_cli_keeps_both_native_paths_for_every_task(
    monkeypatch, tmp_path, failure, cleanup_failure, interleave
):
    from benchmarks.interactive.adapters import playwright_scrape_bridge

    calls = []

    class Native:
        last_engine = "browser"

        def __init__(self, name):
            self.name = name
            self.arm = SimpleNamespace(binary="native", browser_version="fake")

        async def start(self):
            calls.append((self.name, "start"))

        async def prepare(self, task):
            calls.append((self.name, "prepare", task))

        async def stop(self):
            calls.append((self.name, "stop"))
            if cleanup_failure:
                raise OSError("native cleanup failed")

        def __getattr__(self, method):
            if not method.startswith("run_"):
                raise AttributeError(method)

            async def run(base):
                calls.append((self.name, method))
                return not failure, "native output", "native oracle rejected" if failure else ""

            return run

    class Fixture:
        base_url = "http://fixture.invalid"
        state = SimpleNamespace(snapshot=lambda: (0, 0))
        serve_forever = shutdown = server_close = lambda self: None

    monkeypatch.setattr(scrape, "FixtureServer", Fixture)
    monkeypatch.setattr(scrape, "_environment", lambda *args: {})
    monkeypatch.setattr(playwright_scrape_bridge, "build_scrape_arm", Native)
    raw = tmp_path / "raw.jsonl"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scrape_compare",
            "--arms",
            "playwright-mcp",
            "playwright-cli",
            "--samples",
            "1",
            "--warmups",
            "0",
            *(["--interleave"] if interleave else []),
            "--output",
            str(summary),
            "--raw-jsonl",
            str(raw),
        ],
    )
    assert asyncio.run(scrape.main()) == int(failure or cleanup_failure)
    rows = [
        json.loads(line) for line in raw.read_text().splitlines() if json.loads(line).get("record_type") == "sample"
    ]
    assert len(rows) == 10
    metadata = json.loads(summary.read_text())["environment"]
    assert metadata["cleanup_completed"] is (not cleanup_failure)
    assert len(metadata["cleanup_errors"]) == (2 if cleanup_failure else 0)
    assert {(row["task"], row["arm"]) for row in rows} == {
        (task, arm) for task in scrape.TASKS for arm in ("playwright-mcp", "playwright-cli")
    }
    assert all(row["ok"] is (not failure) for row in rows)
    if failure:
        assert all(row["detail"] == "native oracle rejected" for row in rows)
    for arm in ("playwright-mcp", "playwright-cli"):
        assert calls.count((arm, "start")) == calls.count((arm, "stop")) == 1
        for task, method in scrape.TASKS.items():
            assert calls.count((arm, "prepare", task)) == calls.count((arm, method)) == 1

    for name in ("playwright-mcp", "playwright-cli"):
        assert calls.count((name, "stop")) == 1
