"""Tests for the local scraping benchmark output and sample order."""

import asyncio
import json
import sys
from pathlib import Path

from benchmarks.scraping import scrape_compare


def test_interleaved_run_writes_each_measured_sample_as_jsonl(monkeypatch, tmp_path: Path):
    class FakeBrowserArm:
        def __init__(self, name, render_mode, engine="auto", chrome_path=None):
            self.name = name

        async def start(self):
            return None

        async def stop(self):
            return None

        async def run_static(self, base):
            return True, "OpenBrowser result", ""

    class FakeFirecrawlServer:
        def __init__(self, **kwargs):
            self.version = "0.1.1"
            self.url = "http://127.0.0.1:3002"
            self.binary = kwargs.get("binary")
            self.chrome_path = kwargs.get("chrome_path")
            self.process = None

        def start(self):
            return None

        def stop(self):
            return None

    class FakeFirecrawlArm:
        def __init__(self, name, server, waits):
            self.name = name

        async def start(self):
            return None

        async def stop(self):
            return None

        async def run_static(self, base):
            return True, "Firecrawl result", ""

    raw_path = tmp_path / "static-results.jsonl"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(scrape_compare, "BrowserArm", FakeBrowserArm)
    monkeypatch.setattr(scrape_compare, "FirecrawlServer", FakeFirecrawlServer)
    monkeypatch.setattr(scrape_compare, "FirecrawlArm", FakeFirecrawlArm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scrape_compare.py",
            "--arms",
            "openbrowser-full",
            "firecrawl-local",
            "--tasks",
            "static_article",
            "--samples",
            "2",
            "--warmups",
            "1",
            "--firecrawl-bin",
            "fake-firecrawl",
            "--interleave",
            "--raw-jsonl",
            str(raw_path),
            "--output",
            str(summary_path),
        ],
    )

    assert asyncio.run(scrape_compare.main()) == 0

    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert records[0]["record_type"] == "run"
    samples = [record for record in records if record["record_type"] == "sample"]
    assert len(samples) == 4
    assert [(row["sample_index"], row["arm"]) for row in samples] == [
        (1, "openbrowser-full"),
        (1, "firecrawl-local"),
        (2, "firecrawl-local"),
        (2, "openbrowser-full"),
    ]
    assert all(row["ok"] for row in samples)
    assert summary_path.exists()
