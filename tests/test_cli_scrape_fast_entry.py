"""Fast entry path for HTTP-only scrape commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_fast_entry_single_http_scrape(monkeypatch, capsys):
    import openbrowser.cli_entry as entry

    calls: dict = {}

    def fake_run(urls, **kwargs):
        calls["urls"] = urls
        calls.update(kwargs)
        return {"url": urls[0], "content": "fast-http", "format": "text", "engine_used": "http"}

    monkeypatch.setattr(entry, "run_http_scrape_sync", fake_run)
    monkeypatch.setattr(entry.runpy, "run_module", lambda *a, **k: pytest.fail("fell back to Click"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["openbrowser-ai", "scrape", "https://a.example", "--engine", "http", "--format", "text"],
    )
    entry.cli()
    assert capsys.readouterr().out.strip() == "fast-http"
    assert calls["urls"] == ["https://a.example"]
    assert calls["batch_mode"] is False


def test_fast_entry_batch_jsonl(monkeypatch, capsys):
    import openbrowser.cli_entry as entry

    def fake_run(urls, **kwargs):
        return [
            {"url": url, "ok": True, "content": f"content-{index}", "engine_used": "http"}
            for index, url in enumerate(urls)
        ]

    monkeypatch.setattr(entry, "run_http_scrape_sync", fake_run)
    monkeypatch.setattr(entry.runpy, "run_module", lambda *a, **k: pytest.fail("fell back to Click"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["openbrowser-ai", "scrape", "https://a.example", "https://b.example", "--engine", "http", "--jsonl"],
    )
    entry.cli()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert '"https://a.example"' in lines[0]
    assert '"https://b.example"' in lines[1]


def test_unknown_option_falls_back_to_click():
    import openbrowser.cli_entry as entry

    assert (
        entry._execute_scrape_fast_path(["scrape", "https://a.example", "--engine", "http", "--future-option", "x"])
        is False
    )


def test_missing_http_engine_falls_back_to_click():
    import openbrowser.cli_entry as entry

    assert entry._execute_scrape_fast_path(["scrape", "https://a.example"]) is False
    assert entry._execute_scrape_fast_path(["scrape", "https://a.example", "--engine", "auto"]) is False


def test_fast_entry_reads_urls_file(monkeypatch, tmp_path, capsys):
    import openbrowser.cli_entry as entry

    path = tmp_path / "urls.txt"
    path.write_text("# comment\nhttps://a.example\n\nhttps://b.example\n")
    captured: dict = {}

    def fake_run(urls, **kwargs):
        captured["urls"] = urls
        return [{"url": url, "ok": True, "content": "x"} for url in urls]

    monkeypatch.setattr(entry, "run_http_scrape_sync", fake_run)
    monkeypatch.setattr(entry.runpy, "run_module", lambda *a, **k: pytest.fail("fell back to Click"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["openbrowser-ai", "scrape", "--urls-file", str(path), "--engine", "http", "--jsonl"],
    )
    entry.cli()
    assert captured["urls"] == ["https://a.example", "https://b.example"]
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_fast_entry_does_not_import_the_browser_stack():
    source = """
import sys, types
fake = types.ModuleType('openbrowser.browser')
class Boom:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('browser stack imported')
fake.BrowserSession = Boom
fake.BrowserProfile = Boom
sys.modules['openbrowser.browser'] = fake

import openbrowser.cli_entry as entry
def fake_run(urls, **kwargs):
    return {'url': urls[0], 'content': 'fast-http', 'format': 'text', 'engine_used': 'http'}
entry.run_http_scrape_sync = fake_run
entry.runpy.run_module = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('fallback'))
sys.argv = ['openbrowser-ai', 'scrape', 'https://a.example', '--engine', 'http', '--format', 'text']
entry.cli()
"""
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fast-http"
