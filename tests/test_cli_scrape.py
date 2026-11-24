"""Tests for the first-class `openbrowser-ai scrape` command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from openbrowser.cli import main


def _invoke(
    args,
    *,
    result=None,
    error=None,
    http_result=None,
    http_batch_result=None,
    batch_result=None,
    batch_error=None,
):
    session = MagicMock()
    session.start = AsyncMock()
    session.kill = AsyncMock()
    namespace = {
        "scrape_page": AsyncMock(
            return_value=result
            or {
                "url": "https://example.com/final",
                "title": "Example",
                "format": "markdown",
                "content_type": "text/markdown",
                "content": "# Example",
                "stats": {},
            },
            side_effect=error,
        ),
        "scrape_pages": AsyncMock(return_value=batch_result, side_effect=batch_error),
    }
    profile_cls = MagicMock()
    with (
        patch("openbrowser.browser.BrowserSession", return_value=session),
        patch("openbrowser.browser.BrowserProfile", profile_cls),
        patch("openbrowser.code_use.namespace.create_namespace", return_value=namespace),
        patch("openbrowser.tools.service.CodeAgentTools"),
        patch("openbrowser.logging_config.setup_logging"),
        patch("openbrowser.cli._quiet_stderr_logging"),
        patch(
            "openbrowser.code_use.namespace.scrape_http",
            new_callable=AsyncMock,
            return_value=http_result,
        ),
        patch(
            "openbrowser.scrape.http.scrape_http",
            new_callable=AsyncMock,
            return_value=http_result,
        ),
        patch(
            "openbrowser.scrape.http.scrape_http_batch",
            new_callable=AsyncMock,
            return_value=http_batch_result,
        ),
    ):
        outcome = CliRunner().invoke(main, ["scrape", *args])
    return outcome, session, profile_cls, namespace


def test_scrape_prints_markdown_content():
    outcome, session, _, namespace = _invoke(["https://example.com"])
    assert outcome.exit_code == 0
    assert outcome.output.strip() == "# Example"
    session.start.assert_awaited_once()
    session.kill.assert_awaited_once()
    namespace["scrape_page"].assert_awaited_once()
    assert namespace["scrape_page"].await_args.kwargs["format"] == "markdown"


def test_scrape_lightweight_sets_render_mode():
    _, _, profile_cls, _ = _invoke(["https://example.com", "--lightweight"])
    profile_kwargs = profile_cls.call_args.kwargs
    assert profile_kwargs["render_mode"] == "lightweight"
    assert profile_kwargs["enable_default_extensions"] is False
    assert "storage_state" not in profile_kwargs


def test_scrape_explicit_render_mode_wins():
    _, _, profile_cls, _ = _invoke(["https://example.com", "--lightweight", "--render-mode", "no-render"])
    assert profile_cls.call_args.kwargs["render_mode"] == "no-render"


def test_scrape_json_flag_prints_full_result():
    payload = {
        "url": "https://example.com",
        "title": "Example",
        "format": "json",
        "content_type": "application/json",
        "content": [{"text": "x"}],
        "stats": {},
    }
    outcome, _, _, _ = _invoke(["https://example.com", "--format", "json", "--json"], result=payload)
    assert outcome.exit_code == 0
    assert json.loads(outcome.output) == payload


def test_scrape_failure_exits_one():
    outcome, session, _, _ = _invoke(
        ["https://example.com"],
        error=RuntimeError("navigation failed"),
    )
    assert outcome.exit_code == 1
    assert "Scrape failed" in outcome.output
    session.kill.assert_awaited_once()


def test_scrape_http_path_skips_browser():
    http_result = {
        "url": "https://example.com",
        "title": "Example",
        "format": "markdown",
        "content_type": "text/markdown",
        "content": "# HTTP",
        "stats": {"engine_used": "http"},
        "engine_used": "http",
    }
    outcome, session, profile_cls, namespace = _invoke(
        ["https://example.com", "--engine", "http"], http_result=http_result
    )
    assert outcome.exit_code == 0
    assert outcome.output.strip() == "# HTTP"
    session.start.assert_not_awaited()
    profile_cls.assert_not_called()
    namespace["scrape_page"].assert_not_awaited()


def test_http_only_cli_does_not_import_the_browser_stack():
    source = """
import sys, types
fake = types.ModuleType('openbrowser.browser')
class Boom:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('browser stack imported')
fake.BrowserSession = Boom
fake.BrowserProfile = Boom
sys.modules['openbrowser.browser'] = fake

import openbrowser.scrape.http as http
async def fake_scrape_http(*args, **kwargs):
    return {
        'url': 'https://example.com',
        'title': '',
        'format': 'text',
        'content_type': 'text/text',
        'content': 'fake-http',
        'engine_used': 'http',
    }
http.scrape_http = fake_scrape_http

from click.testing import CliRunner
from openbrowser.cli import main
outcome = CliRunner().invoke(main, ['scrape', 'https://example.com', '--engine', 'http'])
print(outcome.output)
raise SystemExit(outcome.exit_code)
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
    assert "fake-http" in result.stdout


def test_scrape_http_batch_skips_browser():
    batch = [
        {"url": "https://a.example", "ok": True, "engine_used": "http", "content": "# A"},
        {"url": "https://b.example", "ok": True, "engine_used": "http", "content": "# B"},
    ]
    outcome, session, profile_cls, namespace = _invoke(
        ["https://a.example", "https://b.example", "--engine", "http", "--jsonl"],
        http_batch_result=batch,
    )
    assert outcome.exit_code == 0
    assert len(outcome.output.strip().splitlines()) == 2
    session.start.assert_not_awaited()
    profile_cls.assert_not_called()
    namespace["scrape_pages"].assert_not_awaited()


def test_scrape_writes_output_file(tmp_path):
    target = tmp_path / "page.md"
    outcome, _, _, _ = _invoke(["https://example.com", "--output", str(target)])
    assert outcome.exit_code == 0
    assert target.read_text(encoding="utf-8") == "# Example"


def _batch_items():
    return [
        {
            "url": "https://a.example",
            "ok": True,
            "engine_used": "http",
            "content": "A",
            "error": None,
            "duration_ms": 1.0,
            "timing": {"total_ms": 1.0},
        },
        {
            "url": "https://b.example",
            "ok": False,
            "engine_used": "http",
            "content": None,
            "error": "HTTP scrape failed",
            "duration_ms": 2.0,
            "timing": {"total_ms": 2.0},
        },
    ]


def test_scrape_multiple_urls_prints_json_array():
    items = _batch_items()
    outcome, session, _, namespace = _invoke(
        ["https://a.example", "https://b.example"],
        batch_result=items,
    )
    assert outcome.exit_code == 0
    assert json.loads(outcome.output) == items
    namespace["scrape_pages"].assert_awaited_once()
    assert namespace["scrape_pages"].await_args.args[0] == ["https://a.example", "https://b.example"]
    assert namespace["scrape_pages"].await_args.kwargs["concurrency"] == 4
    namespace["scrape_page"].assert_not_awaited()
    session.kill.assert_awaited_once()


def test_scrape_urls_file_combines_with_positional_and_concurrency(tmp_path):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("# comment\nhttps://b.example\n\nhttps://c.example\n", encoding="utf-8")
    items = _batch_items()
    outcome, _, _, namespace = _invoke(
        ["https://a.example", "--urls-file", str(urls_file), "--concurrency", "2"],
        batch_result=items,
    )
    assert outcome.exit_code == 0
    assert namespace["scrape_pages"].await_args.args[0] == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]
    assert namespace["scrape_pages"].await_args.kwargs["concurrency"] == 2


def test_scrape_batch_jsonl_stdout():
    items = _batch_items()
    outcome, _, _, _ = _invoke(
        ["https://a.example", "https://b.example", "--jsonl"],
        batch_result=items,
    )
    assert outcome.exit_code == 0
    lines = [json.loads(line) for line in outcome.output.splitlines() if line.strip()]
    assert lines == items


def test_scrape_batch_infers_jsonl_from_output_suffix(tmp_path):
    target = tmp_path / "results.jsonl"
    items = _batch_items()
    outcome, _, _, _ = _invoke(
        ["https://a.example", "https://b.example", "--output", str(target)],
        batch_result=items,
    )
    assert outcome.exit_code == 0
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == items


def test_scrape_batch_writes_json_array_file(tmp_path):
    target = tmp_path / "results.json"
    items = _batch_items()
    outcome, _, _, _ = _invoke(
        ["https://a.example", "https://b.example", "--output", str(target)],
        batch_result=items,
    )
    assert outcome.exit_code == 0
    assert json.loads(target.read_text(encoding="utf-8")) == items


def test_scrape_batch_requires_a_url():
    outcome, _, _, namespace = _invoke([])
    assert outcome.exit_code == 2
    assert "Provide at least one URL" in outcome.output
    namespace["scrape_pages"].assert_not_awaited()
    namespace["scrape_page"].assert_not_awaited()


def test_scrape_single_url_behavior_is_unchanged():
    outcome, session, _, namespace = _invoke(["https://example.com"])
    assert outcome.exit_code == 0
    assert outcome.output.strip() == "# Example"
    session.start.assert_awaited_once()
    namespace["scrape_page"].assert_awaited_once()
    namespace["scrape_pages"].assert_not_awaited()


def test_scrape_batch_explicit_jsonl_format_overrides_suffix(tmp_path):
    target = tmp_path / "results.json"
    items = _batch_items()
    outcome, _, _, _ = _invoke(
        [
            "https://a.example",
            "https://b.example",
            "--output-format",
            "jsonl",
            "--output",
            str(target),
        ],
        batch_result=items,
    )
    assert outcome.exit_code == 0
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == items


def test_scrape_single_output_format_json_forces_full_object():
    payload = {
        "url": "https://example.com",
        "title": "Example",
        "format": "markdown",
        "content_type": "text/markdown",
        "content": "# Example",
        "stats": {},
    }
    outcome, _, _, _ = _invoke(
        ["https://example.com", "--output-format", "json"],
        result=payload,
    )
    assert outcome.exit_code == 0
    assert json.loads(outcome.output) == payload


def test_scrape_urls_file_only_uses_batch_mode(tmp_path):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://a.example\nhttps://b.example\n", encoding="utf-8")
    items = _batch_items()
    outcome, _, _, namespace = _invoke(
        ["--urls-file", str(urls_file)],
        batch_result=items,
    )
    assert outcome.exit_code == 0
    assert namespace["scrape_pages"].await_args.args[0] == ["https://a.example", "https://b.example"]
    namespace["scrape_page"].assert_not_awaited()


_STATIC_ONLY_SCRIPT = r"""
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class BlockBrowserStack:
    def find_spec(self, name, path=None, target=None):
        if name == 'openbrowser.browser' or name.startswith('openbrowser.browser.'):
            raise ImportError('the browser stack must not load for a static scrape')
        return None


sys.meta_path.insert(0, BlockBrowserStack())


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        return

    def do_GET(self):
        body = (
            b'<html><head><title>Static Only</title></head><body>'
            b'<p>Static content that is long enough to pass the render probe.</p>'
            b'</body></html>'
        )
        self.send_response(200)
        self.send_header('content-type', 'text/html')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = 'http://127.0.0.1:%d/page' % server.server_address[1]

from openbrowser.cli_entry import cli

sys.argv = ['openbrowser-ai', 'scrape', url, '--format', 'markdown']
cli()
print('BROWSER_IMPORTED', 'openbrowser.browser' in sys.modules)
server.shutdown()
"""


def test_static_scrape_does_not_require_the_browser_stack():
    """A static page must scrape over HTTP without importing the browser stack."""
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", _STATIC_ONLY_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "BROWSER_IMPORTED False" in completed.stdout, completed.stdout[-2000:]
    assert "Static content" in completed.stdout
