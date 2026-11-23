"""Tests for the first-class `openbrowser-ai scrape` command."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from openbrowser.cli import main


def _invoke(args, *, result=None, error=None, http_result=None):
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
        )
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


def test_scrape_writes_output_file(tmp_path):
    target = tmp_path / "page.md"
    outcome, _, _, _ = _invoke(["https://example.com", "--output", str(target)])
    assert outcome.exit_code == 0
    assert target.read_text(encoding="utf-8") == "# Example"
