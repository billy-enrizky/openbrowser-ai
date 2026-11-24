from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.scraping.scrape_compare import LightpandaArm, main


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _arm(*responses):
    with patch("benchmarks.scraping.scrape_compare.subprocess.run", side_effect=list(responses)) as run:
        arm = LightpandaArm("/tmp/lightpanda")
    return arm, run


def test_lightpanda_arm_records_the_version():
    arm, run = _arm(_completed("1.0.5\n"))

    assert arm.version == "1.0.5"
    assert run.call_args.args[0] == ["/tmp/lightpanda", "version"]


def test_lightpanda_fetch_uses_wait_script_and_parses_content():
    arm, _ = _arm(_completed("1.0.5\n"))
    with patch(
        "benchmarks.scraping.scrape_compare.subprocess.run",
        return_value=_completed(json.dumps({"error": None, "content": "rendered"})),
    ) as run:
        content = arm._fetch(
            "http://127.0.0.1:1/js",
            "markdown",
            wait_script="document.querySelectorAll('#items').length === 1",
        )

    assert content == "rendered"
    command = run.call_args.args[0]
    assert "--wait-script" in command
    assert "--wait-ms" not in command


def test_lightpanda_fetch_reports_payload_errors():
    arm, _ = _arm(_completed("1.0.5\n"))
    with (
        patch(
            "benchmarks.scraping.scrape_compare.subprocess.run",
            return_value=_completed(json.dumps({"error": "navigation", "content": ""})),
        ),
        pytest.raises(RuntimeError, match="navigation"),
    ):
        arm._fetch("http://127.0.0.1:1/", "markdown", wait_ms=100)


@pytest.mark.asyncio
async def test_setup_failure_stops_firecrawl_and_the_fixture_server(monkeypatch):
    fixture = MagicMock()
    fixture.base_url = "http://127.0.0.1:1"
    fixture.state = SimpleNamespace()
    firecrawl = MagicMock()
    firecrawl.url = "http://127.0.0.1:2"
    firecrawl.version = "0.1.1"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scrape_compare.py",
            "--arms",
            "firecrawl-local",
            "lightpanda",
            "--samples",
            "1",
            "--warmups",
            "0",
            "--firecrawl-bin",
            "/tmp/firecrawl",
            "--lightpanda-bin",
            "/tmp/lightpanda",
        ],
    )
    with (
        patch("benchmarks.scraping.scrape_compare.FixtureServer", return_value=fixture),
        patch("benchmarks.scraping.scrape_compare.FirecrawlServer", return_value=firecrawl),
        patch("benchmarks.scraping.scrape_compare._resolve_firecrawl_binary", return_value="/tmp/firecrawl"),
        patch("benchmarks.scraping.scrape_compare.LightpandaArm", side_effect=SystemExit("missing")),
        pytest.raises(SystemExit, match="missing"),
    ):
        await main()

    firecrawl.start.assert_called_once()
    firecrawl.stop.assert_called_once()
    fixture.shutdown.assert_called_once()
    fixture.server_close.assert_called_once()
