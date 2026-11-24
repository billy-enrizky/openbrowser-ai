from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.scraping.scrape_compare import LightpandaArm


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
