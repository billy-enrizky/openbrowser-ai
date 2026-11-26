"""Shared helpers for the native Playwright MCP and CLI benchmark arms."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from ..fixtures.server import BENCH_PASSWORD, REPORT_SHA256, REPORT_SIZE
from .base import RunContext, matched_chrome, verify_mock_keychain
from .playwright_tasks import SECRET_PASSWORD

MCP_VERSION = "0.0.86"
CLI_VERSION = "0.1.25"
PLAYWRIGHT_VERSION = "1.66.0-alpha-2026-11-18"
CACHE_ROOT = Path.home() / ".cache" / "openbrowser-interactive-bench" / f"playwright-native-{MCP_VERSION}-{CLI_VERSION}"
DEFAULT_MCP_BINARY = CACHE_ROOT / "node_modules" / ".bin" / "playwright-mcp"
DEFAULT_CLI_BINARY = CACHE_ROOT / "node_modules" / ".bin" / "playwright-cli"
SECRETS_ENV = "PLAYWRIGHT_MCP_SECRETS_FILE"
_VERSION_CACHE: dict[str, str] = {}


def chrome_version() -> str:
    """Return the pinned Chrome version once, before any timed observation.

    A harness preflight can export PLAYWRIGHT_BENCH_BROWSER_VERSION so fresh
    resource workers never spawn a metadata probe inside the measured tree.
    """
    configured = os.environ.get("PLAYWRIGHT_BENCH_BROWSER_VERSION")
    if configured:
        return configured
    chrome = matched_chrome()
    if chrome not in _VERSION_CACHE:
        try:
            completed = subprocess.run(
                [chrome, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            value = (completed.stdout or completed.stderr).strip()
        except (OSError, subprocess.TimeoutExpired):
            value = ""
        _VERSION_CACHE[chrome] = value
    return _VERSION_CACHE[chrome]


def write_secrets(ctx: RunContext) -> Path:
    """Write the native secrets file with the exact lookup key."""
    path = ctx.config / "playwright-secrets.env"
    path.write_text(f"{SECRET_PASSWORD}={BENCH_PASSWORD}\n")
    path.chmod(0o600)
    return path


def write_browser_config(ctx: RunContext) -> Path:
    """Write one native config shared by the MCP and CLI surfaces."""
    path = ctx.config / "playwright-config.json"
    path.write_text(
        json.dumps(
            {
                "browser": {
                    "launchOptions": {
                        "executablePath": matched_chrome(),
                        "headless": True,
                        "args": ["--use-mock-keychain"],
                        "downloadsPath": str(ctx.downloads),
                    },
                    "contextOptions": {"viewport": {"width": 1280, "height": 800}},
                },
                "outputDir": str(ctx.downloads),
                "codegen": "none",
                # Pinned playwright-core reads config.timeouts.settle in
                # waitForCompletion. The shared config applies to MCP and CLI;
                # the scripted programs already wait exact readiness.
                "timeouts": {"settle": 0},
            },
            indent=2,
        )
    )
    return path


def result_text(result: dict) -> str:
    if result.get("isError"):
        raise RuntimeError("Playwright tool error: " + text_content(result)[:2000])
    return text_content(result)


def text_content(result: dict) -> str:
    return "\n".join(
        item.get("text", "")
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )


def last_json(text: str) -> dict:
    """Return the last JSON object line in a native tool response."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Playwright tool output did not contain a JSON result")


def wait_for_download(directory: Path, *, timeout: float = 90.0) -> Path:
    """Wait for the native automatic download save to finish exactly."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            candidates = list(directory.iterdir())
        except OSError:
            candidates = []
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) == REPORT_SIZE and hashlib.sha256(data).hexdigest() == REPORT_SHA256:
                return path
        time.sleep(0.05)
    raise TimeoutError(f"report.csv did not complete in {directory}")


def assert_no_secret(*texts: str) -> None:
    """Fail the observation if native output ever contains the run password."""
    for text in texts:
        if BENCH_PASSWORD in text:
            raise RuntimeError("native output leaked the benchmark password")


def keychain(ctx: RunContext) -> bool | None:
    return verify_mock_keychain(str(ctx.profile))
