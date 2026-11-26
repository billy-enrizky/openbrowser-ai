"""Shared helpers for the native Playwright MCP and CLI benchmark arms."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..fixtures.server import BENCH_PASSWORD, REPORT_SHA256, REPORT_SIZE
from .base import RunContext, verify_mock_keychain

MCP_VERSION = "0.0.86"
CLI_VERSION = "0.1.25"
PLAYWRIGHT_VERSION = "1.66.0-alpha-2026-11-18"
CACHE_ROOT = Path.home() / ".cache" / "openbrowser-interactive-bench" / f"playwright-native-{MCP_VERSION}-{CLI_VERSION}"
DEFAULT_MCP_BINARY = CACHE_ROOT / "node_modules" / ".bin" / "playwright-mcp"
DEFAULT_CLI_BINARY = CACHE_ROOT / "node_modules" / ".bin" / "playwright-cli"
SECRETS_ENV = "PLAYWRIGHT_MCP_SECRETS_FILE"


def write_secrets(ctx: RunContext) -> Path:
    """Write the native secrets file for the run, readable only by the owner."""
    path = ctx.config / "playwright-secrets.env"
    path.write_text(f"BENCH_PASSWORD={BENCH_PASSWORD}\n")
    path.chmod(0o600)
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
