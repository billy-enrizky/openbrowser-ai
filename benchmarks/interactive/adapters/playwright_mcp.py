"""Native Playwright MCP arm for the interactive and scraping benchmarks.

The arm drives the official ``@playwright/mcp`` server over stdio. The primary
scripted path is one ``browser_run_code_unsafe`` program per task. Downloads
use the native automatic save with the run output directory. Browser launch is
pinned to the benchmark Chrome with an explicit mock keychain, 1280x800
viewport, and the same native config the CLI surface uses.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

from .base import ArmResult, RunContext, terminate_matching
from .mcp_stdio import StdioMcpClient
from .playwright_common import (
    DEFAULT_MCP_BINARY,
    MCP_VERSION,
    PLAYWRIGHT_VERSION,
    assert_no_secret,
    chrome_version,
    keychain,
    last_json,
    result_text,
    wait_for_download,
    write_browser_config,
    write_secrets,
)
from .playwright_tasks import SECRET_PASSWORD, account_program, login_program, task_program

READINESS_PROGRAM = """async (page) => {
  const browser_version = await page.evaluate(() => navigator.userAgent);
  return { browser_version };
}"""

AUTH_READY_PROGRAM = """async (page) => {
  await page.waitForSelector('#secret', { timeout: 15000 });
  return { ready: true };
}"""


class PlaywrightMcpArm:
    name = "playwright-mcp"

    def __init__(
        self,
        binary: str | None = None,
        *,
        secrets_file: Path | None = None,
        browser_version: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.binary = str(binary or os.environ.get("PLAYWRIGHT_MCP_BIN") or DEFAULT_MCP_BINARY)
        self.secrets_file = Path(secrets_file) if secrets_file else None
        self.timeout = timeout
        self.browser_version = browser_version or chrome_version()
        self._result = ArmResult(ok=False)
        self._client: StdioMcpClient | None = None

    # -- native lifecycle --------------------------------------------------
    def _server_command(self, ctx: RunContext) -> list[str]:
        secrets = self.secrets_file or write_secrets(ctx)
        return [
            self.binary,
            "--headless",
            "--user-data-dir",
            str(ctx.profile),
            "--output-dir",
            str(ctx.downloads),
            "--secrets",
            str(secrets),
            "--config",
            str(write_browser_config(ctx)),
            "--viewport-size",
            "1280x800",
            "--codegen",
            "none",
            "--snapshot-mode",
            "none",
        ]

    def _archive_stderr(self, client: StdioMcpClient) -> None:
        lines = getattr(client, "stderr_lines", [])
        archived = int(getattr(client, "_bench_stderr_archived", 0))
        fresh = "\n".join(lines[archived:])
        vars(client)["_bench_stderr_archived"] = len(lines)
        if fresh:
            self._result.stderr = (self._result.stderr + "\n" + fresh).strip("\n")

    def _start(self, ctx: RunContext) -> StdioMcpClient:
        if self._client is not None:
            return self._client
        client = StdioMcpClient(self._server_command(ctx), env=ctx.env, timeout=self.timeout)
        self._client = client
        try:
            client.initialize()
        except Exception:
            self._archive_stderr(client)
            self._client = None
            with contextlib.suppress(Exception):
                client.close()
            self._archive_stderr(client)
            raise
        self._result.process_starts += 1
        return client

    def _call(self, ctx: RunContext, name: str, arguments: dict, *, timeout: float | None = None) -> str:
        client = self._start(ctx)
        self._result.tool_calls += 1
        result = client.call_tool(name, arguments, timeout=timeout or self.timeout)
        text = result_text(result)
        self._result.stdout += text + "\n"
        return text

    def _restart(self, ctx: RunContext) -> None:
        if self._client is not None:
            self._archive_stderr(self._client)
            with contextlib.suppress(Exception):
                self._call(ctx, "browser_close", {}, timeout=30)
            self._archive_stderr(self._client)
            with contextlib.suppress(Exception):
                self._client.close()
            self._archive_stderr(self._client)
            self._client = None
        self._start(ctx)

    # -- benchmark surface -------------------------------------------------
    def warmup(self, ctx: RunContext) -> tuple[dict, float]:
        started = time.monotonic()
        self._result = ArmResult(ok=False)
        self._start(ctx)
        text = self._call(ctx, "browser_run_code_unsafe", {"code": READINESS_PROGRAM})
        payload = last_json(text)
        payload["user_agent"] = payload.pop("browser_version", "")
        payload.update(
            {
                "server": "playwright-mcp",
                "package": MCP_VERSION,
                "playwright": PLAYWRIGHT_VERSION,
                "browser_version": self.browser_version,
            }
        )
        return payload, (time.monotonic() - started) * 1000.0

    def execute_program(self, ctx: RunContext, code: str, *, timeout: float | None = None) -> ArmResult:
        """Run one native program. Seam for the scraping bridge."""
        result = ArmResult(ok=False)
        calls, starts = self._result.tool_calls, self._result.process_starts
        try:
            text = self._call(ctx, "browser_run_code_unsafe", {"code": code}, timeout=timeout)
            result.stdout = text
            result.reported = last_json(text)
            result.browser_version = self.browser_version
            result.ok = True
        except Exception as exc:
            result.error = str(exc)
            result.stdout = self._result.stdout
            result.stderr = self._result.stderr
        result.tool_calls = self._result.tool_calls - calls
        result.process_starts = self._result.process_starts - starts
        return result

    def run(self, ctx: RunContext, *, cleanup: bool = True, restart_for_auth: bool = True) -> ArmResult:
        self._result = ArmResult(ok=False)
        try:
            if ctx.task == "auth":
                self._call(ctx, "browser_run_code_unsafe", {"code": login_program(ctx.base_url)}, timeout=120)
                self._call(
                    ctx,
                    "browser_type",
                    {"element": "Password", "target": "#password", "text": SECRET_PASSWORD, "submit": True},
                    timeout=120,
                )
                self._call(ctx, "browser_run_code_unsafe", {"code": AUTH_READY_PROGRAM}, timeout=120)
                if restart_for_auth:
                    self._restart(ctx)
                text = self._call(ctx, "browser_run_code_unsafe", {"code": account_program(ctx.base_url)}, timeout=120)
                self._result.reported = last_json(text)
            else:
                self._call(
                    ctx,
                    "browser_run_code_unsafe",
                    {"code": task_program(ctx.task, ctx.base_url, ctx.downloads)},
                    timeout=150,
                )
                if ctx.task == "download":
                    path = wait_for_download(ctx.downloads)
                    self._result.reported = {"download": path.name}
                else:
                    self._result.reported = last_json(self._result.stdout)
            self._result.browser_version = self.browser_version
            self._result.keychain_verified = keychain(ctx)
            if self._result.keychain_verified is False:
                raise RuntimeError("Playwright Chrome launched without --use-mock-keychain")
            self._result.ok = True
        except Exception as exc:
            self._result.ok = False
            self._result.error = str(exc)
        finally:
            if cleanup:
                self.cleanup(ctx)
        try:
            assert_no_secret(self._result.stdout, self._result.stderr)
        except RuntimeError as exc:
            self._result.ok = False
            self._result.error = str(exc)
        return self._result

    def cleanup(self, ctx: RunContext) -> None:
        if self._client is not None:
            self._archive_stderr(self._client)
            with contextlib.suppress(Exception):
                self._client.call_tool("browser_close", {}, timeout=30)
            self._archive_stderr(self._client)
            with contextlib.suppress(Exception):
                self._client.close()
            self._archive_stderr(self._client)
            self._client = None
        terminate_matching(str(ctx.profile))
        terminate_matching(str(ctx.downloads))
