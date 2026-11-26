"""Native Playwright MCP arm for the interactive and scraping benchmarks.

The arm drives the official ``@playwright/mcp`` server over stdio. The primary
scripted path is one ``browser_run_code_unsafe`` program per task. Downloads
use the native automatic save with ``--output-dir`` set to the run directory.
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
    keychain,
    last_json,
    result_text,
    wait_for_download,
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
        timeout: float = 180.0,
    ) -> None:
        self.binary = str(binary or os.environ.get("PLAYWRIGHT_MCP_BIN") or DEFAULT_MCP_BINARY)
        self.secrets_file = Path(secrets_file) if secrets_file else None
        self.timeout = timeout
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
            "--codegen",
            "none",
            "--snapshot-mode",
            "none",
        ]

    def _start(self, ctx: RunContext) -> StdioMcpClient:
        if self._client is not None:
            return self._client
        client = StdioMcpClient(self._server_command(ctx), env=ctx.env, timeout=self.timeout)
        client.initialize()
        self._result.process_starts += 1
        self._client = client
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
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None
        self._start(ctx)

    # -- benchmark surface -------------------------------------------------
    def warmup(self, ctx: RunContext) -> tuple[dict, float]:
        started = time.monotonic()
        self._result = ArmResult(ok=False)
        self._start(ctx)
        text = self._call(ctx, "browser_run_code_unsafe", {"code": READINESS_PROGRAM})
        payload = last_json(text)
        payload.update({"server": "playwright-mcp", "package": MCP_VERSION, "playwright": PLAYWRIGHT_VERSION})
        return payload, (time.monotonic() - started) * 1000.0

    def execute_program(self, ctx: RunContext, code: str, *, timeout: float | None = None) -> ArmResult:
        """Run one native program. Seam for the scraping bridge."""
        result = ArmResult(ok=False)
        calls, starts = self._result.tool_calls, self._result.process_starts
        try:
            text = self._call(ctx, "browser_run_code_unsafe", {"code": code}, timeout=timeout)
            result.stdout = text
            result.reported = last_json(text)
            result.ok = True
        except Exception as exc:
            result.error = str(exc)
            result.stdout = self._result.stdout
            result.stderr = self._stderr()
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
            self._result.keychain_verified = keychain(ctx)
            if self._result.keychain_verified is False:
                raise RuntimeError("Playwright Chrome launched without --use-mock-keychain")
            assert_no_secret(self._result.stdout, self._result.stderr)
            self._result.ok = True
        except Exception as exc:
            self._result.ok = False
            self._result.error = str(exc)
        finally:
            if cleanup:
                self.cleanup(ctx)
            self._result.stderr = self._stderr()
        return self._result

    def _stderr(self) -> str:
        if self._client is None:
            return ""
        return "\n".join(getattr(self._client, "stderr_lines", []))

    def cleanup(self, ctx: RunContext) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.call_tool("browser_close", {}, timeout=30)
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None
        terminate_matching(str(ctx.profile))
        terminate_matching(str(ctx.downloads))
