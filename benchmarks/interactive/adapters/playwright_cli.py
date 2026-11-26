"""Native Playwright CLI arm for the interactive and scraping benchmarks.

The arm drives the official ``@playwright/cli`` session. The primary scripted
path is one ``run-code --filename`` program per task. Downloads use the native
automatic save with the run output directory. Browser launch is pinned to the
benchmark Chrome with an explicit mock keychain, a 1280x800 viewport, and the
same native config the MCP surface uses.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

from ..measurement import browser_launch_checkpoint
from .base import ArmResult, RunContext, run_process, terminate_matching
from .playwright_common import (
    CLI_VERSION,
    DEFAULT_CLI_BINARY,
    PLAYWRIGHT_VERSION,
    SECRETS_ENV,
    assert_no_secret,
    chrome_version,
    keychain,
    last_json,
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


class PlaywrightCliArm:
    name = "playwright-cli"

    def __init__(
        self,
        binary: str | None = None,
        *,
        secrets_file: Path | None = None,
        browser_version: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.binary = str(binary or os.environ.get("PLAYWRIGHT_CLI_BIN") or DEFAULT_CLI_BINARY)
        self.secrets_file = Path(secrets_file) if secrets_file else None
        self.timeout = timeout
        self.browser_version = browser_version or chrome_version()
        self._result = ArmResult(ok=False)
        self._keychain_verified: bool | None = None
        self._opened = False
        self._attempted_open = False
        self._config_path: Path | None = None
        self._program_index = 0

    # -- native lifecycle --------------------------------------------------
    def _config(self, ctx: RunContext) -> Path:
        if self._config_path is None:
            self._config_path = write_browser_config(ctx)
        return self._config_path

    def _run(self, ctx: RunContext, *args: str, timeout: float | None = None) -> str:
        env = ctx.env
        env[SECRETS_ENV] = str(self.secrets_file or write_secrets(ctx))
        command = [self.binary, "-s", ctx.session, *args, "--raw"]
        returncode, stdout, stderr, _elapsed = run_process(
            command, env=env, cwd=ctx.root, timeout=timeout or self.timeout
        )
        self._result.stdout += stdout
        self._result.stderr += stderr
        self._result.process_starts += 1
        if args[0] not in {"open", "close"}:
            self._result.tool_calls += 1
        if returncode != 0:
            detail = (stderr.strip() or stdout.strip())[-800:]
            raise RuntimeError(f"playwright-cli {' '.join(args[:2])} failed: {detail}")
        return stdout

    def _open(self, ctx: RunContext) -> None:
        if self._opened:
            return
        self._attempted_open = True
        self._keychain_verified = None
        self._run(
            ctx,
            "open",
            "--profile",
            str(ctx.profile),
            "--config",
            str(self._config(ctx)),
            timeout=120,
        )
        self._opened = True

    def _program_file(self, ctx: RunContext, code: str) -> Path:
        self._program_index += 1
        path = ctx.config / f"program-{self._program_index:02}.js"
        path.write_text(code)
        return path

    def _code(self, ctx: RunContext, code: str, *, timeout: float | None = None) -> str:
        self._open(ctx)
        return self._run(
            ctx,
            "run-code",
            "--filename",
            str(self._program_file(ctx, code)),
            timeout=timeout or self.timeout,
        )

    def _restart(self, ctx: RunContext) -> None:
        self._keychain_verified = None
        if self._opened or self._attempted_open:
            with contextlib.suppress(Exception):
                self._run(ctx, "close", timeout=60)
            self._opened = False
            self._attempted_open = False
        self._open(ctx)

    def _ensure_keychain(self, ctx: RunContext) -> bool:
        """Verify the mock keychain once per launched browser."""
        if self._keychain_verified is True:
            return True
        value = keychain(ctx)
        if value is None:
            raise RuntimeError("Playwright keychain verification unavailable")
        if value is not True:
            raise RuntimeError("Playwright Chrome launched without --use-mock-keychain")
        self._keychain_verified = True
        return True

    # -- benchmark surface -------------------------------------------------
    def warmup(self, ctx: RunContext) -> tuple[dict, float]:
        started = time.monotonic()
        self._result = ArmResult(ok=False)
        try:
            text = self._code(ctx, READINESS_PROGRAM)
        except Exception:
            assert_no_secret(self._result.stdout, self._result.stderr)
            raise
        payload = last_json(text)
        payload["user_agent"] = payload.pop("browser_version", "")
        payload.update(
            {
                "server": "playwright-cli",
                "package": CLI_VERSION,
                "playwright": PLAYWRIGHT_VERSION,
                "browser_version": self.browser_version,
            }
        )
        self._ensure_keychain(ctx)
        payload["keychain_verified"] = True
        assert_no_secret(self._result.stdout, self._result.stderr)
        return payload, (time.monotonic() - started) * 1000.0

    def execute_program(self, ctx: RunContext, code: str, *, timeout: float | None = None) -> ArmResult:
        """Run one native program. Seam for the scraping bridge."""
        result = ArmResult(ok=False)
        if self._keychain_verified is not True:
            result.error = "Playwright launch readiness was not verified; call warmup first"
            return result
        calls, starts = self._result.tool_calls, self._result.process_starts
        try:
            text = self._code(ctx, code, timeout=timeout)
            result.stdout = text
            result.stderr = self._result.stderr
            result.reported = last_json(text)
            result.browser_version = self.browser_version
            assert_no_secret(result.stdout, result.stderr)
            result.keychain_verified = True
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
                self._code(ctx, login_program(ctx.base_url), timeout=120)
                self._open(ctx)
                self._run(
                    ctx,
                    "fill",
                    "#password",
                    SECRET_PASSWORD,
                    "--submit",
                    timeout=120,
                )
                self._code(ctx, AUTH_READY_PROGRAM, timeout=120)
                if restart_for_auth:
                    browser_launch_checkpoint(str(ctx.profile))
                    self._restart(ctx)
                text = self._code(ctx, account_program(ctx.base_url), timeout=120)
                self._result.reported = last_json(text)
            else:
                self._code(ctx, task_program(ctx.task, ctx.base_url, ctx.downloads), timeout=150)
                if ctx.task == "download":
                    path = wait_for_download(ctx.downloads)
                    self._result.reported = {"download": path.name}
                else:
                    self._result.reported = last_json(self._result.stdout)
            self._result.browser_version = self.browser_version
            self._result.keychain_verified = self._ensure_keychain(ctx)
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
        if self._opened or self._attempted_open:
            with contextlib.suppress(Exception):
                self._run(ctx, "close", timeout=60)
            self._opened = False
            self._attempted_open = False
        self._keychain_verified = None
        terminate_matching(str(ctx.profile))
        terminate_matching(str(ctx.downloads))
