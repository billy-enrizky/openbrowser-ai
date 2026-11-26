"""agent-browser 0.43.0 CLI adapter."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..fixtures.server import ACCOUNT_SECRET, BENCH_PASSWORD, BENCH_USER
from ..measurement import browser_launch_checkpoint
from .base import ArmResult, RunContext, matched_chrome, run_process, terminate_matching, verify_mock_keychain

BINARY = os.environ.get(
    "AGENT_BROWSER_BIN",
    str(Path.home() / ".cache" / "openbrowser-interactive-bench" / "npm" / "node_modules" / ".bin" / "agent-browser"),
)


class AgentBrowserArm:
    name = "agent-browser"

    def __init__(self) -> None:
        self.binary = BINARY
        self._result = ArmResult(ok=False)
        self._warm_keychain_verified: bool | None = None

    def _command(self, ctx: RunContext, *args: str) -> list[str]:
        return [
            BINARY,
            "--session",
            ctx.session,
            "--json",
            "--executable-path",
            matched_chrome(),
            "--args",
            "--use-mock-keychain",
            "--profile",
            str(ctx.profile),
            "--download-path",
            str(ctx.downloads),
            *args,
        ]

    def _run(self, ctx: RunContext, *args: str, timeout: float = 120) -> dict:
        command = self._command(ctx, *args)
        env = ctx.env
        env["AGENT_BROWSER_SOCKET_DIR"] = f"/tmp/abib-{ctx.session}"
        returncode, stdout, stderr, _elapsed = run_process(command, env=env, timeout=timeout)
        self._result.process_starts += 1
        if args and args[0] not in {"close"}:
            self._result.tool_calls += 1
        self._result.stdout += stdout
        self._result.stderr += stderr
        payload: dict = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if returncode != 0 or payload.get("success") is False:
            detail = payload.get("error") or stderr.strip() or stdout.strip()
            raise RuntimeError(f"agent-browser {' '.join(args[:2])} failed: {detail[:600]}")
        return payload.get("data") or {}

    def _snapshot(self, ctx: RunContext) -> dict:
        data = self._run(ctx, "snapshot", "-i")
        return data.get("refs") or {}

    def _find(self, refs: dict, *, role: str | None = None, name: str | None = None) -> str:
        for ref, meta in refs.items():
            if role is not None and meta.get("role") != role:
                continue
            if name is not None and meta.get("name") != name:
                continue
            return ref
        raise RuntimeError(f"ref not found role={role} name={name}; refs={list(refs.items())[:30]}")

    def _text(self, ctx: RunContext, selector: str) -> str:
        data = self._run(ctx, "get", "text", selector)
        return str(data.get("text", data.get("value", "")))

    def _title(self, ctx: RunContext) -> str:
        data = self._run(ctx, "get", "title")
        return str(data.get("title", ""))

    def _tabs(self, ctx: RunContext) -> list[dict]:
        data = self._run(ctx, "tab", "list")
        tabs = data.get("tabs") or data.get("items") or []
        return [tab for tab in tabs if isinstance(tab, dict)]

    @staticmethod
    def _tab_id(tab: dict) -> str:
        for key in ("id", "tabId", "tab_id", "targetId", "target_id"):
            value = tab.get(key)
            if value:
                return str(value)
        return ""

    def _run_wizard(self, ctx: RunContext) -> None:
        self._run(ctx, "open", ctx.base_url + "/wizard")
        self._run(ctx, "set", "viewport", "1280", "800")
        self._run(ctx, "fill", "#name", "Alice Bench")
        self._run(ctx, "fill", "#email", "alice@example.com")
        self._run(ctx, "click", "#country-ca")
        self._run(ctx, "wait", "#province-on")
        self._run(ctx, "click", "#province-on")
        self._run(ctx, "click", "#step1-next")
        self._run(ctx, "wait", "#step2")
        self._run(ctx, "click", "#plan-pro")
        self._run(ctx, "fill", "#seats", "5")
        self._run(ctx, "click", "#step2-next")
        self._run(ctx, "wait", "#step3")
        self._run(ctx, "check", "#terms")
        self._run(ctx, "click", "#submit")
        self._run(ctx, "wait", "#quote")
        self._result.reported = {"quote": self._text(ctx, "#quote")}

    def _run_modal(self, ctx: RunContext) -> None:
        self._run(ctx, "open", ctx.base_url + "/modal")
        self._run(ctx, "set", "viewport", "1280", "800")
        self._run(ctx, "click", "#open-modal")
        self._run(ctx, "wait", "#modal")
        self._run(ctx, "press", "Escape")
        self._run(ctx, "wait", "350")
        self._run(ctx, "click", "#open-modal")
        self._run(ctx, "fill", "#project-search", "Atlas")
        self._run(ctx, "wait", "400")
        refs = self._snapshot(ctx)
        self._run(ctx, "click", self._find(refs, role="option", name="Project Atlas"))
        self._run(ctx, "wait", "#selected-value")
        self._result.reported = {"selected": self._text(ctx, "#selected-value")}

    def _run_download(self, ctx: RunContext) -> None:
        self._run(ctx, "open", ctx.base_url + "/download")
        self._run(ctx, "set", "viewport", "1280", "800")
        target = ctx.downloads / "report.csv"
        self._run(ctx, "download", "#download-report", str(target), timeout=120)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if target.exists() and target.stat().st_size > 0:
                self._result.reported = {"path": str(target)}
                return
            files = [path for path in ctx.downloads.rglob("*") if path.is_file()]
            if files:
                self._result.reported = {"files": [str(path) for path in files]}
                return
            time.sleep(0.2)
        raise RuntimeError(f"download command returned but no file appeared in {ctx.downloads}")

    def _run_tabs(self, ctx: RunContext) -> None:
        self._run(ctx, "open", ctx.base_url + "/tabs")
        self._run(ctx, "set", "viewport", "1280", "800")
        before = self._tabs(ctx)
        if not before:
            raise RuntimeError("tab list was empty before opening the detail tab")
        original = self._tab_id(before[0])
        self._run(ctx, "click", "#compare")
        deadline = time.monotonic() + 20
        detail = ""
        while time.monotonic() < deadline:
            tabs = self._tabs(ctx)
            others = [tab for tab in tabs if self._tab_id(tab) != original and "/detail" in str(tab.get("url", ""))]
            if others:
                detail = self._tab_id(others[-1])
                break
            time.sleep(0.15)
        if not detail:
            raise RuntimeError("detail tab did not open")
        self._run(ctx, "tab", detail)
        self._run(ctx, "wait", "#approve")
        self._run(ctx, "click", "#approve")
        self._run(ctx, "wait", "#approved")
        self._run(ctx, "tab", original)
        deadline = time.monotonic() + 20
        status = ""
        while time.monotonic() < deadline:
            status = self._text(ctx, "#status")
            if status == "approved:42":
                break
            time.sleep(0.15)
        self._run(ctx, "tab", "close", detail)
        self._result.reported = {"status": status, "original_title": self._title(ctx)}

    def _run_auth(self, ctx: RunContext) -> None:
        self._run(ctx, "open", ctx.base_url + "/login")
        self._run(ctx, "set", "viewport", "1280", "800")
        self._run(ctx, "fill", "#username", BENCH_USER)
        self._run(ctx, "fill", "#password", BENCH_PASSWORD)
        self._run(ctx, "click", "#login-submit")
        self._run(ctx, "wait", "#secret")
        phase1 = self._text(ctx, "#secret")
        browser_launch_checkpoint(str(ctx.profile))
        self._run(ctx, "close", timeout=60)
        self._run(ctx, "open", ctx.base_url + "/account")
        self._run(ctx, "wait", "#secret")
        secret = self._text(ctx, "#secret")
        self._result.reported = {"secret": secret, "phase1_secret": phase1}
        if secret != ACCOUNT_SECRET:
            raise RuntimeError(f"secret after restart was {secret!r}")

    def warmup(self, ctx: RunContext) -> tuple[dict, float]:
        """Start an agent-browser session once for a warm lane."""
        started = time.monotonic()
        self._result = ArmResult(ok=False)
        self._run(ctx, "open", "about:blank")
        self._run(ctx, "set", "viewport", "1280", "800")
        verified = verify_mock_keychain(str(ctx.profile))
        self._warm_keychain_verified = verified
        if verified is False:
            raise RuntimeError("agent-browser Chrome launched without --use-mock-keychain")
        elapsed_ms = (time.monotonic() - started) * 1000.0
        payload = {"ok": True, "session": ctx.session, "keychain_verified": verified}
        self._result = ArmResult(ok=False)
        return payload, elapsed_ms

    def run(self, ctx: RunContext, *, cleanup: bool = True) -> ArmResult:
        self._result = ArmResult(ok=False)
        try:
            if ctx.task == "wizard":
                self._run_wizard(ctx)
            elif ctx.task == "modal":
                self._run_modal(ctx)
            elif ctx.task == "download":
                self._run_download(ctx)
            elif ctx.task == "tabs":
                self._run_tabs(ctx)
            elif ctx.task == "auth":
                self._run_auth(ctx)
            else:
                raise ValueError(f"unknown task {ctx.task}")
            verified = verify_mock_keychain(str(ctx.profile))
            self._warm_keychain_verified = verified
            self._result.keychain_verified = verified
            if verified is False:
                raise RuntimeError("agent-browser Chrome launched without --use-mock-keychain")
            self._result.ok = True
        except Exception as exc:
            self._result.ok = False
            self._result.error = str(exc)
        finally:
            if cleanup:
                self.cleanup(ctx)
        return self._result

    def cleanup(self, ctx: RunContext) -> None:
        try:
            env = ctx.env
            env["AGENT_BROWSER_SOCKET_DIR"] = f"/tmp/abib-{ctx.session}"
            run_process(self._command(ctx, "close"), env=env, timeout=30)
        except Exception:
            pass
        terminate_matching(str(ctx.profile))
        terminate_matching(str(ctx.root))
