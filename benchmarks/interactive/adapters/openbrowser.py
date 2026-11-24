"""OpenBrowser CLI daemon adapter for the interactive benchmark."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from ..fixtures.server import BENCH_PASSWORD, BENCH_USER
from .base import (
    ArmResult,
    RunContext,
    kill_pidfile,
    processes_matching,
    run_process,
    terminate_matching,
    verify_mock_keychain,
)

HEADER = """
import asyncio
import json as _json

async def find(role=None, name=None, element_id=None, tag=None):
    summary = await browser.get_browser_state_summary(include_screenshot=False)
    for idx, node in summary.dom_state.selector_map.items():
        ax = node.ax_node
        if element_id and node.attributes.get('id') == element_id:
            return idx
        if role and ax and ax.role == role and (name is None or ax.name == name):
            return idx
        if tag and node.tag_name == tag and (name is None or (ax and ax.name == name)):
            return idx
    raise RuntimeError('element not found role=%s name=%s id=%s' % (role, name, element_id))

async def wait_js(expression, timeout=15.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await evaluate(expression):
            return True
        await asyncio.sleep(0.1)
    raise RuntimeError('timeout waiting for: ' + expression)

async def current_tabs():
    summary = await browser.get_browser_state_summary(include_screenshot=False)
    return [(tab.target_id, tab.url, tab.title) for tab in summary.tabs]
"""


def _wizard(url: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
await input_text(index=await find('textbox', 'Full name'), text='Alice Bench')
await input_text(index=await find('textbox', 'Email'), text='alice@example.com')
await click(index=await find('radio', 'Canada'))
await wait_js("!document.getElementById('province-group').hidden")
await click(index=await find('radio', 'Ontario'))
await click(index=await find('button', 'Next'))
await wait_js("!!document.getElementById('step2')")
await click(index=await find('radio', 'Pro'))
await input_text(index=await find(element_id='seats'), text='5')
await click(index=await find('button', 'Next'))
await wait_js("!!document.getElementById('step3')")
await click(index=await find('checkbox', 'I accept the terms'))
await click(index=await find('button', 'Submit'))
await wait_js("!!document.getElementById('quote')")
print(_json.dumps({{"quote": await evaluate("document.getElementById('quote').textContent")}}))
"""
    )


def _modal(url: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
await click(index=await find('button', 'Choose project'))
await wait_js("!document.getElementById('modal').hidden")
await send_keys(keys='Escape')
await wait_js("document.getElementById('modal').hidden")
await click(index=await find('button', 'Choose project'))
await wait_js("!document.getElementById('modal').hidden")
await input_text(index=await find(element_id='project-search'), text='Atlas')
await wait_js("[...document.querySelectorAll('#project-list [role=option]')].filter(o => !o.hidden).length === 1")
await click(index=await find('option', 'Project Atlas'))
await wait_js("document.getElementById('selected-value').textContent === 'Project Atlas'")
print(_json.dumps({{"selected": await evaluate("document.getElementById('selected-value').textContent")}}))
"""
    )


def _download(url: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
waiter = asyncio.create_task(wait_for_download(timeout=90))
await click(index=await find('button', 'Download report'))
outcome = await waiter
print(_json.dumps({{"download": outcome}}))
"""
    )


def _tabs(url: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
summary = await browser.get_browser_state_summary()
original = summary.tabs[0].target_id if summary.tabs else None
await click(index=await find('link', 'Compare plans'))
deadline = asyncio.get_event_loop().time() + 20
detail = None
while asyncio.get_event_loop().time() < deadline:
    tabs = await current_tabs()
    others = [tab for tab in tabs if tab[0] != original]
    if others:
        detail = others[0][0]
        break
    await asyncio.sleep(0.1)
if detail is None:
    raise RuntimeError('detail tab did not open')
await switch(tab_id=detail)
await wait_js("!!document.getElementById('approve')")
await click(index=await find('button', 'Approve'))
await wait_js("document.getElementById('approved').textContent === 'approved'")
await close(tab_id=detail)
await switch(tab_id=original)
await wait_js("document.getElementById('status').textContent === 'approved:42'")
print(_json.dumps({{
    "status": await evaluate("document.getElementById('status').textContent"),
    "original_title": await evaluate("document.title"),
}}))
"""
    )


def _login(url: str, state_path: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
await input_text(index=await find('textbox', 'Username'), text={BENCH_USER!r})
await input_text(index=await find(element_id='password'), text={BENCH_PASSWORD!r})
await click(index=await find('button', 'Sign in'))
await wait_js("!!document.getElementById('secret')")
secret = await evaluate("document.getElementById('secret').textContent")
await browser.export_storage_state({state_path!r})
print(_json.dumps({{"secret": secret}}))
"""
    )


def _read_account(url: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
await wait_js("!!document.getElementById('secret')")
print(_json.dumps({{"secret": await evaluate("document.getElementById('secret').textContent")}}))
"""
    )


def _write_config(ctx: RunContext, *, extensions: bool) -> None:
    config = {
        "browser_profile": {
            "bench": {
                "id": "bench",
                "default": True,
                "headless": True,
                "channel": "chrome",
                "window_size": {"width": 1280, "height": 800},
                "downloads_path": str(ctx.downloads),
                "wait_between_actions": 0.0,
                "args": ["--use-mock-keychain"],
                "enable_default_extensions": extensions,
            }
        },
        "llm": {"bench": {"id": "bench", "default": True}},
        "agent": {"bench": {"id": "bench", "default": True}},
    }
    if ctx.task == "auth":
        state_path = ctx.root / "storage_state.json"
        state_path.write_text('{"cookies": [], "origins": []}')
        config["browser_profile"]["bench"]["storage_state"] = str(state_path)
    ctx.config.mkdir(parents=True, exist_ok=True)
    (ctx.config / "config.json").write_text(json.dumps(config))


def _env(ctx: RunContext) -> dict[str, str]:
    env = ctx.env
    env.update(
        {
            "OPENBROWSER_CONFIG_DIR": str(ctx.config),
            "OPENBROWSER_SOCKET": str(ctx.home / ".openbrowser" / "daemon.sock"),
            "OPENBROWSER_LOGGING_LEVEL": "result",
            "OPENBROWSER_SETUP_LOGGING": "false",
            "OPENBROWSER_HEADLESS": "true",
        }
    )
    return env


class OpenBrowserArm:
    name = "openbrowser"

    def __init__(self, binary: str | None = None, *, extensions: bool = False) -> None:
        self.binary = binary or os.environ.get("OPENBROWSER_BIN") or _default_binary()
        self.extensions = extensions
        self._result = ArmResult(ok=False)

    def _invoke(self, ctx: RunContext, code: str, timeout: float) -> tuple[int, str, str]:
        returncode, stdout, stderr, _elapsed = run_process(
            [self.binary, "-c", code],
            env=_env(ctx),
            timeout=timeout,
        )
        self._result.process_starts += 1
        self._result.tool_calls += 1
        self._result.stdout += stdout
        self._result.stderr += stderr
        if returncode != 0:
            raise RuntimeError(f"openbrowser-ai exited {returncode}: {stderr.strip()[:600]}")
        return returncode, stdout, stderr

    def _parse_last_json(self) -> dict:
        for line in reversed(self._result.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {}

    def run(
        self,
        ctx: RunContext,
        *,
        cleanup: bool = True,
        restart_for_auth: bool = True,
    ) -> ArmResult:
        self._result = ArmResult(ok=False)
        _write_config(ctx, extensions=self.extensions)
        url = ctx.base_url
        try:
            if ctx.task == "wizard":
                self._invoke(ctx, _wizard(url + "/wizard"), 120)
            elif ctx.task == "modal":
                self._invoke(ctx, _modal(url + "/modal"), 120)
            elif ctx.task == "download":
                self._invoke(ctx, _download(url + "/download"), 120)
            elif ctx.task == "tabs":
                self._invoke(ctx, _tabs(url + "/tabs"), 120)
            elif ctx.task == "auth":
                self._invoke(ctx, _login(url + "/login", str(ctx.root / "storage_state.json")), 120)
                if restart_for_auth:
                    self._stop_daemon(ctx)
                self._invoke(ctx, _read_account(url + "/account"), 120)
            else:
                raise ValueError(f"unknown task {ctx.task}")
            verified = verify_mock_keychain(str(ctx.config / "profiles" / "daemon"))
            self._result.keychain_verified = verified
            if verified is False:
                raise RuntimeError("OpenBrowser Chrome launched without --use-mock-keychain")
            self._result.reported = self._parse_last_json()
            self._result.ok = True
        except Exception as exc:
            self._result.ok = False
            self._result.error = str(exc)
        finally:
            if cleanup:
                self.cleanup(ctx)
        return self._result

    def warmup(self, ctx: RunContext) -> tuple[dict, float]:
        """Start the daemon and browser once and return its JSON plus wall time."""
        started = time.monotonic()
        returncode, stdout, stderr, _elapsed = run_process(
            [self.binary, "warm"],
            env=_env(ctx),
            timeout=180,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if returncode != 0:
            raise RuntimeError(f"openbrowser-ai warm exited {returncode}: {stderr.strip()[:600]}")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"openbrowser-ai warm returned invalid JSON: {stdout[:600]}") from exc
        return payload, elapsed_ms

    def _stop_daemon(self, ctx: RunContext) -> None:
        pid_file = ctx.home / ".openbrowser" / "daemon.pid"
        socket_file = ctx.home / ".openbrowser" / "daemon.sock"
        try:
            run_process([self.binary, "daemon", "stop"], env=_env(ctx), timeout=60)
        except Exception:
            pass
        profile_marker = str(ctx.config / "profiles" / "daemon")
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            chrome_alive = processes_matching(profile_marker)
            if not pid_file.exists() and not socket_file.exists() and not chrome_alive:
                return
            time.sleep(0.2)
        try:
            run_process([self.binary, "daemon", "force-stop"], env=_env(ctx), timeout=30)
        except Exception:
            pass
        kill_pidfile(pid_file)
        terminate_matching(profile_marker)
        time.sleep(0.5)

    def cleanup(self, ctx: RunContext) -> None:
        try:
            run_process([self.binary, "daemon", "force-stop"], env=_env(ctx), timeout=30)
        except Exception:
            pass
        kill_pidfile(ctx.home / ".openbrowser" / "daemon.pid")
        terminate_matching(str(ctx.profile))
        terminate_matching(str(ctx.root))


def _default_binary() -> str:
    candidates = [
        Path.home() / ".cache" / "openbrowser-interactive-bench" / "openbrowser-env" / "bin" / "openbrowser-ai",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which("openbrowser-ai")
    if found:
        return found
    raise FileNotFoundError("openbrowser-ai not found; set OPENBROWSER_BIN")
