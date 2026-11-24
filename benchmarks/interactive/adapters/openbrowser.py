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
    return [(tab["target_id"], tab["url"], tab["title"]) for tab in await get_tabs()]
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
tabs = await get_tabs()
original = next((tab["target_id"] for tab in tabs if tab["active"]), tabs[0]["target_id"] if tabs else None)
await click(index=await find('link', 'Compare plans'))

async def open_detail():
    deadline = asyncio.get_event_loop().time() + 20
    last_error = None
    while asyncio.get_event_loop().time() < deadline:
        tabs = await current_tabs()
        detail_tabs = [tab for tab in reversed(tabs) if tab[0] != original and "/detail" in tab[1]]
        for tab in detail_tabs:
            try:
                await switch(tab_id=tab[0])
            except RuntimeError as error:
                last_error = error
                if 'closed' not in str(error).lower():
                    raise
                continue
            await asyncio.sleep(0.15)
            return tab[0]
        await asyncio.sleep(0.1)
    raise RuntimeError('detail tab did not open: %s' % last_error)

detail = await open_detail()
await wait_js("!!document.getElementById('approve')")
for attempt in range(3):
    try:
        await evaluate("document.getElementById('approve').click()")
        await wait_js("document.getElementById('approved').textContent === 'approved'", timeout=5)
        break
    except RuntimeError:
        if attempt == 2:
            raise
        detail = await open_detail()
        await wait_js("!!document.getElementById('approve')")
await close(tab_id=detail)
await switch(tab_id=original)
await wait_js("document.getElementById('status').textContent === 'approved:42'")
print(_json.dumps({{
    "status": await evaluate("document.getElementById('status').textContent"),
    "original_title": await evaluate("document.title"),
}}))
"""
    )


def _fast_wizard(url: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
await fill_selector('#name', 'Alice Bench')
await fill_selector('#email', 'alice@example.com')
await click_selector('#country-ca')
await wait_for_selector('#province-group:not([hidden])', timeout=15)
await click_selector('#province-on')
await click_selector('#step1-next')
await wait_for_selector('#step2', timeout=15)
await click_selector('#plan-pro')
await fill_selector('#seats', '5')
await click_selector('#step2-next')
await wait_for_selector('#step3', timeout=15)
await click_selector('#terms')
await click_selector('#submit')
await wait_for_selector('#quote', timeout=15)
print(_json.dumps({{"quote": await evaluate("document.getElementById('quote').textContent")}}))
"""
    )


def _fast_modal(url: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
await click_selector('#open-modal')
await wait_for_selector('#modal:not([hidden])')
await send_keys(keys='Escape')
await wait_for_selector('#modal[hidden]')
await click_selector('#open-modal')
await fill_selector('#project-search', 'Atlas')
await wait_js("[...document.querySelectorAll('#project-list [role=option]')].filter(o => !o.hidden).length === 1")
option_id = await evaluate("Array.from(document.querySelectorAll('#project-list [role=option]')).find(o => !o.hidden && o.textContent.trim() === 'Project Atlas')?.id")
if not option_id:
    raise RuntimeError('Project Atlas option has no stable fixture id')
await click_selector('#' + option_id)
await wait_js("document.getElementById('selected-value').textContent === 'Project Atlas'")
print(_json.dumps({{"selected": await evaluate("document.getElementById('selected-value').textContent")}}))
"""
    )


def _fast_download(url: str) -> str:
    return (
        HEADER
        + f"""
await navigate({url!r})
waiter = asyncio.create_task(wait_for_download(timeout=90))
await click_selector('#download-report')
outcome = await waiter
print(_json.dumps({{"download": outcome}}))
"""
    )


def _fast_tabs(url: str) -> str:
    return (
        HEADER
        + f"""
_step_ms = {{}}

async def _time_step(name, operation):
    started = asyncio.get_running_loop().time()
    try:
        return await operation
    finally:
        _step_ms[name] = _step_ms.get(name, 0.0) + (asyncio.get_running_loop().time() - started) * 1000

await _time_step('navigate', navigate({url!r}))
tabs = await _time_step('enumerate_initial_tabs', get_tabs())
original = next((tab["target_id"] for tab in tabs if tab["active"]), tabs[0]["target_id"] if tabs else None)

async def open_detail():
    deadline = asyncio.get_event_loop().time() + 20
    last_error = None
    while asyncio.get_event_loop().time() < deadline:
        tabs = await current_tabs()
        detail_tabs = [tab for tab in reversed(tabs) if tab[0] != original and "/detail" in tab[1]]
        for tab in detail_tabs:
            try:
                await switch(tab_id=tab[0])
            except RuntimeError as error:
                last_error = error
                if 'closed' not in str(error).lower():
                    raise
                continue
            await asyncio.sleep(0.15)
            return tab[0]
        await asyncio.sleep(0.1)
    raise RuntimeError('detail tab did not open: %s' % last_error)

detail = await _time_step(
    'click_selector_and_wait_for_new_tab',
    click_selector_and_wait_for_new_tab('#compare'),
)
await _time_step('switch_to_detail', switch(tab_id=detail))
await _time_step('wait_for_approve', wait_for_selector('#approve', timeout=15))
for attempt in range(3):
    try:
        await _time_step('click_approve', click_selector('#approve'))
        await _time_step(
            'wait_for_approval',
            wait_js("document.getElementById('approved').textContent === 'approved'", timeout=5),
        )
        break
    except RuntimeError:
        if attempt == 2:
            raise
        detail = await open_detail()
        await wait_for_selector('#approve', timeout=15)
await _time_step('close_detail', close(tab_id=detail))
await _time_step('switch_to_original', switch(tab_id=original))
await _time_step(
    'wait_for_original_status',
    wait_js("document.getElementById('status').textContent === 'approved:42'"),
)
print(_json.dumps({{
    "status": await _time_step('read_status', evaluate("document.getElementById('status').textContent")),
    "original_title": await _time_step('read_original_title', evaluate("document.title")),
    "step_ms": {{name: round(value, 3) for name, value in _step_ms.items()}},
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

    def __init__(
        self,
        binary: str | None = None,
        *,
        extensions: bool = False,
        fast_selectors: bool = False,
    ) -> None:
        self.binary = binary or os.environ.get("OPENBROWSER_BIN") or _default_binary()
        self.extensions = extensions
        self.fast_selectors = fast_selectors
        self.name = "openbrowser-fast" if fast_selectors else "openbrowser"
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
            raise RuntimeError(
                f"openbrowser-ai exited {returncode}: {stderr.strip()[:2500]} stdout={stdout.strip()[:1500]}"
            )
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
                program = _fast_wizard(url + "/wizard") if self.fast_selectors else _wizard(url + "/wizard")
                self._invoke(ctx, program, 120)
            elif ctx.task == "modal":
                program = _fast_modal(url + "/modal") if self.fast_selectors else _modal(url + "/modal")
                self._invoke(ctx, program, 120)
            elif ctx.task == "download":
                program = _fast_download(url + "/download") if self.fast_selectors else _download(url + "/download")
                self._invoke(ctx, program, 120)
            elif ctx.task == "tabs":
                program = _fast_tabs(url + "/tabs") if self.fast_selectors else _tabs(url + "/tabs")
                self._invoke(ctx, program, 120)
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
