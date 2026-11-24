"""browser-use 1.0.8 CLI adapter in an isolated uv environment."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..fixtures.server import ACCOUNT_SECRET, BENCH_PASSWORD, BENCH_USER
from .base import (
    ArmResult,
    ChromeInstance,
    RunContext,
    kill_pidfile,
    run_process,
    terminate_matching,
    verify_mock_keychain,
)

CHROME = os.environ.get(
    "CHROME_BIN",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
BINARY = os.environ.get(
    "BROWSER_USE_BIN",
    str(Path.home() / ".cache" / "openbrowser-interactive-bench" / "browser-use-venv" / "bin" / "browser-use"),
)

HEADER = """
import json as _json
import time as _time

_calls = 0

def _tick():
    global _calls
    _calls += 1

def wait_js(expression, timeout=20.0):
    _tick()
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        if js(expression):
            return True
        _time.sleep(0.1)
    raise RuntimeError('timeout waiting for: ' + expression)

def emit(payload):
    payload['tool_calls'] = _calls
    print('BENCH_RESULT ' + _json.dumps(payload))

def close_current():
    try:
        _tick(); close_tab(current_tab()['targetId'])
    except Exception:
        pass
"""


def _wizard(url: str) -> str:
    return (
        HEADER
        + f"""
_tick(); new_tab({url!r})
_tick(); fill_input('#name', 'Alice Bench')
_tick(); fill_input('#email', 'alice@example.com')
_tick(); js("document.querySelector('#country-ca').click()")
wait_js("!document.getElementById('province-group').hidden")
_tick(); js("document.querySelector('#province-on').click()")
_tick(); js("document.querySelector('#step1-next').click()")
wait_js("!!document.getElementById('step2')")
_tick(); js("document.querySelector('#plan-pro').click()")
_tick(); fill_input('#seats', '5')
_tick(); js("document.querySelector('#step2-next').click()")
wait_js("!!document.getElementById('step3')")
_tick(); js("document.querySelector('#terms').click()")
_tick(); js("document.querySelector('#submit').click()")
wait_js("!!document.getElementById('quote')")
quote = js("document.getElementById('quote').textContent")
_tick(); close_current()
emit({{'quote': quote}})
"""
    )


def _modal(url: str) -> str:
    return (
        HEADER
        + f"""
_tick(); new_tab({url!r})
_tick(); js("document.querySelector('#open-modal').click()")
wait_js("!document.getElementById('modal').hidden")
_tick(); press_key('Escape')
wait_js("document.getElementById('modal').hidden")
_tick(); js("document.querySelector('#open-modal').click()")
wait_js("!document.getElementById('modal').hidden")
_tick(); fill_input('#project-search', 'Atlas')
wait_js("[...document.querySelectorAll('#project-list [role=option]')].filter(o => !o.hidden).length === 1")
_tick(); js("document.querySelector('[data-value=\\"Project Atlas\\"]').click()")
wait_js("document.getElementById('selected-value').textContent === 'Project Atlas'")
selected = js("document.getElementById('selected-value').textContent")
_tick(); close_current()
emit({{'selected': selected}})
"""
    )


def _download(url: str) -> str:
    return (
        HEADER
        + f"""
_tick(); new_tab({url!r})
_tick(); js("document.querySelector('#download-report').click()")
_tick(); wait_js("document.getElementById('download-status').textContent === 'requested'")
status = js("document.getElementById('download-status').textContent")
_tick(); close_current()
emit({{'status': status}})
"""
    )


def _tabs(url: str) -> str:
    return (
        HEADER
        + f"""
_tick(); new_tab({url!r})
_tick(); original = current_tab()['targetId']
_tick(); box = js(
    "(()=>{{const r=document.querySelector('#compare').getBoundingClientRect(); "
    "return [r.x+r.width/2, r.y+r.height/2];}})()"
)
_tick(); click_at_xy(box[0], box[1])
_tick(); deadline = _time.time() + 20
detail = ''
while _time.time() < deadline:
    tabs = list_tabs()
    others = [tab for tab in tabs if tab['targetId'] != original and '/detail' in str(tab.get('url', ''))]
    if others:
        detail = others[-1]['targetId']
        break
    _time.sleep(0.1)
if not detail:
    raise RuntimeError('detail tab did not open')
_tick(); switch_tab(detail)
_tick(); js("document.querySelector('#approve').click()")
wait_js("document.getElementById('approved').textContent === 'approved'")
_tick(); switch_tab(original)
wait_js("document.getElementById('status').textContent === 'approved:42'")
_tick(); title = js("document.title")
_tick(); close_tab(detail)
status = js("document.getElementById('status').textContent")
_tick(); close_current()
emit({{'status': status, 'original_title': title}})
"""
    )


def _login(url: str) -> str:
    return (
        HEADER
        + f"""
_tick(); new_tab({url!r})
_tick(); fill_input('#username', {BENCH_USER!r})
_tick(); fill_input('#password', {BENCH_PASSWORD!r})
_tick(); js("document.querySelector('#login-submit').click()")
wait_js("!!document.getElementById('secret')")
secret = js("document.getElementById('secret').textContent")
_tick(); emit({{'secret': secret}})
_tick(); cdp('Browser.close')
"""
    )


def _read_account(url: str) -> str:
    return (
        HEADER
        + f"""
_tick(); new_tab({url!r})
wait_js("!!document.getElementById('secret')")
emit({{'secret': js("document.getElementById('secret').textContent")}})
"""
    )


class BrowserUseArm:
    name = "browser-use"

    def __init__(self) -> None:
        self.binary = BINARY
        self._result = ArmResult(ok=False)
        self._chrome: ChromeInstance | None = None
        self._warm_keychain_verified: bool | None = None

    def _env(self, ctx: RunContext) -> dict[str, str]:
        assert self._chrome is not None
        env = ctx.env
        env.update(
            {
                "BU_CDP_URL": self._chrome.cdp_url,
                "BH_RUNTIME_DIR": str(ctx.root / "bu-runtime"),
                "BH_TMP_DIR": str(ctx.root / "bu-tmp"),
            }
        )
        return env

    def _start_chrome(self, ctx: RunContext) -> None:
        self._chrome = ChromeInstance(CHROME, ctx.profile, downloads=ctx.downloads, headless=True)
        self._chrome.start()
        verified = verify_mock_keychain(str(ctx.profile))
        self._warm_keychain_verified = verified
        self._result.keychain_verified = verified
        if verified is False:
            raise RuntimeError("browser-use Chrome launched without --use-mock-keychain")

    def _stop_chrome(self) -> None:
        if self._chrome is not None:
            self._chrome.stop()
            self._chrome = None

    def _script(self, ctx: RunContext, code: str, timeout: float) -> dict:
        returncode, stdout, stderr, _elapsed = run_process(
            [BINARY],
            env=self._env(ctx),
            input_text=code,
            timeout=timeout,
        )
        self._result.process_starts += 1
        self._result.stdout += stdout
        self._result.stderr += stderr
        reported: dict = {}
        for line in stdout.splitlines():
            if line.startswith("BENCH_RESULT "):
                try:
                    reported = json.loads(line[len("BENCH_RESULT ") :])
                except json.JSONDecodeError:
                    reported = {}
        self._result.tool_calls += int(reported.pop("tool_calls", 0))
        if returncode != 0:
            raise RuntimeError(f"browser-use exited {returncode}: {stderr.strip()[:2000]}")
        return reported

    def warmup(self, ctx: RunContext) -> tuple[dict, float]:
        """Start the dedicated Chrome once for a warm browser-use session."""
        started = time.monotonic()
        self._result = ArmResult(ok=False)
        self._start_chrome(ctx)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        assert self._chrome is not None
        payload = {
            "ok": True,
            "cdp_url": self._chrome.cdp_url,
            "keychain_verified": self._warm_keychain_verified,
        }
        self._result = ArmResult(ok=False)
        return payload, elapsed_ms

    def run(self, ctx: RunContext, *, cleanup: bool = True) -> ArmResult:
        self._result = ArmResult(ok=False)
        try:
            if self._chrome is None:
                self._start_chrome(ctx)
            else:
                self._result.keychain_verified = self._warm_keychain_verified
            if ctx.task == "wizard":
                self._result.reported = self._script(ctx, _wizard(ctx.base_url + "/wizard"), 120)
            elif ctx.task == "modal":
                self._result.reported = self._script(ctx, _modal(ctx.base_url + "/modal"), 120)
            elif ctx.task == "download":
                self._result.reported = self._script(ctx, _download(ctx.base_url + "/download"), 120)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    files = [path for path in ctx.downloads.rglob("*") if path.is_file()]
                    if files:
                        self._result.reported["files"] = [str(path) for path in files]
                        break
                    time.sleep(0.2)
            elif ctx.task == "tabs":
                self._result.reported = self._script(ctx, _tabs(ctx.base_url + "/tabs"), 120)
            elif ctx.task == "auth":
                phase1 = self._script(ctx, _login(ctx.base_url + "/login"), 120)
                if self._chrome is not None and self._chrome.process is not None:
                    try:
                        self._chrome.process.wait(timeout=10)
                    except Exception:
                        pass
                self._stop_chrome()
                self._start_chrome(ctx)
                phase2 = self._script(ctx, _read_account(ctx.base_url + "/account"), 120)
                self._result.reported = {
                    "secret": phase2.get("secret", ""),
                    "phase1_secret": phase1.get("secret", ""),
                }
                if self._result.reported["secret"] != ACCOUNT_SECRET:
                    raise RuntimeError(f"secret after restart was {self._result.reported['secret']!r}")
            else:
                raise ValueError(f"unknown task {ctx.task}")
            self._result.ok = True
        except Exception as exc:
            self._result.ok = False
            self._result.error = str(exc)
        finally:
            if cleanup:
                self.cleanup(ctx)
        return self._result

    def cleanup(self, ctx: RunContext) -> None:
        self._stop_chrome()
        kill_pidfile(ctx.root / "bu-runtime" / "bu-default.pid")
        terminate_matching(str(ctx.profile))
        terminate_matching(str(ctx.root))
