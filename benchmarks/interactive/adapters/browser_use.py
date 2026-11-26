"""browser-use 1.0.8 CLI adapter in an isolated uv environment."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from ..arms import WARM_METHOD_ID
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

ADAPTER_VERSION = WARM_METHOD_ID
STDERR_TAIL_BYTES = 32768
WARMUP_SENTINEL = "NATIVE_SESSION_READY_V4"
WARMUP_CODE = f"""
import json
tabs = list_tabs()
if not isinstance(tabs, list):
    raise RuntimeError('native list_tabs did not return a list')
print('BENCH_RESULT ' + json.dumps({{'execution_ready': True, 'sentinel': {WARMUP_SENTINEL!r}}}))
"""

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

# The native wait_for_element uses a separate wall-clock timeout. An atomic
# native-js read keeps visibility, geometry and context retries on one deadline.
TABS_READINESS = """
_readiness_checks = 0
_readiness_retries = 0
_last_readiness_error = ''
_context_messages = {
    'Execution context was destroyed.',
    'Execution context was destroyed, most likely because of a navigation.',
    'Cannot find context with specified id',
}

def ready_box(selector, target, timeout=20.0):
    global _readiness_checks, _readiness_retries, _last_readiness_error
    deadline = _time.monotonic() + timeout
    box = None
    expression = (
        "(()=>{if(document.readyState==='loading')return null;"
        'const e=document.querySelector(' + _json.dumps(selector) + ');'
        'if(!e)return null;'
        'const s=getComputedStyle(e);'
        'if(s.display===\"none\"||s.visibility===\"hidden\"||s.opacity===\"0\")return null;'
        'if(e.checkVisibility&&!e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}))return null;'
        'const r=e.getBoundingClientRect();'
        'return r.width>0&&r.height>0?[r.x+r.width/2,r.y+r.height/2]:null;})()'
    )
    while _time.monotonic() < deadline:
        try:
            _readiness_checks += 1
            box = js(expression, target_id=target)
            if box is not None and _time.monotonic() < deadline:
                return box
        except RuntimeError as error:
            value = error.args[0] if error.args else ''
            if isinstance(value, dict):
                if value.get('code') != -32000:
                    raise
                value = value.get('message')
            if not isinstance(value, str) or value not in _context_messages:
                raise
            _last_readiness_error = str(error)
        _readiness_retries += 1
        remaining = deadline - _time.monotonic()
        if remaining > 0:
            _time.sleep(min(0.1, remaining))
    raise RuntimeError(
        'timeout waiting for ' + selector + ' target=' + str(target)
        + ' retries=' + str(_readiness_retries) + ' last_predicate=' + repr(box)
        + ' last_error=' + _last_readiness_error
    )
"""


def _stderr_tail(stderr: str, ctx: RunContext) -> tuple[str, bool]:
    """Redact complete paths before keeping a UTF-8 bounded diagnostic tail."""
    paths = {
        str(Path.home()),
        str(ctx.root),
        str(ctx.home),
        str(ctx.tmp),
        tempfile.gettempdir(),
        "/tmp",
        "/private/tmp",
    }
    # Redact an entire working-copy path, even when it belongs to another home.
    stderr = re.sub(r"/[^\s'\"<>]*?/\.worktrees/[^\s'\"<>]*", "<working-copy-path>", stderr)
    for path in sorted(paths, key=len, reverse=True):
        stderr = re.sub(re.escape(path) + r"(?=/|\b)[^\s'\"<>]*", "<path>", stderr)
    data = stderr.encode("utf-8")
    if len(data) <= STDERR_TAIL_BYTES:
        return stderr, False
    marker = b"[stderr tail truncated]\n"
    tail = data[-(STDERR_TAIL_BYTES - len(marker)) :].decode("utf-8", errors="ignore")
    return marker.decode() + tail, True


def _ready_header(url: str, selector: str) -> str:
    return (
        HEADER
        + TABS_READINESS
        + f"""
_tick(); initial_target = new_tab({url!r})
if not isinstance(initial_target, str) or not initial_target:
    raise RuntimeError('new_tab did not return a target ID')
_tick(); ready_box({selector!r}, initial_target)
"""
    )


def _wizard(url: str) -> str:
    return (
        _ready_header(url, "#name")
        + """
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
emit({'quote': quote})
"""
    )


def _modal(url: str) -> str:
    return (
        _ready_header(url, "#open-modal")
        + """
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
emit({'selected': selected})
"""
    )


def _download(url: str) -> str:
    return (
        _ready_header(url, "#download-report")
        + """
_tick(); js("document.querySelector('#download-report').click()")
_tick(); wait_js("document.getElementById('download-status').textContent === 'requested'")
status = js("document.getElementById('download-status').textContent")
_tick(); close_current()
emit({'status': status})
"""
    )


def _tabs(url: str) -> str:
    return (
        HEADER
        + TABS_READINESS
        + f"""
_tick(); original = new_tab({url!r})
if not isinstance(original, str) or not original:
    raise RuntimeError('new_tab did not return a target ID')
_tick(); switch_tab(original)
box = ready_box('#compare', original)
_tick(); click_at_xy(box[0], box[1])
_tick(); deadline = _time.monotonic() + 20
detail = ''
while _time.monotonic() < deadline:
    tabs = list_tabs()
    others = [tab for tab in tabs if tab['targetId'] != original and '/detail' in str(tab.get('url', ''))]
    if others:
        detail = others[-1]['targetId']
        break
    _time.sleep(0.1)
if not detail:
    raise RuntimeError('detail tab did not open')
_tick(); switch_tab(detail)
ready_box('#approve', detail)
_tick(); js("document.querySelector('#approve').click()", target_id=detail)
wait_js("document.getElementById('approved').textContent === 'approved'")
_tick(); switch_tab(original)
wait_js("document.getElementById('status').textContent === 'approved:42'")
_tick(); title = js("document.title")
_tick(); close_tab(detail)
status = js("document.getElementById('status').textContent")
_tick(); close_current()
emit({{'status': status, 'original_title': title,
       'readiness_checks': _readiness_checks, 'readiness_retries': _readiness_retries,
       'last_readiness_error': _last_readiness_error}})
"""
    )


def _login(url: str) -> str:
    return (
        _ready_header(url, "#username")
        + f"""
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
        _ready_header(url, "#secret")
        + """
wait_js("!!document.getElementById('secret')")
emit({'secret': js("document.getElementById('secret').textContent")})
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
        total_bytes = self._result.reported.get("stderr_total_bytes", 0) + len(stderr.encode("utf-8"))
        tail, truncated = _stderr_tail(self._result.stderr + stderr, ctx)
        truncated = truncated or self._result.reported.get("stderr_truncated", False)
        self._result.stderr = tail
        diagnostics = {
            "adapter_version": ADAPTER_VERSION,
            "stderr_total_bytes": total_bytes,
            "stderr_truncated": truncated,
        }
        self._result.reported.update(diagnostics)
        reported: dict = {}
        for line in stdout.splitlines():
            if line.startswith("BENCH_RESULT "):
                try:
                    reported = json.loads(line[len("BENCH_RESULT ") :])
                except json.JSONDecodeError:
                    reported = {}
        self._result.tool_calls += int(reported.pop("tool_calls", 0))
        if returncode != 0:
            raise RuntimeError(
                f"browser-use exited {returncode}: stderr_total_bytes={total_bytes} "
                f"stderr_truncated={truncated}\n{tail}"
            )
        reported.update(diagnostics)
        return reported

    def warmup(self, ctx: RunContext) -> tuple[dict, float]:
        """Prepare Chrome and the native execution interface without a task."""
        started = time.monotonic()
        self._result = ArmResult(ok=False)
        try:
            self._start_chrome(ctx)
            ready = self._script(ctx, WARMUP_CODE, 30)
            if ready.get("execution_ready") is not True or ready.get("sentinel") != WARMUP_SENTINEL:
                raise RuntimeError("browser-use warmup did not return the native readiness sentinel")
        except Exception:
            with suppress(Exception):
                self.cleanup(ctx)
            raise
        elapsed_ms = (time.monotonic() - started) * 1000.0
        assert self._chrome is not None
        payload = {
            "ok": True,
            "cdp_url": self._chrome.cdp_url,
            "keychain_verified": self._warm_keychain_verified,
            "execution_ready": True,
            "adapter_version": ADAPTER_VERSION,
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
                    with suppress(Exception):
                        self._chrome.process.wait(timeout=10)
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
