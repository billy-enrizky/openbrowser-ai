"""Real-browser proof that the render resource policy blocks before the network.

Each test launches a headless Chromium against a local fixture server. The
server records every path it serves. Blocked resource types must never appear
in that record, while document, script, XHR, and fetch requests must still work.
"""

import asyncio
import base64
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from openbrowser.browser import BrowserProfile, BrowserSession

pytestmark = pytest.mark.integration

PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
FONT_BYTES = b"wOF2" + b"\x00" * 60
MEDIA_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64

PAGE_HTML = b"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Render Policy Bench</title>
<link rel="stylesheet" href="/style.css">
<script src="/app.js"></script>
</head>
<body>
<h1>Render Policy Bench</h1>
<img id="pixel" src="/pixel.png" alt="pixel">
<video id="clip" preload="auto" muted src="/clip.mp4">
  <track kind="captions" src="/captions.vtt" default>
</video>
<p class="bench-font">font probe</p>
<script>
window.__bench = {api: 'pending', xhr: 'pending'};
const api = fetch('/api/data').then(r => r.json()).then(d => { window.__bench.api = d.status; });
const xhr = new Promise(resolve => {
  const request = new XMLHttpRequest();
  request.open('GET', '/api/xhr');
  request.onload = () => { window.__bench.xhr = JSON.parse(request.responseText).status; resolve(); };
  request.send();
});
Promise.all([api, xhr]).then(() => {
  document.title = 'READY';
});
</script>
</body></html>"""

STYLE_CSS = b"""@font-face { font-family: BenchFont; src: url('/font.woff2') format('woff2'); }
body { background-color: rgb(1, 2, 3); }
.bench-font { font-family: BenchFont, monospace; }"""

APP_JS = b"document.documentElement.dataset.scriptRan = '1';"

CAPTIONS_VTT = b"""WEBVTT

00:00.000 --> 00:01.000
caption probe
"""


class _RecordingServer(ThreadingHTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths: list[str] = []

    def record(self, path: str) -> None:
        self.paths.append(path)


class _FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        self.server.record(path)
        if path == "/":
            self._send(PAGE_HTML, "text/html; charset=utf-8")
        elif path == "/style.css":
            self._send(STYLE_CSS, "text/css")
        elif path == "/app.js":
            self._send(APP_JS, "application/javascript")
        elif path == "/pixel.png":
            self._send(PIXEL_PNG, "image/png")
        elif path == "/clip.mp4":
            self._send(MEDIA_BYTES, "video/mp4")
        elif path == "/font.woff2":
            self._send(FONT_BYTES, "font/woff2")
        elif path == "/captions.vtt":
            self._send(CAPTIONS_VTT, "text/vtt")
        elif path in {"/api/data", "/api/xhr"}:
            self._send(b'{"status": "ok"}', "application/json")
        else:
            self._send(b"not found", "text/plain", status=404)


async def _wait_ready(browser: BrowserSession, served: list[str], budget_seconds: float = 20.0) -> dict:
    focus = browser.agent_focus
    assert focus is not None
    expression = (
        "JSON.stringify({title: document.title, bench: window.__bench, "
        "scriptRan: document.documentElement.dataset.scriptRan === '1', "
        "css: getComputedStyle(document.body).backgroundColor})"
    )
    deadline = time.monotonic() + budget_seconds
    last: dict = {}
    while time.monotonic() < deadline:
        response = await focus.cdp_client.send.Runtime.evaluate(
            params={"expression": expression, "returnByValue": True},
            session_id=focus.session_id,
        )
        value = response.get("result", {}).get("value")
        if value:
            last = json.loads(value)
            if last.get("title") == "READY":
                return last
        await asyncio.sleep(0.05)
    raise AssertionError(f"page did not become ready: {last}, served={sorted(set(served))}")


async def _wait_paths(served: list[str], expected: set[str], budget_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + budget_seconds
    while time.monotonic() < deadline:
        if expected <= set(served):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"missing expected requests: {sorted(expected - set(served))}, served={sorted(set(served))}")


async def _run_mode(render_mode: str) -> tuple[set[str], dict]:
    server = _RecordingServer(("127.0.0.1", 0), _FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        browser = BrowserSession(
            browser_profile=BrowserProfile(
                headless=True,
                user_data_dir=str(tmp_path / "profile"),
                downloads_path=str(tmp_path / "downloads"),
                render_mode=render_mode,
                args=["--use-mock-keychain"],
            )
        )
        try:
            await asyncio.wait_for(browser.start(), timeout=60)
            await asyncio.wait_for(browser.navigate_to(f"{base}/"), timeout=30)
            state = await _wait_ready(browser, server.paths)
            if render_mode == "full":
                await _wait_paths(server.paths, {"/style.css", "/pixel.png", "/font.woff2"})
            return set(server.paths), state
        finally:
            await asyncio.wait_for(browser.kill(), timeout=30)
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.asyncio
async def test_full_mode_requests_every_asset():
    seen, state = await _run_mode("full")

    assert {"/style.css", "/pixel.png", "/font.woff2"} <= seen
    assert state["bench"] == {"api": "ok", "xhr": "ok"}
    assert state["scriptRan"] is True
    assert state["css"] == "rgb(1, 2, 3)"


@pytest.mark.asyncio
async def test_lightweight_blocks_optional_assets_and_keeps_css():
    seen, state = await _run_mode("lightweight")

    assert {"/", "/app.js", "/style.css", "/api/data", "/api/xhr"} <= seen
    assert state["bench"] == {"api": "ok", "xhr": "ok"}
    assert state["scriptRan"] is True
    assert state["css"] == "rgb(1, 2, 3)", "lightweight must keep stylesheets"
    assert "/pixel.png" not in seen
    assert "/font.woff2" not in seen
    assert "/clip.mp4" not in seen
    assert "/captions.vtt" not in seen


@pytest.mark.asyncio
async def test_no_render_blocks_stylesheets_and_optional_assets():
    seen, state = await _run_mode("no-render")

    assert {"/", "/app.js", "/api/data", "/api/xhr"} <= seen
    assert state["bench"] == {"api": "ok", "xhr": "ok"}
    assert state["scriptRan"] is True
    assert state["css"] != "rgb(1, 2, 3)", "no-render must block external stylesheets"
    assert "/style.css" not in seen
    assert "/pixel.png" not in seen
    assert "/font.woff2" not in seen
    assert "/clip.mp4" not in seen
    assert "/captions.vtt" not in seen
