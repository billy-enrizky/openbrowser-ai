"""Compare OpenBrowser render modes against a local asset-heavy fixture.

Arms:
  - full: every resource loads
  - lightweight: images, fonts, media, and text tracks are blocked, CSS loads
  - no-render: the lightweight set plus external stylesheets are blocked

Each sample launches a fresh headless browser against a fresh fixture server and
records wall time, server-observed requests, and optional asset bytes avoided.
The check is server-side, so a blocked type must never reach the fixture.

Run:
    uv run python benchmarks/scraping/no_render_compare.py --samples 3

The wall time includes browser startup, so this measures the product path, not
an isolated engine comparison. No network, cloud, or LLM is used.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openbrowser.browser import BrowserProfile, BrowserSession

PIXEL_PNG = base64.b64decode(
	'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
)
FONT_BYTES = b'wOF2' + b'\x00' * 4096
MEDIA_BYTES = b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 65536
STYLE_CSS = b"""@font-face { font-family: BenchFont; src: url('/font.woff2') format('woff2'); }
body { background-color: rgb(1, 2, 3); }
.bench-font { font-family: BenchFont, monospace; }"""
APP_JS = b"document.documentElement.dataset.scriptRan = '1';"
PAGE_HTML = b"""<!doctype html>
<html><head>
<meta charset="utf-8"><title>Render Mode Bench</title>
<link rel="stylesheet" href="/style.css">
<script src="/app.js"></script>
</head><body>
<h1>Render Mode Bench</h1>
<img id="pixel" src="/pixel.png" alt="pixel">
<video id="clip" preload="auto" muted src="/clip.mp4"><track kind="captions" src="/captions.vtt" default></video>
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
Promise.all([api, xhr]).then(() => { document.title = 'READY'; });
</script>
</body></html>"""

BODIES: dict[str, tuple[bytes, str]] = {
	'/': (PAGE_HTML, 'text/html; charset=utf-8'),
	'/style.css': (STYLE_CSS, 'text/css'),
	'/app.js': (APP_JS, 'application/javascript'),
	'/pixel.png': (PIXEL_PNG, 'image/png'),
	'/clip.mp4': (MEDIA_BYTES, 'video/mp4'),
	'/font.woff2': (FONT_BYTES, 'font/woff2'),
	'/captions.vtt': (b'WEBVTT\n\n00:00.000 --> 00:01.000\ncaption probe\n', 'text/vtt'),
	'/api/data': (b'{"status": "ok"}', 'application/json'),
	'/api/xhr': (b'{"status": "ok"}', 'application/json'),
}

MODES: dict[str, set[str]] = {
	'full': set(),
	'lightweight': {'/pixel.png', '/font.woff2', '/clip.mp4', '/captions.vtt'},
	'no-render': {'/style.css', '/pixel.png', '/font.woff2', '/clip.mp4', '/captions.vtt'},
}
REQUIRED = {'/', '/app.js', '/api/data', '/api/xhr'}


class _Server(ThreadingHTTPServer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.paths: list[str] = []
		self.bytes_served = 0


class _Handler(BaseHTTPRequestHandler):
	def log_message(self, *_args):
		return

	def do_GET(self):
		path = self.path.split('?', 1)[0]
		self.server.paths.append(path)
		body, content_type = BODIES.get(path, (b'not found', 'text/plain'))
		self.send_response(200 if path in BODIES else 404)
		self.send_header('Content-Type', content_type)
		self.send_header('Content-Length', str(len(body)))
		self.send_header('Cache-Control', 'no-store')
		self.end_headers()
		self.wfile.write(body)
		self.server.bytes_served += len(body)


@dataclass
class RunResult:
	mode: str
	ok: bool
	wall_ms: float
	requests: list[str]
	optional_requests: list[str]
	bytes_served: int
	bytes_avoided: int
	title: str
	text_length: int
	problems: list[str] = field(default_factory=list)


async def _wait_ready(browser: BrowserSession, timeout: float = 20.0) -> tuple[str, int]:
	focus = browser.agent_focus
	assert focus is not None
	expression = (
		'JSON.stringify({title: document.title, text: document.body ? document.body.innerText.length : 0})'
	)
	deadline = time.monotonic() + timeout
	last: dict = {}
	while time.monotonic() < deadline:
		response = await focus.cdp_client.send.Runtime.evaluate(
			params={'expression': expression, 'returnByValue': True},
			session_id=focus.session_id,
		)
		value = response.get('result', {}).get('value')
		if value:
			last = json.loads(value)
			if last.get('title') == 'READY':
				return str(last['title']), int(last.get('text') or 0)
		await asyncio.sleep(0.05)
	raise RuntimeError(f'page did not become ready: {last}')


async def _run(mode: str) -> RunResult:
	server = _Server(('127.0.0.1', 0), _Handler)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	base = f'http://127.0.0.1:{server.server_port}'
	problems: list[str] = []
	title = ''
	text_length = 0
	try:
		with tempfile.TemporaryDirectory() as tmp:
			tmp_path = Path(tmp)
			browser = BrowserSession(
				browser_profile=BrowserProfile(
					headless=True,
					user_data_dir=str(tmp_path / 'profile'),
					downloads_path=str(tmp_path / 'downloads'),
					render_mode=mode,
					args=['--use-mock-keychain'],
				)
			)
			started = time.perf_counter()
			try:
				await asyncio.wait_for(browser.start(), timeout=60)
				await asyncio.wait_for(browser.navigate_to(f'{base}/'), timeout=30)
				title, text_length = await _wait_ready(browser)
			finally:
				await asyncio.wait_for(browser.kill(), timeout=30)
			wall_ms = (time.perf_counter() - started) * 1000
		# Let optional asset requests arrive before inspecting the server log.
		# This happens after timing so full mode is not penalized for assets.
		await asyncio.sleep(0.3)
	finally:
		server.shutdown()
		thread.join(timeout=5)

	requests = list(server.paths)
	blocked_paths = MODES[mode]
	optional_requests = [path for path in requests if path in {'/style.css', '/pixel.png', '/font.woff2', '/clip.mp4', '/captions.vtt'}]
	for path in sorted(blocked_paths & set(requests)):
		problems.append(f'{mode} fetched blocked resource {path}')
	if not REQUIRED <= set(requests):
		problems.append(f'missing required requests: {sorted(REQUIRED - set(requests))}')
	if title != 'READY':
		problems.append(f'title was {title!r}')
	if text_length <= 0:
		problems.append('empty page text')
	if mode == 'lightweight' and '/style.css' not in requests:
		problems.append('lightweight did not load the stylesheet')
	bytes_avoided = sum(len(BODIES[path][0]) for path in blocked_paths if path not in requests)
	return RunResult(
		mode=mode,
		ok=not problems,
		wall_ms=round(wall_ms, 1),
		requests=requests,
		optional_requests=optional_requests,
		bytes_served=server.bytes_served,
		bytes_avoided=bytes_avoided,
		title=title,
		text_length=text_length,
		problems=problems,
	)


def _summary(values: list[float]) -> dict[str, float]:
	ordered = sorted(values)
	pick = lambda p: ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))]  # noqa: E731
	return {'p50': pick(0.5), 'p95': pick(0.95), 'min': ordered[0], 'max': ordered[-1]}


async def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--samples', type=int, default=3)
	parser.add_argument('--modes', default='full,lightweight,no-render')
	args = parser.parse_args()

	modes = [mode.strip() for mode in args.modes.split(',') if mode.strip()]
	unknown = set(modes) - set(MODES)
	if unknown:
		parser.error(f'unknown modes: {sorted(unknown)}')

	results: dict[str, list[RunResult]] = {mode: [] for mode in modes}
	for sample in range(args.samples):
		for mode in modes:
			result = await _run(mode)
			results[mode].append(result)
			print(
				f'sample {sample + 1}/{args.samples} {mode}: '
				f'{result.wall_ms:.1f} ms, {len(result.requests)} requests, '
				f'{result.bytes_served} bytes, {result.bytes_avoided} avoided, ok={result.ok}'
			)
			if result.problems:
				for problem in result.problems:
					print(f'  problem: {problem}')

	report = {
		'samples': args.samples,
		'arms': {
			mode: {
				'wall_ms': _summary([result.wall_ms for result in results[mode]]),
				'mean_requests': round(statistics.mean(len(result.requests) for result in results[mode]), 2),
				'mean_optional_requests': round(
					statistics.mean(len(result.optional_requests) for result in results[mode]), 2
				),
				'mean_bytes_served': round(statistics.mean(result.bytes_served for result in results[mode]), 1),
				'mean_bytes_avoided': round(statistics.mean(result.bytes_avoided for result in results[mode]), 1),
				'ok': all(result.ok for result in results[mode]),
			}
			for mode in modes
		},
		'runs': {mode: [asdict(result) for result in results[mode]] for mode in modes},
	}
	print(json.dumps(report, indent=2))
	return 0 if all(result.ok for result in results.get(mode, []) for mode in modes) else 1


if __name__ == '__main__':
	raise SystemExit(asyncio.run(main()))
