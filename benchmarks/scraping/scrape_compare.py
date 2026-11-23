"""Deterministic local scraping benchmark for OpenBrowser render modes.

Arms:
  - openbrowser-full
  - openbrowser-lightweight
  - openbrowser-no-render
  - http (httpx plus markdownify, no browser)

Tasks:
  - static_article
  - js_list
  - pagination
  - table
  - asset_heavy

The fixture server is local and deterministic. No external network, cloud, or
LLM is used. Every task has an exact correctness check before latency counts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

STATIC_ROWS = 50
JS_ITEMS = 30
PAGE_SIZE = 20
PAGE_COUNT = 5
TABLE_ROWS = 100
TABLE_COLS = 6
ASSET_COUNT = 20


def _static_page() -> str:
    rows = "\n".join(
        f'<article data-id="row-{i}"><h2>Item {i}</h2><p>Value {i * 7}</p></article>' for i in range(STATIC_ROWS)
    )
    return f"""<!doctype html>
<html><head><title>Static Bench</title></head>
<body>
<nav><a href="/docs">Docs</a> <a href="/about">About</a></nav>
<h1>Static Bench</h1>
{rows}
<script>const decoy = "DO NOT INCLUDE";</script>
</body></html>"""


def _js_list_page() -> str:
    return """<!doctype html>
<html><head><title>JS List</title></head>
<body>
<h1>JS List</h1>
<div id="items"></div>
<script>
fetch('/api/items?count=30').then(r => r.json()).then(items => {
  const root = document.getElementById('items');
  root.innerHTML = items.map(i =>
    `<div data-sku="${i.sku}"><span>${i.name}</span><span>${i.price}</span></div>`
  ).join('');
  root.dataset.count = String(items.length);
});
</script>
</body></html>"""


def _pagination_page(page: int) -> str:
    start = (page - 1) * PAGE_SIZE
    rows = "\n".join(
        f'<tr data-id="page-{page}-row-{i}"><td>{start + i}</td><td>Name {start + i}</td></tr>'
        for i in range(PAGE_SIZE)
    )
    next_link = (
        f'<a id="next" href="/pagination?page={page + 1}">Next</a>'
        if page < PAGE_COUNT
        else '<span id="end">End</span>'
    )
    return f"""<!doctype html>
<html><head><title>Pagination {page}</title></head>
<body><h1>Pagination {page}</h1>
<table><tbody>{rows}</tbody></table>
{next_link}
</body></html>"""


def _table_page() -> str:
    rows = []
    for i in range(TABLE_ROWS):
        cells = "".join(f"<td>{i if c == 0 else f'cell-{i}-{c}'}</td>" for c in range(TABLE_COLS))
        rows.append(f'<tr data-id="table-{i}">{cells}</tr>')
    return f"""<!doctype html>
<html><head><title>Table Bench</title></head>
<body><h1>Table Bench</h1>
<table><thead><tr><th>id</th><th>name</th><th>qty</th><th>price</th><th>date</th><th>status</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
</body></html>"""


def _assets_page() -> str:
    images = "".join(f'<img src="/img/{i}.png" alt="image {i}">' for i in range(ASSET_COUNT))
    return f"""<!doctype html>
<html><head><title>Asset Heavy</title>
<link rel="stylesheet" href="/style-a.css">
<link rel="stylesheet" href="/style-b.css">
</head>
<body>
<h1>Asset Heavy</h1>
{images}
<div id="headlines"></div>
<script>
const hs = Array.from({{length: 20}}, (_, i) => `<h2>Headline ${{i + 1}}</h2>`).join('');
const root = document.getElementById('headlines');
root.innerHTML = hs;
root.dataset.count = '20';
</script>
</body></html>"""


@dataclass
class FixtureState:
    requests: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, path: str, status: int, body: bytes) -> None:
        with self.lock:
            self.requests.append({"path": path, "status": status, "bytes": len(body), "at": time.monotonic()})

    def snapshot(self) -> tuple[int, int]:
        with self.lock:
            return len(self.requests), sum(item["bytes"] for item in self.requests)


class FixtureHandler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, *_args, **_kwargs) -> None:
        return

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.server.state.record(self.path, status, body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/static":
            self._send(200, _static_page().encode())
        elif path == "/js-list":
            self._send(200, _js_list_page().encode())
        elif path == "/api/items":
            count = int(parse_qs(parsed.query).get("count", [JS_ITEMS])[0])
            items = [{"sku": f"SKU-{i:03d}", "name": f"Product {i}", "price": i * 10} for i in range(count)]
            self._send(200, json.dumps(items).encode(), "application/json")
        elif path == "/pagination":
            page = int(parse_qs(parsed.query).get("page", ["1"])[0])
            self._send(200, _pagination_page(page).encode())
        elif path == "/table":
            self._send(200, _table_page().encode())
        elif path == "/assets":
            self._send(200, _assets_page().encode())
        elif path.startswith("/img/"):
            self._send(200, b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png")
        elif path.startswith("/style-"):
            css = b"body{font-family:'Bench';}@font-face{font-family:'Bench';src:url('/font.woff2');}"
            self._send(200, css, "text/css")
        elif path == "/font.woff2":
            self._send(200, b"wOF2" + b"\x00" * 128, "font/woff2")
        elif path == "/login":
            login_page = (
                b"<!doctype html><html><head><title>Login</title></head><body>"
                b'<form method="post" action="/login"><input name="username">'
                b'<input name="password" type="password"><button>Sign in</button></form>'
                b"</body></html>"
            )
            self._send(200, login_page)
        elif path == "/private":
            cookie = self.headers.get("Cookie", "")
            if "bench_session=ok" in cookie:
                private_page = (
                    b"<!doctype html><html><head><title>Private</title></head>"
                    b"<body><h1>Private</h1><p>secret-value</p></body></html>"
                )
                self._send(200, private_page)
            else:
                self._send(403, b"forbidden", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(302)
        self.send_header("Location", "/private")
        self.send_header("Set-Cookie", "bench_session=ok; Path=/; HttpOnly; SameSite=Lax")
        self.end_headers()
        self.server.state.record(self.path, 302, b"")


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address=("127.0.0.1", 0)):
        super().__init__(address, FixtureHandler)
        self.state = FixtureState()

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


def _skus_from_json(result: dict) -> list[str]:
    content = result.get("content")
    if not isinstance(content, list):
        return []
    skus = []
    for item in content:
        attributes = item.get("attributes") or {}
        sku = attributes.get("data-sku")
        if sku:
            skus.append(sku)
    return skus


@dataclass
class TaskResult:
    task: str
    arm: str
    ok: bool
    latency_ms: float
    output_bytes: int
    requests: int
    response_bytes: int
    engine: str = ""
    detail: str = ""


class BrowserArm:
    def __init__(self, name: str, render_mode: str, engine: str = "auto"):
        self.name = name
        self.render_mode = render_mode
        self.engine = engine
        self.session = None
        self.namespace = None
        self.last_engine = ""

    async def start(self) -> None:
        from openbrowser.browser import BrowserProfile, BrowserSession
        from openbrowser.code_use.namespace import create_namespace
        from openbrowser.tools.service import CodeAgentTools

        self.session = BrowserSession(
            browser_profile=BrowserProfile(
                headless=True,
                render_mode=self.render_mode,
                enable_default_extensions=False,
            )
        )
        await self.session.start()
        self.namespace = create_namespace(browser_session=self.session, tools=CodeAgentTools())

    async def stop(self) -> None:
        if self.session is not None:
            with contextlib.suppress(Exception):
                await self.session.kill()

    async def scrape(self, **kwargs):
        kwargs.setdefault("engine", self.engine)
        result = await self.namespace["scrape_page"](**kwargs)
        self.last_engine = result.get("engine_used", "browser")
        return result

    async def run_static(self, base: str):
        result = await self.scrape(url=f"{base}/static", format="markdown")
        content = result.get("content") or ""
        ok = "Static Bench" in content and "Item 49" in content and "DO NOT INCLUDE" not in content
        return ok, content, ""

    async def run_js_list(self, base: str):
        result = await self.scrape(
            url=f"{base}/js-list",
            format="json",
            selector="#items [data-sku]",
            wait_for='#items[data-count="30"]',
        )
        skus = _skus_from_json(result)
        ok = len(skus) == JS_ITEMS and skus[0] == "SKU-000" and skus[-1] == "SKU-029"
        return ok, json.dumps(skus), ""

    async def run_pagination(self, base: str):
        ids: list[str] = []
        for page in range(1, PAGE_COUNT + 1):
            result = await self.scrape(
                url=f"{base}/pagination?page={page}",
                format="json",
                selector="tbody tr",
            )
            for item in result.get("content") or []:
                row_id = (item.get("attributes") or {}).get("data-id")
                if row_id:
                    ids.append(row_id)
        ok = len(ids) == PAGE_COUNT * PAGE_SIZE and len(set(ids)) == len(ids)
        return ok, json.dumps(ids), ""

    async def run_table(self, base: str):
        result = await self.scrape(url=f"{base}/table", format="json", selector="tbody tr")
        rows = result.get("content") or []
        ok = len(rows) == TABLE_ROWS and all(
            len(re.findall(r"<td>", item.get("html") or "")) == TABLE_COLS for item in rows
        )
        return ok, json.dumps([item.get("attributes") for item in rows]), ""

    async def run_assets(self, base: str):
        result = await self.scrape(
            url=f"{base}/assets",
            format="text",
            wait_for='#headlines[data-count="20"]',
        )
        content = result.get("content") or ""
        headlines = re.findall(r"Headline \d+", content)
        ok = len(headlines) == ASSET_COUNT
        return ok, content, ""


class HttpArm:
    name = "http"

    def __init__(self):
        self.client = httpx.Client(follow_redirects=True, timeout=30)
        self.last_engine = "http"

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        self.client.close()

    async def run_static(self, base: str):
        text = self.client.get(f"{base}/static").text
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        content = markdownify(str(soup), heading_style="ATX")
        ok = "Static Bench" in content and "Item 49" in content and "DO NOT INCLUDE" not in content
        return ok, content, ""

    async def run_js_list(self, base: str):
        text = self.client.get(f"{base}/js-list").text
        ok = "SKU-000" in text
        return ok, text, "no JavaScript execution"

    async def run_pagination(self, base: str):
        ids: list[str] = []
        for page in range(1, PAGE_COUNT + 1):
            text = self.client.get(f"{base}/pagination?page={page}").text
            ids.extend(re.findall(r'data-id="([^"]+)"', text))
        ok = len(ids) == PAGE_COUNT * PAGE_SIZE and len(set(ids)) == len(ids)
        return ok, json.dumps(ids), ""

    async def run_table(self, base: str):
        text = self.client.get(f"{base}/table").text
        rows = re.findall(r'<tr data-id="[^"]+">(.*?)</tr>', text, re.S)
        ok = len(rows) == TABLE_ROWS and all(len(re.findall(r"<td>", row)) == TABLE_COLS for row in rows)
        return ok, text, ""

    async def run_assets(self, base: str):
        text = self.client.get(f"{base}/assets").text
        headlines = re.findall(r"Headline \d+", text)
        ok = len(headlines) == ASSET_COUNT
        return ok, text, "no JavaScript execution"


TASKS = {
    "static_article": "run_static",
    "js_list": "run_js_list",
    "pagination": "run_pagination",
    "table": "run_table",
    "asset_heavy": "run_assets",
}


async def _measure(arm, task_name: str, method: str, base: str, state: FixtureState):
    before_requests, before_bytes = state.snapshot()
    started = time.perf_counter_ns()
    error = ""
    try:
        ok, output, detail = await getattr(arm, method)(base)
    except Exception as exc:
        ok, output, detail = False, "", f"{type(exc).__name__}: {exc}"
        error = detail
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    after_requests, after_bytes = state.snapshot()
    return TaskResult(
        task=task_name,
        arm=arm.name,
        ok=bool(ok),
        latency_ms=latency_ms,
        output_bytes=len(output.encode()),
        requests=after_requests - before_requests,
        response_bytes=after_bytes - before_bytes,
        engine=getattr(arm, "last_engine", ""),
        detail=error,
    )


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(values),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "mean_ms": round(statistics.fmean(values), 3),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    default_arms = ["openbrowser-full", "openbrowser-lightweight", "openbrowser-no-render", "http"]
    parser.add_argument("--arms", nargs="+", default=default_arms)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/scraping/results.json"))
    args = parser.parse_args()

    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = server.base_url
    arms = []
    for name in args.arms:
        if name == "http":
            arms.append(HttpArm())
        elif name == "openbrowser-full":
            arms.append(BrowserArm(name, "full"))
        elif name == "openbrowser-lightweight":
            arms.append(BrowserArm(name, "lightweight"))
        elif name == "openbrowser-no-render":
            arms.append(BrowserArm(name, "no-render"))
        else:
            raise SystemExit(f"unknown arm: {name}")

    results: list[TaskResult] = []
    try:
        for arm in arms:
            await arm.start()
            for task_name in args.tasks:
                method = TASKS[task_name]
                for _ in range(args.warmups):
                    await _measure(arm, task_name, method, base, server.state)
                for _ in range(args.samples):
                    results.append(await _measure(arm, task_name, method, base, server.state))
            await arm.stop()
    finally:
        await asyncio.gather(*(arm.stop() for arm in arms), return_exceptions=True)
        server.shutdown()
        server.server_close()

    grouped: dict[tuple[str, str], list[TaskResult]] = {}
    for result in results:
        grouped.setdefault((result.task, result.arm), []).append(result)

    report = {"samples": args.samples, "warmups": args.warmups, "arms": args.arms, "tasks": args.tasks, "groups": []}
    print(f"{'task':18} {'arm':24} {'ok':>3} {'p50 ms':>8} {'p95 ms':>8} {'bytes':>8} {'req':>4}")
    for (task, arm_name), group in sorted(grouped.items()):
        ok_count = sum(1 for item in group if item.ok)
        latency = [item.latency_ms for item in group if item.ok]
        summary = _summary(latency)
        entry = {
            "task": task,
            "arm": arm_name,
            "successes": ok_count,
            "runs": len(group),
            "ok": ok_count == len(group),
            "latency": summary,
            "mean_output_bytes": round(statistics.fmean(item.output_bytes for item in group), 1),
            "mean_requests": round(statistics.fmean(item.requests for item in group), 1),
            "mean_response_bytes": round(statistics.fmean(item.response_bytes for item in group), 1),
            "engines": sorted({item.engine for item in group if item.engine}),
            "errors": sorted({item.detail for item in group if item.detail}),
        }
        report["groups"].append(entry)
        print(
            f"{task:18} {arm_name:24} {ok_count}/{len(group):<2} "
            f"{summary.get('p50_ms', 0):8.1f} {summary.get('p95_ms', 0):8.1f} "
            f"{entry['mean_output_bytes']:8.0f} {entry['mean_requests']:4.1f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
