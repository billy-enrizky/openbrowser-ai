"""Deterministic local scraping benchmark for OpenBrowser render modes.

Arms:
  - openbrowser-full
  - openbrowser-lightweight
  - openbrowser-no-render
  - firecrawl-local (Firecrawl Local, native per-task wait)
  - firecrawl-local-auto (Firecrawl Local, no waitFor override)

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
import html
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

STATIC_ROWS = 50
JS_ITEMS = 30
PAGE_SIZE = 20
PAGE_COUNT = 5
TABLE_ROWS = 100
TABLE_COLS = 6
ASSET_COUNT = 20

EXPECTED_JS_SKUS = [f"SKU-{index:03d}" for index in range(JS_ITEMS)]
EXPECTED_PAGINATION_IDS = [f"page-{page}-row-{row}" for page in range(1, PAGE_COUNT + 1) for row in range(PAGE_SIZE)]
EXPECTED_TABLE_CELLS = [
    [str(index)] + [f"cell-{index}-{column}" for column in range(1, TABLE_COLS)] for index in range(TABLE_ROWS)
]
EXPECTED_TABLE_ROWS = [[f"table-{index}", *EXPECTED_TABLE_CELLS[index]] for index in range(TABLE_ROWS)]


def exact_static(content: str) -> bool:
    """Require every article label and value in order, with no decoy text."""
    if not isinstance(content, str):
        return False
    fields = re.findall(r"\b(Item|Value)\s+([0-9]+)(?![\w.+-])", content)
    expected = [
        (label, str(value)) for index in range(STATIC_ROWS) for label, value in (("Item", index), ("Value", index * 7))
    ]
    return fields == expected and "DO NOT INCLUDE" not in content


def exact_assets(content: str) -> bool:
    """Require every rendered headline exactly once in order."""
    if not isinstance(content, str):
        return False
    headlines = re.findall(r"\bHeadline\s+([0-9]+)(?![\w.+-])", content)
    return headlines == [str(index) for index in range(1, ASSET_COUNT + 1)]


def exact_js_list(skus: list[str]) -> bool:
    """Require the complete ordered SKU sequence."""
    return skus == EXPECTED_JS_SKUS


def exact_pagination(ids: list[str]) -> bool:
    """Require the complete ordered page and row ID sequence."""
    return ids == EXPECTED_PAGINATION_IDS


def exact_table(rows: list[list[str]]) -> bool:
    """Require every row ID and every cell in order."""
    return rows == EXPECTED_TABLE_ROWS


def _cell_text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


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
    def __init__(self, name: str, render_mode: str, engine: str = "auto", chrome_path: str | None = None):
        self.name = name
        self.render_mode = render_mode
        self.engine = engine
        self.chrome_path = chrome_path
        self.session = None
        self.namespace = None
        self.last_engine = ""
        self._started = False

    async def start(self) -> None:
        from openbrowser.browser import BrowserProfile, BrowserSession
        from openbrowser.code_use.namespace import create_namespace
        from openbrowser.tools.service import CodeAgentTools

        profile_kwargs: dict = {
            "headless": True,
            "render_mode": self.render_mode,
            "enable_default_extensions": False,
            # Never touch the macOS login keychain from a headless run.
            "args": ["--use-mock-keychain"],
        }
        if self.chrome_path:
            profile_kwargs["executable_path"] = self.chrome_path
        self.session = BrowserSession(browser_profile=BrowserProfile(**profile_kwargs))
        self.namespace = create_namespace(browser_session=self.session, tools=CodeAgentTools())

    async def _ensure_started(self) -> None:
        if self.session is None:
            raise RuntimeError("start() must complete before browser use")
        if not self._started:
            await self.session.start()
            self._started = True

    async def prepare(self, task_name: str) -> None:
        """Start the browser before timing when the task or engine needs it."""
        if self.engine == "browser" or task_name in {"js_list", "asset_heavy"}:
            await self._ensure_started()

    async def stop(self) -> None:
        if self.session is not None:
            await self.session.kill()

    async def scrape(self, **kwargs):
        if self.namespace is None:
            raise RuntimeError("start() must complete before scraping")
        kwargs.setdefault("engine", self.engine)
        result = await self.namespace["scrape_page"](**kwargs)
        self.last_engine = result.get("engine_used", "browser")
        return result

    async def run_static(self, base: str):
        result = await self.scrape(url=f"{base}/static", format="markdown")
        content = result.get("content") or ""
        ok = exact_static(content)
        return ok, content, ""

    async def run_js_list(self, base: str):
        await self._ensure_started()
        result = await self.scrape(
            url=f"{base}/js-list",
            format="json",
            selector="#items [data-sku]",
            wait_for='#items[data-count="30"]',
        )
        skus = _skus_from_json(result)
        ok = exact_js_list(skus)
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
        ok = exact_pagination(ids)
        return ok, json.dumps(ids), ""

    async def run_table(self, base: str):
        result = await self.scrape(url=f"{base}/table", format="json", selector="tbody tr")
        items = result.get("content") or []
        rows = [
            [
                str((item.get("attributes") or {}).get("data-id") or ""),
                *[_cell_text(cell) for cell in re.findall(r"<td>(.*?)</td>", item.get("html") or "", re.S)],
            ]
            for item in items
        ]
        ok = exact_table(rows)
        return ok, json.dumps(rows), ""

    async def run_assets(self, base: str):
        await self._ensure_started()
        result = await self.scrape(
            url=f"{base}/assets",
            format="text",
            wait_for='#headlines[data-count="20"]',
        )
        content = result.get("content") or ""
        ok = exact_assets(content)
        return ok, content, ""


# Firecrawl Local arms. Firecrawl Local serves the Firecrawl v2 API on loopback
# and runs the same cleaning code as the hosted API, so the adapter posts the
# same request body a Firecrawl SDK would.
FIRECRAWL_ASSET_WAIT_MS = 1


def _pipe_rows(markdown: str) -> list[list[str]]:
    """Return pipe-table rows, keeping the header and dropping separator rows."""
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        line = line.strip()
        if len(line) < 2 or not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if cells and all(cell and set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append(cells)
    return rows


class FirecrawlServer:
    """Run or attach to a Firecrawl Local server.

    A managed server always starts with ``--allow-private`` because the fixture
    runs on loopback, and with ``FIRECRAWL_CLOUD=off`` so no page leaves the
    machine. Both are Firecrawl Local requirements for a local-only run.
    """

    def __init__(self, binary: str | None = None, url: str | None = None, chrome_path: str | None = None):
        self.binary = binary
        self.chrome_path = chrome_path
        self.external_url = url
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.version = ""
        self._url = url.rstrip("/") if url else ""

    @property
    def url(self) -> str:
        if not self._url:
            raise RuntimeError("firecrawl server is not running")
        return self._url

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _health(self) -> dict:
        response = httpx.get(f"{self._url}/health", timeout=2.0)
        response.raise_for_status()
        return response.json()

    def start(self) -> None:
        if self.external_url:
            payload = self._health()
            self.version = str(payload.get("version") or "")
            return
        if not self.binary:
            raise SystemExit("no firecrawl-local binary configured")
        port = self._free_port()
        self._url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env["FIRECRAWL_CLOUD"] = "off"
        env["FIRECRAWL_LOCAL_BROWSER"] = "auto"
        if self.chrome_path:
            env["FIRECRAWL_CHROME_PATH"] = self.chrome_path
        fd, log_name = tempfile.mkstemp(prefix="firecrawl-serve-", suffix=".log")
        self.log_path = Path(log_name)
        with os.fdopen(fd, "wb") as log:
            self.process = subprocess.Popen(
                [self.binary, "serve", "--port", str(port), "--allow-private", "--browser", "auto"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 40.0
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = self.log_path.read_text(errors="replace")[-2000:] if self.log_path.exists() else ""
                raise SystemExit(f"firecrawl-local serve exited with {self.process.returncode}: {tail}")
            try:
                payload = self._health()
                self.version = str(payload.get("version") or "")
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.1)
        raise SystemExit(f"firecrawl-local serve did not become healthy: {last_error}")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self.process = None
        if self.log_path is not None and self.log_path.exists():
            self.log_path.unlink()


class FirecrawlArm:
    """Scrape through Firecrawl Local with the documented v2 request body.

    Firecrawl Local skips the ``json`` format and ignores ``actions``, so every
    task scrapes markdown and the oracle is expressed on markdown output.
    ``waitFor`` is the only request-level way to force the local browser: one
    millisecond is enough to move the asset page off the static fetch path and
    adds no measurable sleep.
    """

    def __init__(self, name: str, server: FirecrawlServer, wait_overrides: dict[str, int] | None = None):
        self.name = name
        self.server = server
        self.wait_overrides = dict(wait_overrides or {})
        self.last_engine = ""
        self.client = httpx.Client(timeout=180.0)

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        self.client.close()

    def _scrape(self, url: str, wait_ms: int | None = None) -> tuple[str, str]:
        payload: dict = {"url": url, "formats": ["markdown"], "maxAge": 0, "storeInCache": False}
        if wait_ms is not None:
            payload["waitFor"] = wait_ms
        response = self.client.post(f"{self.server.url}/v2/scrape", json=payload)
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            raise RuntimeError(f"firecrawl scrape failed: {str(body)[:300]}")
        data = body.get("data") or {}
        engine = str((data.get("metadata") or {}).get("engine") or "")
        return data.get("markdown") or "", engine

    async def run_static(self, base: str):
        content, engine = self._scrape(f"{base}/static")
        self.last_engine = engine
        ok = exact_static(content)
        return ok, content, ""

    async def run_js_list(self, base: str):
        content, engine = self._scrape(f"{base}/js-list")
        self.last_engine = engine
        # Firecrawl markdown drops data-sku, so the exact oracle is the ordered
        # name plus price sequence rendered by the same 30 cards.
        expected = [f"Product {index}{index * 10}" for index in range(JS_ITEMS)]
        cursor = 0
        ordered = True
        for token in expected:
            found = content.find(token, cursor)
            if found < 0:
                ordered = False
                break
            cursor = found + len(token)
        ok = ordered and content.count("Product") == JS_ITEMS
        return ok, content, ""

    async def run_pagination(self, base: str):
        names: list[int] = []
        pages: list[str] = []
        engines: set[str] = set()
        for page in range(1, PAGE_COUNT + 1):
            content, engine = self._scrape(f"{base}/pagination?page={page}")
            pages.append(content)
            engines.add(engine)
            for row in _pipe_rows(content)[1:]:
                if len(row) < 2:
                    continue
                match = re.fullmatch(r"Name (\d+)", row[1])
                if match:
                    names.append(int(match.group(1)))
        self.last_engine = "local-browser" if "local-browser" in engines else (sorted(engines)[0] if engines else "")
        ok = names == list(range(PAGE_COUNT * PAGE_SIZE))
        return ok, "\n".join(pages), ""

    async def run_table(self, base: str):
        content, engine = self._scrape(f"{base}/table")
        self.last_engine = engine
        rows = _pipe_rows(content)[1:]
        expected = [
            [str(index)] + [f"cell-{index}-{column}" for column in range(1, TABLE_COLS)] for index in range(TABLE_ROWS)
        ]
        ok = rows == expected
        return ok, content, ""

    async def run_assets(self, base: str):
        content, engine = self._scrape(f"{base}/assets", self.wait_overrides.get("asset_heavy"))
        self.last_engine = engine
        ok = exact_assets(content)
        return ok, content, ""


class LightpandaArm:
    """Scrape through the native Lightpanda fetch command."""

    name = "lightpanda"

    def __init__(self, binary: str):
        self.binary = binary
        self.version = self._read_version()
        self.last_engine = "lightpanda"

    def _read_version(self) -> str:
        completed = subprocess.run(
            [self.binary, "version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"lightpanda version failed: {(completed.stderr or completed.stdout)[:300]}")
        return (completed.stdout or completed.stderr).strip()

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        return

    def _fetch(
        self,
        url: str,
        dump: str,
        *,
        wait_ms: int | None = None,
        wait_script: str | None = None,
    ) -> str:
        command = [self.binary, "fetch", url, "--dump", dump, "--json"]
        if wait_ms is not None:
            command.extend(["--wait-ms", str(wait_ms), "--wait-until", "load"])
        if wait_script is not None:
            command.extend(["--wait-script", wait_script])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-500:]
            raise RuntimeError(f"lightpanda fetch exited {completed.returncode}: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"lightpanda returned invalid JSON: {exc}") from exc
        if payload.get("error"):
            raise RuntimeError(f"lightpanda fetch error: {payload['error']}")
        return str(payload.get("content") or "")

    async def run_static(self, base: str):
        content = self._fetch(f"{base}/static", "markdown", wait_ms=100)
        ok = exact_static(content)
        return ok, content, ""

    async def run_js_list(self, base: str):
        content = self._fetch(
            f"{base}/js-list",
            "markdown",
            wait_script="document.querySelectorAll('#items [data-sku]').length === 30",
        )
        expected = [f"Product {index}{index * 10}" for index in range(JS_ITEMS)]
        cursor = 0
        ordered = True
        for token in expected:
            found = content.find(token, cursor)
            if found < 0:
                ordered = False
                break
            cursor = found + len(token)
        ok = ordered and content.count("Product") == JS_ITEMS
        return ok, content, ""

    async def run_pagination(self, base: str):
        ids: list[str] = []
        pages: list[str] = []
        for page in range(1, PAGE_COUNT + 1):
            content = self._fetch(f"{base}/pagination?page={page}", "html", wait_ms=100)
            pages.append(content)
            ids.extend(re.findall(r'data-id="(page-[^"]+)"', content))
        expected = [f"page-{page}-row-{row}" for page in range(1, PAGE_COUNT + 1) for row in range(PAGE_SIZE)]
        ok = ids == expected
        return ok, "\n".join(pages), ""

    async def run_table(self, base: str):
        content = self._fetch(f"{base}/table", "html", wait_ms=100)
        rows = [
            [row_id, *[_cell_text(cell) for cell in re.findall(r"<td>(.*?)</td>", body, re.S)]]
            for row_id, body in re.findall(r'<tr data-id="(table-[^"]+)">(.*?)</tr>', content, re.S)
        ]
        ok = exact_table(rows)
        return ok, content, ""

    async def run_assets(self, base: str):
        content = self._fetch(
            f"{base}/assets",
            "markdown",
            wait_script="document.querySelectorAll('#headlines h2').length === 20",
        )
        ok = exact_assets(content)
        return ok, content, ""


TASKS = {
    "static_article": "run_static",
    "js_list": "run_js_list",
    "pagination": "run_pagination",
    "table": "run_table",
    "asset_heavy": "run_assets",
}


async def _prepare(arm, task_name: str) -> None:
    prepare = getattr(arm, "prepare", None)
    if prepare is not None:
        await prepare(task_name)


async def _measure(arm, task_name: str, method: str, base: str, state: FixtureState):
    before_requests, before_bytes = state.snapshot()
    started = time.perf_counter_ns()
    error = ""
    try:
        ok, output, detail = await getattr(arm, method)(base)
        error = detail if not ok else ""
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


def _resolve_firecrawl_binary(explicit: str | None) -> str | None:
    candidate = explicit or os.environ.get("FIRECRAWL_LOCAL_BIN")
    if candidate:
        return candidate
    return shutil.which("firecrawl-local")


def _chrome_version(path: str | None) -> str:
    if not path:
        return ""
    try:
        completed = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or completed.stderr).strip()


def _environment(
    firecrawl_server: FirecrawlServer | None,
    firecrawl_waits: dict[str, dict[str, int]],
    chrome_path: str | None = None,
    lightpanda_arm: LightpandaArm | None = None,
) -> dict:
    try:
        from openbrowser.utils import get_openbrowser_version

        openbrowser_version = get_openbrowser_version()
    except Exception:
        openbrowser_version = ""
    environment: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "openbrowser": openbrowser_version,
        "chrome_path": chrome_path,
        "chrome_version": _chrome_version(chrome_path),
        "firecrawl": None,
        "lightpanda": None,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }
    if lightpanda_arm is not None:
        environment["lightpanda"] = {
            "version": lightpanda_arm.version,
            "binary": lightpanda_arm.binary,
            "wait_policy": {
                "static": "wait-until load, wait-ms 100",
                "js": "wait-script, default wait budget",
                "resources": "Lightpanda fetch default, no subresources",
            },
        }
    if firecrawl_server is not None:
        environment["firecrawl"] = {
            "version": firecrawl_server.version,
            "url": firecrawl_server.url,
            "binary": firecrawl_server.binary,
            "managed_process": firecrawl_server.process is not None,
            "cloud": "off",
            "serve_flags": "--allow-private --browser auto",
            "chrome_path": firecrawl_server.chrome_path,
            "wait_ms_by_arm": firecrawl_waits,
        }
    return environment


def make_scrape_arm(
    name: str,
    *,
    chrome_path: str | None = None,
    firecrawl_server: FirecrawlServer | None = None,
    firecrawl_waits: dict[str, dict[str, int]] | None = None,
    lightpanda_bin: str | None = None,
):
    """Construct the requested native scraping surface without starting it."""
    modes = {"openbrowser-full": "full", "openbrowser-lightweight": "lightweight", "openbrowser-no-render": "no-render"}
    if name in modes:
        return BrowserArm(name, modes[name], chrome_path=chrome_path)
    if name in {"playwright-mcp", "playwright-cli"}:
        from benchmarks.interactive.adapters.playwright_scrape_bridge import build_scrape_arm

        return build_scrape_arm(name)
    if name in {"firecrawl-local", "firecrawl-local-auto"}:
        if firecrawl_server is None:
            raise ValueError("Firecrawl scraping needs a configured service")
        waits = {"asset_heavy": FIRECRAWL_ASSET_WAIT_MS} if name == "firecrawl-local" else {}
        if firecrawl_waits is not None:
            firecrawl_waits[name] = waits
        return FirecrawlArm(name, firecrawl_server, waits)
    if name == "lightpanda":
        binary = lightpanda_bin or os.environ.get("LIGHTPANDA_BIN") or shutil.which("lightpanda")
        if not binary:
            raise SystemExit("Lightpanda arm needs a binary. Pass --lightpanda-bin or set LIGHTPANDA_BIN.")
        return LightpandaArm(binary)
    raise SystemExit(f"unknown arm: {name}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    default_arms = ["openbrowser-full", "openbrowser-lightweight", "openbrowser-no-render"]
    parser.add_argument("--arms", nargs="+", default=default_arms)
    parser.add_argument(
        "--firecrawl-url",
        default=None,
        help="Attach to a running firecrawl-local serve endpoint, for example http://127.0.0.1:3002.",
    )
    parser.add_argument(
        "--firecrawl-bin",
        default=None,
        help="Path to the firecrawl-local binary. Starts a managed server on a free loopback port.",
    )
    parser.add_argument(
        "--chrome",
        default=None,
        help="Chrome or Chromium binary used by both arms: BrowserProfile.executable_path and FIRECRAWL_CHROME_PATH.",
    )
    parser.add_argument(
        "--lightpanda-bin",
        default=None,
        help="Path to the Lightpanda binary. Falls back to LIGHTPANDA_BIN or PATH.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/scraping/results.json"))
    parser.add_argument("--raw-jsonl", type=Path, default=None, help="Write one JSON object per measured sample.")
    parser.add_argument("--interleave", action="store_true", help="Alternate arm order for each measured sample.")
    args = parser.parse_args()
    load_average_start = list(os.getloadavg()) if hasattr(os, "getloadavg") else None

    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = server.base_url

    firecrawl_server: FirecrawlServer | None = None
    firecrawl_waits: dict[str, dict[str, int]] = {}
    arms: list = []
    results: list[TaskResult] = []
    raw_samples: list[dict] = []
    environment: dict = {}
    cleanup_errors: list[str] = []
    stopped: set[str] = set()

    async def stop_arm(arm) -> None:
        if arm.name in stopped:
            return
        stopped.add(arm.name)
        try:
            await arm.stop()
        except Exception as error:
            cleanup_errors.append(f"{arm.name}: {type(error).__name__}")

    try:
        if any(name.startswith("firecrawl") for name in args.arms):
            if args.firecrawl_url:
                firecrawl_server = FirecrawlServer(url=args.firecrawl_url, chrome_path=args.chrome)
            else:
                binary = _resolve_firecrawl_binary(args.firecrawl_bin)
                if not binary:
                    raise SystemExit(
                        "Firecrawl arms need firecrawl-local. Install it with "
                        "`cargo install firecrawl-local --version 0.1.1 --root <tools-dir>`, "
                        "pass --firecrawl-bin, set FIRECRAWL_LOCAL_BIN, or attach with --firecrawl-url."
                    )
                firecrawl_server = FirecrawlServer(binary=binary, chrome_path=args.chrome)
            firecrawl_server.start()
            print(f"firecrawl-local {firecrawl_server.version} at {firecrawl_server.url}")

        for name in args.arms:
            arms.append(
                make_scrape_arm(
                    name,
                    chrome_path=args.chrome,
                    firecrawl_server=firecrawl_server,
                    firecrawl_waits=firecrawl_waits,
                    lightpanda_bin=args.lightpanda_bin,
                )
            )

        lightpanda_arm = next((arm for arm in arms if isinstance(arm, LightpandaArm)), None)
        environment = _environment(firecrawl_server, firecrawl_waits, args.chrome, lightpanda_arm)
        environment["load_average_start"] = load_average_start
        if args.interleave:
            for arm in arms:
                await arm.start()
            for task_name in args.tasks:
                method = TASKS[task_name]
                for arm in arms:
                    await _prepare(arm, task_name)
                for warmup_index in range(args.warmups):
                    order = arms if warmup_index % 2 == 0 else list(reversed(arms))
                    for arm in order:
                        await _measure(arm, task_name, method, base, server.state)
                for sample_index in range(args.samples):
                    order = arms if sample_index % 2 == 0 else list(reversed(arms))
                    for arm_position, arm in enumerate(order, start=1):
                        result = await _measure(arm, task_name, method, base, server.state)
                        results.append(result)
                        raw_samples.append(
                            {
                                "record_type": "sample",
                                "sample_index": sample_index + 1,
                                "arm_position": arm_position,
                                **asdict(result),
                            }
                        )
        else:
            for arm_position, arm in enumerate(arms, start=1):
                await arm.start()
                for task_name in args.tasks:
                    method = TASKS[task_name]
                    await _prepare(arm, task_name)
                    for _ in range(args.warmups):
                        await _measure(arm, task_name, method, base, server.state)
                    for sample_index in range(args.samples):
                        result = await _measure(arm, task_name, method, base, server.state)
                        results.append(result)
                        raw_samples.append(
                            {
                                "record_type": "sample",
                                "sample_index": sample_index + 1,
                                "arm_position": arm_position,
                                **asdict(result),
                            }
                        )
                await stop_arm(arm)
    finally:
        await asyncio.gather(*(stop_arm(arm) for arm in arms))
        for name, stop in (
            ("firecrawl-service", firecrawl_server.stop if firecrawl_server is not None else None),
            ("fixture-shutdown", server.shutdown),
            ("fixture-close", server.server_close),
        ):
            if stop is not None:
                try:
                    stop()
                except Exception as error:
                    cleanup_errors.append(f"{name}: {type(error).__name__}")
        environment["cleanup_completed"] = not cleanup_errors
        environment["cleanup_errors"] = cleanup_errors

    grouped: dict[tuple[str, str], list[TaskResult]] = {}
    for result in results:
        grouped.setdefault((result.task, result.arm), []).append(result)

    environment["load_average_end"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    report = {
        "samples": args.samples,
        "warmups": args.warmups,
        "arms": args.arms,
        "tasks": args.tasks,
        "environment": environment,
        "groups": [],
    }
    print(f"{'task':18} {'arm':24} {'ok':>4} {'p50 ms':>8} {'p95 ms':>8} {'bytes':>8} {'req':>4} {'engine':<12}")
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
        engine = ",".join(entry["engines"])
        p50 = f"{summary['p50_ms']:8.1f}" if summary.get("p50_ms") is not None else f"{'n/a':>8}"
        p95 = f"{summary['p95_ms']:8.1f}" if summary.get("p95_ms") is not None else f"{'n/a':>8}"
        print(
            f"{task:18} {arm_name:24} {ok_count}/{len(group):<2} "
            f"{p50} {p95} "
            f"{entry['mean_output_bytes']:8.0f} {entry['mean_requests']:4.1f} {engine:<12}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    if args.raw_jsonl is not None:
        raw_run = {
            "record_type": "run",
            "samples": args.samples,
            "warmups": args.warmups,
            "arms": args.arms,
            "tasks": args.tasks,
            "sample_order": "alternating" if args.interleave else "grouped",
            "environment": environment,
        }
        args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(record, separators=(",", ":"), sort_keys=True) for record in [raw_run, *raw_samples]]
        args.raw_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.raw_jsonl}")
    expected = len(args.arms) * len(args.tasks) * args.samples
    return 0 if results and len(results) == expected and all(item.ok for item in results) and not cleanup_errors else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
