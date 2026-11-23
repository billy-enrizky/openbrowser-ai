"""Compare sequential and concurrent HTTP scraping on deterministic local pages.

The fixture serves five paginated pages with 20 rows each. Both arms use the
same OpenBrowser HTTP path and return the same 100 row IDs. The script measures
wall time, request count, and response bytes. No LLM, cloud service, or
external network is used.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE_COUNT = 5
PAGE_SIZE = 20


def _pagination_page(page: int) -> str:
    start = (page - 1) * PAGE_SIZE
    rows = "\n".join(
        f'<tr data-id="page-{page}-row-{index}"><td>{start + index}</td><td>Name {start + index}</td></tr>'
        for index in range(PAGE_SIZE)
    )
    return f"""<!doctype html>
<html><head><title>Pagination {page}</title></head>
<body><h1>Pagination {page}</h1>
<table><thead><tr><th>id</th><th>name</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""


@dataclass
class FixtureState:
    requests: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, path: str, status: int, body: bytes) -> None:
        with self.lock:
            self.requests.append({"path": path, "status": status, "bytes": len(body)})

    def snapshot(self) -> tuple[int, int]:
        with self.lock:
            return len(self.requests), sum(item["bytes"] for item in self.requests)

    def clear(self) -> None:
        with self.lock:
            self.requests.clear()


class FixtureHandler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, *_args, **_kwargs) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/pagination":
            body = b"not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.server.state.record(self.path, 404, body)
            return

        page = int(parse_qs(parsed.query).get("page", ["1"])[0])
        body = _pagination_page(page).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.server.state.record(self.path, 200, body)


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address=("127.0.0.1", 0)):
        super().__init__(address, FixtureHandler)
        self.state = FixtureState()

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


@dataclass
class Sample:
    arm: str
    ok: bool
    latency_ms: float
    requests: int
    response_bytes: int
    output_bytes: int
    detail: str = ""


def _ids_from_result(result: dict) -> list[str]:
    content = result.get("content")
    if not isinstance(content, list):
        return []
    ids = []
    for item in content:
        attributes = item.get("attributes") or {}
        row_id = attributes.get("data-id")
        if row_id:
            ids.append(row_id)
    return ids


def _ids_from_batch(results: list[dict]) -> list[str]:
    ids = []
    for result in results:
        if result.get("ok"):
            ids.extend(_ids_from_result(result))
    return ids


def _valid_ids(ids: list[str]) -> bool:
    expected = PAGE_COUNT * PAGE_SIZE
    return len(ids) == expected and len(set(ids)) == expected


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


async def _measure(arm: str, operation, server: FixtureServer) -> Sample:
    server.state.clear()
    before_requests, before_bytes = server.state.snapshot()
    started = time.perf_counter_ns()
    detail = ""
    try:
        ok, output = await operation()
    except Exception as error:
        ok, output = False, ""
        detail = f"{type(error).__name__}: {error}"
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    after_requests, after_bytes = server.state.snapshot()
    return Sample(
        arm=arm,
        ok=ok,
        latency_ms=latency_ms,
        requests=after_requests - before_requests,
        response_bytes=after_bytes - before_bytes,
        output_bytes=len(output.encode()),
        detail=detail,
    )


async def _run(args: argparse.Namespace) -> int:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    from openbrowser.browser import BrowserProfile, BrowserSession
    from openbrowser.code_use.namespace import create_namespace
    from openbrowser.tools.service import CodeAgentTools

    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    urls = [f"{server.base_url}/pagination?page={page}" for page in range(1, PAGE_COUNT + 1)]
    session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            enable_default_extensions=False,
            args=['--use-mock-keychain'],
        )
    )
    namespace = create_namespace(browser_session=session, tools=CodeAgentTools())
    samples: list[Sample] = []

    async def sequential():
        ids: list[str] = []
        for url in urls:
            result = await namespace["scrape_page"](
                url,
                format="json",
                selector="tbody tr",
                engine="http",
            )
            ids.extend(_ids_from_result(result))
        return _valid_ids(ids), json.dumps(ids)

    async def batch():
        results = await namespace["scrape_pages"](
            urls,
            concurrency=args.concurrency,
            format="json",
            selector="tbody tr",
            engine="http",
        )
        ids = _ids_from_batch(results)
        return _valid_ids(ids), json.dumps(ids)

    try:
        for _ in range(args.warmups):
            await _measure("sequential", sequential, server)
            await _measure("batch", batch, server)
        for index in range(args.samples):
            order = [("sequential", sequential), ("batch", batch)]
            if index % 2:
                order.reverse()
            for arm, operation in order:
                samples.append(await _measure(arm, operation, server))
    finally:
        with contextlib.suppress(Exception):
            await session.kill()
        server.shutdown()
        server.server_close()

    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.arm, []).append(sample)

    report = {
        "samples": args.samples,
        "warmups": args.warmups,
        "concurrency": args.concurrency,
        "tasks": {"pages": PAGE_COUNT, "rows": PAGE_COUNT * PAGE_SIZE},
        "groups": [],
    }
    for arm in ("sequential", "batch"):
        group = grouped.get(arm, [])
        ok_count = sum(1 for sample in group if sample.ok)
        latency = [sample.latency_ms for sample in group if sample.ok]
        output = [sample.output_bytes for sample in group if sample.ok]
        entry = {
            "arm": arm,
            "successes": ok_count,
            "runs": len(group),
            "ok": ok_count == len(group) and bool(group),
            "latency": _summary(latency),
            "mean_output_bytes": round(statistics.fmean(output), 1) if output else 0,
            "mean_requests": round(statistics.fmean(sample.requests for sample in group), 1) if group else 0,
            "mean_response_bytes": round(statistics.fmean(sample.response_bytes for sample in group), 1) if group else 0,
            "errors": sorted({sample.detail for sample in group if sample.detail}),
        }
        report["groups"].append(entry)

    sequential_p50 = report["groups"][0]["latency"].get("p50_ms")
    batch_p50 = report["groups"][1]["latency"].get("p50_ms")
    if sequential_p50 and batch_p50:
        report["p50_speedup"] = round(sequential_p50 / batch_p50, 3)

    print(f"{'arm':12} {'ok':>8} {'p50 ms':>10} {'p95 ms':>10} {'requests':>9} {'bytes':>9}")
    for entry in report["groups"]:
        latency = entry["latency"]
        print(
            f"{entry['arm']:12} {entry['successes']}/{entry['runs']:<6} "
            f"{latency.get('p50_ms', 0):10.1f} {latency.get('p95_ms', 0):10.1f} "
            f"{entry['mean_requests']:9.1f} {entry['mean_output_bytes']:9.0f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/scraping/batch_results.json"))
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
