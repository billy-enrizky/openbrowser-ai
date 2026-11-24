"""Compare OpenBrowser HTTP batch scraping with Firecrawl Local batch scraping.

Both arms fetch the same five deterministic pagination pages and return the
same 100 ordered names. Firecrawl Local's native batch endpoint is asynchronous:
the arm measures the POST, job polling, and result download as one task. The
OpenBrowser arm uses ``scrape_pages(engine="http")`` and never starts a browser
for these static pages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .batch_compare import PAGE_COUNT, PAGE_SIZE, FixtureServer
from .scrape_compare import FirecrawlServer, _pipe_rows

POLL_INTERVAL_S = 0.01


def _openbrowser_binary() -> str | None:
    return os.environ.get("FIRECRAWL_LOCAL_BIN")


def _firecrawl_binary(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("FIRECRAWL_LOCAL_BIN"),
        str(Path.home() / ".local" / "firecrawl-tools" / "bin" / "firecrawl-local"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("firecrawl-local not found; pass --firecrawl-bin or set FIRECRAWL_LOCAL_BIN")


@dataclass
class Sample:
    arm: str
    ok: bool
    latency_ms: float
    detail: str = ""
    output_bytes: int = 0
    requests: int = 0


class OpenBrowserBatchArm:
    name = "openbrowser-http-batch"

    def __init__(self) -> None:
        self.session = None
        self.namespace: dict[str, Any] | None = None
        self.browser_started = False

    async def start(self) -> None:
        from openbrowser.browser import BrowserProfile, BrowserSession
        from openbrowser.code_use.namespace import create_namespace
        from openbrowser.tools.service import CodeAgentTools

        self.session = BrowserSession(
            browser_profile=BrowserProfile(
                headless=True,
                enable_default_extensions=False,
                args=["--use-mock-keychain"],
            )
        )
        self.namespace = create_namespace(browser_session=self.session, tools=CodeAgentTools())

    async def stop(self) -> None:
        if self.session is not None:
            with __import__("contextlib").suppress(Exception):
                await self.session.kill()

    async def run(self, urls: list[str], concurrency: int) -> tuple[bool, str]:
        assert self.namespace is not None
        results = await self.namespace["scrape_pages"](
            urls,
            concurrency=concurrency,
            format="json",
            selector="tbody tr",
            engine="http",
        )
        ids: list[str] = []
        for result in results:
            if not result.get("ok"):
                return False, str(result.get("error") or "batch item failed")
            for item in result.get("content") or []:
                attributes = item.get("attributes") or {}
                row_id = attributes.get("data-id")
                if row_id:
                    ids.append(row_id)
        self.browser_started = self.session._cdp_client_root is not None
        ok = len(ids) == PAGE_COUNT * PAGE_SIZE and len(set(ids)) == len(ids)
        return ok, json.dumps(ids)


class FirecrawlBatchArm:
    name = "firecrawl-local-batch"

    def __init__(self, binary: str) -> None:
        self.server = FirecrawlServer(binary=binary)
        self.client = httpx.Client(timeout=180.0)

    async def start(self) -> None:
        self.server.start()

    async def stop(self) -> None:
        self.client.close()
        self.server.stop()

    async def run(self, urls: list[str], concurrency: int) -> tuple[bool, str]:
        del concurrency  # The server owns batch concurrency.
        response = await asyncio.to_thread(
            self.client.post,
            f"{self.server.url}/v2/batch/scrape",
            json={"urls": urls, "formats": ["markdown"], "maxAge": 0, "storeInCache": False},
        )
        response.raise_for_status()
        job = response.json()
        job_id = job.get("id")
        if not job_id:
            return False, f"batch job did not return an id: {job}"
        polls = 0
        while True:
            polls += 1
            status_response = await asyncio.to_thread(
                self.client.get,
                f"{self.server.url}/v2/batch/scrape/{job_id}",
            )
            status_response.raise_for_status()
            payload = status_response.json()
            status = payload.get("status")
            if status in {"completed", "failed", "cancelled"}:
                break
            if polls > 3000:
                return False, f"batch job did not finish: {status}"
            await asyncio.sleep(POLL_INTERVAL_S)
        if status != "completed":
            return False, f"batch job ended {status}: {payload.get('errors')}"
        names: list[int] = []
        pages = 0
        for item in payload.get("data") or []:
            markdown = item.get("markdown") or ""
            rows = _pipe_rows(markdown)
            if rows:
                pages += 1
            for row in rows[1:]:
                if len(row) < 2:
                    continue
                match = re.fullmatch(r"Name (\d+)", row[1])
                if match:
                    names.append(int(match.group(1)))
        ok = pages == PAGE_COUNT and sorted(names) == list(range(PAGE_COUNT * PAGE_SIZE))
        return ok, "\n".join(item.get("markdown") or "" for item in payload.get("data") or [])


async def _measure(arm: Any, urls: list[str], concurrency: int, server: FixtureServer) -> Sample:
    server.state.clear()
    requests_before = server.state.snapshot()[0]
    started = time.perf_counter_ns()
    try:
        ok, output = await arm.run(urls, concurrency)
        detail = ""
    except Exception as error:
        ok, output, detail = False, "", f"{type(error).__name__}: {error}"
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    requests_after = server.state.snapshot()[0]
    return Sample(
        arm=arm.name,
        ok=ok,
        latency_ms=latency_ms,
        detail=detail,
        output_bytes=len(output.encode()),
        requests=requests_after - requests_before,
    )


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(values),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


async def _run(args: argparse.Namespace) -> int:
    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    urls = [f"{server.base_url}/pagination?page={page}" for page in range(1, PAGE_COUNT + 1)]
    arms = [OpenBrowserBatchArm(), FirecrawlBatchArm(_firecrawl_binary(args.firecrawl_bin))]
    for arm in arms:
        await arm.start()
    samples: list[Sample] = []
    try:
        for _ in range(args.warmups):
            for arm in arms:
                await _measure(arm, urls, args.concurrency, server)
        for index in range(args.samples):
            order = list(arms)
            if index % 2:
                order.reverse()
            for arm in order:
                samples.append(await _measure(arm, urls, args.concurrency, server))
    finally:
        for arm in arms:
            await arm.stop()
        server.shutdown()
        server.server_close()

    by_arm: dict[str, list[Sample]] = {}
    for sample in samples:
        by_arm.setdefault(sample.arm, []).append(sample)
    report: dict[str, Any] = {
        "samples": args.samples,
        "warmups": args.warmups,
        "concurrency": args.concurrency,
        "firecrawl_poll_interval_ms": POLL_INTERVAL_S * 1000,
        "openbrowser_browser_started": arms[0].browser_started,
        "tasks": {"pages": PAGE_COUNT, "rows": PAGE_COUNT * PAGE_SIZE},
        "groups": [],
    }
    latencies: dict[str, list[float]] = {}
    for arm in arms:
        group = by_arm.get(arm.name, [])
        ok_samples = [sample for sample in group if sample.ok]
        latencies[arm.name] = [sample.latency_ms for sample in ok_samples]
        report["groups"].append(
            {
                "arm": arm.name,
                "successes": len(ok_samples),
                "runs": len(group),
                "ok": len(ok_samples) == len(group) and bool(group),
                "latency": _summary(latencies[arm.name]),
                "mean_output_bytes": round(statistics.fmean(sample.output_bytes for sample in group), 1)
                if group
                else 0,
                "mean_requests": round(statistics.fmean(sample.requests for sample in group), 1) if group else 0,
                "errors": sorted({sample.detail for sample in group if sample.detail}),
            }
        )
    by_sample: dict[int, dict[str, float]] = {}
    for index, sample in enumerate(samples):
        if sample.ok:
            by_sample.setdefault(index // len(arms), {})[sample.arm] = sample.latency_ms
    deltas = [pair[arms[1].name] - pair[arms[0].name] for pair in by_sample.values() if len(pair) == 2]
    if deltas:
        report["paired"] = {
            "delta_ms_firecrawl_minus_openbrowser": {
                "p50": round(statistics.median(deltas), 3),
                "min": round(min(deltas), 3),
                "max": round(max(deltas), 3),
            },
            "openbrowser_faster_pairs": sum(1 for delta in deltas if delta > 0),
            "firecrawl_faster_pairs": sum(1 for delta in deltas if delta < 0),
            "pairs": len(deltas),
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(
                        {
                            "arm": sample.arm,
                            "ok": sample.ok,
                            "latency_ms": sample.latency_ms,
                            "detail": sample.detail,
                            "output_bytes": sample.output_bytes,
                            "requests": sample.requests,
                        }
                    )
                    + "\n"
                )
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all(group["ok"] for group in report["groups"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=PAGE_COUNT)
    parser.add_argument("--firecrawl-bin", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
