"""Measure the browser-free MCP scrape_web memory and concurrency lane."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "results" / "results-mcp-scrape-web-resource-2026-11-26-n5.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BROWSER_NAME_TOKENS = ("chrome", "chromium", "headless_shell", "msedge")
EXPECTED_TEXT = ("OpenBrowser static content. " * 500).rstrip()


class _StaticPageHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler method
        body = (
            b"<html><head><title>Memory page</title></head><body><article>"
            + (b"OpenBrowser static content. " * 500)
            + b"</article></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):
        return


def _binary() -> str:
    override = os.environ.get("OPENBROWSER_BIN")
    if override:
        return override
    sibling = Path(sys.executable).with_name("openbrowser-ai")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    found = shutil.which("openbrowser-ai")
    if found:
        return found
    raise SystemExit("Set OPENBROWSER_BIN or install openbrowser-ai")


def _client_env(home: Path) -> dict[str, str]:
    config = home / "config"
    config.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "OPENBROWSER_CONFIG_DIR": str(config),
            "OPENBROWSER_SETUP_LOGGING": "false",
            "OPENBROWSER_DISABLE_TELEMETRY": "true",
            "ANONYMIZED_TELEMETRY": "false",
        }
    )
    return env


def _is_browser_process(child: psutil.Process) -> bool:
    try:
        name = (child.name() or "").lower()
    except psutil.Error:
        return False
    return any(token in name for token in BROWSER_NAME_TOKENS)


def _peak_tree_rss(process: psutil.Process, stop: threading.Event, result: dict[str, object]) -> None:
    peak: int | None = None
    max_children: int | None = None
    browser_seen: bool | None = None
    while not stop.is_set():
        try:
            children = process.children(recursive=True)
            max_children = len(children) if max_children is None else max(max_children, len(children))
            total = process.memory_info().rss
            found_any = False
            for child in children:
                found_any = found_any or _is_browser_process(child)
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    continue
            browser_seen = found_any if browser_seen is None else (browser_seen or found_any)
            peak = total if peak is None else max(peak, total)
        except psutil.Error:
            pass
        time.sleep(0.005)
    result["peak_rss_mib"] = None if peak is None else round(peak / (1024 * 1024), 3)
    result["max_children"] = max_children
    result["browser_processes_observed"] = browser_seen


def _result_passes(text: str) -> bool:
    """Decode one scrape_web response and check the exact fixture evidence."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if payload.get("ok") is not True:
        return False
    if payload.get("engine_used") != "http":
        return False
    content = payload.get("content")
    return (
        isinstance(content, str)
        and content == EXPECTED_TEXT[:5000]
        and payload.get("truncated") is True
        and payload.get("total_chars") == len(EXPECTED_TEXT)
    )


def _one_client(binary: str, url: str, calls: int) -> dict:
    from benchmarks.interactive.adapters.mcp_modern import ModernStdioMcpClient

    home = Path(tempfile.mkdtemp(prefix="mcp-http-resource-"))
    client = ModernStdioMcpClient([binary, "--mcp"], env=_client_env(home))
    process = psutil.Process(client.process.pid)
    stop = threading.Event()
    result: dict[str, object] = {}
    sampler = threading.Thread(target=_peak_tree_rss, args=(process, stop, result), daemon=True)
    sampler.start()
    texts: list[str] = []
    try:
        client.discover()
        started = time.monotonic()
        if calls == 1:
            texts = [
                client.call_tool_text(
                    "scrape_web",
                    {"url": url, "format": "text", "max_chars": 5000},
                    timeout=60,
                )
            ]
        else:
            with ThreadPoolExecutor(max_workers=calls) as executor:
                texts = list(
                    executor.map(
                        lambda index: client.call_tool_text(
                            "scrape_web",
                            {"url": f"{url}?i={index}", "format": "text", "max_chars": 5000},
                            timeout=60,
                        ),
                        range(calls),
                    )
                )
        elapsed = (time.monotonic() - started) * 1000.0
        children = len(process.children(recursive=True))
    finally:
        stop.set()
        sampler.join(timeout=2)
        client.close()
    return {
        "calls": calls,
        "latency_ms": round(elapsed, 3),
        "peak_rss_mib": result.get("peak_rss_mib"),
        "max_children": result.get("max_children"),
        "browser_processes_observed": result.get("browser_processes_observed"),
        "children_at_end": children,
        "ok": sum(_result_passes(text) for text in texts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--concurrency-samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _StaticPageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        binary = _binary()
        sequential = [_one_client(binary, url, 1) for _ in range(args.samples)]
        concurrent = [_one_client(binary, url, args.concurrency) for _ in range(args.concurrency_samples)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    rows = sequential + concurrent
    browser_started = any(row["browser_processes_observed"] is True for row in rows)
    valid = all(
        row["ok"] == row["calls"] and row["peak_rss_mib"] is not None and row["browser_processes_observed"] is False
        for row in rows
    )
    report = {
        "lane": "mcp-scrape-web-resource",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "surface": "stateless scrape_web over MCP stdio",
        "browser_started": browser_started,
        "valid": valid,
        "sequential": sequential,
        "concurrent": concurrent,
        "concurrency": args.concurrency,
        "limits": [
            "One local static fixture, not the live web.",
            "Peak RSS is process-tree resident set size in MiB and can count shared pages more than once.",
            "The Python MCP server process and any observed descendants are included.",
            "The no-browser mode cannot execute JavaScript, take screenshots, or use browser cookies.",
            "Each concurrent row is one observation unless concurrency-samples is raised.",
            "Do not compare this RSS total to a PSS or per-process figure without stating the scope.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not valid:
        print("Resource lane is not graph-eligible: an oracle, RSS sample, or browser check failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
