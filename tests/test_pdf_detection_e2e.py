"""Real-browser PDF detection and validated auto-download integration test."""

# Test assertions intentionally inspect files synchronously after awaiting the browser.
# ruff: noqa: ASYNC240

import asyncio
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import pytest

from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.browser.download_state import PdfDetectionPhase

pytestmark = pytest.mark.integration

MINIMAL_PDF = (
    b"%PDF-1.0\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n212\n%%EOF\n"
)


class _PdfHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _pdf_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/report.pdf"

    def do_GET(self):
        if self.path.startswith("/report.pdf") or self.path.startswith("/export?format=pdf"):
            body = MINIMAL_PDF
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/viewer-embed"):
            body = (
                f'<!doctype html><html><body><embed type="application/pdf" src="{self._pdf_url()}"></body></html>'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/viewer-query") or self.path.startswith("/viewer-about-blank"):
            embed = ""
            if self.path.startswith("/viewer-about-blank"):
                embed = '<embed type="application/pdf" src="about:blank">'
            body = f"<!doctype html><html><body>{embed}<p>viewer</p></body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"<!doctype html><html><body>home</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.parametrize("pdf_path", ["/report.pdf", "/export?format=pdf"])
async def test_real_pdf_navigation_detects_and_saves_validated_bytes(pdf_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PdfHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    pdf_url = f"http://127.0.0.1:{server.server_port}{pdf_path}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        browser = BrowserSession(
            browser_profile=BrowserProfile(
                headless=True,
                user_data_dir=str(tmp_path / "profile"),
                downloads_path=str(tmp_path / "downloads"),
                auto_download_pdfs=True,
            )
        )
        try:
            await asyncio.wait_for(browser.start(), timeout=60)
            await asyncio.wait_for(browser.navigate_to(pdf_url), timeout=30)
            watchdog = browser._downloads_watchdog
            deadline = time.monotonic() + 20
            snapshot = watchdog.pdf_detection_snapshot()
            while time.monotonic() < deadline:
                snapshot = watchdog.pdf_detection_snapshot()
                if snapshot.phase in {
                    PdfDetectionPhase.PDF,
                    PdfDetectionPhase.NOT_PDF,
                    PdfDetectionPhase.CHECK_FAILED,
                }:
                    break
                await asyncio.sleep(0.05)

            assert snapshot.phase is PdfDetectionPhase.PDF, snapshot.to_dict()
            if pdf_path.startswith("/export"):
                assert snapshot.detail and "main document response mime" in snapshot.detail
            target_id = browser.agent_focus.target_id if browser.agent_focus else None
            assert target_id is not None

            deadline = time.monotonic() + 10
            auto_completed = []
            while time.monotonic() < deadline:
                auto_completed = [
                    outcome
                    for outcome in browser.download_outcomes()
                    if outcome.get("phase") == "completed"
                    and outcome.get("source") in {"pdf", "session_cache", "cdp", "native_poll", "playwright"}
                    and outcome.get("path")
                ]
                if auto_completed:
                    break
                await asyncio.sleep(0.05)
            assert auto_completed, "navigation did not complete an automatic PDF download"
            auto_path = Path(str(auto_completed[-1]["path"]))
            assert auto_path.exists()
            assert auto_path.read_bytes().startswith(b"%PDF-")

            # The observer and navigation trigger share one session-cached file.
            download_path = await watchdog.trigger_pdf_download(
                target_id,
                resource_url=snapshot.resource_url,
            )
            assert download_path is not None
            cache_url = str(snapshot.resource_url or pdf_url)
            final_cached_path = watchdog._session_pdf_urls.get(cache_url)
            assert final_cached_path is not None
            assert download_path == final_cached_path
            saved = Path(download_path)
            assert saved.exists()
            assert saved.read_bytes().startswith(b"%PDF-")

            matching = [outcome for outcome in browser.download_outcomes() if outcome.get("path") == download_path]
            completed = [outcome for outcome in matching if outcome.get("phase") == "completed"]
            assert completed
            assert completed[-1]["file_size"] == saved.stat().st_size
            files = list((tmp_path / "downloads").glob("*.pdf"))
            assert files
            assert all(path.read_bytes().startswith(b"%PDF-") for path in files)
            assert list((tmp_path / "downloads").glob("*.tmp")) == []
        finally:
            await asyncio.wait_for(browser.kill(), timeout=30)
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("viewer_route", ["viewer-embed", "viewer-query", "viewer-about-blank"])
async def test_real_viewer_resolves_the_underlying_pdf_resource(viewer_route):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PdfHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    viewer_url = f"{base}/{viewer_route}"
    if viewer_route != "viewer-embed":
        viewer_url = f"{viewer_url}?file={quote(f'{base}/report.pdf', safe='')}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        browser = BrowserSession(
            browser_profile=BrowserProfile(
                headless=True,
                user_data_dir=str(tmp_path / "profile"),
                downloads_path=str(tmp_path / "downloads"),
                auto_download_pdfs=False,
            )
        )
        try:
            await asyncio.wait_for(browser.start(), timeout=60)
            await asyncio.wait_for(browser.navigate_to(viewer_url), timeout=30)
            watchdog = browser._downloads_watchdog
            watchdog._pdf_detection_resource_url = None
            watchdog._session_pdf_urls.clear()
            target_id = browser.agent_focus.target_id if browser.agent_focus else None
            assert target_id is not None

            download_path = await watchdog.trigger_pdf_download(target_id)

            assert download_path is not None
            saved = Path(download_path)
            assert saved.exists()
            assert saved.read_bytes() == MINIMAL_PDF
            assert len(list((tmp_path / "downloads").glob("*.pdf"))) == 1
        finally:
            await asyncio.wait_for(browser.kill(), timeout=30)
            server.shutdown()
            thread.join(timeout=5)
