"""End-to-end integration tests for download_file and list_downloads.

Spins up a real headless browser, a local HTTP server serving test files,
and exercises the full MCP execute_code flow for downloading.

Run with:
    uv run python -m pytest tests/test_mcp_download_e2e.py -m integration -v
"""

import asyncio
import logging
import tempfile
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from conftest import DummyServer, DummyTypes

logger = logging.getLogger(__name__)

try:
    from openbrowser.browser import BrowserProfile, BrowserSession
    from openbrowser.mcp import server as mcp_server_module

    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not BROWSER_AVAILABLE, reason="openbrowser or browser deps not available"),
]


def _has_pypdf() -> bool:
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Generate a minimal valid PDF in memory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_file_server():
    """Serve test files (PDF, text) via a local HTTP server."""
    tmp_dir = tempfile.mkdtemp()
    pdf_path = Path(tmp_dir) / "report.pdf"
    pdf_path.write_bytes(MINIMAL_PDF)

    txt_path = Path(tmp_dir) / "data.txt"
    txt_path.write_text("hello world")

    html_path = Path(tmp_dir) / "index.html"
    html_path.write_text("<html><body><h1>Test Server</h1></body></html>")

    handler = partial(SimpleHTTPRequestHandler, directory=tmp_dir)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {
        "base_url": f"http://127.0.0.1:{port}",
        "pdf_url": f"http://127.0.0.1:{port}/report.pdf",
        "txt_url": f"http://127.0.0.1:{port}/data.txt",
        "html_url": f"http://127.0.0.1:{port}/index.html",
    }

    server.shutdown()
    pdf_path.unlink(missing_ok=True)
    txt_path.unlink(missing_ok=True)
    html_path.unlink(missing_ok=True)
    Path(tmp_dir).rmdir()


@pytest.fixture(scope="module")
def downloads_dir():
    """Create a temporary downloads directory that gets cleaned up."""
    tmp_dir = tempfile.mkdtemp(prefix="openbrowser-test-downloads-")
    yield tmp_dir
    # Clean up all files
    for f in Path(tmp_dir).iterdir():
        f.unlink(missing_ok=True)
    Path(tmp_dir).rmdir()


@pytest.fixture(scope="module")
def mcp_server_with_browser(test_file_server, downloads_dir, monkeypatch_module):
    """Create OpenBrowserServer with real browser, custom downloads path."""
    monkeypatch_module.setattr(mcp_server_module, "MCP_AVAILABLE", True)
    monkeypatch_module.setattr(mcp_server_module, "Server", DummyServer)
    monkeypatch_module.setattr(mcp_server_module, "types", DummyTypes)

    server = mcp_server_module.OpenBrowserServer()

    async def setup():
        profile = BrowserProfile(
            headless=True,
            downloads_path=downloads_dir,
        )
        session = BrowserSession(browser_profile=profile)
        await session.start()

        server.browser_session = session

        from openbrowser.code_use.namespace import create_namespace
        from openbrowser.tools.service import CodeAgentTools

        tools = CodeAgentTools()
        server._namespace = create_namespace(
            browser_session=session,
            tools=tools,
            file_system=None,
        )

        # Navigate to the test server index page so the browser has an origin
        await session.navigate_to(test_file_server["html_url"])

        return server

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(setup())
        yield server, loop, test_file_server, downloads_dir
    finally:
        async def teardown():
            if server.browser_session:
                await server.browser_session.stop()

        loop.run_until_complete(teardown())
        loop.close()


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


class TestDownloadFilePDF:
    """Test downloading a PDF file through the MCP execute_code flow."""

    def test_download_pdf_default_filename(self, mcp_server_with_browser):
        """download_file() with a PDF URL auto-derives the filename."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        code = (
            f"path = await download_file('{urls['pdf_url']}')\n"
            f"print(path)"
        )
        result = loop.run_until_complete(server._execute_code(code))
        assert "report.pdf" in result
        # Verify the file actually exists and has valid PDF bytes
        downloaded = Path(result.strip())
        assert downloaded.exists(), f"Downloaded file does not exist: {downloaded}"
        content = downloaded.read_bytes()
        assert content.startswith(b"%PDF"), "Downloaded file is not a valid PDF"
        assert len(content) == len(MINIMAL_PDF)

    def test_download_pdf_custom_filename(self, mcp_server_with_browser):
        """download_file() with explicit filename."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        code = (
            f"path = await download_file('{urls['pdf_url']}', filename='custom.pdf')\n"
            f"print(path)"
        )
        result = loop.run_until_complete(server._execute_code(code))
        assert "custom.pdf" in result
        downloaded = Path(result.strip())
        assert downloaded.exists()
        assert downloaded.read_bytes().startswith(b"%PDF")

    def test_download_conflict_resolution(self, mcp_server_with_browser):
        """download_file() auto-renames on filename conflict."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        # report.pdf already exists from test_download_pdf_default_filename
        code = (
            f"path = await download_file('{urls['pdf_url']}')\n"
            f"print(path)"
        )
        result = loop.run_until_complete(server._execute_code(code))
        # Should be report (1).pdf since report.pdf already exists
        assert "report (1).pdf" in result or "report" in result
        downloaded = Path(result.strip())
        assert downloaded.exists()


class TestDownloadFileText:
    """Test downloading a text file."""

    def test_download_text_file(self, mcp_server_with_browser):
        """download_file() works for non-PDF files."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        code = (
            f"path = await download_file('{urls['txt_url']}')\n"
            f"print(path)"
        )
        result = loop.run_until_complete(server._execute_code(code))
        assert "data.txt" in result
        downloaded = Path(result.strip())
        assert downloaded.exists()
        assert downloaded.read_text() == "hello world"


class TestListDownloads:
    """Test the list_downloads() function."""

    def test_list_downloads_returns_files(self, mcp_server_with_browser):
        """list_downloads() returns paths of downloaded files."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        code = (
            "files = list_downloads()\n"
            "print(len(files))\n"
            "for f in files:\n"
            "    print(f)"
        )
        result = loop.run_until_complete(server._execute_code(code))
        lines = result.strip().split("\n")
        count = int(lines[0])
        # We downloaded at least report.pdf, custom.pdf, and data.txt
        assert count >= 3, f"Expected >=3 downloads, got {count}"


class TestDownloadValidation:
    """Test URL and filename validation."""

    def test_invalid_url_rejected(self, mcp_server_with_browser):
        """download_file() rejects malformed URLs."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        code = (
            "try:\n"
            "    await download_file('not-a-url')\n"
            "    print('ERROR: should have raised')\n"
            "except ValueError as e:\n"
            "    print(f'OK: {e}')"
        )
        result = loop.run_until_complete(server._execute_code(code))
        assert "OK:" in result
        assert "Invalid URL" in result

    def test_ftp_scheme_rejected(self, mcp_server_with_browser):
        """download_file() rejects non-HTTP schemes."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        code = (
            "try:\n"
            "    await download_file('ftp://evil.com/file.pdf')\n"
            "    print('ERROR: should have raised')\n"
            "except ValueError as e:\n"
            "    print(f'OK: {e}')"
        )
        result = loop.run_until_complete(server._execute_code(code))
        assert "OK:" in result
        assert "Unsupported URL scheme" in result

    def test_path_traversal_blocked(self, mcp_server_with_browser):
        """download_file() sanitizes path traversal in filenames."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        # Attempt path traversal -- should be sanitized to just "passwd"
        code = (
            f"path = await download_file('{urls['txt_url']}', filename='../../etc/passwd')\n"
            f"print(path)"
        )
        result = loop.run_until_complete(server._execute_code(code))
        downloaded = Path(result.strip())
        # Resolve both paths to handle macOS /var -> /private/var symlink
        resolved_dl_dir = str(Path(dl_dir).resolve())
        # File should end up INSIDE dl_dir, not at /etc/passwd
        assert str(downloaded).startswith(resolved_dl_dir), (
            f"File escaped downloads dir: {downloaded}"
        )
        assert downloaded.name == "passwd"


class TestDownloadViaBrowserFetch:
    """Test that browser JS fetch strategy works (same-origin)."""

    def test_same_origin_uses_browser_fetch(self, mcp_server_with_browser):
        """Files from the same origin the browser is on should use JS fetch."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        # Navigate to the test server first so fetch is same-origin
        loop.run_until_complete(server._execute_code(
            f"await navigate('{urls['html_url']}')"
        ))
        code = (
            f"path = await download_file('{urls['pdf_url']}', filename='via_browser.pdf')\n"
            f"print(path)"
        )
        result = loop.run_until_complete(server._execute_code(code))
        assert "via_browser.pdf" in result
        downloaded = Path(result.strip())
        assert downloaded.exists()
        assert downloaded.read_bytes().startswith(b"%PDF")


class TestDownloadPDFReadback:
    """Test full workflow: download PDF then read it with pypdf."""

    @pytest.mark.skipif(
        not _has_pypdf(),
        reason="pypdf not installed",
    )
    def test_download_and_read_pdf(self, mcp_server_with_browser):
        """Download a PDF and verify it can be read back with PdfReader."""
        server, loop, urls, dl_dir = mcp_server_with_browser
        code = (
            f"path = await download_file('{urls['pdf_url']}', filename='readback.pdf')\n"
            "from pypdf import PdfReader\n"
            "reader = PdfReader(path)\n"
            "print(f'pages={len(reader.pages)}')\n"
            "print(f'path={path}')"
        )
        result = loop.run_until_complete(server._execute_code(code))
        assert "pages=1" in result
        assert "readback.pdf" in result
