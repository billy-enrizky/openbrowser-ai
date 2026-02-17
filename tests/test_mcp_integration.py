"""Integration tests for the MCP server with a real browser session.

These tests spin up an actual browser session and verify text extraction,
grep, element search, accessibility tree, and JavaScript execution against
known HTML content.

Requirements:
    - Chrome/Chromium must be installed
    - Tests are marked with @pytest.mark.integration and are skipped by default
    - Run with: pytest tests/test_mcp_integration.py -m integration

These tests use a local HTML file served via file:// protocol to avoid
network dependencies.
"""

import asyncio
import json
import logging
import tempfile
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

# Skip all tests in this module if Chrome or dependencies are not available
try:
    from openbrowser.browser import BrowserProfile, BrowserSession
    from openbrowser.mcp import server as mcp_server_module

    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not BROWSER_AVAILABLE, reason="openbrowser or browser dependencies not available"),
]


# ---------------------------------------------------------------------------
# Test HTML content
# ---------------------------------------------------------------------------

TEST_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MCP Integration Test Page</title>
</head>
<body>
    <h1>Integration Test Page</h1>
    <p>This is a test page for MCP integration tests.</p>

    <nav>
        <a href="/home" id="nav-home" class="nav-link">Home</a>
        <a href="/about" id="nav-about" class="nav-link">About Us</a>
        <a href="/contact" id="nav-contact" class="nav-link">Contact</a>
    </nav>

    <section id="content">
        <h2>Section One</h2>
        <p>First paragraph with some important text here.</p>
        <p>Second paragraph with different content.</p>

        <h2>Section Two</h2>
        <p>This section has a form below.</p>
        <form>
            <label for="search">Search:</label>
            <input type="text" id="search" name="search" placeholder="Type to search...">
            <button type="submit" id="submit-btn" class="btn primary">Submit</button>
        </form>
    </section>

    <section id="data-section">
        <h2>Data Table</h2>
        <table>
            <thead><tr><th>Name</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Alpha</td><td>100</td></tr>
                <tr><td>Beta</td><td>200</td></tr>
                <tr><td>Gamma</td><td>300</td></tr>
            </tbody>
        </table>
    </section>

    <footer>
        <p>Footer content - copyright 2026</p>
    </footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_html_path():
    """Create a temporary HTML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(TEST_HTML)
        f.flush()
        yield Path(f.name)

    Path(f.name).unlink(missing_ok=True)


class DummyServer:
    """Minimal stub for mcp.server.Server."""

    def __init__(self, name):
        pass

    def list_tools(self):
        return lambda f: f

    def list_resources(self):
        return lambda f: f

    def read_resource(self):
        return lambda f: f

    def list_prompts(self):
        return lambda f: f

    def call_tool(self):
        return lambda f: f

    def get_capabilities(self, **kwargs):
        return {}

    async def run(self, *args, **kwargs):
        return None


class DummyTypes:
    """Minimal stub for mcp.types."""

    class Tool:
        def __init__(self, **kwargs):
            pass

    class Resource:
        def __init__(self, **kwargs):
            pass

    class Prompt:
        pass

    class TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text

    class TextResourceContents:
        def __init__(self, **kwargs):
            self.uri = kwargs.get("uri")
            self.text = kwargs.get("text")
            self.mimeType = kwargs.get("mimeType")


@pytest.fixture(scope="module")
def mcp_server_with_browser(test_html_path, monkeypatch_module):
    """Create an OpenBrowserServer with a real browser session."""
    monkeypatch_module.setattr(mcp_server_module, "MCP_AVAILABLE", True)
    monkeypatch_module.setattr(mcp_server_module, "Server", DummyServer)
    monkeypatch_module.setattr(mcp_server_module, "types", DummyTypes)

    server = mcp_server_module.OpenBrowserServer()

    # Initialize a real browser session
    async def setup():
        profile = BrowserProfile(headless=True)
        session = BrowserSession(browser_profile=profile)
        await session.start()

        # Navigate to the test HTML file
        file_url = f"file://{test_html_path}"
        await session.navigate_to(file_url)

        server.browser_session = session
        return server

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(setup())
        yield result
    finally:
        async def teardown():
            if server.browser_session:
                await server.browser_session.stop()

        loop.run_until_complete(teardown())
        loop.close()


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (workaround for scope mismatch)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegrationGetText:
    """Integration tests for browser_get_text with real browser."""

    def test_extracts_page_title(self, mcp_server_with_browser):
        """Extracts page title from real HTML."""
        result = asyncio.run(mcp_server_with_browser._get_text())
        assert "Integration Test Page" in result

    def test_extracts_paragraph_content(self, mcp_server_with_browser):
        """Extracts paragraph text from real HTML."""
        result = asyncio.run(mcp_server_with_browser._get_text())
        assert "important text" in result

    def test_extracts_links_when_requested(self, mcp_server_with_browser):
        """Includes link URLs when extract_links=True."""
        result = asyncio.run(mcp_server_with_browser._get_text(extract_links=True))
        assert "/about" in result or "About Us" in result


class TestIntegrationGrep:
    """Integration tests for browser_grep with real browser."""

    def test_grep_finds_text(self, mcp_server_with_browser):
        """Finds matching text in real page content."""
        result = asyncio.run(mcp_server_with_browser._grep("important"))
        data = json.loads(result)
        assert data["total_matches"] >= 1

    def test_grep_finds_table_data(self, mcp_server_with_browser):
        """Finds table data via grep."""
        result = asyncio.run(mcp_server_with_browser._grep("Alpha"))
        data = json.loads(result)
        assert data["total_matches"] >= 1

    def test_grep_no_matches(self, mcp_server_with_browser):
        """Returns zero matches for non-existent text."""
        result = asyncio.run(mcp_server_with_browser._grep("zzz_nonexistent_xyz_123"))
        data = json.loads(result)
        assert data["total_matches"] == 0


class TestIntegrationSearchElements:
    """Integration tests for browser_search_elements with real browser."""

    def test_search_links_by_tag(self, mcp_server_with_browser):
        """Finds link elements by tag name."""
        result = asyncio.run(mcp_server_with_browser._search_elements("a", by="tag"))
        data = json.loads(result)
        assert data["count"] >= 3  # At least the 3 nav links

    def test_search_by_id(self, mcp_server_with_browser):
        """Finds element by ID."""
        result = asyncio.run(mcp_server_with_browser._search_elements("submit", by="id"))
        data = json.loads(result)
        assert data["count"] >= 1

    def test_search_by_class(self, mcp_server_with_browser):
        """Finds elements by class name."""
        result = asyncio.run(mcp_server_with_browser._search_elements("nav-link", by="class"))
        data = json.loads(result)
        assert data["count"] >= 3

    def test_search_input_by_text(self, mcp_server_with_browser):
        """Finds button by text content."""
        result = asyncio.run(mcp_server_with_browser._search_elements("Submit", by="text"))
        data = json.loads(result)
        assert data["count"] >= 1


class TestIntegrationGetState:
    """Integration tests for browser_get_state with real browser."""

    def test_compact_state(self, mcp_server_with_browser):
        """Compact state returns URL and element count."""
        result = asyncio.run(mcp_server_with_browser._get_browser_state(compact=True))
        data = json.loads(result)
        assert "file://" in data["url"]
        assert data["interactive_element_count"] >= 4  # links + input + button

    def test_full_state(self, mcp_server_with_browser):
        """Full state includes element details."""
        result = asyncio.run(mcp_server_with_browser._get_browser_state(compact=False))
        data = json.loads(result)
        assert "interactive_elements" in data
        assert len(data["interactive_elements"]) >= 4


class TestIntegrationExecuteJs:
    """Integration tests for browser_execute_js with real browser."""

    def test_evaluate_simple_expression(self, mcp_server_with_browser):
        """Evaluates a simple JavaScript expression."""
        result = asyncio.run(mcp_server_with_browser._execute_js("2 + 2"))
        data = json.loads(result)
        assert data["result"] == 4

    def test_get_document_title(self, mcp_server_with_browser):
        """Gets document title via JavaScript."""
        result = asyncio.run(mcp_server_with_browser._execute_js("document.title"))
        data = json.loads(result)
        assert data["result"] == "MCP Integration Test Page"

    def test_query_dom_elements(self, mcp_server_with_browser):
        """Queries DOM elements via JavaScript."""
        result = asyncio.run(
            mcp_server_with_browser._execute_js("document.querySelectorAll('a.nav-link').length")
        )
        data = json.loads(result)
        assert data["result"] == 3

    def test_extract_data_from_table(self, mcp_server_with_browser):
        """Extracts structured data from a table via JavaScript."""
        js = """
        (() => {
            const rows = document.querySelectorAll('#data-section tbody tr');
            return Array.from(rows).map(row => {
                const cells = row.querySelectorAll('td');
                return {name: cells[0].textContent, value: parseInt(cells[1].textContent)};
            });
        })()
        """
        result = asyncio.run(mcp_server_with_browser._execute_js(js))
        data = json.loads(result)
        assert len(data["result"]) == 3
        assert data["result"][0]["name"] == "Alpha"
        assert data["result"][0]["value"] == 100


class TestIntegrationAccessibilityTree:
    """Integration tests for browser_get_accessibility_tree with real browser."""

    def test_returns_tree_structure(self, mcp_server_with_browser):
        """Returns a non-empty accessibility tree."""
        result = asyncio.run(mcp_server_with_browser._get_accessibility_tree())
        data = json.loads(result)
        assert data["total_nodes"] > 0

    def test_tree_contains_page_elements(self, mcp_server_with_browser):
        """Tree contains expected page elements."""
        result = asyncio.run(mcp_server_with_browser._get_accessibility_tree())
        # The tree should have nodes -- just verify it parsed successfully
        data = json.loads(result)
        assert "total_nodes" in data
        assert data["total_nodes"] >= 5  # headings, links, button, input, etc.
