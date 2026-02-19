"""Integration tests for the MCP server with a real browser session.

These tests spin up an actual browser session and verify the execute_code
tool can navigate, extract text, run JavaScript, and interact with elements
on a known HTML page.

Requirements:
    - Chrome/Chromium must be installed
    - Tests are marked with @pytest.mark.integration and are skipped by default
    - Run with: pytest tests/test_mcp_integration.py -m integration
"""

import asyncio
import json
import logging
import tempfile
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from conftest import DummyServer, DummyTypes

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
def test_html_server():
    """Serve test HTML via a local HTTP server.

    Uses HTTP instead of file:// because the browser security policy
    blocks URLs without hostnames (file:// has no hostname).
    """
    tmp_dir = tempfile.mkdtemp()
    html_path = Path(tmp_dir) / "test.html"
    html_path.write_text(TEST_HTML)

    handler = partial(SimpleHTTPRequestHandler, directory=tmp_dir)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}/test.html"

    server.shutdown()
    html_path.unlink(missing_ok=True)
    Path(tmp_dir).rmdir()


@pytest.fixture(scope="module")
def mcp_server_with_browser(test_html_server, monkeypatch_module):
    """Create an OpenBrowserServer with a real browser session and namespace.

    Yields (server, loop) so tests can run async methods on the same
    event loop that owns the browser session.
    """
    monkeypatch_module.setattr(mcp_server_module, "MCP_AVAILABLE", True)
    monkeypatch_module.setattr(mcp_server_module, "Server", DummyServer)
    monkeypatch_module.setattr(mcp_server_module, "types", DummyTypes)

    server = mcp_server_module.OpenBrowserServer()

    async def setup():
        profile = BrowserProfile(headless=True)
        session = BrowserSession(browser_profile=profile)
        await session.start()

        server.browser_session = session

        # Initialize the CodeAgent namespace with the real browser
        from openbrowser.code_use.namespace import create_namespace
        from openbrowser.tools.service import CodeAgentTools

        tools = CodeAgentTools()
        server._namespace = create_namespace(
            browser_session=session,
            tools=tools,
        )

        # Navigate to test page
        await session.navigate_to(test_html_server)

        return server

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(setup())
        yield server, loop
    finally:
        async def teardown():
            if server.browser_session:
                await server.browser_session.stop()

        loop.run_until_complete(teardown())
        loop.close()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegrationNavigate:
    """Integration tests for navigation via execute_code."""

    def test_get_page_title(self, mcp_server_with_browser):
        """Gets page title via browser state."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "print(state.title)"
        ))
        assert "MCP Integration Test Page" in result

    def test_get_page_url(self, mcp_server_with_browser):
        """Gets page URL via browser state."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "print(state.url)"
        ))
        assert "127.0.0.1" in result
        assert "test.html" in result


class TestIntegrationJavaScript:
    """Integration tests for JavaScript evaluation via execute_code."""

    def test_evaluate_simple_expression(self, mcp_server_with_browser):
        """Evaluates a simple JavaScript expression."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "result = await evaluate('2 + 2')\n"
            "print(result)"
        ))
        assert "4" in result

    def test_get_document_title(self, mcp_server_with_browser):
        """Gets document title via JavaScript."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "title = await evaluate('document.title')\n"
            "print(title)"
        ))
        assert "MCP Integration Test Page" in result

    def test_query_dom_elements(self, mcp_server_with_browser):
        """Queries DOM elements via JavaScript."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "count = await evaluate('document.querySelectorAll(\"a.nav-link\").length')\n"
            "print(count)"
        ))
        assert "3" in result

    def test_extract_data_from_table(self, mcp_server_with_browser):
        """Extracts structured data from a table via JavaScript."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "data = await evaluate(\"\"\"\n"
            "Array.from(document.querySelectorAll('#data-section tbody tr')).map(row => {\n"
            "    const cells = row.querySelectorAll('td');\n"
            "    return {name: cells[0].textContent, value: parseInt(cells[1].textContent)};\n"
            "})\n"
            "\"\"\")\n"
            "print(json.dumps(data))"
        ))
        data = json.loads(result)
        assert len(data) == 3
        assert data[0]["name"] == "Alpha"
        assert data[0]["value"] == 100


class TestIntegrationBrowserState:
    """Integration tests for browser state inspection via execute_code."""

    def test_get_interactive_elements(self, mcp_server_with_browser):
        """Gets interactive elements from browser state."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "count = len(state.dom_state.selector_map)\n"
            "print(f'Elements: {count}')"
        ))
        # Should have at least links + input + button
        assert "Elements:" in result

    def test_find_element_by_text(self, mcp_server_with_browser):
        """Finds an element by its text content."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "found = []\n"
            "for idx, el in state.dom_state.selector_map.items():\n"
            "    text = el.get_all_children_text(max_depth=1)\n"
            "    if 'Submit' in text:\n"
            "        found.append((idx, el.tag_name, text))\n"
            "print(found)"
        ))
        assert "Submit" in result

    def test_get_tabs(self, mcp_server_with_browser):
        """Gets tab information from browser state."""
        server, loop = mcp_server_with_browser
        result = loop.run_until_complete(server._execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "for tab in state.tabs:\n"
            "    print(f'Tab: {tab.url}')"
        ))
        assert "Tab:" in result


class TestIntegrationVariablePersistence:
    """Integration tests for variable persistence between execute_code calls."""

    def test_variables_persist(self, mcp_server_with_browser):
        """Variables set in one call are available in the next."""
        server, loop = mcp_server_with_browser
        loop.run_until_complete(server._execute_code(
            "integration_test_var = 'persisted_value'"
        ))
        result = loop.run_until_complete(server._execute_code(
            "print(integration_test_var)"
        ))
        assert "persisted_value" in result

    def test_extracted_data_persists(self, mcp_server_with_browser):
        """Data extracted from the page persists for later processing."""
        server, loop = mcp_server_with_browser
        loop.run_until_complete(server._execute_code(
            "page_title = await evaluate('document.title')"
        ))
        result = loop.run_until_complete(server._execute_code(
            "print(f'Saved title: {page_title}')"
        ))
        assert "Saved title: MCP Integration Test Page" in result
