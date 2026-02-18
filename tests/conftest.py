"""Pytest configuration and fixtures for openbrowser test suite.

This module provides shared configuration and fixtures used across the entire
test suite. It sets up the Python path to allow importing from openbrowser
and defines common fixtures for browser sessions, mock objects, and test data.

Configuration:
    - Adds src/ directory to Python path for test imports
    - Configures pytest-asyncio for async test support
    - Sets up logging for test visibility

Shared Stubs:
    DummyServer and DummyTypes provide minimal MCP SDK stubs so that
    OpenBrowserServer can initialise without the real mcp package installed.
    These are defined once here to prevent stub drift between unit and
    integration tests.

Path Setup:
    The src directory is added to sys.path to enable imports like:
    ``from openbrowser.browser.session import BrowserSession``
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add the src directory to the path so tests can import using openbrowser
# This allows consistent import paths across all test modules
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path.parent))

from openbrowser.mcp import server as mcp_server_module  # noqa: E402


# ---------------------------------------------------------------------------
# Shared dummy MCP SDK stubs (single source of truth)
# ---------------------------------------------------------------------------


class DummyServer:
    """Minimal stub for mcp.server.Server so OpenBrowserServer can initialise."""

    def __init__(self, name):
        pass

    def list_tools(self):
        return lambda f: f

    def list_resources(self):
        return lambda f: f

    def read_resource(self):
        return lambda f: f

    def subscribe_resource(self):
        return lambda f: f

    def unsubscribe_resource(self):
        return lambda f: f

    def list_resource_templates(self):
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

    class ResourceTemplate:
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


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def make_mock_element(
    tag_name="a",
    text="Click here",
    attributes=None,
    node_id=1,
):
    """Create a mock EnhancedDOMTreeNode-like object."""
    elem = MagicMock()
    elem.tag_name = tag_name
    elem.node_name = tag_name
    elem.attributes = attributes or {}
    elem.get_all_children_text = MagicMock(return_value=text)
    elem.node_id = node_id
    return elem


def make_mock_browser_state(url="https://example.com", title="Example", tabs=None, selector_map=None):
    """Create a mock BrowserStateSummary-like object."""
    state = MagicMock()
    state.url = url
    state.title = title

    tab = MagicMock()
    tab.url = url
    tab.title = title
    state.tabs = tabs or [tab]

    dom_state = MagicMock()
    dom_state.selector_map = selector_map or {}
    state.dom_state = dom_state

    return state


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mcp_server(monkeypatch):
    """Create an OpenBrowserServer with dummy MCP SDK stubs."""
    monkeypatch.setattr(mcp_server_module, "MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_server_module, "Server", DummyServer)
    monkeypatch.setattr(mcp_server_module, "types", DummyTypes)
    return mcp_server_module.OpenBrowserServer()


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (workaround for scope mismatch)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()
