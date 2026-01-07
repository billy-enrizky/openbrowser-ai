"""Tests for the MCP (Model Context Protocol) server module.

This module provides test coverage for the OpenBrowser MCP server,
which exposes browser automation capabilities through the Model Context
Protocol. It validates:

    - Graceful handling when MCP SDK is not available
    - Server initialization with proper error handling
    - Tool execution for unknown tools with informative messages
    - Session listing when no active sessions exist

The MCP server enables integration with MCP-compatible clients like
Claude Desktop, allowing AI assistants to control browser sessions.
"""

import asyncio
import sys

import pytest

from openbrowser import mcp as mcp_module


def test_main_exits_when_mcp_missing(monkeypatch, capsys):
    """Verify main() exits with error when MCP SDK is not available."""
    monkeypatch.setattr(mcp_module.server, "MCP_AVAILABLE", False)

    with pytest.raises(SystemExit) as exc:
        asyncio.run(mcp_module.server.main())

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "MCP SDK is required" in captured.err


def test_openbrowser_server_init_raises_when_mcp_missing(monkeypatch):
    """Verify OpenBrowserServer raises ImportError when MCP SDK is missing."""
    monkeypatch.setattr(mcp_module.server, "MCP_AVAILABLE", False)

    with pytest.raises(ImportError):
        mcp_module.server.OpenBrowserServer()


def test_execute_tool_unknown_returns_message(monkeypatch):
    """Verify _execute_tool returns informative message for unknown tools."""

    # Provide fake MCP runtime objects so we can construct the server without the real SDK
    monkeypatch.setattr(mcp_module.server, "MCP_AVAILABLE", True)

    class DummyServer:
        def __init__(self, name):
            pass

        def list_tools(self):
            def deco(f):
                return f

            return deco

        def list_resources(self):
            def deco(f):
                return f

            return deco

        def list_prompts(self):
            def deco(f):
                return f

            return deco

        def call_tool(self):
            def deco(f):
                return f

            return deco

        def get_capabilities(self, **kwargs):
            return {}

        async def run(self, *args, **kwargs):
            return None

    class DummyTypes:
        class Tool:
            def __init__(self, **kwargs):
                pass

        class Resource:
            pass

        class Prompt:
            pass

        class TextContent:
            def __init__(self, type: str, text: str):
                self.type = type
                self.text = text

    monkeypatch.setattr(mcp_module.server, "Server", DummyServer)
    monkeypatch.setattr(mcp_module.server, "types", DummyTypes)

    server = mcp_module.server.OpenBrowserServer()

    result = asyncio.run(server._execute_tool("unknown_tool", {}))

    assert "Unknown tool: unknown_tool" in result


def test_list_sessions_when_none_returns_string(monkeypatch):
    """Verify _list_sessions returns readable message when no sessions exist."""
    monkeypatch.setattr(mcp_module.server, "MCP_AVAILABLE", True)

    class DummyServer:
        def __init__(self, name):
            pass

        def list_tools(self):
            def deco(f):
                return f

            return deco

        def list_resources(self):
            def deco(f):
                return f

            return deco

        def list_prompts(self):
            def deco(f):
                return f

            return deco

        def call_tool(self):
            def deco(f):
                return f

            return deco

        def get_capabilities(self, **kwargs):
            return {}

        async def run(self, *args, **kwargs):
            return None

    class DummyTypes:
        class Tool:
            def __init__(self, **kwargs):
                pass

        class Resource:
            pass

        class Prompt:
            pass

        class TextContent:
            def __init__(self, type: str, text: str):
                self.type = type
                self.text = text

    monkeypatch.setattr(mcp_module.server, "Server", DummyServer)
    monkeypatch.setattr(mcp_module.server, "types", DummyTypes)

    server = mcp_module.server.OpenBrowserServer()

    result = asyncio.run(server._list_sessions())

    assert "No active browser sessions" in result or result == "No active browser sessions"
