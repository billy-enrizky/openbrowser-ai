"""MCP (Model Context Protocol) server for openbrowser-ai.

This module provides MCP server and client integration for browser automation.
"""

from openbrowser.mcp.server import OpenBrowserServer, main

__all__ = [
    "OpenBrowserServer",
    "main",
]

