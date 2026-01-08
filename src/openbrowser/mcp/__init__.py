"""MCP (Model Context Protocol) support for openbrowser.

This module provides integration with MCP servers and clients for browser automation.
"""

from openbrowser.mcp.client import MCPClient
from openbrowser.mcp.controller import MCPToolWrapper

__all__ = ['MCPClient', 'MCPToolWrapper', 'OpenBrowserServer']  # type: ignore


def __getattr__(name):
	"""Lazy import to avoid importing server module when only client is needed."""
	if name == 'OpenBrowserServer':
		from openbrowser.mcp.server import OpenBrowserServer

		return OpenBrowserServer
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
