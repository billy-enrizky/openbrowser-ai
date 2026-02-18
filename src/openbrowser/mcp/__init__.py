"""MCP (Model Context Protocol) support for openbrowser.

This module provides integration with MCP servers and clients for browser automation.

All imports are lazy to avoid pulling in heavy dependencies when only the MCP server
is needed (e.g., ``python -m openbrowser.mcp`` or ``uvx openbrowser-ai[mcp] --mcp``).
"""

__all__ = ['MCPClient', 'MCPToolWrapper', 'OpenBrowserServer']


def __getattr__(name: str):
	"""Lazy import to avoid pulling in heavy openbrowser dependencies for MCP server mode."""
	if name == 'MCPClient':
		from openbrowser.mcp.client import MCPClient

		return MCPClient
	if name == 'MCPToolWrapper':
		from openbrowser.mcp.controller import MCPToolWrapper

		return MCPToolWrapper
	if name == 'OpenBrowserServer':
		from openbrowser.mcp.server import OpenBrowserServer

		return OpenBrowserServer
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
