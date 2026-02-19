"""MCP (Model Context Protocol) support for openbrowser.

Provides a CodeAgent-based MCP server with a single ``execute_code`` tool
that runs Python code in a persistent namespace with browser automation
functions.

All imports are lazy to avoid pulling in heavy dependencies when only the
MCP server is needed (e.g., ``python -m openbrowser.mcp`` or
``uvx openbrowser-ai[mcp] --mcp``).
"""

__all__ = ['OpenBrowserServer']


def __getattr__(name: str):
	"""Lazy import to avoid pulling in heavy openbrowser dependencies for MCP server mode."""
	if name == 'OpenBrowserServer':
		from openbrowser.mcp.server import OpenBrowserServer

		return OpenBrowserServer
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
