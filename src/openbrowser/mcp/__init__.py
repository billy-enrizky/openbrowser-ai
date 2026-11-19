"""MCP (Model Context Protocol) support for openbrowser.

Provides a CodeAgent-based MCP server. Modern 2026-07-28 clients use
``create_context``, ``execute_code``, ``get_status``, ``reset_browser``, and
``close_context`` with an explicit context handle. Older clients receive the
original three-tool catalog and a legacy no-handle context.

All imports are lazy to avoid pulling in heavy dependencies when only the
MCP server is needed (e.g., ``python -m openbrowser.mcp`` or
``uvx openbrowser-ai --mcp``).
"""

__all__ = ['OpenBrowserServer']


def __getattr__(name: str):
	"""Lazy import to avoid pulling in heavy openbrowser dependencies for MCP server mode."""
	if name == 'OpenBrowserServer':
		from openbrowser.mcp.server import OpenBrowserServer

		return OpenBrowserServer
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
