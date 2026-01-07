"""Entry point for running openbrowser-ai MCP server.

Usage:
    python -m openbrowser.mcp
"""

import asyncio

from openbrowser.mcp.server import main

if __name__ == "__main__":
    asyncio.run(main())

