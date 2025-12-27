"""Entry point for running openbrowser MCP server.

Usage:
    python -m src.openbrowser.mcp
"""

import asyncio

from src.openbrowser.mcp.server import main

if __name__ == "__main__":
    asyncio.run(main())

