"""Entry point for running MCP server as a module.

Usage:
    python -m openbrowser.mcp
"""

import asyncio
import os
import sys

transport = os.environ.get('OPENBROWSER_MCP_TRANSPORT', 'lean').strip().lower()
if transport not in {'lean', 'sdk'}:
	print("ERROR: OPENBROWSER_MCP_TRANSPORT must be 'lean' or 'sdk'", file=sys.stderr)
	sys.exit(2)
if transport == 'lean':
	from openbrowser.mcp.lean import main
else:
	from openbrowser.mcp.server import main

if __name__ == '__main__':
	asyncio.run(main())
