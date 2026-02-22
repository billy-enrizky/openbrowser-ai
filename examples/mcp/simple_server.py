"""
Simple example of connecting to openbrowser MCP server as a client.

This example demonstrates how to use the MCP client library to connect to
a running openbrowser MCP server and call its single execute_code tool.

Prerequisites:
1. Install required packages:
   pip install mcp

2. Run this client example (it starts the server automatically via stdio):
   python simple_server.py

This shows the actual MCP protocol flow between a client and the openbrowser server.
"""

import asyncio
import json
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("simple_server")


async def run_simple_browser_automation():
	"""Connect to openbrowser MCP server and perform basic browser automation."""

	# Create connection parameters for the openbrowser MCP server
	server_params = StdioServerParameters(
		command="uvx", args=["openbrowser-ai[mcp]", "--mcp"], env={}
	)

	async with stdio_client(server_params) as (read, write):
		async with ClientSession(read, write) as session:
			# Initialize the connection
			await session.initialize()

			logger.info("Connected to openbrowser MCP server")

			# List available tools -- should be exactly 1: execute_code
			tools_result = await session.list_tools()
			tools = tools_result.tools
			logger.info("Available tools: %d", len(tools))
			for tool in tools:
				logger.info("  - %s: %s", tool.name, tool.description[:80])

			# Example 1: Navigate to a website and get the title
			logger.info("Navigating to example.com...")
			result = await session.call_tool(
				"execute_code",
				arguments={
					"code": (
						'await navigate("https://example.com")\n'
						"state = await browser.get_browser_state_summary()\n"
						'print(f"Title: {state.title}")\n'
						'print(f"URL: {state.url}")'
					)
				},
			)
			content = result.content[0]
			if isinstance(content, TextContent):
				logger.info("Result:\n%s", content.text)

			# Example 2: Variable persistence across calls
			logger.info("Testing variable persistence...")
			await session.call_tool(
				"execute_code",
				arguments={"code": 'saved_title = "example.com visited"'},
			)
			result = await session.call_tool(
				"execute_code",
				arguments={"code": "print(saved_title)"},
			)
			content = result.content[0]
			if isinstance(content, TextContent):
				logger.info("Persisted variable: %s", content.text)

			# Example 3: Use evaluate() for JS execution
			logger.info("Running JavaScript in page context...")
			result = await session.call_tool(
				"execute_code",
				arguments={
					"code": (
						"links = await evaluate(\n"
						'    "Array.from(document.querySelectorAll(\'a\')).map(a => ({text: a.textContent, href: a.href}))"\n'
						")\n"
						"for link in links:\n"
						'    print(f"  {link[\'text\']}: {link[\'href\']}")'
					)
				},
			)
			content = result.content[0]
			if isinstance(content, TextContent):
				logger.info("Links found:\n%s", content.text)

			# Example 4: Get interactive elements and click one
			logger.info("Getting interactive elements...")
			result = await session.call_tool(
				"execute_code",
				arguments={
					"code": (
						"state = await browser.get_browser_state_summary()\n"
						"for idx, elem in state.dom_state.selector_map.items():\n"
						'    print(f"  [{idx}] <{elem.tag_name}> {elem.get_all_children_text(max_depth=1)[:60]}")'
					)
				},
			)
			content = result.content[0]
			if isinstance(content, TextContent):
				logger.info("Elements:\n%s", content.text)

			# Example 5: Open a new tab and list tabs
			logger.info("Opening python.org in a new tab...")
			result = await session.call_tool(
				"execute_code",
				arguments={
					"code": (
						'await navigate("https://python.org", new_tab=True)\n'
						"state = await browser.get_browser_state_summary()\n"
						'print(f"Current page: {state.title}")\n'
						"for tab in state.tabs:\n"
						'    print(f"  Tab: {tab.title} - {tab.url}")'
					)
				},
			)
			content = result.content[0]
			if isinstance(content, TextContent):
				logger.info("Tabs:\n%s", content.text)

			logger.info("Simple browser automation demo complete")


async def main():
	"""Main entry point."""
	logger.info("OpenBrowser MCP Client - Simple Example")
	logger.info("=" * 50)

	try:
		await run_simple_browser_automation()
	except Exception:
		logger.exception("Error running demo")
		logger.info("Make sure openbrowser-ai[mcp] is installed: pip install openbrowser-ai[mcp]")


if __name__ == "__main__":
	asyncio.run(main())
