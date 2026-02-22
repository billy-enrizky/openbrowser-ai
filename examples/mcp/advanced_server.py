"""
Advanced example of building an AI assistant that uses openbrowser MCP server.

This example shows how to build a more sophisticated MCP client that:
- Connects to the openbrowser MCP server via stdio
- Orchestrates complex multi-step workflows using execute_code
- Leverages persistent namespace for stateful automation
- Handles errors gracefully

Prerequisites:
1. Install required packages:
   pip install mcp

2. Run this example (it starts the server automatically):
   python advanced_server.py

This demonstrates real-world usage patterns for the MCP protocol
with openbrowser's single execute_code tool.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("advanced_server")


@dataclass
class TaskResult:
	"""Result of executing a task."""

	success: bool
	data: Any
	error: str | None = None
	timestamp: datetime = field(default_factory=datetime.now)


class BrowserAssistant:
	"""An assistant that uses openbrowser MCP server for complex browser workflows."""

	def __init__(self):
		self._session: ClientSession | None = None
		self.history: list[TaskResult] = []

	async def connect(self):
		"""Connect to the openbrowser MCP server."""
		logger.info("Connecting to openbrowser MCP server...")
		server_params = StdioServerParameters(
			command="uvx", args=["openbrowser-ai[mcp]", "--mcp"], env={}
		)
		self._read_ctx = stdio_client(server_params)
		read, write = await self._read_ctx.__aenter__()
		self._session_ctx = ClientSession(read, write)
		self._session = await self._session_ctx.__aenter__()
		await self._session.initialize()

		tools_result = await self._session.list_tools()
		logger.info(
			"Connected. Tools: %s",
			[t.name for t in tools_result.tools],
		)

	async def disconnect(self):
		"""Disconnect from the MCP server."""
		if self._session:
			try:
				await self._session_ctx.__aexit__(None, None, None)
				await self._read_ctx.__aexit__(None, None, None)
				logger.info("Disconnected from openbrowser server")
			except Exception:
				logger.exception("Error during disconnect")
			self._session = None

	async def execute(self, code: str) -> TaskResult:
		"""Execute Python code via the execute_code tool and return a TaskResult."""
		if not self._session:
			return TaskResult(False, None, "Not connected")
		try:
			result = await self._session.call_tool(
				"execute_code", arguments={"code": code}
			)
			text_parts = [
				c.text for c in result.content if isinstance(c, TextContent)
			]
			data = "\n".join(text_parts) if text_parts else str(result.content)
			task_result = TaskResult(True, data)
		except Exception as exc:
			task_result = TaskResult(False, None, str(exc))
		self.history.append(task_result)
		return task_result

	async def search_and_extract(self, query: str) -> TaskResult:
		"""Search DuckDuckGo and extract top results."""
		logger.info("Searching for: %s", query)

		# Navigate to search results
		result = await self.execute(
			f'await navigate("https://duckduckgo.com/?q={query}")\n'
			"await wait(2)"
		)
		if not result.success:
			return result

		# Extract search results using JS
		result = await self.execute(
			"results = await evaluate(\"\"\"\n"
			"  Array.from(document.querySelectorAll('.result__title, .result__a, [data-testid=\"result-title-a\"]'))\n"
			"    .slice(0, 5)\n"
			"    .map(el => ({text: el.textContent.trim(), href: el.href}))\n"
			"\"\"\")\n"
			"for i, r in enumerate(results, 1):\n"
			"    print(f'{i}. {r[\"text\"]}')\n"
			"    print(f'   {r[\"href\"]}')"
		)
		logger.info("Search results extracted: %s", "OK" if result.success else "FAILED")
		return result

	async def monitor_page(self, url: str, duration: int = 10, interval: int = 3) -> TaskResult:
		"""Monitor a webpage for changes over time."""
		logger.info("Monitoring %s for %ds...", url, duration)

		# Navigate and set up monitoring state
		result = await self.execute(
			f'await navigate("{url}")\n'
			"await wait(2)\n"
			"monitor_snapshots = []"
		)
		if not result.success:
			return result

		# Collect snapshots
		n_snapshots = duration // interval
		for i in range(n_snapshots):
			result = await self.execute(
				"import datetime as _dt\n"
				"state = await browser.get_browser_state_summary()\n"
				"monitor_snapshots.append({\n"
				"    'timestamp': _dt.datetime.now().isoformat(),\n"
				"    'title': state.title,\n"
				"    'n_elements': len(state.dom_state.selector_map),\n"
				"})\n"
				f'print(f"Snapshot {len(monitor_snapshots)}: {{state.title}} ({{len(state.dom_state.selector_map)}} elements)")'
			)
			if i < n_snapshots - 1:
				await asyncio.sleep(interval)

		# Summary
		result = await self.execute(
			"print(f'Collected {len(monitor_snapshots)} snapshots')\n"
			"for s in monitor_snapshots:\n"
			"    print(f'  {s[\"timestamp\"]}: {s[\"title\"]} ({s[\"n_elements\"]} elements)')"
		)
		return result

	async def fill_form(self, url: str, form_data: dict[str, str]) -> TaskResult:
		"""Navigate to a form, discover fields, and fill them."""
		logger.info("Form filling workflow for %s", url)

		# Navigate
		result = await self.execute(
			f'await navigate("{url}")\n'
			"await wait(2)"
		)
		if not result.success:
			return result

		# Discover form fields
		result = await self.execute(
			"state = await browser.get_browser_state_summary()\n"
			"form_fields = []\n"
			"for idx, elem in state.dom_state.selector_map.items():\n"
			"    if elem.tag_name in ('input', 'textarea', 'select'):\n"
			"        attrs = elem.attributes\n"
			"        name = attrs.get('name', attrs.get('id', attrs.get('placeholder', '')))\n"
			"        form_fields.append({'index': idx, 'tag': elem.tag_name, 'name': name, 'type': attrs.get('type', '')})\n"
			"        print(f'  [{idx}] <{elem.tag_name}> name={name} type={attrs.get(\"type\", \"\")}')\n"
			"print(f'Found {len(form_fields)} form fields')"
		)
		if not result.success:
			return result

		# Fill each field by matching name
		for field_name, field_value in form_data.items():
			result = await self.execute(
				"matched = False\n"
				"for f in form_fields:\n"
				f"    if '{field_name}'.lower() in f['name'].lower():\n"
				f"        await input_text(f['index'], '{field_value}')\n"
				f"        print(f'Filled {{f[\"name\"]}} with {field_value}')\n"
				"        matched = True\n"
				"        break\n"
				"if not matched:\n"
				f"    print(f'No match for field: {field_name}')"
			)

		return TaskResult(True, {"url": url, "fields": list(form_data.keys())})

	async def multi_tab_comparison(self, sites: list[tuple[str, str]]) -> TaskResult:
		"""Open multiple sites in tabs and compare their titles."""
		logger.info("Multi-tab comparison: %d sites", len(sites))

		for i, (name, url) in enumerate(sites):
			new_tab = "True" if i > 0 else "False"
			result = await self.execute(
				f'await navigate("{url}", new_tab={new_tab})\n'
				"await wait(2)\n"
				"state = await browser.get_browser_state_summary()\n"
				f'print(f"{name}: {{state.title}}")'
			)
			if not result.success:
				logger.warning("Failed to open %s: %s", name, result.error)

		# List all tabs
		result = await self.execute(
			"state = await browser.get_browser_state_summary()\n"
			"print(f'Open tabs: {len(state.tabs)}')\n"
			"for tab in state.tabs:\n"
			"    print(f'  {tab.title} - {tab.url}')"
		)
		return result


async def main():
	"""Main demonstration of advanced MCP client usage."""
	logger.info("OpenBrowser MCP Client - Advanced Example")
	logger.info("=" * 50)

	assistant = BrowserAssistant()

	try:
		await assistant.connect()

		# Demo 1: Search and extract
		logger.info("--- Demo 1: Web Search and Extraction ---")
		await assistant.search_and_extract("MCP protocol browser automation")

		# Demo 2: Multi-tab comparison
		logger.info("--- Demo 2: Multi-tab Comparison ---")
		await assistant.multi_tab_comparison([
			("BBC News", "https://bbc.com/news"),
			("CNN", "https://cnn.com"),
			("Reuters", "https://reuters.com"),
		])

		# Demo 3: Form filling
		logger.info("--- Demo 3: Automated Form Filling ---")
		await assistant.fill_form(
			"https://httpbin.org/forms/post",
			{
				"custname": "AI Assistant",
				"custtel": "555-0123",
				"custemail": "ai@example.com",
				"comments": "Testing MCP browser automation",
			},
		)

		# Demo 4: Page monitoring
		logger.info("--- Demo 4: Page Monitoring ---")
		await assistant.monitor_page("https://time.is/", duration=9, interval=3)

		# Session summary
		success_count = sum(1 for r in assistant.history if r.success)
		total_count = len(assistant.history)
		logger.info("=" * 50)
		logger.info("Session Summary")
		logger.info("  Total operations: %d", total_count)
		logger.info("  Successful: %d", success_count)
		logger.info("  Failed: %d", total_count - success_count)
		if total_count > 0:
			logger.info("  Success rate: %.1f%%", success_count / total_count * 100)

	except Exception:
		logger.exception("Fatal error")

	finally:
		logger.info("Cleaning up...")
		await assistant.disconnect()
		logger.info("Demo complete")


if __name__ == "__main__":
	asyncio.run(main())
