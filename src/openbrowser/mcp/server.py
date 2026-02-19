"""MCP Server for openbrowser -- CodeAgent code execution via Model Context Protocol.

Exposes a single ``execute_code`` tool that runs Python code in a persistent
namespace with browser automation functions (navigate, click, evaluate, etc.).

Usage:
    uvx openbrowser-ai[mcp] --mcp

Or as an MCP server in Claude Desktop or other MCP clients:
    {
        "mcpServers": {
            "openbrowser": {
                "command": "uvx",
                "args": ["openbrowser-ai[mcp]", "--mcp"]
            }
        }
    }
"""

import os
import sys


# Set environment variables BEFORE any openbrowser imports to prevent early logging
os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'critical'
os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'

import asyncio
import io
import logging
import time
import traceback
from pathlib import Path
from typing import Any

# Configure logging for MCP mode - redirect to stderr but preserve critical diagnostics
logging.basicConfig(
	stream=sys.stderr, level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True
)

try:
	import psutil

	PSUTIL_AVAILABLE = True
except ImportError:
	PSUTIL_AVAILABLE = False

# Add src/ to path if running from source (NOT openbrowser/ which would shadow the pip mcp package)
_src_dir = str(Path(__file__).parent.parent.parent)
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

# Import and configure logging to use stderr before other imports
from openbrowser.logging_config import setup_logging


def _configure_mcp_server_logging():
	"""Configure logging for MCP server mode -- redirect all logs to stderr to prevent JSON RPC interference."""
	os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'warning'
	os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'

	setup_logging(stream=sys.stderr, log_level='warning', force_setup=True)

	logging.root.handlers = []
	stderr_handler = logging.StreamHandler(sys.stderr)
	stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	logging.root.addHandler(stderr_handler)
	logging.root.setLevel(logging.CRITICAL)

	for name in list(logging.root.manager.loggerDict.keys()):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = []
		logger_obj.setLevel(logging.CRITICAL)
		logger_obj.addHandler(stderr_handler)
		logger_obj.propagate = False


_configure_mcp_server_logging()

# Suppress all logging for MCP mode
logging.disable(logging.CRITICAL)

# Import openbrowser modules
from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.code_use.namespace import create_namespace
from openbrowser.config import get_default_profile, load_openbrowser_config
from openbrowser.tools.service import CodeAgentTools

logger = logging.getLogger(__name__)


def _ensure_all_loggers_use_stderr():
	"""Ensure ALL loggers only output to stderr, not stdout."""
	stderr_handler = None
	for handler in logging.root.handlers:
		if hasattr(handler, 'stream') and handler.stream == sys.stderr:  # type: ignore
			stderr_handler = handler
			break

	if not stderr_handler:
		stderr_handler = logging.StreamHandler(sys.stderr)
		stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

	logging.root.handlers = [stderr_handler]
	logging.root.setLevel(logging.CRITICAL)

	for name in list(logging.root.manager.loggerDict.keys()):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = [stderr_handler]
		logger_obj.setLevel(logging.CRITICAL)
		logger_obj.propagate = False


_ensure_all_loggers_use_stderr()


# Try to import MCP SDK
try:
	import mcp.server.stdio
	import mcp.types as types
	from mcp.server import NotificationOptions, Server
	from mcp.server.models import InitializationOptions

	MCP_AVAILABLE = True

	mcp_logger = logging.getLogger('mcp')
	mcp_logger.handlers = []
	mcp_logger.addHandler(logging.root.handlers[0] if logging.root.handlers else logging.StreamHandler(sys.stderr))
	mcp_logger.setLevel(logging.ERROR)
	mcp_logger.propagate = False
except ImportError:
	MCP_AVAILABLE = False
	logger.error('MCP SDK not installed. Install with: pip install mcp')
	sys.exit(1)

try:
	from openbrowser.telemetry import MCPServerTelemetryEvent, ProductTelemetry

	TELEMETRY_AVAILABLE = True
except ImportError:
	TELEMETRY_AVAILABLE = False

from openbrowser.utils import get_openbrowser_version


def get_parent_process_cmdline() -> str | None:
	"""Get the command line of all parent processes up the chain."""
	if not PSUTIL_AVAILABLE:
		return None

	try:
		cmdlines = []
		current_process = psutil.Process()
		parent = current_process.parent()

		while parent:
			try:
				cmdline = parent.cmdline()
				if cmdline:
					cmdlines.append(' '.join(cmdline))
			except (psutil.AccessDenied, psutil.NoSuchProcess):
				pass

			try:
				parent = parent.parent()
			except (psutil.AccessDenied, psutil.NoSuchProcess):
				break

		return ';'.join(cmdlines) if cmdlines else None
	except Exception:
		return None


_EXECUTE_CODE_DESCRIPTION = """Execute Python code in a persistent namespace with browser automation functions. All functions are async -- use `await`. Use print() to return output. Variables persist between calls.

## Navigation

- `await navigate(url: str, new_tab: bool = False)` -- Navigate to a URL. Set new_tab=True to open in a new tab.
- `await go_back()` -- Go back to the previous page in browser history.
- `await wait(seconds: int = 3)` -- Wait for specified seconds (max 30). Use after actions that trigger page loads.

## Element Interaction

- `await click(index: int)` -- Click an element by its index from browser state. Index must be >= 1. Works for buttons, links, checkboxes, radio buttons. Does NOT work for <select> elements (use select_dropdown instead).
- `await input_text(index: int, text: str, clear: bool = True)` -- Type text into an input field. clear=True (default) clears the field first; clear=False appends.
- `await scroll(down: bool = True, pages: float = 1.0, index: int | None = None)` -- Scroll the page. down=True scrolls down, down=False scrolls up. pages=0.5 for half page, 1 for full page, 10 for top/bottom. Pass index to scroll within a specific container element.
- `await send_keys(keys: str)` -- Send keyboard keys or shortcuts. Examples: "Escape", "Enter", "PageDown", "Control+o", "Control+a", "ArrowDown".
- `await upload_file(index: int, path: str)` -- Upload a file to a file input element. index is the file input element index, path is the local file path.

## Dropdowns

- `await select_dropdown(index: int, text: str)` -- Select an option in a <select> dropdown by its visible text. text must be the exact option text.
- `await dropdown_options(index: int)` -- Get all available options for a <select> dropdown. Returns the options as text. Call this first to see what options are available.

## Tab Management

- `await switch(tab_id: str)` -- Switch to a different browser tab. tab_id is a 4-character ID (get IDs from browser state tabs list).
- `await close(tab_id: str)` -- Close a browser tab by its 4-character tab_id.

## JavaScript Execution

- `await evaluate(code: str)` -- Execute JavaScript in the browser page context and return the result as a Python object. Auto-wraps code in an IIFE if not already wrapped. Returns Python dicts/lists/primitives directly. Raises EvaluateError on JS errors.
  Example: `data = await evaluate('document.title')` returns the page title string.
  Example: `items = await evaluate('Array.from(document.querySelectorAll(".item")).map(e => e.textContent)')` returns a Python list of strings.

## CSS Selectors

- `await get_selector_from_index(index: int)` -- Get the CSS selector for an element by its interactive index. Useful for building JS queries targeting specific elements. Returns a CSS selector string.

## Task Completion

- `await done(text: str, success: bool = True)` -- Signal that the task is complete. text is the final output/result. success=True if the task completed successfully. Call this only when the task is truly finished.

## Browser State

- `browser` -- The BrowserSession object. Use `state = await browser.get_browser_state_summary()` to get current page state including:
  - `state.url` -- current URL
  - `state.title` -- page title
  - `state.tabs` -- list of open tabs (each has .target_id, .url, .title). For switch()/close(), use the LAST 4 CHARS of target_id as tab_id.
  - `state.dom_state.selector_map` -- dict of {index: element} for all interactive elements
  - Each element has: `.tag_name`, `.attributes` (dict), `.get_all_children_text(max_depth=N)` (text content)

## File System

- `file_system` -- FileSystem object for file operations.

## Libraries (pre-imported)

- `json` -- JSON encoding/decoding
- `asyncio` -- async utilities
- `Path` -- pathlib.Path for file paths
- `csv` -- CSV reading/writing
- `re` -- regular expressions
- `datetime` -- date/time operations
- `requests` -- HTTP requests (synchronous)

## Optional Libraries (available if installed)

- `numpy` / `np` -- numerical computing
- `pandas` / `pd` -- data analysis and DataFrames
- `matplotlib` / `plt` -- plotting and charts
- `BeautifulSoup` / `bs4` -- HTML parsing
- `PdfReader` / `pypdf` -- PDF reading
- `tabulate` -- table formatting

## Typical Workflow

1. Navigate: `await navigate('https://example.com')`
2. Get state: `state = await browser.get_browser_state_summary()`
3. Inspect elements: iterate `state.dom_state.selector_map` to find element indices
4. Interact: `await click(index)`, `await input_text(index, 'text')`, etc.
5. Extract data: `data = await evaluate('JS expression')` -- returns Python objects
6. Process with Python: use json, pandas, re, etc.
7. Print results: `print(output)` -- this is what gets returned to the client
"""


class OpenBrowserServer:
	"""MCP Server exposing CodeAgent code execution environment.

	Provides a single ``execute_code`` tool that runs Python code in a
	persistent namespace populated with browser automation functions from
	``create_namespace()`` (navigate, click, evaluate, etc.).
	"""

	def __init__(self, session_timeout_minutes: int = 10):
		_ensure_all_loggers_use_stderr()

		self.server = Server('openbrowser')
		self.config = load_openbrowser_config()
		self.browser_session: BrowserSession | None = None
		self._telemetry = ProductTelemetry() if TELEMETRY_AVAILABLE else None
		self._start_time = time.time()

		# CodeAgent namespace -- persistent across execute_code calls
		self._namespace: dict[str, Any] | None = None
		self._tools: CodeAgentTools | None = None

		# Session management
		self.session_timeout_minutes = session_timeout_minutes
		self._last_activity = time.time()
		self._cleanup_task: Any = None

		self._setup_handlers()

	def _setup_handlers(self):
		"""Setup MCP server handlers."""

		@self.server.list_tools()
		async def handle_list_tools() -> list[types.Tool]:
			"""List the single execute_code tool."""
			return [
				types.Tool(
					name='execute_code',
					description=_EXECUTE_CODE_DESCRIPTION,
					inputSchema={
						'type': 'object',
						'properties': {
							'code': {
								'type': 'string',
								'description': 'Python code to execute. All browser functions are async (use await).',
							},
						},
						'required': ['code'],
					},
					annotations=types.ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
				),
			]

		@self.server.list_resources()
		async def handle_list_resources() -> list[types.Resource]:
			return []

		@self.server.list_resource_templates()
		async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
			return []

		@self.server.list_prompts()
		async def handle_list_prompts() -> list[types.Prompt]:
			return []

		@self.server.call_tool()
		async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
			"""Handle tool execution."""
			start_time = time.time()
			error_msg = None
			try:
				if name != 'execute_code':
					return [types.TextContent(type='text', text=f'Unknown tool: {name}')]

				code = (arguments or {}).get('code', '')
				if not code.strip():
					return [types.TextContent(type='text', text='Error: No code provided')]

				result = await self._execute_code(code)
				return [types.TextContent(type='text', text=result)]
			except Exception as e:
				error_msg = str(e)
				logger.error(f'Tool execution failed: {e}', exc_info=True)
				return [types.TextContent(type='text', text=f'Error: {str(e)}')]
			finally:
				if self._telemetry and TELEMETRY_AVAILABLE:
					duration = time.time() - start_time
					self._telemetry.capture(
						MCPServerTelemetryEvent(
							version=get_openbrowser_version(),
							action='tool_call',
							tool_name=name,
							duration_seconds=duration,
							error_message=error_msg,
						)
					)

	async def _ensure_namespace(self):
		"""Lazily initialize browser session, tools, and namespace on first use."""
		if self._namespace is not None:
			return

		_ensure_all_loggers_use_stderr()

		# Initialize browser session
		profile_config = get_default_profile(self.config)
		profile_data = {
			'downloads_path': str(Path.home() / 'Downloads' / 'openbrowser-mcp'),
			'wait_between_actions': 0.5,
			'keep_alive': True,
			'user_data_dir': '~/.config/openbrowser/profiles/default',
			'device_scale_factor': 1.0,
			'disable_security': False,
			'headless': False,
			**profile_config,
		}
		profile = BrowserProfile(**profile_data)
		session = BrowserSession(browser_profile=profile)

		try:
			await session.start()
		except Exception as e:
			logger.error(f'Failed to start browser session: {e}')
			try:
				from openbrowser.browser.events import BrowserStopEvent
				event = session.event_bus.dispatch(BrowserStopEvent())
				await event
			except Exception:
				pass
			raise

		self.browser_session = session

		# Create CodeAgent tools and namespace
		self._tools = CodeAgentTools()
		self._namespace = create_namespace(
			browser_session=self.browser_session,
			tools=self._tools,
		)

	async def _execute_code(self, code: str) -> str:
		"""Execute Python code in the persistent namespace.

		The code is wrapped in an async function so ``await`` expressions work.
		stdout is captured and returned as the result text.  Variables defined
		in user code are persisted back to the namespace after execution.
		"""
		await self._ensure_namespace()
		assert self._namespace is not None

		self._last_activity = time.time()

		# Capture stdout
		stdout_capture = io.StringIO()

		# Wrap code in an async function so await works.
		# We inject a __locals_capture__ dict and copy locals into it at
		# the end so we can persist user-defined variables back to the
		# namespace after the function returns.
		indented_code = '\n'.join(f'    {line}' for line in code.split('\n'))
		wrapped = (
			f"async def __mcp_exec__(__ns__):\n"
			f"{indented_code}\n"
			f"    __ns__.update({{k: v for k, v in locals().items() if not k.startswith('__')}})\n"
		)

		try:
			# Compile and exec the async function definition
			compiled = compile(wrapped, '<execute_code>', 'exec')
			exec(compiled, self._namespace)

			# Call the async function with stdout capture, passing namespace
			# so locals get persisted
			old_stdout = sys.stdout
			sys.stdout = stdout_capture
			try:
				result = await self._namespace['__mcp_exec__'](self._namespace)
			finally:
				sys.stdout = old_stdout

			output = stdout_capture.getvalue()

			# If the function returned a value and nothing was printed, show the return value
			if result is not None and not output.strip():
				output = repr(result)

			# If nothing was printed and no return value, confirm execution
			if not output.strip():
				output = '(executed successfully, no output)'

			return output

		except Exception as e:
			# Restore stdout in case of exception during capture
			sys.stdout = sys.__stdout__

			captured_output = stdout_capture.getvalue()

			# Format the error with traceback
			tb = traceback.format_exc()
			# Strip the wrapper function frames from traceback for cleaner output
			error_output = f'Error: {type(e).__name__}: {e}\n\nTraceback:\n{tb}'

			if captured_output.strip():
				error_output = f'Output before error:\n{captured_output}\n\n{error_output}'

			return error_output

		finally:
			# Clean up the temporary function from namespace
			self._namespace.pop('__mcp_exec__', None)

	async def _cleanup_expired_session(self) -> None:
		"""Close browser session if idle beyond timeout."""
		if not self.browser_session:
			return

		current_time = time.time()
		timeout_seconds = self.session_timeout_minutes * 60

		if current_time - self._last_activity > timeout_seconds:
			logger.info('Auto-closing idle browser session')
			try:
				from openbrowser.browser.events import BrowserStopEvent
				event = self.browser_session.event_bus.dispatch(BrowserStopEvent())
				await event
			except Exception as e:
				logger.error(f'Error closing idle session: {e}')
			finally:
				self.browser_session = None
				self._namespace = None

	async def _start_cleanup_task(self) -> None:
		"""Start the background cleanup task."""

		async def cleanup_loop():
			while True:
				try:
					await self._cleanup_expired_session()
					await asyncio.sleep(120)
				except Exception as e:
					logger.error(f'Error in cleanup task: {e}')
					await asyncio.sleep(120)

		self._cleanup_task = asyncio.create_task(cleanup_loop())

	async def run(self):
		"""Run the MCP server."""
		await self._start_cleanup_task()

		async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
			await self.server.run(
				read_stream,
				write_stream,
				InitializationOptions(
					server_name='openbrowser',
					server_version='0.1.0',
					capabilities=self.server.get_capabilities(
						notification_options=NotificationOptions(),
						experimental_capabilities={},
					),
				),
			)


async def main(session_timeout_minutes: int = 10):
	if not MCP_AVAILABLE:
		print('MCP SDK is required. Install with: pip install mcp', file=sys.stderr)
		sys.exit(1)

	server = OpenBrowserServer(session_timeout_minutes=session_timeout_minutes)
	if server._telemetry and TELEMETRY_AVAILABLE:
		server._telemetry.capture(
			MCPServerTelemetryEvent(
				version=get_openbrowser_version(),
				action='start',
				parent_process_cmdline=get_parent_process_cmdline(),
			)
		)

	try:
		await server.run()
	finally:
		if server._telemetry and TELEMETRY_AVAILABLE:
			duration = time.time() - server._start_time
			server._telemetry.capture(
				MCPServerTelemetryEvent(
					version=get_openbrowser_version(),
					action='stop',
					duration_seconds=duration,
					parent_process_cmdline=get_parent_process_cmdline(),
				)
			)
			server._telemetry.flush()


if __name__ == '__main__':
	asyncio.run(main())
