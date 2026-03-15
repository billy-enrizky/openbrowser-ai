"""MCP Server for openbrowser -- CodeAgent code execution via Model Context Protocol.

Exposes a single ``execute_code`` tool that runs Python code in a persistent
namespace with browser automation functions (navigate, click, evaluate, etc.).

Usage:
    uvx openbrowser-ai --mcp

Or as an MCP server in Claude Desktop or other MCP clients:
    {
        "mcpServers": {
            "openbrowser": {
                "command": "uvx",
                "args": ["openbrowser-ai", "--mcp"]
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
import logging
import time
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
from openbrowser.code_use.executor import DEFAULT_MAX_OUTPUT_CHARS, CodeExecutor

try:
	from openbrowser.filesystem.file_system import FileSystem

	FILESYSTEM_AVAILABLE = True
except ModuleNotFoundError:
	FILESYSTEM_AVAILABLE = False
except Exception:
	FILESYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)

_MCP_WORKSPACE_DIR = Path.home() / 'Downloads' / 'openbrowser-mcp' / 'workspace'


def _create_mcp_file_system() -> Any:
	"""Create a FileSystem instance for MCP mode, or None if unavailable."""
	if not FILESYSTEM_AVAILABLE:
		return None
	return FileSystem(base_dir=str(_MCP_WORKSPACE_DIR), create_default_files=False)


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


from openbrowser.code_use.descriptions import (
	EXECUTE_CODE_DESCRIPTION as _EXECUTE_CODE_DESCRIPTION,
	EXECUTE_CODE_DESCRIPTION_COMPACT as _EXECUTE_CODE_DESCRIPTION_COMPACT,
)


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
		try:
			max_output = int(os.environ.get('OPENBROWSER_MAX_OUTPUT', '0'))
		except (ValueError, TypeError):
			max_output = 0
		self._executor = CodeExecutor(max_output_chars=max_output if max_output > 0 else DEFAULT_MAX_OUTPUT_CHARS)
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
			description = _EXECUTE_CODE_DESCRIPTION
			if os.environ.get('OPENBROWSER_COMPACT_DESCRIPTION', '').lower() in ('1', 'true', 'yes'):
				description = _EXECUTE_CODE_DESCRIPTION_COMPACT
			return [
				types.Tool(
					name='execute_code',
					description=description,
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
					annotations=types.ToolAnnotations(
					readOnlyHint=False,
					destructiveHint=True,
					openWorldHint=True,
					idempotentHint=False,
				),
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
				logger.error('Tool execution failed: %s', e, exc_info=True)
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

	def _build_browser_profile(self):
		"""Build a BrowserProfile from config with MCP defaults."""
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
		return BrowserProfile(**profile_data)

	async def _ensure_namespace(self):
		"""Lazily initialize browser session, tools, and namespace on first use."""
		if self._namespace is not None:
			return

		_ensure_all_loggers_use_stderr()

		# Initialize browser session
		profile = self._build_browser_profile()
		session = BrowserSession(browser_profile=profile)

		try:
			await session.start()
		except Exception as e:
			logger.error('Failed to start browser session: %s', e)
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
			file_system=_create_mcp_file_system(),
		)
		self._executor.set_namespace(self._namespace)

	async def _is_cdp_alive(self) -> bool:
		"""Check if the browser session's CDP WebSocket is still connected."""
		if not self.browser_session:
			return False
		root = getattr(self.browser_session, '_cdp_client_root', None)
		if root is None:
			return False
		try:
			await root.send.Browser.getVersion()
			return True
		except Exception:
			return False

	async def _recover_browser_session(self) -> None:
		"""Kill the dead browser session and create a fresh one.

		Called when we detect the CDP WebSocket has disconnected (e.g. Chrome
		crashed or a navigation broke the connection).  Re-creates the
		BrowserSession and rebuilds the namespace so the next execute_code
		call works transparently.
		"""
		logger.info('CDP connection lost -- recovering browser session')

		# 1. Tear down the old session
		if self.browser_session:
			try:
				await self.browser_session.kill()
			except Exception:
				# Session may already be half-dead; best-effort cleanup
				try:
					await self.browser_session.reset()
				except Exception:
					pass

		# 2. Kill any stale Chrome holding the profile lock
		from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

		user_data_dir = '~/.config/openbrowser/profiles/default'
		if self.browser_session and self.browser_session.browser_profile.user_data_dir:
			user_data_dir = self.browser_session.browser_profile.user_data_dir
		await LocalBrowserWatchdog._kill_stale_chrome_for_profile(user_data_dir)

		# 3. Create a brand-new session
		profile = self._build_browser_profile()
		session = BrowserSession(browser_profile=profile)
		await session.start()
		self.browser_session = session

		# 4. Rebuild namespace with the new session (preserving user variables)
		old_ns = self._namespace or {}

		self._tools = CodeAgentTools()
		self._namespace = create_namespace(
			browser_session=self.browser_session,
			tools=self._tools,
			file_system=_create_mcp_file_system(),
		)
		# Copy user-defined variables from old namespace
		for key, val in old_ns.items():
			if not key.startswith('__') and key not in self._namespace:
				self._namespace[key] = val

		self._executor.set_namespace(self._namespace)

		logger.info('Browser session recovered successfully')

	def _is_connection_error(self, exc_or_text) -> bool:
		"""Return True if the error indicates a dead CDP connection."""
		keywords = ('connectionclosederror', 'no close frame', 'websocket', 'connection closed')
		if isinstance(exc_or_text, str):
			text = exc_or_text.lower()
		else:
			text = f'{type(exc_or_text).__name__}: {exc_or_text}'.lower()
		return any(kw in text for kw in keywords)

	async def _execute_code(self, code: str) -> str:
		"""Execute Python code in the persistent namespace."""
		await self._ensure_namespace()
		assert self._namespace is not None

		# Keep executor in sync (namespace may be set externally, e.g. tests)
		if not self._executor.initialized:
			self._executor.set_namespace(self._namespace)

		self._last_activity = time.time()

		# Pre-flight: check if CDP is still alive and recover if needed
		# Only attempt recovery when we already had a browser session (i.e. it died).
		# If browser_session is None, _ensure_namespace hasn't launched one yet
		# or the code doesn't need a browser -- skip recovery to avoid hanging.
		if self.browser_session and not await self._is_cdp_alive():
			try:
				await self._recover_browser_session()
			except Exception as recovery_err:
				logger.error('Pre-flight CDP recovery failed: %s', recovery_err)

		# Execute via shared CodeExecutor
		result = await self._executor.execute(code)

		# If error looks like CDP connection issue, recover and retry once.
		# Check result.error (raw, not truncated) to avoid missing errors
		# when large stdout causes the error message to be truncated away.
		error_text = result.error or result.output
		if not result.success and self._is_connection_error(error_text):
			logger.info('CDP connection error during execution, recovering')
			try:
				await self._recover_browser_session()
				result = await self._executor.execute(code)
			except Exception as recovery_err:
				logger.error('CDP recovery failed: %s', recovery_err)

		return result.output

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
				logger.error('Error closing idle session: %s', e)
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
					logger.error('Error in cleanup task: %s', e)
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
					server_version=get_openbrowser_version(),
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
