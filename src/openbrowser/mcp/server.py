"""MCP Server for openbrowser - exposes browser automation capabilities via Model Context Protocol.

This server provides tools for:
- Running autonomous browser tasks with an AI agent
- Direct browser control (navigation, clicking, typing, etc.)
- Content extraction from web pages
- File system operations

Usage:
    uvx openbrowser --mcp

Or as an MCP server in Claude Desktop or other MCP clients:
    {
        "mcpServers": {
            "openbrowser": {
                "command": "uvx",
                "args": ["openbrowser-ai[cli]", "--mcp"],
                "env": {
                    "OPENAI_API_KEY": "sk-proj-1234567890",
                }
            }
        }
    }
"""

import os
import sys

from openbrowser.llm import ChatAWSBedrock

# Set environment variables BEFORE any openbrowser imports to prevent early logging
os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'critical'
os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'

import asyncio
import json
import logging
import re
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

# Add openbrowser to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and configure logging to use stderr before other imports
from openbrowser.logging_config import setup_logging


def _configure_mcp_server_logging():
	"""Configure logging for MCP server mode - redirect all logs to stderr to prevent JSON RPC interference."""
	# Set environment to suppress openbrowser logging during server mode
	os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'warning'
	os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'  # Prevent automatic logging setup

	# Configure logging to stderr for MCP mode - preserve warnings and above for troubleshooting
	setup_logging(stream=sys.stderr, log_level='warning', force_setup=True)

	# Also configure the root logger and all existing loggers to use stderr
	logging.root.handlers = []
	stderr_handler = logging.StreamHandler(sys.stderr)
	stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	logging.root.addHandler(stderr_handler)
	logging.root.setLevel(logging.CRITICAL)

	# Configure all existing loggers to use stderr and CRITICAL level
	for name in list(logging.root.manager.loggerDict.keys()):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = []
		logger_obj.setLevel(logging.CRITICAL)
		logger_obj.addHandler(stderr_handler)
		logger_obj.propagate = False


# Configure MCP server logging before any openbrowser imports to capture early log lines
_configure_mcp_server_logging()

# Additional suppression - disable all logging completely for MCP mode
logging.disable(logging.CRITICAL)

# Import openbrowser modules
from openbrowser import ActionModel, Agent
from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.browser.events import ScrollToTextEvent
from openbrowser.config import get_default_llm, get_default_profile, load_openbrowser_config
from openbrowser.dom.markdown_extractor import extract_clean_markdown
from openbrowser.dom.service import DomService
from openbrowser.filesystem.file_system import FileSystem
from openbrowser.llm.google.chat import ChatGoogle
from openbrowser.llm.openai.chat import ChatOpenAI
from openbrowser.tools.service import Tools

logger = logging.getLogger(__name__)


def _ensure_all_loggers_use_stderr():
	"""Ensure ALL loggers only output to stderr, not stdout."""
	# Get the stderr handler
	stderr_handler = None
	for handler in logging.root.handlers:
		if hasattr(handler, 'stream') and handler.stream == sys.stderr:  # type: ignore
			stderr_handler = handler
			break

	if not stderr_handler:
		stderr_handler = logging.StreamHandler(sys.stderr)
		stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

	# Configure root logger
	logging.root.handlers = [stderr_handler]
	logging.root.setLevel(logging.CRITICAL)

	# Configure all existing loggers
	for name in list(logging.root.manager.loggerDict.keys()):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = [stderr_handler]
		logger_obj.setLevel(logging.CRITICAL)
		logger_obj.propagate = False


# Ensure stderr logging after all imports
_ensure_all_loggers_use_stderr()


# Try to import MCP SDK
try:
	import mcp.server.stdio
	import mcp.types as types
	from mcp.server import NotificationOptions, Server
	from mcp.server.lowlevel.helper_types import ReadResourceContents
	from mcp.server.models import InitializationOptions

	MCP_AVAILABLE = True

	# Configure MCP SDK logging to stderr as well
	mcp_logger = logging.getLogger('mcp')
	mcp_logger.handlers = []
	mcp_logger.addHandler(logging.root.handlers[0] if logging.root.handlers else logging.StreamHandler(sys.stderr))
	mcp_logger.setLevel(logging.ERROR)
	mcp_logger.propagate = False
except ImportError:
	MCP_AVAILABLE = False
	logger.error('MCP SDK not installed. Install with: pip install mcp')
	sys.exit(1)

from openbrowser.telemetry import MCPServerTelemetryEvent, ProductTelemetry
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
				# Skip processes we can't access (like system processes)
				pass

			try:
				parent = parent.parent()
			except (psutil.AccessDenied, psutil.NoSuchProcess):
				# Can't go further up the chain
				break

		return ';'.join(cmdlines) if cmdlines else None
	except Exception:
		# If we can't get parent process info, just return None
		return None


class OpenBrowserServer:
	"""MCP Server for openbrowser capabilities."""

	def __init__(self, session_timeout_minutes: int = 10):
		# Ensure all logging goes to stderr (in case new loggers were created)
		_ensure_all_loggers_use_stderr()

		self.server = Server('openbrowser')
		self.config = load_openbrowser_config()
		self.agent: Agent | None = None
		self.browser_session: BrowserSession | None = None
		self.tools: Tools | None = None
		self.llm: ChatOpenAI | ChatGoogle | None = None
		self.file_system: FileSystem | None = None
		self._telemetry = ProductTelemetry()
		self._start_time = time.time()

		# Session management
		self.active_sessions: dict[str, dict[str, Any]] = {}  # session_id -> session info
		self.session_timeout_minutes = session_timeout_minutes
		self._cleanup_task: Any = None

		# Setup handlers
		self._setup_handlers()

	def _setup_handlers(self):
		"""Setup MCP server handlers."""

		@self.server.list_tools()
		async def handle_list_tools() -> list[types.Tool]:
			"""List all available openbrowser tools."""
			return [
				# Agent tools
				# Direct browser control tools
				types.Tool(
					name='browser_navigate',
					description='Navigate to a URL in the browser',
					inputSchema={
						'type': 'object',
						'properties': {
							'url': {'type': 'string', 'description': 'The URL to navigate to'},
							'new_tab': {'type': 'boolean', 'description': 'Whether to open in a new tab', 'default': False},
						},
						'required': ['url'],
					},
				),
				types.Tool(
					name='browser_click',
					description='Click an element on the page by its index',
					inputSchema={
						'type': 'object',
						'properties': {
							'index': {
								'type': 'integer',
								'description': 'The index of the link or element to click (from browser_get_state)',
							},
							'new_tab': {
								'type': 'boolean',
								'description': 'Whether to open any resulting navigation in a new tab',
								'default': False,
							},
						},
						'required': ['index'],
					},
				),
				types.Tool(
					name='browser_type',
					description='Type text into an input field',
					inputSchema={
						'type': 'object',
						'properties': {
							'index': {
								'type': 'integer',
								'description': 'The index of the input element (from browser_get_state)',
							},
							'text': {'type': 'string', 'description': 'The text to type'},
						},
						'required': ['index', 'text'],
					},
				),
				types.Tool(
					name='browser_get_state',
					description='Get the current page state. Use compact=true (default) for a lightweight summary with URL, title, and element count. Use compact=false for the full list of interactive elements.',
					inputSchema={
						'type': 'object',
						'properties': {
							'compact': {
								'type': 'boolean',
								'description': 'If true, returns only URL, title, tab count, and interactive element count. If false, returns full element details.',
								'default': True,
							}
						},
					},
				),
				types.Tool(
					name='browser_scroll',
					description='Scroll the page',
					inputSchema={
						'type': 'object',
						'properties': {
							'direction': {
								'type': 'string',
								'enum': ['up', 'down'],
								'description': 'Direction to scroll',
								'default': 'down',
							}
						},
					},
				),
				types.Tool(
					name='browser_go_back',
					description='Go back to the previous page',
					inputSchema={'type': 'object', 'properties': {}},
				),
				# Tab management
				types.Tool(
					name='browser_list_tabs', description='List all open tabs', inputSchema={'type': 'object', 'properties': {}}
				),
				types.Tool(
					name='browser_switch_tab',
					description='Switch to a different tab',
					inputSchema={
						'type': 'object',
						'properties': {'tab_id': {'type': 'string', 'description': '4 Character Tab ID of the tab to switch to'}},
						'required': ['tab_id'],
					},
				),
				types.Tool(
					name='browser_close_tab',
					description='Close a tab',
					inputSchema={
						'type': 'object',
						'properties': {'tab_id': {'type': 'string', 'description': '4 Character Tab ID of the tab to close'}},
						'required': ['tab_id'],
					},
				),
				# types.Tool(
				# 	name="browser_close",
				# 	description="Close the browser session",
				# 	inputSchema={
				# 		"type": "object",
				# 		"properties": {}
				# 	}
				# ),
				# Browser session management tools
				types.Tool(
					name='browser_list_sessions',
					description='List all active browser sessions with their details and last activity time',
					inputSchema={'type': 'object', 'properties': {}},
				),
				types.Tool(
					name='browser_close_session',
					description='Close a specific browser session by its ID',
					inputSchema={
						'type': 'object',
						'properties': {
							'session_id': {
								'type': 'string',
								'description': 'The browser session ID to close (get from browser_list_sessions)',
							}
						},
						'required': ['session_id'],
					},
				),
				types.Tool(
					name='browser_close_all',
					description='Close all active browser sessions and clean up resources',
					inputSchema={'type': 'object', 'properties': {}},
				),
				# Text-first content tools (efficient alternatives to screenshots)
				types.Tool(
					name='browser_get_text',
					description='Get the current page content as clean markdown text. Use this instead of screenshots for reading page content efficiently.',
					inputSchema={
						'type': 'object',
						'properties': {
							'extract_links': {
								'type': 'boolean',
								'description': 'Whether to include href URLs in the output',
								'default': False,
							},
						},
					},
				),
				types.Tool(
					name='browser_grep',
					description='Search page text content using a regex or string pattern. Returns matching lines with context, like grep. Use this to find specific content on a page without reading the entire page.',
					inputSchema={
						'type': 'object',
						'properties': {
							'pattern': {
								'type': 'string',
								'description': 'Regex or string pattern to search for in page content',
							},
							'context_lines': {
								'type': 'integer',
								'description': 'Number of lines before and after each match to include',
								'default': 2,
							},
							'max_matches': {
								'type': 'integer',
								'description': 'Maximum number of matches to return',
								'default': 20,
							},
							'case_insensitive': {
								'type': 'boolean',
								'description': 'Whether to ignore case when matching',
								'default': True,
							},
						},
						'required': ['pattern'],
					},
				),
				types.Tool(
					name='browser_search_elements',
					description='Search interactive DOM elements by text content, tag name, id, class, or attribute value. Returns element indices that can be used with browser_click and browser_type.',
					inputSchema={
						'type': 'object',
						'properties': {
							'query': {
								'type': 'string',
								'description': 'Text or pattern to search for in elements',
							},
							'by': {
								'type': 'string',
								'enum': ['text', 'tag', 'id', 'class', 'attribute'],
								'description': 'What property to search by',
								'default': 'text',
							},
							'max_results': {
								'type': 'integer',
								'description': 'Maximum number of results to return',
								'default': 20,
							},
						},
						'required': ['query'],
					},
				),
				types.Tool(
					name='browser_find_and_scroll',
					description='Find text on the page and scroll to it. Use this to locate and navigate to specific content.',
					inputSchema={
						'type': 'object',
						'properties': {
							'text': {
								'type': 'string',
								'description': 'The text to find and scroll to on the page',
							},
						},
						'required': ['text'],
					},
				),
				# Advanced inspection tools
				types.Tool(
					name='browser_get_accessibility_tree',
					description='Get the accessibility tree (a11y tree) of the current page. Returns structured data including roles, names, and properties of all accessible elements. Useful for understanding page structure, testing accessibility, and finding elements by their ARIA roles.',
					inputSchema={
						'type': 'object',
						'properties': {
							'max_depth': {
								'type': 'integer',
								'description': 'Maximum depth of the tree to return. Use -1 for unlimited depth.',
								'default': -1,
							},
							'include_ignored': {
								'type': 'boolean',
								'description': 'Whether to include nodes marked as ignored in the accessibility tree.',
								'default': False,
							},
						},
					},
				),
				types.Tool(
					name='browser_execute_js',
					description='Execute JavaScript code on the current page and return the result. The expression is evaluated in the page context. Use for advanced page interactions, reading page state, or extracting data not available through other tools.',
					inputSchema={
						'type': 'object',
						'properties': {
							'expression': {
								'type': 'string',
								'description': 'JavaScript expression or IIFE to evaluate in the page context. For multi-statement code, wrap in an IIFE: (()=>{ ... return result; })()',
							},
						},
						'required': ['expression'],
					},
				),
			]

		@self.server.list_resources()
		async def handle_list_resources() -> list[types.Resource]:
			"""List available resources for the current browser state."""
			resources = []
			if self.browser_session:
				try:
					url = await self.browser_session.get_current_page_url()
				except Exception:
					url = 'unknown'
				resources.append(
					types.Resource(
						uri='browser://current-page/content',
						name='Page Content',
						description=f'Clean markdown text of the current page ({url})',
						mimeType='text/markdown',
					)
				)
				resources.append(
					types.Resource(
						uri='browser://current-page/state',
						name='Page State',
						description=f'Interactive elements and metadata of the current page ({url})',
						mimeType='application/json',
					)
				)
				resources.append(
					types.Resource(
						uri='browser://current-page/accessibility',
						name='Accessibility Tree',
						description=f'Accessibility tree of the current page ({url})',
						mimeType='application/json',
					)
				)
			return resources

		@self.server.read_resource()
		async def handle_read_resource(uri: str) -> list[ReadResourceContents]:
			"""Read a browser resource by URI."""
			uri_str = str(uri)

			if uri_str == 'browser://current-page/content':
				if not self.browser_session:
					return [ReadResourceContents(content='No browser session active', mime_type='text/plain')]
				content = await self._get_text(extract_links=True)
				return [ReadResourceContents(content=content, mime_type='text/markdown')]

			elif uri_str == 'browser://current-page/state':
				if not self.browser_session:
					return [ReadResourceContents(content='{"error": "No browser session active"}', mime_type='application/json')]
				state_json = await self._get_browser_state(compact=False)
				return [ReadResourceContents(content=state_json, mime_type='application/json')]

			elif uri_str == 'browser://current-page/accessibility':
				if not self.browser_session:
					return [ReadResourceContents(content='{"error": "No browser session active"}', mime_type='application/json')]
				a11y_json = await self._get_accessibility_tree()
				return [ReadResourceContents(content=a11y_json, mime_type='application/json')]

			return [ReadResourceContents(content=f'Unknown resource: {uri_str}', mime_type='text/plain')]

		@self.server.list_prompts()
		async def handle_list_prompts() -> list[types.Prompt]:
			"""List available prompts (none for openbrowser)."""
			return []

		@self.server.call_tool()
		async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
			"""Handle tool execution."""
			start_time = time.time()
			error_msg = None
			try:
				result = await self._execute_tool(name, arguments or {})
				return [types.TextContent(type='text', text=result)]
			except Exception as e:
				error_msg = str(e)
				logger.error(f'Tool execution failed: {e}', exc_info=True)
				return [types.TextContent(type='text', text=f'Error: {str(e)}')]
			finally:
				# Capture telemetry for tool calls
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

	async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
		"""Execute an openbrowser tool."""

		# Browser session management tools (don't require active session)
		if tool_name == 'browser_list_sessions':
			return await self._list_sessions()

		elif tool_name == 'browser_close_session':
			return await self._close_session(arguments['session_id'])

		elif tool_name == 'browser_close_all':
			return await self._close_all_sessions()

		# Direct browser control tools (require active session)
		elif tool_name.startswith('browser_'):
			# Ensure browser session exists
			if not self.browser_session:
				await self._init_browser_session()

			if tool_name == 'browser_navigate':
				return await self._navigate(arguments['url'], arguments.get('new_tab', False))

			elif tool_name == 'browser_click':
				return await self._click(arguments['index'], arguments.get('new_tab', False))

			elif tool_name == 'browser_type':
				return await self._type_text(arguments['index'], arguments['text'])

			elif tool_name == 'browser_get_state':
				return await self._get_browser_state(arguments.get('compact', True))

			elif tool_name == 'browser_scroll':
				return await self._scroll(arguments.get('direction', 'down'))

			elif tool_name == 'browser_go_back':
				return await self._go_back()

			elif tool_name == 'browser_close':
				return await self._close_browser()

			elif tool_name == 'browser_list_tabs':
				return await self._list_tabs()

			elif tool_name == 'browser_switch_tab':
				return await self._switch_tab(arguments['tab_id'])

			elif tool_name == 'browser_close_tab':
				return await self._close_tab(arguments['tab_id'])

			# Text-first content tools
			elif tool_name == 'browser_get_text':
				return await self._get_text(arguments.get('extract_links', False))

			elif tool_name == 'browser_grep':
				return await self._grep(
					pattern=arguments['pattern'],
					context_lines=arguments.get('context_lines', 2),
					max_matches=arguments.get('max_matches', 20),
					case_insensitive=arguments.get('case_insensitive', True),
				)

			elif tool_name == 'browser_search_elements':
				return await self._search_elements(
					query=arguments['query'],
					by=arguments.get('by', 'text'),
					max_results=arguments.get('max_results', 20),
				)

			elif tool_name == 'browser_find_and_scroll':
				return await self._find_and_scroll(arguments['text'])

			# Advanced inspection tools
			elif tool_name == 'browser_get_accessibility_tree':
				return await self._get_accessibility_tree(
					max_depth=arguments.get('max_depth', -1),
					include_ignored=arguments.get('include_ignored', False),
				)

			elif tool_name == 'browser_execute_js':
				return await self._execute_js(arguments['expression'])

		return f'Unknown tool: {tool_name}'

	async def _init_browser_session(self, allowed_domains: list[str] | None = None, **kwargs):
		"""Initialize browser session using config"""
		if self.browser_session:
			return

		# Ensure all logging goes to stderr before browser initialization
		_ensure_all_loggers_use_stderr()

		logger.debug('Initializing browser session...')

		# Get profile config
		profile_config = get_default_profile(self.config)

		# Merge profile config with defaults and overrides
		profile_data = {
			'downloads_path': str(Path.home() / 'Downloads' / 'openbrowser-mcp'),
			'wait_between_actions': 0.5,
			'keep_alive': True,
			'user_data_dir': '~/.config/openbrowser/profiles/default',
			'device_scale_factor': 1.0,
			'disable_security': False,
			'headless': False,
			**profile_config,  # Config values override defaults
		}

		# Tool parameter overrides (highest priority)
		if allowed_domains is not None:
			profile_data['allowed_domains'] = allowed_domains

		# Merge any additional kwargs that are valid BrowserProfile fields
		for key, value in kwargs.items():
			profile_data[key] = value

		# Create browser profile
		profile = BrowserProfile(**profile_data)

		# Create browser session
		self.browser_session = BrowserSession(browser_profile=profile)
		await self.browser_session.start()

		# Track the session for management
		self._track_session(self.browser_session)

		# Create tools for direct actions
		self.tools = Tools()

		# Initialize LLM from config
		llm_config = get_default_llm(self.config)
		model_provider = llm_config.get('model_provider') or os.getenv('MODEL_PROVIDER', '')
		if model_provider.lower() == 'google':
			google_api_key = llm_config.get('api_key') or os.getenv('GOOGLE_API_KEY')
			if google_api_key:
				self.llm = ChatGoogle(
					model=llm_config.get('model', 'gemini-2.5-flash'),
					api_key=google_api_key,
					temperature=llm_config.get('temperature', 0.7),
				)
		elif api_key := llm_config.get('api_key') or os.getenv('OPENAI_API_KEY'):
			self.llm = ChatOpenAI(
				model=llm_config.get('model', 'gpt-4o-mini'),
				api_key=api_key,
				temperature=llm_config.get('temperature', 0.7),
			)

		# Initialize FileSystem for extraction actions
		file_system_path = profile_config.get('file_system_path', '~/.openbrowser-mcp')
		self.file_system = FileSystem(base_dir=Path(file_system_path).expanduser())

		logger.debug('Browser session initialized')

	async def _navigate(self, url: str, new_tab: bool = False) -> str:
		"""Navigate to a URL."""
		if not self.browser_session:
			return 'Error: No browser session active'

		# Update session activity
		self._update_session_activity(self.browser_session.id)

		from openbrowser.browser.events import NavigateToUrlEvent

		if new_tab:
			event = self.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=True))
			await event
			return f'Opened new tab with URL: {url}'
		else:
			event = self.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url))
			await event
			return f'Navigated to: {url}'

	async def _click(self, index: int, new_tab: bool = False) -> str:
		"""Click an element by index."""
		if not self.browser_session:
			return 'Error: No browser session active'

		# Update session activity
		self._update_session_activity(self.browser_session.id)

		# Get the element
		element = await self.browser_session.get_dom_element_by_index(index)
		if not element:
			return f'Element with index {index} not found'

		if new_tab:
			# For links, extract href and open in new tab
			href = element.attributes.get('href')
			if href:
				# Convert relative href to absolute URL
				state = await self.browser_session.get_browser_state_summary()
				current_url = state.url
				if href.startswith('/'):
					# Relative URL - construct full URL
					from urllib.parse import urlparse

					parsed = urlparse(current_url)
					full_url = f'{parsed.scheme}://{parsed.netloc}{href}'
				else:
					full_url = href

				# Open link in new tab
				from openbrowser.browser.events import NavigateToUrlEvent

				event = self.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=full_url, new_tab=True))
				await event
				return f'Clicked element {index} and opened in new tab {full_url[:20]}...'
			else:
				# For non-link elements, just do a normal click
				# Opening in new tab without href is not reliably supported
				from openbrowser.browser.events import ClickElementEvent

				event = self.browser_session.event_bus.dispatch(ClickElementEvent(node=element))
				await event
				return f'Clicked element {index} (new tab not supported for non-link elements)'
		else:
			# Normal click
			from openbrowser.browser.events import ClickElementEvent

			event = self.browser_session.event_bus.dispatch(ClickElementEvent(node=element))
			await event
			return f'Clicked element {index}'

	async def _type_text(self, index: int, text: str) -> str:
		"""Type text into an element."""
		if not self.browser_session:
			return 'Error: No browser session active'

		element = await self.browser_session.get_dom_element_by_index(index)
		if not element:
			return f'Element with index {index} not found'

		from openbrowser.browser.events import TypeTextEvent

		# Conservative heuristic to detect potentially sensitive data
		# Only flag very obvious patterns to minimize false positives
		is_potentially_sensitive = len(text) >= 6 and (
			# Email pattern: contains @ and a domain-like suffix
			('@' in text and '.' in text.split('@')[-1] if '@' in text else False)
			# Mixed alphanumeric with reasonable complexity (likely API keys/tokens)
			or (
				len(text) >= 16
				and any(char.isdigit() for char in text)
				and any(char.isalpha() for char in text)
				and any(char in '.-_' for char in text)
			)
		)

		# Use generic key names to avoid information leakage about detection patterns
		sensitive_key_name = None
		if is_potentially_sensitive:
			if '@' in text and '.' in text.split('@')[-1]:
				sensitive_key_name = 'email'
			else:
				sensitive_key_name = 'credential'

		event = self.browser_session.event_bus.dispatch(
			TypeTextEvent(node=element, text=text, is_sensitive=is_potentially_sensitive, sensitive_key_name=sensitive_key_name)
		)
		await event

		if is_potentially_sensitive:
			if sensitive_key_name:
				return f'Typed <{sensitive_key_name}> into element {index}'
			else:
				return f'Typed <sensitive> into element {index}'
		else:
			return f"Typed '{text}' into element {index}"

	async def _get_browser_state(self, compact: bool = True) -> str:
		"""Get current browser state."""
		if not self.browser_session:
			return 'Error: No browser session active'

		state = await self.browser_session.get_browser_state_summary(include_screenshot=False)

		element_count = len(state.dom_state.selector_map)

		result: dict[str, Any] = {
			'url': state.url,
			'title': state.title,
			'tabs': [{'url': tab.url, 'title': tab.title} for tab in state.tabs],
			'interactive_element_count': element_count,
		}

		if not compact:
			elements = []
			for index, element in state.dom_state.selector_map.items():
				elem_info: dict[str, Any] = {
					'index': index,
					'tag': element.tag_name,
					'text': element.get_all_children_text(max_depth=2)[:100],
				}
				if element.attributes.get('placeholder'):
					elem_info['placeholder'] = element.attributes['placeholder']
				if element.attributes.get('href'):
					elem_info['href'] = element.attributes['href']
				if element.attributes.get('id'):
					elem_info['id'] = element.attributes['id']
				if element.attributes.get('class'):
					elem_info['class'] = element.attributes['class']
				elements.append(elem_info)
			result['interactive_elements'] = elements

		return json.dumps(result, indent=2)

	async def _scroll(self, direction: str = 'down') -> str:
		"""Scroll the page."""
		if not self.browser_session:
			return 'Error: No browser session active'

		from openbrowser.browser.events import ScrollEvent

		# Scroll by a standard amount (500 pixels)
		event = self.browser_session.event_bus.dispatch(
			ScrollEvent(
				direction=direction,  # type: ignore
				amount=500,
			)
		)
		await event
		return f'Scrolled {direction}'

	async def _go_back(self) -> str:
		"""Go back in browser history."""
		if not self.browser_session:
			return 'Error: No browser session active'

		from openbrowser.browser.events import GoBackEvent

		event = self.browser_session.event_bus.dispatch(GoBackEvent())
		await event
		return 'Navigated back'

	async def _close_browser(self) -> str:
		"""Close the browser session."""
		if self.browser_session:
			from openbrowser.browser.events import BrowserStopEvent

			event = self.browser_session.event_bus.dispatch(BrowserStopEvent())
			await event
			self.browser_session = None
			self.tools = None
			return 'Browser closed'
		return 'No browser session to close'

	async def _list_tabs(self) -> str:
		"""List all open tabs."""
		if not self.browser_session:
			return 'Error: No browser session active'

		tabs_info = await self.browser_session.get_tabs()
		tabs = []
		for i, tab in enumerate(tabs_info):
			tabs.append({'tab_id': tab.target_id[-4:], 'url': tab.url, 'title': tab.title or ''})
		return json.dumps(tabs, indent=2)

	async def _switch_tab(self, tab_id: str) -> str:
		"""Switch to a different tab."""
		if not self.browser_session:
			return 'Error: No browser session active'

		from openbrowser.browser.events import SwitchTabEvent

		target_id = await self.browser_session.get_target_id_from_tab_id(tab_id)
		event = self.browser_session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
		await event
		state = await self.browser_session.get_browser_state_summary()
		return f'Switched to tab {tab_id}: {state.url}'

	async def _close_tab(self, tab_id: str) -> str:
		"""Close a specific tab."""
		if not self.browser_session:
			return 'Error: No browser session active'

		from openbrowser.browser.events import CloseTabEvent

		target_id = await self.browser_session.get_target_id_from_tab_id(tab_id)
		event = self.browser_session.event_bus.dispatch(CloseTabEvent(target_id=target_id))
		await event
		current_url = await self.browser_session.get_current_page_url()
		return f'Closed tab # {tab_id}, now on {current_url}'

	async def _get_text(self, extract_links: bool = False) -> str:
		"""Get page content as clean markdown text."""
		if not self.browser_session:
			return 'Error: No browser session active'

		try:
			content, stats = await extract_clean_markdown(
				browser_session=self.browser_session,
				extract_links=extract_links,
			)
			if not content or not content.strip():
				return 'No text content found on page'
			return content
		except Exception as e:
			logger.error(f'Failed to extract text: {e}', exc_info=True)
			return f'Error extracting text: {str(e)}'

	async def _grep(
		self,
		pattern: str,
		context_lines: int = 2,
		max_matches: int = 20,
		case_insensitive: bool = True,
	) -> str:
		"""Search page text content using regex or string pattern."""
		if not self.browser_session:
			return 'Error: No browser session active'

		try:
			content, _ = await extract_clean_markdown(browser_session=self.browser_session)
			if not content or not content.strip():
				return json.dumps({'matches': [], 'total_matches': 0, 'message': 'No text content on page'})

			lines = content.split('\n')
			flags = re.IGNORECASE if case_insensitive else 0

			try:
				compiled = re.compile(pattern, flags)
			except re.error:
				# Fall back to literal string search
				escaped = re.escape(pattern)
				compiled = re.compile(escaped, flags)

			matches = []
			for i, line in enumerate(lines):
				if compiled.search(line):
					context_before = lines[max(0, i - context_lines) : i]
					context_after = lines[i + 1 : min(len(lines), i + 1 + context_lines)]
					matches.append(
						{
							'line_number': i + 1,
							'line': line.strip(),
							'context_before': [l.strip() for l in context_before],
							'context_after': [l.strip() for l in context_after],
						}
					)
					if len(matches) >= max_matches:
						break

			total_found = sum(1 for line in lines if compiled.search(line))

			return json.dumps(
				{
					'pattern': pattern,
					'matches': matches,
					'matches_shown': len(matches),
					'total_matches': total_found,
				},
				indent=2,
			)
		except Exception as e:
			logger.error(f'Grep failed: {e}', exc_info=True)
			return f'Error during grep: {str(e)}'

	async def _search_elements(self, query: str, by: str = 'text', max_results: int = 20) -> str:
		"""Search interactive DOM elements by various criteria."""
		if not self.browser_session:
			return 'Error: No browser session active'

		try:
			# Use get_browser_state_summary() for a fresh DOM extraction instead of
			# get_selector_map() which may return stale cached data
			state = await self.browser_session.get_browser_state_summary(include_screenshot=False)
			selector_map = state.dom_state.selector_map if state.dom_state else {}
			results = []
			query_lower = query.lower()

			for index, element in selector_map.items():
				matched = False

				if by == 'text':
					elem_text = element.get_all_children_text(max_depth=3).lower()
					matched = query_lower in elem_text
				elif by == 'tag':
					matched = query_lower == element.tag_name.lower()
				elif by == 'id':
					elem_id = element.attributes.get('id', '').lower()
					matched = query_lower in elem_id
				elif by == 'class':
					elem_class = element.attributes.get('class', '').lower()
					matched = query_lower in elem_class
				elif by == 'attribute':
					for attr_val in element.attributes.values():
						if query_lower in str(attr_val).lower():
							matched = True
							break

				if matched:
					elem_info: dict[str, Any] = {
						'index': index,
						'tag': element.tag_name,
						'text': element.get_all_children_text(max_depth=2)[:100],
					}
					if element.attributes.get('id'):
						elem_info['id'] = element.attributes['id']
					if element.attributes.get('class'):
						elem_info['class'] = element.attributes['class']
					if element.attributes.get('placeholder'):
						elem_info['placeholder'] = element.attributes['placeholder']
					if element.attributes.get('href'):
						elem_info['href'] = element.attributes['href']
					if element.attributes.get('type'):
						elem_info['type'] = element.attributes['type']
					results.append(elem_info)

					if len(results) >= max_results:
						break

			return json.dumps(
				{
					'query': query,
					'by': by,
					'results': results,
					'count': len(results),
				},
				indent=2,
			)
		except Exception as e:
			logger.error(f'Element search failed: {e}', exc_info=True)
			return f'Error searching elements: {str(e)}'

	async def _find_and_scroll(self, text: str) -> str:
		"""Find text on the page and scroll to it."""
		if not self.browser_session:
			return 'Error: No browser session active'

		try:
			event = self.browser_session.event_bus.dispatch(ScrollToTextEvent(text=text))
			await event
			return f"Found and scrolled to: '{text}'"
		except Exception as e:
			logger.error(f'Find and scroll failed: {e}', exc_info=True)
			return f"Text '{text}' not found or not visible on page"

	async def _get_accessibility_tree(self, max_depth: int = -1, include_ignored: bool = False) -> str:
		"""Get the accessibility tree of the current page."""
		if not self.browser_session:
			return 'Error: No browser session active'

		try:
			target_id = self.browser_session.current_target_id
			if not target_id:
				return 'Error: No active page target'

			dom_service = DomService(self.browser_session)
			ax_tree_result = await dom_service._get_ax_tree_for_all_frames(target_id)

			nodes = []
			node_map: dict[str, dict[str, Any]] = {}

			for ax_node in ax_tree_result.get('nodes', []):
				if not include_ignored and ax_node.get('ignored', False):
					continue

				enhanced = dom_service._build_enhanced_ax_node(ax_node)
				node_info: dict[str, Any] = {
					'id': enhanced.ax_node_id,
					'role': enhanced.role,
					'name': enhanced.name,
				}
				if enhanced.description:
					node_info['description'] = enhanced.description
				if enhanced.properties:
					props = {}
					for prop in enhanced.properties:
						if prop.value is not None:
							props[prop.name] = prop.value
					if props:
						node_info['properties'] = props
				if enhanced.child_ids:
					node_info['children'] = enhanced.child_ids

				node_map[enhanced.ax_node_id] = node_info
				nodes.append(node_info)

			def _build_tree(node_id: str, depth: int) -> dict[str, Any] | None:
				"""Recursively build tree structure with depth limit."""
				node = node_map.get(node_id)
				if not node:
					return None
				if max_depth >= 0 and depth > max_depth:
					return None

				result = {k: v for k, v in node.items() if k != 'children'}
				child_ids = node.get('children', [])
				if child_ids:
					children = []
					for child_id in child_ids:
						child = _build_tree(child_id, depth + 1)
						if child:
							children.append(child)
					if children:
						result['children'] = children
				return result

			# Build tree from root nodes (nodes not referenced as children)
			all_child_ids = set()
			for node in nodes:
				for child_id in node.get('children', []):
					all_child_ids.add(child_id)

			root_nodes = [n for n in nodes if n['id'] not in all_child_ids]

			if root_nodes:
				tree = []
				for root in root_nodes:
					built = _build_tree(root['id'], 0)
					if built:
						tree.append(built)

				return json.dumps(
					{
						'tree': tree if len(tree) > 1 else tree[0] if tree else {},
						'total_nodes': len(nodes),
					},
					indent=2,
				)
			else:
				return json.dumps({'nodes': nodes, 'total_nodes': len(nodes)}, indent=2)

		except Exception as e:
			logger.error(f'Accessibility tree extraction failed: {e}', exc_info=True)
			return f'Error getting accessibility tree: {str(e)}'

	async def _execute_js(self, expression: str) -> str:
		"""Execute JavaScript on the current page and return the result."""
		if not self.browser_session:
			return 'Error: No browser session active'

		try:
			target_id = self.browser_session.current_target_id
			if not target_id:
				return 'Error: No active page target'

			cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=target_id, focus=False)

			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={
					'expression': expression,
					'returnByValue': True,
					'awaitPromise': True,
				},
				session_id=cdp_session.session_id,
			)

			if 'exceptionDetails' in result:
				exception = result['exceptionDetails']
				error_text = exception.get('text', 'Unknown error')
				if 'exception' in exception:
					error_text = exception['exception'].get('description', error_text)
				return json.dumps({'error': error_text})

			value = result.get('result', {}).get('value')
			result_type = result.get('result', {}).get('type', 'undefined')

			if result_type == 'undefined':
				return json.dumps({'result': None, 'type': 'undefined'})

			return json.dumps({'result': value, 'type': result_type}, indent=2, default=str)

		except Exception as e:
			logger.error(f'JavaScript execution failed: {e}', exc_info=True)
			return f'Error executing JavaScript: {str(e)}'

	def _track_session(self, session: BrowserSession) -> None:
		"""Track a browser session for management."""
		self.active_sessions[session.id] = {
			'session': session,
			'created_at': time.time(),
			'last_activity': time.time(),
			'url': getattr(session, 'current_url', None),
		}

	def _update_session_activity(self, session_id: str) -> None:
		"""Update the last activity time for a session."""
		if session_id in self.active_sessions:
			self.active_sessions[session_id]['last_activity'] = time.time()

	async def _list_sessions(self) -> str:
		"""List all active browser sessions."""
		if not self.active_sessions:
			return 'No active browser sessions'

		sessions_info = []
		for session_id, session_data in self.active_sessions.items():
			session = session_data['session']
			created_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_data['created_at']))
			last_activity = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_data['last_activity']))

			# Check if session is still active
			is_active = hasattr(session, 'cdp_client') and session.cdp_client is not None

			sessions_info.append(
				{
					'session_id': session_id,
					'created_at': created_at,
					'last_activity': last_activity,
					'active': is_active,
					'current_url': session_data.get('url', 'Unknown'),
					'age_minutes': (time.time() - session_data['created_at']) / 60,
				}
			)

		return json.dumps(sessions_info, indent=2)

	async def _close_session(self, session_id: str) -> str:
		"""Close a specific browser session."""
		if session_id not in self.active_sessions:
			return f'Session {session_id} not found'

		session_data = self.active_sessions[session_id]
		session = session_data['session']

		try:
			# Close the session
			if hasattr(session, 'kill'):
				await session.kill()
			elif hasattr(session, 'close'):
				await session.close()

			# Remove from tracking
			del self.active_sessions[session_id]

			# If this was the current session, clear it
			if self.browser_session and self.browser_session.id == session_id:
				self.browser_session = None
				self.tools = None

			return f'Successfully closed session {session_id}'
		except Exception as e:
			return f'Error closing session {session_id}: {str(e)}'

	async def _close_all_sessions(self) -> str:
		"""Close all active browser sessions."""
		if not self.active_sessions:
			return 'No active sessions to close'

		closed_count = 0
		errors = []

		for session_id in list(self.active_sessions.keys()):
			try:
				result = await self._close_session(session_id)
				if 'Successfully closed' in result:
					closed_count += 1
				else:
					errors.append(f'{session_id}: {result}')
			except Exception as e:
				errors.append(f'{session_id}: {str(e)}')

		# Clear current session references
		self.browser_session = None
		self.tools = None

		result = f'Closed {closed_count} sessions'
		if errors:
			result += f'. Errors: {"; ".join(errors)}'

		return result

	async def _cleanup_expired_sessions(self) -> None:
		"""Background task to clean up expired sessions."""
		current_time = time.time()
		timeout_seconds = self.session_timeout_minutes * 60

		expired_sessions = []
		for session_id, session_data in self.active_sessions.items():
			last_activity = session_data['last_activity']
			if current_time - last_activity > timeout_seconds:
				expired_sessions.append(session_id)

		for session_id in expired_sessions:
			try:
				await self._close_session(session_id)
				logger.info(f'Auto-closed expired session {session_id}')
			except Exception as e:
				logger.error(f'Error auto-closing session {session_id}: {e}')

	async def _start_cleanup_task(self) -> None:
		"""Start the background cleanup task."""

		async def cleanup_loop():
			while True:
				try:
					await self._cleanup_expired_sessions()
					# Check every 2 minutes
					await asyncio.sleep(120)
				except Exception as e:
					logger.error(f'Error in cleanup task: {e}')
					await asyncio.sleep(120)

		self._cleanup_task = asyncio.create_task(cleanup_loop())

	async def run(self):
		"""Run the MCP server."""
		# Start the cleanup task
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
