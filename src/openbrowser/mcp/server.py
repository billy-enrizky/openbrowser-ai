"""MCP Server for openbrowser -- CodeAgent code execution via Model Context Protocol.

Exposes ``execute_code`` plus lifecycle diagnostics and recovery tools.
``execute_code`` runs Python code in a persistent namespace with browser
automation functions (navigate, click, evaluate, etc.).

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
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

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
from openbrowser.browser.lifecycle import LifecycleMachine, LifecyclePhase, LifecycleTransitionError
from openbrowser.code_use.executor import DEFAULT_MAX_OUTPUT_CHARS, CodeExecutor
from openbrowser.code_use.namespace import create_namespace
from openbrowser.config import (
	CONFIG,
	apply_managed_browser_profile_defaults,
	get_default_profile,
	load_openbrowser_config,
)
from openbrowser.tools.service import CodeAgentTools

try:
	from openbrowser.filesystem.file_system import FileSystem

	FILESYSTEM_AVAILABLE = True
except ModuleNotFoundError:
	FILESYSTEM_AVAILABLE = False
except Exception:
	FILESYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)

_MCP_WORKSPACE_DIR = Path.home() / 'Downloads' / 'openbrowser-mcp' / 'workspace'
_CDP_HEALTH_CHECK_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class _CleanupOutcome:
	succeeded: bool
	caller_cancellation: asyncio.CancelledError | None = None


def _start_timeout_seconds() -> float:
	raw = os.environ.get('OPENBROWSER_MCP_START_TIMEOUT', '')
	try:
		return max(1.0, float(raw)) if raw else 30.0
	except ValueError:
		return 30.0


def _exec_timeout_seconds() -> float:
	raw = os.environ.get('OPENBROWSER_MCP_EXEC_TIMEOUT', '')
	try:
		return max(1.0, float(raw)) if raw else 110.0
	except ValueError:
		return 110.0


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
	from mcp import MCPError
	from mcp.server import CacheHint, Server, ServerRequestContext

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
)
from openbrowser.code_use.descriptions import (
	EXECUTE_CODE_DESCRIPTION_COMPACT as _EXECUTE_CODE_DESCRIPTION_COMPACT,
)


class OpenBrowserServer:
	"""MCP Server exposing CodeAgent code execution environment.

	Provides ``execute_code`` plus ``get_status`` and ``reset_browser``.
	``execute_code`` runs Python code in a persistent namespace populated with
	browser automation functions from ``create_namespace()`` (navigate, click,
	evaluate, etc.).
	"""

	def __init__(self, session_timeout_minutes: int = 10):
		_ensure_all_loggers_use_stderr()

		self.config = load_openbrowser_config()
		self.browser_session: BrowserSession | None = None
		self._lifecycle = LifecycleMachine('mcp_server')
		self._browser_lifecycle = LifecycleMachine('mcp_browser')
		self._browser_generation = 0
		self._pending_variables: dict[str, Any] = {}
		self._profile_summary: dict[str, Any] | None = None
		self._instance_id = uuid.uuid4().hex
		self._managed_profile_dir: Path | None = None
		self._managed_storage_state: Path | None = None
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
		self._lifecycle_lock = asyncio.Lock()
		self._execution_lock = asyncio.Lock()
		self._execution_task: asyncio.Task[Any] | None = None

		self._setup_handlers()

	def _setup_handlers(self):
		"""Configure SDK callbacks from one instance-owned catalog and schema source."""

		description = _EXECUTE_CODE_DESCRIPTION
		if os.environ.get('OPENBROWSER_COMPACT_DESCRIPTION', '').lower() in ('1', 'true', 'yes'):
			description = _EXECUTE_CODE_DESCRIPTION_COMPACT
		tool_catalog = (
			types.Tool(
				name='execute_code',
				description=description,
				input_schema={
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
					read_only_hint=False,
					destructive_hint=True,
					open_world_hint=True,
					idempotent_hint=False,
				),
			),
			types.Tool(
				name='get_status',
				description=(
					'Return OpenBrowser server, browser, profile, and tab state as JSON. '
					'Read-only and does not start a browser. Use this before retrying a failed task.'
				),
				input_schema={
					'type': 'object',
					'properties': {},
					'additionalProperties': False,
				},
				annotations=types.ToolAnnotations(
					read_only_hint=True,
					destructive_hint=False,
					open_world_hint=False,
					idempotent_hint=True,
				),
			),
			types.Tool(
				name='reset_browser',
				description=(
					'Close the current browser session and release its profile. '
					'User-defined Python variables are preserved unless clear_variables is true. '
					'The next execute_code call starts a fresh browser.'
				),
				input_schema={
					'type': 'object',
					'properties': {
						'clear_variables': {
							'type': 'boolean',
							'description': 'Also discard user-defined Python variables.',
							'default': False,
						},
					},
					'additionalProperties': False,
				},
				annotations=types.ToolAnnotations(
					read_only_hint=False,
					destructive_hint=True,
					open_world_hint=False,
					idempotent_hint=True,
				),
			),
		)
		tools_by_name = {tool.name: tool for tool in tool_catalog}
		validators = {}
		for tool in tool_catalog:
			Draft202012Validator.check_schema(tool.input_schema)
			validators[tool.name] = Draft202012Validator(tool.input_schema)

		async def handle_list_tools(
			ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
		) -> types.ListToolsResult:
			return types.ListToolsResult(tools=[tool.model_copy(deep=True) for tool in tool_catalog])

		async def handle_list_resources(
			ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
		) -> types.ListResourcesResult:
			return types.ListResourcesResult(resources=[])

		async def handle_list_resource_templates(
			ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
		) -> types.ListResourceTemplatesResult:
			return types.ListResourceTemplatesResult(resource_templates=[])

		async def handle_list_prompts(
			ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
		) -> types.ListPromptsResult:
			return types.ListPromptsResult(prompts=[])

		async def handle_call_tool(
			ctx: ServerRequestContext, params: types.CallToolRequestParams
		) -> types.CallToolResult:
			start_time = time.time()
			error_msg = None
			try:
				if params.name not in tools_by_name:
					raise MCPError(code=-32602, message=f'Unknown tool: {params.name}')
				arguments = params.arguments if params.arguments is not None else {}
				validation_error = next(validators[params.name].iter_errors(arguments), None)
				if validation_error is not None:
					error_msg = f'Invalid arguments: {validation_error.message}'
				elif params.name == 'execute_code' and not arguments['code'].strip():
					error_msg = 'Error: No code provided'
				if error_msg is not None:
					return types.CallToolResult(content=[types.TextContent(type='text', text=error_msg)], is_error=True)
				async def owned_call() -> str | BaseException:
					try:
						return await self._handle_tool(params.name, arguments)
					except asyncio.CancelledError:
						raise
					except BaseException as error:
						return error

				operation = asyncio.create_task(owned_call())
				completion = asyncio.gather(operation, return_exceptions=True)
				caller_cancellation = None
				while not completion.done():
					try:
						await asyncio.shield(completion)
					except asyncio.CancelledError as error:
						if caller_cancellation is None:
							caller_cancellation = error
							operation.cancel()
				result = completion.result()[0]
				if caller_cancellation is not None:
					raise caller_cancellation
				if operation.cancelled():
					raise RuntimeError('Tool operation was cancelled')
				if isinstance(result, BaseException):
					raise result
				return types.CallToolResult(content=[types.TextContent(type='text', text=result)])
			except asyncio.CancelledError as error:
				error_msg = f'{type(error).__name__}: {error}'
				raise
			except MCPError as error:
				error_msg = str(error)
				raise
			except Exception as error:
				error_msg = str(error)
				logger.error('Tool execution failed: %s', error, exc_info=True)
				return types.CallToolResult(content=[types.TextContent(type='text', text=f'Error: {error}')], is_error=True)
			except BaseException as error:
				error_msg = f'{type(error).__name__}: {error}'
				raise
			finally:
				if self._telemetry and TELEMETRY_AVAILABLE:
					self._telemetry.capture(
						MCPServerTelemetryEvent(
							version=get_openbrowser_version(),
							action='tool_call',
							tool_name=params.name,
							duration_seconds=time.time() - start_time,
							error_message=error_msg,
						)
					)

		def get_tool_input_schema(name: str) -> dict[str, Any] | None:
			tool = tools_by_name.get(name)
			return deepcopy(tool.input_schema) if tool is not None else None

		self.server = Server(
			'openbrowser',
			version=get_openbrowser_version(),
			on_list_tools=handle_list_tools,
			on_list_resources=handle_list_resources,
			on_list_resource_templates=handle_list_resource_templates,
			on_list_prompts=handle_list_prompts,
			on_call_tool=handle_call_tool,
			get_tool_input_schema=get_tool_input_schema,
			cache_hints={
				method: CacheHint(ttl_ms=60_000, scope='private')
				for method in ('tools/list', 'resources/list', 'resources/templates/list', 'prompts/list')
			},
		)

	async def _handle_tool(self, name: str, arguments: dict[str, Any]) -> str:
		"""Dispatch one MCP tool call to its implementation."""
		if name == 'execute_code':
			code = arguments.get('code', '')
			if not code.strip():
				return 'Error: No code provided'
			return await self._execute_code(code)
		if name == 'get_status':
			return json.dumps(self._status_payload(), indent=2)
		if name == 'reset_browser':
			return await self._reset_browser(clear_variables=bool(arguments.get('clear_variables', False)))
		return f'Unknown tool: {name}'

	def _build_browser_profile(self):
		"""Build a BrowserProfile from config with MCP defaults."""
		profile_config = dict(get_default_profile(self.config))
		explicit_user_data_dir = profile_config.get('user_data_dir')
		profile_data = {
			'downloads_path': str(Path.home() / 'Downloads' / 'openbrowser-mcp'),
			'wait_between_actions': 0.5,
			'keep_alive': True,
			'device_scale_factor': 1.0,
			'disable_security': False,
			'headless': True,
			**profile_config,
		}
		env_headless = os.environ.get('OPENBROWSER_HEADLESS', '').lower()
		if env_headless in ('true', '1', 'yes'):
			profile_data['headless'] = True
		elif env_headless in ('false', '0', 'no'):
			profile_data['headless'] = False

		if explicit_user_data_dir:
			# Explicit paths are intentionally shared and protected by the
			# profile lease. Never silently replace one with a temporary path.
			managed_default_dir = None
		else:
			managed_default_dir = self._get_managed_profile_dir()
			profile_data['user_data_dir'] = str(managed_default_dir)
			configured_storage_state = profile_data.get('storage_state')
			if configured_storage_state is None or isinstance(configured_storage_state, (str, Path)):
				storage_state_path = managed_default_dir / 'storage_state.json'
				self._seed_managed_storage_state(storage_state_path, source=configured_storage_state)
				profile_data['storage_state'] = str(storage_state_path)
			else:
				# In-memory Playwright storage state is already isolated per profile
				# object. Do not coerce it into a shared filesystem path.
				profile_data['storage_state'] = deepcopy(configured_storage_state)

		profile_data = apply_managed_browser_profile_defaults(profile_data, managed_default_dir)
		args = list(profile_data.get('args') or [])
		ownership_marker = f'--openbrowser-instance-id={self._instance_id}'
		if not any(argument.startswith('--openbrowser-instance-id=') for argument in args):
			args.append(ownership_marker)
		profile_data['args'] = args
		profile = BrowserProfile(**profile_data)
		storage_state = profile.storage_state
		self._profile_summary = {
			'user_data_dir': str(profile.user_data_dir) if profile.user_data_dir else None,
			'headless': profile.headless,
			'storage_state': str(storage_state) if isinstance(storage_state, (str, Path)) else None,
			'instance_id': self._instance_id,
		}
		return profile

	def _get_managed_profile_dir(self) -> Path:
		"""Return the stable isolated profile directory for this MCP instance."""
		if self._managed_profile_dir is None:
			self._managed_profile_dir = CONFIG.OPENBROWSER_PROFILES_DIR / f'mcp-{self._instance_id}'
		self._managed_profile_dir.mkdir(parents=True, exist_ok=True)
		return self._managed_profile_dir

	def _seed_managed_storage_state(self, destination: Path, source: str | Path | None = None) -> None:
		"""Copy configured or default state once into this instance's profile."""
		if destination.exists():
			self._managed_storage_state = destination
			return

		source_path = Path(source).expanduser() if source else CONFIG.OPENBROWSER_PROFILES_DIR / 'default' / 'storage_state.json'
		if source_path.is_file():
			destination.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(source_path, destination)
		self._managed_storage_state = destination

	async def _ensure_namespace(self):
		"""Lazily initialize browser session, tools, and namespace on first use."""
		if self._namespace is not None:
			return

		async with self._lifecycle_lock:
			if self._namespace is not None:
				return
			if self.browser_session is not None:
				raise RuntimeError('A previous browser session still owns resources after cleanup failed')

			_ensure_all_loggers_use_stderr()
			if self._browser_lifecycle.phase not in (LifecyclePhase.STOPPED, LifecyclePhase.FAILED):
				try:
					self._browser_lifecycle.transition(LifecyclePhase.STOPPING, reason='reinitializing browser')
					self._browser_lifecycle.transition(
						LifecyclePhase.STOPPED,
						reason='browser reinitialized',
						increment_generation=True,
					)
				except LifecycleTransitionError:
					pass
			self._browser_lifecycle.transition(LifecyclePhase.STARTING, reason='browser start requested')

			# Initialize browser session
			session = None
			try:
				profile = self._build_browser_profile()
				session = BrowserSession(browser_profile=profile)
				self.browser_session = session
				await asyncio.wait_for(session.start(), timeout=_start_timeout_seconds())
				session.seed_generation(self._browser_generation)

				# Create CodeAgent tools and namespace
				self._tools = CodeAgentTools()
				self._namespace = create_namespace(
					browser_session=self.browser_session,
					tools=self._tools,
					file_system=_create_mcp_file_system(),
				)
				for key, value in self._pending_variables.items():
					if not key.startswith('__') and key not in self._namespace:
						self._namespace[key] = value
				self._pending_variables.clear()
				self._executor.set_namespace(self._namespace)
				self._browser_lifecycle.transition(LifecyclePhase.READY, reason='browser ready')
			except BaseException as error:
				message = f'{type(error).__name__}: {error}'
				self._browser_lifecycle.fail(message, reason='browser start failed')
				logger.error('Failed to initialize browser session: %s', error)
				if session is not None:
					cleanup = await self._kill_session_safely(session)
					if not cleanup.succeeded:
						logger.error('Failed to clean up browser session after initialization failure')
					if cleanup.succeeded and self.browser_session is session:
						self.browser_session = None
					if cleanup.succeeded:
						self._namespace = None
					if cleanup.caller_cancellation is not None and not isinstance(error, asyncio.CancelledError):
						raise cleanup.caller_cancellation
				raise

	async def _is_cdp_alive(self) -> bool:
		"""Check if the browser session's CDP WebSocket is still connected."""
		if not self.browser_session:
			return False
		root = getattr(self.browser_session, '_cdp_client_root', None)
		if root is None:
			return False
		try:
			await asyncio.wait_for(root.send.Browser.getVersion(), timeout=_CDP_HEALTH_CHECK_TIMEOUT_SECONDS)
			return True
		except Exception:
			return False

	async def _kill_session_safely(self, session: BrowserSession) -> _CleanupOutcome:
		"""Complete owned cleanup without confusing worker and caller cancellation."""
		async def owned_cleanup() -> bool:
			try:
				result = session.kill()
				if inspect.isawaitable(result):
					await result
				else:
					from openbrowser.browser.events import BrowserStopEvent
					event = session.event_bus.dispatch(BrowserStopEvent(force=True))
					if inspect.isawaitable(event):
						await event
					event_result = getattr(event, 'event_result', None)
					if callable(event_result):
						result = event_result(raise_if_any=True, raise_if_none=False)
						if inspect.isawaitable(result):
							await result
				return True
			except BaseException:
				return False

		cleanup_task = asyncio.create_task(owned_cleanup())
		completion = asyncio.gather(cleanup_task, return_exceptions=True)
		caller_cancellation = None
		while not completion.done():
			try:
				await asyncio.shield(completion)
			except asyncio.CancelledError as error:
				if caller_cancellation is None:
					caller_cancellation = error
		result = completion.result()[0]
		return _CleanupOutcome(result is True, caller_cancellation)

	async def _recover_browser_session(self) -> None:
		"""Serialize recovery with execution and lifecycle cleanup."""
		if self._execution_task is asyncio.current_task():
			await self._recover_browser_session_locked()
			return

		async with self._execution_lock:
			await self._recover_browser_session_locked()

	async def _recover_browser_session_locked(self) -> None:
		"""Kill the dead browser session and create a fresh one.

		Called when we detect the CDP WebSocket has disconnected (e.g. Chrome
		crashed or a navigation broke the connection).  Re-creates the
		BrowserSession and rebuilds the namespace so the next execute_code
		call works transparently.
		"""
		async with self._lifecycle_lock:
			if self.browser_session and await self._is_cdp_alive():
				return

			logger.info('CDP connection lost, recovering browser session')
			try:
				self._browser_lifecycle.transition(LifecyclePhase.RECOVERING, reason='cdp connection lost')
			except LifecycleTransitionError:
				self._browser_lifecycle.transition(LifecyclePhase.STARTING, reason='cdp connection lost')

			# 1. Tear down the old session
			old_session = self.browser_session
			old_ns = self._namespace or {}
			if old_session:
				cleanup = await self._kill_session_safely(old_session)
				if cleanup.succeeded and self.browser_session is old_session:
					self.browser_session = None
				if cleanup.succeeded:
					self._namespace = None
				if not cleanup.succeeded:
					self._browser_lifecycle.fail('Unable to safely close the old browser session during recovery', reason='browser recovery cleanup failed')
				if cleanup.caller_cancellation is not None:
					raise cleanup.caller_cancellation
				if not cleanup.succeeded:
					raise RuntimeError('Unable to safely close the old browser session during recovery')
			else:
				self._namespace = None

			# 2. Create a brand-new session. The local watchdog owns its profile
			# lease and can reclaim only its recorded browser process.
			try:
				profile = self._build_browser_profile()
				session = BrowserSession(browser_profile=profile)
			except BaseException:
				self.browser_session = None
				self._namespace = None
				raise

			self.browser_session = session
			try:
				await asyncio.wait_for(session.start(), timeout=_start_timeout_seconds())
				self._browser_generation += 1
				session.seed_generation(self._browser_generation)

				# 3. Rebuild namespace with the new session (preserving user variables)
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
				self._browser_lifecycle.transition(
					LifecyclePhase.READY,
					reason='browser recovered',
					increment_generation=True,
				)
			except BaseException as error:
				self._browser_lifecycle.fail(error, reason='browser recovery failed')
				cleanup = await self._kill_session_safely(session)
				if not cleanup.succeeded:
					logger.error('Failed to clean up replacement browser session')
				if cleanup.succeeded and self.browser_session is session:
					self.browser_session = None
				self._namespace = None
				if cleanup.caller_cancellation is not None and not isinstance(error, asyncio.CancelledError):
					raise cleanup.caller_cancellation
				raise

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
		"""Serialize execution with browser lifecycle operations."""
		async with self._execution_lock:
			self._execution_task = asyncio.current_task()
			try:
				return await self._execute_code_locked(code)
			finally:
				self._last_activity = time.time()
				self._execution_task = None

	async def _execute_code_locked(self, code: str) -> str:
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
				return (
					f'Error: Browser connection lost and recovery failed: {type(recovery_err).__name__}: {recovery_err}\n\n'
					'Run get_status, then reset_browser before retrying.'
				)

		# Execute via shared CodeExecutor with a bound that leaves room for the
		# calling agent to receive the timeout message before its own timeout.
		try:
			result = await asyncio.wait_for(
				self._executor.execute(code),
				timeout=_exec_timeout_seconds(),
			)
		except asyncio.TimeoutError:
			return (
				f'Error: Execution timed out after {_exec_timeout_seconds():g}s. '
				'The code may still be running. Run get_status before retrying.'
			)

		# If the error looks like a CDP connection issue, recover the browser
		# but do not run the code again. The first attempt may have changed
		# the page before the connection failed.
		error_text = result.error or result.output
		if not result.success and self._is_connection_error(error_text):
			logger.info('CDP connection error during execution, recovering')
			try:
				await self._recover_browser_session()
			except Exception as recovery_err:
				logger.error('CDP recovery failed: %s', recovery_err)
				return (
					f'{result.output}\n\n[openbrowser] Browser recovery failed: '
					f'{type(recovery_err).__name__}: {recovery_err}. Run get_status before retrying.'
				)
			return (
				f'{result.output}\n\n[openbrowser] Browser was recovered. '
				'The code may have partially run. Run get_status before retrying.'
			)

		return result.output

	def _status_payload(self) -> dict[str, Any]:
		"""Build a JSON-safe status snapshot without starting the browser."""
		browser = self._browser_lifecycle.snapshot().to_dict()
		if self.browser_session is not None and hasattr(self.browser_session, 'lifecycle_snapshot'):
			try:
				snapshot = self.browser_session.lifecycle_snapshot()
				if isinstance(snapshot, dict):
					browser = snapshot
			except Exception:
				pass

		namespace_variables = 0
		if self._namespace is not None:
			namespace_variables = sum(1 for key in self._namespace if not key.startswith('__'))

		try:
			version = get_openbrowser_version()
		except Exception:
			version = None

		return {
			'server': self._lifecycle.snapshot().to_dict(),
			'browser': browser,
			'profile': self._profile_summary,
			'instance_id': self._instance_id,
			'namespace_variables': namespace_variables,
			'pending_variables': len(self._pending_variables),
			'session_timeout_minutes': self.session_timeout_minutes,
			'exec_timeout': _exec_timeout_seconds(),
			'uptime_seconds': max(0.0, time.time() - self._start_time),
			'version': version,
		}

	async def _reset_browser(self, *, clear_variables: bool = False) -> str:
		"""Close the current browser and preserve user variables by default."""
		async with self._execution_lock:
			async with self._lifecycle_lock:
				session = self.browser_session
				old_namespace = self._namespace or {}
				cleanup = _CleanupOutcome(True)

				if session is not None:
					cleanup = await self._kill_session_safely(session)
					if not cleanup.succeeded:
						self._browser_lifecycle.fail(
							'Unable to close the browser session during reset',
							reason='browser reset failed',
						)
						if cleanup.caller_cancellation is not None:
							raise cleanup.caller_cancellation
						return json.dumps(
							{
								'ok': False,
								'error': 'Unable to close the browser session during reset',
								'browser': self._status_payload()['browser'],
							},
							indent=2,
						)
					if self.browser_session is session:
						self.browser_session = None

				self._namespace = None
				self._executor.set_namespace({})
				if clear_variables:
					self._pending_variables.clear()
				else:
					merged = {**self._pending_variables, **old_namespace}
					self._pending_variables = {key: value for key, value in merged.items() if not key.startswith('__')}
				self._browser_generation += 1
				try:
					self._browser_lifecycle.transition(
						LifecyclePhase.STOPPED,
						reason='browser reset',
						increment_generation=True,
					)
				except LifecycleTransitionError:
					pass
				if cleanup.caller_cancellation is not None:
					raise cleanup.caller_cancellation

				return json.dumps(
					{
						'ok': True,
						'message': 'Browser reset',
						'preserved_variables': len(self._pending_variables),
						'browser': self._status_payload()['browser'],
					},
					indent=2,
				)

	async def _shutdown(self) -> None:
		"""Release ownership, separating the shutdown worker from its caller."""
		async def owned_shutdown() -> BaseException | None:
			try:
				await self._shutdown_locked()
			except asyncio.CancelledError:
				raise
			except BaseException as error:
				return error
			return None

		shutdown_task = asyncio.create_task(owned_shutdown())
		completion = asyncio.gather(shutdown_task, return_exceptions=True)
		caller_cancellation = None
		while not completion.done():
			try:
				await asyncio.shield(completion)
			except asyncio.CancelledError as error:
				if caller_cancellation is None:
					caller_cancellation = error
		result = completion.result()[0]
		if isinstance(result, BaseException):
			if self._browser_lifecycle.phase is not LifecyclePhase.FAILED:
				self._browser_lifecycle.fail(result, reason='shutdown failed')
			if caller_cancellation is not None:
				raise caller_cancellation
			if shutdown_task.cancelled():
				raise RuntimeError('Owned shutdown cleanup was cancelled') from result
			raise result
		if caller_cancellation is not None:
			raise caller_cancellation

	async def _shutdown_locked(self) -> None:
		"""Perform shutdown while isolated from cancellation of the caller."""
		cleanup_task = self._cleanup_task
		self._cleanup_task = None
		if cleanup_task:
			cleanup_task.cancel()
			try:
				await cleanup_task
			except asyncio.CancelledError:
				pass
			except Exception as e:
				logger.error('Error stopping cleanup task: %s', e)

		async with self._execution_lock:
			async with self._lifecycle_lock:
				session = self.browser_session
				if not session:
					self._namespace = None
					try:
						self._browser_lifecycle.transition(
							LifecyclePhase.STOPPED,
							reason='server shutdown',
							increment_generation=True,
						)
					except LifecycleTransitionError:
						pass
					return

				cleanup = await self._kill_session_safely(session)
				if cleanup.succeeded and self.browser_session is session:
					self.browser_session = None
					self._namespace = None
					try:
						self._browser_lifecycle.transition(
							LifecyclePhase.STOPPED,
							reason='server shutdown',
							increment_generation=True,
						)
					except LifecycleTransitionError:
						pass
				elif not cleanup.succeeded:
					self._browser_lifecycle.fail(
						'Unable to safely close the browser session during shutdown',
						reason='shutdown failed',
					)
					if cleanup.caller_cancellation is not None:
						raise cleanup.caller_cancellation
					raise RuntimeError('Unable to safely close the browser session during shutdown')
				if cleanup.caller_cancellation is not None:
					raise cleanup.caller_cancellation

	async def _cleanup_expired_session(self) -> None:
		"""Close browser session if idle beyond timeout."""
		async with self._execution_lock:
			async with self._lifecycle_lock:
				session = self.browser_session
				if not session:
					return

				current_time = time.time()
				timeout_seconds = self.session_timeout_minutes * 60

				if current_time - self._last_activity > timeout_seconds:
					logger.info('Auto-closing idle browser session')
					old_namespace = self._namespace or {}
					cleanup = await self._kill_session_safely(session)
					if cleanup.succeeded and self.browser_session is session:
						self.browser_session = None
						self._namespace = None
						self._pending_variables = {
							key: value
							for key, value in old_namespace.items()
							if not key.startswith('__')
						}
						self._browser_generation += 1
						try:
							self._browser_lifecycle.transition(
								LifecyclePhase.STOPPED,
								reason='idle timeout',
								increment_generation=True,
							)
						except LifecycleTransitionError:
							pass
					elif not cleanup.succeeded:
						self._browser_lifecycle.fail('Unable to close the idle browser session', reason='idle cleanup failed')
						logger.error('Error closing idle session: cleanup failed')
					if cleanup.caller_cancellation is not None:
						raise cleanup.caller_cancellation

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
		self._lifecycle.transition(LifecyclePhase.STARTING, reason='mcp server starting')
		try:
			await self._start_cleanup_task()
			self._lifecycle.transition(LifecyclePhase.READY, reason='mcp server ready')

			async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
				await self.server.run(
					read_stream,
					write_stream,
					self.server.create_initialization_options(),
				)
		finally:
			try:
				self._lifecycle.transition(LifecyclePhase.STOPPING, reason='mcp server stopping')
			except LifecycleTransitionError:
				pass
			await self._shutdown()
			try:
				self._lifecycle.transition(
					LifecyclePhase.STOPPED,
					reason='mcp server stopped',
					increment_generation=True,
				)
			except LifecycleTransitionError:
				pass


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
