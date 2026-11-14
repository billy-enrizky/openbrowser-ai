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
import inspect
import json
import logging
import shutil
import time
import uuid
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

		self.server = Server('openbrowser')
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
		"""Setup MCP server handlers."""

		@self.server.list_tools()
		async def handle_list_tools() -> list[types.Tool]:
			"""List the code execution and recovery tools."""
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
				types.Tool(
					name='get_status',
					description=(
						'Return OpenBrowser server, browser, profile, and tab state as JSON. '
						'Read-only and does not start a browser. Use this before retrying a failed task.'
					),
					inputSchema={
						'type': 'object',
						'properties': {},
						'additionalProperties': False,
					},
					annotations=types.ToolAnnotations(
						readOnlyHint=True,
						destructiveHint=False,
						openWorldHint=False,
						idempotentHint=True,
					),
				),
				types.Tool(
					name='reset_browser',
					description=(
						'Close the current browser session and release its profile. '
						'User-defined Python variables are preserved unless clear_variables is true. '
						'The next execute_code call starts a fresh browser.'
					),
					inputSchema={
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
						readOnlyHint=False,
						destructiveHint=True,
						openWorldHint=False,
						idempotentHint=True,
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
				result = await self._handle_tool(name, arguments or {})
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
		else:
			profile_data['headless'] = True

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
			profile = self._build_browser_profile()
			session = BrowserSession(browser_profile=profile)
			self.browser_session = session
			try:
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
				cleanup_succeeded, was_cancelled = await self._kill_session_safely(session)
				if not cleanup_succeeded:
					logger.error('Failed to clean up browser session after initialization failure')
				if cleanup_succeeded and self.browser_session is session:
					self.browser_session = None
				if cleanup_succeeded:
					self._namespace = None
				if was_cancelled:
					raise asyncio.CancelledError
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

	async def _kill_session_safely(self, session: BrowserSession) -> tuple[bool, bool]:
		"""Kill a session without allowing cancellation to interrupt cleanup."""
		try:
			kill_result = session.kill()
		except BaseException:
			return False, False
		if not inspect.isawaitable(kill_result):
			# Keep compatibility with lightweight session doubles and older session
			# implementations that expose only the stop event lifecycle.
			try:
				from openbrowser.browser.events import BrowserStopEvent
				event = session.event_bus.dispatch(BrowserStopEvent(force=True))
				if inspect.isawaitable(event):
					await event
				event_result = getattr(event, 'event_result', None)
				if callable(event_result):
					result = event_result(raise_if_any=True, raise_if_none=False)
					if inspect.isawaitable(result):
						await result
				return True, False
			except BaseException:
				return False, False

		kill_task = asyncio.create_task(kill_result)
		was_cancelled = False
		try:
			await asyncio.shield(kill_task)
		except asyncio.CancelledError:
			was_cancelled = True
		except BaseException:
			return False, False

		# A caller can be cancelled more than once while the owned browser is
		# being stopped. Keep draining the shielded task until it reaches a
		# terminal state so a second cancellation cannot orphan the browser.
		while not kill_task.done():
			try:
				await asyncio.shield(kill_task)
			except asyncio.CancelledError:
				was_cancelled = True
			except BaseException:
				return False, was_cancelled

		try:
			kill_task.result()
		except BaseException:
			return False, was_cancelled
		return True, was_cancelled

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
				cleanup_succeeded, was_cancelled = await self._kill_session_safely(old_session)
				if cleanup_succeeded and self.browser_session is old_session:
					self.browser_session = None
				if cleanup_succeeded:
					self._namespace = None
				if was_cancelled:
					raise asyncio.CancelledError
				if not cleanup_succeeded:
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
				cleanup_succeeded, was_cancelled = await self._kill_session_safely(session)
				if not cleanup_succeeded:
					logger.error('Failed to clean up replacement browser session')
				if cleanup_succeeded and self.browser_session is session:
					self.browser_session = None
				self._namespace = None
				if was_cancelled:
					raise asyncio.CancelledError
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

				if session is not None:
					cleanup_succeeded, was_cancelled = await self._kill_session_safely(session)
					if not cleanup_succeeded:
						self._browser_lifecycle.fail(
							'Unable to close the browser session during reset',
							reason='browser reset failed',
						)
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
					if was_cancelled:
						raise asyncio.CancelledError

				self._namespace = None
				self._executor.set_namespace({})
				if clear_variables:
					self._pending_variables.clear()
				else:
					self._pending_variables = {
						key: value
						for key, value in old_namespace.items()
						if not key.startswith('__')
					}
				self._browser_generation += 1
				try:
					self._browser_lifecycle.transition(
						LifecyclePhase.STOPPED,
						reason='browser reset',
						increment_generation=True,
					)
				except LifecycleTransitionError:
					pass

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
		"""Stop background cleanup and release this server's browser ownership."""
		shutdown_task = asyncio.create_task(self._shutdown_locked())
		cancellation_requested = False
		while not shutdown_task.done():
			try:
				await asyncio.shield(shutdown_task)
			except asyncio.CancelledError:
				cancellation_requested = True

		shutdown_task.result()
		if cancellation_requested:
			raise asyncio.CancelledError

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

				cleanup_succeeded, was_cancelled = await self._kill_session_safely(session)
				if cleanup_succeeded and self.browser_session is session:
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
				elif not cleanup_succeeded:
					self._browser_lifecycle.fail(
						'Unable to safely close the browser session during shutdown',
						reason='shutdown failed',
					)
					raise RuntimeError('Unable to safely close the browser session during shutdown')
				if was_cancelled:
					raise asyncio.CancelledError

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
					cleanup_succeeded, was_cancelled = await self._kill_session_safely(session)
					if cleanup_succeeded and self.browser_session is session:
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
					elif not cleanup_succeeded:
						logger.error('Error closing idle session: cleanup failed')
					if was_cancelled:
						raise asyncio.CancelledError

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
					InitializationOptions(
						server_name='openbrowser',
						server_version=get_openbrowser_version(),
						capabilities=self.server.get_capabilities(
							notification_options=NotificationOptions(),
							experimental_capabilities={},
						),
					),
				),
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
