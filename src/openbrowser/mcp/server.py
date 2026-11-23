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

from __future__ import annotations

import os
import sys

# Set environment variables BEFORE any openbrowser imports to prevent early logging
os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'critical'
os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'

import asyncio
import contextlib
import inspect
import json
import logging
import shutil
import time
import uuid
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
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
from openbrowser.browser.download_state import downloads_summary
from openbrowser.browser.lifecycle import (
	LifecycleMachine,
	LifecyclePhase,
	LifecycleTransitionError,
)
from openbrowser.code_use.executor import DEFAULT_MAX_OUTPUT_CHARS, CodeExecutor, output_limit_for_executor
from openbrowser.code_use.namespace import create_namespace
from openbrowser.config import (
	CONFIG,
	ConfigStartupError,
	apply_managed_browser_profile_defaults,
	get_default_profile,
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

from openbrowser.mcp.cancellation import _CleanupOutcome, _drain_owned_completion

_MCP_WORKSPACE_ROOT = os.environ.get('OPENBROWSER_MCP_WORKSPACE_DIR', '').strip()

_PARTIAL_RUN_WARNING = 'The code may have partially run; run get_status before retrying.'
_MCP_WORKSPACE_DIR = (
	Path(_MCP_WORKSPACE_ROOT).expanduser()
	if _MCP_WORKSPACE_ROOT
	else Path.home() / 'Downloads' / 'openbrowser-mcp' / 'workspace'
)
_CDP_HEALTH_CHECK_TIMEOUT_SECONDS = 2.0

MODERN_PROTOCOL_VERSION = '2026-07-28'
LEGACY_PROTOCOL_VERSIONS = frozenset({'2024-11-05', '2025-03-26', '2025-06-18', '2025-11-25'})
_DEFAULT_MAX_CONTEXTS = 16


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


def _shutdown_timeout_seconds() -> float:
	raw = os.environ.get('OPENBROWSER_SHUTDOWN_TIMEOUT', '')
	try:
		return max(1.0, float(raw)) if raw else 20.0
	except ValueError:
		return 20.0


def _create_mcp_file_system() -> Any:
	"""Create a FileSystem instance for MCP mode, or None if unavailable."""
	if not FILESYSTEM_AVAILABLE:
		return None
	return FileSystem(base_dir=str(_MCP_WORKSPACE_DIR), create_default_files=False)


def _max_contexts() -> int:
	"""Read the positive context capacity override, or use the default."""
	raw = os.environ.get('OPENBROWSER_MCP_MAX_CONTEXTS', '')
	try:
		value = int(raw)
	except (TypeError, ValueError):
		return _DEFAULT_MAX_CONTEXTS
	return value if value > 0 else _DEFAULT_MAX_CONTEXTS


_PROCESS_OUTPUT_LOCKS: weakref.WeakKeyDictionary[Any, asyncio.Lock] = weakref.WeakKeyDictionary()


def _process_output_lock() -> asyncio.Lock:
	"""Return one stdout-capture lock for every server on this event loop."""
	loop = asyncio.get_running_loop()
	lock = _PROCESS_OUTPUT_LOCKS.get(loop)
	if lock is None:
		lock = asyncio.Lock()
		_PROCESS_OUTPUT_LOCKS[loop] = lock
	return lock


def _tool_catalog(description: str, *, modern: bool) -> tuple[types.Tool, ...]:
	"""Build the exact legacy or modern catalog from one source."""
	context_property = {
		'type': 'string',
		'description': (
			'Opaque context handle returned by create_context. Pass the same value '
			'on every stateful call for that context.'
		),
	}
	execute_properties: dict[str, Any] = {
		'code': {
			'type': 'string',
			'description': 'Python code to execute. All browser functions are async (use await).',
		},
	}
	if modern:
		execute_properties = {'context_id': context_property, **execute_properties}
	execute_schema: dict[str, Any] = {
		'type': 'object',
		'properties': execute_properties,
		'required': ['context_id', 'code'] if modern else ['code'],
	}
	status_schema: dict[str, Any] = {
		'type': 'object',
		'properties': {'context_id': context_property} if modern else {},
		'additionalProperties': False,
	}
	if modern:
		status_schema['required'] = ['context_id']
	reset_properties: dict[str, Any] = {
		'clear_variables': {
			'type': 'boolean',
			'description': 'Also discard user-defined Python variables.',
			'default': False,
		},
	}
	if modern:
		reset_properties = {'context_id': context_property, **reset_properties}
	reset_schema: dict[str, Any] = {
		'type': 'object',
		'properties': reset_properties,
		'additionalProperties': False,
	}
	if modern:
		reset_schema['required'] = ['context_id']

	execute = types.Tool(
		name='execute_code',
		description=description,
		input_schema=execute_schema,
		annotations=types.ToolAnnotations(
			read_only_hint=False,
			destructive_hint=True,
			open_world_hint=True,
			idempotent_hint=False,
		),
	)
	status = types.Tool(
		name='get_status',
		description=(
			'Return OpenBrowser server, browser, profile, and tab state as JSON. '
			'Read-only and does not start a browser or probe CDP health. Use this before retrying a failed task. '
			'When browser.cdp is present, cdp.healthy is the last accepted connect or probe observation '
			'while the lifecycle preserves an active claim; inactive snapshots can report null '
			'and keep the history fields. '
			'Use cdp.phase and cdp.last_error_code before retrying. '
			'cdp.phase is one of disconnected, resolving, connecting, connected, degraded, failed. '
			'cdp.last_error_code is null or not_connected, timeout, connection_closed, '
			'session_not_found, probe_failed, connect_failed, cancelled. '
			'session_not_found can come from target attachment during connect setup. '
			'Use cdp.last_probe_at to tell whether a probe ran. '
			'cdp_connected only means a client object is present. '
			'Root CDP health does not prove a target session, tab, or page can accept commands. '
			'When browser.health is present, phase is unknown, healthy, degraded, crashed, stopped, or failed. '
			'health is the last event-driven crash or setup observation plus a status-time local process check, '
			'not a live page probe. '
			'Use last_focus_recovery_outcome, target_crash_count, and process_present before retrying. '
			'Wait while recovery is pending. Reset only when no usable target exists. '
			'process_present is null when the published root is not bound to the recorded local owner, '
			'or when process identity is inconclusive. That includes remote, stopped, or ambiguous local sessions. '
			'A missing health block means no browser session exists.'
			+ (' Requires context_id from create_context.' if modern else '')
		),
		input_schema=status_schema,
		annotations=types.ToolAnnotations(
			read_only_hint=True,
			destructive_hint=False,
			open_world_hint=False,
			idempotent_hint=True,
		),
	)
	reset = types.Tool(
		name='reset_browser',
		description=(
			'Close the current browser session and release its profile. '
			'Scalar and scalar-only container variables are preserved unless clear_variables is true. '
			'Functions, modules, browser-bound objects, cyclic values, and names that shadow '
			'framework bindings are dropped and reported in namespace_transfer. '
			'The next execute_code call starts a fresh browser.'
			+ (' Requires context_id from create_context.' if modern else '')
		),
		input_schema=reset_schema,
		annotations=types.ToolAnnotations(
			read_only_hint=False,
			destructive_hint=True,
			open_world_hint=False,
			idempotent_hint=True,
		),
	)
	if not modern:
		return (execute, status, reset)

	create = types.Tool(
		name='create_context',
		description=(
			'Create an isolated OpenBrowser context and return its context_id. '
			'No browser starts until the first execute_code call. '
			'Call this once, then pass context_id to execute_code, get_status, and reset_browser. '
			'The context lives until close_context or server exit and cannot be resumed by a new server process. '
			'Close it with close_context when finished.'
		),
		input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
		output_schema={
			'type': 'object',
			'properties': {'context_id': context_property},
			'required': ['context_id'],
			'additionalProperties': False,
		},
		annotations=types.ToolAnnotations(
			read_only_hint=False,
			destructive_hint=False,
			open_world_hint=False,
			idempotent_hint=False,
		),
	)
	close = types.Tool(
		name='close_context',
		description=(
			'Close an OpenBrowser context and release its browser and profile. '
			'The context_id cannot be used after a successful close.'
		),
		input_schema={
			'type': 'object',
			'properties': {'context_id': context_property},
			'required': ['context_id'],
			'additionalProperties': False,
		},
		output_schema={
			'type': 'object',
			'properties': {
				'context_id': context_property,
				'status': {'type': 'string', 'enum': ['closed', 'close_failed']},
				'error_code': {'type': ['string', 'null']},
				'retryable': {'type': 'boolean'},
			},
			'required': ['context_id', 'status', 'retryable'],
			'additionalProperties': False,
		},
		annotations=types.ToolAnnotations(
			read_only_hint=False,
			destructive_hint=True,
			open_world_hint=False,
			idempotent_hint=True,
		),
	)
	return (create, execute, status, reset, close)


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
from openbrowser.code_use.namespace_transfer import (
	NamespaceTransfer,
	transfer_namespace,
)


class _BrowserOwner:
	"""Own the browser, namespace, profile, and lifecycle for one context."""

	def __init__(
		self,
		*,
		owner_id: str,
		config: dict[str, Any],
		session_timeout_minutes: int = 10,
		adapter: Any = None,
		workspace_dir: Path | None = None,
		downloads_dir: Path | None = None,
	):
		self.owner_id = owner_id
		self._adapter = adapter
		self.config = config
		self.browser_session: BrowserSession | None = None
		self._browser_lifecycle = LifecycleMachine(f'mcp_browser:{owner_id}')
		self._browser_generation = 0
		self._pending_variables: dict[str, Any] = {}
		self._last_namespace_transfer: dict[str, object] = {}
		self._profile_summary: dict[str, Any] | None = None
		self._managed_profile_dir: Path | None = None
		self._managed_storage_state: Path | None = None
		self._workspace_dir = workspace_dir or (_MCP_WORKSPACE_DIR / owner_id)
		self._downloads_dir = downloads_dir or (Path.home() / 'Downloads' / 'openbrowser-mcp' / owner_id)
		self._file_system: Any = None
		try:
			max_output = int(os.environ.get('OPENBROWSER_MAX_OUTPUT', '0'))
		except (ValueError, TypeError):
			max_output = 0
		self._executor = CodeExecutor(max_output_chars=max_output if max_output > 0 else DEFAULT_MAX_OUTPUT_CHARS)
		self._tools: CodeAgentTools | None = None
		self._tools_generation: int | None = None

		# Session management
		self.session_timeout_minutes = session_timeout_minutes
		self._last_activity = time.time()
		self._lifecycle_lock = asyncio.Lock()
		self._execution_lock = asyncio.Lock()
		self._execution_task: asyncio.Task[Any] | None = None

	def _carry_namespace(
		self,
		old_namespace: dict[str, Any] | None,
		old_session: Any,
	) -> NamespaceTransfer:
		"""Classify the old namespace before rebuilding the browser."""
		transfer = transfer_namespace(old_namespace or {}, old_browser_session=old_session)
		if transfer.warnings:
			logger.warning('Namespace transfer: %s', '; '.join(transfer.warnings))
		self._last_namespace_transfer = transfer.to_dict()
		return transfer

	def _merge_pending_variables(self, transfer: NamespaceTransfer) -> None:
		"""Merge carried values into the pending set, dropping internal names."""
		merged = {**self._pending_variables, **transfer.variables}
		self._pending_variables = {
			key: value for key, value in merged.items() if isinstance(key, str) and not key.startswith('_')
		}

	def _server(self) -> Any:
		"""Return the adapter that owns server-wide state, or this legacy owner."""
		return self._adapter if self._adapter is not None else self

	def _startup_config(self) -> Any:
		"""Return the server-level frozen config result."""
		return self._server()._startup_config_result

	@property
	def _namespace(self) -> dict[str, Any] | None:
		"""Compatibility view of the executor's single active namespace."""
		return self._executor.namespace

	@_namespace.setter
	def _namespace(self, namespace: dict[str, Any] | None) -> None:
		self._executor.set_namespace(namespace)

	def _create_owner_file_system(self) -> Any:
		"""Create one FileSystem per owner and reuse it across namespace rebuilds."""
		if not FILESYSTEM_AVAILABLE:
			return None
		if self._file_system is None:
			self._file_system = FileSystem(base_dir=str(self._workspace_dir), create_default_files=False)
		return self._file_system

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

	def _configured_storage_source(self) -> object:
		"""Resolve the configured storage source without starting a browser."""
		if self._profile_summary:
			mode = self._profile_summary.get('storage_state_mode')
			if mode == 'memory':
				return {}
			if mode == 'file':
				return self._profile_summary.get('storage_state')
			if mode == 'none':
				return None
			return self._profile_summary.get('storage_state')
		try:
			profile_config = dict(get_default_profile(self.config))
		except Exception:
			return None
		configured = profile_config.get('storage_state')
		if isinstance(configured, dict):
			return configured
		if configured is not None and not isinstance(configured, (str, Path)):
			return configured
		if profile_config.get('user_data_dir'):
			return configured
		return self._managed_storage_state_path_preview()

	def _managed_storage_state_path_preview(self) -> Path:
		"""Compute the managed state path without creating configuration directories."""
		config_dir_env = os.getenv('OPENBROWSER_CONFIG_DIR')
		if config_dir_env:
			config_dir = Path(config_dir_env).expanduser().resolve()
		else:
			config_dir = CONFIG.XDG_CONFIG_HOME / 'openbrowser'
		return config_dir / 'profiles' / f'mcp-{self.owner_id}' / 'storage_state.json'

	def _build_browser_profile(self):
		"""Build a BrowserProfile from config with MCP defaults."""
		from openbrowser.browser.profile_provenance import ProfileProvenance, ProfileValueSource

		profile_config = dict(get_default_profile(self.config))
		explicit_user_data_dir = profile_config.get('user_data_dir')
		defaults = {
			'downloads_path': str(self._downloads_dir),
			'wait_between_actions': 0.5,
			'keep_alive': True,
			'device_scale_factor': 1.0,
			'disable_security': False,
			'headless': True,
		}
		provenance = ProfileProvenance()
		provenance.record_mapping(defaults, ProfileValueSource.DEFAULT)
		provenance.record_mapping(profile_config, ProfileValueSource.CONFIG)
		provenance.record_defaults(
			{'channel': None, 'executable_path': None, 'args': [], 'proxy': None}
		)
		env_user_data = os.environ.get('OPENBROWSER_USER_DATA_DIR')
		if env_user_data and str(profile_config.get('user_data_dir') or '') == env_user_data:
			provenance.record(
				'user_data_dir', profile_config.get('user_data_dir'), ProfileValueSource.ENVIRONMENT
			)
		env_storage = os.environ.get('OPENBROWSER_STORAGE_STATE')
		if env_storage and str(profile_config.get('storage_state') or '') == env_storage:
			provenance.record(
				'storage_state', profile_config.get('storage_state'), ProfileValueSource.ENVIRONMENT
			)
		if isinstance(profile_config.get('proxy'), dict) and any(
			os.environ.get(name)
			for name in (
				'OPENBROWSER_PROXY_URL',
				'OPENBROWSER_NO_PROXY',
				'OPENBROWSER_PROXY_USERNAME',
				'OPENBROWSER_PROXY_PASSWORD',
			)
		):
			provenance.record('proxy', profile_config.get('proxy'), ProfileValueSource.ENVIRONMENT)
		profile_data = {**defaults, **profile_config}
		env_headless = os.environ.get('OPENBROWSER_HEADLESS', '').lower()
		if env_headless in ('true', '1', 'yes'):
			profile_data['headless'] = True
			provenance.record('headless', True, ProfileValueSource.ENVIRONMENT)
		elif env_headless in ('false', '0', 'no'):
			profile_data['headless'] = False
			provenance.record('headless', False, ProfileValueSource.ENVIRONMENT)

		if explicit_user_data_dir:
			# Explicit paths are intentionally shared and protected by the
			# profile lease. Never silently replace one with a temporary path.
			managed_default_dir = None
		else:
			managed_default_dir = self._get_managed_profile_dir()
			profile_data['user_data_dir'] = str(managed_default_dir)
			provenance.record('user_data_dir', profile_data['user_data_dir'], ProfileValueSource.MANAGED)
			configured_storage_state = profile_data.get('storage_state')
			if configured_storage_state is None or isinstance(configured_storage_state, (str, Path)):
				storage_state_path = managed_default_dir / 'storage_state.json'
				self._seed_managed_storage_state(storage_state_path, source=configured_storage_state)
				profile_data['storage_state'] = str(storage_state_path)
				provenance.record(
					'storage_state', profile_data['storage_state'], ProfileValueSource.MANAGED
				)
			else:
				# In-memory Playwright storage state is already isolated per profile
				# object. Do not coerce it into a shared filesystem path.
				profile_data['storage_state'] = deepcopy(configured_storage_state)
				provenance.record(
					'storage_state', profile_data['storage_state'], ProfileValueSource.CONFIG
				)

		before_managed = dict(profile_data)
		profile_data = apply_managed_browser_profile_defaults(profile_data, managed_default_dir)
		provenance.record_changes(before_managed, profile_data, ProfileValueSource.MANAGED)
		if profile_data.get('storage_state') is None:
			provenance.record_defaults({'storage_state': None})
		args = list(profile_data.get('args') or [])
		ownership_marker = f'--openbrowser-instance-id={self.owner_id}'
		if not any(argument.startswith('--openbrowser-instance-id=') for argument in args):
			args.append(ownership_marker)
			provenance.record('args', args, ProfileValueSource.MANAGED)
		profile_data['args'] = args
		profile = BrowserProfile(**profile_data)
		provenance.refresh_from_profile(profile)
		storage_state = profile.storage_state
		if isinstance(storage_state, (str, Path)):
			storage_state_mode = 'file'
			storage_state_summary = str(storage_state)
		elif isinstance(storage_state, dict):
			storage_state_mode = 'memory'
			storage_state_summary = None
		else:
			storage_state_mode = 'none'
			storage_state_summary = None
		self._profile_summary = {
			'user_data_dir': str(profile.user_data_dir) if profile.user_data_dir else None,
			'headless': profile.headless,
			'storage_state': storage_state_summary,
			'storage_state_mode': storage_state_mode,
			'instance_id': self.owner_id,
			'resolution': provenance.to_dict(),
		}
		return profile

	def _get_managed_profile_dir(self) -> Path:
		"""Return the stable isolated profile directory for this MCP instance."""
		if self._managed_profile_dir is None:
			self._managed_profile_dir = CONFIG.OPENBROWSER_PROFILES_DIR / f'mcp-{self.owner_id}'
		self._managed_profile_dir.mkdir(parents=True, exist_ok=True)
		return self._managed_profile_dir

	def _seed_managed_storage_state(self, destination: Path, source: str | Path | None = None) -> None:
		"""Copy configured or default state once into this instance's profile."""
		if destination.exists():
			self._managed_storage_state = destination
			return

		source_path = (
			Path(source).expanduser() if source else CONFIG.OPENBROWSER_PROFILES_DIR / 'default' / 'storage_state.json'
		)
		if source_path.is_file():
			destination.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(source_path, destination)
		self._managed_storage_state = destination

	async def _ensure_namespace(self, *, start_timeout: float | None = None):
		"""Lazily initialize browser session, tools, and namespace on first use."""
		if self._namespace is not None:
			return

		async with self._lifecycle_lock:
			if self._namespace is not None:
				return
			self._tools = None
			self._tools_generation = None
			if self.browser_session is not None:
				raise RuntimeError('A previous browser session still owns resources after cleanup failed')

			CONFIG.require_browser_start_allowed(self._startup_config())
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
				await asyncio.wait_for(
					session.start(),
					timeout=start_timeout if start_timeout is not None else _start_timeout_seconds(),
				)
				session.seed_generation(self._browser_generation)

				# Create CodeAgent tools and namespace
				self._tools = CodeAgentTools()
				self._tools_generation = self._browser_generation
				self._namespace = create_namespace(
					browser_session=self.browser_session,
					tools=self._tools,
					file_system=self._create_owner_file_system(),
				)
				for key, value in self._pending_variables.items():
					if not key.startswith('__') and key not in self._namespace:
						self._namespace[key] = value
				self._pending_variables.clear()
				self._browser_lifecycle.transition(LifecyclePhase.READY, reason='browser ready')
			except BaseException as error:
				from openbrowser.browser.cdp_connection import sanitize_error_detail

				message = f'{type(error).__name__}: {sanitize_error_detail(error) or type(error).__name__}'
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
						self._tools = None
						self._tools_generation = None
					if cleanup.caller_cancellation is not None and not isinstance(error, asyncio.CancelledError):
						raise cleanup.caller_cancellation
				raise

	async def _is_cdp_alive(self, *, deadline: float | None = None) -> bool:
		"""Check if the browser session's CDP WebSocket is still connected."""
		if not self.browser_session:
			return False
		timeout = _CDP_HEALTH_CHECK_TIMEOUT_SECONDS
		if deadline is not None:
			remaining = deadline - time.monotonic()
			if remaining <= 0:
				return False
			timeout = min(timeout, remaining)
		probe = getattr(self.browser_session, 'probe_cdp_connection', None)
		if not inspect.iscoroutinefunction(probe):
			return False
		try:
			health = await probe(timeout_seconds=timeout)
		except asyncio.CancelledError:
			raise
		except Exception:
			return False
		return getattr(health, 'client_present', False) is True and getattr(health, 'healthy', False) is True

	async def _kill_session_safely(self, session: BrowserSession) -> _CleanupOutcome:
		"""Complete owned cleanup without confusing worker and caller cancellation."""

		async def owned_cleanup() -> bool:
			try:
				timeout = _shutdown_timeout_seconds()
				result = session.kill()
				if inspect.isawaitable(result):
					await asyncio.wait_for(result, timeout=timeout)
				else:
					from openbrowser.browser.events import BrowserStopEvent

					event = session.event_bus.dispatch(BrowserStopEvent(force=True))
					if inspect.isawaitable(event):
						await asyncio.wait_for(event, timeout=timeout)
					event_result = getattr(event, 'event_result', None)
					if callable(event_result):
						result = event_result(raise_if_any=True, raise_if_none=False)
						if inspect.isawaitable(result):
							await asyncio.wait_for(result, timeout=timeout)
				return True
			except BaseException:
				return False

		cleanup_task = asyncio.create_task(owned_cleanup())
		completion = asyncio.gather(cleanup_task, return_exceptions=True)
		caller_cancellation = await _drain_owned_completion(completion)
		result = completion.result()[0]
		return _CleanupOutcome(result is True, caller_cancellation)

	async def _recover_browser_session(self, *, deadline: float | None = None) -> None:
		"""Serialize recovery with execution and lifecycle cleanup."""
		if self._execution_task is asyncio.current_task():
			await self._recover_browser_session_locked(deadline=deadline)
			return

		async with self._execution_lock:
			await self._recover_browser_session_locked(deadline=deadline)

	async def _recover_browser_session_locked(self, *, deadline: float | None = None) -> None:
		"""Kill the dead browser session and create a fresh one.

		Called when we detect the CDP WebSocket has disconnected (e.g. Chrome
		crashed or a navigation broke the connection).  Re-creates the
		BrowserSession and rebuilds the namespace so the next execute_code
		call works transparently.
		"""
		async with self._lifecycle_lock:
			if self.browser_session and await self._is_cdp_alive(deadline=deadline):
				return

			CONFIG.require_browser_start_allowed(self._startup_config())
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
					self._browser_lifecycle.fail(
						'Unable to safely close the old browser session during recovery',
						reason='browser recovery cleanup failed',
					)
				if cleanup.caller_cancellation is not None:
					if cleanup.succeeded:
						self._merge_pending_variables(self._carry_namespace(old_ns, old_session))
						try:
							self._browser_lifecycle.transition(
								LifecyclePhase.STOPPED,
								reason='browser recovery interrupted',
								increment_generation=True,
							)
						except LifecycleTransitionError:
							pass
					raise cleanup.caller_cancellation
				if not cleanup.succeeded:
					raise RuntimeError('Unable to safely close the old browser session during recovery')
			else:
				self._namespace = None

			if deadline is not None and time.monotonic() >= deadline:
				self._merge_pending_variables(self._carry_namespace(old_ns, old_session))
				self._browser_generation += 1
				try:
					self._browser_lifecycle.transition(
						LifecyclePhase.STOPPED,
						reason='recovery deadline exhausted',
						increment_generation=True,
					)
				except LifecycleTransitionError:
					pass
				raise TimeoutError('Browser recovery exceeded the execution deadline')

			# 2. Create a brand-new session. The local watchdog owns its profile
			# lease and can reclaim only its recorded browser process.
			try:
				profile = self._build_browser_profile()
				session = BrowserSession(browser_profile=profile)
			except BaseException as error:
				from openbrowser.browser.cdp_connection import sanitize_error_detail

				safe_error = f'{type(error).__name__}: {sanitize_error_detail(error) or type(error).__name__}'
				self._browser_lifecycle.fail(safe_error, reason='browser recovery failed')
				self.browser_session = None
				self._namespace = None
				self._merge_pending_variables(self._carry_namespace(old_ns, old_session))
				raise

			self.browser_session = session
			try:
				start_timeout = (
					min(_start_timeout_seconds(), max(0.1, deadline - time.monotonic()))
					if deadline is not None
					else _start_timeout_seconds()
				)
				await asyncio.wait_for(session.start(), timeout=start_timeout)
				self._browser_generation += 1
				session.seed_generation(self._browser_generation)

				# 3. Rebuild namespace with the new session (preserving user variables)
				self._tools = CodeAgentTools()
				self._tools_generation = self._browser_generation
				self._namespace = create_namespace(
					browser_session=self.browser_session,
					tools=self._tools,
					file_system=self._create_owner_file_system(),
				)
				# Carry only scalars and scalar containers from the old namespace.
				transfer = self._carry_namespace(old_ns, old_session)
				for key, value in transfer.variables.items():
					if key not in self._namespace:
						self._namespace[key] = value

				self._executor.set_namespace(self._namespace)
				self._browser_lifecycle.transition(
					LifecyclePhase.READY,
					reason='browser recovered',
					increment_generation=True,
				)
			except BaseException as error:
				from openbrowser.browser.cdp_connection import sanitize_error_detail

				safe_error = f'{type(error).__name__}: {sanitize_error_detail(error) or type(error).__name__}'
				self._browser_lifecycle.fail(safe_error, reason='browser recovery failed')
				cleanup = await self._kill_session_safely(session)
				if not cleanup.succeeded:
					logger.error('Failed to clean up replacement browser session')
				if cleanup.succeeded and self.browser_session is session:
					self.browser_session = None
				self._namespace = None
				# The next start must not lose ordinary user values because this
				# recovery failed after the old session was torn down.
				self._merge_pending_variables(self._carry_namespace(old_ns, old_session))
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
		"""Serialize execution with browser lifecycle operations and stdout capture."""
		server = self._server()
		loop = asyncio.get_running_loop()
		if server._output_lock is None or server._output_lock_loop is not loop:
			server._output_lock = _process_output_lock()
			server._output_lock_loop = loop
		timeout = _exec_timeout_seconds()
		deadline = time.monotonic() + timeout

		async def acquire(lock: asyncio.Lock) -> bool:
			remaining = deadline - time.monotonic()
			if remaining <= 0:
				return False
			try:
				await asyncio.wait_for(lock.acquire(), timeout=remaining)
				return True
			except TimeoutError:
				return False

		if not await acquire(self._execution_lock):
			return (
				f'Error: Execution timed out after {timeout:g}s while waiting for the context execution lock. '
				'The code may still be running. Run get_status before retrying.'
			)
		try:
			self._execution_task = asyncio.current_task()
			try:
				if not await acquire(server._output_lock):
					return (
						f'Error: Execution timed out after {timeout:g}s while waiting for the shared output lock. '
						'Another context is still executing. Run get_status before retrying.'
					)
				try:
					return await self._execute_code_locked(code, deadline=deadline)
				finally:
					server._output_lock.release()
			finally:
				self._last_activity = time.time()
				self._execution_task = None
		finally:
			self._execution_lock.release()

	async def _execute_code_locked(self, code: str, *, deadline: float | None = None) -> str:
		"""Execute Python code in the persistent namespace."""
		now = time.monotonic()
		if deadline is not None and now >= deadline:
			return 'Error: Execution timed out before browser startup. Run get_status before retrying.'
		start_timeout = min(_start_timeout_seconds(), deadline - now) if deadline is not None else None
		try:
			await self._ensure_namespace(start_timeout=start_timeout)
		except TimeoutError:
			return 'Error: Browser startup timed out before the code could run. Run get_status before retrying.'
		assert self._namespace is not None

		# Keep executor in sync (namespace may be set externally, e.g. tests)
		if not self._executor.initialized:
			self._executor.set_namespace(self._namespace)

		self._last_activity = time.time()

		# Pre-flight: check if CDP is still alive and recover if needed
		# Only attempt recovery when we already had a browser session (i.e. it died).
		# If browser_session is None, _ensure_namespace hasn't launched one yet
		# or the code doesn't need a browser -- skip recovery to avoid hanging.
		if self.browser_session and not await self._is_cdp_alive(deadline=deadline):
			if deadline is not None and time.monotonic() >= deadline:
				return 'Error: Execution timed out during the browser health check. Run get_status before retrying.'
			try:
				await self._recover_browser_session(deadline=deadline)
			except ConfigStartupError:
				raise
			except Exception as recovery_err:
				from openbrowser.browser.cdp_connection import sanitize_error_detail

				logger.error('Pre-flight CDP recovery failed: %s', recovery_err)
				safe_error = sanitize_error_detail(recovery_err) or type(recovery_err).__name__
				return (
					'Error: Browser connection lost and recovery failed: '
					f'{type(recovery_err).__name__}: {safe_error}\n\n'
					'Run get_status, then reset_browser before retrying.'
				)
		if deadline is not None and time.monotonic() >= deadline:
			return 'Error: Execution timed out after browser recovery. Run get_status before retrying.'

		# Execute via shared CodeExecutor with a bound that leaves room for the
		# calling agent to receive the timeout message before its own timeout.
		try:
			remaining = deadline - time.monotonic() if deadline is not None else _exec_timeout_seconds()
			if remaining <= 0:
				return 'Error: Execution timed out before user code could run. Run get_status before retrying.'
			result = await asyncio.wait_for(
				self._executor.execute(code),
				timeout=remaining,
			)
		except TimeoutError:
			return (
				f'Error: Execution timed out after {_exec_timeout_seconds():g}s. '
				'The code may still be running. Run get_status before retrying.'
			)

		# If the error looks like a CDP connection issue, recover the browser
		# but do not run the code again. The first attempt may have changed
		# the page before the connection failed.
		error_text = result.error or result.output
		if not result.success:
			from openbrowser.browser.cdp_connection import sanitize_error_detail

			output_limit = output_limit_for_executor(self._executor)
			safe_output = sanitize_error_detail(
				result.output, max_chars=output_limit, preserve_lines=True
			)
			safe_error = sanitize_error_detail(
				result.error, max_chars=output_limit, preserve_lines=True
			)
			safe_output = safe_output or safe_error or 'Execution failed without diagnostic text'
		else:
			safe_output = result.output
		if not result.success and self._is_connection_error(error_text):
			logger.info('CDP connection error during execution, recovering')
			try:
				await self._recover_browser_session()
			except ConfigStartupError as error:
				from openbrowser.browser.cdp_connection import (
					MAX_ERROR_CHARS,
					bound_text_with_suffix,
					sanitize_error_detail,
				)

				recovery_budget = max(1, output_limit // 4)
				recovery_detail = sanitize_error_detail(
					error.detail, max_chars=recovery_budget, preserve_lines=True
				)
				if recovery_detail:
					partial_budget = max(0, output_limit - len(recovery_detail) - 1)
					partial_detail = (
						sanitize_error_detail(
							result.output or result.error,
							max_chars=partial_budget,
							preserve_lines=True,
						)
						if partial_budget > 0
						else None
					)
					combined_detail = (
						f'{partial_detail}\n{recovery_detail}' if partial_detail else recovery_detail
					)
				else:
					combined_detail = sanitize_error_detail(
						result.output or result.error,
						max_chars=output_limit,
						preserve_lines=True,
					) or 'Execution failed without diagnostic text'
				safe_hint = bound_text_with_suffix(
					error.hint,
					_PARTIAL_RUN_WARNING,
					max_chars=MAX_ERROR_CHARS,
				)
				raise ConfigStartupError(
					error.snapshot,
					code=error.code,
					hint=safe_hint or error.code,
					detail=combined_detail or None,
				) from error
			except Exception as recovery_err:
				from openbrowser.browser.cdp_connection import sanitize_error_detail

				logger.error('CDP recovery failed: %s', recovery_err)
				safe_error = sanitize_error_detail(recovery_err) or type(recovery_err).__name__
				return (
					f'{safe_output}\n\n[openbrowser] Browser recovery failed: '
					f'{type(recovery_err).__name__}: {safe_error}. {_PARTIAL_RUN_WARNING}'
				)
			return (
				f'{safe_output}\n\n[openbrowser] Browser was recovered. '
				'The code may have partially run. Run get_status before retrying.'
			)

		return safe_output

	def _downloads_snapshot(self) -> dict[str, Any]:
		"""Return a bounded download outcome summary without starting the browser."""
		return downloads_summary(self.browser_session)

	def _status_payload(self) -> dict[str, Any]:
		"""Build a JSON-safe status snapshot without starting the browser."""
		from openbrowser.config_lifecycle import restart_required
		from openbrowser.tools.action_availability import tool_availability_payload

		browser = self._browser_lifecycle.snapshot().to_dict()
		if self.browser_session is not None and hasattr(self.browser_session, 'lifecycle_snapshot'):
			try:
				snapshot = self.browser_session.lifecycle_snapshot()
				if isinstance(snapshot, dict):
					browser = snapshot
			except Exception:
				health_snapshot = getattr(self.browser_session, 'cdp_connection_health', None)
				if callable(health_snapshot):
					with contextlib.suppress(Exception):
						health_value = health_snapshot()
						to_dict = getattr(health_value, 'to_dict', None)
						if callable(to_dict):
							browser['cdp'] = to_dict()
							browser['cdp_connected'] = bool(getattr(health_value, 'client_present', False))
				focus_snapshot = getattr(self.browser_session, 'focus_snapshot', None)
				if callable(focus_snapshot):
					with contextlib.suppress(Exception):
						focus_value = focus_snapshot()
						to_dict = getattr(focus_value, 'to_dict', None)
						if callable(to_dict):
							browser['focus'] = to_dict()
				browser_health = getattr(self.browser_session, 'browser_health_snapshot', None)
				if callable(browser_health):
					with contextlib.suppress(Exception):
						health_value = browser_health()
						if isinstance(health_value, dict):
							browser['health'] = health_value
		if 'focus' not in browser:
			from openbrowser.browser.focus import FocusSnapshot

			browser['focus'] = FocusSnapshot.uninitialized().to_dict()

		namespace_variables = 0
		if self._namespace is not None:
			namespace_variables = sum(1 for key in self._namespace if not key.startswith('__'))

		configured_storage_state = self._configured_storage_source()
		if 'storage_state' in browser and 'storage_state_save' in browser:
			storage_state = browser['storage_state']
			storage_state_save = browser['storage_state_save']
		else:
			from openbrowser.browser.storage_state import (
				load_snapshot_for_configuration,
				save_snapshot_for_configuration,
			)

			storage_state = load_snapshot_for_configuration(configured_storage_state).to_dict()
			storage_state_save = save_snapshot_for_configuration(configured_storage_state).to_dict()

		try:
			version = get_openbrowser_version()
		except Exception:
			version = None

		tools_for_status = (
			self._tools
			if (
				self._tools is not None
				and self._tools_generation == self._browser_generation
				and self._namespace is not None
				and self.browser_session is not None
			)
			else None
		)
		tools_snapshot = tool_availability_payload(tools_for_status, browser)
		startup_config = self._startup_config().snapshot
		try:
			current_config = CONFIG.load_result().snapshot
			needs_restart = restart_required(startup_config, current_config)
		except Exception:
			current_config = startup_config
			needs_restart = True

		return {
			'server': self._server()._lifecycle.snapshot().to_dict(),
			'browser': browser,
			'profile': self._profile_summary,
			'instance_id': self._server()._instance_id,
			'context_id': getattr(self, 'context_id', None),
			'namespace_variables': namespace_variables,
			'pending_variables': len(self._pending_variables),
			'namespace_transfer': self._last_namespace_transfer,
			'session_timeout_minutes': self.session_timeout_minutes,
			'exec_timeout': _exec_timeout_seconds(),
			'uptime_seconds': max(0.0, time.time() - self._server()._start_time),
			'version': version,
			'tools': tools_snapshot,
			'downloads': self._downloads_snapshot(),
			'storage_state': storage_state,
			'storage_state_save': storage_state_save,
			'config': {
				'startup': startup_config.to_dict(),
				'current': current_config.to_dict(),
				'restart_required': needs_restart,
			},
		}

	async def _reset_browser(self, *, clear_variables: bool = False) -> str:
		"""Close the current browser and preserve scalar user variables by default."""
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
				self._tools = None
				self._tools_generation = None
				if clear_variables:
					self._pending_variables.clear()
					self._last_namespace_transfer = NamespaceTransfer().to_dict()
				else:
					self._merge_pending_variables(self._carry_namespace(old_namespace, session))
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
						'namespace_transfer': self._last_namespace_transfer,
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
		caller_cancellation = await _drain_owned_completion(completion)
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

	async def _close_owned(self) -> None:
		"""Run owner cleanup as an owned worker and drain caller cancellation."""

		async def owned_cleanup() -> None:
			await self._shutdown_owner_locked()

		task = asyncio.create_task(owned_cleanup())
		completion = asyncio.gather(task, return_exceptions=True)
		caller_cancellation = await _drain_owned_completion(completion)
		result = completion.result()[0]
		if caller_cancellation is not None:
			raise caller_cancellation
		if isinstance(result, BaseException):
			raise result

	async def _shutdown_locked(self) -> None:
		"""Subclasses provide their own shutdown boundary."""
		raise NotImplementedError

	async def _shutdown_owner_locked(self) -> None:
		"""Close one owner's browser and namespace while its caller is isolated."""
		async with self._execution_lock:
			async with self._lifecycle_lock:
				session = self.browser_session
				if not session:
					self._namespace = None
					self._tools = None
					self._tools_generation = None
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
					self._tools = None
					self._tools_generation = None
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
						self._tools = None
						self._tools_generation = None
						self._merge_pending_variables(self._carry_namespace(old_namespace, session))
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
						self._browser_lifecycle.fail(
							'Unable to close the idle browser session', reason='idle cleanup failed'
						)
						logger.error('Error closing idle session: cleanup failed')
					if cleanup.caller_cancellation is not None:
						raise cleanup.caller_cancellation


class ContextPhase(str, Enum):
	"""The explicit lifecycle of one modern MCP application context."""

	ACTIVE = 'active'
	CLOSING = 'closing'
	CLOSE_FAILED = 'close_failed'
	CLOSED = 'closed'


class ServerAdmission(str, Enum):
	"""Whether the server may admit new context work."""

	RUNNING = 'running'
	STOPPING = 'stopping'
	STOPPED = 'stopped'
	FAILED = 'failed'


@dataclass(frozen=True)
class CloseResult:
	"""Typed outcome of one close attempt, keyed by its context."""

	context_id: str
	status: str
	error_code: str | None = None
	retryable: bool = False

	def __post_init__(self) -> None:
		if self.status == 'closed' and (self.error_code is not None or self.retryable):
			raise ValueError('A closed result has no error and is not retryable')
		if self.status == 'close_failed' and (self.error_code is None or not self.retryable):
			raise ValueError('A close_failed result needs an error code and is retryable')
		if self.status not in {'closed', 'close_failed'}:
			raise ValueError(f'Unknown close status: {self.status}')


class ContextUnavailable(RuntimeError):
	"""A stable error code for a context that cannot be used."""

	def __init__(self, code: str, message: str):
		self.code = code
		super().__init__(message)


@dataclass
class ContextEntry:
	"""Registry state for one context, including its in-flight lease count."""

	context_id: str
	owner: Any
	phase: ContextPhase = ContextPhase.ACTIVE
	active_calls: int = 0
	close_task: asyncio.Task[CloseResult] | None = None
	drained: asyncio.Event = field(default_factory=asyncio.Event)
	drain_waiting: asyncio.Event = field(default_factory=asyncio.Event)

	def __post_init__(self) -> None:
		self.drained.set()


class ContextLease:
	"""One admitted tool call against a context."""

	def __init__(self, registry: Any, entry: ContextEntry):
		self._registry = registry
		self.entry = entry

	@property
	def owner(self) -> Any:
		return self.entry.owner

	async def __aenter__(self) -> Any:
		return self.entry.owner

	async def __aexit__(self, *_: object) -> None:
		self._registry.release(self.entry)


class OpenBrowserContext(_BrowserOwner):
	"""One isolated browser and Python namespace selected by an explicit handle."""

	def __init__(self, server: Any, context_id: str):
		owner_id = uuid.uuid4().hex
		super().__init__(
			owner_id=owner_id,
			config=server.config,
			session_timeout_minutes=server.session_timeout_minutes,
			adapter=server,
			workspace_dir=_MCP_WORKSPACE_DIR / owner_id,
			downloads_dir=Path.home() / 'Downloads' / 'openbrowser-mcp' / owner_id,
		)
		self.context_id = context_id

	async def _shutdown_locked(self) -> None:
		await self._shutdown_owner_locked()


class ContextRegistry:
	"""Admission, lookup, leasing, and close serialization for application contexts."""

	def __init__(self, server: Any):
		self._server = server
		self._lock = asyncio.Lock()
		self._entries: dict[str, ContextEntry] = {}
		self._legacy_entry: ContextEntry | None = None
		self._admission = ServerAdmission.RUNNING
		self.max_contexts = _max_contexts()

	@property
	def admission(self) -> ServerAdmission:
		return self._admission

	def _open_count_unlocked(self) -> int:
		return len(self._entries) + (1 if self._legacy_entry is not None else 0)

	def snapshot(self) -> list[ContextEntry]:
		entries = list(self._entries.values())
		if self._legacy_entry is not None:
			entries.append(self._legacy_entry)
		return entries

	async def create(self) -> str:
		async with self._lock:
			if self._admission is not ServerAdmission.RUNNING:
				raise ContextUnavailable('server_stopping', 'Server is shutting down')
			if self._open_count_unlocked() >= self.max_contexts:
				raise ContextUnavailable(
					'context_limit_reached',
					f'At most {self.max_contexts} contexts may be open at once',
				)
			context_id = uuid.uuid4().hex
			owner = OpenBrowserContext(self._server, context_id)
			self._entries[context_id] = ContextEntry(context_id=context_id, owner=owner)
			return context_id

	async def acquire(self, context_id: str) -> ContextLease:
		async with self._lock:
			if self._admission is not ServerAdmission.RUNNING:
				raise ContextUnavailable('server_stopping', 'Server is shutting down')
			entry = self._entries.get(context_id)
			if entry is None:
				raise ContextUnavailable('context_not_found', f'No open context: {context_id}')
			if entry.phase is ContextPhase.CLOSING:
				raise ContextUnavailable('context_closing', f'Context is closing: {context_id}')
			if entry.phase is ContextPhase.CLOSE_FAILED:
				raise ContextUnavailable('context_close_failed', f'Context close failed: {context_id}')
			if entry.phase is not ContextPhase.ACTIVE:
				raise ContextUnavailable('context_not_found', f'No open context: {context_id}')
			entry.active_calls += 1
			entry.drained.clear()
			return ContextLease(self, entry)

	async def acquire_legacy(self) -> ContextLease:
		async with self._lock:
			if self._admission is not ServerAdmission.RUNNING:
				raise ContextUnavailable('server_stopping', 'Server is shutting down')
			if self._legacy_entry is None:
				if self._open_count_unlocked() >= self.max_contexts:
					raise ContextUnavailable(
						'context_limit_reached',
						f'At most {self.max_contexts} contexts may be open at once',
					)
				self._legacy_entry = ContextEntry(context_id='__legacy__', owner=self._server)
			entry = self._legacy_entry
			entry.active_calls += 1
			entry.drained.clear()
			return ContextLease(self, entry)

	def release(self, entry: ContextEntry) -> None:
		"""Release one lease atomically on the event-loop thread."""
		if entry.active_calls > 0:
			entry.active_calls -= 1
		if entry.active_calls == 0:
			entry.drained.set()

	async def _close_entry_task(self, entry: ContextEntry) -> asyncio.Task[CloseResult]:
		async with self._lock:
			if entry.phase is ContextPhase.CLOSED:
				raise ContextUnavailable('context_not_found', f'No open context: {entry.context_id}')
			if entry.close_task is None:
				entry.phase = ContextPhase.CLOSING
				entry.drain_waiting.clear()
				entry.close_task = asyncio.create_task(self._close_entry(entry))
			return entry.close_task

	async def close(self, context_id: str) -> CloseResult:
		async with self._lock:
			entry = self._entries.get(context_id)
		if entry is None:
			raise ContextUnavailable('context_not_found', f'No open context: {context_id}')
		task = await self._close_entry_task(entry)
		return await self._join_close(task)

	async def _join_close(self, task: asyncio.Task[CloseResult]) -> CloseResult:
		completion = asyncio.gather(task, return_exceptions=True)
		caller_cancellation = await _drain_owned_completion(completion)
		result = completion.result()[0]
		if caller_cancellation is not None:
			raise caller_cancellation
		if isinstance(result, BaseException):
			raise result
		return result

	async def _close_entry(self, entry: ContextEntry) -> CloseResult:
		try:
			entry.drain_waiting.set()
			await entry.drained.wait()
			await entry.owner._close_owned()
			result = CloseResult(context_id=entry.context_id, status='closed', error_code=None, retryable=False)
		except asyncio.CancelledError:
			logger.error('Context cleanup was cancelled by its own worker: %s', entry.context_id)
			result = CloseResult(
				context_id=entry.context_id, status='close_failed', error_code='cleanup_cancelled', retryable=True
			)
		except Exception as error:
			logger.error('Context cleanup failed for %s: %s', entry.context_id, error, exc_info=True)
			result = CloseResult(
				context_id=entry.context_id, status='close_failed', error_code='cleanup_failed', retryable=True
			)
		async with self._lock:
			if result.status == 'closed':
				entry.phase = ContextPhase.CLOSED
				self._entries.pop(entry.context_id, None)
				if self._legacy_entry is entry:
					self._legacy_entry = None
			else:
				entry.phase = ContextPhase.CLOSE_FAILED
			entry.close_task = None
		return result

	async def stop_admission(self) -> None:
		async with self._lock:
			self._admission = ServerAdmission.STOPPING

	async def set_admission(self, phase: ServerAdmission) -> None:
		async with self._lock:
			self._admission = phase

	async def mark_legacy_closed(self) -> None:
		"""Retire a legacy entry after a direct successful cleanup retry."""
		async with self._lock:
			if self._legacy_entry is not None:
				self._legacy_entry.phase = ContextPhase.CLOSED
				self._legacy_entry.close_task = None
				self._legacy_entry = None

	async def close_all(self) -> list[CloseResult]:
		async with self._lock:
			self._admission = ServerAdmission.STOPPING
			entries = list(self._entries.values())
			if self._legacy_entry is not None:
				entries.append(self._legacy_entry)
		results: list[CloseResult] = []
		for entry in entries:
			try:
				task = await self._close_entry_task(entry)
				results.append(await self._join_close(task))
			except ContextUnavailable:
				continue
		async with self._lock:
			self._admission = (
				ServerAdmission.FAILED
				if any(result.status != 'closed' for result in results)
				else ServerAdmission.STOPPED
			)
		return results


class OpenBrowserServer(_BrowserOwner):
	"""MCP adapter and registry, plus the exact legacy no-handle owner."""

	def __init__(self, session_timeout_minutes: int = 10):
		_ensure_all_loggers_use_stderr()

		self._startup_config_result = CONFIG.load_result()
		config = CONFIG.config_dict_from_result(self._startup_config_result)
		self._instance_id = uuid.uuid4().hex
		self._lifecycle = LifecycleMachine('mcp_server')
		self._telemetry = ProductTelemetry() if TELEMETRY_AVAILABLE else None
		self._start_time = time.time()
		self._output_lock: asyncio.Lock | None = None
		self._output_lock_loop: Any = None
		self._cleanup_task: Any = None
		super().__init__(
			owner_id=self._instance_id,
			config=config,
			session_timeout_minutes=session_timeout_minutes,
			adapter=None,
		)
		self._contexts = ContextRegistry(self)
		self._setup_handlers()

	def _protocol_era(self, version: str | None) -> str:
		if version is None:
			return 'legacy'
		if version == MODERN_PROTOCOL_VERSION:
			return 'modern'
		if version in LEGACY_PROTOCOL_VERSIONS:
			return 'legacy'
		raise MCPError(code=-32602, message=f'Unsupported protocol version: {version}')

	def _setup_handlers(self):
		"""Configure SDK callbacks from explicit legacy and modern catalogs."""

		description = _EXECUTE_CODE_DESCRIPTION
		if os.environ.get('OPENBROWSER_COMPACT_DESCRIPTION', '').lower() in ('1', 'true', 'yes'):
			description = _EXECUTE_CODE_DESCRIPTION_COMPACT
		self._legacy_tools = _tool_catalog(description, modern=False)
		self._modern_tools = _tool_catalog(description, modern=True)
		self._legacy_by_name = {tool.name: tool for tool in self._legacy_tools}
		self._modern_by_name = {tool.name: tool for tool in self._modern_tools}
		self._legacy_validators = {tool.name: Draft202012Validator(tool.input_schema) for tool in self._legacy_tools}
		self._modern_validators = {tool.name: Draft202012Validator(tool.input_schema) for tool in self._modern_tools}
		for tool in (*self._legacy_tools, *self._modern_tools):
			Draft202012Validator.check_schema(tool.input_schema)
			if tool.output_schema is not None:
				Draft202012Validator.check_schema(tool.output_schema)

		async def handle_list_tools(
			ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
		) -> types.ListToolsResult:
			era = self._protocol_era(getattr(ctx, 'protocol_version', None))
			catalog = self._modern_tools if era == 'modern' else self._legacy_tools
			return types.ListToolsResult(tools=[tool.model_copy(deep=True) for tool in catalog])

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
			telemetry_error = None
			telemetry_tool_name = 'unknown_tool'
			try:
				era = self._protocol_era(getattr(ctx, 'protocol_version', None))
				catalog = self._modern_tools if era == 'modern' else self._legacy_tools
				by_name = self._modern_by_name if era == 'modern' else self._legacy_by_name
				validators = self._modern_validators if era == 'modern' else self._legacy_validators
				if params.name not in by_name:
					telemetry_error = 'unknown_tool'
					raise MCPError(code=-32602, message=f'Unknown tool: {params.name}')
				telemetry_tool_name = params.name
				arguments = dict(params.arguments or {})
				validation_error = next(validators[params.name].iter_errors(arguments), None)
				if validation_error is not None:
					location = (
						f' at {validation_error.json_path}'
						if validation_error.json_path and validation_error.json_path != '$'
						else ''
					)
					error_msg = f'Invalid arguments{location}: {validation_error.message}'
					# The validator message can contain a repr of user data.
					# Keep it client-facing, never in outbound telemetry.
					telemetry_error = 'invalid_tool_arguments'
					return types.CallToolResult(content=[types.TextContent(type='text', text=error_msg)], is_error=True)
				if params.name == 'execute_code' and not arguments['code'].strip():
					telemetry_error = 'empty_code'
					error_msg = 'Error: No code provided'
					return types.CallToolResult(content=[types.TextContent(type='text', text=error_msg)], is_error=True)

				if era == 'modern':
					if params.name == 'create_context':
						try:
							return await self._create_context_result()
						except ContextUnavailable as error:
							telemetry_error = error.code
							error_msg = f'Error: {error.code}: {error}'
							return types.CallToolResult(
								content=[types.TextContent(type='text', text=error_msg)], is_error=True
							)
					if params.name == 'close_context':
						try:
							return await self._close_context_result(arguments['context_id'])
						except ContextUnavailable as error:
							telemetry_error = error.code
							error_msg = f'Error: {error.code}: {error}'
							return types.CallToolResult(
								content=[types.TextContent(type='text', text=error_msg)], is_error=True
							)
					context_id = arguments.pop('context_id')
					try:
						lease = await self._contexts.acquire(context_id)
					except ContextUnavailable as error:
						telemetry_error = error.code
						error_msg = f'Error: {error.code}: {error}'
						return types.CallToolResult(
							content=[types.TextContent(type='text', text=error_msg)], is_error=True
						)
					try:
						text, is_error = await self._run_owned_tool(lease.owner, params.name, arguments)
					finally:
						self._contexts.release(lease.entry)
				else:
					if params.name in ('create_context', 'close_context'):
						telemetry_error = 'unknown_tool'
						raise MCPError(code=-32602, message=f'Unknown tool: {params.name}')
					try:
						lease = await self._contexts.acquire_legacy()
					except ContextUnavailable as error:
						telemetry_error = error.code
						error_msg = f'Error: {error.code}: {error}'
						return types.CallToolResult(
							content=[types.TextContent(type='text', text=error_msg)], is_error=True
						)
					try:
						text, is_error = await self._run_owned_tool(self, params.name, arguments)
					finally:
						self._contexts.release(lease.entry)
				return types.CallToolResult(content=[types.TextContent(type='text', text=text)], is_error=is_error)
			except asyncio.CancelledError as error:
				telemetry_error = 'cancelled'
				error_msg = f'{type(error).__name__}: {error}'
				raise
			except MCPError as error:
				if telemetry_error is None:
					telemetry_error = 'mcp_error'
				error_msg = str(error)
				raise
			except ConfigStartupError as error:
				from openbrowser.browser.cdp_connection import MAX_ERROR_CHARS, sanitize_error_detail

				telemetry_error = error.code
				output_limit = output_limit_for_executor(self._executor)
				safe_detail = sanitize_error_detail(
					error.detail, max_chars=output_limit, preserve_lines=True
				)
				safe_path = sanitize_error_detail(str(error.path), max_chars=MAX_ERROR_CHARS) or str(error.path)
				raw_hint = error.hint or error.code
				if _PARTIAL_RUN_WARNING in raw_hint:
					base_hint = raw_hint.replace(_PARTIAL_RUN_WARNING, '').strip()
					base_budget = MAX_ERROR_CHARS - len(_PARTIAL_RUN_WARNING) - 1
					safe_base = sanitize_error_detail(base_hint, max_chars=base_budget) or error.code
					safe_hint = f'{safe_base} {_PARTIAL_RUN_WARNING}'
				else:
					safe_hint = sanitize_error_detail(raw_hint, max_chars=MAX_ERROR_CHARS) or error.code
				message = f'{error.code}: {safe_path}. {safe_hint}'
				error_msg = f'Error: {safe_detail}\n{message}' if safe_detail else f'Error: {message}'
				logger.error('Tool execution blocked by config: %s', error, exc_info=True)
				return types.CallToolResult(
					content=[types.TextContent(type='text', text=error_msg)], is_error=True
				)
			except Exception as error:
				from openbrowser.browser.cdp_connection import sanitize_error_detail

				telemetry_error = 'tool_error'
				error_msg = sanitize_error_detail(error) or type(error).__name__
				logger.error('Tool execution failed: %s', error, exc_info=True)
				return types.CallToolResult(
					content=[types.TextContent(type='text', text=f'Error: {error_msg}')], is_error=True
				)
			except BaseException as error:
				telemetry_error = 'internal_error'
				error_msg = f'{type(error).__name__}: {error}'
				raise
			finally:
				if self._telemetry and TELEMETRY_AVAILABLE:
					self._telemetry.capture(
						MCPServerTelemetryEvent(
							version=get_openbrowser_version(),
							action='tool_call',
							tool_name=telemetry_tool_name,
							duration_seconds=time.time() - start_time,
							error_message=telemetry_error if telemetry_error is not None else error_msg,
						)
					)

		def get_tool_input_schema(name: str) -> dict[str, Any] | None:
			tool = self._modern_by_name.get(name) or self._legacy_by_name.get(name)
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

	async def _run_owned_tool(self, owner: Any, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
		"""Run one validated application call as an owned, drained task."""

		async def owned_call() -> str | BaseException:
			try:
				return await owner._handle_tool(name, arguments)
			except asyncio.CancelledError:
				raise
			except BaseException as error:
				return error

		operation = asyncio.create_task(owned_call())
		completion = asyncio.gather(operation, return_exceptions=True)
		caller_cancellation = await _drain_owned_completion(completion, on_first_cancellation=operation.cancel)
		result = completion.result()[0]
		if caller_cancellation is not None:
			raise caller_cancellation
		if operation.cancelled():
			return 'Error: Tool operation was cancelled', True
		if isinstance(result, BaseException):
			raise result
		return result, False

	async def _create_context_result(self) -> types.CallToolResult:
		context_id = await self._contexts.create()
		payload = {'context_id': context_id}
		return types.CallToolResult(
			content=[types.TextContent(type='text', text=json.dumps(payload))],
			structured_content=payload,
		)

	async def _close_context_result(self, context_id: str) -> types.CallToolResult:
		result = await self._contexts.close(context_id)
		payload = {
			'context_id': result.context_id,
			'status': result.status,
			'error_code': result.error_code,
			'retryable': result.retryable,
		}
		return types.CallToolResult(
			content=[types.TextContent(type='text', text=json.dumps(payload))],
			structured_content=payload,
			is_error=result.status != 'closed',
		)

	async def _shutdown_locked(self) -> None:
		"""Fence new work, close every context, then close the legacy owner."""
		cleanup_task = self._cleanup_task
		self._cleanup_task = None
		if cleanup_task:
			cleanup_task.cancel()
			try:
				await cleanup_task
			except asyncio.CancelledError:
				pass
			except Exception as error:
				logger.error('Error stopping cleanup task: %s', error)

		await self._contexts.stop_admission()
		results = await self._contexts.close_all()
		failures = [result for result in results if result.status != 'closed']
		for failure in list(failures):
			if failure.context_id == '__legacy__':
				continue
			try:
				retry = await self._contexts.close(failure.context_id)
			except Exception:
				continue
			if retry.status == 'closed':
				failures.remove(failure)
		owner_error: Exception | None = None
		if self.browser_session is not None or self._executor.namespace is not None:
			try:
				await self._shutdown_owner_locked()
			except Exception as error:
				owner_error = error
				failures.append(
					CloseResult(
						context_id='__legacy__',
						status='close_failed',
						error_code='cleanup_failed',
						retryable=True,
					)
				)
			else:
				await self._contexts.mark_legacy_closed()
				failures = [result for result in failures if result.context_id != '__legacy__']
		if failures:
			await self._contexts.set_admission(ServerAdmission.FAILED)
			if owner_error is not None and len(failures) == 1:
				raise owner_error
			message = 'Context cleanup failed: ' + ', '.join(
				f'{result.context_id}:{result.error_code or result.status}' for result in failures
			)
			raise RuntimeError(message)
		await self._contexts.set_admission(ServerAdmission.STOPPED)

	async def _start_cleanup_task(self) -> None:
		"""Start the background cleanup task for every active owner."""

		async def cleanup_loop():
			while True:
				try:
					await self._cleanup_expired_session()
					for entry in self._contexts.snapshot():
						if entry.phase is not ContextPhase.ACTIVE or entry.owner is self:
							continue
						try:
							await entry.owner._cleanup_expired_session()
						except Exception as error:
							logger.error('Idle cleanup failed for %s: %s', entry.context_id, error)
					await asyncio.sleep(120)
				except asyncio.CancelledError:
					raise
				except Exception as error:
					logger.error('Error in cleanup task: %s', error)
					await asyncio.sleep(120)

		self._cleanup_task = asyncio.create_task(cleanup_loop())

	async def run(self):
		"""Run the MCP server and clean up before the transport context exits."""
		self._lifecycle.transition(LifecyclePhase.STARTING, reason='mcp server starting')
		shutdown_attempted = False
		try:
			await self._start_cleanup_task()
			self._lifecycle.transition(LifecyclePhase.READY, reason='mcp server ready')

			async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
				try:
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
					shutdown_attempted = True
					await self._shutdown()
		except BaseException as error:
			if not shutdown_attempted:
				try:
					await self._shutdown()
				except BaseException as shutdown_error:
					logger.error('Shutdown after startup failure also failed: %s', shutdown_error)
			try:
				self._lifecycle.fail(error, reason='mcp server shutdown failed')
			except LifecycleTransitionError:
				pass
			raise
		else:
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
