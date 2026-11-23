"""Event-driven browser session with backwards compatibility."""

import asyncio
import contextlib
import contextvars
import inspect
import logging
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, Union, cast, overload
from uuid import UUID

import httpx
from bubus import EventBus
from cdp_use import CDPClient
from cdp_use.cdp.fetch import AuthRequiredEvent, RequestPausedEvent
from cdp_use.cdp.network import Cookie
from cdp_use.cdp.target import AttachedToTargetEvent, SessionID, TargetID
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from uuid_extensions import uuid7str

# CDP logging is now handled by setup_logging() in logging_config.py
# It automatically sets CDP logs to the same level as openbrowser logs
from openbrowser.browser.cdp_connection import (
	CdpConnectionHealth,
	CdpConnectionHealthTracker,
	CdpConnectionPhase,
	sanitize_error_detail,
)
from openbrowser.browser.dom_cache import DocumentIdentity, DomCacheMachine
from openbrowser.browser.events import (
	AgentFocusChangedEvent,
	BrowserConnectedEvent,
	BrowserErrorEvent,
	BrowserKillEvent,
	BrowserLaunchEvent,
	BrowserLaunchResult,
	BrowserStartEvent,
	BrowserStateRequestEvent,
	BrowserStopEvent,
	BrowserStoppedEvent,
	CloseTabEvent,
	FileDownloadedEvent,
	NavigateToUrlEvent,
	NavigationCompleteEvent,
	NavigationFinishedEvent,
	NavigationStartedEvent,
	SwitchTabEvent,
	TabClosedEvent,
	TabClosingEvent,
	TabCreatedEvent,
)
from openbrowser.browser.focus import (
	CLOSE_RECONCILIATION_SECONDS,
	FOCUS_NAVIGATION_OBSERVER_SECONDS,
	FocusChangeReason,
	FocusPhase,
	FocusSnapshot,
	FocusTargetCode,
	FocusTargetError,
	FocusTransitionBudget,
	FocusTransitionResult,
)
from openbrowser.browser.health import (
	BrowserHealthMachine,
	BrowserHealthReason,
	OwnedRootBinding,
	canonical_cdp_endpoint,
)
from openbrowser.browser.lifecycle import (
	LifecycleMachine,
	LifecyclePhase,
	LifecycleSnapshot,
	LifecycleTransitionError,
)
from openbrowser.browser.navigation import (
	ContainmentAttempt,
	ContainmentPhase,
	NavigationBlockedError,
	NavigationBusyError,
	NavigationCommit,
	NavigationError,
	NavigationMonitor,
	NavigationOperation,
	NavigationPhase,
	NavigationTimeoutError,
	WaitUntil,
	await_document_ready,
	read_document_snapshot,
	security_policy_blocked_message,
	validate_actor_wait_until,
)
from openbrowser.browser.profile import BrowserProfile, ProxySettings, RenderMode
from openbrowser.browser.tabs import TabError, TabRegistry, TabState
from openbrowser.browser.views import BrowserStateSummary, TabInfo
from openbrowser.dom.views import DOMRect, EnhancedDOMTreeNode, TargetInfo
from openbrowser.observability import observe_debug
from openbrowser.utils import _log_pretty_url, is_new_tab_page

if TYPE_CHECKING:
	from openbrowser.actor.page import Page
	from openbrowser.browser.download_state import PdfDetectionSnapshot

DEFAULT_BROWSER_PROFILE = BrowserProfile()
NAVIGATION_CONTAINMENT_GRACE = 1.0
MAX_RETAINED_PROBES = 8
MAX_RETAINED_FETCH_DECISIONS = 4096

# Resource types the render policy keeps off the network. Stylesheet is only
# blocked for no-render; lightweight keeps CSS so layout and visibility match.
LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES: tuple[str, ...] = ('Image', 'Font', 'Media', 'TextTrack')
NO_RENDER_BLOCKED_RESOURCE_TYPES: tuple[str, ...] = (*LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES, 'Stylesheet')


def blocked_resource_types_for_render_mode(mode: RenderMode | str | None) -> tuple[str, ...]:
	"""Return the CDP resource types a render mode must not fetch."""
	try:
		resolved = RenderMode(mode) if mode is not None else RenderMode.FULL
	except ValueError:
		resolved = RenderMode.FULL
	if resolved is RenderMode.NO_RENDER:
		return NO_RENDER_BLOCKED_RESOURCE_TYPES
	if resolved is RenderMode.LIGHTWEIGHT:
		return LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES
	return ()


class _ProbeLifecycleClosed(RuntimeError):
	"""Internal signal that shutdown closed probe admission before the command ran."""

_POLICY_RETIREMENT_HINT = 'Runtime retained. Close the actual browser, then create a new BrowserSession; kill() can verify a managed local browser exit.'
_CONTAIN_DOCUMENT = """function(expectedUrl) {
	const active = this.defaultView && this.defaultView.document;
	if (this.nodeType !== 9 || this !== document || this !== active ||
		this.URL !== expectedUrl || active.URL !== expectedUrl) return {initiated:false};
	this.location.replace('about:blank');
	return {initiated:true};
}"""


def _shutdown_timeout() -> float:
	raw = os.environ.get('OPENBROWSER_SHUTDOWN_TIMEOUT', '')
	try:
		return max(1.0, float(raw)) if raw else 20.0
	except ValueError:
		return 20.0


_LOGGED_UNIQUE_SESSION_IDS = set()  # track unique session IDs that have been logged to make sure we always assign a unique enough id to new sessions and avoid ambiguity in logs
red = '\033[91m'
reset = '\033[0m'


class CDPSession(BaseModel):
	"""Info about a single CDP session bound to a specific target.

	Can optionally use its own WebSocket connection for better isolation.
	"""

	model_config = ConfigDict(arbitrary_types_allowed=True, revalidate_instances='never')

	cdp_client: CDPClient

	target_id: TargetID
	session_id: SessionID
	title: str = 'Unknown title'
	url: str = 'about:blank'

	@classmethod
	async def for_target(
		cls,
		cdp_client: CDPClient,
		target_id: TargetID,
		domains: list[str] | None = None,
	):
		"""Create a CDP session for a target using the shared WebSocket.

		Args:
                cdp_client: The shared CDP client (root WebSocket connection)
                target_id: Target ID to attach to
                domains: List of CDP domains to enable. If None, enables default domains.
		"""
		# Always use shared CDP client (event-driven approach)
		cdp_session = cls(
			cdp_client=cdp_client,
			target_id=target_id,
			session_id='connecting',
		)
		return await cdp_session.attach(domains=domains)

	async def attach(self, domains: list[str] | None = None) -> Self:
		result = await self.cdp_client.send.Target.attachToTarget(
			params={
				'targetId': self.target_id,
				'flatten': True,  # removed filter as a param because it doesn't exist at https://chromedevtools.github.io/devtools-protocol/tot/Target/#method-attachToTarget
			}
		)
		self.session_id = result['sessionId']

		# Use specified domains or default domains
		domains = domains or ['Page', 'DOM', 'DOMSnapshot', 'Accessibility', 'Runtime', 'Inspector']

		# Enable all domains in parallel
		enable_tasks = []
		for domain in domains:
			# Get the enable method, e.g. self.cdp_client.send.Page.enable(session_id=self.session_id)
			domain_api = getattr(self.cdp_client.send, domain, None)
			# Browser and Target domains don't use session_id, dont pass it for those
			enable_kwargs = {} if domain in ['Browser', 'Target'] else {'session_id': self.session_id}
			assert domain_api and hasattr(domain_api, 'enable'), (
				f'{domain_api} is not a recognized CDP domain with a .enable() method'
			)
			enable_tasks.append(domain_api.enable(**enable_kwargs))

		results = await asyncio.gather(*enable_tasks, return_exceptions=True)
		if any(isinstance(result, Exception) for result in results):
			raise RuntimeError(f'Failed to enable requested CDP domain: {results}')

		# in case 'Debugger' domain is enabled, disable breakpoints on the page so it doesnt pause on crashes / debugger statements
		# also covered by Runtime.runIfWaitingForDebugger() calls in get_or_create_cdp_session()
		try:
			await self.cdp_client.send.Debugger.setSkipAllPauses(params={'skip': True}, session_id=self.session_id)
			# if 'Debugger' not in domains:
			# 	await self.cdp_client.send.Debugger.disable()
			# await cdp_session.cdp_client.send.EventBreakpoints.disable(session_id=cdp_session.session_id)
		except Exception:
			# self.logger.warning(f'Failed to disable page JS breakpoints: {e}')
			pass

		target_info = await self.get_target_info()
		self.title = target_info['title']
		self.url = target_info['url']
		return self

	async def disconnect(self) -> None:
		"""Disconnect session (no-op since we use shared WebSocket)."""
		# With event-driven approach, all sessions share the root WebSocket
		# Nothing to disconnect - only the root client is disconnected on browser.stop()
		pass

	async def get_tab_info(self) -> TabInfo:
		target_info = await self.get_target_info()
		return TabInfo(
			target_id=target_info['targetId'],
			url=target_info['url'],
			title=target_info['title'],
		)

	async def get_target_info(self) -> TargetInfo:
		result = await self.cdp_client.send.Target.getTargetInfo(params={'targetId': self.target_id})
		return result['targetInfo']


@dataclass(frozen=True)
class _FocusCheckpoint:
	"""Focus state captured before one serialized transition."""

	snapshot: FocusSnapshot
	session: CDPSession | None


@dataclass(frozen=True)
class _PolicyRetentionFailure:
	operation: NavigationOperation
	root: CDPClient
	failure: LifecycleSnapshot
	revision: int


@dataclass
class _PendingEventBusRetirement:
	"""One failed event-bus retirement and any run loop still unwinding."""

	event_bus: EventBus
	runloop_tasks: list[asyncio.Future] = field(default_factory=list)


class BrowserSession(BaseModel):
	"""Event-driven browser session with backwards compatibility.

	This class provides a 2-layer architecture:
	- High-level event handling for agents/tools
	- Direct CDP/Playwright calls for browser operations

	Supports both event-driven and imperative calling styles.

	Browser configuration is stored in the browser_profile, session identity in direct fields:
	```python
	# Direct settings (recommended for most users)
	session = BrowserSession(headless=True, user_data_dir='./profile')

	# Or use a profile (for advanced use cases)
	session = BrowserSession(browser_profile=BrowserProfile(...))

	# Access session fields directly, browser settings via profile or property
	print(session.id)  # Session field
	```
	"""

	model_config = ConfigDict(
		arbitrary_types_allowed=True,
		validate_assignment=True,
		extra='forbid',
		revalidate_instances='never',  # resets private attrs on every model rebuild
	)

	# Local browser mode init
	@overload
	def __init__(
		self,
		*,
		# Core configuration for local
		id: str | None = None,
		cdp_url: str | None = None,
		browser_profile: BrowserProfile | None = None,
		# Local browser launch params
		executable_path: str | Path | None = None,
		headless: bool | None = None,
		user_data_dir: str | Path | None = None,
		args: list[str] | None = None,
		downloads_path: str | Path | None = None,
		# Common params
		headers: dict[str, str] | None = None,
		allowed_domains: list[str] | None = None,
		keep_alive: bool | None = None,
		minimum_wait_page_load_time: float | None = None,
		wait_for_network_idle_page_load_time: float | None = None,
		wait_between_actions: float | None = None,
		auto_download_pdfs: bool | None = None,
		cookie_whitelist_domains: list[str] | None = None,
		cross_origin_iframes: bool | None = None,
		highlight_elements: bool | None = None,
		dom_highlight_elements: bool | None = None,
		paint_order_filtering: bool | None = None,
		max_iframes: int | None = None,
		max_iframe_depth: int | None = None,
		# All other local params
		env: dict[str, str | float | bool] | None = None,
		ignore_default_args: list[str] | Literal[True] | None = None,
		channel: str | None = None,
		chromium_sandbox: bool | None = None,
		devtools: bool | None = None,
		traces_dir: str | Path | None = None,
		accept_downloads: bool | None = None,
		permissions: list[str] | None = None,
		user_agent: str | None = None,
		screen: dict | None = None,
		viewport: dict | None = None,
		no_viewport: bool | None = None,
		device_scale_factor: float | None = None,
		record_har_content: str | None = None,
		record_har_mode: str | None = None,
		record_har_path: str | Path | None = None,
		record_video_dir: str | Path | None = None,
		record_video_framerate: int | None = None,
		record_video_size: dict | None = None,
		storage_state: str | Path | dict[str, Any] | None = None,
		disable_security: bool | None = None,
		deterministic_rendering: bool | None = None,
		proxy: ProxySettings | None = None,
		enable_default_extensions: bool | None = None,
		window_size: dict | None = None,
		window_position: dict | None = None,
		filter_highlight_ids: bool | None = None,
		profile_directory: str | None = None,
	) -> None: ...

	def __init__(
		self,
		# Core configuration
		id: str | None = None,
		cdp_url: str | None = None,
		is_local: bool | None = None,
		browser_profile: BrowserProfile | None = None,
		# BrowserProfile fields that can be passed directly
		# From BrowserConnectArgs
		headers: dict[str, str] | None = None,
		# From BrowserLaunchArgs
		env: dict[str, str | float | bool] | None = None,
		executable_path: str | Path | None = None,
		headless: bool | None = None,
		args: list[str] | None = None,
		ignore_default_args: list[str] | Literal[True] | None = None,
		channel: str | None = None,
		chromium_sandbox: bool | None = None,
		devtools: bool | None = None,
		downloads_path: str | Path | None = None,
		traces_dir: str | Path | None = None,
		# From BrowserContextArgs
		accept_downloads: bool | None = None,
		permissions: list[str] | None = None,
		user_agent: str | None = None,
		screen: dict | None = None,
		viewport: dict | None = None,
		no_viewport: bool | None = None,
		device_scale_factor: float | None = None,
		record_har_content: str | None = None,
		record_har_mode: str | None = None,
		record_har_path: str | Path | None = None,
		record_video_dir: str | Path | None = None,
		record_video_framerate: int | None = None,
		record_video_size: dict | None = None,
		# From BrowserLaunchPersistentContextArgs
		user_data_dir: str | Path | None = None,
		# From BrowserNewContextArgs
		storage_state: str | Path | dict[str, Any] | None = None,
		# BrowserProfile specific fields
		disable_security: bool | None = None,
		deterministic_rendering: bool | None = None,
		allowed_domains: list[str] | None = None,
		keep_alive: bool | None = None,
		proxy: ProxySettings | None = None,
		enable_default_extensions: bool | None = None,
		window_size: dict | None = None,
		window_position: dict | None = None,
		minimum_wait_page_load_time: float | None = None,
		wait_for_network_idle_page_load_time: float | None = None,
		wait_between_actions: float | None = None,
		filter_highlight_ids: bool | None = None,
		auto_download_pdfs: bool | None = None,
		profile_directory: str | None = None,
		cookie_whitelist_domains: list[str] | None = None,
		# DOM extraction layer configuration
		cross_origin_iframes: bool | None = None,
		highlight_elements: bool | None = None,
		dom_highlight_elements: bool | None = None,
		paint_order_filtering: bool | None = None,
		# Iframe processing limits
		max_iframes: int | None = None,
		max_iframe_depth: int | None = None,
	):
		# Following the same pattern as AgentSettings in service.py
		# Only pass non-None values to avoid validation errors
		profile_kwargs = {
			k: v
			for k, v in locals().items()
			if k
			not in [
				'self',
				'browser_profile',
				'id',
			]
			and v is not None
		}

		# An explicit executable_path means this process launches the browser.
		if executable_path is not None:
			profile_kwargs['is_local'] = True
		# Set is_local=True only when neither the direct argument nor the
		# supplied profile has a CDP URL to connect to.
		if not cdp_url and (browser_profile is None or not browser_profile.cdp_url):
			profile_kwargs['is_local'] = True

		# Create browser profile from direct parameters or use provided one
		if browser_profile is not None:
			# Merge any direct kwargs into the provided browser_profile (direct kwargs take precedence)
			merged_kwargs = {**browser_profile.model_dump(exclude_unset=True), **profile_kwargs}
			resolved_browser_profile = BrowserProfile(**merged_kwargs)
		else:
			resolved_browser_profile = BrowserProfile(**profile_kwargs)

		# Initialize the Pydantic model
		super().__init__(
			id=id or str(uuid7str()),
			browser_profile=resolved_browser_profile,
		)

	# Session configuration (session identity only)
	id: str = Field(default_factory=lambda: str(uuid7str()), description='Unique identifier for this browser session')

	# Browser configuration (reusable profile)
	browser_profile: BrowserProfile = Field(
		default_factory=lambda: DEFAULT_BROWSER_PROFILE,
		description='BrowserProfile() options to use for the session, otherwise a default profile will be used',
	)

	# Convenience properties for common browser settings
	@property
	def cdp_url(self) -> str | None:
		"""CDP URL from browser profile."""
		return self.browser_profile.cdp_url

	@property
	def is_local(self) -> bool:
		"""Whether this is a local browser instance from browser profile."""
		return self.browser_profile.is_local

	# Main shared event bus for all browser session + all watchdogs
	event_bus: EventBus = Field(default_factory=EventBus)

	# Mutable public state
	agent_focus: CDPSession | None = None

	# Mutable private state shared between watchdogs
	_cdp_client_root: CDPClient | None = PrivateAttr(default=None)
	_cdp_session_pool: dict[str, CDPSession] = PrivateAttr(default_factory=dict)
	_session_manager: Any = PrivateAttr(default=None)  # SessionManager instance
	_cdp_health: CdpConnectionHealthTracker = PrivateAttr(default_factory=CdpConnectionHealthTracker)
	_lifecycle: LifecycleMachine = PrivateAttr(default_factory=lambda: LifecycleMachine('browser_session'))
	_health: BrowserHealthMachine = PrivateAttr(default_factory=BrowserHealthMachine)
	_health_owned_root_binding: OwnedRootBinding | None = PrivateAttr(default=None)
	_tab_registry: TabRegistry = PrivateAttr(default_factory=TabRegistry)
	_lifecycle_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
	_connect_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
	_connect_admission_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
	_start_admission_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
	_direct_connect_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
	_active_cleanup_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
	_detached_cleanup_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
	_detached_probe_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
	_stop_reset_error: BaseException | None = PrivateAttr(default=None)
	_detached_shutdown_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
	_watchdog_cleanup_error: BaseException | None = PrivateAttr(default=None)
	_probe_slots: set[asyncio.Task] = PrivateAttr(default_factory=set)
	_connect_admission_closed: bool = PrivateAttr(default=False)
	_connect_admission_epoch: int = PrivateAttr(default=0)
	_shutdown_fence_epoch: int = PrivateAttr(default=0)
	_reset_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
	_reset_task: asyncio.Task | None = PrivateAttr(default=None)
	_navigation_monitor: NavigationMonitor = PrivateAttr(default_factory=NavigationMonitor)
	_navigation_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
	_focus_state: FocusSnapshot = PrivateAttr(default_factory=FocusSnapshot.uninitialized)
	_focus_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
	_focus_epoch: int = PrivateAttr(default=0)
	_focus_observer_tasks: set[asyncio.Task[Any]] = PrivateAttr(default_factory=set)
	_navigation_task: asyncio.Task | None = PrivateAttr(default=None)
	_navigation_task_operation: NavigationOperation | None = PrivateAttr(default=None)
	_navigation_event_pumps: dict[asyncio.Task, EventBus] = PrivateAttr(default_factory=dict)
	_navigation_policies: dict[str, NavigationOperation] = PrivateAttr(default_factory=dict)
	_policy_retention_failure: _PolicyRetentionFailure | None = PrivateAttr(default=None)
	_start_event: asyncio.Event | None = PrivateAttr(default=None)
	_start_error: BaseException | None = PrivateAttr(default=None)
	_start_task: asyncio.Task | None = PrivateAttr(default=None)
	_start_cancelled_by_lifecycle: bool = PrivateAttr(default=False)
	_stop_task: asyncio.Task | None = PrivateAttr(default=None)
	_watchdogs_attached: bool = PrivateAttr(default=False)
	_event_bus_retirement_pending: list[_PendingEventBusRetirement] = PrivateAttr(default_factory=list)
	_core_handlers_attached: bool = PrivateAttr(default=False)
	_cached_browser_state_summary: Any = PrivateAttr(default=None)
	_cached_selector_map: dict[int, EnhancedDOMTreeNode] = PrivateAttr(default_factory=dict)
	_dom_cache: DomCacheMachine = PrivateAttr(default_factory=DomCacheMachine)
	_downloaded_files: list[str] = PrivateAttr(default_factory=list)  # Track files downloaded during this session
	_download_history: list[dict[str, object]] = PrivateAttr(default_factory=list)  # Last outcomes, kept after reset
	_download_outcomes_error: str | None = PrivateAttr(default=None)  # Last reader failure, surfaced in status
	_closed_popup_messages: list[str] = PrivateAttr(
		default_factory=list
	)  # Store messages from auto-closed JavaScript dialogs
	_storage_state_warnings: list[str] = PrivateAttr(default_factory=list)
	_fetch_handled_requests: dict[tuple[str | None, str], None] = PrivateAttr(default_factory=dict)
	_fetch_interceptor_tasks: set[asyncio.Task[Any]] = PrivateAttr(default_factory=set)

	# Watchdogs
	_crash_watchdog: Any | None = PrivateAttr(default=None)
	_downloads_watchdog: Any | None = PrivateAttr(default=None)
	_aboutblank_watchdog: Any | None = PrivateAttr(default=None)
	_security_watchdog: Any | None = PrivateAttr(default=None)
	_storage_state_watchdog: Any | None = PrivateAttr(default=None)
	_local_browser_watchdog: Any | None = PrivateAttr(default=None)
	_default_action_watchdog: Any | None = PrivateAttr(default=None)
	_dom_watchdog: Any | None = PrivateAttr(default=None)
	_screenshot_watchdog: Any | None = PrivateAttr(default=None)
	_permissions_watchdog: Any | None = PrivateAttr(default=None)
	_recording_watchdog: Any | None = PrivateAttr(default=None)

	_logger: Any = PrivateAttr(default=None)

	@property
	def logger(self) -> Any:
		"""Get instance-specific logger with session ID in the name"""
		# **regenerate it every time** because our id and str(self) can change as browser connection state changes
		# if self._logger is None or not self._cdp_client_root:
		# 	self._logger = logging.getLogger(f'openbrowser.{self}')
		return logging.getLogger(f'openbrowser.{self}')

	@cached_property
	def _id_for_logs(self) -> str:
		"""Get human-friendly semi-unique identifier for differentiating different BrowserSession instances in logs"""
		str_id = self.id[
			-4:
		]  # default to last 4 chars of truly random uuid, less helpful than cdp port but always unique enough
		port_number = (self.cdp_url or 'no-cdp').rsplit(':', 1)[-1].split('/', 1)[0].strip()
		port_is_random = not port_number.startswith('922')
		port_is_unique_enough = port_number not in _LOGGED_UNIQUE_SESSION_IDS
		if port_number and port_number.isdigit() and port_is_random and port_is_unique_enough:
			# if cdp port is random/unique enough to identify this session, use it as our id in logs
			_LOGGED_UNIQUE_SESSION_IDS.add(port_number)
			str_id = port_number
		return str_id

	@property
	def _tab_id_for_logs(self) -> str:
		return self.agent_focus.target_id[-2:] if self.agent_focus and self.agent_focus.target_id else f'{red}--{reset}'

	def __repr__(self) -> str:
		return f'BrowserSession🅑 {self._id_for_logs} 🅣 {self._tab_id_for_logs} (cdp_url={self.cdp_url}, profile={self.browser_profile})'

	def __str__(self) -> str:
		return f'BrowserSession🅑 {self._id_for_logs} 🅣 {self._tab_id_for_logs}'

	@property
	def lifecycle(self) -> LifecycleSnapshot:
		"""Return the current lifecycle snapshot."""
		return self._lifecycle.snapshot()

	def lifecycle_snapshot(self) -> dict[str, Any]:
		"""Return lifecycle, tab, connection, and storage-state state in one JSON-safe shape."""
		from openbrowser.browser.storage_state import (
			load_snapshot_for_configuration,
			save_snapshot_for_configuration,
		)

		snapshot = self._lifecycle.snapshot().to_dict()
		snapshot['last_error'] = sanitize_error_detail(snapshot.get('last_error'))
		cdp_health = self.cdp_connection_health()
		snapshot['cdp_connected'] = cdp_health.client_present
		snapshot['cdp'] = cdp_health.to_dict()
		snapshot['health'] = self.browser_health_snapshot()
		snapshot['active_target_id'] = self.agent_focus.target_id if self.agent_focus else None
		snapshot['tabs'] = self.tab_snapshot()
		snapshot['navigation'] = self._navigation_monitor.snapshot()
		snapshot['focus'] = self.focus_snapshot().to_dict()
		snapshot['dom_cache'] = self.dom_cache_snapshot()
		snapshot['warnings'] = list(self._storage_state_warnings)
		configured = self.browser_profile.storage_state if self.browser_profile else None
		storage_state = None
		storage_state_save = None
		watchdog = getattr(self, '_storage_state_watchdog', None)
		if watchdog is not None:
			try:
				storage_state = watchdog.storage_state_snapshot()
				storage_state_save = watchdog.storage_state_save_snapshot()
			except Exception:
				storage_state = None
				storage_state_save = None
		if storage_state is None:
			storage_state = load_snapshot_for_configuration(configured)
		if storage_state_save is None:
			storage_state_save = save_snapshot_for_configuration(configured)
		snapshot['storage_state'] = storage_state.to_dict()
		snapshot['storage_state_save'] = storage_state_save.to_dict()
		return snapshot

	def cdp_connection_health(self) -> CdpConnectionHealth:
		"""Return the last known root CDP health without network I/O."""
		connection_expected = self._lifecycle.phase in (
			LifecyclePhase.STARTING,
			LifecyclePhase.READY,
			LifecyclePhase.DEGRADED,
			LifecyclePhase.RECOVERING,
		)
		return self._cdp_health.snapshot(
			client_present=self._cdp_client_root is not None,
			connection_expected=connection_expected,
		)

	def _health_binding_is_current(self, binding: OwnedRootBinding | None) -> bool:
		"""Return True only while the published root is the exact bound owner root."""
		if binding is None:
			return False
		root = self._cdp_client_root
		if root is None or root is not binding.root:
			return False
		try:
			websocket = getattr(root, 'ws', None)
			transport = getattr(websocket, 'transport', None)
		except Exception:
			return False
		if websocket is not binding.websocket or transport is not binding.transport:
			return False
		if self._cdp_health.connection_generation != binding.connection_generation:
			return False
		if canonical_cdp_endpoint(self.cdp_url) != binding.endpoint:
			return False
		watchdog = self._local_browser_watchdog
		if watchdog is None or getattr(watchdog, '_owns_browser_resources', False) is not True:
			return False
		record = getattr(watchdog, '_browser_record', None)
		lease = getattr(watchdog, '_profile_lease', None)
		if not isinstance(record, dict) or lease is None:
			return False
		try:
			profile = str(Path(record['profile_dir']).expanduser().resolve())
			if (
				int(record['pid']) != binding.pid
				or float(record['start_time']) != binding.start_time
				or profile != binding.profile_dir
				or str(record['instance_id']) != binding.instance_id
				or str(record['ownership_marker']) != binding.ownership_marker
				or str(record['executable']) != binding.executable
				or int(record['cdp_port']) != binding.cdp_port
				or str(Path(lease.profile_dir).expanduser().resolve()) != binding.profile_dir
				or str(lease.instance_id) != binding.instance_id
				or str(getattr(watchdog, '_instance_id', '')) != binding.instance_id
			):
				return False
		except (AttributeError, KeyError, TypeError, ValueError, OSError, RuntimeError):
			return False
		return True

	def _establish_owned_root_binding(self, generation: int) -> OwnedRootBinding | None:
		"""Prove the live root socket belongs to the recorded local process and bind it."""
		self._health_owned_root_binding = None
		if not self.is_local:
			return None
		root = self._cdp_client_root
		watchdog = self._local_browser_watchdog
		if root is None or watchdog is None or getattr(watchdog, '_owns_browser_resources', False) is not True:
			return None
		record = getattr(watchdog, '_browser_record', None)
		lease = getattr(watchdog, '_profile_lease', None)
		process = getattr(watchdog, '_subprocess', None)
		if not isinstance(record, dict) or lease is None or process is None:
			return None

		import ipaddress
		import psutil
		import socket
		from openbrowser.browser.watchdogs.profile_lease import ProfileLease

		try:
			user_data_dir = self.browser_profile.user_data_dir
			if user_data_dir is None:
				return None
			profile = str(Path(record['profile_dir']).expanduser().resolve())
			if (
				profile != str(Path(user_data_dir).expanduser().resolve())
				or profile != str(Path(lease.profile_dir).expanduser().resolve())
				or str(record['instance_id']) != str(lease.instance_id)
				or str(record['instance_id']) != str(getattr(watchdog, '_instance_id', ''))
				or not ProfileLease.process_matches_browser(process, record)
			):
				return None
			endpoint = canonical_cdp_endpoint(self.cdp_url)
			if endpoint is None:
				return None
			websocket = getattr(root, 'ws', None)
			if websocket is None:
				return None
			transport = getattr(websocket, 'transport', None)
			if transport is None:
				return None
			connection_socket = transport.get_extra_info('socket')
			family = connection_socket.family
			if family not in {socket.AF_INET, socket.AF_INET6}:
				return None

			def endpoint_for(value: Any) -> tuple[Any, int] | None:
				if not isinstance(value, tuple) or len(value) < 2 or not isinstance(value[1], int) or any(value[2:]):
					return None
				address = ipaddress.ip_address(value[0])
				if (
					address.version != (4 if family == socket.AF_INET else 6)
					or getattr(address, 'scope_id', None) is not None
				):
					return None
				return address, value[1]

			peer = endpoint_for(connection_socket.getpeername())
			local = endpoint_for(connection_socket.getsockname())
			cdp_port = int(record['cdp_port'])
			if peer is None or local is None or not peer[0].is_loopback or not local[0].is_loopback:
				return None
			if peer[1] != cdp_port:
				return None
			connections = [connection for connection in process.net_connections(kind='tcp') if connection.family == family]
			listeners = [
				endpoint_for(connection.laddr) for connection in connections if connection.status == psutil.CONN_LISTEN
			]
			covers_peer = any(
				listener is not None
				and listener[1] == peer[1]
				and (listener[0] == peer[0] or listener[0].is_unspecified)
				for listener in listeners
			)
			accepted_peer = any(
				connection.status == psutil.CONN_ESTABLISHED
				and endpoint_for(connection.laddr) == peer
				and endpoint_for(connection.raddr) == local
				for connection in connections
			)
			if not covers_peer or not accepted_peer:
				return None
			binding = OwnedRootBinding(
				connection_generation=int(generation),
				endpoint=endpoint,
				pid=int(record['pid']),
				start_time=float(record['start_time']),
				profile_dir=profile,
				instance_id=str(record['instance_id']),
				ownership_marker=str(record['ownership_marker']),
				executable=str(record['executable']),
				cdp_port=cdp_port,
				root=root,
				websocket=websocket,
				transport=transport,
			)
		except Exception:
			return None
		self._health_owned_root_binding = binding
		return binding

	def browser_health_snapshot(self) -> dict[str, Any]:
		"""Return browser health, including a status-time owned-process check."""
		from openbrowser.browser.watchdogs.profile_lease import ProfileLease

		binding = self._health_owned_root_binding
		try:
			binding_current = self._health_binding_is_current(binding)
		except Exception:
			binding_current = False
		process_present: bool | None = None
		process_pid: int | None = None
		process_checked_at: float | None = None
		projection_active = False
		if binding_current and binding is not None and self._lifecycle.phase is not LifecyclePhase.STOPPING:
			process_checked_at = time.time()
			process_pid = binding.pid
			try:
				identity = ProfileLease.process_identity_status(binding.pid, binding.start_time)
			except Exception:
				identity = 'unknown'
			if identity == 'mismatch':
				owner_identity_clearly_differs = self._process_ownership_marker_matches(
					binding,
					getattr(self._local_browser_watchdog, '_subprocess', None),
				) is False
				if not owner_identity_clearly_differs:
					identity = 'unknown'
			if identity == 'match':
				process_present = True
			elif identity in {'missing', 'mismatch'}:
				process_present = False
				projection_active = self._cdp_client_root is not None
		return self._health.snapshot(
			process_present=process_present,
			process_pid=process_pid,
			process_checked_at=process_checked_at,
			projection_active=projection_active,
		).to_dict()

	@staticmethod
	def _process_ownership_marker_matches(binding: OwnedRootBinding, process: Any | None) -> bool | None:
		"""Return False only when a reused PID clearly lacks the bound owner identity."""
		if process is None:
			return None
		import psutil

		try:
			if int(process.pid) != binding.pid:
				return False
			cmdline = process.cmdline()
		except psutil.NoSuchProcess:
			return False
		except (AttributeError, OSError, psutil.Error):
			return None
		if not isinstance(cmdline, (list, tuple)):
			return None
		profile_matches = False
		for argument in cmdline:
			if not isinstance(argument, str) or not argument.startswith('--user-data-dir='):
				continue
			try:
				profile_matches = str(Path(argument.split('=', 1)[1]).expanduser().resolve()) == binding.profile_dir
			except (OSError, RuntimeError):
				return None
			if profile_matches:
				break
		return profile_matches and binding.ownership_marker in cmdline

	def _record_stopped_health(self, reason: BrowserHealthReason) -> None:
		"""Record one completed public terminal operation."""
		self._health.record_stopped(reason, self._cdp_health.connection_generation)

	def _record_cleanup_failure_health(self) -> None:
		"""Record one retained reset, stop, or kill cleanup failure."""
		self._health.record_cleanup_failed(self._cdp_health.connection_generation)

	def _record_startup_failure_health(self) -> None:
		"""Record a health-bearing start failure without overriding shutdown."""
		if self._lifecycle.phase in (LifecyclePhase.STOPPING, LifecyclePhase.STOPPED):
			return
		self._health.record_startup_failed(self._cdp_health.connection_generation)

	async def probe_cdp_connection(self, *, timeout_seconds: float = 2.0) -> CdpConnectionHealth:
		"""Send one root-level health probe and return the last known snapshot."""
		if self._lifecycle.phase not in (
			LifecyclePhase.READY,
			LifecyclePhase.DEGRADED,
			LifecyclePhase.RECOVERING,
		):
			return self.cdp_connection_health()
		current = self.cdp_connection_health()
		if current.phase in (
			CdpConnectionPhase.RESOLVING,
			CdpConnectionPhase.CONNECTING,
			CdpConnectionPhase.FAILED,
		):
			return current
		active_probes = sum(1 for task in self._probe_slots if not task.done())
		if active_probes >= MAX_RETAINED_PROBES:
			self.logger.warning('CDP probe skipped: %d probe commands are still running', active_probes)
			return current
		try:
			timeout = float(timeout_seconds)
		except (TypeError, ValueError, OverflowError):
			timeout = 2.0
		if not math.isfinite(timeout) or timeout <= 0:
			timeout = 2.0
		root = self._cdp_client_root
		token = self._cdp_health.begin_probe(current.connection_generation)
		if root is None:
			self._cdp_health.complete_probe_failure(
				token,
				RuntimeError('CDP client is not connected'),
				latency_ms=0.0,
				client_present=False,
			)
			return self.cdp_connection_health()
		started = time.perf_counter()

		async def run_probe():
			if (
				self._connect_admission_closed
				or self._lifecycle.phase not in (
					LifecyclePhase.READY,
					LifecyclePhase.DEGRADED,
					LifecyclePhase.RECOVERING,
				)
				or self._cdp_client_root is not root
				or self._cdp_health.connection_generation != token.generation
			):
				raise _ProbeLifecycleClosed('CDP probe ownership changed before command start')
			return await root.send.Browser.getVersion()

		probe_task = asyncio.create_task(run_probe())
		self._probe_slots.add(probe_task)
		probe_task.add_done_callback(self._probe_slots.discard)
		try:
			await self._await_bounded_operation(
				probe_task,
				timeout,
				'CDP probe',
				retained_tasks=self._detached_probe_tasks,
			)
		except asyncio.CancelledError as error:
			if self._caller_cancellation_requested():
				self._cdp_health.discard_probe(token)
				raise
			latency_ms = round((time.perf_counter() - started) * 1000, 3)
			if self._cdp_client_root is root:
				self._cdp_health.complete_probe_failure(
					token,
					error,
					latency_ms=latency_ms,
					client_present=True,
					worker_cancelled=True,
				)
			else:
				self._cdp_health.discard_probe(token)
		except _ProbeLifecycleClosed:
			self._cdp_health.discard_probe(token)
		except Exception as error:
			latency_ms = round((time.perf_counter() - started) * 1000, 3)
			if self._cdp_client_root is root:
				self._cdp_health.complete_probe_failure(
					token,
					error,
					latency_ms=latency_ms,
					client_present=True,
				)
			else:
				self._cdp_health.discard_probe(token)
		else:
			latency_ms = round((time.perf_counter() - started) * 1000, 3)
			if self._cdp_client_root is root:
				self._cdp_health.complete_probe_success(token, latency_ms=latency_ms)
			else:
				self._cdp_health.discard_probe(token)
		return self.cdp_connection_health()

	def tab_snapshot(self) -> list[dict[str, Any]]:
		"""Return the registry view of all known tabs."""
		active_target_id = self.agent_focus.target_id if self.agent_focus else None
		return self._tab_registry.snapshot(active_target_id=active_target_id)

	def focus_snapshot(self) -> FocusSnapshot:
		"""Return the current focus state machine snapshot."""
		return self._focus_state

	def _focus_checkpoint_locked(self) -> _FocusCheckpoint:
		return _FocusCheckpoint(snapshot=self._focus_state, session=self.agent_focus)

	def _begin_focus_change_locked(
		self,
		reason: FocusChangeReason,
		target_id: str | None,
		phase: FocusPhase,
	) -> None:
		current = self._focus_state
		previous_target_id = current.target_id or current.previous_target_id
		self._focus_state = FocusSnapshot(
			phase=phase,
			target_id=target_id if phase is FocusPhase.SWITCHING else None,
			session_id=None,
			previous_target_id=previous_target_id,
			generation=current.generation,
			reason=reason,
			last_error=None,
			updated_at=time.time(),
		)
		if phase is FocusPhase.RECOVERING:
			self.agent_focus = None

	def _restore_focus_locked(self, checkpoint: _FocusCheckpoint) -> None:
		self._focus_state = checkpoint.snapshot
		self.agent_focus = checkpoint.session

	def _focus_lost_locked(
		self,
		error: FocusTargetError,
		reason: FocusChangeReason,
		previous_target_id: str | None = None,
	) -> FocusSnapshot:
		current = self._focus_state
		prior_target = previous_target_id or current.target_id or current.previous_target_id
		generation = current.generation
		if current.phase not in (FocusPhase.LOST, FocusPhase.UNINITIALIZED):
			generation += 1
		self._clear_focus_caches_locked()
		self.agent_focus = None
		self._focus_state = FocusSnapshot(
			phase=FocusPhase.LOST,
			target_id=None,
			session_id=None,
			previous_target_id=prior_target,
			generation=generation,
			reason=reason,
			last_error=error.to_snapshot(),
			updated_at=time.time(),
		)
		return self._focus_state

	def _clear_focus_caches_locked(self) -> None:
		self.reset_dom_cache(reason='focus_change')

	async def _target_liveness(self, target_id: str, budget: FocusTransitionBudget) -> bool | None:
		"""Return target existence, or None when the browser cannot prove it."""
		root = self._cdp_client_root
		if root is None:
			return None
		try:
			result = await self._await_focus_operation(
				root.send.Target.getTargets(),
				budget,
				target_id=target_id,
				phase=self._focus_state.phase,
			)
		except Exception:
			return None
		if not isinstance(result, dict):
			return None
		target_infos = result.get('targetInfos')
		if not isinstance(target_infos, list):
			return None
		return any(isinstance(target, dict) and target.get('targetId') == target_id for target in target_infos)

	def _set_agent_focus_locked(
		self,
		session: CDPSession,
		reason: FocusChangeReason,
		*,
		previous_snapshot: FocusSnapshot | None = None,
	) -> FocusSnapshot:
		"""Commit one validated page session.

		The caller holds `_focus_lock`. The comparison snapshot is the state
		before the transaction began, not the temporary `switching` snapshot.
		"""
		comparison = previous_snapshot or self._focus_state
		record = self._tab_registry.get(session.target_id)
		if record is not None and record.state in (TabState.CLOSING, TabState.CLOSED, TabState.CRASHED):
			raise FocusTargetError(
				f'Target {session.target_id[-8:]} is {record.state.value} and cannot receive focus.',
				code=FocusTargetCode.TAB_NOT_LIVE,
				phase=self._focus_state.phase,
				target_id=session.target_id,
				target_type='page',
			)
		if comparison.phase is FocusPhase.FOCUSED and comparison.is_exact_pair(session.target_id, session.session_id):
			self._focus_state = comparison
			self.agent_focus = session
			return comparison

		self._clear_focus_caches_locked()
		session_url = getattr(session, 'url', 'about:blank')
		session_title = getattr(session, 'title', '')
		if record is None:
			self._tab_registry.register(
				session.target_id,
				url=session_url,
				title=session_title,
				state=TabState.OPEN,
			)
		elif record.state is TabState.OPENING:
			self._tab_registry.update(
				session.target_id,
				url=session_url,
				title=session_title,
				state=TabState.OPEN,
			)
		else:
			self._tab_registry.update(session.target_id, url=session_url, title=session_title)
		previous_target_id = comparison.target_id or comparison.previous_target_id
		self.agent_focus = session
		self._focus_state = FocusSnapshot(
			phase=FocusPhase.FOCUSED,
			target_id=session.target_id,
			session_id=session.session_id,
			previous_target_id=previous_target_id,
			generation=comparison.generation + 1,
			reason=reason,
			last_error=None,
			updated_at=time.time(),
		)
		return self._focus_state

	@staticmethod
	def _focus_timeout_error(
		target_id: str | None,
		phase: FocusPhase,
		*,
		target_type: str | None = None,
		message: str = 'Focus transition timed out.',
		created_target_id: str | None = None,
	) -> FocusTargetError:
		return FocusTargetError(
			message,
			code=FocusTargetCode.TIMEOUT,
			phase=phase,
			target_id=target_id,
			target_type=target_type,
			created_target_id=created_target_id,
		)

	async def _await_focus_operation(
		self,
		awaitable: Awaitable[Any],
		budget: FocusTransitionBudget,
		*,
		target_id: str | None,
		phase: FocusPhase,
		target_type: str | None = None,
		created_target_id: str | None = None,
	) -> Any:
		remaining = budget.remaining()
		if remaining <= 0:
			if inspect.iscoroutine(awaitable):
				awaitable.close()
			raise self._focus_timeout_error(
				target_id,
				phase,
				target_type=target_type,
				created_target_id=created_target_id,
			)
		try:
			return await asyncio.wait_for(awaitable, timeout=remaining)
		except TimeoutError as error:
			raise self._focus_timeout_error(
				target_id,
				phase,
				target_type=target_type,
				created_target_id=created_target_id,
			) from error

	async def _validate_focus_target_locked(
		self,
		target_id: str,
		budget: FocusTransitionBudget,
		*,
		created_target_id: str | None = None,
	) -> dict[str, Any]:
		root = self._cdp_client_root
		if root is None:
			raise FocusTargetError(
				'CDP client is not initialized.',
				code=FocusTargetCode.TARGET_MISSING,
				phase=self._focus_state.phase,
				target_id=target_id,
				created_target_id=created_target_id,
			)
		try:
			result = await self._await_focus_operation(
				root.send.Target.getTargetInfo(params={'targetId': target_id}),
				budget,
				target_id=target_id,
				phase=self._focus_state.phase,
				created_target_id=created_target_id,
			)
		except FocusTargetError:
			raise
		except Exception as error:
			raise FocusTargetError(
				f'Target {target_id[-8:]} is not available: {type(error).__name__}: {error}',
				code=FocusTargetCode.TARGET_MISSING,
				phase=self._focus_state.phase,
				target_id=target_id,
				created_target_id=created_target_id,
			) from error
		target_info = result.get('targetInfo') if isinstance(result, dict) else None
		if isinstance(target_info, dict) and target_info.get('targetId') not in (None, target_id):
			targets = await self._await_focus_operation(
				root.send.Target.getTargets(),
				budget,
				target_id=target_id,
				phase=self._focus_state.phase,
				created_target_id=created_target_id,
			)
			target_info = next(
				(item for item in targets.get('targetInfos', []) if item.get('targetId') == target_id),
				None,
			)
		if not isinstance(target_info, dict):
			raise FocusTargetError(
				f'Target {target_id[-8:]} returned no target information.',
				code=FocusTargetCode.TARGET_MISSING,
				phase=self._focus_state.phase,
				target_id=target_id,
			)
		target_type = str(target_info.get('type') or 'unknown')
		if target_type != 'page':
			raise FocusTargetError(
				f'Target {target_id[-8:]} is a {target_type}, not a page.',
				code=FocusTargetCode.NOT_A_PAGE,
				phase=self._focus_state.phase,
				target_id=target_id,
				target_type=target_type,
			)
		record = self._tab_registry.get(target_id)
		if record is not None and record.state in (TabState.CLOSING, TabState.CLOSED, TabState.CRASHED):
			raise FocusTargetError(
				f'Target {target_id[-8:]} is {record.state.value}.',
				code=FocusTargetCode.TAB_NOT_LIVE,
				phase=self._focus_state.phase,
				target_id=target_id,
				target_type=target_type,
			)
		return target_info

	async def _get_focus_session_locked(
		self,
		target_id: str,
		budget: FocusTransitionBudget,
		*,
		created_target_id: str | None = None,
	) -> CDPSession:
		manager = self._session_manager
		if manager is None:
			raise FocusTargetError(
				'CDP session manager is not initialized.',
				code=FocusTargetCode.TARGET_MISSING,
				phase=self._focus_state.phase,
				target_id=target_id,
				created_target_id=created_target_id,
			)
		wait_deadline = min(budget.deadline, asyncio.get_running_loop().time() + 2.0)
		for _attempt in range(20):
			try:
				session = await manager.get_live_session_for_target(target_id, budget=budget)
			except TimeoutError as error:
				raise FocusTargetError(
					f'Focus session lookup timed out for target {target_id[-8:]}.',
					code=FocusTargetCode.TIMEOUT,
					phase=self._focus_state.phase,
					target_id=target_id,
					created_target_id=created_target_id,
				) from error
			if session is not None:
				return session
			remaining = wait_deadline - asyncio.get_running_loop().time()
			if remaining <= 0:
				raise FocusTargetError(
					f'No live session is available for target {target_id[-8:]}.',
					code=FocusTargetCode.TARGET_MISSING,
					phase=self._focus_state.phase,
					target_id=target_id,
					created_target_id=created_target_id,
				)
			await asyncio.sleep(min(0.1, remaining))
		raise FocusTargetError(
			f'No live session is available for target {target_id[-8:]}.',
			code=FocusTargetCode.TARGET_MISSING,
			phase=self._focus_state.phase,
			target_id=target_id,
			created_target_id=created_target_id,
		)

	@staticmethod
	def _is_missing_cdp_session_error(error: Exception) -> bool:
		"""Return True for a stale or detached exact CDP session."""
		text = str(error).lower()
		return any(
			marker in text
			for marker in (
				'session with given id not found',
				'session not found',
				'no session with given id',
				'target closed',
				'detached from target',
				'-32001',
			)
		)

	async def _focus_exact_session_is_live(
		self,
		target_id: str | None,
		session_id: str | None,
		budget: FocusTransitionBudget,
		*,
		phase: FocusPhase,
		use_active_ids: bool = False,
	) -> bool:
		"""Return True only when the exact committed session is still active."""
		if target_id is None or session_id is None:
			return False
		record = self._tab_registry.get(target_id)
		if record is not None and record.state in (TabState.CLOSING, TabState.CLOSED, TabState.CRASHED):
			return False
		manager = self._session_manager
		if manager is None:
			return False
		try:
			if use_active_ids:
				active_ids = await self._await_focus_operation(
					manager.active_session_ids(target_id, budget=budget),
					budget,
					target_id=target_id,
					phase=phase,
					target_type='page',
				)
				return session_id in active_ids
			live = await self._await_focus_operation(
				manager.get_live_session_for_target(target_id, budget=budget),
				budget,
				target_id=target_id,
				phase=phase,
				target_type='page',
			)
		except FocusTargetError:
			raise
		except Exception:
			return False
		return live is not None and live.target_id == target_id and live.session_id == session_id

	async def _focus_checkpoint_liveness(
		self,
		checkpoint: _FocusCheckpoint,
		budget: FocusTransitionBudget,
	) -> bool | None:
		"""Return live, dead, or unknown for a prior focus checkpoint."""
		snapshot = checkpoint.snapshot
		if snapshot.phase is not FocusPhase.FOCUSED:
			return True
		if snapshot.target_id is None or snapshot.session_id is None:
			return False
		record = self._tab_registry.get(snapshot.target_id)
		if record is not None and record.state in (TabState.CLOSING, TabState.CLOSED, TabState.CRASHED):
			return False
		manager = self._session_manager
		if manager is None:
			return None
		try:
			active_ids = await self._await_focus_operation(
				manager.active_session_ids(snapshot.target_id, budget=budget),
				budget,
				target_id=snapshot.target_id,
				phase=snapshot.phase,
				target_type='page',
			)
		except Exception:
			return None
		return snapshot.session_id in active_ids

	async def _restore_or_lose_focus_locked(
		self,
		checkpoint: _FocusCheckpoint,
		budget: FocusTransitionBudget,
	) -> FocusChangeReason | None:
		"""Restore a still-live prior focus, or publish loss and request recovery."""
		rollback_budget = budget if budget.remaining() > 0 else FocusTransitionBudget.start(CLOSE_RECONCILIATION_SECONDS)
		checkpoint_liveness = await self._focus_checkpoint_liveness(checkpoint, rollback_budget)
		previous_target_id = checkpoint.snapshot.target_id
		if checkpoint_liveness is True:
			self._restore_focus_locked(checkpoint)
			return None
		if checkpoint_liveness is None:
			self._focus_lost_locked(
				FocusTargetError(
					'The previous focus target liveness could not be verified.',
					code=FocusTargetCode.TARGET_LIVENESS_UNKNOWN,
					phase=self._focus_state.phase,
					target_id=previous_target_id,
					target_type='page',
				),
				FocusChangeReason.LIVENESS_RECOVERY,
				previous_target_id=previous_target_id,
			)
			return None
		reason = (
			self._focus_reason_for_detach(previous_target_id)
			if previous_target_id
			else FocusChangeReason.DETACH_RECOVERY
		)
		self._focus_lost_locked(
			FocusTargetError(
				'The previous focus target is no longer live.',
				code=FocusTargetCode.TARGET_DETACHED,
				phase=self._focus_state.phase,
				target_id=previous_target_id,
				target_type='page',
			),
			reason,
			previous_target_id=previous_target_id,
		)
		return reason

	async def _select_focus_target_locked(
		self,
		budget: FocusTransitionBudget,
		*,
		excluded_target_id: str | None,
		create_if_missing: bool,
	) -> tuple[str, str | None]:
		try:
			pages = await self._await_focus_operation(
				self._cdp_get_all_pages(),
				budget,
				target_id=excluded_target_id,
				phase=self._focus_state.phase,
			)
		except FocusTargetError:
			raise
		except Exception as error:
			raise FocusTargetError(
				f'Could not list browser pages: {type(error).__name__}: {error}',
				code=FocusTargetCode.NO_PAGE_AVAILABLE,
				phase=self._focus_state.phase,
				target_id=excluded_target_id,
			) from error
		candidates = self._tab_registry.live_page_candidates(
			pages,
			exclude_target_id=excluded_target_id,
		)
		if candidates:
			return candidates[-1].target_id, None
		if not create_if_missing:
			raise FocusTargetError(
				'No live page target is available.',
				code=FocusTargetCode.NO_PAGE_AVAILABLE,
				phase=self._focus_state.phase,
				target_id=excluded_target_id,
			)
		try:
			target_id = await self._await_focus_operation(
				self._cdp_create_new_page('about:blank'),
				budget,
				target_id=excluded_target_id,
				phase=self._focus_state.phase,
			)
		except FocusTargetError:
			raise
		except Exception as error:
			raise FocusTargetError(
				f'Could not create a recovery page: {type(error).__name__}: {error}',
				code=FocusTargetCode.NO_PAGE_AVAILABLE,
				phase=self._focus_state.phase,
				target_id=excluded_target_id,
			) from error
		self._tab_registry.register(target_id, url='about:blank', state=TabState.OPEN)
		return target_id, target_id

	async def _activate_focus_locked(self, session: CDPSession, budget: FocusTransitionBudget) -> None:
		root = self._cdp_client_root
		if root is None:
			raise FocusTargetError(
				'CDP client is not initialized.',
				code=FocusTargetCode.ACTIVATION_FAILED,
				phase=self._focus_state.phase,
				target_id=session.target_id,
				target_type='page',
			)
		try:
			await self._await_focus_operation(
				root.send.Target.activateTarget(params={'targetId': session.target_id}),
				budget,
				target_id=session.target_id,
				phase=self._focus_state.phase,
				target_type='page',
			)
		except FocusTargetError:
			raise
		except Exception as error:
			raise FocusTargetError(
				f'Could not activate target {session.target_id[-8:]}: {type(error).__name__}: {error}',
				code=FocusTargetCode.ACTIVATION_FAILED,
				phase=self._focus_state.phase,
				target_id=session.target_id,
				target_type='page',
			) from error

	async def _probe_focus_locked(self, session: CDPSession, budget: FocusTransitionBudget) -> None:
		try:
			result = await self._await_focus_operation(
				session.cdp_client.send.Runtime.evaluate(
					params={'expression': '1 + 1', 'returnByValue': True},
					session_id=session.session_id,
				),
				budget,
				target_id=session.target_id,
				phase=self._focus_state.phase,
				target_type='page',
			)
		except FocusTargetError:
			raise
		except Exception as error:
			raise FocusTargetError(
				f'Target {session.target_id[-8:]} is unresponsive: {type(error).__name__}: {error}',
				code=FocusTargetCode.UNRESPONSIVE,
				phase=self._focus_state.phase,
				target_id=session.target_id,
				target_type='page',
			) from error
		if not isinstance(result, dict) or result.get('result', {}).get('value') != 2:
			raise FocusTargetError(
				f'Target {session.target_id[-8:]} failed the focus responsiveness probe.',
				code=FocusTargetCode.UNRESPONSIVE,
				phase=self._focus_state.phase,
				target_id=session.target_id,
				target_type='page',
			)

	async def _focus_transition_to_target(
		self,
		target_id: str | None,
		*,
		reason: FocusChangeReason,
		budget: FocusTransitionBudget,
		excluded_target_id: str | None = None,
		create_if_missing: bool = False,
		activate: bool = True,
		probe: bool = True,
		failure_mode: Literal['restore', 'lost'] = 'restore',
		phase: FocusPhase = FocusPhase.SWITCHING,
		commit_guard: Callable[[], None] | None = None,
		expected_generation: int | None = None,
	) -> FocusTransitionResult:
		"""Run one serialized focus transaction and commit at the end."""
		remaining = budget.remaining()
		if remaining <= 0:
			raise self._focus_timeout_error(target_id, self._focus_state.phase)
		try:
			await asyncio.wait_for(self._focus_lock.acquire(), timeout=remaining)
		except TimeoutError as error:
			raise self._focus_timeout_error(target_id, self._focus_state.phase) from error

		checkpoint = self._focus_checkpoint_locked()
		if expected_generation is not None and checkpoint.snapshot.generation != expected_generation:
			self._focus_lock.release()
			raise FocusTargetError(
				'Focus changed before the transition could start.',
				code=FocusTargetCode.STALE_TRANSITION,
				phase=checkpoint.snapshot.phase,
				target_id=checkpoint.snapshot.target_id,
			)
		focus_epoch = self._focus_epoch
		lifecycle_generation = self._lifecycle.generation
		created_target_id: str | None = None
		recovery_after_failure: tuple[str, FocusChangeReason] | None = None
		try:
			self._begin_focus_change_locked(reason, target_id, phase)
			if target_id is None:
				target_id, created_target_id = await self._select_focus_target_locked(
					budget,
					excluded_target_id=excluded_target_id,
					create_if_missing=create_if_missing,
				)
				self._focus_state = FocusSnapshot(
					phase=phase,
					target_id=target_id if phase is FocusPhase.SWITCHING else None,
					session_id=None,
					previous_target_id=self._focus_state.previous_target_id,
					generation=self._focus_state.generation,
					reason=reason,
					last_error=None,
					updated_at=time.time(),
				)
			await self._validate_focus_target_locked(target_id, budget, created_target_id=created_target_id)
			session = await self._get_focus_session_locked(target_id, budget, created_target_id=created_target_id)
			if activate:
				await self._activate_focus_locked(session, budget)
			if probe:
				await self._probe_focus_locked(session, budget)
			record = self._tab_registry.get(target_id)
			if record is not None and record.state in (TabState.CLOSING, TabState.CLOSED, TabState.CRASHED):
				raise FocusTargetError(
					f'Target {target_id[-8:]} is no longer live before the focus commit.',
					code=FocusTargetCode.TAB_NOT_LIVE,
					phase=self._focus_state.phase,
					target_id=target_id,
					target_type='page',
					created_target_id=created_target_id,
				)
			if not await self._focus_exact_session_is_live(
				session.target_id,
				session.session_id,
				budget,
				phase=self._focus_state.phase,
			):
				raise FocusTargetError(
					f'Target {target_id[-8:]} detached before the focus commit.',
					code=FocusTargetCode.TARGET_DETACHED,
					phase=self._focus_state.phase,
					target_id=target_id,
					target_type='page',
					created_target_id=created_target_id,
				)
			if commit_guard is not None:
				commit_guard()
			if self._focus_epoch != focus_epoch or self._lifecycle.generation != lifecycle_generation:
				raise FocusTargetError(
					'Reset, stop, or browser replacement changed the focus state during the transition.',
					code=FocusTargetCode.STALE_TRANSITION,
					phase=self._focus_state.phase,
					target_id=target_id,
					target_type='page',
					created_target_id=created_target_id,
				)
			snapshot = self._set_agent_focus_locked(
				session,
				reason,
				previous_snapshot=checkpoint.snapshot,
			)
			return FocusTransitionResult(
				session=session,
				snapshot=snapshot,
				created_target_id=created_target_id,
			)
		except asyncio.CancelledError:
			if self._focus_epoch == focus_epoch and self._lifecycle.generation == lifecycle_generation:
				if failure_mode == 'lost':
					self._focus_lost_locked(
						FocusTargetError(
							'Focus transition was cancelled.',
							code=FocusTargetCode.RECOVERY_FAILED,
							phase=self._focus_state.phase,
							target_id=target_id,
							created_target_id=created_target_id,
						),
						reason,
					)
				else:
					recovery_reason = await self._restore_or_lose_focus_locked(checkpoint, budget)
					if recovery_reason is not None and checkpoint.snapshot.target_id is not None:
						recovery_after_failure = (checkpoint.snapshot.target_id, recovery_reason)
			raise
		except FocusTargetError as error:
			if created_target_id and error.created_target_id is None:
				error.created_target_id = created_target_id
			if self._focus_epoch != focus_epoch or self._lifecycle.generation != lifecycle_generation:
				if error.code is FocusTargetCode.STALE_TRANSITION:
					raise
				raise FocusTargetError(
					'Reset, stop, or browser replacement changed the focus state during the transition.',
					code=FocusTargetCode.STALE_TRANSITION,
					phase=self._focus_state.phase,
					target_id=target_id,
					target_type='page',
					created_target_id=created_target_id,
				) from error
			if error.code is FocusTargetCode.STALE_TRANSITION:
				raise
			if failure_mode == 'lost':
				self._focus_lost_locked(error, reason)
			else:
				recovery_reason = await self._restore_or_lose_focus_locked(checkpoint, budget)
				if recovery_reason is not None and checkpoint.snapshot.target_id is not None:
					recovery_after_failure = (checkpoint.snapshot.target_id, recovery_reason)
			raise
		except Exception as error:
			focus_error = FocusTargetError(
				f'Focus transition failed: {type(error).__name__}: {error}',
				code=FocusTargetCode.RECOVERY_FAILED,
				phase=self._focus_state.phase,
				target_id=target_id,
				created_target_id=created_target_id,
			)
			if self._focus_epoch != focus_epoch or self._lifecycle.generation != lifecycle_generation:
				raise FocusTargetError(
					'Reset, stop, or browser replacement changed the focus state during the transition.',
					code=FocusTargetCode.STALE_TRANSITION,
					phase=self._focus_state.phase,
					target_id=target_id,
					target_type='page',
					created_target_id=created_target_id,
				) from error
			if failure_mode == 'lost':
				self._focus_lost_locked(focus_error, reason)
			else:
				recovery_reason = await self._restore_or_lose_focus_locked(checkpoint, budget)
				if recovery_reason is not None and checkpoint.snapshot.target_id is not None:
					recovery_after_failure = (checkpoint.snapshot.target_id, recovery_reason)
			raise focus_error from error
		finally:
			self._focus_lock.release()
			if recovery_after_failure is not None:
				self._schedule_focus_recovery(
					*recovery_after_failure,
					expected_generation=self._focus_state.generation,
				)

	async def _set_agent_focus(
		self,
		session: CDPSession,
		reason: FocusChangeReason,
		*,
		expected_generation: int | None = None,
	) -> FocusSnapshot:
		"""Commit a live page session through the focus transaction."""
		result = await self._focus_transition_to_target(
			session.target_id,
			reason=reason,
			budget=FocusTransitionBudget.start(),
			activate=False,
			probe=False,
			expected_generation=expected_generation,
		)
		return result.snapshot

	def _focus_observer_done(self, task: asyncio.Task[Any]) -> None:
		self._focus_observer_tasks.discard(task)
		if task.cancelled():
			return
		try:
			task.exception()
		except asyncio.CancelledError:
			return

	def _cancel_focus_observers(self) -> None:
		for task in list(self._focus_observer_tasks):
			if not task.done():
				task.cancel()
		self._focus_observer_tasks.clear()

	def _schedule_navigation_complete_observation(self, target_id: str, url: str) -> None:
		# Dispatch from a clean context so the observation is not a child of
		# the focus event and cannot delay focus completion.
		event = contextvars.Context().run(
			self.event_bus.dispatch,
			NavigationCompleteEvent(target_id=target_id, url=url),
		)

		async def observe() -> None:
			try:
				await asyncio.wait_for(
					event.event_result(
						timeout=FOCUS_NAVIGATION_OBSERVER_SECONDS,
						raise_if_any=False,
						raise_if_none=False,
					),
					timeout=FOCUS_NAVIGATION_OBSERVER_SECONDS,
				)
			except (TimeoutError, asyncio.CancelledError):
				return

		task = asyncio.create_task(observe())
		self._focus_observer_tasks.add(task)
		task.add_done_callback(self._focus_observer_done)

	@staticmethod
	def _focus_event_timeout(event_type: type[Any], budget: FocusTransitionBudget) -> float:
		remaining = budget.remaining()
		default = getattr(event_type, 'model_fields', {}).get('event_timeout')
		default_value = getattr(default, 'default', None) if default is not None else None
		if isinstance(default_value, (int, float)):
			return min(remaining, float(default_value))
		return remaining

	async def _await_event_result(
		self,
		event: Any,
		budget: FocusTransitionBudget,
		*,
		raise_if_any: bool,
		raise_if_none: bool,
	) -> Any:
		"""Await one child event without deadlocking the handler that queued it."""
		await asyncio.wait_for(event, timeout=max(0.001, budget.remaining()))
		return await event.event_result(
			timeout=max(0.001, budget.remaining()),
			raise_if_any=raise_if_any,
			raise_if_none=raise_if_none,
		)

	async def _notify_focus_changed(
		self,
		target_id: str,
		url: str,
		reason: FocusChangeReason,
		budget: FocusTransitionBudget,
		snapshot: FocusSnapshot,
	) -> None:
		"""Dispatch and await one generation-stamped focus observation."""
		remaining = budget.remaining()
		if remaining <= 0:
			raise self._focus_timeout_error(target_id, FocusPhase.FOCUSED)
		event_timeout = self._focus_event_timeout(AgentFocusChangedEvent, budget)
		event = self.event_bus.dispatch(
			AgentFocusChangedEvent(
				target_id=target_id,
				url=url,
				focus_generation=snapshot.generation,
				focus_session_id=snapshot.session_id,
				focus_reason=reason,
				focus_budget_seconds=remaining,
				focus_deadline_monotonic=budget.deadline,
				event_timeout=max(0.001, event_timeout),
			)
		)
		try:
			await self._await_event_result(
				event,
				budget,
				raise_if_any=True,
				raise_if_none=False,
			)
		except asyncio.CancelledError:
			raise
		except Exception as error:
			raise FocusTargetError(
				f'Focus committed to target {target_id[-8:]}, but its observation failed: '
				f'{type(error).__name__}: {error}',
				code=FocusTargetCode.OBSERVATION_FAILED,
				phase=FocusPhase.FOCUSED,
				target_id=target_id,
				target_type='page',
			) from error

	async def _initialize_focus_page(self, target_id: str, budget: FocusTransitionBudget) -> None:
		"""Dispatch TabCreatedEvent for a page created by focus recovery."""
		try:
			event = self.event_bus.dispatch(TabCreatedEvent(target_id=target_id, url='about:blank'))
			await self._await_event_result(
				event,
				budget,
				raise_if_any=True,
				raise_if_none=False,
			)
		except Exception as error:
			self.logger.warning(
				f'Focus recovery page {target_id[-8:]} was created, but TabCreatedEvent setup failed: '
				f'{type(error).__name__}: {error}'
			)

	def _focus_reason_for_detach(self, target_id: str) -> FocusChangeReason:
		record = self._tab_registry.get(target_id)
		if record is not None and record.state is TabState.CRASHED:
			return FocusChangeReason.CRASH_RECOVERY
		if record is not None and record.state in (TabState.CLOSING, TabState.CLOSED):
			return FocusChangeReason.CLOSE_REPAIR
		return FocusChangeReason.DETACH_RECOVERY

	async def _handle_attached_session(self, target_id: str, session: CDPSession) -> None:
		"""Rebind focus when an attach changes its exact session pair."""
		snapshot = self._focus_state
		if (
			snapshot.phase is FocusPhase.FOCUSED
			and snapshot.target_id == target_id
			and snapshot.session_id != session.session_id
		):
			await self._set_agent_focus(
				session,
				FocusChangeReason.SESSION_REBIND,
				expected_generation=snapshot.generation,
			)

	async def _handle_detached_session(
		self,
		target_id: str,
		session_id: str,
		remaining_session_ids: tuple[str, ...],
	) -> None:
		"""Repair focus after one exact CDP session detaches."""
		snapshot = self._focus_state
		if (
			snapshot.phase is not FocusPhase.FOCUSED
			or snapshot.target_id != target_id
			or snapshot.session_id != session_id
		):
			return
		if remaining_session_ids:
			manager = self._session_manager
			if manager is not None:
				session = await manager.get_live_session_for_target(target_id)
				if session is not None and session.session_id != session_id:
					await self._set_agent_focus(
						session,
						FocusChangeReason.SESSION_REBIND,
						expected_generation=snapshot.generation,
					)
					return
		reason = self._focus_reason_for_detach(target_id)
		try:
			await self._recover_agent_focus(
				target_id,
				reason=reason,
				budget=FocusTransitionBudget.start(),
				expected_generation=snapshot.generation,
			)
		except FocusTargetError as error:
			self._report_focus_recovery_failure(target_id, error)

	async def _recover_agent_focus(
		self,
		excluded_target_id: str | None,
		*,
		reason: FocusChangeReason,
		budget: FocusTransitionBudget,
		expected_generation: int | None = None,
	) -> FocusSnapshot:
		"""Select, validate, and commit one recovery page within one budget."""
		created_target_ids: list[str] = []
		starting_generation = expected_generation if expected_generation is not None else self._focus_state.generation
		try:
			result = await self._focus_transition_to_target(
				None,
				reason=reason,
				budget=budget,
				excluded_target_id=excluded_target_id,
				create_if_missing=True,
				activate=True,
				probe=True,
				failure_mode='lost',
				phase=FocusPhase.RECOVERING,
				expected_generation=starting_generation,
			)
		except FocusTargetError as error:
			if error.code is FocusTargetCode.STALE_TRANSITION:
				raise
			if error.created_target_id:
				created_target_ids.append(error.created_target_id)
			if budget.expired():
				for target_id in created_target_ids:
					await self._initialize_focus_page(target_id, budget)
				raise
			recovery_root = self._cdp_client_root
			recovery_epoch = self._focus_epoch
			recovery_lifecycle_generation = self._lifecycle.generation
			retry_focus_generation = self._focus_state.generation
			try:
				retry_target_id = await self._await_focus_operation(
					self._cdp_create_new_page('about:blank'),
					budget,
					target_id=excluded_target_id,
					phase=FocusPhase.RECOVERING,
				)
			except FocusTargetError as create_error:
				create_error.created_target_id = created_target_ids[0] if created_target_ids else None
				raise create_error from error
			except asyncio.CancelledError:
				raise
			except Exception as create_error:
				raise FocusTargetError(
					f'Focus recovery could not create an emergency page: '
					f'{type(create_error).__name__}: {create_error}',
					code=FocusTargetCode.RECOVERY_FAILED,
					phase=self._focus_state.phase,
					target_id=excluded_target_id,
					created_target_id=created_target_ids[0] if created_target_ids else None,
				) from create_error
			if (
				self._cdp_client_root is not recovery_root
				or self._focus_epoch != recovery_epoch
				or self._lifecycle.generation != recovery_lifecycle_generation
				or self._focus_state.generation != retry_focus_generation
			):
				raise FocusTargetError(
					'Browser or focus state changed while the recovery page was being created.',
					code=FocusTargetCode.STALE_TRANSITION,
					phase=self._focus_state.phase,
					target_id=excluded_target_id,
					created_target_id=retry_target_id,
				) from error
			self._tab_registry.register(retry_target_id, url='about:blank', state=TabState.OPEN)
			created_target_ids.append(retry_target_id)
			if self._focus_state.phase is FocusPhase.FOCUSED:
				raise FocusTargetError(
					'Focus changed while the recovery page was being created.',
					code=FocusTargetCode.STALE_TRANSITION,
					phase=self._focus_state.phase,
					target_id=self._focus_state.target_id,
				) from error
			retry_generation = self._focus_state.generation
			try:
				result = await self._focus_transition_to_target(
					retry_target_id,
					reason=reason,
					budget=budget,
					activate=True,
					probe=True,
					failure_mode='lost',
					phase=FocusPhase.RECOVERING,
					expected_generation=retry_generation,
				)
			except FocusTargetError as retry_error:
				if retry_error.created_target_id and retry_error.created_target_id not in created_target_ids:
					created_target_ids.append(retry_error.created_target_id)
				for target_id in created_target_ids:
					await self._initialize_focus_page(target_id, budget)
				if retry_error.code is FocusTargetCode.RECOVERY_FAILED:
					raise
				raise FocusTargetError(
					f'Focus recovery failed: {retry_error}',
					code=FocusTargetCode.RECOVERY_FAILED,
					phase=FocusPhase.LOST,
					target_id=excluded_target_id,
					created_target_id=retry_error.created_target_id,
				) from retry_error
		if result.created_target_id and result.created_target_id not in created_target_ids:
			created_target_ids.append(result.created_target_id)
		for target_id in created_target_ids:
			await self._initialize_focus_page(target_id, budget)
		try:
			await self._notify_focus_changed(
				result.session.target_id,
				result.session.url or 'about:blank',
				reason,
				budget,
				result.snapshot,
			)
		except FocusTargetError:
			raise
		return result.snapshot

	def _report_focus_recovery_failure(self, target_id: str, error: FocusTargetError) -> None:
		"""Publish one bounded browser error for a failed background recovery."""
		self.logger.warning(f'Focus recovery failed for target {target_id[-8:]}: {error.code.value}: {error}')
		try:
			self.event_bus.dispatch(
				BrowserErrorEvent(
					error_type='FocusRecoveryError',
					message=f'Focus recovery failed for target {target_id[-8:]}.',
					details=error.to_snapshot().to_dict(),
				)
			)
		except Exception:
			pass

	def _schedule_close_reconciliation(
		self,
		target_id: str,
		*,
		expected_generation: int | None = None,
		expected_close_attempt: int | None = None,
	) -> None:
		"""Reconcile an uncertain close outside the action deadline."""
		focus_generation = self._focus_state.generation if expected_generation is None else expected_generation
		if expected_close_attempt is None:
			record = self._tab_registry.get(target_id)
			expected_close_attempt = record.close_attempt if record is not None else None

		async def reconcile() -> None:
			budget = FocusTransitionBudget.start(CLOSE_RECONCILIATION_SECONDS)
			try:
				await self._reconcile_uncertain_close(
					target_id,
					budget,
					expected_generation=focus_generation,
					expected_close_attempt=expected_close_attempt,
				)
			except asyncio.CancelledError:
				raise
			except Exception as error:
				self.logger.warning(
					f'Close reconciliation failed for target {target_id[-8:]}: '
					f'{type(error).__name__}: {error}'
				)

		task = asyncio.create_task(reconcile())
		self._focus_observer_tasks.add(task)
		task.add_done_callback(self._focus_observer_done)

	async def _reconcile_uncertain_close(
		self,
		target_id: str,
		budget: FocusTransitionBudget,
		*,
		expected_generation: int,
		expected_close_attempt: int | None,
	) -> None:
		"""Resolve CLOSING to OPEN or CLOSED without extending the close action."""
		record = self._tab_registry.get(target_id)
		if (
			record is None
			or record.state in (TabState.OPEN, TabState.OPENING)
			or (expected_close_attempt is not None and record.close_attempt != expected_close_attempt)
		):
			return
		if budget.remaining() <= 0:
			return
		liveness_seconds = max(0.001, budget.remaining() / 2)
		liveness_budget = FocusTransitionBudget(
			min(budget.deadline, asyncio.get_running_loop().time() + liveness_seconds)
		)
		try:
			still_open = await asyncio.wait_for(
				self._target_liveness(target_id, liveness_budget),
				timeout=max(0.001, liveness_budget.remaining()),
			)
		except TimeoutError:
			still_open = None
		record = self._tab_registry.get(target_id)
		if record is None:
			return
		if expected_close_attempt is not None and record.close_attempt != expected_close_attempt:
			return
		if still_open is True:
			if record.state is TabState.CLOSING:
				self._tab_registry.update(target_id, state=TabState.OPEN, allow_reopen=True)
			await self._restore_focus_after_close_reopen(target_id, budget)
			return
		focus_loss_applied = await self._mark_focus_lost_for_uncertain_close(
			target_id,
			budget,
			expected_generation=expected_generation,
			expected_close_attempt=expected_close_attempt,
		)
		if not focus_loss_applied:
			return
		if still_open is False:
			if record.state is TabState.CLOSING:
				self._tab_registry.mark_closed(target_id)
			try:
				await self._repair_focus_after_closed_tab(target_id, budget)
			except FocusTargetError as error:
				self.logger.warning(
					f'Focus repair after confirmed close absence failed: {error.code.value}: {error}'
				)

	async def _mark_focus_lost_for_uncertain_close(
		self,
		target_id: str,
		budget: FocusTransitionBudget,
		*,
		expected_generation: int,
		expected_close_attempt: int | None,
	) -> bool:
		"""Stop reporting focus on a target whose close outcome is unknown.

		Returns False when the close attempt is no longer the current owner.
		"""
		remaining = budget.remaining()
		if remaining <= 0:
			return False
		try:
			await asyncio.wait_for(self._focus_lock.acquire(), timeout=remaining)
		except TimeoutError:
			return False
		try:
			record = self._tab_registry.get(target_id)
			if (
				record is None
				or record.state in (TabState.OPEN, TabState.OPENING)
				or (expected_close_attempt is not None and record.close_attempt != expected_close_attempt)
			):
				return False
			snapshot = self._focus_state
			if (
				snapshot.phase is FocusPhase.FOCUSED
				and snapshot.target_id == target_id
				and snapshot.generation == expected_generation
			):
				self._focus_lost_locked(
					FocusTargetError(
						f'Close outcome for target {target_id[-8:]} is unknown.',
						code=FocusTargetCode.TARGET_DETACHED,
						phase=FocusPhase.FOCUSED,
						target_id=target_id,
						target_type='page',
					),
					FocusChangeReason.CLOSE_REPAIR,
					previous_target_id=target_id,
				)
			return True
		finally:
			self._focus_lock.release()

	async def _ensure_closed_target_focus_lost(
		self,
		target_id: str,
		snapshot: FocusSnapshot,
		error: FocusTargetError,
	) -> None:
		"""Publish lost immediately when close repair timed out before its lock."""
		if self._focus_lock.locked():
			self._schedule_close_reconciliation(target_id, expected_generation=snapshot.generation)
			return
		await self._focus_lock.acquire()
		try:
			current = self._focus_state
			if (
				current.phase is FocusPhase.FOCUSED
				and current.target_id == target_id
				and current.generation == snapshot.generation
			):
				self._focus_lost_locked(
					error,
					FocusChangeReason.CLOSE_REPAIR,
					previous_target_id=target_id,
				)
		finally:
			self._focus_lock.release()

	def _schedule_reopened_focus_restore(self, target_id: str, *, expected_generation: int) -> None:
		"""Retry one confirmed reopen on a separate bounded deadline."""

		async def restore() -> None:
			snapshot = self._focus_state
			if (
				snapshot.phase is not FocusPhase.LOST
				or snapshot.previous_target_id != target_id
				or snapshot.generation != expected_generation
			):
				return
			budget = FocusTransitionBudget.start(CLOSE_RECONCILIATION_SECONDS)
			await self._restore_focus_after_close_reopen(
				target_id,
				budget,
				reschedule_on_timeout=False,
			)

		task = asyncio.create_task(restore())
		self._focus_observer_tasks.add(task)
		task.add_done_callback(self._focus_observer_done)

	async def _restore_focus_after_close_reopen(
		self,
		target_id: str,
		budget: FocusTransitionBudget,
		*,
		reschedule_on_timeout: bool = True,
	) -> None:
		"""Restore focus when a timed-out close proves the target is still open."""
		snapshot = self._focus_state
		if snapshot.phase is not FocusPhase.LOST or snapshot.previous_target_id != target_id:
			return
		try:
			result = await self._focus_transition_to_target(
				target_id,
				reason=FocusChangeReason.CLOSE_REPAIR,
				budget=budget,
				activate=True,
				probe=True,
				failure_mode='lost',
				phase=FocusPhase.RECOVERING,
				expected_generation=snapshot.generation,
			)
			await self._notify_focus_changed(
				result.session.target_id,
				result.session.url or 'about:blank',
				FocusChangeReason.CLOSE_REPAIR,
				budget,
				result.snapshot,
			)
		except FocusTargetError as error:
			record = self._tab_registry.get(target_id)
			current = self._focus_state
			if (
				reschedule_on_timeout
				and error.code is FocusTargetCode.TIMEOUT
				and record is not None
				and record.state is TabState.OPEN
				and current.phase is FocusPhase.LOST
				and current.previous_target_id == target_id
			):
				self._schedule_reopened_focus_restore(
					target_id,
					expected_generation=current.generation,
				)
				return
			self.logger.warning(
				f'Could not restore focus after a reopened close: {error.code.value}: {error}'
			)

	def _schedule_focus_recovery(
		self,
		target_id: str,
		reason: FocusChangeReason,
		*,
		expected_generation: int | None = None,
	) -> None:
		"""Run one recovery after the current focus lock holder exits."""

		async def recover() -> None:
			try:
				await self._recover_agent_focus(
					target_id,
					reason=reason,
					budget=FocusTransitionBudget.start(),
					expected_generation=expected_generation,
				)
			except FocusTargetError as error:
				self._report_focus_recovery_failure(target_id, error)

		task = asyncio.create_task(recover())
		self._focus_observer_tasks.add(task)
		task.add_done_callback(self._focus_observer_done)

	def _focus_lost_on_reset(self) -> None:
		current = self._focus_state
		if current.phase not in (FocusPhase.FOCUSED, FocusPhase.SWITCHING, FocusPhase.RECOVERING):
			return
		previous_target_id = (
			current.previous_target_id
			if current.phase is FocusPhase.SWITCHING
			else current.target_id or current.previous_target_id
		)
		self._focus_lost_locked(
			FocusTargetError(
				'Browser focus was cleared by reset or stop.',
				code=FocusTargetCode.TARGET_DETACHED,
				phase=current.phase,
				target_id=previous_target_id,
			),
			FocusChangeReason.RESET,
			previous_target_id=previous_target_id,
		)

	def seed_generation(self, generation: int) -> None:
		"""Keep the lifecycle generation monotonic across replacement sessions."""
		self._lifecycle.seed_generation(generation)

	async def mark_degraded(self, reason: str) -> None:
		"""Record a non-fatal browser health problem in the lifecycle state."""
		if self._lifecycle.phase in (LifecyclePhase.READY, LifecyclePhase.RECOVERING):
			try:
				self._lifecycle.transition(LifecyclePhase.DEGRADED, reason=reason)
			except LifecycleTransitionError:
				pass

	def _has_live_resources(self) -> bool:
		"""Return True when cleanup still has browser-owned resources to release."""
		return bool(
			self._cdp_client_root
			or self._session_manager
			or self._watchdogs_attached
			or self._event_bus_retirement_pending
			or (self.is_local and self.browser_profile.cdp_url)
			or self._watchdog_has_live_resources(self._local_browser_watchdog)
		)

	def _start_publication_is_current(
		self, generation: int, owner_task: asyncio.Task | None
	) -> bool:
		"""Return True while a start still owns lifecycle publication."""
		if self._lifecycle.generation != generation or self._lifecycle.phase not in (
			LifecyclePhase.STARTING,
			LifecyclePhase.READY,
			LifecyclePhase.DEGRADED,
			LifecyclePhase.RECOVERING,
		):
			return False
		if owner_task is None:
			return True
		return (
			self._start_task is owner_task
			and not owner_task.cancelled()
			and not owner_task.cancelling()
		)

	def _has_start_in_flight(self) -> bool:
		"""Return True while public start or admitted connect work is active."""
		start_task = self._start_task
		return (
			bool(self._start_admission_tasks)
			or bool(self._direct_connect_tasks)
			or (start_task is not None and not start_task.done())
		)

	def _has_cleanup_in_flight(self) -> bool:
		"""Return True while this session still owns an active cleanup child."""
		return any(
			not task.done()
			for task in (*tuple(self._active_cleanup_tasks), *tuple(self._detached_cleanup_tasks))
		)

	def _raise_if_connect_admission_closed(self, *, start_owned: bool = False) -> None:
		"""Refuse connection work that began before the current shutdown fence closed."""
		if not self._connect_admission_closed:
			return
		if start_owned:
			raise RuntimeError('Browser session shutdown closed connection admission')
		raise RuntimeError('Browser session shutdown closed direct connections; call start() before connecting')

	def _close_connect_admission(self, *, shutdown: bool = False) -> None:
		"""Close direct-connect admission and invalidate already queued attempts."""
		self._connect_admission_closed = True
		self._connect_admission_epoch += 1
		if shutdown:
			self._shutdown_fence_epoch += 1

	@staticmethod
	def _caller_cancellation_requested() -> bool:
		"""Return True only when the current caller task received a cancellation request."""
		task = asyncio.current_task()
		return task is not None and task.cancelling() > 0

	@staticmethod
	def _watchdog_has_live_resources(watchdog: Any | None) -> bool:
		"""Return True while a local watchdog still owns a process or profile lease."""
		if watchdog is None:
			return False
		for attribute in ('_subprocess', '_spawned_subprocess', '_profile_lease'):
			if getattr(watchdog, attribute, None) is not None:
				return True
		return False

	async def reset(
		self,
		*,
		stop_event_bus: bool = True,
		wait_for_operations: bool = True,
		_reopen_admission: bool = True,
		_health_reason: BrowserHealthReason | None = BrowserHealthReason.RESET,
	) -> None:
		"""Clear all cached CDP sessions with proper cleanup."""
		current_task = asyncio.current_task()
		shutdown_fence_epoch = self._shutdown_fence_epoch
		async with self._reset_lock:
			self._reset_task = current_task
			try:
				await self._reset_transaction(
					stop_event_bus=stop_event_bus,
					wait_for_operations=wait_for_operations,
					reopen_admission=_reopen_admission,
					shutdown_fence_epoch=shutdown_fence_epoch,
				)
				if _health_reason is not None:
					self._record_stopped_health(_health_reason)
			except BaseException:
				if self._lifecycle.phase is LifecyclePhase.FAILED or self._has_live_resources():
					self._record_cleanup_failure_health()
				raise
			finally:
				if self._reset_task is current_task:
					self._reset_task = None

	async def _reset_transaction(
		self,
		*,
		stop_event_bus: bool,
		wait_for_operations: bool,
		reopen_admission: bool,
		shutdown_fence_epoch: int,
	) -> None:
		if self._has_cleanup_in_flight():
			raise TimeoutError('Prior browser cleanup is still running; retry after it finishes')
		admission_was_closed = self._connect_admission_closed
		cleanup_started = False
		keep_alive_retry = bool(
			self.browser_profile.keep_alive
			and self._cdp_health.cleanup_pending
			and self._watchdog_has_live_resources(self._local_browser_watchdog)
		)
		if (
			self._lifecycle.phase is LifecyclePhase.FAILED
			and self._watchdog_has_live_resources(self._local_browser_watchdog)
			and not keep_alive_retry
			and not self._event_bus_retirement_pending
		):
			raise RuntimeError('Browser resources are still live; run kill() before reset()')
		self._close_connect_admission()
		try:
			if wait_for_operations:
				try:
					await self._drain_reset_work(_shutdown_timeout())
				except TimeoutError as error:
					self._cdp_health.mark_cleanup_failure(error)
					self._watchdogs_attached = True
					if self._lifecycle.phase is not LifecyclePhase.FAILED:
						self._lifecycle.fail(error, reason='browser reset drain timed out')
					raise
			async with self._connect_admission_lock, self._lifecycle_lock:
				cleanup_started = True
				await self._reset_locked(stop_event_bus=stop_event_bus)
				if reopen_admission and self._shutdown_fence_epoch == shutdown_fence_epoch:
					self._connect_admission_closed = False
		except asyncio.CancelledError as error:
			if cleanup_started:
				if self._lifecycle.phase not in (LifecyclePhase.STOPPED, LifecyclePhase.FAILED):
					self._lifecycle.fail(error, reason='browser reset cancelled during cleanup')
				if (
					self._cdp_client_root is not None
					or self._session_manager is not None
					or self._watchdog_has_live_resources(self._local_browser_watchdog)
				):
					self._cdp_health.mark_cleanup_failure(error)
					self._watchdogs_attached = True
			else:
				if self._shutdown_fence_epoch == shutdown_fence_epoch:
					self._connect_admission_closed = admission_was_closed
			raise
		except BaseException as error:
			if (
				cleanup_started
				and (
					self._cdp_client_root is not None
					or self._session_manager is not None
					or self._watchdog_has_live_resources(self._local_browser_watchdog)
				)
			):
				self._cdp_health.mark_cleanup_failure(error)
				self._watchdogs_attached = True
			raise

	async def _reset_locked(self, *, stop_event_bus: bool) -> None:
		"""Reset runtime state for the active lifecycle operation."""
		await self._cancel_navigation()
		retirement_errors: list[BaseException] = []
		had_pending_retirement = bool(self._event_bus_retirement_pending)
		for retirement in tuple(self._event_bus_retirement_pending):
			if retirement.event_bus is not self.event_bus:
				try:
					await self._retire_event_bus(retirement.event_bus)
				except asyncio.CancelledError:
					raise
				except BaseException as error:
					retirement_errors.append(error)
		replace_event_bus = stop_event_bus
		current_bus_retired = True
		try:
			await self._retire_event_bus(self.event_bus, stop_bus=replace_event_bus)
		except asyncio.CancelledError:
			raise
		except BaseException as error:
			retirement_errors.append(error)
			current_bus_retired = False
		if replace_event_bus and current_bus_retired:
			self.event_bus = EventBus()
			self._core_handlers_attached = False
			self._attach_core_handlers()

		if had_pending_retirement and stop_event_bus and self._watchdog_has_live_resources(
			self._local_browser_watchdog
		):
			force_error = await self._force_stop_retained_browser_bounded()
			if force_error is not None:
				retirement_errors.append(force_error)

		if retirement_errors:
			primary_error = retirement_errors[0]
			for additional_error in retirement_errors[1:]:
				primary_error.add_note(f'Additional retirement failure: {additional_error!r}')
			if self._lifecycle.phase is not LifecyclePhase.FAILED:
				self._lifecycle.fail(primary_error, reason='event bus retirement failed')
			raise primary_error

		# Invalidate in-flight focus work before manager cleanup can cancel it.
		# A cancelled recovery must not publish its own failure reason after reset.
		self._focus_epoch += 1
		self._cancel_focus_observers()

		if self._watchdog_cleanup_error is not None:
			if any(not task.done() for task in self._detached_cleanup_tasks):
				cleanup_error = TimeoutError('Prior browser cleanup is still running')
				self._cdp_health.mark_cleanup_failure(cleanup_error)
				self._watchdogs_attached = True
				if self._lifecycle.phase is not LifecyclePhase.FAILED:
					self._lifecycle.fail(cleanup_error, reason='browser reset cleanup still running')
				raise cleanup_error
			if self._watchdog_has_live_resources(self._local_browser_watchdog):
				retry_error = await self._force_stop_retained_browser_bounded()
				if retry_error is not None or self._watchdog_has_live_resources(self._local_browser_watchdog):
					cleanup_error = retry_error or self._watchdog_cleanup_error
					self._cdp_health.mark_cleanup_failure(cleanup_error)
					self._watchdogs_attached = True
					if self._lifecycle.phase is not LifecyclePhase.FAILED:
						self._lifecycle.fail(cleanup_error, reason='browser reset watchdog cleanup failed')
					raise cleanup_error
			self._watchdog_cleanup_error = None

		was_cleanup_pending = self._cdp_health.cleanup_pending
		cleanup_error = await self._cleanup_current_cdp_ownership()
		if cleanup_error is not None:
			self._watchdogs_attached = True
			if self._lifecycle.phase is not LifecyclePhase.FAILED:
				self._lifecycle.fail(cleanup_error, reason='browser reset cleanup failed')
			if isinstance(cleanup_error, asyncio.CancelledError):
				raise cleanup_error
			if was_cleanup_pending:
				raise RuntimeError('Retained CDP resources could not be cleaned up') from cleanup_error
			raise cleanup_error

		# TODO: clear the event bus queue here, implement this helper
		# await self.event_bus.wait_for_idle(timeout=5.0)
		# await self.event_bus.clear()

		async with self._focus_lock:
			self._focus_lost_on_reset()

		# Clear session pool (all sessions share the root WebSocket, so no disconnect needed)
		self._cdp_session_pool.clear()
		self._tab_registry.clear()
		self._watchdogs_attached = False

		if self._lifecycle.phase in (
			LifecyclePhase.STARTING,
			LifecyclePhase.READY,
			LifecyclePhase.DEGRADED,
			LifecyclePhase.RECOVERING,
			LifecyclePhase.STOPPING,
			LifecyclePhase.FAILED,
		):
			self._lifecycle.transition(LifecyclePhase.STOPPED, reason='reset', increment_generation=True)
		self.reset_dom_cache(reason='reset', purge_metadata=True)
		self._downloaded_files.clear()
		self._download_history = await self._capture_download_history()
		self._storage_state_warnings.clear()

		self.agent_focus = None
		keep_local_browser = bool(
			self.is_local
			and self.browser_profile.keep_alive
			and self._watchdog_has_live_resources(self._local_browser_watchdog)
		)
		if self.is_local and not keep_local_browser:
			self.browser_profile.cdp_url = None

		self._crash_watchdog = None
		self._downloads_watchdog = None
		self._aboutblank_watchdog = None
		self._security_watchdog = None
		self._storage_state_watchdog = None
		if not self._watchdog_has_live_resources(self._local_browser_watchdog):
			self._local_browser_watchdog = None
		self._health_owned_root_binding = None
		self._default_action_watchdog = None
		self._dom_watchdog = None
		self._screenshot_watchdog = None
		self._permissions_watchdog = None
		self._recording_watchdog = None

	def model_post_init(self, __context) -> None:
		"""Register event handlers after model initialization."""
		self._attach_core_handlers()

	def _attach_core_handlers(self) -> None:
		"""Attach the BrowserSession core handlers to its current event bus."""
		if self._core_handlers_attached:
			return
		from openbrowser.browser.watchdog_base import BaseWatchdog

		BaseWatchdog.attach_handler_to_session(self, BrowserStartEvent, self.on_BrowserStartEvent)
		BaseWatchdog.attach_handler_to_session(self, BrowserStopEvent, self.on_BrowserStopEvent)
		BaseWatchdog.attach_handler_to_session(self, NavigateToUrlEvent, self.on_NavigateToUrlEvent)
		BaseWatchdog.attach_handler_to_session(self, SwitchTabEvent, self.on_SwitchTabEvent)
		BaseWatchdog.attach_handler_to_session(self, TabCreatedEvent, self.on_TabCreatedEvent)
		BaseWatchdog.attach_handler_to_session(self, TabClosedEvent, self.on_TabClosedEvent)
		BaseWatchdog.attach_handler_to_session(self, AgentFocusChangedEvent, self.on_AgentFocusChangedEvent)
		BaseWatchdog.attach_handler_to_session(self, FileDownloadedEvent, self.on_FileDownloadedEvent)
		BaseWatchdog.attach_handler_to_session(self, CloseTabEvent, self.on_CloseTabEvent)
		self._core_handlers_attached = True

	@observe_debug(ignore_input=True, ignore_output=True, name='browser_session_start')
	async def start(self) -> None:
		"""Run one start while direct connect calls are excluded."""
		if self._event_bus_retirement_pending:
			raise RuntimeError('Event bus retirement is still pending; run reset() before start()')
		if self._has_cleanup_in_flight():
			raise RuntimeError('Browser cleanup is still running; retry after it finishes')
		if self._cdp_health.cleanup_pending:
			raise RuntimeError('CDP cleanup is still pending; run kill() before starting')
		current_task = asyncio.current_task()
		if current_task is not None:
			self._start_admission_tasks.add(current_task)
		try:
			while True:
				reset_task = self._reset_task
				stop_task = self._stop_task
				blocking_task = None
				if reset_task is not None and not reset_task.done() and reset_task is not asyncio.current_task():
					blocking_task = reset_task
				elif stop_task is not None and not stop_task.done() and stop_task is not asyncio.current_task():
					blocking_task = stop_task
				if blocking_task is None:
					break
				await asyncio.shield(blocking_task)
			if self._lifecycle.phase is LifecyclePhase.READY:
				return
			await self._wait_for_start_prerequisites()
			async with self._connect_admission_lock:
				await self._start_admitted()
		finally:
			if current_task is not None:
				self._start_admission_tasks.discard(current_task)

	async def _wait_for_start_prerequisites(self) -> None:
		"""Wait for an earlier start or shutdown without holding the admission lock."""
		previous_start = self._start_task
		if previous_start is not None and not previous_start.done() and previous_start is not asyncio.current_task():
			await asyncio.wait({previous_start})
		if self._has_cleanup_in_flight():
			raise RuntimeError('Browser cleanup is still running; retry after it finishes')
		if self._cdp_health.cleanup_pending:
			raise RuntimeError('CDP cleanup is still pending; run kill() before starting')
		reset_task = self._reset_task
		if reset_task is not None and not reset_task.done() and reset_task is not asyncio.current_task():
			await asyncio.shield(reset_task)
		stop_task = self._stop_task
		if stop_task is not None and not stop_task.done() and stop_task is not asyncio.current_task():
			await asyncio.shield(stop_task)
			reset_task = self._reset_task
			if reset_task is not None and not reset_task.done() and reset_task is not asyncio.current_task():
				await asyncio.shield(reset_task)

	async def _start_admitted(self) -> None:
		"""Start the browser session while holding connection admission."""
		if self._event_bus_retirement_pending:
			raise RuntimeError('Event bus retirement is still pending; run reset() before start()')
		if self._has_cleanup_in_flight():
			raise RuntimeError('Browser cleanup is still running; retry after it finishes')
		if self._cdp_health.cleanup_pending:
			raise RuntimeError('CDP cleanup is still pending; run kill() before starting')
		if self._lifecycle.phase is LifecyclePhase.READY:
			return

		owner = False
		admission_was_closed = False
		async with self._lifecycle_lock:
			if self._lifecycle.phase is LifecyclePhase.READY:
				return
			if self._stop_task is not None and not self._stop_task.done():
				raise RuntimeError('Browser start cancelled by shutdown')
			self._check_failed_runtime_retirement()
			if self._lifecycle.phase is LifecyclePhase.FAILED and self._has_live_resources():
				raise RuntimeError('Browser resources are still live; run kill() or reset() before start()')
			if self._lifecycle.phase is LifecyclePhase.STARTING and self._start_event is not None:
				start_event = self._start_event
			else:
				self._lifecycle.transition(LifecyclePhase.STARTING, reason='start requested')
				admission_was_closed = self._connect_admission_closed
				self._connect_admission_closed = False
				self._start_event = asyncio.Event()
				self._start_error = None
				start_event = self._start_event
				owner = True

		if not owner:
			await start_event.wait()
			if self._lifecycle.phase is not LifecyclePhase.READY:
				error = self._start_error
				if error is not None:
					raise RuntimeError(f'Browser start failed: {error}') from error
				raise RuntimeError('Browser start did not reach the ready state')
			return

		self._start_cancelled_by_lifecycle = False
		start_task = asyncio.create_task(self._start_locked())
		self._start_task = start_task
		try:
			await start_task
		except asyncio.CancelledError:
			if admission_was_closed:
				self._close_connect_admission()
			if self._start_cancelled_by_lifecycle:
				raise RuntimeError('Browser start cancelled by shutdown') from None
			raise
		except BaseException as error:
			if admission_was_closed:
				self._close_connect_admission()
			self._start_error = error
			raise
		finally:
			start_event.set()
			if self._start_event is start_event:
				self._start_event = None
			if self._start_task is start_task:
				self._start_task = None

	async def _start_locked(self) -> None:
		"""Run the browser start event and update lifecycle state."""
		try:
			start_event = self.event_bus.dispatch(BrowserStartEvent())
			await start_event
			# Ensure any exceptions from the event handler are propagated
			await start_event.event_result(raise_if_any=True, raise_if_none=False)
		except asyncio.CancelledError:
			self._record_startup_failure_health()
			if self._lifecycle.phase is not LifecyclePhase.STOPPING:
				self._lifecycle.fail('Browser start cancelled', reason='browser start cancelled')
			await self._cleanup_cancelled_start()
			raise
		except BaseException as error:
			self._record_startup_failure_health()
			safe_error = f'{type(error).__name__}: {sanitize_error_detail(error) or type(error).__name__}'
			self._lifecycle.fail(safe_error, reason='browser start failed')
			raise
		else:
			self._health.record_connected(self._cdp_health.connection_generation)
			self._lifecycle.transition(LifecyclePhase.READY, reason='browser ready')

	async def _cleanup_cancelled_start(self) -> None:
		"""Release resources that a cancelled start acquired before failing."""
		force_error: BaseException | None = None
		if self._watchdog_has_live_resources(self._local_browser_watchdog):
			force_error = await self._force_stop_retained_browser_bounded()
			if force_error is not None:
				self._watchdog_cleanup_error = force_error
				self._cdp_health.mark_cleanup_failure(force_error)
				self._watchdogs_attached = True
		await self._drain_navigation_policies()
		await self._cleanup_current_cdp_ownership()
		if force_error is not None and self._watchdog_has_live_resources(self._local_browser_watchdog):
			self._cdp_health.mark_cleanup_failure(force_error)
			self._watchdogs_attached = True
		# A shutdown task already owns the current event bus and must be allowed
		# to stop and replace it after this cancellation cleanup returns.
		replace_event_bus = self._lifecycle.phase is not LifecyclePhase.STOPPING
		await self._retire_event_bus(self.event_bus, stop_bus=replace_event_bus, timeout=0)
		self._force_reset_runtime_state(replace_event_bus=replace_event_bus)
		if not self._watchdog_has_live_resources(self._local_browser_watchdog):
			self._local_browser_watchdog = None

	@staticmethod
	async def _await_task(task: asyncio.Task) -> None:
		"""Wait for a lifecycle task even when the caller is cancelled."""
		cancelled = False
		while not task.done():
			try:
				await asyncio.shield(task)
			except asyncio.CancelledError:
				cancelled = True
			except BaseException:
				break
		if task.done():
			task.result()
		if cancelled:
			raise asyncio.CancelledError

	async def _await_task_bounded(self, task: asyncio.Task, timeout: float) -> None:
		"""Wait for cleanup, then retain a cancellation-resistant task past the bound."""
		try:
			await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
		except asyncio.TimeoutError as error:
			self._record_cleanup_failure_health()
			if task.done():
				try:
					task.result()
				except asyncio.CancelledError:
					pass
				else:
					raise
			task.cancel()
			done, _pending = await asyncio.wait({task}, timeout=0.05)
			timeout_error = TimeoutError(f'Browser shutdown timed out after {timeout:g}s')
			if task not in done:
				self._cdp_health.mark_cleanup_failure(timeout_error)
				self._watchdogs_attached = True
				self._detached_shutdown_tasks.add(task)
				task.add_done_callback(self._consume_detached_operation_task)
				task.add_done_callback(self._detached_shutdown_tasks.discard)
				if self._lifecycle.phase is not LifecyclePhase.FAILED:
					self._lifecycle.fail(
						timeout_error, reason='browser shutdown timed out with retained work'
					)
				raise timeout_error from error
			try:
				task.result()
			except BaseException:
				pass
			if self._has_live_resources():
				self._cdp_health.mark_cleanup_failure(timeout_error)
				self._watchdogs_attached = True
				if self._lifecycle.phase is not LifecyclePhase.FAILED:
					self._lifecycle.fail(
						timeout_error, reason='browser shutdown timed out with retained work'
					)
				raise timeout_error from error
			await self._retire_event_bus(self.event_bus, timeout=0)
			self._force_reset_runtime_state()
			try:
				self._lifecycle.transition(
					LifecyclePhase.STOPPED,
					reason='browser force-stopped after timeout',
					error=timeout_error,
					increment_generation=True,
				)
			except LifecycleTransitionError:
				pass
			raise timeout_error from error
		except asyncio.CancelledError:
			try:
				await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
			except BaseException:
				pass
			raise

	def _force_reset_runtime_state(self, *, replace_event_bus: bool = True) -> None:
		"""Clear a quiescent runtime without discarding unretired authority."""
		self._abort_navigation()
		self._revoke_exited_policy_authorities()
		pending = next((op for op in self._navigation_policies.values() if op.policy_pending), None)
		if pending is not None:
			error = NavigationError(f'Policy authority has not retired. {_POLICY_RETIREMENT_HINT}')
			self._fail_policy_retirement(pending, error, reason='policy retirement incomplete')
			raise error
		self._cancel_navigation_event_pumps(self.event_bus)
		if self._live_navigation_event_pumps(self.event_bus):
			error = NavigationError('Event bus pumps have not retired; runtime is retained')
			self._lifecycle.fail(error, reason='event bus retirement incomplete')
			raise error
		if self._session_manager is not None:
			cancel_pending = getattr(self._session_manager, 'cancel_pending', None)
			if callable(cancel_pending):
				cancel_pending()
		self._focus_epoch += 1
		self._cancel_focus_observers()
		self._focus_lost_on_reset()
		if not self._cdp_health.cleanup_pending:
			self._session_manager = None
		self._cdp_session_pool.clear()
		if not self._cdp_health.cleanup_pending:
			self._cdp_client_root = None
			self._health_owned_root_binding = None
		self._tab_registry.clear()
		self._watchdogs_attached = False
		self.reset_dom_cache(reason='shutdown', purge_metadata=True)
		self._storage_state_warnings.clear()
		self.agent_focus = None
		keep_local_browser = bool(
			self.is_local
			and self.browser_profile.keep_alive
			and self._watchdog_has_live_resources(self._local_browser_watchdog)
		)
		if self.is_local and not keep_local_browser:
			self.browser_profile.cdp_url = None
		if not self._watchdog_has_live_resources(self._local_browser_watchdog):
			self._local_browser_watchdog = None
		if replace_event_bus:
			self.event_bus = EventBus()
			self._core_handlers_attached = False
			self._attach_core_handlers()

	async def kill(self) -> None:
		"""Kill the browser session and reset all state.

		When event-bus retirement is pending, reset recovery runs first because the
		closed bus cannot safely receive save or stop events.
		"""
		if self._has_cleanup_in_flight():
			raise TimeoutError('Prior browser cleanup is still running; retry after it finishes')
		if any(not task.done() for task in self._detached_shutdown_tasks):
			raise TimeoutError('Prior browser shutdown cleanup is still running; retry after it finishes')
		if self._event_bus_retirement_pending:
			await self.reset(_reopen_admission=False, _health_reason=BrowserHealthReason.STOP)
			return
		self._close_connect_admission(shutdown=True)
		if (
			self._lifecycle.phase is LifecyclePhase.STOPPED
			and not self._has_live_resources()
			and not self._has_start_in_flight()
		):
			return

		async with self._lifecycle_lock:
			if self._has_cleanup_in_flight():
				raise TimeoutError('Prior browser cleanup is still running; retry after it finishes')
			if (
				self._lifecycle.phase is LifecyclePhase.STOPPED
				and not self._has_live_resources()
				and not self._has_start_in_flight()
			):
				return
			pending_after_lock = bool(self._event_bus_retirement_pending)
			if self._cdp_health.cleanup_pending and (self._stop_task is None or self._stop_task.done()):
				cleanup_error = await self._retry_pending_cdp_cleanup()
				if cleanup_error is not None:
					self._record_cleanup_failure_health()
					if self._lifecycle.phase is not LifecyclePhase.FAILED:
						self._lifecycle.fail(cleanup_error, reason='browser kill cleanup failed')
					raise RuntimeError('Failed to release retained CDP resources') from cleanup_error
			if not pending_after_lock:
				if self._lifecycle.phase is LifecyclePhase.STOPPED:
					self._lifecycle.transition(LifecyclePhase.STOPPING, reason='kill requested')
				if self._stop_task is None or self._stop_task.done():
					if self._lifecycle.phase is not LifecyclePhase.STOPPING:
						self._lifecycle.transition(LifecyclePhase.STOPPING, reason='kill requested')
					self._stop_task = asyncio.create_task(self._shutdown_locked(force=True))
				stop_task = self._stop_task

		if pending_after_lock:
			await self.reset(_reopen_admission=False, _health_reason=BrowserHealthReason.STOP)
			return

		await self._await_task_bounded(stop_task, _shutdown_timeout())
		if self._watchdog_has_live_resources(self._local_browser_watchdog):
			force_error = await self._force_stop_retained_browser()
			if force_error is not None:
				self._cdp_health.mark_cleanup_failure(force_error)
				self._watchdogs_attached = True
				if self._lifecycle.phase is not LifecyclePhase.FAILED:
					self._lifecycle.fail(force_error, reason='browser kill cleanup failed')
				self._record_cleanup_failure_health()
				raise force_error from None
			if not self._watchdog_has_live_resources(self._local_browser_watchdog):
				self._local_browser_watchdog = None

	async def stop(self) -> None:
		"""Stop the browser session without forcing the browser process closed.

		Pending event-bus retirement uses reset recovery first. That recovery may
		force retained browser resources when the closed bus cannot drain normally.
		"""
		if self._has_cleanup_in_flight():
			raise TimeoutError('Prior browser cleanup is still running; retry after it finishes')
		if any(not task.done() for task in self._detached_shutdown_tasks):
			raise TimeoutError('Prior browser shutdown cleanup is still running; retry after it finishes')
		if self._event_bus_retirement_pending:
			await self.reset(_reopen_admission=False, _health_reason=BrowserHealthReason.STOP)
			return
		self._close_connect_admission(shutdown=True)
		if (
			self._lifecycle.phase is LifecyclePhase.STOPPED
			and not self._has_live_resources()
			and not self._has_start_in_flight()
		):
			return

		async with self._lifecycle_lock:
			if self._has_cleanup_in_flight():
				raise TimeoutError('Prior browser cleanup is still running; retry after it finishes')
			if (
				self._lifecycle.phase is LifecyclePhase.STOPPED
				and not self._has_live_resources()
				and not self._has_start_in_flight()
			):
				return
			pending_after_lock = bool(self._event_bus_retirement_pending)
			if not pending_after_lock:
				if self._lifecycle.phase is LifecyclePhase.STOPPED:
					self._lifecycle.transition(LifecyclePhase.STOPPING, reason='stop requested')
				if self._stop_task is None or self._stop_task.done():
					if self._lifecycle.phase is not LifecyclePhase.STOPPING:
						self._lifecycle.transition(LifecyclePhase.STOPPING, reason='stop requested')
					self._stop_task = asyncio.create_task(self._shutdown_locked(force=False))
				stop_task = self._stop_task

		if pending_after_lock:
			await self.reset(_reopen_admission=False, _health_reason=BrowserHealthReason.STOP)
			return

		await self._await_task_bounded(stop_task, _shutdown_timeout())

	async def _shutdown_locked(self, *, force: bool) -> None:
		"""Save state, dispatch the stop event, then release the event bus."""
		if self._event_bus_retirement_pending:
			try:
				await self.reset(_reopen_admission=False, _health_reason=BrowserHealthReason.STOP)
			finally:
				self._stop_task = None
			return
		self._close_connect_admission(shutdown=True)
		start_cancel_generation = self._lifecycle.generation
		start_cancel_bus = self.event_bus
		await self._cancel_owned_start()
		if self.event_bus is not start_cancel_bus or self._lifecycle.generation != start_cancel_generation:
			self._record_stopped_health(BrowserHealthReason.STOP)
			return
		if self._has_cleanup_in_flight():
			error = TimeoutError('Prior browser cleanup is still running; retry after it finishes')
			if self._lifecycle.phase is not LifecyclePhase.FAILED:
				self._lifecycle.fail(error, reason='browser shutdown waits for cleanup')
			raise error
		try:
			await self._cancel_navigation()
		except NavigationError:
			if not force or not any(
				op.containment and op.containment.requires_retirement for op in self._navigation_policies.values()
			):
				raise
			force_error = await self._force_stop_retained_browser()
			if force_error is not None:
				raise force_error from None
			self._revoke_exited_policy_authorities()
			await self._cancel_navigation()
			if self._lifecycle.phase is LifecyclePhase.FAILED:
				self._lifecycle.transition(
					LifecyclePhase.STOPPING, reason='owned browser exit retired policy authority'
				)
		shutdown_generation = self._lifecycle.generation
		event_bus = self.event_bus

		# First save storage state while CDP is still connected
		from openbrowser.browser.events import SaveStorageStateEvent

		save_error: BaseException | None = None
		try:
			save_event = event_bus.dispatch(SaveStorageStateEvent())
			await save_event
		except asyncio.CancelledError:
			raise
		except BaseException as error:
			# A storage-state failure must not prevent the browser from receiving
			# its force-stop event. Preserve the error for the caller after cleanup.
			save_error = error

		stop_error: BaseException | None = None
		self._stop_reset_error = None
		try:
			# Dispatch stop event to release the browser.
			stop_event = event_bus.dispatch(BrowserStopEvent(force=force))
			await stop_event
			event_result = getattr(stop_event, 'event_result', None)
			if callable(event_result):
				result = event_result(raise_if_any=True, raise_if_none=False)
				if inspect.isawaitable(result):
					await result
		except asyncio.CancelledError:
			raise
		except BaseException as error:
			# Keep the event bus and watchdog references intact when ownership
			# cleanup fails, so the caller can retry without releasing the lease.
			stop_error = error

		if stop_error is None and self._stop_reset_error is not None:
			stop_error = self._stop_reset_error

		if force:
			force_error = await self._force_stop_retained_browser()
			if stop_error is None and force_error is not None:
				stop_error = force_error

		if stop_error is not None and self._has_live_resources():
			self._record_cleanup_failure_health()
			self._cdp_health.mark_cleanup_failure(stop_error)
			self._watchdogs_attached = True
			if self._lifecycle.phase is not LifecyclePhase.FAILED:
				self._lifecycle.fail(stop_error, reason='browser stop failed')
			raise stop_error

		if self.event_bus is not event_bus:
			self._record_stopped_health(BrowserHealthReason.STOP)
			return
		if self._lifecycle.generation != shutdown_generation:
			# The stop handler already reset the lifecycle on this bus. Retire
			# the old bus and install a fresh one for the next start.
			try:
				await self._retire_event_bus(event_bus)
			except BaseException as error:
				self._watchdogs_attached = True
				self._record_cleanup_failure_health()
				self._lifecycle.fail(error, reason='event bus retirement failed after stop reset')
				raise
			self.event_bus = EventBus()
			self._core_handlers_attached = False
			self._attach_core_handlers()
			self._record_stopped_health(BrowserHealthReason.STOP)
			return

		try:
			if stop_error is not None:
				self._record_cleanup_failure_health()
				# A force-stop may still be able to release the watchdog after the
				# normal stop handler failed.
				if force:
					force_cleanup_error = await self._force_stop_retained_browser()
					if force_cleanup_error is not None:
						stop_error = force_cleanup_error

				if self._has_live_resources():
					# Retain the bus and every live owner so kill() can retry the
					# same cleanup. Marking the watchdog set attached prevents a later
					# start from registering duplicate handlers.
					self._cdp_health.mark_cleanup_failure(stop_error)
					self._watchdogs_attached = True
					if save_error is not None:
						raise stop_error from save_error
					raise stop_error

				# No browser-owned resources remain. Retire the failed bus so a
				# retry starts with a clean handler set.
				await self._retire_event_bus(event_bus)
				self._force_reset_runtime_state(replace_event_bus=True)
				if save_error is not None:
					raise stop_error from save_error
				raise stop_error

			finalization_error: BaseException | None = None
			event_bus_stopped = False
			try:
				# Stop the event bus only after the browser cleanup handlers complete.
				await self._retire_event_bus(event_bus)
				event_bus_stopped = True
			except asyncio.CancelledError:
				raise
			except BaseException as error:
				finalization_error = error
			if event_bus_stopped:
				try:
					# Reset all state. This task already owns the shutdown operation,
					# so it must not wait on a public reset admission slot.
					await self._reset_transaction(
						stop_event_bus=False,
						wait_for_operations=False,
						reopen_admission=False,
						shutdown_fence_epoch=self._shutdown_fence_epoch,
					)
				except asyncio.CancelledError:
					raise
				except BaseException as error:
					finalization_error = error
					if self._has_live_resources():
						self._cdp_health.mark_cleanup_failure(error)
						self._watchdogs_attached = True
				finally:
					self._force_reset_runtime_state(replace_event_bus=True)

			if finalization_error is not None:
				self._record_cleanup_failure_health()
				raise finalization_error
			if self.event_bus is event_bus:
				# The old bus was stopped before finalization. Install a fresh
				# bus so a keep_alive retry starts with an attachable handler set.
				self.event_bus = EventBus()
				self._core_handlers_attached = False
				self._attach_core_handlers()
			if save_error is not None:
				if self._lifecycle.phase is not LifecyclePhase.STOPPED:
					self._lifecycle.transition(
						LifecyclePhase.STOPPED,
						reason='browser stopped with storage-state error',
						error=save_error,
						increment_generation=True,
					)
				self._record_stopped_health(BrowserHealthReason.STOP)
				raise save_error
			if self._lifecycle.phase is not LifecyclePhase.STOPPED:
				self._lifecycle.transition(
					LifecyclePhase.STOPPED, reason='browser stopped', increment_generation=True
				)
			self._record_stopped_health(BrowserHealthReason.STOP)
		except BaseException as error:
			if self._lifecycle.phase is not LifecyclePhase.FAILED and (
				self._lifecycle.phase is not LifecyclePhase.STOPPED or self._has_live_resources()
			):
				self._record_cleanup_failure_health()
				self._lifecycle.fail(error, reason='browser stop failed')
			raise
		finally:
			self._stop_task = None

	def _owned_start_tasks(self) -> set[asyncio.Task]:
		"""Return in-flight start and admitted connect tasks owned by this session."""
		current = asyncio.current_task()
		return {
			task
			for task in (
				self._start_task,
				*tuple(self._start_admission_tasks),
				*tuple(self._direct_connect_tasks),
			)
			if task is not None and not task.done() and task is not current
		}

	async def _drain_reset_work(self, budget_seconds: float) -> None:
		"""Cancel owned work and wait for the prior stop within one reset deadline."""
		loop = asyncio.get_running_loop()
		deadline = loop.time() + budget_seconds
		if any(not task.done() for task in self._detached_shutdown_tasks):
			raise TimeoutError('Prior browser shutdown cleanup is still running')
		tasks = self._owned_start_tasks()
		if tasks:
			self._start_cancelled_by_lifecycle = True
			for task in tasks:
				task.cancel()
			remaining = max(0.0, deadline - loop.time())
			done, pending = await asyncio.wait(tasks, timeout=remaining)
			for task in done:
				with contextlib.suppress(BaseException):
					task.result()
			if pending:
				raise TimeoutError('Browser reset start or connect work did not quiesce before the deadline')
		stop_task = self._stop_task
		if stop_task is not None and not stop_task.done() and stop_task is not asyncio.current_task():
			remaining = deadline - loop.time()
			if remaining <= 0:
				raise TimeoutError('Browser reset deadline expired while waiting for the prior stop')
			done, pending = await asyncio.wait({stop_task}, timeout=remaining)
			for task in done:
				with contextlib.suppress(BaseException):
					task.result()
			if pending:
				raise TimeoutError('Browser reset did not drain the prior stop before the deadline')

	async def _cancel_owned_start(self) -> None:
		"""Cancel in-flight start and admitted connect work before cleanup."""
		tasks = self._owned_start_tasks()
		if not tasks:
			return
		self._start_cancelled_by_lifecycle = True
		for task in tasks:
			task.cancel()
		await asyncio.gather(*tasks, return_exceptions=True)

	async def _force_stop_retained_browser_bounded(self) -> BaseException | None:
		"""Bound watchdog cleanup while retaining cancellation-resistant work."""
		if not self._watchdog_has_live_resources(self._local_browser_watchdog):
			return None
		if self._has_cleanup_in_flight():
			return TimeoutError('Prior browser cleanup is still running')
		cleanup_task = asyncio.create_task(self._force_stop_retained_browser())
		self._active_cleanup_tasks.add(cleanup_task)
		cleanup_task.add_done_callback(self._active_cleanup_tasks.discard)
		try:
			await self._await_bounded_operation(
				cleanup_task,
				_shutdown_timeout(),
				'Local browser cleanup',
				retained_tasks=self._detached_cleanup_tasks,
			)
		except asyncio.CancelledError as error:
			return error
		except BaseException as error:
			return error
		try:
			return cleanup_task.result()
		except BaseException as error:
			return error

	async def _force_stop_retained_browser(self) -> BaseException | None:
		"""Kill and release a local browser retained by keep_alive."""
		watchdog = self._local_browser_watchdog
		if not self._watchdog_has_live_resources(watchdog):
			return None
		handler = getattr(watchdog, 'on_BrowserKillEvent', None)
		if not callable(handler):
			return None
		try:
			result = handler(BrowserKillEvent())
			if inspect.isawaitable(result):
				await result
		except asyncio.CancelledError:
			raise
		except BaseException as error:
			return error
		return None

	@observe_debug(ignore_input=True, ignore_output=True, name='browser_start_event_handler')
	async def on_BrowserStartEvent(self, event: BrowserStartEvent) -> dict[str, str]:
		"""Handle browser start request.

		Returns:
                Dict with 'cdp_url' key containing the CDP URL
		"""

		# await self.reset()
		start_generation = self._lifecycle.generation
		owner_task = self._start_task

		# Initialize and attach all watchdogs FIRST so LocalBrowserWatchdog can handle BrowserLaunchEvent
		await self.attach_all_watchdogs()

		try:
			# If no CDP URL, launch local browser
			if not self.cdp_url:
				if self.is_local:
					# Launch local browser using event-driven approach
					launch_event = self.event_bus.dispatch(BrowserLaunchEvent())
					await launch_event

					# Get the CDP URL from LocalBrowserWatchdog handler result
					launch_result: BrowserLaunchResult = cast(
						BrowserLaunchResult, await launch_event.event_result(raise_if_none=True, raise_if_any=True)
					)
					self.browser_profile.cdp_url = launch_result.cdp_url
				else:
					raise ValueError('Got BrowserSession(is_local=False) but no cdp_url was provided to connect to!')

			assert self.cdp_url and '://' in self.cdp_url

			if self._cdp_health.cleanup_pending:
				raise RuntimeError('CDP cleanup is still pending; run kill() before starting')

			# Only connect if not already connected
			if self._cdp_client_root is None:
				# Setup browser via CDP (for both local and remote cases)
				await self.connect(cdp_url=self.cdp_url, _start_owned=True)
				assert self.cdp_client is not None
				if not self._start_publication_is_current(start_generation, owner_task):
					raise RuntimeError('Browser start no longer owns connection publication')

				# Notify that browser is connected (single place)
				self.event_bus.dispatch(BrowserConnectedEvent(cdp_url=self.cdp_url))
			else:
				self.logger.debug('Already connected to CDP, skipping reconnection')
				targets = await self._cdp_get_all_pages()
				if not self._start_publication_is_current(start_generation, owner_task):
					raise RuntimeError('Browser start no longer owns connection publication')
				self.event_bus.dispatch(BrowserConnectedEvent(cdp_url=self.cdp_url))
				for target in targets:
					self.event_bus.dispatch(
						TabCreatedEvent(
							url=target.get('url', '') or 'about:blank',
							target_id=target['targetId'],
						)
					)

			# Return the CDP URL for other components
			return {'cdp_url': self.cdp_url}

		except Exception as e:
			self.event_bus.dispatch(
				BrowserErrorEvent(
					error_type='BrowserStartEventError',
					message=f'Failed to start browser: {type(e).__name__} {e}',
					details={'cdp_url': self.cdp_url, 'is_local': self.is_local},
				)
			)
			raise

	def _reserve_navigation(self, operation: NavigationOperation) -> None:
		self._check_failed_runtime_retirement()
		if self._lifecycle.phase is LifecyclePhase.STOPPING or self._lifecycle_lock.locked():
			raise NavigationBusyError('Browser lifecycle cleanup is in progress')
		if self._navigation_task is not None and not self._navigation_task.done():
			raise NavigationBusyError('Another navigation is in progress')
		self._navigation_monitor.reserve(operation)

	def _check_failed_runtime_retirement(self) -> None:
		self._revoke_exited_policy_authorities()
		if self._event_bus_retirement_pending:
			raise NavigationError('Event bus retirement is pending; run reset() before navigating')
		if any(op.containment and op.containment.requires_retirement for op in self._navigation_policies.values()):
			raise NavigationError(f'Policy action retirement is unconfirmed. {_POLICY_RETIREMENT_HINT}')
		if self._lifecycle.phase is LifecyclePhase.FAILED and self._live_navigation_event_pumps(self.event_bus):
			raise NavigationError('Event bus retirement is incomplete; finish cleanup before reusing this runtime')

	def _publish_navigation_finished(self, operation: NavigationOperation, event_bus: EventBus) -> None:
		if operation.publication_attempted:
			return
		operation.publication_attempted = True
		try:
			contextvars.Context().run(
				event_bus.dispatch,
				NavigationFinishedEvent(
					navigation_id=operation.navigation_id,
					generation=operation.generation,
					target_id=operation.target_id,
					requested_url=operation.requested_url,
					observed_url=operation.observed_url,
					outcome=operation.phase.value,
					error_text=operation.last_error,
				),
			)
		except Exception as error:
			self.logger.debug('Could not enqueue navigation terminal status: %s', error)

	def _retire_navigation(self, operation: NavigationOperation) -> None:
		if (
			operation.terminal
			and not operation.child_tasks
			and not operation.policy_pending
			and self._navigation_task_operation is not operation
		):
			self._navigation_monitor.complete(operation)

	def _cancel_navigation_tasks(self, operation: NavigationOperation) -> set[asyncio.Task]:
		tasks = set(operation.child_tasks)
		if self._navigation_task_operation is operation and self._navigation_task is not None:
			tasks.add(self._navigation_task)
		if not tasks:
			return tasks
		current = asyncio.current_task()
		tasks = {task for task in tasks if task is not current and not task.done()}
		for task in tasks:
			task.cancel()
		return tasks

	def _abort_navigation(self) -> set[asyncio.Task]:
		operation = self._navigation_monitor.active
		tasks = set()
		if operation is not None:
			operation.close_policy_capture()
			if (
				operation.policy_task is not None
				and not operation.policy_task.done()
				and not (operation.containment and operation.containment.requires_retirement)
			):
				operation.policy_task.cancel()
			operation.finish(NavigationPhase.CANCELLED, 'Browser session reset')
			tasks = self._cancel_navigation_tasks(operation)
			self._publish_navigation_finished(operation, self.event_bus)
			self._retire_navigation(operation)
		self._navigation_monitor.invalidate()
		return tasks

	async def _cancel_navigation(self) -> None:
		tasks = self._abort_navigation()
		if tasks:
			await asyncio.wait(tasks, timeout=1.0)
		await self._drain_navigation_policies()

	def _check_navigation_owner(
		self, operation: NavigationOperation, root: CDPClient, session: CDPSession | None = None
	) -> None:
		if operation.deadline is not None and asyncio.get_running_loop().time() >= operation.deadline:
			raise NavigationTimeoutError('Navigation deadline expired')
		if (
			self._navigation_monitor.active is not operation
			or operation.terminal
			or self._lifecycle.generation != operation.generation
			or self._cdp_client_root is not root
		):
			raise NavigationError('Navigation ownership or browser binding changed')
		if session is not None and (
			session.cdp_client is not root
			or session.session_id != operation.session_id
			or session.target_id != operation.target_id
		):
			raise NavigationError('Navigation CDP session binding changed')

	def _marked_navigation(self, event: Any, root: CDPClient) -> NavigationOperation:
		operation = self._navigation_monitor.active
		if (
			operation is None
			or event.navigation_id != operation.navigation_id
			or event.navigation_generation != operation.generation
		):
			raise NavigationError('Stale navigation child event')
		self._check_navigation_owner(operation, root)
		return operation

	async def _await_navigation_event(self, event_bus: EventBus, event: Any) -> None:
		dispatched = event_bus.dispatch(event)

		async def pump() -> None:
			await dispatched
			await dispatched.event_result(raise_if_any=True, raise_if_none=False)

		task = asyncio.create_task(pump())
		self._navigation_event_pumps[task] = event_bus
		task.add_done_callback(self._navigation_event_pump_done)
		# Bubus event-await can execute unrelated queued handlers. Its pump is
		# bus-owned, while cancellation of marked handlers is request-owned.
		await asyncio.shield(task)

	def _navigation_event_pump_done(self, task: asyncio.Task) -> None:
		self._navigation_event_pumps.pop(task, None)
		if not task.cancelled():
			task.exception()
		self._reconcile_policy_retention()

	def _live_navigation_event_pumps(self, event_bus: EventBus) -> set[asyncio.Task]:
		return {task for task, bus in self._navigation_event_pumps.items() if bus is event_bus and not task.done()}

	def _cancel_navigation_event_pumps(self, event_bus: EventBus) -> set[asyncio.Task]:
		tasks = self._live_navigation_event_pumps(event_bus)
		if tasks:
			tasks.discard(asyncio.current_task())
		for task in tasks:
			task.cancel()
		return tasks

	async def _drain_navigation_event_pumps(self, event_bus: EventBus) -> None:
		tasks = self._cancel_navigation_event_pumps(event_bus)
		if tasks:
			await asyncio.wait(tasks, timeout=1.0)
		if self._live_navigation_event_pumps(event_bus):
			raise TimeoutError('Navigation event pump retirement timed out')

	async def _retire_event_bus(self, event_bus: EventBus, *, stop_bus: bool = True, timeout: float = 5.0) -> None:
		await self._drain_navigation_policies()
		runloop_tasks = self._event_bus_runloop_tasks(event_bus)
		stop_error: BaseException | None = None
		if stop_bus:
			try:
				await event_bus.stop(clear=True, timeout=timeout)
			except BaseException as error:
				stop_error = error
		try:
			await self._drain_navigation_event_pumps(event_bus)
		except BaseException as error:
			self._mark_event_bus_retirement_pending(
				event_bus, self._merge_event_bus_runloop_tasks(event_bus, runloop_tasks)
			)
			self._lifecycle.fail(error, reason='event bus pump retirement failed')
			if stop_error is not None:
				raise error from stop_error
			raise
		if stop_error is not None:
			self._mark_event_bus_retirement_pending(
				event_bus, self._merge_event_bus_runloop_tasks(event_bus, runloop_tasks)
			)
			self._lifecycle.fail(stop_error, reason='event bus stop failed')
			raise stop_error
		if stop_bus:
			runloop_tasks = self._merge_event_bus_runloop_tasks(event_bus, runloop_tasks)
			active_runloops = [task for task in runloop_tasks if not task.done()]
			if active_runloops:
				self._mark_event_bus_retirement_pending(event_bus, runloop_tasks)
				await self._wait_for_event_bus_runloops(active_runloops)
				runloop_tasks = self._merge_event_bus_runloop_tasks(event_bus, runloop_tasks)
				active_runloops = [task for task in runloop_tasks if not task.done()]
			if active_runloops:
				error = TimeoutError('Event bus run loop did not stop')
				self._mark_event_bus_retirement_pending(event_bus, runloop_tasks)
				self._lifecycle.fail(error, reason='event bus run loop did not stop')
				raise error
			self._clear_event_bus_retirement_pending(event_bus)

	@staticmethod
	async def _wait_for_event_bus_runloops(runloop_tasks: list[asyncio.Future]) -> None:
		"""Give cancelled bus run loops one bounded chance to finish."""
		await asyncio.wait(set(runloop_tasks), timeout=0.1)

	@staticmethod
	def _event_bus_runloop_task(event_bus: EventBus) -> asyncio.Future | None:
		"""Return the installed bus run loop when it exists."""
		task = getattr(event_bus, '_runloop_task', None)
		return task if isinstance(task, asyncio.Future) else None

	def _event_bus_runloop_tasks(self, event_bus: EventBus) -> list[asyncio.Future]:
		"""Return every run loop retained for one bus, including a replacement."""
		tasks: list[asyncio.Future] = []
		pending_entry = self._pending_event_bus_retirement(event_bus)
		if pending_entry is not None:
			tasks.extend(pending_entry.runloop_tasks)
		current_task = self._event_bus_runloop_task(event_bus)
		if current_task is not None and not any(current_task is task for task in tasks):
			tasks.append(current_task)
		return tasks

	def _merge_event_bus_runloop_tasks(
		self, event_bus: EventBus, retained_tasks: list[asyncio.Future]
	) -> list[asyncio.Future]:
		"""Keep the pre-stop tasks while adding any current or replacement run loop."""
		tasks = list(retained_tasks)
		for task in self._event_bus_runloop_tasks(event_bus):
			if not any(task is retained for retained in tasks):
				tasks.append(task)
		return tasks

	def _pending_event_bus_retirement(self, event_bus: EventBus) -> _PendingEventBusRetirement | None:
		"""Return the retained retirement entry for one bus identity."""
		return next(
			(entry for entry in self._event_bus_retirement_pending if entry.event_bus is event_bus),
			None,
		)

	def _is_event_bus_retirement_pending(self, event_bus: EventBus) -> bool:
		"""Return whether one bus still owns cleanup authority."""
		return self._pending_event_bus_retirement(event_bus) is not None

	def _mark_event_bus_retirement_pending(
		self, event_bus: EventBus, runloop_tasks: list[asyncio.Future] | None = None
	) -> None:
		"""Retain one retiring bus exactly once until its run loop is quiescent."""
		entry = self._pending_event_bus_retirement(event_bus)
		if entry is None:
			entry = _PendingEventBusRetirement(event_bus)
			self._event_bus_retirement_pending.append(entry)
		for task in runloop_tasks or ():
			if not any(task is retained for retained in entry.runloop_tasks):
				entry.runloop_tasks.append(task)

	def _clear_event_bus_retirement_pending(self, event_bus: EventBus) -> None:
		"""Release pending ownership for one bus that was actually stopped."""
		self._event_bus_retirement_pending = [
			entry for entry in self._event_bus_retirement_pending if entry.event_bus is not event_bus
		]

	def _navigation_deadline(self, event: NavigateToUrlEvent, operation: NavigationOperation) -> float:
		if operation.deadline is None:
			duration = event.timeout_ms / 1000 if event.timeout_ms is not None else event.event_timeout
			duration = duration if duration is not None else 30.0
			if event.event_timeout is not None:
				duration = min(duration, event.event_timeout)
			if not math.isfinite(duration) or duration <= 0:
				raise NavigationError('Navigation timeout must be positive and finite')
			operation.deadline = asyncio.get_running_loop().time() + duration
		return operation.deadline

	def _policy_owned(self, operation: NavigationOperation, root: CDPClient, session: CDPSession) -> bool:
		return (
			self._navigation_monitor.active is operation
			and self._cdp_client_root is root
			and self._lifecycle.generation == operation.generation
			and session.cdp_client is root
			and session.target_id == operation.target_id
			and session.session_id == operation.session_id
			and (operation.containment is None or operation.containment.phase is not ContainmentPhase.REVOKED)
		)

	def _require_policy_owner(self, operation: NavigationOperation, root: CDPClient, session: CDPSession) -> None:
		if not self._policy_owned(operation, root, session):
			raise NavigationError('Policy document authority changed')

	def _start_navigation_policy(
		self, operation: NavigationOperation, root: CDPClient, session: CDPSession, policy: Any, event_bus: EventBus
	) -> None:
		operation.arm_policy()
		self._navigation_policies[operation.navigation_id] = operation
		operation.policy_task = asyncio.create_task(
			self._run_navigation_policy(operation, root, session, policy, event_bus)
		)
		operation.policy_task.add_done_callback(lambda task: self._navigation_policy_done(operation, task))

	def _navigation_policy_done(self, operation: NavigationOperation, task: asyncio.Task) -> None:
		if not task.cancelled():
			task.exception()
		if not operation.policy_pending:
			self._navigation_policies.pop(operation.navigation_id, None)
		self._reconcile_policy_retention()
		self._retire_navigation(operation)

	def _fail_policy_retirement(
		self, operation: NavigationOperation, error: BaseException | str, *, reason: str
	) -> None:
		snapshot = self._lifecycle.snapshot()
		retained = self._policy_retention_failure
		root = self._cdp_client_root
		owns_failure = (
			retained is not None
			and retained.operation is operation
			and retained.root is root
			and retained.failure == snapshot
			and retained.revision == self._lifecycle.revision
			and operation.generation == snapshot.generation
		)
		if snapshot.phase is LifecyclePhase.FAILED and not owns_failure:
			self._policy_retention_failure = None
			return
		can_recover = owns_failure or snapshot.phase in {LifecyclePhase.READY, LifecyclePhase.DEGRADED}
		failure = self._lifecycle.fail(error, reason=reason)
		self._policy_retention_failure = (
			_PolicyRetentionFailure(operation, root, failure, self._lifecycle.revision)
			if (can_recover and root is not None and operation.generation == failure.generation)
			else None
		)

	def _reconcile_policy_retention(self) -> None:
		retained = self._policy_retention_failure
		if retained is None:
			return
		operation = retained.operation
		if (
			retained.failure != self._lifecycle.snapshot()
			or retained.root is not self._cdp_client_root
			or retained.revision != self._lifecycle.revision
			or operation.generation != self._lifecycle.generation
			or (self._stop_task is not None and not self._stop_task.done())
		):
			self._policy_retention_failure = None
			return
		task = operation.policy_task
		if task is None or not task.done():
			return
		if (
			task.cancelled()
			or task.exception() is not None
			or operation.containment is None
			or operation.containment.phase not in {ContainmentPhase.CONTAINED, ContainmentPhase.SKIPPED}
		):
			self._policy_retention_failure = None
			return
		if any(op.policy_pending for op in self._navigation_policies.values()) or self._live_navigation_event_pumps(
			self.event_bus
		):
			return
		self._policy_retention_failure = None
		self._lifecycle.transition(LifecyclePhase.RECOVERING, reason='policy authority retired')
		self._lifecycle.transition(
			LifecyclePhase.DEGRADED, reason='policy authority retired; explicit recovery is available'
		)

	async def _owned_policy_browser(
		self, operation: NavigationOperation, root: CDPClient
	) -> tuple[int, float, str, str] | None:
		import ipaddress
		import psutil
		import socket
		from openbrowser.browser.watchdogs.profile_lease import ProfileLease

		watchdog = self._local_browser_watchdog
		if not self.is_local or watchdog is None or getattr(watchdog, '_owns_browser_resources', False) is not True:
			return None
		record = getattr(watchdog, '_browser_record', None)
		lease = getattr(watchdog, '_profile_lease', None)
		process = getattr(watchdog, '_subprocess', None)
		if not isinstance(record, dict) or lease is None or process is None:
			return None
		try:
			ws = root.ws
			transport = ws.transport
			connection_socket = transport.get_extra_info('socket')
			family = connection_socket.family
			if family not in {socket.AF_INET, socket.AF_INET6}:
				return None

			def endpoint(value: Any) -> tuple[Any, int] | None:
				if not isinstance(value, tuple) or len(value) < 2 or not isinstance(value[1], int) or any(value[2:]):
					return None
				address = ipaddress.ip_address(value[0])
				if (
					address.version != (4 if family == socket.AF_INET else 6)
					or getattr(address, 'scope_id', None) is not None
				):
					return None
				return address, value[1]

			peer = endpoint(connection_socket.getpeername())
			local = endpoint(connection_socket.getsockname())
			profile = str(Path(self.browser_profile.user_data_dir).expanduser().resolve())
			if (
				peer is None
				or local is None
				or not peer[0].is_loopback
				or not local[0].is_loopback
				or peer[1] != record['cdp_port']
				or profile != record['profile_dir']
				or str(lease.profile_dir) != profile
				or lease.instance_id != record['instance_id']
				or watchdog._instance_id != record['instance_id']
				or not ProfileLease.process_matches_browser(process, record)
			):
				return None

			def endpoint_owned() -> bool:
				if (
					root.ws is not ws
					or ws.transport is not transport
					or connection_socket.family != family
					or endpoint(connection_socket.getpeername()) != peer
					or endpoint(connection_socket.getsockname()) != local
				):
					return False
				connections = [
					connection for connection in process.net_connections(kind='tcp') if connection.family == family
				]
				listeners = [
					endpoint(connection.laddr) for connection in connections if connection.status == psutil.CONN_LISTEN
				]
				covers_peer = any(
					listener is not None
					and listener[1] == peer[1]
					and (listener[0] == peer[0] or listener[0].is_unspecified)
					for listener in listeners
				)
				accepted_peer = any(
					connection.status == psutil.CONN_ESTABLISHED
					and endpoint(connection.laddr) == peer
					and endpoint(connection.raddr) == local
					for connection in connections
				)
				return covers_peer and accepted_peer

			if not endpoint_owned():
				return None
			info = await self._policy_read(operation, root.send.SystemInfo.getProcessInfo())
			if (
				not any(
					entry.get('type') == 'browser' and entry.get('id') == record['pid']
					for entry in info.get('processInfo', [])
				)
				or not ProfileLease.process_matches_browser(process, record)
				or not endpoint_owned()
			):
				return None
			return (int(record['pid']), float(record['start_time']), profile, str(record['instance_id']))
		except Exception:
			return None

	def _revoke_exited_policy_authorities(self) -> None:
		if not self._navigation_policies:
			return
		from openbrowser.browser.watchdogs.profile_lease import ProfileLease

		for operation in list(self._navigation_policies.values()):
			attempt = operation.containment
			if attempt is None or attempt.owned_browser is None or not attempt.requires_retirement:
				continue
			pid, start_time, _profile, _instance = attempt.owned_browser
			if ProfileLease.process_identity_status(pid, start_time) not in {'missing', 'mismatch'}:
				continue
			attempt.phase = ContainmentPhase.REVOKED
			attempt.error = 'Authority revoked by verified exit of its owned browser process'
			if operation.policy_task is not None:
				if not operation.policy_task.done():
					operation.policy_task.cancel()
				else:
					self._navigation_policy_done(operation, operation.policy_task)
			self._retire_navigation(operation)

	async def _run_navigation_policy(
		self, operation: NavigationOperation, root: CDPClient, session: CDPSession, policy: Any, event_bus: EventBus
	) -> None:
		seen = None
		while True:
			commit = operation.policy_commit
			if commit is not None and commit != seen:
				seen = commit
				if not policy._is_url_allowed(commit.url):
					attempt = ContainmentAttempt(
						commit,
						world_name=f'openbrowser-policy-{uuid7str()}',
						deadline=asyncio.get_running_loop().time() + NAVIGATION_CONTAINMENT_GRACE,
					)
					operation.containment = attempt
					contextvars.Context().run(
						event_bus.dispatch,
						BrowserErrorEvent(
							error_type='NavigationBlocked',
							message=f'Navigation blocked to {commit.url}',
							details={
								'url': commit.url,
								'target_id': operation.target_id,
								'navigation_id': operation.navigation_id,
							},
						),
					)
					await self._contain_navigation_document(operation, root, session)
					return
			if not operation.policy_open:
				return
			operation.policy_changed.clear()
			await operation.policy_changed.wait()

	async def _policy_read(self, operation: NavigationOperation, awaitable: Any, *, verification: bool = False) -> Any:
		attempt = operation.containment
		assert attempt is not None and attempt.deadline is not None
		remaining = attempt.deadline - asyncio.get_running_loop().time()
		return await asyncio.wait_for(
			awaitable, timeout=max(0.001, remaining) if not verification else NAVIGATION_CONTAINMENT_GRACE
		)

	@staticmethod
	def _frame_commit(frame: dict[str, Any]) -> NavigationCommit:
		return NavigationCommit(
			frame['id'], frame.get('loaderId', ''), frame.get('url', '') + frame.get('urlFragment', '')
		)

	def _observe_contained_document(self, operation: NavigationOperation, root: CDPClient, session: CDPSession) -> None:
		self._require_policy_owner(operation, root, session)
		session.url, session.title = 'about:blank', ''
		record = self._tab_registry.get(operation.target_id)
		if record is not None and record.state is not TabState.CLOSED:
			self._tab_registry.update(operation.target_id, url='about:blank', title='')
		self.invalidate_dom_cache(target_id=operation.target_id, reason='policy_containment')
		focus = self.agent_focus
		if (
			focus is not None
			and focus.cdp_client is root
			and focus.target_id == operation.target_id
			and focus.session_id == operation.session_id
		):
			focus.url, focus.title = 'about:blank', ''

	async def _resolve_policy_protocol_error(
		self, error: BaseException, operation: NavigationOperation, root: CDPClient, session: CDPSession
	) -> bool:
		details = error.args[0] if isinstance(error, RuntimeError) and error.args else None
		if not isinstance(details, dict) or not isinstance(details.get('code'), int):
			return False
		attempt = operation.containment
		assert attempt is not None
		try:
			self._require_policy_owner(operation, root, session)
			frame = (
				await self._policy_read(
					operation, root.send.Page.getFrameTree(session_id=operation.session_id), verification=True
				)
			)['frameTree']['frame']
			self._require_policy_owner(operation, root, session)
			current = self._frame_commit(frame)
			if current.url == 'about:blank':
				self._observe_contained_document(operation, root, session)
				attempt.phase = ContainmentPhase.CONTAINED
				return True
			if details.get('code') == -32000 and details.get('message') in {
				'Cannot find context with specified id',
				'Could not find object with given id',
			}:
				attempt.phase = ContainmentPhase.SKIPPED if current != attempt.commit else ContainmentPhase.FAILED
				return True
		except Exception:
			pass
		return False

	async def _contain_navigation_document(
		self, operation: NavigationOperation, root: CDPClient, session: CDPSession
	) -> None:
		attempt = operation.containment
		assert attempt is not None
		try:
			self._require_policy_owner(operation, root, session)
			attempt.owned_browser = await self._owned_policy_browser(operation, root)
			self._require_policy_owner(operation, root, session)
			frame = (await self._policy_read(operation, root.send.Page.getFrameTree(session_id=operation.session_id)))[
				'frameTree'
			]['frame']
			self._require_policy_owner(operation, root, session)
			if self._frame_commit(frame) != attempt.commit:
				attempt.phase = ContainmentPhase.SKIPPED
				return
			await self._policy_read(operation, root.send.Runtime.enable(session_id=operation.session_id))
			self._require_policy_owner(operation, root, session)
			world = await self._policy_read(
				operation,
				root.send.Page.createIsolatedWorld(
					params={
						'frameId': attempt.commit.frame_id,
						'worldName': attempt.world_name,
						'grantUniveralAccess': False,
					},
					session_id=operation.session_id,
				),
			)
			self._require_policy_owner(operation, root, session)
			while attempt.context is None or attempt.context[0] != world['executionContextId']:
				operation.policy_changed.clear()
				await self._policy_read(operation, operation.policy_changed.wait())
				self._require_policy_owner(operation, root, session)
			result = await self._policy_read(
				operation,
				root.send.Runtime.evaluate(
					params={
						'expression': 'document',
						'uniqueContextId': attempt.context[1],
						'returnByValue': False,
						'objectGroup': attempt.world_name,
					},
					session_id=operation.session_id,
				),
			)
			self._require_policy_owner(operation, root, session)
			document = result.get('result', {})
			if (
				result.get('exceptionDetails')
				or document.get('subtype') != 'node'
				or not isinstance(document.get('objectId'), str)
			):
				raise NavigationError('Could not acquire the policy Document object')
			attempt.object_id = document['objectId']
			frame = (await self._policy_read(operation, root.send.Page.getFrameTree(session_id=operation.session_id)))[
				'frameTree'
			]['frame']
			self._require_policy_owner(operation, root, session)
			if self._frame_commit(frame) != attempt.commit:
				attempt.phase = ContainmentPhase.SKIPPED
				return
			if asyncio.get_running_loop().time() >= attempt.deadline:
				raise TimeoutError('Policy preparation exceeded its containment grace')
			attempt.phase = ContainmentPhase.ACTION_IN_FLIGHT
			result = await root.send.Runtime.callFunctionOn(
				params={
					'objectId': attempt.object_id,
					'functionDeclaration': _CONTAIN_DOCUMENT,
					'arguments': [{'value': attempt.commit.url}],
					'returnByValue': True,
					'awaitPromise': False,
				},
				session_id=operation.session_id,
			)
			self._require_policy_owner(operation, root, session)
			value = result.get('result', {}).get('value', {})
			if (
				result.get('exceptionDetails')
				or not isinstance(value, dict)
				or not isinstance(value.get('initiated'), bool)
			):
				raise NavigationError('Policy action returned no conclusive result')
			if not value['initiated']:
				attempt.phase = ContainmentPhase.SKIPPED
				return
			attempt.phase = ContainmentPhase.VERIFYING
			verify_deadline = asyncio.get_running_loop().time() + NAVIGATION_CONTAINMENT_GRACE
			while True:
				frame = (
					await self._policy_read(
						operation, root.send.Page.getFrameTree(session_id=operation.session_id), verification=True
					)
				)['frameTree']['frame']
				self._require_policy_owner(operation, root, session)
				current = self._frame_commit(frame)
				if current.url == 'about:blank':
					self._observe_contained_document(operation, root, session)
					attempt.phase = ContainmentPhase.CONTAINED
					return
				remaining = verify_deadline - asyncio.get_running_loop().time()
				if remaining <= 0:
					raise TimeoutError('Policy replacement was not observed')
				operation.policy_changed.clear()
				try:
					await asyncio.wait_for(operation.policy_changed.wait(), timeout=min(0.05, remaining))
				except TimeoutError:
					pass
		except BaseException as error:
			if attempt.phase is ContainmentPhase.REVOKED:
				return
			attempt.error = f'{type(error).__name__}: {error}'
			if attempt.requires_retirement:
				if await self._resolve_policy_protocol_error(error, operation, root, session):
					return
				attempt.phase = ContainmentPhase.UNCONFIRMED
				self._fail_policy_retirement(
					operation,
					attempt.error,
					reason='policy action is unconfirmed; close the owned browser or retain this remote runtime',
				)
			else:
				attempt.phase = (
					ContainmentPhase.FAILED
					if self._policy_owned(operation, root, session)
					else ContainmentPhase.SKIPPED
				)
		finally:
			if attempt.object_id is not None:
				try:
					await asyncio.wait_for(
						root.send.Runtime.releaseObject(
							params={'objectId': attempt.object_id}, session_id=operation.session_id
						),
						timeout=0.2,
					)
				except BaseException:
					pass

	async def _join_navigation_policy(
		self, operation: NavigationOperation, *, close: bool = True, retirement: bool = False
	) -> None:
		if close:
			operation.close_policy_capture()
		task = operation.policy_task
		if task is not None and task is not asyncio.current_task() and not task.done():
			attempt = operation.containment
			budget = (
				NAVIGATION_CONTAINMENT_GRACE
				if retirement or attempt is None or attempt.deadline is None
				else max(0.0, attempt.deadline - asyncio.get_running_loop().time())
			)
			await asyncio.wait({task}, timeout=budget)
		if task is not None and task.done() and not task.cancelled():
			task.result()
		if operation.policy_pending:
			if operation.containment and operation.containment.requires_retirement:
				operation.containment.phase = ContainmentPhase.UNCONFIRMED
			self._fail_policy_retirement(
				operation, 'Policy cleanup has not retired', reason='policy retirement incomplete'
			)
			raise NavigationError(f'Policy cleanup has not retired. {_POLICY_RETIREMENT_HINT}')

	async def _drain_navigation_policies(self) -> None:
		self._revoke_exited_policy_authorities()
		operations = list(self._navigation_policies.values())
		for operation in operations:
			operation.close_policy_capture()
			if (
				operation.policy_task
				and not operation.policy_task.done()
				and not (operation.containment and operation.containment.requires_retirement)
			):
				operation.policy_task.cancel()
		for operation in operations:
			await self._join_navigation_policy(operation, retirement=True)

	async def _navigation_step(self, operation: NavigationOperation, awaitable: Any) -> Any:
		task = asyncio.ensure_future(awaitable)
		operation.child_tasks.add(task)
		try:
			policy_task = operation.policy_task
			if policy_task is not None:
				await asyncio.wait({task, policy_task}, return_when=asyncio.FIRST_COMPLETED)
				if policy_task.done() and operation.containment is None:
					policy_task.result()
				if operation.containment is not None:
					await self._join_navigation_policy(operation, close=False)
					raise NavigationBlockedError(
						f'Navigation to {operation.containment.commit.url} blocked by security policy'
					)
			return await task
		finally:
			if not task.done():
				task.cancel()
			operation.child_tasks.discard(task)

	async def on_NavigateToUrlEvent(self, event: NavigateToUrlEvent) -> None:
		"""Run one correlated request under one deadline, without implicit replay."""
		operation = event.get_navigation_operation(self._lifecycle.generation)
		event_bus = self.event_bus
		loop = asyncio.get_running_loop()
		deadline = float('inf')
		acquired = False
		task = asyncio.current_task()
		try:
			if operation.terminal:
				raise NavigationError(f'Navigation is already {operation.phase.value}')
			deadline = self._navigation_deadline(event, operation)
			self._reserve_navigation(operation)
			if self._navigation_lock.locked():
				raise NavigationBusyError('Another navigation is in progress')
			await self._navigation_lock.acquire()
			acquired = True
			self._navigation_task, self._navigation_task_operation = task, operation
			async with asyncio.timeout_at(deadline):
				root = self._cdp_client_root
				if root is None or self.agent_focus is None:
					raise NavigationError('Cannot navigate: browser not connected')
				self._check_navigation_owner(operation, root)
				from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

				policy = self._security_watchdog or SecurityWatchdog(event_bus=event_bus, browser_session=self)
				if not policy._is_url_allowed(event.url):
					raise NavigationBlockedError(security_policy_blocked_message(event.url))

				target_id = None
				if event.new_tab:
					try:
						current_url = await self.get_current_page_url()
						self._check_navigation_owner(operation, root)
						if is_new_tab_page(current_url):
							event.new_tab = False
					except NavigationError:
						raise
					except Exception as error:
						self.logger.debug('Could not check current URL: %s', error)
				self._check_navigation_owner(operation, root)
				targets = await self._cdp_get_all_pages()
				self._check_navigation_owner(operation, root)
				for target in targets:
					if (
						target.get('url') == event.url
						and target['targetId'] != self.agent_focus.target_id
						and not event.new_tab
					):
						target_id = target['targetId']
						event.new_tab = False
				if event.new_tab:
					targets = await self._cdp_get_all_pages()
					self._check_navigation_owner(operation, root)
					for target in targets:
						if target.get('url') == 'about:blank' and target['targetId'] != self.agent_focus.target_id:
							target_id = target['targetId']
							break
					if not target_id:
						try:
							target_id = await self._cdp_create_new_page('about:blank')
						except NavigationError:
							raise
						except Exception as error:
							self._check_navigation_owner(operation, root)
							self.logger.debug('Tab creation failed, using current tab: %s', error)
							target_id = self.agent_focus.target_id
						else:
							self._check_navigation_owner(operation, root)
							await self._await_navigation_event(
								event_bus,
								TabCreatedEvent(
									target_id=target_id,
									url='about:blank',
									navigation_id=operation.navigation_id,
									navigation_generation=operation.generation,
								),
							)
							self._check_navigation_owner(operation, root)
				else:
					target_id = target_id or self.agent_focus.target_id
				if self.agent_focus.target_id != target_id:
					await self._await_navigation_event(
						event_bus,
						SwitchTabEvent(
							target_id=target_id,
							navigation_id=operation.navigation_id,
							navigation_generation=operation.generation,
						),
					)
					self._check_navigation_owner(operation, root)
				if self.agent_focus is None or self.agent_focus.target_id != target_id:
					raise NavigationError('Navigation target did not acquire focus')
				session = self.agent_focus
				operation.select_target(target_id, str(session.session_id))
				self._check_navigation_owner(operation, root, session)
				self.invalidate_dom_cache(target_id=target_id, reason='navigation_request')
				await self._await_navigation_event(
					event_bus,
					NavigationStartedEvent(
						target_id=target_id,
						url=event.url,
						navigation_id=operation.navigation_id,
						navigation_generation=operation.generation,
					),
				)
				self._check_navigation_owner(operation, root, session)
				self._navigation_monitor.bind(root)
				tree = await root.send.Page.getFrameTree(session_id=operation.session_id)
				self._check_navigation_owner(operation, root, session)
				operation.capture_document(tree['frameTree']['frame'])
				await root.send.Page.setLifecycleEventsEnabled(
					params={'enabled': True}, session_id=operation.session_id
				)
				self._check_navigation_owner(operation, root, session)
				operation.transition(NavigationPhase.NAVIGATING)
				self._start_navigation_policy(operation, root, session, policy, event_bus)
				response = await self._navigation_step(
					operation,
					root.send.Page.navigate(
						params={'url': event.url, 'transitionType': 'address_bar'}, session_id=operation.session_id
					),
				)
				self._check_navigation_owner(operation, root, session)
				operation.accept_response(response)
				self.invalidate_dom_cache(target_id=operation.target_id, reason='navigation_accepted')
				if not operation.is_download:
					await self._navigation_step(operation, operation.wait_ready())
					self._check_navigation_owner(operation, root, session)
					operation.transition(NavigationPhase.NOTIFYING)
				tree = await root.send.Page.getFrameTree(session_id=operation.session_id)
				self._check_navigation_owner(operation, root, session)
				frame = tree['frameTree']['frame']
				self._record_committed_document(operation.target_id, frame)
				if not operation.is_download and (
					frame.get('id') != operation.frame_id or frame.get('loaderId') != operation.loader_id
				):
					raise NavigationError('Navigation live document binding changed')
				operation.observed_url = frame.get('url', '') + frame.get('urlFragment', '')
				if not policy._is_url_allowed(operation.observed_url):
					operation.record_commit(frame)
					await self._join_navigation_policy(operation)
					raise NavigationBlockedError(f'Navigation to {operation.observed_url} blocked by security policy')
				await self._close_extension_options_pages()
				self._check_navigation_owner(operation, root, session)
				await self._await_navigation_event(
					event_bus,
					NavigationCompleteEvent(
						target_id=target_id,
						url=operation.observed_url,
						navigation_id=operation.navigation_id,
						navigation_generation=operation.generation,
					),
				)
				self._check_navigation_owner(operation, root, session)
				await self._await_navigation_event(
					event_bus,
					AgentFocusChangedEvent(
						target_id=target_id,
						url=operation.observed_url,
						navigation_id=operation.navigation_id,
						navigation_generation=operation.generation,
						navigation_session_id=operation.session_id,
					),
				)
				self._check_navigation_owner(operation, root, session)
				await self._join_navigation_policy(operation)
				self._check_navigation_owner(operation, root, session)
				if operation.containment is not None:
					raise NavigationBlockedError(
						f'Navigation to {operation.containment.commit.url} blocked by security policy'
					)
				operation.finish(NavigationPhase.DOWNLOAD if operation.is_download else NavigationPhase.SUCCEEDED)
		except asyncio.CancelledError:
			operation.finish(
				NavigationPhase.TIMED_OUT if loop.time() >= deadline else NavigationPhase.CANCELLED,
				'Navigation cancelled',
			)
			raise
		except TimeoutError as error:
			if loop.time() >= deadline:
				operation.finish(NavigationPhase.TIMED_OUT, 'Navigation deadline expired')
				raise NavigationTimeoutError('Navigation deadline expired') from error
			operation.finish(NavigationPhase.FAILED, error)
			raise NavigationError(str(error)) from error
		except Exception as error:
			operation.finish(NavigationPhase.FAILED, error)
			if isinstance(error, NavigationError):
				raise
			raise NavigationError(str(error)) from error
		finally:
			try:
				if operation.phase not in (NavigationPhase.SUCCEEDED, NavigationPhase.DOWNLOAD):
					tasks = self._cancel_navigation_tasks(operation)
					if tasks:
						await asyncio.wait(tasks, timeout=1.0)
				try:
					await self._join_navigation_policy(operation)
				except BaseException as error:
					self.logger.debug('Policy cleanup remains reserved: %s', error)
			finally:
				self._publish_navigation_finished(operation, event_bus)
				if self._navigation_task is task and self._navigation_task_operation is operation:
					self._navigation_task = self._navigation_task_operation = None
				self._retire_navigation(operation)
				if acquired:
					self._navigation_lock.release()

	@staticmethod
	def _owned_navigation_timeout(event_timeout: float | None) -> float:
		"""Keep the request deadline just inside the event bus deadline."""
		if event_timeout is None:
			return 30.0
		if not math.isfinite(event_timeout) or event_timeout < 0.01:
			raise NavigationError('Navigation timeout must be at least 0.01 seconds')
		return event_timeout - min(0.1, event_timeout / 2)

	async def _owned_ready_navigation(
		self,
		*,
		prepare: Callable[
			[CDPClient, CDPSession], Awaitable[tuple[str, Literal['history', 'reload'], Callable[[], Awaitable[None]]]]
		],
		navigation_id: str | None,
		event_timeout: float | None,
		wait_until: WaitUntil = 'load',
	) -> str:
		"""Run a caller-owned navigation that has no Page.navigate response body.

		Used by history and reload. The caller owns the target and the trigger,
		while the operation owns admission, the deadline, the commit evidence,
		the final binding check, and the terminal publication.
		"""
		event_bus = self.event_bus
		loop = asyncio.get_running_loop()
		timeout = self._owned_navigation_timeout(event_timeout)
		deadline = loop.time() + timeout
		operation: NavigationOperation | None = None
		blocked_url: str | None = None
		acquired = False
		task = asyncio.current_task()
		try:
			root = self._cdp_client_root
			focus = self.agent_focus
			if root is None or focus is None:
				raise NavigationError('Cannot navigate: browser not connected')
			if self._navigation_lock.locked():
				raise NavigationBusyError('Another navigation is in progress')
			await self._navigation_lock.acquire()
			acquired = True
			async with asyncio.timeout_at(deadline):
				requested_url, commit_kind, trigger = await prepare(root, focus)
				policy = self._security_watchdog
				if policy is None:
					from openbrowser.browser.watchdogs.security_watchdog import (
						SecurityWatchdog,
					)

					policy = SecurityWatchdog(event_bus=event_bus, browser_session=self)
				if not policy._is_url_allowed(requested_url):
					raise NavigationBlockedError(f'Navigation to {requested_url} blocked by security policy')
				operation = NavigationOperation(
					navigation_id or uuid7str(),
					self._lifecycle.generation,
					requested_url,
					wait_until,
					deadline=deadline,
				)
				self._reserve_navigation(operation)
				self._navigation_task, self._navigation_task_operation = task, operation
				operation.select_target(focus.target_id, focus.session_id)
				self.invalidate_dom_cache(target_id=focus.target_id, reason='navigation_request')
				await self._await_navigation_event(
					event_bus,
					NavigationStartedEvent(
						target_id=focus.target_id,
						url=requested_url,
						navigation_id=operation.navigation_id,
						navigation_generation=operation.generation,
					),
				)
				self._check_navigation_owner(operation, root, focus)
				self._navigation_monitor.bind(root)
				tree = await root.send.Page.getFrameTree(session_id=operation.session_id)
				operation.capture_document(tree['frameTree']['frame'])
				self._check_navigation_owner(operation, root, focus)
				await root.send.Page.setLifecycleEventsEnabled(
					params={'enabled': True}, session_id=operation.session_id
				)
				self._check_navigation_owner(operation, root, focus)
				operation.transition(NavigationPhase.NAVIGATING)
				operation.begin_trigger()
				if commit_kind == 'history':
					operation.expect_history_commit(requested_url)
				else:
					operation.expect_reload_commit()
				await trigger()
				self._check_navigation_owner(operation, root, focus)
				await operation.wait_commit()
				self._check_navigation_owner(operation, root, focus)
				tree = await root.send.Page.getFrameTree(session_id=operation.session_id)
				self._check_navigation_owner(operation, root, focus)
				committed_frame = tree['frameTree']['frame']
				self._record_committed_document(operation.target_id, committed_frame)
				if (
					committed_frame.get('id') != operation.frame_id
					or committed_frame.get('loaderId') != operation.loader_id
				):
					raise NavigationError('Navigation live document binding changed after commit')
				operation.observed_url = committed_frame.get('url', '') + committed_frame.get('urlFragment', '')
				committed_url = operation.observed_url or requested_url
				if not policy._is_url_allowed(committed_url):
					# Contain at commit, before a blocked document can finish loading.
					blocked_url = committed_url
					operation.transition(NavigationPhase.NOTIFYING)
				else:
					await operation.wait_ready()
					self._check_navigation_owner(operation, root, focus)
					tree = await root.send.Page.getFrameTree(session_id=operation.session_id)
					frame = tree['frameTree']['frame']
					if frame.get('id') != operation.frame_id or frame.get('loaderId') != operation.loader_id:
						raise NavigationError('Navigation live document binding changed')
					operation.observed_url = frame.get('url', '') + frame.get('urlFragment', '')
					if not policy._is_url_allowed(operation.observed_url):
						blocked_url = operation.observed_url
						operation.transition(NavigationPhase.NOTIFYING)
					else:
						if not self._navigation_focus_owns(focus, operation):
							raise NavigationError('Agent focus changed during navigation')
						operation.transition(NavigationPhase.NOTIFYING)
						await self._await_navigation_event(
							event_bus,
							NavigationCompleteEvent(
								target_id=focus.target_id,
								url=operation.observed_url,
								navigation_id=operation.navigation_id,
								navigation_generation=operation.generation,
							),
						)
						self._check_navigation_owner(operation, root, focus)
						await self._await_navigation_event(
							event_bus,
							AgentFocusChangedEvent(
								target_id=focus.target_id,
								url=operation.observed_url,
								navigation_id=operation.navigation_id,
								navigation_generation=operation.generation,
								navigation_session_id=operation.session_id,
							),
						)
						self._check_navigation_owner(operation, root, focus)
						tree = await root.send.Page.getFrameTree(session_id=operation.session_id)
						self._check_navigation_owner(operation, root, focus)
						live = tree['frameTree']['frame']
						if live.get('id') != operation.frame_id or live.get('loaderId') != operation.loader_id:
							raise NavigationError('Navigation live document binding changed after notification')
						if not self._navigation_focus_owns(focus, operation):
							raise NavigationError('Agent focus changed during navigation')
						operation.finish(NavigationPhase.SUCCEEDED)
						return operation.observed_url or requested_url
			if blocked_url is not None:
				assert operation is not None
				# The completion observation is unmarked on purpose: the request
				# deadline has expired, and containment must not depend on it.
				await self._await_navigation_event(
					event_bus,
					NavigationCompleteEvent(target_id=focus.target_id, url=blocked_url),
				)
				if not await self._confirm_blocked_containment(root, focus):
					raise NavigationError(
						f'Navigation to {blocked_url} was blocked, but page containment '
						'was not confirmed. Run get_status and reset_browser before continuing.'
					)
				raise NavigationBlockedError(f'Navigation to {blocked_url} blocked by security policy')
			raise NavigationError('Navigation ended without a terminal result')
		except asyncio.CancelledError:
			if operation is not None:
				operation.finish(
					NavigationPhase.TIMED_OUT if loop.time() >= deadline else NavigationPhase.CANCELLED,
					'Navigation cancelled',
				)
			raise
		except TimeoutError as error:
			expired = loop.time() >= deadline
			if operation is not None:
				if expired:
					operation.finish(NavigationPhase.TIMED_OUT, 'Navigation deadline expired')
				else:
					operation.finish(NavigationPhase.FAILED, error)
			if expired:
				raise NavigationTimeoutError('Navigation deadline expired') from error
			raise NavigationError(str(error)) from error
		except Exception as error:
			if operation is not None:
				operation.finish(NavigationPhase.FAILED, error)
			if isinstance(error, NavigationError):
				raise
			raise NavigationError(str(error)) from error
		finally:
			try:
				if operation is not None and operation.phase not in (
					NavigationPhase.SUCCEEDED,
					NavigationPhase.DOWNLOAD,
				):
					tasks = self._cancel_navigation_tasks(operation)
					if tasks:
						await asyncio.wait(tasks, timeout=1.0)
			finally:
				if operation is not None:
					self._publish_navigation_finished(operation, event_bus)
					if self._navigation_task is task and self._navigation_task_operation is operation:
						self._navigation_task = self._navigation_task_operation = None
					self._retire_navigation(operation)
				if acquired:
					self._navigation_lock.release()

	def _navigation_focus_owns(self, focus: CDPSession, operation: NavigationOperation) -> bool:
		return (
			self.agent_focus is not None
			and self.agent_focus.target_id == operation.target_id
			and self.agent_focus.session_id == operation.session_id
			and focus.target_id == operation.target_id
		)

	async def _confirm_blocked_containment(self, root: CDPClient, focus: CDPSession, timeout: float = 0.5) -> bool:
		"""Wait briefly for the security watchdog to blank a blocked page.

		Returns True only when the live frame is confirmed to be about:blank.
		"""
		loop = asyncio.get_running_loop()
		deadline = loop.time() + timeout
		while loop.time() < deadline:
			remaining = max(0.01, deadline - loop.time())
			try:
				tree = await asyncio.wait_for(
					root.send.Page.getFrameTree(session_id=focus.session_id), timeout=remaining
				)
				if tree['frameTree']['frame'].get('url') == 'about:blank':
					return True
			except Exception:
				return False
			await asyncio.sleep(min(0.05, max(0.0, deadline - loop.time())))
		self.logger.warning('Blocked navigation containment was not confirmed before the error returned')
		return False

	async def _history_target(
		self, root: CDPClient, focus: CDPSession, delta: int
	) -> tuple[str, Literal['history'], Callable[[], Awaitable[None]]]:
		history = await root.send.Page.getNavigationHistory(session_id=focus.session_id)
		entries = history.get('entries') or []
		current = history.get('currentIndex')
		if not isinstance(current, int) or not isinstance(entries, list):
			raise NavigationError('Page.getNavigationHistory returned a malformed response')
		index = current + delta
		if index < 0:
			raise NavigationError('Cannot go back: no previous history entry')
		if index >= len(entries):
			raise NavigationError('Cannot go forward: no next history entry')
		entry = entries[index]
		if not isinstance(entry, dict) or not isinstance(entry.get('id'), int) or not isinstance(entry.get('url'), str):
			raise NavigationError('Page.getNavigationHistory returned a malformed entry')
		entry_id = entry['id']

		async def trigger() -> None:
			await root.send.Page.navigateToHistoryEntry(params={'entryId': entry_id}, session_id=focus.session_id)

		return entry['url'], 'history', trigger

	async def go_back(self, *, navigation_id: str | None = None, event_timeout: float | None = None) -> str:
		"""Go back one history entry and wait for the page load to complete."""
		return await self._owned_ready_navigation(
			prepare=lambda root, focus: self._history_target(root, focus, -1),
			navigation_id=navigation_id,
			event_timeout=event_timeout,
		)

	async def go_forward(self, *, navigation_id: str | None = None, event_timeout: float | None = None) -> str:
		"""Go forward one history entry and wait for the page load to complete."""
		return await self._owned_ready_navigation(
			prepare=lambda root, focus: self._history_target(root, focus, 1),
			navigation_id=navigation_id,
			event_timeout=event_timeout,
		)

	async def refresh(self, *, navigation_id: str | None = None, event_timeout: float | None = None) -> str:
		"""Reload the current page and wait for the new document to load."""

		async def prepare(
			root: CDPClient, focus: CDPSession
		) -> tuple[str, Literal['reload'], Callable[[], Awaitable[None]]]:
			tree = await root.send.Page.getFrameTree(session_id=focus.session_id)
			frame = tree['frameTree']['frame']
			url = frame.get('url', '') + frame.get('urlFragment', '')

			async def trigger() -> None:
				await root.send.Page.reload(session_id=focus.session_id)

			return url, 'reload', trigger

		return await self._owned_ready_navigation(
			prepare=prepare,
			navigation_id=navigation_id,
			event_timeout=event_timeout,
		)

	async def on_SwitchTabEvent(self, event: SwitchTabEvent) -> TargetID:
		"""Switch focus to a live page through the serialized focus transaction."""
		budget = FocusTransitionBudget.from_event(
			deadline_monotonic=event.focus_deadline_monotonic,
			budget_seconds=None,
		)
		if event.navigation_id is not None:
			return await self._switch_navigation_target(event, budget)
		result = await self._focus_transition_to_target(
			event.target_id,
			reason=FocusChangeReason.SWITCH,
			budget=budget,
			excluded_target_id=None,
			create_if_missing=True,
			activate=True,
			probe=True,
		)
		event.target_id = result.session.target_id
		if result.created_target_id:
			await self._initialize_focus_page(result.created_target_id, budget)
		await self._notify_focus_changed(
			result.session.target_id,
			result.session.url or 'about:blank',
			FocusChangeReason.SWITCH,
			budget,
			result.snapshot,
		)
		return result.session.target_id

	async def _switch_navigation_target(
		self,
		event: SwitchTabEvent,
		budget: FocusTransitionBudget | None = None,
	) -> TargetID:
		root = self._cdp_client_root
		operation = self._marked_navigation(event, root)
		if not event.target_id:
			raise NavigationError('Navigation switch requires an explicit target')
		transition_budget = budget or FocusTransitionBudget.start()
		try:
			result = await self._focus_transition_to_target(
				event.target_id,
				reason=FocusChangeReason.NAVIGATION,
				budget=transition_budget,
				activate=True,
				probe=True,
				commit_guard=lambda: self._check_navigation_owner(operation, root),
			)
		except FocusTargetError as error:
			if isinstance(error.__cause__, NavigationError):
				raise error.__cause__ from error
			raise NavigationError(str(error)) from error
		session = result.session
		if session.cdp_client is not root:
			raise NavigationError('Navigation target root binding changed')
		await self._await_navigation_event(
			self.event_bus,
			AgentFocusChangedEvent(
				target_id=session.target_id,
				url=session.url,
				navigation_id=operation.navigation_id,
				navigation_generation=operation.generation,
				navigation_session_id=session.session_id,
				focus_generation=result.snapshot.generation,
				focus_session_id=result.snapshot.session_id,
				focus_reason=FocusChangeReason.NAVIGATION,
				focus_budget_seconds=transition_budget.remaining(),
				focus_deadline_monotonic=transition_budget.deadline,
				event_timeout=max(0.001, transition_budget.remaining()),
			),
		)
		self._check_navigation_owner(operation, root)
		return session.target_id

	async def on_CloseTabEvent(self, event: CloseTabEvent) -> None:
		"""Close one tab and run the owned focus repair within the same budget."""
		budget = FocusTransitionBudget.from_event(
			deadline_monotonic=event.focus_deadline_monotonic,
			budget_seconds=None,
		)
		focus_snapshot_before_close = self._focus_state
		record = self._tab_registry.get(event.target_id)
		if record is not None and record.state is TabState.CLOSED:
			await self._repair_focus_after_closed_tab(event.target_id, budget)
			return

		self._tab_registry.mark_closing(event.target_id)
		closed = False
		try:
			try:
				closing_event = self.event_bus.dispatch(TabClosingEvent(target_id=event.target_id))
				await self._await_event_result(
					closing_event,
					budget,
					raise_if_any=False,
					raise_if_none=False,
				)
			except Exception as error:
				self.logger.warning(f'Tab closing observer failed: {type(error).__name__}: {error}')
			try:
				close_result = await self._await_focus_operation(
					self.cdp_client.send.Target.closeTarget(params={'targetId': event.target_id}),
					budget,
					target_id=event.target_id,
					phase=self._focus_state.phase,
				)
				if isinstance(close_result, dict) and close_result.get('success') is False:
					raise RuntimeError('Target.closeTarget returned success=false')
			except FocusTargetError:
				raise
			except Exception as error:
				error_text = str(error).lower()
				already_closed = any(
					marker in error_text
					for marker in ('already closed', 'no target', 'target closed', 'not found', '-32001')
				)
				still_open = False if already_closed else await self._target_liveness(event.target_id, budget)
				if still_open is True:
					self._tab_registry.update(event.target_id, state=TabState.OPEN, allow_reopen=True)
					raise
				if still_open is False:
					closed = True
					self.logger.debug(f'Target was already closed: {error}')
				else:
					raise RuntimeError('Close target liveness is unknown') from error
			else:
				closed = True
		except asyncio.CancelledError:
			if not closed:
				record = self._tab_registry.get(event.target_id)
				if record is not None and record.state is TabState.CLOSING:
					self._schedule_close_reconciliation(event.target_id)
			raise
		except BaseException:
			if not closed:
				record = self._tab_registry.get(event.target_id)
				if record is not None and record.state is TabState.CLOSING:
					if budget.remaining() <= 0:
						self._schedule_close_reconciliation(event.target_id)
					else:
						try:
							still_open = await self._target_liveness(event.target_id, budget)
						except asyncio.CancelledError:
							self._schedule_close_reconciliation(event.target_id)
							raise
						except Exception:
							still_open = None
						record = self._tab_registry.get(event.target_id)
						if record is not None and record.state is TabState.CLOSING:
							if still_open is True:
								self._tab_registry.update(
									event.target_id,
									state=TabState.OPEN,
									allow_reopen=True,
								)
							elif still_open is False:
								self._tab_registry.mark_closed(event.target_id)
								self._schedule_close_reconciliation(event.target_id)
							else:
								self._schedule_close_reconciliation(event.target_id)
			raise

		self._tab_registry.mark_closed(event.target_id)
		event_timeout = self._focus_event_timeout(TabClosedEvent, budget)
		closed_event = self.event_bus.dispatch(
			TabClosedEvent(
				target_id=event.target_id,
				focus_deadline_monotonic=budget.deadline,
				event_timeout=max(0.001, event_timeout),
			)
		)
		try:
			await self._await_event_result(
				closed_event,
				budget,
				raise_if_any=True,
				raise_if_none=False,
			)
		except asyncio.CancelledError:
			self._schedule_close_reconciliation(event.target_id)
			raise
		except BaseException as error:
			post_close_error = FocusTargetError(
				f'Tab {event.target_id[-4:]} is closed, but post-close handling failed: '
				f'{type(error).__name__}: {error}',
				code=FocusTargetCode.RECOVERY_FAILED,
				phase=self._focus_state.phase,
				target_id=event.target_id,
				target_type='page',
			)
			await self._ensure_closed_target_focus_lost(
				event.target_id,
				focus_snapshot_before_close,
				post_close_error,
			)
			raise

	async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
		"""Handle tab creation - apply viewport settings to new tab."""
		if event.navigation_id is not None:
			await self._prepare_navigation_tab(event)
			return
		# Note: Tab switching prevention is handled by the Force Background Tab extension
		# The extension automatically keeps focus on the current tab when new tabs are created

		self._tab_registry.register(
			event.target_id,
			url=event.url,
			state=TabState.OPENING,
		)

		# Apply viewport settings if configured
		if self.browser_profile.viewport and not self.browser_profile.no_viewport:
			try:
				viewport_width = self.browser_profile.viewport.width
				viewport_height = self.browser_profile.viewport.height
				device_scale_factor = self.browser_profile.device_scale_factor or 1.0

				# Use the helper method with the new tab's target_id
				await self._cdp_set_viewport(
					viewport_width, viewport_height, device_scale_factor, target_id=event.target_id
				)

				self.logger.debug(f'Applied viewport {viewport_width}x{viewport_height} to tab {event.target_id[-8:]}')
			except Exception as e:
				self.logger.warning(f'Failed to set viewport for new tab {event.target_id[-8:]}: {e}')

	async def _prepare_navigation_tab(self, event: TabCreatedEvent) -> None:
		root = self._cdp_client_root
		operation = self._marked_navigation(event, root)
		if self.browser_profile.viewport and not self.browser_profile.no_viewport:
			session = await self.get_or_create_cdp_session(target_id=event.target_id, focus=False)
			self._check_navigation_owner(operation, root)
			if session.cdp_client is not root:
				raise NavigationError('Navigation new-tab root binding changed')
			session_id = str(session.session_id)
			await root.send.Emulation.setDeviceMetricsOverride(
				params={
					'width': self.browser_profile.viewport.width,
					'height': self.browser_profile.viewport.height,
					'deviceScaleFactor': self.browser_profile.device_scale_factor or 1.0,
					'mobile': False,
				},
				session_id=session_id,
			)
			self._check_navigation_owner(operation, root)
			if session.session_id != session_id:
				raise NavigationError('Navigation new-tab session binding changed')
		self._tab_registry.register(event.target_id, url=event.url, state=TabState.OPENING)

	async def on_TabClosedEvent(self, event: TabClosedEvent) -> None:
		"""Record one closed tab and repair focus if it pointed there."""
		self._tab_registry.mark_closed(event.target_id)
		budget = FocusTransitionBudget.from_event(
			deadline_monotonic=event.focus_deadline_monotonic,
			budget_seconds=None,
		)
		await self._repair_focus_after_closed_tab(event.target_id, budget)

	async def _repair_focus_after_closed_tab(
		self,
		target_id: str,
		budget: FocusTransitionBudget,
	) -> FocusSnapshot:
		"""Move focus away from one closed target or raise a typed failure."""
		snapshot = self._focus_state
		if snapshot.phase is FocusPhase.FOCUSED and snapshot.target_id != target_id:
			return snapshot
		if snapshot.phase is FocusPhase.LOST and snapshot.previous_target_id != target_id:
			return snapshot
		if snapshot.phase not in (FocusPhase.FOCUSED, FocusPhase.LOST):
			return snapshot
		created_target_id: str | None = None
		try:
			result = await self._focus_transition_to_target(
				None,
				reason=FocusChangeReason.CLOSE_REPAIR,
				budget=budget,
				excluded_target_id=target_id,
				create_if_missing=True,
				activate=True,
				probe=True,
				failure_mode='lost',
				phase=FocusPhase.RECOVERING,
				expected_generation=snapshot.generation,
			)
			created_target_id = result.created_target_id
			if created_target_id:
				await self._initialize_focus_page(created_target_id, budget)
			await self._notify_focus_changed(
				result.session.target_id,
				result.session.url or 'about:blank',
				FocusChangeReason.CLOSE_REPAIR,
				budget,
				result.snapshot,
			)
			return result.snapshot
		except FocusTargetError as error:
			created_target_id = created_target_id or error.created_target_id
			if created_target_id:
				await self._initialize_focus_page(created_target_id, budget)
			if error.code is FocusTargetCode.STALE_TRANSITION:
				current = self._focus_state
				if current.phase is FocusPhase.FOCUSED and current.target_id != target_id:
					return current
			if error.code is FocusTargetCode.OBSERVATION_FAILED:
				raise
			inner_message = error.to_snapshot().message
			repair_error = FocusTargetError(
				f'Tab {target_id[-4:]} is closed, but focus repair failed: {inner_message}',
				code=FocusTargetCode.RECOVERY_FAILED,
				phase=FocusPhase.LOST,
				target_id=target_id,
			)
			await self._ensure_closed_target_focus_lost(target_id, snapshot, repair_error)
			raise repair_error from error

	async def on_AgentFocusChangedEvent(self, event: AgentFocusChangedEvent) -> None:
		"""Validate one focus observation without clearing caches twice."""
		budget = FocusTransitionBudget.from_event(
			deadline_monotonic=event.focus_deadline_monotonic,
			budget_seconds=event.focus_budget_seconds,
		)
		if event.navigation_id is not None:
			await self._refresh_navigation_focus(event, budget)
			return
		if not event.target_id:
			raise FocusTargetError(
				'AgentFocusChangedEvent has no target ID.',
				code=FocusTargetCode.TARGET_MISSING,
				phase=self._focus_state.phase,
			)

		snapshot = self._focus_state
		if event.focus_generation is not None:
			if snapshot.generation != event.focus_generation:
				return
			if snapshot.target_id != event.target_id or snapshot.session_id != event.focus_session_id:
				return
			if self.agent_focus is None or not snapshot.is_exact_pair(
				self.agent_focus.target_id,
				self.agent_focus.session_id,
			):
				return
			if event.url:
				self._schedule_navigation_complete_observation(event.target_id, event.url)
			return

		result = await self._focus_transition_to_target(
			event.target_id,
			reason=event.focus_reason or FocusChangeReason.INITIAL,
			budget=budget,
			activate=False,
			probe=True,
			failure_mode='lost' if snapshot.phase is FocusPhase.UNINITIALIZED else 'restore',
		)
		if result.created_target_id:
			await self._initialize_focus_page(result.created_target_id, budget)
		if event.url:
			self._schedule_navigation_complete_observation(event.target_id, event.url)

	async def _refresh_navigation_focus(
		self,
		event: AgentFocusChangedEvent,
		budget: FocusTransitionBudget,
	) -> None:
		root = self._cdp_client_root
		operation = self._marked_navigation(event, root)

		def check() -> bool:
			self._check_navigation_owner(operation, root)
			return (
				self.agent_focus is not None
				and self.agent_focus.target_id == event.target_id
				and self.agent_focus.session_id == event.navigation_session_id
			)

		if not check():
			return
		session = await self.get_or_create_cdp_session(
			target_id=event.target_id,
			focus=False,
			budget=budget,
		)
		if not check():
			return
		if session.cdp_client is not root or session.session_id != event.navigation_session_id:
			raise NavigationError('Navigation focus session binding changed')
		try:
			result = await self._await_focus_operation(
				root.send.Runtime.evaluate(
					params={'expression': '1 + 1', 'returnByValue': True},
					session_id=event.navigation_session_id,
				),
				budget,
				target_id=event.target_id,
				phase=FocusPhase.FOCUSED,
				target_type='page',
			)
		except FocusTargetError as error:
			raise NavigationError(str(error)) from error
		if not check():
			return
		if result.get('result', {}).get('value') != 2:
			raise NavigationError('Navigation focus did not respond to JavaScript')
		self._tab_registry.update(session.target_id, url=event.url or session.url, title=session.title)
		if event.url:
			try:
				await self._await_focus_operation(
					self._await_navigation_event(
						self.event_bus,
						NavigationCompleteEvent(
							target_id=event.target_id,
							url=event.url,
							navigation_id=operation.navigation_id,
							navigation_generation=operation.generation,
						),
					),
					budget,
					target_id=event.target_id,
					phase=FocusPhase.FOCUSED,
					target_type='page',
				)
			except FocusTargetError as error:
				raise NavigationError(str(error)) from error
			check()

	async def on_FileDownloadedEvent(self, event: FileDownloadedEvent) -> None:
		"""Track downloaded files during this session."""
		self.logger.debug(f'FileDownloadedEvent received: {event.file_name} at {event.path}')
		if event.path and event.path not in self._downloaded_files:
			self._downloaded_files.append(event.path)
			self.logger.info(
				f'📁 Tracked download: {event.file_name} ({len(self._downloaded_files)} total downloads in session)'
			)
		else:
			if not event.path:
				self.logger.warning(f'FileDownloadedEvent has no path: {event}')
			else:
				self.logger.debug(f'File already tracked: {event.path}')

	@property
	def downloads_registry(self):
		"""Return the downloads registry, or None before the watchdog is attached."""
		return getattr(self._downloads_watchdog, '_download_registry', None)

	@property
	def download_outcomes_error(self) -> str | None:
		"""Return the last download-outcome reader failure, if any."""
		return self._download_outcomes_error

	async def _capture_download_history(self) -> list[dict[str, object]]:
		"""Cancel and drain in-flight downloads, then keep the final history."""
		watchdog = self._downloads_watchdog
		reader = getattr(watchdog, 'shutdown', None)
		if not callable(reader):
			return list(self._download_history)
		try:
			return [dict(outcome) for outcome in await reader()]
		except Exception:
			self.logger.warning('Could not capture download history', exc_info=True)
			snapshot = getattr(watchdog, 'download_outcomes', None)
			if callable(snapshot):
				try:
					return [dict(outcome) for outcome in snapshot()]
				except Exception:
					pass
			return [dict(outcome) for outcome in self._download_history]

	def download_outcomes(self) -> list[dict[str, object]]:
		"""Return bounded download outcomes, including history after a stop."""
		watchdog = self._downloads_watchdog
		reader = getattr(watchdog, 'download_outcomes', None)
		if callable(reader):
			try:
				outcomes = reader()
			except Exception as error:
				self._download_outcomes_error = f'{type(error).__name__}: {error}'
				self.logger.warning('Could not read download outcomes', exc_info=True)
			else:
				self._download_outcomes_error = None
				return [dict(outcome) for outcome in outcomes]
		return [dict(outcome) for outcome in self._download_history]

	def latest_download_outcome(self) -> dict[str, object] | None:
		"""Return the most recent download outcome, terminal or not."""
		watchdog = self._downloads_watchdog
		reader = getattr(watchdog, 'latest_download_outcome', None)
		if callable(reader):
			try:
				return reader()
			except Exception as error:
				self._download_outcomes_error = f'{type(error).__name__}: {error}'
				self.logger.warning('Could not read the latest download outcome', exc_info=True)
		return dict(self._download_history[-1]) if self._download_history else None

	def pdf_detection_snapshot(self) -> 'PdfDetectionSnapshot':
		"""Return the latest PDF detection snapshot, or an idle default.

		A reader failure propagates so ``downloads_summary`` can report it as
		``pdf_detection_error`` in the same response.
		"""
		from openbrowser.browser.download_state import (
			PdfDetectionPhase,
			PdfDetectionSnapshot,
			redacted_pdf_detection_snapshot,
		)

		watchdog = self._downloads_watchdog
		reader = getattr(watchdog, 'pdf_detection_snapshot', None)
		if callable(reader):
			value = reader()
			if isinstance(value, PdfDetectionSnapshot):
				return redacted_pdf_detection_snapshot(value)
		return PdfDetectionSnapshot(phase=PdfDetectionPhase.IDLE)

	async def wait_for_download(
		self,
		*,
		timeout: float = 30.0,
		url: str | None = None,
		download_id: str | None = None,
		download_ref: str | None = None,
		lookback: float = 0.0,
	) -> dict[str, object]:
		"""Wait for one terminal download outcome recorded by the watchdog."""
		watchdog = self._downloads_watchdog
		waiter = getattr(watchdog, 'wait_for_download', None)
		if not callable(waiter):
			raise RuntimeError('Downloads watchdog is not attached. Start the browser before waiting for a download.')
		return await waiter(
			timeout=timeout,
			url=url,
			download_id=download_id,
			download_ref=download_ref,
			lookback=lookback,
		)

	async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
		"""Handle browser stop request."""

		try:
			# Check if we should keep the browser alive
			if self.browser_profile.keep_alive and not event.force:
				self.event_bus.dispatch(BrowserStoppedEvent(reason='Kept alive due to keep_alive=True'))
				return

			# Clear CDP session cache before stopping. The active shutdown task
			# owns the lifecycle operation, so it must not wait on itself.
			active_reset = self._reset_task
			if active_reset is None or active_reset.done() or active_reset is asyncio.current_task():
				await self.reset(
					stop_event_bus=False,
					wait_for_operations=False,
					_reopen_admission=False,
					_health_reason=None,
				)

			# Reset state
			if self.is_local:
				self.browser_profile.cdp_url = None

			# Notify stop and wait for all handlers to complete
			# LocalBrowserWatchdog listens for BrowserStopEvent and dispatches BrowserKillEvent
			stop_event = self.event_bus.dispatch(BrowserStoppedEvent(reason='Stopped by request'))
			await stop_event

		except Exception as e:
			self._stop_reset_error = e
			self.event_bus.dispatch(
				BrowserErrorEvent(
					error_type='BrowserStopEventError',
					message=f'Failed to stop browser: {type(e).__name__} {e}',
					details={'cdp_url': self.cdp_url, 'is_local': self.is_local},
				)
			)
			raise

	# region - ========== CDP-based replacements for browser_context operations ==========
	@property
	def cdp_client(self) -> CDPClient:
		"""Get the cached root CDP cdp_session.cdp_client. The client is created and started in self.connect()."""
		assert self._cdp_client_root is not None, 'CDP client not initialized - browser may not be connected yet'
		return self._cdp_client_root

	async def new_page(self, url: str | None = None) -> 'Page':
		"""Create a new page (tab)."""
		from cdp_use.cdp.target.commands import CreateTargetParameters

		params: CreateTargetParameters = {'url': url or 'about:blank'}
		result = await self.cdp_client.send.Target.createTarget(params)

		target_id = result['targetId']

		# Import here to avoid circular import
		from openbrowser.actor.page import Page as Target

		return Target(self, target_id)

	async def get_current_page(self) -> 'Page | None':
		"""Get the current page as an actor Page."""
		target_info = await self.get_current_target_info()

		if not target_info:
			return None

		from openbrowser.actor.page import Page as Target

		return Target(self, target_info['targetId'])

	async def must_get_current_page(self) -> 'Page':
		"""Get the current page as an actor Page."""
		page = await self.get_current_page()
		if not page:
			raise RuntimeError('No current target found')

		return page

	async def get_pages(self) -> list['Page']:
		"""Get all available pages."""
		result = await self.cdp_client.send.Target.getTargets()

		targets = []
		# Import here to avoid circular import
		from openbrowser.actor.page import Page as Target

		for target_info in result['targetInfos']:
			if target_info['type'] in ['page', 'iframe']:
				targets.append(Target(self, target_info['targetId']))

		return targets

	async def close_page(self, page: 'Union[Page, str]') -> None:
		"""Close a page by Page object or target ID."""
		from cdp_use.cdp.target.commands import CloseTargetParameters

		# Import here to avoid circular import
		from openbrowser.actor.page import Page as Target

		if isinstance(page, Target):
			target_id = page._target_id
		else:
			target_id = str(page)

		params: CloseTargetParameters = {'targetId': target_id}
		await self.cdp_client.send.Target.closeTarget(params)

	async def cookies(self, urls: list[str] | None = None) -> list['Cookie']:
		"""Get cookies, optionally filtered by URLs."""
		# `library` re-exports this only under TYPE_CHECKING; at runtime it lives in `commands`.
		from cdp_use.cdp.network.commands import GetCookiesParameters

		params: GetCookiesParameters = {}
		if urls:
			params['urls'] = urls

		result = await self.cdp_client.send.Network.getCookies(params)
		return result['cookies']

	async def clear_cookies(self) -> None:
		"""Clear all cookies."""
		await self.cdp_client.send.Network.clearBrowserCookies()

	async def export_storage_state(self, output_path: str | Path | None = None) -> dict[str, Any]:
		"""Export all browser cookies and storage to storage_state format.

		Extracts decrypted cookies via CDP, bypassing keychain encryption.

		Args:
                output_path: Optional path to save storage_state.json. If None, returns dict only.

		Returns:
                Storage state dict with cookies in Playwright format.

		"""
		from pathlib import Path

		# Get all cookies using Storage.getCookies (returns decrypted cookies from all domains)
		cookies = await self._cdp_get_cookies()

		# Convert CDP cookie format to Playwright storage_state format
		storage_state = {
			'cookies': [
				{
					'name': c['name'],
					'value': c['value'],
					'domain': c['domain'],
					'path': c['path'],
					'expires': c.get('expires', -1),
					'httpOnly': c.get('httpOnly', False),
					'secure': c.get('secure', False),
					'sameSite': c.get('sameSite', 'Lax'),
				}
				for c in cookies
			],
			'origins': [],  # Could add localStorage/sessionStorage extraction if needed
		}

		if output_path:
			import json

			output_file = Path(output_path).expanduser().resolve()
			output_file.parent.mkdir(parents=True, exist_ok=True)
			output_file.write_text(json.dumps(storage_state, indent=2))
			self.logger.info(f'💾 Exported {len(cookies)} cookies to {output_file}')

		return storage_state

	def _ensure_session_returnable(
		self,
		session: CDPSession,
		*,
		implicit_snapshot: FocusSnapshot | None = None,
	) -> None:
		"""Reject a stale or closed binding before returning it to a caller."""
		record = self._tab_registry.get(session.target_id)
		if record is not None and record.state in (TabState.CLOSING, TabState.CLOSED, TabState.CRASHED):
			raise FocusTargetError(
				f'Target {session.target_id[-8:]} is {record.state.value} and cannot return a session.',
				code=FocusTargetCode.TAB_NOT_LIVE,
				phase=self._focus_state.phase,
				target_id=session.target_id,
				target_type='page',
			)
		if implicit_snapshot is None:
			return
		current = self._focus_state
		if (
			current.generation != implicit_snapshot.generation
			or not current.is_exact_pair(session.target_id, session.session_id)
		):
			raise FocusTargetError(
				'Focus changed while resolving the implicit browser session.',
				code=FocusTargetCode.STALE_TRANSITION,
				phase=current.phase,
				target_id=current.target_id or session.target_id,
				target_type='page',
			)

	async def get_or_create_cdp_session(
		self,
		target_id: TargetID | None = None,
		focus: bool = True,
		*,
		focus_reason: FocusChangeReason = FocusChangeReason.SWITCH,
		budget: FocusTransitionBudget | None = None,
	) -> CDPSession:
		"""Get a CDP session for a target from the event-driven pool.

		Args:
                target_id: Target ID to get session for. If None, uses current agent focus.
                focus: If True, commits focus after page validation, activation, and probe.
                focus_reason: Reason recorded for a focus commit.
                budget: Optional shared focus deadline.

		Returns:
                CDPSession for the specified target.

		Raises:
                FocusTargetError: If the target is not a live page or has no exact session.
		"""
		assert self._cdp_client_root is not None, 'Root CDP client not initialized'
		assert self._session_manager is not None, 'SessionManager not initialized'

		transition_budget = budget or FocusTransitionBudget.start()
		snapshot = self._focus_state
		implicit_target = target_id is None
		if target_id is None:
			if snapshot.phase is FocusPhase.FOCUSED and snapshot.target_id:
				target_id = snapshot.target_id
			elif self.agent_focus is not None:
				target_id = self.agent_focus.target_id
			else:
				raise FocusTargetError(
					'No browser page is focused.',
					code=FocusTargetCode.NO_PAGE_AVAILABLE,
					phase=snapshot.phase,
				)

		expected_generation = snapshot.generation if implicit_target else None
		if focus:
			if snapshot.phase is FocusPhase.FOCUSED and snapshot.target_id == target_id:
				session = await self._await_focus_operation(
					self._session_manager.get_live_session_for_target(
						target_id,
						budget=transition_budget,
					),
					transition_budget,
					target_id=target_id,
					phase=snapshot.phase,
					target_type='page',
				)
				if session is not None and session.session_id == snapshot.session_id:
					try:
						await self._await_focus_operation(
							session.cdp_client.send.Runtime.runIfWaitingForDebugger(session_id=session.session_id),
							transition_budget,
							target_id=target_id,
							phase=FocusPhase.FOCUSED,
							target_type='page',
						)
					except FocusTargetError:
						raise
					except Exception as error:
						missing_session = self._is_missing_cdp_session_error(error)
						still_live = False
						if not missing_session:
							still_live = await self._focus_exact_session_is_live(
								target_id,
								snapshot.session_id,
								transition_budget,
								phase=FocusPhase.FOCUSED,
								use_active_ids=True,
							)
						if not missing_session and still_live:
							self._ensure_session_returnable(
								session,
								implicit_snapshot=snapshot if implicit_target else None,
							)
							return session
						remaining_session_ids: tuple[str, ...] = ()
						try:
							active_ids = await self._session_manager.active_session_ids(
								target_id,
								budget=transition_budget,
							)
							remaining_session_ids = tuple(
								session_id for session_id in active_ids if session_id != snapshot.session_id
							)
						except Exception:
							remaining_session_ids = ()
						await self._handle_detached_session(
							target_id,
							session.session_id,
							remaining_session_ids,
						)
						if implicit_target and self._focus_state.phase is FocusPhase.FOCUSED:
							replacement = self.agent_focus
							if replacement is not None and replacement.target_id == target_id:
								self._ensure_session_returnable(
									replacement,
									implicit_snapshot=self._focus_state,
								)
								return replacement
							raise FocusTargetError(
								'Focus changed to a different target while resolving the implicit browser session.',
								code=FocusTargetCode.STALE_TRANSITION,
								phase=self._focus_state.phase,
								target_id=self._focus_state.target_id,
								target_type='page',
							) from error
						raise FocusTargetError(
							f'Focus session for target {target_id[-8:]} is no longer live.',
							code=FocusTargetCode.TARGET_DETACHED,
							phase=self._focus_state.phase,
							target_id=target_id,
							target_type='page',
						) from error
					self._ensure_session_returnable(
						session,
						implicit_snapshot=snapshot if implicit_target else None,
					)
					return session
				if session is not None:
					result = await self._focus_transition_to_target(
						target_id,
						reason=FocusChangeReason.SESSION_REBIND,
						budget=transition_budget,
						activate=False,
						probe=False,
						expected_generation=snapshot.generation,
					)
					self._ensure_session_returnable(
						result.session,
						implicit_snapshot=result.snapshot if implicit_target else None,
					)
					return result.session
			result = await self._focus_transition_to_target(
				target_id,
				reason=focus_reason,
				budget=transition_budget,
				activate=True,
				probe=True,
				expected_generation=expected_generation,
			)
			try:
				await self._await_focus_operation(
					result.session.cdp_client.send.Runtime.runIfWaitingForDebugger(
						session_id=result.session.session_id
					),
					transition_budget,
					target_id=result.session.target_id,
					phase=FocusPhase.FOCUSED,
					target_type='page',
				)
			except FocusTargetError:
				raise
			except Exception as error:
				if self._is_missing_cdp_session_error(error):
					remaining_session_ids: tuple[str, ...] = ()
					try:
						active_ids = await self._session_manager.active_session_ids(
							target_id,
							budget=transition_budget,
						)
						remaining_session_ids = tuple(
							session_id for session_id in active_ids if session_id != result.session.session_id
						)
					except Exception:
						remaining_session_ids = ()
					if remaining_session_ids:
						replacement = await self._session_manager.get_live_session_for_target(
							target_id,
							budget=transition_budget,
						)
						if replacement is not None and replacement.session_id != result.session.session_id:
							await self._set_agent_focus(
								replacement,
								FocusChangeReason.SESSION_REBIND,
								expected_generation=result.snapshot.generation,
							)
							self._ensure_session_returnable(
								replacement,
								implicit_snapshot=self._focus_state if implicit_target else None,
							)
							return replacement
					await self._handle_detached_session(target_id, result.session.session_id, ())
					raise FocusTargetError(
						f'Focus session for target {target_id[-8:]} detached before it could be returned.',
						code=FocusTargetCode.TARGET_DETACHED,
						phase=self._focus_state.phase,
						target_id=target_id,
						target_type='page',
					) from error
			self._ensure_session_returnable(
				result.session,
				implicit_snapshot=result.snapshot if implicit_target else None,
			)
			return result.session

		session = await self._get_focus_session_locked(target_id, transition_budget)
		self._ensure_session_returnable(
			session,
			implicit_snapshot=snapshot if implicit_target else None,
		)
		return session

	@property
	def current_target_id(self) -> str | None:
		return self.agent_focus.target_id if self.agent_focus else None

	@property
	def current_session_id(self) -> str | None:
		return self.agent_focus.session_id if self.agent_focus else None

	# endregion - ========== CDP-based ... ==========

	# region - ========== Helper Methods ==========
	@observe_debug(ignore_input=True, ignore_output=True, name='get_browser_state_summary')
	async def get_browser_state_summary(
		self,
		include_screenshot: bool = True,
		cached: bool = False,
		include_recent_events: bool = False,
	) -> BrowserStateSummary:
		if self.browser_profile.effective_render_mode is RenderMode.NO_RENDER:
			include_screenshot = False
		cached_summary = self._cached_browser_state_summary
		cached_target_id = self.current_target_id
		if (
			cached
			and cached_summary is not None
			and getattr(cached_summary, 'dom_state', None)
			and cached_target_id is not None
		):
			selector_map = cached_summary.dom_state.selector_map
			if not self._dom_cache.has_entry(cached_target_id):
				identity = await self._read_document_identity(cached_target_id)
				if identity is None or not self._dom_cache.adopt(
					cached_target_id,
					selector_count=len(selector_map),
					has_summary=True,
					identity=identity,
				):
					self._cached_browser_state_summary = None
					cached_summary = None
			if cached_summary is not None and (
				not self._dom_cache.summary_is_current(cached_target_id)
				or not await self._validate_dom_cache_identity(cached_target_id)
				or self._cached_browser_state_summary is not cached_summary
			):
				self.logger.debug('⚠️ Cached browser state belongs to an older document, fetching fresh state')
				self._cached_browser_state_summary = None
				cached_summary = None
			if cached_summary is not None:
				# Don't use cached state if it has 0 interactive elements
				# Don't use cached state if we need a screenshot but the cached state doesn't have one
				if include_screenshot and not cached_summary.screenshot:
					self.logger.debug('⚠️ Cached browser state has no screenshot, fetching fresh state with screenshot')
					# Fall through to fetch fresh state with screenshot
				elif selector_map and len(selector_map) > 0:
					self.logger.debug('🔄 Using pre-cached browser state summary for open tab')
					return cached_summary
				else:
					self.logger.debug('⚠️ Cached browser state has 0 interactive elements, fetching fresh state')
					# Fall through to fetch fresh state

		# Dispatch the event and wait for result
		event: BrowserStateRequestEvent = cast(
			BrowserStateRequestEvent,
			self.event_bus.dispatch(
				BrowserStateRequestEvent(
					include_dom=True,
					include_screenshot=include_screenshot,
					include_recent_events=include_recent_events,
				)
			),
		)

		# The handler returns the BrowserStateSummary directly
		result = await event.event_result(raise_if_none=True, raise_if_any=True)
		assert result is not None and result.dom_state is not None
		return result

	async def get_state_as_text(self) -> str:
		"""Get the browser state as text."""
		state = await self.get_browser_state_summary()
		assert state.dom_state is not None
		dom_state = state.dom_state
		return dom_state.llm_representation()

	async def attach_all_watchdogs(self) -> None:
		"""Initialize and attach all watchdogs with explicit handler registration."""
		# Prevent duplicate watchdog attachment
		if hasattr(self, '_watchdogs_attached') and self._watchdogs_attached:
			self.logger.debug('Watchdogs already attached, skipping duplicate attachment')
			return

		from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

		# from openbrowser.browser.crash_watchdog import CrashWatchdog
		from openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
		from openbrowser.browser.watchdogs.dom_watchdog import DOMWatchdog
		from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog
		from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
		from openbrowser.browser.watchdogs.permissions_watchdog import PermissionsWatchdog
		from openbrowser.browser.watchdogs.popups_watchdog import PopupsWatchdog
		from openbrowser.browser.watchdogs.recording_watchdog import RecordingWatchdog
		from openbrowser.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog
		from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog
		from openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

		retained_local = (
			self._local_browser_watchdog
			if self._watchdog_has_live_resources(self._local_browser_watchdog)
			else None
		)

		# Initialize CrashWatchdog
		# CrashWatchdog.model_rebuild()
		# self._crash_watchdog = CrashWatchdog(event_bus=self.event_bus, browser_session=self)
		# self.event_bus.on(BrowserConnectedEvent, self._crash_watchdog.on_BrowserConnectedEvent)
		# self.event_bus.on(BrowserStoppedEvent, self._crash_watchdog.on_BrowserStoppedEvent)
		# self._crash_watchdog.attach_to_session()

		# Initialize DownloadsWatchdog
		DownloadsWatchdog.model_rebuild()
		self._downloads_watchdog = DownloadsWatchdog(event_bus=self.event_bus, browser_session=self)
		# self.event_bus.on(BrowserLaunchEvent, self._downloads_watchdog.on_BrowserLaunchEvent)
		# self.event_bus.on(TabCreatedEvent, self._downloads_watchdog.on_TabCreatedEvent)
		# self.event_bus.on(TabClosedEvent, self._downloads_watchdog.on_TabClosedEvent)
		# self.event_bus.on(BrowserStoppedEvent, self._downloads_watchdog.on_BrowserStoppedEvent)
		# self.event_bus.on(NavigationCompleteEvent, self._downloads_watchdog.on_NavigationCompleteEvent)
		self._downloads_watchdog.attach_to_session()
		if self.browser_profile.auto_download_pdfs:
			self.logger.debug('📄 PDF auto-download enabled for this session')

		# Initialize StorageStateWatchdog conditionally
		# Enable when user provides either storage_state or user_data_dir (indicating they want persistence)
		should_enable_storage_state = (
			self.browser_profile.storage_state is not None or self.browser_profile.user_data_dir is not None
		)

		if should_enable_storage_state:
			StorageStateWatchdog.model_rebuild()
			self._storage_state_watchdog = StorageStateWatchdog(
				event_bus=self.event_bus,
				browser_session=self,
				# More conservative defaults when auto-enabled
				auto_save_interval=60.0,  # 1 minute instead of 30 seconds
				save_on_change=False,  # Only save on shutdown by default
			)
			self._storage_state_watchdog.attach_to_session()
			self.logger.debug(
				f'🍪 StorageStateWatchdog enabled (storage_state: {bool(self.browser_profile.storage_state)}, user_data_dir: {bool(self.browser_profile.user_data_dir)})'
			)
		else:
			self.logger.debug('🍪 StorageStateWatchdog disabled (no storage_state or user_data_dir configured)')

		# Initialize LocalBrowserWatchdog. A keep_alive stop retains the process
		# and profile lease, so the replacement watchdog adopts that owner
		# instead of starting a second browser or abandoning the lease.
		LocalBrowserWatchdog.model_rebuild()
		local_browser_watchdog = LocalBrowserWatchdog(event_bus=self.event_bus, browser_session=self)
		if retained_local is not None:
			local_browser_watchdog.adopt_owner(retained_local)
		self._local_browser_watchdog = local_browser_watchdog
		# self.event_bus.on(BrowserLaunchEvent, self._local_browser_watchdog.on_BrowserLaunchEvent)
		# self.event_bus.on(BrowserKillEvent, self._local_browser_watchdog.on_BrowserKillEvent)
		# self.event_bus.on(BrowserStopEvent, self._local_browser_watchdog.on_BrowserStopEvent)
		self._local_browser_watchdog.attach_to_session()

		# Initialize SecurityWatchdog (hooks NavigationWatchdog and implements allowed_domains restriction)
		SecurityWatchdog.model_rebuild()
		self._security_watchdog = SecurityWatchdog(event_bus=self.event_bus, browser_session=self)
		# Core navigation is now handled in BrowserSession directly
		# SecurityWatchdog only handles security policy enforcement
		self._security_watchdog.attach_to_session()

		# Initialize AboutBlankWatchdog (handles about:blank pages and DVD loading animation on first load)
		AboutBlankWatchdog.model_rebuild()
		self._aboutblank_watchdog = AboutBlankWatchdog(event_bus=self.event_bus, browser_session=self)
		# self.event_bus.on(BrowserStopEvent, self._aboutblank_watchdog.on_BrowserStopEvent)
		# self.event_bus.on(BrowserStoppedEvent, self._aboutblank_watchdog.on_BrowserStoppedEvent)
		# self.event_bus.on(TabCreatedEvent, self._aboutblank_watchdog.on_TabCreatedEvent)
		# self.event_bus.on(TabClosedEvent, self._aboutblank_watchdog.on_TabClosedEvent)
		self._aboutblank_watchdog.attach_to_session()

		# Initialize PopupsWatchdog (handles accepting and dismissing JS dialogs, alerts, confirm, onbeforeunload, etc.)
		PopupsWatchdog.model_rebuild()
		self._popups_watchdog = PopupsWatchdog(event_bus=self.event_bus, browser_session=self)
		# self.event_bus.on(TabCreatedEvent, self._popups_watchdog.on_TabCreatedEvent)
		# self.event_bus.on(DialogCloseEvent, self._popups_watchdog.on_DialogCloseEvent)
		self._popups_watchdog.attach_to_session()

		# Initialize PermissionsWatchdog (handles granting and revoking browser permissions like clipboard, microphone, camera, etc.)
		PermissionsWatchdog.model_rebuild()
		self._permissions_watchdog = PermissionsWatchdog(event_bus=self.event_bus, browser_session=self)
		# self.event_bus.on(BrowserConnectedEvent, self._permissions_watchdog.on_BrowserConnectedEvent)
		self._permissions_watchdog.attach_to_session()

		# Initialize DefaultActionWatchdog (handles all default actions like click, type, scroll, go back, go forward, refresh, wait, send keys, upload file, scroll to text, etc.)
		DefaultActionWatchdog.model_rebuild()
		self._default_action_watchdog = DefaultActionWatchdog(event_bus=self.event_bus, browser_session=self)
		# self.event_bus.on(ClickElementEvent, self._default_action_watchdog.on_ClickElementEvent)
		# self.event_bus.on(TypeTextEvent, self._default_action_watchdog.on_TypeTextEvent)
		# self.event_bus.on(ScrollEvent, self._default_action_watchdog.on_ScrollEvent)
		# self.event_bus.on(GoBackEvent, self._default_action_watchdog.on_GoBackEvent)
		# self.event_bus.on(GoForwardEvent, self._default_action_watchdog.on_GoForwardEvent)
		# self.event_bus.on(RefreshEvent, self._default_action_watchdog.on_RefreshEvent)
		# self.event_bus.on(WaitEvent, self._default_action_watchdog.on_WaitEvent)
		# self.event_bus.on(SendKeysEvent, self._default_action_watchdog.on_SendKeysEvent)
		# self.event_bus.on(UploadFileEvent, self._default_action_watchdog.on_UploadFileEvent)
		# self.event_bus.on(ScrollToTextEvent, self._default_action_watchdog.on_ScrollToTextEvent)
		self._default_action_watchdog.attach_to_session()

		# Initialize ScreenshotWatchdog (handles taking screenshots of the browser)
		ScreenshotWatchdog.model_rebuild()
		self._screenshot_watchdog = ScreenshotWatchdog(event_bus=self.event_bus, browser_session=self)
		# self.event_bus.on(BrowserStartEvent, self._screenshot_watchdog.on_BrowserStartEvent)
		# self.event_bus.on(BrowserStoppedEvent, self._screenshot_watchdog.on_BrowserStoppedEvent)
		# self.event_bus.on(ScreenshotEvent, self._screenshot_watchdog.on_ScreenshotEvent)
		self._screenshot_watchdog.attach_to_session()

		# Initialize DOMWatchdog (handles building the DOM tree and detecting interactive elements, depends on ScreenshotWatchdog)
		DOMWatchdog.model_rebuild()
		self._dom_watchdog = DOMWatchdog(event_bus=self.event_bus, browser_session=self)
		# self.event_bus.on(TabCreatedEvent, self._dom_watchdog.on_TabCreatedEvent)
		# self.event_bus.on(BrowserStateRequestEvent, self._dom_watchdog.on_BrowserStateRequestEvent)
		self._dom_watchdog.attach_to_session()

		# Initialize RecordingWatchdog (handles video recording)
		RecordingWatchdog.model_rebuild()
		self._recording_watchdog = RecordingWatchdog(event_bus=self.event_bus, browser_session=self)
		self._recording_watchdog.attach_to_session()

		# Mark watchdogs as attached to prevent duplicate attachment
		self._watchdogs_attached = True

	async def connect(self, cdp_url: str | None = None, *, _start_owned: bool = False) -> Self:
		"""Serialize one CDP connection attempt."""
		if self._event_bus_retirement_pending:
			raise RuntimeError('Event bus retirement is still pending; run reset() before connecting')
		acquired = False
		current_task = asyncio.current_task() if not _start_owned else None
		admission_epoch = self._connect_admission_epoch
		if not _start_owned:
			self._raise_if_connect_admission_closed()
			if self._start_admission_tasks:
				raise RuntimeError('Browser start cleanup is still in progress; retry after it finishes')
			if current_task is not None:
				self._direct_connect_tasks.add(current_task)
		try:
			if not _start_owned:
				await self._connect_admission_lock.acquire()
				acquired = True
				if self._connect_admission_epoch != admission_epoch:
					self._raise_if_connect_admission_closed()
					raise RuntimeError(
						'Browser session shutdown closed direct connections; call start() before connecting'
					)
			async with self._connect_lock:
				return await self._connect_locked(cdp_url, _start_owned=_start_owned)
		finally:
			if current_task is not None:
				self._direct_connect_tasks.discard(current_task)
			if acquired:
				self._connect_admission_lock.release()

	async def _connect_locked(self, cdp_url: str | None = None, *, _start_owned: bool = False) -> Self:
		"""Connect to a remote chromium-based browser via CDP using cdp-use.

		This MUST succeed or the browser is unusable. Fails hard on any error.
		"""
		start_generation = self._lifecycle.generation
		start_owner = self._start_task if _start_owned else None
		self._raise_if_connect_admission_closed(start_owned=_start_owned)
		if self._event_bus_retirement_pending:
			raise RuntimeError('Event bus retirement is still pending; run reset() before connecting')
		active_start = self._start_task
		if (
			not _start_owned
			and active_start is not None
			and not active_start.done()
			and active_start is not asyncio.current_task()
		):
			raise RuntimeError('Browser start cleanup is still in progress; retry after it finishes')
		if self._has_cleanup_in_flight():
			raise RuntimeError('Browser cleanup is still running; retry after it finishes')
		if self._cdp_health.cleanup_pending:
			raise RuntimeError('CDP connection cleanup is still pending; run kill() before reconnecting')
		self._check_failed_runtime_retirement()
		if self._cdp_client_root is not None or self._session_manager is not None:
			raise RuntimeError('CDP connection already exists; run kill() or reset() before connecting again')
		generation = self._cdp_health.begin_connect()
		self._health_owned_root_binding = None
		try:
			self.browser_profile.cdp_url = cdp_url or self.cdp_url
			if not self.cdp_url:
				raise RuntimeError('Cannot setup CDP connection without CDP URL')

			if not self.cdp_url.startswith('ws'):
				# If it's an HTTP URL, fetch the WebSocket URL from /json/version endpoint
				url = self.cdp_url.rstrip('/')
				if not url.endswith('/json/version'):
					url = url + '/json/version'

				# Run a tiny HTTP client to query for the WebSocket URL from the /json/version endpoint
				# Retry up to 30s -- browser may still be starting and not serving CDP yet
				async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
					headers = self.browser_profile.headers or {}
					max_wait = 30.0
					start = asyncio.get_event_loop().time()
					while True:
						try:
							version_info = await client.get(url, headers=headers)
							self._check_failed_runtime_retirement()
							discovered_url = version_info.json().get('webSocketDebuggerUrl')
							if not isinstance(discovered_url, str) or not discovered_url:
								raise RuntimeError('Browser CDP endpoint did not return a debugger URL')
							self.browser_profile.cdp_url = discovered_url
							break
						except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError):
							elapsed = asyncio.get_event_loop().time() - start
							if elapsed >= max_wait:
								message = f'Browser CDP endpoint {url} did not respond within {max_wait}s'
								raise TimeoutError(sanitize_error_detail(message) or message) from None
							await asyncio.sleep(0.3)
			self._raise_if_connect_admission_closed(start_owned=_start_owned)
		except asyncio.CancelledError as error:
			if self._caller_cancellation_requested():
				await self._await_connect_failure(generation, error, None, None, cancelled=True)
				raise
			worker_error = RuntimeError('CDP connect worker was cancelled')
			await self._await_connect_failure(
				generation, error, None, None, cancelled=False, worker_cancelled=True
			)
			raise worker_error from error
		except Exception as error:
			if isinstance(error, NavigationError):
				self._cdp_health.mark_connect_failed(
					generation, error, client_present=self._cdp_client_root is not None
				)
				self._health.record_connect_failed(generation)
				raise
			await self._await_connect_failure(generation, error, None, None, cancelled=False)
			raise

		assert self.cdp_url is not None

		browser_location = 'local browser' if self.is_local else 'remote browser'
		self.logger.debug(
			f'🌎 Connecting to existing chromium-based browser via CDP: {self.cdp_url} -> ({browser_location})'
		)
		try:
			self._check_failed_runtime_retirement()
			self._raise_if_connect_admission_closed(start_owned=_start_owned)
		except asyncio.CancelledError as error:
			self._cdp_health.mark_connect_failed(
				generation, error, client_present=self._cdp_client_root is not None, cancelled=True
			)
			self._health.record_connect_failed(generation)
			raise
		except Exception as error:
			self._cdp_health.mark_connect_failed(
				generation, error, client_present=self._cdp_client_root is not None
			)
			self._health.record_connect_failed(generation)
			raise
		attempt_root: CDPClient | None = None
		attempt_manager: Any = None
		try:
			# Create and store the CDP client for direct CDP communication
			self._cdp_health.mark_connecting(generation)
			attempt_root = CDPClient(self.cdp_url)
			self._cdp_client_root = attempt_root
			assert self._cdp_client_root is not None
			await self._cdp_client_root.start()

			# Initialize event-driven session manager FIRST (before enabling autoAttach)
			from openbrowser.browser.session_manager import SessionManager

			attempt_manager = SessionManager(self)
			self._session_manager = attempt_manager
			await self._session_manager.start_monitoring()
			self._navigation_monitor.bind(self._cdp_client_root, observer=self.observe_document_event)
			self.logger.debug('Event-driven session manager started')

			# Enable auto-attach so Chrome automatically notifies us when NEW targets attach/detach
			# This is the foundation of event-driven session management
			await self._cdp_client_root.send.Target.setAutoAttach(
				params={'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True}
			)
			self.logger.debug('CDP client connected with auto-attach enabled')

			# Get browser targets to find available contexts/pages
			targets = await self._cdp_client_root.send.Target.getTargets()

			# Manually attach to ALL EXISTING targets (autoAttach only fires for new ones)
			# We attach to everything (pages, iframes, workers) for complete coverage
			for target in targets['targetInfos']:
				target_id = target['targetId']
				target_type = target.get('type', 'unknown')

				try:
					# Attach to target - this triggers attachedToTarget event
					result = await self._cdp_client_root.send.Target.attachToTarget(
						params={'targetId': target_id, 'flatten': True}
					)
					session_id = result['sessionId']

					# Enable auto-attach for this target's children
					await self._cdp_client_root.send.Target.setAutoAttach(
						params={'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True},
						session_id=session_id,
					)

					self.logger.debug(
						f'Attached to existing target: {target_id[:8]}... (type={target_type}, session={session_id[:8]}...)'
					)
				except Exception as e:
					self.logger.debug(
						f'Failed to attach to existing target {target_id[:8]}... (type={target_type}): {e}'
					)

			# Find main browser pages (avoiding iframes, workers, extensions, etc.)
			page_targets: list[TargetInfo] = [
				t
				for t in targets['targetInfos']
				if self._is_valid_target(
					t,
					include_http=True,
					include_about=True,
					include_pages=True,
					include_iframes=False,
					include_workers=False,
				)
			]

			# Check for chrome://newtab pages and redirect them to about:blank
			from openbrowser.utils import is_new_tab_page

			for target in page_targets:
				target_url = target.get('url', '')
				if is_new_tab_page(target_url) and target_url != 'about:blank':
					target_id = target['targetId']
					self.logger.debug(f'🔄 Redirecting {target_url} to about:blank for target {target_id}')
					try:
						# Sessions now exist from manual attachment above
						session = await self._session_manager.get_session_for_target(target_id)
						if session:
							await session.cdp_client.send.Page.navigate(
								params={'url': 'about:blank'}, session_id=session.session_id
							)
							target['url'] = 'about:blank'
							await asyncio.sleep(0.05)  # Let navigation start
					except Exception as e:
						self.logger.warning(f'Failed to redirect {target_url}: {e}')

			# Ensure we have at least one page
			if not page_targets:
				new_target = await self._cdp_client_root.send.Target.createTarget(params={'url': 'about:blank'})
				target_id = new_target['targetId']
				self.logger.debug(f'📄 Created new blank page: {target_id}')
			else:
				target_id = [page for page in page_targets if page.get('type') == 'page'][0]['targetId']
				self.logger.debug(f'📄 Using existing page: {target_id}')

			budget = FocusTransitionBudget.start()
			result = await self._focus_transition_to_target(
				target_id,
				reason=FocusChangeReason.INITIAL,
				budget=budget,
				activate=False,
				probe=True,
			)
			self.logger.debug(f'📄 Agent focus set to {result.session.target_id[:8]}...')

			# Learn the initial document identity so the first DOM build is stamped.
			initial_identity = await self._read_document_identity(result.session.target_id)
			if initial_identity is not None:
				self._dom_cache.note_identity(result.session.target_id, initial_identity)

			# Enable proxy authentication handling if configured
			await self._setup_proxy_auth()

			# Verify the session is working
			if result.session.title == 'Unknown title':
				self.logger.warning('Session created but title is unknown (may be normal for about:blank)')

			# Dispatch TabCreatedEvent for all initial tabs (so watchdogs can initialize)
			initial_pages = page_targets or [
				{
					'targetId': result.session.target_id,
					'url': getattr(result.session, 'url', 'about:blank') or 'about:blank',
				}
			]
			if _start_owned:
				if not self._start_publication_is_current(start_generation, start_owner):
					raise RuntimeError('Browser start no longer owns connection publication')
			else:
				self._raise_if_connect_admission_closed(start_owned=False)
			for idx, target in enumerate(initial_pages):
				target_url = target.get('url', '')
				self.logger.debug(f'Dispatching TabCreatedEvent for initial tab {idx}: {target_url}')
				self.event_bus.dispatch(TabCreatedEvent(url=target_url, target_id=target['targetId']))

			try:
				await self._notify_focus_changed(
					result.session.target_id,
					getattr(result.session, 'url', 'about:blank') or 'about:blank',
					FocusChangeReason.INITIAL,
					budget,
					result.snapshot,
				)
			except FocusTargetError as error:
				if error.code not in (FocusTargetCode.OBSERVATION_FAILED, FocusTargetCode.TIMEOUT):
					raise
				self.logger.warning(f'Initial focus observation failed after commit: {error}')
			self.logger.debug(f'Initial agent focus set to tab 0: {getattr(result.session, "url", "about:blank")}')
			self._raise_if_connect_admission_closed(start_owned=_start_owned)

		except asyncio.CancelledError as error:
			if self._caller_cancellation_requested():
				await self._await_connect_failure(
					generation, error, attempt_root, attempt_manager, cancelled=True
				)
				raise
			worker_error = RuntimeError('CDP connect worker was cancelled')
			await self._await_connect_failure(
				generation,
				error,
				attempt_root,
				attempt_manager,
				cancelled=False,
				worker_cancelled=True,
			)
			raise worker_error from error
		except Exception as error:
			# Fatal error - browser is not usable without CDP connection
			self.logger.error(f'❌ FATAL: Failed to setup CDP connection: {error}')
			self.logger.error('❌ Browser cannot continue without CDP connection')
			await self._await_connect_failure(
				generation, error, attempt_root, attempt_manager, cancelled=False
			)
			# Re-raise as a fatal error
			safe_error = sanitize_error_detail(error) or type(error).__name__
			raise RuntimeError(f'Failed to establish CDP connection to browser: {safe_error}') from error

		self._cdp_health.mark_connected(generation)
		self._establish_owned_root_binding(generation)
		self._health.record_connected(generation)
		return self

	async def _handle_connect_failure(
		self,
		generation: int,
		error: BaseException,
		root: CDPClient | None,
		manager: Any,
		*,
		cancelled: bool,
		worker_cancelled: bool = False,
	) -> None:
		"""Record a failed connect and retire only resources owned by that attempt."""
		self._cdp_health.mark_connect_failed(
			generation,
			error,
			client_present=self._cdp_client_root is not None,
			cancelled=cancelled,
			worker_cancelled=worker_cancelled,
		)
		self._health.record_connect_failed(generation)
		self._focus_epoch += 1
		self._cancel_focus_observers()
		error_detail = sanitize_error_detail(error) or type(error).__name__
		self._focus_lost_locked(
			FocusTargetError(
				f'CDP connection failed: {type(error).__name__}: {error_detail}',
				code=FocusTargetCode.RECOVERY_FAILED,
				phase=self._focus_state.phase,
				target_id=self._focus_state.target_id or self._focus_state.previous_target_id,
			),
			FocusChangeReason.INITIAL,
			previous_target_id=self._focus_state.target_id or self._focus_state.previous_target_id,
		)
		self.agent_focus = None
		cleanup_manager = manager if manager is not None else self._session_manager
		cleanup_error = await self._cleanup_failed_cdp_connect(root, cleanup_manager)
		if cleanup_error is not None:
			self._cdp_health.mark_cleanup_failure(cleanup_error)
		elif self._cdp_health.cleanup_pending:
			self._cdp_health.mark_cleanup_success()

	async def _await_connect_failure(
		self,
		generation: int,
		error: BaseException,
		root: CDPClient | None,
		manager: Any,
		*,
		cancelled: bool,
		worker_cancelled: bool = False,
	) -> None:
		"""Finish failed-connect cleanup even when the caller is cancelled again."""
		cleanup_task = asyncio.create_task(
			self._handle_connect_failure(
				generation,
				error,
				root,
				manager,
				cancelled=cancelled,
				worker_cancelled=worker_cancelled,
			)
		)
		while not cleanup_task.done():
			try:
				await asyncio.shield(cleanup_task)
			except asyncio.CancelledError:
				continue
		cleanup_task.result()

	async def _cleanup_failed_cdp_connect(self, root: CDPClient | None, manager: Any) -> BaseException | None:
		"""Retire a failed attempt's resources within one shutdown-style budget."""
		if any(not task.done() for task in self._detached_cleanup_tasks):
			return TimeoutError('Prior CDP cleanup is still running')
		loop = asyncio.get_running_loop()
		deadline = loop.time() + _shutdown_timeout()
		cleanup_error: BaseException | None = None
		manager_cleared = manager is None
		root_stopped = root is None

		if manager is not None:
			remaining = deadline - loop.time()
			if remaining <= 0:
				cleanup_error = TimeoutError('CDP session manager cleanup deadline expired')
			else:
				try:
					clear = getattr(manager, 'clear', None)
					if callable(clear):
						result = clear()
						if inspect.isawaitable(result):
							await self._await_bounded_operation(
								result,
								remaining,
								'CDP session manager',
								retained_tasks=self._detached_cleanup_tasks,
							)
				except BaseException as error:
					cleanup_error = error
				else:
					manager_cleared = True

		self._abort_navigation()
		self._navigation_monitor.invalidate()
		self._cdp_session_pool.clear()
		self._tab_registry.clear()

		if root is not None:
			remaining = deadline - loop.time()
			if remaining <= 0:
				cleanup_error = cleanup_error or TimeoutError('CDP root cleanup deadline expired')
			else:
				try:
					stop = getattr(root, 'stop', None)
					if callable(stop):
						result = stop()
						if inspect.isawaitable(result):
							await self._await_bounded_operation(
								result,
								remaining,
								'CDP root',
								retained_tasks=self._detached_cleanup_tasks,
							)
				except BaseException as error:
					cleanup_error = cleanup_error or error
				else:
					root_stopped = True

		if manager_cleared and root_stopped:
			if self._session_manager is manager:
				self._session_manager = None
			if self._cdp_client_root is root:
				self._cdp_client_root = None

		return cleanup_error

	@staticmethod
	def _consume_detached_operation_task(task: asyncio.Future) -> None:
		"""Consume a late operation exception after its deadline was reported."""
		if not task.cancelled():
			task.exception()

	def _retain_detached_task(
		self, task: asyncio.Task, retained_tasks: set[asyncio.Task] | None
	) -> None:
		"""Keep a canceled child session-owned until its late result is consumed."""
		task.add_done_callback(self._consume_detached_operation_task)
		if retained_tasks is not None:
			retained_tasks.add(task)
			task.add_done_callback(retained_tasks.discard)

	async def _await_bounded_operation(
		self,
		awaitable: Awaitable[Any],
		remaining_seconds: float,
		label: str,
		*,
		retained_tasks: set[asyncio.Task] | None = None,
	) -> None:
		"""Apply a hard deadline even when the operation suppresses cancellation."""
		task = asyncio.ensure_future(awaitable)
		try:
			done, _pending = await asyncio.wait({task}, timeout=remaining_seconds)
		except asyncio.CancelledError:
			if not task.done():
				task.cancel()
			self._retain_detached_task(task, retained_tasks)
			raise
		if task not in done:
			task.cancel()
			self._retain_detached_task(task, retained_tasks)
			raise TimeoutError(f'{label} deadline expired')
		task.result()

	async def _retry_pending_cdp_cleanup(self) -> BaseException | None:
		"""Retry retained failed-connect resources and clear pending only on success."""
		if not self._cdp_health.cleanup_pending:
			return None
		return await self._cleanup_current_cdp_ownership()

	async def _cleanup_current_cdp_ownership(self) -> BaseException | None:
		"""Retire current CDP resources and retain ownership when cleanup fails."""
		if (
			self._cdp_client_root is None
			and self._session_manager is None
			and not self._cdp_health.cleanup_pending
		):
			return None
		cleanup_error = await self._cleanup_failed_cdp_connect(self._cdp_client_root, self._session_manager)
		if cleanup_error is not None:
			self._cdp_health.mark_cleanup_failure(cleanup_error)
		elif self._cdp_health.cleanup_pending:
			self._cdp_health.mark_cleanup_success()
		return cleanup_error

	async def _setup_proxy_auth(self) -> None:
		"""Install the CDP Fetch interceptor for render policy and proxy auth.

		The render resource policy blocks optional asset types before they leave
		the browser. Proxy authentication reuses the same Fetch domain so exactly
		one requestPaused handler decides each paused request.
		"""
		await self._setup_fetch_interception()

	async def _setup_fetch_interception(self) -> None:
		"""Install a single Fetch.requestPaused handler for render and proxy policy.

		`lightweight` blocks Image, Font, Media, and TextTrack resources.
		`no-render` also blocks Stylesheet resources. Blocked requests fail with
		BlockedByClient before reaching the network. When proxy credentials are
		configured every request is paused so auth challenges can be answered,
		and the same handler continues the non-blocked requests.
		"""
		assert self._cdp_client_root

		try:
			proxy_cfg = self.browser_profile.proxy
			username = (proxy_cfg.username if proxy_cfg else None) or ''
			password = (proxy_cfg.password if proxy_cfg else None) or ''
			handle_auth = bool(username and password)
			blocked_types = blocked_resource_types_for_render_mode(self.browser_profile.effective_render_mode)
			blocked_set = frozenset(blocked_types)
			if not handle_auth and not blocked_set:
				self.logger.debug('Fetch interception skipped: full render mode and no proxy credentials')
				return

			root = self._cdp_client_root
			effective_types = list(blocked_types)

			def _fetch_patterns() -> list[dict[str, str]]:
				if handle_auth:
					# Pause every request so proxy challenges on documents and
					# subresources are answered through Fetch.authRequired.
					return [{'urlPattern': '*', 'requestStage': 'Request'}]
				return [
					{'urlPattern': '*', 'resourceType': resource_type, 'requestStage': 'Request'}
					for resource_type in effective_types
				]

			handled = self._fetch_handled_requests
			handled.clear()

			def _spawn(coro: Coroutine[Any, Any, Any]) -> None:
				task = asyncio.create_task(coro)
				self._fetch_interceptor_tasks.add(task)
				task.add_done_callback(self._fetch_interceptor_tasks.discard)

			def _claim_request(request_id: str, event_session_id: SessionID | None) -> bool:
				"""Return True once per delivered requestId, even across duplicate events."""
				key = (event_session_id, request_id)
				if key in handled:
					return False
				handled[key] = None
				while len(handled) > MAX_RETAINED_FETCH_DECISIONS:
					handled.pop(next(iter(handled)))
				return True

			def _on_auth_required(event: AuthRequiredEvent, session_id: SessionID | None = None):
				# event keys may be snake_case or camelCase depending on generator; handle both
				request_id = event.get('requestId') or event.get('request_id')
				if not request_id:
					return

				challenge = event.get('authChallenge') or event.get('auth_challenge') or {}
				source = (challenge.get('source') or '').lower()
				# Only respond to proxy challenges
				if source == 'proxy' and request_id:

					async def _respond():
						try:
							await root.send.Fetch.continueWithAuth(
								params={
									'requestId': request_id,
									'authChallengeResponse': {
										'response': 'ProvideCredentials',
										'username': username,
										'password': password,
									},
								},
								session_id=session_id,
							)
						except Exception as e:
							self.logger.debug(f'Proxy auth respond failed: {type(e).__name__}: {e}')

					# schedule
					_spawn(_respond())
				else:
					# Default behaviour for non-proxy challenges: let browser handle
					async def _default():
						try:
							await root.send.Fetch.continueWithAuth(
								params={'requestId': request_id, 'authChallengeResponse': {'response': 'Default'}},
								session_id=session_id,
							)
						except Exception as e:
							self.logger.debug(f'Default auth respond failed: {type(e).__name__}: {e}')

					if request_id:
						_spawn(_default())

			def _on_request_paused(event: RequestPausedEvent, session_id: SessionID | None = None):
				# Single decision point for every paused request. Blocked resource
				# types fail before network; everything else continues untouched.
				request_id = event.get('requestId') or event.get('request_id')
				if not request_id:
					return
				if not _claim_request(request_id, session_id):
					return

				resource_type = str(event.get('resourceType') or event.get('resource_type') or '')
				if resource_type in blocked_set:

					async def _block():
						try:
							await root.send.Fetch.failRequest(
								params={'requestId': request_id, 'errorReason': 'BlockedByClient'},
								session_id=session_id,
							)
							self.logger.debug(f'Blocked {resource_type} request {request_id} ({sorted(blocked_set)})')
						except Exception as e:
							self.logger.debug(f'Fetch.failRequest failed for {request_id}: {type(e).__name__}: {e}')
							try:
								await root.send.Fetch.continueRequest(
									params={'requestId': request_id},
									session_id=session_id,
								)
							except Exception:
								pass

					_spawn(_block())
					return

				async def _continue():
					try:
						await root.send.Fetch.continueRequest(
							params={'requestId': request_id},
							session_id=session_id,
						)
					except Exception:
						pass

				_spawn(_continue())

			try:
				root.register.Fetch.requestPaused(_on_request_paused)
				focused_client = None
				if self.agent_focus is not None:
					focused_client = self.agent_focus.cdp_client
					if focused_client is not root:
						focused_client.register.Fetch.requestPaused(_on_request_paused)
				if handle_auth:
					root.register.Fetch.authRequired(_on_auth_required)
					if focused_client is not None and focused_client is not root:
						focused_client.register.Fetch.authRequired(_on_auth_required)
				self.logger.debug(
					f'Registered Fetch interception handlers (blocked={sorted(blocked_set)}, auth={handle_auth})'
				)
			except Exception as e:
				self.logger.debug(f'Failed to register Fetch handlers: {type(e).__name__}: {e}')

			async def _enable(target_session_id: SessionID | None) -> None:
				# Older Chromium builds reject unknown Fetch resource filters as a whole.
				# Retry without the unsupported type instead of losing the policy.
				while True:
					params: Any = {'handleAuthRequests': handle_auth, 'patterns': _fetch_patterns()}
					try:
						await root.send.Fetch.enable(params=params, session_id=target_session_id)
						self.logger.debug(
							f'Fetch.enable applied to session {target_session_id} (blocked={sorted(effective_types)})'
						)
						return
					except Exception as e:
						match = re.search(r"Unknown resource type in fetch filter: '([^']+)'", str(e), re.IGNORECASE)
						unsupported = match.group(1) if match else None
						if unsupported and unsupported in effective_types:
							effective_types.remove(unsupported)
							self.logger.warning(
								f'Chromium does not support the Fetch {unsupported!r} resource filter; '
								'that type is not blocked by this policy'
							)
							if not effective_types and not handle_auth:
								self.logger.warning('No supported Fetch resource filters remain; interception disabled')
								return
							continue
						self.logger.debug(
							f'Fetch.enable failed for session {target_session_id}: {type(e).__name__}: {e}'
						)
						return

			# Enable Fetch on the root client and the focused page before any navigation.
			await _enable(None)
			if self.agent_focus is not None:
				await _enable(self.agent_focus.session_id)

			# Reapply the same policy to every newly attached target.
			def _on_attached(event: AttachedToTargetEvent, session_id: SessionID | None = None):
				attached_session_id = event.get('sessionId') or event.get('session_id') or session_id
				if not attached_session_id:
					return
				_spawn(_enable(attached_session_id))

			try:
				root.register.Target.attachedToTarget(_on_attached)
				self.logger.debug('Registered Target.attachedToTarget handler for Fetch.enable')
			except Exception as e:
				self.logger.debug(f'Failed to register attachedToTarget handler: {type(e).__name__}: {e}')
		except Exception as e:
			self.logger.debug(f'Skipping Fetch interception setup: {type(e).__name__}: {e}')

	async def get_tabs(self) -> list[TabInfo]:
		"""Get information about all open tabs using CDP Target.getTargetInfo for speed."""
		tabs = []

		# Safety check - return empty list if browser not connected yet
		if not self._cdp_client_root:
			return tabs

		# Reconcile against every page target, then use the filtered page query
		# for display. A chrome-error:// or chrome:// tab is still live even
		# when it is not shown as an actionable tab.
		registry_revisions = self._tab_registry.state_revisions()
		authoritative_pages = await self._cdp_get_all_pages(
			include_chrome=True,
			include_chrome_extensions=True,
			include_chrome_error=True,
		)
		self._tab_registry.sync_from_pages(
			authoritative_pages,
			allow_reopen=True,
			state_revisions=registry_revisions,
		)
		pages = await self._cdp_get_all_pages()

		for i, page_target in enumerate(pages):
			target_id = page_target['targetId']
			url = page_target['url']

			# Try to get the title directly from Target.getTargetInfo - much faster!
			# The initial getTargets() doesn't include title, but getTargetInfo does
			try:
				target_info = await self.cdp_client.send.Target.getTargetInfo(params={'targetId': target_id})
				# The title is directly available in targetInfo
				title = target_info.get('targetInfo', {}).get('title', '')

				# Skip JS execution for chrome:// pages and new tab pages
				if is_new_tab_page(url) or url.startswith('chrome://'):
					# Use URL as title for chrome pages, or mark new tabs as unusable
					if is_new_tab_page(url):
						title = ''
					elif not title:
						# For chrome:// pages without a title, use the URL itself
						title = url

				# Special handling for PDF pages without titles
				if (not title or title == '') and (url.endswith('.pdf') or 'pdf' in url):
					# PDF pages might not have a title, use URL filename
					try:
						from urllib.parse import urlparse

						filename = urlparse(url).path.split('/')[-1]
						if filename:
							title = filename
					except Exception:
						pass

			except Exception as e:
				# Fallback to basic title handling
				self.logger.debug(
					f'⚠️ Failed to get target info for tab #{i}: {_log_pretty_url(url)} - {type(e).__name__}'
				)

				if is_new_tab_page(url):
					title = ''
				elif url.startswith('chrome://'):
					title = url
				else:
					title = ''

			tab_info = TabInfo(
				target_id=target_id,
				url=url,
				title=title,
				parent_target_id=None,
			)
			self._tab_registry.update(target_id, url=url, title=title, state=TabState.OPEN)
			tabs.append(tab_info)

		return tabs

	# endregion - ========== Helper Methods ==========

	# region - ========== ID Lookup Methods ==========
	async def get_current_target_info(self) -> TargetInfo | None:
		"""Get info about the current active target using CDP."""
		if not self.agent_focus or not self.agent_focus.target_id:
			return None

		targets = await self.cdp_client.send.Target.getTargets()
		for target in targets.get('targetInfos', []):
			if target.get('targetId') == self.agent_focus.target_id:
				# Still return even if it's not a "valid" target since we're looking for a specific ID
				return target
		return None

	async def get_current_page_url(self) -> str:
		"""Get the URL of the current page using CDP."""
		target = await self.get_current_target_info()
		if target:
			return target.get('url', '')
		return 'about:blank'

	async def get_current_page_title(self) -> str:
		"""Get the title of the current page using CDP."""
		target_info = await self.get_current_target_info()
		if target_info:
			return target_info.get('title', 'Unknown page title')
		return 'Unknown page title'

	async def navigate_to(
		self, url: str, new_tab: bool = False, *, wait_until: WaitUntil = 'load', timeout_ms: int | None = None
	) -> None:
		"""Navigate to a URL using the standard event system.

		Args:
                url: URL to navigate to
                new_tab: Whether to open in a new tab
		"""
		event = NavigateToUrlEvent(url=url, new_tab=new_tab, wait_until=wait_until, timeout_ms=timeout_ms)
		operation = event.get_navigation_operation(self._lifecycle.generation)
		event_bus = self.event_bus
		deadline = self._navigation_deadline(event, operation)
		self._reserve_navigation(operation)
		try:
			async with asyncio.timeout_at(deadline):
				await self._await_navigation_event(event_bus, event)
		except (asyncio.CancelledError, TimeoutError) as error:
			expired = asyncio.get_running_loop().time() >= deadline
			phase = (
				NavigationPhase.TIMED_OUT
				if expired
				else NavigationPhase.CANCELLED
				if isinstance(error, asyncio.CancelledError)
				else NavigationPhase.FAILED
			)
			operation.finish(phase, 'Navigation deadline expired' if expired else error)
			tasks = self._cancel_navigation_tasks(operation)
			if tasks:
				await asyncio.wait(tasks, timeout=1.0)
			self._publish_navigation_finished(operation, event_bus)
			self._retire_navigation(operation)
			if expired and not isinstance(error, asyncio.CancelledError):
				raise NavigationTimeoutError('Navigation deadline expired') from error
			raise
		except BaseException as error:
			operation.finish(NavigationPhase.FAILED, error)
			self._publish_navigation_finished(operation, event_bus)
			self._retire_navigation(operation)
			raise

	# endregion - ========== ID Lookup Methods ==========

	# region - ========== DOM Helper Methods ==========

	async def get_dom_element_by_index(self, index: int) -> EnhancedDOMTreeNode | None:
		"""Get DOM element by index.

		Get element from cached selector map.

		Args:
                index: The element index from the serialized DOM

		Returns:
                EnhancedDOMTreeNode or None if index not found
		"""
		#  Check cached selector map
		if self._cached_selector_map and index in self._cached_selector_map:
			return self._cached_selector_map[index]

		return None

	def update_cached_selector_map(
		self,
		selector_map: dict[int, EnhancedDOMTreeNode],
		*,
		target_id: str | None = None,
		loader_id: str | None = None,
		epoch: tuple[int, int] | None = None,
		sequence: int | None = None,
		identity: DocumentIdentity | None = None,
	) -> bool:
		"""Publish one selector map for the current document view.

		Returns False and leaves the cache untouched when the document changed
		while the map was being built.
		"""
		target = target_id or self.current_target_id
		if target is None or target != self.current_target_id:
			return False
		captured = epoch if epoch is not None else self._dom_cache.epoch(target)
		if captured is None:
			return False
		resolved = identity or self._dom_cache.known_identity(target)
		if resolved is None:
			return False
		published = self._dom_cache.publish(
			target,
			epoch=captured,
			loader_id=loader_id or resolved.loader_id,
			url=resolved.url,
			selector_count=len(selector_map),
			sequence=sequence,
		)
		if not published:
			return False
		self._cached_selector_map = selector_map
		return True

	def dom_cache_epoch(self, target_id: str | None = None) -> tuple[int, int] | None:
		"""Return the generation a DOM build must publish against."""
		target = target_id or self.current_target_id
		if target is None:
			return None
		return self._dom_cache.epoch(target)

	def dom_cache_build_token(self) -> int:
		"""Return a monotonic token for one DOM build."""
		return self._dom_cache.begin_build()

	def publish_browser_state_summary(
		self,
		summary: Any,
		*,
		target_id: str | None = None,
		epoch: tuple[int, int] | None = None,
		sequence: int | None = None,
	) -> bool:
		"""Cache a browser state summary for the document it was built from."""
		target = target_id or self.current_target_id
		if target is None or target != self.current_target_id:
			return False
		captured = epoch if epoch is not None else self._dom_cache.epoch(target)
		if captured is None:
			return False
		if not self._dom_cache.mark_summary(target, epoch=captured, sequence=sequence):
			return False
		self._cached_browser_state_summary = summary
		return True

	def invalidate_dom_cache(self, target_id: str | None = None, *, reason: str) -> None:
		"""Invalidate the cache for one target, or for the focused target."""
		target = target_id or self.current_target_id
		if target is None:
			self.reset_dom_cache(reason=reason)
			return
		self._dom_cache.invalidate(target, reason)
		focus = self.agent_focus
		if focus is not None and focus.target_id == target:
			self._cached_selector_map.clear()
			self._cached_browser_state_summary = None
			self._clear_watchdog_dom_cache()

	def reset_dom_cache(self, *, reason: str, purge_metadata: bool = False) -> None:
		"""Clear all cached DOM state for focus change, reset, or shutdown."""
		self._dom_cache.clear(reason, purge_metadata=purge_metadata)
		self._cached_selector_map.clear()
		self._cached_browser_state_summary = None
		self._clear_watchdog_dom_cache()

	def forget_dom_cache(self, target_id: str) -> None:
		"""Drop all cache state for a target that is gone.

		The storage is cleared synchronously so a concurrent read cannot adopt
		the detached target's map before focus recovery runs.
		"""
		self._dom_cache.forget(target_id)
		focus = self.agent_focus
		if focus is not None and focus.target_id == target_id:
			self._cached_selector_map.clear()
			self._cached_browser_state_summary = None
			self._clear_watchdog_dom_cache()

	def mark_dom_cache_provisional(self, target_id: str | None = None, *, reason: str = 'action') -> None:
		"""Mark a published view as possibly stale after an action."""
		target = target_id or self.current_target_id
		if target is None:
			return
		self._dom_cache.mark_provisional(target, reason)

	def dom_cache_snapshot(self) -> dict[str, Any]:
		"""Return the DOM cache state machine without page content."""
		return self._dom_cache.snapshot(self.current_target_id)

	def observe_document_event(self, kind: str, event: dict[str, Any], session_id: str | None) -> None:
		"""Apply a CDP document event to the cache machine.

		This is a best-effort early signal. Read-time revalidation remains the
		correctness fence when an event is missing.
		"""
		if kind not in ('frame', 'within'):
			return
		target_id = self._target_id_for_session(session_id)
		if target_id is None:
			return
		if kind == 'frame':
			frame = event.get('frame')
			if not isinstance(frame, dict) or frame.get('parentId'):
				return
			self._record_committed_document(target_id, frame)
			return
		main_frame = self._dom_cache.main_frame(target_id)
		if main_frame is None or event.get('frameId') != main_frame:
			return
		known = self._dom_cache.known_identity(target_id)
		self.invalidate_dom_cache(target_id=target_id, reason='same_document_navigation')
		new_url = event.get('url')
		if known is not None and isinstance(new_url, str) and new_url:
			self._dom_cache.note_identity(target_id, DocumentIdentity(known.loader_id, new_url))

	def target_id_for_session(self, session_id: str | None) -> str | None:
		"""Resolve a CDP session id to its target without changing focus."""
		return self._target_id_for_session(session_id)

	def _target_id_for_session(self, session_id: str | None) -> str | None:
		if not session_id:
			return None
		focus = self.agent_focus
		if focus is not None and focus.session_id == session_id:
			return focus.target_id
		for target_id, session in self._cdp_session_pool.items():
			if getattr(session, 'session_id', None) == session_id:
				return target_id
		return None

	def _record_committed_document(self, target_id: str | None, frame: Mapping[str, Any]) -> None:
		if target_id is None:
			return
		frame_id = frame.get('id')
		if isinstance(frame_id, str) and frame_id:
			self._dom_cache.note_main_frame(target_id, frame_id)
		loader_id = frame.get('loaderId')
		url = str(frame.get('url') or '') + str(frame.get('urlFragment') or '')
		self.invalidate_dom_cache(target_id=target_id, reason='document_commit')
		if isinstance(loader_id, str) and loader_id:
			self._dom_cache.note_identity(target_id, DocumentIdentity(loader_id, url))

	def _clear_watchdog_dom_cache(self) -> None:
		try:
			watchdog = self._dom_watchdog
		except (AttributeError, KeyError, TypeError):
			return
		clear_cache = getattr(watchdog, 'clear_cache', None)
		if callable(clear_cache):
			clear_cache()

	async def _read_document_identity(self, target_id: str) -> DocumentIdentity | None:
		root = self._cdp_client_root
		focus = self.agent_focus
		if root is None or focus is None or focus.target_id != target_id:
			return None
		try:
			tree = await root.send.Page.getFrameTree(session_id=focus.session_id)
		except Exception as error:
			self.logger.debug('DOM cache revalidation could not read the document identity: %s', error)
			return None
		frame = ((tree or {}).get('frameTree') or {}).get('frame')
		if not isinstance(frame, dict):
			return None
		frame_id = frame.get('id')
		if isinstance(frame_id, str) and frame_id:
			self._dom_cache.note_main_frame(target_id, frame_id)
		loader_id = frame.get('loaderId')
		url = str(frame.get('url') or '') + str(frame.get('urlFragment') or '')
		return DocumentIdentity(loader_id if isinstance(loader_id, str) and loader_id else None, url)

	# Alias for backwards compatibility
	async def get_element_by_index(self, index: int) -> EnhancedDOMTreeNode | None:
		"""Alias for get_dom_element_by_index for backwards compatibility."""
		return await self.get_dom_element_by_index(index)

	async def get_target_id_from_tab_id(self, tab_id: str) -> TargetID:
		"""Get the full-length TargetID from the truncated 4-char tab_id."""
		try:
			return self._tab_registry.resolve(tab_id)
		except TabError as first_error:
			if first_error.candidates:
				raise

		cached_candidates = [
			full_target_id for full_target_id in self._cdp_session_pool if full_target_id.endswith(tab_id)
		]
		live_cached: list[str] = []
		for full_target_id in cached_candidates:
			try:
				if await self._is_target_valid(full_target_id):
					live_cached.append(full_target_id)
			except Exception:
				continue
		if len(live_cached) == 1:
			return live_cached[0]
		if len(live_cached) > 1:
			raise TabError(
				f'Tab ID {tab_id} is ambiguous; it matches {len(live_cached)} cached CDP sessions.',
				tab_id=tab_id,
				candidates=live_cached,
				hint='Pass the full CDP target_id, not the four-character suffix.',
			)

		if self._cdp_client_root is not None:
			registry_revisions = self._tab_registry.state_revisions()
			targets = await self.cdp_client.send.Target.getTargets()
			pages = [target for target in targets.get('targetInfos', []) if target.get('type') in ('page', 'tab')]
			self._tab_registry.sync_from_pages(
				pages,
				allow_reopen=True,
				state_revisions=registry_revisions,
			)

		return self._tab_registry.resolve(tab_id)

	async def _is_target_valid(self, target_id: TargetID) -> bool:
		"""Check if a target ID is still valid."""
		try:
			await self.cdp_client.send.Target.getTargetInfo(params={'targetId': target_id})
			return True
		except Exception:
			return False

	async def get_target_id_from_url(self, url: str) -> TargetID:
		"""Get the TargetID from a URL."""
		all_targets = await self.cdp_client.send.Target.getTargets()
		for target in all_targets.get('targetInfos', []):
			if target['url'] == url and target['type'] == 'page':
				return target['targetId']

		# still not found, try substring match as fallback
		for target in all_targets.get('targetInfos', []):
			if url in target['url'] and target['type'] == 'page':
				return target['targetId']

		raise ValueError(f'No TargetID found for url={url}')

	async def get_most_recently_opened_target_id(self) -> TargetID:
		"""Get the most recently opened target ID."""
		all_targets = await self.cdp_client.send.Target.getTargets()
		return (await self._cdp_get_all_pages())[-1]['targetId']

	def is_file_input(self, element: Any) -> bool:
		"""Check if element is a file input.

		Args:
                element: The DOM element to check

		Returns:
                True if element is a file input, False otherwise
		"""
		if self._dom_watchdog:
			return self._dom_watchdog.is_file_input(element)
		# Fallback if watchdog not available
		return (
			hasattr(element, 'node_name')
			and element.node_name.upper() == 'INPUT'
			and hasattr(element, 'attributes')
			and element.attributes.get('type', '').lower() == 'file'
		)

	async def get_selector_map(self) -> dict[int, EnhancedDOMTreeNode]:
		"""Return the selector map only while it belongs to the live document.

		The watchdog is never a read source. It publishes through the session,
		so a cleared cache cannot be resurrected from its private copy.
		"""
		target_id = self.current_target_id
		if target_id is None or not self._cached_selector_map:
			return {}
		if not self._dom_cache.has_entry(target_id):
			# Adopt state written outside the session for backward compatibility.
			identity = await self._read_document_identity(target_id)
			if identity is None or not self._dom_cache.adopt(
				target_id,
				selector_count=len(self._cached_selector_map),
				has_summary=self._cached_browser_state_summary is not None,
				identity=identity,
			):
				self._cached_selector_map.clear()
				self._cached_browser_state_summary = None
				return {}
		if not self._dom_cache.is_current(target_id):
			self._cached_selector_map.clear()
			self._cached_browser_state_summary = None
			return {}
		if not await self._validate_dom_cache_identity(target_id):
			return {}
		return self._cached_selector_map

	async def _validate_dom_cache_identity(self, target_id: str) -> bool:
		"""Compare the published document identity with the live one.

		Every selector read goes through this check. A page-initiated or timer
		navigation that no observer saw is still refused here. The entry token
		prevents a slow read from invalidating a newer entry that replaced the
		one it checked.
		"""
		before = self._dom_cache.entry_token(target_id)
		identity = await self._read_document_identity(target_id)
		if self._dom_cache.entry_token(target_id) != before:
			return False
		if identity is None:
			self.invalidate_dom_cache(target_id=target_id, reason='document_identity_unavailable')
			return False
		if not self._dom_cache.confirm(target_id, identity):
			self.invalidate_dom_cache(target_id=target_id, reason='document_changed')
			return False
		return True

	async def dom_cache_capture_identity(
		self, target_id: str | None = None
	) -> DocumentIdentity | None:
		"""Read and record the live document identity for a build."""
		target = target_id or self.current_target_id
		if target is None:
			return None
		identity = await self._read_document_identity(target)
		if identity is not None:
			self._dom_cache.note_identity(target, identity)
		return identity

	async def get_index_by_id(self, element_id: str) -> int | None:
		"""Find element index by its id attribute.

		Args:
                element_id: The id attribute value to search for

		Returns:
                Index of the element, or None if not found
		"""
		selector_map = await self.get_selector_map()
		for idx, element in selector_map.items():
			if element.attributes and element.attributes.get('id') == element_id:
				return idx
		return None

	async def get_index_by_class(self, class_name: str) -> int | None:
		"""Find element index by its class attribute (matches if class contains the given name).

		Args:
                class_name: The class name to search for

		Returns:
                Index of the first matching element, or None if not found
		"""
		selector_map = await self.get_selector_map()
		for idx, element in selector_map.items():
			if element.attributes:
				element_class = element.attributes.get('class', '')
				if class_name in element_class.split():
					return idx
		return None

	async def remove_highlights(self) -> None:
		"""Remove highlights from the page using CDP."""
		if not self.browser_profile.highlight_elements:
			return

		try:
			# Get cached session
			cdp_session = await self.get_or_create_cdp_session()

			# Remove highlights via JavaScript - be thorough
			script = """
			(function() {
				// Remove all openbrowser highlight elements
				const highlights = document.querySelectorAll('[data-openbrowser-highlight]');
				console.log('Removing', highlights.length, 'openbrowser highlight elements');
				highlights.forEach(el => el.remove());

				// Also remove by ID in case selector missed anything
				const highlightContainer = document.getElementById('openbrowser-debug-highlights');
				if (highlightContainer) {
					console.log('Removing highlight container by ID');
					highlightContainer.remove();
				}

				// Final cleanup - remove any orphaned tooltips
				const orphanedTooltips = document.querySelectorAll('[data-openbrowser-highlight="tooltip"]');
				orphanedTooltips.forEach(el => el.remove());

				return { removed: highlights.length };
			})();
			"""
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': script, 'returnByValue': True}, session_id=cdp_session.session_id
			)

			# Log the result for debugging
			if result and 'result' in result and 'value' in result['result']:
				removed_count = result['result']['value'].get('removed', 0)
				self.logger.debug(f'Successfully removed {removed_count} highlight elements')
			else:
				self.logger.debug('Highlight removal completed')

		except Exception as e:
			self.logger.warning(f'Failed to remove highlights: {e}')

	@observe_debug(ignore_input=True, ignore_output=True, name='get_element_coordinates')
	async def get_element_coordinates(self, backend_node_id: int, cdp_session: CDPSession) -> DOMRect | None:
		"""Get element coordinates for a backend node ID using multiple methods.

		This method tries DOM.getContentQuads first, then falls back to DOM.getBoxModel,
		and finally uses JavaScript getBoundingClientRect as a last resort.

		Args:
                backend_node_id: The backend node ID to get coordinates for
                cdp_session: The CDP session to use

		Returns:
                DOMRect with coordinates or None if element not found/no bounds
		"""
		session_id = cdp_session.session_id
		quads = []

		# Method 1: Try DOM.getContentQuads first (best for inline elements and complex layouts)
		try:
			content_quads_result = await cdp_session.cdp_client.send.DOM.getContentQuads(
				params={'backendNodeId': backend_node_id}, session_id=session_id
			)
			if 'quads' in content_quads_result and content_quads_result['quads']:
				quads = content_quads_result['quads']
				self.logger.debug(f'Got {len(quads)} quads from DOM.getContentQuads')
			else:
				self.logger.debug(f'No quads found from DOM.getContentQuads {content_quads_result}')
		except Exception as e:
			self.logger.debug(f'DOM.getContentQuads failed: {e}')

		# Method 2: Fall back to DOM.getBoxModel
		if not quads:
			try:
				box_model = await cdp_session.cdp_client.send.DOM.getBoxModel(
					params={'backendNodeId': backend_node_id}, session_id=session_id
				)
				if 'model' in box_model and 'content' in box_model['model']:
					content_quad = box_model['model']['content']
					if len(content_quad) >= 8:
						# Convert box model format to quad format
						quads = [
							[
								content_quad[0],
								content_quad[1],  # x1, y1
								content_quad[2],
								content_quad[3],  # x2, y2
								content_quad[4],
								content_quad[5],  # x3, y3
								content_quad[6],
								content_quad[7],  # x4, y4
							]
						]
						self.logger.debug('Got quad from DOM.getBoxModel')
			except Exception as e:
				self.logger.debug(f'DOM.getBoxModel failed: {e}')

		# Method 3: Fall back to JavaScript getBoundingClientRect
		if not quads:
			try:
				result = await cdp_session.cdp_client.send.DOM.resolveNode(
					params={'backendNodeId': backend_node_id},
					session_id=session_id,
				)
				if 'object' in result and 'objectId' in result['object']:
					object_id = result['object']['objectId']
					js_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'objectId': object_id,
							'functionDeclaration': """
							function() {
								const rect = this.getBoundingClientRect();
								return {
									x: rect.x,
									y: rect.y,
									width: rect.width,
									height: rect.height
								};
							}
							""",
							'returnByValue': True,
						},
						session_id=session_id,
					)
					if 'result' in js_result and 'value' in js_result['result']:
						rect_data = js_result['result']['value']
						if rect_data['width'] > 0 and rect_data['height'] > 0:
							return DOMRect(
								x=rect_data['x'], y=rect_data['y'], width=rect_data['width'], height=rect_data['height']
							)
			except Exception as e:
				self.logger.debug(f'JavaScript getBoundingClientRect failed: {e}')

		# Convert quads to bounding rectangle if we have them
		if quads:
			# Use the first quad (most relevant for the element)
			quad = quads[0]
			if len(quad) >= 8:
				# Calculate bounding rect from quad points
				x_coords = [quad[i] for i in range(0, 8, 2)]
				y_coords = [quad[i] for i in range(1, 8, 2)]

				min_x = min(x_coords)
				min_y = min(y_coords)
				max_x = max(x_coords)
				max_y = max(y_coords)

				width = max_x - min_x
				height = max_y - min_y

				if width > 0 and height > 0:
					return DOMRect(x=min_x, y=min_y, width=width, height=height)

		return None

	async def highlight_interaction_element(self, node: 'EnhancedDOMTreeNode') -> None:
		"""Temporarily highlight an element during interaction for user visibility.

		This creates a visual highlight on the browser that shows the user which element
		is being interacted with. The highlight automatically fades after the configured duration.

		Args:
                node: The DOM node to highlight with backend_node_id for coordinate lookup
		"""
		if not self.browser_profile.highlight_elements:
			return

		try:
			import json

			cdp_session = await self.get_or_create_cdp_session()

			# Get current coordinates
			rect = await self.get_element_coordinates(node.backend_node_id, cdp_session)

			color = self.browser_profile.interaction_highlight_color
			duration_ms = int(self.browser_profile.interaction_highlight_duration * 1000)

			if not rect:
				self.logger.debug(f'No coordinates found for backend node {node.backend_node_id}')
				return

			# Create animated corner brackets that start offset and animate inward
			script = f"""
			(function() {{
				const rect = {json.dumps({'x': rect.x, 'y': rect.y, 'width': rect.width, 'height': rect.height})};
				const color = {json.dumps(color)};
				const duration = {duration_ms};

				// Scale corner size based on element dimensions to ensure gaps between corners
				const maxCornerSize = 20;
				const minCornerSize = 8;
				const cornerSize = Math.max(
					minCornerSize,
					Math.min(maxCornerSize, Math.min(rect.width, rect.height) * 0.35)
				);
				const borderWidth = 3;
				const startOffset = 10; // Starting offset in pixels
				const finalOffset = -3; // Final position slightly outside the element

				// Get current scroll position
				const scrollX = window.pageXOffset || document.documentElement.scrollLeft || 0;
				const scrollY = window.pageYOffset || document.documentElement.scrollTop || 0;

				// Create container for all corners
				const container = document.createElement('div');
				container.setAttribute('data-openbrowser-interaction-highlight', 'true');
				container.style.cssText = `
					position: absolute;
					left: ${{rect.x + scrollX}}px;
					top: ${{rect.y + scrollY}}px;
					width: ${{rect.width}}px;
					height: ${{rect.height}}px;
					pointer-events: none;
					z-index: 2147483647;
				`;

				// Create 4 corner brackets
				const corners = [
					{{ pos: 'top-left', startX: -startOffset, startY: -startOffset, finalX: finalOffset, finalY: finalOffset }},
					{{ pos: 'top-right', startX: startOffset, startY: -startOffset, finalX: -finalOffset, finalY: finalOffset }},
					{{ pos: 'bottom-left', startX: -startOffset, startY: startOffset, finalX: finalOffset, finalY: -finalOffset }},
					{{ pos: 'bottom-right', startX: startOffset, startY: startOffset, finalX: -finalOffset, finalY: -finalOffset }}
				];

				corners.forEach(corner => {{
					const bracket = document.createElement('div');
					bracket.style.cssText = `
						position: absolute;
						width: ${{cornerSize}}px;
						height: ${{cornerSize}}px;
						pointer-events: none;
						transition: all 0.15s ease-out;
					`;

					// Position corners with rounded outer edges
					if (corner.pos === 'top-left') {{
						bracket.style.top = '0';
						bracket.style.left = '0';
						bracket.style.borderTop = `${{borderWidth}}px solid ${{color}}`;
						bracket.style.borderLeft = `${{borderWidth}}px solid ${{color}}`;
						bracket.style.borderTopLeftRadius = '8px';
						bracket.style.transform = `translate(${{corner.startX}}px, ${{corner.startY}}px)`;
					}} else if (corner.pos === 'top-right') {{
						bracket.style.top = '0';
						bracket.style.right = '0';
						bracket.style.borderTop = `${{borderWidth}}px solid ${{color}}`;
						bracket.style.borderRight = `${{borderWidth}}px solid ${{color}}`;
						bracket.style.borderTopRightRadius = '8px';
						bracket.style.transform = `translate(${{corner.startX}}px, ${{corner.startY}}px)`;
					}} else if (corner.pos === 'bottom-left') {{
						bracket.style.bottom = '0';
						bracket.style.left = '0';
						bracket.style.borderBottom = `${{borderWidth}}px solid ${{color}}`;
						bracket.style.borderLeft = `${{borderWidth}}px solid ${{color}}`;
						bracket.style.borderBottomLeftRadius = '8px';
						bracket.style.transform = `translate(${{corner.startX}}px, ${{corner.startY}}px)`;
					}} else if (corner.pos === 'bottom-right') {{
						bracket.style.bottom = '0';
						bracket.style.right = '0';
						bracket.style.borderBottom = `${{borderWidth}}px solid ${{color}}`;
						bracket.style.borderRight = `${{borderWidth}}px solid ${{color}}`;
						bracket.style.borderBottomRightRadius = '8px';
						bracket.style.transform = `translate(${{corner.startX}}px, ${{corner.startY}}px)`;
					}}

					container.appendChild(bracket);

					// Animate to final position slightly outside the element
					setTimeout(() => {{
						bracket.style.transform = `translate(${{corner.finalX}}px, ${{corner.finalY}}px)`;
					}}, 10);
				}});

				document.body.appendChild(container);

				// Auto-remove after duration
				setTimeout(() => {{
					container.style.opacity = '0';
					container.style.transition = 'opacity 0.3s ease-out';
					setTimeout(() => container.remove(), 300);
				}}, duration);

				return {{ created: true }};
			}})();
			"""

			# Fire and forget - don't wait for completion

			await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': script, 'returnByValue': True}, session_id=cdp_session.session_id
			)

		except Exception as e:
			# Don't fail the action if highlighting fails
			self.logger.debug(f'Failed to highlight interaction element: {e}')

	async def add_highlights(self, selector_map: dict[int, 'EnhancedDOMTreeNode']) -> None:
		"""Add visual highlights to the browser DOM for user visibility."""
		if not self.browser_profile.dom_highlight_elements or not selector_map:
			return

		try:
			import json

			# Convert selector_map to the format expected by the highlighting script
			elements_data = []
			for _, node in selector_map.items():
				# Get bounding box using absolute position (includes iframe translations) if available
				if node.absolute_position:
					# Use absolute position which includes iframe coordinate translations
					rect = node.absolute_position
					bbox = {'x': rect.x, 'y': rect.y, 'width': rect.width, 'height': rect.height}

					# Only include elements with valid bounding boxes
					if bbox and bbox.get('width', 0) > 0 and bbox.get('height', 0) > 0:
						element = {
							'x': bbox['x'],
							'y': bbox['y'],
							'width': bbox['width'],
							'height': bbox['height'],
							'element_name': node.node_name,
							'is_clickable': node.snapshot_node.is_clickable if node.snapshot_node else True,
							'is_scrollable': getattr(node, 'is_scrollable', False),
							'attributes': node.attributes or {},
							'frame_id': getattr(node, 'frame_id', None),
							'node_id': node.node_id,
							'backend_node_id': node.backend_node_id,
							'xpath': node.xpath,
							'text_content': node.get_all_children_text()[:50]
							if hasattr(node, 'get_all_children_text')
							else node.node_value[:50],
						}
						elements_data.append(element)

			if not elements_data:
				self.logger.debug('⚠️ No valid elements to highlight')
				return

			self.logger.debug(f'📍 Creating highlights for {len(elements_data)} elements')

			# Always remove existing highlights first
			await self.remove_highlights()

			# Add a small delay to ensure removal completes
			import asyncio

			await asyncio.sleep(0.05)

			# Get CDP session
			cdp_session = await self.get_or_create_cdp_session()

			# Create the proven highlighting script from v0.6.0 with fixed positioning
			script = f"""
			(function() {{
				// Interactive elements data
				const interactiveElements = {json.dumps(elements_data)};

				console.log('=== OPENBROWSER HIGHLIGHTING ===');
				console.log('Highlighting', interactiveElements.length, 'interactive elements');

				// Double-check: Remove any existing highlight container first
				const existingContainer = document.getElementById('openbrowser-debug-highlights');
				if (existingContainer) {{
					console.log('⚠️ Found existing highlight container, removing it first');
					existingContainer.remove();
				}}

				// Also remove any stray highlight elements
				const strayHighlights = document.querySelectorAll('[data-openbrowser-highlight]');
				if (strayHighlights.length > 0) {{
					console.log('⚠️ Found', strayHighlights.length, 'stray highlight elements, removing them');
					strayHighlights.forEach(el => el.remove());
				}}

				// Use maximum z-index for visibility
				const HIGHLIGHT_Z_INDEX = 2147483647;

				// Create container for all highlights - use FIXED positioning (key insight from v0.6.0)
				const container = document.createElement('div');
				container.id = 'openbrowser-debug-highlights';
				container.setAttribute('data-openbrowser-highlight', 'container');

				container.style.cssText = `
					position: absolute;
					top: 0;
					left: 0;
					width: 100vw;
					height: 100vh;
					pointer-events: none;
					z-index: ${{HIGHLIGHT_Z_INDEX}};
					overflow: visible;
					margin: 0;
					padding: 0;
					border: none;
					outline: none;
					box-shadow: none;
					background: none;
					font-family: inherit;
				`;

				// Helper function to create text elements safely
				function createTextElement(tag, text, styles) {{
					const element = document.createElement(tag);
					element.textContent = text;
					if (styles) element.style.cssText = styles;
					return element;
				}}

				// Add highlights for each element
				interactiveElements.forEach((element, index) => {{
					const highlight = document.createElement('div');
					highlight.setAttribute('data-openbrowser-highlight', 'element');
					highlight.setAttribute('data-element-id', element.backend_node_id);
					highlight.style.cssText = `
						position: absolute;
						left: ${{element.x}}px;
						top: ${{element.y}}px;
						width: ${{element.width}}px;
						height: ${{element.height}}px;
						outline: 2px dashed #4a90e2;
						outline-offset: -2px;
						background: transparent;
						pointer-events: none;
						box-sizing: content-box;
						transition: outline 0.2s ease;
						margin: 0;
						padding: 0;
						border: none;
					`;

					// Enhanced label with backend node ID
					const label = createTextElement('div', element.backend_node_id, `
						position: absolute;
						top: -20px;
						left: 0;
						background-color: #4a90e2;
						color: white;
						padding: 2px 6px;
						font-size: 11px;
						font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
						font-weight: bold;
						border-radius: 3px;
						white-space: nowrap;
						z-index: ${{HIGHLIGHT_Z_INDEX + 1}};
						box-shadow: 0 2px 4px rgba(0,0,0,0.3);
						border: none;
						outline: none;
						margin: 0;
						line-height: 1.2;
					`);

					highlight.appendChild(label);
					container.appendChild(highlight);
				}});

				// Add container to document
				document.body.appendChild(container);

				console.log('Highlighting complete - added', interactiveElements.length, 'highlights');
				return {{ added: interactiveElements.length }};
			}})();
			"""

			# Execute the script
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': script, 'returnByValue': True}, session_id=cdp_session.session_id
			)

			# Log the result
			if result and 'result' in result and 'value' in result['result']:
				added_count = result['result']['value'].get('added', 0)
				self.logger.debug(f'Successfully added {added_count} highlight elements to browser DOM')
			else:
				self.logger.debug('Browser highlight injection completed')

		except Exception as e:
			self.logger.warning(f'Failed to add browser highlights: {e}')
			import traceback

			self.logger.debug(f'Browser highlight traceback: {traceback.format_exc()}')

	async def _close_extension_options_pages(self) -> None:
		"""Close any extension options/welcome pages that have opened."""
		try:
			# Get all open pages
			targets = await self._cdp_get_all_pages()

			for target in targets:
				target_url = target.get('url', '')
				target_id = target.get('targetId', '')

				# Check if this is an extension options/welcome page
				if 'chrome-extension://' in target_url and (
					'options.html' in target_url or 'welcome.html' in target_url or 'onboarding.html' in target_url
				):
					self.logger.info(f'[BrowserSession] 🚫 Closing extension options page: {target_url}')
					try:
						await self._cdp_close_page(target_id)
					except Exception as e:
						self.logger.debug(f'[BrowserSession] Could not close extension page {target_id}: {e}')

		except Exception as e:
			self.logger.debug(f'[BrowserSession] Error closing extension options pages: {e}')

	@property
	def downloaded_files(self) -> list[str]:
		"""Get list of files downloaded during this browser session.

		Returns:
                list[str]: List of absolute file paths to downloaded files in this session
		"""
		return self._downloaded_files.copy()

	# endregion - ========== Helper Methods ==========

	# region - ========== CDP-based replacements for browser_context operations ==========

	async def _cdp_get_all_pages(
		self,
		include_http: bool = True,
		include_about: bool = True,
		include_pages: bool = True,
		include_iframes: bool = False,
		include_workers: bool = False,
		include_chrome: bool = False,
		include_chrome_extensions: bool = False,
		include_chrome_error: bool = False,
	) -> list[TargetInfo]:
		"""Get all browser pages/tabs using CDP Target.getTargets."""
		# Safety check - return empty list if browser not connected yet
		if not self._cdp_client_root:
			return []
		targets = await self.cdp_client.send.Target.getTargets()
		# Filter for valid page/tab targets only
		return [
			t
			for t in targets.get('targetInfos', [])
			if self._is_valid_target(
				t,
				include_http=include_http,
				include_about=include_about,
				include_pages=include_pages,
				include_iframes=include_iframes,
				include_workers=include_workers,
				include_chrome=include_chrome,
				include_chrome_extensions=include_chrome_extensions,
				include_chrome_error=include_chrome_error,
			)
		]

	async def _cdp_create_new_page(
		self, url: str = 'about:blank', background: bool = False, new_window: bool = False
	) -> str:
		"""Create a new page/tab using CDP Target.createTarget. Returns target ID."""
		# Use the root CDP client to create tabs at the browser level
		if self._cdp_client_root:
			result = await self._cdp_client_root.send.Target.createTarget(
				params={'url': url, 'newWindow': new_window, 'background': background}
			)
		else:
			# Fallback to using cdp_client if root is not available
			result = await self.cdp_client.send.Target.createTarget(
				params={'url': url, 'newWindow': new_window, 'background': background}
			)
		return result['targetId']

	async def _cdp_close_page(self, target_id: TargetID) -> None:
		"""Close a page/tab using CDP Target.closeTarget."""
		await self.cdp_client.send.Target.closeTarget(params={'targetId': target_id})

	async def _cdp_get_cookies(self) -> list[Cookie]:
		"""Get cookies using CDP Network.getCookies."""
		cdp_session = await self.get_or_create_cdp_session(target_id=None)
		result = await asyncio.wait_for(
			cdp_session.cdp_client.send.Storage.getCookies(session_id=cdp_session.session_id), timeout=8.0
		)
		return result.get('cookies', [])

	async def _cdp_set_cookies(self, cookies: list[Cookie]) -> None:
		"""Set cookies using CDP Storage.setCookies."""
		if not self.agent_focus or not cookies:
			return

		cdp_session = await self.get_or_create_cdp_session(target_id=None)
		# Storage.setCookies expects params dict with 'cookies' key
		await cdp_session.cdp_client.send.Storage.setCookies(
			params={'cookies': cookies},  # type: ignore[arg-type]
			session_id=cdp_session.session_id,
		)

	async def _cdp_clear_cookies(self) -> None:
		"""Clear all cookies using CDP Network.clearBrowserCookies."""
		cdp_session = await self.get_or_create_cdp_session()
		await cdp_session.cdp_client.send.Storage.clearCookies(session_id=cdp_session.session_id)

	async def _cdp_set_extra_headers(self, headers: dict[str, str]) -> None:
		"""Set extra HTTP headers using CDP Network.setExtraHTTPHeaders."""
		if not self.agent_focus:
			return

		cdp_session = await self.get_or_create_cdp_session()
		# await cdp_session.cdp_client.send.Network.setExtraHTTPHeaders(params={'headers': headers}, session_id=cdp_session.session_id)
		raise NotImplementedError('Not implemented yet')

	async def _cdp_grant_permissions(self, permissions: list[str], origin: str | None = None) -> None:
		"""Grant permissions using CDP Browser.grantPermissions."""
		params = {'permissions': permissions}
		# if origin:
		# 	params['origin'] = origin
		cdp_session = await self.get_or_create_cdp_session()
		# await cdp_session.cdp_client.send.Browser.grantPermissions(params=params, session_id=cdp_session.session_id)
		raise NotImplementedError('Not implemented yet')

	async def _cdp_set_geolocation(self, latitude: float, longitude: float, accuracy: float = 100) -> None:
		"""Set geolocation using CDP Emulation.setGeolocationOverride."""
		await self.cdp_client.send.Emulation.setGeolocationOverride(
			params={'latitude': latitude, 'longitude': longitude, 'accuracy': accuracy}
		)

	async def _cdp_clear_geolocation(self) -> None:
		"""Clear geolocation override using CDP."""
		await self.cdp_client.send.Emulation.clearGeolocationOverride()

	async def _cdp_add_init_script(self, script: str) -> str:
		"""Add script to evaluate on new document using CDP Page.addScriptToEvaluateOnNewDocument."""
		assert self._cdp_client_root is not None
		cdp_session = await self.get_or_create_cdp_session()

		result = await cdp_session.cdp_client.send.Page.addScriptToEvaluateOnNewDocument(
			params={'source': script, 'runImmediately': True}, session_id=cdp_session.session_id
		)
		return result['identifier']

	async def _cdp_remove_init_script(self, identifier: str) -> None:
		"""Remove script added with addScriptToEvaluateOnNewDocument."""
		cdp_session = await self.get_or_create_cdp_session(target_id=None)
		await cdp_session.cdp_client.send.Page.removeScriptToEvaluateOnNewDocument(
			params={'identifier': identifier}, session_id=cdp_session.session_id
		)

	async def _cdp_set_viewport(
		self,
		width: int,
		height: int,
		device_scale_factor: float = 1.0,
		mobile: bool = False,
		target_id: str | None = None,
	) -> None:
		"""Set viewport using CDP Emulation.setDeviceMetricsOverride.

		Args:
                width: Viewport width
                height: Viewport height
                device_scale_factor: Device scale factor (default 1.0)
                mobile: Whether to emulate mobile device (default False)
                target_id: Optional target ID to set viewport for. If not provided, uses agent_focus.
		"""
		if target_id:
			# Set viewport for specific target
			cdp_session = await self.get_or_create_cdp_session(target_id, focus=False)
		elif self.agent_focus:
			# Use current focus
			cdp_session = self.agent_focus
		else:
			self.logger.warning('Cannot set viewport: no target_id provided and agent_focus not initialized')
			return

		await cdp_session.cdp_client.send.Emulation.setDeviceMetricsOverride(
			params={'width': width, 'height': height, 'deviceScaleFactor': device_scale_factor, 'mobile': mobile},
			session_id=cdp_session.session_id,
		)

	async def _cdp_get_origins(self) -> list[dict[str, Any]]:
		"""Get origins with localStorage and sessionStorage using CDP."""
		origins = []
		cdp_session = await self.get_or_create_cdp_session(target_id=None)

		try:
			# Enable DOMStorage domain to track storage
			await cdp_session.cdp_client.send.DOMStorage.enable(session_id=cdp_session.session_id)

			try:
				# Get all frames to find unique origins
				frames_result = await cdp_session.cdp_client.send.Page.getFrameTree(session_id=cdp_session.session_id)

				# Extract unique origins from frames
				unique_origins = set()

				def _extract_origins(frame_tree):
					"""Recursively extract origins from frame tree."""
					frame = frame_tree.get('frame', {})
					origin = frame.get('securityOrigin')
					if origin and origin != 'null':
						unique_origins.add(origin)

					# Process child frames
					for child in frame_tree.get('childFrames', []):
						_extract_origins(child)

				async def _get_storage_items(origin: str, is_local_storage: bool) -> list[dict[str, str]] | None:
					"""Helper to get storage items for an origin."""
					storage_type = 'localStorage' if is_local_storage else 'sessionStorage'
					try:
						result = await cdp_session.cdp_client.send.DOMStorage.getDOMStorageItems(
							params={'storageId': {'securityOrigin': origin, 'isLocalStorage': is_local_storage}},
							session_id=cdp_session.session_id,
						)

						items = []
						for item in result.get('entries', []):
							if len(item) == 2:  # Each item is [key, value]
								items.append({'name': item[0], 'value': item[1]})

						return items if items else None
					except Exception as e:
						self.logger.debug(f'Failed to get {storage_type} for {origin}: {e}')
						return None

				_extract_origins(frames_result.get('frameTree', {}))

				# For each unique origin, get localStorage and sessionStorage
				for origin in unique_origins:
					origin_data = {'origin': origin}

					# Get localStorage
					local_storage = await _get_storage_items(origin, is_local_storage=True)
					if local_storage:
						origin_data['localStorage'] = local_storage

					# Get sessionStorage
					session_storage = await _get_storage_items(origin, is_local_storage=False)
					if session_storage:
						origin_data['sessionStorage'] = session_storage

					# Only add origin if it has storage data
					if 'localStorage' in origin_data or 'sessionStorage' in origin_data:
						origins.append(origin_data)

			finally:
				# Always disable DOMStorage tracking when done
				await cdp_session.cdp_client.send.DOMStorage.disable(session_id=cdp_session.session_id)

		except Exception as e:
			self.logger.warning(f'Failed to get origins: {e}')

		return origins

	async def _cdp_get_storage_state(self) -> dict:
		"""Get storage state (cookies, localStorage, sessionStorage) using CDP."""
		# Use the _cdp_get_cookies helper which handles session attachment
		cookies = await self._cdp_get_cookies()

		# Get origins with localStorage/sessionStorage
		origins = await self._cdp_get_origins()

		return {
			'cookies': cookies,
			'origins': origins,
		}

	async def _cdp_navigate(
		self,
		url: str,
		target_id: TargetID | None = None,
		*,
		wait_until: WaitUntil = 'commit',
		timeout_seconds: float = 30.0,
	) -> None:
		"""Navigate to URL using CDP Page.navigate and wait for the target document.

		This is a target-scoped actor barrier, not the full navigation machine.
		It does not emit navigation events, own policy, or wait for network idle.
		"""
		# Use provided target_id or fall back to current_target_id
		validate_actor_wait_until(wait_until)
		loop = asyncio.get_running_loop()
		deadline = loop.time() + max(0.01, timeout_seconds)

		assert self._cdp_client_root is not None, 'CDP client not initialized - browser may not be connected yet'
		assert self.agent_focus is not None, 'CDP session not initialized - browser may not be connected yet'
		target_label = target_id or self.agent_focus.target_id

		try:
			async with asyncio.timeout(max(0.01, timeout_seconds)):
				session = await self.get_or_create_cdp_session(target_label, focus=True)

				baseline_remaining = deadline - loop.time()
				try:
					baseline = await asyncio.wait_for(
						read_document_snapshot(session.cdp_client, session.session_id),
						timeout=min(2.0, max(0.01, baseline_remaining)),
					)
				except Exception:
					baseline = None

				# A raw Page.navigate bypasses the navigation machine. Fence the
				# cache before the command so no pre-navigation index survives.
				self.invalidate_dom_cache(target_id=session.target_id, reason='cdp_navigate')

				async def run() -> None:
					response = await session.cdp_client.send.Page.navigate(
						params={'url': url}, session_id=session.session_id
					)
					if not isinstance(response, dict):
						raise NavigationError('Page.navigate returned a malformed response')
					if response.get('isDownload') is True:
						return
					error_text = response.get('errorText')
					if error_text:
						raise NavigationError(str(error_text))
					raw_loader = response.get('loaderId')
					await await_document_ready(
						session.cdp_client,
						session.session_id,
						target_id=session.target_id,
						baseline=baseline,
						requested_url=url,
						expected_loader=raw_loader
						if isinstance(raw_loader, str) and raw_loader
						else None,
						wait_until=wait_until,
						timeout_seconds=max(0.01, deadline - loop.time() - 0.01),
					)

				await run()
		except NavigationTimeoutError:
			raise
		except TimeoutError as error:
			raise NavigationTimeoutError(
				f'Actor navigation on target {target_label} to {url} did not complete '
				f'within {timeout_seconds:g}s (command or document did not answer).'
			) from error
	@staticmethod
	def _is_valid_target(
		target_info: TargetInfo,
		include_http: bool = True,
		include_chrome: bool = False,
		include_chrome_extensions: bool = False,
		include_chrome_error: bool = False,
		include_about: bool = True,
		include_iframes: bool = True,
		include_pages: bool = True,
		include_workers: bool = False,
	) -> bool:
		"""Check if a target should be processed.

		Args:
                target_info: Target info dict from CDP

		Returns:
                True if target should be processed, False if it should be skipped
		"""
		target_type = target_info.get('type', '')
		url = target_info.get('url', '')

		url_allowed, type_allowed = False, False

		# Always allow new tab pages (chrome://new-tab-page/, chrome://newtab/, about:blank)
		# so they can be redirected to about:blank in connect()
		from openbrowser.utils import is_new_tab_page

		if is_new_tab_page(url):
			url_allowed = True

		if url.startswith('chrome-error://') and include_chrome_error:
			url_allowed = True

		if url.startswith('chrome://') and include_chrome:
			url_allowed = True

		if url.startswith('chrome-extension://') and include_chrome_extensions:
			url_allowed = True

		# dont allow about:srcdoc! there are also other rare about: pages that we want to avoid
		if url == 'about:blank' and include_about:
			url_allowed = True

		if (url.startswith('http://') or url.startswith('https://')) and include_http:
			url_allowed = True

		if target_type in ('service_worker', 'shared_worker', 'worker') and include_workers:
			type_allowed = True

		if target_type in ('page', 'tab') and include_pages:
			type_allowed = True

		if target_type in ('iframe', 'webview') and include_iframes:
			type_allowed = True

		return url_allowed and type_allowed

	async def get_all_frames(
		self,
		*,
		include_background_targets: bool = False,
	) -> tuple[dict[str, dict], dict[str, str]]:
		"""Get a complete frame hierarchy from all browser targets.

		When ``cross_origin_iframes`` is disabled, only the focused target is
		inspected by default. Background tabs are included when
		``include_background_targets`` is true, which download completion needs
		because its frame can belong to a tab that is not focused.

		Returns:
                Tuple of (all_frames, target_sessions) where:
                - all_frames: dict mapping frame_id -> frame info dict with all metadata
                - target_sessions: dict mapping target_id -> session_id for active sessions
		"""
		all_frames = {}  # frame_id -> FrameInfo dict
		target_sessions = {}  # target_id -> session_id (keep sessions alive during collection)

		# Check if cross-origin iframe support is enabled
		include_cross_origin = self.browser_profile.cross_origin_iframes

		# Get all targets - only include iframes if cross-origin support is enabled
		targets = await self._cdp_get_all_pages(
			include_http=True,
			include_about=True,
			include_pages=True,
			include_iframes=include_cross_origin,  # Only include iframe targets if flag is set
			include_workers=False,
			include_chrome=False,
			include_chrome_extensions=False,
			include_chrome_error=include_cross_origin,  # Only include error pages if cross-origin is enabled
		)
		all_targets = targets

		# First pass: collect frame trees from ALL targets
		for target in all_targets:
			target_id = target['targetId']

			# Skip iframe targets if cross-origin support is disabled
			if not include_cross_origin and target.get('type') == 'iframe':
				continue

			# A target can detach between enumeration and session lookup. Keep
			# that failure scoped to this target so another tab can still
			# provide the frame that initiated a download.
			try:
				# Without cross-origin support, the default path is focus-only. A
				# download completion may still name a frame in a background tab.
				if not include_cross_origin and not include_background_targets:
					if self.agent_focus and target_id != self.agent_focus.target_id:
						continue
					cdp_session = self.agent_focus
				else:
					# Get cached session for this target without changing focus.
					cdp_session = await self.get_or_create_cdp_session(target_id, focus=False)
			except Exception as session_error:
				self.logger.debug(
					f'Skipping frame scan for detached target {target_id}: '
					f'{type(session_error).__name__}: {session_error}'
				)
				continue

			if cdp_session:
				target_sessions[target_id] = cdp_session.session_id

				try:
					# Try to get frame tree (not all target types support this)
					frame_tree_result = await cdp_session.cdp_client.send.Page.getFrameTree(
						session_id=cdp_session.session_id
					)

					# Process the frame tree recursively
					def process_frame_tree(node, parent_frame_id=None):
						"""Recursively process frame tree and add to all_frames."""
						frame = node.get('frame', {})
						current_frame_id = frame.get('id')

						if current_frame_id:
							# For iframe targets, check if the frame has a parentId field
							# This indicates it's an OOPIF with a parent in another target
							actual_parent_id = frame.get('parentId') or parent_frame_id

							# Create frame info with all CDP response data plus our additions
							frame_info = {
								**frame,  # Include all original frame data: id, url, parentId, etc.
								'frameTargetId': target_id,  # Target that can access this frame
								'parentFrameId': actual_parent_id,  # Use parentId from frame if available
								'childFrameIds': [],  # Will be populated below
								'isCrossOrigin': False,  # Will be determined based on context
								'isValidTarget': self._is_valid_target(
									target,
									include_http=True,
									include_about=True,
									include_pages=True,
									include_iframes=True,
									include_workers=False,
									include_chrome=False,  # chrome://newtab, chrome://settings, etc. are not valid frames we can control (for sanity reasons)
									include_chrome_extensions=False,  # chrome-extension://
									include_chrome_error=False,  # chrome-error://  (e.g. when iframes fail to load or are blocked by uBlock Origin)
								),
							}

							# Check if frame is cross-origin based on crossOriginIsolatedContextType
							cross_origin_type = frame.get('crossOriginIsolatedContextType')
							if cross_origin_type and cross_origin_type != 'NotIsolated':
								frame_info['isCrossOrigin'] = True

							# For iframe targets, the frame itself is likely cross-origin
							if target.get('type') == 'iframe':
								frame_info['isCrossOrigin'] = True

							# Skip cross-origin frames if support is disabled
							if not include_cross_origin and frame_info.get('isCrossOrigin'):
								return  # Skip this frame and its children

							# Add child frame IDs (note: OOPIFs won't appear here)
							child_frames = node.get('childFrames', [])
							for child in child_frames:
								child_frame = child.get('frame', {})
								child_frame_id = child_frame.get('id')
								if child_frame_id:
									frame_info['childFrameIds'].append(child_frame_id)

							# Store or merge frame info
							if current_frame_id in all_frames:
								# Frame already seen from another target, merge info
								existing = all_frames[current_frame_id]
								# If this is an iframe target, it has direct access to the frame
								if target.get('type') == 'iframe':
									existing['frameTargetId'] = target_id
									existing['isCrossOrigin'] = True
							else:
								all_frames[current_frame_id] = frame_info

							# Process child frames recursively (only if we're not skipping this frame)
							if include_cross_origin or not frame_info.get('isCrossOrigin'):
								for child in child_frames:
									process_frame_tree(child, current_frame_id)

					# Process the entire frame tree
					process_frame_tree(frame_tree_result.get('frameTree', {}))

				except Exception as e:
					# Target doesn't support Page domain or has no frames
					self.logger.debug(f'Failed to get frame tree for target {target_id}: {e}')

		# Second pass: populate backend node IDs and parent target IDs
		# Only do this if cross-origin support is enabled
		if include_cross_origin:
			await self._populate_frame_metadata(all_frames, target_sessions)

		return all_frames, target_sessions

	async def _populate_frame_metadata(self, all_frames: dict[str, dict], target_sessions: dict[str, str]) -> None:
		"""Populate additional frame metadata like backend node IDs and parent target IDs.

		Args:
                all_frames: Frame hierarchy dict to populate
                target_sessions: Active target sessions
		"""
		for frame_id_iter, frame_info in all_frames.items():
			parent_frame_id = frame_info.get('parentFrameId')

			if parent_frame_id and parent_frame_id in all_frames:
				parent_frame_info = all_frames[parent_frame_id]
				parent_target_id = parent_frame_info.get('frameTargetId')

				# Store parent target ID
				frame_info['parentTargetId'] = parent_target_id

				# Try to get backend node ID from parent context
				if parent_target_id in target_sessions:
					assert parent_target_id is not None
					parent_session_id = target_sessions[parent_target_id]
					try:
						# Enable DOM domain
						await self.cdp_client.send.DOM.enable(session_id=parent_session_id)

						# Get frame owner info to find backend node ID
						frame_owner = await self.cdp_client.send.DOM.getFrameOwner(
							params={'frameId': frame_id_iter}, session_id=parent_session_id
						)

						if frame_owner:
							frame_info['backendNodeId'] = frame_owner.get('backendNodeId')
							frame_info['nodeId'] = frame_owner.get('nodeId')

					except Exception:
						# Frame owner not available (likely cross-origin)
						pass

	async def find_frame_target(self, frame_id: str, all_frames: dict[str, dict] | None = None) -> dict | None:
		"""Find the frame info for a specific frame ID.

		Args:
                frame_id: The frame ID to search for
                all_frames: Optional pre-built frame hierarchy. If None, will call get_all_frames()

		Returns:
                Frame info dict if found, None otherwise
		"""
		if all_frames is None:
			all_frames, _ = await self.get_all_frames()

		return all_frames.get(frame_id)

	async def cdp_client_for_target(self, target_id: TargetID) -> CDPSession:
		return await self.get_or_create_cdp_session(target_id, focus=False)

	def get_target_id_from_session_id(self, session_id: SessionID | None) -> TargetID | None:
		"""Look up target_id from a CDP session_id.

		Args:
                session_id: The CDP session ID to look up

		Returns:
                The target_id for this session, or None if not found
		"""
		if not session_id:
			return None
		for cdp_session in self._cdp_session_pool.values():
			if cdp_session.session_id == session_id:
				return cdp_session.target_id
		return None

	async def cdp_client_for_frame(self, frame_id: str) -> CDPSession:
		"""Get a CDP client attached to the target containing the specified frame.

		Builds a unified frame hierarchy from all targets to find the correct target
		for any frame, including OOPIFs (Out-of-Process iframes).

		Args:
                frame_id: The frame ID to search for

		Returns:
                Tuple of (cdp_cdp_session, target_id) for the target containing the frame

		Raises:
                ValueError: If the frame is not found in any target
		"""
		# If cross-origin iframes are disabled, just use the main session
		if not self.browser_profile.cross_origin_iframes:
			return await self.get_or_create_cdp_session()

		# Get complete frame hierarchy
		all_frames, target_sessions = await self.get_all_frames()

		# Find the requested frame
		frame_info = await self.find_frame_target(frame_id, all_frames)

		if frame_info:
			target_id = frame_info.get('frameTargetId')

			if target_id in target_sessions:
				assert target_id is not None
				# Use existing session
				session_id = target_sessions[target_id]
				# Return the client with session attached (don't change focus)
				return await self.get_or_create_cdp_session(target_id, focus=False)

		# Frame not found
		raise ValueError(f"Frame with ID '{frame_id}' not found in any target")

	async def cdp_client_for_node(self, node: EnhancedDOMTreeNode) -> CDPSession:
		"""Get CDP client for a specific DOM node based on its frame.

		IMPORTANT: backend_node_id is only valid in the session where the DOM was captured.
		We trust the node's session_id/frame_id/target_id instead of searching all sessions.
		"""

		# Strategy 1: If node has session_id, try to use that exact session (most specific)
		if node.session_id:
			try:
				# Find the CDP session by session_id
				for cdp_session in self._cdp_session_pool.values():
					if cdp_session.session_id == node.session_id:
						self.logger.debug(
							f'✅ Using session from node.session_id for node {node.backend_node_id}: {cdp_session.url}'
						)
						return cdp_session
			except Exception as e:
				self.logger.debug(f'Failed to get session by session_id {node.session_id}: {e}')

		# Strategy 2: If node has frame_id, use that frame's session
		if node.frame_id:
			try:
				cdp_session = await self.cdp_client_for_frame(node.frame_id)
				self.logger.debug(
					f'✅ Using session from node.frame_id for node {node.backend_node_id}: {cdp_session.url}'
				)
				return cdp_session
			except Exception as e:
				self.logger.debug(f'Failed to get session for frame {node.frame_id}: {e}')

		# Strategy 3: If node has target_id, use that target's session
		if node.target_id:
			try:
				cdp_session = await self.get_or_create_cdp_session(target_id=node.target_id, focus=False)
				self.logger.debug(
					f'✅ Using session from node.target_id for node {node.backend_node_id}: {cdp_session.url}'
				)
				return cdp_session
			except Exception as e:
				self.logger.debug(f'Failed to get session for target {node.target_id}: {e}')

		# Strategy 4: Fallback to agent_focus (the page where agent is currently working)
		if self.agent_focus:
			self.logger.warning(
				f'⚠️ Node {node.backend_node_id} has no session/frame/target info. '
				f'Using agent_focus session: {self.agent_focus.url}'
			)
			return self.agent_focus

		# Last resort: use main session
		self.logger.error(
			f'❌ No session info for node {node.backend_node_id} and no agent_focus available. Using main session.'
		)
		return await self.get_or_create_cdp_session()

	@observe_debug(ignore_input=True, ignore_output=True, name='take_screenshot')
	async def take_screenshot(
		self,
		path: str | None = None,
		full_page: bool = False,
		format: str = 'png',
		quality: int | None = None,
		clip: dict | None = None,
	) -> bytes:
		"""Take a screenshot using CDP.

		Args:
                path: Optional file path to save screenshot
                full_page: Capture entire scrollable page beyond viewport
                format: Image format ('png', 'jpeg', 'webp')
                quality: Quality 0-100 for JPEG format
                clip: Region to capture {'x': int, 'y': int, 'width': int, 'height': int}

		Returns:
                Screenshot data as bytes
		"""
		if self.browser_profile.effective_render_mode is RenderMode.NO_RENDER:
			from openbrowser.browser.views import BrowserError

			raise BrowserError('Screenshots are disabled in no-render mode')
		import base64

		from cdp_use.cdp.page import CaptureScreenshotParameters

		cdp_session = await self.get_or_create_cdp_session()

		# Build parameters dict explicitly to satisfy TypedDict expectations
		params: CaptureScreenshotParameters = {
			'format': format,
			'captureBeyondViewport': full_page,
		}

		if quality is not None and format == 'jpeg':
			params['quality'] = quality

		if clip:
			params['clip'] = {
				'x': clip['x'],
				'y': clip['y'],
				'width': clip['width'],
				'height': clip['height'],
				'scale': 1,
			}

		params = CaptureScreenshotParameters(**params)

		result = await cdp_session.cdp_client.send.Page.captureScreenshot(
			params=params, session_id=cdp_session.session_id
		)

		if not result or 'data' not in result:
			raise Exception('Screenshot failed - no data returned')

		screenshot_data = base64.b64decode(result['data'])

		if path:
			Path(path).write_bytes(screenshot_data)

		return screenshot_data

	async def screenshot_element(
		self,
		selector: str,
		path: str | None = None,
		format: str = 'png',
		quality: int | None = None,
	) -> bytes:
		"""Take a screenshot of a specific element.

		Args:
                selector: CSS selector for the element
                path: Optional file path to save screenshot
                format: Image format ('png', 'jpeg', 'webp')
                quality: Quality 0-100 for JPEG format

		Returns:
                Screenshot data as bytes
		"""
		if self.browser_profile.effective_render_mode is RenderMode.NO_RENDER:
			from openbrowser.browser.views import BrowserError

			raise BrowserError('Screenshots are disabled in no-render mode')

		bounds = await self._get_element_bounds(selector)
		if not bounds:
			raise ValueError(f"Element '{selector}' not found or has no bounds")

		return await self.take_screenshot(
			path=path,
			format=format,
			quality=quality,
			clip=bounds,
		)

	async def _get_element_bounds(self, selector: str) -> dict | None:
		"""Get element bounding box using CDP."""

		cdp_session = await self.get_or_create_cdp_session()

		# Get document
		doc = await cdp_session.cdp_client.send.DOM.getDocument(params={'depth': 1}, session_id=cdp_session.session_id)

		# Query selector
		node_result = await cdp_session.cdp_client.send.DOM.querySelector(
			params={'nodeId': doc['root']['nodeId'], 'selector': selector}, session_id=cdp_session.session_id
		)

		node_id = node_result.get('nodeId')
		if not node_id:
			return None

		# Get bounding box
		box_result = await cdp_session.cdp_client.send.DOM.getBoxModel(
			params={'nodeId': node_id}, session_id=cdp_session.session_id
		)

		box_model = box_result.get('model')
		if not box_model:
			return None

		content = box_model['content']
		return {
			'x': min(content[0], content[2], content[4], content[6]),
			'y': min(content[1], content[3], content[5], content[7]),
			'width': max(content[0], content[2], content[4], content[6])
			- min(content[0], content[2], content[4], content[6]),
			'height': max(content[1], content[3], content[5], content[7])
			- min(content[1], content[3], content[5], content[7]),
		}
