"""Storage state watchdog for managing browser cookies and storage persistence."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, ClassVar

from bubus import BaseEvent
from cdp_use.cdp.network import Cookie
from pydantic import Field, PrivateAttr

from openbrowser.browser.events import (
	BrowserConnectedEvent,
	BrowserStopEvent,
	LoadStorageStateEvent,
	SaveStorageStateEvent,
	StorageStateLoadedEvent,
	StorageStateSavedEvent,
)
from openbrowser.browser.storage_state import StorageStatePhase, StorageStateSnapshot
from openbrowser.browser.watchdog_base import BaseWatchdog


class StorageStateWatchdog(BaseWatchdog):
	"""Monitors and persists browser storage state including cookies and localStorage."""

	# Event contracts
	LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [
		BrowserConnectedEvent,
		BrowserStopEvent,
		SaveStorageStateEvent,
		LoadStorageStateEvent,
	]
	EMITS: ClassVar[list[type[BaseEvent]]] = [
		StorageStateSavedEvent,
		StorageStateLoadedEvent,
	]

	# Configuration
	auto_save_interval: float = Field(default=30.0)  # Auto-save every 30 seconds
	save_on_change: bool = Field(default=True)  # Save immediately when cookies change
	max_script_bytes_per_origin: int = Field(default=256_000)  # Bound one injected storage script

	# Private state
	_monitoring_task: asyncio.Task | None = PrivateAttr(default=None)
	_last_cookie_state: list[dict] = PrivateAttr(default_factory=list)
	_save_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
	_load_phase: StorageStatePhase | None = PrivateAttr(default=None)
	_load_path: str | None = PrivateAttr(default=None)
	_load_bytes: int | None = PrivateAttr(default=None)
	_load_started: float | None = PrivateAttr(default=None)
	_load_ms: float | None = PrivateAttr(default=None)
	_load_elapsed: float | None = PrivateAttr(default=None)
	_load_error: str | None = PrivateAttr(default=None)
	_load_detail: str | None = PrivateAttr(default=None)
	_load_cookies: int = PrivateAttr(default=0)
	_load_origins: int = PrivateAttr(default=0)
	_load_scripts: int = PrivateAttr(default=0)
	_load_storage_script_attempts: int = PrivateAttr(default=0)

	async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
		"""Start monitoring when browser starts."""
		self.logger.debug('[StorageStateWatchdog] 🍪 Initializing auth/cookies sync <-> with storage_state.json file')

		# Start monitoring
		await self._start_monitoring()

		# Automatically load storage state after browser start
		await self.event_bus.dispatch(LoadStorageStateEvent())

	async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
		"""Stop monitoring when browser stops."""
		self.logger.debug('[StorageStateWatchdog] Stopping storage_state monitoring')
		await self._stop_monitoring()

	async def on_SaveStorageStateEvent(self, event: SaveStorageStateEvent) -> None:
		"""Handle storage state save request."""
		# Use provided path or fall back to profile default
		path = event.path
		if path is None:
			# Use profile default path if available
			if self.browser_session.browser_profile.storage_state:
				path = str(self.browser_session.browser_profile.storage_state)
			else:
				path = None  # Skip saving if no path available
		await self._save_storage_state(path)

	async def on_LoadStorageStateEvent(self, event: LoadStorageStateEvent) -> None:
		"""Handle storage state load request."""
		# Use provided path or fall back to profile default
		path = event.path
		if path is None:
			# Use profile default path if available
			configured = self.browser_session.browser_profile.storage_state
			if isinstance(configured, (str, Path)):
				path = str(configured)
			else:
				path = None  # Skip loading if no path available
		await self._load_storage_state(path)

	def storage_state_snapshot(self) -> StorageStateSnapshot:
		"""Return the explicit file-load state, including a pre-load view."""
		if self._load_phase is not None:
			elapsed_ms = self._load_elapsed
			if self._load_phase is StorageStatePhase.LOADING and self._load_started is not None:
				elapsed_ms = max(0.0, (time.monotonic() - self._load_started) * 1000.0)
			return StorageStateSnapshot(
				phase=self._load_phase,
				path=self._load_path,
				bytes=self._load_bytes,
				load_ms=self._load_ms,
				elapsed_ms=elapsed_ms,
				error=self._load_error,
				detail=self._load_detail,
				cookies_count=self._load_cookies,
				origins_count=self._load_origins,
				script_count=self._load_scripts,
				storage_script_attempts=self._load_storage_script_attempts,
			)

		configured = self.browser_session.browser_profile.storage_state
		if configured is None or configured == '':
			return StorageStateSnapshot(phase=StorageStatePhase.NOT_CONFIGURED)
		if not isinstance(configured, (str, Path)):
			return StorageStateSnapshot(
				phase=StorageStatePhase.SKIPPED,
				detail='in-memory storage state is not loaded from a file',
			)
		path: Path | None = None
		try:
			path = self._expand_storage_path(configured)
			size = path.stat().st_size
		except FileNotFoundError:
			return StorageStateSnapshot(
				phase=StorageStatePhase.SKIPPED,
				path=str(path) if path is not None else str(configured),
				detail='storage-state file does not exist',
			)
		except (OSError, RuntimeError) as error:
			return StorageStateSnapshot(
				phase=StorageStatePhase.FAILED,
				path=str(configured),
				error=f'{type(error).__name__}: {error}',
				detail='storage-state path could not be resolved or inspected',
			)
		return StorageStateSnapshot(
			phase=StorageStatePhase.PENDING,
			path=str(path),
			bytes=size,
		)

	@staticmethod
	def _expand_storage_path(path: str | Path) -> Path:
		return Path(path).expanduser()

	def _finish_load(
		self,
		phase: StorageStatePhase,
		*,
		path: Path | None,
		error: str | None = None,
		detail: str | None = None,
		cookies_count: int = 0,
		origins_count: int = 0,
		script_count: int = 0,
		storage_script_attempts: int = 0,
		bytes_count: int | None = None,
	) -> None:
		started = self._load_started
		self._load_phase = phase
		self._load_path = str(path) if path is not None else None
		if bytes_count is not None:
			self._load_bytes = bytes_count
		self._load_ms = max(0.0, (time.monotonic() - started) * 1000.0) if started is not None else None
		elapsed_ms = self._load_ms
		self._load_started = None
		self._load_error = error
		self._load_detail = detail
		self._load_cookies = cookies_count
		self._load_origins = origins_count
		self._load_scripts = script_count
		self._load_storage_script_attempts = storage_script_attempts
		self._load_elapsed = elapsed_ms

	async def _start_monitoring(self) -> None:
		"""Start the monitoring task."""
		if self._monitoring_task and not self._monitoring_task.done():
			return

		assert self.browser_session.cdp_client is not None

		self._monitoring_task = asyncio.create_task(self._monitor_storage_changes())
		# self.logger'[StorageStateWatchdog] Started storage monitoring task')

	async def _stop_monitoring(self) -> None:
		"""Stop the monitoring task."""
		if self._monitoring_task and not self._monitoring_task.done():
			self._monitoring_task.cancel()
			try:
				await self._monitoring_task
			except asyncio.CancelledError:
				pass
			# self.logger.debug('[StorageStateWatchdog] Stopped storage monitoring task')

	async def _check_for_cookie_changes_cdp(self, event: dict) -> None:
		"""Check if a CDP network event indicates cookie changes.

		This would be called by Network.responseReceivedExtraInfo events
		if we set up CDP event listeners.
		"""
		try:
			# Check for Set-Cookie headers in the response
			headers = event.get('headers', {})
			if 'set-cookie' in headers or 'Set-Cookie' in headers:
				self.logger.debug('[StorageStateWatchdog] Cookie change detected via CDP')

				# If save on change is enabled, trigger save immediately
				if self.save_on_change:
					await self._save_storage_state()
		except Exception as e:
			self.logger.warning(f'[StorageStateWatchdog] Error checking for cookie changes: {e}')

	async def _monitor_storage_changes(self) -> None:
		"""Periodically check for storage changes and auto-save."""
		while True:
			try:
				await asyncio.sleep(self.auto_save_interval)

				# Check if cookies have changed
				if await self._have_cookies_changed():
					self.logger.debug('[StorageStateWatchdog] Detected changes to sync with storage_state.json')
					await self._save_storage_state()

			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f'[StorageStateWatchdog] Error in monitoring loop: {e}')

	async def _have_cookies_changed(self) -> bool:
		"""Check if cookies have changed since last save."""
		if not self.browser_session.cdp_client:
			return False

		try:
			# Get current cookies using CDP
			current_cookies = await self.browser_session._cdp_get_cookies()

			# Convert to comparable format, using .get() for optional fields
			current_cookie_set = {
				(c.get('name', ''), c.get('domain', ''), c.get('path', '')): c.get('value', '') for c in current_cookies
			}

			last_cookie_set = {
				(c.get('name', ''), c.get('domain', ''), c.get('path', '')): c.get('value', '') for c in self._last_cookie_state
			}

			return current_cookie_set != last_cookie_set
		except Exception as e:
			self.logger.debug(f'[StorageStateWatchdog] Error comparing cookies: {e}')
			return False

	async def _save_storage_state(self, path: str | None = None) -> None:
		"""Save browser storage state to file."""
		async with self._save_lock:
			# Check if CDP client is available
			assert await self.browser_session.get_or_create_cdp_session(target_id=None)

			save_path = path or self.browser_session.browser_profile.storage_state
			if not save_path:
				return

			# Skip saving if the storage state is already a dict (indicates it was loaded from memory)
			# We only save to file if it started as a file path
			if isinstance(save_path, dict):
				self.logger.debug('[StorageStateWatchdog] Storage state is already a dict, skipping file save')
				return

			try:
				# Get current storage state using CDP
				storage_state = await self.browser_session._cdp_get_storage_state()

				# Update our last known state
				self._last_cookie_state = storage_state.get('cookies', []).copy()

				# Convert path to Path object
				json_path = Path(save_path).expanduser().resolve()
				json_path.parent.mkdir(parents=True, exist_ok=True)

				# Merge with existing state if file exists
				merged_state = storage_state
				if json_path.exists():
					try:
						existing_state = json.loads(json_path.read_text())
						merged_state = self._merge_storage_states(existing_state, dict(storage_state))
					except Exception as e:
						self.logger.error(f'[StorageStateWatchdog] Failed to merge with existing state: {e}')

				# Write atomically
				temp_path = json_path.with_suffix('.json.tmp')
				temp_path.write_text(json.dumps(merged_state, indent=4))

				# Backup existing file
				if json_path.exists():
					backup_path = json_path.with_suffix('.json.bak')
					json_path.replace(backup_path)

				# Move temp to final
				temp_path.replace(json_path)

				# Emit success event
				self.event_bus.dispatch(
					StorageStateSavedEvent(
						path=str(json_path),
						cookies_count=len(merged_state.get('cookies', [])),
						origins_count=len(merged_state.get('origins', [])),
					)
				)

				self.logger.debug(
					f'[StorageStateWatchdog] Saved storage state to {json_path} '
					f'({len(merged_state.get("cookies", []))} cookies, '
					f'{len(merged_state.get("origins", []))} origins)'
				)

			except Exception as e:
				self.logger.error(f'[StorageStateWatchdog] Failed to save storage state: {e}')

	async def _load_storage_state(self, path: str | None = None) -> None:
		"""Load browser storage state from file."""
		load_path = path or self.browser_session.browser_profile.storage_state
		if load_path is None or load_path == '':
			self._load_bytes = None
			self._finish_load(
				StorageStatePhase.NOT_CONFIGURED,
				path=None,
				detail='storage state is disabled',
			)
			return
		if not isinstance(load_path, (str, Path)):
			self._load_bytes = None
			self._finish_load(
				StorageStatePhase.SKIPPED,
				path=None,
				detail='in-memory storage state is not loaded from a file',
			)
			return

		try:
			json_path = self._expand_storage_path(load_path)
		except (OSError, RuntimeError) as error:
			self._load_bytes = None
			self._finish_load(
				StorageStatePhase.FAILED,
				path=None,
				error=f'{type(error).__name__}: {error}',
				detail=f'storage-state path could not be resolved or inspected: {load_path}',
			)
			return
		try:
			preflight_bytes = json_path.stat().st_size
		except FileNotFoundError:
			self._load_bytes = None
			self._finish_load(
				StorageStatePhase.SKIPPED,
				path=json_path,
				detail='storage-state file does not exist',
			)
			return
		except OSError as error:
			self._load_bytes = None
			self._finish_load(
				StorageStatePhase.FAILED,
				path=None,
				error=f'{type(error).__name__}: {error}',
				detail=f'storage-state path could not be resolved or inspected: {load_path}',
			)
			return
		if getattr(self.browser_session, '_cdp_client_root', None) is None:
			self._load_bytes = preflight_bytes
			self._finish_load(
				StorageStatePhase.FAILED,
				path=json_path,
				error='Browser is not connected; no CDP client is available',
			)
			self.logger.warning('[StorageStateWatchdog] No CDP client available for loading')
			return

		self._load_phase = StorageStatePhase.LOADING
		self._load_path = str(json_path)
		self._load_bytes = preflight_bytes
		self._load_started = time.monotonic()
		self._load_ms = None
		self._load_elapsed = None
		self._load_error = None
		self._load_detail = None
		self._load_cookies = 0
		self._load_origins = 0
		self._load_scripts = 0
		self._load_storage_script_attempts = 0
		cookies_count = 0
		origins_count = 0
		script_count = 0
		storage_script_attempts = 0
		content_bytes: int | None = None

		try:
			import anyio

			content = await anyio.Path(str(json_path)).read_bytes()
			content_bytes = len(content)
			self._load_bytes = content_bytes
			storage = json.loads(content)
			if not isinstance(storage, dict):
				raise ValueError('storage state must be a JSON object')

			if 'cookies' in storage and storage['cookies']:
				await self.browser_session._cdp_set_cookies(storage['cookies'])
				cookies_count = len(storage['cookies'])
				self._load_cookies = cookies_count
				self._last_cookie_state = storage['cookies'].copy()
				self.logger.debug(f'[StorageStateWatchdog] Added {len(storage["cookies"])} cookies from storage state')

			if 'origins' in storage and storage['origins']:
				for origin in storage['origins']:
					origin_name = str(origin.get('origin') or '')
					for storage_type, key in (('localStorage', 'localStorage'), ('sessionStorage', 'sessionStorage')):
						items = origin.get(key) or []
						if not items:
							continue
						scripts = self._build_storage_scripts(storage_type, items, origin_name)
						if len(scripts) > 1:
							warning = (
								f'{storage_type} for {origin_name or "<unknown origin>"} was split into '
								f'{len(scripts)} scripts to respect the size bound'
							)
							self.logger.warning('[StorageStateWatchdog] %s', warning)
							self._record_warning(warning)
						for script in scripts:
							storage_script_attempts += 1
							self._load_storage_script_attempts = storage_script_attempts
							await self.browser_session._cdp_add_init_script(script)
							script_count += 1
							self._load_scripts = script_count
					origins_count += 1
					self._load_origins = origins_count
				self.logger.debug(
					f'[StorageStateWatchdog] Applied localStorage/sessionStorage from {len(storage["origins"])} origins'
				)

			self._finish_load(
				StorageStatePhase.LOADED,
				path=json_path,
				detail='storage state loaded',
				cookies_count=cookies_count,
				origins_count=origins_count,
				script_count=script_count,
				storage_script_attempts=storage_script_attempts,
				bytes_count=content_bytes,
			)

			self.event_bus.dispatch(
				StorageStateLoadedEvent(
					path=str(json_path),
					cookies_count=cookies_count,
					origins_count=origins_count,
				)
			)

			self.logger.debug(f'[StorageStateWatchdog] Loaded storage state from: {json_path}')

		except asyncio.CancelledError:
			self._finish_load(
				StorageStatePhase.CANCELLED,
				path=json_path,
				error='storage-state load cancelled',
				detail=(
					f'cancelled after sending {storage_script_attempts} initialization-script registration(s); '
					f'{script_count} confirmed'
					if storage_script_attempts
					else 'cancelled before any initialization script was sent'
				),
				cookies_count=cookies_count,
				origins_count=origins_count,
				script_count=script_count,
				storage_script_attempts=storage_script_attempts,
				bytes_count=content_bytes,
			)
			raise
		except Exception as e:
			error = f'{type(e).__name__}: {e}'
			self._finish_load(
				StorageStatePhase.FAILED,
				path=json_path,
				error=error,
				detail=(
					f'failed after sending {storage_script_attempts} initialization-script registration(s); '
					f'{script_count} confirmed'
					if storage_script_attempts
					else 'failed before any initialization script was sent'
				),
				cookies_count=cookies_count,
				origins_count=origins_count,
				script_count=script_count,
				storage_script_attempts=storage_script_attempts,
				bytes_count=content_bytes,
			)
			self._record_warning(f'Failed to load storage state: {error}')
			self.logger.error(f'[StorageStateWatchdog] Failed to load storage state: {e}')

	def _record_warning(self, warning: str) -> None:
		warnings = getattr(self.browser_session, '_storage_state_warnings', None)
		if warnings is None:
			return
		if len(warnings) < 20:
			warnings.append(warning)

	def _build_storage_scripts(
		self,
		storage_type: str,
		items: list[dict[str, Any]],
		origin: str = '',
	) -> list[str]:
		"""Build one bounded initialization script per chunk of storage items."""
		clean_items = [
			[str(item.get('name', '')), str(item.get('value', ''))]
			for item in items
			if isinstance(item, dict) and item.get('name') is not None
		]
		if not clean_items:
			return []

		accessor = 'window.localStorage' if storage_type == 'localStorage' else 'window.sessionStorage'
		origin_guard = f'if (window.location.origin !== {json.dumps(origin)}) {{ return; }}' if origin else ''
		prefix = f'(() => {{{origin_guard}const items = '
		suffix = (
			f';for (const [key, value] of items) {{ try {{ {accessor}.setItem(key, value); }} '
			'catch (error) {} }})();'
		)
		overhead = len(prefix.encode('utf-8')) + len(suffix.encode('utf-8'))
		limit = max(1, int(self.max_script_bytes_per_origin))

		chunks: list[list[list[str]]] = []
		current: list[list[str]] = []
		current_size = 2  # JSON array brackets
		for item in clean_items:
			item_size = len(json.dumps(item, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
			candidate_size = current_size + (1 if current else 0) + item_size
			if current and overhead + candidate_size > limit:
				chunks.append(current)
				current = [item]
				current_size = 2 + item_size
			else:
				current.append(item)
				current_size = candidate_size
		if current:
			chunks.append(current)

		scripts = []
		for chunk in chunks:
			payload = json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))
			script = f'{prefix}{payload}{suffix}'
			if len(chunk) == 1 and len(script.encode('utf-8')) > limit:
				warning = (
					f'{storage_type} entry {chunk[0][0]!r} for {origin or "<unknown origin>"} '
					f'exceeds max_script_bytes_per_origin={limit}; keeping it in one script to avoid data loss'
				)
				self.logger.warning('[StorageStateWatchdog] %s', warning)
				self._record_warning(warning)
			scripts.append(script)
		return scripts

	@staticmethod
	def _merge_storage_states(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
		"""Merge two storage states, with new values taking precedence."""
		merged = existing.copy()

		# Merge cookies
		existing_cookies = {(c['name'], c['domain'], c['path']): c for c in existing.get('cookies', [])}

		for cookie in new.get('cookies', []):
			key = (cookie['name'], cookie['domain'], cookie['path'])
			existing_cookies[key] = cookie

		merged['cookies'] = list(existing_cookies.values())

		# Merge origins
		existing_origins = {origin['origin']: origin for origin in existing.get('origins', [])}

		for origin in new.get('origins', []):
			existing_origins[origin['origin']] = origin

		merged['origins'] = list(existing_origins.values())

		return merged

	async def get_current_cookies(self) -> list[dict[str, Any]]:
		"""Get current cookies using CDP."""
		if not self.browser_session.cdp_client:
			return []

		try:
			cookies = await self.browser_session._cdp_get_cookies()
			# Cookie is a TypedDict, cast to dict for compatibility
			return [dict(cookie) for cookie in cookies]
		except Exception as e:
			self.logger.error(f'[StorageStateWatchdog] Failed to get cookies: {e}')
			return []

	async def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
		"""Add cookies using CDP."""
		if not self.browser_session.cdp_client:
			self.logger.warning('[StorageStateWatchdog] No CDP client available for adding cookies')
			return

		try:
			# Convert dicts to Cookie objects
			cookie_objects = [Cookie(**cookie_dict) if isinstance(cookie_dict, dict) else cookie_dict for cookie_dict in cookies]
			# Set cookies using CDP
			await self.browser_session._cdp_set_cookies(cookie_objects)
			self.logger.debug(f'[StorageStateWatchdog] Added {len(cookies)} cookies')
		except Exception as e:
			self.logger.error(f'[StorageStateWatchdog] Failed to add cookies: {e}')
