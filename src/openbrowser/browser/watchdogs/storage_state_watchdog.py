"""Storage state watchdog for managing browser cookies and storage persistence."""

import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit
from weakref import WeakValueDictionary

from bubus import BaseEvent
from cdp_use.cdp.network import Cookie
from filelock import FileLock
from pydantic import Field, PrivateAttr

from openbrowser.browser.events import (
	BrowserConnectedEvent,
	BrowserStopEvent,
	LoadStorageStateEvent,
	SaveStorageStateEvent,
	StorageStateFailedEvent,
	StorageStateLoadedEvent,
	StorageStateSavedEvent,
)
from openbrowser.browser.storage_state import (
	StorageStatePhase,
	StorageStateSavePhase,
	StorageStateSaveSnapshot,
	StorageStateSnapshot,
	load_snapshot_for_configuration,
	save_snapshot_for_configuration,
)
from openbrowser.browser.watchdog_base import BaseWatchdog

_PATH_SAVE_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def _path_save_lock(path: Path) -> asyncio.Lock:
	key = str(path.expanduser().resolve())
	lock = _PATH_SAVE_LOCKS.get(key)
	if lock is None:
		lock = asyncio.Lock()
		_PATH_SAVE_LOCKS[key] = lock
	return lock


def _is_safe_storage_origin(origin: str) -> bool:
	"""Accept only a serialized http or https origin with no path or credentials."""
	if not origin or origin in {'null', 'about:blank'}:
		return False
	try:
		parsed = urlsplit(origin)
	except ValueError:
		return False
	return (
		parsed.scheme in {'http', 'https'}
		and bool(parsed.netloc)
		and parsed.path == ''
		and not parsed.query
		and not parsed.fragment
		and parsed.username is None
		and parsed.password is None
	)


class StorageStateMergeError(RuntimeError):
	"""An existing storage-state file could not be read or merged."""


@dataclass(frozen=True)
class _StorageWriteOutcome:
	"""Result of the owned storage-state writer thread."""

	cancelled: bool
	bytes: int
	cookies_count: int
	origins_count: int


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
		StorageStateFailedEvent,
	]

	# Configuration
	auto_save_interval: float = Field(default=30.0)  # Auto-save every 30 seconds
	save_on_change: bool = Field(default=True)  # Save immediately when cookies change
	max_script_bytes_per_origin: int = Field(default=256_000)  # Bound one packed injected script
	save_lock_timeout: float = Field(default=5.0)  # Cross-process writer lock timeout

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
	_load_skipped_items: int = PrivateAttr(default=0)
	_load_skipped_origins: int = PrivateAttr(default=0)
	_save_phase: StorageStateSavePhase | None = PrivateAttr(default=None)
	_save_path: str | None = PrivateAttr(default=None)
	_save_bytes: int | None = PrivateAttr(default=None)
	_save_started: float | None = PrivateAttr(default=None)
	_save_ms: float | None = PrivateAttr(default=None)
	_save_elapsed: float | None = PrivateAttr(default=None)
	_save_error: str | None = PrivateAttr(default=None)
	_save_detail: str | None = PrivateAttr(default=None)
	_save_cookies: int = PrivateAttr(default=0)
	_save_origins: int = PrivateAttr(default=0)
	_warnings_suppressed: int = PrivateAttr(default=0)

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
				skipped_items=self._load_skipped_items,
				skipped_origins=self._load_skipped_origins,
			)

		configured = self.browser_session.browser_profile.storage_state
		return load_snapshot_for_configuration(configured)

	def storage_state_save_snapshot(self) -> StorageStateSaveSnapshot:
		"""Return the explicit file-save state, including a pre-save view."""
		if self._save_phase is not None:
			elapsed_ms = self._save_elapsed
			if self._save_phase is StorageStateSavePhase.SAVING and self._save_started is not None:
				elapsed_ms = max(0.0, (time.monotonic() - self._save_started) * 1000.0)
			return StorageStateSaveSnapshot(
				phase=self._save_phase,
				path=self._save_path,
				bytes=self._save_bytes,
				save_ms=self._save_ms,
				elapsed_ms=elapsed_ms,
				error=self._save_error,
				detail=self._save_detail,
				cookies_count=self._save_cookies,
				origins_count=self._save_origins,
			)

		configured = self.browser_session.browser_profile.storage_state
		return save_snapshot_for_configuration(configured)

	@staticmethod
	def _expand_storage_path(path: str | Path) -> Path:
		return Path(path).expanduser().resolve()

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
		skipped_items: int = 0,
		skipped_origins: int = 0,
	) -> None:
		started = self._load_started
		self._load_phase = phase
		self._load_path = str(path) if path is not None else None
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
		self._load_skipped_items = skipped_items
		self._load_skipped_origins = skipped_origins
		if phase is StorageStatePhase.FAILED:
			self.event_bus.dispatch(
				StorageStateFailedEvent(
					operation='load',
					path=self._load_path,
					phase=phase.value,
					error=error,
					detail=detail,
				)
			)

	def _finish_save(
		self,
		phase: StorageStateSavePhase,
		*,
		path: Path | None,
		error: str | None = None,
		detail: str | None = None,
		cookies_count: int = 0,
		origins_count: int = 0,
		bytes_count: int | None = None,
	) -> None:
		started = self._save_started
		self._save_phase = phase
		self._save_path = str(path) if path is not None else None
		self._save_bytes = bytes_count
		self._save_ms = max(0.0, (time.monotonic() - started) * 1000.0) if started is not None else None
		elapsed_ms = self._save_ms
		self._save_started = None
		self._save_error = error
		self._save_detail = detail
		self._save_cookies = cookies_count
		self._save_origins = origins_count
		self._save_elapsed = elapsed_ms
		if phase is StorageStateSavePhase.FAILED:
			self.event_bus.dispatch(
				StorageStateFailedEvent(
					operation='save',
					path=self._save_path,
					phase=phase.value,
					error=error,
					detail=detail,
				)
			)

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
			save_path = path or self.browser_session.browser_profile.storage_state
			if not save_path:
				self._finish_save(
					StorageStateSavePhase.NOT_CONFIGURED,
					path=None,
					detail='storage state is not configured',
				)
				return

			if isinstance(save_path, dict):
				self._finish_save(
					StorageStateSavePhase.SKIPPED,
					path=None,
					detail='in-memory storage state is not saved to a file',
				)
				return

			try:
				json_path = Path(save_path).expanduser().resolve()
			except (OSError, RuntimeError) as error:
				self._finish_save(
					StorageStateSavePhase.FAILED,
					path=Path(save_path),
					error=f'{type(error).__name__}: {error}',
					detail='storage-state save path could not be resolved',
				)
				return

			self._save_phase = StorageStateSavePhase.SAVING
			self._save_path = str(json_path)
			self._save_bytes = None
			self._save_started = time.monotonic()
			self._save_ms = None
			self._save_elapsed = None
			self._save_error = None
			self._save_detail = None
			self._save_cookies = 0
			self._save_origins = 0
			try:
				async with _path_save_lock(json_path):
					await self._save_storage_state_locked(json_path)
			except asyncio.CancelledError:
				if self._save_phase is StorageStateSavePhase.SAVING:
					self._finish_save(
						StorageStateSavePhase.CANCELLED,
						path=json_path,
						error='storage-state save cancelled',
						detail='cancelled while waiting for the path lock; previous file preserved',
					)
				raise

	async def _save_storage_state_locked(self, json_path: Path) -> None:
		"""Run CDP collection and the owned writer while holding the path lock."""
		caller_cancelled = False
		try:
			cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=None)
			if cdp_session is None:
				raise RuntimeError('No CDP session available for storage-state save')
			storage_state = await self.browser_session._cdp_get_storage_state()
			if not isinstance(storage_state, dict):
				raise ValueError('storage state must be a mapping')
			cookies = storage_state.get('cookies', [])
			origins = storage_state.get('origins', [])
			if not isinstance(cookies, list) or not all(isinstance(cookie, dict) for cookie in cookies):
				raise ValueError('storage-state cookies must be a list of objects')
			if not isinstance(origins, list) or not all(isinstance(origin, dict) for origin in origins):
				raise ValueError('storage-state origins must be a list of objects')
			last_cookie_state = [dict(cookie) for cookie in cookies]
			cancel_event = threading.Event()
			writer_task = asyncio.create_task(
				asyncio.to_thread(
					self._write_storage_state,
					json_path,
					dict(storage_state),
					cancel_event,
				)
			)
			while not writer_task.done():
				try:
					await asyncio.shield(writer_task)
				except asyncio.CancelledError:
					caller_cancelled = True
					cancel_event.set()
			if writer_task.cancelled():
				raise RuntimeError('storage-state writer was cancelled')
			outcome = writer_task.result()
			if outcome.cancelled:
				self._finish_save(
					StorageStateSavePhase.CANCELLED,
					path=json_path,
					detail='cancelled before the commit gate; previous file preserved',
					cookies_count=outcome.cookies_count,
					origins_count=outcome.origins_count,
				)
			else:
				self._last_cookie_state = last_cookie_state
				self._finish_save(
					StorageStateSavePhase.SAVED,
					path=json_path,
					cookies_count=outcome.cookies_count,
					origins_count=outcome.origins_count,
					bytes_count=outcome.bytes,
				)
				self.event_bus.dispatch(
					StorageStateSavedEvent(
						path=str(json_path),
						cookies_count=outcome.cookies_count,
						origins_count=outcome.origins_count,
						bytes=outcome.bytes,
						save_ms=self._save_ms,
					)
				)
				self.logger.debug(
					f'[StorageStateWatchdog] Saved storage state to {json_path} '
					f'({outcome.cookies_count} cookies, {outcome.origins_count} origins, '
					f'{outcome.bytes} bytes)'
				)
		except asyncio.CancelledError:
			if self._save_phase is StorageStateSavePhase.SAVING:
				self._finish_save(
					StorageStateSavePhase.CANCELLED,
					path=json_path,
					error='storage-state save cancelled',
					detail='cancelled before the commit gate; previous file preserved',
				)
			raise
		except Exception as error:
			error_text = f'{type(error).__name__}: {error}'
			if self._save_phase is StorageStateSavePhase.SAVING:
				self._finish_save(
					StorageStateSavePhase.FAILED,
					path=json_path,
					error=error_text,
					detail='save failed; previous file preserved',
				)
				self._record_warning(f'Failed to save storage state: {error_text}')
				self.logger.error(f'[StorageStateWatchdog] Failed to save storage state: {error}')
			if caller_cancelled:
				raise asyncio.CancelledError() from error
		else:
			if caller_cancelled:
				raise asyncio.CancelledError()

	def _write_storage_state(
		self,
		json_path: Path,
		storage_state: dict[str, Any],
		cancel_event: threading.Event,
	) -> _StorageWriteOutcome:
		"""Read, merge, write, gate, and commit one storage-state file."""
		json_path.parent.mkdir(parents=True, exist_ok=True)
		lock_path = json_path.with_name(f'{json_path.name}.lock')
		with FileLock(str(lock_path), timeout=self.save_lock_timeout):
			merged = dict(storage_state)
			if json_path.exists():
				try:
					existing = json.loads(json_path.read_text())
				except Exception as error:
					raise StorageStateMergeError(
						f'Could not read existing storage state {json_path}: {error}'
					) from error
				if not isinstance(existing, dict):
					raise StorageStateMergeError(
						f'Existing storage state {json_path} is not a JSON object'
					)
				merged = self._merge_storage_states(existing, merged)

			data = json.dumps(merged, indent=4).encode('utf-8')
			temp_path = json_path.with_name(
				f'.{json_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
			)
			try:
				self._write_temp(temp_path, data)
				if cancel_event.is_set():
					return _StorageWriteOutcome(
						cancelled=True,
						bytes=len(data),
						cookies_count=len(merged.get('cookies', [])),
						origins_count=len(merged.get('origins', [])),
					)
				if json_path.exists():
					self._backup_existing(json_path, json_path.with_suffix('.json.bak'))
				self._commit_temp(temp_path, json_path)
				return _StorageWriteOutcome(
					cancelled=False,
					bytes=len(data),
					cookies_count=len(merged.get('cookies', [])),
					origins_count=len(merged.get('origins', [])),
				)
			finally:
				temp_path.unlink(missing_ok=True)

	@staticmethod
	def _write_temp(temp_path: Path, data: bytes) -> Path:
		with open(temp_path, 'wb') as handle:
			handle.write(data)
			handle.flush()
			os.fsync(handle.fileno())
		return temp_path

	@staticmethod
	def _backup_existing(json_path: Path, backup_path: Path) -> None:
		shutil.copy2(json_path, backup_path)

	@staticmethod
	def _commit_temp(temp_path: Path, json_path: Path) -> None:
		os.replace(temp_path, json_path)

	async def _load_storage_state(self, path: str | None = None) -> None:
		"""Load browser storage state from file."""
		self._load_phase = None
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
		self._load_skipped_items = 0
		self._load_skipped_origins = 0
		cookies_count = 0
		origins_count = 0
		script_count = 0
		storage_script_attempts = 0
		skipped_items = 0
		skipped_origins = 0
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

			groups: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
				'localStorage': [],
				'sessionStorage': [],
			}
			for origin in storage.get('origins', []) or []:
				origins_count += 1
				self._load_origins = origins_count
				if not isinstance(origin, dict):
					skipped_origins += 1
					self._load_skipped_origins = skipped_origins
					self._load_skipped_items = skipped_items
					self._emit_warning('storage state contains an invalid origin entry; skipped')
					continue
				origin_name = str(origin.get('origin') or '').strip()
				if not _is_safe_storage_origin(origin_name):
					origin_items = sum(
						len(origin.get(storage_type) or [])
						for storage_type in ('localStorage', 'sessionStorage')
					)
					skipped_origins += 1
					skipped_items += origin_items
					self._load_skipped_origins = skipped_origins
					self._load_skipped_items = skipped_items
					warning = (
						f'storage state contains an unsafe origin {origin_name!r}; skipped '
						f'{origin_items} item(s) to avoid injecting values into an unrelated site'
					)
					self._emit_warning(warning)
					continue
				for storage_type in ('localStorage', 'sessionStorage'):
					items = origin.get(storage_type) or []
					if items:
						groups[storage_type].append((origin_name, items))

			for storage_type, entries in groups.items():
				scripts, split_origins, skipped = self._build_storage_scripts(storage_type, entries)
				skipped_items += skipped
				self._load_skipped_items = skipped_items
				for origin_name in split_origins:
					warning = (
						f'{storage_type} for {origin_name or "<unknown origin>"} was split into '
						f'multiple bounded scripts'
					)
					self._emit_warning(warning)
				for script in scripts:
					storage_script_attempts += 1
					self._load_storage_script_attempts = storage_script_attempts
					await self.browser_session._cdp_add_init_script(script)
					script_count += 1
					self._load_scripts = script_count

			detail = 'storage state loaded'
			if skipped_items or skipped_origins:
				detail = (
					f'storage state loaded with {skipped_items} skipped item(s) '
					f'and {skipped_origins} skipped origin(s)'
				)

			self._finish_load(
				StorageStatePhase.LOADED,
				path=json_path,
				detail=detail,
				cookies_count=cookies_count,
				origins_count=origins_count,
				script_count=script_count,
				storage_script_attempts=storage_script_attempts,
				bytes_count=content_bytes,
				skipped_items=skipped_items,
				skipped_origins=skipped_origins,
			)

			self.event_bus.dispatch(
				StorageStateLoadedEvent(
					path=str(json_path),
					cookies_count=cookies_count,
					origins_count=origins_count,
					bytes=content_bytes,
					load_ms=self._load_ms,
					script_count=script_count,
					skipped_items=skipped_items,
					skipped_origins=skipped_origins,
				)
			)

			self.logger.debug(f'[StorageStateWatchdog] Loaded storage state from: {json_path}')

		except asyncio.CancelledError:
			self._finish_load(
				StorageStatePhase.CANCELLED,
				path=json_path,
				error='storage-state load cancelled',
				detail=(
					f'cancelled after attempting {storage_script_attempts} initialization-script registration(s); '
					f'{script_count} confirmed'
					if storage_script_attempts
					else 'cancelled before any initialization script was sent'
				),
				cookies_count=cookies_count,
				origins_count=origins_count,
				script_count=script_count,
				storage_script_attempts=storage_script_attempts,
				bytes_count=content_bytes,
				skipped_items=skipped_items,
				skipped_origins=skipped_origins,
			)
			raise
		except Exception as e:
			error = f'{type(e).__name__}: {e}'
			self._finish_load(
				StorageStatePhase.FAILED,
				path=json_path,
				error=error,
				detail=(
					f'failed after attempting {storage_script_attempts} initialization-script registration(s); '
					f'{script_count} confirmed'
					if storage_script_attempts
					else 'failed before any initialization script was sent'
				),
				cookies_count=cookies_count,
				origins_count=origins_count,
				script_count=script_count,
				storage_script_attempts=storage_script_attempts,
				bytes_count=content_bytes,
				skipped_items=skipped_items,
				skipped_origins=skipped_origins,
			)
			self._record_warning(f'Failed to load storage state: {error}')
			self.logger.error(f'[StorageStateWatchdog] Failed to load storage state: {e}')

	def _emit_warning(self, warning: str) -> None:
		"""Record and log a warning with a shared bounded cap."""
		warnings = getattr(self.browser_session, '_storage_state_warnings', None)
		if warnings is None:
			self.logger.warning('[StorageStateWatchdog] %s', warning)
			return
		if len(warnings) < 20:
			warnings.append(warning)
			self.logger.warning('[StorageStateWatchdog] %s', warning)
			return
		self._warnings_suppressed += 1
		if self._warnings_suppressed == 1:
			self.logger.warning('[StorageStateWatchdog] additional storage-state warnings suppressed')

	def _record_warning(self, warning: str) -> None:
		warnings = getattr(self.browser_session, '_storage_state_warnings', None)
		if warnings is None:
			return
		if len(warnings) < 20:
			warnings.append(warning)

	def _build_storage_scripts(
		self,
		storage_type: str,
		groups: list[tuple[str, list[dict[str, Any]]]],
	) -> tuple[list[str], list[str], int]:
		"""Pack many origin blocks into bounded scripts."""
		prefix = '(() => {'
		suffix = '})();'
		limit = max(1, int(self.max_script_bytes_per_origin))
		blocks: list[str] = []
		split_origins: list[str] = []
		skipped_items = 0

		for origin, items in groups:
			clean_items: list[list[str]] = []
			for item in items:
				if not isinstance(item, dict) or item.get('name') is None:
					skipped_items += 1
					continue
				clean_items.append([str(item.get('name', '')), str(item.get('value', ''))])
			if not clean_items:
				continue

			origin_blocks, origin_skipped = self._split_origin_blocks(
				storage_type,
				origin,
				clean_items,
				limit,
				prefix,
				suffix,
			)
			skipped_items += origin_skipped
			if len(origin_blocks) > 1:
				split_origins.append(origin)
			blocks.extend(origin_blocks)

		scripts: list[str] = []
		current = prefix
		for block in blocks:
			candidate = current + block
			if current != prefix and len((candidate + suffix).encode('utf-8')) > limit:
				scripts.append(current + suffix)
				current = prefix + block
			else:
				current = candidate
		if current != prefix:
			scripts.append(current + suffix)
		return scripts, split_origins, skipped_items

	def _split_origin_blocks(
		self,
		storage_type: str,
		origin: str,
		items: list[list[str]],
		limit: int,
		prefix: str,
		suffix: str,
	) -> tuple[list[str], int]:
		"""Split one origin into blocks that fit, skipping indivisible oversize items."""
		script_overhead = len(prefix.encode('utf-8')) + len(suffix.encode('utf-8'))
		block_overhead = len(self._render_storage_block(storage_type, origin, []).encode('utf-8'))
		item_budget = max(1, limit - script_overhead - block_overhead)

		chunks: list[list[list[str]]] = []
		current: list[list[str]] = []
		current_size = 2
		for item in items:
			item_size = len(json.dumps(item, ensure_ascii=True, separators=(',', ':')).encode('utf-8'))
			candidate_size = current_size + (1 if current else 0) + item_size
			if current and candidate_size > item_budget:
				chunks.append(current)
				current = [item]
				current_size = 2 + item_size
			else:
				current.append(item)
				current_size = candidate_size
		if current:
			chunks.append(current)

		blocks: list[str] = []
		skipped = 0
		for chunk in chunks:
			if len((prefix + self._render_storage_block(storage_type, origin, chunk) + suffix).encode('utf-8')) <= limit:
				blocks.append(self._render_storage_block(storage_type, origin, chunk))
				continue
			sub_blocks, sub_skipped = self._split_oversized_items(
				storage_type,
				origin,
				chunk,
				limit,
				prefix,
				suffix,
			)
			blocks.extend(sub_blocks)
			skipped += sub_skipped
		return blocks, skipped

	def _split_oversized_items(
		self,
		storage_type: str,
		origin: str,
		items: list[list[str]],
		limit: int,
		prefix: str,
		suffix: str,
	) -> tuple[list[str], int]:
		block = self._render_storage_block(storage_type, origin, items)
		if len((prefix + block + suffix).encode('utf-8')) <= limit:
			return [block], 0
		if len(items) == 1:
			emitted_bytes = len((prefix + block + suffix).encode('utf-8'))
			warning = (
				f'{storage_type} entry {items[0][0]!r} for {origin} would emit '
				f'{emitted_bytes} bytes, exceeding max_script_bytes_per_origin={limit}; skipped'
			)
			self._emit_warning(warning)
			return [], 1
		middle = len(items) // 2
		left_blocks, left_skipped = self._split_oversized_items(
			storage_type,
			origin,
			items[:middle],
			limit,
			prefix,
			suffix,
		)
		right_blocks, right_skipped = self._split_oversized_items(
			storage_type,
			origin,
			items[middle:],
			limit,
			prefix,
			suffix,
		)
		return left_blocks + right_blocks, left_skipped + right_skipped

	@staticmethod
	def _render_storage_block(storage_type: str, origin: str, items: list[list[str]]) -> str:
		accessor = 'window.localStorage' if storage_type == 'localStorage' else 'window.sessionStorage'
		items_json = json.dumps(items, ensure_ascii=True, separators=(',', ':'))
		body = (
			f'const items={items_json};'
			f'for(const [key,value] of items){{try{{{accessor}.setItem(key,value);}}catch(error){{}}}}'
		)
		origin_json = json.dumps(origin, ensure_ascii=True)
		return f'if(window.location.origin==={origin_json}){{{body}}}'

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
