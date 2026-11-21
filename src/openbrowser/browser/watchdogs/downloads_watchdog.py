"""Downloads watchdog for monitoring and handling file downloads."""

import asyncio
import contextlib
import hashlib
import json
import os
import re
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import parse_qsl, urlparse, urlsplit

import anyio
from bubus import BaseEvent
from cdp_use.cdp.browser import DownloadProgressEvent, DownloadWillBeginEvent
from cdp_use.cdp.network import ResponseReceivedEvent
from cdp_use.cdp.target import SessionID, TargetID
from pydantic import PrivateAttr

from openbrowser.browser.download_state import (
	DownloadPhase,
	DownloadRegistry,
	PdfDetectionPhase,
	PdfDetectionSnapshot,
	pdf_validation_error,
	redacted_outcome,
	redacted_text,
)
from openbrowser.browser.events import (
	BrowserLaunchEvent,
	BrowserStateRequestEvent,
	BrowserStoppedEvent,
	FileDownloadedEvent,
	NavigationCompleteEvent,
	TabClosedEvent,
	TabCreatedEvent,
)
from openbrowser.browser.watchdog_base import BaseWatchdog

if TYPE_CHECKING:
	pass


PDF_VALIDATION_TIMEOUT_SECONDS = 30.0


class DownloadsWatchdog(BaseWatchdog):
	"""Monitors downloads and handles file download events."""

	# Events this watchdog listens to (for documentation)
	LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
		BrowserLaunchEvent,
		BrowserStateRequestEvent,
		BrowserStoppedEvent,
		TabCreatedEvent,
		TabClosedEvent,
		NavigationCompleteEvent,
	]

	# Events this watchdog emits
	EMITS: ClassVar[list[type[BaseEvent[Any]]]] = [
		FileDownloadedEvent,
	]

	# Private state
	_sessions_with_listeners: set[str] = PrivateAttr(default_factory=set)  # Track sessions that already have download listeners
	_active_downloads: dict[str, Any] = PrivateAttr(default_factory=dict)
	_pdf_document_mime: OrderedDict[tuple[str, str], str] = PrivateAttr(default_factory=OrderedDict)
	_pdf_url_epoch: OrderedDict[str, int] = PrivateAttr(default_factory=OrderedDict)
	_pdf_detection_phase: PdfDetectionPhase = PrivateAttr(default=PdfDetectionPhase.IDLE)
	_pdf_detection_url: str | None = PrivateAttr(default=None)
	_pdf_detection_resource_url: str | None = PrivateAttr(default=None)
	_pdf_detection_error: str | None = PrivateAttr(default=None)
	_pdf_detection_detail: str | None = PrivateAttr(default=None)
	_pdf_detection_started: float | None = PrivateAttr(default=None)
	_pdf_detection_finished: float | None = PrivateAttr(default=None)
	_pdf_detection_ms: float | None = PrivateAttr(default=None)
	_pdf_detection_serial: int = PrivateAttr(default=0)
	_pdf_published_serial: int = PrivateAttr(default=0)
	_download_cdp_session_setup: bool = PrivateAttr(default=False)  # Track if CDP session is set up
	_download_cdp_session: Any = PrivateAttr(default=None)  # Store CDP session reference
	_cdp_event_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)  # Track CDP event handler tasks
	_native_completion_pending: set[str] = PrivateAttr(default_factory=set)
	_cdp_downloads_info: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)  # Map guid -> info
	_use_js_fetch_for_local: bool = PrivateAttr(default=False)  # Guard JS fetch path for local regular downloads
	_session_pdf_urls: dict[str, str] = PrivateAttr(default_factory=dict)  # URL -> path for PDFs downloaded this session
	_network_monitored_targets: set[str] = PrivateAttr(default_factory=set)  # Track targets with network monitoring enabled
	_detected_downloads: set[str] = PrivateAttr(default_factory=set)  # Track detected download URLs to avoid duplicates
	_network_callback_registered: bool = PrivateAttr(default=False)  # Track if global network callback is registered
	_download_registry: DownloadRegistry = PrivateAttr(default_factory=DownloadRegistry)  # Terminal outcome per download
	_js_fetch_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)  # Serialize browser-fetch writes
	_download_serial: int = PrivateAttr(default=0)  # Unique synthetic id for guid-less events
	_polling_names: dict[str, int] = PrivateAttr(default_factory=dict)  # Active pollers per name family
	_ambiguous_paths: set[str] = PrivateAttr(default_factory=set)  # Files seen while two pollers competed
	_downloads_admission: bool = PrivateAttr(default=True)  # False once shutdown starts

	async def on_BrowserLaunchEvent(self, event: BrowserLaunchEvent) -> None:
		self.logger.debug(f'[DownloadsWatchdog] Received BrowserLaunchEvent, EventBus ID: {id(self.event_bus)}')
		# Ensure downloads directory exists
		downloads_path = self.browser_session.browser_profile.downloads_path
		if downloads_path:
			expanded_path = Path(downloads_path).expanduser().resolve()
			expanded_path.mkdir(parents=True, exist_ok=True)
			self.logger.debug(f'[DownloadsWatchdog] Ensured downloads directory exists: {expanded_path}')
		self._download_registry.clear()
		self._polling_names.clear()
		self._ambiguous_paths.clear()
		self._native_completion_pending.clear()
		self._pdf_url_epoch.clear()
		self._downloads_admission = True
		self._cdp_downloads_info.clear()

	async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
		"""Monitor new tabs for downloads."""
		# logger.info(f'[DownloadsWatchdog] TabCreatedEvent received for tab {event.target_id[-4:]}: {event.url}')

		# Assert downloads path is configured (should always be set by BrowserProfile default)
		assert self.browser_session.browser_profile.downloads_path is not None, 'Downloads path must be configured'

		if event.target_id:
			# logger.info(f'[DownloadsWatchdog] Found target for tab {event.target_id}, calling attach_to_target')
			await self.attach_to_target(event.target_id)
		else:
			self.logger.warning(f'[DownloadsWatchdog] No target found for tab {event.target_id}')

	async def on_TabClosedEvent(self, event: TabClosedEvent) -> None:
		"""Stop monitoring closed tabs."""
		pass  # No cleanup needed, browser context handles target lifecycle

	async def on_BrowserStateRequestEvent(self, event: BrowserStateRequestEvent) -> None:
		"""Handle browser state request events."""
		cdp_session = self.browser_session.agent_focus
		if not cdp_session:
			return

		url = await self.browser_session.get_current_page_url()
		if not url:
			return

		target_id = cdp_session.target_id
		self.event_bus.dispatch(
			NavigationCompleteEvent(
				event_type='NavigationCompleteEvent',
				url=url,
				target_id=target_id,
				event_parent_id=event.event_id,
			)
		)

	async def on_BrowserStoppedEvent(self, event: BrowserStoppedEvent) -> None:
		"""Clean up when browser stops."""
		await self.shutdown()

		# Clean up CDP session
		# CDP sessions are now cached and managed by BrowserSession
		self._download_cdp_session = None
		self._download_cdp_session_setup = False

		# Clear other state
		self._sessions_with_listeners.clear()
		self._active_downloads.clear()
		self._polling_names.clear()
		self._cdp_downloads_info.clear()
		self._pdf_document_mime.clear()
		self._pdf_url_epoch.clear()
		self._session_pdf_urls.clear()
		self._network_monitored_targets.clear()
		self._detected_downloads.clear()
		self._native_completion_pending.clear()
		self._network_callback_registered = False

	async def on_NavigationCompleteEvent(self, event: NavigationCompleteEvent) -> None:
		"""Check for PDFs after navigation completes."""
		self.logger.debug(
			f'[DownloadsWatchdog] NavigationCompleteEvent received for '
			f'{self._safe_log_url(event.url)}, tab #{event.target_id[-4:]}'
		)

		if not self._is_auto_download_enabled():
			return

		target_id = event.target_id
		self.logger.debug(f'[DownloadsWatchdog] Got target_id={target_id} for tab #{event.target_id[-4:]}')

		detection = await self._detect_pdf(target_id, url=event.url)
		if detection.phase is PdfDetectionPhase.PDF:
			self.logger.debug(
				f'[DownloadsWatchdog] 📄 PDF detected at {self._safe_log_url(event.url)}, '
				'triggering auto-download...'
			)
			download_path = await self.trigger_pdf_download(
				target_id,
				resource_url=detection.resource_url,
			)
			if not download_path:
				self.logger.warning(
					f'[DownloadsWatchdog] ⚠️ PDF download failed for {self._safe_log_url(event.url)}'
				)
			return

		if detection.phase is PdfDetectionPhase.CHECK_FAILED:
			self.logger.warning(
				f'[DownloadsWatchdog] PDF check failed for {self._safe_log_url(event.url)}; '
				'retry on next navigation: '
				f'{detection.error or detection.detail or "unknown error"}'
			)

	def _is_auto_download_enabled(self) -> bool:
		"""Check if auto-download PDFs is enabled in browser profile."""
		return self.browser_session.browser_profile.auto_download_pdfs

	async def attach_to_target(self, target_id: TargetID) -> None:
		"""Set up download monitoring for a specific target."""

		# Define CDP event handlers outside of try to avoid indentation/scope issues
		def download_will_begin_handler(event: DownloadWillBeginEvent, session_id: SessionID | None) -> None:
			self.logger.debug(
				f'[DownloadsWatchdog] Download will begin: guid={event.get("guid", "")!r}, '
				f'url={self._safe_log_url(event.get("url", ""))}, '
				f'filename={self._safe_filename(event.get("suggestedFilename"))!r}'
			)
			# Cache info for later completion event handling (esp. remote browsers)
			guid = event.get('guid', '')
			download_url = event.get('url', '')
			raw_suggested_filename = event.get('suggestedFilename')
			suggested_filename = (
				self._safe_filename(raw_suggested_filename)
				if raw_suggested_filename
				else None
			)
			if guid:
				self._cdp_downloads_info[guid] = {
					'url': download_url,
					'suggested_filename': suggested_filename,
					'frame_id': event.get('frameId'),
					'handled': False,
				}
			download_id = guid or self._next_synthetic_download_id('cdp')
			if not self._downloads_admission:
				self.logger.debug('[DownloadsWatchdog] Shutdown started; ignoring new download')
				return
			if self._download_registry.active_count() >= self._download_registry.max_active:
				self._download_registry.reject(
					download_id,
					stage='capacity',
					error=f"Too many active downloads (limit {self._download_registry.max_active})",
					url=download_url,
					source='cdp',
					detail='The browser transfer was not tracked',
				)
				self.logger.warning(
					f'[DownloadsWatchdog] Download capacity reached; rejecting '
					f'{self._safe_log_url(download_url)}'
				)
				return
			self._download_registry.start(
				download_id,
				url=download_url,
				file_name=suggested_filename,
				source='cdp',
			)
			if guid and guid in self._cdp_downloads_info:
				self._cdp_downloads_info[guid]['download_id'] = download_id
			# Create and track the task
			task = asyncio.create_task(self._handle_cdp_download(event, None, session_id, download_id))
			self._cdp_event_tasks.add(task)
			# Remove from set when done
			task.add_done_callback(lambda t: self._cdp_event_tasks.discard(t))

		def download_progress_handler(event: DownloadProgressEvent, session_id: SessionID | None) -> None:
			guid = event.get('guid', '')
			info = self._cdp_downloads_info.get(guid, {}) if guid else {}
			download_id = info.get('download_id') or guid or None
			if download_id is None:
				# Without a guid or a stored mapping this event cannot be tied
				# to one attempt. Never guess from the set of pending downloads.
				self.logger.debug('[DownloadsWatchdog] Progress event without a guid; ignoring')
				return
			# A browser-canceled transfer is a terminal failure, not silence.
			if event.get('state') == 'canceled':
				if download_id in self._native_completion_pending:
					self.logger.debug('[DownloadsWatchdog] Ignoring cancel after a native completion was observed')
					return
				self._download_registry.cancel(
					download_id,
					detail='download canceled by the browser',
				)
				return
			# Check if download is complete
			if event.get('state') == 'completed':
				existing = self._download_registry.get(download_id)
				if (
					existing is not None
					and existing.terminal
					and (existing.phase is not DownloadPhase.FAILED or existing.rejected)
				):
					return
				file_path = event.get('filePath')
				total_bytes = event.get('totalBytes')
				if self.browser_session.is_local:
					if file_path and not Path(file_path).exists():
						# Reported but not written yet. Leave the attempt started
						# so the polling path can confirm it.
						self.logger.debug(
							f'[DownloadsWatchdog] Reported path is not present yet: {file_path}'
						)
					elif file_path:
						self.logger.debug(f'[DownloadsWatchdog] Download completed: {file_path}')
						info = self._cdp_downloads_info.get(guid, {})
						try:
							event_target_id = (
								self.browser_session.get_target_id_from_session_id(session_id)
								if session_id
								else None
							)
						except Exception:
							event_target_id = None
						event_target_id = event_target_id or info.get('target_id')
						task = asyncio.create_task(
							self._complete_native_download(
								download_id=download_id,
								file_path=file_path,
								url=str(info.get('url') or ''),
								total_bytes=int(total_bytes) if isinstance(total_bytes, int) else None,
								source='cdp',
								guid=guid,
								target_id=event_target_id,
								session_id=session_id,
								frame_id=info.get('frame_id'),
							)
						)
						self._native_completion_pending.add(download_id)
						self._cdp_event_tasks.add(task)
						task.add_done_callback(lambda t: self._cdp_event_tasks.discard(t))
					else:
						# No local file path provided, local polling in _handle_cdp_download will handle it
						self.logger.debug(
							'[DownloadsWatchdog] No filePath in progress event (local); polling will handle detection'
						)
				else:
					# Remote browser: do not touch local filesystem. Fallback to downloadPath+suggestedFilename
					info = self._cdp_downloads_info.get(guid, {})
					try:
						suggested_filename = info.get('suggested_filename') or (Path(file_path).name if file_path else 'download')
						downloads_path = str(self.browser_session.browser_profile.downloads_path or '')
						effective_path = file_path or str(Path(downloads_path) / suggested_filename)
						file_name = Path(effective_path).name
						file_ext = Path(file_name).suffix.lower().lstrip('.')
						download_url = str(info.get('url') or '')
						remote_pdf = self._is_pdf_file_candidate(download_url, str(effective_path))
						self.event_bus.dispatch(
							FileDownloadedEvent(
								url=download_url,
								path=str(effective_path),
								file_name=file_name,
								file_size=0,
								file_type=file_ext if file_ext else None,
							)
						)
						self._download_registry.complete(
							download_id,
							path=str(effective_path),
							file_size=int(total_bytes) if isinstance(total_bytes, int) else None,
							file_name=file_name,
							file_type=file_ext if file_ext else None,
							url=download_url,
							source='cdp',
							detail=(
								'remote browser did not expose file bytes; content validation unavailable'
								if remote_pdf
								else None
							),
						)
						self.logger.debug(f'[DownloadsWatchdog] ✅ (remote) Download completed: {effective_path}')
					finally:
						if guid in self._cdp_downloads_info:
							del self._cdp_downloads_info[guid]

		try:
			downloads_path_raw = self.browser_session.browser_profile.downloads_path
			if not downloads_path_raw:
				# logger.info(f'[DownloadsWatchdog] No downloads path configured, skipping target: {target_id}')
				return  # No downloads path configured

			# Check if we already have a download listener on this session
			# to prevent duplicate listeners from being added
			# Note: Since download listeners are set up once per browser session, not per target,
			# we just track if we've set up the browser-level listener
			if self._download_cdp_session_setup:
				self.logger.debug('[DownloadsWatchdog] Download listener already set up for browser session')
				return

			# logger.debug(f'[DownloadsWatchdog] Setting up CDP download listener for target: {target_id}')

			# Use CDP session for download events but store reference in watchdog
			if not self._download_cdp_session_setup:
				# Set up CDP session for downloads (only once per browser session)
				cdp_client = self.browser_session.cdp_client

				# Set download behavior to allow downloads and enable events
				downloads_path = self.browser_session.browser_profile.downloads_path
				if not downloads_path:
					self.logger.warning('[DownloadsWatchdog] No downloads path configured, skipping CDP download setup')
					return
				# Ensure path is properly expanded (~ -> absolute path)
				expanded_downloads_path = Path(downloads_path).expanduser().resolve()
				await cdp_client.send.Browser.setDownloadBehavior(
					params={
						'behavior': 'allow',
						'downloadPath': str(expanded_downloads_path),  # Use expanded absolute path
						'eventsEnabled': True,
					}
				)

				# Register the handlers with CDP
				cdp_client.register.Browser.downloadWillBegin(download_will_begin_handler)  # type: ignore[arg-type]
				cdp_client.register.Browser.downloadProgress(download_progress_handler)  # type: ignore[arg-type]

				self._download_cdp_session_setup = True
				self.logger.debug('[DownloadsWatchdog] Set up CDP download listeners')

			# No need to track individual targets since download listener is browser-level
			# logger.debug(f'[DownloadsWatchdog] Successfully set up CDP download listener for target: {target_id}')

		except Exception as e:
			self.logger.warning(
				f'[DownloadsWatchdog] Failed to set up CDP download listener for target '
				f'{target_id}: {self._safe_log_error(e)}'
			)

		# Set up network monitoring for this target (catches ALL download variants)
		await self._setup_network_monitoring(target_id)

	async def _setup_network_monitoring(self, target_id: TargetID) -> None:
		"""Set up network monitoring to detect PDFs and downloads from ALL sources.

		This catches:
		- Direct PDF navigation
		- PDFs in iframes
		- PDFs with embed/object tags
		- JavaScript-triggered downloads
		- Any Content-Disposition: attachment headers
		"""
		# Skip if already monitoring this target
		if target_id in self._network_monitored_targets:
			self.logger.debug(f'[DownloadsWatchdog] Network monitoring already enabled for target {target_id[-4:]}')
			return

		# Check if auto-download is enabled
		if not self._is_auto_download_enabled():
			self.logger.debug('[DownloadsWatchdog] Auto-download disabled, skipping network monitoring')
			return

		try:
			cdp_client = self.browser_session.cdp_client

			# Register the global callback once
			if not self._network_callback_registered:

				def on_response_received(event: ResponseReceivedEvent, session_id: str | None) -> None:
					"""Handle Network.responseReceived event to detect downloadable content.

					This callback is registered globally and uses session_id to determine the correct target.
					"""
					try:
						# Look up target_id from session_id
						event_target_id = self.browser_session.get_target_id_from_session_id(session_id)
						if not event_target_id:
							# Session not in pool - might be a stale session or not yet tracked
							return

						# Only process events for targets we're monitoring
						if event_target_id not in self._network_monitored_targets:
							return

						response = event.get('response', {})
						url = response.get('url', '')
						content_type = response.get('mimeType', '').lower()
						raw_headers = response.get('headers', {})
						headers = (
							{str(key).lower(): str(value) for key, value in raw_headers.items()}
							if isinstance(raw_headers, dict)
							else {}
						)

						# Skip non-HTTP URLs (data:, about:, chrome-extension:, etc.)
						if not url.startswith('http'):
							return

						response_type = str(event.get('type') or '').lower()
						# Chrome may classify a PDF navigation as "Other" rather than
						# "Document". Record PDF MIME regardless of the response type;
						# the page URL lookup keeps subresource evidence isolated.
						if content_type and (response_type == 'document' or 'application/pdf' in content_type):
							self._note_document_mime(event_target_id, url, content_type)

						# Check if it's a PDF
						is_pdf = 'application/pdf' in content_type

						# Check if it's marked as download via Content-Disposition header
						content_disposition = headers.get('content-disposition', '').lower()
						is_download_attachment = 'attachment' in content_disposition

						# Filter out image/video/audio files even if marked as attachment
						# These are likely resources, not intentional downloads
						unwanted_content_types = [
							'image/',
							'video/',
							'audio/',
							'text/css',
							'text/javascript',
							'application/javascript',
							'application/x-javascript',
							'text/html',
							'application/json',
							'font/',
							'application/font',
							'application/x-font',
						]
						is_unwanted_type = any(content_type.startswith(prefix) for prefix in unwanted_content_types)
						if is_unwanted_type:
							return

						# Check URL extension to filter out obvious images/resources
						url_lower = url.lower().split('?')[0]  # Remove query params
						unwanted_extensions = [
							'.jpg',
							'.jpeg',
							'.png',
							'.gif',
							'.webp',
							'.svg',
							'.ico',
							'.css',
							'.js',
							'.woff',
							'.woff2',
							'.ttf',
							'.eot',
							'.mp4',
							'.webm',
							'.mp3',
							'.wav',
							'.ogg',
						]
						if any(url_lower.endswith(ext) for ext in unwanted_extensions):
							return

						# Only process if it's a PDF or download
						if not (is_pdf or is_download_attachment):
							return

						# Check if we've already processed this URL in this session
						if url in self._detected_downloads:
							self.logger.debug(
								f'[DownloadsWatchdog] Already detected download: '
								f'{self._safe_log_url(url)}'
							)
							return

						# Mark as detected to avoid duplicates
						self._detected_downloads.add(url)

						# Extract filename from Content-Disposition if available
						suggested_filename = None
						if 'filename=' in content_disposition:
							# Parse filename from Content-Disposition header
							import re

							filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
							if filename_match:
								suggested_filename = filename_match.group(1).strip('\'"')

						self.logger.info(
							f'[DownloadsWatchdog] 🔍 Detected downloadable content via network: '
							f'{self._safe_log_url(url)}'
						)
						self.logger.debug(
							f'[DownloadsWatchdog]   Content-Type: {content_type}, Is PDF: {is_pdf}, Is Attachment: {is_download_attachment}'
						)

						# Trigger download asynchronously in background (don't block event handler)
						async def download_in_background():
							download_path = None
							try:
								download_path = await self.download_file_from_url(
									url=url,
									target_id=event_target_id,  # Use target_id from session_id lookup
									content_type=content_type,
									suggested_filename=suggested_filename,
								)

								if download_path:
									self.logger.info(f'[DownloadsWatchdog] ✅ Successfully downloaded: {download_path}')
								else:
									self.logger.warning(
										f'[DownloadsWatchdog] ⚠️  Failed to download: '
										f'{self._safe_log_url(url)}'
									)
							except Exception as e:
								self.logger.error(
									f'[DownloadsWatchdog] Error downloading in background: '
									f'{self._safe_log_error(e)}'
								)
							finally:
								if not download_path:
									self._detected_downloads.discard(url)

						# Create background task
						task = asyncio.create_task(download_in_background())
						self._cdp_event_tasks.add(task)
						task.add_done_callback(lambda t: self._cdp_event_tasks.discard(t))

					except Exception as e:
						self.logger.error(
							f'[DownloadsWatchdog] Error in network response handler: '
							f'{self._safe_log_error(e)}'
						)

				# Register the callback globally (once)
				cdp_client.register.Network.responseReceived(on_response_received)
				self._network_callback_registered = True
				self.logger.debug('[DownloadsWatchdog] ✅ Registered global network response callback')

			# Get or create CDP session for this target
			cdp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=False)

			# Enable Network domain to monitor HTTP responses (per-target/per-session)
			await cdp_client.send.Network.enable(session_id=cdp_session.session_id)
			self.logger.debug(f'[DownloadsWatchdog] Enabled Network domain for target {target_id[-4:]}')

			# Mark this target as monitored
			self._network_monitored_targets.add(target_id)
			self.logger.debug(f'[DownloadsWatchdog] ✅ Network monitoring enabled for target {target_id[-4:]}')

		except Exception as e:
			self.logger.warning(
				f'[DownloadsWatchdog] Failed to set up network monitoring for target '
				f'{target_id}: {self._safe_log_error(e)}'
			)

	async def download_file_from_url(
		self, url: str, target_id: TargetID, content_type: str | None = None, suggested_filename: str | None = None
	) -> str | None:
		"""Generic method to download any file from a URL.

		Args:
			url: The URL to download
			target_id: The target ID for CDP session
			content_type: Optional content type (e.g., 'application/pdf')
			suggested_filename: Optional filename from Content-Disposition header

		Returns:
			Path to downloaded file, or None if download failed
		"""
		download_id = f'url:{url}:{time.time_ns()}'
		url_epoch = self._pdf_url_epoch.get(url, 0)
		try:
			async with self._js_fetch_lock:
				return await self._download_file_from_url_unlocked(
					url,
					target_id,
					content_type=content_type,
					suggested_filename=suggested_filename,
					download_id=download_id,
					url_epoch=url_epoch,
				)
		except asyncio.CancelledError:
			self._download_registry.cancel(
				download_id,
				detail='download cancelled while queued',
				url=url,
				source='js_fetch',
			)
			raise

	async def _download_file_from_url_unlocked(
		self,
		url: str,
		target_id: TargetID,
		content_type: str | None = None,
		suggested_filename: str | None = None,
		*,
		download_id: str,
		url_epoch: int,
	) -> str | None:
		safe_suggested_filename = self._safe_filename(suggested_filename) if suggested_filename else None
		if not self.browser_session.browser_profile.downloads_path:
			self._download_registry.fail(
				download_id,
				stage='configuration',
				error='No downloads path configured',
				url=url,
				source='js_fetch',
			)
			self.logger.warning('[DownloadsWatchdog] No downloads path configured')
			return None

		self._download_registry.start(download_id, url=url, source='js_fetch')

		# Check if already downloaded in this session.
		is_pdf_request = (
			'application/pdf' in (content_type or '').lower()
			or self._check_url_for_pdf(url)
			or bool(safe_suggested_filename and Path(safe_suggested_filename).suffix.lower() == '.pdf')
		)
		if url in self._session_pdf_urls:
			if is_pdf_request:
				existing_path, cache_error = await self._validated_pdf_cache_entry(url)
			else:
				existing_path = self._session_pdf_urls[url]
				cache_error = None
				if not await asyncio.to_thread(os.path.isfile, existing_path):
					self._session_pdf_urls.pop(url, None)
					existing_path = None
					cache_error = 'Cached download path is not a file'
			if existing_path is not None:
				self._download_registry.complete(
					download_id,
					path=existing_path,
					file_size=await asyncio.to_thread(os.path.getsize, existing_path),
					file_name=Path(existing_path).name,
					from_cache=True,
					source='session_cache',
					detail='already downloaded in this session',
				)
				self.logger.debug(f'[DownloadsWatchdog] File already downloaded in session: {existing_path}')
				return existing_path
			if cache_error is not None:
				self.logger.warning(
					f'[DownloadsWatchdog] Cached download failed validation; refetching: {cache_error}'
				)

		try:
			# Get or create CDP session for this target
			temp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=False)

			# Determine filename
			if safe_suggested_filename:
				filename = safe_suggested_filename
			else:
				# Extract from URL
				filename = self._safe_filename(
					os.path.basename(url.split('?')[0]),
					fallback='download',
				)  # Remove query params and confine
				if not filename or '.' not in filename:
					# Fallback: use content type to determine extension
					if content_type and 'pdf' in content_type:
						filename = 'document.pdf'
					else:
						filename = 'download'

			# Ensure downloads directory exists
			downloads_dir = str(self.browser_session.browser_profile.downloads_path)
			os.makedirs(downloads_dir, exist_ok=True)

			# Generate unique filename if file exists
			final_filename = filename
			existing_files = os.listdir(downloads_dir)
			if filename in existing_files:
				base, ext = os.path.splitext(filename)
				counter = 1
				while f'{base} ({counter}){ext}' in existing_files:
					counter += 1
				final_filename = f'{base} ({counter}){ext}'
				self.logger.debug(f'[DownloadsWatchdog] File exists, using: {final_filename}')

			self.logger.debug(f'[DownloadsWatchdog] Downloading from: {self._safe_log_url(url)}')

			# Download using JavaScript fetch to leverage browser cache
			escaped_url = json.dumps(url)

			result = await asyncio.wait_for(
				temp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': f"""
				(async () => {{
					try {{
						const response = await fetch({escaped_url}, {{
							cache: 'force-cache'
						}});
						if (!response.ok) {{
							throw new Error(`HTTP error! status: ${{response.status}}`);
						}}
						const blob = await response.blob();
						const arrayBuffer = await blob.arrayBuffer();
						const uint8Array = new Uint8Array(arrayBuffer);

						return {{
							data: Array.from(uint8Array),
							responseSize: uint8Array.length
						}};
					}} catch (error) {{
						throw new Error(`Fetch failed: ${{error.message}}`);
					}}
				}})()
				""",
						'awaitPromise': True,
						'returnByValue': True,
					},
					session_id=temp_session.session_id,
				),
				timeout=15.0,  # 15 second timeout
			)

			download_result = result.get('result', {}).get('value', {})

			payload = download_result.get('data') if isinstance(download_result, dict) else None
			if isinstance(payload, list):
				data_bytes = bytes(payload)
				if is_pdf_request:
					validation_error = await self._pdf_validation_error_async(data_bytes)
					if validation_error:
						self._download_registry.fail(
							download_id,
							stage='content_validation',
							error=validation_error,
							detail=f'content-type={content_type or "unknown"}',
							url=url,
							source='js_fetch',
						)
						self.logger.warning(
							f'[DownloadsWatchdog] Refusing non-PDF content from {self._safe_log_url(url)}: '
							f'{validation_error}'
						)
						return None
				download_path = os.path.join(downloads_dir, final_filename)

				actual_size, failure_stage, write_error = await self._write_bytes_atomically(
					download_path,
					data_bytes,
				)
				if write_error is not None:
					self._download_registry.fail(
						download_id,
						stage=failure_stage or 'write',
						error=write_error,
						url=url,
						source='js_fetch',
					)
					return None
				assert actual_size is not None
				self.logger.debug(f'[DownloadsWatchdog] File written: {download_path} ({actual_size} bytes)')

				# Determine file type
				file_ext = Path(final_filename).suffix.lower().lstrip('.')
				mime_type = content_type or f'application/{file_ext}'

				# Store URL->path mapping for this session
				if self._pdf_url_epoch.get(url, 0) == url_epoch:
					self._session_pdf_urls[url] = download_path
				else:
					self.logger.debug(
						f'[DownloadsWatchdog] Newer response invalidated fetched bytes; not caching '
						f'{self._safe_log_url(url)}'
					)

				# Emit file downloaded event
				self.logger.debug(f'[DownloadsWatchdog] Dispatching FileDownloadedEvent for {final_filename}')
				self.event_bus.dispatch(
					FileDownloadedEvent(
						url=url,
						path=download_path,
						file_size=actual_size,
						file_name=final_filename,
						file_type=file_ext if file_ext else None,
						mime_type=mime_type,
						auto_download=True,
					)
				)
				self._download_registry.complete(
					download_id,
					path=download_path,
					file_size=actual_size,
					file_name=final_filename,
					file_type=file_ext if file_ext else None,
					mime_type=mime_type,
					url=url,
					source='js_fetch',
				)
				return download_path
			else:
				self.logger.warning(
					f'[DownloadsWatchdog] No data received when downloading from '
					f'{self._safe_log_url(url)}'
				)
				self._download_registry.fail(
					download_id,
					stage='fetch',
					error='No data received from browser fetch',
					url=url,
					source='js_fetch',
				)
				return None

		except asyncio.CancelledError:
			self._download_registry.cancel(download_id, detail='download fetch cancelled')
			raise
		except TimeoutError:
			self.logger.warning(
				f'[DownloadsWatchdog] Download timed out: {self._safe_log_url(url)}'
			)
			self._download_registry.fail(
				download_id,
				stage='fetch_timeout',
				error='Browser fetch timed out after 15s',
				url=url,
				source='js_fetch',
			)
			return None
		except Exception as e:
			self.logger.warning(
				f'[DownloadsWatchdog] Download failed: {self._safe_log_error(e)}'
			)
			self._download_registry.fail(
				download_id,
				stage='fetch',
				error=f'{type(e).__name__}: {e}',
				url=url,
				source='js_fetch',
			)
			return None

	def _track_download(self, file_path: str) -> None:
		"""Track a completed download and dispatch the appropriate event.

		Args:
			file_path: The path to the downloaded file
		"""
		try:
			# Get file info
			path = Path(file_path)
			if path.exists():
				file_size = path.stat().st_size
				self.logger.debug(f'[DownloadsWatchdog] Tracked download: {path.name} ({file_size} bytes)')

				# Dispatch download event
				from openbrowser.browser.events import FileDownloadedEvent

				self.event_bus.dispatch(
					FileDownloadedEvent(
						url=str(path),  # Use the file path as URL for local files
						path=str(path),
						file_name=path.name,
						file_size=file_size,
					)
				)
			else:
				self.logger.warning(f'[DownloadsWatchdog] Downloaded file not found: {file_path}')
		except Exception as e:
			self.logger.error(
				f'[DownloadsWatchdog] Error tracking download: {self._safe_log_error(e)}'
			)

	async def _resolve_download_target_id(
		self,
		session_id: SessionID | None,
		frame_id: str | None,
	) -> TargetID | None:
		if session_id:
			try:
				target_id = self.browser_session.get_target_id_from_session_id(session_id)
			except Exception:
				target_id = None
			if target_id:
				return str(target_id)
		if frame_id:
			try:
				all_frames, _target_sessions = await self.browser_session.get_all_frames()
				frame_info = await self.browser_session.find_frame_target(frame_id, all_frames)
			except Exception:
				return None
			target_id = frame_info.get('frameTargetId') if frame_info else None
			return str(target_id) if target_id else None
		return None

	async def _handle_cdp_download(
		self,
		event: DownloadWillBeginEvent,
		target_id: TargetID | None,
		session_id: SessionID | None,
		download_id: str | None = None,
	) -> None:
		"""Handle a CDP Page.downloadWillBegin event."""
		if target_id is None:
			target_id = await self._resolve_download_target_id(session_id, event.get('frameId'))
			if target_id:
				guid = str(event.get('guid') or '')
				if guid and guid in self._cdp_downloads_info:
					self._cdp_downloads_info[guid]['target_id'] = target_id
		downloads_dir = (
			Path(
				self.browser_session.browser_profile.downloads_path
				or f'{tempfile.gettempdir()}/openbrowser_downloads.{str(self.browser_session.id)[-4:]}'
			)
			.expanduser()
			.resolve()
		)  # Ensure path is properly expanded

		# Initialize variables that may be used outside try blocks
		unique_filename = None
		file_size = 0
		expected_path = None
		download_result = None
		download_url = event.get('url', '')
		suggested_filename = self._safe_filename(event.get('suggestedFilename'), fallback='download')
		guid = event.get('guid', '')
		if download_id is None:
			download_id = guid or self._next_synthetic_download_id('cdp')

		try:
			self.logger.debug(
				f'[DownloadsWatchdog] ⬇️ File download starting: {suggested_filename} from '
				f'{self._safe_log_url(download_url)}'
			)
			self.logger.debug(
				f'[DownloadsWatchdog] CDP event: guid={guid!r}, '
				f'url={self._safe_log_url(download_url)}, filename={suggested_filename!r}'
			)

			# Since Browser.setDownloadBehavior is already configured, the browser will download the file
			# We just need to wait for it to appear in the downloads directory
			expected_path = downloads_dir / suggested_filename

			# Debug: List current directory contents
			self.logger.debug(f'[DownloadsWatchdog] Downloads directory: {downloads_dir}')
			if downloads_dir.exists():
				files_before = list(downloads_dir.iterdir())
				self.logger.debug(f'[DownloadsWatchdog] Files before download: {[f.name for f in files_before]}')

			# Try manual JavaScript fetch as a fallback for local browsers (disabled for regular local downloads)
			if target_id and self.browser_session.is_local and self._use_js_fetch_for_local:
				self.logger.debug(
					f'[DownloadsWatchdog] Attempting JS fetch fallback for '
					f'{self._safe_log_url(download_url)}'
				)

				unique_filename = None
				file_size = None
				download_result = None
				try:
					# Escape the URL for JavaScript
					import json

					escaped_url = json.dumps(download_url)

					# Get the proper session for the frame that initiated the download
					cdp_session = await self.browser_session.cdp_client_for_frame(event.get('frameId'))
					assert cdp_session

					result = await cdp_session.cdp_client.send.Runtime.evaluate(
						params={
							'expression': f"""
						(async () => {{
							try {{
								const response = await fetch({escaped_url});
								if (!response.ok) {{
									throw new Error(`HTTP error! status: ${{response.status}}`);
								}}
								const blob = await response.blob();
								const arrayBuffer = await blob.arrayBuffer();
								const uint8Array = new Uint8Array(arrayBuffer);
								return {{
									data: Array.from(uint8Array),
									size: uint8Array.length,
									contentType: response.headers.get('content-type') || 'application/octet-stream'
								}};
							}} catch (error) {{
								throw new Error(`Fetch failed: ${{error.message}}`);
							}}
						}})()
						""",
							'awaitPromise': True,
							'returnByValue': True,
						},
						session_id=cdp_session.session_id,
					)
					download_result = result.get('result', {}).get('value')

					if download_result and download_result.get('data'):
						# Save the file
						file_data = bytes(download_result['data'])
						content_type = str(download_result.get('contentType') or '')

						# Ensure unique filename
						unique_filename = await self._get_unique_filename(str(downloads_dir), suggested_filename)
						final_path = downloads_dir / unique_filename
						if self._is_pdf_file_candidate(download_url, str(final_path), content_type):
							validation_error = await self._pdf_validation_error_async(file_data)
							if validation_error:
								self._download_registry.fail(
									download_id,
									stage='content_validation',
									error=validation_error,
									detail=f'content-type={content_type or "unknown"}',
									url=download_url,
									source='js_fetch',
								)
								return

						file_size, failure_stage, write_error = await self._write_bytes_atomically(
							str(final_path),
							file_data,
						)
						if write_error is not None:
							self._download_registry.fail(
								download_id,
								stage=failure_stage or 'write',
								error=write_error,
								url=download_url,
								source='js_fetch',
							)
							return
						assert file_size is not None

						self.logger.debug(f'[DownloadsWatchdog] ✅ Downloaded and saved file: {final_path} ({file_size} bytes)')
						expected_path = final_path
						if self._is_pdf_file_candidate(download_url, str(final_path), content_type):
							self._session_pdf_urls[download_url] = str(final_path)
						# Emit download event immediately
						file_ext = expected_path.suffix.lower().lstrip('.')
						file_type = file_ext if file_ext else None
						self.event_bus.dispatch(
							FileDownloadedEvent(
								url=download_url,
								path=str(expected_path),
								file_name=unique_filename or expected_path.name,
								file_size=file_size or 0,
								file_type=file_type,
								mime_type=(download_result.get('contentType') if download_result else None),
								from_cache=False,
								auto_download=False,
							)
						)
						# Mark as handled to prevent duplicate dispatch from progress/polling paths
						try:
							if guid in self._cdp_downloads_info:
								self._cdp_downloads_info[guid]['handled'] = True
						except (KeyError, AttributeError):
							pass
						self._download_registry.complete(
							download_id,
							path=str(expected_path),
							file_size=file_size or 0,
							file_name=unique_filename or expected_path.name,
							file_type=file_type,
							mime_type=(download_result.get('contentType') if download_result else None),
							url=download_url,
							source='js_fetch',
						)
						self.logger.debug(
							f'[DownloadsWatchdog] ✅ File download completed via CDP: {suggested_filename} ({file_size} bytes) saved to {expected_path}'
						)
						return
					else:
						self.logger.error('[DownloadsWatchdog] ❌ No data received from fetch')

				except asyncio.CancelledError:
					self._download_registry.cancel(download_id, detail='download fetch cancelled')
					raise
				except Exception as fetch_error:
					self.logger.error(
						f'[DownloadsWatchdog] ❌ Failed to download file via fetch: '
						f'{self._safe_log_error(fetch_error)}'
					)

			# For remote browsers, don't poll local filesystem; downloadProgress handler will emit the event
			if not self.browser_session.is_local:
				return
		except Exception as e:
			self.logger.error(
				f'[DownloadsWatchdog] ❌ Error handling CDP download: {self._safe_log_error(e)}'
			)

		# If we reach here, the fetch method failed, so wait for native download
		await self._poll_for_download_file(
			download_id=download_id,
			download_url=download_url,
			downloads_dir=downloads_dir,
			suggested_filename=suggested_filename,
			guid=guid,
			target_id=target_id,
		)

	async def _poll_for_download_file(
		self,
		*,
		download_id: str,
		download_url: str,
		downloads_dir: Path,
		suggested_filename: str,
		guid: str,
		target_id: str | None = None,
	) -> None:
		"""Poll for a name-matched file and complete this attempt at most once."""
		family = self._download_family(suggested_filename)
		url_epoch = self._pdf_url_epoch.get(download_url, 0)
		self._polling_names[family] = self._polling_names.get(family, 0) + 1
		try:
			initial_files = set()
			if Path(downloads_dir).exists():
				for existing in Path(downloads_dir).iterdir():
					if existing.is_file() and not existing.name.startswith('.'):
						initial_files.add(existing.name)

			max_wait = 20  # seconds
			start_time = asyncio.get_event_loop().time()

			while asyncio.get_event_loop().time() - start_time < max_wait:
				existing = self._download_registry.get(download_id)
				if existing is not None and existing.terminal:
					return
				try:
					await asyncio.sleep(5.0)  # Check every 5 seconds
				except asyncio.CancelledError:
					self._download_registry.cancel(download_id, detail='download polling cancelled')
					raise

				# With two attempts suggesting the same name, a name match cannot
				# prove ownership. Leave both to the CDP progress event.
				self._prune_ambiguous_paths()
				if self._polling_names.get(family, 0) > 1:
					if Path(downloads_dir).exists():
						for candidate in Path(downloads_dir).iterdir():
							if (
								candidate.is_file()
								and candidate.name not in initial_files
								and self._download_name_matches(candidate.name, suggested_filename)
							):
								self._ambiguous_paths.add(str(candidate))
					continue

				if Path(downloads_dir).exists():
					for file_path in Path(downloads_dir).iterdir():
						if (
							file_path.is_file()
							and not file_path.name.startswith('.')
							and file_path.name not in initial_files
							and str(file_path) not in self._ambiguous_paths
							and self._download_name_matches(file_path.name, suggested_filename)
						):
							try:
								file_size = file_path.stat().st_size
								if file_size >= 0:
									self.logger.debug(
										f'[DownloadsWatchdog] ✅ Found downloaded file: {file_path} ({file_size} bytes)'
									)
									file_ext = file_path.suffix.lower().lstrip('.')
									file_type = file_ext if file_ext else None
									existing = self._download_registry.get(download_id)
									if existing is not None and existing.terminal:
										return
									if not await self._validate_pdf_file_path(
										str(file_path),
										url=download_url,
										download_id=download_id,
										source='native_poll',
										target_id=target_id,
									):
										return
									self._download_registry.complete(
										download_id,
										path=str(file_path),
										file_size=file_size,
										file_name=file_path.name,
										file_type=file_type,
										source='native_poll',
										url=download_url,
									)
									if (
										self._is_pdf_file_candidate(
											download_url,
											str(file_path),
											target_id=target_id,
										)
										and self._pdf_url_epoch.get(download_url, 0) == url_epoch
									):
										self._session_pdf_urls[download_url] = str(file_path)
									info = self._cdp_downloads_info.get(guid, {})
									if info.get('handled'):
										return
									self.event_bus.dispatch(
										FileDownloadedEvent(
											url=download_url,
											path=str(file_path),
											file_name=file_path.name,
											file_size=file_size,
											file_type=file_type,
										)
									)
									try:
										if guid in self._cdp_downloads_info:
											self._cdp_downloads_info[guid]['handled'] = True
									except (KeyError, AttributeError):
										pass
									return
							except Exception as e:
								self.logger.debug(
									f'[DownloadsWatchdog] Error checking file {file_path}: '
									f'{self._safe_log_error(e)}'
								)

			self._download_registry.fail(
				download_id,
				stage='native_poll',
				error=f'Download did not complete within {max_wait} seconds',
				url=download_url,
				source='cdp',
			)
			self.logger.warning(f'[DownloadsWatchdog] Download did not complete within {max_wait} seconds')
		finally:
			remaining = self._polling_names.get(family, 1) - 1
			if remaining <= 0:
				self._polling_names.pop(family, None)
			else:
				self._polling_names[family] = remaining

	async def _handle_download(self, download: Any) -> None:
		"""Handle a download event."""
		download_id = self._next_synthetic_download_id('playwright')
		self._active_downloads[download_id] = download
		self._download_registry.start(
			download_id,
			url=download.url,
			file_name=download.suggested_filename,
			source='playwright',
		)
		self.logger.debug(
			f'[DownloadsWatchdog] ⬇️ Handling download: {download.suggested_filename} from '
			f'{self._safe_log_url(download.url)}'
		)

		try:
			# Debug: Check if download is already being handled elsewhere
			failure = (
				await download.failure()
			)  # TODO: it always fails for some reason, figure out why connect_over_cdp makes accept_downloads not work
			self.logger.warning(
				f'[DownloadsWatchdog] ❌ Download state - canceled: {redacted_text(failure)}, '
				f'url: {self._safe_log_url(download.url)}'
			)
			current_step = 'getting_download_info'
			# Get download info immediately
			url = download.url
			suggested_filename = self._safe_filename(download.suggested_filename)

			current_step = 'determining_download_directory'
			# Determine download directory from browser profile
			downloads_dir = self.browser_session.browser_profile.downloads_path
			if not downloads_dir:
				downloads_dir = str(Path.home() / 'Downloads')
			else:
				downloads_dir = str(downloads_dir)  # Ensure it's a string

			# Check if Playwright already auto-downloaded the file (due to CDP setup)
			original_path = Path(downloads_dir) / suggested_filename
			if original_path.exists() and original_path.stat().st_size > 0:
				self.logger.debug(
					f'[DownloadsWatchdog] File already downloaded by Playwright: {original_path} ({original_path.stat().st_size} bytes)'
				)

				# Use the existing file instead of creating a duplicate
				download_path = original_path
				file_size = original_path.stat().st_size
				unique_filename = suggested_filename
			else:
				current_step = 'generating_unique_filename'
				# Ensure unique filename
				unique_filename = await self._get_unique_filename(downloads_dir, suggested_filename)
				download_path = Path(downloads_dir) / unique_filename

				self.logger.debug(
					f'[DownloadsWatchdog] Download started: {unique_filename} from '
					f'{self._safe_log_url(url)}'
				)

				current_step = 'calling_save_as'
				# Save the download using Playwright's save_as method
				self.logger.debug(f'[DownloadsWatchdog] Saving download to: {download_path}')
				self.logger.debug(f'[DownloadsWatchdog] Download path exists: {download_path.parent.exists()}')
				self.logger.debug(f'[DownloadsWatchdog] Download path writable: {os.access(download_path.parent, os.W_OK)}')

				try:
					self.logger.debug('[DownloadsWatchdog] About to call download.save_as()...')
					await download.save_as(str(download_path))
					self.logger.debug(f'[DownloadsWatchdog] Successfully saved download to: {download_path}')
					current_step = 'save_as_completed'
				except Exception as save_error:
					self.logger.error(f'[DownloadsWatchdog] save_as() failed with error: {save_error}')
					raise save_error

				# Get file info
				file_size = download_path.stat().st_size if download_path.exists() else 0

			# Determine file type from extension
			file_ext = download_path.suffix.lower().lstrip('.')
			file_type = file_ext if file_ext else None
			if not await self._validate_pdf_file_path(
				str(download_path),
				url=url,
				download_id=download_id,
				source='playwright',
			):
				return
			if self._is_pdf_file_candidate(url, str(download_path)):
				self._session_pdf_urls[url] = str(download_path)

			# Try to get MIME type from response headers if available
			mime_type = None
			# Note: Playwright doesn't expose response headers directly from Download object

			# Check if this was a PDF auto-download
			auto_download = False
			if file_type == 'pdf':
				auto_download = self._is_auto_download_enabled()

			# Emit download event
			self.event_bus.dispatch(
				FileDownloadedEvent(
					url=url,
					path=str(download_path),
					file_name=suggested_filename,
					file_size=file_size,
					file_type=file_type,
					mime_type=mime_type,
					from_cache=False,
					auto_download=auto_download,
				)
			)

			self._download_registry.complete(
				download_id,
				path=str(download_path),
				file_size=file_size,
				file_name=suggested_filename,
				file_type=file_type,
				mime_type=mime_type,
				auto_download=auto_download,
				url=url,
				source='playwright',
			)
			self.logger.debug(
				f'[DownloadsWatchdog] ✅ Download completed: {suggested_filename} ({file_size} bytes) saved to {download_path}'
			)

			# File is now tracked on filesystem, no need to track in memory

		except asyncio.CancelledError:
			self._download_registry.cancel(download_id, detail='playwright download cancelled')
			raise
		except Exception as e:
			stage = locals().get('current_step', 'unknown')
			self._download_registry.fail(
				download_id,
				stage=str(stage),
				error=f'{type(e).__name__}: {e}',
				url=download.url,
				source='playwright',
			)
			self.logger.error(
				f'[DownloadsWatchdog] Error handling download at step "{stage}", error: {e}'
			)
			self.logger.error(
				f'[DownloadsWatchdog] Download state - URL: {self._safe_log_url(download.url)}, '
				f'filename: {download.suggested_filename}'
			)
		finally:
			# Clean up tracking
			if download_id in self._active_downloads:
				del self._active_downloads[download_id]

	async def _complete_native_download(
		self,
		*,
		download_id: str,
		file_path: str,
		url: str,
		total_bytes: int | None,
		source: str,
		target_id: str | None = None,
		guid: str | None = None,
		session_id: str | None = None,
		frame_id: str | None = None,
	) -> None:
		"""Validate a finished native download before publishing completion."""
		self._native_completion_pending.add(download_id)
		url_epoch = self._pdf_url_epoch.get(url, 0)
		try:
			if target_id is None:
				target_id = await self._resolve_download_target_id(session_id, frame_id)
			existing = self._download_registry.get(download_id)
			if (
				existing is not None
				and existing.terminal
				and (existing.phase is not DownloadPhase.FAILED or existing.rejected)
			):
				return
			content_type = None
			if not await self._validate_pdf_file_path(
				file_path,
				url=url,
				download_id=download_id,
				source=source,
				content_type=content_type,
				target_id=target_id,
			):
				return
			path = Path(file_path)
			file_size = total_bytes if total_bytes is not None else (await asyncio.to_thread(path.stat)).st_size
			file_ext = path.suffix.lower().lstrip('.')
			existing = self._download_registry.get(download_id)
			if (
				existing is not None
				and existing.terminal
				and (existing.phase is not DownloadPhase.FAILED or existing.rejected)
			):
				return
			self._track_download(file_path)
			self._download_registry.complete(
				download_id,
				path=str(file_path),
				file_size=file_size,
				file_name=path.name,
				file_type=file_ext if file_ext else None,
				url=url,
				source=source,
			)
			if (
				self._is_pdf_file_candidate(url, file_path, target_id=target_id)
				and self._pdf_url_epoch.get(url, 0) == url_epoch
			):
				self._session_pdf_urls[url] = str(file_path)
			if guid:
				try:
					if guid in self._cdp_downloads_info:
						self._cdp_downloads_info[guid]['handled'] = True
				except (KeyError, AttributeError):
					pass
		except asyncio.CancelledError:
			raise
		except Exception as completion_error:
			self._download_registry.fail(
				download_id,
				stage='native_completion',
				error=f'{type(completion_error).__name__}: {completion_error}',
				url=url,
				source=source,
			)
			self.logger.warning(
				f'[DownloadsWatchdog] Native download completion failed for '
				f'{self._safe_log_url(url)}: {self._safe_log_error(completion_error)}'
			)
		finally:
			self._native_completion_pending.discard(download_id)

	def pdf_detection_snapshot(self) -> PdfDetectionSnapshot:
		"""Return the latest PDF detection attempt."""
		duration_ms = self._pdf_detection_ms
		if (
			self._pdf_detection_phase is PdfDetectionPhase.CHECKING
			and self._pdf_detection_started is not None
		):
			duration_ms = max(0.0, (time.time() - self._pdf_detection_started) * 1000.0)
		return PdfDetectionSnapshot(
			phase=self._pdf_detection_phase,
			url=self._pdf_detection_url,
			resource_url=self._pdf_detection_resource_url,
			error=self._pdf_detection_error,
			detail=self._pdf_detection_detail,
			started_at=self._pdf_detection_started,
			finished_at=self._pdf_detection_finished,
			duration_ms=duration_ms,
		)

	@staticmethod
	def _pdf_snapshot(
		phase: PdfDetectionPhase,
		*,
		url: str | None,
		started_at: float,
		resource_url: str | None = None,
		error: str | None = None,
		detail: str | None = None,
		finished_at: float | None = None,
	) -> PdfDetectionSnapshot:
		duration_ms = None
		if finished_at is not None:
			duration_ms = max(0.0, (finished_at - started_at) * 1000.0)
		return PdfDetectionSnapshot(
			phase=phase,
			url=url,
			resource_url=resource_url,
			error=error,
			detail=detail,
			started_at=started_at,
			finished_at=finished_at,
			duration_ms=duration_ms,
		)

	def _publish_pdf_detection(self, snapshot: PdfDetectionSnapshot, serial: int) -> None:
		"""Publish an attempt unless a newer attempt has already started."""
		if serial < self._pdf_published_serial:
			return
		self._pdf_published_serial = serial
		self._pdf_detection_phase = snapshot.phase
		self._pdf_detection_url = snapshot.url
		self._pdf_detection_resource_url = snapshot.resource_url
		self._pdf_detection_error = snapshot.error
		self._pdf_detection_detail = snapshot.detail
		self._pdf_detection_started = snapshot.started_at
		self._pdf_detection_finished = snapshot.finished_at
		self._pdf_detection_ms = snapshot.duration_ms

	def _begin_pdf_detection(self, url: str | None) -> tuple[int, float]:
		self._pdf_detection_serial += 1
		serial = self._pdf_detection_serial
		started_at = time.time()
		self._publish_pdf_detection(
			self._pdf_snapshot(PdfDetectionPhase.CHECKING, url=url, started_at=started_at),
			serial,
		)
		return serial, started_at

	def _note_document_mime(self, target_id: str, url: str, mime_type: str) -> None:
		"""Record observed response MIME for one target and URL."""
		if not target_id or not url or not mime_type:
			return
		key = (str(target_id), url)
		self._pdf_document_mime[key] = mime_type.lower()
		self._pdf_document_mime.move_to_end(key)
		while len(self._pdf_document_mime) > 200:
			self._pdf_document_mime.popitem(last=False)
		if 'application/pdf' not in mime_type.lower():
			self._pdf_url_epoch[url] = self._pdf_url_epoch.get(url, 0) + 1
			self._pdf_url_epoch.move_to_end(url)
			while len(self._pdf_url_epoch) > 200:
				self._pdf_url_epoch.popitem(last=False)
			if self._session_pdf_urls.pop(url, None) is not None:
				self.logger.debug(
					f'[DownloadsWatchdog] Non-PDF response invalidated the PDF cache: '
					f'{self._safe_log_url(url)}'
				)

	async def _pdf_validation_error_async(self, data: bytes) -> str | None:
		"""Run PDF validation off the event loop with a bounded timeout."""
		try:
			return await asyncio.wait_for(
				asyncio.to_thread(pdf_validation_error, data),
				timeout=PDF_VALIDATION_TIMEOUT_SECONDS,
			)
		except TimeoutError:
			return f'PDF validation timed out after {PDF_VALIDATION_TIMEOUT_SECONDS:g} seconds'
		except Exception as validation_error:
			return f'PDF validation failed: {type(validation_error).__name__}: {validation_error}'

	async def _validated_pdf_cache_entry(self, url: str) -> tuple[str | None, str | None]:
		"""Return a validated cached PDF path, dropping an invalid entry."""
		existing_path = self._session_pdf_urls.get(url)
		if existing_path is None:
			return None, None
		if not os.path.isfile(existing_path):
			if self._session_pdf_urls.get(url) == existing_path:
				self._session_pdf_urls.pop(url, None)
			return None, 'Cached PDF path is not a file'
		try:
			cached_bytes = await asyncio.to_thread(Path(existing_path).read_bytes)
		except OSError as read_error:
			if self._session_pdf_urls.get(url) == existing_path:
				self._session_pdf_urls.pop(url, None)
			return None, f'{type(read_error).__name__}: {read_error}'
		validation_error = await self._pdf_validation_error_async(cached_bytes)
		if validation_error is not None:
			if self._session_pdf_urls.get(url) == existing_path:
				self._session_pdf_urls.pop(url, None)
			return None, validation_error
		if self._session_pdf_urls.get(url) != existing_path:
			return None, 'Cached PDF entry changed during validation'
		return existing_path, None

	def _is_pdf_file_candidate(
		self,
		url: str,
		path: str,
		content_type: str | None = None,
		target_id: str | None = None,
	) -> bool:
		mime_evidence = any(
			known_url == url
			and 'application/pdf' in mime_type
			and target_id is not None
			and known_target_id == str(target_id)
			for (known_target_id, known_url), mime_type in self._pdf_document_mime.items()
		)
		return (
			'application/pdf' in (content_type or '').lower()
			or self._check_url_for_pdf(url)
			or Path(path).suffix.lower() == '.pdf'
			or mime_evidence
		)

	@staticmethod
	def _is_definitive_non_pdf_mime(mime_type: str) -> bool:
		normalized = mime_type.lower().split(';', 1)[0].strip()
		return normalized.startswith(('text/', 'image/', 'audio/', 'video/', 'font/')) or normalized in {
			'application/xhtml+xml',
			'application/xml',
			'application/json',
			'application/zip',
		}

	async def _validate_pdf_file_path(
		self,
		path: str,
		*,
		url: str,
		download_id: str,
		source: str,
		content_type: str | None = None,
		target_id: str | None = None,
	) -> bool:
		"""Validate a completed local PDF download, removing rejected content."""
		if not self._is_pdf_file_candidate(url, path, content_type, target_id=target_id):
			return True
		try:
			data = await asyncio.to_thread(Path(path).read_bytes)
		except OSError as read_error:
			self._download_registry.fail(
				download_id,
				stage='content_validation',
				error=f'{type(read_error).__name__}: {read_error}',
				url=url,
				source=source,
			)
			return False
		validation_error = await self._pdf_validation_error_async(data)
		if validation_error is None:
			return True
		with contextlib.suppress(OSError):
			os.unlink(path)
		self._download_registry.fail(
			download_id,
			stage='content_validation',
			error=validation_error,
			detail=f'content-type={content_type or "unknown"}',
			url=url,
			source=source,
		)
		self.logger.warning(
			f'[DownloadsWatchdog] Refusing non-PDF native download from '
			f'{self._safe_log_url(url)}: {validation_error}'
		)
		return False

	async def _detect_pdf(self, target_id: TargetID, url: str | None = None) -> PdfDetectionSnapshot:
		"""Run one isolated detection attempt and publish its terminal snapshot."""
		serial, started_at = self._begin_pdf_detection(url)
		page_url = url
		self.logger.debug(f'[DownloadsWatchdog] Checking if target {target_id} is PDF viewer...')

		try:
			cdp_client = self.browser_session.cdp_client
			targets = await cdp_client.send.Target.getTargets()
			target_info = next((t for t in targets['targetInfos'] if t['targetId'] == target_id), None)
			if not target_info:
				snapshot = self._pdf_snapshot(
					PdfDetectionPhase.CHECK_FAILED,
					url=url,
					started_at=started_at,
					finished_at=time.time(),
					error=f'No target info found for {target_id}',
					detail='target lookup failed',
				)
				self._publish_pdf_detection(snapshot, serial)
				self.logger.warning(f'[DownloadsWatchdog] No target info found for {target_id}')
				return snapshot

			page_url = str(target_info.get('url', '') or url or '')

			document_mime = self._pdf_document_mime.get((str(target_id), page_url), '')
			if 'application/pdf' in document_mime:
				self.logger.debug(
					f'[DownloadsWatchdog] PDF detected via document MIME: {self._safe_log_url(page_url)}'
				)
				snapshot = self._pdf_snapshot(
					PdfDetectionPhase.PDF,
					url=page_url,
					started_at=started_at,
					finished_at=time.time(),
					resource_url=page_url,
					detail=f'main document response mime {document_mime}',
				)
				self._publish_pdf_detection(snapshot, serial)
				return snapshot

			if self._is_definitive_non_pdf_mime(document_mime):
				snapshot = self._pdf_snapshot(
					PdfDetectionPhase.NOT_PDF,
					url=page_url,
					started_at=started_at,
					finished_at=time.time(),
					detail=f'observed non-PDF document mime {document_mime}',
				)
				self._publish_pdf_detection(snapshot, serial)
				return snapshot

			if self._check_url_for_pdf(page_url):
				self.logger.debug(
					f'[DownloadsWatchdog] PDF detected via URL pattern: {self._safe_log_url(page_url)}'
				)
				snapshot = self._pdf_snapshot(
					PdfDetectionPhase.PDF,
					url=page_url,
					started_at=started_at,
					finished_at=time.time(),
					resource_url=page_url,
					detail='strict URL pattern',
				)
				self._publish_pdf_detection(snapshot, serial)
				return snapshot

			if self._is_chrome_pdf_viewer_url(page_url):
				self.logger.debug(
					f'[DownloadsWatchdog] Chrome PDF viewer detected: {self._safe_log_url(page_url)}'
				)
				snapshot = self._pdf_snapshot(
					PdfDetectionPhase.PDF,
					url=page_url,
					started_at=started_at,
					finished_at=time.time(),
					detail='Chrome PDF viewer identity',
				)
				self._publish_pdf_detection(snapshot, serial)
				return snapshot

			history_result = await self._check_network_headers_for_pdf(target_id)
			if history_result is True:
				self.logger.debug(
					f'[DownloadsWatchdog] PDF detected via navigation history: {self._safe_log_url(page_url)}'
				)
				snapshot = self._pdf_snapshot(
					PdfDetectionPhase.PDF,
					url=page_url,
					started_at=started_at,
					finished_at=time.time(),
					resource_url=page_url,
					detail='navigation history URL',
				)
				self._publish_pdf_detection(snapshot, serial)
				return snapshot
			if history_result is None:
				snapshot = self._pdf_snapshot(
					PdfDetectionPhase.CHECK_FAILED,
					url=page_url,
					started_at=started_at,
					finished_at=time.time(),
					error='PDF navigation history check was unavailable',
					detail='transient failure; retry on the next navigation',
				)
				self._publish_pdf_detection(snapshot, serial)
				return snapshot

			snapshot = self._pdf_snapshot(
				PdfDetectionPhase.NOT_PDF,
				url=page_url,
				started_at=started_at,
				finished_at=time.time(),
				detail='no PDF source matched',
			)
			self._publish_pdf_detection(snapshot, serial)
			return snapshot

		except asyncio.CancelledError:
			snapshot = self._pdf_snapshot(
				PdfDetectionPhase.CANCELLED,
				url=page_url,
				started_at=started_at,
				finished_at=time.time(),
				detail='PDF detection attempt was cancelled; retry on the next navigation',
			)
			self._publish_pdf_detection(snapshot, serial)
			raise

		except Exception as e:
			self.logger.warning(
				f'[DownloadsWatchdog] ❌ Error checking for PDF viewer: {self._safe_log_error(e)}'
			)
			snapshot = self._pdf_snapshot(
				PdfDetectionPhase.CHECK_FAILED,
				url=page_url,
				started_at=started_at,
				finished_at=time.time(),
				error=f'{type(e).__name__}: {e}',
				detail='PDF detection failed before a definitive result',
			)
			self._publish_pdf_detection(snapshot, serial)
			return snapshot

	async def check_for_pdf_viewer(self, target_id: TargetID, url: str | None = None) -> bool:
		"""Detect a PDF from document MIME, strict URL, viewer identity, or history."""
		snapshot = await self._detect_pdf(target_id, url=url)
		return snapshot.phase is PdfDetectionPhase.PDF

	def _check_url_for_pdf(self, url: str) -> bool:
		"""Check whether a URL is a strict PDF candidate."""
		if not url:
			return False
		try:
			parsed = urlsplit(url)
		except ValueError:
			return False
		scheme = parsed.scheme.lower()
		if scheme == 'data':
			return re.match(r'^data:application/pdf(?:[;,]|$)', url, re.IGNORECASE) is not None
		if scheme not in {'http', 'https'}:
			return False
		if (
			not parsed.netloc
			or '\\' in parsed.netloc
			or any(char.isspace() for char in parsed.netloc)
			or re.search(r'%(?![0-9A-Fa-f]{2})', parsed.netloc) is not None
		):
			return False
		hostname = parsed.hostname
		if not hostname or '%' in hostname or any(char.isspace() or ord(char) < 32 for char in hostname):
			return False
		try:
			_ = parsed.port
		except ValueError:
			return False
		if parsed.path.lower().endswith('.pdf'):
			return True
		for key, value in parse_qsl(parsed.query, keep_blank_values=True):
			if key.lower() in {'content-type', 'mimetype', 'type'} and value.lower() == 'application/pdf':
				return True
		return False

	def _is_chrome_pdf_viewer_url(self, url: str) -> bool:
		"""Check for the exact Chrome PDF viewer extension or internal page."""
		if not url:
			return False
		try:
			parsed = urlsplit(url)
		except ValueError:
			return False
		scheme = parsed.scheme.lower()
		host = parsed.netloc.lower()
		if scheme == 'chrome-extension' and host == 'mhjfbmdgcfjbbpaeojofohoefgiehjai':
			return True
		return scheme == 'chrome' and host == 'pdf-viewer'

	async def _check_network_headers_for_pdf(self, target_id: TargetID) -> bool | None:
		"""Infer PDF from navigation history; return None when the check is unavailable."""
		try:
			temp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=False)
			history = await asyncio.wait_for(
				temp_session.cdp_client.send.Page.getNavigationHistory(session_id=temp_session.session_id), timeout=3.0
			)
			entries = history.get('entries', [])
			if not entries:
				return None
			current_index = history.get('currentIndex', 0)
			if not 0 <= current_index < len(entries):
				return None
			current_url = entries[current_index].get('url', '')
			if not current_url:
				return None
			return self._check_url_for_pdf(current_url)
		except Exception as e:
			self.logger.debug(
				f'[DownloadsWatchdog] Navigation history PDF check failed: {self._safe_log_error(e)}'
			)
			return None

	async def trigger_pdf_download(
		self,
		target_id: TargetID,
		*,
		resource_url: str | None = None,
	) -> str | None:
		"""Trigger download of a PDF from Chrome's PDF viewer.

		Returns the download path if successful, None otherwise.
		"""
		attempt_id = f'pdf:{target_id}:{time.time_ns()}'
		queued_url_epoch = (
			self._pdf_url_epoch.get(resource_url, 0) if resource_url is not None else None
		)
		try:
			async with self._js_fetch_lock:
				return await self._trigger_pdf_download_unlocked(
					target_id,
					resource_url=resource_url,
					attempt_id=attempt_id,
					url_epoch=queued_url_epoch,
				)
		except asyncio.CancelledError:
			self._download_registry.cancel(
				attempt_id,
				detail='PDF download cancelled while queued',
				url=resource_url,
				source='pdf',
			)
			raise

	async def _trigger_pdf_download_unlocked(
		self,
		target_id: TargetID,
		*,
		resource_url: str | None = None,
		attempt_id: str,
		url_epoch: int | None = None,
	) -> str | None:
		self.logger.debug(f'[DownloadsWatchdog] trigger_pdf_download called for target_id={target_id}')

		if not self.browser_session.browser_profile.downloads_path:
			self._download_registry.fail(
				attempt_id,
				stage='configuration',
				error='No downloads path configured, cannot save PDF download',
				source='pdf',
			)
			self.logger.warning('[DownloadsWatchdog] ❌ No downloads path configured, cannot save PDF download')
			return None

		downloads_path = self.browser_session.browser_profile.downloads_path
		self.logger.debug(f'[DownloadsWatchdog] Downloads path: {downloads_path}')

		try:
			# Create a temporary CDP session for this target without switching focus
			self.logger.debug(f'[DownloadsWatchdog] Creating CDP session for PDF download from target {target_id}')
			temp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=False)

			# Resolve the underlying PDF resource. A Chrome viewer page is not the PDF.
			pdf_url = resource_url or ''
			if not pdf_url:
				result = await asyncio.wait_for(
					temp_session.cdp_client.send.Runtime.evaluate(
						params={
							'expression': """
					(() => {
						const embedElement = document.querySelector('embed[type="application/x-google-chrome-pdf"]') ||
											document.querySelector('embed[type="application/pdf"]');
						const params = new URLSearchParams(window.location.search);
						return {
							embed: embedElement && embedElement.src ? embedElement.src : null,
							file: params.get('file'),
							src: params.get('src')
						};
					})()
					""",
							'returnByValue': True,
						},
						session_id=temp_session.session_id,
					),
					timeout=5.0,
				)
				pdf_info = result.get('result', {}).get('value', {}) or {}
				pdf_url = str(pdf_info.get('url') or '')
				if not pdf_url:
					for key in ('embed', 'file', 'src'):
						candidate = str(pdf_info.get(key) or '')
						if not candidate or candidate.startswith('about:') or self._is_chrome_pdf_viewer_url(candidate):
							continue
						try:
							candidate_scheme = urlsplit(candidate).scheme.lower()
						except ValueError:
							continue
						if candidate_scheme in {'http', 'https', 'data'}:
							pdf_url = candidate
							break

			if not pdf_url or self._is_chrome_pdf_viewer_url(pdf_url) or pdf_url.startswith('about:'):
				self.logger.warning(f'[DownloadsWatchdog] ❌ Could not resolve the PDF resource for {target_id}')
				self._download_registry.fail(
					attempt_id,
					stage='resolve_url',
					error='Could not resolve the underlying PDF resource from the viewer',
					source='pdf',
				)
				return None
			try:
				pdf_scheme = urlsplit(pdf_url).scheme.lower()
			except ValueError:
				pdf_scheme = ''
			if pdf_scheme not in {'http', 'https', 'data'}:
				self._download_registry.fail(
					attempt_id,
					stage='resolve_url',
					error=f'Unsupported PDF resource scheme: {pdf_scheme or "unknown"}',
					url=pdf_url,
					source='pdf',
				)
				return None

			# Generate filename from URL. A data URL has no safe basename.
			if pdf_scheme == 'data':
				pdf_filename = 'document.pdf'
			else:
				pdf_filename = self._safe_filename(
					os.path.basename(pdf_url.split('?')[0]),
					fallback='document.pdf',
				)  # Remove query params and confine
				if not pdf_filename or not pdf_filename.lower().endswith('.pdf'):
					parsed = urlparse(pdf_url)
					pdf_filename = self._safe_filename(
						os.path.basename(parsed.path),
						fallback='document.pdf',
					)
					if not pdf_filename.lower().endswith('.pdf'):
						pdf_filename += '.pdf'

			self.logger.debug(f'[DownloadsWatchdog] Generated filename: {pdf_filename}')
			self._download_registry.start(
				attempt_id,
				url=pdf_url,
				file_name=pdf_filename,
				source='pdf',
			)

			# Check if already downloaded in this session.
			self.logger.debug(f'[DownloadsWatchdog] PDF_URL: {self._safe_log_url(pdf_url)}')
			existing_path, cache_error = await self._validated_pdf_cache_entry(pdf_url)
			if existing_path is not None:
				existing_size = os.path.getsize(existing_path)
				self._download_registry.complete(
					attempt_id,
					path=existing_path,
					file_size=existing_size,
					file_name=Path(existing_path).name,
					file_type='pdf',
					mime_type='application/pdf',
					from_cache=True,
					source='session_cache',
					detail='already downloaded in this session',
				)
				self.logger.debug(f'[DownloadsWatchdog] PDF already downloaded in session: {existing_path}')
				return existing_path
			if cache_error is not None:
				self.logger.warning(
					f'[DownloadsWatchdog] Cached PDF failed validation; refetching: {cache_error}'
				)

			# Generate unique filename if file exists from previous run
			downloads_dir = str(self.browser_session.browser_profile.downloads_path)
			os.makedirs(downloads_dir, exist_ok=True)
			final_filename = pdf_filename
			existing_files = os.listdir(downloads_dir)
			if pdf_filename in existing_files:
				# Generate unique name with (1), (2), etc.
				base, ext = os.path.splitext(pdf_filename)
				counter = 1
				while f'{base} ({counter}){ext}' in existing_files:
					counter += 1
				final_filename = f'{base} ({counter}){ext}'
				self.logger.debug(f'[DownloadsWatchdog] File exists, using: {final_filename}')

			if url_epoch is None:
				url_epoch = self._pdf_url_epoch.get(pdf_url, 0)
			self.logger.debug(f'[DownloadsWatchdog] Starting PDF download from: {self._safe_log_url(pdf_url)}')

			# Download using JavaScript fetch to leverage browser cache
			try:
				# Properly escape the URL to prevent JavaScript injection
				escaped_pdf_url = json.dumps(pdf_url)

				result = await asyncio.wait_for(
					temp_session.cdp_client.send.Runtime.evaluate(
						params={
							'expression': f"""
					(async () => {{
						try {{
							// Use fetch with cache: 'force-cache' to prioritize cached version
							const response = await fetch({escaped_pdf_url}, {{
								cache: 'force-cache'
							}});
							if (!response.ok) {{
								throw new Error(`HTTP error! status: ${{response.status}}`);
							}}
							const blob = await response.blob();
							const arrayBuffer = await blob.arrayBuffer();
							const uint8Array = new Uint8Array(arrayBuffer);
							
							// Check if served from cache
							const fromCache = response.headers.has('age') || 
											 !response.headers.has('date');
											 
							return {{ 
								data: Array.from(uint8Array),
								fromCache: fromCache,
								responseSize: uint8Array.length,
								contentType: response.headers.get('content-type') || '',
								transferSize: response.headers.get('content-length') || 'unknown'
							}};
						}} catch (error) {{
							throw new Error(`Fetch failed: ${{error.message}}`);
						}}
					}})()
					""",
							'awaitPromise': True,
							'returnByValue': True,
						},
						session_id=temp_session.session_id,
					),
					timeout=10.0,  # 10 second timeout for download operation
				)
				download_result = result.get('result', {}).get('value', {})

				if download_result and download_result.get('data') and len(download_result['data']) > 0:
					data_bytes = bytes(download_result['data'])
					content_type = str(download_result.get('contentType') or '')
					validation_error = await self._pdf_validation_error_async(data_bytes)
					if validation_error:
						self._download_registry.fail(
							attempt_id,
							stage='content_validation',
							error=validation_error,
							detail=f'content-type={content_type or "unknown"}',
							url=pdf_url,
							source='pdf',
						)
						self.logger.warning(
							f'[DownloadsWatchdog] ❌ Refusing non-PDF content from '
							f'{self._safe_log_url(pdf_url)}: {validation_error}'
						)
						return None
					# Ensure downloads directory exists
					downloads_dir = str(self.browser_session.browser_profile.downloads_path)
					os.makedirs(downloads_dir, exist_ok=True)
					download_path = os.path.join(downloads_dir, final_filename)

					actual_size, failure_stage, write_error = await self._write_bytes_atomically(
						download_path,
						data_bytes,
					)
					if write_error is not None:
						self._download_registry.fail(
							attempt_id,
							stage=failure_stage or 'write',
							error=write_error,
							url=pdf_url,
							source='pdf',
						)
						return None
					assert actual_size is not None
					self.logger.debug(
						f'[DownloadsWatchdog] PDF file written successfully: {download_path} ({actual_size} bytes)'
					)

					# Log cache information
					cache_status = 'from cache' if download_result.get('fromCache') else 'from network'
					response_size = actual_size
					self.logger.debug(
						f'[DownloadsWatchdog] ✅ Auto-downloaded PDF ({cache_status}, {response_size:,} bytes): {download_path}'
					)

					# Store URL->path mapping for this session
					if self._pdf_url_epoch.get(pdf_url, 0) == url_epoch:
						self._session_pdf_urls[pdf_url] = download_path
					else:
						self.logger.debug(
							f'[DownloadsWatchdog] Newer response invalidated fetched PDF; not caching '
							f'{self._safe_log_url(pdf_url)}'
						)

					# Emit file downloaded event
					self.logger.debug(f'[DownloadsWatchdog] Dispatching FileDownloadedEvent for {final_filename}')
					self.event_bus.dispatch(
						FileDownloadedEvent(
							url=pdf_url,
							path=download_path,
							file_name=final_filename,
							file_size=response_size,
							file_type='pdf',
							mime_type='application/pdf',
							from_cache=download_result.get('fromCache', False),
							auto_download=True,
						)
					)

					self._download_registry.complete(
						attempt_id,
						path=download_path,
						file_size=response_size,
						file_name=final_filename,
						file_type='pdf',
						mime_type='application/pdf',
						from_cache=download_result.get('fromCache', False),
						auto_download=True,
						url=pdf_url,
						source='pdf',
					)
					# No need to detach - session is cached
					return download_path
				else:
					self.logger.warning(
						f'[DownloadsWatchdog] No data received when downloading PDF from '
						f'{self._safe_log_url(pdf_url)}'
					)
					self._download_registry.fail(
						attempt_id,
						stage='pdf_fetch',
						error='No data received from the PDF fetch',
						url=pdf_url,
						source='pdf',
					)
					return None

			except Exception as e:
				self._download_registry.fail(
					attempt_id,
					stage='pdf_fetch',
					error=f'{type(e).__name__}: {e}',
					url=pdf_url,
					source='pdf',
				)
				self.logger.warning(
					f'[DownloadsWatchdog] Failed to auto-download PDF from '
					f'{self._safe_log_url(pdf_url)}: {self._safe_log_error(e)}'
				)
				return None

		except asyncio.CancelledError:
			self._download_registry.cancel(attempt_id, detail='PDF download cancelled')
			raise
		except TimeoutError:
			self.logger.debug('[DownloadsWatchdog] PDF download operation timed out')
			self._download_registry.fail(
				attempt_id,
				stage='pdf_timeout',
				error='PDF download operation timed out',
				source='pdf',
			)
			return None
		except Exception as e:
			self.logger.error(
				f'[DownloadsWatchdog] Error in PDF download: {self._safe_log_error(e)}'
			)
			self._download_registry.fail(
				attempt_id,
				stage='pdf',
				error=f'{type(e).__name__}: {e}',
				source='pdf',
			)
			return None

	@staticmethod
	def _sha256_file(path: str) -> str:
		digest = hashlib.sha256()
		with open(path, 'rb') as handle:
			for chunk in iter(lambda: handle.read(1024 * 1024), b''):
				digest.update(chunk)
		return digest.hexdigest()

	async def _write_bytes_atomically(
		self,
		download_path: str,
		data: bytes,
	) -> tuple[int | None, str | None, str | None]:
		"""Write and verify a same-directory temp file, then replace the destination."""
		temp_path: str | None = None
		reserved_path = False
		try:
			fd, temp_path = tempfile.mkstemp(
				dir=os.path.dirname(download_path) or '.',
				prefix='.openbrowser-download-',
				suffix='.tmp',
			)
			os.close(fd)
			async with await anyio.open_file(temp_path, 'wb') as handle:
				await handle.write(data)
			expected_hash = (await asyncio.to_thread(hashlib.sha256, data)).hexdigest()
			actual_hash = await asyncio.to_thread(self._sha256_file, temp_path)
			if actual_hash != expected_hash:
				return None, 'content_validation', 'Written bytes did not match the fetched payload'
			file_size = os.path.getsize(temp_path)
			fd = os.open(download_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
			reserved_path = True
			os.close(fd)
			os.chmod(temp_path, os.stat(download_path).st_mode & 0o777)
			os.replace(temp_path, download_path)
			temp_path = None
			reserved_path = False
			return file_size, None, None
		except FileExistsError:
			return None, 'destination_exists', f'Download destination already exists: {download_path}'
		except OSError as write_error:
			return None, 'write', f'{type(write_error).__name__}: {write_error}'
		finally:
			if temp_path is not None and os.path.exists(temp_path):
				with contextlib.suppress(OSError):
					os.unlink(temp_path)
			if reserved_path:
				with contextlib.suppress(OSError):
					os.unlink(download_path)

	@staticmethod
	def _safe_log_url(url: str) -> str:
		"""Keep useful URL context without logging inline data payloads."""
		return str(redacted_outcome({'url': url}).get('url') or '<invalid-url>')[:120]

	@staticmethod
	def _safe_log_error(error: object) -> str:
		redacted = redacted_text(f'{type(error).__name__}: {error}')
		return str(redacted)

	@staticmethod
	def _safe_filename(name: str | None, fallback: str = 'download') -> str:
		"""Confine a server-suggested filename to one safe path component."""
		if not name:
			return fallback
		candidate = name.replace('\\', '/').rsplit('/', 1)[-1]
		candidate = re.sub(r'[<>:\"|?*\x00-\x1f]', '_', candidate)
		candidate = candidate.strip().rstrip(' .')
		if candidate in {'', '.', '..'}:
			return fallback
		encoded = candidate.encode('utf-8')
		if len(encoded) > 240:
			suffix = Path(candidate).suffix[:32]
			stem_bytes = 240 - len(suffix.encode('utf-8'))
			candidate = encoded[:max(1, stem_bytes)].decode('utf-8', 'ignore') + suffix
		return candidate or fallback

	def _next_synthetic_download_id(self, prefix: str) -> str:
		"""Allocate a process-unique id for events that omit a guid."""
		self._download_serial += 1
		return f'{prefix}-{self._download_serial}'

	def _prune_ambiguous_paths(self) -> None:
		"""Release tombstoned paths that no longer exist on disk."""
		if not self._ambiguous_paths:
			return
		self._ambiguous_paths = {path for path in self._ambiguous_paths if Path(path).exists()}

	@staticmethod
	@staticmethod
	def _download_family(suggested: str) -> str:
		"""Return the base name shared by all browser duplicate suffixes."""
		stem, ext = os.path.splitext(suggested)
		while True:
			match = re.fullmatch(r'(.+) \([0-9]+\)', stem)
			if match is None:
				break
			stem = match.group(1)
		return f'{stem}{ext}'

	@staticmethod
	def _download_name_matches(candidate: str, suggested: str) -> bool:
		"""Whether a discovered file belongs to the suggested download family.

		Accepts the exact name and one or more numeric duplicate suffixes such
		as `report (1).pdf` or `report (1) (2).pdf`. Only ASCII digits count.
		"""
		if candidate == suggested:
			return True
		family = DownloadsWatchdog._download_family(suggested)
		stem, ext = os.path.splitext(family)
		pattern = rf'^{re.escape(stem)}( \([0-9]+\))+{re.escape(ext)}$'
		return re.fullmatch(pattern, candidate) is not None

	def download_outcomes(self) -> list[dict[str, object]]:
		"""Return bounded download outcomes in creation order."""
		return [redacted_outcome(outcome.to_dict()) for outcome in self._download_registry.outcomes()]

	def cancel_pending_downloads(self, *, detail: str) -> list[dict[str, object]]:
		"""Cancel every non-terminal attempt and return the final history."""
		self._download_registry.cancel_pending(detail=detail)
		return self.download_outcomes()

	async def shutdown(self) -> list[dict[str, object]]:
		"""Cancel and drain in-flight download tasks, then return final history."""
		self._downloads_admission = False
		for task in list(self._cdp_event_tasks):
			if not task.done():
				task.cancel()
		if self._cdp_event_tasks:
			await asyncio.gather(*self._cdp_event_tasks, return_exceptions=True)
		self._cdp_event_tasks.clear()
		return self.cancel_pending_downloads(detail='browser stopped')

	def latest_download_outcome(self) -> dict[str, object] | None:
		outcome = self._download_registry.latest()
		return redacted_outcome(outcome.to_dict()) if outcome is not None else None

	@staticmethod
	def _download_ref(download_id: str) -> str:
		return hashlib.sha256(download_id.encode('utf-8')).hexdigest()[:12]

	def _resolve_download_ref(self, download_ref: str) -> str | None:
		for outcome in self._download_registry.outcomes():
			if self._download_ref(outcome.download_id) == download_ref:
				return outcome.download_id
		return None

	async def wait_for_download(
		self,
		*,
		timeout: float,
		url: str | None = None,
		download_id: str | None = None,
		download_ref: str | None = None,
		lookback: float = 0.0,
	) -> dict[str, object]:
		"""Wait for one terminal download outcome and return it as JSON-safe data."""
		if download_ref is not None:
			if download_id is not None:
				raise ValueError('Pass either download_id or download_ref, not both')
			download_id = self._resolve_download_ref(download_ref)
			if download_id is None:
				raise ValueError(f'Unknown download_ref: {download_ref}')
		outcome = await self._download_registry.wait_for_terminal(
			timeout=timeout,
			url=url,
			download_id=download_id,
			lookback=lookback,
		)
		return redacted_outcome(outcome.to_dict())

	@staticmethod
	async def _get_unique_filename(directory: str, filename: str) -> str:
		"""Generate a unique filename for downloads by appending (1), (2), etc., if a file already exists."""
		base, ext = os.path.splitext(filename)
		counter = 1
		new_filename = filename
		while os.path.exists(os.path.join(directory, new_filename)):
			new_filename = f'{base} ({counter}){ext}'
			counter += 1
		return new_filename


# Fix Pydantic circular dependency - this will be called from session.py after BrowserSession is defined
