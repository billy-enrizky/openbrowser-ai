"""Downloads watchdog for monitoring and handling file downloads."""

import asyncio
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse

import anyio
from bubus import BaseEvent
from cdp_use.cdp.browser import DownloadProgressEvent, DownloadWillBeginEvent
from cdp_use.cdp.network import ResponseReceivedEvent
from cdp_use.cdp.target import SessionID, TargetID
from pydantic import PrivateAttr

from openbrowser.browser.download_state import DownloadRegistry
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
	_pdf_viewer_cache: dict[str, bool] = PrivateAttr(default_factory=dict)  # Cache PDF viewer status by target URL
	_download_cdp_session_setup: bool = PrivateAttr(default=False)  # Track if CDP session is set up
	_download_cdp_session: Any = PrivateAttr(default=None)  # Store CDP session reference
	_cdp_event_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)  # Track CDP event handler tasks
	_cdp_downloads_info: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)  # Map guid -> info
	_use_js_fetch_for_local: bool = PrivateAttr(default=False)  # Guard JS fetch path for local regular downloads
	_session_pdf_urls: dict[str, str] = PrivateAttr(default_factory=dict)  # URL -> path for PDFs downloaded this session
	_network_monitored_targets: set[str] = PrivateAttr(default_factory=set)  # Track targets with network monitoring enabled
	_detected_downloads: set[str] = PrivateAttr(default_factory=set)  # Track detected download URLs to avoid duplicates
	_network_callback_registered: bool = PrivateAttr(default=False)  # Track if global network callback is registered
	_download_registry: DownloadRegistry = PrivateAttr(default_factory=DownloadRegistry)  # Terminal outcome per download
	_download_serial: int = PrivateAttr(default=0)  # Unique synthetic id for guid-less events
	_polling_names: dict[str, int] = PrivateAttr(default_factory=dict)  # Active pollers per suggested name

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
		self._pdf_viewer_cache.clear()
		self._session_pdf_urls.clear()
		self._network_monitored_targets.clear()
		self._detected_downloads.clear()
		self._network_callback_registered = False

	async def on_NavigationCompleteEvent(self, event: NavigationCompleteEvent) -> None:
		"""Check for PDFs after navigation completes."""
		self.logger.debug(f'[DownloadsWatchdog] NavigationCompleteEvent received for {event.url}, tab #{event.target_id[-4:]}')

		# Clear PDF cache for the navigated URL since content may have changed
		if event.url in self._pdf_viewer_cache:
			del self._pdf_viewer_cache[event.url]

		# Check if auto-download is enabled
		auto_download_enabled = self._is_auto_download_enabled()
		if not auto_download_enabled:
			return

		# Note: Using network-based PDF detection that doesn't require JavaScript

		target_id = event.target_id
		self.logger.debug(f'[DownloadsWatchdog] Got target_id={target_id} for tab #{event.target_id[-4:]}')

		is_pdf = await self.check_for_pdf_viewer(target_id)
		if is_pdf:
			self.logger.debug(f'[DownloadsWatchdog] 📄 PDF detected at {event.url}, triggering auto-download...')
			download_path = await self.trigger_pdf_download(target_id)
			if not download_path:
				self.logger.warning(f'[DownloadsWatchdog] ⚠️ PDF download failed for {event.url}')

	def _is_auto_download_enabled(self) -> bool:
		"""Check if auto-download PDFs is enabled in browser profile."""
		return self.browser_session.browser_profile.auto_download_pdfs

	async def attach_to_target(self, target_id: TargetID) -> None:
		"""Set up download monitoring for a specific target."""

		# Define CDP event handlers outside of try to avoid indentation/scope issues
		def download_will_begin_handler(event: DownloadWillBeginEvent, session_id: SessionID | None) -> None:
			self.logger.debug(f'[DownloadsWatchdog] Download will begin: {event}')
			# Cache info for later completion event handling (esp. remote browsers)
			guid = event.get('guid', '')
			download_url = event.get('url', '')
			try:
				suggested_filename = event.get('suggestedFilename')
				assert suggested_filename, 'CDP DownloadWillBegin missing suggestedFilename'
				self._cdp_downloads_info[guid] = {
					'url': download_url,
					'suggested_filename': suggested_filename,
					'handled': False,
				}
			except (AssertionError, KeyError):
				suggested_filename = None
			download_id = guid or self._next_synthetic_download_id('cdp')
			if self._download_registry.active_count() >= self._download_registry.max_active:
				self._download_registry.fail(
					download_id,
					stage='capacity',
					error=f"Too many active downloads (limit {self._download_registry.max_active})",
					url=download_url,
					source='cdp',
					detail='The browser transfer was not tracked',
				)
				self.logger.warning(
					f'[DownloadsWatchdog] Download capacity reached; rejecting {download_url[:80]}'
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
			task = asyncio.create_task(self._handle_cdp_download(event, target_id, session_id, download_id))
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
				self._download_registry.fail(
					download_id,
					stage='cdp_progress',
					error='download canceled by the browser',
				)
				return
			# Check if download is complete
			if event.get('state') == 'completed':
				file_path = event.get('filePath')
				total_bytes = event.get('totalBytes')
				if self.browser_session.is_local:
					if file_path:
						self.logger.debug(f'[DownloadsWatchdog] Download completed: {file_path}')
						# Track the download
						self._track_download(file_path)
						self._download_registry.complete(
							download_id,
							path=str(file_path),
							file_size=int(total_bytes) if isinstance(total_bytes, int) else None,
							file_name=Path(file_path).name,
							source='cdp',
						)
						# Mark as handled to prevent fallback duplicate dispatch.
						# Entries are retained for the session and cleared on
						# launch or stop; the registry check is the primary guard.
						try:
							if guid in self._cdp_downloads_info:
								self._cdp_downloads_info[guid]['handled'] = True
						except (KeyError, AttributeError):
							pass
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
						self.event_bus.dispatch(
							FileDownloadedEvent(
								url=info.get('url', ''),
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
							source='cdp',
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
			self.logger.warning(f'[DownloadsWatchdog] Failed to set up CDP download listener for target {target_id}: {e}')

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
						headers = response.get('headers', {})

						# Skip non-HTTP URLs (data:, about:, chrome-extension:, etc.)
						if not url.startswith('http'):
							return

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
							self.logger.debug(f'[DownloadsWatchdog] Already detected download: {url[:80]}...')
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

						self.logger.info(f'[DownloadsWatchdog] 🔍 Detected downloadable content via network: {url[:80]}...')
						self.logger.debug(
							f'[DownloadsWatchdog]   Content-Type: {content_type}, Is PDF: {is_pdf}, Is Attachment: {is_download_attachment}'
						)

						# Trigger download asynchronously in background (don't block event handler)
						async def download_in_background():
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
									self.logger.warning(f'[DownloadsWatchdog] ⚠️  Failed to download: {url[:80]}...')
							except Exception as e:
								self.logger.error(f'[DownloadsWatchdog] Error downloading in background: {type(e).__name__}: {e}')

						# Create background task
						task = asyncio.create_task(download_in_background())
						self._cdp_event_tasks.add(task)
						task.add_done_callback(lambda t: self._cdp_event_tasks.discard(t))

					except Exception as e:
						self.logger.error(f'[DownloadsWatchdog] Error in network response handler: {type(e).__name__}: {e}')

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
			self.logger.warning(f'[DownloadsWatchdog] Failed to set up network monitoring for target {target_id}: {e}')

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

		# Check if already downloaded in this session
		if url in self._session_pdf_urls:
			existing_path = self._session_pdf_urls[url]
			self._download_registry.complete(
				download_id,
				path=existing_path,
				file_name=Path(existing_path).name,
				source='session_cache',
				detail='already downloaded in this session',
			)
			self.logger.debug(f'[DownloadsWatchdog] File already downloaded in session: {existing_path}')
			return existing_path

		try:
			# Get or create CDP session for this target
			temp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=False)

			# Determine filename
			if suggested_filename:
				filename = suggested_filename
			else:
				# Extract from URL
				filename = os.path.basename(url.split('?')[0])  # Remove query params
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

			self.logger.debug(f'[DownloadsWatchdog] Downloading from: {url[:100]}...')

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
				download_path = os.path.join(downloads_dir, final_filename)

				# Save the file asynchronously
				async with await anyio.open_file(download_path, 'wb') as f:
					await f.write(bytes(download_result['data']))

				# Verify file was written successfully
				if os.path.exists(download_path):
					actual_size = os.path.getsize(download_path)
					self.logger.debug(f'[DownloadsWatchdog] File written: {download_path} ({actual_size} bytes)')

					# Determine file type
					file_ext = Path(final_filename).suffix.lower().lstrip('.')
					mime_type = content_type or f'application/{file_ext}'

					# Store URL->path mapping for this session
					self._session_pdf_urls[url] = download_path

					# Emit file downloaded event
					self.logger.debug(f'[DownloadsWatchdog] Dispatching FileDownloadedEvent for {final_filename}')
					self.event_bus.dispatch(
						FileDownloadedEvent(
							url=url,
							path=download_path,
							file_name=final_filename,
							file_size=actual_size,
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
					self.logger.error(f'[DownloadsWatchdog] Failed to write file: {download_path}')
					self._download_registry.fail(
						download_id,
						stage='write',
						error=f'Failed to write file: {download_path}',
						url=url,
						source='js_fetch',
					)
					return None
			else:
				self.logger.warning(f'[DownloadsWatchdog] No data received when downloading from {url}')
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
			self.logger.warning(f'[DownloadsWatchdog] Download timed out: {url[:80]}...')
			self._download_registry.fail(
				download_id,
				stage='fetch_timeout',
				error='Browser fetch timed out after 15s',
				url=url,
				source='js_fetch',
			)
			return None
		except Exception as e:
			self.logger.warning(f'[DownloadsWatchdog] Download failed: {type(e).__name__}: {e}')
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
			self.logger.error(f'[DownloadsWatchdog] Error tracking download: {e}')

	async def _handle_cdp_download(
		self,
		event: DownloadWillBeginEvent,
		target_id: TargetID,
		session_id: SessionID | None,
		download_id: str | None = None,
	) -> None:
		"""Handle a CDP Page.downloadWillBegin event."""
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
		suggested_filename = event.get('suggestedFilename', 'download')
		guid = event.get('guid', '')
		if download_id is None:
			download_id = guid or self._next_synthetic_download_id('cdp')

		try:
			self.logger.debug(f'[DownloadsWatchdog] ⬇️ File download starting: {suggested_filename} from {download_url[:100]}...')
			self.logger.debug(f'[DownloadsWatchdog] Full CDP event: {event}')

			# Since Browser.setDownloadBehavior is already configured, the browser will download the file
			# We just need to wait for it to appear in the downloads directory
			expected_path = downloads_dir / suggested_filename

			# Debug: List current directory contents
			self.logger.debug(f'[DownloadsWatchdog] Downloads directory: {downloads_dir}')
			if downloads_dir.exists():
				files_before = list(downloads_dir.iterdir())
				self.logger.debug(f'[DownloadsWatchdog] Files before download: {[f.name for f in files_before]}')

			# Try manual JavaScript fetch as a fallback for local browsers (disabled for regular local downloads)
			if self.browser_session.is_local and self._use_js_fetch_for_local:
				self.logger.debug(f'[DownloadsWatchdog] Attempting JS fetch fallback for {download_url}')

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
						file_size = len(file_data)

						# Ensure unique filename
						unique_filename = await self._get_unique_filename(str(downloads_dir), suggested_filename)
						final_path = downloads_dir / unique_filename

						# Write the file
						import anyio

						async with await anyio.open_file(final_path, 'wb') as f:
							await f.write(file_data)

						self.logger.debug(f'[DownloadsWatchdog] ✅ Downloaded and saved file: {final_path} ({file_size} bytes)')
						expected_path = final_path
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
					self.logger.error(f'[DownloadsWatchdog] ❌ Failed to download file via fetch: {fetch_error}')

			# For remote browsers, don't poll local filesystem; downloadProgress handler will emit the event
			if not self.browser_session.is_local:
				return
		except Exception as e:
			self.logger.error(f'[DownloadsWatchdog] ❌ Error handling CDP download: {type(e).__name__} {e}')

		# If we reach here, the fetch method failed, so wait for native download
		await self._poll_for_download_file(
			download_id=download_id,
			download_url=download_url,
			downloads_dir=downloads_dir,
			suggested_filename=suggested_filename,
			guid=guid,
		)

	async def _poll_for_download_file(
		self,
		*,
		download_id: str,
		download_url: str,
		downloads_dir: Path,
		suggested_filename: str,
		guid: str,
	) -> None:
		"""Poll for a name-matched file and complete this attempt at most once."""
		family = self._download_family(suggested_filename)
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
				try:
					await asyncio.sleep(5.0)  # Check every 5 seconds
				except asyncio.CancelledError:
					self._download_registry.cancel(download_id, detail='download polling cancelled')
					raise

				# With two attempts suggesting the same name, a name match cannot
				# prove ownership. Leave both to the CDP progress event.
				if self._polling_names.get(family, 0) > 1:
					continue

				if Path(downloads_dir).exists():
					for file_path in Path(downloads_dir).iterdir():
						if (
							file_path.is_file()
							and not file_path.name.startswith('.')
							and file_path.name not in initial_files
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
									self._download_registry.complete(
										download_id,
										path=str(file_path),
										file_size=file_size,
										file_name=file_path.name,
										file_type=file_type,
										source='native_poll',
									)
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
								self.logger.debug(f'[DownloadsWatchdog] Error checking file {file_path}: {e}')

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
		self.logger.debug(f'[DownloadsWatchdog] ⬇️ Handling download: {download.suggested_filename} from {download.url[:100]}...')

		try:
			# Debug: Check if download is already being handled elsewhere
			failure = (
				await download.failure()
			)  # TODO: it always fails for some reason, figure out why connect_over_cdp makes accept_downloads not work
			self.logger.warning(f'[DownloadsWatchdog] ❌ Download state - canceled: {failure}, url: {download.url}')
			current_step = 'getting_download_info'
			# Get download info immediately
			url = download.url
			suggested_filename = download.suggested_filename

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

				self.logger.debug(f'[DownloadsWatchdog] Download started: {unique_filename} from {url[:100]}...')

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
				f'[DownloadsWatchdog] Download state - URL: {download.url}, filename: {download.suggested_filename}'
			)
		finally:
			# Clean up tracking
			if download_id in self._active_downloads:
				del self._active_downloads[download_id]

	async def check_for_pdf_viewer(self, target_id: TargetID) -> bool:
		"""Check if the current target is a PDF using network-based detection.

		This method avoids JavaScript execution that can crash WebSocket connections.
		Returns True if a PDF is detected and should be downloaded.
		"""
		self.logger.debug(f'[DownloadsWatchdog] Checking if target {target_id} is PDF viewer...')

		# Get target info to get URL
		cdp_client = self.browser_session.cdp_client
		targets = await cdp_client.send.Target.getTargets()
		target_info = next((t for t in targets['targetInfos'] if t['targetId'] == target_id), None)
		if not target_info:
			self.logger.warning(f'[DownloadsWatchdog] No target info found for {target_id}')
			return False

		page_url = target_info.get('url', '')

		# Check cache first
		if page_url in self._pdf_viewer_cache:
			cached_result = self._pdf_viewer_cache[page_url]
			self.logger.debug(f'[DownloadsWatchdog] Using cached PDF check result for {page_url}: {cached_result}')
			return cached_result

		try:
			# Method 1: Check URL patterns (fastest, most reliable)
			url_is_pdf = self._check_url_for_pdf(page_url)
			if url_is_pdf:
				self.logger.debug(f'[DownloadsWatchdog] PDF detected via URL pattern: {page_url}')
				self._pdf_viewer_cache[page_url] = True
				return True

			# Method 2: Check network response headers via CDP (safer than JavaScript)
			header_is_pdf = await self._check_network_headers_for_pdf(target_id)
			if header_is_pdf:
				self.logger.debug(f'[DownloadsWatchdog] PDF detected via network headers: {page_url}')
				self._pdf_viewer_cache[page_url] = True
				return True

			# Method 3: Check Chrome's PDF viewer specific URLs
			chrome_pdf_viewer = self._is_chrome_pdf_viewer_url(page_url)
			if chrome_pdf_viewer:
				self.logger.debug(f'[DownloadsWatchdog] Chrome PDF viewer detected: {page_url}')
				self._pdf_viewer_cache[page_url] = True
				return True

			# Not a PDF
			self._pdf_viewer_cache[page_url] = False
			return False

		except Exception as e:
			self.logger.warning(f'[DownloadsWatchdog] ❌ Error checking for PDF viewer: {e}')
			self._pdf_viewer_cache[page_url] = False
			return False

	def _check_url_for_pdf(self, url: str) -> bool:
		"""Check if URL indicates a PDF file."""
		if not url:
			return False

		url_lower = url.lower()

		# Direct PDF file extensions
		if url_lower.endswith('.pdf'):
			return True

		# PDF in path
		if '.pdf' in url_lower:
			return True

		# PDF MIME type in URL parameters
		if any(
			param in url_lower
			for param in [
				'content-type=application/pdf',
				'content-type=application%2fpdf',
				'mimetype=application/pdf',
				'type=application/pdf',
			]
		):
			return True

		return False

	def _is_chrome_pdf_viewer_url(self, url: str) -> bool:
		"""Check if this is Chrome's internal PDF viewer URL."""
		if not url:
			return False

		url_lower = url.lower()

		# Chrome PDF viewer uses chrome-extension:// URLs
		if 'chrome-extension://' in url_lower and 'pdf' in url_lower:
			return True

		# Chrome PDF viewer internal URLs
		if url_lower.startswith('chrome://') and 'pdf' in url_lower:
			return True

		return False

	async def _check_network_headers_for_pdf(self, target_id: TargetID) -> bool:
		"""Infer PDF via navigation history/URL; headers are not available post-navigation in this context."""
		try:
			# Get CDP session
			temp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=False)

			# Get navigation history to find the main resource
			history = await asyncio.wait_for(
				temp_session.cdp_client.send.Page.getNavigationHistory(session_id=temp_session.session_id), timeout=3.0
			)

			current_entry = history.get('entries', [])
			if current_entry:
				current_index = history.get('currentIndex', 0)
				if 0 <= current_index < len(current_entry):
					current_url = current_entry[current_index].get('url', '')

					# Check if the URL itself suggests PDF
					if self._check_url_for_pdf(current_url):
						return True

			# Note: CDP doesn't easily expose response headers for completed navigations
			# For more complex cases, we'd need to set up Network.responseReceived listeners
			# before navigation, but that's overkill for most PDF detection cases

			return False

		except Exception as e:
			self.logger.debug(f'[DownloadsWatchdog] Network headers check failed (non-critical): {e}')
			return False

	async def trigger_pdf_download(self, target_id: TargetID) -> str | None:
		"""Trigger download of a PDF from Chrome's PDF viewer.

		Returns the download path if successful, None otherwise.
		"""
		self.logger.debug(f'[DownloadsWatchdog] trigger_pdf_download called for target_id={target_id}')

		attempt_id = f'pdf:{target_id}:{time.time_ns()}'
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

			# Try to get the PDF URL with timeout
			result = await asyncio.wait_for(
				temp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': """
				(() => {
					// For Chrome's PDF viewer, the actual URL is in window.location.href
					// The embed element's src is often "about:blank"
					const embedElement = document.querySelector('embed[type="application/x-google-chrome-pdf"]') ||
										document.querySelector('embed[type="application/pdf"]');
					if (embedElement) {
						// Chrome PDF viewer detected - use the page URL
						return { url: window.location.href };
					}
					// Fallback to window.location.href anyway
					return { url: window.location.href };
				})()
				""",
						'returnByValue': True,
					},
					session_id=temp_session.session_id,
				),
				timeout=5.0,  # 5 second timeout to prevent hanging
			)
			pdf_info = result.get('result', {}).get('value', {})

			pdf_url = pdf_info.get('url', '')
			if not pdf_url:
				self.logger.warning(f'[DownloadsWatchdog] ❌ Could not determine PDF URL for download {pdf_info}')
				self._download_registry.fail(
					attempt_id,
					stage='pdf_probe',
					error='Could not determine the PDF URL from the viewer',
					source='pdf',
				)
				return None

			# Generate filename from URL
			pdf_filename = os.path.basename(pdf_url.split('?')[0])  # Remove query params
			if not pdf_filename or not pdf_filename.endswith('.pdf'):
				parsed = urlparse(pdf_url)
				pdf_filename = os.path.basename(parsed.path) or 'document.pdf'
				if not pdf_filename.endswith('.pdf'):
					pdf_filename += '.pdf'

			self.logger.debug(f'[DownloadsWatchdog] Generated filename: {pdf_filename}')
			self._download_registry.start(
				attempt_id,
				url=pdf_url,
				file_name=pdf_filename,
				source='pdf',
			)

			# Check if already downloaded in this session
			self.logger.debug(f'[DownloadsWatchdog] PDF_URL: {pdf_url}, session_pdf_urls: {self._session_pdf_urls}')
			if pdf_url in self._session_pdf_urls:
				existing_path = self._session_pdf_urls[pdf_url]
				self._download_registry.complete(
					attempt_id,
					path=existing_path,
					file_name=Path(existing_path).name,
					file_type='pdf',
					mime_type='application/pdf',
					source='session_cache',
					detail='already downloaded in this session',
				)
				self.logger.debug(f'[DownloadsWatchdog] PDF already downloaded in session: {existing_path}')
				return existing_path

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

			self.logger.debug(f'[DownloadsWatchdog] Starting PDF download from: {pdf_url[:100]}...')

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
					# Ensure downloads directory exists
					downloads_dir = str(self.browser_session.browser_profile.downloads_path)
					os.makedirs(downloads_dir, exist_ok=True)
					download_path = os.path.join(downloads_dir, final_filename)

					# Save the PDF asynchronously
					async with await anyio.open_file(download_path, 'wb') as f:
						await f.write(bytes(download_result['data']))

					# Verify file was written successfully
					if os.path.exists(download_path):
						actual_size = os.path.getsize(download_path)
						self.logger.debug(
							f'[DownloadsWatchdog] PDF file written successfully: {download_path} ({actual_size} bytes)'
						)
					else:
						self.logger.error(f'[DownloadsWatchdog] ❌ Failed to write PDF file to: {download_path}')
						self._download_registry.fail(
							attempt_id,
							stage='write',
							error=f'Failed to write PDF file to: {download_path}',
							url=pdf_url,
							source='pdf',
						)
						return None

					# Log cache information
					cache_status = 'from cache' if download_result.get('fromCache') else 'from network'
					response_size = download_result.get('responseSize', 0)
					self.logger.debug(
						f'[DownloadsWatchdog] ✅ Auto-downloaded PDF ({cache_status}, {response_size:,} bytes): {download_path}'
					)

					# Store URL->path mapping for this session
					self._session_pdf_urls[pdf_url] = download_path

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
					self.logger.warning(f'[DownloadsWatchdog] No data received when downloading PDF from {pdf_url}')
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
				self.logger.warning(f'[DownloadsWatchdog] Failed to auto-download PDF from {pdf_url}: {type(e).__name__}: {e}')
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
			self.logger.error(f'[DownloadsWatchdog] Error in PDF download: {type(e).__name__}: {e}')
			self._download_registry.fail(
				attempt_id,
				stage='pdf',
				error=f'{type(e).__name__}: {e}',
				source='pdf',
			)
			return None

	def _next_synthetic_download_id(self, prefix: str) -> str:
		"""Allocate a process-unique id for events that omit a guid."""
		self._download_serial += 1
		return f'{prefix}-{self._download_serial}'

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
		return [outcome.to_dict() for outcome in self._download_registry.outcomes()]

	def cancel_pending_downloads(self, *, detail: str) -> list[dict[str, object]]:
		"""Cancel every non-terminal attempt and return the final history."""
		self._download_registry.cancel_pending(detail=detail)
		return self.download_outcomes()

	async def shutdown(self) -> list[dict[str, object]]:
		"""Cancel and drain in-flight download tasks, then return final history."""
		for task in list(self._cdp_event_tasks):
			if not task.done():
				task.cancel()
		if self._cdp_event_tasks:
			await asyncio.gather(*self._cdp_event_tasks, return_exceptions=True)
		self._cdp_event_tasks.clear()
		return self.cancel_pending_downloads(detail='browser stopped')

	def latest_download_outcome(self) -> dict[str, object] | None:
		outcome = self._download_registry.latest()
		return outcome.to_dict() if outcome is not None else None

	async def wait_for_download(
		self,
		*,
		timeout: float,
		url: str | None = None,
		download_id: str | None = None,
		lookback: float = 0.0,
	) -> dict[str, object]:
		"""Wait for one terminal download outcome and return it as JSON-safe data."""
		outcome = await self._download_registry.wait_for_terminal(
			timeout=timeout,
			url=url,
			download_id=download_id,
			lookback=lookback,
		)
		return outcome.to_dict()

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
