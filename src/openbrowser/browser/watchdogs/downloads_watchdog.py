"""Downloads watchdog for monitoring and handling file downloads.

This module provides the DownloadsWatchdog which monitors CDP download
events and emits FileDownloadedEvent when downloads complete.

Classes:
    DownloadsWatchdog: Monitors downloads and emits completion events.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar

from bubus import BaseEvent
from cdp_use.cdp.browser import DownloadProgressEvent, DownloadWillBeginEvent
from cdp_use.cdp.target import SessionID, TargetID
from pydantic import PrivateAttr

from src.openbrowser.browser.events import (
    BrowserConnectedEvent,
    BrowserStoppedEvent,
    FileDownloadedEvent,
    NavigationCompleteEvent,
    TabClosedEvent,
    TabCreatedEvent,
)
from src.openbrowser.browser.watchdogs.base import BaseWatchdog

logger = logging.getLogger(__name__)


class DownloadsWatchdog(BaseWatchdog):
    """Monitors downloads and handles file download events.

    Sets up CDP download behavior and listens for download events.
    Tracks download progress and emits FileDownloadedEvent on completion.

    Listens to:
        BrowserConnectedEvent: Sets up download listeners.
        BrowserStoppedEvent: Cleans up download state.
        TabCreatedEvent: Monitors new tabs for downloads.
        TabClosedEvent: Handles closed tab cleanup.
        NavigationCompleteEvent: Checks for downloads after navigation.

    Emits:
        FileDownloadedEvent: When a file download completes.

    Note:
        Requires downloads_path to be configured in BrowserProfile.
    """

    # Events this watchdog listens to
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
        BrowserConnectedEvent,
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
    _download_cdp_session_setup: bool = PrivateAttr(default=False)
    _cdp_downloads_info: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _cdp_event_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to browser lifecycle and tab events for
        download monitoring.
        """
        self.event_bus.on(BrowserConnectedEvent, self.on_BrowserConnectedEvent)
        self.event_bus.on(BrowserStoppedEvent, self.on_BrowserStoppedEvent)
        self.event_bus.on(TabCreatedEvent, self.on_TabCreatedEvent)
        self.event_bus.on(TabClosedEvent, self.on_TabClosedEvent)
        self.event_bus.on(NavigationCompleteEvent, self.on_NavigationCompleteEvent)

    async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
        """Set up download monitoring when browser connects.

        Initializes CDP download listeners and behavior.

        Args:
            event: BrowserConnectedEvent from session.
        """
        await self._setup_download_listeners()

    async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
        """Monitor new tabs for downloads.

        Ensures download listeners are set up for new tabs.

        Args:
            event: TabCreatedEvent with target info.
        """
        if event.target_id:
            await self._setup_download_listeners()

    async def on_TabClosedEvent(self, event: TabClosedEvent) -> None:
        """Stop monitoring closed tabs.

        No cleanup needed as browser context handles target lifecycle.

        Args:
            event: TabClosedEvent with target info.
        """
        pass  # No cleanup needed, browser context handles target lifecycle

    async def on_BrowserStoppedEvent(self, event: BrowserStoppedEvent) -> None:
        """Clean up when browser stops.

        Cancels pending CDP event tasks and clears download state.

        Args:
            event: BrowserStoppedEvent from session.
        """
        # Cancel all CDP event handler tasks
        for task in list(self._cdp_event_tasks):
            if not task.done():
                task.cancel()
        # Wait for all tasks to complete cancellation
        if self._cdp_event_tasks:
            await asyncio.gather(*self._cdp_event_tasks, return_exceptions=True)
        self._cdp_event_tasks.clear()

        # Clear state
        self._download_cdp_session_setup = False
        self._cdp_downloads_info.clear()

    async def on_NavigationCompleteEvent(self, event: NavigationCompleteEvent) -> None:
        """Check for downloads after navigation completes.

        Downloads are handled via CDP events, no action needed here.

        Args:
            event: NavigationCompleteEvent with URL info.
        """
        # Downloads are handled via CDP events, no action needed here
        pass

    async def _setup_download_listeners(self) -> None:
        """Set up CDP download listeners.

        Configures Browser.setDownloadBehavior and registers handlers
        for DownloadWillBegin and DownloadProgress events.
        """
        if self._download_cdp_session_setup:
            return

        downloads_path = self.browser_session.browser_profile.downloads_path
        if not downloads_path:
            self.logger.debug('[DownloadsWatchdog] No downloads path configured, skipping setup')
            return

        try:
            cdp_client = self.browser_session.cdp_client

            # Ensure downloads directory exists
            expanded_downloads_path = Path(downloads_path).expanduser().resolve()
            expanded_downloads_path.mkdir(parents=True, exist_ok=True)

            # Set download behavior to allow downloads and enable events
            await cdp_client.send.Browser.setDownloadBehavior(
                params={
                    'behavior': 'allow',
                    'downloadPath': str(expanded_downloads_path),
                    'eventsEnabled': True,
                }
            )

            # Define CDP event handlers
            def download_will_begin_handler(event: DownloadWillBeginEvent, session_id: SessionID | None) -> None:
                guid = event.get('guid', '')
                try:
                    suggested_filename = event.get('suggestedFilename', 'download')
                    self._cdp_downloads_info[guid] = {
                        'url': event.get('url', ''),
                        'suggested_filename': suggested_filename,
                        'handled': False,
                    }
                except Exception as e:
                    self.logger.debug(f'[DownloadsWatchdog] Error in download_will_begin_handler: {e}')

                # Create task to handle download
                task = asyncio.create_task(self._handle_cdp_download(event, session_id))
                self._cdp_event_tasks.add(task)
                task.add_done_callback(lambda t: self._cdp_event_tasks.discard(t))

            def download_progress_handler(event: DownloadProgressEvent, session_id: SessionID | None) -> None:
                # Check if download is complete
                if event.get('state') == 'completed':
                    file_path = event.get('filePath')
                    guid = event.get('guid', '')
                    
                    if file_path:
                        self.logger.debug(f'[DownloadsWatchdog] Download completed: {file_path}')
                        self._track_download(file_path, guid)
                    else:
                        # Fallback: use suggested filename from download_will_begin
                        info = self._cdp_downloads_info.get(guid, {})
                        suggested_filename = info.get('suggested_filename', 'download')
                        downloads_path = self.browser_session.browser_profile.downloads_path
                        if downloads_path:
                            effective_path = str(Path(downloads_path) / suggested_filename)
                            self._track_download(effective_path, guid)

                    # Clean up
                    if guid in self._cdp_downloads_info:
                        del self._cdp_downloads_info[guid]

            # Register the handlers with CDP
            cdp_client.register.Browser.downloadWillBegin(download_will_begin_handler)  # type: ignore[arg-type]
            cdp_client.register.Browser.downloadProgress(download_progress_handler)  # type: ignore[arg-type]

            self._download_cdp_session_setup = True
            self.logger.debug('[DownloadsWatchdog] Set up CDP download listeners')

        except Exception as e:
            self.logger.warning(f'[DownloadsWatchdog] Failed to set up CDP download listener: {e}')

    async def _handle_cdp_download(
        self, event: DownloadWillBeginEvent, session_id: SessionID | None
    ) -> None:
        """Handle a CDP download event."""
        guid = event.get('guid', '')
        url = event.get('url', '')
        suggested_filename = event.get('suggestedFilename', 'download')

        self.logger.debug(f'[DownloadsWatchdog] Download will begin: {suggested_filename} from {url}')

        # Download completion is handled in download_progress_handler
        # This method can be extended for additional download handling logic

    def _track_download(self, file_path: str, guid: str) -> None:
        """Track a completed download and dispatch FileDownloadedEvent."""
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                self.logger.warning(f'[DownloadsWatchdog] Download file not found: {file_path}')
                return

            file_name = path_obj.name
            file_size = path_obj.stat().st_size
            file_ext = path_obj.suffix.lower().lstrip('.')

            info = self._cdp_downloads_info.get(guid, {})
            url = info.get('url', '')

            # Dispatch event
            self.event_bus.dispatch(
                FileDownloadedEvent(
                    url=url,
                    path=str(file_path),
                    file_name=file_name,
                    file_size=file_size,
                    file_type=file_ext if file_ext else None,
                )
            )

            self.logger.info(f'[DownloadsWatchdog] Tracked download: {file_name} ({file_size} bytes)')

        except Exception as e:
            self.logger.error(f'[DownloadsWatchdog] Error tracking download: {e}')

