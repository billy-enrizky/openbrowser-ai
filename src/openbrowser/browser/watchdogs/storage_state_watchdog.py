"""Storage state watchdog for managing browser cookies and storage persistence.

This module provides the StorageStateWatchdog which persists cookies and
localStorage to a JSON file, enabling session persistence across browser
restarts.

Classes:
    StorageStateWatchdog: Monitors and persists browser storage state.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from bubus import BaseEvent
from pydantic import Field, PrivateAttr

from src.openbrowser.browser.events import (
    BrowserConnectedEvent,
    BrowserStopEvent,
    LoadStorageStateEvent,
    SaveStorageStateEvent,
    StorageStateLoadedEvent,
    StorageStateSavedEvent,
)
from src.openbrowser.browser.watchdogs.base import BaseWatchdog

logger = logging.getLogger(__name__)


class StorageStateWatchdog(BaseWatchdog):
    """Monitors and persists browser storage state including cookies and localStorage.

    Automatically loads storage state on browser connect and saves on stop.
    Optionally monitors for cookie changes and auto-saves periodically.

    Listens to:
        BrowserConnectedEvent: Loads storage state and starts monitoring.
        BrowserStopEvent: Saves final state and stops monitoring.
        SaveStorageStateEvent: Manual save trigger.
        LoadStorageStateEvent: Manual load trigger.

    Emits:
        StorageStateSavedEvent: After successful save.
        StorageStateLoadedEvent: After successful load.

    Configuration (in BrowserProfile):
        storage_state: Path to storage state JSON file.

    Attributes:
        auto_save_interval: Seconds between auto-saves (default: 30).
        save_on_change: Save immediately when cookies change (default: True).

    Example:
        >>> profile = BrowserProfile(
        ...     storage_state='./auth_state.json'
        ... )
    """

    # Events this watchdog listens to
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
        BrowserConnectedEvent,
        BrowserStopEvent,
        SaveStorageStateEvent,
        LoadStorageStateEvent,
    ]

    # Events this watchdog emits
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = [
        StorageStateSavedEvent,
        StorageStateLoadedEvent,
    ]

    # Configuration
    auto_save_interval: float = Field(default=30.0)  # Auto-save every 30 seconds
    save_on_change: bool = Field(default=True)  # Save immediately when cookies change

    # Private state
    _monitoring_task: asyncio.Task | None = PrivateAttr(default=None)
    _last_cookie_state: list[dict] = PrivateAttr(default_factory=list)
    _save_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to browser lifecycle and storage events.
        """
        self.event_bus.on(BrowserConnectedEvent, self.on_BrowserConnectedEvent)
        self.event_bus.on(BrowserStopEvent, self.on_BrowserStopEvent)
        self.event_bus.on(SaveStorageStateEvent, self.on_SaveStorageStateEvent)
        self.event_bus.on(LoadStorageStateEvent, self.on_LoadStorageStateEvent)

    async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
        """Start monitoring when browser starts.

        Initializes storage monitoring and loads existing storage state.

        Args:
            event: BrowserConnectedEvent from session.
        """
        self.logger.debug('[StorageStateWatchdog] Initializing auth/cookies sync with storage_state.json file')

        # Start monitoring
        await self._start_monitoring()

        # Automatically load storage state after browser start
        await self.event_bus.dispatch(LoadStorageStateEvent())

    async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
        """Stop monitoring when browser stops.

        Stops the monitoring task and saves final state.

        Args:
            event: BrowserStopEvent from session.
        """
        self.logger.debug('[StorageStateWatchdog] Stopping storage_state monitoring')
        await self._stop_monitoring()

    async def on_SaveStorageStateEvent(self, event: SaveStorageStateEvent) -> None:
        """Handle storage state save request.

        Saves current cookies and storage to the specified path
        or the default path from browser_profile.

        Args:
            event: SaveStorageStateEvent with optional path override.
        """
        path = event.path
        if path is None:
            if self.browser_session.browser_profile.storage_state:
                path = str(self.browser_session.browser_profile.storage_state)
        await self._save_storage_state(path)

    async def on_LoadStorageStateEvent(self, event: LoadStorageStateEvent) -> None:
        """Handle storage state load request.

        Loads cookies and storage from the specified path or
        the default path from browser_profile.

        Args:
            event: LoadStorageStateEvent with optional path override.
        """
        path = event.path
        if path is None:
            if self.browser_session.browser_profile.storage_state:
                path = str(self.browser_session.browser_profile.storage_state)
        await self._load_storage_state(path)

    async def _start_monitoring(self) -> None:
        """Start the monitoring task.

        Creates async task for periodic storage change monitoring.
        """
        if self._monitoring_task and not self._monitoring_task.done():
            return

        self._monitoring_task = asyncio.create_task(self._monitor_storage_changes())

    async def _stop_monitoring(self) -> None:
        """Stop the monitoring task.

        Cancels the async monitoring task if running.
        """
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

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
        if not self.browser_session._cdp_client_root:
            return False

        try:
            # Get current cookies using CDP
            current_cookies = await self.browser_session._cdp_get_cookies()

            # Convert to comparable format
            current_cookie_set = {
                (c.get('name', ''), c.get('domain', ''), c.get('path', '')): c.get('value', '')
                for c in current_cookies
            }

            last_cookie_set = {
                (c.get('name', ''), c.get('domain', ''), c.get('path', '')): c.get('value', '')
                for c in self._last_cookie_state
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
                return

            # Skip saving if the storage state is already a dict
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
        if not self.browser_session._cdp_client_root:
            self.logger.warning('[StorageStateWatchdog] No CDP client available for loading')
            return

        load_path = path or self.browser_session.browser_profile.storage_state
        if not load_path or not os.path.exists(str(load_path)):
            return

        try:
            # Read the storage state file
            content = Path(load_path).read_text()
            storage = json.loads(content)

            # Apply cookies if present
            if 'cookies' in storage and storage['cookies']:
                await self.browser_session._cdp_set_cookies(storage['cookies'])
                self._last_cookie_state = storage['cookies'].copy()

                self.logger.debug(
                    f'[StorageStateWatchdog] Loaded {len(storage["cookies"])} cookies from {load_path}'
                )

                # Emit success event
                self.event_bus.dispatch(
                    StorageStateLoadedEvent(
                        path=str(load_path),
                        cookies_count=len(storage.get('cookies', [])),
                        origins_count=len(storage.get('origins', [])),
                    )
                )

        except Exception as e:
            self.logger.error(f'[StorageStateWatchdog] Failed to load storage state: {e}')

    def _merge_storage_states(self, existing: dict, new: dict) -> dict:
        """Merge existing storage state with new state.
        
        Args:
            existing: Existing storage state
            new: New storage state
            
        Returns:
            Merged storage state
        """
        merged = existing.copy()

        # Merge cookies (new cookies override existing ones with same name/domain/path)
        existing_cookies = {(c.get('name'), c.get('domain'), c.get('path')): c for c in existing.get('cookies', [])}
        new_cookies = {(c.get('name'), c.get('domain'), c.get('path')): c for c in new.get('cookies', [])}

        existing_cookies.update(new_cookies)
        merged['cookies'] = list(existing_cookies.values())

        # Merge origins
        existing_origins = {o.get('origin'): o for o in existing.get('origins', [])}
        new_origins = {o.get('origin'): o for o in new.get('origins', [])}

        existing_origins.update(new_origins)
        merged['origins'] = list(existing_origins.values())

        return merged

