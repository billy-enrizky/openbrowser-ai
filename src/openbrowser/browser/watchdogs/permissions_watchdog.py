"""Permissions watchdog for granting browser permissions on connection."""

import logging
from typing import Any, ClassVar

from bubus import BaseEvent

from src.openbrowser.browser.events import BrowserConnectedEvent
from src.openbrowser.browser.watchdogs.base import BaseWatchdog

logger = logging.getLogger(__name__)


class PermissionsWatchdog(BaseWatchdog):
    """Grants browser permissions when browser connects."""

    # Events this watchdog listens to
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
        BrowserConnectedEvent,
    ]

    # Events this watchdog emits
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

    def attach_to_session(self) -> None:
        """Register event handlers."""
        self.event_bus.on(BrowserConnectedEvent, self.on_BrowserConnectedEvent)

    async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
        """Grant permissions when browser connects."""
        permissions = self.browser_session.browser_profile.permissions

        if not permissions:
            self.logger.debug('[PermissionsWatchdog] No permissions to grant')
            return

        self.logger.debug(f'[PermissionsWatchdog] Granting browser permissions: {permissions}')

        try:
            # Grant permissions using CDP Browser.grantPermissions
            # origin=None means grant to all origins
            # Browser domain commands don't use session_id
            await self.browser_session.cdp_client.send.Browser.grantPermissions(
                params={'permissions': permissions}  # type: ignore
            )
            self.logger.debug(f'[PermissionsWatchdog] Successfully granted permissions: {permissions}')
        except Exception as e:
            self.logger.error(f'[PermissionsWatchdog] Failed to grant permissions: {str(e)}')
            # Don't raise - permissions are not critical to browser operation

