"""Permissions watchdog for granting browser permissions on connection.

This module provides the PermissionsWatchdog which grants configured
browser permissions (geolocation, camera, etc.) when the browser connects.

Classes:
    PermissionsWatchdog: Grants browser permissions on connection.
"""

import logging
from typing import Any, ClassVar

from bubus import BaseEvent

from openbrowser.browser.events import BrowserConnectedEvent
from openbrowser.browser.watchdogs.base import BaseWatchdog

logger = logging.getLogger(__name__)


class PermissionsWatchdog(BaseWatchdog):
    """Grants browser permissions when browser connects.

    Reads permissions from BrowserProfile and grants them globally
    using CDP Browser.grantPermissions on connection.

    Listens to:
        BrowserConnectedEvent: Triggers permission granting.

    Supported Permissions:
        geolocation, notifications, camera, microphone, midi,
        background-sync, accelerometer, gyroscope, magnetometer, etc.

    Example:
        >>> # In BrowserProfile:
        >>> profile = BrowserProfile(
        ...     permissions=['geolocation', 'notifications']
        ... )
    """

    # Events this watchdog listens to
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
        BrowserConnectedEvent,
    ]

    # Events this watchdog emits
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to BrowserConnectedEvent for permission granting.
        """
        self.event_bus.on(BrowserConnectedEvent, self.on_BrowserConnectedEvent)

    async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
        """Grant permissions when browser connects.

        Reads permissions from browser_profile and grants them
        globally using CDP Browser.grantPermissions.

        Args:
            event: BrowserConnectedEvent from session.

        Note:
            Failures are logged but don't raise - permissions are
            not critical to browser operation.
        """
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

