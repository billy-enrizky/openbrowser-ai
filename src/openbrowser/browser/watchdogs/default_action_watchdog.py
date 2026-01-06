"""Default action watchdog following browser-use pattern.

This module provides the DefaultActionWatchdog for handling common UI
patterns and default action behaviors like dropdown interactions.

Classes:
    DefaultActionWatchdog: Handles default UI action behaviors.
"""

import logging
from typing import TYPE_CHECKING

from openbrowser.browser.watchdogs.base import BaseWatchdog

if TYPE_CHECKING:
    from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class DefaultActionWatchdog(BaseWatchdog):
    """Watchdog that handles default action behaviors.

    Handles common UI patterns including dropdown interactions
    and other default browser behaviors. Extensible for
    additional default action handling.

    Example:
        >>> watchdog = DefaultActionWatchdog(
        ...     event_bus=bus,
        ...     browser_session=session
        ... )
    """

    browser_session: "BrowserSession"

    def attach(self) -> None:
        """Attach event handlers.

        Currently no specific events registered. Override to add
        handlers for default action behaviors.
        """
        # No specific events to handle yet
        # This watchdog can be extended for default action behaviors
        logger.info("DefaultActionWatchdog attached")

