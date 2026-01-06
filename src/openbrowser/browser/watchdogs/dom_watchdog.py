"""DOM watchdog for browser DOM tree management using CDP.

This module provides the DOMWatchdog which maintains cached DOM state
and selector maps for the browser session. Acts as a bridge between
the event-driven architecture and DomService implementation.

Classes:
    DOMWatchdog: Handles DOM tree caching and element access.
"""

import logging
from typing import TYPE_CHECKING, ClassVar

from bubus import BaseEvent

from openbrowser.browser.dom import DomState
from openbrowser.browser.events import TabCreatedEvent
from openbrowser.browser.watchdogs.base import BaseWatchdog

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DOMWatchdog(BaseWatchdog):
    """Handles DOM tree building and element access via CDP.

    Acts as a bridge between the event-driven browser session and the
    DomService implementation. Maintains cached DOM state and provides
    helper methods for other watchdogs and tools.

    Following browser-use pattern, caches DOM state and provides access
    to selector maps for element interaction.

    Attributes:
        current_dom_state: Cached DomState from last update.
        selector_map: Mapping from distinct_id to backend_node_id.

    Listens to:
        TabCreatedEvent: Handles new tab creation.

    Example:
        >>> watchdog = DOMWatchdog(event_bus=bus, browser_session=session)
        >>> watchdog.update_dom_state(new_dom_state)
        >>> backend_id = watchdog.selector_map[1]
    """

    LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [TabCreatedEvent]
    EMITS: ClassVar[list[type[BaseEvent]]] = []

    # Public properties for other watchdogs
    current_dom_state: DomState | None = None
    selector_map: dict[int, int] | None = None

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to TabCreatedEvent for new tab handling.
        """
        self.event_bus.on(TabCreatedEvent, self.on_TabCreatedEvent)

    async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
        """Handle tab creation.

        Currently no special setup needed for new tabs.

        Args:
            event: TabCreatedEvent with target info.
        """
        return None

    def update_dom_state(self, dom_state: DomState) -> None:
        """Update cached DOM state.

        Updates the watchdog's cached state and propagates to browser
        session if it supports cached selector maps.

        Args:
            dom_state: New DomState to cache.
        """
        self.current_dom_state = dom_state
        self.selector_map = dom_state.selector_map

        # Update browser session's cached selector map if it has that method
        if hasattr(self.browser_session, "update_cached_selector_map"):
            self.browser_session.update_cached_selector_map(self.selector_map)

        self.logger.debug(f"DOM state updated: {len(self.selector_map)} elements")

