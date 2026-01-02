"""Base watchdog class for browser monitoring components.

This module defines the BaseWatchdog class that all watchdogs inherit from.
Watchdogs are event-driven components that monitor and react to browser
state changes.

Classes:
    BaseWatchdog: Abstract base class for all browser watchdogs.

Example:
    >>> class MyWatchdog(BaseWatchdog):
    ...     LISTENS_TO = [TabCreatedEvent]
    ...     EMITS = [MyCustomEvent]
    ...
    ...     def attach_to_session(self):
    ...         self.event_bus.on(TabCreatedEvent, self.on_tab_created)
    ...
    ...     async def on_tab_created(self, event):
    ...         self.logger.info(f'New tab: {event.target_id}')
"""

import logging
from typing import Any, ClassVar

from bubus import BaseEvent, EventBus
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class BaseWatchdog(BaseModel):
    """Base class for all browser watchdogs.

    Watchdogs monitor browser state and emit events based on changes.
    They automatically register event handlers based on method names.

    Subclasses should:
    1. Define LISTENS_TO with event types to subscribe to
    2. Define EMITS with event types the watchdog may dispatch
    3. Override attach_to_session() to register handlers
    4. Implement handler methods (async def on_EventName)

    Class Attributes:
        LISTENS_TO: List of event types this watchdog subscribes to.
        EMITS: List of event types this watchdog may dispatch.

    Instance Attributes:
        event_bus: Shared EventBus for event communication.
        browser_session: Reference to the parent BrowserSession.

    Example:
        >>> class TabWatchdog(BaseWatchdog):
        ...     LISTENS_TO = [TabCreatedEvent, TabClosedEvent]
        ...
        ...     def attach_to_session(self):
        ...         self.event_bus.on(TabCreatedEvent, self.on_tab_created)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid',
        validate_assignment=False,
        revalidate_instances='never',
    )

    # Class variables to statically define the list of events relevant to each watchdog
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = []
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

    # Core dependencies
    event_bus: EventBus = Field()
    browser_session: Any = Field()  # BrowserSession type

    @property
    def logger(self) -> logging.Logger:
        """Get the logger from the browser session.

        Returns:
            Logger instance for this watchdog's browser session.
        """
        return self.browser_session.logger

    def attach_to_session(self) -> None:
        """Attach event handlers to the event bus.

        Subclasses should override this to register their specific handlers.
        Called by BrowserSession during initialization.

        Example:
            >>> def attach_to_session(self):
            ...     self.event_bus.on(TabCreatedEvent, self.on_TabCreatedEvent)
        """
        pass

