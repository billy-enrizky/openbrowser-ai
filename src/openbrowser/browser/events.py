"""Event definitions for browser communication.

This module defines all event types used in the event-driven browser architecture.
Events are dispatched via bubus EventBus to coordinate actions between browser
components, watchdogs, and external consumers.

Event Categories:
    Browser Lifecycle: BrowserStartEvent, BrowserStopEvent, BrowserConnectedEvent, etc.
    Navigation: NavigateToUrlEvent, NavigationStartedEvent, NavigationCompleteEvent
    Tab Management: TabCreatedEvent, TabClosedEvent, SwitchTabEvent, CloseTabEvent
    Browser Actions: ClickElementEvent, TypeTextEvent, PressKeyEvent, ScreenshotEvent
    File Downloads: FileDownloadedEvent
    Storage State: SaveStorageStateEvent, LoadStorageStateEvent
    Errors: BrowserErrorEvent

Each event class inherits from bubus.BaseEvent and may specify:
    - Typed result (generic parameter)
    - Event-specific timeout via event_timeout
    - Pydantic fields for event data

Timeouts can be overridden via environment variables (e.g., TIMEOUT_NavigateToUrlEvent).

Example:
    >>> from src.openbrowser.browser.events import NavigateToUrlEvent
    >>> event = NavigateToUrlEvent(url='https://example.com', new_tab=True)
    >>> await event_bus.dispatch(event)
"""

import os
from typing import Any, Literal

from bubus import BaseEvent
from bubus.models import T_EventResultType
from cdp_use.cdp.target import TargetID
from pydantic import BaseModel, Field


def _get_timeout(env_var: str, default: float) -> float | None:
    """Safely parse environment variable timeout values with robust error handling.

    Retrieves timeout value from environment variable, falling back to default
    if not set or invalid.

    Args:
        env_var: Environment variable name (e.g., 'TIMEOUT_NavigateToUrlEvent').
        default: Default timeout value as float (e.g., 15.0).

    Returns:
        Parsed float value from environment, or default if parsing fails
        or value is negative.

    Example:
        >>> os.environ['TIMEOUT_MyEvent'] = '30.0'
        >>> _get_timeout('TIMEOUT_MyEvent', 10.0)
        30.0
    """
    env_value = os.getenv(env_var)
    if env_value:
        try:
            parsed = float(env_value)
            if parsed < 0:
                return default
            return parsed
        except (ValueError, TypeError):
            pass

    return default


# ============================================================================
# Browser Lifecycle Events
# ============================================================================


class BrowserStartEvent(BaseEvent[dict[str, str]]):
    """Event to start or connect to a browser.

    Triggers browser launch and CDP connection establishment.
    Returns dict with 'cdp_url' on success.

    Attributes:
        cdp_url: Optional CDP URL for connecting to existing browser.
        launch_options: Additional options for browser launch.
        event_timeout: Timeout for event handling (default: 30s).
    """

    cdp_url: str | None = None
    launch_options: dict[str, Any] = Field(default_factory=dict)

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserStartEvent', 30.0)


class BrowserStopEvent(BaseEvent[None]):
    """Event to stop or disconnect from browser.

    Triggers browser shutdown and cleanup.

    Attributes:
        force: If True, forcefully kill browser process.
            If False, keep browser alive for potential reattachment.
        event_timeout: Timeout for event handling (default: 45s).
    """

    force: bool = False

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserStopEvent', 45.0)


class BrowserLaunchResult(BaseModel):
    """Result of launching a browser process.

    Returned by BrowserLaunchEvent handlers to communicate
    the CDP WebSocket URL.

    Attributes:
        cdp_url: WebSocket URL for CDP connection.
    """

    cdp_url: str


class BrowserLaunchEvent(BaseEvent[BrowserLaunchResult]):
    """Event to launch a local browser process.

    Handled by LocalBrowserWatchdog to spawn Chrome subprocess.
    Returns BrowserLaunchResult with CDP URL on success.

    Attributes:
        event_timeout: Timeout for browser launch (default: 30s).
    """

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserLaunchEvent', 30.0)


class BrowserKillEvent(BaseEvent[None]):
    """Event to kill local browser subprocess.

    Forcefully terminates the browser process. Used when force=True
    in BrowserStopEvent.

    Attributes:
        event_timeout: Timeout for process termination (default: 30s).
    """

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserKillEvent', 30.0)


class BrowserConnectedEvent(BaseEvent[None]):
    """Event indicating browser has started and CDP is connected.

    Dispatched after successful browser launch and CDP connection.
    Watchdogs listen to this to initialize their functionality.

    Attributes:
        cdp_url: The CDP WebSocket URL of the connected browser.
        event_timeout: Timeout for event handling (default: 30s).
    """

    cdp_url: str

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserConnectedEvent', 30.0)


class BrowserStoppedEvent(BaseEvent[None]):
    """Event indicating browser has stopped or disconnected.

    Dispatched when browser session ends. Watchdogs listen to this
    for cleanup.

    Attributes:
        reason: Optional description of why browser stopped.
        event_timeout: Timeout for event handling (default: 30s).
    """

    reason: str | None = None

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserStoppedEvent', 30.0)


# ============================================================================
# Navigation Events
# ============================================================================


class NavigateToUrlEvent(BaseEvent[None]):
    """Event to navigate to a specific URL.

    Triggers page navigation in the browser. Can optionally open
    in a new tab.

    Attributes:
        url: The URL to navigate to.
        wait_until: Navigation completion condition ('load', 'domcontentloaded',
            'networkidle', 'commit').
        timeout_ms: Optional navigation timeout in milliseconds.
        new_tab: If True, opens URL in a new tab.
        event_timeout: Timeout for navigation (default: 15s).
    """

    url: str
    wait_until: Literal['load', 'domcontentloaded', 'networkidle', 'commit'] = 'load'
    timeout_ms: int | None = None
    new_tab: bool = Field(
        default=False, description='Set True to open URL in a new tab'
    )

    event_timeout: float | None = _get_timeout('TIMEOUT_NavigateToUrlEvent', 15.0)


class NavigationStartedEvent(BaseEvent[None]):
    """Event indicating navigation has started.

    Dispatched just before Page.navigate CDP command is sent.

    Attributes:
        target_id: CDP target ID of the navigating page.
        url: The URL being navigated to.
        event_timeout: Timeout for event handling (default: 30s).
    """

    target_id: TargetID
    url: str

    event_timeout: float | None = _get_timeout('TIMEOUT_NavigationStartedEvent', 30.0)


class NavigationCompleteEvent(BaseEvent[None]):
    """Event indicating navigation has completed.

    Dispatched after page load completes. SecurityWatchdog uses this
    to check for redirect violations.

    Attributes:
        target_id: CDP target ID of the navigated page.
        url: The final URL after navigation (may differ due to redirects).
        status: HTTP status code of the navigation response.
        error_message: Error message if navigation failed.
        event_timeout: Timeout for event handling (default: 30s).
    """

    target_id: TargetID
    url: str
    status: int | None = None
    error_message: str | None = None

    event_timeout: float | None = _get_timeout('TIMEOUT_NavigationCompleteEvent', 30.0)


# ============================================================================
# Tab Management Events
# ============================================================================


class TabCreatedEvent(BaseEvent[None]):
    """Event indicating a new tab was created.

    Dispatched when a new browser tab or popup is opened.
    Watchdogs can use this to set up per-tab functionality.

    Attributes:
        target_id: CDP target ID of the new tab.
        url: Initial URL of the new tab.
        event_timeout: Timeout for event handling (default: 30s).
    """

    target_id: TargetID
    url: str

    event_timeout: float | None = _get_timeout('TIMEOUT_TabCreatedEvent', 30.0)


class TabClosedEvent(BaseEvent[None]):
    """Event indicating a tab was closed.

    Dispatched when a browser tab is closed. Used for cleanup.

    Attributes:
        target_id: CDP target ID of the closed tab.
        event_timeout: Timeout for event handling (default: 10s).
    """

    target_id: TargetID

    event_timeout: float | None = _get_timeout('TIMEOUT_TabClosedEvent', 10.0)


class SwitchTabEvent(BaseEvent[TargetID]):
    """Event to switch to a different tab.

    Triggers agent focus change and visual tab activation.
    Returns the target_id of the switched-to tab.

    Attributes:
        target_id: CDP target ID to switch to. If None, switches to
            the most recently opened tab.
        event_timeout: Timeout for tab switch (default: 10s).
    """

    target_id: TargetID | None = Field(
        default=None, description='None means switch to the most recently opened tab'
    )

    event_timeout: float | None = _get_timeout('TIMEOUT_SwitchTabEvent', 10.0)


class CloseTabEvent(BaseEvent[None]):
    """Event to close a tab.

    Triggers tab closure via CDP Target.closeTarget.

    Attributes:
        target_id: CDP target ID of the tab to close.
        event_timeout: Timeout for tab closure (default: 10s).
    """

    target_id: TargetID

    event_timeout: float | None = _get_timeout('TIMEOUT_CloseTabEvent', 10.0)


class AgentFocusChangedEvent(BaseEvent[None]):
    """Event indicating agent focus changed to a different tab.

    Dispatched when the browser session switches which tab is
    the "active" target for agent interactions.

    Attributes:
        target_id: CDP target ID of the newly focused tab.
        url: URL of the newly focused tab.
        event_timeout: Timeout for event handling (default: 10s).
    """

    target_id: TargetID
    url: str

    event_timeout: float | None = _get_timeout('TIMEOUT_AgentFocusChangedEvent', 10.0)


# ============================================================================
# Browser Action Events
# ============================================================================


class ClickElementEvent(BaseEvent[dict[str, Any] | None]):
    """Event to click an element by index.

    Triggers element click using the selector map from DOM state.
    Returns dict with click result or None.

    Attributes:
        index: Element index from the serialized DOM tree.
        button: Mouse button to use ('left', 'right', 'middle').
        event_timeout: Timeout for click action (default: 15s).
    """

    index: int
    button: Literal['left', 'right', 'middle'] = 'left'

    event_timeout: float | None = _get_timeout('TIMEOUT_ClickElementEvent', 15.0)


class TypeTextEvent(BaseEvent[dict | None]):
    """Event to type text into an element by index.

    Triggers text input into the specified element.

    Attributes:
        index: Element index from the serialized DOM tree.
        text: Text content to type.
        clear: If True, clears existing content before typing.
        event_timeout: Timeout for typing action (default: 15s).
    """

    index: int
    text: str
    clear: bool = True

    event_timeout: float | None = _get_timeout('TIMEOUT_TypeTextEvent', 15.0)


class PressKeyEvent(BaseEvent[None]):
    """Event to press a keyboard key.

    Triggers keyboard key press via CDP Input.dispatchKeyEvent.

    Attributes:
        key: Key identifier (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown').
            Uses DOM key values.
        event_timeout: Timeout for key press (default: 15s).
    """

    key: str  # e.g., "Enter", "Tab", "Escape", "ArrowDown", "ArrowUp"

    event_timeout: float | None = _get_timeout('TIMEOUT_PressKeyEvent', 15.0)


# ============================================================================
# Browser State Events
# ============================================================================


class ScreenshotEvent(BaseEvent[str]):
    """Event to request a screenshot.

    Triggers screenshot capture via CDP Page.captureScreenshot.
    Returns base64-encoded image data.

    Attributes:
        full_page: If True, captures entire scrollable page.
        clip: Optional dict with {x, y, width, height} for partial capture.
        event_timeout: Timeout for screenshot (default: 8s).
    """

    full_page: bool = False
    clip: dict[str, float] | None = None  # {x, y, width, height}

    event_timeout: float | None = _get_timeout('TIMEOUT_ScreenshotEvent', 8.0)


# ============================================================================
# File Download Events
# ============================================================================


class FileDownloadedEvent(BaseEvent[None]):
    """Event indicating a file has been downloaded.

    Dispatched by DownloadsWatchdog when a download completes.

    Attributes:
        url: Source URL of the downloaded file.
        path: Local file path where file was saved.
        file_name: Name of the downloaded file.
        file_size: Size of file in bytes.
        file_type: File extension (e.g., 'pdf', 'zip').
        mime_type: MIME type (e.g., 'application/pdf').
        from_cache: Whether file was served from cache.
        auto_download: Whether this was an automatic download (e.g., PDF).
        event_timeout: Timeout for event handling (default: 30s).
    """

    url: str
    path: str
    file_name: str
    file_size: int
    file_type: str | None = None  # e.g., 'pdf', 'zip', 'docx', etc.
    mime_type: str | None = None  # e.g., 'application/pdf'
    from_cache: bool = False
    auto_download: bool = False  # Whether this was an automatic download (e.g., PDF auto-download)

    event_timeout: float | None = _get_timeout('TIMEOUT_FileDownloadedEvent', 30.0)


# ============================================================================
# Storage State Events
# ============================================================================


class SaveStorageStateEvent(BaseEvent[None]):
    """Event to save browser storage state.

    Triggers saving of cookies and localStorage to file.

    Attributes:
        path: Optional path to save to. If None, uses profile setting.
        event_timeout: Timeout for save operation (default: 30s).
    """

    path: str | None = Field(default=None, description='Optional path to save to (overrides profile setting)')

    event_timeout: float | None = _get_timeout('TIMEOUT_SaveStorageStateEvent', 30.0)


class LoadStorageStateEvent(BaseEvent[None]):
    """Event to load browser storage state.

    Triggers loading of cookies and localStorage from file.

    Attributes:
        path: Optional path to load from. If None, uses profile setting.
        event_timeout: Timeout for load operation (default: 30s).
    """

    path: str | None = Field(default=None, description='Optional path to load from (overrides profile setting)')

    event_timeout: float | None = _get_timeout('TIMEOUT_LoadStorageStateEvent', 30.0)


class StorageStateSavedEvent(BaseEvent[None]):
    """Event indicating storage state has been saved.

    Dispatched after successful storage state save operation.

    Attributes:
        path: File path where state was saved.
        cookies_count: Number of cookies saved.
        origins_count: Number of localStorage origins saved.
        event_timeout: Timeout for event handling (default: 10s).
    """

    path: str
    cookies_count: int = 0
    origins_count: int = 0

    event_timeout: float | None = _get_timeout('TIMEOUT_StorageStateSavedEvent', 10.0)


class StorageStateLoadedEvent(BaseEvent[None]):
    """Event indicating storage state has been loaded.

    Dispatched after successful storage state load operation.

    Attributes:
        path: File path from which state was loaded.
        cookies_count: Number of cookies loaded.
        origins_count: Number of localStorage origins loaded.
        event_timeout: Timeout for event handling (default: 10s).
    """

    path: str
    cookies_count: int = 0
    origins_count: int = 0

    event_timeout: float | None = _get_timeout('TIMEOUT_StorageStateLoadedEvent', 10.0)


# ============================================================================
# Error Events
# ============================================================================


class BrowserErrorEvent(BaseEvent[None]):
    """Event indicating an error occurred in the browser layer.

    Dispatched when errors occur during browser operations. Can be
    used for logging, recovery, or error aggregation.

    Attributes:
        error_type: Category of error (e.g., 'NavigationBlocked',
            'BrowserStartEventError').
        message: Human-readable error description.
        details: Additional error context as dict.
        event_timeout: Timeout for event handling (default: 30s).
    """

    error_type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserErrorEvent', 30.0)

