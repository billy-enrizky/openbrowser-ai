"""Event definitions for browser communication."""

import os
from typing import Any, Literal

from bubus import BaseEvent
from bubus.models import T_EventResultType
from cdp_use.cdp.target import TargetID
from pydantic import BaseModel, Field


def _get_timeout(env_var: str, default: float) -> float | None:
    """Safely parse environment variable timeout values with robust error handling.

    Args:
        env_var: Environment variable name (e.g. 'TIMEOUT_NavigateToUrlEvent')
        default: Default timeout value as float (e.g. 15.0)

    Returns:
        Parsed float value or the default if parsing fails
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
    """Start/connect to browser."""

    cdp_url: str | None = None
    launch_options: dict[str, Any] = Field(default_factory=dict)

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserStartEvent', 30.0)


class BrowserStopEvent(BaseEvent[None]):
    """Stop/disconnect from browser."""

    force: bool = False

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserStopEvent', 45.0)


class BrowserLaunchResult(BaseModel):
    """Result of launching a browser."""

    cdp_url: str


class BrowserLaunchEvent(BaseEvent[BrowserLaunchResult]):
    """Launch a local browser process."""

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserLaunchEvent', 30.0)


class BrowserKillEvent(BaseEvent[None]):
    """Kill local browser subprocess."""

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserKillEvent', 30.0)


class BrowserConnectedEvent(BaseEvent[None]):
    """Browser has started/connected."""

    cdp_url: str

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserConnectedEvent', 30.0)


class BrowserStoppedEvent(BaseEvent[None]):
    """Browser has stopped/disconnected."""

    reason: str | None = None

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserStoppedEvent', 30.0)


# ============================================================================
# Navigation Events
# ============================================================================


class NavigateToUrlEvent(BaseEvent[None]):
    """Navigate to a specific URL."""

    url: str
    wait_until: Literal['load', 'domcontentloaded', 'networkidle', 'commit'] = 'load'
    timeout_ms: int | None = None
    new_tab: bool = Field(
        default=False, description='Set True to open URL in a new tab'
    )

    event_timeout: float | None = _get_timeout('TIMEOUT_NavigateToUrlEvent', 15.0)


class NavigationStartedEvent(BaseEvent[None]):
    """Navigation started."""

    target_id: TargetID
    url: str

    event_timeout: float | None = _get_timeout('TIMEOUT_NavigationStartedEvent', 30.0)


class NavigationCompleteEvent(BaseEvent[None]):
    """Navigation completed."""

    target_id: TargetID
    url: str
    status: int | None = None
    error_message: str | None = None

    event_timeout: float | None = _get_timeout('TIMEOUT_NavigationCompleteEvent', 30.0)


# ============================================================================
# Tab Management Events
# ============================================================================


class TabCreatedEvent(BaseEvent[None]):
    """A new tab was created."""

    target_id: TargetID
    url: str

    event_timeout: float | None = _get_timeout('TIMEOUT_TabCreatedEvent', 30.0)


class TabClosedEvent(BaseEvent[None]):
    """A tab was closed."""

    target_id: TargetID

    event_timeout: float | None = _get_timeout('TIMEOUT_TabClosedEvent', 10.0)


class SwitchTabEvent(BaseEvent[TargetID]):
    """Switch to a different tab."""

    target_id: TargetID | None = Field(
        default=None, description='None means switch to the most recently opened tab'
    )

    event_timeout: float | None = _get_timeout('TIMEOUT_SwitchTabEvent', 10.0)


class CloseTabEvent(BaseEvent[None]):
    """Close a tab."""

    target_id: TargetID

    event_timeout: float | None = _get_timeout('TIMEOUT_CloseTabEvent', 10.0)


class AgentFocusChangedEvent(BaseEvent[None]):
    """Agent focus changed to a different tab."""

    target_id: TargetID
    url: str

    event_timeout: float | None = _get_timeout('TIMEOUT_AgentFocusChangedEvent', 10.0)


# ============================================================================
# Browser Action Events
# ============================================================================


class ClickElementEvent(BaseEvent[dict[str, Any] | None]):
    """Click an element by index."""

    index: int
    button: Literal['left', 'right', 'middle'] = 'left'

    event_timeout: float | None = _get_timeout('TIMEOUT_ClickElementEvent', 15.0)


class TypeTextEvent(BaseEvent[dict | None]):
    """Type text into an element by index."""

    index: int
    text: str
    clear: bool = True

    event_timeout: float | None = _get_timeout('TIMEOUT_TypeTextEvent', 15.0)


class PressKeyEvent(BaseEvent[None]):
    """Press a keyboard key."""

    key: str  # e.g., "Enter", "Tab", "Escape", "ArrowDown", "ArrowUp"

    event_timeout: float | None = _get_timeout('TIMEOUT_PressKeyEvent', 15.0)


# ============================================================================
# Browser State Events
# ============================================================================


class ScreenshotEvent(BaseEvent[str]):
    """Request to take a screenshot."""

    full_page: bool = False
    clip: dict[str, float] | None = None  # {x, y, width, height}

    event_timeout: float | None = _get_timeout('TIMEOUT_ScreenshotEvent', 8.0)


# ============================================================================
# File Download Events
# ============================================================================


class FileDownloadedEvent(BaseEvent[None]):
    """A file has been downloaded."""

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
    """Save browser storage state (cookies, localStorage, etc.) to file."""

    path: str | None = Field(default=None, description='Optional path to save to (overrides profile setting)')

    event_timeout: float | None = _get_timeout('TIMEOUT_SaveStorageStateEvent', 30.0)


class LoadStorageStateEvent(BaseEvent[None]):
    """Load browser storage state (cookies, localStorage, etc.) from file."""

    path: str | None = Field(default=None, description='Optional path to load from (overrides profile setting)')

    event_timeout: float | None = _get_timeout('TIMEOUT_LoadStorageStateEvent', 30.0)


class StorageStateSavedEvent(BaseEvent[None]):
    """Storage state has been saved."""

    path: str
    cookies_count: int = 0
    origins_count: int = 0

    event_timeout: float | None = _get_timeout('TIMEOUT_StorageStateSavedEvent', 10.0)


class StorageStateLoadedEvent(BaseEvent[None]):
    """Storage state has been loaded."""

    path: str
    cookies_count: int = 0
    origins_count: int = 0

    event_timeout: float | None = _get_timeout('TIMEOUT_StorageStateLoadedEvent', 10.0)


# ============================================================================
# Error Events
# ============================================================================


class BrowserErrorEvent(BaseEvent[None]):
    """An error occurred in the browser layer."""

    error_type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    event_timeout: float | None = _get_timeout('TIMEOUT_BrowserErrorEvent', 30.0)

