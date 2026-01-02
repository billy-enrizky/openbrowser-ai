"""Screenshot watchdog for handling screenshot requests using CDP.

This module provides the ScreenshotWatchdog which captures screenshots
using CDP Page.captureScreenshot in response to ScreenshotEvent.

Classes:
    ScreenshotWatchdog: Handles screenshot requests via CDP.
"""

import logging
from typing import Any, ClassVar

from bubus import BaseEvent
from cdp_use.cdp.page import CaptureScreenshotParameters

from src.openbrowser.browser.events import ScreenshotEvent
from src.openbrowser.browser.watchdogs.base import BaseWatchdog

logger = logging.getLogger(__name__)


class ScreenshotWatchdog(BaseWatchdog):
    """Handles screenshot requests using CDP.

    Captures screenshots in response to ScreenshotEvent using
    CDP Page.captureScreenshot. Supports full page screenshots
    and viewport clipping.

    Listens to:
        ScreenshotEvent: Triggers screenshot capture.

    Returns:
        Base64-encoded JPEG screenshot data.

    Example:
        >>> result = await event_bus.dispatch(
        ...     ScreenshotEvent(full_page=True)
        ... )
        >>> base64_data = result  # JPEG image data
    """

    # Events this watchdog listens to
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
        ScreenshotEvent,
    ]

    # Events this watchdog emits
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to ScreenshotEvent for capture requests.
        """
        self.event_bus.on(ScreenshotEvent, self.on_ScreenshotEvent)

    async def on_ScreenshotEvent(self, event: ScreenshotEvent) -> str:
        """Handle screenshot request.

        Captures a screenshot using CDP Page.captureScreenshot with
        JPEG format and configurable quality.

        Args:
            event: ScreenshotEvent with parameters:
                - full_page: Whether to capture beyond viewport.
                - clip: Optional viewport region to capture.

        Returns:
            Base64-encoded JPEG screenshot data.

        Raises:
            RuntimeError: If screenshot capture fails.
        """
        self.logger.debug('[ScreenshotWatchdog] Handler START - on_ScreenshotEvent called')
        try:
            # Get CDP client and session for current target
            cdp_session = await self.browser_session.get_or_create_cdp_session()

            # Prepare screenshot parameters
            params = CaptureScreenshotParameters(
                format='jpeg',
                quality=60,
                captureBeyondViewport=event.full_page,
            )

            if event.clip:
                params['clip'] = {
                    'x': event.clip['x'],
                    'y': event.clip['y'],
                    'width': event.clip['width'],
                    'height': event.clip['height'],
                    'scale': 1,
                }

            # Take screenshot using CDP
            self.logger.debug(f'[ScreenshotWatchdog] Taking screenshot with params: {params}')
            result = await cdp_session.cdp_client.send.Page.captureScreenshot(
                params=params, session_id=cdp_session.session_id
            )

            # Return base64-encoded screenshot data
            if result and 'data' in result:
                self.logger.debug('[ScreenshotWatchdog] Screenshot captured successfully')
                return result['data']

            raise RuntimeError('[ScreenshotWatchdog] Screenshot result missing data')
        except Exception as e:
            self.logger.error(f'[ScreenshotWatchdog] Screenshot failed: {e}')
            raise

