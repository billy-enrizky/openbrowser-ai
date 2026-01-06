"""Recording Watchdog for Browser Use Sessions.

This module provides the RecordingWatchdog which manages video recording
of browser sessions using CDP screencasting and the VideoRecorderService.

Classes:
    RecordingWatchdog: Manages video recording via CDP screencast.

Requirements:
    - record_video_dir set in BrowserProfile
    - Optional: imageio, imageio-ffmpeg, numpy for encoding
"""

import asyncio
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from bubus import BaseEvent
from cdp_use.cdp.page.events import ScreencastFrameEvent
from pydantic import PrivateAttr

from openbrowser.browser.events import BrowserConnectedEvent, BrowserStopEvent
from openbrowser.browser.profile import ViewportSize
from openbrowser.browser.video_recorder import VideoRecorderService
from openbrowser.browser.watchdogs.base import BaseWatchdog


class RecordingWatchdog(BaseWatchdog):
    """Manages video recording of a browser session using CDP screencasting.

    Captures frames via CDP Page.startScreencast and encodes them using
    VideoRecorderService. Automatically detects viewport size if not
    specified in profile.

    Listens to:
        BrowserConnectedEvent: Starts recording if configured.
        BrowserStopEvent: Stops recording and saves video.

    Configuration (in BrowserProfile):
        record_video_dir: Output directory for video files.
        record_video_size: Optional ViewportSize for video dimensions.
        record_video_framerate: Frames per second (default varies).
        record_video_format: Output format (default: 'mp4').

    Example:
        >>> profile = BrowserProfile(
        ...     record_video_dir='/tmp/videos',
        ...     record_video_framerate=30
        ... )
    """

    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [BrowserConnectedEvent, BrowserStopEvent]
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

    _recorder: VideoRecorderService | None = PrivateAttr(default=None)

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to browser lifecycle events for recording control.
        """
        self.event_bus.on(BrowserConnectedEvent, self.on_BrowserConnectedEvent)
        self.event_bus.on(BrowserStopEvent, self.on_BrowserStopEvent)

    async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
        """Starts video recording if configured in the browser profile.

        Initializes VideoRecorderService, registers screencast handler,
        and starts CDP screencast. Auto-detects viewport size if not
        specified.

        Args:
            event: BrowserConnectedEvent from session.
        """
        profile = self.browser_session.browser_profile
        if not profile.record_video_dir:
            return

        # Dynamically determine video size
        size = profile.record_video_size
        if not size:
            self.logger.debug('[RecordingWatchdog] record_video_size not specified, detecting viewport size...')
            size = await self._get_current_viewport_size()

        if not size:
            self.logger.warning('[RecordingWatchdog] Cannot start video recording: viewport size could not be determined.')
            return

        video_format = getattr(profile, 'record_video_format', 'mp4').strip('.')
        output_path = Path(profile.record_video_dir) / f'{uuid4().hex}.{video_format}'

        self.logger.debug(f'[RecordingWatchdog] Initializing video recorder for format: {video_format}')
        self._recorder = VideoRecorderService(output_path=output_path, size=size, framerate=profile.record_video_framerate)
        self._recorder.start()

        if not self._recorder._is_active:
            self._recorder = None
            return

        self.browser_session.cdp_client.register.Page.screencastFrame(self.on_screencastFrame)

        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            await cdp_session.cdp_client.send.Page.startScreencast(
                params={
                    'format': 'png',
                    'quality': 90,
                    'maxWidth': size['width'],
                    'maxHeight': size['height'],
                    'everyNthFrame': 1,
                },
                session_id=cdp_session.session_id,
            )
            self.logger.info(f'[RecordingWatchdog] Started video recording to {output_path}')
        except Exception as e:
            self.logger.error(f'[RecordingWatchdog] Failed to start screencast via CDP: {e}')
            if self._recorder:
                self._recorder.stop_and_save()
                self._recorder = None

    async def _get_current_viewport_size(self) -> ViewportSize | None:
        """Gets the current viewport size directly from the browser via CDP.

        Uses Page.getLayoutMetrics to get accurate visible area dimensions.

        Returns:
            ViewportSize with current dimensions, or None on failure.
        """
        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            metrics = await cdp_session.cdp_client.send.Page.getLayoutMetrics(session_id=cdp_session.session_id)

            # Use cssVisualViewport for the most accurate representation of the visible area
            viewport = metrics.get('cssVisualViewport', {})
            width = viewport.get('clientWidth')
            height = viewport.get('clientHeight')

            if width and height:
                self.logger.debug(f'[RecordingWatchdog] Detected viewport size: {width}x{height}')
                return ViewportSize(width=int(width), height=int(height))
        except Exception as e:
            self.logger.warning(f'[RecordingWatchdog] Failed to get viewport size from browser: {e}')

        return None

    def on_screencastFrame(self, event: ScreencastFrameEvent, session_id: str | None) -> None:
        """Synchronous handler for incoming screencast frames.

        Adds frame to recorder and schedules async acknowledgment.

        Args:
            event: ScreencastFrameEvent with frame data.
            session_id: CDP session ID for acknowledgment.
        """
        if not self._recorder:
            return
        self._recorder.add_frame(event['data'])
        asyncio.create_task(self._ack_screencast_frame(event, session_id))

    async def _ack_screencast_frame(self, event: ScreencastFrameEvent, session_id: str | None) -> None:
        """Asynchronously acknowledges a screencast frame.

        Required by CDP to continue receiving frames.

        Args:
            event: ScreencastFrameEvent with sessionId for ack.
            session_id: CDP session ID.
        """
        try:
            await self.browser_session.cdp_client.send.Page.screencastFrameAck(
                params={'sessionId': event['sessionId']}, session_id=session_id
            )
        except Exception as e:
            self.logger.debug(f'[RecordingWatchdog] Failed to acknowledge screencast frame: {e}')

    async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
        """
        Stops the video recording and finalizes the video file.
        """
        if self._recorder:
            recorder = self._recorder
            self._recorder = None

            self.logger.debug('[RecordingWatchdog] Stopping video recording and saving file...')
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, recorder.stop_and_save)

