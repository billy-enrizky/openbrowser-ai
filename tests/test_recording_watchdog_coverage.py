"""Tests for openbrowser.browser.watchdogs.recording_watchdog module -- 100% coverage target."""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_browser_session():
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test_recording_watchdog")
    session.event_bus = MagicMock()
    session._cdp_client_root = MagicMock()
    session.agent_focus = MagicMock()
    session.get_or_create_cdp_session = AsyncMock()
    session.cdp_client = MagicMock()
    session.cdp_client.register = MagicMock()
    session.cdp_client.register.Page = MagicMock()
    session.cdp_client.register.Page.screencastFrame = MagicMock()
    session.cdp_client.send = MagicMock()
    session.cdp_client.send.Page = MagicMock()
    session.cdp_client.send.Page.screencastFrameAck = AsyncMock()
    session.id = "test-recording-session"
    session.is_local = True

    profile = MagicMock()
    profile.record_video_dir = None
    profile.record_video_size = None
    profile.record_video_format = "mp4"
    profile.record_video_framerate = 10
    session.browser_profile = profile

    return session


def _make_recording_watchdog(session=None, event_bus=None):
    from openbrowser.browser.watchdogs.recording_watchdog import RecordingWatchdog

    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = MagicMock()
    session.event_bus = event_bus
    return RecordingWatchdog.model_construct(
        event_bus=event_bus,
        browser_session=session,
    )


# ---------------------------------------------------------------------------
# on_BrowserConnectedEvent
# ---------------------------------------------------------------------------

class TestOnBrowserConnectedEvent:
    @pytest.mark.asyncio
    async def test_returns_if_no_record_dir(self):
        session = _make_mock_browser_session()
        session.browser_profile.record_video_dir = None
        watchdog = _make_recording_watchdog(session=session)

        event = MagicMock()
        await watchdog.on_BrowserConnectedEvent(event)
        assert watchdog._recorder is None

    @pytest.mark.asyncio
    async def test_detects_viewport_size_when_not_specified(self):
        session = _make_mock_browser_session()
        session.browser_profile.record_video_dir = "/tmp/videos"
        session.browser_profile.record_video_size = None

        watchdog = _make_recording_watchdog(session=session)

        with patch.object(watchdog, "_get_current_viewport_size", new_callable=AsyncMock, return_value=None):
            event = MagicMock()
            await watchdog.on_BrowserConnectedEvent(event)
            # Cannot determine viewport => recorder stays None
            assert watchdog._recorder is None

    @pytest.mark.asyncio
    async def test_starts_recording_successfully(self):
        session = _make_mock_browser_session()
        session.browser_profile.record_video_dir = "/tmp/videos"
        session.browser_profile.record_video_size = {"width": 1280, "height": 720}
        session.browser_profile.record_video_format = "mp4"
        session.browser_profile.record_video_framerate = 10

        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.startScreencast = AsyncMock()
        cdp_session_mock.session_id = "sess-1"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_recording_watchdog(session=session)

        mock_recorder = MagicMock()
        mock_recorder._is_active = True
        mock_recorder.start = MagicMock()

        event = MagicMock()
        with patch("openbrowser.browser.watchdogs.recording_watchdog.VideoRecorderService", return_value=mock_recorder):
            with patch("openbrowser.browser.watchdogs.recording_watchdog.uuid7str", return_value="test-uuid"):
                await watchdog.on_BrowserConnectedEvent(event)

        assert watchdog._recorder is mock_recorder
        mock_recorder.start.assert_called_once()
        cdp_session_mock.cdp_client.send.Page.startScreencast.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recorder_not_active_after_start(self):
        session = _make_mock_browser_session()
        session.browser_profile.record_video_dir = "/tmp/videos"
        session.browser_profile.record_video_size = {"width": 800, "height": 600}

        watchdog = _make_recording_watchdog(session=session)

        mock_recorder = MagicMock()
        mock_recorder._is_active = False
        mock_recorder.start = MagicMock()

        event = MagicMock()
        with patch("openbrowser.browser.watchdogs.recording_watchdog.VideoRecorderService", return_value=mock_recorder):
            with patch("openbrowser.browser.watchdogs.recording_watchdog.uuid7str", return_value="uuid"):
                await watchdog.on_BrowserConnectedEvent(event)

        assert watchdog._recorder is None

    @pytest.mark.asyncio
    async def test_screencast_start_fails(self):
        session = _make_mock_browser_session()
        session.browser_profile.record_video_dir = "/tmp/videos"
        session.browser_profile.record_video_size = {"width": 1280, "height": 720}

        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.startScreencast = AsyncMock(side_effect=RuntimeError("CDP fail"))
        cdp_session_mock.session_id = "sess-err"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_recording_watchdog(session=session)

        mock_recorder = MagicMock()
        mock_recorder._is_active = True
        mock_recorder.start = MagicMock()
        mock_recorder.stop_and_save = MagicMock()

        event = MagicMock()
        with patch("openbrowser.browser.watchdogs.recording_watchdog.VideoRecorderService", return_value=mock_recorder):
            with patch("openbrowser.browser.watchdogs.recording_watchdog.uuid7str", return_value="uuid"):
                await watchdog.on_BrowserConnectedEvent(event)

        assert watchdog._recorder is None
        mock_recorder.stop_and_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_detects_viewport_when_size_not_set(self):
        session = _make_mock_browser_session()
        session.browser_profile.record_video_dir = "/tmp/videos"
        session.browser_profile.record_video_size = None

        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.startScreencast = AsyncMock()
        cdp_session_mock.session_id = "sess-2"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_recording_watchdog(session=session)

        mock_recorder = MagicMock()
        mock_recorder._is_active = True
        mock_recorder.start = MagicMock()

        event = MagicMock()
        with patch.object(watchdog, "_get_current_viewport_size", new_callable=AsyncMock, return_value={"width": 1920, "height": 1080}):
            with patch("openbrowser.browser.watchdogs.recording_watchdog.VideoRecorderService", return_value=mock_recorder):
                with patch("openbrowser.browser.watchdogs.recording_watchdog.uuid7str", return_value="uuid2"):
                    await watchdog.on_BrowserConnectedEvent(event)

        assert watchdog._recorder is mock_recorder


# ---------------------------------------------------------------------------
# _get_current_viewport_size
# ---------------------------------------------------------------------------

class TestGetCurrentViewportSize:
    @pytest.mark.asyncio
    async def test_returns_viewport_size(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            return_value={
                "cssVisualViewport": {"clientWidth": 1280, "clientHeight": 720}
            }
        )
        cdp_session_mock.session_id = "sess-vp"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_recording_watchdog(session=session)
        result = await watchdog._get_current_viewport_size()
        assert result is not None
        assert result["width"] == 1280
        assert result["height"] == 720

    @pytest.mark.asyncio
    async def test_returns_none_when_no_dimensions(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            return_value={"cssVisualViewport": {}}
        )
        cdp_session_mock.session_id = "sess-vp2"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_recording_watchdog(session=session)
        result = await watchdog._get_current_viewport_size()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        session = _make_mock_browser_session()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("fail"))

        watchdog = _make_recording_watchdog(session=session)
        result = await watchdog._get_current_viewport_size()
        assert result is None


# ---------------------------------------------------------------------------
# on_screencastFrame
# ---------------------------------------------------------------------------

class TestOnScreencastFrame:
    def test_noop_if_no_recorder(self):
        watchdog = _make_recording_watchdog()
        watchdog._recorder = None

        event = {"data": "base64data", "sessionId": "sess1"}
        watchdog.on_screencastFrame(event, "session-x")
        # Should not raise

    def test_adds_frame_and_acks(self):
        watchdog = _make_recording_watchdog()
        mock_recorder = MagicMock()
        watchdog._recorder = mock_recorder

        event = {"data": "frame-data", "sessionId": "sess2"}

        with patch("asyncio.create_task") as mock_create_task:
            watchdog.on_screencastFrame(event, "session-y")

        mock_recorder.add_frame.assert_called_once_with("frame-data")
        mock_create_task.assert_called_once()


# ---------------------------------------------------------------------------
# _ack_screencast_frame
# ---------------------------------------------------------------------------

class TestAckScreencastFrame:
    @pytest.mark.asyncio
    async def test_ack_success(self):
        session = _make_mock_browser_session()
        session.cdp_client.send.Page.screencastFrameAck = AsyncMock()
        watchdog = _make_recording_watchdog(session=session)

        event = {"sessionId": "screencast-session-1"}
        await watchdog._ack_screencast_frame(event, "session-z")
        session.cdp_client.send.Page.screencastFrameAck.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ack_exception(self):
        session = _make_mock_browser_session()
        session.cdp_client.send.Page.screencastFrameAck = AsyncMock(side_effect=RuntimeError("ack fail"))
        watchdog = _make_recording_watchdog(session=session)

        event = {"sessionId": "screencast-session-2"}
        # Should not raise
        await watchdog._ack_screencast_frame(event, "session-w")


# ---------------------------------------------------------------------------
# on_BrowserStopEvent
# ---------------------------------------------------------------------------

class TestOnBrowserStopEvent:
    @pytest.mark.asyncio
    async def test_noop_if_no_recorder(self):
        watchdog = _make_recording_watchdog()
        watchdog._recorder = None

        event = MagicMock()
        await watchdog.on_BrowserStopEvent(event)

    @pytest.mark.asyncio
    async def test_stops_and_saves_recorder(self):
        watchdog = _make_recording_watchdog()
        mock_recorder = MagicMock()
        mock_recorder.stop_and_save = MagicMock()
        watchdog._recorder = mock_recorder

        event = MagicMock()

        with patch("asyncio.get_event_loop") as mock_loop:
            loop_mock = MagicMock()
            loop_mock.run_in_executor = AsyncMock()
            mock_loop.return_value = loop_mock

            await watchdog.on_BrowserStopEvent(event)

        assert watchdog._recorder is None
        loop_mock.run_in_executor.assert_awaited_once()
