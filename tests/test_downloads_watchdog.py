"""Tests for openbrowser.browser.watchdogs.downloads_watchdog module."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec

import pytest
from bubus import EventBus
from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


def _make_mock_browser_session(downloads_path=None):
    """Create a mock BrowserSession for DownloadsWatchdog tests."""
    # Use create_autospec to satisfy Pydantic type validation
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger('test_downloads_watchdog')
    session.event_bus = EventBus()
    session._cdp_client_root = MagicMock()
    session.agent_focus = None
    session.get_or_create_cdp_session = AsyncMock()
    session.get_current_page_url = AsyncMock(return_value='https://example.com')
    session.cdp_client = MagicMock()
    session.cdp_client.send.Browser.setDownloadBehavior = AsyncMock()
    session.cdp_client.send.Network.enable = AsyncMock()
    session.cdp_client.register.Browser.downloadWillBegin = MagicMock()
    session.cdp_client.register.Browser.downloadProgress = MagicMock()
    session.cdp_client.register.Network.responseReceived = MagicMock()
    session.id = 'test-session-1234'
    session.is_local = True
    session.get_target_id_from_session_id = MagicMock(return_value=None)
    session.cdp_client_for_frame = AsyncMock()

    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = downloads_path or tempfile.mkdtemp(prefix='openbrowser-test-')
    session.browser_profile.auto_download_pdfs = False

    return session


def _make_watchdog(downloads_path=None):
    from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

    session = _make_mock_browser_session(downloads_path)
    event_bus = EventBus()
    return DownloadsWatchdog(event_bus=event_bus, browser_session=session), session


class TestDownloadsWatchdogInit:
    """Tests for DownloadsWatchdog initialization."""

    def test_init(self):
        watchdog, session = _make_watchdog()
        assert watchdog._sessions_with_listeners == set()
        assert watchdog._active_downloads == {}
        assert watchdog._download_cdp_session_setup is False


@pytest.mark.asyncio
class TestOnBrowserLaunchEvent:
    """Tests for on_BrowserLaunchEvent."""

    async def test_creates_downloads_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloads_path = os.path.join(tmpdir, 'downloads')
            watchdog, session = _make_watchdog(downloads_path)

            event = MagicMock()
            await watchdog.on_BrowserLaunchEvent(event)

            assert os.path.isdir(downloads_path)

    async def test_no_path_configured(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.downloads_path = None

        event = MagicMock()
        # Should not raise
        await watchdog.on_BrowserLaunchEvent(event)


@pytest.mark.asyncio
class TestOnTabCreatedEvent:
    """Tests for on_TabCreatedEvent."""

    async def test_attaches_to_target(self):
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog
        watchdog, session = _make_watchdog()

        with patch.object(DownloadsWatchdog, 'attach_to_target') as mock_attach:
            mock_attach.return_value = AsyncMock()
            event = MagicMock()
            event.target_id = 'target-1'

            await watchdog.on_TabCreatedEvent(event)
            mock_attach.assert_called_once_with('target-1')

    async def test_no_target_id(self):
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog
        watchdog, session = _make_watchdog()

        with patch.object(DownloadsWatchdog, 'attach_to_target') as mock_attach:
            mock_attach.return_value = AsyncMock()
            event = MagicMock()
            event.target_id = None

            await watchdog.on_TabCreatedEvent(event)
            mock_attach.assert_not_called()


@pytest.mark.asyncio
class TestOnTabClosedEvent:
    """Tests for on_TabClosedEvent."""

    async def test_tab_closed_noop(self):
        watchdog, _ = _make_watchdog()
        event = MagicMock()
        # Should not raise
        await watchdog.on_TabClosedEvent(event)


@pytest.mark.asyncio
class TestOnBrowserStoppedEvent:
    """Tests for on_BrowserStoppedEvent."""

    async def test_cleanup(self):
        watchdog, _ = _make_watchdog()
        watchdog._sessions_with_listeners = {'s1'}
        watchdog._active_downloads = {'d1': {}}
        watchdog._download_cdp_session_setup = True
        watchdog._download_cdp_session = MagicMock()
        watchdog._pdf_viewer_cache = {'url1': True}
        watchdog._session_pdf_urls = {'url1': '/path'}
        watchdog._network_monitored_targets = {'t1'}
        watchdog._detected_downloads = {'url1'}
        watchdog._network_callback_registered = True

        # Create mock tasks - need to be actual awaitables for asyncio.gather
        import asyncio
        async def mock_task():
            return
        task = asyncio.create_task(mock_task())
        task.cancel = MagicMock()
        watchdog._cdp_event_tasks = {task}

        event = MagicMock()
        await watchdog.on_BrowserStoppedEvent(event)

        assert watchdog._sessions_with_listeners == set()
        assert watchdog._active_downloads == {}
        assert watchdog._download_cdp_session_setup is False
        assert watchdog._download_cdp_session is None
        assert watchdog._pdf_viewer_cache == {}
        assert watchdog._session_pdf_urls == {}
        assert watchdog._network_monitored_targets == set()
        assert watchdog._detected_downloads == set()
        assert watchdog._network_callback_registered is False


@pytest.mark.asyncio
class TestOnBrowserStateRequestEvent:
    """Tests for on_BrowserStateRequestEvent."""

    async def test_no_cdp_session(self):
        watchdog, session = _make_watchdog()
        session.agent_focus = None

        event = MagicMock()
        # Should return early without dispatching
        await watchdog.on_BrowserStateRequestEvent(event)

    async def test_no_url(self):
        watchdog, session = _make_watchdog()
        session.agent_focus = MagicMock()
        session.agent_focus.target_id = 'target-1'
        session.get_current_page_url.return_value = ''

        event = MagicMock()
        await watchdog.on_BrowserStateRequestEvent(event)

    async def test_dispatches_navigation_complete(self):
        watchdog, session = _make_watchdog()
        cdp = MagicMock()
        cdp.target_id = 'target-1'
        session.agent_focus = cdp
        session.get_current_page_url.return_value = 'https://example.com/page'

        # Replace real EventBus with MagicMock so we can assert on dispatch
        watchdog.event_bus = MagicMock()

        event = MagicMock()
        event.event_id = '00000000-0000-0000-0000-000000000001'

        await watchdog.on_BrowserStateRequestEvent(event)
        watchdog.event_bus.dispatch.assert_called_once()
        # Verify the dispatched event contains navigation data
        dispatched_event = watchdog.event_bus.dispatch.call_args[0][0]
        assert hasattr(dispatched_event, 'event_parent_id') or dispatched_event is not None


@pytest.mark.asyncio
class TestOnNavigationCompleteEvent:
    """Tests for on_NavigationCompleteEvent."""

    async def test_clears_pdf_cache(self):
        watchdog, _ = _make_watchdog()
        watchdog._pdf_viewer_cache = {'https://example.com/doc.pdf': True}

        event = MagicMock()
        event.url = 'https://example.com/doc.pdf'
        event.target_id = 'target-1'

        await watchdog.on_NavigationCompleteEvent(event)

        assert 'https://example.com/doc.pdf' not in watchdog._pdf_viewer_cache

    async def test_auto_download_disabled_returns_early(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = False

        event = MagicMock()
        event.url = 'https://example.com/doc.pdf'
        event.target_id = 'target-1'

        await watchdog.on_NavigationCompleteEvent(event)


class TestIsAutoDownloadEnabled:
    """Tests for _is_auto_download_enabled."""

    def test_enabled(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = True
        assert watchdog._is_auto_download_enabled() is True

    def test_disabled(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = False
        assert watchdog._is_auto_download_enabled() is False


@pytest.mark.asyncio
class TestAttachToTarget:
    """Tests for attach_to_target."""

    async def test_no_downloads_path(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.downloads_path = None

        await watchdog.attach_to_target('target-1')

        # Should return early
        assert watchdog._download_cdp_session_setup is False

    async def test_setup_cdp_listeners(self):
        watchdog, session = _make_watchdog()
        mock_cdp_session = MagicMock()
        mock_cdp_session.session_id = 'sid-123'
        session.get_or_create_cdp_session.return_value = mock_cdp_session

        await watchdog.attach_to_target('target-1')

        assert watchdog._download_cdp_session_setup is True
        session.cdp_client.send.Browser.setDownloadBehavior.assert_called_once()
        session.cdp_client.register.Browser.downloadWillBegin.assert_called_once()
        session.cdp_client.register.Browser.downloadProgress.assert_called_once()

    async def test_skip_if_already_setup(self):
        watchdog, session = _make_watchdog()
        watchdog._download_cdp_session_setup = True

        await watchdog.attach_to_target('target-1')

        # Should not call setDownloadBehavior again
        session.cdp_client.send.Browser.setDownloadBehavior.assert_not_called()

    async def test_handles_setup_error(self):
        watchdog, session = _make_watchdog()
        session.cdp_client.send.Browser.setDownloadBehavior.side_effect = Exception('fail')

        # Should not raise
        await watchdog.attach_to_target('target-1')


@pytest.mark.asyncio
class TestSetupNetworkMonitoring:
    """Tests for _setup_network_monitoring."""

    async def test_skip_already_monitored(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = True
        watchdog._network_monitored_targets.add('target-1')

        await watchdog._setup_network_monitoring('target-1')

        # Should not register another callback
        session.cdp_client.register.Network.responseReceived.assert_not_called()

    async def test_skip_when_auto_download_disabled(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = False

        await watchdog._setup_network_monitoring('target-1')

        session.cdp_client.register.Network.responseReceived.assert_not_called()

    async def test_enables_network_domain(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = True
        mock_cdp = MagicMock()
        mock_cdp.session_id = 'sid-123'
        session.get_or_create_cdp_session.return_value = mock_cdp

        await watchdog._setup_network_monitoring('target-1')

        session.cdp_client.send.Network.enable.assert_called_once()
        assert 'target-1' in watchdog._network_monitored_targets


class TestTrackDownload:
    """Tests for _track_download."""

    def test_tracks_existing_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
            f.write(b'test content')
            temp_path = f.name

        try:
            watchdog, _ = _make_watchdog()
            watchdog.event_bus = MagicMock()

            watchdog._track_download(temp_path)

            watchdog.event_bus.dispatch.assert_called_once()
        finally:
            os.unlink(temp_path)

    def test_nonexistent_file(self):
        watchdog, _ = _make_watchdog()
        watchdog.event_bus = MagicMock()

        watchdog._track_download('/nonexistent/file.pdf')

        watchdog.event_bus.dispatch.assert_not_called()


@pytest.mark.asyncio
class TestDownloadFileFromUrl:
    """Tests for download_file_from_url."""

    async def test_no_downloads_path(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.downloads_path = None

        result = await watchdog.download_file_from_url('https://example.com/file.pdf', 'target-1')
        assert result is None

    async def test_already_downloaded_returns_cached(self):
        watchdog, _ = _make_watchdog()
        watchdog._session_pdf_urls['https://example.com/file.pdf'] = '/cached/path.pdf'

        result = await watchdog.download_file_from_url('https://example.com/file.pdf', 'target-1')
        assert result == '/cached/path.pdf'

    async def test_successful_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            watchdog, session = _make_watchdog(tmpdir)
            watchdog.event_bus = MagicMock()

            mock_cdp = MagicMock()
            mock_cdp.session_id = 'sid-123'
            mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
                'result': {'value': {'data': list(b'PDF content here'), 'responseSize': 16}}
            })
            session.get_or_create_cdp_session.return_value = mock_cdp

            result = await watchdog.download_file_from_url(
                'https://example.com/document.pdf',
                'target-1',
                content_type='application/pdf',
            )

            assert result is not None
            assert os.path.exists(result)
            watchdog.event_bus.dispatch.assert_called_once()

    async def test_download_timeout(self):
        watchdog, session = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.session_id = 'sid-123'
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=asyncio.TimeoutError())
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await watchdog.download_file_from_url('https://example.com/file.pdf', 'target-1')
        assert result is None

    async def test_download_no_data(self):
        watchdog, session = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.session_id = 'sid-123'
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
            'result': {'value': {'data': [], 'responseSize': 0}}
        })
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await watchdog.download_file_from_url('https://example.com/file.pdf', 'target-1')
        assert result is None

    async def test_filename_from_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            watchdog, session = _make_watchdog(tmpdir)
            watchdog.event_bus = MagicMock()

            mock_cdp = MagicMock()
            mock_cdp.session_id = 'sid-123'
            mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
                'result': {'value': {'data': list(b'test'), 'responseSize': 4}}
            })
            session.get_or_create_cdp_session.return_value = mock_cdp

            result = await watchdog.download_file_from_url(
                'https://example.com/report.pdf?token=abc',
                'target-1',
            )

            assert result is not None
            assert 'report.pdf' in result

    async def test_fallback_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            watchdog, session = _make_watchdog(tmpdir)
            watchdog.event_bus = MagicMock()

            mock_cdp = MagicMock()
            mock_cdp.session_id = 'sid-123'
            mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
                'result': {'value': {'data': list(b'test'), 'responseSize': 4}}
            })
            session.get_or_create_cdp_session.return_value = mock_cdp

            result = await watchdog.download_file_from_url(
                'https://example.com/download',
                'target-1',
                content_type='application/pdf',
            )

            assert result is not None
            assert 'document.pdf' in result

    async def test_unique_filename_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an existing file
            Path(os.path.join(tmpdir, 'report.pdf')).touch()

            watchdog, session = _make_watchdog(tmpdir)
            watchdog.event_bus = MagicMock()

            mock_cdp = MagicMock()
            mock_cdp.session_id = 'sid-123'
            mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
                'result': {'value': {'data': list(b'test'), 'responseSize': 4}}
            })
            session.get_or_create_cdp_session.return_value = mock_cdp

            result = await watchdog.download_file_from_url(
                'https://example.com/report.pdf',
                'target-1',
            )

            assert result is not None
            assert 'report (1).pdf' in result
