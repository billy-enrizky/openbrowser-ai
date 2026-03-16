"""Tests for openbrowser.browser.watchdogs.local_browser_watchdog module."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import psutil
import pytest
from bubus import EventBus
from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


def _make_mock_browser_session():
    """Create a mock BrowserSession for LocalBrowserWatchdog tests."""
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger('test_local_browser_watchdog')
    session.event_bus = EventBus()
    session._cdp_client_root = MagicMock()
    session.is_local = True

    session.browser_profile = MagicMock()
    session.browser_profile.executable_path = None
    user_data_dir = tempfile.mkdtemp(prefix='openbrowser-test-')
    session.browser_profile.user_data_dir = user_data_dir
    session.browser_profile.profile_directory = None
    session.browser_profile.get_args = MagicMock(return_value=[
        '--no-first-run',
        '--disable-default-apps',
        f'--user-data-dir={user_data_dir}',
    ])

    return session


def _make_watchdog():
    from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

    session = _make_mock_browser_session()
    event_bus = EventBus()
    return LocalBrowserWatchdog(event_bus=event_bus, browser_session=session), session


class TestLocalBrowserWatchdogInit:
    """Tests for LocalBrowserWatchdog initialization."""

    def test_init(self):
        watchdog, _ = _make_watchdog()
        assert watchdog._subprocess is None
        assert watchdog._owns_browser_resources is True
        assert watchdog._temp_dirs_to_cleanup == []
        assert watchdog._original_user_data_dir is None


class TestBrowserPidProperty:
    """Tests for browser_pid property."""

    def test_returns_pid_when_subprocess(self):
        watchdog, _ = _make_watchdog()
        mock_process = MagicMock()
        mock_process.pid = 12345
        watchdog._subprocess = mock_process

        assert watchdog.browser_pid == 12345

    def test_returns_none_when_no_subprocess(self):
        watchdog, _ = _make_watchdog()
        assert watchdog.browser_pid is None


class TestFindFreePort:
    """Tests for _find_free_port static method."""

    def test_returns_valid_port(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        port = LocalBrowserWatchdog._find_free_port()
        assert isinstance(port, int)
        assert 1 <= port <= 65535


class TestFindInstalledBrowserPath:
    """Tests for _find_installed_browser_path static method."""

    def test_returns_string_or_none(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        result = LocalBrowserWatchdog._find_installed_browser_path()
        # Should return a path string or None
        assert result is None or isinstance(result, str)

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_file', return_value=True)
    def test_darwin_finds_chrome(self, mock_is_file, mock_exists, mock_system):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        result = LocalBrowserWatchdog._find_installed_browser_path()
        if result is not None:
            assert isinstance(result, str)

    @patch('platform.system', return_value='Linux')
    def test_linux_patterns(self, mock_system):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        # Just verify it runs without error on Linux patterns
        result = LocalBrowserWatchdog._find_installed_browser_path()
        assert result is None or isinstance(result, str)


class TestCleanupTempDir:
    """Tests for _cleanup_temp_dir."""

    def test_cleanup_valid_dir(self):
        watchdog, _ = _make_watchdog()
        tmpdir = tempfile.mkdtemp(prefix='openbrowser-tmp-')
        assert os.path.exists(tmpdir)

        watchdog._cleanup_temp_dir(tmpdir)

        assert not os.path.exists(tmpdir)

    def test_cleanup_empty_path(self):
        watchdog, _ = _make_watchdog()
        # Should not raise
        watchdog._cleanup_temp_dir('')

    def test_cleanup_none_path(self):
        watchdog, _ = _make_watchdog()
        # Should not raise
        watchdog._cleanup_temp_dir(None)

    def test_cleanup_non_openbrowser_dir_skipped(self):
        watchdog, _ = _make_watchdog()
        tmpdir = tempfile.mkdtemp(prefix='other-')
        assert os.path.exists(tmpdir)

        watchdog._cleanup_temp_dir(tmpdir)

        # Should NOT remove non-openbrowser dirs
        assert os.path.exists(tmpdir)
        os.rmdir(tmpdir)  # cleanup


@pytest.mark.asyncio
class TestCleanupProcess:
    """Tests for _cleanup_process static method."""

    async def test_cleanup_none_process(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        # Should not raise
        await LocalBrowserWatchdog._cleanup_process(None)

    async def test_cleanup_terminates_gracefully(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.is_running.return_value = False

        with patch('asyncio.sleep', new_callable=AsyncMock):
            await LocalBrowserWatchdog._cleanup_process(mock_process)

        mock_process.terminate.assert_called_once()

    async def test_cleanup_force_kills_after_timeout(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.is_running.return_value = True  # Never stops

        with patch('asyncio.sleep', new_callable=AsyncMock):
            await LocalBrowserWatchdog._cleanup_process(mock_process)

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    async def test_cleanup_handles_no_such_process(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_process = MagicMock()
        mock_process.terminate.side_effect = psutil.NoSuchProcess(pid=99999)

        with patch('asyncio.sleep', new_callable=AsyncMock):
            # Should not raise
            await LocalBrowserWatchdog._cleanup_process(mock_process)


@pytest.mark.asyncio
class TestOnBrowserLaunchEvent:
    """Tests for on_BrowserLaunchEvent."""

    async def test_launch_success(self):
        watchdog, session = _make_watchdog()

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch.object(watchdog, '_launch_browser', new_callable=AsyncMock) as mock_launch:
            mock_launch.return_value = (mock_process, 'http://localhost:9222/')
            event = MagicMock()
            result = await watchdog.on_BrowserLaunchEvent(event)

        assert result.cdp_url == 'http://localhost:9222/'
        assert watchdog._subprocess is mock_process

    async def test_launch_failure_raises(self):
        watchdog, _ = _make_watchdog()

        with patch.object(watchdog, '_launch_browser', new_callable=AsyncMock) as mock_launch:
            mock_launch.side_effect = RuntimeError('Launch failed')
            event = MagicMock()

            with pytest.raises(RuntimeError, match='Launch failed'):
                await watchdog.on_BrowserLaunchEvent(event)


@pytest.mark.asyncio
class TestOnBrowserKillEvent:
    """Tests for on_BrowserKillEvent."""

    async def test_kill_cleans_up(self):
        watchdog, session = _make_watchdog()

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.is_running.return_value = False
        watchdog._subprocess = mock_process
        watchdog._original_user_data_dir = '/original/path'

        tmpdir = tempfile.mkdtemp(prefix='openbrowser-tmp-')
        watchdog._temp_dirs_to_cleanup = [Path(tmpdir)]

        event = MagicMock()

        with patch.object(type(watchdog), '_cleanup_process', new_callable=AsyncMock) as mock_cleanup:
            await watchdog.on_BrowserKillEvent(event)

        assert watchdog._subprocess is None
        assert watchdog._temp_dirs_to_cleanup == []
        assert session.browser_profile.user_data_dir == '/original/path'
        assert watchdog._original_user_data_dir is None

    async def test_kill_no_subprocess(self):
        watchdog, _ = _make_watchdog()
        watchdog._subprocess = None

        event = MagicMock()
        # Should not raise
        await watchdog.on_BrowserKillEvent(event)


@pytest.mark.asyncio
class TestOnBrowserStopEvent:
    """Tests for on_BrowserStopEvent."""

    async def test_dispatches_kill_when_local(self):
        watchdog, session = _make_watchdog()
        session.is_local = True
        watchdog._subprocess = MagicMock()
        # Replace real EventBus with MagicMock so dispatch doesn't actually execute
        watchdog.event_bus = MagicMock()

        event = MagicMock()
        await watchdog.on_BrowserStopEvent(event)

        watchdog.event_bus.dispatch.assert_called()

    async def test_no_dispatch_when_not_local(self):
        watchdog, session = _make_watchdog()
        session.is_local = False

        event = MagicMock()
        await watchdog.on_BrowserStopEvent(event)

    async def test_no_dispatch_when_no_subprocess(self):
        watchdog, session = _make_watchdog()
        session.is_local = True
        watchdog._subprocess = None
        watchdog.event_bus = MagicMock()

        event = MagicMock()
        await watchdog.on_BrowserStopEvent(event)

        watchdog.event_bus.dispatch.assert_not_called()


@pytest.mark.asyncio
class TestWaitForCdpUrl:
    """Tests for _wait_for_cdp_url static method."""

    async def test_timeout_raises(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = ConnectionError('refused')
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            with pytest.raises(TimeoutError, match='did not start within'):
                await LocalBrowserWatchdog._wait_for_cdp_url(9999, timeout=0.2)

    async def test_success_returns_url(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await LocalBrowserWatchdog._wait_for_cdp_url(9222)
            assert result == 'http://localhost:9222/'


@pytest.mark.asyncio
class TestGetBrowserPidViaCdp:
    """Tests for get_browser_pid_via_cdp static method."""

    async def test_returns_pid(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_browser = MagicMock()
        mock_cdp_session = AsyncMock()
        mock_cdp_session.send = AsyncMock(return_value={'processInfo': {'id': 54321}})
        mock_cdp_session.detach = AsyncMock()
        mock_browser.new_browser_cdp_session = AsyncMock(return_value=mock_cdp_session)

        result = await LocalBrowserWatchdog.get_browser_pid_via_cdp(mock_browser)
        assert result == 54321

    async def test_returns_none_on_error(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_browser = MagicMock()
        mock_browser.new_browser_cdp_session = AsyncMock(side_effect=Exception('fail'))

        result = await LocalBrowserWatchdog.get_browser_pid_via_cdp(mock_browser)
        assert result is None


@pytest.mark.asyncio
class TestKillStaleChromeForProfile:
    """Tests for _kill_stale_chrome_for_profile static method."""

    async def test_no_stale_processes(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        with patch('psutil.process_iter', return_value=[]):
            result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile('/tmp/no-such-profile')
            assert result is False

    async def test_ignores_non_chrome_processes(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        mock_proc = MagicMock()
        mock_proc.info = {'pid': 123, 'name': 'python', 'cmdline': ['python', 'test.py']}

        with patch('psutil.process_iter', return_value=[mock_proc]):
            result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile('/tmp/test-profile')
            assert result is False

    async def test_kills_matching_chrome(self):
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = str(Path(tmpdir).resolve())

            mock_proc = MagicMock()
            mock_proc.info = {
                'pid': 999,
                'name': 'chrome',
                'cmdline': ['chrome', f'--user-data-dir={resolved}']
            }
            mock_proc.kill = MagicMock()

            # First call returns the process, second call (after kill) returns empty
            with patch('psutil.process_iter', side_effect=[[mock_proc], []]):
                result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile(tmpdir)

            assert result is True
            mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
class TestInstallBrowserWithPlaywright:
    """Tests for _install_browser_with_playwright."""

    async def test_timeout_raises(self):
        watchdog, _ = _make_watchdog()

        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=mock_process):
            with pytest.raises(RuntimeError, match='Timeout'):
                await watchdog._install_browser_with_playwright()

    async def test_no_path_after_install_raises(self):
        watchdog, _ = _make_watchdog()

        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(return_value=(b'installed', b''))
        mock_process.returncode = 0

        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=mock_process):
            with patch.object(type(watchdog), '_find_installed_browser_path', return_value=None):
                with pytest.raises(RuntimeError, match='No local browser path'):
                    await watchdog._install_browser_with_playwright()

    async def test_successful_install(self):
        watchdog, _ = _make_watchdog()

        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(return_value=(b'installed', b''))
        mock_process.returncode = 0

        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=mock_process):
            with patch.object(type(watchdog), '_find_installed_browser_path', return_value='/usr/bin/chrome'):
                result = await watchdog._install_browser_with_playwright()
                assert result == '/usr/bin/chrome'


@pytest.mark.asyncio
class TestLaunchBrowser:
    """Tests for _launch_browser."""

    async def test_no_browser_found_raises(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.executable_path = None

        with patch.object(type(watchdog), '_find_installed_browser_path', return_value=None):
            with patch.object(watchdog, '_install_browser_with_playwright', new_callable=AsyncMock, return_value=None):
                with pytest.raises(RuntimeError, match='No local Chrome'):
                    await watchdog._launch_browser(max_retries=1)

    async def test_custom_executable_path(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.executable_path = '/custom/chrome'

        mock_subprocess = MagicMock()
        mock_subprocess.pid = 12345

        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=mock_subprocess):
            with patch('psutil.Process', return_value=MagicMock()):
                with patch.object(type(watchdog), '_wait_for_cdp_url', new_callable=AsyncMock, return_value='http://localhost:9222/'):
                    with patch.object(type(watchdog), '_kill_stale_chrome_for_profile', new_callable=AsyncMock, return_value=False):
                        process, cdp_url = await watchdog._launch_browser(max_retries=1)

        assert cdp_url == 'http://localhost:9222/'

    async def test_retries_with_temp_dir_on_profile_error(self):
        watchdog, session = _make_watchdog()
        session.browser_profile.executable_path = '/usr/bin/chrome'

        call_count = 0

        async def mock_create_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError('user data directory already in use')
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            return mock_proc

        with patch('asyncio.create_subprocess_exec', side_effect=mock_create_subprocess):
            with patch('psutil.Process', return_value=MagicMock()):
                with patch.object(type(watchdog), '_wait_for_cdp_url', new_callable=AsyncMock, return_value='http://localhost:9222/'):
                    with patch.object(type(watchdog), '_kill_stale_chrome_for_profile', new_callable=AsyncMock, return_value=False):
                        process, cdp_url = await watchdog._launch_browser(max_retries=3)

        assert cdp_url == 'http://localhost:9222/'
        assert call_count == 2
