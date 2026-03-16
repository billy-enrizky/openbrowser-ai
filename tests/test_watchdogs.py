"""Tests for smaller watchdog modules:
- permissions_watchdog
- screenshot_watchdog
- popups_watchdog
- aboutblank_watchdog
- security_watchdog
- dom_watchdog
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus
from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


# ====================================================================
# Shared mock helpers
# ====================================================================

def _make_mock_browser_session():
    """Create a mock BrowserSession for watchdog tests."""
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger('test_watchdogs')
    session.event_bus = EventBus()
    session._cdp_client_root = MagicMock()
    session.agent_focus = None
    session.get_or_create_cdp_session = AsyncMock()
    session.remove_highlights = AsyncMock()
    session.add_highlights = AsyncMock()
    session._cdp_get_all_pages = AsyncMock(return_value=[])
    session._cdp_close_page = AsyncMock()
    session.get_current_page_url = AsyncMock(return_value='https://example.com')
    session.get_current_page_title = AsyncMock(return_value='Example')
    session.get_tabs = AsyncMock(return_value=[])
    session._cached_browser_state_summary = None
    session._closed_popup_messages = []
    session.cdp_client = MagicMock()
    session.id = 'test-session-id-1234'
    session.is_local = True

    # Browser profile defaults
    session.browser_profile = MagicMock()
    session.browser_profile.permissions = []
    session.browser_profile.allowed_domains = None
    session.browser_profile.prohibited_domains = None
    session.browser_profile.block_ip_addresses = False
    session.browser_profile.viewport = {'width': 1280, 'height': 720}
    session.browser_profile.highlight_elements = False
    session.browser_profile.dom_highlight_elements = False
    session.browser_profile.cross_origin_iframes = False
    session.browser_profile.paint_order_filtering = True
    session.browser_profile.max_iframes = 3
    session.browser_profile.max_iframe_depth = 2
    session.browser_profile.filter_highlight_ids = None
    session.browser_profile.minimum_wait_page_load_time = 0
    session.browser_profile.wait_for_network_idle_page_load_time = 0
    session.browser_profile.auto_download_pdfs = False
    session.browser_profile.downloads_path = '/tmp/downloads'

    return session


# ====================================================================
# PermissionsWatchdog Tests
# ====================================================================

class TestPermissionsWatchdog:
    """Tests for PermissionsWatchdog."""

    @pytest.mark.asyncio
    async def test_no_permissions_to_grant(self):
        from openbrowser.browser.watchdogs.permissions_watchdog import PermissionsWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.permissions = []
        event_bus = EventBus()
        watchdog = PermissionsWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        await watchdog.on_BrowserConnectedEvent(event)

        # Should not call grantPermissions
        session.cdp_client.send.Browser.grantPermissions.assert_not_called()

    @pytest.mark.asyncio
    async def test_grants_permissions(self):
        from openbrowser.browser.watchdogs.permissions_watchdog import PermissionsWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.permissions = ['geolocation', 'notifications']
        session.cdp_client.send.Browser.grantPermissions = AsyncMock()
        event_bus = EventBus()
        watchdog = PermissionsWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        await watchdog.on_BrowserConnectedEvent(event)

        session.cdp_client.send.Browser.grantPermissions.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_grant_error_gracefully(self):
        from openbrowser.browser.watchdogs.permissions_watchdog import PermissionsWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.permissions = ['geolocation']
        session.cdp_client.send.Browser.grantPermissions = AsyncMock(side_effect=Exception('fail'))
        event_bus = EventBus()
        watchdog = PermissionsWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        # Should not raise
        await watchdog.on_BrowserConnectedEvent(event)


# ====================================================================
# ScreenshotWatchdog Tests
# ====================================================================

class TestScreenshotWatchdog:
    """Tests for ScreenshotWatchdog."""

    @pytest.mark.asyncio
    async def test_screenshot_success(self):
        from openbrowser.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog

        session = _make_mock_browser_session()
        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={'data': 'base64screenshot'}
        )
        mock_cdp_session.session_id = 'sid-123'
        session.get_or_create_cdp_session.return_value = mock_cdp_session

        event_bus = EventBus()
        watchdog = ScreenshotWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        result = await watchdog.on_ScreenshotEvent(event)

        assert result == 'base64screenshot'
        session.remove_highlights.assert_called_once()

    @pytest.mark.asyncio
    async def test_screenshot_missing_data_raises(self):
        from openbrowser.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog

        session = _make_mock_browser_session()
        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={}
        )
        mock_cdp_session.session_id = 'sid-123'
        session.get_or_create_cdp_session.return_value = mock_cdp_session

        event_bus = EventBus()
        watchdog = ScreenshotWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        with pytest.raises(Exception):
            await watchdog.on_ScreenshotEvent(event)

    @pytest.mark.asyncio
    async def test_screenshot_removes_highlights_on_failure(self):
        from openbrowser.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog

        session = _make_mock_browser_session()
        session.get_or_create_cdp_session.side_effect = Exception('no session')

        event_bus = EventBus()
        watchdog = ScreenshotWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        with pytest.raises(Exception):
            await watchdog.on_ScreenshotEvent(event)

        session.remove_highlights.assert_called_once()


# ====================================================================
# PopupsWatchdog Tests
# ====================================================================

class TestPopupsWatchdog:
    """Tests for PopupsWatchdog."""

    @pytest.mark.asyncio
    async def test_setup_dialog_handler(self):
        from openbrowser.browser.watchdogs.popups_watchdog import PopupsWatchdog

        session = _make_mock_browser_session()
        mock_cdp_session = MagicMock()
        mock_cdp_session.cdp_client.send.Page.enable = AsyncMock()
        mock_cdp_session.cdp_client.register.Page.javascriptDialogOpening = MagicMock()
        mock_cdp_session.session_id = 'sid-123'
        session.get_or_create_cdp_session.return_value = mock_cdp_session
        session._cdp_client_root.send.Page.enable = AsyncMock()
        session._cdp_client_root.register.Page.javascriptDialogOpening = MagicMock()

        event_bus = EventBus()
        watchdog = PopupsWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        event.target_id = 'target-1'

        await watchdog.on_TabCreatedEvent(event)

        assert 'target-1' in watchdog._dialog_listeners_registered
        mock_cdp_session.cdp_client.register.Page.javascriptDialogOpening.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_already_registered_target(self):
        from openbrowser.browser.watchdogs.popups_watchdog import PopupsWatchdog

        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = PopupsWatchdog(event_bus=event_bus, browser_session=session)
        watchdog._dialog_listeners_registered.add('target-1')

        event = MagicMock()
        event.target_id = 'target-1'

        await watchdog.on_TabCreatedEvent(event)

        # Should not call get_or_create_cdp_session
        session.get_or_create_cdp_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_setup_error_gracefully(self):
        from openbrowser.browser.watchdogs.popups_watchdog import PopupsWatchdog

        session = _make_mock_browser_session()
        session.get_or_create_cdp_session.side_effect = Exception('fail')

        event_bus = EventBus()
        watchdog = PopupsWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        event.target_id = 'target-1'

        # Should not raise
        await watchdog.on_TabCreatedEvent(event)


# ====================================================================
# AboutBlankWatchdog Tests
# ====================================================================

class TestAboutBlankWatchdog:
    """Tests for AboutBlankWatchdog."""

    @pytest.mark.asyncio
    async def test_stop_event_sets_stopping(self):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = AboutBlankWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        await watchdog.on_BrowserStopEvent(event)
        assert watchdog._stopping is True

    @pytest.mark.asyncio
    async def test_stopped_event_sets_stopping(self):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = AboutBlankWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        await watchdog.on_BrowserStoppedEvent(event)
        assert watchdog._stopping is True

    @pytest.mark.asyncio
    async def test_tab_created_about_blank_shows_screensaver(self):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

        session = _make_mock_browser_session()
        session._cdp_get_all_pages.return_value = [
            {'targetId': 'target-1', 'url': 'about:blank'}
        ]
        mock_temp_session = MagicMock()
        mock_temp_session.cdp_client.send.Runtime.evaluate = AsyncMock()
        mock_temp_session.session_id = 'sid-123'
        session.get_or_create_cdp_session.return_value = mock_temp_session

        event_bus = EventBus()
        watchdog = AboutBlankWatchdog(event_bus=event_bus, browser_session=session)
        # Replace real EventBus with MagicMock to prevent unawaited coroutine from dispatch
        watchdog.event_bus = MagicMock()

        event = MagicMock()
        event.url = 'about:blank'
        await watchdog.on_TabCreatedEvent(event)

        # Should have called Runtime.evaluate to inject screensaver
        mock_temp_session.cdp_client.send.Runtime.evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_tab_created_non_blank_no_screensaver(self):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = AboutBlankWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        event.url = 'https://example.com'
        await watchdog.on_TabCreatedEvent(event)

        # Should not call _cdp_get_all_pages since url is not about:blank
        session._cdp_get_all_pages.assert_not_called()

    @pytest.mark.asyncio
    async def test_tab_closed_when_stopping(self):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = AboutBlankWatchdog(event_bus=event_bus, browser_session=session)
        watchdog._stopping = True

        event = MagicMock()
        await watchdog.on_TabClosedEvent(event)

        # Should not check pages when stopping
        session._cdp_get_all_pages.assert_not_called()

    @pytest.mark.asyncio
    async def test_tab_closed_last_tab_creates_new(self):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

        session = _make_mock_browser_session()
        session._cdp_get_all_pages.return_value = [{'targetId': 'target-1', 'url': 'about:blank'}]

        event_bus = EventBus()
        # dispatch must return an awaitable (the source does `await navigate_event`)
        event_bus.dispatch = AsyncMock()

        watchdog = AboutBlankWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        # Patch _show_dvd_screensaver to avoid complex CDP setup
        watchdog._show_dvd_screensaver_on_about_blank_tabs = AsyncMock()
        await watchdog.on_TabClosedEvent(event)

        event_bus.dispatch.assert_called()

    @pytest.mark.asyncio
    async def test_tab_closed_multiple_tabs_checks_about_blank(self):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

        session = _make_mock_browser_session()
        session._cdp_get_all_pages.return_value = [
            {'targetId': 't1', 'url': 'https://example.com'},
            {'targetId': 't2', 'url': 'https://other.com'},
        ]

        event_bus = EventBus()
        watchdog = AboutBlankWatchdog(event_bus=event_bus, browser_session=session)
        watchdog._check_and_ensure_about_blank_tab = AsyncMock()

        event = MagicMock()
        await watchdog.on_TabClosedEvent(event)

        watchdog._check_and_ensure_about_blank_tab.assert_called_once()

    @pytest.mark.asyncio
    async def test_attach_to_target_is_noop(self):
        from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog

        session = _make_mock_browser_session()
        event_bus = EventBus()
        watchdog = AboutBlankWatchdog(event_bus=event_bus, browser_session=session)

        # Should not raise
        await watchdog.attach_to_target('target-1')


# ====================================================================
# SecurityWatchdog Tests
# ====================================================================

class TestSecurityWatchdog:
    """Tests for SecurityWatchdog."""

    def _make_watchdog(self, allowed_domains=None, prohibited_domains=None, block_ip=False):
        from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = allowed_domains
        session.browser_profile.prohibited_domains = prohibited_domains
        session.browser_profile.block_ip_addresses = block_ip

        event_bus = EventBus()
        return SecurityWatchdog(event_bus=event_bus, browser_session=session)

    def test_internal_urls_always_allowed(self):
        watchdog = self._make_watchdog(allowed_domains={'example.com'})
        assert watchdog._is_url_allowed('about:blank') is True
        assert watchdog._is_url_allowed('chrome://new-tab-page/') is True
        assert watchdog._is_url_allowed('chrome://newtab/') is True

    def test_no_restrictions_allows_all(self):
        watchdog = self._make_watchdog()
        assert watchdog._is_url_allowed('https://anything.com') is True

    def test_allowed_domains_set(self):
        watchdog = self._make_watchdog(allowed_domains={'example.com', 'test.com'})
        assert watchdog._is_url_allowed('https://example.com/page') is True
        assert watchdog._is_url_allowed('https://www.example.com/page') is True
        assert watchdog._is_url_allowed('https://evil.com') is False

    def test_allowed_domains_list_with_glob(self):
        watchdog = self._make_watchdog(allowed_domains=['*.example.com'])
        assert watchdog._is_url_allowed('https://sub.example.com/page') is True
        assert watchdog._is_url_allowed('https://example.com/page') is True
        assert watchdog._is_url_allowed('https://evil.com') is False

    def test_prohibited_domains_set(self):
        watchdog = self._make_watchdog(prohibited_domains={'evil.com'})
        assert watchdog._is_url_allowed('https://evil.com') is False
        assert watchdog._is_url_allowed('https://www.evil.com') is False
        assert watchdog._is_url_allowed('https://good.com') is True

    def test_prohibited_domains_list_with_glob(self):
        watchdog = self._make_watchdog(prohibited_domains=['*.evil.com'])
        assert watchdog._is_url_allowed('https://sub.evil.com') is False
        assert watchdog._is_url_allowed('https://good.com') is True

    def test_block_ip_addresses(self):
        watchdog = self._make_watchdog(block_ip=True)
        assert watchdog._is_url_allowed('http://192.168.1.1') is False
        assert watchdog._is_url_allowed('http://127.0.0.1:8080') is False
        assert watchdog._is_url_allowed('https://example.com') is True

    def test_data_and_blob_urls_allowed(self):
        watchdog = self._make_watchdog(allowed_domains={'example.com'})
        assert watchdog._is_url_allowed('data:text/html,<h1>hello</h1>') is True
        assert watchdog._is_url_allowed('blob:https://example.com/abc') is True

    def test_invalid_url_returns_false(self):
        watchdog = self._make_watchdog(allowed_domains={'example.com'})
        assert watchdog._is_url_allowed('') is False

    def test_is_root_domain(self):
        watchdog = self._make_watchdog()
        assert watchdog._is_root_domain('example.com') is True
        assert watchdog._is_root_domain('sub.example.com') is False
        assert watchdog._is_root_domain('*.example.com') is False
        assert watchdog._is_root_domain('http://example.com') is False

    def test_get_domain_variants(self):
        watchdog = self._make_watchdog()
        assert watchdog._get_domain_variants('example.com') == ('example.com', 'www.example.com')
        assert watchdog._get_domain_variants('www.example.com') == ('www.example.com', 'example.com')

    def test_is_ip_address(self):
        watchdog = self._make_watchdog()
        assert watchdog._is_ip_address('192.168.1.1') is True
        assert watchdog._is_ip_address('::1') is True
        assert watchdog._is_ip_address('example.com') is False

    def test_url_match_full_url_pattern(self):
        watchdog = self._make_watchdog()
        assert watchdog._is_url_match('https://example.com/path', 'example.com', 'https', 'https://example.com') is True

    def test_url_match_domain_only(self):
        watchdog = self._make_watchdog()
        assert watchdog._is_url_match('https://example.com', 'example.com', 'https', 'example.com') is True

    def test_url_match_root_domain_www(self):
        watchdog = self._make_watchdog()
        assert watchdog._is_url_match('https://www.example.com', 'www.example.com', 'https', 'example.com') is True

    def test_url_match_glob_star_prefix(self):
        watchdog = self._make_watchdog()
        assert watchdog._is_url_match('https://sub.example.com', 'sub.example.com', 'https', '*.example.com') is True

    def test_url_match_glob_star_suffix(self):
        watchdog = self._make_watchdog()
        assert watchdog._is_url_match('brave://settings/privacy', 'settings', 'brave', 'brave://*') is True

    @pytest.mark.asyncio
    async def test_navigate_event_blocks_disallowed(self):
        from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = {'example.com'}
        event_bus = EventBus()
        watchdog = SecurityWatchdog(event_bus=event_bus, browser_session=session)
        # Replace real EventBus to prevent unawaited coroutine from dispatch
        watchdog.event_bus = MagicMock()

        event = MagicMock()
        event.url = 'https://evil.com'

        with pytest.raises(ValueError, match='blocked by security policy'):
            await watchdog.on_NavigateToUrlEvent(event)

    @pytest.mark.asyncio
    async def test_navigate_event_allows_permitted(self):
        from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = {'example.com'}
        event_bus = EventBus()
        watchdog = SecurityWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        event.url = 'https://example.com/page'

        # Should not raise
        await watchdog.on_NavigateToUrlEvent(event)

    @pytest.mark.asyncio
    async def test_navigation_complete_redirects_blocked(self):
        from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = {'example.com'}
        mock_cdp = MagicMock()
        mock_cdp.cdp_client.send.Page.navigate = AsyncMock()
        mock_cdp.session_id = 'sid-123'
        session.get_or_create_cdp_session.return_value = mock_cdp
        event_bus = EventBus()
        watchdog = SecurityWatchdog(event_bus=event_bus, browser_session=session)
        # Replace real EventBus to prevent unawaited coroutine from dispatch
        watchdog.event_bus = MagicMock()

        event = MagicMock()
        event.url = 'https://evil.com'
        event.target_id = 'target-1'

        await watchdog.on_NavigationCompleteEvent(event)

        mock_cdp.cdp_client.send.Page.navigate.assert_called_once()

    @pytest.mark.asyncio
    async def test_tab_created_closes_blocked(self):
        from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = {'example.com'}
        event_bus = EventBus()
        watchdog = SecurityWatchdog(event_bus=event_bus, browser_session=session)
        # Replace real EventBus to prevent unawaited coroutine from dispatch
        watchdog.event_bus = MagicMock()

        event = MagicMock()
        event.url = 'https://evil.com'
        event.target_id = 'target-1'

        await watchdog.on_TabCreatedEvent(event)

        session._cdp_close_page.assert_called_once_with('target-1')

    @pytest.mark.asyncio
    async def test_tab_created_allows_permitted(self):
        from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

        session = _make_mock_browser_session()
        session.browser_profile.allowed_domains = {'example.com'}
        event_bus = EventBus()
        watchdog = SecurityWatchdog(event_bus=event_bus, browser_session=session)

        event = MagicMock()
        event.url = 'https://example.com'
        event.target_id = 'target-1'

        await watchdog.on_TabCreatedEvent(event)

        session._cdp_close_page.assert_not_called()


# ====================================================================
# DOMWatchdog Tests
# ====================================================================

class TestDOMWatchdog:
    """Tests for DOMWatchdog."""

    def _make_watchdog(self):
        from openbrowser.browser.watchdogs.dom_watchdog import DOMWatchdog

        session = _make_mock_browser_session()
        event_bus = EventBus()
        return DOMWatchdog(event_bus=event_bus, browser_session=session), session

    @pytest.mark.asyncio
    async def test_tab_created_returns_none(self):
        watchdog, _ = self._make_watchdog()
        event = MagicMock()
        result = await watchdog.on_TabCreatedEvent(event)
        assert result is None

    def test_clear_cache(self):
        watchdog, _ = self._make_watchdog()
        watchdog.selector_map = {1: MagicMock()}
        watchdog.current_dom_state = MagicMock()
        watchdog.enhanced_dom_tree = MagicMock()

        watchdog.clear_cache()

        assert watchdog.selector_map is None
        assert watchdog.current_dom_state is None
        assert watchdog.enhanced_dom_tree is None

    def test_is_file_input(self):
        watchdog, _ = self._make_watchdog()
        element = MagicMock()
        element.node_name = 'INPUT'
        element.attributes = {'type': 'file'}

        assert watchdog.is_file_input(element) is True

        element.attributes = {'type': 'text'}
        assert watchdog.is_file_input(element) is False

        element.node_name = 'DIV'
        element.attributes = {'type': 'file'}
        assert watchdog.is_file_input(element) is False

    @pytest.mark.asyncio
    async def test_get_element_by_index_empty(self):
        watchdog, _ = self._make_watchdog()
        watchdog._build_dom_tree_without_highlights = AsyncMock(return_value=None)
        watchdog.selector_map = None

        result = await watchdog.get_element_by_index(1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_element_by_index_found(self):
        watchdog, _ = self._make_watchdog()
        mock_node = MagicMock()
        watchdog.selector_map = {1: mock_node}

        result = await watchdog.get_element_by_index(1)
        assert result is mock_node

    def test_get_recent_events_str(self):
        watchdog, session = self._make_watchdog()
        # Use spec to restrict attributes so hasattr checks for url/error_message/target_id
        # return False, preventing json.dumps from trying to serialize MagicMock objects
        mock_event = MagicMock(spec=['event_type', 'event_created_at', 'event_id'])
        mock_event.event_type = 'TestEvent'
        mock_event.event_created_at.timestamp.return_value = 1234567890.0
        mock_event.event_created_at.isoformat.return_value = '2025-01-01T00:00:00'
        mock_event.event_id = 'evt-1234'

        session.event_bus.event_history = {'evt-1234': mock_event}

        result = watchdog._get_recent_events_str(limit=5)
        assert result is not None
        assert 'TestEvent' in result

    def test_get_recent_events_str_empty(self):
        watchdog, session = self._make_watchdog()
        session.event_bus.event_history = {}
        watchdog.event_bus = session.event_bus

        result = watchdog._get_recent_events_str()
        assert result == '[]'

    def test_detect_pagination_buttons(self):
        """Test pagination detection with mock selector map."""
        watchdog, _ = self._make_watchdog()

        # Empty selector map
        result = watchdog._detect_pagination_buttons({})
        assert result == []

    @pytest.mark.asyncio
    async def test_aexit_cleans_up(self):
        watchdog, _ = self._make_watchdog()
        mock_dom_service = MagicMock()
        mock_dom_service.__aexit__ = AsyncMock()
        watchdog._dom_service = mock_dom_service

        await watchdog.__aexit__(None, None, None)

        mock_dom_service.__aexit__.assert_called_once()
        assert watchdog._dom_service is None

    def test_del_cleans_dom_service(self):
        watchdog, _ = self._make_watchdog()
        watchdog._dom_service = MagicMock()

        watchdog.__del__()

        assert watchdog._dom_service is None
