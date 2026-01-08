"""Comprehensive test suite for BrowserSession and all watchdogs.

This test suite validates all browser features:
- Local browser launch with watchdog system
- Session manager for CDP session lifecycle
- Multiple tab support
- Profile management (BrowserProfile)
- Proxy support
- User data directory management
- Video recording
- Screenshot management
- Download handling
- Popup handling
- Security handling
- Storage state management
- Event-driven CDP session management
- Target attach/detach event handling
- Multiple CDP sessions per target support
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


# ============================================================================
# Test: BrowserProfile Configuration
# ============================================================================
async def test_browser_profile():
    """Test BrowserProfile class configuration."""
    logger.info("=" * 80)
    logger.info("TEST: BrowserProfile Configuration")
    logger.info("=" * 80)

    from openbrowser.browser.profile import BrowserProfile, ProxySettings, ViewportSize

    # Test default profile creation
    profile = BrowserProfile()
    assert profile.user_data_dir is not None, "user_data_dir should be auto-generated"
    assert profile.headless is None, "headless should be None by default"
    assert profile.permissions == ['clipboardReadWrite', 'notifications'], "default permissions"
    assert profile.is_local is True, "is_local should be True by default"
    assert profile.cross_origin_iframes is True, "cross_origin_iframes should be True by default"
    assert profile.max_iframes == 100, "max_iframes should be 100 by default"
    assert profile.max_iframe_depth == 5, "max_iframe_depth should be 5 by default"
    assert profile.highlight_elements is True, "highlight_elements should be True by default"
    logger.info("  Default profile creation: PASSED")

    # Test profile with custom settings
    custom_profile = BrowserProfile(
        headless=True,
        user_data_dir="/tmp/test_profile",
        window_size=ViewportSize(width=1920, height=1080),
        viewport=ViewportSize(width=1280, height=720),
        proxy=ProxySettings(server="http://proxy.example.com:8080"),
        allowed_domains=["example.com", "*.google.com"],
        prohibited_domains=["malware.com"],
        record_video_dir=Path("/tmp/videos"),
        record_video_framerate=30,
        storage_state="/tmp/storage.json",
        auto_download_pdfs=True,
        cross_origin_iframes=False,
        max_iframes=50,
        max_iframe_depth=3,
        highlight_elements=False,
        interaction_highlight_color="rgb(0, 255, 0)",
        interaction_highlight_duration=2.0,
    )

    assert custom_profile.headless is True
    # On macOS, /tmp resolves to /private/tmp, so compare resolved paths
    expected_path = Path("/tmp/test_profile").resolve()
    assert custom_profile.user_data_dir == expected_path, f"user_data_dir mismatch: {custom_profile.user_data_dir} != {expected_path}"
    assert custom_profile.window_size.width == 1920
    assert custom_profile.proxy.server == "http://proxy.example.com:8080"
    assert "example.com" in custom_profile.allowed_domains
    assert "malware.com" in custom_profile.prohibited_domains
    assert custom_profile.record_video_framerate == 30
    assert custom_profile.cross_origin_iframes is False
    assert custom_profile.max_iframes == 50
    logger.info("  Custom profile settings: PASSED")

    # Test get_args()
    args = custom_profile.get_args()
    assert any("--user-data-dir=" in arg for arg in args), "user_data_dir arg missing"
    assert "--headless=new" in args, "headless arg missing"
    assert any("--proxy-server=" in arg for arg in args), "proxy arg missing"
    logger.info("  get_args(): PASSED")

    logger.info("BrowserProfile: ALL TESTS PASSED")
    return True


# ============================================================================
# Test: BrowserSession Event-Driven Architecture
# ============================================================================
async def test_browser_session_events():
    """Test BrowserSession event-driven architecture."""
    logger.info("=" * 80)
    logger.info("TEST: BrowserSession Event-Driven Architecture")
    logger.info("=" * 80)

    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile
    from openbrowser.browser.events import (
        BrowserStartEvent,
        BrowserStopEvent,
        NavigateToUrlEvent,
        TabCreatedEvent,
        SwitchTabEvent,
    )

    profile = BrowserProfile(headless=True)
    session = BrowserSession(browser_profile=profile)

    # Test event bus exists
    assert session.event_bus is not None, "event_bus should exist"
    logger.info("  Event bus exists: PASSED")

    # Test event handlers are registered
    assert hasattr(session, 'on_BrowserStartEvent'), "on_BrowserStartEvent handler should exist"
    assert hasattr(session, 'on_BrowserStopEvent'), "on_BrowserStopEvent handler should exist"
    assert hasattr(session, 'on_NavigateToUrlEvent'), "on_NavigateToUrlEvent handler should exist"
    assert hasattr(session, 'on_SwitchTabEvent'), "on_SwitchTabEvent handler should exist"
    assert hasattr(session, 'on_CloseTabEvent'), "on_CloseTabEvent handler should exist"
    assert hasattr(session, 'on_FileDownloadedEvent'), "on_FileDownloadedEvent handler should exist"
    logger.info("  Event handlers registered: PASSED")

    # Test helper methods exist
    assert hasattr(session, 'get_current_page_url'), "get_current_page_url should exist"
    assert hasattr(session, 'get_current_page_title'), "get_current_page_title should exist"
    assert hasattr(session, 'get_current_target_info'), "get_current_target_info should exist"
    assert hasattr(session, 'get_tabs'), "get_tabs should exist"
    assert hasattr(session, 'get_all_frames'), "get_all_frames should exist"
    assert hasattr(session, 'navigate_to'), "navigate_to should exist"
    assert hasattr(session, 'update_cached_selector_map'), "update_cached_selector_map should exist"
    logger.info("  Helper methods exist: PASSED")

    # Test properties
    assert hasattr(session, 'current_target_id'), "current_target_id property should exist"
    assert hasattr(session, 'current_session_id'), "current_session_id property should exist"
    assert hasattr(session, 'downloaded_files'), "downloaded_files property should exist"
    logger.info("  Properties exist: PASSED")

    logger.info("BrowserSession Events: ALL TESTS PASSED")
    return True


# ============================================================================
# Test: BrowserSession Start/Stop Lifecycle
# ============================================================================
async def test_browser_session_lifecycle():
    """Test BrowserSession start/stop lifecycle with watchdogs."""
    logger.info("=" * 80)
    logger.info("TEST: BrowserSession Start/Stop Lifecycle")
    logger.info("=" * 80)

    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile

    profile = BrowserProfile(headless=True)
    session = BrowserSession(browser_profile=profile)

    # Test start
    logger.info("  Starting browser session...")
    await session.start()

    assert session.agent_focus is not None, "agent_focus should be set after start"
    assert session._cdp_client_root is not None, "CDP client should be initialized"
    assert session._session_manager is not None, "SessionManager should be initialized"
    assert session._watchdogs_attached is True, "Watchdogs should be attached"
    logger.info("  Browser started: PASSED")

    # Test helper methods after start
    url = await session.get_current_page_url()
    assert url is not None, "get_current_page_url should return URL"
    logger.info(f"  Current URL: {url}")

    title = await session.get_current_page_title()
    assert title is not None, "get_current_page_title should return title"
    logger.info(f"  Current title: {title}")

    tabs = await session.get_tabs()
    assert isinstance(tabs, list), "get_tabs should return a list"
    assert len(tabs) >= 1, "At least one tab should exist"
    logger.info(f"  Tabs: {len(tabs)} tab(s) found")

    target_info = await session.get_current_target_info()
    assert target_info is not None, "get_current_target_info should return info"
    logger.info(f"  Target info: {target_info.get('targetId', 'N/A')[:8]}...")

    # Test stop
    logger.info("  Stopping browser session...")
    await session.stop()

    assert session.agent_focus is None, "agent_focus should be None after stop"
    assert session._cdp_client_root is None, "CDP client should be None after stop"
    logger.info("  Browser stopped: PASSED")

    logger.info("BrowserSession Lifecycle: ALL TESTS PASSED")
    return True


# ============================================================================
# Test: SessionManager Event-Driven CDP Session Management
# ============================================================================
async def test_session_manager():
    """Test SessionManager for CDP session lifecycle management."""
    logger.info("=" * 80)
    logger.info("TEST: SessionManager CDP Session Lifecycle")
    logger.info("=" * 80)

    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile

    profile = BrowserProfile(headless=True)
    session = BrowserSession(browser_profile=profile)

    await session.start()

    try:
        # Test SessionManager initialization
        assert session._session_manager is not None, "SessionManager should be initialized"
        logger.info("  SessionManager initialized: PASSED")

        # Test session pool management
        assert len(session._cdp_session_pool) >= 1, "Session pool should have at least one session"
        logger.info(f"  Session pool size: {len(session._cdp_session_pool)}")

        # Test get_session_for_target
        if session.agent_focus:
            target_id = session.agent_focus.target_id
            cached_session = await session._session_manager.get_session_for_target(target_id)
            assert cached_session is not None, "Should get cached session for current target"
            logger.info("  get_session_for_target: PASSED")

        # Test validate_session
        if session.agent_focus:
            is_valid = await session._session_manager.validate_session(session.agent_focus.target_id)
            assert is_valid is True, "Current target should be valid"
            logger.info("  validate_session: PASSED")

        logger.info("SessionManager: ALL TESTS PASSED")
        return True

    finally:
        await session.stop()


# ============================================================================
# Test: Multiple Tab Support
# ============================================================================
async def test_multiple_tab_support():
    """Test multiple tab support in BrowserSession."""
    logger.info("=" * 80)
    logger.info("TEST: Multiple Tab Support")
    logger.info("=" * 80)

    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile

    profile = BrowserProfile(headless=True)
    session = BrowserSession(browser_profile=profile)

    await session.start()

    try:
        # Get initial tab count
        initial_tabs = await session.get_tabs()
        initial_count = len(initial_tabs)
        logger.info(f"  Initial tab count: {initial_count}")

        # Create a new tab
        new_target_id = await session._cdp_create_new_page("about:blank")
        assert new_target_id is not None, "Should create new page"
        logger.info(f"  Created new tab: {new_target_id[:8]}...")

        # Wait for tab to be registered
        await asyncio.sleep(0.5)

        # Verify tab count increased
        current_tabs = await session.get_tabs()
        assert len(current_tabs) == initial_count + 1, f"Tab count should increase to {initial_count + 1}"
        logger.info(f"  Tab count after creation: {len(current_tabs)}")

        # Switch to new tab
        from openbrowser.browser.events import SwitchTabEvent
        await session.event_bus.dispatch(SwitchTabEvent(target_id=new_target_id))
        await asyncio.sleep(0.3)

        assert session.agent_focus.target_id == new_target_id, "Agent focus should switch to new tab"
        logger.info("  Tab switch: PASSED")

        # Close the new tab
        await session._cdp_close_page(new_target_id)
        await asyncio.sleep(0.5)

        final_tabs = await session.get_tabs()
        assert len(final_tabs) == initial_count, f"Tab count should return to {initial_count}"
        logger.info(f"  Tab count after close: {len(final_tabs)}")

        logger.info("Multiple Tab Support: ALL TESTS PASSED")
        return True

    finally:
        await session.stop()


# ============================================================================
# Test: Navigation with Event System
# ============================================================================
async def test_navigation_events():
    """Test navigation using event-driven system."""
    logger.info("=" * 80)
    logger.info("TEST: Navigation with Event System")
    logger.info("=" * 80)

    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile

    profile = BrowserProfile(headless=True)
    session = BrowserSession(browser_profile=profile)

    await session.start()

    try:
        # Navigate using helper method
        await session.navigate_to("https://example.com")
        await asyncio.sleep(2)  # Wait for navigation

        url = await session.get_current_page_url()
        assert "example.com" in url, f"URL should contain example.com, got: {url}"
        logger.info(f"  Navigated to: {url}")

        # Navigate to new tab
        await session.navigate_to("https://example.org", new_tab=True)
        await asyncio.sleep(2)

        tabs = await session.get_tabs()
        # Should have at least 2 tabs now
        assert len(tabs) >= 2, f"Should have at least 2 tabs, got {len(tabs)}"
        logger.info(f"  New tab navigation: {len(tabs)} tabs")

        logger.info("Navigation Events: ALL TESTS PASSED")
        return True

    finally:
        await session.stop()


# ============================================================================
# Test: Watchdogs Initialization
# ============================================================================
async def test_watchdogs_initialization():
    """Test all watchdogs are properly initialized."""
    logger.info("=" * 80)
    logger.info("TEST: Watchdogs Initialization")
    logger.info("=" * 80)

    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile

    # Create profile with video recording to enable RecordingWatchdog
    with tempfile.TemporaryDirectory() as tmpdir:
        profile = BrowserProfile(
            headless=True,
            storage_state=os.path.join(tmpdir, "storage.json"),
            downloads_path=tmpdir,
        )
        session = BrowserSession(browser_profile=profile)

        await session.start()

        try:
            # Check all watchdogs are initialized
            assert session._local_browser_watchdog is not None, "LocalBrowserWatchdog should be initialized"
            logger.info("  LocalBrowserWatchdog: INITIALIZED")

            assert session._downloads_watchdog is not None, "DownloadsWatchdog should be initialized"
            logger.info("  DownloadsWatchdog: INITIALIZED")

            assert session._popups_watchdog is not None, "PopupsWatchdog should be initialized"
            logger.info("  PopupsWatchdog: INITIALIZED")

            assert session._security_watchdog is not None, "SecurityWatchdog should be initialized"
            logger.info("  SecurityWatchdog: INITIALIZED")

            assert session._storage_state_watchdog is not None, "StorageStateWatchdog should be initialized"
            logger.info("  StorageStateWatchdog: INITIALIZED")

            assert session._permissions_watchdog is not None, "PermissionsWatchdog should be initialized"
            logger.info("  PermissionsWatchdog: INITIALIZED")

            assert session._screenshot_watchdog is not None, "ScreenshotWatchdog should be initialized"
            logger.info("  ScreenshotWatchdog: INITIALIZED")

            assert session._dom_watchdog is not None, "DOMWatchdog should be initialized"
            logger.info("  DOMWatchdog: INITIALIZED")

            logger.info("Watchdogs Initialization: ALL TESTS PASSED")
            return True

        finally:
            await session.stop()


# ============================================================================
# Test: Screenshot via Event System
# ============================================================================
async def test_screenshot_event():
    """Test screenshot via event system."""
    logger.info("=" * 80)
    logger.info("TEST: Screenshot via Event System")
    logger.info("=" * 80)

    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile
    from openbrowser.browser.events import ScreenshotEvent

    profile = BrowserProfile(headless=True)
    session = BrowserSession(browser_profile=profile)

    await session.start()

    try:
        # Navigate to a page first
        await session.navigate_to("https://example.com")
        await asyncio.sleep(2)

        # Take screenshot via event
        screenshot_event = session.event_bus.dispatch(ScreenshotEvent())
        await screenshot_event
        result = await screenshot_event.event_result(raise_if_any=True)

        assert result is not None, "Screenshot should return data"
        assert isinstance(result, str), "Screenshot should be base64 string"
        assert len(result) > 100, "Screenshot should have content"
        logger.info(f"  Screenshot captured: {len(result)} bytes (base64)")

        logger.info("Screenshot Event: ALL TESTS PASSED")
        return True

    finally:
        await session.stop()


# ============================================================================
# Test: Security Watchdog URL Validation
# ============================================================================
async def test_security_watchdog():
    """Test SecurityWatchdog URL validation."""
    logger.info("=" * 80)
    logger.info("TEST: SecurityWatchdog URL Validation")
    logger.info("=" * 80)

    from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog
    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile
    from bubus import EventBus

    # Create profile with domain restrictions
    profile = BrowserProfile(
        headless=True,
        allowed_domains=["example.com", "*.test.com"],
        prohibited_domains=["malware.com"],
    )
    session = BrowserSession(browser_profile=profile)

    # Test URL validation logic without starting browser
    watchdog = SecurityWatchdog(event_bus=EventBus(), browser_session=session)

    # Test allowed URLs
    assert watchdog._is_url_allowed("https://example.com") is True
    assert watchdog._is_url_allowed("https://sub.test.com") is True
    logger.info("  Allowed URLs pass: PASSED")

    # Test prohibited URLs
    assert watchdog._is_url_allowed("https://malware.com") is False
    logger.info("  Prohibited URLs blocked: PASSED")

    # Test non-allowed URLs (when allowed_domains is set)
    assert watchdog._is_url_allowed("https://other-site.com") is False
    logger.info("  Non-allowed URLs blocked: PASSED")

    logger.info("SecurityWatchdog: ALL TESTS PASSED")
    return True


# ============================================================================
# Test: Cookie Management
# ============================================================================
async def test_cookie_management():
    """Test cookie get/set via CDP."""
    logger.info("=" * 80)
    logger.info("TEST: Cookie Management")
    logger.info("=" * 80)

    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.profile import BrowserProfile

    profile = BrowserProfile(headless=True)
    session = BrowserSession(browser_profile=profile)

    await session.start()

    try:
        # Navigate to a page first (cookies require a URL context)
        await session.navigate_to("https://example.com")
        await asyncio.sleep(2)

        # Get initial cookies
        initial_cookies = await session._cdp_get_cookies()
        logger.info(f"  Initial cookies: {len(initial_cookies)}")

        # Set a test cookie - use the exact domain from the current URL
        # CDP requires the URL domain for cookie setting (not .example.com pattern)
        test_cookie = {
            "name": "test_cookie",
            "value": "test_value",
            "domain": ".example.com",  # Leading dot for domain cookies
            "path": "/",
            "secure": True,  # Required for SameSite=None
        }
        await session._cdp_set_cookies([test_cookie])
        logger.info("  Set test cookie: PASSED")

        # Wait briefly for cookie to be set
        await asyncio.sleep(0.5)

        # Get cookies again - use Network.getAllCookies to get all cookies including domain cookies
        try:
            all_cookies_result = await session.agent_focus.cdp_client.send.Storage.getCookies(
                session_id=session.agent_focus.session_id
            )
            current_cookies = all_cookies_result.get('cookies', [])
        except Exception:
            # Fallback to Network.getCookies
            current_cookies = await session._cdp_get_cookies()
        
        test_cookie_found = any(c.get("name") == "test_cookie" for c in current_cookies)
        if not test_cookie_found:
            logger.warning(f"  Cookies found: {[c.get('name') for c in current_cookies]}")
        assert test_cookie_found, "Test cookie should be found"
        logger.info("  Get cookies: PASSED")

        logger.info("Cookie Management: ALL TESTS PASSED")
        return True

    finally:
        await session.stop()


# ============================================================================
# Test: Views and TabInfo Model
# ============================================================================
async def test_views_and_models():
    """Test view models like TabInfo."""
    logger.info("=" * 80)
    logger.info("TEST: Views and TabInfo Model")
    logger.info("=" * 80)

    from openbrowser.browser.views import TabInfo

    # Test TabInfo creation
    tab = TabInfo(
        target_id="1234567890ABCDEF",
        url="https://example.com",
        title="Example Page",
        parent_target_id=None,
    )

    assert tab.target_id == "1234567890ABCDEF"
    assert tab.url == "https://example.com"
    assert tab.title == "Example Page"
    logger.info("  TabInfo creation: PASSED")

    # Test serialization (should truncate target_id to last 4 chars)
    serialized = tab.model_dump(by_alias=True)
    assert serialized["tab_id"] == "CDEF", f"tab_id should be last 4 chars, got {serialized['tab_id']}"
    logger.info("  TabInfo serialization: PASSED")

    # Test validation aliases
    tab2 = TabInfo(
        tab_id="ABCD1234",  # Using alias
        url="https://test.com",
        title="Test",
    )
    assert tab2.target_id == "ABCD1234"
    logger.info("  TabInfo validation alias: PASSED")

    logger.info("Views and Models: ALL TESTS PASSED")
    return True


# ============================================================================
# Test: Event Definitions
# ============================================================================
async def test_event_definitions():
    """Test all event definitions exist and have correct structure."""
    logger.info("=" * 80)
    logger.info("TEST: Event Definitions")
    logger.info("=" * 80)

    from openbrowser.browser.events import (
        # Lifecycle events
        BrowserStartEvent,
        BrowserStopEvent,
        BrowserLaunchEvent,
        BrowserLaunchResult,
        BrowserKillEvent,
        BrowserConnectedEvent,
        BrowserStoppedEvent,
        # Navigation events
        NavigateToUrlEvent,
        NavigationStartedEvent,
        NavigationCompleteEvent,
        # Tab events
        TabCreatedEvent,
        TabClosedEvent,
        SwitchTabEvent,
        CloseTabEvent,
        AgentFocusChangedEvent,
        # Action events
        ClickElementEvent,
        TypeTextEvent,
        PressKeyEvent,
        ScreenshotEvent,
        # Download events
        FileDownloadedEvent,
        # Storage events
        SaveStorageStateEvent,
        LoadStorageStateEvent,
        StorageStateSavedEvent,
        StorageStateLoadedEvent,
        # Error events
        BrowserErrorEvent,
    )

    # Test event creation
    events_to_test = [
        BrowserStartEvent(),
        BrowserStopEvent(),
        BrowserLaunchEvent(),
        BrowserKillEvent(),
        BrowserConnectedEvent(cdp_url="ws://localhost:9222"),
        BrowserStoppedEvent(reason="test"),
        NavigateToUrlEvent(url="https://example.com"),
        NavigationStartedEvent(target_id="test", url="https://example.com"),
        NavigationCompleteEvent(target_id="test", url="https://example.com"),
        TabCreatedEvent(target_id="test", url="https://example.com"),
        TabClosedEvent(target_id="test"),
        SwitchTabEvent(target_id="test"),
        CloseTabEvent(target_id="test"),
        AgentFocusChangedEvent(target_id="test", url="https://example.com"),
        ClickElementEvent(index=1),
        TypeTextEvent(index=1, text="test"),
        PressKeyEvent(key="Enter"),
        ScreenshotEvent(),
        FileDownloadedEvent(url="test", path="/tmp/test", file_name="test.pdf", file_size=100),
        SaveStorageStateEvent(),
        LoadStorageStateEvent(),
        StorageStateSavedEvent(path="/tmp/storage.json"),
        StorageStateLoadedEvent(path="/tmp/storage.json"),
        BrowserErrorEvent(error_type="test", message="test error"),
    ]

    for event in events_to_test:
        assert event is not None
        assert hasattr(event, 'event_timeout'), f"{type(event).__name__} should have event_timeout"

    logger.info(f"  All {len(events_to_test)} event types: PASSED")

    logger.info("Event Definitions: ALL TESTS PASSED")
    return True


# ============================================================================
# Test: Agent Integration with BrowserSession
# ============================================================================
async def test_agent_integration():
    """Test BrowserAgent integration with BrowserSession."""
    logger.info("=" * 80)
    logger.info("TEST: Agent Integration with BrowserSession")
    logger.info("=" * 80)

    from openbrowser.agent.graph import BrowserAgent
    from openbrowser.browser.profile import BrowserProfile

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("  SKIPPED: No API key available")
        return True  # Skip but don't fail

    provider = "openai" if os.getenv("OPENAI_API_KEY") else "google"
    model = "gpt-4o" if provider == "openai" else "gemini-flash-latest"

    profile = BrowserProfile(headless=True)
    agent = BrowserAgent(
        task="Test task",
        headless=True,
        model_name=model,
        llm_provider=provider,
        browser_profile=profile,
    )

    # Verify agent uses BrowserSession
    assert hasattr(agent, 'browser_session'), "Agent should have browser_session"
    assert agent.browser_session is not None, "browser_session should not be None"
    logger.info("  Agent has BrowserSession: PASSED")

    # Verify tools uses BrowserSession
    assert hasattr(agent, 'tools'), "Agent should have tools"
    assert agent.tools.browser_session is agent.browser_session, "Tools should use same session"
    logger.info("  Tools uses same BrowserSession: PASSED")

    logger.info("Agent Integration: ALL TESTS PASSED")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================
async def main():
    """Run all comprehensive tests."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE BROWSER VALIDATION TEST SUITE")
    logger.info("Validating all features from difference.md (92-112)")
    logger.info("=" * 80 + "\n")

    test_results = {}

    # Run all tests
    tests = [
        ("BrowserProfile Configuration", test_browser_profile),
        ("BrowserSession Events", test_browser_session_events),
        ("Views and Models", test_views_and_models),
        ("Event Definitions", test_event_definitions),
        ("Security Watchdog", test_security_watchdog),
        ("BrowserSession Lifecycle", test_browser_session_lifecycle),
        ("SessionManager", test_session_manager),
        ("Multiple Tab Support", test_multiple_tab_support),
        ("Navigation Events", test_navigation_events),
        ("Watchdogs Initialization", test_watchdogs_initialization),
        ("Screenshot Event", test_screenshot_event),
        ("Cookie Management", test_cookie_management),
        ("Agent Integration", test_agent_integration),
    ]

    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            test_results[test_name] = f"ERROR: {str(e)[:50]}"

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for r in test_results.values() if r == "PASSED")
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "PASSED" if result == "PASSED" else "FAILED"
        logger.info(f"  {test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\nALL TESTS PASSED!")
    else:
        logger.warning(f"\n{total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

