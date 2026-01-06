"""Event-driven browser session following browser-use pattern.

This module provides the core browser session management using Chrome DevTools
Protocol (CDP) for direct communication with Chromium-based browsers. It implements
an event-driven architecture with watchdogs for handling various browser behaviors.

Key Components:
    CDPSession: Manages a single CDP session bound to a specific browser target.
    BrowserSession: Main orchestrator for browser lifecycle, navigation, and state.

The module uses bubus EventBus for event-driven communication between components
and cdp-use for type-safe CDP interactions.

Example:
    >>> session = BrowserSession(headless=True, debug_port=9222)
    >>> await session.start()
    >>> await session.navigate_to("https://example.com")
    >>> tabs = await session.get_tabs()
    >>> await session.stop()
"""

import asyncio
import logging
from typing import Any, Optional

from bubus import EventBus
from cdp_use import CDPClient
from cdp_use.cdp.target import SessionID, TargetID
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)


class CDPSession(BaseModel):
    """Info about a single CDP session bound to a specific browser target.

    Manages the connection to a specific browser target (page, iframe, worker)
    via the Chrome DevTools Protocol. Handles session attachment, domain
    enabling, and provides target info access.

    Attributes:
        cdp_client: Shared CDPClient for WebSocket communication.
        target_id: The CDP target ID this session is attached to.
        session_id: The CDP session ID for this attachment.
        title: Current page/target title.
        url: Current URL of the target.

    Example:
        >>> session = await CDPSession.for_target(cdp_client, target_id)
        >>> info = await session.get_target_info()
        >>> print(session.title, session.url)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, revalidate_instances='never')

    cdp_client: CDPClient
    target_id: TargetID
    session_id: SessionID
    title: str = 'Unknown title'
    url: str = 'about:blank'

    @classmethod
    async def for_target(
        cls,
        cdp_client: CDPClient,
        target_id: TargetID,
        domains: list[str] | None = None,
    ):
        """Create a CDP session for a target using the shared WebSocket.

        Factory method that creates and attaches a CDP session to a browser target.
        Enables specified CDP domains for the session.

        Args:
            cdp_client: The shared CDP client (root WebSocket connection).
            target_id: Target ID to attach to (page, iframe, worker).
            domains: List of CDP domains to enable (e.g., ['Page', 'DOM']).
                If None, enables default domains: Page, DOM, DOMSnapshot,
                Accessibility, Runtime, Inspector.

        Returns:
            Attached CDPSession instance ready for use.

        Raises:
            RuntimeError: If domain enabling fails.

        Example:
            >>> session = await CDPSession.for_target(
            ...     cdp_client, target_id,
            ...     domains=['Page', 'DOM', 'Network']
            ... )
        """
        cdp_session = cls(
            cdp_client=cdp_client,
            target_id=target_id,
            session_id='connecting',
        )
        return await cdp_session.attach(domains=domains)

    async def attach(self, domains: list[str] | None = None):
        """Attach to target and enable CDP domains.

        Establishes a CDP session with the target and enables the specified
        domains for interacting with the target.

        Args:
            domains: List of CDP domain names to enable (e.g., ['Page', 'DOM']).
                If None, uses default domains.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If any domain fails to enable.
        """
        result = await self.cdp_client.send.Target.attachToTarget(
            params={
                'targetId': self.target_id,
                'flatten': True,
            }
        )
        self.session_id = result['sessionId']

        # Use specified domains or default domains
        domains = domains or ['Page', 'DOM', 'DOMSnapshot', 'Accessibility', 'Runtime', 'Inspector']

        # Enable all domains in parallel
        enable_tasks = []
        for domain in domains:
            domain_api = getattr(self.cdp_client.send, domain, None)
            enable_kwargs = {} if domain in ['Browser', 'Target'] else {'session_id': self.session_id}
            assert domain_api and hasattr(domain_api, 'enable'), (
                f'{domain_api} is not a recognized CDP domain with a .enable() method'
            )
            enable_tasks.append(domain_api.enable(**enable_kwargs))

        results = await asyncio.gather(*enable_tasks, return_exceptions=True)
        if any(isinstance(result, Exception) for result in results):
            raise RuntimeError(f'Failed to enable requested CDP domain: {results}')

        # Disable breakpoints if Debugger domain is enabled
        try:
            await self.cdp_client.send.Debugger.setSkipAllPauses(
                params={'skip': True}, session_id=self.session_id
            )
        except Exception:
            pass

        # Get target info
        target_info = await self.get_target_info()
        self.title = target_info.get('title', 'Unknown title')
        self.url = target_info.get('url', 'about:blank')
        return self

    async def get_target_info(self) -> dict:
        """Get target info from CDP.

        Retrieves current information about the attached target including
        URL, title, type, and other metadata.

        Returns:
            Dictionary containing target info from CDP Target.getTargetInfo.
            Keys include 'targetId', 'type', 'title', 'url', etc.
        """
        result = await self.cdp_client.send.Target.getTargetInfo(params={'targetId': self.target_id})
        return result['targetInfo']


class BrowserSession(BaseModel):
    """Event-driven browser session following browser-use pattern.

    Main orchestrator for browser lifecycle management using CDP. Handles:
    - Browser process launch and termination
    - CDP connection establishment and session management
    - Tab/target creation, navigation, and switching
    - Watchdog attachment for downloads, popups, security, etc.
    - DOM state caching and element selection

    Uses bubus EventBus for event-driven architecture, allowing watchdogs
    and other components to react to browser events.

    Attributes:
        event_bus: EventBus for dispatching and handling browser events.
        agent_focus: Currently focused CDPSession for agent interactions.
        debug_port: Chrome remote debugging port.
        headless: Whether browser runs without visible UI.
        user_data_dir: Path to Chrome user data directory.
        browser_profile: Full BrowserProfile configuration.
        downloaded_files: List of files downloaded during session.

    Example:
        >>> session = BrowserSession(
        ...     browser_profile=BrowserProfile(headless=True),
        ...     debug_port=9222
        ... )
        >>> await session.start()
        >>> await session.navigate_to("https://example.com")
        >>> url = await session.get_current_page_url()
        >>> await session.stop()
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid',
        revalidate_instances='never',
    )

    # Main shared event bus for all browser session operations
    event_bus: EventBus = Field(default_factory=EventBus)

    # Mutable public state
    agent_focus: CDPSession | None = None

    # Mutable private state
    _cdp_client_root: CDPClient | None = PrivateAttr(default=None)
    _cdp_session_pool: dict[str, CDPSession] = PrivateAttr(default_factory=dict)
    _cdp_url: Optional[str] = PrivateAttr(default=None)
    _logger: Optional[logging.Logger] = PrivateAttr(default=None)
    _process: Optional[asyncio.subprocess.Process] = PrivateAttr(default=None)
    _debug_port: int = PrivateAttr(default=9222)
    _headless: bool = PrivateAttr(default=True)
    _user_data_dir: Optional[str] = PrivateAttr(default=None)
    _session_manager: Optional[Any] = PrivateAttr(default=None)
    _browser_profile: Optional[Any] = PrivateAttr(default=None)
    _downloaded_files: list[str] = PrivateAttr(default_factory=list)
    _closed_popup_messages: list[str] = PrivateAttr(default_factory=list)
    _watchdogs_attached: bool = PrivateAttr(default=False)
    # Watchdog instances
    _downloads_watchdog: Optional[Any] = PrivateAttr(default=None)
    _popups_watchdog: Optional[Any] = PrivateAttr(default=None)
    _security_watchdog: Optional[Any] = PrivateAttr(default=None)
    _storage_state_watchdog: Optional[Any] = PrivateAttr(default=None)
    _recording_watchdog: Optional[Any] = PrivateAttr(default=None)
    _screenshot_watchdog: Optional[Any] = PrivateAttr(default=None)
    _permissions_watchdog: Optional[Any] = PrivateAttr(default=None)
    _local_browser_watchdog: Optional[Any] = PrivateAttr(default=None)
    _dom_watchdog: Optional[Any] = PrivateAttr(default=None)
    _cached_selector_map: Optional[dict[int, Any]] = PrivateAttr(default=None)
    
    def __init__(
        self,
        debug_port: int = 9222,
        headless: bool = True,
        user_data_dir: Optional[str] = None,
        browser_profile: Optional[Any] = None,
        **kwargs
    ):
        """Initialize BrowserSession with browser launch parameters.

        Creates a new browser session with the specified configuration.
        Either provide a full BrowserProfile or individual parameters.

        Args:
            debug_port: Port for Chrome remote debugging protocol.
            headless: Whether to run Chrome in headless mode (no UI).
            user_data_dir: Path to Chrome user data directory for profile
                persistence. If None, uses a temporary directory.
            browser_profile: Optional BrowserProfile instance for advanced
                configuration. If provided, overrides other parameters.
            **kwargs: Additional Pydantic model arguments.

        Example:
            >>> # Simple initialization
            >>> session = BrowserSession(headless=True)
            >>> # With full profile
            >>> profile = BrowserProfile(headless=False, downloads_path='/tmp')
            >>> session = BrowserSession(browser_profile=profile)
        """
        super().__init__(**kwargs)
        import tempfile
        from openbrowser.browser.profile import BrowserProfile
        
        self._debug_port = debug_port
        
        # Use browser_profile if provided, otherwise create from simple params
        if browser_profile:
            self._browser_profile = browser_profile
            self._user_data_dir = str(browser_profile.user_data_dir) if browser_profile.user_data_dir else None
            # CRITICAL: Use headless from browser_profile, not from parameter default
            self._headless = browser_profile.headless if browser_profile.headless is not None else headless
        else:
            self._user_data_dir = user_data_dir or tempfile.mkdtemp(prefix="openbrowser_chrome_")
            self._headless = headless
            # Create minimal profile from simple params
            self._browser_profile = BrowserProfile(
                user_data_dir=self._user_data_dir,
                headless=headless,
            )
    
    @property
    def debug_port(self) -> int:
        """Get the Chrome remote debugging port.

        Returns:
            The port number used for CDP connections.
        """
        return self._debug_port
    
    @property
    def headless(self) -> bool:
        """Get whether browser runs in headless mode.

        Returns:
            True if browser runs without visible UI, False otherwise.
        """
        return self._headless
    
    @property
    def user_data_dir(self) -> Optional[str]:
        """Get the Chrome user data directory path.

        Returns:
            Path to user data directory, or None if using default.
        """
        return self._user_data_dir
    
    @property
    def browser_profile(self) -> Any:
        """Get the BrowserProfile configuration.

        Returns:
            BrowserProfile instance with all browser settings.
        """
        return self._browser_profile
    
    @property
    def downloaded_files(self) -> list[str]:
        """Get list of files downloaded during this session.

        Returns:
            Copy of list containing paths to downloaded files.
        """
        return self._downloaded_files.copy()

    @property
    def logger(self) -> logging.Logger:
        """Get instance-specific logger for this session.

        Returns:
            Logger instance for this browser session.
        """
        if self._logger is None:
            self._logger = logging.getLogger(f'openbrowser.browser_session')
        return self._logger

    @property
    def cdp_url(self) -> str | None:
        """Get the CDP WebSocket URL.

        Returns:
            WebSocket URL for CDP connection, or None if not connected.
        """
        return self._cdp_url

    @cdp_url.setter
    def cdp_url(self, value: str | None) -> None:
        """Set the CDP WebSocket URL.

        Args:
            value: WebSocket URL for CDP connection.
        """
        self._cdp_url = value

    @property
    def cdp_client(self) -> CDPClient:
        """Get the cached root CDP client.

        Returns:
            The CDPClient instance for sending CDP commands.

        Raises:
            AssertionError: If CDP client is not initialized (browser not connected).
        """
        assert self._cdp_client_root is not None, 'CDP client not initialized - browser may not be connected yet'
        return self._cdp_client_root

    async def reset(self) -> None:
        """Clear all cached CDP sessions with proper cleanup.

        Resets the session state by clearing the session pool,
        disconnecting from CDP, and clearing agent focus.
        Call this when reconnecting or before starting a new session.
        """
        # Clear session pool
        self._cdp_session_pool.clear()
        self._cdp_client_root = None
        self.agent_focus = None
        self._cdp_url = None

    def model_post_init(self, __context) -> None:
        """Register event handlers after model initialization.

        Sets up event bus subscriptions for core browser events including
        start, stop, navigation, tab switching, and downloads.

        Args:
            __context: Pydantic model context (unused).
        """
        from openbrowser.browser.events import (
            BrowserStartEvent,
            BrowserStopEvent,
            NavigateToUrlEvent,
            SwitchTabEvent,
            CloseTabEvent,
            FileDownloadedEvent,
        )

        # Register event handlers
        self.event_bus.on(BrowserStartEvent, self.on_BrowserStartEvent)
        self.event_bus.on(BrowserStopEvent, self.on_BrowserStopEvent)
        self.event_bus.on(NavigateToUrlEvent, self.on_NavigateToUrlEvent)
        self.event_bus.on(SwitchTabEvent, self.on_SwitchTabEvent)
        self.event_bus.on(CloseTabEvent, self.on_CloseTabEvent)
        self.event_bus.on(FileDownloadedEvent, self.on_FileDownloadedEvent)

    async def start(self) -> None:
        """Start the browser session.

        Launches the browser process (if local), establishes CDP connection,
        attaches all watchdogs, and sets up initial page focus.

        Raises:
            RuntimeError: If browser launch or CDP connection fails.

        Example:
            >>> session = BrowserSession(headless=True)
            >>> await session.start()
            >>> # Browser is now ready for navigation
        """
        from openbrowser.browser.events import BrowserStartEvent

        start_event = self.event_bus.dispatch(BrowserStartEvent())
        await start_event
        await start_event.event_result(raise_if_any=True, raise_if_none=False)

    async def stop(self, force: bool = False) -> None:
        """Stop the browser session.

        Cleanly shuts down the browser session, stops watchdogs, and
        optionally terminates the browser process.

        Args:
            force: If True, kill the browser process immediately.
                If False, keep browser alive (for reattachment).

        Example:
            >>> await session.stop()  # Keep browser alive
            >>> await session.stop(force=True)  # Kill browser
        """
        from openbrowser.browser.events import BrowserStopEvent

        await self.event_bus.dispatch(BrowserStopEvent(force=force))
        await self.event_bus.stop(clear=True, timeout=5)
        await self.reset()
        self.event_bus = EventBus()

    async def on_BrowserStartEvent(self, event) -> dict[str, str]:
        """Handle browser start request.

        Initializes watchdogs, launches browser if needed, establishes
        CDP connection, and sets up initial session state.

        Args:
            event: BrowserStartEvent with optional launch options.

        Returns:
            Dict with 'cdp_url' key containing the CDP WebSocket URL.

        Raises:
            Exception: If browser launch or CDP connection fails.
        """
        # Initialize and attach all watchdogs FIRST so LocalBrowserWatchdog can handle BrowserLaunchEvent
        await self.attach_all_watchdogs()

        try:
            # If no CDP URL, launch local browser
            if not self._cdp_url:
                from openbrowser.browser.events import BrowserLaunchEvent, BrowserLaunchResult

                launch_event = self.event_bus.dispatch(BrowserLaunchEvent())
                await launch_event

                launch_result: BrowserLaunchResult = await launch_event.event_result(
                    raise_if_none=True, raise_if_any=True
                )
                self._cdp_url = launch_result.cdp_url

            assert self._cdp_url and '://' in self._cdp_url

            # Only connect if not already connected
            if self._cdp_client_root is None:
                await self.connect(cdp_url=self._cdp_url)
                assert self._cdp_client_root is not None

                # SessionManager is initialized inside connect() method
                assert self._session_manager is not None, 'SessionManager should be initialized in connect()'

                # Notify that browser is connected (this triggers watchdogs that listen to BrowserConnectedEvent)
                from openbrowser.browser.events import BrowserConnectedEvent

                self.event_bus.dispatch(BrowserConnectedEvent(cdp_url=self._cdp_url))
            else:
                self.logger.debug('Already connected to CDP, skipping reconnection')

            return {'cdp_url': self._cdp_url}

        except Exception as e:
            from openbrowser.browser.events import BrowserErrorEvent

            self.event_bus.dispatch(
                BrowserErrorEvent(
                    error_type='BrowserStartEventError',
                    message=f'Failed to start browser: {type(e).__name__} {e}',
                    details={'cdp_url': self._cdp_url},
                )
            )
            raise

    async def on_BrowserStopEvent(self, event) -> None:
        """Handle browser stop request.

        Clears CDP session cache, resets state, and dispatches
        BrowserStoppedEvent to notify watchdogs.

        Args:
            event: BrowserStopEvent with force flag.
        """
        try:
            # Clear CDP session cache before stopping
            await self.reset()

            # Reset state
            self._cdp_url = None

            # Notify stop
            from openbrowser.browser.events import BrowserStoppedEvent

            stop_event = self.event_bus.dispatch(BrowserStoppedEvent(reason='Stopped by request'))
            await stop_event

        except Exception as e:
            from openbrowser.browser.events import BrowserErrorEvent

            self.event_bus.dispatch(
                BrowserErrorEvent(
                    error_type='BrowserStopEventError',
                    message=f'Failed to stop browser: {type(e).__name__} {e}',
                    details={'cdp_url': self._cdp_url},
                )
            )

    async def on_NavigateToUrlEvent(self, event) -> None:
        """Handle navigation requests - core browser functionality.

        Navigates to the specified URL, optionally in a new tab.
        Reuses existing about:blank tabs when opening new tabs.
        Dispatches NavigationStartedEvent and NavigationCompleteEvent.

        Args:
            event: NavigateToUrlEvent with url, new_tab, and wait_until options.

        Raises:
            Exception: If navigation fails.
        """
        self.logger.debug(f'[on_NavigateToUrlEvent] Received NavigateToUrlEvent: url={event.url}, new_tab={event.new_tab}')
        if not self.agent_focus:
            self.logger.warning('Cannot navigate - browser not connected')
            return

        target_id = None

        # Check if the url is already open in a tab somewhere that we're not currently on
        targets = await self._cdp_get_all_pages()
        for target in targets:
            if target.get('url') == event.url and target['targetId'] != self.agent_focus.target_id and not event.new_tab:
                target_id = target['targetId']
                event.new_tab = False
                break

        try:
            # Find or create target for navigation
            self.logger.debug(f'[on_NavigateToUrlEvent] Processing new_tab={event.new_tab}')
            if event.new_tab:
                # Look for existing about:blank tab that's not the current one
                targets = await self._cdp_get_all_pages()
                self.logger.debug(f'[on_NavigateToUrlEvent] Found {len(targets)} existing tabs')
                current_target_id = self.agent_focus.target_id if self.agent_focus else None

                for target in targets:
                    if target.get('url') == 'about:blank' and target['targetId'] != current_target_id:
                        target_id = target['targetId']
                        self.logger.debug(f'[on_NavigateToUrlEvent] Reusing existing about:blank tab #{target_id[-4:]}')
                        break

                # Create new tab if no reusable one found
                if not target_id:
                    self.logger.debug('[on_NavigateToUrlEvent] No reusable about:blank tab found, creating new tab...')
                    try:
                        target_id = await self._cdp_create_new_page('about:blank')
                        self.logger.debug(f'[on_NavigateToUrlEvent] Created new page with target_id: {target_id}')

                        # Dispatch TabCreatedEvent for new tab
                        from openbrowser.browser.events import TabCreatedEvent

                        await self.event_bus.dispatch(TabCreatedEvent(target_id=target_id, url='about:blank'))
                    except Exception as e:
                        self.logger.error(f'[on_NavigateToUrlEvent] Failed to create new tab: {type(e).__name__}: {e}')
                        # Fall back to using current tab
                        target_id = self.agent_focus.target_id
                        self.logger.warning(f'[on_NavigateToUrlEvent] Falling back to current tab #{target_id[-4:]}')
            else:
                # Use current tab
                target_id = target_id or self.agent_focus.target_id

            # Only switch tab if we're not already on the target tab
            if self.agent_focus is None or self.agent_focus.target_id != target_id:
                self.logger.debug(
                    f'[on_NavigateToUrlEvent] Switching to target tab {target_id[-4:]} '
                    f'(current: {self.agent_focus.target_id[-4:] if self.agent_focus else "none"})'
                )
                # Activate target (bring to foreground)
                from openbrowser.browser.events import SwitchTabEvent

                await self.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
            else:
                self.logger.debug(f'[on_NavigateToUrlEvent] Already on target tab {target_id[-4:]}, skipping SwitchTabEvent')

            # Ensure agent_focus is set to the target
            if not self.agent_focus or self.agent_focus.target_id != target_id:
                await self.get_or_create_cdp_session(target_id=target_id, focus=True)

            assert self.agent_focus is not None and self.agent_focus.target_id == target_id, (
                'Agent focus not updated to new target_id after SwitchTabEvent'
            )

            # Dispatch navigation started
            from openbrowser.browser.events import NavigationStartedEvent

            await self.event_bus.dispatch(NavigationStartedEvent(target_id=target_id, url=event.url))

            # Navigate to URL
            await self.agent_focus.cdp_client.send.Page.navigate(
                params={
                    'url': event.url,
                    'transitionType': 'address_bar',
                },
                session_id=self.agent_focus.session_id,
            )

            # Wait a bit to ensure page starts loading
            await asyncio.sleep(1)

            # Dispatch navigation complete
            from openbrowser.browser.events import NavigationCompleteEvent

            self.logger.debug(f'[on_NavigateToUrlEvent] Dispatching NavigationCompleteEvent for {event.url}')
            await self.event_bus.dispatch(
                NavigationCompleteEvent(
                    target_id=target_id,
                    url=event.url,
                    status=None,
                )
            )

        except Exception as e:
            self.logger.error(f'[on_NavigateToUrlEvent] Navigation failed: {type(e).__name__}: {e}')
            raise

    async def connect(self, cdp_url: str | None = None) -> None:
        """Connect to a remote chromium-based browser via CDP using cdp-use.

        Establishes WebSocket connection to browser, enables auto-attach for
        new targets, attaches to existing targets, and sets initial agent focus.

        Args:
            cdp_url: WebSocket URL for CDP connection. If HTTP URL provided,
                fetches WebSocket URL from /json/version endpoint.

        Raises:
            RuntimeError: If CDP URL is not provided or connection fails.
        """
        self._cdp_url = cdp_url or self._cdp_url
        if not self._cdp_url:
            raise RuntimeError('Cannot setup CDP connection without CDP URL')

        import httpx

        if not self._cdp_url.startswith('ws'):
            # If it's an HTTP URL, fetch the WebSocket URL from /json/version endpoint
            url = self._cdp_url.rstrip('/')
            if not url.endswith('/json/version'):
                url = url + '/json/version'

            async with httpx.AsyncClient() as client:
                version_info = await client.get(url)
                self._cdp_url = version_info.json()['webSocketDebuggerUrl']

        assert self._cdp_url is not None

        browser_location = 'local browser'
        self.logger.debug(f'Connecting to existing chromium-based browser via CDP: {self._cdp_url} -> ({browser_location})')

        try:
            # Create and store the CDP client for direct CDP communication
            self._cdp_client_root = CDPClient(self._cdp_url)
            assert self._cdp_client_root is not None
            await self._cdp_client_root.start()

            # Initialize SessionManager FIRST (before enabling autoAttach)
            # This ensures session manager is ready to handle attach/detach events
            from openbrowser.browser.session_manager import SessionManager
            
            self._session_manager = SessionManager(self)
            await self._session_manager.start_monitoring()
            self.logger.debug('Event-driven session manager started')

            # Enable auto-attach so Chrome automatically notifies us when NEW targets attach/detach
            # This is the foundation of event-driven session management
            await self._cdp_client_root.send.Target.setAutoAttach(
                params={'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True}
            )
            self.logger.debug('CDP client connected with auto-attach enabled')

            # Get browser targets to find available contexts/pages
            targets = await self._cdp_client_root.send.Target.getTargets()

            # Manually attach to ALL EXISTING targets (autoAttach only fires for new ones)
            # We attach to everything (pages, iframes, workers) for complete coverage
            for target in targets['targetInfos']:
                target_id = target['targetId']
                target_type = target.get('type', 'unknown')

                try:
                    # Attach to target - this triggers attachedToTarget event
                    result = await self._cdp_client_root.send.Target.attachToTarget(
                        params={'targetId': target_id, 'flatten': True}
                    )
                    session_id = result['sessionId']

                    # Enable auto-attach for this target's children
                    await self._cdp_client_root.send.Target.setAutoAttach(
                        params={'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True},
                        session_id=session_id
                    )

                    # Note: SessionManager will create CDPSession via attachedToTarget event
                    # We just attach here and let SessionManager handle session creation
                    self.logger.debug(
                        f'Attached to existing target: {target_id[:8]}... (type={target_type}, session={session_id[:8]}...)'
                    )
                except Exception as e:
                    self.logger.debug(f'Failed to attach to existing target {target_id[:8]}... (type={target_type}): {e}')

            # Find main browser pages (avoiding iframes, workers, extensions, etc.)
            page_targets = [
                t
                for t in targets['targetInfos']
                if self._is_valid_target(
                    t, include_http=True, include_about=True, include_pages=True, include_iframes=False, include_workers=False
                )
            ]

            # Check for chrome://newtab pages and redirect them to about:blank
            for target in page_targets:
                target_url = target.get('url', '')
                if target_url in ('chrome://new-tab-page/', 'chrome://newtab/', 'chrome://newtab/') and target_url != 'about:blank':
                    target_id = target['targetId']
                    self.logger.debug(f'[connect] Redirecting {target_url} to about:blank for target {target_id}')
                    try:
                        # Wait for SessionManager to create session
                        for _ in range(20):
                            await asyncio.sleep(0.1)
                            session = await self._session_manager.get_session_for_target(target_id)
                            if session:
                                await session.cdp_client.send.Page.navigate(
                                    params={'url': 'about:blank'}, session_id=session.session_id
                                )
                                target['url'] = 'about:blank'
                                await asyncio.sleep(0.05)  # Let navigation start
                                break
                    except Exception as e:
                        self.logger.warning(f'[connect] Failed to redirect {target_url}: {e}')

            # Ensure we have at least one page
            if not page_targets:
                new_target = await self._cdp_client_root.send.Target.createTarget(params={'url': 'about:blank'})
                target_id = new_target['targetId']
                self.logger.debug(f'Created new blank page: {target_id}')
                
                # Attach to the newly created target
                try:
                    result = await self._cdp_client_root.send.Target.attachToTarget(
                        params={'targetId': target_id, 'flatten': True}
                    )
                    session_id = result['sessionId']
                    
                    # Enable auto-attach for this target's children
                    await self._cdp_client_root.send.Target.setAutoAttach(
                        params={'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True},
                        session_id=session_id
                    )
                    
                    # Note: SessionManager will create CDPSession via attachedToTarget event
                    # Wait a bit for SessionManager to create the session
                    for _ in range(20):  # Wait up to 2 seconds
                        await asyncio.sleep(0.1)
                        if target_id in self._cdp_session_pool:
                            self.agent_focus = self._cdp_session_pool[target_id]
                            self.logger.debug(f'Agent focus set to new target {target_id[:8]}...')
                            break
                except Exception as e:
                    self.logger.error(f'Failed to attach to newly created target {target_id[:8]}...: {e}')
                    raise
            else:
                # Find first page type target (not iframe)
                page_target = next((p for p in page_targets if p.get('type') == 'page'), None)
                if page_target:
                    target_id = page_target['targetId']
                else:
                    target_id = page_targets[0]['targetId']
                self.logger.debug(f'[connect] Using existing page: {target_id}')

            # Wait for SessionManager to receive the attach event for this target
            # (Chrome will fire Target.attachedToTarget event which SessionManager handles)
            for _ in range(20):  # Wait up to 2 seconds
                await asyncio.sleep(0.1)
                session = await self._session_manager.get_session_for_target(target_id)
                if session:
                    self.agent_focus = session
                    # SessionManager already added it to pool - no need to do it manually
                    self.logger.debug(f'[connect] Agent focus set to {target_id[:8]}...')
                    break

            if not self.agent_focus:
                raise RuntimeError(f'Failed to get session for initial target {target_id}')

            # Dispatch TabCreatedEvent for all initial tabs (so watchdogs can initialize)
            for idx, target in enumerate(page_targets):
                target_url = target.get('url', '')
                self.logger.debug(f'[connect] Dispatching TabCreatedEvent for initial tab {idx}: {target_url}')
                from openbrowser.browser.events import TabCreatedEvent

                self.event_bus.dispatch(TabCreatedEvent(url=target_url, target_id=target['targetId']))

            # Dispatch initial focus event
            if page_targets:
                initial_url = page_targets[0].get('url', '')
                from openbrowser.browser.events import AgentFocusChangedEvent

                self.event_bus.dispatch(AgentFocusChangedEvent(target_id=page_targets[0]['targetId'], url=initial_url))
                self.logger.debug(f'[connect] Initial agent focus set to tab 0: {initial_url}')

            # Verify the session is working
            if self.agent_focus.title == 'Unknown title':
                self.logger.warning('Session created but title is unknown (may be normal for about:blank)')

        except Exception as e:
            self.logger.error(f'FATAL: Failed to setup CDP connection: {e}')
            self._cdp_client_root = None
            self.agent_focus = None
            raise RuntimeError(f'Failed to establish CDP connection to browser: {e}') from e

    async def get_or_create_cdp_session(self, target_id: TargetID | None = None, focus: bool = True) -> CDPSession:
        """Get CDP session for a target from the pool or create it.

        Retrieves an existing session from the pool or creates a new one
        for the specified target. Optionally switches agent focus to the target.

        Args:
            target_id: Target ID to get session for. If None, uses current
                agent focus target.
            focus: If True, switches agent_focus to this target.

        Returns:
            CDPSession for the specified target.

        Raises:
            ValueError: If target_id is None and agent_focus is not set.
            AssertionError: If root CDP client is not initialized.

        Example:
            >>> session = await browser.get_or_create_cdp_session(target_id)
            >>> await session.cdp_client.send.DOM.enable(session_id=session.session_id)
        """
        assert self._cdp_client_root is not None, 'Root CDP client not initialized'

        # If no target_id specified, use current agent focus (if it exists)
        if target_id is None:
            if self.agent_focus is None:
                raise ValueError('target_id must be provided when agent_focus is not initialized')
            target_id = self.agent_focus.target_id

        # Check if session exists in pool
        if target_id in self._cdp_session_pool:
            session = self._cdp_session_pool[target_id]
        else:
            # Create new session
            session = await CDPSession.for_target(
                cdp_client=self._cdp_client_root,
                target_id=target_id,
            )
            self._cdp_session_pool[target_id] = session

        # Update focus if requested
        if focus:
            if self.agent_focus is None or self.agent_focus.target_id != target_id:
                if self.agent_focus is not None:
                    self.logger.debug(f'Switching focus: {self.agent_focus.target_id[:8]}... → {target_id[:8]}...')
                self.agent_focus = session

        # Resume if waiting for debugger
        if focus:
            try:
                await session.cdp_client.send.Runtime.runIfWaitingForDebugger(session_id=session.session_id)
            except Exception:
                pass

        return session

    async def on_SwitchTabEvent(self, event) -> TargetID:
        """Handle tab switching requests.

        Switches agent focus to the specified target and activates it
        visually in the browser.

        Args:
            event: SwitchTabEvent with optional target_id. If None,
                switches to the most recently opened tab.

        Returns:
            The target_id of the newly focused tab.

        Raises:
            RuntimeError: If browser not connected or no tabs available.
        """
        if not self._cdp_client_root:
            raise RuntimeError('Browser not connected')

        target_id = event.target_id

        # If no target_id specified, switch to most recent tab
        if target_id is None:
            all_pages = await self._cdp_get_all_pages()
            if not all_pages:
                raise RuntimeError('No tabs available to switch to')
            target_id = all_pages[-1]['targetId']

        # Get or create session for target
        session = await self.get_or_create_cdp_session(target_id=target_id, focus=True)

        # Activate the tab visually in browser
        try:
            await self._cdp_client_root.send.Target.activateTarget(params={'targetId': target_id})
        except Exception as e:
            self.logger.debug(f'Failed to activate tab visually: {e}')

        # Dispatch focus changed event
        from openbrowser.browser.events import AgentFocusChangedEvent

        self.event_bus.dispatch(AgentFocusChangedEvent(target_id=target_id, url=session.url))

        return target_id

    async def on_CloseTabEvent(self, event) -> None:
        """Handle tab close requests.

        Closes the specified tab and switches focus to another tab
        if the closed tab was the current focus.

        Args:
            event: CloseTabEvent with target_id of tab to close.

        Raises:
            RuntimeError: If browser not connected.
        """
        if not self._cdp_client_root:
            raise RuntimeError('Browser not connected')

        target_id = event.target_id

        # Close the target
        await self._cdp_close_page(target_id)

        # If this was the agent focus, switch to another tab
        if self.agent_focus and self.agent_focus.target_id == target_id:
            all_pages = await self._cdp_get_all_pages()
            if all_pages:
                # Switch to most recent remaining tab
                new_target_id = all_pages[-1]['targetId']
                await self.get_or_create_cdp_session(target_id=new_target_id, focus=True)
            else:
                # No tabs left, create a new one
                new_target_id = await self._cdp_create_new_page('about:blank')
                await self.get_or_create_cdp_session(target_id=new_target_id, focus=True)

    async def on_FileDownloadedEvent(self, event) -> None:
        """Track downloaded files during this session.

        Adds the downloaded file path to the session's download list
        for later reference.

        Args:
            event: FileDownloadedEvent with file_name and path.
        """
        self.logger.debug(f'FileDownloadedEvent received: {event.file_name} at {event.path}')
        if event.path and event.path not in self._downloaded_files:
            self._downloaded_files.append(event.path)
            self.logger.info(
                f'Tracked download: {event.file_name} ({len(self._downloaded_files)} total downloads in session)'
            )

    async def _cdp_get_all_pages(
        self,
        include_http: bool = True,
        include_about: bool = True,
        include_pages: bool = True,
        include_iframes: bool = False,
        include_workers: bool = False,
    ) -> list[dict]:
        """Get all browser pages/tabs using CDP Target.getTargets.

        Retrieves target info for all matching browser targets based on
        filter criteria.

        Args:
            include_http: Include HTTP/HTTPS pages in results.
            include_about: Include about: pages (e.g., about:blank).
            include_pages: Include page/tab type targets.
            include_iframes: Include iframe type targets.
            include_workers: Include worker type targets.

        Returns:
            List of target info dictionaries with keys like 'targetId',
            'url', 'title', 'type', etc.

        Example:
            >>> pages = await session._cdp_get_all_pages(include_iframes=True)
            >>> for page in pages:
            ...     print(page['url'])
        """
        if not self._cdp_client_root:
            return []

        targets = await self._cdp_client_root.send.Target.getTargets()

        def _is_valid_target(target: dict) -> bool:
            target_type = target.get('type', 'unknown')
            url = target.get('url', '')

            # Filter by type
            if target_type == 'page' and not include_pages:
                return False
            if target_type == 'iframe' and not include_iframes:
                return False
            if target_type == 'worker' and not include_workers:
                return False

            # Filter by URL
            if url.startswith('about:') and not include_about:
                return False
            if (url.startswith('http://') or url.startswith('https://')) and not include_http:
                return False

            # Only include page/tab types
            return target_type in ('page', 'tab')

        return [t for t in targets.get('targetInfos', []) if _is_valid_target(t)]

    async def _cdp_create_new_page(
        self, url: str = 'about:blank', background: bool = False, new_window: bool = False
    ) -> TargetID:
        """Create a new page/tab using CDP Target.createTarget.

        Creates a new browser tab or window and optionally navigates to a URL.

        Args:
            url: Initial URL for the new page (default: 'about:blank').
            background: If True, open tab in background without focusing.
            new_window: If True, open in new browser window instead of tab.

        Returns:
            Target ID of the newly created page.

        Raises:
            RuntimeError: If browser not connected.

        Example:
            >>> target_id = await session._cdp_create_new_page('https://example.com')
        """
        if not self._cdp_client_root:
            raise RuntimeError('Browser not connected')

        result = await self._cdp_client_root.send.Target.createTarget(
            params={'url': url, 'newWindow': new_window, 'background': background}
        )
        return result['targetId']

    async def _cdp_close_page(self, target_id: TargetID) -> None:
        """Close a page/tab using CDP Target.closeTarget.

        Closes the specified browser target. Does not switch focus if
        the closed target was focused - caller must handle that.

        Args:
            target_id: Target ID of the page to close.

        Raises:
            RuntimeError: If browser not connected.
        """
        if not self._cdp_client_root:
            raise RuntimeError('Browser not connected')

        await self._cdp_client_root.send.Target.closeTarget(params={'targetId': target_id})

    async def _cdp_get_storage_state(self) -> dict:
        """Get browser storage state (cookies, localStorage, etc.) using CDP.

        Retrieves current browser storage state for persistence or analysis.

        Returns:
            Dictionary with 'cookies' list and 'origins' list.
            Each cookie contains name, value, domain, path, etc.

        Raises:
            RuntimeError: If no active session.
        """
        if not self.agent_focus:
            raise RuntimeError('No active session')

        # Get cookies
        cookies_result = await self.agent_focus.cdp_client.send.Network.getCookies(
            session_id=self.agent_focus.session_id
        )
        cookies = cookies_result.get('cookies', [])

        # Get storage state structure
        storage_state = {
            'cookies': cookies,
            'origins': [],
        }

        return storage_state

    @staticmethod
    def _is_valid_target(
        target_info: dict,
        include_http: bool = True,
        include_chrome: bool = False,
        include_chrome_extensions: bool = False,
        include_chrome_error: bool = False,
        include_about: bool = True,
        include_iframes: bool = True,
        include_pages: bool = True,
        include_workers: bool = False,
    ) -> bool:
        """Check if a target should be processed based on filter criteria.

        Static method that evaluates whether a CDP target matches the
        specified inclusion filters.

        Args:
            target_info: Target info dict from CDP Target.getTargets.
            include_http: Include HTTP/HTTPS URLs.
            include_chrome: Include chrome:// URLs.
            include_chrome_extensions: Include chrome-extension:// URLs.
            include_chrome_error: Include chrome-error:// URLs.
            include_about: Include about: URLs (only about:blank).
            include_iframes: Include iframe/webview type targets.
            include_pages: Include page/tab type targets.
            include_workers: Include worker type targets.

        Returns:
            True if target matches filters and should be processed.
        """
        target_type = target_info.get('type', '')
        url = target_info.get('url', '')

        url_allowed, type_allowed = False, False

        # Always allow new tab pages (chrome://new-tab-page/, chrome://newtab/, about:blank)
        # so they can be redirected to about:blank in connect()
        if url in ('chrome://new-tab-page/', 'chrome://newtab/', 'about:blank', 'about:newtab'):
            url_allowed = True

        if url.startswith('chrome-error://') and include_chrome_error:
            url_allowed = True

        if url.startswith('chrome://') and include_chrome:
            url_allowed = True

        if url.startswith('chrome-extension://') and include_chrome_extensions:
            url_allowed = True

        # Don't allow about:srcdoc! there are also other rare about: pages that we want to avoid
        if url == 'about:blank' and include_about:
            url_allowed = True

        if (url.startswith('http://') or url.startswith('https://')) and include_http:
            url_allowed = True

        if target_type in ('service_worker', 'shared_worker', 'worker') and include_workers:
            type_allowed = True

        if target_type in ('page', 'tab') and include_pages:
            type_allowed = True

        if target_type in ('iframe', 'webview') and include_iframes:
            type_allowed = True

        return url_allowed and type_allowed

    async def _cdp_get_cookies(self) -> list[dict]:
        """Get all cookies from the browser using CDP.

        Retrieves all cookies accessible to the current page.

        Returns:
            List of cookie dictionaries with keys: name, value, domain,
            path, secure, httpOnly, sameSite, expires, etc.

        Raises:
            RuntimeError: If no active session.
        """
        if not self.agent_focus:
            raise RuntimeError('No active session')

        cookies_result = await self.agent_focus.cdp_client.send.Network.getCookies(
            session_id=self.agent_focus.session_id
        )
        return cookies_result.get('cookies', [])

    async def _cdp_set_cookies(self, cookies: list[dict]) -> None:
        """Set cookies in the browser using CDP.

        Sets cookies one by one. Silently logs failures for individual
        cookies without raising exceptions.

        Args:
            cookies: List of cookie dictionaries to set. Each should have
                'name', 'value', 'domain' at minimum.

        Raises:
            RuntimeError: If no active session.
        """
        if not self.agent_focus:
            raise RuntimeError('No active session')

        # Set cookies one by one
        for cookie in cookies:
            try:
                await self.agent_focus.cdp_client.send.Network.setCookie(
                    params={
                        'name': cookie.get('name', ''),
                        'value': cookie.get('value', ''),
                        'domain': cookie.get('domain', ''),
                        'path': cookie.get('path', '/'),
                        'secure': cookie.get('secure', False),
                        'httpOnly': cookie.get('httpOnly', False),
                        'sameSite': cookie.get('sameSite', 'None'),
                        'expires': cookie.get('expires', -1),
                    },
                    session_id=self.agent_focus.session_id,
                )
            except Exception as e:
                self.logger.debug(f'Failed to set cookie {cookie.get("name")}: {e}')

    async def attach_all_watchdogs(self) -> None:
        """Attach all watchdogs to the browser session.

        Initializes and attaches all watchdog components following the
        browser-use pattern. Watchdogs handle various browser behaviors:

        - LocalBrowserWatchdog: Browser process lifecycle
        - DownloadsWatchdog: File download monitoring
        - PopupsWatchdog: JavaScript dialog handling
        - SecurityWatchdog: URL access policy enforcement
        - StorageStateWatchdog: Cookie/localStorage persistence
        - PermissionsWatchdog: Browser permission grants
        - RecordingWatchdog: Video recording (if enabled)
        - ScreenshotWatchdog: Screenshot capture
        - DOMWatchdog: DOM tree management

        Called automatically during session start. Prevents duplicate
        attachment if called multiple times.
        """
        # Prevent duplicate watchdog attachment
        if self._watchdogs_attached:
            self.logger.debug('Watchdogs already attached, skipping duplicate attachment')
            return

        # Initialize LocalBrowserWatchdog FIRST (needs to handle BrowserLaunchEvent)
        from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        LocalBrowserWatchdog.model_rebuild()
        self._local_browser_watchdog = LocalBrowserWatchdog(event_bus=self.event_bus, browser_session=self)
        self._local_browser_watchdog.attach_to_session()

        # Initialize DownloadsWatchdog
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        DownloadsWatchdog.model_rebuild()
        self._downloads_watchdog = DownloadsWatchdog(event_bus=self.event_bus, browser_session=self)
        self._downloads_watchdog.attach_to_session()
        if self.browser_profile.auto_download_pdfs:
            self.logger.debug('PDF auto-download enabled for this session')

        # Initialize PopupsWatchdog
        from openbrowser.browser.watchdogs.popups_watchdog import PopupsWatchdog

        PopupsWatchdog.model_rebuild()
        self._popups_watchdog = PopupsWatchdog(event_bus=self.event_bus, browser_session=self)
        self._popups_watchdog.attach_to_session()

        # Initialize SecurityWatchdog
        from openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

        SecurityWatchdog.model_rebuild()
        self._security_watchdog = SecurityWatchdog(event_bus=self.event_bus, browser_session=self)
        self._security_watchdog.attach_to_session()

        # Initialize StorageStateWatchdog conditionally
        # Enable when user provides either storage_state or user_data_dir
        should_enable_storage_state = (
            self.browser_profile.storage_state is not None or self.browser_profile.user_data_dir is not None
        )

        if should_enable_storage_state:
            from openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

            StorageStateWatchdog.model_rebuild()
            self._storage_state_watchdog = StorageStateWatchdog(
                event_bus=self.event_bus,
                browser_session=self,
                auto_save_interval=60.0,  # 1 minute instead of 30 seconds
                save_on_change=False,  # Only save on shutdown by default
            )
            self._storage_state_watchdog.attach_to_session()
            self.logger.debug(
                f'StorageStateWatchdog enabled (storage_state: {bool(self.browser_profile.storage_state)}, '
                f'user_data_dir: {bool(self.browser_profile.user_data_dir)})'
            )
        else:
            self.logger.debug('StorageStateWatchdog disabled (no storage_state or user_data_dir configured)')

        # Initialize PermissionsWatchdog
        from openbrowser.browser.watchdogs.permissions_watchdog import PermissionsWatchdog

        PermissionsWatchdog.model_rebuild()
        self._permissions_watchdog = PermissionsWatchdog(event_bus=self.event_bus, browser_session=self)
        self._permissions_watchdog.attach_to_session()

        # Initialize RecordingWatchdog if video recording is enabled
        if self.browser_profile.record_video_dir:
            from openbrowser.browser.watchdogs.recording_watchdog import RecordingWatchdog

            RecordingWatchdog.model_rebuild()
            self._recording_watchdog = RecordingWatchdog(event_bus=self.event_bus, browser_session=self)
            self._recording_watchdog.attach_to_session()

        # Initialize ScreenshotWatchdog
        from openbrowser.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog

        ScreenshotWatchdog.model_rebuild()
        self._screenshot_watchdog = ScreenshotWatchdog(event_bus=self.event_bus, browser_session=self)
        self._screenshot_watchdog.attach_to_session()

        # Initialize DOMWatchdog
        from openbrowser.browser.watchdogs.dom_watchdog import DOMWatchdog

        DOMWatchdog.model_rebuild()
        self._dom_watchdog = DOMWatchdog(event_bus=self.event_bus, browser_session=self)
        self._dom_watchdog.attach_to_session()

        # Mark watchdogs as attached to prevent duplicate attachment
        self._watchdogs_attached = True

        self.logger.debug('All watchdogs attached to browser session')

    # region - ========== Helper Methods ==========

    async def get_current_target_info(self) -> dict | None:
        """Get info about the current active target using CDP.

        Retrieves full target information for the currently focused target.

        Returns:
            Target info dict with 'targetId', 'type', 'title', 'url', etc.
            Returns None if no agent focus is set.
        """
        if not self.agent_focus or not self.agent_focus.target_id:
            return None

        targets = await self.cdp_client.send.Target.getTargets()
        for target in targets.get('targetInfos', []):
            if target.get('targetId') == self.agent_focus.target_id:
                return target
        return None

    async def get_current_page_url(self) -> str:
        """Get the URL of the current page using CDP.

        Returns:
            Current page URL string. Returns 'about:blank' if not available.

        Example:
            >>> url = await session.get_current_page_url()
            >>> print(url)  # 'https://example.com'
        """
        target = await self.get_current_target_info()
        if target:
            return target.get('url', '')
        return 'about:blank'

    async def get_current_page_title(self) -> str:
        """Get the title of the current page using CDP.

        Returns:
            Current page title string. Returns 'Unknown page title'
            if not available.
        """
        target_info = await self.get_current_target_info()
        if target_info:
            return target_info.get('title', 'Unknown page title')
        return 'Unknown page title'

    async def navigate_to(self, url: str, new_tab: bool = False) -> None:
        """Navigate to a URL using the standard event system.

        High-level navigation method that dispatches NavigateToUrlEvent
        and waits for completion.

        Args:
            url: URL to navigate to.
            new_tab: Whether to open in a new tab.

        Raises:
            Exception: If navigation fails or is blocked by security policy.

        Example:
            >>> await session.navigate_to('https://example.com')
            >>> await session.navigate_to('https://other.com', new_tab=True)
        """
        from openbrowser.browser.events import NavigateToUrlEvent

        event = self.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=new_tab))
        await event
        await event.event_result(raise_if_any=True, raise_if_none=False)

    async def get_tabs(self) -> list[Any]:
        """Get information about all open tabs using CDP Target.getTargetInfo.

        Retrieves TabInfo for each open browser tab with URL and title.

        Returns:
            List of TabInfo objects with target_id, url, title, and
            parent_target_id (for popups/iframes).

        Example:
            >>> tabs = await session.get_tabs()
            >>> for tab in tabs:
            ...     print(f"{tab.title}: {tab.url}")
        """
        from openbrowser.browser.views import TabInfo

        tabs = []

        # Safety check - return empty list if browser not connected yet
        if not self._cdp_client_root:
            return tabs

        # Get all page targets using CDP
        pages = await self._cdp_get_all_pages()

        for i, page_target in enumerate(pages):
            target_id = page_target['targetId']
            url = page_target['url']

            # Try to get the title directly from Target.getTargetInfo - much faster!
            try:
                target_info = await self.cdp_client.send.Target.getTargetInfo(params={'targetId': target_id})
                title = target_info.get('targetInfo', {}).get('title', '')

                # Skip JS execution for chrome:// pages and new tab pages
                if url in ('chrome://new-tab-page/', 'chrome://newtab/', 'about:blank', 'about:newtab'):
                    if url in ('chrome://new-tab-page/', 'chrome://newtab/', 'about:newtab'):
                        title = ''
                    elif not title:
                        title = url
                elif url.startswith('chrome://'):
                    if not title:
                        title = url

                # Special handling for PDF pages without titles
                if (not title or title == '') and (url.endswith('.pdf') or 'pdf' in url):
                    try:
                        from urllib.parse import urlparse
                        filename = urlparse(url).path.split('/')[-1]
                        if filename:
                            title = filename
                    except Exception:
                        pass

            except Exception as e:
                # Fallback to basic title handling
                self.logger.debug(f'Failed to get target info for tab #{i}: {url} - {type(e).__name__}')

                if url in ('chrome://new-tab-page/', 'chrome://newtab/', 'about:blank', 'about:newtab'):
                    title = ''
                elif url.startswith('chrome://'):
                    title = url
                else:
                    title = ''

            tab_info = TabInfo(
                target_id=target_id,
                url=url,
                title=title,
                parent_target_id=None,
            )
            tabs.append(tab_info)

        return tabs

    async def get_all_frames(self) -> tuple[dict[str, dict], dict[str, str]]:
        """Get a complete frame hierarchy from all browser targets.

        Collects frame information including cross-origin iframes (OOPIFs)
        if enabled in the browser profile.

        Returns:
            Tuple of (all_frames, target_sessions) where:
            - all_frames: Dict mapping frame_id to frame info dict with
              'url', 'parentFrameId', 'childFrameIds', 'isCrossOrigin', etc.
            - target_sessions: Dict mapping target_id to session_id for
              targets with active sessions.

        Example:
            >>> frames, sessions = await session.get_all_frames()
            >>> for frame_id, info in frames.items():
            ...     print(f"{frame_id}: {info['url']}")
        """
        all_frames = {}  # frame_id -> FrameInfo dict
        target_sessions = {}  # target_id -> session_id (keep sessions alive during collection)

        # Check if cross-origin iframe support is enabled
        include_cross_origin = getattr(self.browser_profile, 'cross_origin_iframes', False)

        # Get all targets - only include iframes if cross-origin support is enabled
        targets = await self._cdp_get_all_pages(
            include_http=True,
            include_about=True,
            include_pages=True,
            include_iframes=include_cross_origin,
            include_workers=False,
        )

        # First pass: collect frame trees from ALL targets
        for target in targets:
            target_id = target['targetId']

            # Skip iframe targets if cross-origin support is disabled
            if not include_cross_origin and target.get('type') == 'iframe':
                continue

            # When cross-origin support is disabled, only process the current target
            if not include_cross_origin:
                # Only process the current focus target
                if self.agent_focus and target_id != self.agent_focus.target_id:
                    continue
                # Use the existing agent_focus session
                cdp_session = self.agent_focus
            else:
                # Get cached session for this target (don't change focus - iterating frames)
                cdp_session = await self.get_or_create_cdp_session(target_id=target_id, focus=False)

            if cdp_session:
                target_sessions[target_id] = cdp_session.session_id

                try:
                    # Try to get frame tree (not all target types support this)
                    frame_tree_result = await cdp_session.cdp_client.send.Page.getFrameTree(
                        session_id=cdp_session.session_id
                    )

                    # Process the frame tree recursively
                    def process_frame_tree(node, parent_frame_id=None):
                        """Recursively process frame tree and add to all_frames."""
                        frame = node.get('frame', {})
                        current_frame_id = frame.get('id')

                        if current_frame_id:
                            # For iframe targets, check if the frame has a parentId field
                            actual_parent_id = frame.get('parentId') or parent_frame_id

                            # Create frame info with all CDP response data plus our additions
                            frame_info = {
                                **frame,  # Include all original frame data: id, url, parentId, etc.
                                'frameTargetId': target_id,  # Target that can access this frame
                                'parentFrameId': actual_parent_id,  # Use parentId from frame if available
                                'childFrameIds': [],  # Will be populated below
                                'isCrossOrigin': False,  # Will be determined based on context
                            }

                            # Check if frame is cross-origin based on crossOriginIsolatedContextType
                            cross_origin_type = frame.get('crossOriginIsolatedContextType')
                            if cross_origin_type and cross_origin_type != 'NotIsolated':
                                frame_info['isCrossOrigin'] = True

                            # For iframe targets, the frame itself is likely cross-origin
                            if target.get('type') == 'iframe':
                                frame_info['isCrossOrigin'] = True

                            # Skip cross-origin frames if support is disabled
                            if not include_cross_origin and frame_info.get('isCrossOrigin'):
                                return  # Skip this frame and its children

                            # Add child frame IDs (note: OOPIFs won't appear here)
                            child_frames = node.get('childFrames', [])
                            for child in child_frames:
                                child_frame = child.get('frame', {})
                                child_frame_id = child_frame.get('id')
                                if child_frame_id:
                                    frame_info['childFrameIds'].append(child_frame_id)

                            # Store or merge frame info
                            if current_frame_id in all_frames:
                                # Frame already seen from another target, merge info
                                existing = all_frames[current_frame_id]
                                # If this is an iframe target, it has direct access to the frame
                                if target.get('type') == 'iframe':
                                    existing['frameTargetId'] = target_id
                                    existing['isCrossOrigin'] = True
                            else:
                                all_frames[current_frame_id] = frame_info

                            # Process child frames recursively (only if we're not skipping this frame)
                            if include_cross_origin or not frame_info.get('isCrossOrigin'):
                                for child in child_frames:
                                    process_frame_tree(child, current_frame_id)

                    # Process the entire frame tree
                    process_frame_tree(frame_tree_result.get('frameTree', {}))

                except Exception as e:
                    # Target doesn't support Page domain or has no frames
                    self.logger.debug(f'Failed to get frame tree for target {target_id}: {e}')

        return all_frames, target_sessions

    def update_cached_selector_map(self, selector_map: dict[int, Any]) -> None:
        """Update the cached selector map with new DOM state.

        Called by the DOM watchdog after rebuilding the DOM tree to
        cache element indices for quick lookup during interactions.

        Args:
            selector_map: Dict mapping element index to EnhancedDOMTreeNode
                or backend_node_id for element resolution.
        """
        self._cached_selector_map = selector_map

    @property
    def current_target_id(self) -> str | None:
        """Get current target ID from agent focus.

        Returns:
            Target ID of the currently focused target, or None if not set.
        """
        return self.agent_focus.target_id if self.agent_focus else None

    @property
    def current_session_id(self) -> str | None:
        """Get current session ID from agent focus.

        Returns:
            CDP session ID for the currently focused target, or None if not set.
        """
        return self.agent_focus.session_id if self.agent_focus else None

    # endregion - ========== Helper Methods ==========

