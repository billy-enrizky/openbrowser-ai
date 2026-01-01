"""Event-driven browser session following browser-use pattern."""

import asyncio
import logging
from typing import Any, Optional

from bubus import EventBus
from cdp_use import CDPClient
from cdp_use.cdp.target import SessionID, TargetID
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)


class CDPSession(BaseModel):
    """Info about a single CDP session bound to a specific target."""

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

        Args:
            cdp_client: The shared CDP client (root WebSocket connection)
            target_id: Target ID to attach to
            domains: List of CDP domains to enable. If None, enables default domains.
        """
        cdp_session = cls(
            cdp_client=cdp_client,
            target_id=target_id,
            session_id='connecting',
        )
        return await cdp_session.attach(domains=domains)

    async def attach(self, domains: list[str] | None = None):
        """Attach to target and enable domains."""
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
        """Get target info."""
        result = await self.cdp_client.send.Target.getTargetInfo(params={'targetId': self.target_id})
        return result['targetInfo']


class BrowserSession(BaseModel):
    """Event-driven browser session following browser-use pattern."""

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
        """Initialize BrowserSession with optional browser launch parameters.
        
        Args:
            debug_port: Port for Chrome remote debugging
            headless: Whether to run Chrome in headless mode
            user_data_dir: Optional user data directory for Chrome profile
            browser_profile: Optional BrowserProfile instance for advanced configuration
        """
        super().__init__(**kwargs)
        import tempfile
        from src.openbrowser.browser.profile import BrowserProfile
        
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
        """Get debug port."""
        return self._debug_port
    
    @property
    def headless(self) -> bool:
        """Get headless mode."""
        return self._headless
    
    @property
    def user_data_dir(self) -> Optional[str]:
        """Get user data directory."""
        return self._user_data_dir
    
    @property
    def browser_profile(self) -> Any:
        """Get browser profile."""
        return self._browser_profile
    
    @property
    def downloaded_files(self) -> list[str]:
        """Get list of downloaded files."""
        return self._downloaded_files.copy()

    @property
    def logger(self) -> logging.Logger:
        """Get instance-specific logger."""
        if self._logger is None:
            self._logger = logging.getLogger(f'openbrowser.browser_session')
        return self._logger

    @property
    def cdp_url(self) -> str | None:
        """Get CDP URL."""
        return self._cdp_url

    @cdp_url.setter
    def cdp_url(self, value: str | None) -> None:
        """Set CDP URL."""
        self._cdp_url = value

    @property
    def cdp_client(self) -> CDPClient:
        """Get the cached root CDP client."""
        assert self._cdp_client_root is not None, 'CDP client not initialized - browser may not be connected yet'
        return self._cdp_client_root

    async def reset(self) -> None:
        """Clear all cached CDP sessions with proper cleanup."""
        # Clear session pool
        self._cdp_session_pool.clear()
        self._cdp_client_root = None
        self.agent_focus = None
        self._cdp_url = None

    def model_post_init(self, __context) -> None:
        """Register event handlers after model initialization."""
        from src.openbrowser.browser.events import (
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
        """Start the browser session."""
        from src.openbrowser.browser.events import BrowserStartEvent

        start_event = self.event_bus.dispatch(BrowserStartEvent())
        await start_event
        await start_event.event_result(raise_if_any=True, raise_if_none=False)

    async def stop(self, force: bool = False) -> None:
        """Stop the browser session.
        
        Args:
            force: If True, kill the browser process. If False, keep browser alive.
        """
        from src.openbrowser.browser.events import BrowserStopEvent

        await self.event_bus.dispatch(BrowserStopEvent(force=force))
        await self.event_bus.stop(clear=True, timeout=5)
        await self.reset()
        self.event_bus = EventBus()

    async def on_BrowserStartEvent(self, event) -> dict[str, str]:
        """Handle browser start request.

        Returns:
            Dict with 'cdp_url' key containing the CDP URL
        """
        # Initialize and attach all watchdogs FIRST so LocalBrowserWatchdog can handle BrowserLaunchEvent
        await self.attach_all_watchdogs()

        try:
            # If no CDP URL, launch local browser
            if not self._cdp_url:
                from src.openbrowser.browser.events import BrowserLaunchEvent, BrowserLaunchResult

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
                from src.openbrowser.browser.events import BrowserConnectedEvent

                self.event_bus.dispatch(BrowserConnectedEvent(cdp_url=self._cdp_url))
            else:
                self.logger.debug('Already connected to CDP, skipping reconnection')

            return {'cdp_url': self._cdp_url}

        except Exception as e:
            from src.openbrowser.browser.events import BrowserErrorEvent

            self.event_bus.dispatch(
                BrowserErrorEvent(
                    error_type='BrowserStartEventError',
                    message=f'Failed to start browser: {type(e).__name__} {e}',
                    details={'cdp_url': self._cdp_url},
                )
            )
            raise

    async def on_BrowserStopEvent(self, event) -> None:
        """Handle browser stop request."""
        try:
            # Clear CDP session cache before stopping
            await self.reset()

            # Reset state
            self._cdp_url = None

            # Notify stop
            from src.openbrowser.browser.events import BrowserStoppedEvent

            stop_event = self.event_bus.dispatch(BrowserStoppedEvent(reason='Stopped by request'))
            await stop_event

        except Exception as e:
            from src.openbrowser.browser.events import BrowserErrorEvent

            self.event_bus.dispatch(
                BrowserErrorEvent(
                    error_type='BrowserStopEventError',
                    message=f'Failed to stop browser: {type(e).__name__} {e}',
                    details={'cdp_url': self._cdp_url},
                )
            )

    async def on_NavigateToUrlEvent(self, event) -> None:
        """Handle navigation requests - core browser functionality."""
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
                        from src.openbrowser.browser.events import TabCreatedEvent

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
                from src.openbrowser.browser.events import SwitchTabEvent

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
            from src.openbrowser.browser.events import NavigationStartedEvent

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
            from src.openbrowser.browser.events import NavigationCompleteEvent

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
        """Connect to a remote chromium-based browser via CDP using cdp-use."""
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
            from src.openbrowser.browser.session_manager import SessionManager
            
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
                from src.openbrowser.browser.events import TabCreatedEvent

                self.event_bus.dispatch(TabCreatedEvent(url=target_url, target_id=target['targetId']))

            # Dispatch initial focus event
            if page_targets:
                initial_url = page_targets[0].get('url', '')
                from src.openbrowser.browser.events import AgentFocusChangedEvent

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

        Args:
            target_id: Target ID to get session for. If None, uses current agent focus.
            focus: If True, switches agent focus to this target.

        Returns:
            CDPSession for the specified target.
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
        """Handle tab switching requests."""
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
        from src.openbrowser.browser.events import AgentFocusChangedEvent

        self.event_bus.dispatch(AgentFocusChangedEvent(target_id=target_id, url=session.url))

        return target_id

    async def on_CloseTabEvent(self, event) -> None:
        """Handle tab close requests."""
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
        """Track downloaded files during this session."""
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
        
        Args:
            include_http: Include HTTP/HTTPS pages
            include_about: Include about: pages
            include_pages: Include page targets
            include_iframes: Include iframe targets
            include_workers: Include worker targets
            
        Returns:
            List of target info dictionaries
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
        
        Args:
            url: URL to navigate to
            background: Whether to open in background
            new_window: Whether to open in new window
            
        Returns:
            Target ID of the new page
        """
        if not self._cdp_client_root:
            raise RuntimeError('Browser not connected')

        result = await self._cdp_client_root.send.Target.createTarget(
            params={'url': url, 'newWindow': new_window, 'background': background}
        )
        return result['targetId']

    async def _cdp_close_page(self, target_id: TargetID) -> None:
        """Close a page/tab using CDP Target.closeTarget.
        
        Args:
            target_id: Target ID to close
        """
        if not self._cdp_client_root:
            raise RuntimeError('Browser not connected')

        await self._cdp_client_root.send.Target.closeTarget(params={'targetId': target_id})

    async def _cdp_get_storage_state(self) -> dict:
        """Get browser storage state (cookies, localStorage, etc.) using CDP.
        
        Returns:
            Dictionary with cookies and origins
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
        """Check if a target should be processed.
        
        Args:
            target_info: Target info dict from CDP
            include_http: Include HTTP/HTTPS pages
            include_chrome: Include chrome:// pages
            include_chrome_extensions: Include chrome-extension:// pages
            include_chrome_error: Include chrome-error:// pages
            include_about: Include about: pages
            include_iframes: Include iframe targets
            include_pages: Include page/tab targets
            include_workers: Include worker targets
            
        Returns:
            True if target should be processed, False if it should be skipped
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
        """Get all cookies using CDP.
        
        Returns:
            List of cookie dictionaries
        """
        if not self.agent_focus:
            raise RuntimeError('No active session')

        cookies_result = await self.agent_focus.cdp_client.send.Network.getCookies(
            session_id=self.agent_focus.session_id
        )
        return cookies_result.get('cookies', [])

    async def _cdp_set_cookies(self, cookies: list[dict]) -> None:
        """Set cookies using CDP.
        
        Args:
            cookies: List of cookie dictionaries to set
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
        
        This method initializes and attaches all watchdogs following browser-use pattern.
        """
        # Prevent duplicate watchdog attachment
        if self._watchdogs_attached:
            self.logger.debug('Watchdogs already attached, skipping duplicate attachment')
            return

        # Initialize LocalBrowserWatchdog FIRST (needs to handle BrowserLaunchEvent)
        from src.openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

        LocalBrowserWatchdog.model_rebuild()
        self._local_browser_watchdog = LocalBrowserWatchdog(event_bus=self.event_bus, browser_session=self)
        self._local_browser_watchdog.attach_to_session()

        # Initialize DownloadsWatchdog
        from src.openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        DownloadsWatchdog.model_rebuild()
        self._downloads_watchdog = DownloadsWatchdog(event_bus=self.event_bus, browser_session=self)
        self._downloads_watchdog.attach_to_session()
        if self.browser_profile.auto_download_pdfs:
            self.logger.debug('PDF auto-download enabled for this session')

        # Initialize PopupsWatchdog
        from src.openbrowser.browser.watchdogs.popups_watchdog import PopupsWatchdog

        PopupsWatchdog.model_rebuild()
        self._popups_watchdog = PopupsWatchdog(event_bus=self.event_bus, browser_session=self)
        self._popups_watchdog.attach_to_session()

        # Initialize SecurityWatchdog
        from src.openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog

        SecurityWatchdog.model_rebuild()
        self._security_watchdog = SecurityWatchdog(event_bus=self.event_bus, browser_session=self)
        self._security_watchdog.attach_to_session()

        # Initialize StorageStateWatchdog conditionally
        # Enable when user provides either storage_state or user_data_dir
        should_enable_storage_state = (
            self.browser_profile.storage_state is not None or self.browser_profile.user_data_dir is not None
        )

        if should_enable_storage_state:
            from src.openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

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
        from src.openbrowser.browser.watchdogs.permissions_watchdog import PermissionsWatchdog

        PermissionsWatchdog.model_rebuild()
        self._permissions_watchdog = PermissionsWatchdog(event_bus=self.event_bus, browser_session=self)
        self._permissions_watchdog.attach_to_session()

        # Initialize RecordingWatchdog if video recording is enabled
        if self.browser_profile.record_video_dir:
            from src.openbrowser.browser.watchdogs.recording_watchdog import RecordingWatchdog

            RecordingWatchdog.model_rebuild()
            self._recording_watchdog = RecordingWatchdog(event_bus=self.event_bus, browser_session=self)
            self._recording_watchdog.attach_to_session()

        # Initialize ScreenshotWatchdog
        from src.openbrowser.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog

        ScreenshotWatchdog.model_rebuild()
        self._screenshot_watchdog = ScreenshotWatchdog(event_bus=self.event_bus, browser_session=self)
        self._screenshot_watchdog.attach_to_session()

        # Initialize DOMWatchdog
        from src.openbrowser.browser.watchdogs.dom_watchdog import DOMWatchdog

        DOMWatchdog.model_rebuild()
        self._dom_watchdog = DOMWatchdog(event_bus=self.event_bus, browser_session=self)
        self._dom_watchdog.attach_to_session()

        # Mark watchdogs as attached to prevent duplicate attachment
        self._watchdogs_attached = True

        self.logger.debug('All watchdogs attached to browser session')

    # region - ========== Helper Methods ==========

    async def get_current_target_info(self) -> dict | None:
        """Get info about the current active target using CDP.
        
        Returns:
            Target info dict or None if no agent focus
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
            Current page URL or 'about:blank' if not available
        """
        target = await self.get_current_target_info()
        if target:
            return target.get('url', '')
        return 'about:blank'

    async def get_current_page_title(self) -> str:
        """Get the title of the current page using CDP.
        
        Returns:
            Current page title or 'Unknown page title' if not available
        """
        target_info = await self.get_current_target_info()
        if target_info:
            return target_info.get('title', 'Unknown page title')
        return 'Unknown page title'

    async def navigate_to(self, url: str, new_tab: bool = False) -> None:
        """Navigate to a URL using the standard event system.
        
        Args:
            url: URL to navigate to
            new_tab: Whether to open in a new tab
        """
        from src.openbrowser.browser.events import NavigateToUrlEvent

        event = self.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=new_tab))
        await event
        await event.event_result(raise_if_any=True, raise_if_none=False)

    async def get_tabs(self) -> list[Any]:
        """Get information about all open tabs using CDP Target.getTargetInfo for speed.
        
        Returns:
            List of TabInfo objects
        """
        from src.openbrowser.browser.views import TabInfo

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
        
        Returns:
            Tuple of (all_frames, target_sessions) where:
            - all_frames: dict mapping frame_id -> frame info dict with all metadata
            - target_sessions: dict mapping target_id -> session_id for active sessions
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
        
        This should be called by the DOM watchdog after rebuilding the DOM.
        
        Args:
            selector_map: The new selector map from DOM serialization
        """
        self._cached_selector_map = selector_map

    @property
    def current_target_id(self) -> str | None:
        """Get current target ID from agent focus."""
        return self.agent_focus.target_id if self.agent_focus else None

    @property
    def current_session_id(self) -> str | None:
        """Get current session ID from agent focus."""
        return self.agent_focus.session_id if self.agent_focus else None

    # endregion - ========== Helper Methods ==========

