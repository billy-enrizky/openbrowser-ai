"""Security watchdog for enforcing URL access policies.

This module provides the SecurityWatchdog which enforces domain allowlists
and blocklists, preventing navigation to unauthorized URLs.

Classes:
    SecurityWatchdog: Monitors and enforces URL access policies.
"""

import logging
from typing import Any, ClassVar
from urllib.parse import urlparse

from bubus import BaseEvent

from src.openbrowser.browser.events import BrowserErrorEvent, NavigateToUrlEvent, NavigationCompleteEvent, TabCreatedEvent
from src.openbrowser.browser.watchdogs.base import BaseWatchdog

logger = logging.getLogger(__name__)


class SecurityWatchdog(BaseWatchdog):
    """Monitors and enforces security policies for URL access.

    Checks navigation URLs against configured allowed_domains in
    BrowserProfile. Blocks disallowed navigations and handles
    redirects to blocked domains.

    Listens to:
        NavigateToUrlEvent: Checks URL before navigation.
        NavigationCompleteEvent: Catches redirects to blocked domains.
        TabCreatedEvent: Checks new tab URLs.

    Emits:
        BrowserErrorEvent: When navigation is blocked.

    Configuration (in BrowserProfile):
        allowed_domains: List of allowed domain patterns.

    Example:
        >>> profile = BrowserProfile(
        ...     allowed_domains=['example.com', '*.trusted.org']
        ... )
    """

    # Events this watchdog listens to
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
        NavigateToUrlEvent,
        NavigationCompleteEvent,
        TabCreatedEvent,
    ]

    # Events this watchdog emits
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = [
        BrowserErrorEvent,
    ]

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to navigation and tab events for security checks.
        """
        self.event_bus.on(NavigateToUrlEvent, self.on_NavigateToUrlEvent)
        self.event_bus.on(NavigationCompleteEvent, self.on_NavigationCompleteEvent)
        self.event_bus.on(TabCreatedEvent, self.on_TabCreatedEvent)

    async def on_NavigateToUrlEvent(self, event: NavigateToUrlEvent) -> None:
        """Check if navigation URL is allowed before navigation starts.

        Blocks navigation to disallowed URLs by raising ValueError.

        Args:
            event: NavigateToUrlEvent with target URL.

        Raises:
            ValueError: If URL is not in allowed_domains.
        """
        # Security check BEFORE navigation
        if not self._is_url_allowed(event.url):
            self.logger.warning(f'[SecurityWatchdog] Blocking navigation to disallowed URL: {event.url}')
            self.event_bus.dispatch(
                BrowserErrorEvent(
                    error_type='NavigationBlocked',
                    message=f'Navigation blocked to disallowed URL: {event.url}',
                    details={'url': event.url, 'reason': 'not_in_allowed_domains'},
                )
            )
            # Stop event propagation by raising exception
            raise ValueError(f'Navigation to {event.url} blocked by security policy')

    async def on_NavigationCompleteEvent(self, event: NavigationCompleteEvent) -> None:
        """Check if navigated URL is allowed (catches redirects to blocked domains).

        If navigation ended on a disallowed URL (via redirect), navigates
        to about:blank to keep session alive.

        Args:
            event: NavigationCompleteEvent with final URL.
        """
        # Check if the navigated URL is allowed (in case of redirects)
        if not self._is_url_allowed(event.url):
            self.logger.warning(f'[SecurityWatchdog] Navigation to non-allowed URL detected: {event.url}')

            # Dispatch browser error
            self.event_bus.dispatch(
                BrowserErrorEvent(
                    error_type='NavigationBlocked',
                    message=f'Navigation blocked to non-allowed URL: {event.url} - redirecting to about:blank',
                    details={'url': event.url, 'target_id': event.target_id},
                )
            )
            # Navigate to about:blank to keep session alive
            try:
                session = await self.browser_session.get_or_create_cdp_session(target_id=event.target_id)
                await session.cdp_client.send.Page.navigate(
                    params={'url': 'about:blank'}, session_id=session.session_id
                )
                self.logger.info(f'[SecurityWatchdog] Navigated to about:blank after blocked URL: {event.url}')
            except Exception as e:
                self.logger.error(f'[SecurityWatchdog] Failed to navigate to about:blank: {type(e).__name__} {e}')

    async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
        """Check if new tab URL is allowed.

        Closes tabs created with disallowed URLs (e.g., popup ads).

        Args:
            event: TabCreatedEvent with URL and target_id.
        """
        if not self._is_url_allowed(event.url):
            self.logger.warning(f'[SecurityWatchdog] New tab created with disallowed URL: {event.url}')

            # Dispatch error and try to close the tab
            self.event_bus.dispatch(
                BrowserErrorEvent(
                    error_type='TabCreationBlocked',
                    message=f'Tab created with non-allowed URL: {event.url}',
                    details={'url': event.url, 'target_id': event.target_id},
                )
            )

            # Try to close the offending tab
            try:
                await self.browser_session._cdp_close_page(event.target_id)
                self.logger.info(f'[SecurityWatchdog] Closed new tab with non-allowed URL: {event.url}')
            except Exception as e:
                self.logger.error(f'[SecurityWatchdog] Failed to close new tab with non-allowed URL: {type(e).__name__} {e}')

    def _is_url_allowed(self, url: str) -> bool:
        """Check if a URL is allowed based on browser profile settings.

        Args:
            url: URL to check.

        Returns:
            True if URL is allowed, False otherwise.
            
        Returns:
            True if URL is allowed, False otherwise
        """
        profile = self.browser_session.browser_profile

        # If no restrictions, allow all
        if not profile.allowed_domains and not profile.prohibited_domains:
            return True

        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path

            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]

            # Check prohibited domains first (more restrictive)
            if profile.prohibited_domains:
                if isinstance(profile.prohibited_domains, set):
                    if domain in profile.prohibited_domains:
                        return False
                else:
                    for prohibited in profile.prohibited_domains:
                        if self._matches_pattern(domain, prohibited):
                            return False

            # Check allowed domains
            if profile.allowed_domains:
                if isinstance(profile.allowed_domains, set):
                    if domain in profile.allowed_domains:
                        return True
                else:
                    for allowed in profile.allowed_domains:
                        if self._matches_pattern(domain, allowed):
                            return True
                # If allowed_domains is set but domain doesn't match, deny
                return False

            # If only prohibited_domains is set and domain doesn't match, allow
            return True

        except Exception as e:
            self.logger.warning(f'[SecurityWatchdog] Error checking URL {url}: {e}')
            # On error, be permissive
            return True

    def _matches_pattern(self, domain: str, pattern: str) -> bool:
        """Check if domain matches a pattern (supports wildcards).
        
        Args:
            domain: Domain to check
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            True if domain matches pattern
        """
        if '*' in pattern:
            # Simple wildcard matching
            pattern_parts = pattern.split('*')
            if len(pattern_parts) == 2:
                prefix, suffix = pattern_parts
                return domain.startswith(prefix) and domain.endswith(suffix)
        return domain == pattern

