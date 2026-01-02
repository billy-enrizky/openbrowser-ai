"""Browser manager for spawning Chrome and managing CDP connections.

This module provides a compatibility wrapper around BrowserSession that maintains
the original API while using the event-driven architecture internally. It handles
browser process lifecycle and provides a simpler interface for basic browser operations.

Note:
    This module also includes a monkeypatch for BaseSubprocessTransport.__del__
    to handle closed event loops gracefully during garbage collection on macOS.

Classes:
    BrowserManager: High-level wrapper for browser lifecycle management.

Example:
    >>> async with BrowserManager(headless=True) as manager:
    ...     client, session_id = await manager.get_session()
    ...     await manager.take_screenshot('/tmp/screenshot.png')
"""

import asyncio
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from playwright.async_api import async_playwright

from cdp_use.client import CDPClient
from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
# This prevents "RuntimeError: Event loop is closed" errors during garbage collection on macOS
# The error occurs when BaseSubprocessTransport.__del__ tries to close pipes after the event loop is closed
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
    """Patched __del__ that handles closed event loops without throwing errors.

    This monkeypatch prevents RuntimeError: Event loop is closed errors during
    garbage collection when the subprocess transport tries to clean up pipes
    after the event loop has already closed. This is particularly common on
    macOS when Python is shutting down.

    The patch checks if the event loop is closed before calling the original
    __del__ method, and silently ignores the specific 'Event loop is closed'
    error if it occurs.
    """
    try:
        # Check if the event loop is closed before calling the original
        if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
            # Event loop is closed, skip cleanup that requires the loop
            return
        _original_del(self)
    except RuntimeError as e:
        if 'Event loop is closed' in str(e):
            # Silently ignore this specific error - it's harmless during GC
            pass
        else:
            raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


class BrowserManager:
    """Manages Chrome browser process and CDP connections.

    Compatibility wrapper around BrowserSession that provides a simpler API
    for browser lifecycle management. Internally uses the event-driven
    BrowserSession architecture.

    This class is suitable for simple use cases where full event-driven
    control is not needed. For advanced usage, use BrowserSession directly.

    Attributes:
        debug_port: Chrome remote debugging port.
        headless: Whether browser runs in headless mode.
        user_data_dir: Path to Chrome user data directory.
        cdp_client: CDPClient for sending CDP commands.
        session_id: Current CDP session ID.

    Example:
        >>> manager = BrowserManager(headless=True)
        >>> await manager.start()
        >>> client, session_id = await manager.get_session()
        >>> await manager.stop()

        # Or use as async context manager:
        >>> async with BrowserManager() as manager:
        ...     await manager.take_screenshot('screenshot.png')
    """

    def __init__(
        self,
        debug_port: int = 9222,
        headless: bool = True,
        user_data_dir: Optional[str] = None,
    ):
        """Initialize BrowserManager.

        Args:
            debug_port: Port for Chrome remote debugging (default: 9222).
            headless: Whether to run Chrome in headless mode (default: True).
            user_data_dir: Optional path to Chrome user data directory for
                profile persistence. If None, uses a temporary directory.

        Example:
            >>> manager = BrowserManager(
            ...     debug_port=9223,
            ...     headless=False,
            ...     user_data_dir='/path/to/profile'
            ... )
        """
        self.debug_port = debug_port
        self.headless = headless
        self.user_data_dir = user_data_dir or tempfile.mkdtemp(prefix="openbrowser_chrome_")
        
        # Create BrowserSession for event-driven management
        self._browser_session = BrowserSession(
            debug_port=debug_port,
            headless=headless,
            user_data_dir=user_data_dir or tempfile.mkdtemp(prefix="openbrowser_chrome_")
        )
        
        # Backward compatibility attributes
        self._process: Optional[asyncio.subprocess.Process] = None
        self._browser_executable: Optional[str] = None
        self._cdp_client: Optional[CDPClient] = None
        self._session_id: Optional[str] = None

    async def start(self) -> None:
        """Start the Chrome browser process and wait for CDP to be ready.

        Launches a Chrome browser process and establishes CDP connection.
        Updates backward compatibility attributes for direct access to
        CDP client and session.

        Raises:
            RuntimeError: If browser fails to start or CDP connection fails.

        Note:
            Logs a warning if browser is already started.
        """
        if self._browser_session._cdp_client_root is not None:
            logger.warning("Browser already started")
            return

        # Use BrowserSession's event-driven start
        await self._browser_session.start()
        
        # Update backward compatibility attributes
        self._cdp_url = self._browser_session._cdp_url
        self._cdp_client = self._browser_session._cdp_client_root
        if self._browser_session.agent_focus:
            self._session_id = self._browser_session.agent_focus.session_id
        self._process = getattr(self._browser_session, '_process', None)


    @property
    def cdp_client(self) -> CDPClient:
        """Get the persistent CDP client.

        Provides access to the underlying CDPClient for sending raw
        CDP commands.

        Returns:
            The persistent CDPClient instance.

        Raises:
            RuntimeError: If browser is not started or connection not established.
        """
        return self._browser_session.cdp_client

    @property
    def session_id(self) -> str:
        """Get the persistent session ID.

        Provides the CDP session ID for the currently attached target.

        Returns:
            The session ID string for CDP commands.

        Raises:
            RuntimeError: If browser is not started or session not established.
        """
        if not self._browser_session.agent_focus:
            raise RuntimeError(
                "Session ID not initialized. Call start() first to establish connection."
            )
        return self._browser_session.agent_focus.session_id

    async def get_session(self) -> tuple[CDPClient, str]:
        """Get the persistent CDP client and session ID.

        Returns the cached persistent connection created in start().
        This method returns the same connection for the lifetime of the
        browser session, eliminating connection overhead.

        Returns:
            Tuple of (CDPClient, session_id) from the persistent connection.

        Raises:
            RuntimeError: If browser is not started or connection not established.

        Example:
            >>> client, session_id = await manager.get_session()
            >>> await client.send.DOM.enable(session_id=session_id)
        """
        if not self._browser_session.agent_focus:
            raise RuntimeError(
                "Persistent CDP connection not established. Call start() first."
            )

        return (
            self._browser_session.agent_focus.cdp_client,
            self._browser_session.agent_focus.session_id,
        )

    async def take_screenshot(
        self,
        path: str,
        session_id: Optional[str] = None,
        client: Optional[CDPClient] = None,
    ) -> None:
        """Take a screenshot of the current page.

        Captures the visible viewport and saves it as a PNG file.

        Args:
            path: File path to save the screenshot (creates parent directories).
            session_id: Optional session ID. Must be provided with client if
                reusing an existing session. If not provided, uses persistent
                connection.
            client: Optional CDPClient. If provided with session_id, uses that
                connection. If not provided, uses persistent connection.

        Raises:
            RuntimeError: If browser is not started, screenshot fails, or
                session_id is provided without client.

        Example:
            >>> await manager.take_screenshot('/tmp/screenshot.png')
        """
        if not self._browser_session.agent_focus:
            raise RuntimeError("Browser not started. Call start() first.")

        # Use persistent connection if no client provided
        if client is None:
            client = self._browser_session.agent_focus.cdp_client
            session_id = self._browser_session.agent_focus.session_id
        else:
            if session_id is None:
                raise RuntimeError(
                    "client provided without session_id. Provide both or neither."
                )

            logger.info(f"Taking screenshot to {path}")

            # Enable Page domain if not already enabled
            try:
                await client.send.Page.enable(session_id=session_id)
            except Exception:
                pass

            # Capture screenshot using raw CDP command
            result = await client.send.Page.captureScreenshot(
                params={"format": "png"}, session_id=session_id
            )

            # Decode base64 image data
            image_data = base64.b64decode(result["data"])

            # Write to file
            screenshot_path = Path(path)
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(image_data)

            logger.info(f"Screenshot saved to {path}")

    async def stop(self) -> None:
        """Stop the Chrome browser process.

        Cleanly shuts down the browser session and clears all connection
        state. The browser process will be terminated.

        Example:
            >>> await manager.stop()
        """
        # Use BrowserSession's event-driven stop
        await self._browser_session.stop()
        
        # Clear backward compatibility attributes
        self._cdp_url = None
        self._cdp_client = None
        self._session_id = None
        self._process = None

    async def __aenter__(self):
        """Async context manager entry.

        Starts the browser and returns the manager instance.

        Returns:
            Self for use in async with statement.

        Example:
            >>> async with BrowserManager() as manager:
            ...     await manager.take_screenshot('screenshot.png')
        """
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.

        Stops the browser when exiting the context. Any exceptions from
        the context body are not suppressed.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        await self.stop()
    
    @property
    def _cdp_url(self) -> Optional[str]:
        """Backward compatibility: expose _cdp_url.

        Returns:
            CDP WebSocket URL from the underlying browser session.
        """
        return self._browser_session._cdp_url

    @_cdp_url.setter
    def _cdp_url(self, value: Optional[str]) -> None:
        """Set CDP URL (updates browser session).

        Args:
            value: New CDP WebSocket URL.
        """
        self._browser_session._cdp_url = value

