"""Browser manager for spawning Chrome and managing CDP connections.

This class is a compatibility wrapper around BrowserSession that maintains
the original API while using the event-driven architecture internally.
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
    
    This prevents RuntimeError: Event loop is closed errors during garbage collection
    when the subprocess transport tries to clean up after the event loop has closed.
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
    
    This class is a compatibility wrapper around BrowserSession that maintains
    the original API while using the event-driven architecture internally.
    """

    def __init__(
        self,
        debug_port: int = 9222,
        headless: bool = True,
        user_data_dir: Optional[str] = None,
    ):
        """Initialize BrowserManager.
        
        Args:
            debug_port: Port for Chrome remote debugging
            headless: Whether to run Chrome in headless mode
            user_data_dir: Optional user data directory for Chrome profile
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
        """Start the Chrome browser process and wait for CDP to be ready."""
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
        
        Returns:
            The persistent CDPClient instance
            
        Raises:
            RuntimeError: If browser is not started or connection not established
        """
        return self._browser_session.cdp_client

    @property
    def session_id(self) -> str:
        """Get the persistent session ID.
        
        Returns:
            The session ID for the attached target
            
        Raises:
            RuntimeError: If browser is not started or session not established
        """
        if not self._browser_session.agent_focus:
            raise RuntimeError(
                "Session ID not initialized. Call start() first to establish connection."
            )
        return self._browser_session.agent_focus.session_id

    async def get_session(self) -> tuple[CDPClient, str]:
        """Get the persistent CDP client and session ID.
        
        Returns the cached persistent connection created in start().
        This method now returns the same connection for the lifetime
        of the browser session, eliminating connection overhead.
        
        Returns:
            Tuple of (CDPClient, session_id) from the persistent connection
            
        Raises:
            RuntimeError: If browser is not started or connection not established
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
        
        Args:
            path: File path to save the screenshot
            session_id: Optional session ID. Must be provided with client if
                reusing an existing session.
            client: Optional CDPClient. If provided with session_id, will use
                the existing connection. If not provided, uses persistent connection.
                
        Raises:
            RuntimeError: If browser is not started, screenshot fails, or
                session_id is provided without client.
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
        """Stop the Chrome browser process."""
        # Use BrowserSession's event-driven stop
        await self._browser_session.stop()
        
        # Clear backward compatibility attributes
        self._cdp_url = None
        self._cdp_client = None
        self._session_id = None
        self._process = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    @property
    def _cdp_url(self) -> Optional[str]:
        """Backward compatibility: expose _cdp_url."""
        return self._browser_session._cdp_url
    
    @_cdp_url.setter
    def _cdp_url(self, value: Optional[str]) -> None:
        """Set CDP URL (updates browser session)."""
        self._browser_session._cdp_url = value

