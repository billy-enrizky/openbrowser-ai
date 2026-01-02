"""Local browser watchdog for managing browser subprocess lifecycle.

This module provides the LocalBrowserWatchdog which launches and manages
local Chrome/Chromium browser processes with CDP debugging enabled.

Classes:
    LocalBrowserWatchdog: Manages local browser subprocess lifecycle.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
from bubus import BaseEvent
from playwright.async_api import async_playwright

from src.openbrowser.browser.events import (
    BrowserKillEvent,
    BrowserLaunchEvent,
    BrowserLaunchResult,
    BrowserStopEvent,
)
from src.openbrowser.browser.watchdogs.base import BaseWatchdog

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LocalBrowserWatchdog(BaseWatchdog):
    """Manages local browser subprocess lifecycle.

    Handles launching Chrome/Chromium with CDP debugging enabled,
    process termination, and cleanup. Uses Playwright for browser
    executable discovery.

    Listens to:
        BrowserLaunchEvent: Launches a new browser process.
        BrowserKillEvent: Terminates the browser process.
        BrowserStopEvent: Graceful shutdown handling.

    Example:
        >>> watchdog = LocalBrowserWatchdog(
        ...     event_bus=bus,
        ...     browser_session=session
        ... )
        >>> result = await bus.dispatch(BrowserLaunchEvent())
        >>> print(result.cdp_url)  # ws://localhost:9222/devtools/...
    """

    # Events this watchdog listens to
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
        BrowserLaunchEvent,
        BrowserKillEvent,
        BrowserStopEvent,
    ]

    # Events this watchdog emits
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to browser lifecycle events for process management.
        """
        self.event_bus.on(BrowserLaunchEvent, self.on_BrowserLaunchEvent)
        self.event_bus.on(BrowserKillEvent, self.on_BrowserKillEvent)
        self.event_bus.on(BrowserStopEvent, self.on_BrowserStopEvent)

    async def on_BrowserLaunchEvent(self, event: BrowserLaunchEvent) -> BrowserLaunchResult:
        """Launch a local browser process.

        Creates a new Chrome/Chromium subprocess with CDP debugging
        enabled. Returns the CDP WebSocket URL for connection.

        Args:
            event: BrowserLaunchEvent triggering the launch.

        Returns:
            BrowserLaunchResult with cdp_url for CDP connection.

        Raises:
            Exception: If browser launch fails.
        """
        try:
            self.logger.debug('[LocalBrowserWatchdog] Received BrowserLaunchEvent, launching local browser...')

            process, cdp_url = await self._launch_browser()
            self.browser_session._process = process

            return BrowserLaunchResult(cdp_url=cdp_url)
        except Exception as e:
            self.logger.error(f'[LocalBrowserWatchdog] Exception in on_BrowserLaunchEvent: {e}', exc_info=True)
            raise

    async def on_BrowserKillEvent(self, event: BrowserKillEvent) -> None:
        """Kill the browser process.

        Terminates the browser subprocess gracefully, falling back
        to forceful kill if needed. Cleans up pipe handles.

        Args:
            event: BrowserKillEvent triggering the kill.
        """
        process = getattr(self.browser_session, '_process', None)
        if process is None:
            self.logger.warning('[LocalBrowserWatchdog] No browser process to kill')
            return

        self.logger.info(f'[LocalBrowserWatchdog] Killing browser process (PID {process.pid})')

        try:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning('[LocalBrowserWatchdog] Process did not terminate gracefully, killing')
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            self.logger.warning(f'[LocalBrowserWatchdog] Error during process kill: {e}')

        # Clean up pipes
        try:
            if process.stdout:
                try:
                    process.stdout.close()
                except (OSError, ValueError, RuntimeError):
                    pass
            if process.stderr:
                try:
                    process.stderr.close()
                except (OSError, ValueError, RuntimeError):
                    pass
            if process.stdin:
                try:
                    process.stdin.close()
                except (OSError, ValueError, RuntimeError):
                    pass
        except Exception as e:
            self.logger.debug(f'[LocalBrowserWatchdog] Error during pipe cleanup: {e}')

        self.browser_session._process = None

    async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
        """Handle browser stop request."""
        if event.force:
            # Kill the process
            kill_event = self.event_bus.dispatch(BrowserKillEvent())
            await kill_event
        # Otherwise, just keep the browser alive (keep_alive behavior)

    async def _launch_browser(self) -> tuple[asyncio.subprocess.Process, str]:
        """Launch Chrome browser and return process and CDP URL."""
        debug_port = getattr(self.browser_session, 'debug_port', 9222)
        headless = getattr(self.browser_session, 'headless', True)
        user_data_dir = getattr(self.browser_session, 'user_data_dir', None)

        if user_data_dir is None:
            user_data_dir = tempfile.mkdtemp(prefix="openbrowser_chrome_")

        self.logger.info(f'[LocalBrowserWatchdog] Starting Chrome browser on port {debug_port}')

        # Get Chrome executable from playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                browser_executable = getattr(browser, "_executable_path", None)
                if not browser_executable:
                    from playwright._impl._driver import get_driver
                    driver = get_driver()
                    browser_paths = driver._get_browser_paths()
                    browser_executable = browser_paths.get("chromium")
            except Exception:
                import platform
                system = platform.system()
                if system == "Darwin":
                    common_paths = [
                        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                        "/Applications/Chromium.app/Contents/MacOS/Chromium",
                    ]
                elif system == "Linux":
                    common_paths = [
                        "/usr/bin/google-chrome",
                        "/usr/bin/chromium-browser",
                        "/usr/bin/chromium",
                    ]
                else:  # Windows
                    common_paths = [
                        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                    ]

                browser_executable = None
                for path in common_paths:
                    if Path(path).exists():
                        browser_executable = path
                        break

            await browser.close()

        if not browser_executable:
            raise RuntimeError(
                "Failed to get Chrome executable. Please install Chrome/Chromium or playwright browsers."
            )

        if "/" in browser_executable or "\\" in browser_executable:
            if not Path(browser_executable).exists():
                raise RuntimeError(f"Chrome executable not found at: {browser_executable}")

        self.logger.info(f'[LocalBrowserWatchdog] Using Chrome executable: {browser_executable}')

        # Build Chrome launch arguments
        launch_args = [
            browser_executable,
            f"--remote-debugging-port={debug_port}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--start-maximized",
            "--window-size=1920,1080",
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]

        if headless:
            launch_args.extend(["--headless=new"])

        # Launch Chrome process
        process = await asyncio.create_subprocess_exec(
            *launch_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self.logger.info(f'[LocalBrowserWatchdog] Chrome process started with PID {process.pid}')

        # Wait for CDP to be ready and get websocket URL
        cdp_url = await self._wait_for_cdp_ready(debug_port)

        self.logger.info(f'[LocalBrowserWatchdog] CDP connection ready at {cdp_url}')

        return process, cdp_url

    async def _wait_for_cdp_ready(self, debug_port: int, timeout: int = 20) -> str:
        """Wait for Chrome CDP to be ready and return websocket URL."""
        version_url = f"http://localhost:{debug_port}/json/version"

        for attempt in range(timeout):
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    response = await client.get(version_url)
                    if response.status_code == 200:
                        data = response.json()
                        ws_url = data.get("webSocketDebuggerUrl")
                        if ws_url:
                            self.logger.info(f'[LocalBrowserWatchdog] CDP connection established: {ws_url}')
                            return ws_url
            except Exception as e:
                if attempt < timeout - 1:
                    await asyncio.sleep(1)
                else:
                    raise RuntimeError(
                        f"Failed to connect to CDP after {timeout} seconds: {e}"
                    )

        raise RuntimeError(f"CDP not ready after {timeout} seconds")

