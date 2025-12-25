"""Browser manager for spawning Chrome and managing CDP connections."""

import asyncio
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from playwright.async_api import async_playwright

from cdp_use.client import CDPClient

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages Chrome browser process and CDP connections.
    
    This class spawns a Chrome process using playwright's binary but connects
    via CDP websocket using cdp-use for raw control.
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
        self._process: Optional[asyncio.subprocess.Process] = None
        self._cdp_url: Optional[str] = None
        self._browser_executable: Optional[str] = None

    async def start(self) -> None:
        """Start the Chrome browser process and wait for CDP to be ready."""
        if self._process is not None:
            logger.warning("Browser already started")
            return

        logger.info(f"Starting Chrome browser on port {self.debug_port}")

        # Get Chrome executable from playwright
        # Use playwright's browser installation path
        async with async_playwright() as p:
            # Launch browser temporarily to get executable path
            browser = await p.chromium.launch(headless=True)
            # Access the executable path from the browser's context
            # Playwright stores this internally
            try:
                # Try to get executable path from browser object
                self._browser_executable = getattr(browser, "_executable_path", None)
                if not self._browser_executable:
                    # Fallback: use playwright's browser path resolution
                    from playwright._impl._driver import get_driver
                    driver = get_driver()
                    browser_paths = driver._get_browser_paths()
                    self._browser_executable = browser_paths.get("chromium")
            except Exception:
                # If that fails, try common system paths
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

                for path in common_paths:
                    if Path(path).exists():
                        self._browser_executable = path
                        break

            await browser.close()

        if not self._browser_executable:
            raise RuntimeError(
                "Failed to get Chrome executable. Please install Chrome/Chromium or playwright browsers."
            )
        
        # Check if it's a valid path (skip check for commands in PATH)
        if "/" in self._browser_executable or "\\" in self._browser_executable:
            if not Path(self._browser_executable).exists():
                raise RuntimeError(
                    f"Chrome executable not found at: {self._browser_executable}"
                )

        logger.info(f"Using Chrome executable: {self._browser_executable}")

        # Build Chrome launch arguments
        launch_args = [
            self._browser_executable,
            f"--remote-debugging-port={self.debug_port}",
            f"--user-data-dir={self.user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            # --- ANTI-BOT EVASION FLAGS ---
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",
            "--window-size=1920,1080",
            # Use a real Mac User-Agent to fool Google
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]

        if self.headless:
            launch_args.extend(["--headless=new"])  # Use new headless mode (better detection evasion)

        # Launch Chrome process
        self._process = await asyncio.create_subprocess_exec(
            *launch_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        logger.info(f"Chrome process started with PID {self._process.pid}")

        # Wait for CDP to be ready and get websocket URL
        self._cdp_url = await self._wait_for_cdp_ready()

        logger.info(f"CDP connection ready at {self._cdp_url}")

    async def _wait_for_cdp_ready(self, timeout: int = 20) -> str:
        """Wait for Chrome CDP to be ready and return websocket URL.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            WebSocket URL for CDP connection
            
        Raises:
            RuntimeError: If CDP is not ready within timeout
        """
        version_url = f"http://localhost:{self.debug_port}/json/version"

        for attempt in range(timeout):
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    response = await client.get(version_url)
                    if response.status_code == 200:
                        data = response.json()
                        ws_url = data.get("webSocketDebuggerUrl")
                        if ws_url:
                            logger.info(f"CDP connection established: {ws_url}")
                            return ws_url
            except Exception as e:
                if attempt < timeout - 1:
                    await asyncio.sleep(1)
                else:
                    raise RuntimeError(
                        f"Failed to connect to CDP after {timeout} seconds: {e}"
                    )

        raise RuntimeError(f"CDP not ready after {timeout} seconds")

    async def get_session(self) -> tuple[CDPClient, str]:
        """Get a CDP client connected to the browser with an attached session.
        
        Returns:
            Tuple of (CDPClient, session_id) where session_id is from attaching
            to the first page target.
            
        Raises:
            RuntimeError: If browser is not started or no page targets found
        """
        if self._cdp_url is None:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.info("Creating CDP client connection")

        # Create and start CDP client
        client = CDPClient(self._cdp_url)
        await client.start()

        logger.info("CDP client connected")

        # Get all targets
        logger.info("Fetching browser targets")
        targets_result = await client.send.Target.getTargets()
        page_targets = [
            t for t in targets_result["targetInfos"] if t["type"] == "page"
        ]

        if not page_targets:
            await client.stop()
            raise RuntimeError("No page targets found in browser")

        target_id = page_targets[0]["targetId"]
        logger.info(f"Attaching to target: {target_id}")

        # Attach to the first page target
        attach_result = await client.send.Target.attachToTarget(
            params={"targetId": target_id, "flatten": True}
        )
        session_id = attach_result["sessionId"]

        if not session_id:
            await client.stop()
            raise RuntimeError("Failed to attach to target: no session ID returned")

        logger.info(f"Attached to target with session_id: {session_id}")

        return client, session_id

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
                the existing connection. If not provided, a new session will be
                created and closed after the screenshot.
                
        Raises:
            RuntimeError: If browser is not started, screenshot fails, or
                session_id is provided without client.
        """
        if self._cdp_url is None:
            raise RuntimeError("Browser not started. Call start() first.")

        # Validate parameters
        if session_id is not None and client is None:
            raise RuntimeError(
                "session_id provided without client. CDP sessions are tied to "
                "specific client connections. Provide both client and session_id, "
                "or neither to create a new session."
            )

        # If no client/session provided, create a temporary session
        if client is None:
            client, session_id = await self.get_session()
            should_close = True
        else:
            # Using existing client - don't close it
            should_close = False
            if session_id is None:
                raise RuntimeError(
                    "client provided without session_id. Provide both or neither."
                )

        try:
            logger.info(f"Taking screenshot to {path}")

            # Enable Page domain if not already enabled
            # (Some CDP commands require domains to be enabled)
            try:
                await client.send.Page.enable(session_id=session_id)
            except Exception:
                # Domain might already be enabled, ignore
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

        finally:
            if should_close:
                await client.stop()

    async def stop(self) -> None:
        """Stop the Chrome browser process."""
        if self._process is None:
            logger.warning("Browser process not running")
            return

        logger.info(f"Stopping Chrome process (PID {self._process.pid})")

        try:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Process did not terminate gracefully, killing")
                self._process.kill()
                await self._process.wait()
        except ProcessLookupError:
            # Process already terminated
            pass

        self._process = None
        self._cdp_url = None
        logger.info("Chrome process stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

