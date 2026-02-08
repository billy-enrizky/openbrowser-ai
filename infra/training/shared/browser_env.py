"""Browser environment for online GRPO training using the openbrowser package.

Wraps openbrowser's BrowserSession and Tools to provide a training-friendly
interface for executing parsed action sequences in a real browser.
"""

import asyncio
import logging
import os
import subprocess

from openbrowser import BrowserSession, Tools

from infra.training.shared.online_reward import BrowserOutcome

logger = logging.getLogger(__name__)


def _find_chromium_binary() -> str | None:
    """Find the Playwright-installed Chromium binary path.

    Uses Playwright's own registry to locate the browser binary (most reliable),
    then falls back to common system paths.
    """
    # Method 1: Ask Playwright where it installed Chromium (most reliable)
    try:
        result = subprocess.run(
            ["python", "-c",
             "from playwright._impl._driver import compute_driver_executable, get_driver_env; "
             "import subprocess, json; "
             "proc = subprocess.run("
             "[str(compute_driver_executable()), 'print-api-json'], "
             "capture_output=True, text=True, env=get_driver_env()); "
             "print('OK')"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        pass

    # Method 2: Search for the Playwright chromium executable via registry
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        executable = pw.chromium.executable_path
        pw.stop()
        if executable and os.path.isfile(executable):
            return executable
    except Exception as e:
        logger.debug(f"Playwright API lookup failed: {e}")

    # Method 3: Check common system paths
    for system_path in [
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]:
        if os.path.isfile(system_path) and os.access(system_path, os.X_OK):
            return system_path

    # Method 4: Broad search of Playwright cache with multiple patterns
    pw_path = os.environ.get(
        "PLAYWRIGHT_BROWSERS_PATH",
        os.path.expanduser("~/.cache/ms-playwright"),
    )
    if os.path.isdir(pw_path):
        import glob
        # Try multiple patterns (Playwright may use different directory layouts)
        for pattern in [
            "chromium-*/chrome-linux/chrome",
            "chromium_headless_shell-*/chrome-linux/headless_shell",
            "chromium-*/chrome-linux/headless_shell",
            "chrome-*/chrome-linux/chrome",
        ]:
            matches = glob.glob(os.path.join(pw_path, pattern))
            if matches:
                return sorted(matches)[-1]

        # Last resort: find any executable named 'chrome' or 'chromium'
        for root, dirs, files in os.walk(pw_path):
            for name in files:
                if name in ("chrome", "chromium", "headless_shell"):
                    path = os.path.join(root, name)
                    if os.access(path, os.X_OK):
                        return path

    return None

# Success indicators on FormFactory pages after form submission
SUCCESS_INDICATORS = [
    "successfully submitted",
    "thank you",
    "submission received",
    "form submitted",
    "success",
]


class BrowserEnvironment:
    """Training browser environment backed by the openbrowser package.

    Uses BrowserSession for browser lifecycle and Tools for action execution.
    Designed for the online Flow GRPO training loop where parsed action
    sequences are executed against FormFactory forms.
    """

    def __init__(self, browser_session: BrowserSession, tools: Tools):
        self.browser_session = browser_session
        self.tools = tools

    @classmethod
    async def create(cls, headless: bool = True) -> "BrowserEnvironment":
        """Create and start a browser environment."""
        executable = _find_chromium_binary()
        if executable:
            logger.info(f"Found browser binary: {executable}")
            browser_session = BrowserSession(
                headless=headless, executable_path=executable
            )
        else:
            logger.warning(
                "No pre-installed browser binary found, "
                "falling back to openbrowser auto-detection"
            )
            browser_session = BrowserSession(headless=headless)
        await browser_session.start()
        tools = Tools()
        logger.info("Browser environment created")
        return cls(browser_session, tools)

    async def get_element_map(self) -> dict[str, int]:
        """Build field_name -> element_index mapping from current page DOM.

        Inspects the selector map and extracts element names from attributes
        (name, placeholder, aria-label, tag type for submit buttons).
        """
        selector_map = await self.browser_session.get_selector_map()
        field_map: dict[str, int] = {}

        for idx, element in selector_map.items():
            attrs = element.attributes or {}

            # Map by name, placeholder, aria-label
            for attr_key in ["name", "placeholder", "aria-label"]:
                value = attrs.get(attr_key, "")
                if value:
                    field_map[value.lower()] = idx

            # Map submit buttons
            tag = getattr(element, "tag_name", "") or ""
            input_type = attrs.get("type", "")
            if tag.lower() in ("button", "input") and input_type.lower() == "submit":
                field_map["submit"] = idx
                btn_text = getattr(element, "node_value", "") or ""
                if btn_text:
                    field_map[btn_text.lower()] = idx

        return field_map

    async def execute_actions(
        self,
        actions: list[dict],
        timeout_per_action: float = 5.0,
    ) -> BrowserOutcome:
        """Execute a sequence of parsed action dicts in the browser.

        Each action dict has format:
            {"action": "navigate", "params": {"url": "...", "new_tab": False}}
            {"action": "input_text", "params": {"index": 3, "text": "John", "clear": True}}
            {"action": "click_element", "params": {"index": 5}}
            {"action": "select_dropdown_option", "params": {"index": 7, "text": "CA"}}

        Args:
            actions: List of action dicts from action_parser.
            timeout_per_action: Max seconds per action before timeout.

        Returns:
            BrowserOutcome with execution results.
        """
        executed = 0
        total = len(actions)

        for action_dict in actions:
            action_name = action_dict.get("action", "")
            params = action_dict.get("params", {})

            try:
                # Use Tools' dynamic wrapper methods (via __getattr__)
                action_fn = getattr(self.tools, action_name, None)
                if action_fn is None:
                    logger.warning(f"Unknown action: {action_name}")
                    continue

                result = await asyncio.wait_for(
                    action_fn(browser_session=self.browser_session, **params),
                    timeout=timeout_per_action,
                )

                if result and result.error:
                    logger.debug(f"Action {action_name} error: {result.error}")
                    return BrowserOutcome(
                        success_page_detected=False,
                        submitted_values={},
                        error=result.error,
                        actions_executed=executed,
                        total_actions=total,
                    )

                executed += 1

            except asyncio.TimeoutError:
                logger.warning(f"Action {action_name} timed out after {timeout_per_action}s")
                return BrowserOutcome(
                    success_page_detected=False,
                    submitted_values={},
                    error=f"Timeout on action {action_name}",
                    actions_executed=executed,
                    total_actions=total,
                )
            except Exception as e:
                logger.warning(f"Action {action_name} failed: {e}")
                return BrowserOutcome(
                    success_page_detected=False,
                    submitted_values={},
                    error=str(e),
                    actions_executed=executed,
                    total_actions=total,
                )

        # Check final page state for success indicators
        success = await self._check_success_page()

        return BrowserOutcome(
            success_page_detected=success,
            submitted_values={},
            error=None,
            actions_executed=executed,
            total_actions=total,
        )

    async def _check_success_page(self) -> bool:
        """Check if current page indicates successful form submission."""
        try:
            state = await self.browser_session.get_browser_state_summary(
                include_screenshot=False
            )
            # Check URL and title for success indicators
            url = (state.url or "").lower()
            title = (state.title or "").lower()
            dom_text = ""
            if state.dom_state:
                dom_text = state.dom_state.llm_representation().lower()

            for indicator in SUCCESS_INDICATORS:
                if indicator in url or indicator in title or indicator in dom_text:
                    return True

            return False
        except Exception as e:
            logger.warning(f"Failed to check success page: {e}")
            return False

    async def reset(self) -> None:
        """Reset browser state between rollouts."""
        try:
            await self.tools.go_to_url(
                url="about:blank", browser_session=self.browser_session
            )
        except Exception:
            # Fallback: try navigate
            try:
                result = await self.tools.navigate(
                    url="about:blank",
                    new_tab=False,
                    browser_session=self.browser_session,
                )
            except Exception as e:
                logger.warning(f"Browser reset failed: {e}")

    async def close(self) -> None:
        """Shutdown browser."""
        try:
            await self.browser_session.kill()
            logger.info("Browser environment closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
