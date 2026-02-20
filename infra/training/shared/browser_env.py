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
# Map action parser names to openbrowser Tools method names
ACTION_NAME_MAP = {
    "input_text": "input",
    "click_element": "click",
    "select_dropdown_option": "select_dropdown",
    "navigate": "navigate",
}

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
        # Allow localhost/127.0.0.1 for FormFactory server
        allowed = ["localhost", "127.0.0.1", "about:blank"]
        if executable:
            logger.info(f"Found browser binary: {executable}")
            browser_session = BrowserSession(
                headless=headless,
                executable_path=executable,
                allowed_domains=allowed,
            )
        else:
            logger.warning(
                "No pre-installed browser binary found, "
                "falling back to openbrowser auto-detection"
            )
            browser_session = BrowserSession(
                headless=headless, allowed_domains=allowed
            )
        await browser_session.start()
        tools = Tools()
        logger.info("Browser environment created")
        return cls(browser_session, tools)

    async def get_element_map(self) -> dict[str, int]:
        """Build field_name -> element_index mapping from current page DOM.

        Inspects the selector map and extracts element names from attributes
        (name, placeholder, aria-label, tag type for submit buttons).
        """
        # Trigger DOM parsing first -- get_selector_map() returns empty
        # unless the DOMWatchdog has built the DOM tree via get_browser_state_summary()
        await self.browser_session.get_browser_state_summary(include_screenshot=False)

        selector_map = await self.browser_session.get_selector_map()
        field_map: dict[str, int] = {}

        logger.info(f"Selector map has {len(selector_map)} elements")

        for idx, element in selector_map.items():
            # Try multiple ways to access attributes (openbrowser API varies)
            attrs = {}
            if hasattr(element, "attributes") and element.attributes:
                if isinstance(element.attributes, dict):
                    attrs = element.attributes
                else:
                    # Some openbrowser versions return attributes as list of tuples
                    try:
                        attrs = dict(element.attributes)
                    except (TypeError, ValueError):
                        attrs = {}

            tag = getattr(element, "tag_name", "") or ""
            tag_lower = tag.lower()

            # Log first few elements for debugging
            if idx < 5 or tag_lower in ("input", "select", "textarea", "button"):
                logger.debug(
                    f"  Element {idx}: tag={tag}, attrs={attrs}, "
                    f"text={getattr(element, 'text_content', '')!r}"
                )

            # Map by name, placeholder, aria-label, id, for
            for attr_key in ["name", "placeholder", "aria-label", "id", "for"]:
                value = attrs.get(attr_key, "")
                if value:
                    field_map[value.lower()] = idx

            # Map by visible text / label content
            text = getattr(element, "text_content", "") or ""
            if text and len(text) < 100:  # Skip long text content
                field_map[text.strip().lower()] = idx

            # For <label> elements, extract the "for" attribute to link label text to input
            if tag_lower == "label" and text:
                label_for = attrs.get("for", "")
                if label_for:
                    # Map the human-readable label text to the input element index
                    # (will be resolved when we find the matching input by id)
                    field_map[text.strip().lower()] = field_map.get(label_for.lower(), idx)

            # Map submit buttons
            input_type = attrs.get("type", "")
            if tag_lower in ("button", "input") and input_type.lower() == "submit":
                field_map["submit"] = idx
                btn_text = getattr(element, "node_value", "") or ""
                if btn_text:
                    field_map[btn_text.lower()] = idx
                # Also map by value attribute (common for submit inputs)
                btn_value = attrs.get("value", "")
                if btn_value:
                    field_map[btn_value.lower()] = idx

            # Map button elements by text
            if tag_lower == "button":
                field_map["submit"] = field_map.get("submit", idx)
                if text:
                    field_map[text.strip().lower()] = idx

        # Post-process: link label text to actual input elements
        # Labels have "for" attribute pointing to input "id" -- resolve the mapping
        for idx, element in selector_map.items():
            tag = (getattr(element, "tag_name", "") or "").lower()
            if tag == "label":
                attrs = {}
                if hasattr(element, "attributes") and element.attributes:
                    if isinstance(element.attributes, dict):
                        attrs = element.attributes
                    else:
                        try:
                            attrs = dict(element.attributes)
                        except (TypeError, ValueError):
                            pass
                label_for = attrs.get("for", "")
                text = (getattr(element, "text_content", "") or "").strip().lower()
                if label_for and text:
                    # Find the input element with matching id
                    target_idx = field_map.get(label_for.lower())
                    if target_idx is not None:
                        field_map[text] = target_idx

        logger.info(f"Element map built: {len(field_map)} entries, keys={list(field_map.keys())[:40]}")
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

        Actions may include a "field_name" key in params (set by action_parser)
        to track which form fields were filled. This enables the field_accuracy
        reward component even when form submission does not occur.

        Args:
            actions: List of action dicts from action_parser.
            timeout_per_action: Max seconds per action before timeout.

        Returns:
            BrowserOutcome with execution results and filled field values.
        """
        executed = 0
        total = len(actions)
        filled_values: dict[str, str] = {}

        for action_dict in actions:
            action_name = action_dict.get("action", "")
            params = action_dict.get("params", {})

            # Extract field tracking metadata before passing to openbrowser
            field_name = params.pop("field_name", None)
            is_checkbox = params.pop("is_checkbox", False)

            # Map parser action names to openbrowser Tools method names
            tools_method = ACTION_NAME_MAP.get(action_name, action_name)

            try:
                action_fn = getattr(self.tools, tools_method, None)
                if action_fn is None:
                    logger.warning(f"Unknown action: {action_name} (mapped to {tools_method})")
                    continue

                result = await asyncio.wait_for(
                    action_fn(browser_session=self.browser_session, **params),
                    timeout=timeout_per_action,
                )

                if result and result.error:
                    logger.debug(f"Action {action_name} error: {result.error}")
                    return BrowserOutcome(
                        success_page_detected=False,
                        submitted_values=filled_values,
                        error=result.error,
                        actions_executed=executed,
                        total_actions=total,
                    )

                executed += 1

                # Track filled values for field_accuracy reward
                if field_name:
                    if action_name in ("input_text", "select_dropdown_option"):
                        filled_values[field_name] = params.get("text", "")
                    elif is_checkbox:
                        filled_values[field_name] = "true"

            except asyncio.TimeoutError:
                logger.warning(f"Action {action_name} timed out after {timeout_per_action}s")
                return BrowserOutcome(
                    success_page_detected=False,
                    submitted_values=filled_values,
                    error=f"Timeout on action {action_name}",
                    actions_executed=executed,
                    total_actions=total,
                )
            except Exception as e:
                logger.warning(f"Action {action_name} failed: {e}")
                return BrowserOutcome(
                    success_page_detected=False,
                    submitted_values=filled_values,
                    error=str(e),
                    actions_executed=executed,
                    total_actions=total,
                )

        # Check final page state for success indicators
        success = await self._check_success_page()

        return BrowserOutcome(
            success_page_detected=success,
            submitted_values=filled_values,
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
