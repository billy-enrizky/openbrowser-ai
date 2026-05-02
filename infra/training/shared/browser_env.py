"""Browser environment for online GRPO training using the openbrowser package.

Wraps openbrowser's BrowserSession and Tools to provide a training-friendly
interface for executing parsed action sequences in a real browser.
"""

import asyncio
import logging
import os
import random
import signal
import string
import subprocess
import threading

from openbrowser import BrowserSession, Tools
from openbrowser.code_use.namespace import evaluate as js_evaluate

from infra.training.shared.online_reward import BrowserOutcome

logger = logging.getLogger(__name__)


class BrowserResetError(Exception):
    """Raised when browser reset fails due to an unrecoverable event bus deadlock."""


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


def _perturb_text_value(value: str) -> str:
    """Apply a random perturbation to a text value for epsilon-greedy exploration."""
    if not value or len(value) < 2:
        return value
    perturbation = random.choice(["swap", "truncate", "append", "delete"])
    if perturbation == "swap":
        idx = random.randint(0, len(value) - 2)
        chars = list(value)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    elif perturbation == "truncate":
        cut = max(1, int(len(value) * 0.75))
        return value[:cut]
    elif perturbation == "append":
        return value + random.choice(string.ascii_lowercase)
    else:  # delete
        idx = random.randint(0, len(value) - 1)
        return value[:idx] + value[idx + 1:]


class BrowserEnvironment:
    """Training browser environment backed by the openbrowser package.

    Uses BrowserSession for browser lifecycle and Tools for action execution.
    Designed for the online Flow GRPO training loop where parsed action
    sequences are executed against FormFactory forms.
    """

    def __init__(
        self, browser_session: BrowserSession, tools: Tools, headless: bool = True,
        video_kwargs: dict | None = None,
    ):
        self.browser_session = browser_session
        self.tools = tools
        self._headless = headless
        self._video_kwargs = video_kwargs or {}

    @classmethod
    async def create(
        cls, headless: bool = True, **video_kwargs,
    ) -> "BrowserEnvironment":
        """Create and start a browser environment.

        Pass record_video_dir, record_video_size, etc. as kwargs
        to forward to BrowserSession for CDP screencast recording.
        """
        executable = _find_chromium_binary()
        # Allow localhost/127.0.0.1 for FormFactory server
        allowed = ["localhost", "127.0.0.1", "about:blank"]
        if executable:
            logger.info(f"Found browser binary: {executable}")
            browser_session = BrowserSession(
                headless=headless,
                executable_path=executable,
                allowed_domains=allowed,
                **video_kwargs,
            )
        else:
            logger.warning(
                "No pre-installed browser binary found, "
                "falling back to openbrowser auto-detection"
            )
            browser_session = BrowserSession(
                headless=headless, allowed_domains=allowed,
                **video_kwargs,
            )
        await browser_session.start()
        tools = Tools()
        logger.info("Browser environment created")
        return cls(browser_session, tools, headless=headless, video_kwargs=video_kwargs)

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
        epsilon: float = 0.0,
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

            # Epsilon-greedy: perturb text values for exploration diversity
            if epsilon > 0 and random.random() < epsilon:
                if action_name == "input_text" and "text" in params:
                    original = params["text"]
                    params["text"] = _perturb_text_value(original)
                    logger.debug(
                        "Epsilon-greedy: perturbed '%s' -> '%s' for %s",
                        original, params["text"], field_name or "unknown",
                    )

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

    async def get_dom_summary(self, max_chars: int = 800) -> str:
        """Get a truncated DOM text representation for multi-turn prompts.

        Uses openbrowser's llm_representation() and truncates to stay within
        the token budget (~200 tokens at ~4 chars/token).
        """
        try:
            state = await self.browser_session.get_browser_state_summary(
                include_screenshot=False
            )
            if state.dom_state:
                dom_text = state.dom_state.llm_representation()
                if len(dom_text) > max_chars:
                    dom_text = dom_text[:max_chars] + "\n[DOM truncated]"
                return dom_text
            return ""
        except Exception as e:
            logger.warning("Failed to get DOM summary: %s", e)
            return ""

    async def bypass_html5_validation(self) -> None:
        """Inject novalidate on all forms to bypass HTML5 client-side validation.

        Construction-manufacturing forms have required file upload fields the
        agent cannot populate.  Without novalidate, the browser blocks the
        POST request entirely, producing task_completion=0 regardless of
        field accuracy.
        """
        try:
            await js_evaluate(
                "(function(){ document.querySelectorAll('form').forEach("
                "f => f.setAttribute('novalidate', '')); return 'ok'; })()",
                self.browser_session,
            )
            logger.debug("Injected novalidate on all forms")
        except Exception as e:
            logger.warning("Failed to inject novalidate: %s", e)

    async def navigate_with_timeout(self, url: str, timeout: float = 30.0) -> None:
        """Navigate to a URL with an asyncio timeout.

        For deadlock protection, callers should use RolloutWatchdog which
        covers all browser operations, not just navigation.
        """
        await asyncio.wait_for(
            self.tools.navigate(
                url=url, new_tab=False,
                browser_session=self.browser_session,
            ),
            timeout=timeout,
        )

    async def reset(self) -> None:
        """Reset browser state between rollouts."""
        try:
            await self.navigate_with_timeout("about:blank", timeout=20.0)
        except Exception as e:
            logger.warning(f"Browser reset failed: {e}")
            raise BrowserResetError(
                "Browser session corrupted after event bus deadlock"
            ) from e

    def _kill_chromium_processes(self) -> None:
        """Force-kill all chromium processes owned by this user.

        This is a last resort when the event bus is deadlocked and
        asyncio.wait_for / CancelledError cannot unblock the session.
        """
        try:
            subprocess.run(
                ["pkill", "-9", "-f", "chromium"],
                timeout=5,
                capture_output=True,
            )
            logger.info("Force-killed chromium processes via pkill")
        except Exception as e:
            logger.warning(f"pkill chromium failed: {e}")

    def _force_kill_browser_processes(self) -> None:
        """Force-kill all chromium/chrome processes at OS level."""
        for proc_name in ("chromium", "chrome", "headless_shell"):
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", proc_name],
                    capture_output=True, timeout=3,
                )
            except Exception:
                pass

    async def _create_fresh_session(self) -> None:
        """Create and start a new browser session, replacing the old one."""
        executable = _find_chromium_binary()
        allowed = ["localhost", "127.0.0.1", "about:blank"]
        if executable:
            self.browser_session = BrowserSession(
                headless=self._headless,
                executable_path=executable,
                allowed_domains=allowed,
                **self._video_kwargs,
            )
        else:
            self.browser_session = BrowserSession(
                headless=self._headless, allowed_domains=allowed,
                **self._video_kwargs,
            )
        await self.browser_session.start()
        self.tools = Tools()

    async def restart(self, timeout: float = 30.0) -> None:
        """Kill and relaunch the browser to prevent session degradation.

        Chromium accumulates memory and event handler state over many
        navigations, eventually causing timeouts.  A full restart clears
        all of that.  We force-kill all chromium/chrome processes at the
        OS level to ensure no zombie processes linger.

        The entire restart operation is timeout-protected.  If it exceeds
        ``timeout`` seconds, we force-kill again and retry once.
        """
        for attempt in range(2):
            logger.info(
                "Restarting browser (attempt %d, killing old session)", attempt + 1
            )
            try:
                await asyncio.wait_for(self.browser_session.kill(), timeout=5.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("browser_session.kill() failed: %s", e)

            self._force_kill_browser_processes()
            await asyncio.sleep(1.0)

            try:
                await asyncio.wait_for(
                    self._create_fresh_session(), timeout=timeout
                )
                logger.info("Browser restarted successfully")
                return
            except asyncio.TimeoutError:
                logger.error(
                    "Browser restart timed out after %.0fs (attempt %d)",
                    timeout, attempt + 1,
                )
                # Force-kill again before retry
                self._force_kill_browser_processes()
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.error("Browser restart failed: %s (attempt %d)", e, attempt + 1)
                self._force_kill_browser_processes()
                await asyncio.sleep(2.0)

        # If both attempts failed, raise so the caller can handle it
        raise RuntimeError("Browser restart failed after 2 attempts")

    async def safe_navigate(
        self, url: str, nav_timeout: float = 30.0, max_retries: int = 2,
    ) -> dict[str, int]:
        """Navigate to a URL and build the element map, with timeout protection.

        Wraps navigate + get_element_map in a hard timeout.  On failure,
        restarts the browser and retries up to ``max_retries`` times.

        Returns:
            Element map (field_name -> element_index).

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                async with asyncio.timeout(nav_timeout):
                    await self.tools.navigate(
                        url=url,
                        new_tab=False,
                        browser_session=self.browser_session,
                    )
                    await asyncio.sleep(0.5)
                    element_map = await self.get_element_map()
                return element_map
            except (asyncio.TimeoutError, TimeoutError, Exception) as e:
                last_error = e
                logger.warning(
                    "safe_navigate attempt %d failed: %s", attempt + 1, e
                )
                if attempt < max_retries:
                    try:
                        await self.restart()
                    except RuntimeError:
                        logger.error("Browser restart failed during safe_navigate retry")

        raise RuntimeError(
            f"safe_navigate failed after {max_retries + 1} attempts: {last_error}"
        )

    async def close(self) -> None:
        """Shutdown browser. Falls back to OS-level kill if graceful shutdown hangs."""
        try:
            await asyncio.wait_for(
                self.browser_session.kill(),
                timeout=10.0,
            )
            logger.info("Browser environment closed")
        except asyncio.TimeoutError:
            logger.warning(
                "Browser kill timed out (event bus deadlock), "
                "force-killing chromium processes"
            )
            self._kill_chromium_processes()
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
            self._kill_chromium_processes()

    @classmethod
    async def force_restart(cls, env: "BrowserEnvironment", headless: bool = True) -> "BrowserEnvironment":
        """Force-restart browser: kill everything and create fresh session.

        Use when the event bus is deadlocked and normal close/create fails.
        """
        logger.warning("Force-restarting browser environment")
        # Try graceful close first
        try:
            await asyncio.wait_for(env.browser_session.kill(), timeout=5.0)
        except Exception:
            pass
        # Nuclear option: kill all chromium processes
        env._kill_chromium_processes()
        await asyncio.sleep(1)  # Brief pause for OS cleanup
        return await cls.create(headless=headless)


class RolloutWatchdog:
    """Thread-based watchdog that kills chromium if a rollout exceeds a timeout.

    The openbrowser event bus can deadlock during ANY browser operation
    (navigation, typing, clicking, DOM queries). When it deadlocks, it blocks
    the asyncio event loop thread, making all async recovery mechanisms
    (asyncio.wait_for, asyncio.Event, CancelledError) ineffective.

    This watchdog runs a threading.Timer completely outside asyncio. When it
    fires, it:
      1. Kills all chromium processes via pkill -9 (breaks the deadlock)
      2. Cancels the current asyncio task via loop.call_soon_threadsafe
         (unblocks the stuck coroutine once the event loop resumes)

    Usage in the trainer::

        watchdog = RolloutWatchdog(browser_env, timeout=120.0)
        watchdog.start()
        try:
            await browser_env.reset()
            await browser_env.navigate_with_timeout(url)
            ...  # all browser ops covered
        except (BrowserResetError, asyncio.CancelledError):
            browser_env = await BrowserEnvironment.force_restart(...)
            reward = 0.0
        finally:
            watchdog.disarm()
    """

    def __init__(self, browser_env: BrowserEnvironment, timeout: float = 120.0):
        self.browser_env = browser_env
        self.timeout = timeout
        self._completed = threading.Event()
        self._timer: threading.Timer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Arm the watchdog. Must be called from the asyncio event loop thread."""
        self._loop = asyncio.get_running_loop()
        # Capture the current task so we can cancel it from the timer thread
        self._task = asyncio.current_task()
        self._completed.clear()

        self._timer = threading.Timer(self.timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def disarm(self) -> None:
        """Disarm the watchdog after successful rollout completion."""
        self._completed.set()
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timeout(self) -> None:
        """Called from the timer thread when the rollout exceeds the timeout."""
        if self._completed.is_set():
            return
        logger.warning(
            "Rollout watchdog fired after %.0fs -- "
            "killing chromium and cancelling task to break deadlock",
            self.timeout,
        )
        # Step 1: Kill chromium processes (works even if event loop is blocked)
        self.browser_env._kill_chromium_processes()
        # Step 2: Cancel the asyncio task (processed once event loop unblocks)
        if self._loop and self._task and not self._task.done():
            self._loop.call_soon_threadsafe(self._task.cancel)
