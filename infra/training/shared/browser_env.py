"""Browser environment for online GRPO training using the openbrowser package.

Wraps openbrowser's BrowserSession and Tools to provide a training-friendly
interface for executing parsed action sequences in a real browser.
"""

import asyncio
import logging

from openbrowser import BrowserSession, Tools

from infra.training.shared.online_reward import BrowserOutcome

logger = logging.getLogger(__name__)

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
            await self.browser_session.close()
            logger.info("Browser environment closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
