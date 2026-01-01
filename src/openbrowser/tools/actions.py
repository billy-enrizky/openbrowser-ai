"""Browser action tools using registry pattern following browser-use."""

from __future__ import annotations

import asyncio
import enum
import json
import logging
from typing import Any, Generic, Optional, TypeVar, TYPE_CHECKING

from pydantic import BaseModel, Field

from src.openbrowser.agent.views import ActionResult
from src.openbrowser.browser.dom import DomState
from src.openbrowser.browser.session import BrowserSession
from src.openbrowser.observability import observe_debug
from src.openbrowser.tools.registry import Registry, ActionModel

T = TypeVar('T', bound=BaseModel)

if TYPE_CHECKING:
    from src.openbrowser.browser.views import BrowserError

logger = logging.getLogger(__name__)

KEY_DEFINITIONS = {
    "Enter": {"key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
    "Tab": {"key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9},
    "Backspace": {"key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8},
    "Escape": {"key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27},
    "ArrowDown": {"key": "ArrowDown", "code": "ArrowDown", "windowsVirtualKeyCode": 40},
    "ArrowUp": {"key": "ArrowUp", "code": "ArrowUp", "windowsVirtualKeyCode": 38},
    "ArrowLeft": {"key": "ArrowLeft", "code": "ArrowLeft", "windowsVirtualKeyCode": 37},
    "ArrowRight": {"key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39},
    "Space": {"key": " ", "code": "Space", "windowsVirtualKeyCode": 32},
}

# CAPTCHA detection patterns
CAPTCHA_INDICATORS = [
    "captcha",
    "recaptcha",
    "i'm not a robot",
    "unusual traffic",
    "verify you're human",
    "robot verification",
    "security check",
    "please verify",
    "automated queries",
]


async def detect_captcha(browser_session: BrowserSession) -> bool:
    """Detect if the current page shows a CAPTCHA.
    
    Returns:
        True if CAPTCHA is detected, False otherwise.
    """
    if not browser_session.agent_focus:
        return False
    
    try:
        cdp_session = browser_session.agent_focus
        client = cdp_session.cdp_client
        session_id = cdp_session.session_id
        
        # Get the page content
        result = await client.send.Runtime.evaluate(
            params={"expression": "document.body ? document.body.innerText.toLowerCase() : ''"},
            session_id=session_id,
        )
        
        if result and "result" in result and "value" in result["result"]:
            page_text = result["result"]["value"]
            
            # Check for CAPTCHA indicators in page text
            for indicator in CAPTCHA_INDICATORS:
                if indicator in page_text:
                    logger.info(f"CAPTCHA detected: found '{indicator}' in page content")
                    return True
        
        # Also check URL for CAPTCHA patterns
        url_result = await client.send.Runtime.evaluate(
            params={"expression": "window.location.href.toLowerCase()"},
            session_id=session_id,
        )
        
        if url_result and "result" in url_result and "value" in url_result["result"]:
            url = url_result["result"]["value"]
            if "captcha" in url or "sorry" in url or "recaptcha" in url:
                logger.info(f"CAPTCHA detected: URL contains CAPTCHA pattern: {url}")
                return True
        
        return False
        
    except Exception as e:
        logger.debug(f"Error detecting CAPTCHA: {e}")
        return False


class NavigateParams(ActionModel):
    url: str = Field(description="The URL to navigate to")


class ClickParams(ActionModel):
    index: int = Field(description="The element index [N] from the DOM tree to click")


class InputParams(ActionModel):
    index: int = Field(description="The element index [N] from the DOM tree to type into")
    text: str = Field(description="The text to type")


class SendKeysParams(ActionModel):
    keys: str = Field(description="The key(s) to press (Enter, Tab, Escape, etc.)")


class ScrollParams(ActionModel):
    direction: str = Field(default="down", description="Direction to scroll: up or down")
    amount: int | None = Field(default=None, description="Amount to scroll in pixels (if None, scrolls one page)")
    pages: float = Field(default=1.0, description="Number of pages to scroll (supports fractional values)")
    index: int | None = Field(default=None, description="Element index [N] to scroll within (for scrollable elements)")


class WaitParams(ActionModel):
    seconds: int = Field(default=3, description="Number of seconds to wait")


class DoneParams(ActionModel):
    text: str = Field(description="Final result or message to return")
    success: bool = Field(default=True, description="Whether the task was successful")
    files_to_display: list[str] | None = Field(default=[], description="List of file names to display in the done message")


class GoBackParams(ActionModel):
    pass


class ScreenshotParams(ActionModel):
    pass


class SearchParams(ActionModel):
    query: str = Field(description="The search query to execute")
    engine: str = Field(default="duckduckgo", description="Search engine: duckduckgo, google, or bing")


class ExtractParams(ActionModel):
    query: str = Field(description="What information to extract from the current page")
    extract_links: bool = Field(
        default=False, description="Set True if the query requires links, else false to save tokens"
    )
    start_from_char: int = Field(
        default=0, description="Use this for long markdowns to start from a specific character (not index in browser_state)"
    )


class FindTextParams(ActionModel):
    text: str = Field(description="The text to find and scroll to on the page")


class SelectDropdownParams(ActionModel):
    index: int = Field(description="The element index [N] of the dropdown")
    value: str = Field(description="The value or text to select")


class DropdownOptionsParams(ActionModel):
    index: int = Field(description="The element index [N] of the dropdown to get options from")


class SwitchTabParams(ActionModel):
    tab_id: int = Field(description="The tab index to switch to (0-based)")


class CloseTabParams(ActionModel):
    tab_id: int | None = Field(default=None, description="The tab index to close. If None, closes current tab")


class UploadFileParams(ActionModel):
    index: int = Field(description="The element index [N] of the file input")
    file_path: str = Field(description="Path to the file to upload")


class EvaluateParams(ActionModel):
    script: str = Field(description="JavaScript code to execute in the browser")


class ReadFileParams(ActionModel):
    file_path: str = Field(description="Path to the file to read")


class WriteFileParams(ActionModel):
    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class ReplaceFileParams(ActionModel):
    file_path: str = Field(description="Path to the file")
    old_text: str = Field(description="Text to find and replace")
    new_text: str = Field(description="Replacement text")


class StructuredOutputAction(BaseModel, Generic[T]):
    """Action model for structured output."""
    success: bool = Field(default=True, description='True if user_request completed successfully')
    data: T = Field(description='The actual output data matching the requested schema')


class Tools:
    """Tools service following browser-use pattern with registry."""

    def __init__(
        self,
        browser_session: BrowserSession,
        exclude_actions: list[str] | None = None,
        output_model: type[T] | None = None,
        display_files_in_done_text: bool = True,
    ):
        self.browser_session = browser_session
        self._selector_map: dict[int, int] = {}
        self.display_files_in_done_text = display_files_in_done_text
        self.registry = Registry(exclude_actions=exclude_actions)
        self._google_blocked = False  # Track if Google is blocked due to CAPTCHA
        self._register_done_action(output_model)
        self._register_default_actions()

    def _register_done_action(self, output_model: type[T] | None) -> None:
        """Register done action with optional structured output support."""
        tools_instance = self
        
        if output_model is not None:
            # Register structured output variant
            @self.registry.action(
                'Complete task with structured output.',
                param_model=StructuredOutputAction[output_model],  # type: ignore
            )
            async def done(params: StructuredOutputAction):  # type: ignore
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(
                    is_done=True,
                    success=params.success,
                    extracted_content=json.dumps(output_dict, ensure_ascii=False),
                    long_term_memory=f'Task completed. Success Status: {params.success}',
                )
        else:
            # Register regular done action
            @self.registry.action("Complete the task and return result", param_model=DoneParams)
            async def done(params: DoneParams, browser_session: BrowserSession = None, file_system: Any = None) -> ActionResult:
                user_message = params.text

                len_text = len(params.text)
                len_max_memory = 100
                memory = f'Task completed: {params.success} - {params.text[:len_max_memory]}'
                if len_text > len_max_memory:
                    memory += f' - {len_text - len_max_memory} more characters'

                attachments = []
                if params.files_to_display:
                    if self.display_files_in_done_text:
                        file_msg = ''
                        for file_name in params.files_to_display:
                            if file_system:
                                file_content = file_system.display_file(file_name)
                                if file_content:
                                    file_msg += f'\n\n{file_name}:\n{file_content}'
                                    attachments.append(file_name)
                        if file_msg:
                            user_message += '\n\nAttachments:'
                            user_message += file_msg
                        else:
                            logger.warning('Agent wanted to display files but none were found')
                    else:
                        if file_system:
                            for file_name in params.files_to_display:
                                file_content = file_system.display_file(file_name)
                                if file_content:
                                    attachments.append(file_name)

                # Convert attachments to full paths if file_system available
                if file_system and attachments:
                    attachments = [str(file_system.get_dir() / file_name) for file_name in attachments]

                return ActionResult(
                    is_done=True,
                    success=params.success,
                    extracted_content=user_message,
                    long_term_memory=memory,
                    attachments=attachments if attachments else None,
                )

    def use_structured_output_action(self, output_model: type[T]) -> None:
        """Register structured output action for done."""
        self._register_done_action(output_model)

    def update_state(self, dom_state: DomState) -> None:
        self._selector_map = dom_state.selector_map
        logger.info(f"Updated selector map with {len(self._selector_map)} elements")

    def _validate_and_fix_javascript(self, code: str) -> str:
        """Validate and fix common JavaScript issues before execution.

        Fixes issues like:
        - Double-escaped quotes from LLM output
        - XPath expressions with mixed quotes
        - querySelector with mixed quotes
        """
        import re

        fixed_code = code

        # Fix double-escaped quotes (common in LLM output)
        if '\\"' in fixed_code:
            # Only fix if there are more escaped quotes than unescaped
            if fixed_code.count('\\"') > fixed_code.count('"') - fixed_code.count('\\"'):
                fixed_code = fixed_code.replace('\\"', '"')
        if "\\'" in fixed_code:
            if fixed_code.count("\\'") > fixed_code.count("'") - fixed_code.count("\\'"):
                fixed_code = fixed_code.replace("\\'", "'")

        # Fix XPath expressions with mixed quotes
        # Pattern: document.evaluate("//div[text()='something']", ...)
        xpath_pattern = r'document\.evaluate\s*\(\s*["\']([^"\']*)["\']'

        def fix_xpath_quotes(match):
            xpath_expr = match.group(1)
            # If XPath uses single quotes inside, wrap with double quotes
            if "'" in xpath_expr and '"' not in xpath_expr:
                return f'document.evaluate("{xpath_expr}"'
            elif '"' in xpath_expr and "'" not in xpath_expr:
                return f"document.evaluate('{xpath_expr}'"
            return match.group(0)

        fixed_code = re.sub(xpath_pattern, fix_xpath_quotes, fixed_code)

        # Fix querySelector with mixed quotes
        selector_pattern = r'querySelector(?:All)?\s*\(\s*["\']([^"\']*)["\']'

        def fix_selector_quotes(match):
            selector = match.group(1)
            if "'" in selector and '"' not in selector:
                return f'querySelector("{selector}"' if 'All' not in match.group(0) else f'querySelectorAll("{selector}"'
            return match.group(0)

        fixed_code = re.sub(selector_pattern, fix_selector_quotes, fixed_code)

        return fixed_code

    def _detect_sensitive_key_name(self, text: str, sensitive_data: dict[str, Any] | None) -> str | None:
        """Detect which sensitive key name corresponds to the given text value.

        Args:
            text: The text value to check
            sensitive_data: Dict of sensitive data (can be nested by domain)

        Returns:
            The key name if found, None otherwise
        """
        if not sensitive_data:
            return None

        for domain_or_key, content in sensitive_data.items():
            if isinstance(content, dict):
                # Nested by domain
                for key, value in content.items():
                    if value == text:
                        return key
            elif content == text:
                return domain_or_key

        return None

    def _mask_sensitive_text(
        self,
        text: str,
        sensitive_data: dict[str, Any] | None,
    ) -> str:
        """Mask sensitive text in a message for logging.

        Args:
            text: Text that may contain sensitive data
            sensitive_data: Dict of sensitive data

        Returns:
            Text with sensitive values replaced by their key names
        """
        if not sensitive_data:
            return text

        masked_text = text
        for domain_or_key, content in sensitive_data.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, str) and value and value in masked_text:
                        masked_text = masked_text.replace(value, f"<{key}>")
            elif isinstance(content, str) and content and content in masked_text:
                masked_text = masked_text.replace(content, f"<{domain_or_key}>")

        return masked_text

    async def _highlight_element(self, client, session_id: str, backend_node_id: int) -> None:
        try:
            await client.send.DOM.enable(session_id=session_id)
            await client.send.Runtime.enable(session_id=session_id)
            resolve_result = await client.send.DOM.resolveNode(
                params={"backendNodeId": backend_node_id}, session_id=session_id
            )
            if "object" not in resolve_result or "objectId" not in resolve_result["object"]:
                return
            object_id = resolve_result["object"]["objectId"]
            await client.send.Runtime.callFunctionOn(
                params={
                    "objectId": object_id,
                    "functionDeclaration": """function() {
                        this.style.outline = "3px solid red";
                        this.style.transition = "outline 0.3s";
                        this.scrollIntoView({behavior: "smooth", block: "center"});
                    }""",
                    "returnByValue": False,
                },
                session_id=session_id,
            )
        except Exception as e:
            logger.warning(f"Failed to highlight element: {e}")

    def _convert_google_to_bing(self, google_url: str) -> str:
        """Convert a Google URL to equivalent Bing URL.
        
        Args:
            google_url: The Google URL to convert
            
        Returns:
            The equivalent Bing URL
        """
        from urllib.parse import urlparse, parse_qs, quote_plus, unquote
        
        parsed = urlparse(google_url)
        
        # If it's a Google search URL, extract the query and build Bing URL
        if "google.com" in parsed.netloc:
            query_params = parse_qs(parsed.query)
            
            search_query = None
            
            # For Google sorry/CAPTCHA pages, the original search is in the 'continue' parameter
            if 'continue' in query_params:
                continue_url = unquote(query_params['continue'][0])
                continue_parsed = urlparse(continue_url)
                continue_params = parse_qs(continue_parsed.query)
                if 'q' in continue_params:
                    search_query = continue_params['q'][0]
            
            # Fallback to direct 'q' parameter (for regular Google search pages)
            if not search_query and 'q' in query_params:
                search_query = query_params['q'][0]
            
            if search_query:
                return f"https://www.bing.com/search?q={quote_plus(search_query)}&setlang=en&cc=US"
            
            # If it's just google.com without search, redirect to bing.com
            return "https://www.bing.com?setlang=en&cc=US"
        
        return google_url

    def _register_default_actions(self) -> None:
        tools_instance = self

        @self.registry.action("Navigate to a URL", param_model=NavigateParams)
        async def navigate(params: NavigateParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            
            target_url = params.url
            
            # If Google is blocked due to CAPTCHA, redirect to Bing automatically
            if tools_instance._google_blocked and "google.com" in target_url.lower():
                logger.warning("Google is blocked due to CAPTCHA, redirecting to Bing")
                target_url = tools_instance._convert_google_to_bing(target_url)
                logger.info(f"Converted Google URL to Bing: {target_url}")
            
            logger.info(f"Navigating to {target_url}")
            from src.openbrowser.browser.events import NavigateToUrlEvent
            await session.event_bus.dispatch(NavigateToUrlEvent(url=target_url, new_tab=False))
            
            # Wait for page to load before checking for CAPTCHA
            await asyncio.sleep(1.5)
            
            # Check for CAPTCHA on Google pages (only if not already blocked)
            if not tools_instance._google_blocked and "google.com" in target_url.lower():
                if await detect_captcha(session):
                    logger.warning("CAPTCHA detected on Google, falling back to Bing")
                    
                    # Mark Google as blocked
                    tools_instance._google_blocked = True
                    
                    # Convert Google URL to Bing URL
                    fallback_url = tools_instance._convert_google_to_bing(target_url)
                    logger.info(f"Redirecting to Bing: {fallback_url}")
                    
                    await session.event_bus.dispatch(NavigateToUrlEvent(url=fallback_url, new_tab=False))
                    return ActionResult(
                        extracted_content=f"CAPTCHA detected on Google. Google is now blocked. Automatically redirected to Bing: {fallback_url}. Use Bing for all future searches."
                    )
            
            if tools_instance._google_blocked and "google.com" in params.url.lower():
                return ActionResult(extracted_content=f"Google is blocked due to CAPTCHA. Navigated to Bing instead: {target_url}")
            
            return ActionResult(extracted_content=f"Navigated to {target_url}")

        @self.registry.action("Click on an element by index", param_model=ClickParams)
        async def click(params: ClickParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            if params.index not in tools_instance._selector_map:
                return ActionResult(error=f"Element index {params.index} not found")
            backend_node_id = tools_instance._selector_map[params.index]
            logger.info(f"Clicking element {params.index} (backend_node_id: {backend_node_id})")
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            try:
                await client.send.DOM.enable(session_id=session_id)
                await client.send.Input.enable(session_id=session_id)
            except Exception:
                pass
            
            # Try to scroll element into view first
            try:
                await client.send.DOM.scrollIntoViewIfNeeded(
                    params={"backendNodeId": backend_node_id}, session_id=session_id
                )
                await asyncio.sleep(0.1)
            except Exception:
                pass
            
            # Try multiple methods to get element coordinates (following browser-use pattern)
            x, y = None, None
            
            # Method 1: Try DOM.getContentQuads first (best for inline elements)
            try:
                content_quads_result = await client.send.DOM.getContentQuads(
                    params={"backendNodeId": backend_node_id}, session_id=session_id
                )
                if "quads" in content_quads_result and content_quads_result["quads"]:
                    quad = content_quads_result["quads"][0]
                    if len(quad) >= 8:
                        x = sum(quad[i] for i in range(0, 8, 2)) / 4
                        y = sum(quad[i] for i in range(1, 8, 2)) / 4
            except Exception:
                pass
            
            # Method 2: Fall back to DOM.getBoxModel
            if x is None:
                try:
                    box_model_result = await client.send.DOM.getBoxModel(
                        params={"backendNodeId": backend_node_id}, session_id=session_id
                    )
                    if "model" in box_model_result and "content" in box_model_result["model"]:
                        content_quad = box_model_result["model"]["content"]
                        if len(content_quad) >= 8:
                            x = (content_quad[0] + content_quad[2] + content_quad[4] + content_quad[6]) / 4
                            y = (content_quad[1] + content_quad[3] + content_quad[5] + content_quad[7]) / 4
                except Exception:
                    pass
            
            # Method 3: Fall back to JavaScript getBoundingClientRect
            if x is None:
                try:
                    resolve_result = await client.send.DOM.resolveNode(
                        params={"backendNodeId": backend_node_id}, session_id=session_id
                    )
                    if "object" in resolve_result and "objectId" in resolve_result["object"]:
                        object_id = resolve_result["object"]["objectId"]
                        bounds_result = await client.send.Runtime.callFunctionOn(
                            params={
                                "functionDeclaration": """
                                    function() {
                                        const rect = this.getBoundingClientRect();
                                        return {
                                            x: rect.left + rect.width / 2,
                                            y: rect.top + rect.height / 2,
                                            width: rect.width,
                                            height: rect.height
                                        };
                                    }
                                """,
                                "objectId": object_id,
                                "returnByValue": True,
                            },
                            session_id=session_id,
                        )
                        if "result" in bounds_result and "value" in bounds_result["result"]:
                            rect = bounds_result["result"]["value"]
                            if rect.get("width", 0) > 0 and rect.get("height", 0) > 0:
                                x = rect["x"]
                                y = rect["y"]
                except Exception:
                    pass
            
            # Method 4: Fall back to JavaScript click if we still don't have coordinates
            if x is None:
                try:
                    resolve_result = await client.send.DOM.resolveNode(
                        params={"backendNodeId": backend_node_id}, session_id=session_id
                    )
                    if "object" in resolve_result and "objectId" in resolve_result["object"]:
                        object_id = resolve_result["object"]["objectId"]
                        await client.send.Runtime.callFunctionOn(
                            params={
                                "functionDeclaration": "function() { this.click(); }",
                                "objectId": object_id,
                            },
                            session_id=session_id,
                        )
                        logger.info(f"Clicked element {params.index} via JavaScript fallback")
                        return ActionResult(extracted_content=f"Clicked element {params.index} (via JS)")
                except Exception as js_e:
                    return ActionResult(error=f"Element {params.index} could not be clicked: {js_e}")
                return ActionResult(error=f"Element {params.index} is not visible or clickable")
            
            await tools_instance._highlight_element(client, session_id, backend_node_id)
            await asyncio.sleep(0.3)
            
            # Move mouse to element first
            await client.send.Input.dispatchMouseEvent(
                params={"type": "mouseMoved", "x": x, "y": y},
                session_id=session_id,
            )
            await asyncio.sleep(0.05)
            
            await client.send.Input.dispatchMouseEvent(
                params={"type": "mousePressed", "button": "left", "x": x, "y": y, "clickCount": 1},
                session_id=session_id,
            )
            await client.send.Input.dispatchMouseEvent(
                params={"type": "mouseReleased", "button": "left", "x": x, "y": y, "clickCount": 1},
                session_id=session_id,
            )
            return ActionResult(extracted_content=f"Clicked element {params.index}")

        @self.registry.action("Type text into an element by index", param_model=InputParams)
        async def input_text(params: InputParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            if params.index not in tools_instance._selector_map:
                return ActionResult(error=f"Element index {params.index} not found")
            click_result = await click(params=ClickParams(index=params.index), browser_session=session)
            if click_result.error:
                return click_result
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            for char in params.text:
                await client.send.Input.dispatchKeyEvent(
                    params={"type": "char", "text": char}, session_id=session_id
                )
            return ActionResult(extracted_content=f"Typed '{params.text}' into element {params.index}")

        @self.registry.action("Press keyboard keys", param_model=SendKeysParams)
        async def send_keys(params: SendKeysParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            key = params.keys
            if key not in KEY_DEFINITIONS:
                return ActionResult(error=f"Key '{key}' not supported. Available: {list(KEY_DEFINITIONS.keys())}")
            key_def = KEY_DEFINITIONS[key]
            logger.info(f"Pressing key: {key}")
            await client.send.Input.dispatchKeyEvent(
                params={
                    "type": "keyDown",
                    "key": key_def["key"],
                    "code": key_def["code"],
                    "windowsVirtualKeyCode": key_def["windowsVirtualKeyCode"],
                },
                session_id=session_id,
            )
            await client.send.Input.dispatchKeyEvent(
                params={
                    "type": "keyUp",
                    "key": key_def["key"],
                    "code": key_def["code"],
                    "windowsVirtualKeyCode": key_def["windowsVirtualKeyCode"],
                },
                session_id=session_id,
            )
            if key == "Enter":
                await asyncio.sleep(1.0)
            return ActionResult(extracted_content=f"Pressed key: {key}")

        @self.registry.action("Scroll the page", param_model=ScrollParams)
        async def scroll(params: ScrollParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            # Get viewport height for page-based scrolling
            try:
                metrics = await client.send.Page.getLayoutMetrics(session_id=session_id)
                viewport_height = metrics.get('cssVisualViewport', {}).get('clientHeight', 600)
            except Exception:
                viewport_height = 600  # Fallback

            # Calculate scroll amount
            if params.amount is not None:
                scroll_amount = params.amount
            else:
                # Scroll by pages (use 90% of viewport to leave context)
                scroll_amount = int(viewport_height * 0.9 * params.pages)

            delta_y = scroll_amount if params.direction == "down" else -scroll_amount

            # Check if scrolling within a specific element
            if params.index is not None:
                backend_node_id = tools_instance._selector_map.get(params.index)
                if backend_node_id:
                    try:
                        # Use JavaScript to scroll within the element
                        result = await client.send.DOM.resolveNode(
                            params={"backendNodeId": backend_node_id},
                            session_id=session_id,
                        )
                        object_id = result.get("object", {}).get("objectId")
                        if object_id:
                            scroll_script = f"""
                            function() {{
                                this.scrollBy(0, {delta_y});
                                return this.scrollTop;
                            }}
                            """
                            await client.send.Runtime.callFunctionOn(
                                params={"functionDeclaration": scroll_script, "objectId": object_id},
                                session_id=session_id,
                            )
                            await asyncio.sleep(0.3)
                            return ActionResult(extracted_content=f"Scrolled element [{params.index}] {params.direction} by {abs(delta_y)}px")
                    except Exception as e:
                        logger.warning(f"Failed to scroll element, falling back to page scroll: {e}")

            # Multi-page scrolling with delays
            pages_to_scroll = int(params.pages) if params.pages > 1 else 1
            page_delta = delta_y // pages_to_scroll if pages_to_scroll > 1 else delta_y

            for i in range(pages_to_scroll):
                await client.send.Input.dispatchMouseEvent(
                    params={"type": "mouseWheel", "x": 400, "y": 300, "deltaX": 0, "deltaY": page_delta},
                    session_id=session_id,
                )
                if pages_to_scroll > 1:
                    await asyncio.sleep(0.3)

            await asyncio.sleep(0.3)
            return ActionResult(extracted_content=f"Scrolled {params.direction} by {abs(delta_y)}px ({params.pages} page(s))")

        @self.registry.action("Wait for a specified time", param_model=WaitParams)
        async def wait(params: WaitParams, browser_session: BrowserSession = None) -> ActionResult:
            # Cap wait time at 30 seconds and account for LLM overhead (reduce by 3 seconds)
            actual_seconds = min(max(params.seconds - 3, 0), 30)
            memory = f'Waited for {params.seconds} seconds'
            logger.info(f'🕒 waited for {params.seconds} second{"" if params.seconds == 1 else "s"}')
            await asyncio.sleep(actual_seconds)
            return ActionResult(extracted_content=memory, long_term_memory=memory)

        @self.registry.action("Go back to the previous page", param_model=GoBackParams)
        async def go_back(params: GoBackParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            await client.send.Page.goBack(session_id=session_id)
            await asyncio.sleep(1.0)
            return ActionResult(extracted_content="Navigated back")


        @self.registry.action("Take a screenshot of the current page", param_model=ScreenshotParams)
        async def screenshot(params: ScreenshotParams, browser_session: BrowserSession = None) -> ActionResult:
            return ActionResult(extracted_content="Screenshot captured", metadata={"include_screenshot": True})

        @self.registry.action("Search the web using a search engine", param_model=SearchParams)
        async def search(params: SearchParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            engine_urls = {
                "duckduckgo": f"https://duckduckgo.com/?q={params.query.replace(' ', '+')}",
                "google": f"https://www.google.com/search?q={params.query.replace(' ', '+')}",
                "bing": f"https://www.bing.com/search?q={params.query.replace(' ', '+')}&setlang=en&cc=US",
            }
            url = engine_urls.get(params.engine.lower(), engine_urls["duckduckgo"])
            logger.info(f"Searching for '{params.query}' using {params.engine}")
            from src.openbrowser.browser.events import NavigateToUrlEvent
            await session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=False))
            return ActionResult(extracted_content=f"Searched for '{params.query}' using {params.engine}")

        @self.registry.action(
            "LLM extracts structured data from page markdown. Use when: on right page, know what to extract, haven't called before on same page+query. Can't get interactive elements. Set extract_links=True for URLs. Use start_from_char if truncated. If fails, use find_text instead.",
            param_model=ExtractParams,
        )
        async def extract(
            params: ExtractParams,
            browser_session: BrowserSession = None,
            page_extraction_llm: Any = None,
            file_system: Any = None,
        ) -> ActionResult:
            # Constants
            MAX_CHAR_LIMIT = 30000
            query = params.query if isinstance(params, dict) else params.query
            extract_links = params.extract_links if isinstance(params, dict) else params.extract_links
            start_from_char = params.start_from_char if isinstance(params, dict) else params.start_from_char

            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")

            # Extract clean markdown using the unified method
            try:
                from src.openbrowser.browser.dom.markdown_extractor import extract_clean_markdown

                content, content_stats = await extract_clean_markdown(
                    browser_session=session, extract_links=extract_links
                )
            except Exception as e:
                raise RuntimeError(f'Could not extract clean markdown: {type(e).__name__}')

            # Original content length for processing
            final_filtered_length = content_stats['final_filtered_chars']

            if start_from_char > 0:
                if start_from_char >= len(content):
                    return ActionResult(
                        error=f'start_from_char ({start_from_char}) exceeds content length {final_filtered_length} characters.'
                    )
                content = content[start_from_char:]
                content_stats['started_from_char'] = start_from_char

            # Smart truncation with context preservation
            truncated = False
            if len(content) > MAX_CHAR_LIMIT:
                # Try to truncate at a natural break point (paragraph, sentence)
                truncate_at = MAX_CHAR_LIMIT

                # Look for paragraph break within last 500 chars of limit
                paragraph_break = content.rfind('\n\n', MAX_CHAR_LIMIT - 500, MAX_CHAR_LIMIT)
                if paragraph_break > 0:
                    truncate_at = paragraph_break
                else:
                    # Look for sentence break within last 200 chars of limit
                    sentence_break = content.rfind('.', MAX_CHAR_LIMIT - 200, MAX_CHAR_LIMIT)
                    if sentence_break > 0:
                        truncate_at = sentence_break + 1

                content = content[:truncate_at]
                truncated = True
                next_start = (start_from_char or 0) + truncate_at
                content_stats['truncated_at_char'] = truncate_at
                content_stats['next_start_char'] = next_start

            # Add content statistics to the result
            original_html_length = content_stats['original_html_chars']
            initial_markdown_length = content_stats['initial_markdown_chars']
            chars_filtered = content_stats['filtered_chars_removed']

            stats_summary = f"""Content processed: {original_html_length:,} HTML chars → {initial_markdown_length:,} initial markdown → {final_filtered_length:,} filtered markdown"""
            if start_from_char > 0:
                stats_summary += f' (started from char {start_from_char:,})'
            if truncated:
                stats_summary += f' → {len(content):,} final chars (truncated, use start_from_char={content_stats["next_start_char"]} to continue)'
            elif chars_filtered > 0:
                stats_summary += f' (filtered {chars_filtered:,} chars of noise)'

            system_prompt = """
You are an expert at extracting data from the markdown of a webpage.

<input>
You will be given a query and the markdown of a webpage that has been filtered to remove noise and advertising content.
</input>

<instructions>
- You are tasked to extract information from the webpage that is relevant to the query.
- You should ONLY use the information available in the webpage to answer the query. Do not make up information or provide guess from your own knowledge.
- If the information relevant to the query is not available in the page, your response should mention that.
- If the query asks for all items, products, etc., make sure to directly list all of them.
- If the content was truncated and you need more information, note that the user can use start_from_char parameter to continue from where truncation occurred.
</instructions>

<output>
- Your output should present ALL the information relevant to the query in a concise way.
- Do not answer in conversational format - directly output the relevant information or that the information is unavailable.
</output>
""".strip()

            prompt = f'<query>\n{query}\n</query>\n\n<content_stats>\n{stats_summary}\n</content_stats>\n\n<webpage_content>\n{content}\n</webpage_content>'

            try:
                # Use page_extraction_llm if provided, otherwise fall back to simple text extraction
                if page_extraction_llm is not None:
                    from src.openbrowser.agent.views import SystemMessage, UserMessage
                    from langchain_core.messages import AIMessage
                    
                    response = await asyncio.wait_for(
                        page_extraction_llm.ainvoke([SystemMessage(content=system_prompt), UserMessage(content=prompt)]),
                        timeout=120.0,
                    )

                    # Extract content from LangChain message
                    if isinstance(response, AIMessage):
                        response_content = response.content
                    elif hasattr(response, 'content'):
                        response_content = response.content
                    elif hasattr(response, 'completion'):
                        response_content = response.completion
                    else:
                        response_content = str(response)

                    current_url = await session.get_current_page_url()
                    extracted_content = (
                        f'<url>\n{current_url}\n</url>\n<query>\n{query}\n</query>\n<result>\n{response_content}\n</result>'
                    )

                    # Simple memory handling
                    MAX_MEMORY_LENGTH = 1000
                    if len(extracted_content) < MAX_MEMORY_LENGTH:
                        memory = extracted_content
                        include_extracted_content_only_once = False
                    else:
                        if file_system is None:
                            # Fallback if file_system not provided
                            memory = f'Query: {query}\nContent too long to store in memory.'
                            include_extracted_content_only_once = True
                        else:
                            file_name = await file_system.save_extracted_content(extracted_content)
                            memory = f'Query: {query}\nContent in {file_name} and once in <read_state>.'
                            include_extracted_content_only_once = True

                    logger.info(f'📄 {memory}')
                    return ActionResult(
                        extracted_content=extracted_content,
                        include_extracted_content_only_once=include_extracted_content_only_once,
                        long_term_memory=memory,
                    )
                else:
                    # Fallback to simple extraction if no LLM provided
                    current_url = await session.get_current_page_url()
                    extracted_content = (
                        f'<url>\n{current_url}\n</url>\n<query>\n{query}\n</query>\n<result>\n{content[:5000]}\n</result>'
                    )
                    return ActionResult(
                        extracted_content=extracted_content,
                        long_term_memory=f'Extracted content for query: {query}',
                    )
            except Exception as e:
                logger.debug(f'Error extracting content: {e}')
                raise RuntimeError(str(e))

        @self.registry.action("Find text on the page and scroll to it", param_model=FindTextParams)
        async def find_text(params: FindTextParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            try:
                # Escape quotes for JavaScript
                escaped_text = params.text.replace('"', '\\"')
                script = f'''
                (function() {{
                    const text = "{escaped_text}";
                    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
                    while (walker.nextNode()) {{
                        if (walker.currentNode.textContent.includes(text)) {{
                            const range = document.createRange();
                            range.selectNodeContents(walker.currentNode);
                            const rect = range.getBoundingClientRect();
                            window.scrollTo({{ top: rect.top + window.scrollY - 100, behavior: 'smooth' }});
                            return true;
                        }}
                    }}
                    return false;
                }})()
                '''
                result = await client.send.Runtime.evaluate(
                    params={"expression": script, "returnByValue": True},
                    session_id=session_id,
                )
                found = result.get("result", {}).get("value", False)
                if found:
                    return ActionResult(extracted_content=f"Found and scrolled to text: '{params.text}'")
                return ActionResult(error=f"Text not found on page: '{params.text}'")
            except Exception as e:
                return ActionResult(error=f"Failed to find text: {e}")

        @self.registry.action("Select an option from a dropdown", param_model=SelectDropdownParams)
        async def select_dropdown(params: SelectDropdownParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            if params.index not in tools_instance._selector_map:
                return ActionResult(error=f"Element index {params.index} not found")
            backend_node_id = tools_instance._selector_map[params.index]
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            try:
                resolve_result = await client.send.DOM.resolveNode(
                    params={"backendNodeId": backend_node_id}, session_id=session_id
                )
                object_id = resolve_result.get("object", {}).get("objectId")
                if not object_id:
                    return ActionResult(error=f"Could not resolve element {params.index}")
                script = f'''
                function(value) {{
                    for (let option of this.options) {{
                        if (option.value === value || option.text === value) {{
                            option.selected = true;
                            this.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            return true;
                        }}
                    }}
                    return false;
                }}
                '''
                result = await client.send.Runtime.callFunctionOn(
                    params={
                        "objectId": object_id,
                        "functionDeclaration": script,
                        "arguments": [{"value": params.value}],
                        "returnByValue": True,
                    },
                    session_id=session_id,
                )
                success = result.get("result", {}).get("value", False)
                if success:
                    return ActionResult(extracted_content=f"Selected '{params.value}' from dropdown {params.index}")
                return ActionResult(error=f"Option '{params.value}' not found in dropdown")
            except Exception as e:
                return ActionResult(error=f"Failed to select dropdown: {e}")

        @self.registry.action("Get all options from a dropdown", param_model=DropdownOptionsParams)
        async def dropdown_options(params: DropdownOptionsParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            if params.index not in tools_instance._selector_map:
                return ActionResult(error=f"Element index {params.index} not found")
            backend_node_id = tools_instance._selector_map[params.index]
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            try:
                resolve_result = await client.send.DOM.resolveNode(
                    params={"backendNodeId": backend_node_id}, session_id=session_id
                )
                object_id = resolve_result.get("object", {}).get("objectId")
                if not object_id:
                    return ActionResult(error=f"Could not resolve element {params.index}")

                # Enhanced dropdown detection supporting native select and ARIA patterns
                result = await client.send.Runtime.callFunctionOn(
                    params={
                        "objectId": object_id,
                        "functionDeclaration": """
                        function() {
                            // Check for native <select> element
                            if (this.tagName && this.tagName.toLowerCase() === 'select') {
                                return {
                                    type: 'native',
                                    options: Array.from(this.options).map(o => ({
                                        value: o.value,
                                        text: o.text,
                                        selected: o.selected
                                    }))
                                };
                            }
                            
                            // Check for ARIA menu patterns
                            const role = this.getAttribute('role');
                            const ariaExpanded = this.getAttribute('aria-expanded');
                            const ariaHasPopup = this.getAttribute('aria-haspopup');
                            
                            // Find associated listbox/menu
                            let options = [];
                            const ariaControls = this.getAttribute('aria-controls');
                            const ariaOwns = this.getAttribute('aria-owns');
                            
                            // Look for controlled element
                            let listbox = null;
                            if (ariaControls) {
                                listbox = document.getElementById(ariaControls);
                            } else if (ariaOwns) {
                                listbox = document.getElementById(ariaOwns);
                            }
                            
                            // If no controlled element, look for nearby listbox/menu
                            if (!listbox) {
                                const parent = this.closest('[role="combobox"], [role="menu"], [role="listbox"]');
                                if (parent) {
                                    listbox = parent.querySelector('[role="listbox"], [role="menu"]') || parent;
                                }
                            }
                            
                            // Also check siblings and children
                            if (!listbox) {
                                listbox = this.querySelector('[role="listbox"], [role="menu"], ul, ol');
                            }
                            if (!listbox && this.nextElementSibling) {
                                const next = this.nextElementSibling;
                                if (next.matches('[role="listbox"], [role="menu"], ul, ol')) {
                                    listbox = next;
                                }
                            }
                            
                            if (listbox) {
                                const optionElements = listbox.querySelectorAll(
                                    '[role="option"], [role="menuitem"], [role="menuitemradio"], li'
                                );
                                optionElements.forEach((opt, idx) => {
                                    const text = opt.textContent.trim();
                                    const value = opt.getAttribute('data-value') || 
                                                  opt.getAttribute('value') || 
                                                  text;
                                    const selected = opt.getAttribute('aria-selected') === 'true' ||
                                                    opt.classList.contains('selected');
                                    if (text) {
                                        options.push({ value, text, selected, index: idx });
                                    }
                                });
                            }
                            
                            return {
                                type: role || 'aria',
                                expanded: ariaExpanded === 'true',
                                hasPopup: ariaHasPopup,
                                options: options
                            };
                        }
                        """,
                        "returnByValue": True,
                    },
                    session_id=session_id,
                )
                data = result.get("result", {}).get("value", {})
                options = data.get("options", [])
                dropdown_type = data.get("type", "unknown")

                if not options:
                    return ActionResult(
                        extracted_content=f"No options found for element [{params.index}] (type: {dropdown_type}). "
                        "If this is an ARIA dropdown, you may need to click it first to expand options."
                    )

                options_text = "\n".join([
                    f"  - {o.get('text', '')} (value: {o.get('value', '')})" +
                    (" [selected]" if o.get('selected') else "")
                    for o in options
                ])
                return ActionResult(
                    extracted_content=f"Dropdown options ({dropdown_type}):\n{options_text}",
                    metadata={"dropdown_type": dropdown_type, "options_count": len(options)}
                )
            except Exception as e:
                return ActionResult(error=f"Failed to get dropdown options: {e}")

        @self.registry.action("Switch to a different browser tab", param_model=SwitchTabParams)
        async def switch_tab(params: SwitchTabParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            try:
                from src.openbrowser.browser.events import SwitchTabEvent
                tabs = await session.get_tabs()
                if params.tab_id < 0 or params.tab_id >= len(tabs):
                    return ActionResult(error=f"Tab index {params.tab_id} out of range (0-{len(tabs)-1})")
                target_id = tabs[params.tab_id].target_id
                await session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
                return ActionResult(extracted_content=f"Switched to tab {params.tab_id}")
            except Exception as e:
                return ActionResult(error=f"Failed to switch tab: {e}")

        @self.registry.action("Close a browser tab", param_model=CloseTabParams)
        async def close_tab(params: CloseTabParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            try:
                from src.openbrowser.browser.events import CloseTabEvent
                tabs = await session.get_tabs()
                if params.tab_id is not None:
                    if params.tab_id < 0 or params.tab_id >= len(tabs):
                        return ActionResult(error=f"Tab index {params.tab_id} out of range")
                    target_id = tabs[params.tab_id].target_id
                else:
                    target_id = session.agent_focus.target_info.target_id
                await session.event_bus.dispatch(CloseTabEvent(target_id=target_id))
                return ActionResult(extracted_content=f"Closed tab {params.tab_id if params.tab_id is not None else 'current'}")
            except Exception as e:
                return ActionResult(error=f"Failed to close tab: {e}")

        @self.registry.action("Upload a file to a file input element", param_model=UploadFileParams)
        async def upload_file(params: UploadFileParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            if params.index not in tools_instance._selector_map:
                return ActionResult(error=f"Element index {params.index} not found")
            backend_node_id = tools_instance._selector_map[params.index]
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            try:
                import os
                if not os.path.exists(params.file_path):
                    return ActionResult(error=f"File not found: {params.file_path}")
                await client.send.DOM.setFileInputFiles(
                    params={"backendNodeId": backend_node_id, "files": [params.file_path]},
                    session_id=session_id,
                )
                return ActionResult(extracted_content=f"Uploaded file: {params.file_path}")
            except Exception as e:
                return ActionResult(error=f"Failed to upload file: {e}")

        @self.registry.action("Execute JavaScript code in the browser", param_model=EvaluateParams)
        async def evaluate(params: EvaluateParams, browser_session: BrowserSession = None) -> ActionResult:
            session = browser_session or tools_instance.browser_session
            if not session.agent_focus:
                raise RuntimeError("Browser not started")
            cdp_session = session.agent_focus
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id
            try:
                # Validate and fix common JavaScript issues
                fixed_script = tools_instance._validate_and_fix_javascript(params.script)

                result = await client.send.Runtime.evaluate(
                    params={"expression": fixed_script, "returnByValue": True, "awaitPromise": True},
                    session_id=session_id,
                )
                value = result.get("result", {}).get("value")
                exception = result.get("exceptionDetails")
                if exception:
                    error_msg = exception.get('text', 'Unknown error')
                    exception_obj = exception.get('exception', {})
                    if exception_obj:
                        error_msg = exception_obj.get('description', error_msg)
                    return ActionResult(error=f"JavaScript error: {error_msg}")
                return ActionResult(extracted_content=f"JavaScript result: {value}")
            except Exception as e:
                return ActionResult(error=f"Failed to evaluate JavaScript: {e}")

        @self.registry.action("Read a file from the filesystem", param_model=ReadFileParams)
        async def read_file(params: ReadFileParams, browser_session: BrowserSession = None) -> ActionResult:
            try:
                import os
                if not os.path.exists(params.file_path):
                    return ActionResult(error=f"File not found: {params.file_path}")
                with open(params.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content) > 10000:
                    content = content[:10000] + "\n... (truncated)"
                return ActionResult(
                    extracted_content=f"File content ({params.file_path}):\n{content}",
                    include_extracted_content_only_once=True,
                )
            except Exception as e:
                return ActionResult(error=f"Failed to read file: {e}")

        @self.registry.action("Write content to a file", param_model=WriteFileParams)
        async def write_file(params: WriteFileParams, browser_session: BrowserSession = None) -> ActionResult:
            try:
                import os
                os.makedirs(os.path.dirname(params.file_path) or '.', exist_ok=True)
                with open(params.file_path, 'w', encoding='utf-8') as f:
                    f.write(params.content)
                return ActionResult(extracted_content=f"Wrote {len(params.content)} characters to {params.file_path}")
            except Exception as e:
                return ActionResult(error=f"Failed to write file: {e}")

        @self.registry.action("Replace text in a file", param_model=ReplaceFileParams)
        async def replace_file(params: ReplaceFileParams, browser_session: BrowserSession = None) -> ActionResult:
            try:
                import os
                if not os.path.exists(params.file_path):
                    return ActionResult(error=f"File not found: {params.file_path}")
                with open(params.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if params.old_text not in content:
                    return ActionResult(error=f"Text '{params.old_text[:50]}...' not found in file")
                new_content = content.replace(params.old_text, params.new_text, 1)
                with open(params.file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return ActionResult(extracted_content=f"Replaced text in {params.file_path}")
            except Exception as e:
                return ActionResult(error=f"Failed to replace in file: {e}")

    async def execute_action(self, action_name: str, params: dict) -> ActionResult:
        return await self.registry.execute_action(
            action_name=action_name, params=params, browser_session=self.browser_session
        )

    def create_action_model(self, page_url: str | None = None) -> type[ActionModel]:
        return self.registry.create_action_model(page_url=page_url)

    def get_prompt_description(self, page_url: str | None = None) -> str:
        return self.registry.get_prompt_description(page_url=page_url)

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions.

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description, **kwargs)

    @observe_debug(ignore_input=True, ignore_output=True, name='act')
    async def act(
        self,
        action: ActionModel,
        browser_session: BrowserSession,
        page_extraction_llm: Any | None = None,
        sensitive_data: dict[str, str | dict[str, str]] | None = None,
        available_file_paths: list[str] | None = None,
        file_system: Any | None = None,
    ) -> ActionResult:
        """Execute an action with unified error handling."""
        from src.openbrowser.browser.views import BrowserError  # noqa: F401

        for action_name, params in action.model_dump(exclude_unset=True).items():
            if params is not None:
                try:
                    result = await self.registry.execute_action(
                        action_name=action_name,
                        params=params,
                        browser_session=browser_session,
                        page_extraction_llm=page_extraction_llm,
                        file_system=file_system,
                        sensitive_data=sensitive_data,
                        available_file_paths=available_file_paths,
                    )
                except BrowserError as e:
                    logger.error(f'❌ Action {action_name} failed with BrowserError: {str(e)}')
                    result = handle_browser_error(e)
                except TimeoutError as e:
                    logger.error(f'❌ Action {action_name} failed with TimeoutError: {str(e)}')
                    result = ActionResult(error=f'{action_name} was not executed due to timeout.')
                except Exception as e:
                    # Log the original exception with traceback for observability
                    logger.error(f"Action '{action_name}' failed with error: {str(e)}")
                    result = ActionResult(error=str(e))

                if isinstance(result, str):
                    return ActionResult(extracted_content=result)
                elif isinstance(result, ActionResult):
                    return result
                elif result is None:
                    return ActionResult()
                else:
                    raise ValueError(f'Invalid action result type: {type(result)} of {result}')
        return ActionResult()

    def __getattr__(self, name: str):
        """
        Enable direct action calls like tools.navigate(url=..., browser_session=...).
        This provides a simpler API for tests and direct usage while maintaining backward compatibility.
        """
        # Check if this is a registered action
        if name in self.registry.registry.actions:
            from typing import Union
            from pydantic import create_model

            action = self.registry.registry.actions[name]

            # Create a wrapper that calls act() to ensure consistent error handling and result normalization
            async def action_wrapper(**kwargs):
                # Extract browser_session (required positional argument for act())
                browser_session = kwargs.get('browser_session')

                # Separate action params from special params (injected dependencies)
                special_param_names = {
                    'browser_session',
                    'page_extraction_llm',
                    'file_system',
                    'available_file_paths',
                    'sensitive_data',
                }

                # Extract action params (params for the action itself)
                action_params = {k: v for k, v in kwargs.items() if k not in special_param_names}

                # Extract special params (injected dependencies) - exclude browser_session as it's positional
                special_kwargs = {k: v for k, v in kwargs.items() if k in special_param_names and k != 'browser_session'}

                # Create the param instance
                params_instance = action.param_model(**action_params)

                # Dynamically create an ActionModel with this action
                # Use Union for type compatibility with create_model
                DynamicActionModel = create_model(
                    'DynamicActionModel',
                    __base__=ActionModel,
                    **{name: (Union[action.param_model, None], None)},  # type: ignore
                )

                # Create the action model instance
                action_model = DynamicActionModel(**{name: params_instance})

                # Call act() which has all the error handling, result normalization, and observability
                # browser_session is passed as positional argument (required by act())
                return await self.act(action=action_model, browser_session=browser_session, **special_kwargs)  # type: ignore

            return action_wrapper

        # If not an action, raise AttributeError for normal Python behavior
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def handle_browser_error(e: 'BrowserError') -> ActionResult:
    """Handle BrowserError exception and convert to ActionResult."""
    if e.long_term_memory is not None:
        if e.short_term_memory is not None:
            return ActionResult(
                extracted_content=e.short_term_memory, error=e.long_term_memory, include_extracted_content_only_once=True
            )
        else:
            return ActionResult(error=e.long_term_memory)
    # Fallback to original error handling if long_term_memory is None
    logger.warning(
        '⚠️ A BrowserError was raised without long_term_memory - always set long_term_memory when raising BrowserError to propagate right messages to LLM.'
    )
    raise e


class CodeAgentTools(Tools):
    """Tools subclass for CodeAgent with optimized action exclusions."""

    def __init__(
        self,
        browser_session: BrowserSession,
        exclude_actions: list[str] | None = None,
        output_model: type[T] | None = None,
        display_files_in_done_text: bool = True,
    ):
        # Default exclusions for CodeAgent
        default_exclusions = [
            'extract',
            'find_text',
            'screenshot',
            'search',
            'write_file',
            'read_file',
            'replace_file',
        ]
        
        # Merge with user-provided exclusions
        if exclude_actions:
            combined_exclusions = list(set(default_exclusions + exclude_actions))
        else:
            combined_exclusions = default_exclusions
        
        super().__init__(
            browser_session=browser_session,
            exclude_actions=combined_exclusions,
            output_model=output_model,
            display_files_in_done_text=display_files_in_done_text,
        )

    def _register_code_use_done_action(self, output_model: type[T] | None) -> None:
        """Register done action optimized for CodeAgent."""
        # Use the same implementation as parent but with CodeAgent-specific behavior if needed
        self._register_done_action(output_model)


Controller = Tools
