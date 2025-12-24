"""Browser action tools for LangChain integration."""

import logging
from typing import Optional

from cdp_use.client import CDPClient
from langchain_core.tools import StructuredTool

from src.openbrowser.browser.dom import DomState
from src.openbrowser.browser.manager import BrowserManager

logger = logging.getLogger(__name__)


class BrowserToolKit:
    """Toolkit for browser actions with LangChain integration.
    
    This class provides methods for browser interactions (click, type, navigate)
    and wraps them as LangChain tools for use with agents.
    """

    def __init__(self, browser_manager: BrowserManager):
        """Initialize BrowserToolKit.
        
        Args:
            browser_manager: BrowserManager instance for CDP operations
        """
        self.browser_manager = browser_manager
        self._selector_map: dict[int, int] = {}

    def update_state(self, dom_state: DomState) -> None:
        """Update the stored selector map with latest DOM state.
        
        Args:
            dom_state: DomState containing the selector_map to store
        """
        self._selector_map = dom_state.selector_map
        logger.info(f"Updated selector map with {len(self._selector_map)} elements")

    async def navigate(
        self,
        url: str,
        client: Optional[CDPClient] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Navigate to a URL.
        
        Args:
            url: URL to navigate to
            client: Optional CDP client. If not provided, creates temporary session
            session_id: Optional session ID. Must be provided with client
            
        Returns:
            Success message
            
        Raises:
            RuntimeError: If browser is not started or navigation fails
        """
        if self.browser_manager._cdp_url is None:
            raise RuntimeError("Browser not started. Call browser_manager.start() first.")

        # Validate parameters
        if session_id is not None and client is None:
            raise RuntimeError(
                "session_id provided without client. Provide both or neither."
            )

        # Create temporary session if needed
        if client is None:
            client, session_id = await self.browser_manager.get_session()
            should_close = True
        else:
            should_close = False
            if session_id is None:
                raise RuntimeError(
                    "client provided without session_id. Provide both or neither."
                )

        try:
            logger.info(f"Navigating to {url}")

            # Enable Page domain
            try:
                await client.send.Page.enable(session_id=session_id)
            except Exception:
                # Domain might already be enabled
                pass

            # Navigate
            await client.send.Page.navigate(
                params={"url": url}, session_id=session_id
            )

            logger.info(f"Navigation initiated to {url}")
            return f"Navigated to {url}"

        finally:
            if should_close:
                await client.stop()

    async def click_element(
        self,
        index: int,
        client: Optional[CDPClient] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Click an element by its index.
        
        Args:
            index: Element index from selector_map
            client: Optional CDP client. If not provided, creates temporary session
            session_id: Optional session ID. Must be provided with client
            
        Returns:
            Success message
            
        Raises:
            ValueError: If index not found in selector_map
            RuntimeError: If element is not visible or click fails
        """
        # Validate index exists
        if index not in self._selector_map:
            raise ValueError(
                f"Element index {index} not found in selector_map. "
                f"Available indices: {list(self._selector_map.keys())}"
            )

        backend_node_id = self._selector_map[index]
        logger.info(f"Clicking element {index} (backend_node_id: {backend_node_id})")

        if self.browser_manager._cdp_url is None:
            raise RuntimeError("Browser not started. Call browser_manager.start() first.")

        # Validate parameters
        if session_id is not None and client is None:
            raise RuntimeError(
                "session_id provided without client. Provide both or neither."
            )

        # Create temporary session if needed
        if client is None:
            client, session_id = await self.browser_manager.get_session()
            should_close = True
        else:
            should_close = False
            if session_id is None:
                raise RuntimeError(
                    "client provided without session_id. Provide both or neither."
                )

        try:
            # Enable required domains
            try:
                await client.send.DOM.enable(session_id=session_id)
                await client.send.Input.enable(session_id=session_id)
            except Exception:
                # Domains might already be enabled
                pass

            # Get element box model
            try:
                box_model_result = await client.send.DOM.getBoxModel(
                    params={"backendNodeId": backend_node_id}, session_id=session_id
                )
            except Exception as e:
                raise RuntimeError(
                    f"Element {index} is not visible. DOM.getBoxModel failed: {e}"
                ) from e

            # Extract content quad and calculate center
            if "model" not in box_model_result or "content" not in box_model_result["model"]:
                raise RuntimeError(f"Element {index} is not visible. Invalid box model response.")

            content_quad = box_model_result["model"]["content"]
            if len(content_quad) < 8:
                raise RuntimeError(f"Element {index} is not visible. Invalid content quad.")

            # Calculate center coordinates
            # Quad format: [x1, y1, x2, y2, x3, y3, x4, y4]
            x = (content_quad[0] + content_quad[2] + content_quad[4] + content_quad[6]) / 4
            y = (content_quad[1] + content_quad[3] + content_quad[5] + content_quad[7]) / 4

            logger.info(f"Clicking at coordinates ({x:.1f}, {y:.1f})")

            # Dispatch mouse press event
            await client.send.Input.dispatchMouseEvent(
                params={
                    "type": "mousePressed",
                    "button": "left",
                    "x": x,
                    "y": y,
                    "clickCount": 1,
                },
                session_id=session_id,
            )

            # Dispatch mouse release event
            await client.send.Input.dispatchMouseEvent(
                params={
                    "type": "mouseReleased",
                    "button": "left",
                    "x": x,
                    "y": y,
                    "clickCount": 1,
                },
                session_id=session_id,
            )

            logger.info(f"Successfully clicked element {index}")
            return f"Clicked element {index}"

        finally:
            if should_close:
                await client.stop()

    async def type_text(
        self,
        index: int,
        text: str,
        client: Optional[CDPClient] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Type text into an element by its index.
        
        First clicks the element to focus it, then types each character.
        
        Args:
            index: Element index from selector_map
            text: Text to type
            client: Optional CDP client. If not provided, creates temporary session
            session_id: Optional session ID. Must be provided with client
            
        Returns:
            Success message
            
        Raises:
            ValueError: If index not found in selector_map
            RuntimeError: If element is not visible or typing fails
        """
        logger.info(f"Typing text into element {index}: {text[:50]}...")

        if self.browser_manager._cdp_url is None:
            raise RuntimeError("Browser not started. Call browser_manager.start() first.")

        # Validate parameters
        if session_id is not None and client is None:
            raise RuntimeError(
                "session_id provided without client. Provide both or neither."
            )

        # Create temporary session if needed (before calling click_element to reuse it)
        if client is None:
            client, session_id = await self.browser_manager.get_session()
            should_close = True
        else:
            should_close = False
            if session_id is None:
                raise RuntimeError(
                    "client provided without session_id. Provide both or neither."
                )

        try:
            # First click to focus (reusing the session)
            await self.click_element(index, client=client, session_id=session_id)

            # Enable Input domain
            try:
                await client.send.Input.enable(session_id=session_id)
            except Exception:
                # Domain might already be enabled
                pass

            # Type each character
            for char in text:
                await client.send.Input.dispatchKeyEvent(
                    params={"type": "char", "text": char}, session_id=session_id
                )

            logger.info(f"Successfully typed text into element {index}")
            return f"Typed text into element {index}"

        finally:
            if should_close:
                await client.stop()

    def get_tools(self) -> list[StructuredTool]:
        """Get LangChain tool wrappers for all browser actions.
        
        Returns:
            List of StructuredTool instances for navigate, click_element, and type_text
        """
        # Create tool wrappers that handle async execution
        async def navigate_tool(url: str) -> str:
            """Navigate to a URL.
            
            Args:
                url: The URL to navigate to
            """
            return await self.navigate(url)

        async def click_element_tool(index: int) -> str:
            """Click an element by its index.
            
            Args:
                index: The element index from the selector map
            """
            return await self.click_element(index)

        async def type_text_tool(index: int, text: str) -> str:
            """Type text into an element by its index.
            
            Args:
                index: The element index from the selector map
                text: The text to type
            """
            return await self.type_text(index, text)

        # Wrap async functions as LangChain tools
        # LangChain tools can handle async functions, but we need to ensure
        # they're properly wrapped
        tools = [
            StructuredTool.from_function(
                func=navigate_tool,
                name="navigate",
                description="Navigate to a URL. Use this to go to a webpage.",
            ),
            StructuredTool.from_function(
                func=click_element_tool,
                name="click_element",
                description=(
                    "Click an element by its index. The index corresponds to "
                    "an interactive element from the DOM state. Use this to click "
                    "buttons, links, or other clickable elements."
                ),
            ),
            StructuredTool.from_function(
                func=type_text_tool,
                name="type_text",
                description=(
                    "Type text into an input element by its index. The element "
                    "will be clicked first to focus it, then the text will be "
                    "typed character by character. Use this to fill in text fields."
                ),
            ),
        ]

        return tools

