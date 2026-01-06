"""MCP Server for openbrowser-ai - exposes browser automation capabilities via Model Context Protocol.

This server provides tools for:
- Running autonomous browser tasks with an AI agent
- Direct browser control (navigation, clicking, typing, etc.)
- Content extraction from web pages

Usage:
    python -m src.openbrowser.mcp

Or as an MCP server in Claude Desktop or other MCP clients:
    {
        "mcpServers": {
            "openbrowser-ai": {
                "command": "python",
                "args": ["-m", "openbrowser.mcp"],
                "env": {
                    "OPENAI_API_KEY": "sk-..."
                }
            }
        }
    }
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Configure logging for MCP mode - redirect to stderr
logging.basicConfig(
    stream=sys.stderr, level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True
)

logger = logging.getLogger(__name__)

# Try to import MCP SDK
try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.error("MCP SDK not installed. Install with: pip install mcp")


class OpenBrowserServer:
    """MCP Server for openbrowser-ai capabilities.
    
    This class implements a Model Context Protocol (MCP) server that exposes
    browser automation capabilities to MCP clients like Claude Desktop.
    
    The server provides tools for:
        - Browser navigation and interaction (navigate, click, type, scroll)
        - Page state inspection (get DOM elements, take screenshots)
        - Tab management (list, switch, close tabs)
        - Autonomous agent execution (run_browser_agent)
        - Session management (multiple concurrent sessions)
    
    Attributes:
        server: The MCP Server instance.
        browser_session: Currently active browser session.
        tools: Browser tools instance for direct browser control.
        active_sessions: Dict mapping session IDs to session data.
        session_timeout_minutes: Timeout for inactive sessions.
    
    Example:
        >>> server = OpenBrowserServer(session_timeout_minutes=15)
        >>> await server.run()  # Start the MCP server
    
    Note:
        Requires the MCP SDK to be installed: pip install mcp
    """

    def __init__(self, session_timeout_minutes: int = 10):
        """Initialize the OpenBrowser MCP server.
        
        Args:
            session_timeout_minutes: Time in minutes before inactive sessions
                are automatically cleaned up. Defaults to 10 minutes.
        
        Raises:
            ImportError: If the MCP SDK is not installed.
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not installed. Install with: pip install mcp")

        self.server = Server("openbrowser-ai")
        self.browser_session = None
        self.tools = None
        self._start_time = time.time()

        # Session management
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.session_timeout_minutes = session_timeout_minutes
        self._cleanup_task: Any = None

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP server handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List all available openbrowser-ai tools."""
            return [
                # Browser navigation
                types.Tool(
                    name="browser_navigate",
                    description="Navigate to a URL in the browser",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to navigate to"},
                            "new_tab": {"type": "boolean", "description": "Whether to open in a new tab", "default": False},
                        },
                        "required": ["url"],
                    },
                ),
                types.Tool(
                    name="browser_click",
                    description="Click an element on the page by its index",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "The index of the element to click (from browser_get_state)",
                            },
                        },
                        "required": ["index"],
                    },
                ),
                types.Tool(
                    name="browser_type",
                    description="Type text into an input field",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "The index of the input element (from browser_get_state)",
                            },
                            "text": {"type": "string", "description": "The text to type"},
                        },
                        "required": ["index", "text"],
                    },
                ),
                types.Tool(
                    name="browser_get_state",
                    description="Get the current state of the page including all interactive elements",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_screenshot": {
                                "type": "boolean",
                                "description": "Whether to include a screenshot of the current page",
                                "default": False,
                            }
                        },
                    },
                ),
                types.Tool(
                    name="browser_scroll",
                    description="Scroll the page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["up", "down"],
                                "description": "Direction to scroll",
                                "default": "down",
                            }
                        },
                    },
                ),
                types.Tool(
                    name="browser_go_back",
                    description="Go back to the previous page",
                    inputSchema={"type": "object", "properties": {}},
                ),
                # Tab management
                types.Tool(
                    name="browser_list_tabs", description="List all open tabs", inputSchema={"type": "object", "properties": {}}
                ),
                types.Tool(
                    name="browser_switch_tab",
                    description="Switch to a different tab",
                    inputSchema={
                        "type": "object",
                        "properties": {"tab_id": {"type": "string", "description": "Tab ID to switch to"}},
                        "required": ["tab_id"],
                    },
                ),
                types.Tool(
                    name="browser_close_tab",
                    description="Close a tab",
                    inputSchema={
                        "type": "object",
                        "properties": {"tab_id": {"type": "string", "description": "Tab ID to close"}},
                        "required": ["tab_id"],
                    },
                ),
                # Agent-based task execution
                types.Tool(
                    name="run_browser_agent",
                    description="Run an autonomous browser agent to complete a task",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task description for the browser agent",
                            },
                            "max_steps": {
                                "type": "integer",
                                "description": "Maximum number of steps",
                                "default": 50,
                            },
                            "use_vision": {
                                "type": "boolean",
                                "description": "Whether to use vision capabilities",
                                "default": True,
                            },
                        },
                        "required": ["task"],
                    },
                ),
                # Session management
                types.Tool(
                    name="browser_list_sessions",
                    description="List all active browser sessions",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="browser_close_all",
                    description="Close all active browser sessions",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.list_resources()
        async def handle_list_resources() -> list[types.Resource]:
            """List available resources."""
            return []

        @self.server.list_prompts()
        async def handle_list_prompts() -> list[types.Prompt]:
            """List available prompts."""
            return []

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
            """Handle tool execution."""
            try:
                result = await self._execute_tool(name, arguments or {})
                return [types.TextContent(type="text", text=result)]
            except Exception as e:
                logger.error(f"Tool execution failed: {e}", exc_info=True)
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute an openbrowser-ai tool."""

        # Agent-based tools
        if tool_name == "run_browser_agent":
            return await self._run_browser_agent(
                task=arguments["task"],
                max_steps=arguments.get("max_steps", 50),
                use_vision=arguments.get("use_vision", True),
            )

        # Session management tools
        if tool_name == "browser_list_sessions":
            return await self._list_sessions()

        if tool_name == "browser_close_all":
            return await self._close_all_sessions()

        # Direct browser control tools
        if tool_name.startswith("browser_"):
            # Ensure browser session exists
            if not self.browser_session:
                await self._init_browser_session()

            if tool_name == "browser_navigate":
                return await self._navigate(arguments["url"], arguments.get("new_tab", False))

            elif tool_name == "browser_click":
                return await self._click(arguments["index"])

            elif tool_name == "browser_type":
                return await self._type_text(arguments["index"], arguments["text"])

            elif tool_name == "browser_get_state":
                return await self._get_browser_state(arguments.get("include_screenshot", False))

            elif tool_name == "browser_scroll":
                return await self._scroll(arguments.get("direction", "down"))

            elif tool_name == "browser_go_back":
                return await self._go_back()

            elif tool_name == "browser_list_tabs":
                return await self._list_tabs()

            elif tool_name == "browser_switch_tab":
                return await self._switch_tab(arguments["tab_id"])

            elif tool_name == "browser_close_tab":
                return await self._close_tab(arguments["tab_id"])

        return f"Unknown tool: {tool_name}"

    async def _init_browser_session(self):
        """Initialize browser session."""
        if self.browser_session:
            return

        logger.debug("Initializing browser session...")

        from openbrowser.browser.profile import BrowserProfile
        from openbrowser.browser.session import BrowserSession
        from openbrowser.tools.actions import Tools

        # Create browser profile
        profile = BrowserProfile(
            headless=False,
            downloads_path=str(Path.home() / "Downloads" / "openbrowser-ai-mcp"),
        )

        # Create browser session
        self.browser_session = BrowserSession(browser_profile=profile)
        await self.browser_session.start()

        # Track the session
        self._track_session(self.browser_session)

        # Create tools
        self.tools = Tools(self.browser_session)

        logger.debug("Browser session initialized")

    async def _run_browser_agent(
        self,
        task: str,
        max_steps: int = 50,
        use_vision: bool = True,
    ) -> str:
        """Run an autonomous browser agent task."""
        logger.debug(f"Running agent task: {task}")

        from openbrowser.agent.graph import BrowserAgent

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY not set in environment"

        try:
            # Create and run agent
            agent = BrowserAgent(
                headless=False,
                model_name="gpt-4o",
                llm_provider="openai",
                api_key=api_key,
                use_vision=use_vision,
                max_steps=max_steps,
            )

            history = await agent.run(goal=task, max_steps=max_steps)

            # Format results
            results = []
            results.append(f"Task completed in {len(history.history)} steps")
            results.append(f"Success: {history.is_successful()}")

            # Get final result if available
            final_result = history.final_result()
            if final_result:
                results.append(f"\nFinal result:\n{final_result}")

            # Include any errors
            errors = history.errors()
            if any(errors):
                results.append(f"\nErrors encountered:\n{json.dumps([e for e in errors if e], indent=2)}")

            return "\n".join(results)

        except Exception as e:
            logger.error(f"Agent task failed: {e}", exc_info=True)
            return f"Agent task failed: {str(e)}"

    async def _navigate(self, url: str, new_tab: bool = False) -> str:
        """Navigate to a URL."""
        if not self.browser_session:
            return "Error: No browser session active"

        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            await client.send.Page.navigate(params={"url": url}, session_id=session_id)
            await asyncio.sleep(1)  # Wait for navigation

            return f"Navigated to: {url}"
        except Exception as e:
            return f"Navigation failed: {str(e)}"

    async def _click(self, index: int) -> str:
        """Click an element by index."""
        if not self.browser_session or not self.tools:
            return "Error: No browser session active"

        try:
            if index not in self.tools._selector_map:
                return f"Element with index {index} not found"

            backend_node_id = self.tools._selector_map[index]
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            # Get element box model
            result = await client.send.DOM.getBoxModel(params={"backendNodeId": backend_node_id}, session_id=session_id)
            content = result.get("model", {}).get("content", [])

            if len(content) >= 4:
                x = (content[0] + content[2] + content[4] + content[6]) / 4
                y = (content[1] + content[3] + content[5] + content[7]) / 4

                # Click at center
                await client.send.Input.dispatchMouseEvent(
                    params={"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1},
                    session_id=session_id,
                )
                await client.send.Input.dispatchMouseEvent(
                    params={"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1},
                    session_id=session_id,
                )

                return f"Clicked element {index}"

            return f"Could not get element position for index {index}"
        except Exception as e:
            return f"Click failed: {str(e)}"

    async def _type_text(self, index: int, text: str) -> str:
        """Type text into an element."""
        if not self.browser_session or not self.tools:
            return "Error: No browser session active"

        try:
            # First click to focus
            await self._click(index)
            await asyncio.sleep(0.1)

            cdp_session = await self.browser_session.get_or_create_cdp_session()
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            # Type the text
            for char in text:
                await client.send.Input.dispatchKeyEvent(params={"type": "char", "text": char}, session_id=session_id)

            return f"Typed '{text}' into element {index}"
        except Exception as e:
            return f"Type failed: {str(e)}"

    async def _get_browser_state(self, include_screenshot: bool = False) -> str:
        """Get current browser state."""
        if not self.browser_session:
            return "Error: No browser session active"

        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            # Get URL
            result = await client.send.Runtime.evaluate(
                params={"expression": "window.location.href", "returnByValue": True},
                session_id=session_id,
            )
            url = result.get("result", {}).get("value", "")

            # Get title
            result = await client.send.Runtime.evaluate(
                params={"expression": "document.title", "returnByValue": True},
                session_id=session_id,
            )
            title = result.get("result", {}).get("value", "")

            state = {
                "url": url,
                "title": title,
                "interactive_elements": [],
            }

            # Add interactive elements if tools available
            if self.tools and self.tools._selector_map:
                for index in list(self.tools._selector_map.keys())[:50]:  # Limit to 50 elements
                    state["interactive_elements"].append({"index": index})

            if include_screenshot:
                try:
                    result = await client.send.Page.captureScreenshot(
                        params={"format": "jpeg", "quality": 70},
                        session_id=session_id,
                    )
                    state["screenshot"] = result.get("data")
                except Exception:
                    pass

            return json.dumps(state, indent=2)
        except Exception as e:
            return f"Failed to get browser state: {str(e)}"

    async def _scroll(self, direction: str = "down") -> str:
        """Scroll the page."""
        if not self.browser_session:
            return "Error: No browser session active"

        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            delta_y = 500 if direction == "down" else -500
            await client.send.Input.dispatchMouseEvent(
                params={"type": "mouseWheel", "x": 400, "y": 400, "deltaX": 0, "deltaY": delta_y},
                session_id=session_id,
            )

            return f"Scrolled {direction}"
        except Exception as e:
            return f"Scroll failed: {str(e)}"

    async def _go_back(self) -> str:
        """Go back in browser history."""
        if not self.browser_session:
            return "Error: No browser session active"

        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            await client.send.Page.navigateToHistoryEntry(params={"entryId": -1}, session_id=session_id)
            return "Navigated back"
        except Exception as e:
            return f"Go back failed: {str(e)}"

    async def _list_tabs(self) -> str:
        """List all open tabs."""
        if not self.browser_session:
            return "Error: No browser session active"

        try:
            targets = await self.browser_session.get_all_targets()
            tabs = []
            for target in targets:
                if target.get("type") == "page":
                    tabs.append({"id": target.get("targetId", "")[-4:], "url": target.get("url", ""), "title": target.get("title", "")})
            return json.dumps(tabs, indent=2)
        except Exception as e:
            return f"Failed to list tabs: {str(e)}"

    async def _switch_tab(self, tab_id: str) -> str:
        """Switch to a different tab."""
        if not self.browser_session:
            return "Error: No browser session active"

        try:
            targets = await self.browser_session.get_all_targets()
            for target in targets:
                if target.get("targetId", "").endswith(tab_id):
                    await self.browser_session.set_active_target(target.get("targetId"))
                    return f"Switched to tab {tab_id}"
            return f"Tab {tab_id} not found"
        except Exception as e:
            return f"Switch tab failed: {str(e)}"

    async def _close_tab(self, tab_id: str) -> str:
        """Close a specific tab."""
        if not self.browser_session:
            return "Error: No browser session active"

        try:
            targets = await self.browser_session.get_all_targets()
            for target in targets:
                if target.get("targetId", "").endswith(tab_id):
                    cdp_client = self.browser_session.cdp_client
                    await cdp_client.send.Target.closeTarget(params={"targetId": target.get("targetId")})
                    return f"Closed tab {tab_id}"
            return f"Tab {tab_id} not found"
        except Exception as e:
            return f"Close tab failed: {str(e)}"

    def _track_session(self, session) -> None:
        """Track a browser session for management."""
        self.active_sessions[session.id] = {
            "session": session,
            "created_at": time.time(),
            "last_activity": time.time(),
        }

    async def _list_sessions(self) -> str:
        """List all active browser sessions."""
        if not self.active_sessions:
            return "No active browser sessions"

        sessions_info = []
        for session_id, session_data in self.active_sessions.items():
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_data["created_at"]))
            last_activity = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_data["last_activity"]))

            sessions_info.append(
                {
                    "session_id": session_id,
                    "created_at": created_at,
                    "last_activity": last_activity,
                    "age_minutes": (time.time() - session_data["created_at"]) / 60,
                }
            )

        return json.dumps(sessions_info, indent=2)

    async def _close_all_sessions(self) -> str:
        """Close all active browser sessions."""
        if not self.active_sessions:
            return "No active sessions to close"

        closed_count = 0
        for session_id in list(self.active_sessions.keys()):
            try:
                session_data = self.active_sessions[session_id]
                session = session_data["session"]
                if hasattr(session, "stop"):
                    await session.stop()
                del self.active_sessions[session_id]
                closed_count += 1
            except Exception as e:
                logger.warning(f"Failed to close session {session_id}: {e}")

        # Clear current session
        self.browser_session = None
        self.tools = None

        return f"Closed {closed_count} sessions"

    async def run(self):
        """Run the MCP server using stdio transport.
        
        This method starts the MCP server and handles incoming requests
        from MCP clients. It runs until the connection is closed.
        
        The server uses stdin/stdout for communication, making it suitable
        for use with MCP clients like Claude Desktop that spawn the server
        as a subprocess.
        """
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="openbrowser-ai",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main(session_timeout_minutes: int = 10):
    """Main entry point for the openbrowser-ai MCP server.
    
    This function initializes and runs the openbrowser-ai MCP server,
    which exposes browser automation capabilities via the Model Context Protocol.
    
    Args:
        session_timeout_minutes: Time in minutes before inactive browser
            sessions are automatically cleaned up. Defaults to 10.
    
    Example:
        Run as a module:
            python -m src.openbrowser.mcp
        
        Or in code:
            asyncio.run(main(session_timeout_minutes=15))
    """
    if not MCP_AVAILABLE:
        print("MCP SDK is required. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = OpenBrowserServer(session_timeout_minutes=session_timeout_minutes)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())

