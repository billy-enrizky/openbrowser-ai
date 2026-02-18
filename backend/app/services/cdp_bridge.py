"""CDP Bridge - local WebSocket server that bridges CDPClient to Chrome extension."""

import asyncio
import json
import logging
import socket
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class CDPBridge:
    """Creates a local WebSocket server that acts as a CDP endpoint.

    BrowserSession.connect() connects to this bridge, and all CDP traffic
    is relayed through the Chrome extension's chrome.debugger API.
    """

    def __init__(self, extension_id: str):
        self.extension_id = extension_id
        self._port: int | None = None
        self._server: Any = None
        self._cdp_client_ws: WebSocketServerProtocol | None = None
        self._session_map: dict[str, str] = {}  # sessionId -> targetId
        self._reverse_session_map: dict[str, str] = {}  # targetId -> sessionId
        self._pending_responses: dict[int, asyncio.Future] = {}  # command id -> future
        self._running = False
        self._connected_event = asyncio.Event()
        self._attached_tab_id: int | None = None
        self._attached_tab_url: str = "about:blank"

    @property
    def cdp_url(self) -> str:
        """Get the WebSocket URL for this bridge."""
        if self._port is None:
            raise RuntimeError("Bridge not started")
        return f"ws://localhost:{self._port}"

    async def start(self) -> str:
        """Start the local WebSocket server and return the CDP URL."""
        from app.websocket.extension_handler import extension_manager

        self._port = _find_free_port()
        self._running = True

        # Register callback for extension messages
        extension_manager.register_message_callback(
            self.extension_id, self._on_extension_message
        )

        # Start WebSocket server
        self._server = await websockets.serve(
            self._handle_cdp_client,
            "localhost",
            self._port,
            # Allow any origin since this is local only
            origins=None,
        )

        logger.info(f"CDP Bridge started on port {self._port} for extension {self.extension_id}")
        return self.cdp_url

    async def stop(self):
        """Stop the bridge and clean up."""
        self._running = False

        from app.websocket.extension_handler import extension_manager
        extension_manager.unregister_message_callback(self.extension_id)

        if self._cdp_client_ws:
            try:
                await self._cdp_client_ws.close()
            except Exception:
                pass
            self._cdp_client_ws = None

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Cancel pending responses
        for future in self._pending_responses.values():
            if not future.done():
                future.cancel()
        self._pending_responses.clear()
        self._session_map.clear()
        self._reverse_session_map.clear()

        logger.info(f"CDP Bridge stopped for extension {self.extension_id}")

    async def _handle_cdp_client(self, websocket: WebSocketServerProtocol):
        """Handle incoming WebSocket connection from CDPClient."""
        self._cdp_client_ws = websocket
        self._connected_event.set()
        logger.info("CDPClient connected to CDP Bridge")

        try:
            async for raw_message in websocket:
                if not self._running:
                    break

                try:
                    message = json.loads(raw_message)
                    await self._on_cdp_client_message(message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from CDPClient: {raw_message[:100]}")
                except Exception as e:
                    logger.error(f"Error handling CDPClient message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("CDPClient disconnected from CDP Bridge")
        finally:
            self._cdp_client_ws = None
            self._connected_event.clear()

    async def _on_cdp_client_message(self, message: dict[str, Any]):
        """Handle a CDP message from CDPClient and relay to extension."""
        from app.websocket.extension_handler import extension_manager

        msg_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params", {})
        session_id = message.get("sessionId")

        logger.info(f"CDP <- CDPClient: id={msg_id} method={method} sessionId={str(session_id)[:8] if session_id else None}")

        # Intercept Target.setAutoAttach: respond locally since chrome.debugger
        # handles target attachment differently.
        if method == "Target.setAutoAttach":
            await self._send_to_cdp_client({"id": msg_id, "result": {}})
            return

        # Intercept Browser-domain commands: chrome.debugger doesn't support
        # the Browser domain at all. Return appropriate responses immediately
        # to prevent watchdog timeouts.
        if method.startswith("Browser."):
            # Browser.getVersion -> return synthetic info
            if method == "Browser.getVersion":
                await self._send_to_cdp_client({
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "1.3",
                        "product": "Chrome (via extension)",
                        "revision": "unknown",
                        "userAgent": "",
                        "jsVersion": "",
                    },
                })
                return
            # All other Browser.* commands -> return error immediately
            error_msg = f"'{method}' is not available through chrome.debugger extension bridge"
            logger.info(f"CDP Bridge intercepted unsupported: {method}")
            cdp_response: dict[str, Any] = {
                "id": msg_id,
                "error": {"code": -32601, "message": error_msg},
            }
            if session_id:
                cdp_response["sessionId"] = session_id
            elif self._attached_tab_id is not None:
                tab_target = f"tab-{self._attached_tab_id}"
                synthetic_session = self._reverse_session_map.get(tab_target)
                if synthetic_session:
                    cdp_response["sessionId"] = synthetic_session
            await self._send_to_cdp_client(cdp_response)
            return

        # Intercept Target.getTargetInfo: chrome.debugger returns "Not allowed"
        # for this method. Return synthetic target info for our tab.
        if method == "Target.getTargetInfo" and not session_id:
            requested_target = params.get("targetId", "")
            target_id = f"tab-{self._attached_tab_id}" if self._attached_tab_id else "tab-0"
            synthetic_response = {
                "id": msg_id,
                "result": {
                    "targetInfo": {
                        "targetId": requested_target or target_id,
                        "type": "page",
                        "title": "",
                        "url": self._attached_tab_url,
                        "attached": True,
                        "browserContextId": "default",
                    }
                },
            }
            await self._send_to_cdp_client(synthetic_response)
            return

        # Intercept Target.getTargets: chrome.debugger doesn't return the tab
        # itself as a target, so we synthesize a response including it.
        if method == "Target.getTargets" and not session_id:
            target_id = f"tab-{self._attached_tab_id}" if self._attached_tab_id else "tab-0"
            synthetic_response = {
                "id": msg_id,
                "result": {
                    "targetInfos": [
                        {
                            "targetId": target_id,
                            "type": "page",
                            "title": "",
                            "url": self._attached_tab_url,
                            "attached": True,
                            "browserContextId": "default",
                        }
                    ]
                },
            }
            await self._send_to_cdp_client(synthetic_response)
            return

        # Intercept Target.attachToTarget for the synthetic tab target:
        # Generate a fake sessionId and map it, since chrome.debugger
        # is already attached to this tab.
        if method == "Target.attachToTarget" and not session_id:
            requested_target = params.get("targetId", "")
            if requested_target.startswith("tab-"):
                import uuid
                fake_session_id = str(uuid.uuid4())
                self._session_map[fake_session_id] = requested_target
                self._reverse_session_map[requested_target] = fake_session_id
                synthetic_response = {
                    "id": msg_id,
                    "result": {"sessionId": fake_session_id},
                }
                await self._send_to_cdp_client(synthetic_response)
                # Also send a synthetic attachedToTarget event
                attached_event = {
                    "method": "Target.attachedToTarget",
                    "params": {
                        "sessionId": fake_session_id,
                        "targetInfo": {
                            "targetId": requested_target,
                            "type": "page",
                            "title": "",
                            "url": self._attached_tab_url,
                            "attached": True,
                            "browserContextId": "default",
                        },
                        "waitingForDebugger": False,
                    },
                }
                await self._send_to_cdp_client(attached_event)
                return

        # Build extension command
        ext_command: dict[str, Any] = {
            "type": "cdp_command",
            "id": msg_id,
            "method": method,
            "params": params,
        }

        # Map sessionId to targetId for the extension.
        # If the sessionId maps to our synthetic tab target (tab-*),
        # strip it so the extension uses its activeTabId.
        if session_id:
            target_id = self._session_map.get(session_id)
            if target_id and target_id.startswith("tab-"):
                # Synthetic session for the main tab -- don't send sessionId
                # to the extension; it will use activeTabId.
                pass
            elif target_id:
                ext_command["sessionId"] = session_id
                ext_command["targetId"] = target_id
            else:
                ext_command["sessionId"] = session_id

        # Send to extension
        await extension_manager.send_command(self.extension_id, ext_command)

    async def _on_extension_message(self, message: dict[str, Any]):
        """Handle a message from the extension (CDP response or event)."""
        msg_type = message.get("type", "")

        if msg_type == "cdp_response":
            await self._handle_cdp_response(message)
        elif msg_type == "cdp_event":
            await self._handle_cdp_event(message)
        elif msg_type == "tab_attached":
            self._attached_tab_id = message.get("tabId")
            logger.info(f"Tab attached: tabId={self._attached_tab_id}")
        elif msg_type == "debugger_detached":
            logger.warning(f"Debugger detached: {message.get('reason')}")

    async def _handle_cdp_response(self, message: dict[str, Any]):
        """Handle a CDP response from the extension."""
        msg_id = message.get("id")
        result = message.get("result")
        error = message.get("error")
        session_id = message.get("sessionId")

        # Log response summary for debugging
        result_summary = str(result)[:200] if result else "None"
        logger.info(f"CDP <- Extension: id={msg_id} error={error} result={result_summary}")

        # Track current tab URL from navigation history responses
        if result and isinstance(result, dict) and "entries" in result and "currentIndex" in result:
            try:
                entries = result["entries"]
                current_idx = result["currentIndex"]
                if 0 <= current_idx < len(entries):
                    self._attached_tab_url = entries[current_idx].get("url", self._attached_tab_url)
            except (IndexError, TypeError):
                pass

        # Track sessionId -> targetId from Target.attachToTarget responses
        if result and isinstance(result, dict) and "sessionId" in result:
            new_session_id = result["sessionId"]
            # The original command had params with targetId
            original_target_id = message.get("originalTargetId")
            if original_target_id:
                self._session_map[new_session_id] = original_target_id
                self._reverse_session_map[original_target_id] = new_session_id
                logger.info(
                    f"Session mapping: {new_session_id[:8]}... -> {original_target_id[:8]}..."
                )

        # Forward to CDPClient
        cdp_response: dict[str, Any] = {"id": msg_id}
        if error:
            cdp_response["error"] = {"message": error} if isinstance(error, str) else error
        else:
            cdp_response["result"] = result or {}

        # If extension didn't include sessionId but the command was for the
        # main tab (synthetic session), add the synthetic sessionId back.
        if session_id:
            cdp_response["sessionId"] = session_id
        elif self._attached_tab_id is not None:
            # Find the synthetic session for the main tab
            tab_target = f"tab-{self._attached_tab_id}"
            synthetic_session = self._reverse_session_map.get(tab_target)
            if synthetic_session:
                cdp_response["sessionId"] = synthetic_session

        await self._send_to_cdp_client(cdp_response)

    async def _handle_cdp_event(self, message: dict[str, Any]):
        """Handle a CDP event from the extension."""
        method = message.get("method", "")
        params = message.get("params", {})
        session_id = message.get("sessionId")
        target_id = message.get("targetId")

        # Track new sessions from Target.attachedToTarget events
        if method == "Target.attachedToTarget":
            event_session_id = params.get("sessionId")
            target_info = params.get("targetInfo", {})
            event_target_id = target_info.get("targetId")
            if event_session_id and event_target_id:
                self._session_map[event_session_id] = event_target_id
                self._reverse_session_map[event_target_id] = event_session_id
                logger.info(
                    f"Session mapping (event): {event_session_id[:8]}... -> {event_target_id[:8]}..."
                )

        # Clean up on detach
        elif method == "Target.detachedFromTarget":
            event_session_id = params.get("sessionId")
            event_target_id = params.get("targetId")
            if event_session_id and event_session_id in self._session_map:
                del self._session_map[event_session_id]
            if event_target_id and event_target_id in self._reverse_session_map:
                del self._reverse_session_map[event_target_id]

        # Resolve sessionId from targetId if needed
        if not session_id and target_id:
            session_id = self._reverse_session_map.get(target_id)

        # If no sessionId resolved, use the synthetic session for the main tab
        if not session_id and self._attached_tab_id is not None:
            tab_target = f"tab-{self._attached_tab_id}"
            session_id = self._reverse_session_map.get(tab_target)

        # Forward to CDPClient as raw CDP event
        cdp_event: dict[str, Any] = {
            "method": method,
            "params": params,
        }
        if session_id:
            cdp_event["sessionId"] = session_id

        await self._send_to_cdp_client(cdp_event)

    async def _send_to_cdp_client(self, message: dict[str, Any]):
        """Send a message to the connected CDPClient."""
        if self._cdp_client_ws:
            try:
                await self._cdp_client_ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send to CDPClient: {e}")
