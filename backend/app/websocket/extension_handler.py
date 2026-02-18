"""WebSocket handler for Chrome extension connections."""

import asyncio
import json
import logging
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ExtensionConnectionManager:
    """Manages WebSocket connections from Chrome extensions."""

    def __init__(self):
        self.active_extensions: dict[str, WebSocket] = {}
        self._message_callbacks: dict[str, Callable] = {}  # extension_id -> callback for CDP bridge
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, extension_id: str):
        """Accept a new extension WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            # Close existing connection if any
            if extension_id in self.active_extensions:
                try:
                    await self.active_extensions[extension_id].close()
                except Exception:
                    pass
            self.active_extensions[extension_id] = websocket
        logger.info(f"Extension {extension_id} connected")

    async def disconnect(self, extension_id: str):
        """Remove an extension WebSocket connection."""
        async with self._lock:
            if extension_id in self.active_extensions:
                del self.active_extensions[extension_id]
            if extension_id in self._message_callbacks:
                del self._message_callbacks[extension_id]
        logger.info(f"Extension {extension_id} disconnected")

    def is_connected(self, extension_id: str | None = None) -> bool:
        """Check if any extension (or specific one) is connected."""
        if extension_id:
            return extension_id in self.active_extensions
        return len(self.active_extensions) > 0

    def get_any_extension_id(self) -> str | None:
        """Get any connected extension ID."""
        if self.active_extensions:
            return next(iter(self.active_extensions))
        return None

    async def send_command(self, extension_id: str, command: dict[str, Any]) -> None:
        """Send a command to a specific extension."""
        websocket = self.active_extensions.get(extension_id)
        if websocket:
            try:
                await websocket.send_json(command)
            except Exception as e:
                logger.error(f"Failed to send command to extension {extension_id}: {e}")
                await self.disconnect(extension_id)

    def register_message_callback(self, extension_id: str, callback: Callable):
        """Register a callback for messages from a specific extension."""
        self._message_callbacks[extension_id] = callback

    def unregister_message_callback(self, extension_id: str):
        """Unregister message callback for an extension."""
        self._message_callbacks.pop(extension_id, None)

    async def _handle_message(self, extension_id: str, message: dict[str, Any]):
        """Route incoming extension message to registered callback."""
        callback = self._message_callbacks.get(extension_id)
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)


# Global extension manager
extension_manager = ExtensionConnectionManager()


async def handle_extension_websocket(websocket: WebSocket, extension_id: str):
    """Handle WebSocket connection from a Chrome extension."""
    await extension_manager.connect(websocket, extension_id)

    # Import here to avoid circular imports
    from app.websocket.handler import connection_manager as frontend_manager
    from app.models.schemas import WSMessage, WSMessageType

    # Notify all frontend clients that extension is connected
    await frontend_manager.broadcast(
        WSMessage(
            type=WSMessageType.EXTENSION_STATUS,
            data={"connected": True, "extension_id": extension_id},
        )
    )

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "ping":
                # Keepalive ping from extension
                await websocket.send_json({"type": "pong"})
            elif msg_type == "extension_connected":
                logger.info(f"Extension {extension_id} reported connected: {data.get('info', {})}")
            else:
                # Route to CDP bridge callback
                await extension_manager._handle_message(extension_id, data)

    except WebSocketDisconnect:
        logger.info(f"Extension {extension_id} disconnected")
    except Exception as e:
        logger.exception(f"Extension WebSocket error for {extension_id}: {e}")
    finally:
        await extension_manager.disconnect(extension_id)
        # Notify frontend clients
        await frontend_manager.broadcast(
            WSMessage(
                type=WSMessageType.EXTENSION_STATUS,
                data={"connected": False, "extension_id": extension_id},
            )
        )
