"""WebSocket proxy for VNC browser viewing.

Proxies WebSocket frames between the frontend (noVNC/react-vnc) and the
local websockify process running inside the Docker container.

Flow:
  Frontend (wss://api-host/api/v1/vnc/ws?task_id=X&token=Y)
    -> API Gateway -> ALB -> FastAPI WebSocket handler
    -> ws://localhost:{port} (local websockify)
"""

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import websockets

from app.core.auth import verify_token_string
from app.core.config import settings
from app.services.vnc_service import vnc_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/vnc/ws")
async def vnc_websocket_proxy(
    websocket: WebSocket,
    task_id: str = Query(...),
    token: str = Query(default=None),
):
    """Proxy WebSocket connection to local websockify for VNC viewing.

    Query params:
        task_id: The task ID whose VNC session to connect to.
        token: JWT auth token (required when AUTH_ENABLED=true).
    """
    # 1. Authenticate
    if settings.AUTH_ENABLED:
        if not token:
            await websocket.close(code=4001, reason="Missing auth token")
            return
        try:
            await verify_token_string(token)
        except Exception:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return

    # 2. Look up VNC session
    session = await vnc_service.get_session(task_id)
    if not session:
        await websocket.accept()
        await websocket.close(code=4004, reason=f"No VNC session for task {task_id}")
        return

    # 3. Accept the frontend WebSocket with binary subprotocol (noVNC requirement)
    await websocket.accept(subprotocol="binary")

    # 4. Connect to local websockify
    local_url = f"ws://localhost:{session.websockify_port}"
    try:
        async with websockets.connect(
            local_url,
            subprotocols=["binary"],
            max_size=None,
            ping_interval=None,
        ) as ws_local:
            logger.info(
                "VNC proxy connected: task=%s, local_port=%d",
                task_id,
                session.websockify_port,
            )

            # 5. Bidirectional proxy
            async def frontend_to_local():
                """Forward frames from frontend -> websockify."""
                try:
                    while True:
                        message = await websocket.receive()
                        msg_type = message.get("type", "")
                        if msg_type == "websocket.disconnect":
                            break
                        if "bytes" in message and message["bytes"]:
                            await ws_local.send(message["bytes"])
                        elif "text" in message and message["text"]:
                            await ws_local.send(message["text"])
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    logger.debug("frontend_to_local ended: %s", e)

            async def local_to_frontend():
                """Forward frames from websockify -> frontend."""
                try:
                    async for frame in ws_local:
                        if websocket.client_state != WebSocketState.CONNECTED:
                            break
                        if isinstance(frame, bytes):
                            await websocket.send_bytes(frame)
                        else:
                            await websocket.send_text(frame)
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    logger.debug("local_to_frontend ended: %s", e)

            # Run both directions concurrently
            tasks = [
                asyncio.create_task(frontend_to_local()),
                asyncio.create_task(local_to_frontend()),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            logger.info("VNC proxy disconnected: task=%s", task_id)

    except (ConnectionRefusedError, OSError) as e:
        logger.error(
            "Cannot connect to local websockify for task %s: %s", task_id, e
        )
        try:
            await websocket.close(code=4502, reason="Cannot reach VNC server")
        except Exception:
            pass
    except Exception as e:
        logger.error("VNC proxy error for task %s: %s", task_id, e)
        try:
            await websocket.close(code=4500, reason="VNC proxy error")
        except Exception:
            pass
