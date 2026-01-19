"""WebSocket handler for real-time agent communication."""

import asyncio
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.models.schemas import (
    AgentType,
    FileAttachment,
    WSMessage,
    WSMessageType,
    WSOutputData,
    WSScreenshotData,
    WSStartTaskData,
    WSStepUpdateData,
    WSTaskCompletedData,
    WSLogData,
    WSVncInfoData,
)
from app.services.agent_service import agent_manager
from app.services.vnc_service import vnc_service

logger = logging.getLogger(__name__)


class WebSocketLogHandler(logging.Handler):
    """Custom logging handler that sends logs to WebSocket clients."""
    
    def __init__(self, client_id: str, task_id: str, connection_manager: "ConnectionManager"):
        super().__init__()
        self.client_id = client_id
        self.task_id = task_id
        self.connection_manager = connection_manager
        self.setLevel(logging.INFO)
        # Filter to only capture openbrowser logs
        self.addFilter(lambda record: "openbrowser" in record.name or record.name.startswith("app."))
    
    def emit(self, record: logging.LogRecord):
        try:
            # Format the log message
            message = self.format(record)
            
            # Extract step number from message if present
            step_number = None
            if "Step " in message and "/" in message:
                try:
                    import re
                    match = re.search(r"Step (\d+)/", message)
                    if match:
                        step_number = int(match.group(1))
                except (ValueError, AttributeError):
                    pass
            
            # Send log to WebSocket
            asyncio.create_task(
                self.connection_manager.send_message(
                    self.client_id,
                    WSMessage(
                        type=WSMessageType.LOG,
                        task_id=self.task_id,
                        data=WSLogData(
                            level=record.levelname.lower(),
                            message=message,
                            source=record.name,
                            step_number=step_number,
                        ).model_dump(),
                    ),
                )
            )
        except Exception:
            # Don't let logging errors break the application
            pass


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: WSMessage):
        """Send a message to a specific client."""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send_json(message.model_dump(mode="json"))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                await self.disconnect(client_id)

    async def broadcast(self, message: WSMessage):
        """Broadcast a message to all connected clients."""
        for client_id in list(self.active_connections.keys()):
            await self.send_message(client_id, message)


# Global connection manager
connection_manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket, client_id: str):
    """Handle WebSocket connection for a client."""
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = WSMessage(**data)
            
            # Handle message based on type
            if message.type == WSMessageType.START_TASK:
                await handle_start_task(client_id, message)
            elif message.type == WSMessageType.CANCEL_TASK:
                await handle_cancel_task(client_id, message)
            elif message.type == WSMessageType.PAUSE_TASK:
                await handle_pause_task(client_id, message)
            elif message.type == WSMessageType.RESUME_TASK:
                await handle_resume_task(client_id, message)
            elif message.type == WSMessageType.REQUEST_VNC:
                await handle_request_vnc(client_id, message)
            else:
                logger.warning(f"Unknown message type: {message.type}")
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error for {client_id}: {e}")
    finally:
        await connection_manager.disconnect(client_id)


async def handle_start_task(client_id: str, message: WSMessage):
    """Handle START_TASK message."""
    try:
        task_data = WSStartTaskData(**message.data)
        
        # Generate task_id early so we can use it for logging
        from uuid import uuid4
        task_id = str(uuid4())
        
        # Set up WebSocket log handler for this task
        ws_log_handler = WebSocketLogHandler(client_id, task_id, connection_manager)
        ws_log_handler.setFormatter(logging.Formatter("%(message)s"))
        
        # Add handler to openbrowser loggers
        openbrowser_logger = logging.getLogger("openbrowser")
        openbrowser_logger.addHandler(ws_log_handler)
        
        # Also add to app logger
        app_logger = logging.getLogger("app")
        app_logger.addHandler(ws_log_handler)
        
        # Create callbacks for real-time updates
        def on_step(step_data: dict[str, Any]):
            asyncio.create_task(
                connection_manager.send_message(
                    client_id,
                    WSMessage(
                        type=WSMessageType.STEP_UPDATE,
                        task_id=task_id,
                        data=WSStepUpdateData(**step_data).model_dump(),
                    ),
                )
            )
        
        def on_output(content: str, is_final: bool):
            asyncio.create_task(
                connection_manager.send_message(
                    client_id,
                    WSMessage(
                        type=WSMessageType.OUTPUT,
                        task_id=task_id,
                        data=WSOutputData(content=content, is_final=is_final).model_dump(),
                    ),
                )
            )
        
        def on_screenshot(screenshot_base64: str | None, step_number: int):
            if screenshot_base64:
                asyncio.create_task(
                    connection_manager.send_message(
                        client_id,
                        WSMessage(
                            type=WSMessageType.SCREENSHOT,
                            task_id=task_id,
                            data=WSScreenshotData(
                                base64=screenshot_base64,
                                step_number=step_number,
                            ).model_dump(),
                        ),
                    )
                )
        
        def on_thinking(thinking: str):
            asyncio.create_task(
                connection_manager.send_message(
                    client_id,
                    WSMessage(
                        type=WSMessageType.THINKING,
                        task_id=task_id,
                        data={"thinking": thinking},
                    ),
                )
            )
        
        def on_error(error: str):
            asyncio.create_task(
                connection_manager.send_message(
                    client_id,
                    WSMessage(
                        type=WSMessageType.ERROR,
                        task_id=task_id,
                        data={"error": error},
                    ),
                )
            )
        
        def on_log(level: str, message_text: str, source: str | None = None, step_number: int | None = None):
            """Send log message to frontend."""
            asyncio.create_task(
                connection_manager.send_message(
                    client_id,
                    WSMessage(
                        type=WSMessageType.LOG,
                        task_id=task_id,
                        data=WSLogData(
                            level=level,
                            message=message_text,
                            source=source,
                            step_number=step_number,
                        ).model_dump(),
                    ),
                )
            )
        
        def on_vnc_info(vnc_data: dict[str, Any]):
            """Send VNC connection info to frontend."""
            asyncio.create_task(
                connection_manager.send_message(
                    client_id,
                    WSMessage(
                        type=WSMessageType.VNC_INFO,
                        task_id=task_id,
                        data=WSVncInfoData(**vnc_data).model_dump(),
                    ),
                )
            )
        
        # Create agent session with pre-generated task_id
        session = await agent_manager.create_session_with_id(
            task_id=task_id,
            task=task_data.task,
            agent_type=task_data.agent_type.value,
            max_steps=task_data.max_steps,
            use_vision=task_data.use_vision,
            llm_model=task_data.llm_model,
            on_step_callback=on_step,
            on_output_callback=on_output,
            on_screenshot_callback=on_screenshot,
            on_thinking_callback=on_thinking,
            on_error_callback=on_error,
            on_log_callback=on_log,
            on_vnc_info_callback=on_vnc_info,
        )
        
        # Send task started message
        await connection_manager.send_message(
            client_id,
            WSMessage(
                type=WSMessageType.TASK_STARTED,
                task_id=session.task_id,
                data={"task": task_data.task, "agent_type": task_data.agent_type.value},
            ),
        )
        
        # Run agent in background
        asyncio.create_task(run_agent_task(client_id, session, ws_log_handler))
        
    except Exception as e:
        logger.exception(f"Failed to start task: {e}")
        await connection_manager.send_message(
            client_id,
            WSMessage(
                type=WSMessageType.TASK_FAILED,
                task_id=message.task_id,
                data={"error": str(e)},
            ),
        )


async def run_agent_task(client_id: str, session, ws_log_handler: WebSocketLogHandler | None = None):
    """Run the agent task and send completion message."""
    try:
        result = await session.start()
        
        # Convert attachments to FileAttachment objects
        raw_attachments = result.get("attachments", [])
        attachments = []
        for att in raw_attachments:
            if isinstance(att, dict):
                attachments.append(FileAttachment(
                    name=att.get("name", "file"),
                    content=att.get("content"),
                    url=att.get("url"),
                    type=att.get("type"),
                    mime_type=att.get("mime_type"),
                    size=att.get("size"),
                ))
            elif isinstance(att, str):
                # Legacy format: just a filename
                attachments.append(FileAttachment(
                    name=att.split("/")[-1],
                    url=att,
                ))
        
        # Send completion message
        await connection_manager.send_message(
            client_id,
            WSMessage(
                type=WSMessageType.TASK_COMPLETED,
                task_id=session.task_id,
                data=WSTaskCompletedData(
                    result=result.get("result", ""),
                    success=result.get("success", False),
                    total_steps=result.get("total_steps", 0),
                    duration_seconds=result.get("duration_seconds", 0),
                    attachments=attachments,
                ).model_dump(),
            ),
        )
    except asyncio.CancelledError:
        await connection_manager.send_message(
            client_id,
            WSMessage(
                type=WSMessageType.TASK_CANCELLED,
                task_id=session.task_id,
                data={"reason": "Task was cancelled"},
            ),
        )
    except Exception as e:
        logger.exception(f"Task failed: {e}")
        await connection_manager.send_message(
            client_id,
            WSMessage(
                type=WSMessageType.TASK_FAILED,
                task_id=session.task_id,
                data={"error": str(e)},
            ),
        )
    finally:
        # Remove WebSocket log handler
        if ws_log_handler:
            openbrowser_logger = logging.getLogger("openbrowser")
            openbrowser_logger.removeHandler(ws_log_handler)
            app_logger = logging.getLogger("app")
            app_logger.removeHandler(ws_log_handler)
        
        # Clean up session after a delay
        await asyncio.sleep(60)  # Keep session for 1 minute for potential follow-ups
        await agent_manager.remove_session(session.task_id)


async def handle_cancel_task(client_id: str, message: WSMessage):
    """Handle CANCEL_TASK message."""
    if message.task_id:
        session = await agent_manager.get_session(message.task_id)
        if session:
            await session.cancel()
            await connection_manager.send_message(
                client_id,
                WSMessage(
                    type=WSMessageType.TASK_CANCELLED,
                    task_id=message.task_id,
                    data={"reason": "Cancelled by user"},
                ),
            )


async def handle_pause_task(client_id: str, message: WSMessage):
    """Handle PAUSE_TASK message."""
    if message.task_id:
        session = await agent_manager.get_session(message.task_id)
        if session:
            await session.pause()


async def handle_resume_task(client_id: str, message: WSMessage):
    """Handle RESUME_TASK message."""
    if message.task_id:
        session = await agent_manager.get_session(message.task_id)
        if session:
            await session.resume()


async def handle_request_vnc(client_id: str, message: WSMessage):
    """Handle REQUEST_VNC message - send VNC info for an existing task."""
    if message.task_id:
        # Get VNC session info
        vnc_session = await vnc_service.get_session(message.task_id)
        if vnc_session:
            await connection_manager.send_message(
                client_id,
                WSMessage(
                    type=WSMessageType.VNC_INFO,
                    task_id=message.task_id,
                    data=WSVncInfoData(**vnc_session.to_dict()).model_dump(),
                ),
            )
        else:
            await connection_manager.send_message(
                client_id,
                WSMessage(
                    type=WSMessageType.ERROR,
                    task_id=message.task_id,
                    data={"error": "No VNC session available for this task"},
                ),
            )

