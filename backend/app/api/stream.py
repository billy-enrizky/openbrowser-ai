"""SSE streaming API for real-time task events.

Replaces WebSocket for production where API Gateway HTTP API
does not support WebSocket protocol upgrade.

Flow:
  1. POST /api/v1/tasks/start  -> creates task, returns {task_id}
  2. GET  /api/v1/tasks/{task_id}/stream  -> SSE event stream
  3. POST /api/v1/tasks/{task_id}/cancel  -> cancel running task
"""

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from app.core.auth import AuthPrincipal, get_current_user, verify_token_string
from app.core.config import settings
from app.models.schemas import (
    AgentType,
    CreateTaskRequest,
    FileAttachment,
    WSLogData,
    WSMessageType,
    WSOutputData,
    WSScreenshotData,
    WSStepUpdateData,
    WSTaskCompletedData,
    WSVncInfoData,
)
from app.services.agent_service import agent_manager
from app.services.event_buffer import event_buffer
from app.websocket.handler import _persist_assistant_message, _persist_task_user_message

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /api/v1/tasks/start
# ---------------------------------------------------------------------------

@router.post("/tasks/start")
async def start_task(
    req: CreateTaskRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
):
    """Start a new agent task and return its ID.

    The caller should then open an SSE stream on
    ``GET /api/v1/tasks/{task_id}/stream`` to receive real-time events.
    """
    task_id = str(uuid4())
    conversation_id = req.conversation_id

    # ---- build callbacks that push to the event buffer ----

    def on_step(step_data: dict[str, Any]):
        event_buffer.push(task_id, "step_update", step_data)

    def on_output(content: str, is_final: bool):
        event_buffer.push(
            task_id,
            "output",
            WSOutputData(content=content, is_final=is_final).model_dump(),
        )

    def on_screenshot(screenshot_base64: str | None, step_number: int):
        if screenshot_base64:
            event_buffer.push(
                task_id,
                "screenshot",
                WSScreenshotData(
                    base64=screenshot_base64, step_number=step_number
                ).model_dump(),
            )

    def on_thinking(thinking: str):
        event_buffer.push(task_id, "thinking", {"thinking": thinking})

    def on_error(error: str):
        event_buffer.push(task_id, "error", {"error": error})

    def on_log(
        level: str,
        message_text: str,
        source: str | None = None,
        step_number: int | None = None,
    ):
        event_buffer.push(
            task_id,
            "log",
            WSLogData(
                level=level,
                message=message_text,
                source=source,
                step_number=step_number,
            ).model_dump(),
        )

    def on_vnc_info(vnc_data: dict[str, Any]):
        event_buffer.push(
            task_id,
            "vnc_info",
            WSVncInfoData(**vnc_data).model_dump(),
        )

    # ---- create session ----

    session = await agent_manager.create_session_with_id(
        task_id=task_id,
        task=req.task,
        agent_type=req.agent_type.value,
        max_steps=req.max_steps,
        use_vision=req.use_vision,
        llm_model=req.llm_model,
        use_current_browser=req.use_current_browser,
        on_step_callback=on_step,
        on_output_callback=on_output,
        on_screenshot_callback=on_screenshot,
        on_thinking_callback=on_thinking,
        on_error_callback=on_error,
        on_log_callback=on_log,
        on_vnc_info_callback=on_vnc_info,
    )

    # ---- persist user message ----
    try:
        conversation_id = await _persist_task_user_message(
            principal=principal,
            task_id=task_id,
            task=req.task,
            conversation_id=conversation_id,
        )
    except Exception as persistence_error:
        logger.exception("Failed to persist start_task message: %s", persistence_error)

    # Push the initial "task_started" event (include conversation_id)
    event_buffer.push(
        task_id,
        "task_started",
        {
            "task": req.task,
            "agent_type": req.agent_type.value,
            "conversation_id": conversation_id,
        },
    )

    # Run agent in background
    asyncio.create_task(
        _run_agent(session, principal=principal, conversation_id=conversation_id)
    )

    return {"task_id": task_id}


async def _run_agent(
    session,
    principal: AuthPrincipal | None = None,
    conversation_id: str | None = None,
) -> None:
    """Run the agent task and push completion/failure events."""
    task_id = session.task_id
    try:
        result = await session.start()

        # Convert attachments
        raw_attachments = result.get("attachments", [])
        attachments = []
        for att in raw_attachments:
            if isinstance(att, dict):
                attachments.append(
                    FileAttachment(
                        name=att.get("name", "file"),
                        content=att.get("content"),
                        url=att.get("url"),
                        type=att.get("type"),
                        mime_type=att.get("mime_type"),
                        size=att.get("size"),
                    ).model_dump()
                )
            elif isinstance(att, str):
                attachments.append(
                    FileAttachment(name=att.split("/")[-1], url=att).model_dump()
                )

        event_buffer.push(
            task_id,
            "task_completed",
            WSTaskCompletedData(
                result=result.get("result", ""),
                success=result.get("success", False),
                total_steps=result.get("total_steps", 0),
                duration_seconds=result.get("duration_seconds", 0),
                attachments=[FileAttachment(**a) for a in attachments],
            ).model_dump(),
        )

        # Persist assistant completion message
        try:
            await _persist_assistant_message(
                principal=principal,
                task_id=task_id,
                conversation_id=conversation_id,
                content=result.get("result", "") or "Task completed successfully.",
                metadata={
                    "success": result.get("success", False),
                    "total_steps": result.get("total_steps", 0),
                    "duration_seconds": result.get("duration_seconds", 0),
                    "attachments": attachments,
                },
            )
        except Exception as persistence_error:
            logger.exception("Failed to persist task completion: %s", persistence_error)

    except asyncio.CancelledError:
        event_buffer.push(
            task_id, "task_cancelled", {"reason": "Task was cancelled"}
        )
        try:
            await _persist_assistant_message(
                principal=principal,
                task_id=task_id,
                conversation_id=conversation_id,
                content="Task was cancelled",
                metadata={"cancelled": True},
            )
        except Exception as persistence_error:
            logger.exception("Failed to persist cancellation: %s", persistence_error)

    except Exception as e:
        logger.exception("Task %s failed: %s", task_id, e)
        event_buffer.push(task_id, "task_failed", {"error": str(e)})
        try:
            await _persist_assistant_message(
                principal=principal,
                task_id=task_id,
                conversation_id=conversation_id,
                content=f"Error: {e}",
                metadata={"is_error": True},
            )
        except Exception as persistence_error:
            logger.exception("Failed to persist error: %s", persistence_error)

    finally:
        event_buffer.mark_complete(task_id)
        # Delayed cleanup
        asyncio.create_task(event_buffer.cleanup_task(task_id))
        # Also clean up agent session after a delay
        await asyncio.sleep(60)
        await agent_manager.remove_session(task_id)


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/{task_id}/events  (polling)
# ---------------------------------------------------------------------------

@router.get("/tasks/{task_id}/events")
async def poll_task_events(
    task_id: str,
    since: int = Query(default=0),
    _user: AuthPrincipal | None = Depends(get_current_user),
):
    """Return new events for a task since the given event ID.

    This is the polling alternative to the SSE stream endpoint.
    API Gateway HTTP API has a 30s integration timeout which kills
    long-lived SSE connections, so the frontend polls this endpoint
    every ~1.5 seconds instead.

    Response::

        {
            "events": [
                {"id": 1, "type": "task_started", "data": {...}},
                {"id": 2, "type": "log", "data": {...}},
                ...
            ],
            "complete": false
        }
    """
    if not event_buffer.has_task(task_id):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"detail": f"No active task {task_id}"},
        )

    events, complete = event_buffer.get_events_since(task_id, since)

    return {
        "events": [
            {"id": e.id, "type": e.event_type, "data": e.data}
            for e in events
        ],
        "complete": complete,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/{task_id}/stream  (SSE -- kept for local dev)
# ---------------------------------------------------------------------------

@router.get("/tasks/{task_id}/stream")
async def stream_task_events(
    task_id: str,
    request: Request,
    token: str | None = Query(default=None),
    last_event_id: int = Query(default=0, alias="lastEventId"),
):
    """Server-Sent Events stream for a running task.

    Auth: pass ``?token=<jwt>`` as query parameter (EventSource cannot
    send custom headers).

    Reconnection: pass ``?lastEventId=<id>`` to resume from a previous
    position (the browser does this automatically via ``Last-Event-ID``).
    """
    # Authenticate via query-string token
    if settings.AUTH_ENABLED:
        if not token:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing token query parameter"},
            )
        try:
            await verify_token_string(token)
        except Exception:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token"},
            )

    # Also check the standard Last-Event-ID header (SSE reconnection)
    header_last_id = request.headers.get("Last-Event-ID")
    if header_last_id is not None:
        try:
            last_event_id = int(header_last_id)
        except ValueError:
            pass

    if not event_buffer.has_task(task_id):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"detail": f"No active stream for task {task_id}"},
        )

    async def generate():
        """Yield SSE-formatted events."""
        try:
            async for event in event_buffer.stream(task_id, last_event_id):
                # SSE format: id, event, data fields separated by newlines
                yield f"id: {event.id}\n"
                yield f"event: {event.event_type}\n"
                yield f"data: {json.dumps(event.data)}\n\n"

            # Send a final "done" event so the client knows the stream ended
            yield "event: done\ndata: {}\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            return

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /api/v1/tasks/{task_id}/cancel
# ---------------------------------------------------------------------------

@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    _user: AuthPrincipal | None = Depends(get_current_user),
):
    """Cancel a running task."""
    session = await agent_manager.get_session(task_id)
    if not session:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"detail": f"No active session for task {task_id}"},
        )

    await session.cancel()
    event_buffer.push(task_id, "task_cancelled", {"reason": "Cancelled by user"})
    return {"status": "cancelled", "task_id": task_id}
