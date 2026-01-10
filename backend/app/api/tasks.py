"""API routes for tasks."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    CreateTaskRequest,
    TaskListResponse,
    TaskListItem,
    TaskResponse,
    TaskStatus,
)
from app.services.agent_service import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[TaskStatus] = None,
    project_id: Optional[str] = None,
):
    """List all tasks with pagination and filtering."""
    # For now, return tasks from active sessions
    # In production, this would query a database
    tasks = []
    for task_id, session in agent_manager.sessions.items():
        status_val = TaskStatus.RUNNING if session.is_running else (
            TaskStatus.COMPLETED if session.success else TaskStatus.FAILED
        )
        if status and status_val != status:
            continue
            
        tasks.append(TaskListItem(
            id=task_id,
            task=session.task,
            status=status_val,
            agent_type=session.agent_type,
            created_at=session.started_at or session.completed_at,
            preview=session.result[:100] if session.result else None,
        ))
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    paginated = tasks[start:end]
    
    return TaskListResponse(
        tasks=paginated,
        total=len(tasks),
        page=page,
        page_size=page_size,
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get details of a specific task."""
    session = await agent_manager.get_session(task_id)
    if not session:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = TaskStatus.RUNNING if session.is_running else (
        TaskStatus.COMPLETED if session.success else TaskStatus.FAILED
    )
    
    return TaskResponse(
        id=task_id,
        task=session.task,
        status=status,
        agent_type=session.agent_type,
        created_at=session.started_at,
        updated_at=session.completed_at or session.started_at,
        completed_at=session.completed_at,
        result=session.result,
        success=session.success,
        error=session.error,
    )


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    session = await agent_manager.get_session(task_id)
    if not session:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if not session.is_running:
        raise HTTPException(status_code=400, detail="Task is not running")
    
    await session.cancel()
    return {"status": "cancelled", "task_id": task_id}

