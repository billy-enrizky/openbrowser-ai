"""Schedule API endpoints.

POST   /api/v1/schedules              -- Create job (starts test run)
GET    /api/v1/schedules              -- List jobs
GET    /api/v1/schedules/{id}         -- Get job + recent executions
PATCH  /api/v1/schedules/{id}         -- Update schedule, pause/resume
DELETE /api/v1/schedules/{id}         -- Delete job
GET    /api/v1/schedules/{id}/executions -- Execution history
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthPrincipal, get_current_user
from app.db.session import get_db_session, is_database_configured
from app.models.schemas import (
    CreateScheduledJobRequest,
    JobExecutionListResponse,
    JobExecutionResponse,
    ScheduledJobDetailResponse,
    ScheduledJobListResponse,
    ScheduledJobResponse,
    UpdateScheduledJobRequest,
)
from app.api.deps import resolve_user_id
from app.services import schedule_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schedules", tags=["schedules"])


def _job_to_response(job) -> ScheduledJobResponse:
    return ScheduledJobResponse(
        id=job.id,
        title=job.title,
        task_description=job.task_description,
        workflow_id=job.workflow_id,
        auth_profile_id=job.auth_profile_id,
        schedule_expression=job.schedule_expression,
        schedule_timezone=job.schedule_timezone,
        status=job.status,
        last_run_at=job.last_run_at,
        next_run_at=job.next_run_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _execution_to_response(ex) -> JobExecutionResponse:
    return JobExecutionResponse(
        id=ex.id,
        status=ex.status,
        started_at=ex.started_at,
        completed_at=ex.completed_at,
        error_message=ex.error_message,
        task_id=ex.task_id,
        created_at=ex.created_at,
    )


@router.post("", response_model=ScheduledJobResponse)
async def create_scheduled_job(
    req: CreateScheduledJobRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a scheduled job. Starts in 'testing' status."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await resolve_user_id(principal, db)

    try:
        job = await schedule_service.create_job(
            db=db,
            user_id=user_id,
            title=req.title,
            task_description=req.task_description,
            schedule_expression=req.schedule_expression,
            schedule_timezone=req.schedule_timezone,
            auth_profile_id=req.auth_profile_id,
        )
        await db.commit()
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))

    return _job_to_response(job)


@router.get("", response_model=ScheduledJobListResponse)
async def list_scheduled_jobs(
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """List all scheduled jobs for the current user."""
    if not is_database_configured():
        return ScheduledJobListResponse(jobs=[])

    user_id = await resolve_user_id(principal, db)
    jobs = await schedule_service.list_jobs(db, user_id)
    return ScheduledJobListResponse(jobs=[_job_to_response(j) for j in jobs])


@router.get("/{job_id}", response_model=ScheduledJobDetailResponse)
async def get_scheduled_job(
    job_id: str,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Get a scheduled job with recent executions."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await resolve_user_id(principal, db)
    job = await schedule_service.get_job(db, job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    executions = await schedule_service.get_executions(db, job_id)
    return ScheduledJobDetailResponse(
        job=_job_to_response(job),
        executions=[_execution_to_response(e) for e in executions],
    )


@router.patch("/{job_id}", response_model=ScheduledJobResponse)
async def update_scheduled_job(
    job_id: str,
    req: UpdateScheduledJobRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Update a scheduled job (title, schedule, or pause/resume)."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await resolve_user_id(principal, db)
    job = await schedule_service.get_job(db, job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    if req.title is not None:
        job.title = req.title

    schedule_changed = False
    if req.schedule_expression is not None:
        job.schedule_expression = req.schedule_expression
        schedule_changed = True
    if req.schedule_timezone is not None:
        job.schedule_timezone = req.schedule_timezone
        schedule_changed = True

    if req.status is not None:
        if req.status == "paused" and job.status == "active":
            await schedule_service.pause_job(db, job)
        elif req.status == "active" and job.status == "paused":
            await schedule_service.resume_job(db, job)
    elif schedule_changed:
        # Sync expression/timezone change to EventBridge (if no status change already did it)
        await schedule_service.update_eventbridge_schedule(job)

    await db.commit()
    return _job_to_response(job)


@router.delete("/{job_id}", status_code=204)
async def delete_scheduled_job(
    job_id: str,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a scheduled job and its EventBridge schedule."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await resolve_user_id(principal, db)
    job = await schedule_service.get_job(db, job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    await schedule_service.delete_job(db, job)
    await db.commit()


@router.get("/{job_id}/executions", response_model=JobExecutionListResponse)
async def list_executions(
    job_id: str,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Get execution history for a scheduled job."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await resolve_user_id(principal, db)
    job = await schedule_service.get_job(db, job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    executions = await schedule_service.get_executions(db, job_id)
    return {"executions": [_execution_to_response(e) for e in executions]}
