"""Schedule service -- job lifecycle and EventBridge Scheduler integration."""

import asyncio
import json
import logging
import os
from functools import lru_cache, partial

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import JobExecution, ScheduledJob

logger = logging.getLogger(__name__)


async def _run_sync(func, *args, **kwargs):
    """Run a blocking boto3 call without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


@lru_cache(maxsize=1)
def _get_scheduler_client():
    import boto3
    return boto3.client("scheduler", region_name=os.getenv("AWS_REGION", "ca-central-1"))


@lru_cache(maxsize=1)
def _get_sqs_arn() -> str:
    """Derive SQS ARN from queue URL.

    URL format: https://sqs.{region}.amazonaws.com/{account}/{name}
    """
    queue_url = settings.SQS_SCHEDULE_QUEUE_URL
    if not queue_url:
        raise RuntimeError("SQS_SCHEDULE_QUEUE_URL not configured")
    parts = queue_url.rstrip("/").split("/")
    if len(parts) < 5 or not parts[-2].isdigit():
        raise RuntimeError(f"Cannot parse SQS queue URL (expected standard format): {queue_url}")
    account_id = parts[-2]
    queue_name = parts[-1]
    region = queue_url.split(".")[1]
    return f"arn:aws:sqs:{region}:{account_id}:{queue_name}"


async def count_active_jobs(db: AsyncSession, user_id: str) -> int:
    result = await db.execute(
        select(func.count())
        .select_from(ScheduledJob)
        .where(ScheduledJob.user_id == user_id, ScheduledJob.status.in_(["testing", "active"]))
    )
    return result.scalar_one()


async def create_job(
    db: AsyncSession,
    user_id: str,
    title: str,
    task_description: str,
    schedule_expression: str,
    schedule_timezone: str = "UTC",
    auth_profile_id: str | None = None,
) -> ScheduledJob:
    count = await count_active_jobs(db, user_id)
    if count >= settings.MAX_SCHEDULED_JOBS_PER_USER:
        raise RuntimeError(f"Maximum active scheduled jobs ({settings.MAX_SCHEDULED_JOBS_PER_USER}) reached")

    job = ScheduledJob(
        user_id=user_id,
        title=title,
        task_description=task_description,
        schedule_expression=schedule_expression,
        schedule_timezone=schedule_timezone,
        auth_profile_id=auth_profile_id,
        status="testing",
    )
    db.add(job)
    await db.flush()
    return job


async def activate_job(db: AsyncSession, job: ScheduledJob, workflow_id: str) -> None:
    """After successful test run, link workflow and create EventBridge schedule."""
    job.workflow_id = workflow_id
    job.status = "active"

    queue_url = settings.SQS_SCHEDULE_QUEUE_URL
    role_arn = settings.SCHEDULER_ROLE_ARN
    if not queue_url or not role_arn:
        logger.warning("SQS_SCHEDULE_QUEUE_URL or SCHEDULER_ROLE_ARN not configured, skipping EventBridge schedule")
        await db.flush()
        return

    sqs_arn = _get_sqs_arn()

    client = _get_scheduler_client()
    schedule_name = f"openbrowser-job-{job.id}"

    response = await _run_sync(
        client.create_schedule,
        Name=schedule_name,
        ScheduleExpression=f"cron({job.schedule_expression})",
        ScheduleExpressionTimezone=job.schedule_timezone,
        FlexibleTimeWindow={"Mode": "OFF"},
        Target={
            "Arn": sqs_arn,
            "RoleArn": role_arn,
            "Input": json.dumps({"job_id": job.id, "user_id": job.user_id}),
        },
        State="ENABLED",
    )
    job.eventbridge_schedule_arn = response.get("ScheduleArn", schedule_name)
    await db.flush()


async def pause_job(db: AsyncSession, job: ScheduledJob) -> None:
    job.status = "paused"
    if job.eventbridge_schedule_arn:
        try:
            client = _get_scheduler_client()
            sqs_arn = _get_sqs_arn()
            await _run_sync(
                client.update_schedule,
                Name=f"openbrowser-job-{job.id}",
                ScheduleExpression=f"cron({job.schedule_expression})",
                ScheduleExpressionTimezone=job.schedule_timezone,
                FlexibleTimeWindow={"Mode": "OFF"},
                Target={
                    "Arn": sqs_arn,
                    "RoleArn": settings.SCHEDULER_ROLE_ARN or "",
                    "Input": json.dumps({"job_id": job.id, "user_id": job.user_id}),
                },
                State="DISABLED",
            )
        except Exception as e:
            logger.warning("Failed to disable EventBridge schedule: %s", e)
    await db.flush()


async def resume_job(db: AsyncSession, job: ScheduledJob) -> None:
    job.status = "active"
    if job.eventbridge_schedule_arn:
        try:
            client = _get_scheduler_client()
            sqs_arn = _get_sqs_arn()

            await _run_sync(
                client.update_schedule,
                Name=f"openbrowser-job-{job.id}",
                ScheduleExpression=f"cron({job.schedule_expression})",
                ScheduleExpressionTimezone=job.schedule_timezone,
                FlexibleTimeWindow={"Mode": "OFF"},
                Target={
                    "Arn": sqs_arn,
                    "RoleArn": settings.SCHEDULER_ROLE_ARN or "",
                    "Input": json.dumps({"job_id": job.id, "user_id": job.user_id}),
                },
                State="ENABLED",
            )
        except Exception as e:
            logger.warning("Failed to enable EventBridge schedule: %s", e)
    await db.flush()


async def update_eventbridge_schedule(job: ScheduledJob) -> None:
    """Sync schedule expression/timezone to EventBridge after a PATCH update."""
    if not job.eventbridge_schedule_arn:
        return

    queue_url = settings.SQS_SCHEDULE_QUEUE_URL
    role_arn = settings.SCHEDULER_ROLE_ARN
    if not queue_url or not role_arn:
        return

    try:
        client = _get_scheduler_client()
        sqs_arn = _get_sqs_arn()
        state = "ENABLED" if job.status == "active" else "DISABLED"

        await _run_sync(
            client.update_schedule,
            Name=f"openbrowser-job-{job.id}",
            ScheduleExpression=f"cron({job.schedule_expression})",
            ScheduleExpressionTimezone=job.schedule_timezone,
            FlexibleTimeWindow={"Mode": "OFF"},
            Target={
                "Arn": sqs_arn,
                "RoleArn": role_arn,
                "Input": json.dumps({"job_id": job.id, "user_id": job.user_id}),
            },
            State=state,
        )
    except Exception as e:
        logger.warning("Failed to update EventBridge schedule for job %s: %s", job.id, e)


async def delete_job(db: AsyncSession, job: ScheduledJob) -> None:
    """Delete job and its EventBridge schedule."""
    if job.eventbridge_schedule_arn:
        try:
            client = _get_scheduler_client()
            await _run_sync(client.delete_schedule, Name=f"openbrowser-job-{job.id}")
        except Exception as e:
            logger.warning("Failed to delete EventBridge schedule: %s", e)
    await db.delete(job)
    await db.flush()


async def list_jobs(db: AsyncSession, user_id: str) -> list[ScheduledJob]:
    result = await db.execute(
        select(ScheduledJob)
        .where(ScheduledJob.user_id == user_id)
        .order_by(ScheduledJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_job(db: AsyncSession, job_id: str, user_id: str) -> ScheduledJob | None:
    result = await db.execute(
        select(ScheduledJob).where(
            ScheduledJob.id == job_id,
            ScheduledJob.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def get_executions(db: AsyncSession, job_id: str, limit: int = 20) -> list[JobExecution]:
    result = await db.execute(
        select(JobExecution)
        .where(JobExecution.job_id == job_id)
        .order_by(JobExecution.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def record_execution(
    db: AsyncSession,
    job_id: str,
    status: str = "running",
    task_id: str | None = None,
) -> JobExecution:
    execution = JobExecution(
        job_id=job_id,
        status=status,
        task_id=task_id,
    )
    db.add(execution)
    await db.flush()
    return execution
