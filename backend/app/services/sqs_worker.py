"""SQS long-poll worker for executing scheduled jobs.

Runs inside the FastAPI lifespan (startup/shutdown).
Picks up messages from the scheduled-jobs queue and replays workflows.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)

_worker_task: asyncio.Task | None = None
_shutdown_event = asyncio.Event()


@lru_cache(maxsize=1)
def _get_sqs_client():
    import boto3
    return boto3.client("sqs", region_name=os.getenv("AWS_REGION", "ca-central-1"))


async def start() -> None:
    """Start the SQS worker loop in background."""
    global _worker_task
    queue_url = settings.SQS_SCHEDULE_QUEUE_URL
    if not queue_url:
        logger.info("SQS_SCHEDULE_QUEUE_URL not configured, SQS worker disabled")
        return

    _shutdown_event.clear()
    _worker_task = asyncio.create_task(_poll_loop(queue_url))
    logger.info("SQS worker started, polling %s", queue_url)

    # Check for stale executions on startup
    await _cleanup_stale_executions()


async def stop() -> None:
    """Graceful shutdown."""
    global _worker_task
    _shutdown_event.set()
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
    logger.info("SQS worker stopped")


async def _poll_loop(queue_url: str) -> None:
    """Long-poll SQS and process messages concurrently up to the semaphore limit."""
    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_SCHEDULED_EXECUTIONS)
    client = _get_sqs_client()

    async def _guarded_process(msg: dict) -> None:
        async with semaphore:
            try:
                await _process_message(msg, queue_url, client)
            except Exception as e:
                logger.exception("Failed to process SQS message: %s", e)

    while not _shutdown_event.is_set():
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.receive_message(
                    QueueUrl=queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20,
                ),
            )

            for msg in response.get("Messages", []):
                asyncio.create_task(_guarded_process(msg))

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("SQS poll error: %s", e)
            await asyncio.sleep(5)


async def _delete_message(client, queue_url: str, receipt_handle: str) -> None:
    """Delete an SQS message without blocking the event loop."""
    await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle),
    )


async def _process_message(msg: dict, queue_url: str, client) -> None:
    """Process a single SQS message: load job, replay workflow, update execution."""
    body = json.loads(msg["Body"])
    receipt_handle = msg["ReceiptHandle"]

    job_id = body.get("job_id")
    user_id = body.get("user_id")

    if not job_id or not user_id:
        logger.warning("Malformed SQS message (missing job_id/user_id), deleting: %s", body)
        await _delete_message(client, queue_url, receipt_handle)
        return

    logger.info("Processing scheduled job %s for user %s", job_id, user_id)

    from app.db.session import get_session_factory
    from app.services import schedule_service

    session_factory = get_session_factory()
    async with session_factory() as db:
        job = await schedule_service.get_job(db, job_id, user_id)
        if not job:
            logger.warning("Job %s not found, deleting SQS message", job_id)
            await _delete_message(client, queue_url, receipt_handle)
            return

        if job.status not in ("active",):
            logger.info("Job %s is %s, skipping", job_id, job.status)
            await _delete_message(client, queue_url, receipt_handle)
            return

        # Create execution record
        execution = await schedule_service.record_execution(db, job_id, "running")
        now = datetime.now(timezone.utc)
        execution.heartbeat_at = now
        await db.commit()

        try:
            # TODO: Implement actual workflow replay here.
            # When workflow replay is added, update execution.heartbeat_at
            # periodically (every ~60s) so _cleanup_stale_executions can
            # detect truly stale runs vs. long-running ones.
            execution.status = "success"
            execution.completed_at = datetime.now(timezone.utc)
            job.last_run_at = datetime.now(timezone.utc)
            await db.commit()
            logger.info("Job %s execution %s completed successfully", job_id, execution.id)

        except Exception as e:
            execution.status = "failed"
            execution.completed_at = datetime.now(timezone.utc)
            execution.error_message = str(e)[:2000]
            job.last_run_at = datetime.now(timezone.utc)

            # Check for consecutive failures
            recent = await schedule_service.get_executions(db, job_id, limit=3)
            consecutive_failures = sum(1 for ex in recent if ex.status == "failed")
            if consecutive_failures >= 3:
                job.status = "paused"
                logger.warning("Job %s auto-paused after 3 consecutive failures", job_id)

            await db.commit()
            logger.exception("Job %s execution failed: %s", job_id, e)

    # Delete message from queue (processed successfully or failed but recorded)
    await _delete_message(client, queue_url, receipt_handle)


async def _cleanup_stale_executions() -> None:
    """On startup, mark stale 'running' executions as failed."""
    try:
        from app.db.session import get_session_factory, is_database_configured

        if not is_database_configured():
            return

        from sqlalchemy import update, func
        from app.db.models import JobExecution

        session_factory = get_session_factory()
        async with session_factory() as db:
            cutoff = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            # Mark executions running for > 15 min as failed
            from datetime import timedelta
            stale_cutoff = cutoff - timedelta(minutes=15)

            # Use heartbeat_at if set, fall back to started_at
            last_active = func.coalesce(JobExecution.heartbeat_at, JobExecution.started_at)

            result = await db.execute(
                update(JobExecution)
                .where(
                    JobExecution.status == "running",
                    last_active < stale_cutoff,
                )
                .values(
                    status="failed",
                    error_message="Stale execution detected on worker restart",
                    completed_at=cutoff,
                )
            )
            if result.rowcount > 0:
                logger.warning("Marked %d stale executions as failed", result.rowcount)
            await db.commit()
    except Exception as e:
        logger.warning("Failed to cleanup stale executions: %s", e)
