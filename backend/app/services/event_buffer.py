"""In-memory event buffer for SSE streaming.

Stores events per task and supports async streaming with reconnection.
Events have incremental IDs so clients can reconnect with Last-Event-ID.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

# Keep completed task events for 5 minutes for potential reconnects
CLEANUP_DELAY_SECONDS = 300


@dataclass
class TaskEvent:
    """A single event in the task stream."""
    id: int
    event_type: str
    task_id: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class EventBuffer:
    """Thread-safe event buffer for streaming task events via SSE.

    Each task gets its own event list with auto-incrementing IDs.
    Supports async iteration with backpressure via asyncio.Event.
    """

    def __init__(self):
        self._events: dict[str, list[TaskEvent]] = {}
        self._notify: dict[str, asyncio.Event] = {}
        self._completed: dict[str, bool] = {}
        self._counters: dict[str, int] = {}

    def _ensure_task(self, task_id: str) -> None:
        if task_id not in self._events:
            self._events[task_id] = []
            self._notify[task_id] = asyncio.Event()
            self._completed[task_id] = False
            self._counters[task_id] = 0

    def push(self, task_id: str, event_type: str, data: dict[str, Any]) -> int:
        """Push a new event for a task. Returns the event ID."""
        self._ensure_task(task_id)
        self._counters[task_id] += 1
        event_id = self._counters[task_id]

        event = TaskEvent(
            id=event_id,
            event_type=event_type,
            task_id=task_id,
            data=data,
        )
        self._events[task_id].append(event)

        # Wake up any waiting streamers
        self._notify[task_id].set()
        self._notify[task_id] = asyncio.Event()

        return event_id

    def mark_complete(self, task_id: str) -> None:
        """Mark a task stream as complete (no more events will arrive)."""
        self._ensure_task(task_id)
        self._completed[task_id] = True
        # Wake up waiters so they see the completion
        self._notify[task_id].set()

    def is_complete(self, task_id: str) -> bool:
        return self._completed.get(task_id, False)

    def has_task(self, task_id: str) -> bool:
        return task_id in self._events

    def get_events_since(
        self, task_id: str, since_id: int = 0
    ) -> tuple[list[TaskEvent], bool]:
        """Return (events_after_since_id, is_complete).

        Used by the polling endpoint as a synchronous alternative to
        the async ``stream()`` generator.
        """
        events = self._events.get(task_id, [])
        new_events = [e for e in events if e.id > since_id]
        complete = self._completed.get(task_id, False)
        return new_events, complete

    async def stream(
        self, task_id: str, last_event_id: int = 0
    ) -> AsyncGenerator[TaskEvent, None]:
        """Async generator that yields events for a task.

        Starts from events after last_event_id (for reconnection support).
        Waits for new events and yields them as they arrive.
        Stops when the task stream is marked complete.
        """
        self._ensure_task(task_id)
        cursor = last_event_id

        while True:
            # Yield any events we haven't sent yet
            events = self._events.get(task_id, [])
            new_events = [e for e in events if e.id > cursor]

            for event in new_events:
                yield event
                cursor = event.id

            # If the task is done and we've sent all events, stop
            if self._completed.get(task_id, False):
                # Check one more time for any final events
                events = self._events.get(task_id, [])
                final_events = [e for e in events if e.id > cursor]
                for event in final_events:
                    yield event
                    cursor = event.id
                return

            # Wait for new events (with timeout for heartbeats)
            notify = self._notify.get(task_id)
            if notify:
                try:
                    await asyncio.wait_for(notify.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Yield nothing -- the caller should send a heartbeat comment
                    pass

    async def cleanup_task(self, task_id: str, delay: float = CLEANUP_DELAY_SECONDS) -> None:
        """Remove task events after a delay. Call after task completion."""
        await asyncio.sleep(delay)
        self._events.pop(task_id, None)
        self._notify.pop(task_id, None)
        self._completed.pop(task_id, None)
        self._counters.pop(task_id, None)
        logger.debug("Cleaned up event buffer for task %s", task_id)


# Global event buffer instance
event_buffer = EventBuffer()
