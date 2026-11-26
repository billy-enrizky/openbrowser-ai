"""Keep harness metadata validation separate from timed native task work.

Only benchmark launch/keychain proof functions use ``validation_only``.
Product startup, IPC, task execution, native readiness, transfer completion,
and output serialization remain inside the task interval. Both durations are
retained so the measurement boundary can be audited from raw rows.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import wraps

MEASUREMENT_METHOD = "native-task-external-validation-v1"


class TaskTimer:
    def __init__(self, *, clock: Callable[[], int] | None = None) -> None:
        self.clock = clock or time.perf_counter_ns
        self._token: Token[TaskTimer | None] | None = None
        self._started: int | None = None
        self._wall_ns: int | None = None
        self._validation_ns = 0
        self._validation_calls = 0
        self._validation_depth = 0

    def __enter__(self) -> TaskTimer:
        if _CURRENT.get() is not None:
            raise RuntimeError("a task timer is already active")
        if self._started is not None:
            raise RuntimeError("a task timer can measure one task")
        self._started = self.clock()
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            assert self._started is not None
            self._wall_ns = self.clock() - self._started
        finally:
            assert self._token is not None
            _CURRENT.reset(self._token)

    def public(self) -> dict[str, int]:
        if self._wall_ns is None:
            raise RuntimeError("task timer has not finished")
        if not 0 <= self._validation_ns <= self._wall_ns:
            raise ValueError("task timer clock moved backwards")
        return {
            "wall_ns": self._wall_ns,
            "validation_ns": self._validation_ns,
            "task_ns": self._wall_ns - self._validation_ns,
            "validation_calls": self._validation_calls,
        }

    def validation[**P, R](self, callback: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self._validation_calls += 1
        outermost = self._validation_depth == 0
        started = self.clock() if outermost else 0
        self._validation_depth += 1
        try:
            return callback(*args, **kwargs)
        finally:
            self._validation_depth -= 1
            if outermost:
                self._validation_ns += self.clock() - started


_CURRENT: ContextVar[TaskTimer | None] = ContextVar("native_task_timer", default=None)
_LAUNCH_OBSERVER: ContextVar[Callable[[str], None] | None] = ContextVar("native_launch_observer", default=None)


def validation_only[**P, R](callback: Callable[P, R]) -> Callable[P, R]:
    """Measure a synchronous harness proof separately when a timer is active."""

    @wraps(callback)
    def measured(*args: P.args, **kwargs: P.kwargs) -> R:
        timer = _CURRENT.get()
        if timer is None:
            return callback(*args, **kwargs)
        return timer.validation(callback, *args, **kwargs)

    return measured


@contextmanager
def browser_launch_observer(callback: Callable[[str], None]) -> Iterator[None]:
    """Own the launch-proof checkpoints for one synchronous native task."""
    token = _LAUNCH_OBSERVER.set(callback)
    try:
        yield
    finally:
        _LAUNCH_OBSERVER.reset(token)


@validation_only
def _observe_browser_launch(callback: Callable[[str], None], marker: str) -> None:
    callback(marker)


def browser_launch_checkpoint(marker: str) -> None:
    """Observe a browser before an auth restart, preserving native batching."""
    observer = _LAUNCH_OBSERVER.get()
    if observer is not None:
        _observe_browser_launch(observer, marker)
