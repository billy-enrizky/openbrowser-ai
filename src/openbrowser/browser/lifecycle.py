"""Explicit lifecycle state for browser, daemon, and MCP resources."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class LifecyclePhase(str, Enum):
	"""The shared lifecycle vocabulary for long-lived OpenBrowser resources."""

	STOPPED = 'stopped'
	STARTING = 'starting'
	READY = 'ready'
	DEGRADED = 'degraded'
	RECOVERING = 'recovering'
	STOPPING = 'stopping'
	FAILED = 'failed'


_ALLOWED_TRANSITIONS: dict[LifecyclePhase, frozenset[LifecyclePhase]] = {
	LifecyclePhase.STOPPED: frozenset(
		{
			LifecyclePhase.STARTING,
			LifecyclePhase.STOPPING,
			LifecyclePhase.FAILED,
		}
	),
	LifecyclePhase.STARTING: frozenset(
		{
			LifecyclePhase.READY,
			LifecyclePhase.FAILED,
			LifecyclePhase.STOPPING,
			LifecyclePhase.STOPPED,
		}
	),
	LifecyclePhase.READY: frozenset(
		{
			LifecyclePhase.DEGRADED,
			LifecyclePhase.RECOVERING,
			LifecyclePhase.STOPPING,
			LifecyclePhase.FAILED,
			LifecyclePhase.STOPPED,
		}
	),
	LifecyclePhase.DEGRADED: frozenset(
		{
			LifecyclePhase.READY,
			LifecyclePhase.STARTING,
			LifecyclePhase.RECOVERING,
			LifecyclePhase.STOPPING,
			LifecyclePhase.FAILED,
			LifecyclePhase.STOPPED,
		}
	),
	LifecyclePhase.RECOVERING: frozenset(
		{
			LifecyclePhase.READY,
			LifecyclePhase.DEGRADED,
			LifecyclePhase.STOPPING,
			LifecyclePhase.FAILED,
			LifecyclePhase.STOPPED,
		}
	),
	LifecyclePhase.STOPPING: frozenset(
		{
			LifecyclePhase.STOPPED,
			LifecyclePhase.FAILED,
		}
	),
	LifecyclePhase.FAILED: frozenset(
		{
			LifecyclePhase.STARTING,
			LifecyclePhase.RECOVERING,
			LifecyclePhase.STOPPING,
			LifecyclePhase.STOPPED,
		}
	),
}


class LifecycleTransitionError(RuntimeError):
	"""Raised when a resource attempts an illegal lifecycle transition."""

	def __init__(
		self,
		name: str,
		current: LifecyclePhase,
		requested: LifecyclePhase,
		reason: str | None = None,
	):
		self.name = name
		self.current = current
		self.requested = requested
		self.reason = reason
		detail = f'Invalid lifecycle transition for {name}: {current.value} -> {requested.value}'
		if reason:
			detail = f'{detail} ({reason})'
		super().__init__(detail)


def _error_text(error: BaseException | str | None) -> str | None:
	if error is None:
		return None
	if isinstance(error, str):
		return error
	return f'{type(error).__name__}: {error}'


@dataclass(frozen=True)
class LifecycleSnapshot:
	"""Read-only view of one lifecycle machine."""

	name: str
	phase: LifecyclePhase
	previous_phase: LifecyclePhase | None
	generation: int
	reason: str | None
	last_error: str | None
	started_at: float | None
	updated_at: float

	def to_dict(self) -> dict[str, Any]:
		return {
			'name': self.name,
			'phase': self.phase.value,
			'previous_phase': self.previous_phase.value if self.previous_phase else None,
			'generation': self.generation,
			'reason': self.reason,
			'last_error': self.last_error,
			'started_at': self.started_at,
			'updated_at': self.updated_at,
		}


class LifecycleMachine:
	"""A small, explicit state machine for one long-lived resource.

	The machine does not own locks. The resource that owns the machine must
	serialize transitions. A transition to the current phase is a no-op so
	start, stop, and reset calls can be repeated safely.
	"""

	def __init__(self, name: str, phase: LifecyclePhase = LifecyclePhase.STOPPED):
		self.name = name
		self._phase = phase
		self._previous_phase: LifecyclePhase | None = None
		self._generation = 0
		self._reason: str | None = None
		self._last_error: str | None = None
		self._started_at: float | None = None
		self._updated_at = time.time()

	@property
	def phase(self) -> LifecyclePhase:
		return self._phase

	@property
	def generation(self) -> int:
		return self._generation

	def seed_generation(self, generation: int) -> None:
		"""Raise the generation floor when a replacement resource is adopted."""
		try:
			candidate = int(generation)
		except (TypeError, ValueError):
			return
		self._generation = max(self._generation, candidate)

	def transition(
		self,
		phase: LifecyclePhase,
		*,
		reason: str | None = None,
		error: BaseException | str | None = None,
		increment_generation: bool = False,
	) -> LifecycleSnapshot:
		"""Move to ``phase`` or raise when the transition is illegal."""
		now = time.time()
		if phase is self._phase:
			if reason is not None:
				self._reason = reason
			if error is not None:
				self._last_error = _error_text(error)
			self._updated_at = now
			return self.snapshot()

		if phase not in _ALLOWED_TRANSITIONS[self._phase]:
			raise LifecycleTransitionError(self.name, self._phase, phase, reason)

		self._previous_phase = self._phase
		self._phase = phase
		self._reason = reason
		if phase is LifecyclePhase.STARTING:
			self._started_at = now
		if error is not None:
			self._last_error = _error_text(error)
		elif phase is LifecyclePhase.STARTING:
			self._last_error = None
		if increment_generation:
			self._generation += 1
		self._updated_at = now
		return self.snapshot()

	def fail(self, error: BaseException | str, *, reason: str | None = None) -> LifecycleSnapshot:
		"""Move to ``failed`` and record the error."""
		return self.transition(LifecyclePhase.FAILED, reason=reason, error=error)

	def snapshot(self) -> LifecycleSnapshot:
		return LifecycleSnapshot(
			name=self.name,
			phase=self._phase,
			previous_phase=self._previous_phase,
			generation=self._generation,
			reason=self._reason,
			last_error=self._last_error,
			started_at=self._started_at,
			updated_at=self._updated_at,
		)
