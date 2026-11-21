"""Explicit focus state for one browser session."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openbrowser.browser.session import CDPSession


FOCUS_TRANSITION_BUDGET_SECONDS = 8.0
FOCUS_NAVIGATION_OBSERVER_SECONDS = 2.0
CLOSE_RECONCILIATION_SECONDS = 2.0
FOCUS_TEXT_LIMIT = 300
FOCUS_TRUNCATION_MARKER = "..."


class FocusPhase(StrEnum):
    """Lifecycle phase for agent focus."""

    UNINITIALIZED = "uninitialized"
    SWITCHING = "switching"
    FOCUSED = "focused"
    RECOVERING = "recovering"
    LOST = "lost"


class FocusChangeReason(StrEnum):
    """Why a focus transition started or completed."""

    INITIAL = "initial"
    SWITCH = "switch"
    NAVIGATION = "navigation"
    CLOSE_REPAIR = "close_repair"
    SESSION_REBIND = "session_rebind"
    DETACH_RECOVERY = "detach_recovery"
    CRASH_RECOVERY = "crash_recovery"
    RESPONSIVENESS_RECOVERY = "responsiveness_recovery"
    RESET = "reset"


class FocusTargetCode(StrEnum):
    """Typed focus failure reasons."""

    NOT_A_PAGE = "not_a_page"
    TAB_NOT_LIVE = "tab_not_live"
    TARGET_MISSING = "target_missing"
    TARGET_DETACHED = "target_detached"
    NO_PAGE_AVAILABLE = "no_page_available"
    UNRESPONSIVE = "unresponsive"
    ACTIVATION_FAILED = "activation_failed"
    RECOVERY_FAILED = "recovery_failed"
    OBSERVATION_FAILED = "observation_failed"
    STALE_TRANSITION = "stale_transition"
    TIMEOUT = "timeout"


_DEFAULT_HINTS: dict[FocusTargetCode, str] = {
    FocusTargetCode.NOT_A_PAGE: "Choose a page target after a browser status check, then retry the action.",
    FocusTargetCode.TAB_NOT_LIVE: "Check browser status and choose an open page target.",
    FocusTargetCode.TARGET_MISSING: "Check browser status, then retry the action or open a new page.",
    FocusTargetCode.TARGET_DETACHED: "Check browser status before retrying the action.",
    FocusTargetCode.NO_PAGE_AVAILABLE: "Check browser status, then retry the explicit switch to create a page.",
    FocusTargetCode.UNRESPONSIVE: "Check browser status, then retry. Recovery may select another page.",
    FocusTargetCode.ACTIVATION_FAILED: "Check browser status, then retry the switch.",
    FocusTargetCode.RECOVERY_FAILED: "Check browser status, then retry to start a new bounded recovery.",
    FocusTargetCode.OBSERVATION_FAILED: "Focus changed. Check browser status before retrying a page-changing action.",
    FocusTargetCode.STALE_TRANSITION: "Check browser status. The browser or focus state changed during the operation.",
    FocusTargetCode.TIMEOUT: "Check browser status, then retry after the browser responds.",
}


def _bounded_text(value: str | None, *, limit: int = FOCUS_TEXT_LIMIT) -> str | None:
    """Return one-line text within a deterministic character bound."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    if limit <= len(FOCUS_TRUNCATION_MARKER):
        return FOCUS_TRUNCATION_MARKER[:limit]
    return text[: limit - len(FOCUS_TRUNCATION_MARKER)] + FOCUS_TRUNCATION_MARKER


def default_focus_hint(code: FocusTargetCode) -> str:
    """Return the concrete next step for a focus failure."""
    return _DEFAULT_HINTS[code]


@dataclass(frozen=True)
class FocusTransitionBudget:
    """One absolute monotonic deadline shared by focus-owned work."""

    deadline: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.deadline):
            raise ValueError("Focus deadline must be finite")

    @classmethod
    def start(cls, seconds: float | None = None) -> FocusTransitionBudget:
        duration = float(FOCUS_TRANSITION_BUDGET_SECONDS if seconds is None else seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Focus transition budget must be positive and finite")
        return cls(asyncio.get_running_loop().time() + duration)

    @classmethod
    def from_event(
        cls,
        *,
        deadline_monotonic: float | None,
        budget_seconds: float | None,
        default_seconds: float | None = None,
    ) -> FocusTransitionBudget:
        if deadline_monotonic is not None:
            deadline = float(deadline_monotonic)
            if not math.isfinite(deadline):
                raise ValueError("Focus deadline must be finite")
            return cls(deadline)
        seconds = budget_seconds if budget_seconds is not None else default_seconds
        return cls.start(seconds)

    def remaining(self) -> float:
        return max(0.0, self.deadline - asyncio.get_running_loop().time())

    def expired(self) -> bool:
        return self.remaining() <= 0

    def bounded_timeout(self, timeout: float) -> float:
        remaining = self.remaining()
        if remaining <= 0:
            return 0.0
        return min(float(timeout), remaining)


@dataclass(frozen=True)
class FocusErrorSnapshot:
    """Bounded, typed focus failure suitable for status output."""

    code: FocusTargetCode
    message: str
    hint: str
    target_id: str | None
    target_type: str | None
    phase: FocusPhase

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", _bounded_text(self.message) or "")
        object.__setattr__(self, "hint", _bounded_text(self.hint) or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "hint": self.hint,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "phase": self.phase.value,
        }


@dataclass(frozen=True)
class FocusSnapshot:
    """Read-only view of the focus state machine."""

    phase: FocusPhase
    target_id: str | None
    session_id: str | None
    previous_target_id: str | None
    generation: int
    reason: FocusChangeReason | None
    last_error: FocusErrorSnapshot | None
    updated_at: float

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("Focus generation cannot be negative")
        if self.phase is FocusPhase.UNINITIALIZED:
            if any((self.target_id, self.session_id, self.previous_target_id, self.reason, self.last_error)):
                raise ValueError("Uninitialized focus cannot carry a target, reason, or error")
            return
        if self.reason is None:
            raise ValueError(f"{self.phase.value} focus requires a reason")
        if self.phase is FocusPhase.SWITCHING:
            if self.session_id is not None or self.last_error is not None:
                raise ValueError("Switching focus cannot commit a session or error")
            return
        if self.phase is FocusPhase.FOCUSED:
            if self.target_id is None or self.session_id is None or self.last_error is not None:
                raise ValueError("Focused state requires target and session IDs and no error")
            return
        if self.phase is FocusPhase.RECOVERING:
            if self.target_id is not None or self.session_id is not None or self.last_error is not None:
                raise ValueError("Recovering focus cannot carry a target, session, or error")
            return
        if self.phase is FocusPhase.LOST:
            if self.target_id is not None or self.session_id is not None or self.last_error is None:
                raise ValueError("Lost focus requires a structured error and no target or session")
            return
        raise ValueError(f"Unknown focus phase: {self.phase}")

    @classmethod
    def uninitialized(cls) -> FocusSnapshot:
        return cls(
            phase=FocusPhase.UNINITIALIZED,
            target_id=None,
            session_id=None,
            previous_target_id=None,
            generation=0,
            reason=None,
            last_error=None,
            updated_at=time.time(),
        )

    def is_exact_pair(self, target_id: str | None, session_id: str | None) -> bool:
        return (
            self.phase is FocusPhase.FOCUSED
            and self.target_id is not None
            and self.session_id is not None
            and self.target_id == target_id
            and self.session_id == session_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "target_id": self.target_id,
            "session_id": self.session_id,
            "previous_target_id": self.previous_target_id,
            "generation": self.generation,
            "reason": self.reason.value if self.reason else None,
            "last_error": self.last_error.to_dict() if self.last_error else None,
            "updated_at": self.updated_at,
        }


class FocusTargetError(ValueError):
    """Typed failure for a rejected or failed focus transition."""

    def __init__(
        self,
        message: str,
        *,
        code: FocusTargetCode,
        phase: FocusPhase,
        target_id: str | None = None,
        target_type: str | None = None,
        hint: str | None = None,
        created_target_id: str | None = None,
    ):
        self.code = code
        self.phase = phase
        self.target_id = target_id
        self.target_type = target_type
        self.hint = _bounded_text(hint or default_focus_hint(code)) or default_focus_hint(code)
        self.created_target_id = created_target_id
        super().__init__(f"{_bounded_text(message) or message} Next step: {self.hint}")

    def to_snapshot(self) -> FocusErrorSnapshot:
        return FocusErrorSnapshot(
            code=self.code,
            message=str(self).split(" Next step: ", 1)[0],
            hint=self.hint,
            target_id=self.target_id,
            target_type=self.target_type,
            phase=self.phase,
        )


@dataclass(frozen=True)
class FocusTransitionResult:
    """Committed focus plus any page created by this transaction."""

    session: CDPSession
    snapshot: FocusSnapshot
    created_target_id: str | None = None

    def __post_init__(self) -> None:
        if self.snapshot.phase is not FocusPhase.FOCUSED:
            raise ValueError("A focus transition result must contain a focused snapshot")
        if self.session.target_id != self.snapshot.target_id or self.session.session_id != self.snapshot.session_id:
            raise ValueError("The focus session must match the committed target and session IDs")
