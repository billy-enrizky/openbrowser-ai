"""Explicit event-driven browser health state.

Lifecycle, focus, tabs, and the root CDP transport keep their existing
authority. This module records browser-level observations and derives one
status-only process-exit projection.
"""

from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openbrowser.browser.cdp_connection import MAX_ERROR_CHARS, sanitize_error_detail


class BrowserHealthPhase(StrEnum):
    """Current observed browser health."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRASHED = "crashed"
    STOPPED = "stopped"
    FAILED = "failed"


class BrowserHealthReason(StrEnum):
    """Last recorded or derived health reason."""

    INITIAL = "initial"
    CONNECTED = "connected"
    TARGET_CRASH = "target_crash"
    FOCUS_RECOVERED = "focus_recovered"
    PROCESS_EXIT = "process_exit"
    RESET = "reset"
    STOP = "stop"
    CONNECT_FAILED = "connect_failed"
    STARTUP_FAILED = "startup_failed"
    CLEANUP_FAILED = "cleanup_failed"


class FocusRecoveryOutcome(StrEnum):
    """Outcome of recovery for the latest page-target crash."""

    NOT_ATTEMPTED = "not_attempted"
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class BrowserCrashToken:
    """Identity of one accepted page-target crash."""

    sequence: int
    connection_generation: int
    target_id: str


@dataclass(frozen=True, eq=False)
class OwnedRootBinding:
    """Private proof that one published CDP root belongs to a local owner."""

    connection_generation: int
    endpoint: str
    pid: int
    start_time: float
    profile_dir: str
    instance_id: str
    ownership_marker: str
    executable: str
    cdp_port: int
    root: Any = field(repr=False)
    websocket: Any = field(repr=False)
    transport: Any = field(repr=False)


@dataclass(frozen=True)
class BrowserHealthSnapshot:
    """Read-only JSON-safe browser health view."""

    phase: BrowserHealthPhase
    healthy: bool | None
    reason: BrowserHealthReason
    connection_generation: int
    process_present: bool | None
    process_pid: int | None
    target_crash_count: int
    last_target_crash_at: float | None
    last_target_crash_target_id: str | None
    last_target_crash_detail: str | None
    last_target_crash_generation: int | None
    last_change_at: float | None
    last_focus_recovery_outcome: FocusRecoveryOutcome
    last_focus_recovery_target_id: str | None
    last_focus_recovery_at: float | None
    process_checked_at: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return one additive browser status block."""
        return {
            "phase": self.phase.value,
            "healthy": self.healthy,
            "reason": self.reason.value,
            "connection_generation": self.connection_generation,
            "process_present": self.process_present,
            "process_pid": self.process_pid,
            "target_crash_count": self.target_crash_count,
            "last_target_crash_at": self.last_target_crash_at,
            "last_target_crash_target_id": self.last_target_crash_target_id,
            "last_target_crash_detail": self.last_target_crash_detail,
            "last_target_crash_generation": self.last_target_crash_generation,
            "last_change_at": self.last_change_at,
            "last_focus_recovery_outcome": self.last_focus_recovery_outcome.value,
            "last_focus_recovery_target_id": self.last_focus_recovery_target_id,
            "last_focus_recovery_at": self.last_focus_recovery_at,
            "process_checked_at": self.process_checked_at,
        }


def _canonical_host(hostname: str) -> str:
    lowered = hostname.lower()
    if ":" not in lowered:
        return lowered
    try:
        return ipaddress.ip_address(lowered).compressed
    except ValueError:
        return lowered


def canonical_cdp_endpoint(value: str | None) -> str | None:
    """Return the identity-bearing canonical form of one WebSocket CDP URL."""
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in {"ws", "wss"} or parts.username is not None or parts.password is not None:
        return None
    hostname = parts.hostname
    if not hostname:
        return None
    host = _canonical_host(hostname)
    if ":" in host:
        host = f"[{host}]"
    effective_port = port if port is not None else (443 if scheme == "wss" else 80)
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunsplit((scheme, f"{host}:{effective_port}", path, parts.query, ""))


class BrowserHealthMachine:
    """Record browser observations without owning lifecycle or cleanup work."""

    def __init__(self) -> None:
        self._phase = BrowserHealthPhase.UNKNOWN
        self._healthy: bool | None = None
        self._reason = BrowserHealthReason.INITIAL
        self._connection_generation = 0
        self._observed_generation = 0
        self._target_crash_count = 0
        self._crash_sequence = 0
        self._latest_crash: BrowserCrashToken | None = None
        self._last_target_crash_at: float | None = None
        self._last_target_crash_target_id: str | None = None
        self._last_target_crash_detail: str | None = None
        self._last_target_crash_generation: int | None = None
        self._last_change_at: float | None = None
        self._last_recovery_outcome = FocusRecoveryOutcome.NOT_ATTEMPTED
        self._last_recovery_target_id: str | None = None
        self._last_recovery_at: float | None = None
        self._last_recovered_crash_sequence: int | None = None

    def _advance_generation(self, generation: int) -> int:
        try:
            value = max(0, int(generation))
        except (TypeError, ValueError, OverflowError):
            value = self._observed_generation
        if value > self._observed_generation:
            self._observed_generation = value
            if self._last_recovery_outcome is FocusRecoveryOutcome.PENDING:
                self._last_recovery_outcome = FocusRecoveryOutcome.NOT_ATTEMPTED
                self._last_recovery_target_id = None
                self._last_recovery_at = None
        return value

    @staticmethod
    def _healthy_for(phase: BrowserHealthPhase) -> bool | None:
        if phase is BrowserHealthPhase.HEALTHY:
            return True
        if phase in (BrowserHealthPhase.UNKNOWN, BrowserHealthPhase.STOPPED):
            return None
        return False

    def _set_phase(self, phase: BrowserHealthPhase, reason: BrowserHealthReason, generation: int) -> None:
        now = time.time()
        self._phase = phase
        self._healthy = self._healthy_for(phase)
        self._reason = reason
        self._connection_generation = generation
        self._last_change_at = now

    def _token_is_current(self, token: BrowserCrashToken, generation: int) -> bool:
        try:
            value = int(generation)
        except (TypeError, ValueError, OverflowError):
            return False
        return self._latest_crash == token and value == self._observed_generation

    def _has_unresolved_crash(self, generation: int) -> bool:
        crash = self._latest_crash
        return bool(
            crash is not None
            and crash.connection_generation == generation
            and self._last_recovered_crash_sequence != crash.sequence
        )

    def record_connected(self, generation: int) -> None:
        """Record one successful connect or start publication."""
        value = self._advance_generation(generation)
        if self._has_unresolved_crash(value):
            self._set_phase(BrowserHealthPhase.DEGRADED, BrowserHealthReason.TARGET_CRASH, value)
            return
        self._set_phase(BrowserHealthPhase.HEALTHY, BrowserHealthReason.CONNECTED, value)

    def record_connect_failed(self, generation: int) -> None:
        """Record one health-bearing connect failure."""
        value = self._advance_generation(generation)
        self._set_phase(BrowserHealthPhase.FAILED, BrowserHealthReason.CONNECT_FAILED, value)

    def record_startup_failed(self, generation: int) -> bool:
        """Record a start failure unless a stronger failure or stop is stored."""
        value = self._advance_generation(generation)
        if self._phase is BrowserHealthPhase.STOPPED:
            return False
        if (
            self._phase is BrowserHealthPhase.FAILED
            and self._reason is BrowserHealthReason.CONNECT_FAILED
            and self._connection_generation == value
        ):
            return False
        self._set_phase(BrowserHealthPhase.FAILED, BrowserHealthReason.STARTUP_FAILED, value)
        return True

    def record_target_crash(self, generation: int, target_id: str, detail: str | None) -> BrowserCrashToken | None:
        """Record one page-target crash, returning None for an immediate duplicate."""
        value = self._advance_generation(generation)
        if value < self._observed_generation:
            return None
        latest = self._latest_crash
        if latest is not None and latest.connection_generation == value and latest.target_id == target_id:
            return None

        self._crash_sequence += 1
        token = BrowserCrashToken(self._crash_sequence, value, target_id)
        now = time.time()
        self._target_crash_count += 1
        self._latest_crash = token
        self._last_target_crash_at = now
        self._last_target_crash_target_id = target_id
        self._last_target_crash_detail = sanitize_error_detail(detail, max_chars=MAX_ERROR_CHARS)
        self._last_target_crash_generation = value
        self._last_recovery_outcome = FocusRecoveryOutcome.NOT_ATTEMPTED
        self._last_recovery_target_id = None
        self._last_recovery_at = None
        if self._phase is not BrowserHealthPhase.FAILED:
            self._set_phase(BrowserHealthPhase.DEGRADED, BrowserHealthReason.TARGET_CRASH, value)
        return token

    def record_focus_recovery_started(self, token: BrowserCrashToken, generation: int) -> bool:
        """Mark the latest accepted crash as having an active focused recovery."""
        if not self._token_is_current(token, generation):
            return False
        if self._phase not in (BrowserHealthPhase.DEGRADED, BrowserHealthPhase.FAILED):
            return False
        self._last_recovery_outcome = FocusRecoveryOutcome.PENDING
        self._last_recovery_target_id = None
        self._last_recovery_at = time.time()
        return True

    def record_focus_recovery_success(self, token: BrowserCrashToken, generation: int, target_id: str) -> bool:
        """Record a successful recovery only for the latest accepted crash."""
        if not self._token_is_current(token, generation):
            return False
        if self._phase not in (BrowserHealthPhase.DEGRADED, BrowserHealthPhase.FAILED):
            return False
        value = self._advance_generation(generation)
        self._last_recovery_outcome = FocusRecoveryOutcome.SUCCEEDED
        self._last_recovery_target_id = target_id
        self._last_recovery_at = time.time()
        self._last_recovered_crash_sequence = token.sequence
        if self._phase is BrowserHealthPhase.DEGRADED:
            self._set_phase(BrowserHealthPhase.HEALTHY, BrowserHealthReason.FOCUS_RECOVERED, value)
        return True

    def record_focus_recovery_failure(self, token: BrowserCrashToken, generation: int) -> bool:
        """Record a failed recovery only for the latest accepted crash."""
        if not self._token_is_current(token, generation):
            return False
        if self._phase not in (BrowserHealthPhase.DEGRADED, BrowserHealthPhase.FAILED):
            return False
        self._last_recovery_outcome = FocusRecoveryOutcome.FAILED
        self._last_recovery_target_id = None
        self._last_recovery_at = time.time()
        return True

    def record_stopped(self, reason: BrowserHealthReason, generation: int) -> None:
        """Record a completed public reset, stop, or kill."""
        value = self._advance_generation(generation)
        self._set_phase(BrowserHealthPhase.STOPPED, reason, value)
        self._last_recovery_outcome = FocusRecoveryOutcome.NOT_ATTEMPTED
        self._last_recovery_target_id = None
        self._last_recovery_at = None

    def record_cleanup_failed(self, generation: int) -> None:
        """Record a reset, stop, or kill cleanup failure."""
        value = self._advance_generation(generation)
        self._set_phase(BrowserHealthPhase.FAILED, BrowserHealthReason.CLEANUP_FAILED, value)

    def snapshot(
        self,
        *,
        process_present: bool | None = None,
        process_pid: int | None = None,
        process_checked_at: float | None = None,
        projection_active: bool = False,
    ) -> BrowserHealthSnapshot:
        """Return the stored observation plus an optional process-exit projection."""
        phase = self._phase
        healthy = self._healthy
        reason = self._reason
        if projection_active and process_present is False:
            phase = BrowserHealthPhase.CRASHED
            healthy = False
            reason = BrowserHealthReason.PROCESS_EXIT
        return BrowserHealthSnapshot(
            phase=phase,
            healthy=healthy,
            reason=reason,
            connection_generation=self._connection_generation,
            process_present=process_present,
            process_pid=process_pid,
            target_crash_count=self._target_crash_count,
            last_target_crash_at=self._last_target_crash_at,
            last_target_crash_target_id=self._last_target_crash_target_id,
            last_target_crash_detail=self._last_target_crash_detail,
            last_target_crash_generation=self._last_target_crash_generation,
            last_change_at=self._last_change_at,
            last_focus_recovery_outcome=self._last_recovery_outcome,
            last_focus_recovery_target_id=self._last_recovery_target_id,
            last_focus_recovery_at=self._last_recovery_at,
            process_checked_at=process_checked_at,
        )
