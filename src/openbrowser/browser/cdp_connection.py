"""Bounded health observation for the root CDP transport.

The browser session lifecycle owns start, stop, reset, and recovery. This
module only records the last observed state of the root CDP client and the
last root-level health probe. It does not own the client or close it.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

MAX_ERROR_CHARS = 240
_URL_RE = re.compile(r"(?i)\b(?:https?|wss?)://\S+")
_QUOTED_HEADER_CREDENTIAL_RE = re.compile(
    r"(?i)(['\"])(proxy-authorization|authorization|cookie)\1\s*[:=]\s*(['\"])(?:\\.|(?!\3).)*\3",
    re.DOTALL,
)
_HEADER_CREDENTIAL_RE = re.compile(r"(?i)(['\"]?)(proxy-authorization|authorization|cookie)\1\s*[:=]\s*")
_QUOTED_CREDENTIAL_RE = re.compile(
    r"(?i)(['\"]?)\b(password|passwd|pwd|token|secret|api[_-]?key|username)\b\1\s*[:=]\s*"
    r"(['\"])(?:\\.|(?!\3).)*\3",
    re.DOTALL,
)
_CREDENTIAL_RE = re.compile(r"(?i)\b(password|passwd|pwd|token|secret|api[_-]?key|username)\b\s*[:=]\s*[^\s,;]+")
_WHITESPACE_RE = re.compile(r"\s+")
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_AUTH_SCHEME_RE = re.compile(r"(?i)^(?:bearer|basic)\s*$")
_BARE_FOLDED_VALUE_RE = re.compile(r"(?i)^[a-z0-9._~+/-]+={0,2}$")
_AUTH_FOLDED_VALUE_RE = re.compile(r"(?i)^(?:bearer|basic)(?:\s+\S+)*$")
_COOKIE_PAIR_RE = re.compile(r"^\S+=[^;]*(?:;\s*\S+=[^;]*)*;?$")


def _folded_header_kind(name: str, remainder: str) -> str | None:
    """Return the continuation kind only when the header value is visibly open."""
    normalized_name = name.lower()
    value = remainder.strip()
    if normalized_name in ("authorization", "proxy-authorization"):
        if not value or _AUTH_SCHEME_RE.fullmatch(value):
            return "auth"
        return None
    if normalized_name == "cookie":
        if not value or value.endswith(";"):
            return "cookie-pair"
        if value.endswith("="):
            return "cookie-value"
    return None


def _is_folded_header_value(kind: str, value: str) -> bool:
    """Return True only for a token shape that belongs to the sensitive header."""
    if kind == "auth":
        return bool(_AUTH_FOLDED_VALUE_RE.fullmatch(value) or _BARE_FOLDED_VALUE_RE.fullmatch(value))
    if kind == "cookie-pair":
        return bool(_COOKIE_PAIR_RE.fullmatch(value))
    if kind == "cookie-value":
        return bool(_BARE_FOLDED_VALUE_RE.fullmatch(value) or _COOKIE_PAIR_RE.fullmatch(value))
    return False


class CdpConnectionPhase(StrEnum):
    """Last known phase of the root CDP transport."""

    DISCONNECTED = "disconnected"
    RESOLVING = "resolving"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    FAILED = "failed"


class CdpConnectionErrorCode(StrEnum):
    """Closed set of diagnostic codes for a connection failure."""

    NOT_CONNECTED = "not_connected"
    TIMEOUT = "timeout"
    CONNECTION_CLOSED = "connection_closed"
    SESSION_NOT_FOUND = "session_not_found"
    PROBE_FAILED = "probe_failed"
    CONNECT_FAILED = "connect_failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CdpProbeToken:
    """Identity of one accepted probe start."""

    generation: int
    sequence: int


@dataclass(frozen=True)
class CdpConnectionHealth:
    """Read-only, JSON-safe health snapshot for one browser session."""

    phase: CdpConnectionPhase
    client_present: bool
    healthy: bool | None
    connection_generation: int
    connect_attempts: int
    reconnect_count: int
    probe_attempts: int
    last_connect_started_at: float | None
    last_connected_at: float | None
    last_probe_at: float | None
    last_success_at: float | None
    last_failure_at: float | None
    last_latency_ms: float | None
    last_error_code: CdpConnectionErrorCode | None
    last_error_type: str | None
    last_error_detail: str | None
    cleanup_pending: bool
    last_cleanup_error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return one additive browser status block."""
        return {
            "phase": self.phase.value,
            "client_present": self.client_present,
            "healthy": self.healthy,
            "connection_generation": self.connection_generation,
            "connect_attempts": self.connect_attempts,
            "reconnect_count": self.reconnect_count,
            "probe_attempts": self.probe_attempts,
            "last_connect_started_at": self.last_connect_started_at,
            "last_connected_at": self.last_connected_at,
            "last_probe_at": self.last_probe_at,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "last_latency_ms": self.last_latency_ms,
            "last_error_code": self.last_error_code.value if self.last_error_code else None,
            "last_error_type": self.last_error_type,
            "last_error_detail": self.last_error_detail,
            "cleanup_pending": self.cleanup_pending,
            "last_cleanup_error": self.last_cleanup_error,
        }


def _redact_header_lines(text: str) -> str:
    """Redact header values and only the continuations their syntax leaves open."""
    lines: list[str] = []
    continuation_kind: str | None = None
    for line in text.split("\n"):
        if (
            continuation_kind is not None
            and line[:1].isspace()
            and _is_folded_header_value(continuation_kind, line.strip())
        ):
            continue
        continuation_kind = None
        header_match = _HEADER_CREDENTIAL_RE.search(line)
        if header_match is not None:
            continuation_kind = _folded_header_kind(header_match.group(2), line[header_match.end() :])
            line = (
                line[: header_match.start()]
                + f"{header_match.group(1)}{header_match.group(2)}{header_match.group(1)}=<redacted>"
            )
        lines.append(line)
    return "\n".join(lines)


def sanitize_error_detail(
    error: BaseException | str | None,
    *,
    max_chars: int | None = MAX_ERROR_CHARS,
    preserve_lines: bool = False,
) -> str | None:
    """Return bounded diagnostic text with common credentials removed."""
    if error is None:
        return None
    text = str(error) if not isinstance(error, str) else error
    text = _LINE_BREAK_RE.sub("\n", text)
    text = _URL_RE.sub("<url>", text)
    text = _QUOTED_HEADER_CREDENTIAL_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(1)}=<redacted>",
        text,
    )
    text = _redact_header_lines(text)
    text = _QUOTED_CREDENTIAL_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(1)}=<redacted>",
        text,
    )
    text = _CREDENTIAL_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _CONTROL_RE.sub(" ", text)
    if preserve_lines:
        preserved_lines: list[str] = []
        for line in text.split("\n"):
            leading = line[: len(line) - len(line.lstrip(" \t"))]
            body = _HORIZONTAL_WHITESPACE_RE.sub(" ", line[len(leading) :]).rstrip()
            preserved_lines.append(leading + body)
        text = "\n".join(preserved_lines).strip()
    else:
        text = _WHITESPACE_RE.sub(" ", text).strip()
    if max_chars is not None:
        if max_chars <= 0:
            return ""
        if len(text) > max_chars:
            text = text[:max_chars] if max_chars < 3 else text[: max_chars - 3] + "..."
    return text


_sanitize_error = sanitize_error_detail


def bound_text_with_suffix(
    text: str | None,
    suffix: str,
    *,
    max_chars: int = MAX_ERROR_CHARS,
) -> str | None:
    """Keep a bounded prefix and reserve enough room for the suffix."""
    if max_chars <= 0:
        return ""
    safe_suffix = sanitize_error_detail(suffix, max_chars=max_chars) or ""
    if not safe_suffix:
        return sanitize_error_detail(text, max_chars=max_chars)
    if len(safe_suffix) >= max_chars:
        return safe_suffix
    prefix_budget = max_chars - len(safe_suffix) - 1
    safe_prefix = sanitize_error_detail(text, max_chars=prefix_budget) or ""
    return f"{safe_prefix} {safe_suffix}".strip() if safe_prefix else safe_suffix


def _error_type(error: BaseException | str | None) -> str | None:
    if error is None:
        return None
    return type(error).__name__ if isinstance(error, BaseException) else "str"


class CdpConnectionHealthTracker:
    """Mutable health observation owned by one BrowserSession."""

    def __init__(self) -> None:
        self._phase = CdpConnectionPhase.DISCONNECTED
        self._healthy: bool | None = None
        self._generation = 0
        self._connect_attempts = 0
        self._reconnect_count = 0
        self._probe_attempts = 0
        self._probe_sequence = 0
        self._last_connect_started_at: float | None = None
        self._last_connected_at: float | None = None
        self._last_probe_at: float | None = None
        self._last_success_at: float | None = None
        self._last_failure_at: float | None = None
        self._last_latency_ms: float | None = None
        self._last_error_code: CdpConnectionErrorCode | None = None
        self._last_error_type: str | None = None
        self._last_error_detail: str | None = None
        self._cleanup_pending = False
        self._last_cleanup_error: str | None = None

    @property
    def connection_generation(self) -> int:
        """Return the current connect-attempt generation."""
        return self._generation

    @property
    def cleanup_pending(self) -> bool:
        """Return True while the session retains cleanup ownership."""
        return self._cleanup_pending

    def classify_error(self, error: BaseException, *, connect: bool) -> CdpConnectionErrorCode:
        """Map one exception to the fixed health error vocabulary."""
        if isinstance(error, asyncio.CancelledError):
            return CdpConnectionErrorCode.CANCELLED
        if isinstance(error, TimeoutError):
            return CdpConnectionErrorCode.TIMEOUT
        text = str(error).lower()
        if "-32001" in text or "session with given id not found" in text:
            return CdpConnectionErrorCode.SESSION_NOT_FOUND
        if any(
            marker in text
            for marker in (
                "connectionclosed",
                "connection closed",
                "no close frame",
                "websocket",
            )
        ):
            return CdpConnectionErrorCode.CONNECTION_CLOSED
        return CdpConnectionErrorCode.CONNECT_FAILED if connect else CdpConnectionErrorCode.PROBE_FAILED

    def begin_connect(self) -> int:
        """Start one serialized connect attempt and return its generation."""
        self._generation += 1
        self._connect_attempts += 1
        if self._connect_attempts > 1:
            self._reconnect_count += 1
        self._phase = CdpConnectionPhase.RESOLVING
        self._healthy = None
        self._last_connect_started_at = time.time()
        self._probe_sequence += 1
        self._last_error_code = None
        self._last_error_type = None
        self._last_error_detail = None
        return self._generation

    def mark_connecting(self, generation: int) -> None:
        """Record that a root client object is being created."""
        if generation != self._generation:
            return
        self._phase = CdpConnectionPhase.CONNECTING
        self._healthy = None

    def mark_connected(self, generation: int) -> None:
        """Record successful root and session-manager setup."""
        if generation != self._generation:
            return
        now = time.time()
        self._phase = CdpConnectionPhase.CONNECTED
        self._healthy = True
        self._last_connected_at = now
        self._last_success_at = now
        self._last_error_code = None
        self._last_error_type = None
        self._last_error_detail = None
        self._cleanup_pending = False
        self._last_cleanup_error = None

    def mark_connect_failed(
        self,
        generation: int,
        error: BaseException,
        *,
        client_present: bool,
        cancelled: bool = False,
        worker_cancelled: bool = False,
    ) -> None:
        """Record a failed attempt. Client presence is rendered from the session."""
        del client_present
        if generation != self._generation:
            return
        now = time.time()
        self._phase = CdpConnectionPhase.FAILED
        self._healthy = False
        self._last_failure_at = now
        self._last_error_code = (
            CdpConnectionErrorCode.CANCELLED
            if cancelled
            else CdpConnectionErrorCode.CONNECT_FAILED
            if worker_cancelled
            else self.classify_error(error, connect=True)
        )
        self._last_error_type = _error_type(error)
        self._last_error_detail = sanitize_error_detail(error)

    def begin_probe(self, generation: int) -> CdpProbeToken:
        """Mark one accepted probe start and return its fencing token."""
        self._probe_attempts += 1
        self._probe_sequence += 1
        self._last_probe_at = time.time()
        return CdpProbeToken(generation=generation, sequence=self._probe_sequence)

    def _token_current(self, token: CdpProbeToken) -> bool:
        return token.generation == self._generation and token.sequence == self._probe_sequence

    def complete_probe_success(self, token: CdpProbeToken, *, latency_ms: float) -> None:
        """Apply a successful probe only when its token is still current."""
        if not self._token_current(token):
            return
        now = time.time()
        self._phase = CdpConnectionPhase.CONNECTED
        self._healthy = True
        self._last_success_at = now
        self._last_latency_ms = latency_ms
        self._last_error_code = None
        self._last_error_type = None
        self._last_error_detail = None

    def complete_probe_failure(
        self,
        token: CdpProbeToken,
        error: BaseException,
        *,
        latency_ms: float,
        client_present: bool,
        worker_cancelled: bool = False,
    ) -> None:
        """Apply a failed probe only when its token is still current."""
        if not self._token_current(token):
            return
        now = time.time()
        self._phase = CdpConnectionPhase.DEGRADED if client_present else CdpConnectionPhase.DISCONNECTED
        self._healthy = False
        self._last_failure_at = now
        self._last_latency_ms = latency_ms
        self._last_error_code = (
            CdpConnectionErrorCode.NOT_CONNECTED
            if not client_present
            else CdpConnectionErrorCode.PROBE_FAILED
            if worker_cancelled
            else self.classify_error(error, connect=False)
        )
        self._last_error_type = _error_type(error)
        self._last_error_detail = sanitize_error_detail(error)

    def discard_probe(self, token: CdpProbeToken) -> None:
        """Invalidate one probe whose root changed before completion."""
        if self._token_current(token):
            self._probe_sequence += 1

    def mark_cleanup_failure(self, error: BaseException | str) -> None:
        """Record that a cleanup failure retained session-owned resources."""
        self._cleanup_pending = True
        self._last_cleanup_error = sanitize_error_detail(error)

    def mark_cleanup_success(self) -> None:
        """Clear retention only after every owned resource is released."""
        self._cleanup_pending = False
        self._last_cleanup_error = None

    def snapshot(self, *, client_present: bool, connection_expected: bool) -> CdpConnectionHealth:
        """Return a lifecycle-aware snapshot without performing I/O."""
        phase = self._phase
        healthy = self._healthy
        if phase is not CdpConnectionPhase.FAILED and (
            not connection_expected
            or (
                not client_present
                and phase
                not in (
                    CdpConnectionPhase.RESOLVING,
                    CdpConnectionPhase.CONNECTING,
                    CdpConnectionPhase.DISCONNECTED,
                )
            )
        ):
            phase = CdpConnectionPhase.DISCONNECTED
            healthy = None
        return CdpConnectionHealth(
            phase=phase,
            client_present=client_present,
            healthy=healthy,
            connection_generation=self._generation,
            connect_attempts=self._connect_attempts,
            reconnect_count=self._reconnect_count,
            probe_attempts=self._probe_attempts,
            last_connect_started_at=self._last_connect_started_at,
            last_connected_at=self._last_connected_at,
            last_probe_at=self._last_probe_at,
            last_success_at=self._last_success_at,
            last_failure_at=self._last_failure_at,
            last_latency_ms=self._last_latency_ms,
            last_error_code=self._last_error_code,
            last_error_type=self._last_error_type,
            last_error_detail=self._last_error_detail,
            cleanup_pending=self._cleanup_pending,
            last_cleanup_error=self._last_cleanup_error,
        )
