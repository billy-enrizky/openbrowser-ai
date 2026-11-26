"""Closed matched-engine session guard for browser benchmark arms.

The guard discovers one main browser process from exact argv data, binds it to
a pinned engine through ``engine_binding.capture_process_binding``, and
re-verifies the same session through ``engine_binding.verify_process_binding``.
It never recaches a changed browser. Public diagnostics contain no absolute
paths, argv, or raw exception strings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.interactive.adapters.launch_proof import canonical_profile_marker
from benchmarks.interactive.engine_binding import (
    EngineBindingError,
    EnginePin,
    ProcessBinding,
    capture_process_binding,
    verify_process_binding,
)

SESSION_SCHEMA = "openbrowser-session-observation-v1"
PROOF_SCHEMA = "openbrowser-session-proof-v1"
_USER_DATA_DIR_PREFIX = "--user-data-dir="
_BROWSER_BASENAMES = frozenset(
    {
        "brave",
        "brave browser",
        "brave-browser",
        "chrome",
        "chrome canary",
        "chrome-headless-shell",
        "chromium",
        "chromium-browser",
        "google chrome",
        "google chrome canary",
        "google chrome for testing",
        "google-chrome",
        "google-chrome-beta",
        "google-chrome-canary",
        "google-chrome-dev",
        "google-chrome-stable",
        "google-chrome-unstable",
        "headless_shell",
        "microsoft edge",
        "msedge",
    }
)

STATUS_VALID = "valid"
STATUS_PIN_CHANGED = "pin_changed"
STATUS_PIN_UNAVAILABLE = "pin_unavailable"
STATUS_PROCESS_MISSING = "process_missing"
STATUS_PROCESS_DENIED = "process_denied"
STATUS_PROCESS_REPLACED = "process_replaced"
STATUS_EXECUTABLE_CHANGED = "executable_changed"
STATUS_KEYCHAIN_MISSING = "keychain_missing"
STATUS_DUPLICATE_MAIN = "duplicate_main"
STATUS_MARKER_CHANGED = "marker_changed"
STATUS_MALFORMED = "malformed"
STATUS_UNKNOWN = "unknown"

ProcessTable = Callable[[], Sequence[tuple[int, Sequence[str]]]]
ProcessFactory = Callable[[int], Any]
StatReader = Callable[[Path], Any]


@dataclass(frozen=True)
class SessionProof:
    schema: str
    lane: str
    revision: str
    version: str
    launcher_sha256: str
    engine_image_sha256: str
    profile_marker: str
    binding: ProcessBinding
    keychain_verified: bool

    @property
    def profile_digest(self) -> str:
        return hashlib.sha256(self.profile_marker.encode("utf-8")).hexdigest()

    def public_proof(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lane": self.lane,
            "revision": self.revision,
            "version": self.version,
            "launcher_sha256": self.launcher_sha256,
            "engine_image_sha256": self.engine_image_sha256,
            "pid": self.binding.pid,
            "created_at": self.binding.created_at,
            "executable_basename": self.binding.executable_basename,
            "keychain_verified": self.keychain_verified,
            "profile_digest": self.profile_digest,
        }


@dataclass(frozen=True)
class SessionObservation:
    status: str
    proof: SessionProof | None = None
    pid: int | None = None
    created_at: float | None = None
    executable_basename: str | None = None
    keychain_verified: bool | None = None
    profile_digest: str | None = None
    lane: str | None = None
    revision: str | None = None
    version: str | None = None
    launcher_sha256: str | None = None
    engine_image_sha256: str | None = None

    @property
    def eligible(self) -> bool:
        return self.status == STATUS_VALID and self.proof is not None

    def public_proof(self) -> dict[str, Any]:
        return {
            "schema": SESSION_SCHEMA,
            "status": self.status,
            "eligible": self.eligible,
            "lane": self.lane,
            "revision": self.revision,
            "version": self.version,
            "launcher_sha256": self.launcher_sha256,
            "engine_image_sha256": self.engine_image_sha256,
            "pid": self.pid,
            "created_at": self.created_at,
            "executable_basename": self.executable_basename,
            "keychain_verified": self.keychain_verified,
            "profile_digest": self.profile_digest,
        }


@dataclass(frozen=True)
class _Candidate:
    pid: int
    marker: str
    keychain: bool


@dataclass(frozen=True)
class _Scan:
    status: str
    candidate: _Candidate | None = None


def observe_session(
    pin: EnginePin,
    profile_marker: str | Path,
    *,
    process_table: ProcessTable | None = None,
    process_factory: ProcessFactory | None = None,
    stat_reader: StatReader | None = None,
) -> SessionObservation:
    """Discover and bind one main browser session for a pinned engine."""
    if not isinstance(pin, EnginePin):
        return _observation(STATUS_MALFORMED)
    expected = _canonical_marker(profile_marker)
    if expected is None:
        return _observation(STATUS_MALFORMED, pin=pin)
    digest = _profile_digest(expected)
    fence = _fence_status(pin, stat_reader)
    if fence is not None:
        return _observation(fence, pin=pin, profile_digest=digest)

    table = process_table or _default_process_table
    try:
        rows = table()
    except Exception:
        return _observation(STATUS_UNKNOWN, pin=pin, profile_digest=digest)
    scan = _scan_main(rows, expected)
    if scan.status != STATUS_VALID or scan.candidate is None:
        return _observation(scan.status, pin=pin, profile_digest=digest)
    candidate = scan.candidate
    if not candidate.keychain:
        return _observation(
            STATUS_KEYCHAIN_MISSING,
            pin=pin,
            pid=candidate.pid,
            keychain_verified=False,
            profile_digest=digest,
        )

    try:
        binding = capture_process_binding(
            pin,
            candidate.pid,
            process_factory=process_factory,
            stat_reader=stat_reader,
        )
    except EngineBindingError as error:
        return _binding_failure(error.reason, pin=pin, candidate=candidate, profile_digest=digest)

    try:
        rows_after = table()
    except Exception:
        return _observation(
            STATUS_UNKNOWN,
            pin=pin,
            profile_digest=digest,
        )
    after = _scan_main(rows_after, expected)
    if after.status != STATUS_VALID or after.candidate is None:
        status = after.status
        if status == STATUS_PROCESS_MISSING and _pid_is_main(rows_after, binding.pid):
            status = STATUS_MARKER_CHANGED
        return _observation(
            status,
            pin=pin,
            pid=binding.pid,
            created_at=binding.created_at,
            executable_basename=binding.executable_basename,
            keychain_verified=True,
            profile_digest=digest,
        )
    if after.candidate.pid != candidate.pid:
        return _observation(
            STATUS_PROCESS_REPLACED,
            pin=pin,
            pid=after.candidate.pid,
            profile_digest=digest,
        )
    if not after.candidate.keychain:
        return _observation(
            STATUS_KEYCHAIN_MISSING,
            pin=pin,
            pid=binding.pid,
            created_at=binding.created_at,
            executable_basename=binding.executable_basename,
            keychain_verified=False,
            profile_digest=digest,
        )

    binding_status = _binding_status(
        verify_process_binding(
            binding,
            pin,
            process_factory=process_factory,
            stat_reader=stat_reader,
        )
    )
    if binding_status != STATUS_VALID:
        if binding_status == STATUS_UNKNOWN:
            return _observation(STATUS_UNKNOWN, pin=pin, profile_digest=digest)
        return _observation(
            binding_status,
            pin=pin,
            pid=binding.pid,
            created_at=binding.created_at,
            executable_basename=binding.executable_basename,
            keychain_verified=True,
            profile_digest=digest,
        )

    fence = _fence_status(pin, stat_reader)
    if fence is not None:
        return _observation(
            fence,
            pin=pin,
            pid=binding.pid,
            created_at=binding.created_at,
            executable_basename=binding.executable_basename,
            keychain_verified=True,
            profile_digest=digest,
        )

    proof = SessionProof(
        schema=PROOF_SCHEMA,
        lane=pin.lane,
        revision=pin.revision,
        version=pin.version,
        launcher_sha256=pin.launcher_sha256,
        engine_image_sha256=pin.engine_image_sha256,
        profile_marker=expected,
        binding=binding,
        keychain_verified=True,
    )
    return _observation(STATUS_VALID, pin=pin, proof=proof)


def verify_session(
    proof: SessionProof,
    pin: EnginePin,
    *,
    process_table: ProcessTable | None = None,
    process_factory: ProcessFactory | None = None,
    stat_reader: StatReader | None = None,
) -> SessionObservation:
    """Re-discover and verify a known session without recaching it."""
    if not isinstance(proof, SessionProof) or not isinstance(pin, EnginePin):
        return _observation(STATUS_MALFORMED)
    if not _valid_proof(proof):
        return _observation(STATUS_MALFORMED, pin=pin if isinstance(pin, EnginePin) else None)
    if not _same_pin(proof, pin):
        return _observation(STATUS_PIN_CHANGED, pin=pin, profile_digest=proof.profile_digest)

    observed = observe_session(
        pin,
        proof.profile_marker,
        process_table=process_table,
        process_factory=process_factory,
        stat_reader=stat_reader,
    )
    if observed.status != STATUS_VALID or observed.proof is None:
        if observed.status == STATUS_PROCESS_MISSING:
            table = process_table or _default_process_table
            try:
                rows = table()
            except Exception:
                rows = None
            main_marker = _main_marker_for_pid(rows, proof.binding.pid) if rows is not None else None
            if main_marker is not None and main_marker != proof.profile_marker:
                return _observation(
                    STATUS_MARKER_CHANGED,
                    pin=pin,
                    pid=proof.binding.pid,
                    created_at=proof.binding.created_at,
                    executable_basename=proof.binding.executable_basename,
                    keychain_verified=True,
                    profile_digest=proof.profile_digest,
                )
        return observed

    binding_status = verify_process_binding(
        proof.binding,
        pin,
        process_factory=process_factory,
        stat_reader=stat_reader,
    )
    mapped = _binding_status(binding_status)
    if mapped != STATUS_VALID:
        return _observation(
            mapped,
            pin=pin,
            profile_digest=proof.profile_digest,
        )

    fresh = observed.proof
    if fresh.binding.pid != proof.binding.pid or fresh.binding.created_at != proof.binding.created_at:
        return _observation(
            STATUS_PROCESS_REPLACED,
            pin=pin,
            pid=fresh.binding.pid,
            created_at=fresh.binding.created_at,
            executable_basename=fresh.binding.executable_basename,
            keychain_verified=True,
            profile_digest=fresh.profile_digest,
        )
    if fresh.binding.executable_basename != proof.binding.executable_basename:
        return _observation(
            STATUS_EXECUTABLE_CHANGED,
            pin=pin,
            pid=fresh.binding.pid,
            created_at=fresh.binding.created_at,
            executable_basename=fresh.binding.executable_basename,
            keychain_verified=True,
            profile_digest=fresh.profile_digest,
        )
    if fresh.profile_marker != proof.profile_marker:
        return _observation(
            STATUS_MARKER_CHANGED,
            pin=pin,
            pid=fresh.binding.pid,
            created_at=fresh.binding.created_at,
            executable_basename=fresh.binding.executable_basename,
            keychain_verified=True,
            profile_digest=fresh.profile_digest,
        )
    return _observation(STATUS_VALID, pin=pin, proof=proof)


def _scan_main(rows: Any, expected_marker: str) -> _Scan:
    try:
        normalized = _normalize_table(rows)
    except ValueError:
        return _Scan(STATUS_MALFORMED)
    candidates: list[_Candidate] = []
    for pid, argv in normalized:
        if _is_helper(argv) or not _is_browser_process(argv):
            continue
        present, marker, malformed = _marker_from_argv(argv)
        if not present:
            continue
        if malformed or marker is None:
            return _Scan(STATUS_MALFORMED)
        if marker != expected_marker:
            continue
        candidates.append(_Candidate(pid=pid, marker=marker, keychain="--use-mock-keychain" in argv))
    if not candidates:
        return _Scan(STATUS_PROCESS_MISSING)
    if len(candidates) > 1:
        return _Scan(STATUS_DUPLICATE_MAIN)
    return _Scan(STATUS_VALID, candidates[0])


def _normalize_table(rows: Any) -> list[tuple[int, list[str]]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("table")
    normalized: list[tuple[int, list[str]]] = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("row")
        pid, argv = row
        if type(pid) is not int or pid <= 0:
            raise ValueError("pid")
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
            raise ValueError("argv")
        if any(not isinstance(token, str) for token in argv):
            raise ValueError("token")
        normalized.append((pid, list(argv)))
    return normalized


def _marker_from_argv(argv: Sequence[str]) -> tuple[bool, str | None, bool]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--user-data-dir":
            if index + 1 >= len(argv):
                return True, None, True
            values.append(argv[index + 1])
            index += 2
            continue
        if token.startswith(_USER_DATA_DIR_PREFIX):
            values.append(token[len(_USER_DATA_DIR_PREFIX) :])
        index += 1
    if not values:
        return False, None, False
    canonical: set[str] = set()
    for value in values:
        if not value:
            return True, None, True
        marker = _canonical_marker(value)
        if marker is None:
            return True, None, True
        canonical.add(marker)
    if len(canonical) != 1:
        return True, None, True
    return True, next(iter(canonical)), False


def _is_helper(argv: Sequence[str]) -> bool:
    if any(token == "--type" or token.startswith("--type=") for token in argv):
        return True
    name = Path(argv[0]).name.lower()
    return "helper" in name or "crashpad" in name


def _is_browser_process(argv: Sequence[str]) -> bool:
    name = Path(argv[0]).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name in _BROWSER_BASENAMES


def _pid_is_main(rows: Any, pid: int) -> bool:
    try:
        normalized = _normalize_table(rows)
    except ValueError:
        return False
    return any(row_pid == pid and not _is_helper(argv) and _is_browser_process(argv) for row_pid, argv in normalized)


def _main_marker_for_pid(rows: Any, pid: int) -> str | None:
    try:
        normalized = _normalize_table(rows)
    except ValueError:
        return None
    for row_pid, argv in normalized:
        if row_pid != pid or _is_helper(argv) or not _is_browser_process(argv):
            continue
        present, marker, malformed = _marker_from_argv(argv)
        if present and not malformed:
            return marker
    return None


def _canonical_marker(value: Any) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    return canonical_profile_marker(str(value))


def _profile_digest(marker: str) -> str:
    return hashlib.sha256(marker.encode("utf-8")).hexdigest()


def _pin_status(reason: str) -> str:
    if reason == "pin_changed":
        return STATUS_PIN_CHANGED
    if reason in {"malformed_pin", "malformed_stat"}:
        return STATUS_MALFORMED
    return STATUS_PIN_UNAVAILABLE


def _fence_status(pin: EnginePin, stat_reader: StatReader | None) -> str | None:
    try:
        pin.fence(stat_reader=stat_reader)
    except EngineBindingError as error:
        return _pin_status(error.reason)
    except (OSError, TypeError, ValueError):
        return STATUS_MALFORMED
    return None


def _binding_status(status: str) -> str:
    return {
        "valid": STATUS_VALID,
        "malformed_binding": STATUS_MALFORMED,
        "binding_mismatch": STATUS_PIN_CHANGED,
        "pin_changed": STATUS_PIN_CHANGED,
        "pin_unavailable": STATUS_PIN_UNAVAILABLE,
        "process_missing": STATUS_PROCESS_MISSING,
        "process_denied": STATUS_PROCESS_DENIED,
        "process_replaced": STATUS_PROCESS_REPLACED,
        "executable_changed": STATUS_EXECUTABLE_CHANGED,
        "process_unknown": STATUS_UNKNOWN,
    }.get(status, STATUS_UNKNOWN)


def _binding_failure(
    reason: str,
    *,
    pin: EnginePin,
    candidate: _Candidate,
    profile_digest: str,
) -> SessionObservation:
    status = {
        "pin_changed": STATUS_PIN_CHANGED,
        "pin_unavailable": STATUS_PIN_UNAVAILABLE,
        "malformed_pin": STATUS_MALFORMED,
        "malformed_pid": STATUS_MALFORMED,
        "process_gone": STATUS_PROCESS_MISSING,
        "process_denied": STATUS_PROCESS_DENIED,
        "process_replaced": STATUS_PROCESS_REPLACED,
        "executable_mismatch": STATUS_EXECUTABLE_CHANGED,
        "executable_missing": STATUS_EXECUTABLE_CHANGED,
        "executable_changed": STATUS_EXECUTABLE_CHANGED,
        "process_unknown": STATUS_UNKNOWN,
    }.get(reason, STATUS_UNKNOWN)
    if status == STATUS_UNKNOWN:
        return _observation(STATUS_UNKNOWN, pin=pin, profile_digest=profile_digest)
    return _observation(
        status,
        pin=pin,
        pid=candidate.pid,
        keychain_verified=True,
        profile_digest=profile_digest,
    )


def _same_pin(proof: SessionProof, pin: EnginePin) -> bool:
    return (
        proof.lane == pin.lane
        and proof.revision == pin.revision
        and proof.version == pin.version
        and proof.launcher_sha256 == pin.launcher_sha256
        and proof.engine_image_sha256 == pin.engine_image_sha256
    )


def _valid_proof(proof: SessionProof) -> bool:
    if proof.schema != PROOF_SCHEMA:
        return False
    if not all(
        isinstance(value, str) and value
        for value in (
            proof.lane,
            proof.revision,
            proof.version,
            proof.launcher_sha256,
            proof.engine_image_sha256,
        )
    ):
        return False
    if not isinstance(proof.profile_marker, str):
        return False
    canonical = _canonical_marker(proof.profile_marker)
    if canonical is None or canonical != proof.profile_marker:
        return False
    if not isinstance(proof.binding, ProcessBinding):
        return False
    return proof.keychain_verified is True


def _observation(
    status: str,
    *,
    pin: EnginePin | None = None,
    proof: SessionProof | None = None,
    pid: int | None = None,
    created_at: float | None = None,
    executable_basename: str | None = None,
    keychain_verified: bool | None = None,
    profile_digest: str | None = None,
) -> SessionObservation:
    if proof is not None:
        return SessionObservation(
            status=status,
            proof=proof,
            pid=proof.binding.pid,
            created_at=proof.binding.created_at,
            executable_basename=proof.binding.executable_basename,
            keychain_verified=proof.keychain_verified,
            profile_digest=proof.profile_digest,
            lane=proof.lane,
            revision=proof.revision,
            version=proof.version,
            launcher_sha256=proof.launcher_sha256,
            engine_image_sha256=proof.engine_image_sha256,
        )
    return SessionObservation(
        status=status,
        proof=None,
        pid=pid,
        created_at=created_at,
        executable_basename=executable_basename,
        keychain_verified=keychain_verified,
        profile_digest=profile_digest,
        lane=pin.lane if pin is not None else None,
        revision=pin.revision if pin is not None else None,
        version=pin.version if pin is not None else None,
        launcher_sha256=pin.launcher_sha256 if pin is not None else None,
        engine_image_sha256=pin.engine_image_sha256 if pin is not None else None,
    )


def _default_process_table() -> list[tuple[int, list[str]]]:
    import psutil

    rows: list[tuple[int, list[str]]] = []
    for process in psutil.process_iter(["pid"]):
        try:
            pid = process.pid
            cmdline = process.cmdline()
        except (psutil.Error, OSError):
            continue
        if not cmdline:
            continue
        rows.append((pid, list(cmdline)))
    return rows
