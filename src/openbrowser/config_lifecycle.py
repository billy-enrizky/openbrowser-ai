"""Explicit lifecycle types for the OpenBrowser configuration file."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openbrowser.config import DBStyleConfigJSON


class ConfigFilePhase(StrEnum):
    """State of the config file at one point in time."""

    READY = "ready"
    CREATED = "created"
    LEGACY_FORMAT = "legacy_format"
    INVALID = "invalid"
    READ_FAILED = "read_failed"
    WRITE_FAILED = "write_failed"


_ALLOWED_START_PHASES = frozenset({ConfigFilePhase.READY, ConfigFilePhase.CREATED})


@dataclass(frozen=True)
class ConfigFileSnapshot:
    """One JSON-safe view of the config file lifecycle."""

    path: str
    resolved_path: str
    phase: ConfigFilePhase
    error_code: str | None
    hint: str | None
    browser_start_allowed: bool
    file_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "resolved_path": self.resolved_path,
            "phase": self.phase.value,
            "error_code": self.error_code,
            "hint": self.hint,
            "browser_start_allowed": self.browser_start_allowed,
            "file_version": self.file_version,
        }


@dataclass(frozen=True)
class ConfigLoadResult:
    """The parsed config plus the snapshot that describes its source."""

    config: DBStyleConfigJSON
    snapshot: ConfigFileSnapshot


class ConfigStartupError(RuntimeError):
    """Raised when a browser may not start from the current config state."""

    def __init__(
        self,
        snapshot: ConfigFileSnapshot,
        *,
        code: str | None = None,
        hint: str | None = None,
        detail: str | None = None,
    ):
        self.snapshot = snapshot
        self.code = code or snapshot.error_code or "config_unavailable"
        self.path = snapshot.path
        self.hint = hint or snapshot.hint or "Inspect the config file, then restart OpenBrowser."
        self.detail = detail
        message = f"{self.code}: {self.path}. {self.hint}"
        super().__init__(f"{detail}\n{message}" if detail else message)


def start_allowed(snapshot: ConfigFileSnapshot) -> bool:
    """Return whether a browser may start from this snapshot."""
    return snapshot.phase in _ALLOWED_START_PHASES


def restart_required(startup: ConfigFileSnapshot, current: ConfigFileSnapshot) -> bool:
    """Return whether a long-lived process must restart before another start."""
    if not start_allowed(startup) or not start_allowed(current):
        return True
    return startup.file_version != current.file_version
