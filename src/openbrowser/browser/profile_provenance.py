"""Explicit provenance for the effective browser profile.

The daemon and MCP merge adapter defaults, stored config, environment
overrides, and managed paths into one BrowserProfile. This module records
where each effective value came from without changing the merge result.

Sensitive or unbounded fields expose a mode and a source, never the value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

MAX_TEXT = 300

VALUE_FIELDS = (
    "user_data_dir",
    "headless",
    "channel",
    "executable_path",
    "downloads_path",
    "keep_alive",
    "wait_between_actions",
    "device_scale_factor",
    "disable_security",
)
MODE_FIELDS = ("storage_state", "args", "proxy")
ALL_FIELDS = VALUE_FIELDS + MODE_FIELDS


class ProfileValueSource(StrEnum):
    """Where an effective profile value came from."""

    DEFAULT = "default"
    CONFIG = "config"
    ENVIRONMENT = "environment"
    MANAGED = "managed"
    CLI = "cli"


@dataclass(frozen=True)
class ProfileField:
    """One effective field and the source that set it."""

    source: ProfileValueSource
    value: Any | None = None
    mode: str | None = None
    count: int | None = None


def _bounded_value(value: Any) -> Any:
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "..."
    if isinstance(value, dict):
        return f"<dict len={len(value)}>"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"<{type(value).__name__} len={len(value)}>"
    return f"<{type(value).__name__}>"


def _mode(name: str, value: Any) -> tuple[str, int | None]:
    if name == "storage_state":
        if value is None:
            return "none", None
        if isinstance(value, (str, Path)):
            return "file", None
        if isinstance(value, dict):
            return "memory", None
        return "configured", None
    if name == "proxy":
        if value is None:
            return "none", None
        if isinstance(value, dict):
            return ("none" if not value else "configured"), None
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                if not model_dump(exclude_none=True):
                    return "none", None
            except (TypeError, ValueError):
                pass
        return "configured", None
    if name == "args":
        if value is None:
            return "none", 0
        if isinstance(value, (list, tuple, set, frozenset)):
            return ("none" if not value else "configured"), len(value)
        return "configured", None
    return ("none" if value is None else "configured"), None


@dataclass
class ProfileProvenance:
    """A bounded, JSON-safe trace of the effective profile fields."""

    fields: dict[str, ProfileField] = field(default_factory=dict)

    def record(self, name: str, value: Any, source: ProfileValueSource) -> None:
        if name not in ALL_FIELDS:
            return
        if name in MODE_FIELDS:
            mode, count = _mode(name, value)
            self.fields[name] = ProfileField(source=source, mode=mode, count=count)
            return
        self.fields[name] = ProfileField(source=source, value=_bounded_value(value))

    def record_mapping(self, mapping: dict[str, Any], source: ProfileValueSource) -> None:
        for name in ALL_FIELDS:
            if name in mapping:
                self.record(name, mapping[name], source)

    def record_defaults(self, defaults: dict[str, Any]) -> None:
        """Record effective defaults only when a higher layer did not set them."""
        for name, value in defaults.items():
            if name not in self.fields:
                self.record(name, value, ProfileValueSource.DEFAULT)

    def refresh(self, name: str, value: Any) -> None:
        """Update a recorded value after BrowserProfile normalizes it."""
        entry = self.fields.get(name)
        if entry is None:
            return
        if name in MODE_FIELDS:
            mode, count = _mode(name, value)
            self.fields[name] = replace(entry, mode=mode, count=count)
        else:
            self.fields[name] = replace(entry, value=_bounded_value(value))

    def refresh_from_profile(self, profile: Any) -> None:
        """Refresh recorded fields from the validated BrowserProfile."""
        for name in ALL_FIELDS:
            if name in self.fields and hasattr(profile, name):
                self.refresh(name, getattr(profile, name))

    def record_changes(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        source: ProfileValueSource,
        *,
        names: tuple[str, ...] = ("user_data_dir", "storage_state"),
    ) -> None:
        for name in names:
            if after.get(name) != before.get(name):
                self.record(name, after.get(name), source)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name in ALL_FIELDS:
            entry = self.fields.get(name)
            if entry is None:
                continue
            item: dict[str, Any] = {"source": entry.source.value}
            if entry.mode is not None:
                item["mode"] = entry.mode
            else:
                item["value"] = entry.value
            if entry.count is not None:
                item["count"] = entry.count
            result[name] = item
        return result
