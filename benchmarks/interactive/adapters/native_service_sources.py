"""Decode native ownership candidates without filesystem or process access.

Candidates need a separate live PID, create-time, executable, and scope proof.
The browser-use pending JSON record and ready decimal record are both native
formats. Neither format establishes process ownership by itself.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

MAX_PID_RECORD_BYTES = 4096


class PidSource(StrEnum):
    OPENBROWSER = "openbrowser"
    AGENT_BROWSER = "agent-browser"
    BROWSER_USE = "browser-use"


class SourceError(ValueError):
    """A closed reason for an unusable native ownership candidate."""


@dataclass(frozen=True)
class PidRecord:
    pid: int
    record_format: Literal["decimal", "json"]
    started: str | int | None


@dataclass(frozen=True)
class ProcessCommand:
    pid: int
    executable: str
    argv: tuple[str, ...]


def _json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SourceError("malformed_record")
        result[key] = value
    return result


def _reject_constant(_value: str) -> object:
    raise SourceError("malformed_record")


def parse_native_pid_record(payload: object, *, source: object) -> PidRecord:
    """Parse bounded bytes and preserve a native start fingerprint unchanged."""
    if not isinstance(source, PidSource):
        raise SourceError("invalid_source")
    if not isinstance(payload, bytes):
        raise SourceError("invalid_payload")
    if len(payload) > MAX_PID_RECORD_BYTES:
        raise SourceError("record_too_large")
    try:
        text = payload.decode("utf-8").strip(" \t\r\n")
    except UnicodeDecodeError as error:
        raise SourceError("malformed_record") from error
    if re.fullmatch(r"[1-9][0-9]*", text):
        return PidRecord(int(text), "decimal", None)
    if source is not PidSource.BROWSER_USE:
        raise SourceError("malformed_record")
    try:
        record = json.loads(text, object_pairs_hook=_json_pairs, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        raise SourceError("malformed_record") from error
    if not isinstance(record, dict) or set(record) != {"pid", "started"}:
        raise SourceError("malformed_record")
    pid, started = record["pid"], record["started"]
    if type(pid) is not int or pid <= 0:
        raise SourceError("invalid_pid")
    if isinstance(started, str):
        if not started.strip() or "\x00" in started or len(started) > 512:
            raise SourceError("invalid_fingerprint")
    elif type(started) is not int or started <= 0:
        raise SourceError("invalid_fingerprint")
    return PidRecord(pid, "json", started)


def _absolute_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\x00" in value or not os.path.isabs(value):
        return None
    return os.path.normcase(os.path.normpath(value))


def select_playwright_daemon(
    processes: Sequence[ProcessCommand],
    *,
    executable: str,
    daemon_entry: str,
    session: str,
    profile: str,
    config: str,
) -> ProcessCommand:
    """Select one exact native daemon candidate, retaining its original tokens.

    The caller supplies canonical paths from its pinned installation and owned
    context. The actual executable can differ from the spelling of argv[0].
    No substring, shell parsing, or first-match fallback establishes ownership.
    """
    paths = [executable, daemon_entry, profile, config]
    if any((canonical := _absolute_path(value)) is None or canonical != value for value in paths):
        raise SourceError("invalid_expectation")
    if not isinstance(session, str) or not session or "\x00" in session:
        raise SourceError("invalid_expectation")
    if not isinstance(processes, (list, tuple)):
        raise SourceError("invalid_process")
    matches = []
    expected = (daemon_entry, session, "--profile=" + profile, "--config=" + config)
    for process in processes:
        if not isinstance(process, ProcessCommand) or type(process.pid) is not int or process.pid <= 0:
            raise SourceError("invalid_process")
        if _absolute_path(process.executable) is None or not isinstance(process.argv, (list, tuple)):
            raise SourceError("invalid_process")
        if not process.argv or any(
            not isinstance(token, str) or not token or "\x00" in token for token in process.argv
        ):
            raise SourceError("invalid_process")
        if process.executable == executable and tuple(process.argv[1:]) == expected:
            matches.append(process)
    if not matches:
        raise SourceError("process_missing")
    if len(matches) != 1:
        raise SourceError("ambiguous_process")
    return matches[0]
