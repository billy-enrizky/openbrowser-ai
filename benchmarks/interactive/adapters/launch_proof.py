"""Fenced positive launch proofs for warm OpenBrowser arms.

The proof binds a positive mock-keychain verification to the exact daemon or
MCP server identity and the main browser process identity. Warm tasks may reuse
it only while every identity still matches. Any change fails the current row
and requires a later explicit warmup.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from .base import _process_table

ProcessFactory = Callable[[int], Any]
CommandTable = Callable[[], list[tuple[int, str]]]
ALIVE_STATUSES = {
    value
    for name in (
        "STATUS_RUNNING",
        "STATUS_SLEEPING",
        "STATUS_DISK_SLEEP",
        "STATUS_IDLE",
        "STATUS_WAKING",
        "STATUS_PARKED",
        "STATUS_LOCKED",
        "STATUS_WAITING",
    )
    if isinstance(value := getattr(psutil, name, None), str)
}


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    created_at: float


@dataclass(frozen=True)
class LaunchProof:
    server: ProcessIdentity
    browser: ProcessIdentity
    profile_marker: str
    context_id: str | None
    keychain_verified: bool


def read_pid_file(path: Path) -> int | None:
    """Read a positive PID from a file without probing the process."""
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return value if value > 0 else None


def identity_status(
    pid: int | None,
    process_factory: ProcessFactory | None = None,
) -> tuple[ProcessIdentity | None, str]:
    """Return the exact identity and a closed status: ok, missing, unknown."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None, "missing"
    factory = process_factory or psutil.Process
    try:
        process = factory(pid)
        created = process.create_time()
        if not process.is_running():
            return None, "missing"
        status = process.status()
        if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            return None, "missing"
        if status not in ALIVE_STATUSES:
            return None, "unknown"
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None, "missing"
    except (psutil.Error, OSError, TypeError, ValueError):
        return None, "unknown"
    if isinstance(created, bool) or not isinstance(created, (int, float)):
        return None, "unknown"
    created_at = float(created)
    if not math.isfinite(created_at) or created_at <= 0:
        return None, "unknown"
    return ProcessIdentity(pid, created_at), "ok"


def main_browser_identity(
    profile_marker: str,
    *,
    command_table: CommandTable | None = None,
    process_factory: ProcessFactory | None = None,
) -> ProcessIdentity | None:
    """Return the unique main browser identity for an owned profile marker."""
    if not isinstance(profile_marker, str) or len(profile_marker) < 8 or not Path(profile_marker).is_absolute():
        return None
    table = command_table or _process_table

    def matching_pids() -> list[int]:
        matches: list[int] = []
        for pid, command in table():
            if not isinstance(command, str):
                continue
            tokens = command.split()
            if not tokens:
                continue
            executable = Path(tokens[0]).name.lower()
            if "helper" in executable or "crashpad" in executable:
                continue
            if any(token.startswith("--type=") for token in tokens):
                continue
            if f"--user-data-dir={profile_marker}" not in tokens:
                continue
            if "--use-mock-keychain" not in tokens:
                continue
            matches.append(pid)
        return matches

    try:
        before = matching_pids()
        if len(before) != 1:
            return None
        pid = before[0]
        first, reason = identity_status(pid, process_factory)
        if reason != "ok" or first is None:
            return None
        after = matching_pids()
        if after != [pid]:
            return None
        second, reason = identity_status(pid, process_factory)
        if reason != "ok" or second != first:
            return None
        return first
    except (psutil.Error, OSError, TypeError, ValueError):
        return None


def capture_launch_proof(
    profile_marker: str,
    server_pid: int | None,
    *,
    context_id: str | None = None,
    keychain_verified: bool | None,
    command_table: CommandTable | None = None,
    process_factory: ProcessFactory | None = None,
) -> LaunchProof | None:
    """Build a proof only from a positive keychain result and stable identities."""
    if keychain_verified is not True:
        return None
    if context_id is not None and not isinstance(context_id, str):
        return None
    server, server_reason = identity_status(server_pid, process_factory)
    if server_reason != "ok" or server is None:
        return None
    browser = main_browser_identity(
        profile_marker,
        command_table=command_table,
        process_factory=process_factory,
    )
    if browser is None:
        return None
    server_after, server_reason = identity_status(server_pid, process_factory)
    if server_reason != "ok" or server_after != server:
        return None
    browser_after, browser_reason = identity_status(browser.pid, process_factory)
    if browser_reason != "ok" or browser_after != browser:
        return None
    return LaunchProof(
        server=server,
        browser=browser,
        profile_marker=profile_marker,
        context_id=context_id,
        keychain_verified=True,
    )


def proof_status(
    proof: LaunchProof,
    *,
    profile_marker: str,
    server_pid: int | None,
    context_id: str | None,
    process_factory: ProcessFactory | None = None,
) -> str:
    """Return a closed proof status for the current task fences."""
    if not isinstance(proof, LaunchProof) or proof.keychain_verified is not True:
        return "keychain_unproven"
    if proof.profile_marker != profile_marker:
        return "profile_changed"
    if proof.context_id != context_id:
        return "context_changed"
    server, server_reason = identity_status(server_pid, process_factory)
    if server is None:
        return "server_missing" if server_reason == "missing" else "identity_unknown"
    if server != proof.server:
        return "server_replaced"
    browser, browser_reason = identity_status(proof.browser.pid, process_factory)
    if browser is None:
        return "browser_missing" if browser_reason == "missing" else "identity_unknown"
    if browser != proof.browser:
        return "browser_replaced"
    return "valid"
