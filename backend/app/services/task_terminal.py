"""Map an agent result to one user-visible terminal outcome."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping

TaskTerminalKind = Literal["completed", "failed", "cancelled"]
_DEFAULT_CANCELLATION_MESSAGES = frozenset({"Task was cancelled", "Cancelled by user"})


@dataclass(frozen=True)
class TaskTerminal:
    kind: TaskTerminalKind
    content: str


def cancellation_display_content(content: str) -> str:
    """Build the persisted cancellation text shown after a reload."""
    normalized = content.strip()
    if not normalized or normalized in _DEFAULT_CANCELLATION_MESSAGES:
        return "Task was cancelled"
    return f"Task was cancelled. {normalized}"


def classify_task_terminal(result: Mapping[str, Any]) -> TaskTerminal:
    """Map an agent result to one user-visible terminal outcome.

    Browser Agent and CodeAgent results carry terminal_phase. The fallback is
    retained for older callers and stored results.
    """
    text = result.get("result")
    content = text if isinstance(text, str) and text else None
    if result.get("cancelled") or result.get("terminal_phase") in {"cancelled", "stopped"}:
        return TaskTerminal("cancelled", content or "Task was cancelled")
    terminal_phase = result.get("terminal_phase")
    if terminal_phase:
        if terminal_phase == "succeeded" and result.get("success") is True:
            return TaskTerminal("completed", content or "Task completed successfully.")
        reason = result.get("terminal_reason") or "unknown error"
        return TaskTerminal("failed", content or f"Task failed: {reason}")
    if result.get("phase"):
        phase = result.get("phase")
        return TaskTerminal("failed", content or f"Task failed: CodeAgent ended in {phase}")
    if result.get("success") is False:
        return TaskTerminal("failed", content or "Task failed")
    return TaskTerminal("completed", content or "Task completed successfully.")
