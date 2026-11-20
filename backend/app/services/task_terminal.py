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
    if content in _DEFAULT_CANCELLATION_MESSAGES:
        return "Task was cancelled"
    return f"Task was cancelled. {content}"


def classify_task_terminal(result: Mapping[str, Any]) -> TaskTerminal:
    """Classify a result without changing the Browser Agent event contract.

    The Browser Agent still reports completion with a success flag until its
    own lifecycle is made explicit. CodeAgent failures carry terminal_phase.
    """
    text = result.get("result")
    content = text if isinstance(text, str) and text else None
    if result.get("cancelled"):
        return TaskTerminal("cancelled", content or "Task was cancelled")
    if result.get("terminal_phase"):
        if result.get("success"):
            return TaskTerminal("completed", content or "Task completed successfully.")
        reason = result.get("terminal_reason") or "unknown error"
        return TaskTerminal("failed", content or f"Task failed: {reason}")
    if result.get("phase"):
        phase = result.get("phase")
        return TaskTerminal("failed", content or f"Task failed: CodeAgent ended in {phase}")
    return TaskTerminal("completed", content or "Task completed successfully.")
