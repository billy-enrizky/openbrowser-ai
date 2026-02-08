"""Browser-based reward computation for online Flow GRPO training.

Computes reward from actual browser execution outcomes rather than
text-matching heuristics. Used by online_flow_grpo_trainer.py.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BrowserOutcome:
    """Result of executing a rollout in the browser."""

    success_page_detected: bool = False
    submitted_values: dict = field(default_factory=dict)
    error: str | None = None
    actions_executed: int = 0
    total_actions: int = 0


# Default reward component weights
DEFAULT_REWARD_WEIGHTS = {
    "task_completion": 0.6,
    "field_accuracy": 0.3,
    "execution_completeness": 0.1,
}


def _compute_field_accuracy(
    submitted: dict, ground_truth: dict
) -> float:
    """Compute fraction of submitted field values matching ground truth.

    Handles string fields (case-insensitive), booleans (exact match),
    and list fields (set overlap).
    """
    if not ground_truth:
        return 0.0

    matches = 0
    total = len(ground_truth)

    for field_name, expected in ground_truth.items():
        actual = submitted.get(field_name)
        if actual is None:
            # Try case-insensitive key lookup
            for k, v in submitted.items():
                if k.lower() == field_name.lower():
                    actual = v
                    break

        if actual is None:
            continue

        if isinstance(expected, bool):
            # Boolean fields: checkbox presence indicates True
            if expected and actual:
                matches += 1
            elif not expected and not actual:
                matches += 1
        elif isinstance(expected, list):
            # List fields: set overlap
            expected_set = {str(v).lower() for v in expected}
            actual_set = {str(v).lower() for v in (actual if isinstance(actual, list) else [actual])}
            if expected_set and expected_set == actual_set:
                matches += 1
            elif expected_set & actual_set:
                matches += 0.5
        else:
            # String fields: case-insensitive comparison
            if str(actual).lower().strip() == str(expected).lower().strip():
                matches += 1

    return matches / total if total > 0 else 0.0


def compute_online_reward(
    outcome: BrowserOutcome,
    ground_truth_fields: dict,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute reward from browser execution outcome.

    Components:
    - task_completion (0.6): 1.0 if success page detected, 0.0 otherwise
    - field_accuracy (0.3): fraction of submitted values matching ground truth
    - execution_completeness (0.1): fraction of actions executed without error

    Args:
        outcome: BrowserOutcome from executing rollout in browser.
        ground_truth_fields: Expected field values from FormFactory ground truth.
        weights: Optional override for reward component weights.

    Returns:
        Float reward in [0, 1].
    """
    w = weights or DEFAULT_REWARD_WEIGHTS

    # Task completion: binary
    task_completion = 1.0 if outcome.success_page_detected else 0.0

    # Field accuracy: fraction of correct field values
    field_accuracy = _compute_field_accuracy(
        outcome.submitted_values, ground_truth_fields
    )

    # Execution completeness: fraction of actions that ran without error
    if outcome.total_actions > 0:
        execution_completeness = outcome.actions_executed / outcome.total_actions
    else:
        execution_completeness = 0.0

    reward = (
        w.get("task_completion", 0.6) * task_completion
        + w.get("field_accuracy", 0.3) * field_accuracy
        + w.get("execution_completeness", 0.1) * execution_completeness
    )

    logger.debug(
        f"Reward: {reward:.3f} (completion={task_completion:.1f}, "
        f"field_acc={field_accuracy:.3f}, exec={execution_completeness:.3f})"
    )

    return reward
