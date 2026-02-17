"""Browser-based reward computation for online GRPO training.

Computes reward from actual browser execution outcomes rather than
text-matching heuristics. Used by both online_grpo_trainer.py (AR)
and online_flow_llm_grpo_trainer.py (Flow).

Uses continuous string similarity (SequenceMatcher) instead of binary
exact match for field accuracy, producing more granular rewards that
prevent zero-advantage batches in GRPO.
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher

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
# task_completion is binary (0/1), so reduce its weight to prevent
# identical rewards within GRPO groups. field_accuracy now produces
# continuous values via string similarity, making it the primary
# source of reward variance across rollouts.
DEFAULT_REWARD_WEIGHTS = {
    "task_completion": 0.4,
    "field_accuracy": 0.4,
    "execution_completeness": 0.2,
}


def _string_similarity(a: str, b: str) -> float:
    """Compute continuous string similarity in [0, 1] using SequenceMatcher.

    Case-insensitive, whitespace-normalized comparison that returns
    continuous values instead of binary 0/1, preventing identical
    rewards within GRPO groups when rollouts differ slightly.
    """
    a_norm = str(a).lower().strip()
    b_norm = str(b).lower().strip()
    if a_norm == b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _compute_field_accuracy(
    submitted: dict, ground_truth: dict
) -> float:
    """Compute field accuracy with continuous string similarity.

    Instead of binary exact match (0 or 1 per field), uses
    SequenceMatcher for string fields to produce continuous [0, 1]
    scores. This makes rewards more granular, preventing zero-advantage
    batches in GRPO when all G rollouts produce nearly-correct values.

    Handles string fields (continuous similarity), booleans (exact match),
    and list fields (set overlap with Jaccard similarity).
    """
    if not ground_truth:
        return 0.0

    score = 0.0
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
                score += 1.0
            elif not expected and not actual:
                score += 1.0
        elif isinstance(expected, list):
            # List fields: Jaccard similarity for continuous score
            expected_set = {str(v).lower() for v in expected}
            actual_set = {str(v).lower() for v in (actual if isinstance(actual, list) else [actual])}
            if expected_set or actual_set:
                intersection = expected_set & actual_set
                union = expected_set | actual_set
                score += len(intersection) / len(union) if union else 0.0
        else:
            # String fields: continuous similarity instead of binary match
            score += _string_similarity(actual, expected)

    return score / total if total > 0 else 0.0


def compute_online_reward(
    outcome: BrowserOutcome,
    ground_truth_fields: dict,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute reward from browser execution outcome.

    Components:
    - task_completion (0.4): 1.0 if success page detected, 0.0 otherwise (binary)
    - field_accuracy (0.4): continuous similarity of submitted values vs ground truth
    - execution_completeness (0.2): fraction of actions executed without error

    field_accuracy uses continuous string similarity (SequenceMatcher) instead
    of binary exact match, producing more granular rewards that create
    non-zero GRPO advantages even when rollouts are similar.

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
        w.get("task_completion", 0.4) * task_completion
        + w.get("field_accuracy", 0.4) * field_accuracy
        + w.get("execution_completeness", 0.2) * execution_completeness
    )

    logger.debug(
        f"Reward: {reward:.3f} (completion={task_completion:.1f}, "
        f"field_acc={field_accuracy:.3f}, exec={execution_completeness:.3f})"
    )

    return reward
