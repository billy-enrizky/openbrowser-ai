"""Structural proxy reward for MDPO intermediate denoising steps.

Uses existing parse_rollout_to_actions() and compute_action_correctness_reward()
to score partially-denoised text without browser execution.
"""

import logging

from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.reward_functions import compute_action_correctness_reward

logger = logging.getLogger(__name__)


def compute_proxy_reward(
    text: str,
    element_map: dict[str, int],
    ground_truth_actions: list[str],
) -> float:
    """Score decoded text structurally without browser execution.

    Reward signal:
    - 0.0: text is unparseable (typical for early noisy denoising steps)
    - 0.1: parseable but no actions extracted
    - 0.2-1.0: scaled by action correctness F1 vs ground truth

    Args:
        text: Decoded model output at denoising step t.
        element_map: Field name -> DOM element index mapping.
        ground_truth_actions: List of ground truth action strings for F1.

    Returns:
        Float reward in [0.0, 1.0].
    """
    try:
        actions = parse_rollout_to_actions(text, element_map)
    except Exception:
        return 0.0

    if len(actions) == 0:
        return 0.1

    predicted_action_strs = []
    for a in actions:
        action_type = a.get("action", "")
        params = a.get("params", {})
        text_val = params.get("text", "")
        predicted_action_strs.append(f"{action_type} {text_val}".strip())

    if not predicted_action_strs or not ground_truth_actions:
        return 0.1

    f1 = compute_action_correctness_reward(
        predicted_actions=predicted_action_strs,
        ground_truth_actions=ground_truth_actions,
    )

    return 0.2 + 0.8 * f1
