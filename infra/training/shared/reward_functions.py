"""Reward functions and GRPO advantage calculation for RL training."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewardSignal:
    """Reward signal for a single trajectory."""
    task_completion: float = 0.0  # 0 or 1 -- did the agent complete the task?
    step_efficiency: float = 0.0  # Reward for fewer steps
    action_correctness: float = 0.0  # Per-action match against ground truth
    total: float = 0.0


def compute_task_completion_reward(
    agent_output: str | None,
    ground_truth: str,
    success: bool,
) -> float:
    """Binary task completion reward."""
    if success:
        return 1.0

    # Partial credit: check if ground truth keywords appear in output
    if agent_output and ground_truth:
        gt_lower = ground_truth.lower()
        out_lower = agent_output.lower()
        if gt_lower in out_lower or "successfully" in out_lower:
            return 0.5

    return 0.0


def compute_step_efficiency_reward(
    steps_taken: int,
    max_steps: int = 50,
    optimal_steps: int | None = None,
) -> float:
    """Reward for completing tasks in fewer steps.

    Returns value in [0, 1] where 1 = optimal steps, 0 = max steps.
    """
    if steps_taken <= 0:
        return 0.0

    target = optimal_steps if optimal_steps else max(1, max_steps // 5)
    if steps_taken <= target:
        return 1.0

    return max(0.0, 1.0 - (steps_taken - target) / (max_steps - target))


def compute_action_correctness_reward(
    predicted_actions: list[str],
    ground_truth_actions: list[str],
) -> float:
    """Compute action sequence overlap reward.

    Simple token-level F1 between predicted and ground truth actions.
    """
    if not predicted_actions or not ground_truth_actions:
        return 0.0

    pred_set = set(" ".join(predicted_actions).lower().split())
    gt_set = set(" ".join(ground_truth_actions).lower().split())

    if not pred_set or not gt_set:
        return 0.0

    intersection = pred_set & gt_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gt_set)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_reward(
    agent_output: str | None,
    ground_truth: str,
    success: bool,
    steps_taken: int,
    max_steps: int = 50,
    predicted_actions: list[str] | None = None,
    ground_truth_actions: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> RewardSignal:
    """Compute combined reward signal."""
    if weights is None:
        weights = {
            "task_completion": 0.6,
            "step_efficiency": 0.2,
            "action_correctness": 0.2,
        }

    signal = RewardSignal()
    signal.task_completion = compute_task_completion_reward(agent_output, ground_truth, success)
    signal.step_efficiency = compute_step_efficiency_reward(steps_taken, max_steps)
    signal.action_correctness = compute_action_correctness_reward(
        predicted_actions or [], ground_truth_actions or []
    )

    signal.total = (
        weights["task_completion"] * signal.task_completion
        + weights["step_efficiency"] * signal.step_efficiency
        + weights["action_correctness"] * signal.action_correctness
    )

    return signal


def compute_grpo_advantages(
    rewards: list[float],
    group_size: int = 4,
) -> list[float]:
    """Compute GRPO advantages using group-relative normalization.

    For each group of G rollouts, advantage = (reward - mean) / std.
    """
    if not rewards:
        return []

    advantages = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]

        if len(group) < 2:
            advantages.extend([0.0] * len(group))
            continue

        mean = sum(group) / len(group)
        variance = sum((r - mean) ** 2 for r in group) / len(group)
        std = max(variance ** 0.5, 1e-8)

        for r in group:
            advantages.append((r - mean) / std)

    return advantages
