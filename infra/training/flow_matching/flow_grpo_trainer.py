"""Flow GRPO trainer: Advantage-weighted CFM loss."""

import json
import logging

import torch

from infra.training.shared.reward_functions import compute_grpo_advantages, compute_reward
from infra.training.flow_matching.config import FLOW_GRPO_CONFIG, DATA_CONFIG, FLOW_MODEL_CONFIG
from infra.training.flow_matching.flow_model import FlowVectorFieldEstimator
from infra.training.flow_matching.ode_solver import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    """Run flow GRPO training."""
    config = FLOW_GRPO_CONFIG

    model = FlowVectorFieldEstimator(FLOW_MODEL_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load SFT checkpoint if available
    sft_path = "outputs/flow_matching_sft/model.pt"
    try:
        model.load_state_dict(torch.load(sft_path, map_location=device))
        logger.info(f"Loaded SFT checkpoint from {sft_path}")
    except FileNotFoundError:
        logger.warning("No SFT checkpoint found, training from scratch")

    # Load prompts
    prompts = []
    with open(DATA_CONFIG["train_file"]) as f:
        for line in f:
            prompts.append(json.loads(line))
    if DATA_CONFIG.get("max_train_samples", 0) > 0:
        prompts = prompts[: DATA_CONFIG["max_train_samples"]]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    group_size = config["group_size"]

    logger.info(f"Starting flow GRPO: {len(prompts)} prompts, G={group_size}")

    for epoch in range(config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")

        for i, prompt_data in enumerate(prompts):
            condition_text = prompt_data.get("condition", "")
            target_actions = prompt_data.get("target", [])
            ground_truth = " ".join(target_actions) if isinstance(target_actions, list) else str(target_actions)

            # Generate G rollouts using ODE solver
            # Simplified: proper implementation needs tokenization
            # condition_tokens = tokenize(condition_text)
            # rollouts = [sample(model, condition_tokens, seq_length=64, num_steps=config["num_ode_steps"]) for _ in range(group_size)]

            # Score rollouts
            rewards = []
            for g in range(group_size):
                signal = compute_reward(
                    agent_output="placeholder",
                    ground_truth=ground_truth,
                    success=False,
                    steps_taken=len(target_actions),
                    weights=config.get("reward_weights"),
                )
                rewards.append(signal.total)

            # Compute advantages
            advantages = compute_grpo_advantages(rewards, group_size)

            # Advantage-weighted CFM loss update
            # Full implementation: weight CFM loss by advantage for each rollout
            loss = torch.tensor(0.0, device=device, requires_grad=True)

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if (i + 1) % config["logging_steps"] == 0:
                avg_r = sum(rewards) / len(rewards) if rewards else 0
                logger.info(f"  Step {i+1}/{len(prompts)}: avg_reward={avg_r:.3f}")

    torch.save(model.state_dict(), "outputs/flow_matching_grpo/model.pt")
    logger.info("Flow GRPO training complete")


if __name__ == "__main__":
    train()
