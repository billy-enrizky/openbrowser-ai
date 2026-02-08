"""Flow GRPO trainer: Advantage-weighted CFM loss."""

import json
import logging
from pathlib import Path

import torch

from infra.training.shared.reward_functions import compute_grpo_advantages, compute_reward
from infra.training.flow_matching.config import FLOW_GRPO_CONFIG, DATA_CONFIG, FLOW_MODEL_CONFIG
from infra.training.flow_matching.flow_model import FlowVectorFieldEstimator
from infra.training.flow_matching.flow_sft_trainer import cfm_loss, tokenize_for_flow
from infra.training.flow_matching.ode_solver import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_flow_tokens(token_ids: torch.Tensor) -> str:
    """Decode flow token IDs back to text (inverse of hash-based tokenization).

    Since hash-based tokenization is lossy, this returns a best-effort string.
    """
    ids = token_ids.squeeze(0).tolist()
    ids = [i for i in ids if i != 0]
    try:
        return bytes(ids).decode("utf-8", errors="replace")
    except Exception:
        return ""


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
    vocab_size = model.vocab_size
    max_seq_length = model.max_seq_length

    logger.info(f"Starting flow GRPO: {len(prompts)} prompts, G={group_size}")

    for epoch in range(config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")

        for i, prompt_data in enumerate(prompts):
            condition_text = prompt_data.get("condition", "")
            target_actions = prompt_data.get("target", [])
            ground_truth = " ".join(target_actions) if isinstance(target_actions, list) else str(target_actions)

            # Tokenize condition
            condition_ids = tokenize_for_flow(
                [condition_text], max_seq_length, vocab_size, device
            )

            # Generate G rollouts using ODE solver
            rollout_ids = []
            for _ in range(group_size):
                output_ids = sample(
                    model, condition_ids,
                    seq_length=max_seq_length,
                    num_steps=config.get("num_ode_steps", 10),
                )
                rollout_ids.append(output_ids)

            # Score rollouts
            rewards = []
            for g in range(group_size):
                rollout_text = decode_flow_tokens(rollout_ids[g])
                signal = compute_reward(
                    agent_output=rollout_text,
                    ground_truth=ground_truth,
                    success=False,
                    steps_taken=len(target_actions),
                    weights=config.get("reward_weights"),
                )
                rewards.append(signal.total)

            # Compute advantages
            advantages = compute_grpo_advantages(rewards, group_size)

            # Advantage-weighted CFM loss: compute CFM loss for each rollout,
            # weight by advantage
            target_ids = tokenize_for_flow(
                [ground_truth], max_seq_length, vocab_size, device
            )

            total_loss = torch.tensor(0.0, device=device)
            for g in range(group_size):
                x_0 = torch.randint(0, vocab_size, target_ids.shape, device=device)
                single_loss = cfm_loss(model, x_0, target_ids, condition_ids)
                total_loss = total_loss + advantages[g] * single_loss

            loss = total_loss / group_size

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if (i + 1) % config["logging_steps"] == 0:
                avg_r = sum(rewards) / len(rewards) if rewards else 0
                logger.info(
                    f"  Step {i+1}/{len(prompts)}: avg_reward={avg_r:.3f}, "
                    f"loss={loss.item():.4f}"
                )

    output_dir = Path("outputs/flow_matching_grpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_dir / "model.pt"))
    logger.info("Flow GRPO training complete")


if __name__ == "__main__":
    train()
