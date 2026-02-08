"""Online Flow GRPO trainer: Advantage-weighted CFM loss with real browser execution.

Unlike the offline flow_grpo_trainer.py which scores rollouts via text-matching,
this trainer executes generated action plans in a real browser (via the openbrowser
package) against a running FormFactory Flask server and computes rewards from
actual form submission outcomes.

Architecture:
    1. Flow model generates G candidate plans via ODE solver (GPU)
    2. Each plan is decoded to text and parsed into executable actions
    3. Actions are executed in a headless browser against FormFactory
    4. Reward = actual form submission success + field accuracy
    5. Advantage-weighted CFM loss backpropagates through the flow model

Usage:
    uv run infra/training/flow_matching/online_flow_grpo_trainer.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import torch

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FLOW_MODEL_CONFIG,
    ONLINE_FLOW_GRPO_CONFIG,
)
from infra.training.flow_matching.flow_model import FlowVectorFieldEstimator
from infra.training.flow_matching.flow_sft_trainer import cfm_loss, tokenize_for_flow
from infra.training.flow_matching.ode_solver import sample
from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.browser_env import BrowserEnvironment
from infra.training.shared.formfactory_server import FormFactoryServer
from infra.training.shared.online_reward import compute_online_reward
from infra.training.shared.reward_functions import compute_grpo_advantages
from infra.training.shared.utils import resolve_data_path, upload_checkpoint_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def decode_flow_tokens(token_ids: torch.Tensor) -> str:
    """Decode flow token IDs back to text (inverse of hash-based tokenization)."""
    ids = token_ids.squeeze(0).tolist()
    ids = [i for i in ids if i != 0]
    try:
        return bytes(ids).decode("utf-8", errors="replace")
    except Exception:
        return ""


async def train():
    """Run online flow GRPO training with browser execution."""
    config = ONLINE_FLOW_GRPO_CONFIG

    # Load flow model
    model = FlowVectorFieldEstimator(FLOW_MODEL_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load SFT checkpoint if available
    sft_path = os.environ.get(
        "FLOW_SFT_CHECKPOINT", "outputs/flow_matching_sft/model.pt"
    )
    try:
        model.load_state_dict(torch.load(sft_path, map_location=device))
        logger.info(f"Loaded SFT checkpoint from {sft_path}")
    except FileNotFoundError:
        logger.warning("No SFT checkpoint found, training from scratch")

    # Load prompts
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    prompts = []
    with open(train_file) as f:
        for line in f:
            prompts.append(json.loads(line))

    max_samples = DATA_CONFIG.get("max_train_samples", 0)
    if max_samples > 0:
        prompts = prompts[:max_samples]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    group_size = config["group_size"]
    vocab_size = model.vocab_size
    max_seq_length = model.max_seq_length

    # Start FormFactory server
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    # Start browser environment (openbrowser)
    headless = config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)
    action_timeout = config.get("action_timeout_s", 5.0)

    logger.info(
        f"Starting online flow GRPO: {len(prompts)} prompts, G={group_size}, "
        f"device={device}"
    )

    total_steps = 0
    try:
        for epoch in range(config["num_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
            epoch_rewards = []

            for i, prompt_data in enumerate(prompts):
                condition_text = prompt_data.get("condition", "")
                target_actions = prompt_data.get("target", [])
                ground_truth_fields = prompt_data.get("ground_truth_fields", {})
                form_url = prompt_data.get("url", "")
                ground_truth_text = (
                    " ".join(target_actions)
                    if isinstance(target_actions, list)
                    else str(target_actions)
                )

                if not condition_text or not form_url:
                    logger.warning(f"Skipping prompt {i}: missing condition or url")
                    continue

                # Tokenize condition
                condition_ids = tokenize_for_flow(
                    [condition_text], max_seq_length, vocab_size, device
                )

                # Generate G rollouts using ODE solver
                rollout_ids = []
                for _ in range(group_size):
                    output_ids = sample(
                        model,
                        condition_ids,
                        seq_length=max_seq_length,
                        num_steps=config.get("num_ode_steps", 10),
                    )
                    rollout_ids.append(output_ids)

                # Execute each rollout in browser and score
                rewards = []
                for g in range(group_size):
                    rollout_text = decode_flow_tokens(rollout_ids[g])

                    # Reset browser and navigate to form page
                    await browser_env.reset()

                    # Navigate to form URL and get element mapping
                    try:
                        await browser_env.tools.navigate(
                            url=form_url,
                            new_tab=False,
                            browser_session=browser_env.browser_session,
                        )
                        # Brief wait for page to load
                        await asyncio.sleep(0.5)
                        element_map = await browser_env.get_element_map()
                    except Exception as e:
                        logger.warning(f"Navigation failed for rollout {g}: {e}")
                        rewards.append(0.0)
                        continue

                    # Parse rollout text into executable actions
                    actions = parse_rollout_to_actions(rollout_text, element_map)

                    if not actions:
                        logger.debug(f"No valid actions parsed from rollout {g}")
                        rewards.append(0.0)
                        continue

                    # Execute actions in browser
                    outcome = await browser_env.execute_actions(
                        actions, timeout_per_action=action_timeout
                    )

                    # Compute reward from browser outcome
                    reward = compute_online_reward(
                        outcome,
                        ground_truth_fields,
                        weights=config.get("reward_weights"),
                    )
                    rewards.append(reward)

                epoch_rewards.extend(rewards)

                # Compute group-relative advantages
                advantages = compute_grpo_advantages(rewards, group_size)

                # Advantage-weighted CFM loss
                target_ids = tokenize_for_flow(
                    [ground_truth_text], max_seq_length, vocab_size, device
                )

                total_loss = torch.tensor(0.0, device=device)
                for g in range(group_size):
                    x_0 = torch.randint(
                        0, vocab_size, target_ids.shape, device=device
                    )
                    single_loss = cfm_loss(model, x_0, target_ids, condition_ids)
                    total_loss = total_loss + advantages[g] * single_loss

                loss = total_loss / group_size

                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_steps += 1

                if (i + 1) % config["logging_steps"] == 0:
                    avg_r = sum(rewards) / len(rewards) if rewards else 0
                    logger.info(
                        f"  Step {i+1}/{len(prompts)}: avg_reward={avg_r:.3f}, "
                        f"loss={loss.item():.4f}"
                    )

            # Epoch summary
            if epoch_rewards:
                epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
                nonzero = sum(1 for r in epoch_rewards if r > 0)
                logger.info(
                    f"Epoch {epoch + 1} complete: avg_reward={epoch_avg:.3f}, "
                    f"nonzero_rewards={nonzero}/{len(epoch_rewards)}"
                )

        # Save final model
        output_dir = Path("outputs/flow_matching_online_grpo")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        torch.save(model.state_dict(), str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Upload to S3 if configured
        s3_bucket = os.environ.get("S3_CHECKPOINT_BUCKET", "")
        if s3_bucket:
            s3_config = {
                "checkpoint_bucket": s3_bucket,
                "checkpoint_prefix": os.environ.get(
                    "S3_CHECKPOINT_PREFIX", "training/checkpoints"
                ),
                "region": os.environ.get("AWS_REGION", "ca-central-1"),
            }
            upload_checkpoint_to_s3(str(output_dir), s3_config, "online-flow-grpo")

        logger.info("Online flow GRPO training complete")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
