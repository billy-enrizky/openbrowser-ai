"""FS-DFM AR Surrogate GRPO trainer: TRUE autoregressive log-probs for fair comparison.

Uses FS-DFM 1.3B (Apple DiT) with LoRA, but computes TRUE autoregressive
surrogate log-probabilities via causal masking at t~1.0, identical to
ReFusion's Standard GRPO formulation. This eliminates the confound between
diffusion-native GKL loss and AR log-probs for fair comparison in Table 3.

Architecture:
    1. FS-DFM (DiT + LoRA) generates G candidate plans via discrete Euler solver
       with prefix conditioning (BIDIRECTIONAL attention, unchanged)
    2. Each plan is decoded to text via GPT-2 tokenizer and parsed into actions
    3. Actions are executed in a headless browser against FormFactory
    4. Reward = form submission success + field accuracy
    5. AR surrogate log-probs computed via causal masking at t~1.0:
       - model(x_1, t=0.9999, is_causal=True) -> logits
       - Shifted cross-entropy -> per-token log-probs
       - REINFORCE loss: -advantage * mean(log_probs)
    6. KL penalty: Schulman k3 (r - log_r - 1) using AR log-probs from
       policy and frozen reference model

Usage:
    uv run infra/training/flow_matching/fsdfm_ar_grpo_trainer.py
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import torch

from infra.training.flow_matching.config import (
    AR_SURROGATE_FSDFM_GRPO_CONFIG,
    DATA_CONFIG,
    FSDFM_MODEL_CONFIG,
)
from infra.training.flow_matching.fsdfm_model import (
    PolynomialConvexScheduler,
    compute_per_token_log_probs_fsdfm,
    generate_with_prefix_conditioning,
    inject_lora,
    load_fsdfm_from_huggingface,
    load_lora_weights,
    save_lora_weights,
)
from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.browser_env import BrowserEnvironment
from infra.training.shared.formfactory_server import FormFactoryServer
from infra.training.shared.online_reward import compute_online_reward
from infra.training.shared.reward_functions import compute_grpo_advantages
from infra.training.shared.utils import persist_checkpoint, resolve_data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info("Loaded %d prompts for FS-DFM AR surrogate GRPO", len(records))
    return records


async def train():
    """Run online FS-DFM AR Surrogate GRPO training with browser execution."""
    model_config = FSDFM_MODEL_CONFIG
    grpo_config = AR_SURROGATE_FSDFM_GRPO_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if grpo_config.get("bf16") else torch.float16

    # Load GPT-2 tokenizer
    from transformers import AutoTokenizer
    logger.info("Loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Scheduler needed for generation (Euler solver)
    exponent = model_config.get("scheduler_exponent", 2.0)
    scheduler = PolynomialConvexScheduler(exponent=exponent)

    # Load policy model
    sft_checkpoint = os.environ.get("FSDFM_SFT_CHECKPOINT", "")
    logger.info("Loading FS-DFM 1.3B policy model")
    policy_model = load_fsdfm_from_huggingface(model_config, device=device, dtype=compute_dtype)
    policy_model = inject_lora(policy_model, model_config)
    policy_model.gradient_checkpointing = True

    if sft_checkpoint and Path(sft_checkpoint).exists():
        logger.info("Loading SFT checkpoint: %s", sft_checkpoint)
        lora_path = Path(sft_checkpoint) / "lora_weights.pt"
        if lora_path.exists():
            load_lora_weights(policy_model, str(lora_path))
        else:
            logger.warning("lora_weights.pt not found at %s, starting from base LoRA init", lora_path)
    elif sft_checkpoint:
        logger.warning("SFT checkpoint not found at %s, training from base LoRA init", sft_checkpoint)

    # Load reference model (frozen, for KL)
    logger.info("Loading FS-DFM 1.3B reference model (frozen)")
    ref_model = load_fsdfm_from_huggingface(model_config, device=device, dtype=compute_dtype)
    ref_model = inject_lora(ref_model, model_config)
    if sft_checkpoint and Path(sft_checkpoint).exists():
        lora_path = Path(sft_checkpoint) / "lora_weights.pt"
        if lora_path.exists():
            load_lora_weights(ref_model, str(lora_path))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Load training prompts
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )
    if grpo_config.get("shuffle_prompts", True):
        random.shuffle(prompts)
        logger.info("Shuffled training prompts")

    # Optimizer (only policy LoRA params)
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=grpo_config["learning_rate"]
    )

    group_size = grpo_config["group_size"]
    kl_coeff = grpo_config["kl_coeff"]
    max_seq_length = model_config["max_seq_length"]
    num_sampling_steps = grpo_config.get("num_sampling_steps", model_config.get("num_sampling_steps", 64))
    gen_temperature = grpo_config.get("generation_temperature", model_config.get("generation_temperature", 1.0))
    action_timeout = grpo_config.get("action_timeout_s", 5.0)
    grad_clip = grpo_config.get("grad_clip", 1.0)
    min_nonzero = grpo_config.get("min_nonzero_for_update", 1)
    early_stop_max_steps = grpo_config.get("early_stop_max_steps", 40)
    checkpoint_every = grpo_config.get("checkpoint_every_steps", 10)

    # Start FormFactory server
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = grpo_config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    # Start browser environment
    headless = grpo_config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(
        "Starting FS-DFM AR Surrogate GRPO: %d prompts, G=%d, "
        "kl_coeff=%s, sampling_steps=%d, LR=%s",
        len(prompts), group_size, kl_coeff, num_sampling_steps,
        grpo_config["learning_rate"],
    )

    total_steps = 0
    best_avg_reward = -1.0
    no_improve_steps = 0
    ckpt_name = os.environ.get("CHECKPOINT_NAME", "ar-surrogate-fsdfm")
    output_dir = Path("outputs/fsdfm_ar_grpo")

    try:
        for epoch in range(grpo_config["num_epochs"]):
            logger.info("Epoch %d/%d", epoch + 1, grpo_config["num_epochs"])
            epoch_rewards = []
            epoch_kl = []

            for i, prompt_data in enumerate(prompts):
                instruction = prompt_data.get(
                    "instruction", prompt_data.get("condition", "")
                )
                form_url = prompt_data.get("url", "")
                ground_truth_fields = prompt_data.get("ground_truth_fields", {})

                if not instruction or not form_url:
                    logger.warning("Skipping prompt %d: missing instruction or url", i)
                    continue

                # Periodic browser restart to prevent session degradation
                if i > 0 and i % 5 == 0:
                    logger.info("Periodic browser restart (prompt %d)", i)
                    await browser_env.restart()

                # Tokenize instruction
                inst_enc = tokenizer(
                    instruction,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_seq_length // 2,
                    return_tensors="pt",
                )
                prefix_ids = inst_enc["input_ids"].to(device)
                prefix_len = prefix_ids.shape[1]
                gen_length = max(1, max_seq_length - prefix_len)

                # Generate G rollouts via discrete Euler solver (BIDIRECTIONAL)
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    generated_ids = generate_with_prefix_conditioning(
                        model=policy_model,
                        prefix_ids=prefix_ids,
                        gen_length=gen_length,
                        config={**model_config, "num_sampling_steps": num_sampling_steps},
                        scheduler=scheduler,
                        temperature=gen_temperature,
                    )
                    text = tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )
                    rollout_texts.append(text)
                policy_model.train()

                # Execute each rollout in browser and score
                rewards = []
                for g, rollout_text in enumerate(rollout_texts):
                    await browser_env.reset()

                    try:
                        await browser_env.tools.navigate(
                            url=form_url,
                            new_tab=False,
                            browser_session=browser_env.browser_session,
                        )
                        await asyncio.sleep(0.5)
                        element_map = await browser_env.get_element_map()
                    except Exception as e:
                        logger.warning("Navigation failed for rollout %d: %s", g, e)
                        logger.info("Restarting browser after navigation failure")
                        await browser_env.restart()
                        try:
                            await browser_env.tools.navigate(
                                url=form_url,
                                new_tab=False,
                                browser_session=browser_env.browser_session,
                            )
                            await asyncio.sleep(0.5)
                            element_map = await browser_env.get_element_map()
                        except Exception as e2:
                            logger.warning(
                                "Navigation still failed after restart for rollout %d: %s",
                                g, e2,
                            )
                            rewards.append(0.0)
                            continue

                    actions = parse_rollout_to_actions(rollout_text, element_map)
                    if not actions:
                        logger.warning(
                            "No valid actions parsed from rollout %d. "
                            "Generated text (first 300 chars): %.300s",
                            g, rollout_text,
                        )
                        rewards.append(0.0)
                        continue

                    logger.info(
                        "Rollout %d: %d actions parsed. Text (first 200 chars): %.200s",
                        g, len(actions), rollout_text,
                    )
                    outcome = await browser_env.execute_actions(
                        actions, timeout_per_action=action_timeout
                    )
                    reward = compute_online_reward(
                        outcome,
                        ground_truth_fields,
                        weights=grpo_config.get("reward_weights"),
                    )
                    rewards.append(reward)
                    logger.info(
                        "Rollout %d: reward=%.3f (actions_executed=%d/%d)",
                        g, reward,
                        outcome.actions_executed if hasattr(outcome, "actions_executed") else -1,
                        len(actions),
                    )

                epoch_rewards.extend(rewards)

                # Check min nonzero for update
                nonzero_count = sum(1 for r in rewards if r > 0)
                if nonzero_count < min_nonzero:
                    logger.info(
                        "Step %d: %d nonzero < %d min, skipping gradient update",
                        total_steps + 1, nonzero_count, min_nonzero,
                    )
                    total_steps += 1
                    # Still save checkpoint on schedule even when skipping
                    if total_steps % checkpoint_every == 0:
                        ckpt_dir = output_dir / "final"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        save_lora_weights(policy_model, str(ckpt_dir / "lora_weights.pt"))
                        tokenizer.save_pretrained(str(ckpt_dir))
                        persist_checkpoint(str(output_dir), ckpt_name)
                        logger.info("  Checkpoint saved at step %d (skipped update)", total_steps)
                    continue

                # Compute GRPO advantages
                advantages = compute_grpo_advantages(rewards, group_size)
                advantages_t = torch.tensor(
                    advantages, dtype=torch.float32, device=device
                )

                # --- AR Surrogate REINFORCE + Schulman k3 KL ---
                total_pg_loss = torch.tensor(0.0, device=device, requires_grad=False)
                total_kl = torch.tensor(0.0, device=device)
                valid_rollouts = 0

                for g in range(group_size):
                    # Tokenize the generated rollout
                    rollout_enc = tokenizer(
                        rollout_texts[g],
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    rollout_ids = rollout_enc["input_ids"].squeeze(0).to(device)
                    if rollout_ids.shape[0] == 0:
                        logger.debug("Empty rollout %d, skipping loss", g)
                        continue

                    # Build full sequence: prefix + rollout
                    full_ids = torch.cat([
                        prefix_ids.squeeze(0), rollout_ids
                    ])[:max_seq_length]
                    full_len = full_ids.shape[0]

                    # Pad to max_seq_length
                    pad_len = max_seq_length - full_len
                    if pad_len > 0:
                        pad_id = tokenizer.pad_token_id or 0
                        full_ids = torch.cat([
                            full_ids,
                            torch.full((pad_len,), pad_id, dtype=torch.long, device=device),
                        ])

                    # Loss mask: 0 for prefix, 1 for response tokens, 0 for padding
                    loss_mask = torch.zeros(max_seq_length, dtype=torch.float32, device=device)
                    resp_start = min(prefix_len, full_len)
                    loss_mask[resp_start:full_len] = 1.0

                    full_ids = full_ids.unsqueeze(0).to(device)  # [1, L]
                    loss_mask = loss_mask.unsqueeze(0)  # [1, L]

                    # Policy AR log-probs (with gradient)
                    policy_token_lp, resp_mask = compute_per_token_log_probs_fsdfm(
                        policy_model, full_ids, prefix_len, loss_mask,
                    )  # [1, T], [1, T]

                    # Reference AR log-probs (no gradient)
                    with torch.no_grad():
                        ref_token_lp, _ = compute_per_token_log_probs_fsdfm(
                            ref_model, full_ids, prefix_len, loss_mask,
                        )  # [1, T]

                    # Per-sample mean log-prob under current policy
                    tokens_per_sample = resp_mask.sum(dim=-1).clamp(min=1)  # [1]
                    sample_log_prob = (
                        (policy_token_lp * resp_mask).sum(dim=-1) / tokens_per_sample
                    )  # [1]

                    # REINFORCE: -advantage * log_prob
                    pg_loss_g = -(advantages_t[g] * sample_log_prob).squeeze()

                    # KL divergence: Schulman k3 (always >= 0)
                    log_r = ref_token_lp - policy_token_lp  # [1, T]
                    r = torch.exp(log_r)
                    kl_per_token = r - log_r - 1  # >= 0 by Jensen's inequality
                    total_resp_tokens = resp_mask.sum().clamp(min=1)
                    kl_g = (kl_per_token * resp_mask).sum() / total_resp_tokens

                    total_pg_loss = total_pg_loss + pg_loss_g + kl_coeff * kl_g
                    total_kl = total_kl + kl_g
                    valid_rollouts += 1

                divisor = max(valid_rollouts, 1)
                loss = total_pg_loss / divisor

                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    optimizer.step()

                total_steps += 1
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                avg_kl = (total_kl / max(valid_rollouts, 1)).item()
                epoch_kl.append(avg_kl)

                if total_steps % grpo_config["logging_steps"] == 0:
                    logger.info(
                        "  Step %d (prompt %d/%d): "
                        "avg_reward=%.3f, loss=%.4f, kl=%.4f",
                        total_steps, i + 1, len(prompts),
                        avg_reward, loss.item(), avg_kl,
                    )

                # Best checkpoint tracking
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    no_improve_steps = 0
                    if avg_reward > 0:
                        best_dir = output_dir / "best"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        save_lora_weights(policy_model, str(best_dir / "lora_weights.pt"))
                        tokenizer.save_pretrained(str(best_dir))
                        logger.info(
                            "  New best reward=%.3f at step %d", best_avg_reward, total_steps,
                        )
                else:
                    no_improve_steps += 1

                # Periodic checkpoint
                if total_steps % checkpoint_every == 0:
                    ckpt_dir = output_dir / "final"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    save_lora_weights(policy_model, str(ckpt_dir / "lora_weights.pt"))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    persist_checkpoint(str(output_dir), ckpt_name)
                    logger.info("  Checkpoint saved at step %d", total_steps)

                # Early stopping
                if no_improve_steps >= early_stop_max_steps:
                    logger.info(
                        "Early stopping: no improvement for %d steps (best=%.3f)",
                        early_stop_max_steps, best_avg_reward,
                    )
                    break

            # Epoch summary
            if epoch_rewards:
                epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
                nonzero = sum(1 for r in epoch_rewards if r > 0)
                logger.info(
                    "Epoch %d complete: avg_reward=%.3f, "
                    "nonzero_rewards=%d/%d, avg_kl=%.4f",
                    epoch + 1, epoch_avg, nonzero, len(epoch_rewards),
                    sum(epoch_kl) / len(epoch_kl) if epoch_kl else 0,
                )

        # Save final model
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        save_lora_weights(policy_model, str(final_dir / "lora_weights.pt"))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("FS-DFM AR Surrogate GRPO complete. Model saved to %s", final_dir)
        persist_checkpoint(str(output_dir), ckpt_name)

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
