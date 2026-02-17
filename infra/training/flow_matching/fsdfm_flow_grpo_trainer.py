"""FS-DFM Flow-GRPO trainer: Proper discrete policy gradients for flow matching.

Implements Flow-GRPO (Liu et al., 2025) adapted for discrete flow matching
with the Poisson jump process.  Unlike the advantage-weighted flow loss in
fsdfm_online_grpo_trainer.py, this computes proper per-step log-probabilities
aligned with the actual generation process, enabling PPO-style clipped policy
gradients.

Key innovation -- discrete analog of continuous Flow-GRPO:
    Continuous: SDE step gives Gaussian policy N(mu, sigma^2 dt I)
                log-prob = Gaussian log-density
    Discrete:   Euler step gives categorical per-position distribution
                P(stay) = exp(-lambda * dt)
                P(jump to j) = (1 - exp(-lambda * dt)) * p(j) / (1 - p(cur))
                log-prob = sum of per-position log-probs over response tokens

Architecture:
    1. Generate G rollouts via discrete Euler solver, recording trajectories
    2. Execute rollouts in headless browser against FormFactory
    3. Compute group-relative advantages from browser rewards
    4. For each rollout, sample K random trajectory steps (denoising reduction):
       a. Recompute policy log-prob at that step (with gradients)
       b. REINFORCE loss: -advantage * log_prob (per-step backward)
       c. KL penalty from Schulman k3 approximation (ref model swapped
          from CPU to GPU on demand to save VRAM)
    5. Gradient accumulation across steps, single optimizer.step()

Reference: github.com/yifan123/flow_grpo (continuous version for images)

Usage:
    uv run infra/training/flow_matching/fsdfm_flow_grpo_trainer.py
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import torch

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FLOW_GRPO_FSDFM_CONFIG,
    FSDFM_MODEL_CONFIG,
)
from infra.training.flow_matching.fsdfm_model import (
    PolynomialConvexScheduler,
    compute_discrete_step_log_prob,
    generate_with_prefix_conditioning_trajectory,
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
    """Load prompts for Flow-GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info("Loaded %d prompts for FS-DFM Flow-GRPO", len(records))
    return records


async def train():
    """Run FS-DFM Flow-GRPO training with browser execution."""
    model_config = FSDFM_MODEL_CONFIG
    grpo_config = FLOW_GRPO_FSDFM_CONFIG
    vocab_size = model_config["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if grpo_config.get("bf16") else torch.float16

    # Load GPT-2 tokenizer (native to FS-DFM)
    from transformers import AutoTokenizer

    logger.info("Loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Flow matching scheduler
    exponent = model_config.get("scheduler_exponent", 2.0)
    scheduler = PolynomialConvexScheduler(exponent=exponent)

    # ---------------------------------------------------------------
    # Load policy model (LoRA, from SFT checkpoint)
    # ---------------------------------------------------------------
    sft_checkpoint = os.environ.get("FSDFM_SFT_CHECKPOINT", "")
    logger.info("Loading FS-DFM 1.3B policy model")
    policy_model = load_fsdfm_from_huggingface(
        model_config, device=device, dtype=compute_dtype
    )
    policy_model = inject_lora(policy_model, model_config)

    if sft_checkpoint and Path(sft_checkpoint).exists():
        logger.info("Loading SFT checkpoint: %s", sft_checkpoint)
        lora_path = Path(sft_checkpoint) / "lora_weights.pt"
        if lora_path.exists():
            load_lora_weights(policy_model, str(lora_path))
        else:
            logger.warning(
                "lora_weights.pt not found at %s, starting from base LoRA init",
                lora_path,
            )
    elif sft_checkpoint:
        logger.warning(
            "SFT checkpoint not found at %s, training from base LoRA init",
            sft_checkpoint,
        )

    # ---------------------------------------------------------------
    # Load reference model (frozen, on CPU to save VRAM)
    # ---------------------------------------------------------------
    logger.info("Loading FS-DFM 1.3B reference model (frozen, CPU)")
    ref_model = load_fsdfm_from_huggingface(
        model_config, device=torch.device("cpu"), dtype=compute_dtype
    )
    ref_model = inject_lora(ref_model, model_config)
    if sft_checkpoint and Path(sft_checkpoint).exists():
        lora_path = Path(sft_checkpoint) / "lora_weights.pt"
        if lora_path.exists():
            load_lora_weights(ref_model, str(lora_path))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ---------------------------------------------------------------
    # Training data and optimizer
    # ---------------------------------------------------------------
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=grpo_config["learning_rate"]
    )

    group_size = grpo_config["group_size"]
    kl_coeff = grpo_config["kl_coeff"]
    adv_clip_max = grpo_config.get("adv_clip_max", 5.0)
    max_seq_length = model_config["max_seq_length"]
    num_gen_steps = grpo_config.get("num_generation_steps", 10)
    gen_temperature = grpo_config.get("generation_temperature", 1.0)
    action_timeout = grpo_config.get("action_timeout_s", 5.0)
    grad_clip = grpo_config.get("grad_clip", 1.0)
    num_sampled_timesteps = grpo_config.get("num_sampled_timesteps", 8)
    dt = 1.0 / num_gen_steps

    # ---------------------------------------------------------------
    # Start FormFactory server and browser
    # ---------------------------------------------------------------
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = grpo_config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    headless = grpo_config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(
        "Starting FS-DFM Flow-GRPO: %d prompts, G=%d, kl=%.3f, "
        "T_gen=%d, temp=%.1f",
        len(prompts),
        group_size,
        kl_coeff,
        num_gen_steps,
        gen_temperature,
    )

    total_steps = 0
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

                # Periodic browser restart to reset DOM indices
                if i > 0 and i % 10 == 0:
                    logger.info("Periodic browser restart (prompt %d)", i)
                    await browser_env.close()
                    browser_env = await BrowserEnvironment.create(headless=headless)

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

                # ==========================================================
                # Phase 1: Generate G rollouts with trajectory recording
                # ==========================================================
                trajectories = []
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    trajectory = generate_with_prefix_conditioning_trajectory(
                        model=policy_model,
                        prefix_ids=prefix_ids,
                        gen_length=gen_length,
                        config={
                            **model_config,
                            "num_generation_steps": num_gen_steps,
                        },
                        scheduler=scheduler,
                        temperature=gen_temperature,
                    )
                    trajectories.append(trajectory)
                    # Decode only the response portion
                    response_ids = trajectory.final_tokens[0, prefix_len:]
                    text = tokenizer.decode(response_ids, skip_special_tokens=True)
                    rollout_texts.append(text)
                policy_model.train()

                # ==========================================================
                # Phase 2: Execute each rollout in browser and score
                # ==========================================================
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
                        outcome.actions_executed if hasattr(outcome, 'actions_executed') else -1,
                        len(actions),
                    )

                epoch_rewards.extend(rewards)

                # ==========================================================
                # Phase 3: Compute group-relative advantages
                # ==========================================================
                advantages = compute_grpo_advantages(rewards, group_size)
                advantages_t = torch.tensor(
                    advantages, dtype=torch.float32, device=device
                )

                # ==========================================================
                # Phase 4: Policy gradient update over ALL trajectory steps
                # ==========================================================
                # REINFORCE with per-step gradient accumulation. We backward()
                # after each step to release activation memory immediately,
                # preventing OOM from accumulated autograd graphs.
                #
                # Note: old_log_prob was removed because with a single
                # optimization step per prompt, the policy weights are
                # identical when computing log_prob and old_log_prob, making
                # ratio = exp(0) = 1.0 always (PPO clipping has no effect).
                optimizer.zero_grad()
                total_loss_val = 0.0
                total_kl_val = 0.0
                kl_terms = 0

                for g in range(group_size):
                    traj = trajectories[g]
                    adv_g = torch.clamp(
                        advantages_t[g], -adv_clip_max, adv_clip_max
                    )

                    if len(traj.steps) == 0:
                        continue

                    # Skip gradient computation when advantage is exactly 0
                    # (no learning signal). This avoids 0 * NaN = NaN when
                    # log_prob has numerical issues from bf16 overflow.
                    if adv_g.abs().item() < 1e-10:
                        continue

                    # Response mask from the trajectory edit_mask (float)
                    response_mask = traj.edit_mask.float()  # [1, L]

                    # Denoising reduction: sample K random timesteps
                    # instead of processing all T steps (Flow-GRPO paper)
                    num_sampled = min(num_sampled_timesteps, len(traj.steps))
                    sampled_indices = sorted(
                        random.sample(range(len(traj.steps)), num_sampled)
                    )
                    sampled_steps = [traj.steps[idx] for idx in sampled_indices]
                    num_steps_g = max(len(sampled_steps), 1)

                    for step in sampled_steps:
                        # Current policy log-prob (WITH gradients)
                        log_prob = compute_discrete_step_log_prob(
                            model=policy_model,
                            x_t=step.x_t,
                            x_next=step.x_next,
                            t_scalar=step.t_value,
                            dt=dt,
                            scheduler=scheduler,
                            vocab_size=vocab_size,
                            response_mask=response_mask,
                            temperature=gen_temperature,
                        )  # [1]

                        # Guard: skip if log_prob is NaN/Inf
                        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                            logger.warning(
                                "NaN/Inf log_prob at rollout %d, step t=%.4f, skipping",
                                g, step.t_value,
                            )
                            continue

                        # REINFORCE policy loss (ratio=1 simplification)
                        policy_loss = (-adv_g * log_prob).mean()

                        # KL penalty (Schulman k3: r - log(r) - 1 >= 0)
                        kl_loss = torch.tensor(0.0, device=device)
                        if kl_coeff > 0:
                            # Move ref_model to GPU for KL computation
                            ref_model.to(device)
                            with torch.no_grad():
                                ref_log_prob = compute_discrete_step_log_prob(
                                    model=ref_model,
                                    x_t=step.x_t,
                                    x_next=step.x_next,
                                    t_scalar=step.t_value,
                                    dt=dt,
                                    scheduler=scheduler,
                                    vocab_size=vocab_size,
                                    response_mask=response_mask,
                                    temperature=gen_temperature,
                                )  # [1]
                            ref_model.to("cpu")
                            torch.cuda.empty_cache()
                            log_r = ref_log_prob - log_prob
                            kl_loss = (torch.exp(log_r) - log_r - 1).mean()
                            # Guard: replace NaN KL with 0
                            if torch.isnan(kl_loss):
                                kl_loss = torch.tensor(0.0, device=device)
                            total_kl_val += kl_loss.detach().item()
                            kl_terms += 1

                        step_loss = (policy_loss + kl_coeff * kl_loss) / num_steps_g
                        # Per-step backward to release activations immediately
                        step_loss.backward()
                        total_loss_val += step_loss.detach().item()

                # Clip gradients and step (guard against NaN accumulated loss)
                if total_loss_val != 0.0 and not (
                    total_loss_val != total_loss_val  # NaN check
                ):
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    optimizer.step()

                torch.cuda.empty_cache()

                total_steps += 1
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                avg_kl = total_kl_val / max(kl_terms, 1)
                epoch_kl.append(avg_kl)

                if total_steps % grpo_config["logging_steps"] == 0:
                    logger.info(
                        "  Step %d (prompt %d/%d): avg_reward=%.3f, "
                        "loss=%.4f, kl=%.4f",
                        total_steps,
                        i + 1,
                        len(prompts),
                        avg_reward,
                        total_loss_val,
                        avg_kl,
                    )

            # Epoch summary
            if epoch_rewards:
                epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
                nonzero = sum(1 for r in epoch_rewards if r > 0)
                logger.info(
                    "Epoch %d complete: avg_reward=%.3f, "
                    "nonzero_rewards=%d/%d, avg_kl=%.4f",
                    epoch + 1,
                    epoch_avg,
                    nonzero,
                    len(epoch_rewards),
                    sum(epoch_kl) / len(epoch_kl) if epoch_kl else 0,
                )

        # Save final model
        final_dir = Path("outputs/fsdfm_flow_grpo/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        save_lora_weights(policy_model, str(final_dir / "lora_weights.pt"))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("FS-DFM Flow-GRPO complete. Model saved to %s", final_dir)

        persist_checkpoint(str(final_dir.parent), "fsdfm-flow-grpo")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
