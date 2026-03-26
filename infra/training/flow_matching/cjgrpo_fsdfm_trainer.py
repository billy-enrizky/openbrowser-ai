"""CJ-GRPO trainer for FS-DFM 1.3B: Trajectory-consistent PPO for discrete flow matching.

Implements Consistent Joint GRPO (CJ-GRPO) for FS-DFM's Poisson jump process.
This adapts the CJ-GRPO algorithm from the ReFusion trainer to FS-DFM's discrete
flow matching architecture with three key differences from plain Flow-GRPO:

1. **Full PPO clipping**: Computes importance ratios exp(logp - old_logp) and
   applies min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) instead of the
   REINFORCE simplification (-A * logp) used when ratio=1 always holds.

2. **Cached sequence-level log-probs**: After generating each trajectory,
   forward-passes the policy and reference models on each step to cache
   old_log_prob and ref_log_prob (sequence-level scalars). These are frozen
   and reused across mu policy update iterations.

3. **mu > 1 policy updates**: Outer loop over mu optimizer steps per prompt
   batch (like ESPO), enabled by the cached old/ref log-probs and importance
   ratios that correctly track policy drift.

FS-DFM specifics (different from ReFusion):
    - GPT-2 tokenizer (vocab_size=50257)
    - PolynomialConvexScheduler(exponent=2.0) for noise schedule
    - EulerTrajectory with EulerTrajectoryStep (x_t, x_next, t_value)
    - Sequence-level log-probs from compute_discrete_step_log_prob (not per-token)
    - PPO clipping operates on [B] scalar ratios, not per-token vectors
    - Reference model on CPU (swapped to GPU only for KL computation)
    - LoRA via custom inject_lora (not PEFT/QLoRA)

Per-step loss (sequence-level):
    log_ratio = cur_log_prob - old_log_prob  (both [B] scalars)
    coef_1 = exp(log_ratio)
    coef_2 = clamp(coef_1, 1 - epsilon, 1 + epsilon)
    policy_loss = -min(coef_1 * A, coef_2 * A).mean()
    kl_ratio = ref_log_prob - cur_log_prob
    kl_loss = (exp(kl_ratio) - kl_ratio - 1).mean()
    step_loss = (policy_loss + beta * kl_loss) / num_sampled_steps

Usage:
    uv run infra/training/flow_matching/cjgrpo_fsdfm_trainer.py
"""

import asyncio
import logging
import os
import random
from pathlib import Path

import torch

from infra.training.flow_matching.config import (
    CJGRPO_FSDFM_CONFIG,
    DATA_CONFIG,
    FSDFM_MODEL_CONFIG,
)
from infra.training.flow_matching.fsdfm_model import (
    EulerTrajectoryStep,
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
from infra.training.shared.utils import load_prompts, persist_checkpoint, resolve_data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent



def cache_discrete_step_logprobs(
    model,
    step: EulerTrajectoryStep,
    dt: float,
    scheduler: PolynomialConvexScheduler,
    vocab_size: int,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cache sequence-level log-prob for one Euler step (no gradients).

    Unlike the ReFusion version which returns per-token log-probs, FS-DFM's
    compute_discrete_step_log_prob returns a [B] scalar (mean over response
    positions). This is the correct granularity for FS-DFM's Poisson jump
    process.

    Args:
        model: FS-DFM policy or reference model.
        step: Recorded trajectory step with x_t and x_next.
        dt: Euler step size (1.0 / num_generation_steps).
        scheduler: PolynomialConvexScheduler instance.
        vocab_size: Token vocabulary size (50257 for GPT-2).
        response_mask: [B, L] float, 1 for response positions, 0 for prefix.
        temperature: Softmax temperature (should match generation).

    Returns:
        [B] tensor of sequence-level log-probs (mean over response positions).
    """
    with torch.no_grad():
        log_prob = compute_discrete_step_log_prob(
            model=model,
            x_t=step.x_t,
            x_next=step.x_next,
            t_scalar=step.t_value,
            dt=dt,
            scheduler=scheduler,
            vocab_size=vocab_size,
            response_mask=response_mask,
            temperature=temperature,
        )
    return log_prob.detach()


def compute_cjgrpo_discrete_step_loss(
    model,
    step: EulerTrajectoryStep,
    dt: float,
    scheduler: PolynomialConvexScheduler,
    vocab_size: int,
    response_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    advantage: float,
    epsilon: float,
    beta: float,
    temperature: float = 1.0,
) -> tuple[torch.Tensor | None, dict]:
    """Compute CJ-GRPO loss for one discrete Euler step with PPO clipping.

    Forward-passes the model with gradients, computes sequence-level importance
    ratios, applies PPO clipping, and adds reverse KL penalty.

    Since FS-DFM's log-prob is sequence-level (mean over response positions),
    the importance ratio is a single scalar per batch element, not per-token.

    Args:
        model: Policy model (with gradients).
        step: Trajectory step with x_t and x_next.
        dt: Euler step size.
        scheduler: PolynomialConvexScheduler instance.
        vocab_size: Token vocabulary size.
        response_mask: [B, L] float, 1 for response, 0 for prefix.
        old_log_prob: Cached old policy log-prob [B] (frozen snapshot).
        ref_log_prob: Cached reference model log-prob [B] (frozen).
        advantage: Scalar advantage for this rollout.
        epsilon: PPO clip range.
        beta: KL penalty coefficient.
        temperature: Softmax temperature.

    Returns:
        (loss_tensor_or_None, metrics_dict) where loss has gradients attached.
        Returns (None, metrics) if the step should be skipped.
    """
    metrics = {
        "policy_loss": 0.0,
        "kl_loss": 0.0,
        "ratio_mean": 0.0,
        "clipped_frac": 0.0,
    }

    # Forward pass WITH gradients -- sequence-level log-prob [B]
    cur_log_prob = compute_discrete_step_log_prob(
        model=model,
        x_t=step.x_t,
        x_next=step.x_next,
        t_scalar=step.t_value,
        dt=dt,
        scheduler=scheduler,
        vocab_size=vocab_size,
        response_mask=response_mask,
        temperature=temperature,
    )

    # Guard: skip if log-prob is NaN/Inf
    if torch.isnan(cur_log_prob).any() or torch.isinf(cur_log_prob).any():
        logger.warning(
            "NaN/Inf log_prob at step t=%.4f, skipping", step.t_value
        )
        return None, metrics

    # Sequence-level importance ratio: exp(current - old)
    log_ratio = cur_log_prob - old_log_prob  # [B]
    coef_1 = torch.exp(log_ratio)  # [B]
    coef_2 = torch.clamp(coef_1, 1.0 - epsilon, 1.0 + epsilon)  # [B]

    # PPO-clipped surrogate loss (sequence-level)
    policy_loss = -torch.min(coef_1 * advantage, coef_2 * advantage).mean()

    # Reverse KL penalty (sequence-level)
    # kl = exp(ref - current) - (ref - current) - 1
    kl_loss = torch.tensor(0.0, device=cur_log_prob.device)
    if beta > 0:
        kl_ratio = ref_log_prob - cur_log_prob  # [B]
        kl_loss = (torch.exp(kl_ratio) - kl_ratio - 1.0).mean()
        # Guard: replace NaN KL with 0
        if torch.isnan(kl_loss):
            kl_loss = torch.tensor(0.0, device=cur_log_prob.device)

    step_loss = policy_loss + beta * kl_loss

    # Metrics (detached)
    with torch.no_grad():
        metrics["policy_loss"] = policy_loss.item()
        metrics["kl_loss"] = kl_loss.item()
        metrics["ratio_mean"] = coef_1.mean().item()
        metrics["clipped_frac"] = (
            (coef_1 < 1.0 - epsilon) | (coef_1 > 1.0 + epsilon)
        ).float().mean().item()

    return step_loss, metrics


async def train():
    """Run FS-DFM CJ-GRPO training with browser execution."""
    model_config = FSDFM_MODEL_CONFIG
    cjgrpo_config = CJGRPO_FSDFM_CONFIG
    vocab_size = model_config["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if cjgrpo_config.get("bf16") else torch.float16

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
        trainable_params,
        lr=cjgrpo_config["learning_rate"],
        weight_decay=cjgrpo_config.get("weight_decay", 0.01),
    )

    group_size = cjgrpo_config["group_size"]
    kl_coeff = cjgrpo_config["kl_coeff"]
    epsilon = cjgrpo_config["epsilon"]
    mu = cjgrpo_config.get("mu", 1)
    max_seq_length = model_config["max_seq_length"]
    num_gen_steps = cjgrpo_config.get("num_generation_steps", 64)
    gen_temperature = cjgrpo_config.get("generation_temperature", 1.0)
    action_timeout = cjgrpo_config.get("action_timeout_s", 5.0)
    grad_clip = cjgrpo_config.get("grad_clip", 1.0)
    num_sampled_timesteps = cjgrpo_config.get("num_sampled_timesteps", 8)
    min_nonzero = cjgrpo_config.get("min_nonzero_for_update", 1)
    max_steps = cjgrpo_config.get("early_stop_max_steps", 40)
    checkpoint_every = cjgrpo_config.get("checkpoint_every_steps", 10)
    dt = 1.0 / num_gen_steps

    # ---------------------------------------------------------------
    # Start FormFactory server and browser
    # ---------------------------------------------------------------
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = cjgrpo_config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    headless = cjgrpo_config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(
        "Starting FS-DFM CJ-GRPO: %d prompts, G=%d, kl=%.4f, eps=%.2f, "
        "mu=%d, K=%d, T_gen=%d, temp=%.1f, max_steps=%d",
        len(prompts),
        group_size,
        kl_coeff,
        epsilon,
        mu,
        num_sampled_timesteps,
        num_gen_steps,
        gen_temperature,
        max_steps,
    )

    total_steps = 0
    best_avg_reward = -1.0
    best_step = -1
    best_checkpoint_dir = Path("outputs/cjgrpo_fsdfm/best")

    try:
        for epoch in range(cjgrpo_config["num_epochs"]):
            logger.info("Epoch %d/%d", epoch + 1, cjgrpo_config["num_epochs"])
            epoch_rewards = []
            epoch_kl = []

            for prompt_idx, prompt_data in enumerate(prompts):
                if total_steps >= max_steps:
                    logger.info(
                        "EARLY STOP at step %d (max_steps=%d). "
                        "best_reward=%.3f at step %d",
                        total_steps, max_steps, best_avg_reward, best_step,
                    )
                    break

                instruction = prompt_data.get(
                    "instruction", prompt_data.get("condition", "")
                )
                form_url = prompt_data.get("url", "")
                ground_truth_fields = prompt_data.get("ground_truth_fields", {})

                if not instruction or not form_url:
                    logger.warning(
                        "Skipping prompt %d: missing instruction or url",
                        prompt_idx,
                    )
                    continue

                # Periodic browser restart to prevent session degradation
                if prompt_idx > 0 and prompt_idx % 5 == 0:
                    logger.info("Periodic browser restart (prompt %d)", prompt_idx)
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
                        # Reactive restart: browser likely degraded
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
                                "Navigation still failed after restart for "
                                "rollout %d: %s",
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
                        weights=cjgrpo_config.get("reward_weights"),
                    )
                    rewards.append(reward)
                    logger.info(
                        "Rollout %d: reward=%.3f (actions_executed=%d/%d)",
                        g, reward,
                        outcome.actions_executed if hasattr(outcome, "actions_executed") else -1,
                        len(actions),
                    )

                epoch_rewards.extend(rewards)

                # ==========================================================
                # Phase 3: Check zero-reward skip
                # ==========================================================
                nonzero_count = sum(1 for r in rewards if r > 0)
                if nonzero_count < min_nonzero:
                    logger.info(
                        "Step %d: skipping update (nonzero=%d < min=%d)",
                        total_steps, nonzero_count, min_nonzero,
                    )
                    total_steps += 1
                    continue

                # ==========================================================
                # Phase 4: Compute group-relative advantages
                # ==========================================================
                advantages = compute_grpo_advantages(rewards, group_size)
                advantages_t = torch.tensor(
                    advantages, dtype=torch.float32, device=device
                )

                # ==========================================================
                # Phase 5: Cache old/ref sequence-level log-probs
                # ==========================================================
                # For each rollout g, for each trajectory step, cache:
                #   old_log_prob: policy model log-prob at generation time [B]
                #   ref_log_prob: reference model log-prob [B]
                # These are frozen and reused across mu iterations.

                # Move ref model to GPU for caching
                if kl_coeff > 0:
                    ref_model.to(device)

                cached_logprobs = []
                for g in range(group_size):
                    traj = trajectories[g]
                    response_mask = traj.edit_mask.float()  # [B, L]
                    rollout_cache = []

                    for step in traj.steps:
                        # Old policy log-prob (frozen snapshot)
                        old_lp = cache_discrete_step_logprobs(
                            model=policy_model,
                            step=step,
                            dt=dt,
                            scheduler=scheduler,
                            vocab_size=vocab_size,
                            response_mask=response_mask,
                            temperature=gen_temperature,
                        )

                        # Reference model log-prob
                        if kl_coeff > 0:
                            ref_lp = cache_discrete_step_logprobs(
                                model=ref_model,
                                step=step,
                                dt=dt,
                                scheduler=scheduler,
                                vocab_size=vocab_size,
                                response_mask=response_mask,
                                temperature=gen_temperature,
                            )
                        else:
                            ref_lp = torch.zeros_like(old_lp)

                        rollout_cache.append((old_lp, ref_lp))

                    cached_logprobs.append(rollout_cache)

                # Move ref model back to CPU after caching
                if kl_coeff > 0:
                    ref_model.to("cpu")
                    torch.cuda.empty_cache()

                # ==========================================================
                # Phase 6: Policy update (mu iterations with PPO clipping)
                # ==========================================================
                total_loss_val = 0.0
                total_kl_val = 0.0
                total_ratio_val = 0.0
                total_clipped_frac = 0.0
                num_loss_terms = 0

                for mu_iter in range(mu):
                    optimizer.zero_grad()
                    iter_loss_val = 0.0

                    for g in range(group_size):
                        traj = trajectories[g]
                        adv_g = advantages_t[g].item()

                        if len(traj.steps) == 0:
                            continue

                        # Skip when advantage is near zero (no learning signal)
                        if abs(adv_g) < 1e-10:
                            continue

                        response_mask = traj.edit_mask.float()  # [B, L]

                        # Denoising reduction: sample K random timesteps
                        num_sampled = min(num_sampled_timesteps, len(traj.steps))
                        sampled_indices = sorted(
                            random.sample(range(len(traj.steps)), num_sampled)
                        )
                        num_steps_denom = max(num_sampled, 1)

                        for step_idx in sampled_indices:
                            step = traj.steps[step_idx]
                            old_lp, ref_lp = cached_logprobs[g][step_idx]

                            step_loss, step_metrics = compute_cjgrpo_discrete_step_loss(
                                model=policy_model,
                                step=step,
                                dt=dt,
                                scheduler=scheduler,
                                vocab_size=vocab_size,
                                response_mask=response_mask,
                                old_log_prob=old_lp,
                                ref_log_prob=ref_lp,
                                advantage=adv_g,
                                epsilon=epsilon,
                                beta=kl_coeff,
                                temperature=gen_temperature,
                            )

                            if step_loss is None:
                                continue

                            # Divide by num_sampled_steps for denoising reduction
                            scaled_loss = step_loss / num_steps_denom
                            # Per-step backward to release activations immediately
                            scaled_loss.backward()
                            iter_loss_val += scaled_loss.detach().item()

                            total_kl_val += step_metrics["kl_loss"]
                            total_ratio_val += step_metrics["ratio_mean"]
                            total_clipped_frac += step_metrics["clipped_frac"]
                            num_loss_terms += 1

                    # Clip gradients and step (guard against all-zero iteration)
                    if iter_loss_val != 0.0 and not (
                        iter_loss_val != iter_loss_val  # NaN check
                    ):
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                        optimizer.step()

                    total_loss_val += iter_loss_val

                torch.cuda.empty_cache()

                total_steps += 1
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                avg_kl = total_kl_val / max(num_loss_terms, 1)
                avg_ratio = total_ratio_val / max(num_loss_terms, 1)
                avg_clipped = total_clipped_frac / max(num_loss_terms, 1)
                epoch_kl.append(avg_kl)

                # Track best checkpoint
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_step = total_steps
                    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_lora_weights(
                        policy_model,
                        str(best_checkpoint_dir / "lora_weights.pt"),
                    )
                    tokenizer.save_pretrained(str(best_checkpoint_dir))
                    logger.info(
                        "New best checkpoint: avg_reward=%.3f at step %d",
                        best_avg_reward, best_step,
                    )

                # Periodic intermediate checkpoint
                if total_steps % checkpoint_every == 0:
                    step_dir = Path(
                        "outputs/cjgrpo_fsdfm/step_%d" % total_steps
                    )
                    step_dir.mkdir(parents=True, exist_ok=True)
                    save_lora_weights(
                        policy_model,
                        str(step_dir / "lora_weights.pt"),
                    )
                    tokenizer.save_pretrained(str(step_dir))
                    logger.info("Intermediate checkpoint saved to %s", step_dir)

                if total_steps % cjgrpo_config["logging_steps"] == 0:
                    logger.info(
                        "Step %d (prompt %d/%d): avg_reward=%.3f, "
                        "loss=%.4f, kl=%.4f, ratio=%.3f, clipped=%.2f, "
                        "best_reward=%.3f@%d",
                        total_steps,
                        prompt_idx + 1,
                        len(prompts),
                        avg_reward,
                        total_loss_val,
                        avg_kl,
                        avg_ratio,
                        avg_clipped,
                        best_avg_reward,
                        best_step,
                    )

            if total_steps >= max_steps:
                break

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
        final_dir = Path("outputs/cjgrpo_fsdfm/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        save_lora_weights(
            policy_model, str(final_dir / "lora_weights.pt")
        )
        tokenizer.save_pretrained(str(final_dir))
        logger.info(
            "FS-DFM CJ-GRPO complete. Final saved to %s, best (%.3f@%d) at %s",
            final_dir, best_avg_reward, best_step, best_checkpoint_dir,
        )

        persist_checkpoint(str(best_checkpoint_dir), "cjgrpo-fsdfm")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
