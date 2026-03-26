"""MDPO (Sequential MDP) trainer for ReFusion 8B: Temporal-advantage PPO for masked diffusion.

Implements Masked Diffusion Policy Optimization (MDPO) for ReFusion's iterative
unmasking process. This extends CJ-GRPO with three key differences:

1. **Per-step rewards**: Computes proxy reward at EVERY denoising step (structural
   similarity of partially-decoded text) and browser reward at the final step.

2. **Temporal advantage**: Uses reward deltas + cumulative future average (adv-v3
   + adv-v4) instead of group-relative advantages. Group-normalizes per step.

3. **Masking-rate scaling**: lambda_t = gen_length / max(num_unmasked_at_step, 1)
   gives earlier steps (fewer unmasked tokens) higher weight.

4. **k2 KL penalty**: Quadratic (ref - cur)^2 / 2 instead of reverse KL.

5. **Top-k step selection**: Selects training steps by advantage magnitude with
   a diversity guard ensuring representation from early trajectory.

Per-step loss:
    coef_1 = exp(per_token_logps - old_per_token_logps)
    coef_2 = clamp(coef_1, 1 - epsilon, 1 + epsilon)
    lambda_t = gen_length / max(num_unmasked, 1)
    per_token_loss = -min(coef_1 * A, coef_2 * A) * lambda_t
    per_token_kl = (ref_logps - per_token_logps)^2 / 2
    per_token_loss = per_token_loss + beta * per_token_kl
    step_loss = per_token_loss.mean()

Architecture:
    1. Generate G rollouts via iterative unmasking, recording trajectories
    2. Compute per-step proxy rewards by decoding cumulative unmaskings
    3. Execute final rollouts in headless browser for terminal reward
    4. Compute temporal advantages (adv-v3 + adv-v4) over [G, T] reward tensor
    5. Select top-k training steps by advantage magnitude (with diversity guard)
    6. Cache old/ref per-token log-probs at SELECTED steps only
    7. For mu iterations:
       a. Recompute per-token log-probs (with gradients)
       b. PPO-clipped surrogate + k2 KL + lambda_t scaling (per-step backward)
       c. Gradient accumulation across steps, single optimizer.step()

Usage:
    uv run infra/training/flow_matching/mdpo_refusion_trainer.py
"""

import asyncio
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FLOW_LLM_CONFIG,
    MDPO_REFUSION_CONFIG,
)
from infra.training.flow_matching.flow_llm_model import (
    FlowLLM,
    UnmaskingTrajectoryStep,
)
from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.browser_env import BrowserEnvironment
from infra.training.shared.formfactory_server import FormFactoryServer
from infra.training.shared.online_reward import compute_online_reward
from infra.training.shared.proxy_reward import compute_proxy_reward
from infra.training.shared.utils import (
    compute_temporal_advantages,
    load_prompts,
    load_quantized_model,
    persist_checkpoint,
    resolve_data_path,
    select_training_steps,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent



# ---------------------------------------------------------------------------
# Cache per-token log-probs (same as CJ-GRPO)
# ---------------------------------------------------------------------------


def cache_step_logprobs(
    model,
    step: UnmaskingTrajectoryStep,
    condition_length: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute per-token log-probs at newly unmasked positions for one step.

    Returns the individual per-token log-probs (not summed) so they can be
    used for importance ratio computation.

    Args:
        model: ReFusion policy or reference model (PEFT/QLoRA).
        step: Recorded trajectory step with masked_state and unmasked info.
        condition_length: L_c (prompt token count).
        temperature: Softmax temperature (should match generation).

    Returns:
        [max_k] tensor of per-token log-probs at newly unmasked positions
        for batch element 0 (B=1 always in our training loop).
        Returns empty tensor if no positions were unmasked.
    """
    device = step.masked_state.device

    indices = step.newly_unmasked_indices[0]  # B=1
    tokens = step.unmasked_tokens[0]

    if not indices:
        return torch.tensor([], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=step.masked_state,
            attention_mask=step.attention_mask,
        )
        response_logits = outputs.logits[:, condition_length:, :]  # [1, L_r, V]
        del outputs

        if temperature != 1.0 and temperature > 0:
            response_logits = response_logits / temperature

        log_probs = F.log_softmax(response_logits, dim=-1)  # [1, L_r, V]
        del response_logits

        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        tok_t = torch.tensor(tokens, dtype=torch.long, device=device)
        per_token_logps = log_probs[0, idx_t, tok_t]  # [k]
        del log_probs

    return per_token_logps.detach()


# ---------------------------------------------------------------------------
# MDPO step loss (PPO + lambda_t + k2 KL)
# ---------------------------------------------------------------------------


def compute_mdpo_step_loss(
    model,
    step: UnmaskingTrajectoryStep,
    condition_length: int,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    step_advantage: float,
    epsilon: float,
    beta: float,
    gen_length: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor | None, dict]:
    """Compute MDPO loss for one denoising step with PPO clipping and lambda_t.

    Differences from CJ-GRPO:
    - step_advantage: per-step temporal advantage (not single scalar for rollout)
    - lambda_t = gen_length / max(num_unmasked, 1): masking-rate scaling
    - k2 KL: (ref - cur)^2 / 2, NOT reverse KL exp(ref-cur) - (ref-cur) - 1

    Args:
        model: Policy model (with gradients).
        step: Trajectory step with masked_state and unmasked info.
        condition_length: L_c (prompt token count).
        old_per_token_logps: Cached old policy per-token log-probs [k].
        ref_per_token_logps: Cached reference model per-token log-probs [k].
        step_advantage: Scalar temporal advantage for this step.
        epsilon: PPO clip range.
        beta: KL penalty coefficient.
        gen_length: Total generation length (for lambda_t computation).
        temperature: Softmax temperature.

    Returns:
        (loss_tensor_or_None, metrics_dict) where loss has gradients attached.
        Returns (None, metrics) if the step should be skipped.
    """
    device = step.masked_state.device
    metrics = {
        "policy_loss": 0.0,
        "kl_loss": 0.0,
        "ratio_mean": 0.0,
        "clipped_frac": 0.0,
        "lambda_t": 0.0,
    }

    indices = step.newly_unmasked_indices[0]  # B=1
    tokens = step.unmasked_tokens[0]

    if not indices:
        return None, metrics

    # Forward pass WITH gradients
    outputs = model(
        input_ids=step.masked_state,
        attention_mask=step.attention_mask,
    )
    response_logits = outputs.logits[:, condition_length:, :]  # [1, L_r, V]
    del outputs

    if temperature != 1.0 and temperature > 0:
        response_logits = response_logits / temperature

    log_probs = F.log_softmax(response_logits, dim=-1)  # [1, L_r, V]
    del response_logits

    idx_t = torch.tensor(indices, dtype=torch.long, device=device)
    tok_t = torch.tensor(tokens, dtype=torch.long, device=device)
    per_token_logps = log_probs[0, idx_t, tok_t]  # [k]
    del log_probs

    # Guard: skip if any log-prob is NaN/Inf
    if torch.isnan(per_token_logps).any() or torch.isinf(per_token_logps).any():
        logger.warning("NaN/Inf in per-token log-probs, skipping step")
        return None, metrics

    # Full importance ratio: exp(current_logp - old_logp)
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)  # [k]
    coef_2 = torch.clamp(coef_1, 1.0 - epsilon, 1.0 + epsilon)  # [k]

    # Masking-rate scaling: earlier steps have fewer unmasked -> larger lambda
    num_unmasked = len(indices)
    lambda_t = gen_length / max(num_unmasked, 1)

    # PPO-clipped surrogate loss (per-token) with lambda_t scaling
    per_token_loss = -torch.min(
        coef_1 * step_advantage, coef_2 * step_advantage
    ) * lambda_t  # [k]

    # k2 KL penalty (quadratic, NOT reverse KL)
    # kl = (ref_logps - per_token_logps)^2 / 2
    if beta > 0 and ref_per_token_logps.numel() > 0:
        log_ratio = ref_per_token_logps - per_token_logps  # [k]
        per_token_kl = log_ratio ** 2 / 2.0  # [k]
        # Guard: replace NaN KL with 0
        per_token_kl = torch.where(
            torch.isnan(per_token_kl),
            torch.zeros_like(per_token_kl),
            per_token_kl,
        )
        per_token_loss = per_token_loss + beta * per_token_kl
        metrics["kl_loss"] = per_token_kl.mean().detach().item()

    # Average over tokens in this step
    step_loss = per_token_loss.mean()

    # Metrics
    with torch.no_grad():
        metrics["policy_loss"] = (
            -torch.min(coef_1 * step_advantage, coef_2 * step_advantage)
        ).mean().item()
        metrics["ratio_mean"] = coef_1.mean().item()
        metrics["clipped_frac"] = (
            (coef_1 < 1.0 - epsilon) | (coef_1 > 1.0 + epsilon)
        ).float().mean().item()
        metrics["lambda_t"] = lambda_t

    return step_loss, metrics


# ---------------------------------------------------------------------------
# Helper: build ground truth action strings for proxy reward
# ---------------------------------------------------------------------------


def build_gt_action_strings(ground_truth_fields: dict) -> list[str]:
    """Convert ground_truth_fields dict to action strings for proxy reward.

    Args:
        ground_truth_fields: {field_name: value} mapping from training data.

    Returns:
        List of action strings like "fill_field <field_name> <value>".
    """
    gt_actions = []
    for field_name, value in ground_truth_fields.items():
        gt_actions.append(f"fill_field {field_name} {value}")
    return gt_actions


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def train():
    """Run ReFusion MDPO training with browser execution."""
    model_config = FLOW_LLM_CONFIG
    mdpo_config = MDPO_REFUSION_CONFIG

    # Load tokenizer
    trust_remote_code = model_config.get("trust_remote_code", True)
    logger.info("Loading tokenizer: %s", model_config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine if loading from SFT checkpoint
    sft_checkpoint = os.environ.get("FLOW_LLM_SFT_CHECKPOINT", "")
    is_peft_checkpoint = sft_checkpoint and Path(sft_checkpoint).exists()

    # ---------------------------------------------------------------
    # Load policy model with QLoRA
    # ---------------------------------------------------------------
    if is_peft_checkpoint:
        logger.info("Loading SFT checkpoint: %s", sft_checkpoint)
        base_model = load_quantized_model(model_config["model_name"], model_config)
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        policy_model = PeftModel.from_pretrained(
            base_model, sft_checkpoint, is_trainable=True
        )
        policy_model.train()
    else:
        if sft_checkpoint:
            logger.warning(
                "SFT checkpoint not found at %s, training from base model",
                sft_checkpoint,
            )
        logger.info("Loading base model: %s", model_config["model_name"])
        policy_model = load_quantized_model(model_config["model_name"], model_config)
        policy_model.config.use_cache = False
        policy_model = prepare_model_for_kbit_training(
            policy_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        lora_config = LoraConfig(
            r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            target_modules=model_config["lora_target_modules"],
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_config)

    policy_model.print_trainable_parameters()

    # Wrap in FlowLLM for generation
    mask_token_id = model_config.get("mask_token_id", 151670)
    flow_policy = FlowLLM(policy_model, tokenizer, mask_token_id=mask_token_id)

    # ---------------------------------------------------------------
    # Load reference model (frozen, for KL penalty)
    # ---------------------------------------------------------------
    logger.info("Loading reference model (frozen)")
    ref_base = load_quantized_model(model_config["model_name"], model_config)
    if is_peft_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_base, sft_checkpoint)
    else:
        ref_model = ref_base
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------------
    # Training data and optimizer
    # ---------------------------------------------------------------
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
        shuffle=mdpo_config.get("shuffle_prompts", True),
    )

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=mdpo_config["learning_rate"],
        weight_decay=mdpo_config.get("weight_decay", 0.01),
    )

    # Warmup scheduler
    warmup_steps = mdpo_config.get("warmup_steps", 5)
    total_training_steps = min(
        len(prompts) * mdpo_config["num_epochs"],
        mdpo_config.get("early_stop_max_steps", 40),
    )

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    group_size = mdpo_config["group_size"]
    kl_coeff = mdpo_config["kl_coeff"]
    epsilon = mdpo_config["epsilon"]
    mu = mdpo_config.get("mu", 1)
    max_seq_length = model_config["max_seq_length"]
    num_gen_steps = mdpo_config.get("num_generation_steps", 64)
    gen_temperature = mdpo_config.get("generation_temperature", 1.0)
    confidence_noise_std = mdpo_config.get("confidence_noise_std", 0.0)
    action_timeout = mdpo_config.get("action_timeout_s", 5.0)
    grad_clip = mdpo_config.get("grad_clip", 1.0)
    sample_train_steps = mdpo_config.get("sample_train_steps", 8)
    min_nonzero = mdpo_config.get("min_nonzero_for_update", 1)
    max_steps = mdpo_config.get("early_stop_max_steps", 40)
    checkpoint_every = mdpo_config.get("checkpoint_every_steps", 10)

    # ---------------------------------------------------------------
    # Start FormFactory server and browser
    # ---------------------------------------------------------------
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = mdpo_config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    headless = mdpo_config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(
        "Starting ReFusion MDPO: %d prompts, G=%d, kl=%.4f, eps=%.2f, "
        "mu=%d, K=%d, T_gen=%d, temp=%.1f, warmup=%d, max_steps=%d",
        len(prompts),
        group_size,
        kl_coeff,
        epsilon,
        mu,
        sample_train_steps,
        num_gen_steps,
        gen_temperature,
        warmup_steps,
        max_steps,
    )

    total_steps = 0
    best_avg_reward = -1.0
    best_step = -1
    best_checkpoint_dir = Path("outputs/mdpo_refusion/best")

    try:
        for epoch in range(mdpo_config["num_epochs"]):
            logger.info("Epoch %d/%d", epoch + 1, mdpo_config["num_epochs"])
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

                # Build ground truth action strings for proxy reward
                gt_action_strs = build_gt_action_strings(ground_truth_fields)

                # Periodic browser restart to prevent session degradation
                if prompt_idx > 0 and prompt_idx % 5 == 0:
                    logger.info("Periodic browser restart (prompt %d)", prompt_idx)
                    await browser_env.restart()

                # Tokenize condition
                condition_enc = tokenizer(
                    instruction,
                    return_tensors="pt",
                    max_length=max_seq_length // 2,
                    truncation=True,
                ).to(flow_policy.device)

                prompt_len = condition_enc["attention_mask"].sum().item()
                gen_length = max(1, max_seq_length - prompt_len)

                # ==========================================================
                # Phase 1: Generate G rollouts with trajectory recording
                # ==========================================================
                trajectories = []
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    trajectory = flow_policy.generate_with_trajectory(
                        condition_ids=condition_enc["input_ids"],
                        condition_mask=condition_enc["attention_mask"],
                        seq_length=gen_length,
                        num_steps=num_gen_steps,
                        temperature=gen_temperature,
                        confidence_noise_std=confidence_noise_std,
                    )
                    trajectories.append(trajectory)
                    text = tokenizer.decode(
                        trajectory.final_tokens[0], skip_special_tokens=True
                    )
                    rollout_texts.append(text)
                policy_model.train()

                # ==========================================================
                # Phase 1.5: Compute per-step proxy rewards
                # ==========================================================
                # Navigate once to get element_map for proxy reward
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
                    logger.warning(
                        "Navigation failed for proxy reward: %s", e
                    )
                    element_map = {}

                num_traj_steps = max(
                    len(traj.steps) for traj in trajectories
                ) if trajectories else 0

                step_rewards = torch.zeros(
                    group_size, num_traj_steps,
                    dtype=torch.float32,
                    device=flow_policy.device,
                )

                condition_ids_list = condition_enc["input_ids"][0].tolist()

                for g in range(group_size):
                    traj = trajectories[g]
                    for s_idx, step in enumerate(traj.steps):
                        # Build current state by applying all unmaskings up to this step
                        current_tokens = (
                            condition_ids_list
                            + [mask_token_id] * gen_length
                        )
                        for prev_s in range(s_idx + 1):
                            prev_step = traj.steps[prev_s]
                            for pos, tok in zip(
                                prev_step.newly_unmasked_indices[0],
                                prev_step.unmasked_tokens[0],
                            ):
                                if prompt_len + pos < len(current_tokens):
                                    current_tokens[prompt_len + pos] = tok

                        # Decode response portion only
                        response_tokens = current_tokens[prompt_len:]
                        text = tokenizer.decode(
                            response_tokens, skip_special_tokens=True
                        )
                        proxy_r = compute_proxy_reward(
                            text, element_map, gt_action_strs
                        )
                        step_rewards[g, s_idx] = proxy_r

                # ==========================================================
                # Phase 2: Execute final rollouts in browser for terminal reward
                # ==========================================================
                browser_rewards = []
                for g, rollout_text in enumerate(rollout_texts):
                    await browser_env.reset()

                    try:
                        await browser_env.tools.navigate(
                            url=form_url,
                            new_tab=False,
                            browser_session=browser_env.browser_session,
                        )
                        await asyncio.sleep(0.5)
                        element_map_g = await browser_env.get_element_map()
                    except Exception as e:
                        logger.warning(
                            "Navigation failed for rollout %d: %s", g, e
                        )
                        browser_rewards.append(0.0)
                        continue

                    actions = parse_rollout_to_actions(rollout_text, element_map_g)
                    if not actions:
                        logger.warning(
                            "No valid actions parsed from rollout %d. "
                            "Generated text (first 300 chars): %.300s",
                            g, rollout_text,
                        )
                        browser_rewards.append(0.0)
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
                        weights=mdpo_config.get("reward_weights"),
                    )
                    browser_rewards.append(reward)
                    logger.info(
                        "Rollout %d: browser_reward=%.3f (actions_executed=%d/%d)",
                        g, reward,
                        outcome.actions_executed
                        if hasattr(outcome, "actions_executed")
                        else -1,
                        len(actions),
                    )

                # Replace final step proxy reward with browser reward
                for g in range(group_size):
                    traj = trajectories[g]
                    if len(traj.steps) > 0:
                        step_rewards[g, len(traj.steps) - 1] = browser_rewards[g]

                epoch_rewards.extend(browser_rewards)

                # ==========================================================
                # Phase 3: Check zero-reward skip
                # ==========================================================
                nonzero_count = sum(1 for r in browser_rewards if r > 0)
                if nonzero_count < min_nonzero:
                    logger.info(
                        "Step %d: skipping update (nonzero=%d < min=%d)",
                        total_steps, nonzero_count, min_nonzero,
                    )
                    total_steps += 1
                    scheduler.step()
                    continue

                # ==========================================================
                # Phase 4: Compute temporal advantages [G, T]
                # ==========================================================
                advantages = compute_temporal_advantages(step_rewards)

                # ==========================================================
                # Phase 5: Select top-k training steps
                # ==========================================================
                selected_steps = select_training_steps(
                    advantages, k=sample_train_steps
                )
                logger.info(
                    "Step %d: selected %d/%d training steps: %s",
                    total_steps, len(selected_steps), num_traj_steps,
                    selected_steps,
                )

                # ==========================================================
                # Phase 6: Cache old/ref per-token log-probs at SELECTED steps
                # ==========================================================
                # Structure: cached_logprobs[g][step_idx] = (old_logps, ref_logps)
                cached_logprobs: dict[int, dict[int, tuple]] = {}

                for g in range(group_size):
                    traj = trajectories[g]
                    cached_logprobs[g] = {}

                    for step_idx in selected_steps:
                        if step_idx >= len(traj.steps):
                            continue

                        step = traj.steps[step_idx]

                        old_logps = cache_step_logprobs(
                            model=policy_model,
                            step=step,
                            condition_length=traj.condition_length,
                            temperature=gen_temperature,
                        )

                        ref_logps = cache_step_logprobs(
                            model=ref_model,
                            step=step,
                            condition_length=traj.condition_length,
                            temperature=gen_temperature,
                        )

                        cached_logprobs[g][step_idx] = (old_logps, ref_logps)

                torch.cuda.empty_cache()

                # ==========================================================
                # Phase 7: Policy update (mu iterations with MDPO loss)
                # ==========================================================
                total_loss_val = 0.0
                total_kl_val = 0.0
                total_ratio_val = 0.0
                total_clipped_frac = 0.0
                total_lambda_t = 0.0
                num_loss_terms = 0

                num_selected = max(len(selected_steps), 1)

                for mu_iter in range(mu):
                    optimizer.zero_grad()
                    iter_loss_val = 0.0

                    for g in range(group_size):
                        traj = trajectories[g]

                        if len(traj.steps) == 0:
                            continue

                        for step_idx in selected_steps:
                            if step_idx >= len(traj.steps):
                                continue

                            if step_idx not in cached_logprobs[g]:
                                continue

                            step = traj.steps[step_idx]
                            old_logps, ref_logps = cached_logprobs[g][step_idx]

                            # Skip steps with no unmasked positions
                            if old_logps.numel() == 0:
                                continue

                            # Per-step temporal advantage for this rollout
                            step_adv = advantages[g, step_idx].item()

                            # Skip when advantage is near zero
                            if abs(step_adv) < 1e-10:
                                continue

                            step_loss, step_metrics = compute_mdpo_step_loss(
                                model=policy_model,
                                step=step,
                                condition_length=traj.condition_length,
                                old_per_token_logps=old_logps,
                                ref_per_token_logps=ref_logps,
                                step_advantage=step_adv,
                                epsilon=epsilon,
                                beta=kl_coeff,
                                gen_length=gen_length,
                                temperature=gen_temperature,
                            )

                            if step_loss is None:
                                continue

                            # Divide by num_selected for denoising reduction
                            scaled_loss = step_loss / num_selected
                            # Per-step backward to release activations immediately
                            scaled_loss.backward()
                            iter_loss_val += scaled_loss.detach().item()

                            total_kl_val += step_metrics["kl_loss"]
                            total_ratio_val += step_metrics["ratio_mean"]
                            total_clipped_frac += step_metrics["clipped_frac"]
                            total_lambda_t += step_metrics["lambda_t"]
                            num_loss_terms += 1

                    # Clip gradients and step (guard against all-zero iteration)
                    if iter_loss_val != 0.0 and not (
                        iter_loss_val != iter_loss_val  # NaN check
                    ):
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                        optimizer.step()

                    total_loss_val += iter_loss_val

                scheduler.step()
                torch.cuda.empty_cache()

                total_steps += 1
                avg_reward = (
                    sum(browser_rewards) / len(browser_rewards)
                    if browser_rewards
                    else 0
                )
                avg_kl = total_kl_val / max(num_loss_terms, 1)
                avg_ratio = total_ratio_val / max(num_loss_terms, 1)
                avg_clipped = total_clipped_frac / max(num_loss_terms, 1)
                avg_lambda = total_lambda_t / max(num_loss_terms, 1)
                epoch_kl.append(avg_kl)

                # Track best checkpoint
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_step = total_steps
                    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    policy_model.save_pretrained(str(best_checkpoint_dir))
                    tokenizer.save_pretrained(str(best_checkpoint_dir))
                    logger.info(
                        "New best checkpoint: avg_reward=%.3f at step %d",
                        best_avg_reward, best_step,
                    )

                # Periodic intermediate checkpoint
                if total_steps % checkpoint_every == 0:
                    step_dir = Path(
                        "outputs/mdpo_refusion/step_%d" % total_steps
                    )
                    step_dir.mkdir(parents=True, exist_ok=True)
                    policy_model.save_pretrained(str(step_dir))
                    tokenizer.save_pretrained(str(step_dir))
                    persist_checkpoint(
                        str(step_dir),
                        "mdpo-refusion/step_%d" % total_steps,
                    )
                    logger.info("Intermediate checkpoint saved to %s", step_dir)

                if total_steps % mdpo_config["logging_steps"] == 0:
                    logger.info(
                        "Step %d (prompt %d/%d): avg_reward=%.3f, "
                        "loss=%.4f, kl=%.4f, ratio=%.3f, clipped=%.2f, "
                        "lambda=%.2f, lr=%.2e, best_reward=%.3f@%d",
                        total_steps,
                        prompt_idx + 1,
                        len(prompts),
                        avg_reward,
                        total_loss_val,
                        avg_kl,
                        avg_ratio,
                        avg_clipped,
                        avg_lambda,
                        scheduler.get_last_lr()[0],
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
        final_dir = Path("outputs/mdpo_refusion/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(
            "ReFusion MDPO complete. Final saved to %s, best (%.3f@%d) at %s",
            final_dir, best_avg_reward, best_step, best_checkpoint_dir,
        )

        persist_checkpoint(str(best_checkpoint_dir), "mdpo-refusion")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
