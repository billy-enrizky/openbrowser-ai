"""MDPO (Sequential MDP) trainer for FS-DFM 1.3B: Temporal-advantage PPO for discrete flow matching.

Implements Masked Diffusion Policy Optimization (MDPO) for FS-DFM's Poisson jump
process. This adapts the MDPO algorithm from the ReFusion trainer to FS-DFM's
discrete flow matching architecture with sequence-level log-probs.

Key differences from ReFusion MDPO:
    - GPT-2 tokenizer (vocab_size=50257)
    - PolynomialConvexScheduler(exponent=2.0) for noise schedule
    - EulerTrajectory with EulerTrajectoryStep (x_t, x_next, t_value)
    - Sequence-level log-probs from compute_discrete_step_log_prob (not per-token)
    - PPO clipping on [B] scalar ratios, not per-token vectors
    - lambda_t = gen_length / max(num_changed, 1) where num_changed counts
      response positions that changed between x_t and x_next
    - k2 KL: (ref - cur)^2 / 2 on sequence-level scalars
    - Reference model on CPU (swapped to GPU only for KL computation)
    - LoRA via custom inject_lora (not PEFT/QLoRA)

Architecture:
    1. Generate G rollouts via Euler trajectory, recording steps
    2. Compute per-step proxy rewards by decoding step.x_next response portion
    3. Execute final rollouts in headless browser for terminal reward
    4. Compute temporal advantages (adv-v3 + adv-v4) over [G, T] reward tensor
    5. Select top-k training steps by advantage magnitude (with diversity guard)
    6. Cache old/ref sequence-level log-probs at SELECTED steps only
    7. For mu iterations:
       a. Recompute sequence-level log-probs (with gradients)
       b. PPO-clipped surrogate + k2 KL + lambda_t scaling (per-step backward)
       c. Gradient accumulation across steps, single optimizer.step()

Per-step loss (sequence-level):
    cur_log_prob = compute_discrete_step_log_prob(...)  # [B]
    log_ratio = cur_log_prob - old_log_prob  # [B]
    coef_1 = exp(log_ratio)
    coef_2 = clamp(coef_1, 1 - epsilon, 1 + epsilon)
    lambda_t = gen_length / max(num_changed, 1)
    policy_loss = -min(coef_1 * A, coef_2 * A) * lambda_t  # [B]
    kl_loss = (ref_log_prob - cur_log_prob)^2 / 2  # [B]
    step_loss = (policy_loss.mean() + beta * kl_loss.mean()) / num_selected_steps

Usage:
    uv run infra/training/flow_matching/mdpo_fsdfm_trainer.py
"""

import asyncio
import logging
import os
from pathlib import Path

import torch

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FSDFM_MODEL_CONFIG,
    MDPO_FSDFM_CONFIG,
)
from infra.training.flow_matching.fsdfm_model import (
    EulerTrajectoryStep,
    PolynomialConvexScheduler,
    compute_discrete_step_log_prob,
    generate_with_prefix_conditioning,
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
from infra.training.shared.proxy_reward import compute_proxy_reward
from infra.training.shared.utils import (
    compute_temporal_advantages,
    load_prompts,
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
# Cache sequence-level log-probs (same pattern as CJ-GRPO FS-DFM)
# ---------------------------------------------------------------------------


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

    FS-DFM's compute_discrete_step_log_prob returns a [B] scalar (mean over
    response positions). This is the correct granularity for FS-DFM's Poisson
    jump process.

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


# ---------------------------------------------------------------------------
# MDPO step loss (PPO + lambda_t + k2 KL, sequence-level)
# ---------------------------------------------------------------------------


def compute_mdpo_discrete_step_loss(
    model,
    step: EulerTrajectoryStep,
    dt: float,
    scheduler: PolynomialConvexScheduler,
    vocab_size: int,
    response_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    step_advantage: float,
    epsilon: float,
    beta: float,
    gen_length: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor | None, dict]:
    """Compute MDPO loss for one discrete Euler step with PPO clipping and lambda_t.

    Differences from CJ-GRPO FS-DFM:
    - step_advantage: per-step temporal advantage (not single scalar for rollout)
    - lambda_t = gen_length / max(num_changed, 1): masking-rate scaling where
      num_changed counts response positions that changed between x_t and x_next
    - k2 KL: (ref - cur)^2 / 2, NOT reverse KL exp(ref-cur) - (ref-cur) - 1

    Args:
        model: Policy model (with gradients).
        step: Trajectory step with x_t and x_next.
        dt: Euler step size.
        scheduler: PolynomialConvexScheduler instance.
        vocab_size: Token vocabulary size.
        response_mask: [B, L] float, 1 for response, 0 for prefix.
        old_log_prob: Cached old policy log-prob [B] (frozen snapshot).
        ref_log_prob: Cached reference model log-prob [B] (frozen).
        step_advantage: Scalar temporal advantage for this step.
        epsilon: PPO clip range.
        beta: KL penalty coefficient.
        gen_length: Total generation length (for lambda_t computation).
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
        "lambda_t": 0.0,
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

    # Guard: skip if ANY log-prob is NaN/Inf (cur, old, or ref).
    # old_log_prob NaN was the silent killer: cur - NaN = NaN ratio,
    # .backward() on NaN loss corrupts ALL accumulated gradients.
    if torch.isnan(cur_log_prob).any() or torch.isinf(cur_log_prob).any():
        logger.warning(
            "NaN/Inf cur_log_prob at step t=%.4f, skipping", step.t_value
        )
        return None, metrics
    if torch.isnan(old_log_prob).any() or torch.isinf(old_log_prob).any():
        logger.warning(
            "NaN/Inf old_log_prob at step t=%.4f, skipping", step.t_value
        )
        return None, metrics
    if torch.isnan(ref_log_prob).any() or torch.isinf(ref_log_prob).any():
        logger.warning(
            "NaN/Inf ref_log_prob at step t=%.4f, skipping", step.t_value
        )
        return None, metrics

    # Sequence-level importance ratio: exp(current - old)
    log_ratio = cur_log_prob - old_log_prob  # [B]
    ratio = torch.exp(log_ratio)  # [B]

    # StableDRL unconditional clipping: ALWAYS clip ratio, regardless of
    # advantage sign. Standard PPO min() has a "trapdoor" when A<0 and
    # ratio>1+eps that allows noise-induced gradient spikes (arXiv:2603.06743).
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)  # [B]

    # Masking-rate scaling: count response positions that changed between x_t and x_next
    # response_mask is [B, L] float; use it to restrict to response portion only
    num_changed = (
        (step.x_t[0] != step.x_next[0]) & response_mask[0].bool()
    ).sum().item()
    lambda_t = min(gen_length / max(num_changed, 1), 10.0)  # clamp to prevent amplification

    # Self-normalized surrogate loss with unconditional clipping + lambda_t scaling
    weighted = clipped_ratio * step_advantage * lambda_t  # [B]
    policy_loss = -weighted.sum() / clipped_ratio.detach().sum().clamp(min=1e-8)

    # k2 KL penalty (quadratic, NOT reverse KL)
    # kl = (ref_log_prob - cur_log_prob)^2 / 2
    kl_loss = torch.zeros_like(cur_log_prob)
    if beta > 0:
        kl_diff = ref_log_prob - cur_log_prob  # [B]
        kl_loss = kl_diff ** 2 / 2.0  # [B]
        # Guard: replace NaN KL with 0
        kl_loss = torch.where(
            torch.isnan(kl_loss),
            torch.zeros_like(kl_loss),
            kl_loss,
        )

    step_loss = policy_loss + beta * kl_loss.mean()

    # Metrics (detached)
    with torch.no_grad():
        metrics["policy_loss"] = policy_loss.item()
        metrics["kl_loss"] = kl_loss.mean().item()
        metrics["ratio_mean"] = ratio.mean().item()
        metrics["clipped_frac"] = (
            (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
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
    """Run FS-DFM MDPO training with browser execution."""
    model_config = FSDFM_MODEL_CONFIG
    mdpo_config = MDPO_FSDFM_CONFIG
    vocab_size = model_config["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 (exponent range up to 3.4e38) prevents activation overflow that
    # causes NaN in the 21-block transformer forward pass. fp16 (max 65504)
    # overflows ~83% of per-step log-prob computations. The old concern
    # about bf16 in log-prob chains is moot after the F.log_softmax refactor.
    compute_dtype = torch.bfloat16 if mdpo_config.get("bf16", False) else torch.float16

    # Load GPT-2 tokenizer (native to FS-DFM)
    from transformers import AutoTokenizer

    logger.info("Loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Flow matching scheduler
    exponent = model_config.get("scheduler_exponent", 2.0)
    flow_scheduler = PolynomialConvexScheduler(exponent=exponent)

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

    # Enable gradient checkpointing: recomputes block activations during
    # backward to avoid bf16 gradient overflow through 21 DDiTBlocks.
    policy_model.gradient_checkpointing = True
    logger.info("Gradient checkpointing enabled for policy model")

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
    max_steps = mdpo_config.get("early_stop_max_steps", 40)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    group_size = mdpo_config["group_size"]
    if group_size < 2:
        raise ValueError(
            "group_size must be >= 2 (1 greedy baseline + at least 1 stochastic)"
        )
    kl_coeff = mdpo_config["kl_coeff"]
    epsilon = mdpo_config["epsilon"]
    mu = mdpo_config.get("mu", 1)
    max_seq_length = model_config["max_seq_length"]
    num_gen_steps = mdpo_config.get("num_generation_steps", 64)
    gen_temperature = mdpo_config.get("generation_temperature", 1.0)
    action_timeout = mdpo_config.get("action_timeout_s", 5.0)
    grad_clip = mdpo_config.get("grad_clip", 1.0)
    sample_train_steps = mdpo_config.get("sample_train_steps", 8)
    min_nonzero = mdpo_config.get("min_nonzero_for_update", 1)
    checkpoint_every = mdpo_config.get("checkpoint_every_steps", 10)
    dt = 1.0 / num_gen_steps

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
        "Starting FS-DFM MDPO: %d prompts, G=%d, kl=%.4f, eps=%.2f, "
        "mu=%d, K=%d, T_gen=%d, temp=%.1f, warmup=%d, max_steps=%d, bf16=%s",
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
        mdpo_config.get("bf16"),
    )

    # ---------------------------------------------------------------
    # Diagnostic: verify model generates coherent text before training
    # ---------------------------------------------------------------
    if prompts:
        diag_prompt = prompts[0]
        diag_inst = diag_prompt.get("instruction", diag_prompt.get("condition", ""))
        diag_enc = tokenizer(
            diag_inst, add_special_tokens=True, truncation=True,
            max_length=max_seq_length // 2, return_tensors="pt",
        )
        diag_prefix = diag_enc["input_ids"].to(device)
        diag_gen_len = max(1, max_seq_length - diag_prefix.shape[1])
        policy_model.eval()
        diag_ids = generate_with_prefix_conditioning(
            model=policy_model,
            prefix_ids=diag_prefix,
            gen_length=diag_gen_len,
            config={**model_config, "num_sampling_steps": num_gen_steps},
            scheduler=flow_scheduler,
            temperature=0.0,
        )
        diag_text = tokenizer.decode(diag_ids[0], skip_special_tokens=True)
        logger.info(
            "DIAGNOSTIC greedy generation (first 300 chars): %.300s",
            diag_text,
        )
        policy_model.train()

    total_steps = 0
    best_avg_reward = -1.0
    best_step = -1
    _ckpt = os.environ.get("CHECKPOINT_NAME", "mdpo-fsdfm")
    best_checkpoint_dir = Path("outputs/%s/best" % _ckpt.replace("-", "_"))

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
                # Rollout 0 = greedy baseline (no trajectory, reward only)
                # Rollouts 1..G-1 = stochastic with trajectory for gradient
                # ==========================================================
                trajectories = []
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    if g == 0:
                        # Greedy baseline: deterministic generation for reward
                        # signal. No trajectory needed (excluded from gradient).
                        response_ids = generate_with_prefix_conditioning(
                            model=policy_model,
                            prefix_ids=prefix_ids,
                            gen_length=gen_length,
                            config={
                                **model_config,
                                "num_sampling_steps": num_gen_steps,
                            },
                            scheduler=flow_scheduler,
                            temperature=0.0,
                        )
                        trajectories.append(None)
                        text = tokenizer.decode(
                            response_ids[0], skip_special_tokens=True
                        )
                    else:
                        # Stochastic rollout with trajectory for gradient
                        trajectory = generate_with_prefix_conditioning_trajectory(
                            model=policy_model,
                            prefix_ids=prefix_ids,
                            gen_length=gen_length,
                            config={
                                **model_config,
                                "num_generation_steps": num_gen_steps,
                            },
                            scheduler=flow_scheduler,
                            temperature=gen_temperature,
                        )
                        trajectories.append(trajectory)
                        response_ids = trajectory.final_tokens[0, prefix_len:]
                        text = tokenizer.decode(
                            response_ids, skip_special_tokens=True
                        )
                    rollout_texts.append(text)
                policy_model.train()

                # ==========================================================
                # Phase 1.5: Compute per-step proxy rewards
                # ==========================================================
                # Navigate once to get element_map for proxy reward
                await browser_env.reset()
                try:
                    element_map = await browser_env.safe_navigate(
                        url=form_url, nav_timeout=30.0, max_retries=2,
                    )
                except RuntimeError as e:
                    logger.error(
                        "Navigation failed for proxy reward (all step "
                        "rewards will be 0 for this prompt): %s", e
                    )
                    element_map = {}

                num_traj_steps = max(
                    (len(traj.steps) for traj in trajectories if traj is not None),
                    default=0,
                )

                step_rewards = torch.zeros(
                    group_size, num_traj_steps,
                    dtype=torch.float32,
                    device=device,
                )

                for g in range(group_size):
                    traj = trajectories[g]
                    if traj is None:
                        # Greedy baseline (g=0): no trajectory steps
                        continue
                    for s_idx, step in enumerate(traj.steps):
                        # Decode the response portion of step.x_next
                        response_ids = step.x_next[0, prefix_len:]
                        text = tokenizer.decode(
                            response_ids, skip_special_tokens=True
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
                        element_map_g = await browser_env.safe_navigate(
                            url=form_url, nav_timeout=30.0, max_retries=2,
                        )
                    except RuntimeError as e:
                        logger.error(
                            "All navigation attempts failed for rollout %d: %s",
                            g, e,
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
                    if traj is not None and len(traj.steps) > 0:
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
                    # Still save periodic checkpoints even when skipping
                    if total_steps % checkpoint_every == 0:
                        ckpt_name = os.environ.get("CHECKPOINT_NAME", "mdpo-fsdfm")
                        step_dir = Path(
                            "outputs/%s/step_%d" % (ckpt_name.replace("-", "_"), total_steps)
                        )
                        step_dir.mkdir(parents=True, exist_ok=True)
                        save_lora_weights(
                            policy_model,
                            str(step_dir / "lora_weights.pt"),
                        )
                        tokenizer.save_pretrained(str(step_dir))
                        persist_checkpoint(
                            str(step_dir),
                            "%s/step_%d" % (ckpt_name, total_steps),
                        )
                        logger.info(
                            "Intermediate checkpoint saved to %s", step_dir
                        )
                    total_steps += 1
                    lr_scheduler.step()
                    continue

                # ==========================================================
                # Phase 4: Compute temporal advantages [G-1, T]
                # Exclude greedy baseline (row 0) to avoid biasing
                # the group normalization with an all-zero phantom row.
                # ==========================================================
                stochastic_step_rewards = step_rewards[1:, :]  # [G-1, T]
                advantages = compute_temporal_advantages(stochastic_step_rewards)

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
                # Phase 6: Cache old/ref sequence-level log-probs at SELECTED steps
                # ==========================================================
                # Structure: cached_logprobs[g][step_idx] = (old_lp, ref_lp)
                cached_logprobs: dict[int, dict[int, tuple]] = {}

                # Move ref model to GPU for caching
                if kl_coeff > 0:
                    ref_model.to(device)

                # eval mode for ALL forward passes in log-prob computation.
                # Disables dropout (0.1) for deterministic, reproducible
                # log-prob values across caching and optimization phases.
                policy_model.eval()

                for g in range(group_size):
                    traj = trajectories[g]
                    if traj is None:
                        # Greedy baseline (g=0): no trajectory, skip caching
                        continue
                    response_mask = traj.edit_mask.float()  # [B, L]
                    cached_logprobs[g] = {}

                    for step_idx in selected_steps:
                        if step_idx >= len(traj.steps):
                            continue

                        step = traj.steps[step_idx]

                        old_lp = cache_discrete_step_logprobs(
                            model=policy_model,
                            step=step,
                            dt=dt,
                            scheduler=flow_scheduler,
                            vocab_size=vocab_size,
                            response_mask=response_mask,
                            temperature=gen_temperature,
                        )

                        if kl_coeff > 0:
                            ref_lp = cache_discrete_step_logprobs(
                                model=ref_model,
                                step=step,
                                dt=dt,
                                scheduler=flow_scheduler,
                                vocab_size=vocab_size,
                                response_mask=response_mask,
                                temperature=gen_temperature,
                            )
                        else:
                            ref_lp = torch.zeros_like(old_lp)

                        cached_logprobs[g][step_idx] = (old_lp, ref_lp)

                # Move ref model back to CPU after caching
                if kl_coeff > 0:
                    ref_model.to("cpu")
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
                        if traj is None:
                            # Greedy baseline (g=0): no trajectory, skip
                            continue

                        if len(traj.steps) == 0:
                            continue

                        response_mask = traj.edit_mask.float()  # [B, L]

                        for step_idx in selected_steps:
                            if step_idx >= len(traj.steps):
                                continue

                            if step_idx not in cached_logprobs[g]:
                                continue

                            step = traj.steps[step_idx]
                            old_lp, ref_lp = cached_logprobs[g][step_idx]

                            # Per-step temporal advantage for this rollout
                            # advantages is [G-1, T] (excludes greedy g=0)
                            step_adv = advantages[g - 1, step_idx].item()

                            # Skip when advantage is near zero
                            if abs(step_adv) < 1e-10:
                                continue

                            step_loss, step_metrics = compute_mdpo_discrete_step_loss(
                                model=policy_model,
                                step=step,
                                dt=dt,
                                scheduler=flow_scheduler,
                                vocab_size=vocab_size,
                                response_mask=response_mask,
                                old_log_prob=old_lp,
                                ref_log_prob=ref_lp,
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

                lr_scheduler.step()
                # Restore train mode after all forward passes are done
                policy_model.train()
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
                    ckpt_name = os.environ.get("CHECKPOINT_NAME", "mdpo-fsdfm")
                    step_dir = Path(
                        "outputs/%s/step_%d" % (ckpt_name.replace("-", "_"), total_steps)
                    )
                    step_dir.mkdir(parents=True, exist_ok=True)
                    save_lora_weights(
                        policy_model,
                        str(step_dir / "lora_weights.pt"),
                    )
                    tokenizer.save_pretrained(str(step_dir))
                    persist_checkpoint(
                        str(step_dir),
                        "%s/step_%d" % (ckpt_name, total_steps),
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
                        lr_scheduler.get_last_lr()[0],
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
        ckpt_name = os.environ.get("CHECKPOINT_NAME", "mdpo-fsdfm")
        final_dir = Path("outputs/%s/final" % ckpt_name.replace("-", "_"))
        final_dir.mkdir(parents=True, exist_ok=True)
        save_lora_weights(
            policy_model, str(final_dir / "lora_weights.pt")
        )
        tokenizer.save_pretrained(str(final_dir))
        logger.info(
            "FS-DFM MDPO complete. Final saved to %s, best (%.3f@%d) at %s",
            final_dir, best_avg_reward, best_step, best_checkpoint_dir,
        )

        persist_checkpoint(str(final_dir), ckpt_name)

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
