"""CJ-GRPO trainer for ReFusion 8B: Trajectory-consistent PPO for masked diffusion.

Implements Consistent Joint GRPO (CJ-GRPO) for ReFusion's iterative unmasking
process. This extends the Flow-GRPO trainer with three key differences:

1. **Full PPO clipping**: Computes importance ratios exp(logp - old_logp) and
   applies min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) instead of the
   REINFORCE simplification (-A * logp) used when ratio=1 always holds.

2. **Cached per-token log-probs**: After generating each trajectory, forward-
   passes the policy and reference models on each step to cache old_per_token
   and ref_per_token log-probs at newly unmasked positions. These are frozen
   and reused across mu policy update iterations.

3. **mu > 1 policy updates**: Outer loop over mu optimizer steps per prompt
   batch (like ESPO), enabled by the cached old/ref log-probs and importance
   ratios that correctly track policy drift.

Per-step loss:
    coef_1 = exp(per_token_logps - old_per_token_logps)
    coef_2 = clamp(coef_1, 1 - epsilon, 1 + epsilon)
    per_token_loss = -min(coef_1 * A, coef_2 * A)
    per_token_kl = exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
    per_token_loss = per_token_loss + beta * per_token_kl
    step_loss = per_token_loss.mean() / num_sampled_steps

Architecture:
    1. Generate G rollouts via iterative unmasking, recording trajectories
    2. Execute rollouts in headless browser against FormFactory
    3. Compute group-relative advantages from browser rewards
    4. Cache old/ref per-token log-probs at newly unmasked positions
    5. For mu iterations:
       a. For each rollout, sample K denoising steps from trajectory
       b. Recompute per-token log-probs (with gradients)
       c. PPO-clipped surrogate + reverse KL penalty (per-step backward)
       d. Gradient accumulation across steps, single optimizer.step()

Usage:
    uv run infra/training/flow_matching/cjgrpo_refusion_trainer.py
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infra.training.flow_matching.config import (
    CJGRPO_REFUSION_CONFIG,
    DATA_CONFIG,
    FLOW_LLM_CONFIG,
)
from infra.training.flow_matching.flow_llm_model import (
    FlowLLM,
    UnmaskingTrajectoryStep,
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


def load_quantized_model(model_name: str, config: dict):
    """Load ReFusion with 4-bit quantization."""
    compute_dtype = (
        torch.bfloat16
        if config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    trust_remote_code = config.get("trust_remote_code", True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for CJ-GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info("Loaded %d prompts for ReFusion CJ-GRPO", len(records))
    return records


def cache_step_logprobs(
    model,
    step: UnmaskingTrajectoryStep,
    condition_length: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute per-token log-probs at newly unmasked positions for one step.

    Unlike compute_unmasking_step_log_prob which sums log-probs into a scalar
    per batch element, this returns the individual per-token log-probs so they
    can be used for importance ratio computation in CJ-GRPO.

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

    # Forward pass (no labels, raw logits)
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


def compute_cjgrpo_step_loss(
    model,
    step: UnmaskingTrajectoryStep,
    condition_length: int,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    advantage: float,
    epsilon: float,
    beta: float,
    temperature: float = 1.0,
) -> tuple[torch.Tensor | None, dict]:
    """Compute CJ-GRPO loss for one denoising step with PPO clipping.

    Forward-passes the model with gradients, computes per-token importance
    ratios, applies PPO clipping, and adds reverse KL penalty.

    Args:
        model: Policy model (with gradients).
        step: Trajectory step with masked_state and unmasked info.
        condition_length: L_c (prompt token count).
        old_per_token_logps: Cached old policy per-token log-probs [k].
        ref_per_token_logps: Cached reference model per-token log-probs [k].
        advantage: Scalar advantage for this rollout.
        epsilon: PPO clip range.
        beta: KL penalty coefficient.
        temperature: Softmax temperature.

    Returns:
        (loss_tensor_or_None, metrics_dict) where loss has gradients attached.
        Returns (None, metrics) if the step should be skipped.
    """
    device = step.masked_state.device
    metrics = {"policy_loss": 0.0, "kl_loss": 0.0, "ratio_mean": 0.0, "clipped_frac": 0.0}

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

    # PPO-clipped surrogate loss (per-token)
    per_token_loss = -torch.min(coef_1 * advantage, coef_2 * advantage)  # [k]

    # Reverse KL penalty (per-token)
    # kl = exp(ref - current) - (ref - current) - 1
    if beta > 0 and ref_per_token_logps.numel() > 0:
        log_r = ref_per_token_logps - per_token_logps  # [k]
        per_token_kl = torch.exp(log_r) - log_r - 1.0  # [k]
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
        metrics["policy_loss"] = (-torch.min(coef_1 * advantage, coef_2 * advantage)).mean().item()
        metrics["ratio_mean"] = coef_1.mean().item()
        metrics["clipped_frac"] = (
            (coef_1 < 1.0 - epsilon) | (coef_1 > 1.0 + epsilon)
        ).float().mean().item()

    return step_loss, metrics


async def train():
    """Run ReFusion CJ-GRPO training with browser execution."""
    model_config = FLOW_LLM_CONFIG
    cjgrpo_config = CJGRPO_REFUSION_CONFIG

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
    confidence_noise_std = cjgrpo_config.get("confidence_noise_std", 0.0)
    action_timeout = cjgrpo_config.get("action_timeout_s", 5.0)
    grad_clip = cjgrpo_config.get("grad_clip", 1.0)
    num_sampled_timesteps = cjgrpo_config.get("num_sampled_timesteps", 8)
    min_nonzero = cjgrpo_config.get("min_nonzero_for_update", 1)
    max_steps = cjgrpo_config.get("early_stop_max_steps", 40)
    checkpoint_every = cjgrpo_config.get("checkpoint_every_steps", 10)

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
        "Starting ReFusion CJ-GRPO: %d prompts, G=%d, kl=%.4f, eps=%.2f, "
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
    best_checkpoint_dir = Path("outputs/cjgrpo_refusion/best")

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

                # Periodic browser restart to reset DOM indices
                if prompt_idx > 0 and prompt_idx % 10 == 0:
                    logger.info("Periodic browser restart (prompt %d)", prompt_idx)
                    await browser_env.close()
                    browser_env = await BrowserEnvironment.create(headless=headless)

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
                    advantages, dtype=torch.float32, device=flow_policy.device
                )

                # ==========================================================
                # Phase 5: Cache old/ref per-token log-probs for each step
                # ==========================================================
                # For each rollout g, for each trajectory step, cache:
                #   old_per_token_logps: policy model log-probs at generation time
                #   ref_per_token_logps: reference model log-probs
                # These are frozen and reused across mu iterations.

                # Structure: cached_logprobs[g] = list of (old_logps, ref_logps) per step
                cached_logprobs = []

                for g in range(group_size):
                    traj = trajectories[g]
                    rollout_cache = []

                    for step in traj.steps:
                        # Old policy log-probs (frozen snapshot at generation time)
                        old_logps = cache_step_logprobs(
                            model=policy_model,
                            step=step,
                            condition_length=traj.condition_length,
                            temperature=gen_temperature,
                        )

                        # Reference model log-probs
                        ref_logps = cache_step_logprobs(
                            model=ref_model,
                            step=step,
                            condition_length=traj.condition_length,
                            temperature=gen_temperature,
                        )

                        rollout_cache.append((old_logps, ref_logps))

                    cached_logprobs.append(rollout_cache)

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

                        # Denoising reduction: sample K random timesteps
                        num_sampled = min(num_sampled_timesteps, len(traj.steps))
                        sampled_indices = sorted(
                            random.sample(range(len(traj.steps)), num_sampled)
                        )
                        num_steps_denom = max(num_sampled, 1)

                        for step_idx in sampled_indices:
                            step = traj.steps[step_idx]
                            old_logps, ref_logps = cached_logprobs[g][step_idx]

                            # Skip steps with no unmasked positions
                            if old_logps.numel() == 0:
                                continue

                            step_loss, step_metrics = compute_cjgrpo_step_loss(
                                model=policy_model,
                                step=step,
                                condition_length=traj.condition_length,
                                old_per_token_logps=old_logps,
                                ref_per_token_logps=ref_logps,
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
                    policy_model.save_pretrained(str(best_checkpoint_dir))
                    tokenizer.save_pretrained(str(best_checkpoint_dir))
                    logger.info(
                        "New best checkpoint: avg_reward=%.3f at step %d",
                        best_avg_reward, best_step,
                    )

                # Periodic intermediate checkpoint
                if total_steps % checkpoint_every == 0:
                    step_dir = Path(
                        "outputs/cjgrpo_refusion/step_%d" % total_steps
                    )
                    step_dir.mkdir(parents=True, exist_ok=True)
                    policy_model.save_pretrained(str(step_dir))
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
        final_dir = Path("outputs/cjgrpo_refusion/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(
            "ReFusion CJ-GRPO complete. Final saved to %s, best (%.3f@%d) at %s",
            final_dir, best_avg_reward, best_step, best_checkpoint_dir,
        )

        persist_checkpoint(str(best_checkpoint_dir), "cjgrpo-refusion")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
