"""ReFusion Flow-GRPO trainer: Masked diffusion policy gradients for ReFusion 8B.

Implements Flow-GRPO (Liu et al., 2025) adapted for ReFusion's iterative
unmasking process.  Unlike the existing online_flow_llm_grpo_trainer.py
which uses autoregressive log-probs, this computes per-step log-probabilities
at the positions that were actually unmasked during generation, aligning the
policy gradient with the masked diffusion trajectory.

Key innovation -- masked diffusion analog of continuous Flow-GRPO:
    Continuous: SDE step gives Gaussian policy N(mu, sigma^2 dt I)
                log-prob = Gaussian log-density
    ReFusion:   Each unmasking step predicts tokens for masked positions,
                unmasks top-k most confident.
                log-prob = sum of log_softmax at newly-unmasked positions

Architecture:
    1. Generate G rollouts via iterative unmasking, recording trajectories
    2. Execute rollouts in headless browser against FormFactory
    3. Compute group-relative advantages from browser rewards
    4. For each rollout, iterate over ALL trajectory steps (denoising reduction):
       a. Recompute policy log-prob at newly-unmasked positions (with gradients)
       b. REINFORCE loss: -advantage * log_prob (per-step backward)
       c. KL penalty from Schulman k3 approximation
    5. Gradient accumulation across steps, single optimizer.step()

Reference: github.com/yifan123/flow_grpo (continuous version for images)

Usage:
    uv run infra/training/flow_matching/refusion_flow_grpo_trainer.py
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
    DATA_CONFIG,
    FLOW_GRPO_REFUSION_CONFIG,
    FLOW_LLM_CONFIG,
)
from infra.training.flow_matching.flow_llm_model import (
    FlowLLM,
    compute_unmasking_step_log_prob,
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
    """Load prompts for Flow-GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info("Loaded %d prompts for ReFusion Flow-GRPO", len(records))
    return records


async def train():
    """Run ReFusion Flow-GRPO training with browser execution."""
    model_config = FLOW_LLM_CONFIG
    grpo_config = FLOW_GRPO_REFUSION_CONFIG

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
        trainable_params, lr=grpo_config["learning_rate"]
    )

    group_size = grpo_config["group_size"]
    kl_coeff = grpo_config["kl_coeff"]
    adv_clip_max = grpo_config.get("adv_clip_max", 5.0)
    max_seq_length = model_config["max_seq_length"]
    num_gen_steps = grpo_config.get("num_generation_steps", 10)
    gen_temperature = grpo_config.get("generation_temperature", 0.7)
    confidence_noise_std = grpo_config.get("confidence_noise_std", 0.0)
    action_timeout = grpo_config.get("action_timeout_s", 5.0)
    grad_clip = grpo_config.get("grad_clip", 1.0)
    num_sampled_timesteps = grpo_config.get("num_sampled_timesteps", 8)

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
        "Starting ReFusion Flow-GRPO: %d prompts, G=%d, kl=%.3f, "
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

                # Tokenize condition
                condition_enc = tokenizer(
                    instruction,
                    add_special_tokens=False,  # Must match eval (no special tokens for ReFusion)
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
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
                    advantages, dtype=torch.float32, device=flow_policy.device
                )

                # ==========================================================
                # Phase 4: Policy gradient update over trajectory steps
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
                    # log_prob has numerical issues.
                    if adv_g.abs().item() < 1e-10:
                        continue

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
                        log_prob = compute_unmasking_step_log_prob(
                            model=policy_model,
                            step=step,
                            condition_length=traj.condition_length,
                            temperature=gen_temperature,
                        )  # [B]

                        # Guard: skip if log_prob is NaN/Inf
                        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                            logger.warning(
                                "NaN/Inf log_prob at rollout %d, skipping",
                                g,
                            )
                            continue

                        # REINFORCE policy loss (ratio=1 simplification)
                        policy_loss = (-adv_g * log_prob).mean()

                        # KL penalty (Schulman k3)
                        kl_loss = torch.tensor(0.0, device=flow_policy.device)
                        if kl_coeff > 0:
                            with torch.no_grad():
                                ref_log_prob = compute_unmasking_step_log_prob(
                                    model=ref_model,
                                    step=step,
                                    condition_length=traj.condition_length,
                                    temperature=gen_temperature,
                                )  # [B]
                            log_r = ref_log_prob - log_prob
                            kl_loss = (torch.exp(log_r) - log_r - 1).mean()
                            # Guard: replace NaN KL with 0
                            if torch.isnan(kl_loss):
                                kl_loss = torch.tensor(0.0, device=flow_policy.device)
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
        final_dir = Path("outputs/refusion_flow_grpo/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("ReFusion Flow-GRPO complete. Model saved to %s", final_dir)

        persist_checkpoint(str(final_dir), "refusion-flow-grpo")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
