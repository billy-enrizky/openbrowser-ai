"""ReFusion ESPO trainer: Sequence-level ELBO policy optimization.

Implements ESPO (Li et al., 2025, arXiv:2512.03759) for ReFusion 8B masked
diffusion. Replaces the token-level per-step REINFORCE of Flow-GRPO with
sequence-level ELBO importance ratios:

    rho = exp((ELBO_theta(y) - ELBO_theta_old(y)) / L)

where ELBO is computed by randomly re-masking the completed output and measuring
cross-entropy at masked positions, weighted by L/l (importance weighting).

Key differences from refusion_flow_grpo_trainer.py:
    - No trajectory recording needed (only final generated sequence)
    - ELBO replaces per-step unmasking log-probs
    - Sequence-level importance ratio instead of per-step REINFORCE
    - k2 quadratic KL estimator instead of k3 Schulman
    - Multiple policy updates (mu) over same rollout batch (configurable)
    - Coupled perturbation for ELBO variance reduction

Reference: https://github.com/ML-GSAI/ESPO

Usage:
    uv run infra/training/flow_matching/espo_refusion_trainer.py
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
    ESPO_REFUSION_CONFIG,
    FLOW_LLM_CONFIG,
)
from infra.training.flow_matching.flow_llm_model import FlowLLM
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


def load_prompts(
    file_path: str, max_samples: int = 0, shuffle: bool = False, seed: int = 42
) -> list[dict]:
    """Load prompts for ESPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)
        logger.info("Shuffled %d prompts with seed %d", len(records), seed)
    logger.info("Loaded %d prompts for ReFusion ESPO", len(records))
    return records


def compute_masked_diffusion_elbo(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    condition_length: int,
    mask_token_id: int,
    num_mc: int = 2,
    coupled: bool = True,
) -> torch.Tensor:
    """Compute the ELBO for a masked diffusion model via random re-masking.

    For each MC sample:
        1. Sample random mask count l from {1, ..., L_response}
        2. Mask l random response positions
        3. Forward pass, compute cross-entropy at masked positions
        4. Weight by L_response / l (importance weighting)

    If coupled=True, also compute the complement mask (unmask the masked
    positions, mask the unmasked ones) and average the two estimates. This
    halves ELBO variance (antithetic sampling).

    Args:
        model: ReFusion model (with gradients if training).
        input_ids: [B, L] full sequence (prompt + response).
        attention_mask: [B, L] attention mask.
        condition_length: Number of prompt tokens (frozen).
        mask_token_id: Token ID for masking (151670 for ReFusion).
        num_mc: Number of MC samples for ELBO estimation.
        coupled: Whether to use coupled (antithetic) perturbation.

    Returns:
        [B] ELBO values (higher = better fit, i.e. negative of the loss).
    """
    B, L = input_ids.shape
    L_resp = L - condition_length
    device = input_ids.device

    if L_resp <= 0:
        return torch.zeros(B, device=device)

    # Response token positions
    resp_indices = torch.arange(condition_length, L, device=device)

    elbo_accum = torch.zeros(B, device=device)
    num_estimates = 0

    for mc_idx in range(num_mc):
        # Sample random mask count l ~ Uniform({1, ..., L_resp})
        l = torch.randint(1, L_resp + 1, (B,), device=device)

        # Create random permutation for each batch element to select l positions
        rand_perm = torch.rand(B, L_resp, device=device).argsort(dim=1)

        # Build mask: True = mask this position
        mask_indices = torch.zeros(B, L_resp, dtype=torch.bool, device=device)
        for b in range(B):
            mask_indices[b, rand_perm[b, :l[b]]] = True

        # Apply mask to input_ids
        masked_ids = input_ids.clone()
        for b in range(B):
            positions_to_mask = resp_indices[mask_indices[b]]
            masked_ids[b, positions_to_mask] = mask_token_id

        # Forward pass (no labels, no prompt_lengths -- raw logits)
        outputs = model(input_ids=masked_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L, V]

        # Extract response logits
        resp_logits = logits[:, condition_length:, :]  # [B, L_resp, V]
        resp_targets = input_ids[:, condition_length:]  # [B, L_resp]

        # Cross-entropy at masked positions only
        # Flatten for cross_entropy, then reshape
        ce_per_token = F.cross_entropy(
            resp_logits.reshape(-1, resp_logits.shape[-1]),
            resp_targets.reshape(-1),
            reduction="none",
        ).reshape(B, L_resp)  # [B, L_resp]

        # Zero out non-masked positions
        ce_masked = ce_per_token * mask_indices.float()

        # Sum CE at masked positions, weight by L_resp / l
        # p_mask = l / (L_resp + 1) for stratified sampling
        # Importance weight = 1 / p_mask = (L_resp + 1) / l
        # But for uniform l, the ELBO estimate is: sum(CE_masked) / l * L_resp
        # Following ESPO: weight each token's CE by 1/p_mask where p_mask = l/(L_resp+1)
        p_mask = l.float() / (L_resp + 1)  # [B]
        elbo_sample = -(ce_masked.sum(dim=1) / l.float()) * L_resp  # [B]

        # Delete large tensors
        del outputs, logits, resp_logits, ce_per_token, ce_masked
        torch.cuda.empty_cache()

        elbo_accum = elbo_accum + elbo_sample
        num_estimates += 1

        # Coupled perturbation: complement mask
        if coupled:
            complement_mask = ~mask_indices
            # l_comp = L_resp - l (number of newly masked positions)
            l_comp = L_resp - l  # [B]

            # Skip if any batch element has l_comp <= 0
            if (l_comp <= 0).any():
                continue

            coupled_ids = input_ids.clone()
            for b in range(B):
                positions_to_mask = resp_indices[complement_mask[b]]
                coupled_ids[b, positions_to_mask] = mask_token_id

            coupled_outputs = model(
                input_ids=coupled_ids, attention_mask=attention_mask
            )
            coupled_logits = coupled_outputs.logits[:, condition_length:, :]
            coupled_ce = F.cross_entropy(
                coupled_logits.reshape(-1, coupled_logits.shape[-1]),
                resp_targets.reshape(-1),
                reduction="none",
            ).reshape(B, L_resp)

            coupled_ce_masked = coupled_ce * complement_mask.float()
            elbo_coupled = -(coupled_ce_masked.sum(dim=1) / l_comp.float()) * L_resp

            del coupled_outputs, coupled_logits, coupled_ce, coupled_ce_masked
            torch.cuda.empty_cache()

            elbo_accum = elbo_accum + elbo_coupled
            num_estimates += 1

    # Average over all estimates
    elbo = elbo_accum / max(num_estimates, 1)
    return elbo  # [B]


async def train():
    """Run ReFusion ESPO training with browser execution."""
    model_config = FLOW_LLM_CONFIG
    espo_config = ESPO_REFUSION_CONFIG

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
    shuffle_prompts = espo_config.get("shuffle_prompts", True)
    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
        shuffle=shuffle_prompts,
    )

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=espo_config["learning_rate"],
        weight_decay=espo_config.get("weight_decay", 0.01),
    )

    group_size = espo_config["group_size"]
    kl_coeff = espo_config["kl_coeff"]
    epsilon_low = espo_config.get("epsilon_low", 0.2)
    epsilon_high = espo_config.get("epsilon_high", 0.2)
    max_seq_length = model_config["max_seq_length"]
    num_gen_steps = espo_config.get("num_generation_steps", 64)
    gen_temperature = espo_config.get("generation_temperature", 1.0)
    action_timeout = espo_config.get("action_timeout_s", 5.0)
    grad_clip = espo_config.get("grad_clip", 1.0)
    mu = espo_config.get("mu", 1)
    num_mc = espo_config.get("num_mc_samples", 2)
    coupled = espo_config.get("coupled_perturbation", True)
    min_nonzero = espo_config.get("min_nonzero_for_update", 1)
    max_steps = espo_config.get("early_stop_max_steps", 40)

    # ---------------------------------------------------------------
    # Start FormFactory server and browser
    # ---------------------------------------------------------------
    formfactory_dir = PROJECT_ROOT / "data" / "formfactory"
    port = espo_config.get("formfactory_port", 5050)
    ff_server = FormFactoryServer(formfactory_dir, port=port)
    if not ff_server.start():
        logger.error("Failed to start FormFactory server, aborting")
        return

    headless = espo_config.get("browser_headless", True)
    browser_env = await BrowserEnvironment.create(headless=headless)

    logger.info(
        "Starting ReFusion ESPO: %d prompts, G=%d, kl=%.4f, mu=%d, "
        "M=%d, coupled=%s, eps=%.2f, T_gen=%d, temp=%.1f, max_steps=%d",
        len(prompts),
        group_size,
        kl_coeff,
        mu,
        num_mc,
        coupled,
        epsilon_low,
        num_gen_steps,
        gen_temperature,
        max_steps,
    )

    total_steps = 0
    best_avg_reward = -1.0
    best_step = -1
    best_checkpoint_dir = Path("outputs/espo_refusion/best")

    try:
        for epoch in range(espo_config["num_epochs"]):
            logger.info("Epoch %d/%d", epoch + 1, espo_config["num_epochs"])
            epoch_rewards = []
            epoch_kl = []

            for i, prompt_data in enumerate(prompts):
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
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                ).to(flow_policy.device)

                prompt_len = condition_enc["attention_mask"].sum().item()
                gen_length = max(1, max_seq_length - prompt_len)

                # ==========================================================
                # Phase 1: Generate G rollouts (no trajectory needed)
                # ==========================================================
                rollout_sequences = []
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    final_tokens = flow_policy.generate(
                        condition_ids=condition_enc["input_ids"],
                        condition_mask=condition_enc["attention_mask"],
                        seq_length=gen_length,
                        num_steps=num_gen_steps,
                        temperature=gen_temperature,
                    )  # [1, gen_length]
                    # Build full sequence: prompt + generated response
                    full_seq = torch.cat(
                        [condition_enc["input_ids"], final_tokens], dim=1
                    )  # [1, L]
                    rollout_sequences.append(full_seq)
                    text = tokenizer.decode(
                        final_tokens[0], skip_special_tokens=True
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
                        weights=espo_config.get("reward_weights"),
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
                # Phase 4: Compute advantages
                # ==========================================================
                advantages = compute_grpo_advantages(rewards, group_size)
                advantages_t = torch.tensor(
                    advantages, dtype=torch.float32, device=flow_policy.device
                )

                # ==========================================================
                # Phase 5: Build attention masks for full sequences
                # ==========================================================
                full_attention_masks = []
                for seq in rollout_sequences:
                    mask = torch.ones_like(seq, dtype=torch.long)
                    full_attention_masks.append(mask)

                # ==========================================================
                # Phase 6: Cache old ELBO and ref ELBO (no gradients)
                # ==========================================================
                old_elbos = []
                ref_elbos = []
                L_resp = gen_length

                with torch.no_grad():
                    for g in range(group_size):
                        old_elbo = compute_masked_diffusion_elbo(
                            model=policy_model,
                            input_ids=rollout_sequences[g],
                            attention_mask=full_attention_masks[g],
                            condition_length=prompt_len,
                            mask_token_id=mask_token_id,
                            num_mc=num_mc,
                            coupled=coupled,
                        )  # [1]
                        old_elbos.append(old_elbo.detach())

                        if kl_coeff > 0:
                            ref_elbo = compute_masked_diffusion_elbo(
                                model=ref_model,
                                input_ids=rollout_sequences[g],
                                attention_mask=full_attention_masks[g],
                                condition_length=prompt_len,
                                mask_token_id=mask_token_id,
                                num_mc=num_mc,
                                coupled=coupled,
                            )  # [1]
                            ref_elbos.append(ref_elbo.detach())

                torch.cuda.empty_cache()

                # ==========================================================
                # Phase 7: Policy update (mu iterations)
                # ==========================================================
                total_loss_val = 0.0
                total_kl_val = 0.0

                for mu_iter in range(mu):
                    optimizer.zero_grad()
                    batch_loss = torch.tensor(0.0, device=flow_policy.device)

                    for g in range(group_size):
                        adv_g = advantages_t[g]

                        # Compute current ELBO (WITH gradients)
                        current_elbo = compute_masked_diffusion_elbo(
                            model=policy_model,
                            input_ids=rollout_sequences[g],
                            attention_mask=full_attention_masks[g],
                            condition_length=prompt_len,
                            mask_token_id=mask_token_id,
                            num_mc=num_mc,
                            coupled=coupled,
                        )  # [1]

                        # Sequence-level importance ratio
                        # rho = exp((ELBO_theta - ELBO_old) / L)
                        log_ratio = (current_elbo - old_elbos[g]) / max(L_resp, 1)
                        rho = torch.exp(log_ratio)

                        # PPO-style clipping
                        clipped_rho = torch.clamp(
                            rho, 1.0 - epsilon_low, 1.0 + epsilon_high
                        )

                        # Surrogate loss: -min(rho * A, clipped_rho * A)
                        surr1 = rho * adv_g
                        surr2 = clipped_rho * adv_g
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # k2 KL penalty: (1/2) * (ELBO_theta - ELBO_ref)^2 / L
                        kl_loss = torch.tensor(0.0, device=flow_policy.device)
                        if kl_coeff > 0 and ref_elbos:
                            kl_log_ratio = (current_elbo - ref_elbos[g]) / max(L_resp, 1)
                            kl_loss = 0.5 * (kl_log_ratio ** 2).mean()
                            total_kl_val += kl_loss.detach().item()

                        rollout_loss = (policy_loss + kl_coeff * kl_loss) / group_size
                        rollout_loss.backward()
                        total_loss_val += rollout_loss.detach().item()

                    # Clip gradients and step
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    optimizer.step()

                torch.cuda.empty_cache()

                total_steps += 1
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                avg_kl = total_kl_val / max(group_size * mu, 1)
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

                if total_steps % espo_config["logging_steps"] == 0:
                    logger.info(
                        "Step %d (prompt %d/%d): avg_reward=%.3f, "
                        "loss=%.4f, kl=%.4f, best_reward=%.3f@%d",
                        total_steps,
                        i + 1,
                        len(prompts),
                        avg_reward,
                        total_loss_val,
                        avg_kl,
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
        final_dir = Path("outputs/espo_refusion/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(
            "ReFusion ESPO complete. Final saved to %s, best (%.3f@%d) at %s",
            final_dir, best_avg_reward, best_step, best_checkpoint_dir,
        )

        persist_checkpoint(str(best_checkpoint_dir), "espo-refusion")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
