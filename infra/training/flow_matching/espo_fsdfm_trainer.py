"""FS-DFM ESPO trainer: Sequence-level ELBO policy optimization.

Implements ESPO (Li et al., 2025, arXiv:2512.03759) adapted for FS-DFM 1.3B
Poisson jump discrete flow matching. The ELBO is computed using the GKL loss
as a proxy: sample random timestep, noise the clean output via the forward
process, compute GKL loss, and negate to get the ELBO.

Key differences from fsdfm_flow_grpo_trainer.py:
    - No trajectory recording needed (only final generated sequence)
    - GKL-based ELBO replaces per-step Poisson jump log-probs
    - Sequence-level importance ratio instead of per-step REINFORCE
    - k2 quadratic KL estimator instead of k3 Schulman
    - Multiple policy updates (mu) over same rollout batch (configurable)
    - Coupled perturbation for ELBO variance reduction

Key differences from ESPO on masked diffusion (ReFusion):
    - Forward process uses MixtureDiscreteProbPath (uniform noise) not binary masking
    - ELBO inner term is the GKL loss from compute_generalized_kl_loss()
    - Timestep sampling is continuous t ~ U(0,1) not discrete mask count

Reference: https://github.com/ML-GSAI/ESPO

Usage:
    uv run infra/training/flow_matching/espo_fsdfm_trainer.py
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    ESPO_FSDFM_CONFIG,
    FSDFM_MODEL_CONFIG,
)
from infra.training.flow_matching.fsdfm_model import (
    MixtureDiscreteProbPath,
    PolynomialConvexScheduler,
    compute_generalized_kl_loss,
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
    logger.info("Loaded %d prompts for FS-DFM ESPO", len(records))
    return records


def compute_fsdfm_elbo(
    model,
    x_1: torch.Tensor,
    edit_mask: torch.Tensor,
    scheduler: PolynomialConvexScheduler,
    vocab_size: int,
    num_mc: int = 2,
    coupled: bool = True,
) -> torch.Tensor:
    """Compute ELBO for FS-DFM via random noising and GKL loss.

    For each MC sample:
        1. Sample random timestep t ~ U(epsilon, 1-epsilon)
        2. Noise clean tokens via forward process: x_t ~ P(x_t | x_0, x_1, t)
        3. Forward pass model at (x_t, t) to get logits
        4. Compute GKL loss at response positions
        5. Negate to get ELBO (GKL loss is negative ELBO)

    If coupled=True, also evaluate at the complementary timestep (1-t)
    and average the two estimates for variance reduction.

    Args:
        model: FS-DFM DiT model (with gradients if training).
        x_1: [B, L] clean (generated) token IDs.
        edit_mask: [B, L] bool tensor (True = response, False = prefix).
        scheduler: PolynomialConvexScheduler instance.
        vocab_size: Vocabulary size for uniform noise sampling.
        num_mc: Number of MC samples.
        coupled: Whether to use coupled (antithetic) timestep sampling.

    Returns:
        [B] ELBO values (higher = better fit).
    """
    B, L = x_1.shape
    device = x_1.device
    prob_path = MixtureDiscreteProbPath(scheduler)

    L_resp = edit_mask.float().sum(dim=1)  # [B]

    elbo_accum = torch.zeros(B, device=device)
    num_estimates = 0

    for mc_idx in range(num_mc):
        # Sample random timestep t ~ U(0.01, 0.99) (avoid boundaries)
        t = torch.rand(B, device=device) * 0.98 + 0.01  # [B]

        # Sample noise (uniform over vocab)
        x_0 = torch.randint(0, vocab_size, (B, L), device=device)

        # Forward process: x_t from mixture path
        path_sample = prob_path.sample(x_0, x_1, t)
        x_t = path_sample.x_t  # [B, L]

        # Freeze prefix positions
        x_t = torch.where(edit_mask, x_t, x_1)

        # Forward pass to get logits
        logits = model(x_t, t)  # [B, L, V]

        # Compute GKL loss per token (without reduction)
        sched = scheduler(t)
        alpha_t = sched["alpha_t"]
        d_alpha_t = sched["d_alpha_t"]

        jump_coeff = d_alpha_t / (1.0 - alpha_t).clamp(min=1e-6)
        jump_coeff = jump_coeff.unsqueeze(-1)  # [B, 1]

        log_probs = F.log_softmax(logits.float(), dim=-1)  # [B, L, V] in float32
        probs = torch.exp(log_probs)

        log_p_x1 = log_probs.gather(2, x_1.unsqueeze(-1)).squeeze(-1)  # [B, L]
        p_xt = probs.gather(2, x_t.unsqueeze(-1)).squeeze(-1)  # [B, L]

        delta = (x_1 == x_t).float()

        # GKL loss per token (positive = loss, negative ELBO)
        gkl_per_token = -jump_coeff * (p_xt - delta + (1.0 - delta) * log_p_x1)

        # Mask to response only
        loss_mask = edit_mask.float()
        gkl_masked = gkl_per_token * loss_mask

        # ELBO = negative of GKL loss, summed over response positions
        # The sum gives the full-sequence ELBO at this timestep
        elbo_sample = -gkl_masked.sum(dim=1)  # [B]

        # Delete large tensors
        del logits, log_probs, probs, gkl_per_token, gkl_masked
        torch.cuda.empty_cache()

        elbo_accum = elbo_accum + elbo_sample
        num_estimates += 1

        # Coupled perturbation: complementary timestep (1 - t)
        if coupled:
            t_comp = 1.0 - t  # [B]
            # Clamp to avoid exact 0 or 1
            t_comp = t_comp.clamp(min=0.01, max=0.99)

            path_sample_comp = prob_path.sample(x_0, x_1, t_comp)
            x_t_comp = path_sample_comp.x_t
            x_t_comp = torch.where(edit_mask, x_t_comp, x_1)

            logits_comp = model(x_t_comp, t_comp)

            sched_comp = scheduler(t_comp)
            alpha_t_comp = sched_comp["alpha_t"]
            d_alpha_t_comp = sched_comp["d_alpha_t"]

            jump_coeff_comp = d_alpha_t_comp / (1.0 - alpha_t_comp).clamp(min=1e-6)
            jump_coeff_comp = jump_coeff_comp.unsqueeze(-1)

            log_probs_comp = F.log_softmax(logits_comp.float(), dim=-1)
            probs_comp = torch.exp(log_probs_comp)

            log_p_x1_comp = log_probs_comp.gather(2, x_1.unsqueeze(-1)).squeeze(-1)
            p_xt_comp = probs_comp.gather(2, x_t_comp.unsqueeze(-1)).squeeze(-1)

            delta_comp = (x_1 == x_t_comp).float()

            gkl_comp = -jump_coeff_comp * (
                p_xt_comp - delta_comp + (1.0 - delta_comp) * log_p_x1_comp
            )
            gkl_comp_masked = gkl_comp * loss_mask
            elbo_coupled = -gkl_comp_masked.sum(dim=1)

            del logits_comp, log_probs_comp, probs_comp, gkl_comp, gkl_comp_masked
            torch.cuda.empty_cache()

            elbo_accum = elbo_accum + elbo_coupled
            num_estimates += 1

    elbo = elbo_accum / max(num_estimates, 1)
    return elbo  # [B]


async def train():
    """Run FS-DFM ESPO training with browser execution."""
    model_config = FSDFM_MODEL_CONFIG
    espo_config = ESPO_FSDFM_CONFIG
    vocab_size = model_config["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if espo_config.get("bf16") else torch.float16

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
        "Starting FS-DFM ESPO: %d prompts, G=%d, kl=%.4f, mu=%d, "
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
    best_checkpoint_dir = Path("outputs/espo_fsdfm/best")

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
                # Phase 1: Generate G rollouts (no trajectory needed)
                # ==========================================================
                rollout_full_seqs = []
                rollout_edit_masks = []
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    response_ids = generate_with_prefix_conditioning(
                        model=policy_model,
                        prefix_ids=prefix_ids,
                        gen_length=gen_length,
                        config={
                            **model_config,
                            "num_generation_steps": num_gen_steps,
                        },
                        scheduler=scheduler,
                        temperature=gen_temperature,
                    )  # [1, gen_length]

                    # Build full sequence and edit mask
                    full_seq = torch.cat([prefix_ids, response_ids], dim=1)  # [1, L]
                    edit_mask = torch.zeros(
                        1, full_seq.shape[1], dtype=torch.bool, device=device
                    )
                    edit_mask[:, prefix_len:] = True

                    rollout_full_seqs.append(full_seq)
                    rollout_edit_masks.append(edit_mask)

                    text = tokenizer.decode(
                        response_ids[0], skip_special_tokens=True
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
                    advantages, dtype=torch.float32, device=device
                )

                # ==========================================================
                # Phase 5: Cache old ELBO and ref ELBO (no gradients)
                # ==========================================================
                old_elbos = []
                ref_elbos = []
                L_resp = gen_length

                with torch.no_grad():
                    for g in range(group_size):
                        old_elbo = compute_fsdfm_elbo(
                            model=policy_model,
                            x_1=rollout_full_seqs[g],
                            edit_mask=rollout_edit_masks[g],
                            scheduler=scheduler,
                            vocab_size=vocab_size,
                            num_mc=num_mc,
                            coupled=coupled,
                        )  # [1]
                        old_elbos.append(old_elbo.detach())

                        if kl_coeff > 0:
                            # Move ref model to GPU
                            ref_model.to(device)
                            ref_elbo = compute_fsdfm_elbo(
                                model=ref_model,
                                x_1=rollout_full_seqs[g],
                                edit_mask=rollout_edit_masks[g],
                                scheduler=scheduler,
                                vocab_size=vocab_size,
                                num_mc=num_mc,
                                coupled=coupled,
                            )  # [1]
                            ref_model.to("cpu")
                            ref_elbos.append(ref_elbo.detach())

                torch.cuda.empty_cache()

                # ==========================================================
                # Phase 6: Policy update (mu iterations)
                # ==========================================================
                total_loss_val = 0.0
                total_kl_val = 0.0

                for mu_iter in range(mu):
                    optimizer.zero_grad()

                    for g in range(group_size):
                        adv_g = advantages_t[g]

                        # Compute current ELBO (WITH gradients)
                        current_elbo = compute_fsdfm_elbo(
                            model=policy_model,
                            x_1=rollout_full_seqs[g],
                            edit_mask=rollout_edit_masks[g],
                            scheduler=scheduler,
                            vocab_size=vocab_size,
                            num_mc=num_mc,
                            coupled=coupled,
                        )  # [1]

                        # Sequence-level importance ratio
                        log_ratio = (current_elbo - old_elbos[g]) / max(L_resp, 1)
                        rho = torch.exp(log_ratio)

                        # PPO-style clipping
                        clipped_rho = torch.clamp(
                            rho, 1.0 - epsilon_low, 1.0 + epsilon_high
                        )

                        # Surrogate loss
                        surr1 = rho * adv_g
                        surr2 = clipped_rho * adv_g
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # k2 KL penalty
                        kl_loss = torch.tensor(0.0, device=device)
                        if kl_coeff > 0 and ref_elbos:
                            kl_log_ratio = (current_elbo - ref_elbos[g]) / max(L_resp, 1)
                            kl_loss = 0.5 * (kl_log_ratio ** 2).mean()
                            total_kl_val += kl_loss.detach().item()

                        rollout_loss = (policy_loss + kl_coeff * kl_loss) / group_size
                        rollout_loss.backward()
                        total_loss_val += rollout_loss.detach().item()

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
                    save_lora_weights(
                        policy_model,
                        str(best_checkpoint_dir / "lora_weights.pt"),
                    )
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
        final_dir = Path("outputs/espo_fsdfm/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        save_lora_weights(policy_model, str(final_dir / "lora_weights.pt"))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(
            "FS-DFM ESPO complete. Final saved to %s, best (%.3f@%d) at %s",
            final_dir, best_avg_reward, best_step, best_checkpoint_dir,
        )

        persist_checkpoint(str(best_checkpoint_dir), "espo-fsdfm")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
