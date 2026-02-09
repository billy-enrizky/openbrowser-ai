"""FS-DFM Online GRPO trainer: True discrete flow matching with browser execution.

Uses FS-DFM 1.3B (Apple) with LoRA for GRPO training against FormFactory
browser forms. This is the STAD80 true discrete flow matching counterpart
to the STAD68 autoregressive GRPO trainer.

Architecture:
    1. FS-DFM (DiT + LoRA) generates G candidate plans via discrete Euler solver
       with prefix conditioning (instruction tokens fixed, response denoised)
    2. Each plan is decoded to text via GPT-2 tokenizer and parsed into actions
    3. Actions are executed in a headless browser against FormFactory
    4. Reward = form submission success + field accuracy
    5. Advantage-weighted flow matching loss updates LoRA parameters
    6. KL penalty via flow-matching logit divergence against frozen reference model

Usage:
    uv run infra/training/flow_matching/fsdfm_online_grpo_trainer.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import torch

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FSDFM_MODEL_CONFIG,
    ONLINE_FSDFM_GRPO_CONFIG,
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


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} prompts for FS-DFM online GRPO")
    return records


def compute_flow_kl(
    policy_model,
    ref_model,
    x_1: torch.Tensor,
    loss_mask: torch.Tensor,
    scheduler: PolynomialConvexScheduler,
    prob_path: MixtureDiscreteProbPath,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute flow-matching KL divergence between policy and reference.

    Sample a random t, create x_t, get logits from both models,
    compute KL on response tokens.
    """
    B, L = x_1.shape

    # Sample noise and timestep
    x_0 = torch.randint(0, vocab_size, (B, L), device=device)
    eps = 1e-4
    t = torch.rand(B, device=device) * (1.0 - 2 * eps) + eps

    # Sample x_t
    path_sample = prob_path.sample(x_0, x_1, t)
    x_t = path_sample.x_t

    # Get logits from both models
    with torch.no_grad():
        ref_logits = ref_model(x_t, t)

    policy_logits = policy_model(x_t, t)

    # KL(policy || ref) on response tokens
    policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

    # Per-token KL: sum_v p(v) * (log p(v) - log q(v))
    policy_probs = torch.exp(policy_log_probs)
    kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)

    # Mask to response tokens
    kl_per_token = kl_per_token * loss_mask
    kl = kl_per_token.sum() / loss_mask.sum().clamp(min=1)

    return kl


async def train():
    """Run online FS-DFM GRPO training with browser execution."""
    model_config = FSDFM_MODEL_CONFIG
    grpo_config = ONLINE_FSDFM_GRPO_CONFIG
    vocab_size = model_config["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if grpo_config.get("bf16") else torch.float16

    # Load GPT-2 tokenizer
    from transformers import AutoTokenizer
    logger.info("Loading GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Flow matching components
    exponent = model_config.get("scheduler_exponent", 2.0)
    scheduler = PolynomialConvexScheduler(exponent=exponent)
    prob_path = MixtureDiscreteProbPath(scheduler)

    # Load policy model
    sft_checkpoint = os.environ.get("FSDFM_SFT_CHECKPOINT", "")
    logger.info("Loading FS-DFM 1.3B policy model")
    policy_model = load_fsdfm_from_huggingface(model_config, device=device, dtype=compute_dtype)
    policy_model = inject_lora(policy_model, model_config)

    if sft_checkpoint and Path(sft_checkpoint).exists():
        logger.info(f"Loading SFT checkpoint: {sft_checkpoint}")
        lora_path = Path(sft_checkpoint) / "lora_weights.pt"
        if lora_path.exists():
            load_lora_weights(policy_model, str(lora_path))
        else:
            logger.warning(f"lora_weights.pt not found at {lora_path}, starting from base LoRA init")
    elif sft_checkpoint:
        logger.warning(f"SFT checkpoint not found at {sft_checkpoint}, training from base LoRA init")

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

    # Optimizer (only policy LoRA params)
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=grpo_config["learning_rate"]
    )

    group_size = grpo_config["group_size"]
    kl_coeff = grpo_config["kl_coeff"]
    max_seq_length = model_config["max_seq_length"]
    num_sampling_steps = grpo_config.get("num_sampling_steps", model_config.get("num_sampling_steps", 64))
    gen_temperature = model_config.get("generation_temperature", 1.0)
    action_timeout = grpo_config.get("action_timeout_s", 5.0)
    grad_clip = grpo_config.get("grad_clip", 1.0)

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
        f"Starting FS-DFM online GRPO: {len(prompts)} prompts, G={group_size}, "
        f"kl_coeff={kl_coeff}, sampling_steps={num_sampling_steps}"
    )

    total_steps = 0
    try:
        for epoch in range(grpo_config["num_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{grpo_config['num_epochs']}")
            epoch_rewards = []
            epoch_kl = []

            for i, prompt_data in enumerate(prompts):
                instruction = prompt_data.get(
                    "instruction", prompt_data.get("condition", "")
                )
                form_url = prompt_data.get("url", "")
                ground_truth_fields = prompt_data.get("ground_truth_fields", {})

                if not instruction or not form_url:
                    logger.warning(f"Skipping prompt {i}: missing instruction or url")
                    continue

                # Periodic browser restart to reset DOM indices
                if i > 0 and i % 10 == 0:
                    logger.info(f"Periodic browser restart (prompt {i})")
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

                # Generate G rollouts via discrete Euler solver
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
                        logger.warning(f"Navigation failed for rollout {g}: {e}")
                        rewards.append(0.0)
                        continue

                    actions = parse_rollout_to_actions(rollout_text, element_map)
                    if not actions:
                        logger.debug(f"No valid actions parsed from rollout {g}")
                        rewards.append(0.0)
                        continue

                    outcome = await browser_env.execute_actions(
                        actions, timeout_per_action=action_timeout
                    )
                    reward = compute_online_reward(
                        outcome,
                        ground_truth_fields,
                        weights=grpo_config.get("reward_weights"),
                    )
                    rewards.append(reward)

                epoch_rewards.extend(rewards)

                # Compute GRPO advantages
                advantages = compute_grpo_advantages(rewards, group_size)
                advantages_t = torch.tensor(
                    advantages, dtype=torch.float32, device=device
                )

                # Compute advantage-weighted flow matching loss
                total_loss = torch.tensor(0.0, device=device, requires_grad=False)
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
                        logger.debug(f"Empty rollout {g}, skipping loss")
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

                    # Flow matching loss for this rollout
                    x_1 = full_ids
                    x_0 = torch.randint(0, vocab_size, x_1.shape, device=device)
                    eps = 1e-4
                    t = torch.rand(1, device=device) * (1.0 - 2 * eps) + eps

                    path_sample = prob_path.sample(x_0, x_1, t)
                    x_t = path_sample.x_t

                    logits = policy_model(x_t, t)
                    flow_loss = compute_generalized_kl_loss(
                        logits=logits,
                        x_1=x_1,
                        x_t=x_t,
                        t=t,
                        scheduler=scheduler,
                        loss_mask=loss_mask,
                    )

                    # KL penalty
                    kl = compute_flow_kl(
                        policy_model, ref_model,
                        x_1, loss_mask,
                        scheduler, prob_path, vocab_size, device,
                    )

                    # Advantage-weighted loss + KL
                    g_loss = advantages_t[g] * flow_loss + kl_coeff * kl
                    total_loss = total_loss + g_loss
                    total_kl = total_kl + kl
                    valid_rollouts += 1

                divisor = max(valid_rollouts, 1)
                loss = total_loss / divisor

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
                        f"  Step {total_steps} (prompt {i+1}/{len(prompts)}): "
                        f"avg_reward={avg_reward:.3f}, "
                        f"loss={loss.item():.4f}, "
                        f"kl={avg_kl:.4f}"
                    )

            # Epoch summary
            if epoch_rewards:
                epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
                nonzero = sum(1 for r in epoch_rewards if r > 0)
                logger.info(
                    f"Epoch {epoch + 1} complete: avg_reward={epoch_avg:.3f}, "
                    f"nonzero_rewards={nonzero}/{len(epoch_rewards)}, "
                    f"avg_kl={sum(epoch_kl) / len(epoch_kl):.4f}"
                )

        # Save final model
        final_dir = Path("outputs/fsdfm_online_grpo/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        save_lora_weights(policy_model, str(final_dir / "lora_weights.pt"))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(f"FS-DFM GRPO complete. Model saved to {final_dir}")

        persist_checkpoint(str(final_dir.parent), "online-fsdfm-grpo")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
