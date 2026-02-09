"""Online Flow LLM GRPO trainer: ReFusion masked diffusion with browser execution.

Uses ReFusion (GSAI-ML/ReFusion) with QLoRA as the backbone for
masked diffusion, trained with GRPO (Group Relative Policy Optimization)
using real browser execution rewards from FormFactory.

Architecture:
    1. FlowLLM (ReFusion + QLoRA) generates G candidate plans via
       iterative unmasking (masked diffusion reverse process)
    2. Each plan is decoded to text and parsed into executable actions
    3. Actions are executed in a headless browser against FormFactory
    4. Reward = actual form submission success + field accuracy
    5. REINFORCE with group-relative advantages updates LoRA parameters
    6. Non-negative KL penalty (Schulman k3) against frozen reference model

ReFusion is built on Qwen3 and uses AutoModelForCausalLM, so standard
HuggingFace/BnB/PEFT loading works without any compatibility patches.
For GRPO, per-token log-probs are extracted from standard causal LM
logits (no masking), while SFT uses ReFusion's native MDM loss.

This is the STAD80 counterpart to the AR GRPO trainer (STAD68).
    - STAD68: Qwen3-8B, left-to-right autoregressive token generation
    - STAD80: ReFusion, slot-based parallel masked diffusion

Usage:
    uv run infra/training/flow_matching/online_flow_llm_grpo_trainer.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infra.training.flow_matching.config import (
    DATA_CONFIG,
    FLOW_LLM_CONFIG,
    ONLINE_FLOW_GRPO_CONFIG,
)
from infra.training.flow_matching.flow_llm_model import FlowLLM
from infra.training.shared.action_parser import parse_rollout_to_actions
from infra.training.shared.browser_env import BrowserEnvironment
from infra.training.shared.formfactory_server import FormFactoryServer
from infra.training.shared.online_reward import compute_online_reward
from infra.training.shared.reward_functions import compute_grpo_advantages
from infra.training.shared.utils import (
    persist_checkpoint,
    resolve_data_path,
)

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

    # ReFusion uses standard AutoModelForCausalLM -- no compatibility patches needed
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} prompts for online flow LLM GRPO")
    return records


def compute_per_token_log_probs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probs for response tokens using standard causal LM forward.

    Uses logits from a plain forward pass (no labels, no masking) so that
    ReFusion's internal masked diffusion is NOT applied -- we get clean
    autoregressive log-probs suitable for REINFORCE and KL computation.

    Args:
        model: Policy or reference model (ReFusion/PEFT).
        input_ids: [B, L] full sequence (prompt + response + padding).
        attention_mask: [B, L] attention mask.
        prompt_length: Length of the prompt prefix.

    Returns:
        token_log_probs: [B, max_resp_len] per-token log-probs for response.
        resp_mask: [B, max_resp_len] mask (1 for real response tokens, 0 for padding).
    """
    B = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    response_start = max(0, prompt_length - 1)  # shifted by 1 for causal LM
    max_resp_len = seq_len - 1 - response_start
    if max_resp_len <= 0:
        device = input_ids.device
        return torch.zeros(B, 1, device=device), torch.zeros(B, 1, device=device)

    all_log_probs = []
    all_masks = []

    for i in range(B):
        ids_i = input_ids[i : i + 1]
        mask_i = attention_mask[i : i + 1]

        with torch.set_grad_enabled(model.training):
            # Plain forward (no labels) to get standard causal LM logits
            outputs = model(input_ids=ids_i, attention_mask=mask_i)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[0, :-1, :]  # [L-1, V]
        shift_labels = ids_i[0, 1:]  # [L-1]

        token_nll = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        token_log_probs = -token_nll  # [L-1]

        # Extract response portion
        response_log_probs = token_log_probs[response_start:]
        mask_shifted = mask_i[0, 1:][response_start:].float()

        all_log_probs.append(response_log_probs[:max_resp_len])
        all_masks.append(mask_shifted[:max_resp_len])

    return torch.stack(all_log_probs), torch.stack(all_masks)


async def train():
    """Run online flow LLM GRPO training with browser execution."""
    model_config = FLOW_LLM_CONFIG
    grpo_config = ONLINE_FLOW_GRPO_CONFIG

    # Load tokenizer
    trust_remote_code = model_config.get("trust_remote_code", True)
    logger.info(f"Loading tokenizer: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine if loading from SFT checkpoint
    sft_checkpoint = os.environ.get("FLOW_LLM_SFT_CHECKPOINT", "")
    is_peft_checkpoint = sft_checkpoint and Path(sft_checkpoint).exists()

    # Load policy model with QLoRA
    if is_peft_checkpoint:
        logger.info(f"Loading SFT checkpoint: {sft_checkpoint}")
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
                f"SFT checkpoint not found at {sft_checkpoint}, "
                "training from base model"
            )
        logger.info(f"Loading base model: {model_config['model_name']}")
        policy_model = load_quantized_model(model_config["model_name"], model_config)
        policy_model.config.use_cache = False
        policy_model = prepare_model_for_kbit_training(
            policy_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        # Standard CAUSAL_LM LoRA
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

    # Load reference model (frozen, for KL computation)
    logger.info("Loading reference model (frozen)")
    ref_base = load_quantized_model(model_config["model_name"], model_config)
    if is_peft_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_base, sft_checkpoint)
    else:
        ref_model = ref_base
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    flow_ref = FlowLLM(ref_model, tokenizer, mask_token_id=mask_token_id)

    # Load training data
    train_file = resolve_data_path(DATA_CONFIG["train_file"])
    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )

    # Optimizer (only LoRA params)
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=grpo_config["learning_rate"]
    )

    group_size = grpo_config["group_size"]
    kl_coeff = grpo_config["kl_coeff"]
    max_seq_length = model_config["max_seq_length"]
    num_denoising_steps = grpo_config.get(
        "num_denoising_steps", model_config.get("num_denoising_steps", 20)
    )
    gen_temperature = model_config.get("generation_temperature", 0.7)
    action_timeout = grpo_config.get("action_timeout_s", 5.0)

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
        f"Starting online flow LLM GRPO: {len(prompts)} prompts, G={group_size}, "
        f"kl_coeff={kl_coeff}, denoising_steps={num_denoising_steps}"
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
                    logger.warning(
                        f"Skipping prompt {i}: missing instruction or url"
                    )
                    continue

                # Periodic browser restart to reset DOM indices
                if i > 0 and i % 10 == 0:
                    logger.info(f"Periodic browser restart (prompt {i}) to reset DOM indices")
                    await browser_env.close()
                    browser_env = await BrowserEnvironment.create(headless=headless)

                # Tokenize condition WITHOUT padding -- padding="max_length"
                # created a 512-pad + 512-mask = 1024 token input, but the model
                # was SFT-trained on 512 total (prompt + response, no padding gap).
                condition_enc = tokenizer(
                    instruction,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                ).to(flow_policy.device)

                # Response length = total budget minus prompt tokens
                prompt_len = condition_enc["attention_mask"].sum().item()
                gen_length = max(1, max_seq_length - prompt_len)

                # Generate G rollouts via iterative denoising
                rollout_texts = []
                policy_model.eval()
                for g in range(group_size):
                    generated_ids = flow_policy.generate(
                        condition_ids=condition_enc["input_ids"],
                        condition_mask=condition_enc["attention_mask"],
                        seq_length=gen_length,
                        num_steps=num_denoising_steps,
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
                    advantages, dtype=torch.float32, device=flow_policy.device
                )

                # Tokenize prompt once for reuse across rollouts
                prompt_enc = tokenizer(
                    instruction, add_special_tokens=True, return_tensors="pt"
                )
                prompt_ids = prompt_enc["input_ids"].squeeze(0)
                prompt_length = prompt_ids.shape[0]

                # Compute REINFORCE loss + Schulman k3 KL over generated rollouts
                total_pg_loss = torch.tensor(
                    0.0, device=flow_policy.device, requires_grad=False
                )
                total_kl = torch.tensor(0.0, device=flow_policy.device)

                valid_rollouts = 0
                for g in range(group_size):
                    # Tokenize the GENERATED rollout (not ground truth)
                    rollout_text = rollout_texts[g]
                    rollout_enc = tokenizer(
                        rollout_text, add_special_tokens=False, return_tensors="pt"
                    )
                    rollout_ids = rollout_enc["input_ids"].squeeze(0)

                    if rollout_ids.shape[0] == 0:
                        logger.debug(f"Empty rollout {g}, skipping loss")
                        continue

                    # Build full sequence: prompt + generated rollout
                    g_full_ids = torch.cat([prompt_ids, rollout_ids])[:max_seq_length]
                    g_full_length = g_full_ids.shape[0]

                    # Pad to max_seq_length
                    g_pad_length = max_seq_length - g_full_length
                    if g_pad_length > 0:
                        pad_id = tokenizer.pad_token_id or 0
                        g_full_ids = torch.cat([
                            g_full_ids,
                            torch.full((g_pad_length,), pad_id, dtype=torch.long),
                        ])

                    g_attn_mask = torch.zeros(max_seq_length, dtype=torch.long)
                    g_attn_mask[:g_full_length] = 1

                    # Move to device, add batch dim [1, L]
                    g_full_ids = g_full_ids.unsqueeze(0).to(flow_policy.device)
                    g_attn_mask = g_attn_mask.unsqueeze(0).to(flow_policy.device)

                    # Per-token policy log-probs (with gradients for REINFORCE)
                    policy_token_lp, resp_mask = compute_per_token_log_probs(
                        policy_model, g_full_ids, g_attn_mask, prompt_length
                    )  # [1, T], [1, T]

                    # Per-token reference log-probs (no gradients)
                    with torch.no_grad():
                        ref_token_lp, _ = compute_per_token_log_probs(
                            ref_model, g_full_ids, g_attn_mask, prompt_length
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
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                total_steps += 1
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                avg_kl = (total_kl / max(valid_rollouts, 1)).item()
                epoch_kl.append(avg_kl)

                if total_steps % grpo_config["logging_steps"] == 0:
                    pg_loss_val = (total_pg_loss / divisor).item()
                    logger.info(
                        f"  Step {total_steps} (prompt {i+1}/{len(prompts)}): "
                        f"avg_reward={avg_reward:.3f}, "
                        f"pg_loss={pg_loss_val:.4f}, "
                        f"kl={avg_kl:.4f}, "
                        f"loss={loss.item():.4f}"
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
        final_dir = Path("outputs/flow_llm_online_grpo/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Flow LLM GRPO complete. Model saved to {final_dir}")

        persist_checkpoint(str(final_dir), "online-flow-llm-grpo")

    finally:
        await browser_env.close()
        ff_server.stop()


if __name__ == "__main__":
    asyncio.run(train())
