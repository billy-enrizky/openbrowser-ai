"""GRPO trainer: Group Relative Policy Optimization on FormFactory data.

Loads the SFT checkpoint (QLoRA adapter), generates rollouts per prompt,
scores them with reward functions, computes group-relative advantages,
and updates the policy with proper log-probability gradients + KL penalty.

Usage:
    SFT_CHECKPOINT_PATH=outputs/finetuning_sft/final uv run infra/training/finetuning/grpo_trainer.py
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from infra.training.finetuning.config import DATA_CONFIG, GRPO_CONFIG, S3_CONFIG
from infra.training.shared.reward_functions import compute_grpo_advantages, compute_reward
from infra.training.shared.utils import (
    format_chat_prompt,
    resolve_data_path,
    upload_checkpoint_to_s3,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompts(file_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts for GRPO rollouts."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} prompts for GRPO")
    return records


def load_quantized_model(model_name: str, config: dict):
    """Load a model with 4-bit quantization."""
    compute_dtype = (
        torch.bfloat16
        if config["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    is_prequantized = "bnb" in model_name.lower()
    load_kwargs = {"device_map": "auto", "dtype": compute_dtype}
    if not is_prequantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["load_in_4bit"],
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model


def generate_rollouts(
    model, tokenizer, prompt: str, group_size: int, max_new_tokens: int = 512
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Generate G rollouts for a single prompt.

    Returns:
        responses: list of decoded response strings
        all_input_ids: tensor of full sequences [G, seq_len]
        prompt_length: length of the prompt tokens
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=False,
        )

    all_sequences = outputs.sequences  # [G, total_len]

    responses = []
    for seq in all_sequences:
        text = tokenizer.decode(
            seq[prompt_length:], skip_special_tokens=True
        )
        responses.append(text)

    return responses, all_sequences, prompt_length


def compute_log_probs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int
) -> torch.Tensor:
    """Compute per-token log probabilities for the response portion.

    Args:
        model: the language model
        input_ids: [B, seq_len] full sequences
        attention_mask: [B, seq_len] mask (1 for real tokens, 0 for padding)
        prompt_length: number of prompt tokens to skip

    Returns:
        log_probs: [B] sum of log-probs over response tokens
    """
    with torch.set_grad_enabled(model.training):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, seq_len, vocab]

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]  # [B, seq_len-1, vocab]
    shift_labels = input_ids[:, 1:]   # [B, seq_len-1]

    # Log softmax over vocab
    log_probs_all = F.log_softmax(shift_logits, dim=-1)  # [B, seq_len-1, vocab]

    # Gather log-probs for actual tokens
    token_log_probs = log_probs_all.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [B, seq_len-1]

    # Only sum over response tokens (skip prompt)
    response_start = max(0, prompt_length - 1)  # -1 for shift
    response_log_probs = token_log_probs[:, response_start:]

    # Use attention_mask shifted to match label positions for padding mask
    mask = attention_mask[:, 1:][:, response_start:].float()
    masked_log_probs = response_log_probs * mask

    # Mean over response tokens (not sum) to keep ratio = exp(policy - ref)
    # in a numerically stable range -- sum over 500 tokens creates exp(>20) explosions
    num_tokens = mask.sum(dim=-1).clamp(min=1)
    return masked_log_probs.sum(dim=-1) / num_tokens  # [B]


def parse_actions_from_rollout(rollout: str) -> list[str]:
    """Extract action steps from a rollout response."""
    actions = []
    for line in rollout.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("step "):
            actions.append(line)
    return actions


def parse_actions_from_ground_truth(ground_truth: str) -> list[str]:
    """Extract action steps from ground truth response."""
    return parse_actions_from_rollout(ground_truth)


def check_form_submission_success(rollout: str) -> bool:
    """Check if the rollout contains expected form submission steps."""
    lower = rollout.lower()
    has_navigation = "navigate to" in lower or "go to" in lower
    has_typing = "type " in lower or "type'" in lower
    has_submit = "submit" in lower or "click" in lower
    return has_navigation and has_typing and has_submit


def train():
    """Run GRPO training loop."""
    config = GRPO_CONFIG

    # Determine model to load: SFT checkpoint or base model
    sft_checkpoint = config["sft_checkpoint"]
    if sft_checkpoint and Path(sft_checkpoint).exists():
        logger.info(f"Loading SFT checkpoint from: {sft_checkpoint}")
        model_name = sft_checkpoint
        is_peft_checkpoint = True
    else:
        if sft_checkpoint:
            logger.warning(
                f"SFT checkpoint not found at {sft_checkpoint}, "
                "falling back to base model"
            )
        model_name = config["model_name"]
        is_peft_checkpoint = False

    logger.info(f"Loading tokenizer from: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model with QLoRA
    logger.info(f"Loading policy model: {model_name}")
    if is_peft_checkpoint:
        base_model = load_quantized_model(config["model_name"], config)
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model = PeftModel.from_pretrained(base_model, model_name, is_trainable=True)
        model.train()
    else:
        model = load_quantized_model(model_name, config)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Load reference model (frozen, for KL computation)
    logger.info("Loading reference model (frozen)")
    ref_model = load_quantized_model(config["model_name"], config)
    if is_peft_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_model, model_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load training data
    train_file = resolve_data_path(DATA_CONFIG["train_file"])

    prompts = load_prompts(
        train_file,
        max_samples=DATA_CONFIG.get("max_train_samples", 0),
    )

    # Optimizer -- only LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"])

    group_size = config["group_size"]
    kl_coeff = config["kl_coeff"]
    clip_range = config["clip_range"]
    max_new_tokens = config.get("max_new_tokens", 512)

    logger.info(
        f"Starting GRPO training: {len(prompts)} prompts, G={group_size}, "
        f"kl_coeff={kl_coeff}, clip_range={clip_range}"
    )

    total_steps = 0
    for epoch in range(config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        epoch_rewards = []
        epoch_kl = []

        for i, prompt_data in enumerate(prompts):
            instruction = prompt_data.get("instruction", "")
            ground_truth = prompt_data.get("response", "")
            gt_actions = parse_actions_from_ground_truth(ground_truth)

            prompt_text = format_chat_prompt(instruction)

            # Generate G rollouts
            model.eval()
            rollouts, sequences, prompt_length = generate_rollouts(
                model, tokenizer, prompt_text, group_size,
                max_new_tokens=max_new_tokens,
            )
            model.train()

            # Pad sequences to same length for batched computation
            max_len = max(seq.shape[0] for seq in sequences)
            padded = torch.zeros(
                group_size, max_len, dtype=torch.long, device=sequences.device
            )
            attention_mask = torch.zeros(
                group_size, max_len, dtype=torch.long, device=sequences.device
            )
            for j, seq in enumerate(sequences):
                padded[j, : seq.shape[0]] = seq
                attention_mask[j, : seq.shape[0]] = 1

            # Score each rollout
            rewards = []
            for rollout in rollouts:
                pred_actions = parse_actions_from_rollout(rollout)
                success = check_form_submission_success(rollout)
                signal = compute_reward(
                    agent_output=rollout,
                    ground_truth=ground_truth,
                    success=success,
                    steps_taken=len(pred_actions),
                    predicted_actions=pred_actions,
                    ground_truth_actions=gt_actions,
                    weights=config.get("reward_weights"),
                )
                rewards.append(signal.total)

            # Compute group-relative advantages
            advantages = compute_grpo_advantages(rewards, group_size)
            advantages_t = torch.tensor(
                advantages, dtype=torch.float32, device=padded.device
            )

            # Compute policy log-probs (with gradients)
            policy_log_probs = compute_log_probs(model, padded, attention_mask, prompt_length)

            # Compute reference log-probs (no gradients)
            with torch.no_grad():
                ref_log_probs = compute_log_probs(ref_model, padded, attention_mask, prompt_length)

            # KL divergence per sample
            kl_div = policy_log_probs - ref_log_probs  # [G]

            # Policy gradient loss with PPO-style clipping
            ratio = torch.exp(policy_log_probs - ref_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            pg_loss1 = -advantages_t * ratio
            pg_loss2 = -advantages_t * clipped_ratio
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # KL penalty
            kl_penalty = kl_coeff * kl_div.mean()

            # Total loss
            loss = pg_loss + kl_penalty

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_steps += 1
            avg_reward = sum(rewards) / len(rewards)
            avg_kl = kl_div.mean().item()
            epoch_rewards.append(avg_reward)
            epoch_kl.append(avg_kl)

            if total_steps % config["logging_steps"] == 0:
                logger.info(
                    f"  Step {total_steps} (prompt {i+1}/{len(prompts)}): "
                    f"avg_reward={avg_reward:.3f}, "
                    f"loss={loss.item():.4f}, "
                    f"pg_loss={pg_loss.item():.4f}, "
                    f"kl={avg_kl:.4f}"
                )

            if config["save_steps"] > 0 and total_steps % config["save_steps"] == 0:
                ckpt_dir = f"outputs/finetuning_grpo/checkpoint-{total_steps}"
                model.save_pretrained(ckpt_dir)
                logger.info(f"Saved checkpoint to {ckpt_dir}")

        if epoch_rewards:
            logger.info(
                f"Epoch {epoch + 1} summary: "
                f"avg_reward={sum(epoch_rewards)/len(epoch_rewards):.3f}, "
                f"avg_kl={sum(epoch_kl)/len(epoch_kl):.4f}"
            )

    # Save final model
    final_dir = "outputs/finetuning_grpo/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"GRPO training complete. Model saved to {final_dir}")

    # Upload checkpoint to S3
    upload_checkpoint_to_s3(final_dir, S3_CONFIG, "grpo")


if __name__ == "__main__":
    train()
