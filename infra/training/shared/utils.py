"""Shared utilities for training pipelines."""

import json
import logging
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Canonical project root: infra/training/shared/utils.py -> 4 parents up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SYSTEM_PROMPT = (
    "You are a web browser automation agent. Given a task, "
    "produce a step-by-step action plan to complete it."
)


def format_chat_prompt(instruction: str) -> str:
    """Format instruction as a ChatML prompt for generation.

    Single source of truth for the prompt template used across
    SFT and GRPO trainers.
    """
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_prompt_parts(instruction: str, response: str) -> tuple[str, str]:
    """Format instruction-response pair into prompt parts.

    Returns (instruction_part, response_part) so callers can mask
    the instruction tokens in labels.
    """
    instruction_part = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    response_part = f"{response}\n<|im_end|>"
    return instruction_part, response_part


def resolve_data_path(relative_path: str) -> str:
    """Resolve a relative data path against the project root.

    If the path is already absolute, return it unchanged.
    """
    p = Path(relative_path)
    if p.is_absolute():
        return relative_path
    return str(PROJECT_ROOT / p)


ANYSCALE_STORAGE = Path("/mnt/user_storage/openbrowser")


def persist_checkpoint(local_dir: str, stage: str):
    """Copy checkpoint to Anyscale persistent storage (/mnt/user_storage).

    Falls back to a no-op when /mnt/user_storage does not exist (local dev).

    Args:
        local_dir: Local directory containing the checkpoint files.
        stage: Sub-path label (e.g. 'online-grpo', 'online-flow-grpo').
    """
    import shutil

    dest = ANYSCALE_STORAGE / "checkpoints" / stage
    if not ANYSCALE_STORAGE.parent.exists():
        logger.info("/mnt/user_storage not available (local dev), skipping persist")
        return

    try:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_dir, str(dest), dirs_exist_ok=True)
        logger.info(f"Checkpoint persisted to {dest}")
    except Exception as e:
        logger.error(f"Failed to persist checkpoint: {e}")


# Keep backward-compatible alias
def upload_checkpoint_to_s3(local_dir: str, s3_config: dict, stage: str):
    """Deprecated: now persists to /mnt/user_storage instead of S3."""
    persist_checkpoint(local_dir, stage)


def load_prompts(
    file_path: str,
    max_samples: int = 0,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Load training/eval prompts from JSONL.

    Each line must have: instruction, url, ground_truth_fields.
    Optionally: ground_truth_actions (for proxy reward).

    Args:
        file_path: Path to JSONL file.
        max_samples: Max samples to load (0 = all).
        shuffle: Whether to shuffle before truncating.
        seed: Random seed for shuffling.

    Returns:
        List of prompt dicts.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))

    logger.info("Loaded %d prompts from %s", len(prompts), file_path)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    if max_samples > 0:
        prompts = prompts[:max_samples]
        logger.info("Truncated to %d prompts", len(prompts))

    return prompts


def load_quantized_model(model_name: str, config: dict):
    """Load 4-bit quantized model for ReFusion.

    Args:
        model_name: HuggingFace model name (e.g. "GSAI-ML/ReFusion").
        config: Model config dict with bnb_4bit_* keys.

    Returns:
        The quantized model (caller loads tokenizer separately).
    """
    compute_dtype = (
        torch.bfloat16
        if config.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16"
        else torch.float16
    )
    trust_remote_code = config.get("trust_remote_code", True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
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


def compute_temporal_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute MDPO temporal advantages from per-step reward tensor.

    Implements adv-v3 (reward delta + 1) and adv-v4 (cumulative future average),
    then group-normalizes per step across G rollouts.

    Args:
        rewards: [G, T] tensor of per-step rewards (proxy for intermediate,
                 browser for final).

    Returns:
        [G, T] tensor of group-normalized temporal advantages.
    """
    G, T = rewards.shape

    # adv-v3: reward delta + 1
    # First step: use raw reward; subsequent steps: delta from previous
    deltas = torch.cat(
        [rewards[:, 0:1], rewards[:, 1:] - rewards[:, :-1]],
        dim=-1,
    )
    all_step_advantages = deltas + 1.0

    # adv-v4: add cumulative future average reward
    if T > 1:
        future_rewards = rewards[:, 1:]  # [G, T-1]
        cum_future = future_rewards.flip(-1).cumsum(-1).flip(-1)  # [G, T-1]
        divisor = torch.arange(
            T - 1, 0, -1, device=rewards.device
        ).unsqueeze(0).float()  # [1, T-1] values: T-1, T-2, ..., 1
        future_avg = cum_future / divisor  # [G, T-1]
        all_step_advantages[:, :-1] += future_avg

    # Add terminal reward to final step advantage
    all_step_advantages[:, -1:] += rewards[:, -1:]

    # Group-normalize per step (across G rollouts)
    mean = all_step_advantages.mean(dim=0, keepdim=True)  # [1, T]
    std = all_step_advantages.std(dim=0, keepdim=True)  # [1, T]
    advantages = (all_step_advantages - mean) / (std + 1e-4)

    return advantages


def select_training_steps(advantages: torch.Tensor, k: int) -> list[int]:
    """Select top-k training steps by advantage magnitude with diversity guard.

    Args:
        advantages: [G, T] tensor of temporal advantages.
        k: Number of steps to select.

    Returns:
        Sorted list of step indices to train on.
    """
    T = advantages.shape[1]
    k = min(k, T)

    if k >= T:
        return list(range(T))

    # Sum absolute advantage across rollouts to get per-step importance
    step_importance = advantages.abs().sum(dim=0)  # [T]
    _, top_indices = step_importance.topk(k)
    selected = top_indices.tolist()

    # Diversity guard: ensure at least 2 of k steps from first half
    midpoint = T // 2
    if midpoint > 0:
        first_half = [s for s in selected if s < midpoint]
        min_first_half = min(2, midpoint)

        if len(first_half) < min_first_half:
            needed = min_first_half - len(first_half)

            first_half_importances = step_importance[:midpoint]
            _, fh_sorted = first_half_importances.sort(descending=True)
            candidates = [
                idx.item() for idx in fh_sorted if idx.item() not in selected
            ]

            second_half_in_selected = [s for s in selected if s >= midpoint]
            second_half_in_selected.sort(
                key=lambda s: step_importance[s].item()
            )

            for i in range(min(needed, len(candidates), len(second_half_in_selected))):
                selected.remove(second_half_in_selected[i])
                selected.append(candidates[i])

    return sorted(selected)
