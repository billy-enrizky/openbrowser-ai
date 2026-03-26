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
        Tuple of (model, tokenizer).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    return model, tokenizer
