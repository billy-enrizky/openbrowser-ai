"""Shared utilities for training pipelines."""

import logging
from pathlib import Path

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
