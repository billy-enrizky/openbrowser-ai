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


def upload_checkpoint_to_s3(local_dir: str, s3_config: dict, stage: str):
    """Upload a checkpoint directory to S3.

    Args:
        local_dir: Local directory containing the checkpoint files.
        s3_config: Dict with keys 'checkpoint_bucket', 'checkpoint_prefix', 'region'.
        stage: Sub-path label (e.g. 'sft', 'grpo') appended to the S3 prefix.
    """
    bucket = s3_config["checkpoint_bucket"]
    prefix = s3_config["checkpoint_prefix"]

    if not bucket:
        logger.info("No S3 bucket configured, skipping upload")
        return

    try:
        import boto3

        s3 = boto3.client("s3", region_name=s3_config["region"])
        local_path = Path(local_dir)

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                s3_key = f"{prefix}/{stage}/{file_path.relative_to(local_path)}"
                logger.info(f"Uploading {file_path} -> s3://{bucket}/{s3_key}")
                s3.upload_file(str(file_path), bucket, s3_key)

        logger.info(f"Checkpoint uploaded to s3://{bucket}/{prefix}/{stage}/")
    except Exception as e:
        logger.error(f"Failed to upload checkpoint to S3: {e}")
