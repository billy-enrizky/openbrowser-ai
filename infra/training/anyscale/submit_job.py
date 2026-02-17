"""Submit training jobs to Anyscale Ray.

Secrets (HF_TOKEN, etc.) are loaded from .env and injected via --env flags
so they never appear in tracked YAML files.

Usage:
    uv run infra/training/anyscale/submit_job.py finetuning-sft
    uv run infra/training/anyscale/submit_job.py finetuning-grpo
    uv run infra/training/anyscale/submit_job.py flow-matching
    uv run infra/training/anyscale/submit_job.py flow-llm-sft
    uv run infra/training/anyscale/submit_job.py online-flow-grpo
    uv run infra/training/anyscale/submit_job.py online-flow-llm-grpo
    uv run infra/training/anyscale/submit_job.py online-grpo
    uv run infra/training/anyscale/submit_job.py fsdfm-sft
    uv run infra/training/anyscale/submit_job.py online-fsdfm-grpo
    uv run infra/training/anyscale/submit_job.py fsdfm-flow-grpo
    uv run infra/training/anyscale/submit_job.py eval-sft
    uv run infra/training/anyscale/submit_job.py eval-grpo
    uv run infra/training/anyscale/submit_job.py eval-fsdfm-sft
    uv run infra/training/anyscale/submit_job.py eval-refusion-sft
    uv run infra/training/anyscale/submit_job.py eval-fsdfm-grpo
    uv run infra/training/anyscale/submit_job.py eval-refusion-grpo
    uv run infra/training/anyscale/submit_job.py eval-fsdfm-flow-grpo
    uv run infra/training/anyscale/submit_job.py --list
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

JOBS_DIR = Path(__file__).parent
PROJECT_ROOT = JOBS_DIR.parents[2]

JOB_CONFIGS = {
    "finetuning-sft": JOBS_DIR / "finetuning_sft_job.yaml",
    "finetuning-grpo": JOBS_DIR / "finetuning_grpo_job.yaml",
    "flow-matching": JOBS_DIR / "flow_matching_job.yaml",
    "flow-llm-sft": JOBS_DIR / "flow_llm_sft_job.yaml",
    "online-flow-grpo": JOBS_DIR / "online_flow_grpo_job.yaml",
    "online-flow-llm-grpo": JOBS_DIR / "online_flow_llm_grpo_job.yaml",
    "online-grpo": JOBS_DIR / "online_grpo_job.yaml",
    "fsdfm-sft": JOBS_DIR / "fsdfm_sft_job.yaml",
    "online-fsdfm-grpo": JOBS_DIR / "online_fsdfm_grpo_job.yaml",
    "fsdfm-flow-grpo": JOBS_DIR / "fsdfm_flow_grpo_job.yaml",
    "eval-sft": JOBS_DIR / "eval_sft_job.yaml",
    "eval-grpo": JOBS_DIR / "eval_grpo_job.yaml",
    "eval-fsdfm-sft": JOBS_DIR / "eval_fsdfm_sft_job.yaml",
    "eval-refusion-sft": JOBS_DIR / "eval_refusion_sft_job.yaml",
    "eval-fsdfm-grpo": JOBS_DIR / "eval_fsdfm_grpo_job.yaml",
    "eval-refusion-grpo": JOBS_DIR / "eval_refusion_grpo_job.yaml",
    "eval-fsdfm-flow-grpo": JOBS_DIR / "eval_fsdfm_flow_grpo_job.yaml",
}

# Secret env vars to inject from .env into Anyscale jobs
SECRET_ENV_KEYS = [
    "HF_TOKEN",
]


def _load_dotenv() -> dict[str, str]:
    """Load key=value pairs from .env file (no dependency on python-dotenv)."""
    env_file = PROJECT_ROOT / ".env"
    env_vars: dict[str, str] = {}
    if not env_file.exists():
        logger.warning(f".env file not found at {env_file}")
        return env_vars
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        env_vars[key] = value
    return env_vars


def _get_secret_env_flags() -> list[str]:
    """Build --env KEY=VALUE flags for secrets from .env or os.environ."""
    dotenv = _load_dotenv()
    flags: list[str] = []
    for key in SECRET_ENV_KEYS:
        value = os.environ.get(key) or dotenv.get(key)
        if value:
            flags.extend(["--env", f"{key}={value}"])
            logger.info(f"Injecting secret env var: {key}=***")
        else:
            logger.warning(f"Secret {key} not found in .env or environment")
    return flags


def submit_job(job_name: str, wait: bool = False):
    """Submit a job to Anyscale."""
    config_path = JOB_CONFIGS.get(job_name)
    if not config_path:
        logger.error(f"Unknown job: {job_name}. Available: {list(JOB_CONFIGS.keys())}")
        sys.exit(1)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    cmd = ["anyscale", "job", "submit", "--config-file", str(config_path)]
    cmd.extend(_get_secret_env_flags())
    if wait:
        cmd.append("--wait")

    logger.info(f"Submitting job: {job_name}")
    logger.info(f"Config: {config_path}")
    # Log command without secret values
    safe_cmd = []
    skip_next = False
    for part in cmd:
        if skip_next:
            safe_cmd.append(part.split("=")[0] + "=***")
            skip_next = False
        elif part == "--env":
            safe_cmd.append(part)
            skip_next = True
        else:
            safe_cmd.append(part)
    logger.info(f"Command: {' '.join(safe_cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error(f"Job submission failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    logger.info(f"Job {job_name} submitted successfully")


def list_jobs():
    """List available job configs."""
    logger.info("Available job configs:")
    for name, path in JOB_CONFIGS.items():
        exists = "OK" if path.exists() else "MISSING"
        logger.info(f"  {name}: {path} [{exists}]")


def main():
    parser = argparse.ArgumentParser(description="Submit Anyscale training jobs")
    parser.add_argument("job", nargs="?", help="Job name to submit")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    parser.add_argument("--list", action="store_true", help="List available jobs")
    args = parser.parse_args()

    if args.list or not args.job:
        list_jobs()
        return

    submit_job(args.job, wait=args.wait)


if __name__ == "__main__":
    main()
