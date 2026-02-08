"""Submit training jobs to Anyscale Ray.

Usage:
    uv run infra/training/anyscale/submit_job.py finetuning-sft
    uv run infra/training/anyscale/submit_job.py finetuning-grpo
    uv run infra/training/anyscale/submit_job.py flow-matching
    uv run infra/training/anyscale/submit_job.py online-flow-grpo
    uv run infra/training/anyscale/submit_job.py online-grpo
    uv run infra/training/anyscale/submit_job.py --list
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

JOBS_DIR = Path(__file__).parent

JOB_CONFIGS = {
    "finetuning-sft": JOBS_DIR / "finetuning_sft_job.yaml",
    "finetuning-grpo": JOBS_DIR / "finetuning_grpo_job.yaml",
    "flow-matching": JOBS_DIR / "flow_matching_job.yaml",
    "online-flow-grpo": JOBS_DIR / "online_flow_grpo_job.yaml",
    "online-grpo": JOBS_DIR / "online_grpo_job.yaml",
}


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
    if wait:
        cmd.append("--wait")

    logger.info(f"Submitting job: {job_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Command: {' '.join(cmd)}")

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
