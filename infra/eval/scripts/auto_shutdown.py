"""Auto-shutdown script for eval EC2: stops instance if idle for 30+ minutes."""

import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IDLE_FILE = Path("/tmp/eval_idle_since")
IDLE_THRESHOLD_SECONDS = 1800  # 30 minutes


def is_eval_running() -> bool:
    """Check if any eval benchmark process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "eval_benchmark"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return True  # assume running on error


def get_idle_seconds() -> float:
    """Get seconds since last activity."""
    if not IDLE_FILE.exists():
        IDLE_FILE.write_text(str(time.time()))
        return 0.0
    try:
        last_active = float(IDLE_FILE.read_text().strip())
        return time.time() - last_active
    except (ValueError, OSError):
        IDLE_FILE.write_text(str(time.time()))
        return 0.0


def stop_instance():
    """Stop this EC2 instance via IMDSv2."""
    try:
        token = subprocess.run(
            ["curl", "-s", "-X", "PUT",
             "http://169.254.169.254/latest/api/token",
             "-H", "X-aws-ec2-metadata-token-ttl-seconds: 60"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()

        instance_id = subprocess.run(
            ["curl", "-s",
             "http://169.254.169.254/latest/meta-data/instance-id",
             "-H", f"X-aws-ec2-metadata-token: {token}"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()

        region = subprocess.run(
            ["curl", "-s",
             "http://169.254.169.254/latest/meta-data/placement/region",
             "-H", f"X-aws-ec2-metadata-token: {token}"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()

        logger.info(f"Stopping instance {instance_id} in {region}")
        subprocess.run(
            ["aws", "ec2", "stop-instances",
             "--instance-ids", instance_id,
             "--region", region],
            timeout=30,
        )
    except Exception as e:
        logger.error(f"Failed to stop instance: {e}")


def main():
    if is_eval_running():
        logger.info("Eval process running, resetting idle timer")
        IDLE_FILE.write_text(str(time.time()))
        return

    idle = get_idle_seconds()
    logger.info(f"No eval process. Idle for {idle:.0f}s / {IDLE_THRESHOLD_SECONDS}s")

    if idle >= IDLE_THRESHOLD_SECONDS:
        logger.info("Idle threshold reached. Stopping instance.")
        stop_instance()


if __name__ == "__main__":
    main()
