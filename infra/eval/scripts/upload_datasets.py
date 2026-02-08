"""Upload local datasets to S3 evaluation datasets bucket."""

import argparse
import logging
import os
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATASET_PATHS = {
    "stress-tests": PROJECT_ROOT / "stress-tests",
    "mind2web": PROJECT_ROOT / "data" / "mind2web",
    "formfactory": PROJECT_ROOT / "data" / "formfactory",
    "webarena": PROJECT_ROOT / "data" / "webarena",
}


def upload_directory(s3_client, local_dir: Path, bucket: str, prefix: str):
    """Upload a local directory to S3."""
    if not local_dir.exists():
        logger.warning(f"Directory not found: {local_dir}")
        return 0

    count = 0
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            s3_key = f"{prefix}/{file_path.relative_to(local_dir)}"
            logger.info(f"Uploading {file_path} -> s3://{bucket}/{s3_key}")
            s3_client.upload_file(str(file_path), bucket, s3_key)
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_PATHS.keys()),
        help="Datasets to upload",
    )
    parser.add_argument("--region", default="ca-central-1", help="AWS region")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)

    total = 0
    for dataset in args.datasets:
        local_dir = DATASET_PATHS.get(dataset)
        if not local_dir:
            logger.warning(f"Unknown dataset: {dataset}")
            continue

        logger.info(f"Uploading {dataset} from {local_dir}")
        count = upload_directory(s3, local_dir, args.bucket, dataset)
        total += count
        logger.info(f"Uploaded {count} files for {dataset}")

    logger.info(f"Total uploaded: {total} files")


if __name__ == "__main__":
    main()
