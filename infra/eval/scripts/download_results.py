"""Download evaluation results from S3."""

import argparse
import logging
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_prefix(s3_client, bucket: str, prefix: str, local_dir: Path):
    """Download all objects under an S3 prefix to a local directory."""
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):].lstrip("/")
            if not relative:
                continue

            local_path = local_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading s3://{bucket}/{key} -> {local_path}")
            s3_client.download_file(bucket, key, str(local_path))
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Download results from S3")
    parser.add_argument("--bucket", required=True, help="S3 results bucket name")
    parser.add_argument("--prefix", default="", help="S3 prefix filter (e.g., benchmarking/)")
    parser.add_argument("--output-dir", default="results", help="Local output directory")
    parser.add_argument("--region", default="ca-central-1", help="AWS region")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    local_dir = Path(args.output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    count = download_prefix(s3, args.bucket, args.prefix, local_dir)
    logger.info(f"Downloaded {count} files to {local_dir}")


if __name__ == "__main__":
    main()
