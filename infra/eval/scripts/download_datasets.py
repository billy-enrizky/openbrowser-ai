"""Download evaluation datasets from their official sources.

Supported datasets:
  - mind2web:     HuggingFace osunlp/Mind2Web (train split)
  - webarena:     GitHub web-arena-x/webarena test tasks
  - formfactory:  GitHub formfactory-ai/formfactory (clone repo)
  - stress_tests: Already included in the repository (no download needed)

Usage:
  uv run infra/eval/scripts/download_datasets.py --datasets mind2web webarena formfactory
  uv run infra/eval/scripts/download_datasets.py --datasets mind2web --max-samples 50
  uv run infra/eval/scripts/download_datasets.py --all
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATASET_DIRS = {
    "mind2web": PROJECT_ROOT / "data" / "mind2web",
    "webarena": PROJECT_ROOT / "data" / "webarena",
    "formfactory": PROJECT_ROOT / "data" / "formfactory",
}

WEBARENA_RAW_URL = (
    "https://raw.githubusercontent.com/web-arena-x/webarena"
    "/main/config_files/test.raw.json"
)

# Docker Hub images (preferred -- compressed layer pulls, ~87 GB total)
WEBARENA_DOCKER_HUB_IMAGES = {
    "shopping_final_0712": "webarenaimages/shopping_final_0712:latest",
    "shopping_admin_final_0719": "webarenaimages/shopping_admin_final_0719:latest",
    "postmill-populated-exposed-withimg": "webarenaimages/postmill-populated-exposed-withimg:latest",
}

# CMU tar mirror (fallback -- uncompressed, ~122 GB total)
WEBARENA_TAR_URLS = {
    "shopping_final_0712": (
        "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar"
    ),
    "shopping_admin_final_0719": (
        "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar"
    ),
    "postmill-populated-exposed-withimg": (
        "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar"
    ),
}

FORMFACTORY_REPO_URL = "https://github.com/formfactory-ai/formfactory.git"


def download_mind2web(output_dir: Path, max_samples: int = 0) -> int:
    """Download Mind2Web dataset from HuggingFace.

    Uses the `datasets` library to fetch the train split from osunlp/Mind2Web.
    Saves each split as a separate JSON file.

    Args:
        output_dir: Directory to save downloaded files.
        max_samples: Maximum number of samples to download (0 = all).

    Returns:
        Number of tasks saved.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required for Mind2Web download. "
            "Install it with: uv pip install datasets"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.json"
    if train_path.exists():
        with open(train_path) as f:
            existing = json.load(f)
        logger.info(
            f"Mind2Web train.json already exists with {len(existing)} tasks, skipping. "
            "Delete the file to re-download."
        )
        return len(existing)

    logger.info("Downloading Mind2Web from HuggingFace (osunlp/Mind2Web)...")
    logger.info("This may take a few minutes on first download.")

    dataset = load_dataset("osunlp/Mind2Web", split="train")

    tasks = []
    limit = max_samples if max_samples > 0 else len(dataset)
    for i in range(min(limit, len(dataset))):
        row = dataset[i]
        tasks.append({
            "annotation_id": row.get("annotation_id", ""),
            "website": row.get("website", ""),
            "domain": row.get("domain", ""),
            "subdomain": row.get("subdomain", ""),
            "confirmed_task": row.get("confirmed_task", ""),
            "action_reprs": row.get("action_reprs", []),
        })

    with open(train_path, "w") as f:
        json.dump(tasks, f, indent=2)

    logger.info(f"Saved {len(tasks)} Mind2Web tasks to {train_path}")
    return len(tasks)


def _download_large_file(url: str, output_path: Path) -> bool:
    """Download a large file with streaming and progress logging.

    Args:
        url: URL to download from.
        output_path: Local path to save the file.

    Returns:
        True if download succeeded.
    """
    try:
        req = Request(url, headers={"User-Agent": "openbrowser-ai/1.0"})
        with urlopen(req, timeout=600) as response:
            total = int(response.headers.get("Content-Length", 0))
            total_mb = total / (1024 * 1024) if total else 0

            output_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            last_logged = 0
            chunk_size = 8 * 1024 * 1024  # 8 MB chunks

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    downloaded_mb = downloaded / (1024 * 1024)

                    # Log progress every 100 MB
                    if downloaded_mb - last_logged >= 100:
                        if total_mb:
                            logger.info(
                                f"  {downloaded_mb:.0f}/{total_mb:.0f} MB "
                                f"({downloaded / total * 100:.0f}%)"
                            )
                        else:
                            logger.info(f"  {downloaded_mb:.0f} MB downloaded")
                        last_logged = downloaded_mb

        logger.info(f"  Download complete: {downloaded / (1024 * 1024):.0f} MB")
        return True

    except (URLError, OSError) as e:
        logger.error(f"  Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def _docker_load_image(tar_path: Path) -> bool:
    """Load a Docker image from a tar file.

    Args:
        tar_path: Path to the .tar file.

    Returns:
        True if load succeeded.
    """
    try:
        result = subprocess.run(
            ["docker", "load", "--input", str(tar_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"  Loaded Docker image from {tar_path.name}")
            return True
        logger.error(f"  docker load failed: {result.stderr}")
        return False
    except FileNotFoundError:
        logger.error("  Docker is not installed or not in PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("  docker load timed out")
        return False


def _docker_pull_image(hub_image: str) -> bool:
    """Pull a Docker image from Docker Hub.

    Args:
        hub_image: Full Docker Hub image reference (e.g. webarenaimages/shopping:latest).

    Returns:
        True if pull succeeded.
    """
    try:
        logger.info(f"  Pulling {hub_image} from Docker Hub...")
        result = subprocess.run(
            ["docker", "pull", hub_image],
            capture_output=False,
            timeout=7200,
        )
        if result.returncode == 0:
            logger.info(f"  Successfully pulled {hub_image}")
            return True
        logger.error(f"  docker pull failed for {hub_image}")
        return False
    except FileNotFoundError:
        logger.error("  Docker is not installed or not in PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"  docker pull timed out for {hub_image}")
        return False


def _docker_image_exists(image_name: str) -> bool:
    """Check if a Docker image is already loaded."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_webarena(
    output_dir: Path, max_samples: int = 0, skip_docker: bool = False
) -> int:
    """Download WebArena test tasks and optionally Docker images.

    Downloads:
    1. test.raw.json task definitions from GitHub
    2. Docker tar images for shopping, shopping_admin, and reddit from CMU mirrors
       (skipped if skip_docker=True)

    Args:
        output_dir: Directory to save downloaded files.
        max_samples: Maximum number of task samples to keep (0 = all).
        skip_docker: If True, only download task JSON, skip Docker images.

    Returns:
        Number of tasks saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download task definitions ---
    test_path = output_dir / "test.raw.json"
    task_count = 0

    if test_path.exists():
        with open(test_path) as f:
            existing = json.load(f)
        task_count = len(existing)
        logger.info(
            f"WebArena test.raw.json already exists with {task_count} tasks, skipping. "
            "Delete the file to re-download."
        )
    else:
        logger.info(f"Downloading WebArena test tasks from {WEBARENA_RAW_URL}...")
        try:
            req = Request(WEBARENA_RAW_URL, headers={"User-Agent": "openbrowser-ai/1.0"})
            with urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except URLError as e:
            logger.error(f"Failed to download WebArena tasks: {e}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebArena JSON: {e}")
            return 0

        if not isinstance(data, list):
            logger.error("Unexpected WebArena data format (expected JSON array)")
            return 0

        if max_samples > 0:
            data = data[:max_samples]

        with open(test_path, "w") as f:
            json.dump(data, f, indent=2)

        task_count = len(data)
        logger.info(f"Saved {task_count} WebArena tasks to {test_path}")

    # --- Step 2: Download Docker images ---
    if skip_docker:
        logger.info("Skipping Docker image downloads (--skip-docker)")
    else:
        for image_name, hub_image in WEBARENA_DOCKER_HUB_IMAGES.items():
            # Check if image is already loaded (check both local name and hub name)
            if _docker_image_exists(hub_image) or _docker_image_exists(image_name):
                logger.info(f"Docker image '{image_name}' already loaded, skipping")
                continue

            logger.info(f"Downloading {image_name}...")

            # Prefer Docker Hub pull (compressed layers, faster)
            if _docker_pull_image(hub_image):
                continue

            # Fallback: download tar from CMU mirror
            tar_url = WEBARENA_TAR_URLS.get(image_name)
            if not tar_url:
                logger.error(f"  No fallback tar URL for {image_name}")
                continue

            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            tar_filename = tar_url.rsplit("/", 1)[-1]
            tar_path = images_dir / tar_filename

            if tar_path.exists():
                logger.info(f"  Tar file {tar_filename} already downloaded")
            else:
                logger.info(f"  Falling back to CMU tar: {tar_url}")
                logger.info("  This is a large file (several GB), please be patient.")
                if not _download_large_file(tar_url, tar_path):
                    continue

            logger.info(f"  Loading {tar_filename} into Docker...")
            _docker_load_image(tar_path)

    return task_count


def check_stress_tests() -> int:
    """Check that stress tests are available in the repository.

    Returns:
        Number of tasks found.
    """
    path = PROJECT_ROOT / "stress-tests" / "InteractionTasks_v8.json"
    if not path.exists():
        logger.warning(f"Stress tests not found at {path}")
        return 0

    with open(path) as f:
        tasks = json.load(f)

    logger.info(f"Stress tests: {len(tasks)} tasks found at {path} (no download needed)")
    return len(tasks)


def download_formfactory(output_dir: Path, max_samples: int = 0) -> int:
    """Download FormFactory by cloning its GitHub repository.

    FormFactory is a Flask app with 25 HTML forms and ground truth JSON.
    The repo is cloned to data/formfactory/ and serves as both the data
    source and the Flask server for evaluation.

    Args:
        output_dir: Directory to clone the repo into.
        max_samples: Not used (FormFactory has a fixed set of forms).

    Returns:
        Number of ground truth files found.
    """
    app_py = output_dir / "app.py"
    if app_py.exists():
        # Count ground truth files
        data_dir = output_dir / "data" / "data1"
        count = len(list(data_dir.glob("*.json"))) if data_dir.exists() else 0
        logger.info(
            f"FormFactory already cloned at {output_dir} with {count} ground truth files, "
            "skipping. Delete the directory to re-download."
        )
        return count

    logger.info(f"Cloning FormFactory from {FORMFACTORY_REPO_URL}...")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", FORMFACTORY_REPO_URL, str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone FormFactory: {e.stderr}")
        return 0
    except FileNotFoundError:
        logger.error("git is not installed. Install git to download FormFactory.")
        return 0

    data_dir = output_dir / "data" / "data1"
    count = len(list(data_dir.glob("*.json"))) if data_dir.exists() else 0
    logger.info(f"Cloned FormFactory with {count} ground truth files to {output_dir}")
    return count


DOWNLOADERS = {
    "mind2web": download_mind2web,
    "webarena": download_webarena,
    "formfactory": download_formfactory,
}

CHECKERS = {
    "stress_tests": check_stress_tests,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets from official sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run infra/eval/scripts/download_datasets.py --datasets mind2web webarena\n"
            "  uv run infra/eval/scripts/download_datasets.py --all\n"
            "  uv run infra/eval/scripts/download_datasets.py --datasets mind2web --max-samples 100\n"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DOWNLOADERS.keys()) + list(CHECKERS.keys()),
        help="Datasets to download or check",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets and check local ones",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples per dataset (0 = all, default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker image downloads (WebArena). Only download task JSON.",
    )
    args = parser.parse_args()

    if not args.datasets and not args.all:
        parser.print_help()
        return

    if args.all:
        selected = list(DOWNLOADERS.keys()) + list(CHECKERS.keys())
    else:
        selected = args.datasets

    summary = {}

    for name in selected:
        logger.info(f"--- {name} ---")

        if name in DOWNLOADERS:
            output_dir = DATASET_DIRS[name]

            if args.force:
                for json_file in output_dir.glob("*.json"):
                    json_file.unlink()
                    logger.info(f"Removed {json_file}")

            kwargs = {"max_samples": args.max_samples}
            if name == "webarena":
                kwargs["skip_docker"] = args.skip_docker
            count = DOWNLOADERS[name](output_dir, **kwargs)
            summary[name] = count

        elif name in CHECKERS:
            count = CHECKERS[name]()
            summary[name] = count

    logger.info("--- Summary ---")
    for name, count in summary.items():
        status = "OK" if count > 0 else "MISSING"
        logger.info(f"  {name}: {count} tasks [{status}]")


if __name__ == "__main__":
    main()
