"""Unified data loaders for evaluation datasets."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Base path for local datasets
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_stress_tests(max_tasks: int = 0) -> list[dict]:
    """Load stress test tasks from InteractionTasks_v8.json.

    Each task has: task_id, name, confirmed_task, category, ground_truth
    """
    path = PROJECT_ROOT / "stress-tests" / "InteractionTasks_v8.json"
    if not path.exists():
        logger.error(f"Stress tests not found at {path}")
        return []

    with open(path) as f:
        tasks = json.load(f)

    logger.info(f"Loaded {len(tasks)} stress test tasks")

    # Normalize to common format
    normalized = []
    for t in tasks:
        normalized.append({
            "task_id": t.get("task_id", ""),
            "name": t.get("name", ""),
            "instruction": t.get("confirmed_task", ""),
            "category": t.get("category", ""),
            "ground_truth": t.get("ground_truth", ""),
            "dataset": "stress_tests",
        })

    if max_tasks > 0:
        normalized = normalized[:max_tasks]
        logger.info(f"Limited to {max_tasks} tasks")

    return normalized


def load_mind2web(max_tasks: int = 0, data_dir: str | None = None) -> list[dict]:
    """Load Mind2Web tasks.

    Expects JSON files in data_dir or PROJECT_ROOT/data/mind2web/
    """
    if data_dir:
        base = Path(data_dir)
    else:
        base = PROJECT_ROOT / "data" / "mind2web"

    if not base.exists():
        logger.warning(f"Mind2Web data not found at {base}")
        return []

    tasks = []
    for json_file in sorted(base.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                tasks.append({
                    "task_id": item.get("annotation_id", item.get("task_id", "")),
                    "name": item.get("confirmed_task", item.get("name", json_file.stem)),
                    "instruction": item.get("confirmed_task", ""),
                    "category": item.get("website", item.get("domain", "")),
                    "ground_truth": item.get("action_reprs", ""),
                    "dataset": "mind2web",
                })
        elif isinstance(data, dict):
            tasks.append({
                "task_id": data.get("annotation_id", json_file.stem),
                "name": data.get("confirmed_task", json_file.stem),
                "instruction": data.get("confirmed_task", ""),
                "category": data.get("website", ""),
                "ground_truth": data.get("action_reprs", ""),
                "dataset": "mind2web",
            })

    logger.info(f"Loaded {len(tasks)} Mind2Web tasks")
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks


def load_formfactory(max_tasks: int = 0, data_dir: str | None = None) -> list[dict]:
    """Load FormFactory tasks."""
    if data_dir:
        base = Path(data_dir)
    else:
        base = PROJECT_ROOT / "data" / "formfactory"

    if not base.exists():
        logger.warning(f"FormFactory data not found at {base}")
        return []

    tasks = []
    for json_file in sorted(base.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                tasks.append({
                    "task_id": item.get("task_id", ""),
                    "name": item.get("name", json_file.stem),
                    "instruction": item.get("instruction", item.get("task", "")),
                    "category": "formfactory",
                    "ground_truth": item.get("ground_truth", ""),
                    "dataset": "formfactory",
                })

    logger.info(f"Loaded {len(tasks)} FormFactory tasks")
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks


def load_webarena(max_tasks: int = 0, data_dir: str | None = None) -> list[dict]:
    """Load WebArena tasks."""
    if data_dir:
        base = Path(data_dir)
    else:
        base = PROJECT_ROOT / "data" / "webarena"

    if not base.exists():
        logger.warning(f"WebArena data not found at {base}")
        return []

    tasks = []
    for json_file in sorted(base.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                tasks.append({
                    "task_id": str(item.get("task_id", "")),
                    "name": item.get("intent", json_file.stem),
                    "instruction": item.get("intent", ""),
                    "category": item.get("sites", ["webarena"])[0] if isinstance(item.get("sites"), list) else "webarena",
                    "ground_truth": item.get("eval", {}).get("reference_answers", ""),
                    "dataset": "webarena",
                })

    logger.info(f"Loaded {len(tasks)} WebArena tasks")
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks


def load_dataset(name: str, max_tasks: int = 0, data_dir: str | None = None) -> list[dict]:
    """Load a dataset by name."""
    loaders = {
        "stress_tests": load_stress_tests,
        "mind2web": load_mind2web,
        "formfactory": load_formfactory,
        "webarena": load_webarena,
    }

    loader = loaders.get(name)
    if not loader:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    if name == "stress_tests":
        return loader(max_tasks=max_tasks)
    return loader(max_tasks=max_tasks, data_dir=data_dir)
