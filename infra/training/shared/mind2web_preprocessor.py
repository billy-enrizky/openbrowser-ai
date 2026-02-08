"""Download, filter, and format Mind2Web data for SFT and Flow training."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_mind2web_raw(data_dir: Path) -> list[dict]:
    """Load raw Mind2Web dataset files."""
    tasks = []
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            tasks.extend(data)
        elif isinstance(data, dict):
            tasks.append(data)
    logger.info(f"Loaded {len(tasks)} raw Mind2Web tasks from {data_dir}")
    return tasks


def filter_tasks(tasks: list[dict], min_actions: int = 2, max_actions: int = 20) -> list[dict]:
    """Filter tasks by action count for tractable training."""
    filtered = []
    for task in tasks:
        actions = task.get("action_reprs", [])
        if isinstance(actions, list) and min_actions <= len(actions) <= max_actions:
            filtered.append(task)
    logger.info(f"Filtered to {len(filtered)} tasks ({min_actions}-{max_actions} actions)")
    return filtered


def format_for_sft(tasks: list[dict]) -> list[dict]:
    """Format Mind2Web tasks as instruction-response pairs for SFT.

    Returns list of dicts with 'instruction' and 'response' keys.
    """
    sft_data = []
    for task in tasks:
        instruction = task.get("confirmed_task", "")
        actions = task.get("action_reprs", [])
        if not instruction or not actions:
            continue

        # Build response as a step-by-step action sequence
        response_lines = []
        for i, action in enumerate(actions, 1):
            if isinstance(action, str):
                response_lines.append(f"Step {i}: {action}")
            elif isinstance(action, dict):
                response_lines.append(f"Step {i}: {action.get('action', str(action))}")
            else:
                logger.warning(
                    f"Unexpected action type {type(action).__name__} in task "
                    f"'{instruction}', skipping action at index {i}"
                )

        sft_data.append({
            "instruction": instruction,
            "response": "\n".join(response_lines),
            "website": task.get("website", ""),
            "domain": task.get("domain", ""),
            "num_actions": len(actions),
        })

    logger.info(f"Formatted {len(sft_data)} SFT examples")
    return sft_data


def format_for_flow(tasks: list[dict]) -> list[dict]:
    """Format Mind2Web tasks for discrete flow matching training.

    Returns list of dicts with 'source' (noise), 'target' (action sequence),
    and 'condition' (instruction) keys.
    """
    flow_data = []
    for task in tasks:
        instruction = task.get("confirmed_task", "")
        actions = task.get("action_reprs", [])
        if not instruction or not actions:
            continue

        # Flatten actions to string tokens
        action_tokens = []
        for action in actions:
            if isinstance(action, str):
                action_tokens.append(action)
            elif isinstance(action, dict):
                action_tokens.append(str(action.get("action", action)))
            else:
                logger.warning(
                    f"Unexpected action type {type(action).__name__} in task "
                    f"'{instruction}', skipping action"
                )

        flow_data.append({
            "condition": instruction,
            "target": action_tokens,
            "website": task.get("website", ""),
            "num_steps": len(action_tokens),
        })

    logger.info(f"Formatted {len(flow_data)} flow training examples")
    return flow_data


def save_jsonl(data: list[dict], output_path: Path):
    """Save data as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(data)} examples to {output_path}")


def preprocess(
    data_dir: str = "data/mind2web",
    output_dir: str = "data/processed",
    min_actions: int = 2,
    max_actions: int = 20,
):
    """Full preprocessing pipeline."""
    logging.basicConfig(level=logging.INFO)

    data_path = Path(data_dir)
    out_path = Path(output_dir)

    tasks = load_mind2web_raw(data_path)
    filtered = filter_tasks(tasks, min_actions, max_actions)

    sft_data = format_for_sft(filtered)
    save_jsonl(sft_data, out_path / "mind2web_sft.jsonl")

    flow_data = format_for_flow(filtered)
    save_jsonl(flow_data, out_path / "mind2web_flow.jsonl")

    logger.info("Preprocessing complete")


if __name__ == "__main__":
    preprocess()
