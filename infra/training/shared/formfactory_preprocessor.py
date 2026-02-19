"""Convert FormFactory ground truth JSON into SFT and flow training data (JSONL).

Reads the 25 ground truth JSON files from data/formfactory/data/data1/,
maps each to its Flask route using the route map from data_loader.py,
and produces:
  - instruction-response pairs for supervised fine-tuning (SFT)
  - condition-target pairs for discrete flow matching (Flow/GRPO)

Usage:
    uv run infra/training/shared/formfactory_preprocessor.py
    uv run infra/training/shared/formfactory_preprocessor.py --output data/processed/formfactory_sft.jsonl
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Import the route map from the eval data loader to stay consistent.
from infra.eval.pipelines.data_loader import (
    FORMFACTORY_ROUTE_MAP,
    FORMFACTORY_DEFAULT_PORT,
)


def build_instruction(form_name: str, route: str, fields: dict, port: int) -> str:
    """Build the instruction prompt for a form-filling task."""
    base_url = f"http://127.0.0.1:{port}"
    parts = [
        f"Go to {base_url}{route} and fill out the {form_name} form "
        "with the following information, then submit it.",
        "",
        "Field values to enter:",
    ]
    for field_name, value in fields.items():
        parts.append(f"  - {field_name}: {value}")
    return "\n".join(parts)


def build_response(form_name: str, route: str, fields: dict, port: int) -> str:
    """Build the expected step-by-step browser action response.

    Generates a sequence of navigate/click/type/submit actions that
    the agent should learn to produce.
    """
    base_url = f"http://127.0.0.1:{port}"
    steps = []
    step_num = 1

    steps.append(f"Step {step_num}: Navigate to {base_url}{route}")
    step_num += 1

    for field_name, value in fields.items():
        if isinstance(value, bool):
            if value:
                steps.append(
                    f"Step {step_num}: Click on the '{field_name}' checkbox to enable it"
                )
                step_num += 1
        elif isinstance(value, list):
            for item in value:
                steps.append(
                    f"Step {step_num}: Select '{item}' from the '{field_name}' field"
                )
                step_num += 1
        else:
            steps.append(
                f"Step {step_num}: Click on the '{field_name}' input field"
            )
            step_num += 1
            steps.append(
                f"Step {step_num}: Type '{value}' into the '{field_name}' field"
            )
            step_num += 1

    steps.append(f"Step {step_num}: Click the Submit button")

    return "\n".join(steps)


def load_and_format(
    data_dir: Path,
    port: int = FORMFACTORY_DEFAULT_PORT,
) -> list[dict]:
    """Load FormFactory ground truth files and format as SFT examples."""
    ground_truth_dir = data_dir / "data" / "data1"
    if not ground_truth_dir.exists():
        logger.error(
            f"FormFactory ground truth not found at {ground_truth_dir}. "
            "Run: uv run infra/eval/scripts/download_datasets.py --datasets formfactory"
        )
        return []

    sft_data = []
    files_processed = 0

    for json_file in sorted(ground_truth_dir.glob("*.json")):
        route_info = FORMFACTORY_ROUTE_MAP.get(json_file.name)
        if not route_info:
            logger.warning(f"No route mapping for {json_file.name}, skipping")
            continue

        with open(json_file) as f:
            records = json.load(f)

        if not isinstance(records, list):
            records = [records]

        form_name = route_info["name"]
        route = route_info["route"]
        domain = route_info["domain"]
        files_processed += 1

        base_url = f"http://127.0.0.1:{port}"
        for record in records:
            instruction = build_instruction(form_name, route, record, port)
            response = build_response(form_name, route, record, port)

            sft_data.append({
                "instruction": instruction,
                "response": response,
                "form_name": form_name,
                "domain": domain,
                "num_fields": len(record),
                "url": f"{base_url}{route}",
                "ground_truth_fields": record,
            })

    logger.info(
        f"Formatted {len(sft_data)} SFT examples from {files_processed} form types"
    )
    return sft_data


def save_jsonl(data: list[dict], output_path: Path):
    """Save data as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(data)} examples to {output_path}")


def format_for_flow(sft_data: list[dict]) -> list[dict]:
    """Convert SFT examples to flow matching format.

    Returns list of dicts with 'condition' (instruction), 'target' (action steps),
    'ground_truth_fields' (raw field values for online reward), and metadata.
    """
    flow_data = []
    for item in sft_data:
        steps = item["response"].split("\n")
        flow_data.append({
            "condition": item["instruction"],
            "target": steps,
            "url": item.get("url", ""),
            "ground_truth_fields": item.get("ground_truth_fields", {}),
            "form_name": item.get("form_name", ""),
            "num_steps": len(steps),
        })
    logger.info(f"Formatted {len(flow_data)} flow training examples")
    return flow_data


def split_dataset(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split dataset into train/val/test, stratified by form_name.

    Groups examples by form_name (25 form types, ~50 each), shuffles each
    group with a fixed seed, then splits each group 80/10/10. Guarantees
    at least 1 example per split per form type.

    Returns:
        (train, val, test) tuple.
    """
    rng = random.Random(seed)

    groups: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        groups[item.get("form_name", "unknown")].append(item)

    train, val, test = [], [], []

    for form_name in sorted(groups.keys()):
        items = groups[form_name]
        rng.shuffle(items)

        n = len(items)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * (1.0 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    logger.info(
        "Split %d examples into train=%d, val=%d, test=%d (seed=%d)",
        len(data), len(train), len(val), len(test), seed,
    )
    return train, val, test


def preprocess(
    data_dir: str = "data/formfactory",
    output_path: str = "data/processed/formfactory_sft.jsonl",
    port: int = FORMFACTORY_DEFAULT_PORT,
):
    """Full preprocessing pipeline. Produces both SFT and flow JSONL."""
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    sft_data = load_and_format(data_path, port=port)
    if not sft_data:
        logger.error("No data produced. Check that FormFactory is downloaded.")
        return

    # Save unsplit files (backward compatibility)
    save_jsonl(sft_data, out_path)

    flow_data = format_for_flow(sft_data)
    flow_out_path = out_path.parent / "formfactory_flow.jsonl"
    save_jsonl(flow_data, flow_out_path)

    # Generate stratified train/val/test splits
    sft_train, sft_val, sft_test = split_dataset(sft_data)
    save_jsonl(sft_train, out_path.parent / "formfactory_sft_train.jsonl")
    save_jsonl(sft_val, out_path.parent / "formfactory_sft_val.jsonl")
    save_jsonl(sft_test, out_path.parent / "formfactory_sft_test.jsonl")

    flow_train, flow_val, flow_test = split_dataset(flow_data)
    save_jsonl(flow_train, out_path.parent / "formfactory_flow_train.jsonl")
    save_jsonl(flow_val, out_path.parent / "formfactory_flow_val.jsonl")
    save_jsonl(flow_test, out_path.parent / "formfactory_flow_test.jsonl")

    logger.info("FormFactory preprocessing complete")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess FormFactory data for SFT training"
    )
    parser.add_argument(
        "--data-dir",
        default="data/formfactory",
        help="Path to FormFactory data directory",
    )
    parser.add_argument(
        "--output",
        default="data/processed/formfactory_sft.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=FORMFACTORY_DEFAULT_PORT,
        help="FormFactory server port (used in generated instructions)",
    )
    args = parser.parse_args()
    preprocess(data_dir=args.data_dir, output_path=args.output, port=args.port)


if __name__ == "__main__":
    main()
