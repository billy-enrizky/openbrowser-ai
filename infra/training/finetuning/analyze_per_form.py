"""Per-form analysis of evaluation results.

Compares GRPO vs SFT per-prompt rewards broken down by form type and domain
category. Reads eval log files (grep output from Anyscale job logs) and the
corresponding data splits to map prompt indices to form types.

Usage:
    python infra/training/finetuning/analyze_per_form.py \
        --grpo-log /tmp/grpo_val_results.txt \
        --sft-log /tmp/sft_val_results.txt \
        --data data/processed/formfactory_sft_val.jsonl

Log files should contain lines matching:
    Prompt N/M: reward=X.XXX
"""

import argparse
import json
import logging
import re
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_eval_log(filepath: str) -> dict[int, float]:
    """Parse per-prompt rewards from eval log grep output."""
    rewards = {}
    with open(filepath) as f:
        for line in f:
            m = re.search(r"Prompt (\d+)/\d+: reward=([\d.]+)", line)
            if m:
                idx = int(m.group(1)) - 1  # Convert to 0-indexed
                reward = float(m.group(2))
                rewards[idx] = reward
    return rewards


def load_form_mapping(filepath: str) -> list[tuple[str, str]]:
    """Load form type mappings from data JSONL."""
    forms = []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line)
            url = d.get("url", "")
            parts = url.rstrip("/").split("/")
            category = parts[-2] if len(parts) >= 2 else "unknown"
            form_name = parts[-1] if len(parts) >= 1 else "unknown"
            forms.append((category, form_name))
    return forms


def analyze(
    grpo_rewards: dict[int, float],
    sft_rewards: dict[int, float],
    forms: list[tuple[str, str]],
    split_name: str,
) -> None:
    """Run per-form and per-category analysis."""
    n = len(forms)
    logger.info("=" * 80)
    logger.info("%s SPLIT (n=%d)", split_name, n)
    logger.info("=" * 80)

    # Per-form breakdown
    form_data: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"grpo": [], "sft": []}
    )
    for idx, (category, form_name) in enumerate(forms):
        key = f"{category}/{form_name}"
        if idx in grpo_rewards:
            form_data[key]["grpo"].append(grpo_rewards[idx])
        if idx in sft_rewards:
            form_data[key]["sft"].append(sft_rewards[idx])

    logger.info(
        "%-50s %3s %10s %10s %10s %10s",
        "Form", "n", "GRPO", "SFT", "Delta", "Winner",
    )
    logger.info("-" * 93)

    grpo_wins = sft_wins = ties = 0
    for form_key in sorted(form_data.keys()):
        data = form_data[form_key]
        cnt = len(data["grpo"])
        g_avg = sum(data["grpo"]) / cnt if data["grpo"] else 0
        s_avg = sum(data["sft"]) / cnt if data["sft"] else 0
        delta = g_avg - s_avg

        if abs(delta) < 0.001:
            winner = "TIE"
            ties += 1
        elif delta > 0:
            winner = "GRPO"
            grpo_wins += 1
        else:
            winner = "SFT"
            sft_wins += 1

        logger.info(
            "%-50s %3d %10.4f %10.4f %+10.4f %10s",
            form_key, cnt, g_avg, s_avg, delta, winner,
        )

    logger.info("")
    logger.info(
        "Form-level: GRPO wins=%d, SFT wins=%d, Ties=%d",
        grpo_wins, sft_wins, ties,
    )

    # Prompt-level W/L/T
    p_grpo = p_sft = p_tie = 0
    for idx in range(n):
        g = grpo_rewards.get(idx, 0)
        s = sft_rewards.get(idx, 0)
        diff = g - s
        if abs(diff) < 0.001:
            p_tie += 1
        elif diff > 0:
            p_grpo += 1
        else:
            p_sft += 1
    logger.info(
        "Prompt-level: GRPO wins=%d, SFT wins=%d, Ties=%d",
        p_grpo, p_sft, p_tie,
    )

    # Per-category
    cat_data: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"grpo": [], "sft": []}
    )
    for idx, (category, _) in enumerate(forms):
        if idx in grpo_rewards:
            cat_data[category]["grpo"].append(grpo_rewards[idx])
        if idx in sft_rewards:
            cat_data[category]["sft"].append(sft_rewards[idx])

    logger.info("")
    logger.info("Per-Category:")
    logger.info(
        "  %-35s %3s %10s %10s %10s", "Category", "n", "GRPO", "SFT", "Delta"
    )
    logger.info("  " + "-" * 73)
    for cat in sorted(cat_data.keys()):
        cnt = len(cat_data[cat]["grpo"])
        g_avg = sum(cat_data[cat]["grpo"]) / cnt
        s_avg = sum(cat_data[cat]["sft"]) / cnt
        logger.info(
            "  %-35s %3d %10.4f %10.4f %+10.4f",
            cat, cnt, g_avg, s_avg, g_avg - s_avg,
        )

    # Perfect scores
    grpo_perfect = sum(1 for v in grpo_rewards.values() if v >= 0.999)
    sft_perfect = sum(1 for v in sft_rewards.values() if v >= 0.999)
    logger.info("")
    logger.info("Perfect scores (1.0): GRPO=%d, SFT=%d", grpo_perfect, sft_perfect)


def main():
    parser = argparse.ArgumentParser(description="Per-form GRPO vs SFT analysis")
    parser.add_argument("--grpo-log", required=True, help="GRPO eval log file")
    parser.add_argument("--sft-log", required=True, help="SFT eval log file")
    parser.add_argument("--data", required=True, help="Data JSONL file (val or test)")
    parser.add_argument("--split", default="val", help="Split name for display")
    args = parser.parse_args()

    grpo_rewards = parse_eval_log(args.grpo_log)
    sft_rewards = parse_eval_log(args.sft_log)
    forms = load_form_mapping(args.data)

    logger.info("Loaded %d GRPO rewards, %d SFT rewards, %d forms",
                len(grpo_rewards), len(sft_rewards), len(forms))

    analyze(grpo_rewards, sft_rewards, forms, args.split.upper())


if __name__ == "__main__":
    main()
