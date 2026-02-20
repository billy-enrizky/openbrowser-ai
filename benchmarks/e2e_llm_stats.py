"""
Statistical E2E LLM Benchmark Runner.

Runs the benchmark N times per server, computes mean/std/CI via bootstrap.

Usage:
    uv run python benchmarks/e2e_llm_stats.py --runs 5 --bootstrap 10000
    uv run python benchmarks/e2e_llm_stats.py --runs 5 --servers openbrowser
"""
import argparse
import asyncio
import json
import logging
import math
import random

import boto3

from e2e_llm_benchmark import (
    SERVERS,
    TASKS,
    DEFAULT_MODEL,
    aggregate_results,
    run_task,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def bootstrap_ci(data: list[float], n_bootstrap: int = 10000, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence interval for the mean."""
    n = len(data)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    means = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - ci) / 2
    low_idx = int(math.floor(alpha * n_bootstrap))
    high_idx = int(math.ceil((1 - alpha) * n_bootstrap)) - 1

    sample_mean = sum(data) / n
    sample_std = (sum((x - sample_mean) ** 2 for x in data) / max(n - 1, 1)) ** 0.5

    return {
        "mean": round(sample_mean, 1),
        "std": round(sample_std, 1),
        "ci_low": round(means[low_idx], 1),
        "ci_high": round(means[high_idx], 1),
        "bootstrap_mean": round(sum(means) / len(means), 1),
    }


async def run_multi(
    server_names: list[str] | None = None,
    task_names: list[str] | None = None,
    n_runs: int = 5,
    n_bootstrap: int = 10000,
    model: str | None = None,
    output_path: str | None = None,
):
    """Run the benchmark n_runs times per server and compute statistics."""
    import pathlib
    if output_path is None:
        output_path = str(pathlib.Path(__file__).parent / "e2e_llm_stats_results.json")
    model_id = model or DEFAULT_MODEL

    servers_to_run = {
        name: config for name, config in SERVERS.items()
        if server_names is None or name in server_names
    }
    tasks_to_run = [
        t for t in TASKS
        if task_names is None or t["name"] in task_names
    ]

    logger.info("=" * 70)
    logger.info("Statistical E2E LLM Benchmark")
    logger.info("=" * 70)
    logger.info("Model: %s", model_id)
    logger.info("Servers: %s", ", ".join(servers_to_run.keys()))
    logger.info("Tasks: %s", ", ".join(t["name"] for t in tasks_to_run))
    logger.info("Runs per server: %d", n_runs)
    logger.info("Bootstrap samples: %d", n_bootstrap)
    logger.info("")

    bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

    all_results = {}

    for server_name, server_config in servers_to_run.items():
        logger.info("=" * 60)
        logger.info("Server: %s (%d runs)", server_name, n_runs)
        logger.info("=" * 60)

        run_durations = []
        run_tool_calls = []
        run_pass_counts = []
        run_bedrock_input_tokens = []
        run_bedrock_output_tokens = []
        run_total_bedrock_tokens = []
        per_task_durations = {t["name"]: [] for t in tasks_to_run}
        per_task_tool_calls = {t["name"]: [] for t in tasks_to_run}
        per_task_pass = {t["name"]: [] for t in tasks_to_run}
        raw_runs = []

        for run_idx in range(n_runs):
            logger.info("")
            logger.info("--- Run %d/%d for %s ---", run_idx + 1, n_runs, server_name)

            task_results = []
            for task in tasks_to_run:
                result = await run_task(
                    bedrock_client, server_name, server_config, task, model_id,
                )
                task_results.append(result)

            summary = aggregate_results(task_results)
            run_durations.append(summary["total_duration_s"])
            run_tool_calls.append(summary["total_tool_calls"])
            run_pass_counts.append(summary["passed"])
            run_bedrock_input_tokens.append(summary.get("bedrock_input_tokens", 0))
            run_bedrock_output_tokens.append(summary.get("bedrock_output_tokens", 0))
            run_total_bedrock_tokens.append(summary.get("total_bedrock_tokens", 0))

            for tr in task_results:
                per_task_durations[tr["name"]].append(tr["duration_s"])
                per_task_tool_calls[tr["name"]].append(tr["tool_calls"])
                per_task_pass[tr["name"]].append(1 if tr["success"] else 0)

            raw_runs.append({
                "run": run_idx + 1,
                "tasks": task_results,
                "summary": summary,
            })

            logger.info(
                "  Run %d: %d/%d passed, %.1fs, %d tool calls, %d bedrock tokens",
                run_idx + 1, summary["passed"], summary["total_tasks"],
                summary["total_duration_s"], summary["total_tool_calls"],
                summary.get("total_bedrock_tokens", 0),
            )

        # Compute statistics
        duration_stats = bootstrap_ci(run_durations, n_bootstrap)
        tool_call_stats = bootstrap_ci(run_tool_calls, n_bootstrap)
        pass_stats = bootstrap_ci(run_pass_counts, n_bootstrap)
        input_token_stats = bootstrap_ci(run_bedrock_input_tokens, n_bootstrap)
        output_token_stats = bootstrap_ci(run_bedrock_output_tokens, n_bootstrap)
        total_token_stats = bootstrap_ci(run_total_bedrock_tokens, n_bootstrap)

        per_task_stats = {}
        for task_name in per_task_durations:
            per_task_stats[task_name] = {
                "duration": bootstrap_ci(per_task_durations[task_name], n_bootstrap),
                "tool_calls": bootstrap_ci(per_task_tool_calls[task_name], n_bootstrap),
                "pass_rate": round(sum(per_task_pass[task_name]) / len(per_task_pass[task_name]), 2),
            }

        all_results[server_name] = {
            "n_runs": n_runs,
            "n_bootstrap": n_bootstrap,
            "raw_runs": raw_runs,
            "stats": {
                "duration_s": duration_stats,
                "tool_calls": tool_call_stats,
                "pass_count": pass_stats,
                "bedrock_input_tokens": input_token_stats,
                "bedrock_output_tokens": output_token_stats,
                "total_bedrock_tokens": total_token_stats,
            },
            "per_task": per_task_stats,
        }

        logger.info("")
        logger.info("Server %s summary (%d runs):", server_name, n_runs)
        logger.info(
            "  Duration: %.1f +/- %.1f s (95%% CI: %.1f - %.1f)",
            duration_stats["mean"], duration_stats["std"],
            duration_stats["ci_low"], duration_stats["ci_high"],
        )
        logger.info(
            "  Tool calls: %.1f +/- %.1f (95%% CI: %.1f - %.1f)",
            tool_call_stats["mean"], tool_call_stats["std"],
            tool_call_stats["ci_low"], tool_call_stats["ci_high"],
        )
        logger.info(
            "  Pass count: %.1f +/- %.1f / %d",
            pass_stats["mean"], pass_stats["std"], len(tasks_to_run),
        )
        logger.info(
            "  Bedrock tokens: %.0f +/- %.0f (95%% CI: %.0f - %.0f)",
            total_token_stats["mean"], total_token_stats["std"],
            total_token_stats["ci_low"], total_token_stats["ci_high"],
        )

    # Print comparison table
    logger.info("")
    logger.info("=" * 70)
    logger.info("Final Comparison (%d runs, %d bootstrap samples)", n_runs, n_bootstrap)
    logger.info("=" * 70)

    names = list(all_results.keys())
    col_w = max(len(n) for n in names) + 2

    header = f"{'Metric':<35s}"
    for name in names:
        header += f"{name:>{col_w + 10}s}"
    logger.info(header)
    logger.info("=" * len(header))

    for label, stat_key, fmt in [
        ("Duration (s)", "duration_s", ".1f"),
        ("Tool Calls", "tool_calls", ".1f"),
        ("Pass Count", "pass_count", ".1f"),
        ("Bedrock Tokens", "total_bedrock_tokens", ".0f"),
    ]:
        row = f"{label:<35s}"
        for name in names:
            s = all_results[name]["stats"][stat_key]
            val = f"{s['mean']:{fmt}} +/- {s['std']:{fmt}}"
            row += f"{val:>{col_w + 10}s}"
        logger.info(row)

    logger.info("")
    logger.info("95%% Confidence Intervals (bootstrap):")
    for label, stat_key, fmt in [
        ("Duration (s)", "duration_s", ".1f"),
        ("Tool Calls", "tool_calls", ".1f"),
        ("Bedrock Tokens", "total_bedrock_tokens", ".0f"),
    ]:
        row = f"  {label:<33s}"
        for name in names:
            s = all_results[name]["stats"][stat_key]
            val = f"[{s['ci_low']:{fmt}}, {s['ci_high']:{fmt}}]"
            row += f"{val:>{col_w + 10}s}"
        logger.info(row)

    # Write results
    output = {
        "model": model_id,
        "n_runs": n_runs,
        "n_bootstrap": n_bootstrap,
        "servers": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("")
    logger.info("Results written to %s", output_path)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Statistical E2E LLM Benchmark")
    parser.add_argument(
        "--servers", nargs="*", choices=list(SERVERS.keys()),
        help="Servers to benchmark (default: all)",
    )
    parser.add_argument(
        "--tasks", nargs="*", choices=[t["name"] for t in TASKS],
        help="Tasks to run (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of runs per server (default: 5)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=10000,
        help="Number of bootstrap samples (default: 10000)",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Bedrock model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: benchmarks/e2e_llm_stats_results.json)",
    )
    args = parser.parse_args()

    asyncio.run(run_multi(
        server_names=args.servers,
        task_names=args.tasks,
        n_runs=args.runs,
        n_bootstrap=args.bootstrap,
        model=args.model,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
