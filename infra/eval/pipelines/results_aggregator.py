"""Cross-project comparison and aggregation of evaluation results."""

import json
import logging
from pathlib import Path

from infra.eval.pipelines.results_schema import RunSummary

logger = logging.getLogger(__name__)


def load_run_summaries(results_dir: str = "results") -> list[RunSummary]:
    """Load all run summaries from the results directory."""
    base = Path(results_dir)
    summaries = []

    for summary_file in sorted(base.rglob("summary.json")):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            summary = RunSummary(**data)
            summaries.append(summary)
            logger.info(f"Loaded run {summary.run_id} ({summary.project})")
        except Exception as e:
            logger.error(f"Failed to load {summary_file}: {e}")

    logger.info(f"Loaded {len(summaries)} run summaries")
    return summaries


def aggregate_by_project(summaries: list[RunSummary]) -> dict[str, dict]:
    """Aggregate results grouped by project (CSC490, STAD68, STAD80)."""
    projects: dict[str, dict] = {}

    for s in summaries:
        if s.project not in projects:
            projects[s.project] = {
                "runs": 0,
                "total_tasks": 0,
                "total_successes": 0,
                "best_success_rate": 0.0,
                "best_run_id": "",
                "agent_totals": {},
                "model_totals": {},
            }

        p = projects[s.project]
        p["runs"] += 1
        p["total_tasks"] += s.total_tasks
        p["total_successes"] += s.total_successes

        if s.success_rate > p["best_success_rate"]:
            p["best_success_rate"] = s.success_rate
            p["best_run_id"] = s.run_id

        for agent_type, stats in s.agent_summaries.items():
            if agent_type not in p["agent_totals"]:
                p["agent_totals"][agent_type] = {"tasks": 0, "successes": 0, "total_time": 0.0}
            p["agent_totals"][agent_type]["tasks"] += stats["total"]
            p["agent_totals"][agent_type]["successes"] += stats["successes"]
            p["agent_totals"][agent_type]["total_time"] += stats["avg_time"] * stats["total"]

        for model, stats in s.model_summaries.items():
            if model not in p["model_totals"]:
                p["model_totals"][model] = {"tasks": 0, "successes": 0, "total_time": 0.0}
            p["model_totals"][model]["tasks"] += stats["total"]
            p["model_totals"][model]["successes"] += stats["successes"]
            p["model_totals"][model]["total_time"] += stats["avg_time"] * stats["total"]

    return projects


def aggregate_by_model(summaries: list[RunSummary]) -> dict[str, dict]:
    """Aggregate results grouped by model across all projects."""
    models: dict[str, dict] = {}

    for s in summaries:
        for model, stats in s.model_summaries.items():
            if model not in models:
                models[model] = {"tasks": 0, "successes": 0, "total_time": 0.0, "projects": set()}
            models[model]["tasks"] += stats["total"]
            models[model]["successes"] += stats["successes"]
            models[model]["total_time"] += stats["avg_time"] * stats["total"]
            models[model]["projects"].add(s.project)

    # Convert sets to lists for serialization
    for m in models.values():
        m["projects"] = sorted(m["projects"])
        m["success_rate"] = m["successes"] / m["tasks"] if m["tasks"] > 0 else 0.0
        m["avg_time"] = m["total_time"] / m["tasks"] if m["tasks"] > 0 else 0.0

    return models


def aggregate_by_dataset(summaries: list[RunSummary]) -> dict[str, dict]:
    """Aggregate results grouped by dataset across all projects."""
    datasets: dict[str, dict] = {}

    for s in summaries:
        for ds, stats in s.dataset_summaries.items():
            if ds not in datasets:
                datasets[ds] = {"tasks": 0, "successes": 0, "total_time": 0.0}
            datasets[ds]["tasks"] += stats["total"]
            datasets[ds]["successes"] += stats["successes"]
            datasets[ds]["total_time"] += stats["avg_time"] * stats["total"]

    for d in datasets.values():
        d["success_rate"] = d["successes"] / d["tasks"] if d["tasks"] > 0 else 0.0
        d["avg_time"] = d["total_time"] / d["tasks"] if d["tasks"] > 0 else 0.0

    return datasets


def print_cross_project_table(summaries: list[RunSummary]):
    """Print cross-project comparison table."""
    projects = aggregate_by_project(summaries)

    logger.info("=" * 80)
    logger.info("CROSS-PROJECT COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Project':<12} {'Runs':<6} {'Tasks':<8} {'Success':<10} {'Best Rate':<12} {'Best Run'}")
    logger.info("-" * 80)

    for project, stats in sorted(projects.items()):
        overall_rate = stats["total_successes"] / stats["total_tasks"] if stats["total_tasks"] > 0 else 0.0
        logger.info(
            f"{project:<12} {stats['runs']:<6} {stats['total_tasks']:<8} "
            f"{stats['total_successes']:<10} {stats['best_success_rate']:.1%}        "
            f"{stats['best_run_id'][:20]}"
        )

    # Model comparison
    models = aggregate_by_model(summaries)
    logger.info("")
    logger.info("MODEL COMPARISON (across all projects)")
    logger.info(f"{'Model':<30} {'Tasks':<8} {'Success Rate':<14} {'Avg Time'}")
    logger.info("-" * 70)

    for model, stats in sorted(models.items()):
        logger.info(
            f"{model:<30} {stats['tasks']:<8} {stats['success_rate']:.1%}          "
            f"{stats['avg_time']:.2f}s"
        )
