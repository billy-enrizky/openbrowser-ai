"""Generate markdown/CSV evaluation reports."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from infra.eval.pipelines.results_aggregator import (
    aggregate_by_dataset,
    aggregate_by_model,
    aggregate_by_project,
    load_run_summaries,
)
from infra.eval.pipelines.results_schema import RunSummary

logger = logging.getLogger(__name__)


def generate_markdown_report(summaries: list[RunSummary], output_path: Path) -> str:
    """Generate a markdown evaluation report."""
    projects = aggregate_by_project(summaries)
    models = aggregate_by_model(summaries)
    datasets = aggregate_by_dataset(summaries)

    lines = []
    lines.append("# OpenBrowser-AI Evaluation Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal runs analyzed: {len(summaries)}")

    # Project summary
    lines.append("\n## Project Summary\n")
    lines.append("| Project | Runs | Tasks | Successes | Success Rate | Best Run |")
    lines.append("|---------|------|-------|-----------|--------------|----------|")
    for project, stats in sorted(projects.items()):
        rate = stats["total_successes"] / stats["total_tasks"] if stats["total_tasks"] > 0 else 0.0
        lines.append(
            f"| {project} | {stats['runs']} | {stats['total_tasks']} | "
            f"{stats['total_successes']} | {rate:.1%} | {stats['best_run_id'][:20]} |"
        )

    # Model comparison
    lines.append("\n## Model Comparison\n")
    lines.append("| Model | Tasks | Success Rate | Avg Time | Projects |")
    lines.append("|-------|-------|--------------|----------|----------|")
    for model, stats in sorted(models.items()):
        lines.append(
            f"| {model} | {stats['tasks']} | {stats['success_rate']:.1%} | "
            f"{stats['avg_time']:.2f}s | {', '.join(stats['projects'])} |"
        )

    # Dataset breakdown
    lines.append("\n## Dataset Breakdown\n")
    lines.append("| Dataset | Tasks | Success Rate | Avg Time |")
    lines.append("|---------|-------|--------------|----------|")
    for ds, stats in sorted(datasets.items()):
        lines.append(
            f"| {ds} | {stats['tasks']} | {stats['success_rate']:.1%} | "
            f"{stats['avg_time']:.2f}s |"
        )

    # Per-run details
    lines.append("\n## Individual Runs\n")
    for s in sorted(summaries, key=lambda x: x.started_at, reverse=True):
        lines.append(f"### {s.run_id}")
        lines.append(f"- Project: {s.project}")
        lines.append(f"- Started: {s.started_at}")
        lines.append(f"- Tasks: {s.total_tasks}, Success: {s.total_successes} ({s.success_rate:.1%})")
        lines.append(f"- Datasets: {', '.join(s.datasets)}")
        lines.append(f"- Models: {', '.join(s.models)}")
        lines.append(f"- Agent types: {', '.join(s.agent_types)}")
        lines.append("")

    report = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Markdown report saved to {output_path}")
    return report


def generate_csv_summary(summaries: list[RunSummary], output_path: Path):
    """Generate a CSV summary across all runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "project", "started_at", "datasets", "models",
            "agent_types", "total_tasks", "successes", "success_rate",
            "avg_time",
        ])
        for s in sorted(summaries, key=lambda x: x.started_at, reverse=True):
            writer.writerow([
                s.run_id,
                s.project,
                s.started_at.isoformat() if s.started_at else "",
                ";".join(s.datasets),
                ";".join(s.models),
                ";".join(s.agent_types),
                s.total_tasks,
                s.total_successes,
                f"{s.success_rate:.3f}",
                f"{s.avg_execution_time:.2f}",
            ])

    logger.info(f"CSV summary saved to {output_path}")


def main():
    """Generate reports from all results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    summaries = load_run_summaries("results")
    if not summaries:
        logger.warning("No results found in results/ directory")
        return

    generate_markdown_report(summaries, Path("results/report.md"))
    generate_csv_summary(summaries, Path("results/summary.csv"))

    logger.info("Reports generated successfully")


if __name__ == "__main__":
    main()
