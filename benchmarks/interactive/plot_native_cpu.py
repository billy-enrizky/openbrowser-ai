"""Render separate cold and warm native CPU charts from recorded summaries.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/interactive/plot_native_cpu.py
"""

# ruff: noqa: E402
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.interactive.plot_common import (
    INTERACTIVE_TASK_LABELS,
    apply_rc,
    bar_layout,
    draw_grouped_bars,
    figure_legend,
    interactive_styles,
    resource_counts,
    resource_values,
    sample_label,
    save_figure,
    style_axes,
)

HERE = Path(__file__).resolve().parent
COLD = HERE / "results/results-cold-resources-seven-arm-2026-11-27-n5.summary.json"
WARM = HERE / "results/results-warm-resources-seven-arm-2026-11-27-n5.summary.json"
IMAGES = ROOT / "docs/images"
TASKS = ("wizard", "modal", "download", "tabs", "auth")


def render(source: Path, output: Path, *, warm: bool) -> None:
    payload = json.loads(source.read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    field = "warm_task_tree_cpu_delta_seconds" if warm else "observed_child_cpu_seconds"
    tasks = TASKS[:-1] if warm else TASKS
    styles = interactive_styles()
    layout = bar_layout(len(styles))
    values = resource_values(rows, tasks, styles, field=field)
    runs, eligible, total = resource_counts(rows, tasks, styles, field=field)
    apply_rc()
    fig, axis = plt.subplots(figsize=(14, 8 if warm else 9.3), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    draw_grouped_bars(
        axis,
        tasks=tasks,
        styles=styles,
        values=values,
        layout=layout,
        formatter=lambda value: f"{value:.3f}",
    )
    axis.set_yticks(range(len(tasks)), [INTERACTIVE_TASK_LABELS[task] for task in tasks], fontsize=12)
    axis.invert_yaxis()
    axis.set_xlabel("Sampled native CPU time (seconds) ↓ Lower is better", fontsize=11)
    style_axes(axis)
    lane = "warm session" if warm else "cold start"
    fig.suptitle(f"Native CPU, {lane} ↓ Lower is better", fontsize=20, x=0.06, ha="left", y=0.99, fontweight="bold")
    window = (
        "Per-task CPU delta after warmup. Native services and browsers included."
        if warm
        else "Fresh native services and browsers. Startup, task, and native cleanup included."
    )
    caption = (
        f"{sample_label(runs)} per measured cell. {eligible}/{total} cells with recorded CPU medians.\n"
        + window
        + "\nSampled CPU is a lower bound. Worker, controller, and fixture CPU are excluded. Unknown values remain n/a."
    )
    fig.text(0.06, 0.94, caption, fontsize=9.5, color="#6b7280", ha="left", va="top")
    figure_legend(fig, styles, layout, y=0.84)
    fig.subplots_adjust(left=0.145, right=0.98, top=0.72, bottom=0.11)
    save_figure(fig, output)
    plt.close(fig)
    logging.info("Wrote %s", output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-input", type=Path, default=COLD)
    parser.add_argument("--warm-input", type=Path, default=WARM)
    parser.add_argument("--cold-output", type=Path, default=IMAGES / "native-interactive-cold-cpu.png")
    parser.add_argument("--warm-output", type=Path, default=IMAGES / "native-interactive-warm-cpu.png")
    args = parser.parse_args()
    render(args.cold_input, args.cold_output, warm=False)
    render(args.warm_input, args.warm_output, warm=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
