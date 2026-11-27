"""Plot the current cold interactive benchmark summary for documentation.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/interactive/plot_cold_results.py

Every required arm appears in the legend. A missing or invalid cell is drawn
as ``n/a`` and never as a zero-length bar.
"""

# ruff: noqa: E402
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.interactive.arms import CANONICAL_ARMS
from benchmarks.interactive.plot_common import (
    INTERACTIVE_TASK_LABELS,
    apply_rc,
    bar_layout,
    caption,
    draw_grouped_bars,
    figure_legend,
    format_ms,
    interactive_styles,
    save_figure,
    style_axes,
    summary_counts,
    summary_latency_values,
    top_margin,
)

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "results-cold-seven-arm-2026-11-27-n5.summary.json"
DEFAULT_OUTPUT = HERE.parents[1] / "docs" / "images" / "native-interactive-cold-p50.png"
TASKS = ("wizard", "modal", "download", "tabs", "auth")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arms", nargs="+", default=list(CANONICAL_ARMS))
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    styles = interactive_styles(tuple(args.arms))
    values = summary_latency_values(rows, TASKS, styles, field="p50_ms")
    runs, eligible, total, ok_samples, total_samples = summary_counts(rows, TASKS, styles, value_key="p50_ms")
    layout = bar_layout(len(styles))

    apply_rc()
    fig, ax = plt.subplots(figsize=(15, 10), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    draw_grouped_bars(
        ax,
        tasks=TASKS,
        styles=styles,
        values=values,
        layout=layout,
        formatter=format_ms,
    )
    positions = list(range(len(TASKS)))
    ax.set_yticks(positions, [INTERACTIVE_TASK_LABELS[task] for task in TASKS], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Task time, p50 (ms) \u2193 Lower is better", fontsize=13)
    style_axes(ax)

    fig.suptitle(
        "Interactive browsing, cold start: task time \u2193 Lower is better",
        x=0.06,
        y=0.99,
        ha="left",
        va="top",
        fontsize=23,
        fontweight="bold",
    )
    note = (
        f"{ok_samples}/{total_samples} samples correct. "
        "Each run starts a fresh process tree, daemon, and browser. Shared host."
    )
    fig.text(
        0.06,
        0.945,
        caption(runs, eligible_cells=eligible, total_cells=total, note=note),
        ha="left",
        va="top",
        fontsize=11.5,
        color="#6b7280",
        wrap=True,
    )
    figure_legend(fig, styles, layout, y=0.875)
    fig.subplots_adjust(left=0.16, right=0.97, top=top_margin(layout, base=0.82), bottom=0.13)
    save_figure(fig, args.output)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
