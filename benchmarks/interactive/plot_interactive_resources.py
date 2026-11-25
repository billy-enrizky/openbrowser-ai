"""Plot the interactive resource lane for documentation.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/interactive/plot_interactive_resources.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "results-interactive-resources-2026-11-25-n5-final.summary.json"
DEFAULT_OUTPUT = HERE.parents[1] / "docs" / "images" / "native-interactive-resources.png"
TASKS = ("wizard", "modal", "download", "tabs", "auth")
ARMS = (
    ("openbrowser-mcp-fast", "OpenBrowser MCP", "#0f6270"),
    ("openbrowser-fast", "OpenBrowser CLI", "#58a69f"),
    ("chrome-devtools-mcp", "chrome-devtools-mcp", "#c2852a"),
    ("agent-browser", "agent-browser", "#4f7f5b"),
    ("browser-use", "browser-use", "#8871b4"),
)
TEXT = "#171717"
MUTED = "#6b7280"
GRID = "#e5e7eb"
MIB = 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    values = {(row["task"], row["arm"]): row for row in rows}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    bar_height = 0.17
    offsets = [index - (len(ARMS) - 1) / 2 for index in range(len(ARMS))]
    positions = list(range(len(TASKS)))
    for axis, field, title in (
        (axes[0], "peak_child_footprint_bytes", "Browser child physical footprint"),
        (axes[1], "peak_tree_rss_bytes", "Process-tree RSS"),
    ):
        axis.set_facecolor("#ffffff")
        for offset, (arm, label, color) in zip(offsets, ARMS, strict=True):
            vals = [
                values[(task, arm)]["metrics"][field]["median"] / MIB
                for task in TASKS
            ]
            axis.barh(
                [position + offset * bar_height for position in positions],
                vals,
                height=bar_height,
                color=color,
                label=label,
                zorder=3,
            )
        axis.set_yticks(positions, [task.capitalize() for task in TASKS], fontsize=12)
        axis.invert_yaxis()
        axis.set_xlabel(f"{title} (MiB) \u2193 Lower is better", fontsize=11)
        axis.set_title(title, fontsize=14, color=TEXT, pad=12)
        axis.grid(axis="x", color=GRID, linewidth=1, zorder=0)
        axis.set_axisbelow(True)
        axis.margins(x=0.18)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=5,
        frameon=False,
        fontsize=10.5,
    )
    fig.suptitle(
        "Interactive resources \u2193 Lower is better",
        fontsize=19,
        color=TEXT,
        x=0.06,
        ha="left",
        y=0.98,
    )
    fig.text(
        0.06,
        0.93,
        "N=5 per arm and task. All 125 observations are resource-eligible. CPU is a sampled lower bound.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
