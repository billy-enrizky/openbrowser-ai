"""Plot the current warm interactive benchmark summary for documentation.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/interactive/plot_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE / "results" / "results-warm-five-arm-final2-2026-11-24-n5.summary.json"
)
DEFAULT_OUTPUT = HERE.parents[1] / "docs" / "images" / "native-interactive-warm-current-p50.png"
TASKS = ("wizard", "modal", "download", "tabs")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = {(row["task"], row["arm"]): row for row in report["rows"]}
    arms = tuple(
        arm
        for arm in ARMS
        if all((task, arm[0]) in rows for task in TASKS)
    )
    if not arms:
        raise SystemExit("No complete arm is present in the summary")
    samples = report["environment"]["samples"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
        }
    )
    fig, ax = plt.subplots(figsize=(14, 9), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bar_height = 0.16
    offsets = [index - (len(arms) - 1) / 2 for index in range(len(arms))]
    positions = list(range(len(TASKS)))
    for offset, (arm, label, color) in zip(offsets, arms, strict=True):
        values = [rows[(task, arm)]["warm_p50_ms"] for task in TASKS]
        y = [position + offset * bar_height for position in positions]
        bars = ax.barh(y, values, height=bar_height, color=color, label=label, zorder=3)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 30,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.0f}",
                va="center",
                ha="left",
                fontsize=10.5,
                color=TEXT,
                zorder=4,
            )

    ax.set_yticks(positions, [task.capitalize() for task in TASKS], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("p50 wall-clock latency (ms)", labelpad=12)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=8)

    fig.suptitle(
        "Interactive browsing, warm p50 with the selector fast path",
        x=0.06,
        y=0.99,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.06,
        0.94,
        f"N={samples} per arm and task. Each arm starts once and uses a fresh fixture per task. Arm-major run on a shared host.",
        ha="left",
        va="top",
        fontsize=12.5,
        color=MUTED,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(arms),
        frameon=False,
        fontsize=12.5,
        handlelength=1.2,
    )
    fig.subplots_adjust(left=0.14, right=0.97, top=0.8, bottom=0.14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
