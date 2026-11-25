"""Plot current scraping benchmark results for the documentation site.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/scraping/plot_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
DEFAULT_FIRECRAWL = HERE / "results" / "results-firecrawl-final6-2026-11-25-n10-interleaved.json"
DEFAULT_LIGHTPANDA = HERE / "results" / "results-lightpanda-final6-2026-11-25-n10-interleaved.json"
DEFAULT_OUTPUT = HERE.parents[1] / "docs" / "images"
TASKS = ("static_article", "pagination", "table", "js_list", "asset_heavy")
TASK_LABELS = {
    "static_article": "static article",
    "pagination": "pagination",
    "table": "table",
    "js_list": "JS list",
    "asset_heavy": "asset heavy",
}
BACKGROUND = "#ffffff"
TEXT = "#171717"
MUTED = "#6b7280"
GRID = "#e5e7eb"


def _load(path: Path) -> tuple[dict, dict[tuple[str, str], dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    groups = {(group["task"], group["arm"]): group for group in report["groups"]}
    return report, groups


def _load_text(report: dict) -> str:
    environment = report["environment"]
    start = environment.get("load_average_start") or environment.get("load_average")
    if start is None:
        return f"N={report['samples']} per arm and task, {report['warmups']} warmups, interleaved."
    load = " / ".join(f"{value:.2f}" for value in start)
    return (
        f"N={report['samples']} per arm and task, {report['warmups']} warmups, interleaved. "
        f"Host load at start: {load} on 18 logical CPUs."
    )


def _ms(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _plot(
    *,
    report: dict,
    groups: dict[tuple[str, str], dict],
    arms: tuple[tuple[str, str, str], ...],
    title: str,
    output: Path,
) -> None:
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
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    bar_height = 0.22
    offsets = [index - (len(arms) - 1) / 2 for index in range(len(arms))]
    positions = list(range(len(TASKS)))
    for offset, (arm, label, color) in zip(offsets, arms, strict=True):
        values = [groups[(task, arm)]["latency"]["p50_ms"] for task in TASKS]
        y = [position + offset * bar_height for position in positions]
        bars = ax.barh(y, values, height=bar_height, color=color, label=label, zorder=3)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value * 1.12,
                bar.get_y() + bar.get_height() / 2,
                _ms(value),
                va="center",
                ha="left",
                fontsize=10.5,
                color=TEXT,
                zorder=4,
            )

    ax.set_yticks(positions, [TASK_LABELS[task] for task in TASKS], fontsize=14)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _ms(value)))
    ax.set_xlabel("p50 wall-clock latency (ms, logarithmic scale) \u2193 Lower is better", labelpad=12)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=8)

    fig.suptitle(title, x=0.06, y=0.99, ha="left", va="top", fontsize=24, fontweight="bold", color=TEXT)
    fig.text(0.06, 0.94, _load_text(report), ha="left", va="top", fontsize=12.5, color=MUTED)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(arms),
        frameon=False,
        fontsize=12.5,
        handlelength=1.2,
    )
    fig.subplots_adjust(left=0.17, right=0.97, top=0.80, bottom=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firecrawl-results", type=Path, default=DEFAULT_FIRECRAWL)
    parser.add_argument("--lightpanda-results", type=Path, default=DEFAULT_LIGHTPANDA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    firecrawl_report, firecrawl_groups = _load(args.firecrawl_results)
    _plot(
        report=firecrawl_report,
        groups=firecrawl_groups,
        arms=(
            ("openbrowser-full", "OpenBrowser full", "#0f6270"),
            ("openbrowser-lightweight", "OpenBrowser lightweight", "#6db3ab"),
            ("firecrawl-local", "Firecrawl Local", "#c25b2a"),
        ),
        title="Scraping p50 latency: OpenBrowser vs Firecrawl Local \u2193 Lower is better",
        output=args.output_dir / "native-scraping-current-p50.png",
    )

    lightpanda_report, lightpanda_groups = _load(args.lightpanda_results)
    _plot(
        report=lightpanda_report,
        groups=lightpanda_groups,
        arms=(
            ("openbrowser-no-render", "OpenBrowser no-render", "#0f6270"),
            ("lightpanda", "Lightpanda", "#8871b4"),
        ),
        title="No-render p50 latency: OpenBrowser vs Lightpanda \u2193 Lower is better",
        output=args.output_dir / "native-lightpanda-current-p50.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
