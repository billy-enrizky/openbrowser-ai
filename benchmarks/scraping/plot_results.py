"""Plot current scraping benchmark results for the documentation site.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/scraping/plot_results.py

The comprehensive chart merges every supplied report and keeps all required
arms, including Playwright MCP and CLI, Firecrawl Local, and Lightpanda. A
missing or invalid cell is drawn as ``n/a`` and never as zero.
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
from matplotlib.ticker import FuncFormatter

from benchmarks.interactive.plot_common import (
    SCRAPING_ARMS,
    apply_rc,
    bar_layout,
    caption,
    draw_grouped_bars,
    figure_legend,
    format_ms,
    group_counts,
    latency_values,
    merge_groups,
    save_figure,
    scraping_styles,
    style_axes,
    top_margin,
)

HERE = Path(__file__).resolve().parent
DEFAULT_FIRECRAWL = HERE / "results" / "results-firecrawl-final6-2026-11-25-n10-interleaved.json"
DEFAULT_LIGHTPANDA = HERE / "results" / "results-lightpanda-final6-2026-11-25-n10-interleaved.json"
DEFAULT_PLAYWRIGHT = HERE / "results" / "results-playwright-configured-2026-11-28-n5.json"
DEFAULT_OUTPUT = HERE.parents[1] / "docs" / "images"
TASKS = ("static_article", "pagination", "table", "js_list", "asset_heavy")
TASK_LABELS = {
    "static_article": "Static article",
    "pagination": "Pagination",
    "table": "Table",
    "js_list": "JS list",
    "asset_heavy": "Asset heavy",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot(
    *,
    groups: dict[tuple[str, str], dict],
    styles,
    title: str,
    note: str,
    output: Path,
) -> None:
    layout = bar_layout(len(styles))
    values = latency_values(groups, TASKS, styles, field="p50_ms")
    runs, eligible, total, ok_samples, total_samples = group_counts(groups, TASKS, styles)

    apply_rc()
    fig, ax = plt.subplots(figsize=(15, 9), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.set_xscale("log")
    draw_grouped_bars(
        ax,
        tasks=TASKS,
        styles=styles,
        values=values,
        layout=layout,
        formatter=format_ms,
        log=True,
    )
    positions = list(range(len(TASKS)))
    ax.set_yticks(positions, [TASK_LABELS[task] for task in TASKS], fontsize=14)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_ms(value)))
    ax.set_xlabel("Task time, p50 (ms, logarithmic scale) \u2193 Lower is better", fontsize=12.5)
    style_axes(ax)

    fig.suptitle(title, x=0.06, y=0.99, ha="left", va="top", fontsize=23, fontweight="bold")
    sample_note = f"{ok_samples}/{total_samples} samples correct. {note}"
    fig.text(
        0.06,
        0.945,
        caption(runs, eligible_cells=eligible, total_cells=total, note=sample_note),
        ha="left",
        va="top",
        fontsize=11,
        color="#6b7280",
        wrap=True,
    )
    figure_legend(fig, styles, layout, y=0.87)
    fig.subplots_adjust(left=0.17, right=0.97, top=top_margin(layout, base=0.81), bottom=0.12)
    save_figure(fig, output)
    plt.close(fig)
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firecrawl-results", type=Path, default=DEFAULT_FIRECRAWL)
    parser.add_argument("--lightpanda-results", type=Path, default=DEFAULT_LIGHTPANDA)
    parser.add_argument("--playwright-results", type=Path, default=DEFAULT_PLAYWRIGHT)
    parser.add_argument("--extra-results", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arms", nargs="+", default=list(SCRAPING_ARMS))
    args = parser.parse_args()

    current = _load(args.playwright_results)
    reports = [current]
    for path, arm in ((args.firecrawl_results, "firecrawl-local"), (args.lightpanda_results, "lightpanda")):
        report = _load(path)
        reports.append({"groups": [group for group in report["groups"] if group["arm"] == arm]})
    reports.extend(_load(path) for path in args.extra_results)
    merged = merge_groups(reports)
    styles = scraping_styles(tuple(args.arms))
    _plot(
        groups=merged,
        styles=styles,
        title="Configured scraping: task time \u2193 Lower is better",
        note=(
            "OpenBrowser and Playwright: 2026-11-28, N=5, ready sessions. "
            "Firecrawl Local and Lightpanda: 2026-11-25 reference lanes, N=10, different lifecycle scopes. "
            "OpenBrowser selects HTTP or browser; Playwright uses Chrome."
        ),
        output=args.output_dir / "native-scraping-current-p50.png",
    )

    _plot(
        groups=merge_groups([current]),
        styles=scraping_styles(tuple(current["arms"])),
        title="Ready-session scraping \u2193 Lower is better",
        note=(
            "2026-11-28, N=5, Chrome 159.0.8154.6. Session startup, warmup and cleanup are outside timing. "
            "OpenBrowser selects HTTP or browser; Playwright uses Chrome. Memory and CPU are unmeasured."
        ),
        output=args.output_dir / "native-scraping-playwright-2026-11-28-p50.png",
    )

    lightpanda = _load(args.lightpanda_results)
    lightpanda_groups = {(group["task"], group["arm"]): group for group in lightpanda.get("groups", [])}
    focused = scraping_styles(("openbrowser-no-render", "lightpanda"))
    _plot(
        groups=lightpanda_groups,
        styles=focused,
        title="No-render: OpenBrowser vs Lightpanda \u2193 Lower is better",
        note="Lightpanda remains a separate no-render comparison.",
        output=args.output_dir / "native-lightpanda-current-p50.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
