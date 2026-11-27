"""Plot cold and warm interactive resource lanes for documentation.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/interactive/plot_interactive_resources.py

Cold memory and warm memory are separate claims. Each chart keeps its own
input and caption. Required arms stay in the legend, and an unknown median is
drawn as ``n/a`` instead of zero.
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
    MIB,
    apply_rc,
    bar_layout,
    draw_grouped_bars,
    figure_legend,
    format_mib,
    has_metric_rows,
    interactive_styles,
    provenance_text,
    resource_counts,
    resource_values,
    sample_label,
    save_figure,
    style_axes,
    warm_footer,
    warm_resource_rows,
)

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "results-cold-resources-seven-arm-2026-11-27-n5.summary.json"
DEFAULT_OUTPUT = HERE.parents[1] / "docs" / "images" / "native-interactive-resources.png"
DEFAULT_WARM_OUTPUT = HERE.parents[1] / "docs" / "images" / "native-interactive-warm-resources.png"
DEFAULT_WARM_INPUT = HERE / "results" / "results-warm-resources-seven-arm-2026-11-27-n5.jsonl"
TASKS = ("wizard", "modal", "download", "tabs", "auth")
WARM_TASKS = ("wizard", "modal", "download", "tabs")
METRICS = (
    ("peak_child_footprint_bytes", "Browser child footprint"),
    ("peak_child_tree_rss_bytes", "Native child-tree RSS"),
)
WARM_SOURCES = {
    "peak_child_footprint_bytes": "warm_peak_child_footprint_bytes",
    "peak_tree_rss_bytes": "warm_peak_tree_rss_bytes",
}


def _load_json_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("rows", []))
    return list(payload)


def _load_warm_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return _load_json_rows(path)


def _footer_note(records: list[dict]) -> str:
    footer = warm_footer(records)
    if footer is None:
        return "Lane footer not recorded."
    if footer.get("valid") is not True:
        details = footer.get("setup_error") or footer.get("abort_reason") or f"failures={footer.get('failures')}"
        raise SystemExit(f"Warm resource lane is not valid: {details}")
    return "Lane validity passed."


def _plot_lane(
    summary_rows: list[dict],
    *,
    tasks: tuple[str, ...],
    styles,
    title: str,
    note: str,
    output: Path,
) -> None:
    layout = bar_layout(len(styles))
    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    eligible_parts: list[str] = []
    runs: list[int] = []
    for axis, (field, label) in zip(axes, METRICS, strict=True):
        axis.set_facecolor("#ffffff")
        values = resource_values(summary_rows, tasks, styles, field=field)
        values_mib = {key: value / MIB if value is not None else None for key, value in values.items()}
        draw_grouped_bars(
            axis,
            tasks=tasks,
            styles=styles,
            values=values_mib,
            layout=layout,
            formatter=format_mib,
        )
        cell_runs, eligible, total = resource_counts(summary_rows, tasks, styles, field=field)
        runs.extend(cell_runs)
        eligible_parts.append(f"{label} cells: {eligible}/{total}")
        positions = list(range(len(tasks)))
        axis.set_yticks(positions, [INTERACTIVE_TASK_LABELS[task] for task in tasks], fontsize=12)
        axis.invert_yaxis()
        axis.set_title(label, fontsize=14, pad=12)
        axis.set_xlabel("Peak memory (MiB) \u2193 Lower is better", fontsize=11)
        style_axes(axis)

    fig.suptitle(title, fontsize=20, x=0.06, ha="left", y=0.99, fontweight="bold")
    caption_text = (
        f"{sample_label(runs)} per measured cell. "
        + ". ".join(eligible_parts)
        + ".\nn/a means no eligible observation. "
        + note
    )
    fig.text(0.06, 0.94, caption_text, fontsize=9.8, color="#6b7280", ha="left", va="top", wrap=True)
    figure_legend(fig, styles, layout, y=0.88)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.13, wspace=0.28)
    save_figure(fig, output)
    plt.close(fig)
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--warm-input", type=Path, default=DEFAULT_WARM_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warm-output", type=Path, default=DEFAULT_WARM_OUTPUT)
    parser.add_argument("--arms", nargs="+", default=list(CANONICAL_ARMS))
    args = parser.parse_args()

    styles = interactive_styles(tuple(args.arms))
    cold_rows = _load_json_rows(args.input)
    cold_note = "Cold start. Medians of resource-eligible observations. " + provenance_text(cold_rows, TASKS, styles)
    _plot_lane(
        cold_rows,
        tasks=TASKS,
        styles=styles,
        title="Interactive resources, cold start \u2193 Lower is better",
        note=cold_note,
        output=args.output,
    )

    if args.warm_input is not None:
        records = _load_warm_records(args.warm_input)
        footer_note = _footer_note(records)
        if has_metric_rows(records):
            warm_rows = records
        else:
            warm_rows = warm_resource_rows(records, WARM_TASKS, styles, metrics=WARM_SOURCES)
        warm_note = f"Warm session after a verified warmup. Medians of resource-eligible observations. {footer_note}"
        _plot_lane(
            warm_rows,
            tasks=WARM_TASKS,
            styles=styles,
            title="Interactive resources, warm session \u2193 Lower is better",
            note=warm_note,
            output=args.warm_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
