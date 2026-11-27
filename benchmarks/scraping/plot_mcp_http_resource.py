"""Plot the browser-free MCP scrape_web memory lane for documentation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "results-mcp-scrape-web-resource-2026-11-26-n5.json"
DEFAULT_OUTPUT = HERE.parents[1] / "docs" / "images" / "native-mcp-http-memory-2026-11-26.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))

    def valid(rows: list[dict]) -> list[dict]:
        return [
            row
            for row in rows
            if row.get("ok") == row.get("calls")
            and row.get("peak_rss_mib") is not None
            and row.get("browser_processes_observed") is False
        ]

    sequential_rows = valid(report["sequential"])
    concurrent_rows = valid(report["concurrent"])
    if not sequential_rows or not concurrent_rows:
        print("No graph-eligible rows: exact oracle, RSS, or browser checks failed.", file=__import__("sys").stderr)
        return 2
    excluded = len(report["sequential"]) + len(report["concurrent"]) - len(sequential_rows) - len(concurrent_rows)
    sequential = [row["peak_rss_mib"] for row in sequential_rows]
    concurrent = [row["peak_rss_mib"] for row in concurrent_rows]
    calls = concurrent_rows[0]["calls"]
    labels = [
        f"One MCP scrape_web call (N={len(sequential)})",
        f"{calls} concurrent MCP calls (N={len(concurrent)})",
    ]
    values = [statistics.median(sequential), statistics.median(concurrent)]
    colors = ["#0f6270", "#58a69f"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    bars = ax.barh(labels, values, color=colors, height=0.45, zorder=3)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} MiB",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xlabel("Peak process-tree RSS (MiB) \u2193 Lower is better")
    ax.set_xlim(0, max(values) * 1.28)
    ax.grid(axis="x", color="#e5e7eb", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        "No-browser MCP scraping memory \u2193 Lower is better",
        loc="left",
        pad=32,
        fontsize=19,
        fontweight="bold",
    )
    ax.text(
        0,
        1.02,
        (
            "One local static page fixture. No Chrome, Blink, or V8 process was observed. "
            "RSS counts resident pages across the process tree."
            + (f" Excluded {excluded} invalid observations." if excluded else "")
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        color="#6b7280",
        va="bottom",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
