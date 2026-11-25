"""Plot the current cold interactive benchmark summary for documentation.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/interactive/plot_cold_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / 'results' / 'results-cold-six-arm-quiet-2026-11-24-n5.summary.json'
DEFAULT_OUTPUT = HERE.parents[1] / 'docs' / 'images' / 'native-interactive-cold-p50.png'
TASKS = ('wizard', 'modal', 'download', 'tabs', 'auth')
ARMS = (
    ('openbrowser-mcp-fast', 'OpenBrowser MCP', '#0f6270'),
    ('openbrowser-fast', 'OpenBrowser CLI', '#58a69f'),
    ('chrome-devtools-mcp', 'chrome-devtools-mcp', '#c2852a'),
    ('agent-browser', 'agent-browser', '#4f7f5b'),
    ('browser-use', 'browser-use', '#8871b4'),
)
TEXT = '#171717'
MUTED = '#6b7280'
GRID = '#e5e7eb'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding='utf-8'))
    values = {(row['task'], row['arm']): row for row in rows}
    samples = rows[0].get('runs', 3) if rows else 3

    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 12,
            'axes.edgecolor': GRID,
            'axes.labelcolor': MUTED,
            'text.color': TEXT,
            'xtick.color': MUTED,
            'ytick.color': TEXT,
        }
    )
    fig, ax = plt.subplots(figsize=(14, 10), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    bar_height = 0.16
    offsets = [index - (len(ARMS) - 1) / 2 for index in range(len(ARMS))]
    positions = list(range(len(TASKS)))
    for offset, (arm, label, color) in zip(offsets, ARMS, strict=True):
        arm_values = [values[(task, arm)]['p50_ms'] for task in TASKS]
        y = [position + offset * bar_height for position in positions]
        bars = ax.barh(y, arm_values, height=bar_height, color=color, label=label, zorder=3)
        for bar, value in zip(bars, arm_values, strict=True):
            ax.text(
                value + 30,
                bar.get_y() + bar.get_height() / 2,
                f'{value:.0f}',
                va='center',
                ha='left',
                fontsize=9.5,
                color=TEXT,
                zorder=4,
            )

    ax.set_yticks(positions, [task.capitalize() for task in TASKS], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel('p50 wall-clock latency (ms)', fontsize=13)
    ax.set_xlim(0, max(row['p50_ms'] for row in rows) * 1.18)
    ax.grid(axis='x', color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=11)
    ax.set_title('Interactive browsing, cold p50', fontsize=19, color=TEXT, loc='left', pad=40)
    ax.text(
        0,
        1.005,
        f'N={samples} per arm and task. Each run starts a fresh process tree, daemon, and browser. Shared host.',
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
        ha='left',
        va='bottom',
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
