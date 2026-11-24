"""Plot the scraping resource lane for documentation.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/scraping/plot_resources.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / 'results-resources-2026-11-24-n3.json'
DEFAULT_OUTPUT = HERE.parents[1] / 'docs' / 'images' / 'native-scraping-resources.png'
TASKS = ('static_article', 'js_list', 'pagination', 'table', 'asset_heavy')
ARMS = (
    ('openbrowser-lightweight', 'OpenBrowser lightweight', '#58a69f'),
    ('firecrawl-local', 'Firecrawl Local', '#c25b2a'),
    ('lightpanda', 'Lightpanda', '#8871b4'),
)
TEXT = '#171717'
MUTED = '#6b7280'
GRID = '#e5e7eb'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding='utf-8'))
    groups = {(group['task'], group['arm']): group for group in report['groups']}
    samples = report['samples']
    warmups = report['warmups']

    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.edgecolor': GRID,
            'axes.labelcolor': MUTED,
            'text.color': TEXT,
            'xtick.color': MUTED,
            'ytick.color': TEXT,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    for axis in axes:
        axis.set_facecolor('#ffffff')

    bar_height = 0.18
    offsets = [index - (len(ARMS) - 1) / 2 for index in range(len(ARMS))]
    positions = list(range(len(TASKS)))
    for panel, axis, field, title, xlabel in (
        (0, axes[0], 'peak_rss_mb_p50', 'Peak process-tree RSS', 'MB'),
        (1, axes[1], 'live_tree_cpu_ms_p50', 'Live process-tree CPU', 'ms'),
    ):
        for offset, (arm, label, color) in zip(offsets, ARMS, strict=True):
            values = []
            texts = []
            for task in TASKS:
                group = groups.get((task, arm))
                value = group.get(field) if group and group.get('ok') else None
                values.append(float(value) if value is not None else 0.0)
                texts.append(f'{value:.1f}' if value is not None else 'n/a')
            y = [position + offset * bar_height for position in positions]
            bars = axis.barh(y, values, height=bar_height, color=color, label=label, zorder=3)
            for bar, value, text in zip(bars, values, texts, strict=True):
                axis.text(
                    value + (1.5 if panel == 0 else 1.0),
                    bar.get_y() + bar.get_height() / 2,
                    text,
                    va='center',
                    ha='left',
                    fontsize=9,
                    color=TEXT,
                    zorder=4,
                )
        axis.set_yticks(positions, [task.replace('_', ' ') for task in TASKS], fontsize=11)
        axis.invert_yaxis()
        axis.set_xlabel(xlabel, fontsize=11)
        axis.set_title(title, fontsize=14, color=TEXT, pad=12)
        axis.grid(axis='x', color=GRID, linewidth=1, zorder=0)
        axis.set_axisbelow(True)
        axis.margins(x=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=False,
        fontsize=10.5,
    )
    fig.suptitle('Scraping process-tree resources, p50', fontsize=19, color=TEXT, x=0.06, ha='left', y=0.98)
    fig.text(
        0.06,
        0.93,
        f'N={samples} per arm and task, {warmups} warmup. Shared host. Use RSS and CPU, not sampled latency.',
        fontsize=9.5,
        color=MUTED,
        ha='left',
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.86))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
