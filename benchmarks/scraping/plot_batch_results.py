"""Plot batch scraping latency and CLI batch peak RSS for documentation.

Run with matplotlib installed outside the base package:
    uv run --with matplotlib python benchmarks/scraping/plot_batch_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_FIRECRAWL = HERE / 'results-batch-firecrawl-2026-11-24-n10.summary.json'
DEFAULT_CLI = HERE / 'results-cli-http-batch-2026-11-24-n10.summary.json'
DEFAULT_LIGHTPANDA_LATENCY = HERE / 'results-lightpanda-after-n10-interleaved.json'
DEFAULT_LIGHTPANDA_RESOURCE = HERE / 'results-resources-2026-11-24-n3.json'
DEFAULT_OUTPUT = HERE.parents[1] / 'docs' / 'images' / 'native-scraping-batch-current.png'
TEXT = '#171717'
MUTED = '#6b7280'
GRID = '#e5e7eb'


def _lightpanda() -> tuple[float, float]:
    latency = json.loads(DEFAULT_LIGHTPANDA_LATENCY.read_text(encoding='utf-8'))
    resource = json.loads(DEFAULT_LIGHTPANDA_RESOURCE.read_text(encoding='utf-8'))
    latency_ms = next(
        group['latency']['p50_ms']
        for group in latency['groups']
        if group['task'] == 'pagination' and group['arm'] == 'lightpanda'
    )
    rss_mb = next(
        group['peak_rss_mb_p50']
        for group in resource['groups']
        if group['task'] == 'pagination' and group['arm'] == 'lightpanda'
    )
    return float(latency_ms), float(rss_mb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firecrawl', type=Path, default=DEFAULT_FIRECRAWL)
    parser.add_argument('--cli', type=Path, default=DEFAULT_CLI)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    firecrawl = json.loads(args.firecrawl.read_text(encoding='utf-8'))
    cli = json.loads(args.cli.read_text(encoding='utf-8'))
    firecrawl_latency = {
        group['arm']: group['latency']['p50_ms'] for group in firecrawl['groups']
    }
    cli_rows = {row['arm']: row for row in cli['rows']}
    lightpanda_ms, lightpanda_mb = _lightpanda()

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
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    for axis in axes:
        axis.set_facecolor('#ffffff')

    def bars(axis, labels, values, colors, title, ylabel, log=False, fmt='{:.2f}'):
        positions = list(range(len(labels)))
        bars = axis.bar(positions, values, color=colors, width=0.62, zorder=3)
        axis.set_xticks(positions, labels, fontsize=10.5)
        axis.set_title(title, fontsize=13, color=TEXT, pad=12)
        axis.set_ylabel(ylabel, fontsize=10.5)
        axis.grid(axis='y', color=GRID, linewidth=1, zorder=0)
        axis.set_axisbelow(True)
        if log:
            axis.set_yscale('log')
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * (1.12 if log else 1.02),
                fmt.format(value),
                ha='center',
                va='bottom',
                fontsize=10.5,
                color=TEXT,
            )
        axis.margins(y=0.18)

    bars(
        axes[0],
        ['OpenBrowser', 'Firecrawl Local'],
        [firecrawl_latency['openbrowser-http-batch'], firecrawl_latency['firecrawl-local-batch']],
        ['#0f6270', '#c25b2a'],
        'API batch p50 latency',
        'milliseconds, log scale',
        log=True,
    )
    bars(
        axes[1],
        ['OpenBrowser fast', 'Pre-change CLI', 'Lightpanda'],
        [cli_rows['new']['latency_p50_ms'], cli_rows['old']['latency_p50_ms'], lightpanda_ms],
        ['#0f6270', '#9ca3af', '#4f7f5b'],
        'CLI batch p50 latency',
        'milliseconds, log scale',
        log=True,
        fmt='{:.0f}',
    )
    bars(
        axes[2],
        ['OpenBrowser fast', 'Pre-change CLI', 'Lightpanda'],
        [cli_rows['new']['peak_rss_p50_mb'], cli_rows['old']['peak_rss_p50_mb'], lightpanda_mb],
        ['#0f6270', '#9ca3af', '#4f7f5b'],
        'CLI batch peak RSS p50',
        'megabytes',
        fmt='{:.0f}',
    )
    fig.suptitle(
        'Five-page batch scraping: 5 URLs, 100 rows',
        fontsize=18,
        color=TEXT,
        x=0.06,
        ha='left',
        y=0.99,
    )
    fig.text(
        0.06,
        0.92,
        'OpenBrowser API batch and Firecrawl Local use one batch request set. CLI bars use the pre-change and fast-entry binaries; Lightpanda is a directional reference from separate lanes.',
        fontsize=9.5,
        color=MUTED,
        ha='left',
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
