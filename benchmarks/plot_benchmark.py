"""Generate benchmark comparison chart for OpenBrowser vs Playwright vs Chrome DevTools MCP."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Data (from authoritative stats JSON files, N=5 runs each)
# ---------------------------------------------------------------------------

labels = ["Playwright MCP", "Chrome DevTools MCP", "OpenBrowser MCP"]

# Duration (seconds) -- mean +/- std (from docs/comparison.mdx E2E benchmark)
duration_mean = np.array([62.7, 103.4, 77.0])
duration_std = np.array([4.8, 2.7, 6.7])

# Response tokens (estimated from MCP tool response chars / 4)
# Playwright: 1,132,173/4 = 283,043 | Chrome DevTools: 1,147,244/4 = 286,811 | OpenBrowser: 7,853/4 = 1,963
response_tokens = np.array([283_043, 286_811, 1_963])

# Colors
COLORS = ["#818cf8", "#fbbf24", "#34d399"]       # indigo-400, amber-400, emerald-400
EDGE   = ["#6366f1", "#f59e0b", "#10b981"]       # indigo-500, amber-500, emerald-500

BG     = "#0f1117"
GRID   = "#2a2d37"
TEXT   = "#e2e4ea"
MUTED  = "#8b8fa3"

# ---------------------------------------------------------------------------
# Single scatter: Response Tokens (x) vs Duration (y)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Spines
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=10)

# Grid
ax.grid(True, linestyle=":", linewidth=0.6, color=GRID, alpha=0.8)

# Scatter
for i, (tok, dur, label) in enumerate(zip(response_tokens, duration_mean, labels)):
    ax.scatter(
        tok, dur,
        s=420,
        color=COLORS[i],
        edgecolors=EDGE[i],
        linewidths=1.8,
        zorder=5,
    )
    # Error bar (vertical, 1 std)
    ax.errorbar(
        tok, dur,
        yerr=duration_std[i],
        fmt="none",
        ecolor=COLORS[i],
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        alpha=0.7,
        zorder=4,
    )

# Labels -- position relative to dots using axis transform for log-scale x
label_cfg = [
    # Playwright -- above-left
    {"ha": "right", "va": "bottom",
     "xytext": (response_tokens[0] * 0.85, duration_mean[0] + 9)},
    # Chrome DevTools -- above-right
    {"ha": "left", "va": "bottom",
     "xytext": (response_tokens[1] * 1.15, duration_mean[1] + 9)},
    # OpenBrowser -- below
    {"ha": "center", "va": "top",
     "xytext": (response_tokens[2], duration_mean[2] - 10)},
]

for i, (tok, dur, label) in enumerate(zip(response_tokens, duration_mean, labels)):
    tok_str = f"{tok:,} tokens"
    cfg = label_cfg[i]
    ax.annotate(
        f"{label}\n{dur:.1f}s  |  {tok_str}",
        xy=(tok, dur),
        xytext=cfg["xytext"],
        fontsize=9.5,
        fontweight="600",
        color=COLORS[i],
        ha=cfg["ha"],
        va=cfg["va"],
        zorder=6,
    )

# "170x fewer tokens" callout arrow from mid-chart to OpenBrowser dot
ax.annotate(
    "144x fewer response tokens",
    xy=(response_tokens[2] * 1.5, duration_mean[2] + 2),
    xytext=(15_000, 120),
    fontsize=13,
    fontweight="800",
    color="#34d399",
    ha="center",
    arrowprops=dict(
        arrowstyle="-|>",
        color="#34d399",
        linewidth=2.2,
        connectionstyle="arc3,rad=-0.15",
        mutation_scale=15,
    ),
    zorder=7,
)

# Axes
ax.set_xlabel("Response Tokens (log scale)", fontsize=12, fontweight="600",
              color=TEXT, labelpad=10)
ax.set_ylabel("Duration (seconds)", fontsize=12, fontweight="600",
              color=TEXT, labelpad=10)
ax.set_xscale("log")
ax.set_xlim(800, 600_000)
ax.set_ylim(40, 130)

# X-axis formatter
def token_fmt(val, pos):
    if val >= 1_000_000:
        return f"{val / 1_000_000:.0f}M"
    if val >= 1_000:
        return f"{val / 1_000:.0f}K"
    return f"{int(val)}"

ax.xaxis.set_major_formatter(ticker.FuncFormatter(token_fmt))
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

# Title
ax.set_title(
    "MCP Server Benchmark: Duration vs Token Usage",
    fontsize=15, fontweight="800", color=TEXT, pad=18,
)

# Subtitle -- inside plot area to avoid overlap with x-axis label
ax.text(
    0.98, 0.02,
    "Claude Sonnet 4.6  |  6 browser tasks  |  N=5 runs  |  bars = 1 std",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=8.5, color=MUTED, fontstyle="italic",
)

# Ideal zone
ax.annotate(
    "ideal",
    xy=(1_200, 46),
    fontsize=10, fontstyle="italic", fontweight="500",
    color="#4ade80", alpha=0.5,
    ha="center",
)
ax.annotate(
    "",
    xy=(900, 42),
    xytext=(1_800, 42),
    arrowprops=dict(arrowstyle="<-", color="#4ade80", linewidth=1.2, alpha=0.4),
)
ax.annotate(
    "",
    xy=(900, 42),
    xytext=(900, 50),
    arrowprops=dict(arrowstyle="<-", color="#4ade80", linewidth=1.2, alpha=0.4),
)

fig.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.90)

out_path = Path(__file__).parent / "benchmark_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

docs_path = Path(__file__).parent.parent / "docs" / "images" / "benchmark_comparison.png"
fig.savefig(docs_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

landing_path = Path(__file__).parent.parent / "landing" / "public" / "benchmark_comparison.png"
fig.savefig(landing_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

print(f"Saved to {out_path}")
print(f"Saved to {docs_path}")
print(f"Saved to {landing_path}")
