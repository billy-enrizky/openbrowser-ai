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

servers = ["Playwright\nMCP", "Chrome DevTools\nMCP", "OpenBrowser\nMCP"]
short_names = ["Playwright", "Chrome\nDevTools", "OpenBrowser"]

# Duration (seconds) -- mean +/- std
duration_mean = np.array([92.2, 128.8, 103.1])
duration_std = np.array([11.4, 6.2, 16.4])

# Response tokens (estimated from MCP tool response chars / 4)
response_tokens = np.array([283_853, 301_030, 1_665])

# Colors -- muted, professional palette
COLORS = {
    "playwright": "#6366f1",      # indigo
    "cdp": "#f59e0b",             # amber
    "openbrowser": "#10b981",     # emerald
}
colors = [COLORS["playwright"], COLORS["cdp"], COLORS["openbrowser"]]
edge_colors = ["#4f46e5", "#d97706", "#059669"]

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"wspace": 0.35})
fig.patch.set_facecolor("#fafafa")

for ax in (ax1, ax2):
    ax.set_facecolor("#fafafa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=10)

x = np.arange(len(servers))
bar_width = 0.52

# ---------------------------------------------------------------------------
# Panel 1 -- Duration (seconds) with error bars
# ---------------------------------------------------------------------------

bars1 = ax1.bar(
    x, duration_mean, bar_width,
    yerr=duration_std,
    capsize=6,
    color=colors,
    edgecolor=edge_colors,
    linewidth=1.2,
    error_kw={"elinewidth": 1.5, "capthick": 1.5, "color": "#555555"},
    zorder=3,
)

ax1.set_ylabel("Duration (seconds)", fontsize=12, fontweight="600", color="#333333")
ax1.set_title("Task Completion Time", fontsize=14, fontweight="700", color="#222222", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, fontsize=10, fontweight="500", color="#444444")
ax1.set_ylim(0, 170)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(30))
ax1.grid(axis="y", alpha=0.3, linestyle="--", color="#cccccc", zorder=0)

# Value labels on bars
for bar, mean, std in zip(bars1, duration_mean, duration_std):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 4,
        f"{mean:.1f}s",
        ha="center", va="bottom",
        fontsize=11, fontweight="600", color="#333333",
    )

# Annotation: "N=5 runs, error bars = 1 std"
ax1.text(
    0.98, 0.97, "N=5 runs, bars = 1 std",
    transform=ax1.transAxes, ha="right", va="top",
    fontsize=8, color="#888888", fontstyle="italic",
)

# ---------------------------------------------------------------------------
# Panel 2 -- Response Tokens (log scale)
# ---------------------------------------------------------------------------

bars2 = ax2.bar(
    x, response_tokens, bar_width,
    color=colors,
    edgecolor=edge_colors,
    linewidth=1.2,
    zorder=3,
)

ax2.set_ylabel("Response Tokens", fontsize=12, fontweight="600", color="#333333")
ax2.set_title("Token Usage per Benchmark", fontsize=14, fontweight="700", color="#222222", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(short_names, fontsize=10, fontweight="500", color="#444444")
ax2.set_yscale("log")
ax2.set_ylim(500, 1_000_000)
ax2.grid(axis="y", alpha=0.3, linestyle="--", color="#cccccc", zorder=0)

# Format y-axis with K/M suffixes
def token_formatter(val, pos):
    if val >= 1_000_000:
        return f"{val / 1_000_000:.0f}M"
    if val >= 1_000:
        return f"{val / 1_000:.0f}K"
    return f"{int(val)}"

ax2.yaxis.set_major_formatter(ticker.FuncFormatter(token_formatter))

# Value labels on bars
for bar, tokens in zip(bars2, response_tokens):
    label = f"{tokens:,}"
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.6,
        label,
        ha="center", va="bottom",
        fontsize=10, fontweight="600", color="#333333",
    )

# Highlight the ratio
ax2.annotate(
    "170x fewer",
    xy=(2, response_tokens[2]),
    xytext=(1.15, 15_000),
    fontsize=11, fontweight="700", color=COLORS["openbrowser"],
    arrowprops=dict(
        arrowstyle="->",
        color=COLORS["openbrowser"],
        linewidth=1.8,
        connectionstyle="arc3,rad=-0.2",
    ),
    zorder=5,
)

# Annotation
ax2.text(
    0.98, 0.97, "6 real-world browser tasks",
    transform=ax2.transAxes, ha="right", va="top",
    fontsize=8, color="#888888", fontstyle="italic",
)

# ---------------------------------------------------------------------------
# Suptitle and save
# ---------------------------------------------------------------------------

fig.suptitle(
    "E2E LLM Benchmark: MCP Server Comparison",
    fontsize=16, fontweight="800", color="#111111",
    y=0.99,
)
fig.text(
    0.5, 0.01,
    "Claude Sonnet 4.6 on AWS Bedrock  |  6 tasks  |  5 runs per server  |  10K bootstrap CI",
    ha="center", fontsize=9, color="#999999",
)

fig.subplots_adjust(left=0.08, right=0.97, bottom=0.1, top=0.85, wspace=0.35)

out_path = Path(__file__).parent / "benchmark_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved to {out_path}")
