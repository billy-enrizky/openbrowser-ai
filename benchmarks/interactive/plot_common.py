"""Shared plotting rules for benchmark charts.

The helpers keep four promises:

1. Every required arm appears in the legend, even when a cell is unknown.
2. An unknown value stays unknown. It is drawn as ``n/a`` and never as zero.
3. Bar groups stay inside their task slot, so adjacent tasks cannot overlap.
4. Captions derive N and eligibility from the artifact, not from a constant.
"""

from __future__ import annotations

# pyright: reportMissingImports=false
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
BACKGROUND = "#ffffff"
TEXT = "#171717"
MUTED = "#6b7280"
GRID = "#e5e7eb"
UNKNOWN_LABEL = "n/a"

CANONICAL_INTERACTIVE_ARMS = (
    "openbrowser-mcp-fast",
    "openbrowser-fast",
    "chrome-devtools-mcp",
    "agent-browser",
    "browser-use",
    "playwright-mcp",
    "playwright-cli",
)

SCRAPING_ARM_STYLES = {
    "openbrowser-full": ("OpenBrowser full", "#0f6270"),
    "openbrowser-lightweight": ("OpenBrowser lightweight", "#6db3ab"),
    "openbrowser-no-render": ("OpenBrowser no-render", "#2f7f8a"),
    "playwright-mcp": ("Playwright MCP", "#b25f48"),
    "playwright-cli": ("Playwright CLI", "#6c7787"),
    "firecrawl-local": ("Firecrawl Local", "#c25b2a"),
    "lightpanda": ("Lightpanda", "#8871b4"),
}
SCRAPING_ARMS = tuple(SCRAPING_ARM_STYLES)

INTERACTIVE_TASK_LABELS = {
    "wizard": "Form flow",
    "modal": "Modal filter",
    "download": "Click to download",
    "tabs": "Multiple tabs",
    "auth": "Login and resume",
}


@dataclass(frozen=True)
class ArmStyle:
    name: str
    label: str
    color: str


@dataclass(frozen=True)
class BarLayout:
    height: float
    offsets: tuple[float, ...]
    span: float
    legend_columns: int
    legend_rows: int
    label_size: float


def bar_layout(
    arm_count: int,
    *,
    max_span: float = 0.72,
    preferred_height: float = 0.16,
    max_legend_columns: int = 4,
) -> BarLayout:
    """Return a non-overlapping grouped-bar layout for one task slot."""
    if arm_count < 1:
        raise ValueError("at least one arm is required")
    if not 0 < max_span < 1:
        raise ValueError("max_span must be between zero and one")
    height = min(preferred_height, max_span / arm_count)
    offsets = tuple((index - (arm_count - 1) / 2) * height for index in range(arm_count))
    columns = min(max_legend_columns, arm_count)
    rows = math.ceil(arm_count / columns)
    label_size = 10.5 if arm_count <= 5 else 8.5
    return BarLayout(height, offsets, height * arm_count, columns, rows, label_size)


def finite_number(value: Any) -> float | None:
    """Return a finite float, or None for a missing or invalid value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def positive_number(value: Any) -> float | None:
    number = finite_number(value)
    return number if number is not None and number > 0 else None


def format_ms(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_mib(value: float) -> str:
    return f"{value:.1f}"


def sample_label(runs: Iterable[Any]) -> str:
    values = {int(value) for value in runs if isinstance(value, int) and not isinstance(value, bool) and value > 0}
    if not values:
        return "N not recorded"
    low, high = min(values), max(values)
    return f"N={low}" if low == high else f"N={low}-{high}"


def eligibility_text(eligible: int, total: int) -> str:
    return f"{eligible}/{total}"


def caption(
    runs: Iterable[Any],
    *,
    eligible_cells: int,
    total_cells: int,
    note: str = "",
) -> str:
    text = (
        f"{sample_label(runs)} per measured cell. "
        f"{eligibility_text(eligible_cells, total_cells)} required cells have eligible observations. "
        f"{UNKNOWN_LABEL} means no eligible observation."
    )
    return f"{text} {note}".strip()


def interactive_styles(
    required: Sequence[str] = CANONICAL_INTERACTIVE_ARMS,
    specs: Mapping[str, Any] | None = None,
) -> tuple[ArmStyle, ...]:
    from .arms import ARM_SPECS

    mapping = specs or ARM_SPECS
    styles: list[ArmStyle] = []
    for name in required:
        spec = mapping.get(name)
        if spec is None:
            styles.append(ArmStyle(name, name, "#9ca3af"))
        else:
            styles.append(ArmStyle(name, spec.label, spec.color))
    return tuple(styles)


def scraping_styles(required: Sequence[str] = SCRAPING_ARMS) -> tuple[ArmStyle, ...]:
    styles: list[ArmStyle] = []
    for name in required:
        label, color = SCRAPING_ARM_STYLES.get(name, (name, "#9ca3af"))
        styles.append(ArmStyle(name, label, color))
    return tuple(styles)


def method_line(environment: Mapping[str, Any] | None) -> str:
    """Return the neutral warm-session method line and the app digest."""
    from .arms import WARM_METHOD_LABEL

    environment = environment or {}
    app = environment.get("application_digest")
    if not isinstance(app, str) or not app:
        application = environment.get("application")
        if isinstance(application, Mapping):
            for key in ("actual_digest", "digest", "expected_digest"):
                candidate = application.get(key)
                if isinstance(candidate, str) and candidate:
                    app = candidate
                    break
    if not isinstance(app, str) or not app:
        arms = environment.get("arms")
        if isinstance(arms, Mapping):
            for arm_name in ("openbrowser-mcp-fast", "openbrowser-fast"):
                arm = arms.get(arm_name)
                if isinstance(arm, Mapping):
                    version = arm.get("version")
                    if isinstance(version, str) and version:
                        app = version
                        break
    app_label = app[:8] if isinstance(app, str) and app else "not recorded"
    return f"Method: {WARM_METHOD_LABEL} | OpenBrowser app: {app_label}"


def merge_groups(reports: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict]:
    """Merge scraping groups and reject conflicting duplicates."""
    merged: dict[tuple[str, str], dict] = {}
    for report in reports:
        for group in report.get("groups", []):
            key = (str(group.get("task")), str(group.get("arm")))
            if key in merged and merged[key] != group:
                raise ValueError(f"conflicting benchmark group for {key}")
            merged[key] = dict(group)
    return merged


def latency_values(
    groups: Mapping[tuple[str, str], Mapping[str, Any]],
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
    *,
    field: str = "p50_ms",
    nested: str = "latency",
) -> dict[tuple[str, str], float | None]:
    values: dict[tuple[str, str], float | None] = {}
    for task in tasks:
        for style in styles:
            group = groups.get((task, style.name), {})
            container = group.get(nested, {}) if nested else group
            values[(task, style.name)] = positive_number(container.get(field))
    return values


def summary_latency_values(
    rows: Iterable[Mapping[str, Any]],
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
    *,
    field: str = "p50_ms",
) -> dict[tuple[str, str], float | None]:
    by_cell = {(row.get("task"), row.get("arm")): row for row in rows}
    return {
        (task, style.name): positive_number(by_cell.get((task, style.name), {}).get(field))
        for task in tasks
        for style in styles
    }


def summary_counts(
    rows: Iterable[Mapping[str, Any]],
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
    *,
    value_key: str = "p50_ms",
) -> tuple[list[int], int, int, int, int]:
    by_cell = {(row.get("task"), row.get("arm")): row for row in rows}
    runs: list[int] = []
    eligible_cells = 0
    ok_samples = 0
    total_samples = 0
    total_cells = len(tasks) * len(styles)
    for task in tasks:
        for style in styles:
            row = by_cell.get((task, style.name), {})
            run_count = row.get("runs")
            if isinstance(run_count, int) and not isinstance(run_count, bool) and run_count > 0:
                runs.append(run_count)
                total_samples += run_count
            ok_count = row.get("ok")
            if isinstance(ok_count, int) and not isinstance(ok_count, bool) and ok_count > 0:
                ok_samples += ok_count
            if positive_number(row.get(value_key)) is not None:
                eligible_cells += 1
    return runs, eligible_cells, total_cells, ok_samples, total_samples


def resource_values(
    rows: Iterable[Mapping[str, Any]],
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
    *,
    field: str,
) -> dict[tuple[str, str], float | None]:
    by_cell = {(row.get("task"), row.get("arm")): row for row in rows}
    values: dict[tuple[str, str], float | None] = {}
    for task in tasks:
        for style in styles:
            row = by_cell.get((task, style.name), {})
            metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
            metric = metrics.get(field) if isinstance(metrics, Mapping) else {}
            median = metric.get("median") if isinstance(metric, Mapping) else None
            values[(task, style.name)] = positive_number(median)
    return values


def resource_counts(
    rows: Iterable[Mapping[str, Any]],
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
    *,
    field: str,
) -> tuple[list[int], int, int]:
    by_cell = {(row.get("task"), row.get("arm")): row for row in rows}
    runs: list[int] = []
    eligible_cells = 0
    total_cells = len(tasks) * len(styles)
    for task in tasks:
        for style in styles:
            row = by_cell.get((task, style.name), {})
            run_count = row.get("runs")
            if isinstance(run_count, int) and not isinstance(run_count, bool) and run_count > 0:
                runs.append(run_count)
            metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
            metric = metrics.get(field) if isinstance(metrics, Mapping) else {}
            median = metric.get("median") if isinstance(metric, Mapping) else None
            if positive_number(median) is not None:
                eligible_cells += 1
    return runs, eligible_cells, total_cells


def group_counts(
    groups: Mapping[tuple[str, str], Mapping[str, Any]],
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
    *,
    metric: str = "latency",
    field: str = "p50_ms",
) -> tuple[list[int], int, int, int, int]:
    runs: list[int] = []
    eligible_cells = 0
    ok_samples = 0
    total_samples = 0
    total_cells = len(tasks) * len(styles)
    for task in tasks:
        for style in styles:
            group = groups.get((task, style.name), {})
            run_count = group.get("runs")
            if isinstance(run_count, int) and not isinstance(run_count, bool) and run_count > 0:
                runs.append(run_count)
                total_samples += run_count
            successes = group.get("successes")
            if isinstance(successes, int) and not isinstance(successes, bool) and successes > 0:
                ok_samples += successes
            container_value = group.get(metric)
            container = container_value if isinstance(container_value, Mapping) else {}
            if positive_number(container.get(field)) is not None:
                eligible_cells += 1
    return runs, eligible_cells, total_cells, ok_samples, total_samples


def _warm_row_eligible(row: Mapping[str, Any]) -> bool:
    gates = ("ok", "adapter_ok", "oracle_ok", "keychain_verified")
    if not all(row.get(gate) is True for gate in gates):
        return False
    if row.get("resource_ok") is not True:
        return False
    if row.get("warm_fatal_read_errors") != 0:
        return False
    if row.get("warm_unverified_processes") != 0:
        return False
    observed_browser = row.get("warm_observed_browser_during_native_run")
    if observed_browser is not None:
        if isinstance(observed_browser, bool) or not isinstance(observed_browser, int):
            return False
        if observed_browser <= 0:
            return False
    return True


def warm_resource_rows(
    records: Iterable[Mapping[str, Any]],
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
    *,
    metrics: Mapping[str, str],
) -> list[dict]:
    """Summarize warm resource samples into chart rows without hiding unknowns."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("record_type") not in (None, "sample"):
            continue
        task = record.get("task")
        arm = record.get("arm")
        if isinstance(task, str) and isinstance(arm, str):
            grouped.setdefault((task, arm), []).append(record)
    rows: list[dict] = []
    for task in tasks:
        for style in styles:
            items = grouped.get((task, style.name), [])
            eligible = [item for item in items if _warm_row_eligible(item)]
            metric_rows: dict[str, dict] = {}
            for field, source_key in metrics.items():
                values = [value for item in eligible if (value := positive_number(item.get(source_key))) is not None]
                metric_rows[field] = {
                    "count": len(values),
                    "median": statistics.median(values) if values else None,
                    "max": max(values) if values else None,
                }
            rows.append(
                {
                    "task": task,
                    "arm": style.name,
                    "runs": len(items),
                    "ok": sum(1 for item in items if item.get("ok") is True),
                    "resource_eligible": len(eligible),
                    "metrics": metric_rows,
                }
            )
    return rows


def provenance_text(
    rows: Iterable[Mapping[str, Any]],
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
) -> str:
    """Count verified provenance against every required task and arm cell."""
    required = {(task, style.name) for task in tasks for style in styles}
    present = {(row.get("task"), row.get("arm")): row for row in rows}
    verified = sum(1 for key in required if present.get(key, {}).get("provenance_verified") is True)
    return f"Provenance: {verified}/{len(required)} required cells verified."


def has_metric_rows(rows: Iterable[Mapping[str, Any]]) -> bool:
    """Return True when rows already carry summarized metric medians."""
    materialized = list(rows)
    return bool(materialized) and all(isinstance(row.get("metrics"), Mapping) for row in materialized)


def warm_footer(records: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    footer = None
    for record in records:
        if record.get("record_type") == "end":
            footer = record
    return footer


def apply_rc() -> None:
    import matplotlib.pyplot as plt

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


def style_axes(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=8)


def draw_grouped_bars(
    ax,
    *,
    tasks: Sequence[str],
    styles: Sequence[ArmStyle],
    values: Mapping[tuple[str, str], float | None],
    layout: BarLayout,
    formatter,
    log: bool = False,
) -> list[float]:
    """Draw one horizontal group per task and mark every unknown explicitly."""
    positions = list(range(len(tasks)))
    drawn: list[tuple[float, float]] = []
    for style, offset in zip(styles, layout.offsets, strict=True):
        for position, task in zip(positions, tasks, strict=True):
            value = positive_number(values.get((task, style.name)))
            if value is None:
                continue
            y = position + offset
            ax.barh([y], [value], height=layout.height, color=style.color, zorder=3)
            drawn.append((value, y))

    if drawn:
        maximum = max(value for value, _ in drawn)
        if log:
            minimum = min(value for value, _ in drawn)
            ax.set_xlim(minimum * 0.7, maximum * 1.55)
        else:
            ax.set_xlim(0, maximum * 1.20)
        for value, y in drawn:
            pad = value * 0.06 if log else maximum * 0.012
            ax.text(
                value + pad,
                y,
                formatter(value),
                va="center",
                ha="left",
                fontsize=layout.label_size,
                color=TEXT,
                zorder=4,
            )
    else:
        ax.set_xlim(0, 1)
        ax.text(
            0.5,
            0.5,
            "No eligible observations",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color=MUTED,
        )

    for style, offset in zip(styles, layout.offsets, strict=True):
        for position, task in zip(positions, tasks, strict=True):
            if positive_number(values.get((task, style.name))) is None:
                ax.text(
                    0.012,
                    position + offset,
                    UNKNOWN_LABEL,
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    fontsize=layout.label_size,
                    color=MUTED,
                    zorder=5,
                )
    return [value for value, _ in drawn]


def figure_legend(fig, styles: Sequence[ArmStyle], layout: BarLayout, *, y: float) -> Any:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=style.color, edgecolor="none", label=style.label) for style in styles]
    return fig.legend(
        handles=handles,
        labels=[style.label for style in styles],
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=layout.legend_columns,
        frameon=False,
        fontsize=10.5,
        handlelength=1.2,
        columnspacing=1.4,
    )


def save_figure(fig, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BACKGROUND, bbox_inches="tight")
    return output


def top_margin(layout: BarLayout, *, base: float = 0.80) -> float:
    """Reserve extra headroom when the legend wraps to a second row."""
    return max(0.66, base - 0.055 * (layout.legend_rows - 1))
