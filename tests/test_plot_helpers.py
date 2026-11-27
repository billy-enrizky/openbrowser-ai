"""Pure tests for honest benchmark plotting rules."""

from __future__ import annotations

import itertools
import math

import pytest

from benchmarks.interactive import plot_common as plot


def test_bar_layout_keeps_seven_arms_inside_one_task_slot():
    layout = plot.bar_layout(7)
    assert layout.span <= 0.72 + 1e-9
    assert layout.height < 0.16
    assert layout.legend_columns == 4
    assert layout.legend_rows == 2
    assert layout.label_size == 8.5
    ordered = sorted(layout.offsets)
    assert all(right - left >= layout.height - 1e-9 for left, right in itertools.pairwise(ordered))


def test_bar_layout_scales_for_two_and_five_arms():
    two = plot.bar_layout(2)
    assert two.height == pytest.approx(0.16)
    assert two.span == pytest.approx(0.32)
    assert two.legend_columns == 2
    assert two.legend_rows == 1

    five = plot.bar_layout(5)
    assert five.span <= 0.72 + 1e-9
    assert five.legend_columns == 4
    assert five.legend_rows == 2


def test_memory_labels_keep_one_decimal():
    assert plot.format_mib(1205.04) == "1205.0"
    assert plot.format_mib(132.94) == "132.9"
    assert plot.format_mib(9.94) == "9.9"


def test_invalid_values_stay_unknown():
    for value in (None, math.nan, math.inf, -math.inf, True, "12", object()):
        assert plot.finite_number(value) is None
        assert plot.positive_number(value) is None
    assert plot.finite_number(0) == 0.0
    assert plot.positive_number(0) is None
    assert plot.positive_number(2.5) == 2.5


def test_caption_derives_n_and_eligibility():
    assert plot.sample_label([5, 5, 5]) == "N=5"
    assert plot.sample_label([4, 5]) == "N=4-5"
    assert plot.sample_label([]) == "N not recorded"
    text = plot.caption([5, 5], eligible_cells=20, total_cells=35)
    assert "N=5" in text
    assert "per measured cell" in text
    assert "20/35 required cells" in text
    assert "n/a means no eligible observation" in text


def test_interactive_styles_include_every_canonical_path():
    styles = plot.interactive_styles()
    names = tuple(style.name for style in styles)
    assert names == plot.CANONICAL_INTERACTIVE_ARMS
    labels = {style.label for style in styles}
    assert {"OpenBrowser MCP", "OpenBrowser CLI", "Playwright MCP", "Playwright CLI"} <= labels
    assert len(styles) == 7


def test_scraping_styles_include_services_and_playwright():
    styles = plot.scraping_styles()
    names = tuple(style.name for style in styles)
    assert "firecrawl-local" in names
    assert "lightpanda" in names
    assert "playwright-mcp" in names
    assert "playwright-cli" in names
    labels = {style.label for style in styles}
    assert {"Firecrawl Local", "Lightpanda", "Playwright MCP", "Playwright CLI"} <= labels


def test_method_line_is_neutral_and_ignores_vendor_fields():
    environment = {"adapter_version": "Vendor-Method-v9", "application_digest": "abc12345deadbeef"}
    line = plot.method_line(environment)
    assert "Warm-session readiness" in line
    assert "OpenBrowser app: abc12345" in line
    assert "Vendor-Method-v9" not in line


def test_method_line_falls_back_to_openbrowser_arm_version():
    environment = {"arms": {"openbrowser-mcp-fast": {"version": "0.1.61.dev173+g9f630f7c6"}}}
    line = plot.method_line(environment)
    assert "OpenBrowser app: 0.1.61.d" in line
    assert "not recorded" not in line


def test_merge_groups_rejects_conflicting_cells():
    first = {"groups": [{"task": "wizard", "arm": "openbrowser-fast", "latency": {"p50_ms": 1.0}}]}
    second = {"groups": [{"task": "modal", "arm": "playwright-mcp", "latency": {"p50_ms": 2.0}}]}
    merged = plot.merge_groups([first, second])
    assert set(merged) == {("wizard", "openbrowser-fast"), ("modal", "playwright-mcp")}

    conflict = {"groups": [{"task": "wizard", "arm": "openbrowser-fast", "latency": {"p50_ms": 3.0}}]}
    with pytest.raises(ValueError):
        plot.merge_groups([first, conflict])


def test_latency_values_keep_missing_and_invalid_as_none():
    groups = {
        ("wizard", "openbrowser-fast"): {"latency": {"p50_ms": 12.5}},
        ("wizard", "playwright-mcp"): {"latency": {"p50_ms": 0}},
        ("wizard", "playwright-cli"): {"latency": {"p50_ms": math.nan}},
    }
    styles = plot.interactive_styles(("openbrowser-fast", "playwright-mcp", "playwright-cli", "browser-use"))
    values = plot.latency_values(groups, ("wizard",), styles)
    assert values[("wizard", "openbrowser-fast")] == 12.5
    assert values[("wizard", "playwright-mcp")] is None
    assert values[("wizard", "playwright-cli")] is None
    assert values[("wizard", "browser-use")] is None


def test_summary_counts_include_required_missing_cells():
    rows = [
        {"task": "wizard", "arm": "openbrowser-fast", "runs": 5, "ok": 5, "p50_ms": 12.0},
        {"task": "wizard", "arm": "playwright-mcp", "runs": 5, "ok": 4, "p50_ms": None},
    ]
    styles = plot.interactive_styles(("openbrowser-fast", "playwright-mcp", "playwright-cli"))
    runs, eligible, total, ok_samples, total_samples = plot.summary_counts(rows, ("wizard",), styles)
    assert runs == [5, 5]
    assert eligible == 1
    assert total == 3
    assert ok_samples == 9
    assert total_samples == 10


def test_resource_values_and_counts_keep_unknown_medians():
    rows = [
        {
            "task": "wizard",
            "arm": "openbrowser-fast",
            "runs": 5,
            "metrics": {"peak_tree_rss_bytes": {"median": 104857600, "count": 5}},
        },
        {
            "task": "wizard",
            "arm": "playwright-mcp",
            "runs": 5,
            "metrics": {"peak_tree_rss_bytes": {"median": None, "count": 0}},
        },
    ]
    styles = plot.interactive_styles(("openbrowser-fast", "playwright-mcp", "playwright-cli"))
    values = plot.resource_values(rows, ("wizard",), styles, field="peak_tree_rss_bytes")
    assert values[("wizard", "openbrowser-fast")] == 104857600
    assert values[("wizard", "playwright-mcp")] is None
    assert values[("wizard", "playwright-cli")] is None
    runs, eligible, total = plot.resource_counts(rows, ("wizard",), styles, field="peak_tree_rss_bytes")
    assert runs == [5, 5]
    assert eligible == 1
    assert total == 3


def test_group_counts_include_missing_required_arms():
    groups = {
        ("wizard", "openbrowser-fast"): {"runs": 7, "successes": 7, "latency": {"p50_ms": 1.0}},
        ("wizard", "playwright-mcp"): {"runs": 7, "successes": 6, "latency": {"p50_ms": None}},
    }
    styles = plot.interactive_styles(("openbrowser-fast", "playwright-mcp", "playwright-cli"))
    runs, eligible, total, ok_samples, total_samples = plot.group_counts(groups, ("wizard",), styles)
    assert runs == [7, 7]
    assert eligible == 1
    assert total == 3
    assert ok_samples == 13
    assert total_samples == 14


def test_warm_resource_rows_exclude_ineligible_observations():
    records = [
        {
            "record_type": "sample",
            "task": "wizard",
            "arm": "openbrowser-fast",
            "ok": True,
            "adapter_ok": True,
            "oracle_ok": True,
            "keychain_verified": True,
            "resource_ok": True,
            "warm_fatal_read_errors": 0,
            "warm_unverified_processes": 0,
            "warm_peak_tree_rss_bytes": 100,
            "warm_peak_child_footprint_bytes": 50,
        },
        {
            "record_type": "sample",
            "task": "wizard",
            "arm": "openbrowser-fast",
            "ok": True,
            "adapter_ok": True,
            "oracle_ok": True,
            "keychain_verified": True,
            "resource_ok": True,
            "warm_fatal_read_errors": 1,
            "warm_unverified_processes": 0,
            "warm_peak_tree_rss_bytes": 900,
            "warm_peak_child_footprint_bytes": 800,
        },
        {
            "record_type": "sample",
            "task": "wizard",
            "arm": "playwright-mcp",
            "ok": True,
            "adapter_ok": True,
            "oracle_ok": True,
            "keychain_verified": True,
            "resource_ok": True,
            "warm_fatal_read_errors": 0,
            "warm_unverified_processes": 0,
            "warm_peak_tree_rss_bytes": 200,
            "warm_peak_child_footprint_bytes": None,
        },
    ]
    styles = plot.interactive_styles(("openbrowser-fast", "playwright-mcp", "playwright-cli"))
    rows = plot.warm_resource_rows(
        records,
        ("wizard",),
        styles,
        metrics={
            "peak_tree_rss_bytes": "warm_peak_tree_rss_bytes",
            "peak_child_footprint_bytes": "warm_peak_child_footprint_bytes",
        },
    )
    by_cell = {(row["task"], row["arm"]): row for row in rows}
    ob = by_cell[("wizard", "openbrowser-fast")]
    assert ob["runs"] == 2
    assert ob["metrics"]["peak_tree_rss_bytes"]["median"] == 100
    assert ob["metrics"]["peak_tree_rss_bytes"]["count"] == 1
    pw = by_cell[("wizard", "playwright-mcp")]
    assert pw["metrics"]["peak_tree_rss_bytes"]["median"] == 200
    assert pw["metrics"]["peak_child_footprint_bytes"]["median"] is None
    missing = by_cell[("wizard", "playwright-cli")]
    assert missing["runs"] == 0
    assert missing["metrics"]["peak_tree_rss_bytes"]["median"] is None


def test_provenance_denominator_covers_required_missing_arms():
    rows = [{"task": "wizard", "arm": arm, "provenance_verified": True} for arm in plot.CANONICAL_INTERACTIVE_ARMS[:5]]
    styles = plot.interactive_styles()
    text = plot.provenance_text(rows, ("wizard", "modal"), styles)
    assert text == "Provenance: 5/14 required cells verified."
    assert "5/5" not in text


def test_has_metric_rows_detects_summary_and_raw_inputs():
    summary = [{"task": "wizard", "arm": "a", "metrics": {"peak_tree_rss_bytes": {"median": 1}}}]
    raw = [
        {"record_type": "run", "metrics": None},
        {"record_type": "sample", "task": "wizard", "arm": "a", "warm_peak_tree_rss_bytes": 1},
    ]
    assert plot.has_metric_rows(summary) is True
    assert plot.has_metric_rows(raw) is False
    assert plot.has_metric_rows([]) is False


def test_warm_footer_returns_terminal_record():
    records = [
        {"record_type": "sample", "task": "wizard"},
        {"record_type": "end", "valid": True, "failures": 0},
    ]
    footer = plot.warm_footer(records)
    assert footer is not None
    assert footer["valid"] is True
    assert plot.warm_footer([{"record_type": "sample"}]) is None


def test_interactive_task_labels_are_reader_friendly():
    assert plot.INTERACTIVE_TASK_LABELS["wizard"] == "Form flow"
    assert plot.INTERACTIVE_TASK_LABELS["auth"] == "Login and resume"
    assert len(plot.INTERACTIVE_TASK_LABELS) == 5
