"""Scraping charts keep current Playwright cells and dated service references."""

import pytest

pytest.importorskip("matplotlib")

from benchmarks.scraping import plot_results as plot


def test_default_charts_use_exact_current_and_reference_cells(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(plot, "_plot", lambda **values: calls.append(values))
    monkeypatch.setattr("sys.argv", ["plot_results", "--output-dir", str(tmp_path)])

    assert plot.main() == 0
    comprehensive, current, lightpanda = calls
    report = plot._load(plot.DEFAULT_PLAYWRIGHT)
    for group in report["groups"]:
        assert comprehensive["groups"][(group["task"], group["arm"])] == group
        assert current["groups"][(group["task"], group["arm"])] == group
        assert group["runs"] == group["successes"] == 5
    assert len(comprehensive["groups"]) == 35
    assert len(current["groups"]) == 25
    assert len(current["styles"]) == 5
    assert "2026-11-28" in comprehensive["note"]
    assert "2026-11-25" in comprehensive["note"]
    assert "startup" in current["note"]
    for path, arm in ((plot.DEFAULT_FIRECRAWL, "firecrawl-local"), (plot.DEFAULT_LIGHTPANDA, "lightpanda")):
        for group in plot._load(path)["groups"]:
            if group["arm"] == arm:
                assert comprehensive["groups"][(group["task"], arm)] == group
                assert group["runs"] == 10
    assert len(lightpanda["groups"]) == 10


def test_current_chart_retains_all_five_arms_and_tasks(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(plot, "_plot", lambda **values: calls.append(values))
    monkeypatch.setattr("sys.argv", ["plot_results", "--output-dir", str(tmp_path)])
    plot.main()
    current = calls[1]
    assert set(current["groups"]) == {(task, arm.name) for task in plot.TASKS for arm in current["styles"]}
    assert current["output"].name == "native-scraping-playwright-2026-11-28-p50.png"
