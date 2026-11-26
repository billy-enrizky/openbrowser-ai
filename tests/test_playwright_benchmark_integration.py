"""Native Playwright paths share the benchmark task and publication registry."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


EXPECTED_CANONICAL = (
    "openbrowser-mcp-fast", "openbrowser-fast", "chrome-devtools-mcp",
    "agent-browser", "browser-use", "playwright-mcp", "playwright-cli",
)


def test_canonical_registry_has_seven_paths_and_four_other_frameworks():
    registry = importlib.import_module("benchmarks.interactive.arms")
    assert registry.CANONICAL_ARMS == EXPECTED_CANONICAL
    assert set(registry.ARM_SPECS) == set(EXPECTED_CANONICAL) | {"openbrowser", "openbrowser-mcp"}
    families = {registry.ARM_SPECS[name].family for name in registry.CANONICAL_ARMS}
    assert families - {"openbrowser"} == {"chrome-devtools", "agent-browser", "browser-use", "playwright"}
    assert registry.ARM_SPECS["playwright-cli"].family == registry.ARM_SPECS["playwright-mcp"].family
    assert len({registry.ARM_SPECS[name].label for name in registry.CANONICAL_ARMS}) == 7


@pytest.mark.parametrize("arm, module_name, class_name", [
    ("playwright-cli", "playwright_cli", "PlaywrightCliArm"),
    ("playwright-mcp", "playwright_mcp", "PlaywrightMcpArm"),
])
def test_factory_uses_the_official_native_adapter(monkeypatch, arm, module_name, class_name):
    from benchmarks.interactive.run_benchmark import make_arm

    class NativeAdapter:
        pass

    name = f"benchmarks.interactive.adapters.{module_name}"
    fake = ModuleType(name)
    setattr(fake, class_name, NativeAdapter)
    monkeypatch.setitem(sys.modules, name, fake)
    assert isinstance(make_arm(arm), NativeAdapter)


def test_default_resource_and_warm_cohorts_cover_all_seven_native_paths():
    from benchmarks.interactive import resource_compare, warm_benchmark

    assert tuple(resource_compare.DEFAULT_ARMS) == EXPECTED_CANONICAL
    assert tuple(warm_benchmark.DEFAULT_ARMS) == EXPECTED_CANONICAL
    assert set(warm_benchmark.SUPPORTED_ARMS) == set(EXPECTED_CANONICAL) | {"openbrowser", "openbrowser-mcp"}
    assert warm_benchmark.WARM_TASKS == ("wizard", "modal", "download", "tabs")


def test_warm_summary_keeps_every_playwright_task():
    from benchmarks.interactive.warm_benchmark import WARM_TASKS, summarize

    records = [{"task": task, "arm": arm, "ok": True, "latency_ms": 1.0, "error": ""}
               for task in WARM_TASKS for arm in ("playwright-mcp", "playwright-cli")]
    rows = summarize(records, {})
    assert {(row["task"], row["arm"]) for row in rows} == {(row["task"], row["arm"]) for row in records}
    assert all(row["runs"] == row["ok"] == 1 for row in rows)


def test_native_session_methodology_has_a_framework_neutral_identity():
    registry = importlib.import_module("benchmarks.interactive.arms")
    from benchmarks.interactive.adapters.browser_use import ADAPTER_VERSION

    assert registry.WARM_METHOD_ID == "native-session-readiness-v3"
    assert registry.WARM_METHOD_LABEL == "Warm-session readiness"
    assert ADAPTER_VERSION == registry.WARM_METHOD_ID
    assert all(name not in registry.WARM_METHOD_ID for name in ("browser-use", "playwright", "openbrowser"))


def test_public_methodology_name_describes_the_workflow():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "benchmarks/interactive/README.md", "docs/comparison.mdx"):
        assert "Warm-session readiness" in (root / name).read_text()
