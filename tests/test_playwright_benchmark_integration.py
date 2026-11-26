"""Native Playwright paths share the benchmark task and publication registry."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

EXPECTED_CANONICAL = (
    "openbrowser-mcp-fast",
    "openbrowser-fast",
    "chrome-devtools-mcp",
    "agent-browser",
    "browser-use",
    "playwright-mcp",
    "playwright-cli",
)


def test_canonical_registry_has_seven_paths_and_four_other_frameworks():
    registry = importlib.import_module("benchmarks.interactive.arms")
    assert registry.CANONICAL_ARMS == EXPECTED_CANONICAL
    assert set(registry.ARM_SPECS) == set(EXPECTED_CANONICAL) | {"openbrowser", "openbrowser-mcp"}
    families = {registry.ARM_SPECS[name].family for name in registry.CANONICAL_ARMS}
    assert families - {"openbrowser"} == {"chrome-devtools", "agent-browser", "browser-use", "playwright"}
    assert registry.ARM_SPECS["playwright-cli"].family == registry.ARM_SPECS["playwright-mcp"].family
    assert len({registry.ARM_SPECS[name].label for name in registry.CANONICAL_ARMS}) == 7


@pytest.mark.parametrize(
    "arm, module_name, class_name",
    [
        ("playwright-cli", "playwright_cli", "PlaywrightCliArm"),
        ("playwright-mcp", "playwright_mcp", "PlaywrightMcpArm"),
    ],
)
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

    records = [
        {"task": task, "arm": arm, "ok": True, "latency_ms": 1.0, "error": ""}
        for task in WARM_TASKS
        for arm in ("playwright-mcp", "playwright-cli")
    ]
    rows = summarize(records, {})
    assert {(row["task"], row["arm"]) for row in rows} == {(row["task"], row["arm"]) for row in records}
    assert all(row["runs"] == row["ok"] == 1 for row in rows)


def test_native_session_methodology_has_a_framework_neutral_identity():
    registry = importlib.import_module("benchmarks.interactive.arms")
    from benchmarks.interactive.adapters.browser_use import ADAPTER_VERSION

    assert registry.WARM_METHOD_ID == "native-session-readiness-v4"
    assert registry.WARM_METHOD_LABEL == "Warm-session readiness"
    assert ADAPTER_VERSION == registry.WARM_METHOD_ID
    assert all(name not in registry.WARM_METHOD_ID for name in ("browser-use", "playwright", "openbrowser"))


def test_public_methodology_name_describes_the_workflow():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "benchmarks/interactive/README.md", "docs/comparison.mdx"):
        assert "Warm-session readiness" in (root / name).read_text()


@pytest.mark.parametrize(
    "stage,error_type",
    [
        ("warmup", ValueError),
        ("warmup", KeyboardInterrupt),
        ("run", ValueError),
        ("run", KeyboardInterrupt),
        ("cleanup", ValueError),
        ("unavailable", FileNotFoundError),
    ],
)
def test_warm_runner_cleans_each_owned_arm_once_on_every_exit(tmp_path, monkeypatch, stage, error_type):
    import socket
    import subprocess
    from types import SimpleNamespace

    from benchmarks.interactive import warm_benchmark as warm
    from benchmarks.interactive.adapters.base import ArmResult

    def forbidden(*args, **kwargs):
        raise AssertionError("fake warm test attempted native I/O")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    calls = []
    roots = {}
    error = error_type("native lifecycle failed")

    class FakeArm:
        binary = ""

        def __init__(self, name):
            self.name = name

        def warmup(self, ctx):
            roots[self.name] = ctx.root
            if self.name == "playwright-cli" and stage in {"warmup", "unavailable"}:
                raise error
            return {"ok": True, "browser_version": "fake"}, 0.0

        def run(self, ctx, *, cleanup=True):
            assert cleanup is False
            calls.append(("run", self.name))
            if stage == "run":
                raise error
            return ArmResult(ok=True, keychain_verified=True)

        def cleanup(self, ctx):
            calls.append(("cleanup", self.name))
            if stage == "cleanup" and self.name == "playwright-cli":
                raise error

    monkeypatch.setattr(warm, "make_arm", FakeArm)
    monkeypatch.setattr(warm, "_loads", lambda: [])
    monkeypatch.setattr(warm, "platform", SimpleNamespace(platform=lambda: "fake", machine=lambda: "fake"))
    fixture = SimpleNamespace(base_url="http://fixture.invalid", state=SimpleNamespace(snapshot=lambda: {}))
    monkeypatch.setattr(warm, "start_fixture", lambda: (fixture, None))
    monkeypatch.setattr(warm, "stop_fixture", lambda *args: calls.append(("fixture", "stopped")))
    monkeypatch.setattr(warm, "ORACLES", {"wizard": lambda *args: SimpleNamespace(ok=True, detail="fake")})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warm_benchmark",
            "--arms",
            "playwright-mcp",
            "playwright-cli",
            "--tasks",
            "wizard",
            "--samples",
            "1",
            "--workdir",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "results.jsonl"),
        ],
    )
    if stage == "unavailable":
        assert warm.main() == 1
        assert calls.index(("cleanup", "playwright-cli")) < calls.index(("run", "playwright-mcp"))
    else:
        with pytest.raises(error_type) as caught:
            warm.main()
        assert caught.value is error
    for name in ("playwright-mcp", "playwright-cli"):
        assert calls.count(("cleanup", name)) == 1
        assert roots[name].exists() is (stage == "cleanup" and name == "playwright-cli")
