"""Fake-only tests for the native Playwright scraping bridge."""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest

from benchmarks.interactive.adapters.base import ArmResult
from benchmarks.interactive.adapters.playwright_scrape_bridge import PlaywrightScrapeArm, build_scrape_arm
from benchmarks.interactive.adapters.playwright_tasks import task_program
from benchmarks.interactive.fixtures.server import BENCH_PASSWORD
from benchmarks.scraping.scrape_compare import (
    ASSET_COUNT,
    EXPECTED_JS_SKUS,
    EXPECTED_PAGINATION_IDS,
    EXPECTED_TABLE_ROWS,
    STATIC_ROWS,
)


@pytest.fixture(autouse=True)
def _no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("bridge test attempted native I/O")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


STATIC_CONTENT = "\n".join(f"Item {index}" for index in range(STATIC_ROWS))
ASSET_CONTENT = "\n".join(f"Headline {index}" for index in range(1, ASSET_COUNT + 1))


class FakeArm:
    def __init__(
        self,
        reported: dict | None = None,
        *,
        ok: bool = True,
        error: str = "",
        stdout: str = "",
    ) -> None:
        self.reported = reported or {}
        self.ok = ok
        self.error = error
        self.stdout_override = stdout
        self.warmups: list = []
        self.calls: list[tuple] = []
        self.cleanups: list = []

    def warmup(self, ctx):
        self.warmups.append(ctx)
        return {"browser_version": "Test"}, 1.0

    def execute_program(self, ctx, code, *, timeout=None):
        self.calls.append((ctx, code, timeout))
        return ArmResult(
            ok=self.ok,
            stdout=self.stdout_override or json.dumps(self.reported),
            stderr="",
            reported=self.reported if self.ok else {},
            error=self.error,
        )

    def cleanup(self, ctx):
        self.cleanups.append(ctx)


def make_arm(reported: dict, *, workdir: Path | None = None, **kwargs) -> tuple[PlaywrightScrapeArm, FakeArm]:
    fake = FakeArm(reported, **kwargs)
    arm = PlaywrightScrapeArm("playwright-mcp", fake, workdir=workdir)
    return arm, fake


EXACT_CASES = [
    ("run_static", "static_article", {"content": STATIC_CONTENT}, STATIC_CONTENT),
    ("run_js_list", "js_list", {"skus": EXPECTED_JS_SKUS}, json.dumps(EXPECTED_JS_SKUS)),
    ("run_pagination", "pagination", {"ids": EXPECTED_PAGINATION_IDS}, json.dumps(EXPECTED_PAGINATION_IDS)),
    ("run_table", "table", {"rows": EXPECTED_TABLE_ROWS}, json.dumps(EXPECTED_TABLE_ROWS)),
    ("run_assets", "asset_heavy", {"content": ASSET_CONTENT}, ASSET_CONTENT),
]

CORRUPT_CASES = [
    ("run_static", "static_article", {"content": "Static Bench\nItem 49"}),
    ("run_js_list", "js_list", {"skus": ["SKU-000", *["WRONG"] * 28, "SKU-029"]}),
    ("run_pagination", "pagination", {"ids": [f"WRONG-{index}" for index in range(100)]}),
    ("run_table", "table", {"rows": [["WRONG"] * 7 for _ in range(100)]}),
    ("run_assets", "asset_heavy", {"content": "\n".join(["Headline 1"] * 20)}),
]


@pytest.mark.parametrize("method,task,reported,expected", EXACT_CASES)
async def test_exact_payloads_pass(method, task, reported, expected, tmp_path):
    arm, fake = make_arm(reported, workdir=tmp_path)
    await arm.start()
    await arm.prepare(task)
    ok, output, detail = await getattr(arm, method)("http://127.0.0.1:9")
    assert ok is True, detail
    assert output == expected
    assert detail == ""
    assert len(fake.warmups) == 1


@pytest.mark.parametrize("method,task,reported", CORRUPT_CASES)
async def test_corrupt_payloads_are_rejected(method, task, reported, tmp_path):
    arm, _fake = make_arm(reported, workdir=tmp_path)
    await arm.start()
    await arm.prepare(task)
    ok, _output, detail = await getattr(arm, method)("http://127.0.0.1:9")
    assert ok is False
    assert "oracle" in detail


@pytest.mark.parametrize("method,task,_reported,_expected", EXACT_CASES)
async def test_each_method_uses_the_shared_task_program(method, task, _reported, _expected, tmp_path):
    arm, fake = make_arm(_reported, workdir=tmp_path)
    await arm.start()
    await arm.prepare(task)
    await getattr(arm, method)("http://127.0.0.1:9")
    ctx, code, _timeout = fake.calls[0]
    assert code == task_program(task, "http://127.0.0.1:9", ctx.downloads)
    assert ctx.task == task
    assert ctx.base_url == "http://127.0.0.1:9"


async def test_prepare_warms_once_and_stop_cleans_once(tmp_path):
    arm, fake = make_arm({"content": STATIC_CONTENT}, workdir=tmp_path)
    await arm.start()
    await arm.prepare("static_article")
    await arm.prepare("static_article")
    assert len(fake.warmups) == 1
    await arm.run_static("http://127.0.0.1:9")
    await arm.run_static("http://127.0.0.1:9")
    assert len(fake.calls) == 2
    assert fake.cleanups == []
    await arm.stop()
    await arm.stop()
    assert len(fake.cleanups) == 1


async def test_task_before_prepare_raises(tmp_path):
    arm, _fake = make_arm({"content": STATIC_CONTENT}, workdir=tmp_path)
    await arm.start()
    with pytest.raises(RuntimeError):
        await arm.run_static("http://127.0.0.1:9")


async def test_native_error_is_returned(tmp_path):
    arm, _fake = make_arm({}, workdir=tmp_path, ok=False, error="native boom")
    await arm.start()
    await arm.prepare("static_article")
    ok, _output, detail = await arm.run_static("http://127.0.0.1:9")
    assert ok is False
    assert "native boom" in detail


async def test_missing_result_key_is_rejected(tmp_path):
    arm, _fake = make_arm({}, workdir=tmp_path)
    await arm.start()
    await arm.prepare("static_article")
    ok, _output, detail = await arm.run_static("http://127.0.0.1:9")
    assert ok is False
    assert "missing content" in detail


async def test_password_leak_is_rejected(tmp_path):
    arm, _fake = make_arm({"content": STATIC_CONTENT}, workdir=tmp_path, stdout=BENCH_PASSWORD)
    await arm.start()
    await arm.prepare("static_article")
    ok, _output, detail = await arm.run_static("http://127.0.0.1:9")
    assert ok is False
    assert "leaked" in detail


def test_unknown_factory_name_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_scrape_arm("not-a-playwright-arm", workdir=tmp_path)
