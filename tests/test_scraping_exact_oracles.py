"""Exact-oracle regressions for the local scraping benchmark."""

from __future__ import annotations

import pytest

from benchmarks.scraping.scrape_compare import (
    EXPECTED_JS_SKUS,
    EXPECTED_PAGINATION_IDS,
    EXPECTED_TABLE_ROWS,
    BrowserArm,
    exact_assets,
    exact_js_list,
    exact_pagination,
    exact_static,
    exact_table,
)


def test_js_list_accepts_the_exact_ordered_sequence():
    assert exact_js_list(EXPECTED_JS_SKUS) is True


def test_js_list_rejects_a_first_and_last_only_match():
    counterexample = ["SKU-000"] + ["WRONG"] * 28 + ["SKU-029"]
    assert exact_js_list(counterexample) is False


def test_js_list_rejects_reordered_skus():
    assert exact_js_list(list(reversed(EXPECTED_JS_SKUS))) is False


def test_pagination_accepts_the_exact_ordered_ids():
    assert exact_pagination(EXPECTED_PAGINATION_IDS) is True


def test_pagination_rejects_arbitrary_unique_ids():
    counterexample = [f"WRONG-{index}" for index in range(len(EXPECTED_PAGINATION_IDS))]
    assert exact_pagination(counterexample) is False


def test_pagination_rejects_reordered_ids():
    assert exact_pagination(list(reversed(EXPECTED_PAGINATION_IDS))) is False


def test_table_accepts_the_exact_rows():
    assert exact_table(EXPECTED_TABLE_ROWS) is True


def test_table_rejects_six_wrong_cells_in_every_row():
    counterexample = [[f"table-{index}"] + ["WRONG"] * 6 for index in range(100)]
    assert exact_table(counterexample) is False


def test_table_rejects_reordered_rows():
    assert exact_table(list(reversed(EXPECTED_TABLE_ROWS))) is False


def test_table_rejects_a_missing_row():
    assert exact_table(EXPECTED_TABLE_ROWS[:-1]) is False


@pytest.mark.asyncio
async def test_prepare_starts_a_forced_browser(monkeypatch):
    arm = BrowserArm("browser", "full", engine="browser")
    calls = []

    async def fake_start():
        calls.append("start")
        arm._started = True

    monkeypatch.setattr(arm, "_ensure_started", fake_start)
    await arm.prepare("static_article")
    assert calls == ["start"]


@pytest.mark.asyncio
async def test_prepare_starts_a_browser_task_in_auto_mode(monkeypatch):
    arm = BrowserArm("auto-js", "full", engine="auto")
    calls = []

    async def fake_start():
        calls.append("start")
        arm._started = True

    monkeypatch.setattr(arm, "_ensure_started", fake_start)
    await arm.prepare("js_list")
    assert calls == ["start"]


@pytest.mark.asyncio
async def test_prepare_does_not_start_for_an_auto_http_task(monkeypatch):
    arm = BrowserArm("auto-http", "full", engine="auto")
    calls = []

    async def fake_start():
        calls.append("start")

    monkeypatch.setattr(arm, "_ensure_started", fake_start)
    await arm.prepare("pagination")
    assert calls == []


def test_static_accepts_all_ordered_items():
    content = "\n".join(f"Item {index}\nValue {index * 7}" for index in range(50))
    assert exact_static(content) is True


def test_static_rejects_first_and_last_only():
    assert exact_static("Static Bench\nItem 49") is False


def test_static_rejects_a_missing_item():
    content = "\n".join(f"Item {index}" for index in range(49))
    assert exact_static(content) is False


def test_static_rejects_the_decoy():
    content = "\n".join(f"Item {index}" for index in range(50)) + "\nDO NOT INCLUDE"
    assert exact_static(content) is False


def test_assets_accepts_all_ordered_headlines():
    content = "\n".join(f"Headline {index}" for index in range(1, 21))
    assert exact_assets(content) is True


def test_assets_rejects_twenty_copies_of_headline_one():
    assert exact_assets("\n".join(["Headline 1"] * 20)) is False


def test_assets_rejects_reordered_headlines():
    content = "\n".join(f"Headline {index}" for index in reversed(range(1, 21)))
    assert exact_assets(content) is False


def test_static_rejects_all_labels_without_values():
    content = "\n".join(f"Item {index}" for index in range(50))
    assert exact_static(content) is False


def test_static_rejects_a_corrupt_value():
    content = "\n".join(f"Item {index}\nValue {index * 7}" for index in range(50))
    assert exact_static(content.replace("Value 168", "Value 999")) is False


def test_static_rejects_reordered_values():
    content = "\n".join(f"Item {index}\nValue {(49 - index) * 7}" for index in range(50))
    assert exact_static(content) is False


def test_static_rejects_additional_value():
    content = "\n".join(f"Item {index}\nValue {index * 7}" for index in range(50))
    assert exact_static(content + "\nValue 350") is False


@pytest.mark.parametrize("value", [None, False, 0, [], {}])
def test_text_scrape_oracles_reject_non_text(value):
    assert exact_static(value) is False
    assert exact_assets(value) is False
