"""Exact-oracle regressions for the local scraping benchmark."""

from __future__ import annotations

from benchmarks.scraping.scrape_compare import (
    EXPECTED_JS_SKUS,
    EXPECTED_PAGINATION_IDS,
    EXPECTED_TABLE_ROWS,
    exact_js_list,
    exact_pagination,
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
