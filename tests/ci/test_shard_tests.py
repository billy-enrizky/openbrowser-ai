"""Tests for the recorded-time shard allocator."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.shard_tests import allocate, load_timings, shard_totals


def test_union_equals_selection_for_every_supported_shard_count() -> None:
    root = Path(__file__).resolve().parents[2]
    files = sorted(str(path.relative_to(root)).replace("\\", "/") for path in (root / "tests").rglob("test_*.py"))
    timings = load_timings(root / "tests" / "ci" / "test_timings.json")
    for shards in range(1, 17):
        bins = allocate(files, timings, shards)
        flat = [file for bin_ in bins for file in bin_]
        assert sorted(flat) == files
        assert len(flat) == len(set(flat))
        assert len(bins) == shards


def test_missing_timing_uses_the_median_not_zero() -> None:
    files = ["a.py", "b.py", "c.py"]
    timings = {"a.py": 10.0, "b.py": 2.0}
    bins = allocate(files, timings, 2)
    totals = shard_totals(bins, timings)
    assert sum(totals) == pytest.approx(18.0)


def test_allocation_is_deterministic() -> None:
    files = ["a.py", "b.py", "c.py", "d.py"]
    timings = {"a.py": 4.0, "b.py": 3.0, "c.py": 2.0, "d.py": 1.0}
    assert allocate(files, timings, 2) == allocate(files, timings, 2)


def test_duplicate_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        allocate(["a.py", "a.py"], {}, 2)


def test_zero_shards_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        allocate(["a.py"], {}, 0)


def test_balance_is_reasonable_on_recorded_data() -> None:
    root = Path(__file__).resolve().parents[2]
    files = sorted(str(path.relative_to(root)).replace("\\", "/") for path in (root / "tests").rglob("test_*.py"))
    timings = load_timings(root / "tests" / "ci" / "test_timings.json")
    totals = shard_totals(allocate(files, timings, 4), timings)
    assert max(totals) - min(totals) <= max(1.0, max(totals) * 0.15)


def test_empty_selection_produces_empty_shards() -> None:
    bins = allocate([], {}, 4)
    assert bins == [[], [], [], []]


def test_one_file_fewer_than_shards() -> None:
    bins = allocate(["a.py"], {"a.py": 1.0}, 4)
    flat = [file for bin_ in bins for file in bin_]
    assert flat == ["a.py"]
    assert len(bins) == 4


def test_all_missing_timings_use_the_fallback() -> None:
    bins = allocate(["a.py", "b.py"], {}, 2)
    assert bins == [["a.py"], ["b.py"]]
    assert shard_totals(bins, {}) == [1.0, 1.0]


def test_ties_break_by_path() -> None:
    bins = allocate(["b.py", "a.py"], {"a.py": 1.0, "b.py": 1.0}, 2)
    assert bins == [["a.py"], ["b.py"]]


def test_allocation_is_order_independent() -> None:
    files = ["a.py", "b.py", "c.py", "d.py"]
    timings = {"a.py": 4.0, "b.py": 3.0, "c.py": 2.0, "d.py": 1.0}
    assert allocate(files, timings, 2) == allocate(list(reversed(files)), timings, 2)


def test_non_finite_timing_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "timings.json"
    path.write_text('{"a.py": Infinity}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not finite"):
        load_timings(path)
