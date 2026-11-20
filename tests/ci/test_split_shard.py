from __future__ import annotations

import pytest

from scripts.ci.split_shard import split_files

NAVIGATION = "tests/test_navigation.py"


def test_parallel_only_partition() -> None:
    parallel, serial = split_files(["tests/test_a.py", "tests/test_b.py"], NAVIGATION)

    assert parallel == ["tests/test_a.py", "tests/test_b.py"]
    assert serial == []


def test_serial_file_is_removed_from_the_parallel_group() -> None:
    parallel, serial = split_files(["tests/test_a.py", NAVIGATION, "tests/test_b.py"], NAVIGATION)

    assert parallel == ["tests/test_a.py", "tests/test_b.py"]
    assert serial == [NAVIGATION]
    assert set(parallel + serial) == {
        "tests/test_a.py",
        NAVIGATION,
        "tests/test_b.py",
    }


def test_serial_only_shard_has_no_parallel_files() -> None:
    parallel, serial = split_files([NAVIGATION], NAVIGATION)

    assert parallel == []
    assert serial == [NAVIGATION]


def test_empty_shard_stays_empty() -> None:
    assert split_files([], NAVIGATION) == ([], [])


def test_duplicate_file_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate test file"):
        split_files(["tests/test_a.py", "tests/test_a.py"], NAVIGATION)
