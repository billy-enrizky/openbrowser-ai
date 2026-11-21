"""Tests for the diagnostic-identity baseline ratchet."""

from __future__ import annotations

from scripts.ci.check_baseline import compare, flatten, message_hash, nest


def entry(count: int, message: str) -> dict[str, object]:
    return {"count": count, "message": message}


def test_round_trip() -> None:
    findings = {("a.py", "E501", "hash1"): entry(1, "line too long")}
    assert flatten(nest(findings)) == findings


def test_new_message_in_same_file_and_rule_is_a_regression() -> None:
    current = {
        ("a.py", "E501", "new"): entry(1, "a different message"),
    }
    baseline = {
        ("a.py", "E501", "old"): entry(1, "the old message"),
    }
    regressions, improvements = compare(current, baseline)
    assert len(regressions) == 1
    assert regressions[0][0] == ("a.py", "E501", "new")
    assert len(improvements) == 1
    assert improvements[0][0] == ("a.py", "E501", "old")


def test_identical_diagnostic_sets_are_not_a_regression() -> None:
    findings = {("a.py", "E501", "same"): entry(1, "line too long")}
    regressions, improvements = compare(findings, findings)
    assert regressions == []
    assert improvements == []


def test_message_hash_ignores_line_numbers() -> None:
    assert message_hash("from line 7") == message_hash("from line 8")
    assert message_hash("Line too long (124 > 120)") != message_hash("Line too long (125 > 120)")


def test_message_hash_keeps_literal_content_distinct() -> None:
    assert message_hash("Literal['line 7']") != message_hash("Literal['line 8']")


def test_increased_count_is_a_regression() -> None:
    current = {("a.py", "E501", "same"): entry(2, "line too long")}
    baseline = {("a.py", "E501", "same"): entry(1, "line too long")}
    regressions, _ = compare(current, baseline)
    assert len(regressions) == 1
    assert regressions[0][2] == 1


def test_new_file_is_a_regression() -> None:
    current = {("new.py", "F401", "hash"): entry(1, "unused import")}
    regressions, _ = compare(current, {})
    assert len(regressions) == 1


def test_removed_file_is_an_improvement() -> None:
    baseline = {("old.py", "F401", "hash"): entry(1, "unused import")}
    regressions, improvements = compare({}, baseline)
    assert regressions == []
    assert len(improvements) == 1
