from __future__ import annotations

from scripts.ci.merge_queue_runner import check_payload, queue_items, trusted_author


def test_queue_is_sorted_and_excludes_closed_or_non_pr_issues() -> None:
    items = [
        {"number": 7, "state": "open", "pull_request": {}},
        {"number": 2, "state": "open", "pull_request": {}},
        {"number": 4, "state": "open"},
        {"number": 1, "state": "closed", "pull_request": {}},
    ]

    assert [item["number"] for item in queue_items(items)] == [2, 7]


def test_only_trusted_associations_enter_the_queue() -> None:
    assert trusted_author({"author_association": "OWNER"})
    assert trusted_author({"author_association": "COLLABORATOR"})
    assert not trusted_author({"author_association": "CONTRIBUTOR"})
    assert not trusted_author({})


def test_check_payload_is_named_and_completed() -> None:
    payload = check_payload("abc", "success", "all checks passed")

    assert payload["name"] == "merge-queue-gate"
    assert payload["head_sha"] == "abc"
    assert payload["status"] == "completed"
    assert payload["conclusion"] == "success"
