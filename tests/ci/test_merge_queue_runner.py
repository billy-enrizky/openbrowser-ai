from __future__ import annotations

import scripts.ci.merge_queue_runner as runner
from scripts.ci.merge_queue_runner import (
    BLOCKED_LABEL,
    FAILED_LABEL,
    check_payload,
    merge_blocked,
    merge_ready,
    queue_items,
    trusted_author,
    wait_for_mergeable,
)


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


# Queue smoke test: this comment exercises the custom merge queue.


def test_merge_ready_accepts_mergeable_states() -> None:
    for state in ("CLEAN", "UNSTABLE", "HAS_HOOKS"):
        assert merge_ready({"mergeable": True, "mergeStateStatus": state})


def test_merge_ready_rejects_pending_or_blocking_states() -> None:
    for state in ("UNKNOWN", "BLOCKED", "BEHIND", "DIRTY", "DRAFT"):
        assert not merge_ready({"mergeable": True, "mergeStateStatus": state})
    assert not merge_ready({"mergeable": None, "mergeStateStatus": "CLEAN"})
    assert not merge_ready({"mergeable": False, "mergeStateStatus": "CLEAN"})


def test_wait_for_mergeable_stops_when_the_check_settles(monkeypatch) -> None:
    states = iter(
        [
            {"mergeable": None, "mergeStateStatus": "UNKNOWN"},
            {"mergeable": True, "mergeStateStatus": "CLEAN"},
        ]
    )
    monkeypatch.setattr(runner, "gh_api", lambda *args: next(states))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    pull = wait_for_mergeable("owner/repo", 12, timeout=10)

    assert pull["mergeStateStatus"] == "CLEAN"
    assert merge_ready(pull)


def test_merge_blocked_preserves_the_successful_gate(monkeypatch) -> None:
    checks = []
    comments = []
    labels = []
    monkeypatch.setattr(runner, "post_check", lambda *args: checks.append(args))
    monkeypatch.setattr(runner, "comment", lambda *args: comments.append(args))
    monkeypatch.setattr(
        runner,
        "set_label",
        lambda repo, number, label, present: labels.append((label, present)),
    )

    merge_blocked("owner/repo", 12, "the base branch policy is unresolved")

    assert checks == []
    assert "the base branch policy is unresolved" in comments[0][2]
    assert (BLOCKED_LABEL, True) in labels
    assert (FAILED_LABEL, False) in labels
