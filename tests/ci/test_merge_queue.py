from __future__ import annotations

from scripts.ci.merge_queue import GATE_CONTEXT, QUEUE_GATE_CONTEXT, owner_type, ruleset_payload, validate_ruleset


def contexts(payload: dict) -> set[str]:
    rule = next(rule for rule in payload["rules"] if rule["type"] == "required_status_checks")
    return {item["context"] for item in rule["parameters"]["required_status_checks"]}


def test_custom_payload_requires_both_gates() -> None:
    payload = ruleset_payload("custom")

    assert validate_ruleset(payload, "custom") == []
    assert contexts(payload) == {GATE_CONTEXT, QUEUE_GATE_CONTEXT}
    assert "merge_queue" not in {rule["type"] for rule in payload["rules"]}


def test_native_payload_contains_merge_queue() -> None:
    payload = ruleset_payload("native")

    assert validate_ruleset(payload, "native") == []
    assert contexts(payload) == {GATE_CONTEXT}
    queue_rule = next(rule for rule in payload["rules"] if rule["type"] == "merge_queue")
    assert queue_rule["parameters"]["grouping_strategy"] == "ALLGREEN"


def test_custom_payload_missing_queue_gate_is_rejected() -> None:
    payload = ruleset_payload("custom")
    rule = next(rule for rule in payload["rules"] if rule["type"] == "required_status_checks")
    rule["parameters"]["required_status_checks"] = [{"context": GATE_CONTEXT}]

    assert any(QUEUE_GATE_CONTEXT in error for error in validate_ruleset(payload, "custom"))


def test_native_payload_missing_merge_queue_is_rejected() -> None:
    payload = ruleset_payload("native")
    payload["rules"] = [rule for rule in payload["rules"] if rule["type"] != "merge_queue"]

    assert "merge_queue rule is missing" in validate_ruleset(payload, "native")


def test_user_owner_is_detected() -> None:
    assert owner_type({"owner": {"type": "User"}}) == "User"
    assert owner_type({"owner": {"type": "Organization"}}) == "Organization"
