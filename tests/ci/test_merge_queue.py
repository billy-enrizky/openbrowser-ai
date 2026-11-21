from __future__ import annotations

from scripts.ci.merge_queue import GATE_CONTEXT, owner_type, ruleset_payload, validate_ruleset


def test_payload_has_merge_queue_and_gate() -> None:
    payload = ruleset_payload()

    assert validate_ruleset(payload) == []
    rule_types = {rule["type"] for rule in payload["rules"]}
    assert {"merge_queue", "required_status_checks"} <= rule_types
    status_rule = next(rule for rule in payload["rules"] if rule["type"] == "required_status_checks")
    contexts = status_rule["parameters"]["required_status_checks"]
    assert {"context": GATE_CONTEXT} in contexts


def test_missing_gate_is_rejected() -> None:
    payload = ruleset_payload()
    status_rule = next(rule for rule in payload["rules"] if rule["type"] == "required_status_checks")
    status_rule["parameters"]["required_status_checks"] = []

    assert any("required status check" in error for error in validate_ruleset(payload))


def test_missing_merge_queue_is_rejected() -> None:
    payload = ruleset_payload()
    payload["rules"] = [rule for rule in payload["rules"] if rule["type"] != "merge_queue"]

    assert "merge_queue rule is missing" in validate_ruleset(payload)


def test_user_owner_is_detected() -> None:
    assert owner_type({"owner": {"type": "User"}}) == "User"
    assert owner_type({"owner": {"type": "Organization"}}) == "Organization"
