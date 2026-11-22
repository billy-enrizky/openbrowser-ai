#!/usr/bin/env python3
"""Apply and verify a merge queue ruleset for main.

The default custom mode works without changing the repository owner. It
requires `ci-gate` plus the `merge-queue-gate` check posted by the serialized
merge queue workflow. Native mode is available if the repository moves to an
organization-owned public repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

RULESET_NAME = "main merge queue and CI gate"
GATE_CONTEXT = "ci-gate"
QUEUE_GATE_CONTEXT = "merge-queue-gate"
MODES = ("custom", "native")


def ruleset_payload(mode: str = "custom") -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    contexts = [GATE_CONTEXT]
    if mode == "custom":
        contexts.append(QUEUE_GATE_CONTEXT)
    rules: list[dict[str, Any]] = [
        {"type": "deletion"},
        {"type": "non_fast_forward"},
        {
            "type": "pull_request",
            "parameters": {
                "dismiss_stale_reviews_on_push": False,
                "require_code_owner_review": False,
                "require_last_push_approval": False,
                "required_approving_review_count": 0,
                "required_review_thread_resolution": False,
                "allowed_merge_methods": ["merge", "squash", "rebase"],
            },
        },
        {
            "type": "required_status_checks",
            "parameters": {
                "do_not_enforce_on_create": True,
                "strict_required_status_checks_policy": False,
                "required_status_checks": [{"context": context} for context in contexts],
            },
        },
    ]
    if mode == "native":
        rules.append(
            {
                "type": "merge_queue",
                "parameters": {
                    "check_response_timeout_minutes": 60,
                    "grouping_strategy": "ALLGREEN",
                    "max_entries_to_build": 5,
                    "max_entries_to_merge": 5,
                    "merge_method": "MERGE",
                    "min_entries_to_merge": 1,
                    "min_entries_to_merge_wait_minutes": 5,
                },
            }
        )
    return {
        "name": RULESET_NAME,
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "rules": rules,
    }


def validate_ruleset(payload: dict[str, Any], mode: str = "custom") -> list[str]:
    if mode not in MODES:
        return [f"unknown mode: {mode}"]
    errors: list[str] = []
    if payload.get("name") != RULESET_NAME:
        errors.append(f"ruleset name must be {RULESET_NAME!r}")
    if payload.get("target") != "branch":
        errors.append("ruleset target must be branch")
    if payload.get("enforcement") != "active":
        errors.append("ruleset must be active")
    include = payload.get("conditions", {}).get("ref_name", {}).get("include", [])
    if "refs/heads/main" not in include:
        errors.append("ruleset must target refs/heads/main")

    rules = {rule.get("type"): rule for rule in payload.get("rules", [])}
    if "required_status_checks" not in rules:
        errors.append("required_status_checks rule is missing")
    else:
        parameters = rules["required_status_checks"].get("parameters", {})
        contexts = {item.get("context") for item in parameters.get("required_status_checks", [])}
        if GATE_CONTEXT not in contexts:
            errors.append(f"required status check {GATE_CONTEXT!r} is missing")
        if mode == "custom" and QUEUE_GATE_CONTEXT not in contexts:
            errors.append(f"required status check {QUEUE_GATE_CONTEXT!r} is missing")
    if mode == "native":
        if "merge_queue" not in rules:
            errors.append("merge_queue rule is missing")
        queue = rules.get("merge_queue", {}).get("parameters", {})
        if queue.get("grouping_strategy") != "ALLGREEN":
            errors.append("merge queue grouping must be ALLGREEN")
        if queue.get("merge_method") != "MERGE":
            errors.append("merge queue method must be MERGE")
    elif "merge_queue" in rules:
        errors.append("custom mode must not contain a native merge_queue rule")
    return errors


def owner_type(repository: dict[str, Any]) -> str:
    return str(repository.get("owner", {}).get("type", "UNKNOWN"))


def gh_api(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    command = ["gh", "api", "--method", method, path]
    if body is not None:
        command.extend(["--input", "-"])
    result = subprocess.run(
        command,
        input=json.dumps(body) if body is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"gh api failed: {result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def find_ruleset(repo: str) -> dict[str, Any] | None:
    rulesets = gh_api("GET", f"/repos/{repo}/rulesets")
    for item in rulesets:
        if item.get("name") == RULESET_NAME:
            return gh_api("GET", f"/repos/{repo}/rulesets/{item['id']}")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="billy-enrizky/openbrowser-ai")
    parser.add_argument("--mode", choices=MODES, default="custom")
    parser.add_argument("--apply", action="store_true", help="create or update the ruleset")
    parser.add_argument("--verify", action="store_true", help="verify the installed ruleset")
    parser.add_argument("--dry-run", action="store_true", help="print the ruleset and stop")
    args = parser.parse_args(argv)

    payload = ruleset_payload(args.mode)
    errors = validate_ruleset(payload, args.mode)
    if errors:
        print("ruleset payload is invalid: " + "; ".join(errors), file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    repository = gh_api("GET", f"/repos/{args.repo}")
    if args.mode == "native" and owner_type(repository) != "Organization":
        print(
            f"{args.repo} is owned by a {owner_type(repository)}. "
            "Native mode requires an organization-owned public repository. "
            "Use the default custom mode to keep the current repository URL.",
            file=sys.stderr,
        )
        return 3

    existing = find_ruleset(args.repo)
    if args.apply:
        if existing:
            installed = gh_api("PUT", f"/repos/{args.repo}/rulesets/{existing['id']}", payload)
        else:
            installed = gh_api("POST", f"/repos/{args.repo}/rulesets", payload)
        print(json.dumps({"id": installed.get("id"), "name": installed.get("name")}, indent=2))
    else:
        installed = existing

    if args.verify:
        if installed is None:
            print("merge queue ruleset is not installed", file=sys.stderr)
            return 1
        errors = validate_ruleset(installed, args.mode)
        if errors:
            print("merge queue ruleset verification failed: " + "; ".join(errors), file=sys.stderr)
            return 1
        print("merge queue ruleset verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
