#!/usr/bin/env python3
"""Apply and verify the native GitHub merge queue ruleset.

GitHub only exposes native merge queues for organization-owned public
repositories. Organization creation itself is a web UI action. Run this script
after the repository has been transferred to a free organization.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

RULESET_NAME = "main merge queue and CI gate"
GATE_CONTEXT = "ci-gate"


def ruleset_payload() -> dict[str, Any]:
    return {
        "name": RULESET_NAME,
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "rules": [
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
                    "required_status_checks": [{"context": GATE_CONTEXT}],
                },
            },
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
            },
        ],
    }


def validate_ruleset(payload: dict[str, Any]) -> list[str]:
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
    if "merge_queue" not in rules:
        errors.append("merge_queue rule is missing")
    if "required_status_checks" not in rules:
        errors.append("required_status_checks rule is missing")
    else:
        parameters = rules["required_status_checks"].get("parameters", {})
        contexts = {item.get("context") for item in parameters.get("required_status_checks", [])}
        if GATE_CONTEXT not in contexts:
            errors.append(f"required status check {GATE_CONTEXT!r} is missing")
    queue = rules.get("merge_queue", {}).get("parameters", {})
    if queue.get("grouping_strategy") != "ALLGREEN":
        errors.append("merge queue grouping must be ALLGREEN")
    if queue.get("merge_method") != "MERGE":
        errors.append("merge queue method must be MERGE")
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
    parser.add_argument("--apply", action="store_true", help="create or update the ruleset")
    parser.add_argument("--verify", action="store_true", help="verify the installed ruleset")
    parser.add_argument("--dry-run", action="store_true", help="print the ruleset and stop")
    args = parser.parse_args(argv)

    payload = ruleset_payload()
    errors = validate_ruleset(payload)
    if errors:
        print("ruleset payload is invalid: " + "; ".join(errors), file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    repository = gh_api("GET", f"/repos/{args.repo}")
    if owner_type(repository) != "Organization":
        print(
            f"{args.repo} is owned by a {owner_type(repository)}. "
            "Create a free GitHub organization in the web UI, transfer the public repository, "
            "and make the transferring user an organization owner. No paid plan is required.",
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
        errors = validate_ruleset(installed)
        if errors:
            print("merge queue ruleset verification failed: " + "; ".join(errors), file=sys.stderr)
            return 1
        print("merge queue ruleset verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
