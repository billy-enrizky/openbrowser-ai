#!/usr/bin/env python3
"""Drain labeled pull requests through a serialized full-suite merge queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

LABEL = "merge-queue"
RUNNING_LABEL = "merge-queue-running"
FAILED_LABEL = "merge-queue-failed"
BLOCKED_LABEL = "merge-queue-blocked"
CHECK_NAME = "merge-queue-gate"
TRUSTED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
QUEUE_CHECKS = "scripts/ci/run_queue_checks.sh"
MERGE_READY_STATES = frozenset({"CLEAN", "UNSTABLE", "HAS_HOOKS"})
MERGE_BLOCKING_STATES = frozenset({"DIRTY", "DRAFT"})
MERGE_RETRY_DELAYS = (2.0, 4.0, 8.0)


def trusted_author(item: dict[str, Any]) -> bool:
    return item.get("author_association") in TRUSTED_ASSOCIATIONS


def queue_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pull_requests = [item for item in items if "pull_request" in item and item.get("state") == "open"]
    return sorted(pull_requests, key=lambda item: int(item["number"]))


def check_payload(head_sha: str, conclusion: str, summary: str) -> dict[str, Any]:
    return {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {
            "title": "Merge queue passed" if conclusion == "success" else "Merge queue failed",
            "summary": summary[:60000],
        },
    }


def command(command: list[str], root: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result


def gh_api(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    cmd = ["gh", "api", "--method", method, path]
    if body is not None:
        cmd.extend(["--input", "-"])
    result = subprocess.run(
        cmd,
        input=json.dumps(body) if body is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh api failed: {result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def list_labeled(repo: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = gh_api(
            "GET",
            f"/repos/{repo}/issues?state=open&labels={LABEL}&per_page=100&page={page}",
        )
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return queue_items(items)


def ensure_labels(repo: str) -> None:
    for name, color, description in (
        (LABEL, "0e8a16", "Queue this pull request for full-suite merge testing"),
        (RUNNING_LABEL, "fbca04", "The serialized merge queue is testing this pull request"),
        (FAILED_LABEL, "d73a4a", "The merge queue rejected this pull request"),
        (BLOCKED_LABEL, "d4c5f9", "The merge queue passed but the merge needs attention"),
    ):
        existing = subprocess.run(
            ["gh", "api", f"/repos/{repo}/labels/{name}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if existing.returncode == 0:
            continue
        result = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"/repos/{repo}/labels",
                "-f",
                f"name={name}",
                "-f",
                f"color={color}",
                "-f",
                f"description={description}",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)


def set_label(repo: str, number: int, label: str, present: bool) -> None:
    if present:
        result = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"/repos/{repo}/issues/{number}/labels",
                "--input",
                "-",
            ],
            input=json.dumps({"labels": [label]}),
            text=True,
            capture_output=True,
            check=False,
        )
    else:
        result = subprocess.run(
            ["gh", "api", "--method", "DELETE", f"/repos/{repo}/issues/{number}/labels/{label}"],
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode != 0 and not any(phrase in result.stderr for phrase in ("Not Found", "does not exist")):
        print(f"label update failed for {label}: {result.stderr.strip()}", file=sys.stderr)


def post_check(repo: str, head_sha: str, conclusion: str, summary: str) -> None:
    gh_api("POST", f"/repos/{repo}/check-runs", check_payload(head_sha, conclusion, summary))


def comment(repo: str, number: int, body: str) -> None:
    gh_api("POST", f"/repos/{repo}/issues/{number}/comments", {"body": body})


def ci_gate_success(repo: str, head_sha: str) -> bool:
    payload = gh_api("GET", f"/repos/{repo}/commits/{head_sha}/check-runs?per_page=100")
    return any(
        run.get("name") == "ci-gate" and run.get("conclusion") == "success" for run in payload.get("check_runs", [])
    )


def check_run_success(repo: str, head_sha: str, name: str) -> bool:
    payload = gh_api("GET", f"/repos/{repo}/commits/{head_sha}/check-runs?per_page=100")
    return any(run.get("name") == name and run.get("conclusion") == "success" for run in payload.get("check_runs", []))


def wait_for_check(repo: str, head_sha: str, name: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_run_success(repo, head_sha, name):
            return True
        time.sleep(2)
    return False


def merge_ready(pull: dict[str, Any]) -> bool:
    """Return true when GitHub allows the merge and no policy state is pending."""
    return pull.get("mergeable") is True and pull.get("mergeStateStatus") in MERGE_READY_STATES


def wait_for_mergeable(repo: str, number: int, timeout: float = 20.0) -> dict[str, Any]:
    """Wait for GitHub to settle the pull request mergeability after a new check."""
    deadline = time.monotonic() + timeout
    pull = gh_api("GET", f"/repos/{repo}/pulls/{number}")
    while time.monotonic() < deadline:
        if merge_ready(pull) or pull.get("mergeStateStatus") in MERGE_BLOCKING_STATES:
            return pull
        time.sleep(2)
        pull = gh_api("GET", f"/repos/{repo}/pulls/{number}")
    return pull


def reject(repo: str, number: int, head_sha: str, reason: str) -> None:
    print(f"merge queue rejected #{number}: {reason}", file=sys.stderr)
    post_check(repo, head_sha, "failure", reason)
    comment(repo, number, f"Merge queue failed: {reason}")
    set_label(repo, number, RUNNING_LABEL, False)
    set_label(repo, number, LABEL, False)
    set_label(repo, number, FAILED_LABEL, True)
    set_label(repo, number, BLOCKED_LABEL, False)


def merge_blocked(repo: str, number: int, reason: str) -> None:
    """Record a post-gate merge problem without revoking the successful gate."""
    print(f"merge queue passed #{number} but could not merge: {reason}", file=sys.stderr)
    comment(
        repo,
        number,
        "Merge queue checks passed, but GitHub did not complete the merge: "
        f"{reason}. The merge-queue-gate success is preserved. Resolve the branch policy "
        "or merge issue, then merge the pull request or rerun the queue.",
    )
    set_label(repo, number, RUNNING_LABEL, False)
    set_label(repo, number, LABEL, False)
    set_label(repo, number, FAILED_LABEL, False)
    set_label(repo, number, BLOCKED_LABEL, True)


def base_sha(repo: str) -> str:
    return str(gh_api("GET", f"/repos/{repo}/commits/main")["sha"])


def process(repo: str, root: Path, number: int, checks_script: Path) -> None:
    item = gh_api("GET", f"/repos/{repo}/issues/{number}")
    pull = gh_api("GET", f"/repos/{repo}/pulls/{number}")
    head_sha = str(pull["head"]["sha"])
    if not trusted_author(item):
        reject(repo, number, head_sha, "author is not an owner, member, or collaborator")
        return
    if pull.get("draft"):
        reject(repo, number, head_sha, "draft pull requests cannot enter the merge queue")
        return
    if pull.get("base", {}).get("ref") != "main":
        reject(repo, number, head_sha, "the pull request does not target main")
        return
    if not ci_gate_success(repo, head_sha):
        reject(repo, number, head_sha, "ci-gate has not succeeded on the current head")
        return

    set_label(repo, number, LABEL, False)
    set_label(repo, number, RUNNING_LABEL, True)
    gate_passed = False
    main_before = base_sha(repo)
    branch = f"queue-pr-{number}"
    try:
        command(["git", "fetch", "origin", "main", "--force"], root)
        command(["git", "fetch", "origin", f"pull/{number}/head:{branch}", "--force"], root)
        command(["git", "checkout", "-B", f"queue-merge-{number}", "origin/main"], root)
        merge = command(["git", "merge", "--no-ff", "--no-edit", branch], root, check=False)
        if merge.returncode != 0:
            command(["git", "merge", "--abort"], root, check=False)
            reject(repo, number, head_sha, "the pull request conflicts with current main")
            return
        checks = command(["bash", str(checks_script)], root, check=False)
        if checks.returncode != 0:
            reject(repo, number, head_sha, "the full non-integration checks failed")
            return
        if base_sha(repo) != main_before:
            reject(
                repo,
                number,
                head_sha,
                "main changed while the pull request was being tested; re-add the merge-queue label",
            )
            return
        current = gh_api("GET", f"/repos/{repo}/pulls/{number}")
        if current["head"]["sha"] != head_sha:
            reject(
                repo,
                number,
                head_sha,
                "the pull request head changed during testing; re-add the merge-queue label after ci-gate passes",
            )
            return
        post_check(
            repo,
            head_sha,
            "success",
            "Full non-integration suite, static checks, frontend checks, backend smoke, "
            "and merge-tree generation passed.",
        )
        gate_passed = True
        if not wait_for_check(repo, head_sha, CHECK_NAME):
            merge_blocked(repo, number, "the merge-queue-gate check did not become visible")
            return
        wait_for_mergeable(repo, number)
        merge_result = command(
            ["gh", "pr", "merge", str(number), "--merge", "--match-head-commit", head_sha],
            root,
            check=False,
        )
        for delay in MERGE_RETRY_DELAYS:
            if merge_result.returncode == 0:
                break
            print(f"merge attempt failed; retrying #{number} in {delay:g}s", file=sys.stderr)
            time.sleep(delay)
            wait_for_mergeable(repo, number)
            merge_result = command(
                ["gh", "pr", "merge", str(number), "--merge", "--match-head-commit", head_sha],
                root,
                check=False,
            )
        if merge_result.returncode != 0:
            detail = (merge_result.stderr or merge_result.stdout or "GitHub rejected the merge").strip()
            merge_blocked(repo, number, f"GitHub rejected the merge after checks passed: {detail}")
            return
        merged_main = base_sha(repo)
        output_path = os.environ.get("GITHUB_OUTPUT")
        if output_path:
            with open(output_path, "a", encoding="utf-8") as output:
                output.write(f"main_sha={merged_main}\n")
        set_label(repo, number, RUNNING_LABEL, False)
        set_label(repo, number, FAILED_LABEL, False)
        set_label(repo, number, BLOCKED_LABEL, False)
        print(f"merged #{number} through the custom queue")
    except Exception as exc:
        if gate_passed:
            merge_blocked(repo, number, f"queue error after checks passed: {exc}")
        else:
            reject(repo, number, head_sha, f"queue error: {exc}")
    finally:
        command(["git", "checkout", "-f", "origin/main"], root, check=False)
        command(["git", "reset", "--hard", "origin/main"], root, check=False)
        command(["git", "clean", "-fd"], root, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    command(["git", "config", "user.name", "github-actions[bot]"], root)
    command(
        ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        root,
    )
    ensure_labels(args.repo)
    checks_source = root / QUEUE_CHECKS
    with tempfile.TemporaryDirectory(prefix="merge-queue-checks-") as tmp:
        checks_script = Path(tmp) / "run_queue_checks.sh"
        shutil.copy2(checks_source, checks_script)
        checks_script.chmod(0o755)
        for item in list_labeled(args.repo):
            number = int(item["number"])
            print(f"queue: processing #{number}", flush=True)
            process(args.repo, root, number, checks_script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
