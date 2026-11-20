#!/usr/bin/env python3
"""Fail when Ruff or Pyright finds a diagnostic that is not in the baseline.

The baseline is keyed by file, rule, and a hash of the normalized diagnostic
message. A line shift is not a new finding. A different message in the same
file and rule is a new finding, even when the total count is unchanged. This
prevents the old count-only ratchet from accepting one new defect when another
defect disappears.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

RUFF_SELECT = "E,F,I,B,UP,SIM,ASYNC,RUF"
LINE_LENGTH = 120
MESSAGE_LIMIT = 160


def tool_path(name: str) -> str:
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def relative(value: str, root: Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def normalize_message(message: str) -> str:
    return " ".join(message.split())


def message_hash(message: str) -> str:
    return hashlib.sha256(normalize_message(message).encode("utf-8")).hexdigest()[:16]


def collect(tool: str, root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    findings: dict[tuple[str, str, str], dict[str, Any]] = {}
    if tool == "ruff":
        command = [
            tool_path("ruff"),
            "check",
            "src",
            "tests",
            "scripts",
            "--output-format",
            "json",
            "--line-length",
            str(LINE_LENGTH),
            "--select",
            RUFF_SELECT,
        ]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        if result.returncode not in (0, 1):
            raise SystemExit(f"ruff failed to run: {result.stderr.strip()}")
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ruff did not return JSON: {exc}") from exc
        for item in payload:
            message = item.get("message") or ""
            key = (
                relative(item["filename"], root),
                item.get("code") or "unknown",
                message_hash(message),
            )
            entry = findings.setdefault(key, {"count": 0, "message": normalize_message(message)[:MESSAGE_LIMIT]})
            entry["count"] += 1
        return findings

    command = [tool_path("pyright"), "--outputjson", "src/openbrowser"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"pyright did not return JSON: {exc}\n{result.stderr}") from exc
    for item in payload.get("generalDiagnostics", []):
        if item.get("severity") != "error":
            continue
        message = item.get("message") or ""
        key = (
            relative(item["file"], root),
            item.get("rule") or "unknown",
            message_hash(message),
        )
        entry = findings.setdefault(key, {"count": 0, "message": normalize_message(message)[:MESSAGE_LIMIT]})
        entry["count"] += 1
    return findings


def collect_format(root: Path) -> set[str]:
    command = [tool_path("ruff"), "format", "--check", "src", "tests", "scripts"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    files: set[str] = set()
    for line in (result.stdout + result.stderr).splitlines():
        stripped = line.strip()
        if stripped.startswith("-->"):
            value = stripped[3:].strip()
            files.add(value.rsplit(":", 2)[0])
    return files


def compare_format(current: set[str], baseline: set[str]) -> tuple[list[str], list[str]]:
    return sorted(current - baseline), sorted(baseline - current)


def nest(
    findings: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    payload: dict[str, dict[str, dict[str, Any]]] = {}
    for (file, code, digest), entry in sorted(findings.items()):
        payload.setdefault(file, {}).setdefault(code, {})[digest] = entry
    return payload


def flatten(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    findings: dict[tuple[str, str, str], dict[str, Any]] = {}
    for file, codes in payload.items():
        if not isinstance(codes, dict):
            continue
        for code, digests in codes.items():
            if not isinstance(digests, dict):
                continue
            for digest, entry in digests.items():
                if isinstance(entry, int):
                    findings[(file, code, digest)] = {"count": entry, "message": ""}
                elif isinstance(entry, dict):
                    findings[(file, code, digest)] = {
                        "count": int(entry.get("count", 0)),
                        "message": str(entry.get("message", "")),
                    }
    return findings


def compare(
    current: dict[tuple[str, str, str], dict[str, Any]],
    baseline: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[
    list[tuple[tuple[str, str, str], dict[str, Any], int]],
    list[tuple[tuple[str, str, str], dict[str, Any], int]],
]:
    regressions = []
    improvements = []
    for key, entry in sorted(current.items()):
        old = int(baseline.get(key, {}).get("count", 0))
        count = int(entry.get("count", 0))
        if count > old:
            regressions.append((key, entry, old))
    for key, entry in sorted(baseline.items()):
        old = int(entry.get("count", 0))
        count = int(current.get(key, {}).get("count", 0))
        if count < old:
            improvements.append((key, entry, count))
    return regressions, improvements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("ruff", "pyright", "format"), required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    baseline_path = Path(args.baseline) if args.baseline else (root / ".github" / "ci-baselines" / f"{args.tool}.json")
    if args.tool == "format":
        current_format = collect_format(root)
        if args.update:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(
                json.dumps({"files": sorted(current_format)}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {len(current_format)} unformatted files to {baseline_path}")
            return 0
        if not baseline_path.exists():
            print(f"baseline missing: {baseline_path}; run with --update", file=sys.stderr)
            return 2
        baseline_format = set(json.loads(baseline_path.read_text(encoding="utf-8")).get("files", []))
        new_files, fixed_files = compare_format(current_format, baseline_format)
        print(
            f"format: current={len(current_format)} baseline={len(baseline_format)} "
            f"new={len(new_files)} fixed={len(fixed_files)}"
        )
        for file in new_files:
            print(f"NEW {file} is not formatted")
        for file in fixed_files[:20]:
            print(f"FIXED {file}")
        if len(fixed_files) > 20:
            print(f"... {len(fixed_files) - 20} more fixed files")
        return 1 if new_files else 0

    current = collect(args.tool, root)
    if args.update:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(nest(current), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {len(current)} baseline entries to {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(f"baseline missing: {baseline_path}; run with --update", file=sys.stderr)
        return 2
    baseline = flatten(json.loads(baseline_path.read_text(encoding="utf-8")))
    regressions, improvements = compare(current, baseline)
    current_total = sum(int(entry["count"]) for entry in current.values())
    baseline_total = sum(int(entry["count"]) for entry in baseline.values())
    print(
        f"{args.tool}: current={current_total} baseline={baseline_total} "
        f"new={len(regressions)} improved={len(improvements)}"
    )
    for (file, code, digest), entry, old in regressions:
        print(f"NEW {file} {code} {digest} count={entry['count']} baseline={old} message={entry.get('message', '')}")
    for (file, code, digest), entry, count in improvements[:20]:
        print(f"IMPROVED {file} {code} {digest} count={count} baseline={entry['count']}")
    if len(improvements) > 20:
        print(f"... {len(improvements) - 20} more improvements")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
