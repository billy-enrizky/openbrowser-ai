#!/usr/bin/env python3
"""Prove that the PR selector catches a real source break.

This script mutates one source file, runs the selector, runs the selected
tests, and requires at least one failing JUnit test case. It restores the file
in ``finally`` and then requires the same selected tests to pass. Generated
files live in a temporary directory outside the repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command))
    return subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="src/openbrowser/code_use/formatting.py")
    parser.add_argument("--test", default="tests/test_code_use_formatting.py")
    parser.add_argument(
        "--marker",
        default="assert state.dom_state is not None",
        help="exact line to replace with a planted break",
    )
    parser.add_argument("--allow-dirty", action="store_true", help="skip the clean-tree check")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    source = root / args.source
    test = args.test
    if not args.allow_dirty:
        status = run(["git", "status", "--porcelain"], root)
        if status.stdout.strip():
            print(
                f"proof error: working tree is not clean; pass --allow-dirty\n{status.stdout}",
                file=sys.stderr,
            )
            return 7

    original = source.read_text(encoding="utf-8")
    if args.marker not in original:
        print(f"proof error: marker not found in {args.source}", file=sys.stderr)
        return 2

    planted = original.replace(
        args.marker,
        'raise RuntimeError("selection proof: planted break")',
        1,
    )
    selected: list[str] = []
    with tempfile.TemporaryDirectory(prefix="openbrowser-selection-proof-") as tmp:
        selection_path = Path(tmp) / "selection.json"
        junit_path = Path(tmp) / "junit.xml"
        try:
            source.write_text(planted, encoding="utf-8")
            select = run(
                [
                    sys.executable,
                    "scripts/ci/select_tests.py",
                    "--changed-file",
                    args.source,
                    "--output",
                    str(selection_path),
                ],
                root,
            )
            if select.returncode != 0:
                print(select.stderr, file=sys.stderr)
                return 3
            selected = json.loads(selection_path.read_text(encoding="utf-8"))["files"]
            if not selected or test not in selected:
                print(
                    f"proof error: selector did not select {test}; selected={selected}",
                    file=sys.stderr,
                )
                return 4
            failing = run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    *selected,
                    "-q",
                    "-n",
                    "auto",
                    "--dist",
                    "load",
                    "--tb=short",
                    "--maxfail=1",
                    f"--junitxml={junit_path}",
                ],
                root,
            )
            if failing.returncode == 0:
                print(
                    "proof error: planted break did not make the selected tests fail",
                    file=sys.stderr,
                )
                return 5
            failures = sum(
                1
                for case in ET.parse(junit_path).getroot().iter("testcase")
                if case.find("failure") is not None or case.find("error") is not None
            )
            if failures == 0:
                print(
                    "proof error: the non-zero exit was not caused by a failing test",
                    file=sys.stderr,
                )
                return 5
            print(f"planted break caught by {len(selected)} selected test files, {failures} failing test(s)")
        finally:
            source.write_text(original, encoding="utf-8")

    restored = run(
        [
            sys.executable,
            "-m",
            "pytest",
            *selected,
            "-q",
            "-n",
            "auto",
            "--dist",
            "load",
            "--tb=short",
            "--maxfail=1",
        ],
        root,
    )
    if restored.returncode != 0:
        print("proof error: selected tests fail after restoring the source", file=sys.stderr)
        return 6
    print(f"restored source passes the same {len(selected)} selected test files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
