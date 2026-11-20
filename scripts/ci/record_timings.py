#!/usr/bin/env python3
"""Record per-file pytest times from JUnit XML for shard balancing."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ALPHA = 0.7


def test_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)).replace("\\", "/") for path in (root / "tests").rglob("test_*.py"))


def real_test_file(classname: str, files: set[str]) -> str | None:
    parts = classname.split(".")
    for index in range(len(parts), 0, -1):
        candidate = "/".join(parts[:index]) + ".py"
        if candidate in files:
            return candidate
    return None


def read_junit(paths: list[Path], root: Path) -> tuple[dict[str, float], Counter[str], int]:
    files = set(test_files(root))
    totals: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    missing = 0
    for path in paths:
        root_node = ET.parse(path).getroot()
        for case in root_node.iter("testcase"):
            filename = real_test_file(case.get("classname", ""), files)
            if filename is None:
                missing += 1
                continue
            totals[filename] += float(case.get("time") or 0.0)
            counts[filename] += 1
    return dict(totals), counts, missing


def merge(existing: dict[str, object], observed: dict[str, float]) -> dict[str, object]:
    old_files = existing.get("files", {}) if isinstance(existing, dict) else {}
    if not isinstance(old_files, dict):
        old_files = {}
    merged: dict[str, dict[str, float | int]] = {}
    for filename in sorted(set(old_files) | set(observed)):
        old = old_files.get(filename)
        new = observed.get(filename)
        if isinstance(old, dict):
            old_seconds = float(old.get("seconds", 0.0))
            old_samples = int(old.get("samples", 1))
        elif isinstance(old, (int, float)):
            old_seconds = float(old)
            old_samples = 1
        else:
            old_seconds = 0.0
            old_samples = 0
        if new is None:
            merged[filename] = {
                "seconds": round(old_seconds, 4),
                "samples": old_samples,
            }
        elif old_samples == 0:
            merged[filename] = {"seconds": round(new, 4), "samples": 1}
        else:
            value = (1.0 - ALPHA) * old_seconds + ALPHA * new
            merged[filename] = {
                "seconds": round(value, 4),
                "samples": old_samples + 1,
            }
    return {
        "_meta": {
            "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
            "method": f"exponential moving average, alpha={ALPHA}",
            "files_recorded": len(observed),
        },
        "files": merged,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--junit", action="append", required=True, help="JUnit XML path")
    parser.add_argument("--output", default="tests/ci/test_timings.json")
    parser.add_argument("--replace", action="store_true", help="ignore existing timings")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    paths = [Path(item) for item in args.junit]
    observed, counts, missing = read_junit(paths, root)
    output = root / args.output
    existing: dict[str, object] = {}
    if output.exists() and not args.replace:
        existing = json.loads(output.read_text(encoding="utf-8"))
    payload = merge(existing, observed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    total = sum(observed.values())
    print(
        f"recorded {len(observed)} files, {sum(counts.values())} tests, "
        f"{total:.1f}s, missing={missing}, output={output}"
    )
    print(f"median file time={statistics.median(observed.values()) if observed else 0:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
