#!/usr/bin/env python3
"""Assign selected test files to shards by recorded file time.

The allocator uses longest-processing-time first. It is deterministic: ties
break by path, and equal shard totals break by shard index. Every input file is
assigned exactly once.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

DEFAULT_TIME = 1.0


def load_timings(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "files" in data and isinstance(data["files"], dict):
        data = data["files"]
    if not isinstance(data, dict):
        raise ValueError("timings must be a JSON object")
    timings: dict[str, float] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            value = value.get("seconds")
        if not isinstance(value, (int, float)):
            raise ValueError(f"timing for {key} is not a number")
        if value < 0:
            raise ValueError(f"timing for {key} is negative")
        timings[str(key)] = float(value)
    return timings


def load_selection(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        files = data
    elif isinstance(data, dict):
        files = data.get("files")
    else:
        raise ValueError("selection must be a JSON list or an object with files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        raise ValueError("selection files must be a list of strings")
    return files


def fill_missing(files: list[str], timings: dict[str, float]) -> dict[str, float]:
    known = [timings[file] for file in files if file in timings]
    fallback = statistics.median(known) if known else DEFAULT_TIME
    return {file: timings.get(file, fallback) for file in files}


def allocate(files: list[str], timings: dict[str, float], shards: int) -> list[list[str]]:
    if shards < 1:
        raise ValueError("shards must be at least 1")
    if len(set(files)) != len(files):
        raise ValueError("selection contains duplicate files")
    weights = fill_missing(files, timings)
    bins: list[list[str]] = [[] for _ in range(shards)]
    totals = [0.0] * shards
    for file in sorted(files, key=lambda item: (-weights[item], item)):
        index = min(range(shards), key=lambda item: (totals[item], item))
        bins[index].append(file)
        totals[index] += weights[file]
    return bins


def shard_totals(bins: list[list[str]], timings: dict[str, float]) -> list[float]:
    weights = fill_missing([file for bin_ in bins for file in bin_], timings)
    return [round(sum(weights[file] for file in bin_), 3) for bin_ in bins]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="selection JSON")
    parser.add_argument("--timings", required=True, help="file timing JSON")
    parser.add_argument("--shards", type=int, required=True, help="number of shards")
    parser.add_argument("--shard", type=int, default=None, help="1-based shard to write")
    parser.add_argument("--output", default=None, help="write this shard's files here")
    parser.add_argument("--json", action="store_true", help="print all shard assignments")
    parser.add_argument("--verify-all", action="store_true", help="prove union and exactly-once")
    args = parser.parse_args(argv)

    try:
        files = load_selection(Path(args.selection))
        timings = load_timings(Path(args.timings))
        bins = allocate(files, timings, args.shards)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"shard error: {exc}", file=sys.stderr)
        return 2

    totals = shard_totals(bins, timings)
    for index, (bin_, total) in enumerate(zip(bins, totals, strict=False), start=1):
        print(f"shard {index}/{args.shards}: files={len(bin_)} recorded_seconds={total:.1f}")

    flat = [file for bin_ in bins for file in bin_]
    if sorted(flat) != sorted(files):
        print("shard error: union does not equal selection", file=sys.stderr)
        return 3
    if len(flat) != len(set(flat)):
        print("shard error: a file appears in more than one shard", file=sys.stderr)
        return 3

    if args.verify_all:
        print(
            f"verified: {len(files)} files, {args.shards} shards, "
            f"max={max(totals) if totals else 0:.1f}s, mean={statistics.mean(totals) if totals else 0:.1f}s"
        )

    if args.json:
        print(json.dumps({"shards": bins, "totals": totals}, indent=2))

    if args.shard is not None:
        if args.shard < 1 or args.shard > args.shards:
            print(
                f"shard error: --shard must be between 1 and {args.shards}",
                file=sys.stderr,
            )
            return 2
        selected = bins[args.shard - 1]
        payload = "\n".join(selected) + ("\n" if selected else "")
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
