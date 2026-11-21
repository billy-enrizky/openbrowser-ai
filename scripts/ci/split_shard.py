"""Split one test shard into parallel and serial groups.

Some test files own process-wide state and must not share an xdist worker pool.
The split is a partition: every input file appears in exactly one output group.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path


def split_files(files: Iterable[str], serial_file: str) -> tuple[list[str], list[str]]:
    """Return ``(parallel, serial)`` files without duplicates or omissions."""
    parallel: list[str] = []
    serial: list[str] = []
    seen: set[str] = set()
    for raw in files:
        path = raw.strip()
        if not path:
            continue
        if path in seen:
            raise ValueError(f"duplicate test file in shard: {path}")
        seen.add(path)
        if path == serial_file:
            serial.append(path)
        else:
            parallel.append(path)
    return parallel, serial


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--serial-file", required=True)
    parser.add_argument("--parallel-output", required=True, type=Path)
    parser.add_argument("--serial-output", required=True, type=Path)
    args = parser.parse_args()

    try:
        parallel, serial = split_files(read_lines(args.input), args.serial_file)
    except ValueError as exc:
        parser.error(str(exc))

    write_lines(args.parallel_output, parallel)
    write_lines(args.serial_output, serial)
    print(f"shard split: parallel={len(parallel)} serial={len(serial)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
