#!/usr/bin/env python3
"""Run a command and fail it when it exceeds its recorded CI budget."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def kill_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="budget key")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--budget", type=float, default=None, help="override seconds")
    parser.add_argument("--hard-timeout", type=float, default=None, help="kill after seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("budget error: no command", file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    budget_file = root / ".github" / "ci-budgets.json"
    budgets = json.loads(budget_file.read_text(encoding="utf-8")) if budget_file.exists() else {}
    budget = args.budget if args.budget is not None else budgets.get(args.name)
    if not isinstance(budget, (int, float)) or budget <= 0:
        print(f"budget error: no positive budget for {args.name}", file=sys.stderr)
        return 2
    hard_timeout = args.hard_timeout if args.hard_timeout is not None else budget * 2

    started = time.monotonic()
    kwargs: dict[str, object] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, cwd=root, **kwargs)
    try:
        code = process.wait(timeout=hard_timeout)
        elapsed = time.monotonic() - started
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        kill_tree(process)
        print(
            f"BUDGET EXCEEDED: {args.name} hit the {hard_timeout:.0f}s hard timeout (budget {budget:.0f}s)",
            file=sys.stderr,
        )
        return 124

    print(f"budget: {args.name} took {elapsed:.1f}s, budget {budget:.0f}s")
    if code != 0:
        print(f"COMMAND FAILED: {args.name} exited {code}", file=sys.stderr)
        return code
    if elapsed > budget:
        print(
            f"BUDGET EXCEEDED: {args.name} took {elapsed:.1f}s, budget {budget:.0f}s",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
