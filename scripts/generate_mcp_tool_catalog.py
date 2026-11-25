#!/usr/bin/env python3
"""Regenerate the static MCP catalog used by the lean stdio transport."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "src" / "openbrowser" / "mcp" / "tool_catalog.json"


def build_catalog() -> dict[str, list[dict]]:
    os.environ.setdefault("OPENBROWSER_SETUP_LOGGING", "false")
    os.environ.setdefault("OPENBROWSER_LOGGING_LEVEL", "critical")
    sys.path.insert(0, str(ROOT / "src"))
    from openbrowser.code_use.descriptions import (
        EXECUTE_CODE_DESCRIPTION,
        EXECUTE_CODE_DESCRIPTION_COMPACT,
    )
    from openbrowser.mcp.server import _tool_catalog

    result: dict[str, list[dict]] = {}
    for variant, description in (
        ("default", EXECUTE_CODE_DESCRIPTION),
        ("compact", EXECUTE_CODE_DESCRIPTION_COMPACT),
    ):
        for era, modern in (("modern", True), ("legacy", False)):
            result[f"{variant}_{era}"] = [
                tool.model_dump(by_alias=True, exclude_none=True, mode="json")
                for tool in _tool_catalog(description, modern=modern)
            ]
    return result


def serialized(catalog: dict[str, list[dict]]) -> str:
    return json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when the file is stale.")
    args = parser.parse_args()

    expected = serialized(build_catalog())
    if args.check:
        actual = CATALOG_PATH.read_text(encoding="utf-8")
        if actual != expected:
            print(f"ERROR: {CATALOG_PATH} is stale. Run this script without --check.", file=sys.stderr)
            return 1
        print(f"{CATALOG_PATH} is current")
        return 0

    CATALOG_PATH.write_text(expected, encoding="utf-8")
    print(f"wrote {CATALOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
