"""Static MCP tool definitions shared by every stdio transport.

The SDK builds typed models from the same source used by the server.  The lean
transport reads this generated artifact so it can answer ``tools/list`` before
it imports pydantic, jsonschema, or the MCP SDK.  A test compares this artifact
with the live server catalog so the two cannot drift.
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).with_name("tool_catalog.json")


@lru_cache(maxsize=1)
def _load() -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("MCP tool catalog must be a JSON object")
    return raw


def load_tool_definitions(*, modern: bool, compact: bool = False) -> tuple[dict[str, Any], ...]:
    """Return deep-copied plain tool definitions for one protocol era."""
    key = f"{'compact' if compact else 'default'}_{'modern' if modern else 'legacy'}"
    try:
        definitions = _load()[key]
    except KeyError as error:
        raise ValueError(f"Missing MCP tool catalog entry: {key}") from error
    if not isinstance(definitions, list):
        raise ValueError(f"MCP tool catalog entry {key} must be a list")
    return tuple(deepcopy(definitions))
