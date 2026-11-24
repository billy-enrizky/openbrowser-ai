"""Serialization for HTTP-only scrape results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def read_scrape_urls_file(path: Path) -> list[str]:
    """Read one scrape URL per line, ignoring blank lines and comments."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read {path}: {error}") from error
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def serialize_scrape_batch(results: list[dict], *, jsonl: bool) -> str:
    """Serialize batch scrape results as a JSON array or JSONL."""
    if jsonl:
        return "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in results)
    return json.dumps(results, indent=2, ensure_ascii=False, default=str)


def emit_scrape_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    batch_mode: bool,
    jsonl: bool,
    batch_output_format: str | None,
    output: str | None,
    as_json: bool,
) -> None:
    """Write a scrape payload with the CLI's single and batch formats."""
    serialization = "jsonl" if jsonl else batch_output_format
    if serialization is None and output and Path(output).suffix.lower() == ".jsonl":
        serialization = "jsonl"
    if serialization is None:
        serialization = "json"
    explicit_serialization = jsonl or batch_output_format is not None

    if batch_mode:
        rendered = serialize_scrape_batch(payload, jsonl=serialization == "jsonl")  # type: ignore[arg-type]
        if output:
            Path(output).write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
            print(f"Wrote {len(rendered)} characters to {output}", file=sys.stderr)
        else:
            print(rendered)
        return

    assert isinstance(payload, dict)
    if serialization == "jsonl":
        rendered = json.dumps(payload, ensure_ascii=False)
    else:
        content = payload.get("content")
        if as_json or (explicit_serialization and serialization == "json") or not isinstance(content, str):
            rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        else:
            rendered = content
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        print(f"Wrote {len(rendered)} characters to {output}")
    else:
        print(rendered)
