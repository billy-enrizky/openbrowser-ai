"""Add numbered section headings and a table of contents to Markdown docs.

Headings become ``## 1. Title``, ``### 1.1. Title``, and so on. A table of
contents is generated after the first H1 and regenerated on every run, so the
script is idempotent. Intra-file anchor links are rewritten to the new slugs.

Usage:
    python scripts/ci/number_markdown_sections.py --check FILE...
    python scripts/ci/number_markdown_sections.py --write FILE...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TOC_START = "<!-- toc:start -->"
TOC_END = "<!-- toc:end -->"
TOC_TITLE = "## Table of Contents"
HEADING = re.compile(r"^(#{2,4})\s+(.*\S)\s*$")
NUMBERED = re.compile(r"^\d+(\.\d+)*\.\s")
LINK = re.compile(r"\]\(#([^)]+)\)")
FENCE = re.compile(r"^\s*(```|~~~)")


def slug(text: str) -> str:
    """GitHub-style anchor slug."""
    lowered = text.strip().lower()
    lowered = re.sub(r"[^\w\s-]", "", lowered)
    lowered = re.sub(r"\s+", "-", lowered)
    return lowered.strip("-")


def split_frontmatter(lines: list[str]) -> tuple[list[str], list[str]]:
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return lines[: index + 1], lines[index + 1 :]
    return [], lines


def number_sections(body: list[str]) -> tuple[list[str], list[tuple[int, str, str]], dict[str, str]]:
    """Rewrite headings in ``body`` and return the new body, TOC entries, and link map."""
    counters = [0, 0, 0]
    toc: list[tuple[int, str, str]] = []
    link_map: dict[str, str] = {}
    output: list[str] = []
    in_fence = False
    for line in body:
        if FENCE.match(line):
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence:
            output.append(line)
            continue
        match = HEADING.match(line)
        if not match:
            output.append(line)
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        old_slug = slug(title)
        if NUMBERED.match(title):
            title = NUMBERED.sub("", title, count=1).strip()
        index = level - 2
        counters[index] += 1
        for deeper in range(index + 1, 3):
            counters[deeper] = 0
        number = ".".join(str(value) for value in counters[: index + 1])
        numbered_title = f"{number}. {title}"
        new_slug = slug(numbered_title)
        if old_slug and old_slug != new_slug:
            link_map[old_slug] = new_slug
        toc.append((level, numbered_title, new_slug))
        output.append(f"{match.group(1)} {numbered_title}")
    return output, toc, link_map


def build_toc(toc: list[tuple[int, str, str]]) -> list[str]:
    lines = [TOC_START, TOC_TITLE, ""]
    for level, title, anchor in toc:
        indent = "  " * (level - 2)
        lines.append(f"{indent}- [{title}](#{anchor})")
    lines.extend(["", TOC_END])
    return lines


def rewrite_links(body: list[str], link_map: dict[str, str]) -> list[str]:
    def replace(match: re.Match[str]) -> str:
        target = match.group(1)
        return f"](#{link_map.get(target, target)})"

    return [LINK.sub(replace, line) for line in body]


def _strip_generated_toc(lines: list[str]) -> tuple[list[str], int | None]:
    if TOC_START not in lines:
        return lines, None
    start = lines.index(TOC_START)
    end = lines.index(TOC_END) if TOC_END in lines else start
    return lines[:start] + lines[end + 1 :], start


def _find_manual_toc(lines: list[str]) -> tuple[int, int] | None:
    """Return the (start, end) range of a manual "Table of Contents" section."""
    in_fence = False
    for index, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match and len(match.group(1)) == 2:
            title = NUMBERED.sub("", match.group(2).strip(), count=1).strip()
            if title.lower() == "table of contents":
                end = index + 1
                while end < len(lines) and not HEADING.match(lines[end]):
                    end += 1
                while end > index + 1 and not lines[end - 1].strip():
                    end -= 1
                return (index, end)
    return None


def process(path: Path) -> str:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    frontmatter, rest = split_frontmatter(lines)
    rest, previous_toc = _strip_generated_toc(rest)

    manual = _find_manual_toc(rest)
    insert_at: int | None = manual[0] if manual else previous_toc
    if manual is not None:
        rest = rest[: manual[0]] + rest[manual[1] :]

    title_index = None
    in_fence = False
    for index, line in enumerate(rest):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("# "):
            title_index = index
            break
    if insert_at is None:
        insert_at = (title_index + 1) if title_index is not None else 0

    numbered, toc, link_map = number_sections(rest)
    numbered = rewrite_links(numbered, link_map)
    toc_block = build_toc(toc) if toc else []
    if toc_block:
        prefix = numbered[:insert_at]
        suffix = numbered[insert_at:]
        while prefix and not prefix[-1].strip():
            prefix = prefix[:-1]
        while suffix and not suffix[0].strip():
            suffix = suffix[1:]
        numbered = [*prefix, "", *toc_block, "", *suffix]
    output = [*frontmatter, *numbered]
    text = "\n".join(output).rstrip() + "\n"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if any file would change.")
    parser.add_argument("--write", action="store_true", help="Rewrite files in place.")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    if not (args.check or args.write):
        parser.error("pass --check or --write")

    changed = []
    for path in args.files:
        before = path.read_text(encoding="utf-8")
        after = process(path)
        if before != after:
            changed.append(path)
            if args.write:
                path.write_text(after, encoding="utf-8")
    if args.check and changed:
        print("Unnumbered sections or TOC missing in:")
        for path in changed:
            print(f"  {path}")
        return 1
    if args.write:
        print(f"Rewrote {len(changed)} of {len(args.files)} files")
    else:
        print(f"All {len(args.files)} files are numbered and current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
