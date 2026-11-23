#!/usr/bin/env python3
"""Validate the root skills.sh manifest against plugin/skills."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "skills.sh.json"
    skills_root = root / "plugin" / "skills"
    errors: list[str] = []

    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        print(f"ERROR: invalid JSON in {manifest_path}: {error}", file=sys.stderr)
        return 1

    paths = manifest.get("paths", [])
    if "./plugin/skills" not in paths:
        errors.append("paths must contain './plugin/skills'")

    if manifest.get("notGrouped") not in (None, "top", "bottom"):
        errors.append("notGrouped must be 'top' or 'bottom'")

    groupings = manifest.get("groupings", [])
    if not groupings:
        errors.append("groupings must contain at least one group")

    grouped: set[str] = set()
    titles: set[str] = set()
    for group in groupings:
        title = group.get("title")
        if not title:
            errors.append("every grouping needs a title")
        elif title in titles:
            errors.append(f"duplicate grouping title: {title}")
        else:
            titles.add(title)
        for skill in group.get("skills", []):
            if skill in grouped:
                errors.append(f"duplicate skill in groupings: {skill}")
                continue
            grouped.add(skill)
            skill_dir = skills_root / skill
            if not skill_dir.is_dir():
                errors.append(f"skill '{skill}' directory not found: {skill_dir}")
            elif not (skill_dir / "SKILL.md").is_file():
                errors.append(f"skill '{skill}' is missing SKILL.md")

    if skills_root.is_dir():
        for entry in sorted(skills_root.iterdir()):
            if (entry / "SKILL.md").is_file() and entry.name not in grouped:
                errors.append(f"skill '{entry.name}' has SKILL.md but is not listed in groupings")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: skills.sh.json valid ({len(grouped)} skills across {len(groupings)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
