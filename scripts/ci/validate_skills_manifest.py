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

    marketplace_path = root / ".claude-plugin" / "marketplace.json"
    marketplace_skill_count = 0
    if not marketplace_path.is_file():
        errors.append(f"missing Claude marketplace manifest: {marketplace_path}")
    else:
        try:
            marketplace = json.loads(marketplace_path.read_text())
        except json.JSONDecodeError as error:
            errors.append(f"invalid JSON in {marketplace_path}: {error}")
        else:
            plugins = [plugin for plugin in marketplace.get("plugins", []) if plugin.get("name") == "openbrowser"]
            if len(plugins) != 1:
                errors.append("marketplace.json must contain exactly one 'openbrowser' plugin")
            else:
                plugin = plugins[0]
                source = plugin.get("source")
                if not isinstance(source, str) or not source.startswith("./"):
                    errors.append("the 'openbrowser' plugin source must be a local path starting with './'")
                else:
                    plugin_root = (root / source).resolve()
                    try:
                        plugin_root.relative_to(root.resolve())
                    except ValueError:
                        errors.append("the 'openbrowser' plugin source must stay inside the repository")
                    else:
                        skill_paths = plugin.get("skills")
                        if not isinstance(skill_paths, list) or not all(isinstance(path, str) for path in skill_paths):
                            errors.append("the 'openbrowser' plugin must declare a list of skill paths")
                        else:
                            marketplace_skill_count = len(skill_paths)
                            expected_paths = {f"./skills/{skill}" for skill in grouped}
                            declared_paths = set(skill_paths)
                            if len(declared_paths) != len(skill_paths):
                                errors.append("marketplace.json contains duplicate skill paths")
                            for skill_path in skill_paths:
                                if not skill_path.startswith("./"):
                                    errors.append(f"marketplace skill path must start with './': {skill_path}")
                                    continue
                                skill_dir = (plugin_root / skill_path[2:]).resolve()
                                try:
                                    skill_dir.relative_to(plugin_root)
                                except ValueError:
                                    errors.append(f"marketplace skill path escapes the plugin source: {skill_path}")
                                    continue
                                if not (skill_dir / "SKILL.md").is_file():
                                    errors.append(f"marketplace skill path is missing SKILL.md: {skill_path}")
                            missing_paths = expected_paths - declared_paths
                            extra_paths = declared_paths - expected_paths
                            if missing_paths:
                                errors.append(
                                    f"marketplace.json is missing skill paths: {', '.join(sorted(missing_paths))}"
                                )
                            if extra_paths:
                                errors.append(
                                    f"marketplace.json has unregistered skill paths: {', '.join(sorted(extra_paths))}"
                                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"OK: skills.sh.json valid ({len(grouped)} skills across {len(groupings)} groups); "
        f"Claude marketplace valid ({marketplace_skill_count} skill paths)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
