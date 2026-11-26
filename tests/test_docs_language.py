"""First-party documentation language checks.

The project keeps public prose affirmative: state what a feature does, what it
covers, and what the reader should do. Define a thing by its own behavior. Fenced code, inline code, and
conditional sentences such as "if it is not ready" keep their technical meaning
and are excluded from the scan.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDED_DIRS = (
    ".codex",
    ".opencode",
    "backend",
    "benchmarks",
    "docs",
    "examples",
    "frontend",
    "infra",
    "landing",
    "plugin",
    "src",
    "stress-tests",
)
PRUNED_DIRS = {".git", ".venv", ".worktrees", "local_docs", "nanochat", "node_modules"}
EXCLUDED_NAMES = {"AGENTS.md", "CLAUDE.md"}

_CONTRASTIVE_PATTERNS = (
    re.compile(r"(?<!if )\b(this|it|that)\s+is\s+not\b", re.IGNORECASE),
    re.compile(r"\bnot\s+just\b", re.IGNORECASE),
    re.compile(r",\s+not\s+\w", re.IGNORECASE),
    re.compile(r"\brather\s+than\b", re.IGNORECASE),
    re.compile(r"\binstead\s+of\b", re.IGNORECASE),
)
_COMPETITIVE_TERMS = re.compile(r"\bcompetitors?\b", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```|~~~)")
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def _markdown_files(root: Path = ROOT) -> list[Path]:
    files = [path for path in root.glob("*.md") if path.is_file() and path.name not in EXCLUDED_NAMES]
    for directory in INCLUDED_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if name not in PRUNED_DIRS]
            for filename in filenames:
                if filename.endswith((".md", ".mdx")):
                    files.append(Path(dirpath) / filename)
    return sorted(set(files))


def _prose_lines(text: str):
    in_fence = False
    for number, line in enumerate(text.splitlines(), 1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        yield number, _INLINE_CODE.sub(" CODE ", line)


def test_first_party_markdown_uses_affirmative_definitions() -> None:
    violations: list[str] = []
    for path in _markdown_files():
        for number, line in _prose_lines(path.read_text(encoding="utf-8")):
            for pattern in _CONTRASTIVE_PATTERNS:
                match = pattern.search(line)
                if match is not None:
                    relative = path.relative_to(ROOT)
                    violations.append(f"{relative}:{number}: {match.group(0)!r} in {line.strip()!r}")
                    break

    assert not violations, (
        "Rewrite these definitions as direct affirmative statements. "
        "State what the feature does and what the reader should do.\n" + "\n".join(violations)
    )


def test_markdown_scan_prunes_ignored_directories(tmp_path: Path) -> None:
    kept = tmp_path / "docs" / "kept.md"
    kept.parent.mkdir()
    kept.write_text("# Kept", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("The phase is `degraded`, not `ready`.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("The phase is `degraded`, not `ready`.", encoding="utf-8")
    for ignored_dir in ("node_modules", ".venv", ".git", "nanochat", "local_docs"):
        ignored = tmp_path / "docs" / ignored_dir / "ignored.md"
        ignored.parent.mkdir(parents=True)
        ignored.write_text("The phase is `degraded`, not `ready`.", encoding="utf-8")

    found = _markdown_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in found] == ["docs/kept.md"]


def test_first_party_markdown_uses_other_framework_terminology() -> None:
    violations: list[str] = []
    for path in _markdown_files():
        for number, line in _prose_lines(path.read_text(encoding="utf-8")):
            match = _COMPETITIVE_TERMS.search(line)
            if match is not None:
                relative = path.relative_to(ROOT)
                violations.append(f"{relative}:{number}: {match.group(0)!r} in {line.strip()!r}")

    assert not violations, "Use 'other framework' or 'other frameworks' for third-party tools.\n" + "\n".join(
        violations
    )


def test_competitive_terminology_pattern_matches() -> None:
    assert _COMPETITIVE_TERMS.search("against every competitor")
    assert _COMPETITIVE_TERMS.search("all three competitors")
