"""Every user-facing Markdown doc keeps numbered sections and a table of contents."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "number_markdown_sections.py"
DOCS = [
    "README.md",
    "plugin/README.md",
    ".codex/INSTALL.md",
    ".opencode/INSTALL.md",
    "plugin/skills/accessibility-audit/SKILL.md",
    "plugin/skills/deep-research/SKILL.md",
    "plugin/skills/e2e-testing/SKILL.md",
    "plugin/skills/file-download/SKILL.md",
    "plugin/skills/form-filling/SKILL.md",
    "plugin/skills/page-analysis/SKILL.md",
    "plugin/skills/setup/SKILL.md",
    "plugin/skills/web-scraping/SKILL.md",
    "src/openbrowser/code_use/README.md",
    "benchmarks/scraping/README.md",
    "benchmarks/interactive/README.md",
]


def test_user_facing_docs_are_numbered_with_a_toc():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", *DOCS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        "Markdown sections or table of contents are stale. "
        f"Run: python scripts/ci/number_markdown_sections.py --write {' '.join(DOCS)}\n"
        f"{completed.stdout}{completed.stderr}"
    )


def test_numbering_is_idempotent():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", *DOCS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    assert "current" in completed.stdout
