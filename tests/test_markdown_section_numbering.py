"""Every user-facing Markdown doc keeps numbered sections and a table of contents."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "number_markdown_sections.py"
EXCLUDED = {
    "src/openbrowser/agent/system_prompt.md",
    "src/openbrowser/agent/system_prompt_flash.md",
    "src/openbrowser/agent/system_prompt_no_thinking.md",
    "src/openbrowser/code_use/system_prompt.md",
}

# These paths belong to the production repository and must never be pushed to
# origin, so they stay out of the origin-facing documentation contract.
PRODUCTION_ONLY_PREFIXES = ("backend/", "frontend/", "landing/", "infra/production/", "CSC490/")


def _tracked_docs() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "*.md", "*.mdx"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return sorted(
        path
        for path in completed.stdout.splitlines()
        if path and path not in EXCLUDED and not path.startswith(PRODUCTION_ONLY_PREFIXES)
    )


def _run(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_user_facing_docs_are_numbered_with_a_toc():
    docs = _tracked_docs()
    assert docs
    completed = _run("--check", *docs)
    assert completed.returncode == 0, (
        "Markdown sections or table of contents are stale. "
        f"Run: python scripts/ci/number_markdown_sections.py --write {' '.join(docs)}\n"
        f"{completed.stdout}{completed.stderr}"
    )


def test_numbering_is_idempotent(tmp_path):
    docs = _tracked_docs()
    copies = []
    for doc in docs:
        destination = tmp_path / doc.replace("/", "__")
        shutil.copy2(ROOT / doc, destination)
        copies.append(str(destination))
    first = _run("--write", *copies)
    assert first.returncode == 0, first.stdout + first.stderr
    before = [Path(path).read_text(encoding="utf-8") for path in copies]
    second = _run("--write", *copies)
    assert second.returncode == 0, second.stdout + second.stderr
    after = [Path(path).read_text(encoding="utf-8") for path in copies]
    assert before == after
    assert _run("--check", *copies).returncode == 0


def test_h3_first_document_numbers_from_one(tmp_path):
    source = tmp_path / "h3-first.md"
    source.write_text(
        "# Title\n\n### First section\n\n#### Child\n\n### Second section\n",
        encoding="utf-8",
    )
    completed = _run("--write", str(source))
    assert completed.returncode == 0
    text = source.read_text(encoding="utf-8")
    assert "## 1. First section" in text
    assert "### 1.1. Child" in text
    assert "## 2. Second section" in text
    assert "0.0.1" not in text


def test_numbering_skips_fenced_code_comments(tmp_path):
    source = tmp_path / "fences.md"
    source.write_text(
        "# Title\n\n```bash\n# not a heading\n```\n\n## Real section\n",
        encoding="utf-8",
    )
    completed = _run("--write", str(source))
    assert completed.returncode == 0
    text = source.read_text(encoding="utf-8")
    assert "# not a heading" in text
    assert "## 1. Real section" in text
