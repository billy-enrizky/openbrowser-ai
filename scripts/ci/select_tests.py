#!/usr/bin/env python3
"""Select the Python test files a change can reach.

The selector is deliberately conservative. A shared input, a deleted source
file, an unparseable file, an unreadable diff, an untested module, or an empty
selection for a code change selects the full suite. The merge queue always runs
the full suite.

Source roots:

- ``src/openbrowser`` maps to the ``openbrowser`` package.
- ``backend/app`` maps to the ``app`` package.
- ``scripts/ci`` maps to the ``scripts.ci`` package.

Package ``__init__`` re-export edges are excluded from the source graph. The
top-level ``openbrowser/__init__.py`` re-exports most of the package, and
following those edges would select almost every test for every deep change.
Changes that reach tests only through a re-export are caught by the
merge-queue full suite, not by the narrower PR selection. A changed
``__init__.py`` is a shared input and selects the full suite.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_ROOTS: dict[str, str] = {
    "src/openbrowser": "openbrowser",
    "backend/app": "app",
    "scripts/ci": "scripts.ci",
}
TEST_PREFIX = "tests"
SHARED_EXACT = {
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "tests/conftest.py",
    "src/openbrowser/__init__.py",
    "backend/app/main.py",
    "backend/pyproject.toml",
    "backend/uv.lock",
}
SHARED_PREFIXES = (
    "tests/support/",
    ".github/workflows/",
    ".github/ci-budgets.json",
    ".github/ci-baselines/",
    "scripts/ci/",
    "benchmarks/",
)
SOURCE_SUFFIX = ".py"
MCP1_TEST = "tests/test_mcp_protocol.py"
DOCS_PREFIXES = ("docs/", "local_docs/")
NON_PYTHON_PREFIXES = (
    "frontend/",
    "landing/",
    "presentation/",
    "infra/",
    "data/",
    "results/",
    "extension/",
    "plugin/",
    "env-for-ica/",
    "CSC490/",
    "STAD68/",
    "STAD80/",
)
NON_PYTHON_FILES = {".gitignore"}
NON_PYTHON_SUFFIXES = {
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".yml",
    ".yaml",
    ".json",
    ".sh",
    ".ps1",
    ".lock",
    ".pkl",
}


@dataclass
class Selection:
    files: list[str]
    mode: str
    reasons: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "files": self.files,
            "mode": self.mode,
            "reasons": self.reasons,
            "changed": self.changed,
            "needs_mcp1": MCP1_TEST in self.files,
        }


def _posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _clean(path: str, root: Path) -> str | None:
    value = _posix(path)
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            value = _posix(candidate.resolve().relative_to(root.resolve()))
        except ValueError:
            return None
    while value.startswith("./"):
        value = value[2:]
    return value or None


def _relative(path: Path, root: Path) -> str:
    try:
        return _posix(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return _posix(path)


def _module_name(path: Path, root: Path) -> str | None:
    resolved = path.resolve()
    if resolved.suffix != SOURCE_SUFFIX:
        return None
    for prefix, package in SOURCE_ROOTS.items():
        try:
            rel = resolved.relative_to((root / prefix).resolve())
        except ValueError:
            continue
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join([package, *parts]) if parts else package
    return None


def _ancestors(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _resolve_from(node: ast.ImportFrom, current_package: str) -> str:
    if node.level == 0:
        return node.module or ""
    parts = current_package.split(".") if current_package else []
    keep = max(0, len(parts) - (node.level - 1))
    base = ".".join(parts[:keep])
    if node.module:
        return f"{base}.{node.module}" if base else node.module
    return base


def _package_for_path(path: Path, root: Path) -> str:
    module = _module_name(path, root)
    if module:
        if path.name == "__init__.py":
            return module
        return module.rsplit(".", 1)[0] if "." in module else module
    rel = path.resolve().relative_to(root.resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    else:
        parts = parts[:-1]
    return ".".join(parts)


def _imports_from_file(path: Path, root: Path) -> set[str] | None:
    """Return every module name the file could import, or None on failure."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    package = _package_for_path(path, root)
    found: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(node, package)
            if base:
                names.append(base)
            for alias in node.names:
                if alias.name != "*" and base:
                    names.append(f"{base}.{alias.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if text.startswith("openbrowser.") or text == "openbrowser" or text.startswith("app.") or text == "app":
                names.append(text.split(":", 1)[0].split("/", 1)[0].removesuffix(".py"))
            elif text.startswith("src/openbrowser/"):
                names.append(text.removeprefix("src/").removesuffix(".py").replace("/", "."))
            elif text.startswith("backend/app/"):
                rel = text.removeprefix("backend/app/").removesuffix(".py")
                names.append("app." + rel.replace("/", "."))
        for name in names:
            if not name:
                continue
            for ancestor in _ancestors(name):
                found.add(ancestor)
    return found


def _source_modules(root: Path) -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for prefix in SOURCE_ROOTS:
        source_root = root / prefix
        if not source_root.exists():
            continue
        for path in source_root.rglob("*.py"):
            name = _module_name(path, root)
            if name:
                modules[name] = path
    return modules


def _source_graph(root: Path, modules: dict[str, Path]) -> tuple[dict[str, set[str]], list[str]]:
    graph: dict[str, set[str]] = {}
    failed: list[str] = []
    for name, path in modules.items():
        if path.name == "__init__.py":
            graph[name] = set()
            continue
        imports = _imports_from_file(path, root)
        if imports is None:
            failed.append(_relative(path, root))
            graph[name] = set()
            continue
        graph[name] = {imported for imported in imports if imported in modules}
    return graph, failed


def _reverse_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = {name: set() for name in graph}
    for module, dependencies in graph.items():
        for dependency in dependencies:
            reverse.setdefault(dependency, set()).add(module)
    return reverse


def _affected_modules(changed: Iterable[str], reverse: dict[str, set[str]], modules: dict[str, Path]) -> set[str]:
    affected: set[str] = set()
    queue: deque[str] = deque()
    for module in changed:
        if module in modules and module not in affected:
            affected.add(module)
            queue.append(module)
    while queue:
        module = queue.popleft()
        for importer in reverse.get(module, set()):
            if importer not in affected:
                affected.add(importer)
                queue.append(importer)
    return affected


def _test_files(root: Path) -> list[str]:
    tests_root = root / TEST_PREFIX
    if not tests_root.exists():
        return []
    return sorted(_relative(path, root) for path in tests_root.rglob("test_*.py"))


def _test_imports(root: Path, tests: Iterable[str]) -> tuple[dict[str, set[str]], list[str]]:
    imports: dict[str, set[str]] = {}
    failed: list[str] = []
    for test in tests:
        names = _imports_from_file(root / test, root)
        if names is None:
            failed.append(test)
            imports[test] = set()
        else:
            imports[test] = names
    return imports, failed


def _is_shared(path: str) -> bool:
    return (
        path.endswith("/__init__.py")
        or path in SHARED_EXACT
        or any(path.startswith(prefix) for prefix in SHARED_PREFIXES)
    )


def _is_non_python(path: str) -> bool:
    if path in NON_PYTHON_FILES:
        return True
    if path.startswith(DOCS_PREFIXES) or path.startswith(NON_PYTHON_PREFIXES):
        return True
    return Path(path).suffix.lower() in NON_PYTHON_SUFFIXES


def _test_module_names(path: str) -> set[str]:
    rel = path.removeprefix("tests/").removesuffix(".py")
    parts = rel.split("/")
    if parts and parts[-1] == "__init__":
        parts.pop()
    names: set[str] = set()
    for index in range(len(parts)):
        names.add(".".join(parts[index:]))
    if parts:
        names.add("tests." + ".".join(parts))
    return {name for name in names if name}


def _is_source(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in SOURCE_ROOTS)


def _is_deleted(root: Path, path: str, base_ref: str | None) -> bool:
    if base_ref is None:
        return False
    result = subprocess.run(
        ["git", "diff", "--name-status", f"{base_ref}...HEAD", "--", path],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return any(line.startswith("D\t") for line in result.stdout.splitlines())


def _changed_from_git(root: Path, base_ref: str) -> list[str] | None:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRD", f"{base_ref}...HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def select(
    root: Path,
    changed: Iterable[str],
    *,
    base_ref: str | None = None,
    force_all: bool = False,
    all_tests: Iterable[str] | None = None,
) -> Selection:
    root = root.resolve()
    tests = sorted(all_tests if all_tests is not None else _test_files(root))
    normalized: list[str] = []
    for item in changed:
        if not item.strip():
            continue
        cleaned = _clean(item, root)
        if cleaned is None:
            return Selection(tests, "all", [f"path is outside the repository: {item}"], [item])
        normalized.append(cleaned)
    normalized = sorted(set(normalized))
    reasons: list[str] = []

    if force_all:
        return Selection(tests, "all", ["forced full suite"], normalized)

    shared = [path for path in normalized if _is_shared(path)]
    if shared:
        reasons.append(f"shared input changed: {', '.join(shared)}")
        return Selection(tests, "all", reasons, normalized)

    source_paths = [path for path in normalized if _is_source(path)]
    test_paths = [path for path in normalized if path.startswith(TEST_PREFIX + "/") and path.endswith(".py")]
    selected_tests = {path for path in test_paths if path in tests}

    test_non_python = [path for path in normalized if path.startswith(TEST_PREFIX + "/") and not path.endswith(".py")]
    if test_non_python:
        reasons.append(f"non-Python test input changed: {', '.join(test_non_python)}")
        return Selection(tests, "all", reasons, normalized)

    non_python_source = [path for path in source_paths if not path.endswith(".py")]
    if non_python_source:
        reasons.append(f"non-Python source input changed: {', '.join(non_python_source)}")
        return Selection(tests, "all", reasons, normalized)

    deleted_sources = [path for path in source_paths if _is_deleted(root, path, base_ref)]
    if deleted_sources:
        reasons.append(f"source file deleted: {', '.join(deleted_sources)}")
        return Selection(tests, "all", reasons, normalized)

    missing_sources = [path for path in source_paths if not (root / path).exists()]
    if missing_sources:
        reasons.append(f"source path does not exist: {', '.join(missing_sources)}")
        return Selection(tests, "all", reasons, normalized)

    modules = _source_modules(root)
    changed_modules = {name for path in source_paths if (name := _module_name(root / path, root))}
    unknown = [
        path
        for path in source_paths
        if (root / path).is_file() and (name := _module_name(root / path, root)) is not None and name not in modules
    ]
    if unknown:
        reasons.append(f"source module is not in the import map: {', '.join(unknown)}")
        return Selection(tests, "all", reasons, normalized)

    imports: dict[str, set[str]] = {}
    if changed_modules or test_paths:
        imports, failed_tests = _test_imports(root, tests)
        if failed_tests:
            reasons.append(f"test parse failed: {', '.join(failed_tests)}")
            return Selection(tests, "all", reasons, normalized)

    if changed_modules:
        graph, failed_sources = _source_graph(root, modules)
        if failed_sources:
            reasons.append(f"source parse failed: {', '.join(failed_sources)}")
            return Selection(tests, "all", reasons, normalized)
        reverse = _reverse_graph(graph)
        affected = _affected_modules(changed_modules, reverse, modules)
        reached = {test for test, names in imports.items() if names & affected}
        if not reached:
            reasons.append("no test imports an affected source module")
            return Selection(tests, "all", reasons, normalized)
        selected_tests |= reached
        reasons.append(f"{len(changed_modules)} source modules affected {len(affected)} modules")

    if test_paths:
        changed_test_modules: set[str] = set()
        for path in test_paths:
            changed_test_modules |= _test_module_names(path)
        dependent = {test for test, names in imports.items() if names & changed_test_modules}
        selected_tests |= dependent
        reasons.append(f"{len(changed_test_modules)} test module names affected {len(dependent)} test files")

    if not normalized:
        reasons.append("empty diff")
        return Selection([], "selected", reasons, normalized)

    if not selected_tests:
        if all(_is_non_python(path) for path in normalized):
            reasons.append("no Python tests apply to this diff")
            return Selection([], "selected", reasons, normalized)
        reasons.append("unclassified paths; selecting all tests")
        return Selection(tests, "all", reasons, normalized)

    reasons.append(f"selected {len(selected_tests)} of {len(tests)} test files")
    return Selection(sorted(selected_tests), "selected", reasons, normalized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--base", default=None, help="git base ref for the diff")
    parser.add_argument("--changed-file", action="append", default=[], help="changed path")
    parser.add_argument("--changed-files-from", default=None, help="file with changed paths")
    parser.add_argument("--force-all", action="store_true", help="select the full suite")
    parser.add_argument("--output", default=None, help="write the selection JSON here")
    parser.add_argument("--summary", action="store_true", help="print a one-line summary")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    changed = list(args.changed_file)
    if args.changed_files_from:
        path = Path(args.changed_files_from)
        if not path.exists():
            print(f"cannot read {path}; selecting all tests", file=sys.stderr)
            args.force_all = True
        else:
            changed.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if args.base and not changed and not args.force_all:
        from_git = _changed_from_git(root, args.base)
        if from_git is None:
            print(
                f"cannot read git diff against {args.base}; selecting all tests",
                file=sys.stderr,
            )
            args.force_all = True
        else:
            changed = from_git

    result = select(root, changed, base_ref=args.base, force_all=args.force_all)
    payload = result.as_dict()
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.summary:
        print(
            f"mode={result.mode} files={len(result.files)} reasons={'; '.join(result.reasons)}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
