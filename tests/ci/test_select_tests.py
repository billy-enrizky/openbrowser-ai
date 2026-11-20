"""Tests for the diff-aware Python test selector."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.select_tests import select


def write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_repo(tmp_path: Path) -> Path:
    write(tmp_path, "src/openbrowser/__init__.py", "")
    write(tmp_path, "src/openbrowser/a.py", "VALUE = 1\n")
    write(tmp_path, "src/openbrowser/b.py", "from openbrowser.a import VALUE\n")
    write(tmp_path, "src/openbrowser/c.py", "from openbrowser.b import VALUE\n")
    write(
        tmp_path,
        "tests/test_c.py",
        "from openbrowser.c import VALUE\n\ndef test_value():\n    assert VALUE == 1\n",
    )
    write(tmp_path, "tests/test_other.py", "def test_other():\n    assert True\n")
    return tmp_path


def test_direct_import_is_selected(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, ["src/openbrowser/c.py"])
    assert result.mode == "selected"
    assert result.files == ["tests/test_c.py"]


def test_transitive_import_is_selected(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, ["src/openbrowser/a.py"])
    assert result.mode == "selected"
    assert "tests/test_c.py" in result.files
    assert "tests/test_other.py" not in result.files


def test_shared_input_selects_all(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    all_tests = ["tests/test_c.py", "tests/test_other.py"]
    result = select(root, ["pyproject.toml"])
    assert result.mode == "all"
    assert result.files == all_tests


def test_untested_module_selects_all(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "src/openbrowser/unused.py", "VALUE = 2\n")
    result = select(root, ["src/openbrowser/unused.py"])
    assert result.mode == "all"


def test_changed_test_selects_itself(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, ["tests/test_other.py"])
    assert result.mode == "selected"
    assert result.files == ["tests/test_other.py"]


def test_empty_diff_selects_nothing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, [])
    assert result.mode == "selected"
    assert result.files == []


def test_missing_source_selects_all(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, ["src/openbrowser/deleted.py"])
    assert result.mode == "all"


def test_non_python_source_selects_all(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "src/openbrowser/system_prompt.md", "prompt\n")
    result = select(root, ["src/openbrowser/system_prompt.md"])
    assert result.mode == "all"


def test_dynamic_string_import_is_selected(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(
        root,
        "tests/test_dynamic.py",
        'import importlib\n\nmodule = importlib.import_module("openbrowser.a")\n',
    )
    result = select(root, ["src/openbrowser/a.py"])
    assert "tests/test_dynamic.py" in result.files


def test_force_all_selects_every_test(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, [], force_all=True)
    assert result.mode == "all"
    assert result.files == ["tests/test_c.py", "tests/test_other.py"]


def test_real_repo_session_change_selects_browser_tests() -> None:
    root = Path(__file__).resolve().parents[2]
    result = select(root, ["src/openbrowser/browser/session.py"])
    assert result.mode == "selected"
    assert "tests/test_browser_session_coverage.py" in result.files
    assert len(result.files) < len(select(root, [], force_all=True).files)


def test_real_repo_leaf_change_selects_less_than_all() -> None:
    root = Path(__file__).resolve().parents[2]
    result = select(root, ["src/openbrowser/llm/oci_raw/chat.py"])
    assert result.mode == "selected"
    assert 0 < len(result.files) < len(select(root, [], force_all=True).files)


@pytest.mark.parametrize("path", [".github/workflows/test.yml", "uv.lock", "scripts/ci/shard_tests.py"])
def test_real_repo_shared_inputs_select_all(path: str) -> None:
    root = Path(__file__).resolve().parents[2]
    result = select(root, [path])
    assert result.mode == "all"


def test_backend_service_change_selects_its_test() -> None:
    root = Path(__file__).resolve().parents[2]
    result = select(root, ["backend/app/services/task_terminal.py"])
    assert result.mode == "selected"
    assert "tests/test_task_terminal.py" in result.files


def test_backend_main_change_selects_all() -> None:
    root = Path(__file__).resolve().parents[2]
    result = select(root, ["backend/app/main.py"])
    assert result.mode == "all"


def test_scripts_change_selects_all() -> None:
    root = Path(__file__).resolve().parents[2]
    result = select(root, ["scripts/ci/select_tests.py"])
    assert result.mode == "all"


def test_python_version_selects_all() -> None:
    root = Path(__file__).resolve().parents[2]
    result = select(root, [".python-version"])
    assert result.mode == "all"


def test_dot_slash_path_is_normalized(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, ["./src/openbrowser/c.py"])
    assert result.mode == "selected"
    assert result.files == ["tests/test_c.py"]


def test_absolute_path_inside_repo_is_normalized(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, [str(root / "src" / "openbrowser" / "c.py")])
    assert result.mode == "selected"
    assert result.files == ["tests/test_c.py"]


def test_absolute_path_outside_repo_selects_all(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, [str(tmp_path.parent / "outside.py")])
    assert result.mode == "all"


def test_parse_failure_selects_all(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "tests/test_broken.py", "def test_broken(:\n")
    result = select(root, ["src/openbrowser/a.py"])
    assert result.mode == "all"


def test_deleted_test_selects_all_remaining_tests(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = select(root, ["tests/test_deleted.py"])
    assert result.mode == "all"


def test_needs_mcp1_flag(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "tests/test_mcp_protocol.py", "def test_protocol():\n    assert True\n")
    selected = select(root, ["src/openbrowser/c.py"])
    full = select(root, [], force_all=True)
    assert selected.as_dict()["needs_mcp1"] is False
    assert full.as_dict()["needs_mcp1"] is True


def test_nested_package_initializer_selects_all() -> None:
    root = Path(__file__).resolve().parents[2]
    result = select(root, ["src/openbrowser/agent/__init__.py"])
    assert result.mode == "all"


def test_changed_test_helper_selects_dependent_tests(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "tests/helper.py", "VALUE = 1\n")
    write(root, "tests/test_a.py", "from tests.helper import VALUE\n")
    write(root, "tests/test_b.py", "from tests.helper import VALUE\n")
    result = select(root, ["tests/helper.py"])
    assert result.mode == "selected"
    assert set(result.files) == {"tests/test_a.py", "tests/test_b.py"}


def test_docs_only_change_selects_nothing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "docs/readme.md", "text\n")
    result = select(root, ["docs/readme.md"])
    assert result.mode == "selected"
    assert result.files == []


def test_frontend_change_selects_nothing(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "frontend/src/app/page.tsx", "export default 1\n")
    result = select(root, ["frontend/src/app/page.tsx"])
    assert result.mode == "selected"
    assert result.files == []


def test_unknown_python_path_selects_all(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "tools/new_tool.py", "VALUE = 1\n")
    result = select(root, ["tools/new_tool.py"])
    assert result.mode == "all"


def test_non_python_test_input_selects_all(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "tests/ci/fixtures/sample.json", "{}\n")
    result = select(root, ["tests/ci/fixtures/sample.json"])
    assert result.mode == "all"


def test_needs_mcp1_true_when_protocol_test_is_selected(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    write(root, "tests/test_mcp_protocol.py", "def test_protocol():\n    assert True\n")
    result = select(root, [], force_all=True)
    payload = result.as_dict()
    assert payload["needs_mcp1"] is True


@pytest.mark.parametrize("path", [".gitignore", "hk.pkl"])
def test_hook_config_selects_nothing(tmp_path: Path, path: str) -> None:
    root = make_repo(tmp_path)
    write(root, path, "text\n")
    result = select(root, [path])
    assert result.mode == "selected"
    assert result.files == []
