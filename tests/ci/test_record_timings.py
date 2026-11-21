from __future__ import annotations

from pathlib import Path

from scripts.ci.record_timings import main


def write_junit(path: Path, classname: str, seconds: float) -> None:
    path.write_text(
        '<testsuite><testcase classname="' + classname + '" name="test" time="' + str(seconds) + '" /></testsuite>\n',
        encoding="utf-8",
    )


def test_one_junit_flag_accepts_multiple_reports(tmp_path: Path) -> None:
    root = tmp_path
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    (root / "tests" / "test_b.py").write_text("def test_b():\n    assert True\n", encoding="utf-8")
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    write_junit(first, "tests.test_a.Test", 1.5)
    write_junit(second, "tests.test_b.Test", 2.5)

    code = main(
        [
            "--root",
            str(root),
            "--junit",
            str(first),
            str(second),
            "--output",
            "timings.json",
            "--replace",
        ]
    )

    assert code == 0
    output = (root / "timings.json").read_text(encoding="utf-8")
    assert '"tests/test_a.py"' in output
    assert '"tests/test_b.py"' in output
