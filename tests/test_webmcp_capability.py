"""Focused checks for the separate WebMCP page-tool capability lane."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.interactive import webmcp_capability as lane


def test_exact_catalog_and_invocation_oracles() -> None:
    result = {"available": True, "catalog": [lane.TOOL], "invoked": lane.EXPECTED_RESULT}
    assert lane.task_results(result) == {"webmcp-discovery": True, "webmcp-invocation": True}
    result["invoked"] = {"content": [{"type": "text", "text": "41"}]}
    assert lane.task_results(result) == {"webmcp-discovery": True, "webmcp-invocation": False}


def test_unavailable_feature_is_explicit() -> None:
    assert lane.task_results({"available": False}) == {
        "webmcp-discovery": False,
        "webmcp-invocation": False,
    }


def test_changed_catalog_fails_discovery() -> None:
    tool = dict(lane.TOOL, description="different")
    result = {"available": True, "catalog": [tool], "invoked": lane.EXPECTED_RESULT}
    assert lane.task_results(result)["webmcp-discovery"] is False


def test_programs_use_the_same_page_api() -> None:
    url = "http://127.0.0.1:8123/"
    assert lane.PAGE_TASK in lane.python_program(url)
    assert lane.PAGE_TASK in lane.playwright_program(url)
    assert "document.modelContext" in lane.PAGE_TASK
    assert "executeTool(tool, {left: 17, right: 25})" in lane.PAGE_TASK
    assert "registerTool" in lane.FIXTURE


@pytest.mark.parametrize("feature", [False, True])
def test_configs_pin_chrome_and_flag(tmp_path: Path, feature: bool) -> None:
    chrome = Path("/example/chrome")
    ob, pw = lane.write_configs(tmp_path, chrome, feature)
    ob_profile = json.loads(ob.read_text())["browser_profile"]["capability"]
    pw_launch = json.loads(pw.read_text())["browser"]["launchOptions"]
    assert ob_profile["executable_path"] == pw_launch["executablePath"] == str(chrome)
    assert ("--enable-features=WebMCP" in ob_profile["args"]) is feature
    assert ob_profile["args"] == pw_launch["args"]
    assert "--use-mock-keychain" in ob_profile["args"]


def test_child_environment_is_keyless_and_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unused-sentinel")
    monkeypatch.setenv("PYTHONPATH", "/unrelated")
    env = lane.child_environment(tmp_path)
    assert "OPENAI_API_KEY" not in env
    assert "PYTHONPATH" not in env
    assert env["OPENBROWSER_SOCKET"] == str(tmp_path / "daemon.sock")
    assert env["HOME"] == str(tmp_path / "home")


def test_default_plan_does_not_import_or_start_browser() -> None:
    source = Path(lane.__file__).read_text()
    code = """
import json,sys
def guard(event, args):
    if event.startswith('socket.') or event == 'subprocess.Popen':
        raise SystemExit('unexpected live operation')
sys.addaudithook(guard)
namespace={'__name__':'__main__'}
exec(compile(SOURCE, 'capability.py', 'exec'), namespace)
""".replace("SOURCE", repr(source))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["arms"] == list(lane.ARMS)
    assert plan["tasks"] == ["webmcp-discovery", "webmcp-invocation"]
    assert plan["performance_claim"] is False


def test_native_output_parser_retains_exact_object() -> None:
    value = {"available": True, "catalog": [lane.TOOL], "invoked": lane.EXPECTED_RESULT}
    assert lane.last_json("native heading\n" + json.dumps(value) + "\n") == value
