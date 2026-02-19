"""Test verification functions for E2E LLM benchmark tasks."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.e2e_llm_benchmark import TASKS, SERVERS, aggregate_results, format_summary_table


def test_task_count():
    assert len(TASKS) == 6


def test_fact_lookup_verify():
    task = next(t for t in TASKS if t["name"] == "fact_lookup")
    assert task["verify"]("Python was created by Guido van Rossum in 1991.")
    assert task["verify"]("guido van rossum made Python around 1991")
    assert not task["verify"]("Python is a programming language")
    assert not task["verify"]("Guido van Rossum is a programmer")


def test_form_fill_verify():
    task = next(t for t in TASKS if t["name"] == "form_fill")
    assert task["verify"]("The form was submitted successfully. Response: custname=John")
    assert task["verify"]("I submitted the form and received a response")
    assert not task["verify"]("I could not find the form")


def test_multi_page_extract_verify():
    task = next(t for t in TASKS if t["name"] == "multi_page_extract")
    assert task["verify"](
        "1. How AI is changing education\n"
        "2. New study on climate\n"
        "3. Tech stocks rise today"
    )
    assert not task["verify"]("Here are the stories")


def test_search_navigate_verify():
    task = next(t for t in TASKS if t["name"] == "search_navigate")
    assert task["verify"]("Rust was originally developed by Mozilla Research")
    assert task["verify"]("The company that developed Rust is mozilla")
    assert not task["verify"]("Rust is a systems programming language")


def test_deep_navigation_verify():
    task = next(t for t in TASKS if t["name"] == "deep_navigation")
    assert task["verify"]("The latest release version is 1.2.3")
    assert task["verify"]("v0.1.17")
    assert not task["verify"]("Claude Code is a CLI tool")


def test_content_analysis_verify():
    task = next(t for t in TASKS if t["name"] == "content_analysis")
    assert task["verify"]("The page has 1 heading, 3 links, and 2 paragraphs")
    assert not task["verify"]("The page looks nice")


def test_server_configs():
    assert len(SERVERS) == 3
    assert "openbrowser" in SERVERS
    assert "playwright" in SERVERS
    assert "chrome-devtools" in SERVERS
    for name, config in SERVERS.items():
        assert "command" in config, f"{name} missing 'command'"
        assert "args" in config, f"{name} missing 'args'"


def test_aggregate_results():
    task_results = [
        {"name": "t1", "success": True, "duration_s": 10.0, "tool_calls": 3, "result": "ok", "error": None},
        {"name": "t2", "success": False, "duration_s": 20.0, "tool_calls": 5, "result": "fail", "error": None},
    ]
    summary = aggregate_results(task_results)
    assert summary["total_tasks"] == 2
    assert summary["passed"] == 1
    assert summary["total_duration_s"] == 30.0
    assert summary["total_tool_calls"] == 8
    assert summary["avg_tool_calls"] == 4.0


def test_format_summary_table():
    results = {
        "openbrowser": {
            "summary": {"total_tasks": 6, "passed": 6, "total_duration_s": 72.1, "total_tool_calls": 18, "avg_tool_calls": 3.0},
        },
        "playwright": {
            "summary": {"total_tasks": 6, "passed": 5, "total_duration_s": 145.3, "total_tool_calls": 24, "avg_tool_calls": 4.0},
        },
    }
    table = format_summary_table(results)
    assert "6/6" in table
    assert "5/6" in table
    assert "72.1" in table
