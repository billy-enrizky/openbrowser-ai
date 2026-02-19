"""Test verification functions for E2E LLM benchmark tasks."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.e2e_llm_benchmark import TASKS


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
