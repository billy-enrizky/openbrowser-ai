"""Differential tests for the stateless MCP schema validator.

The stateless tools must validate their arguments without importing
``jsonschema``. These tests compare the strict hand-written validator against
``Draft202012Validator`` for both advertised schemas, including integer
semantics, bounds, Unicode lengths, arrays, required properties, additional
properties, JSON path formatting, and error text.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openbrowser.mcp.catalog import load_tool_definitions
from openbrowser.mcp.schema import UnsupportedSchemaError, compile_schema, iter_errors

REPO_ROOT = Path(__file__).parents[1]
_STATELESS_TOOLS = ("web_search", "scrape_web")


def _schemas() -> dict[str, dict[str, Any]]:
    return {
        tool["name"]: tool["inputSchema"]
        for tool in load_tool_definitions(modern=True)
        if tool["name"] in _STATELESS_TOOLS
    }


def _ours(schema: dict[str, Any], instance: Any) -> Any:
    return next(iter_errors(instance, compile_schema(schema)), None)


def _theirs(schema: dict[str, Any], instance: Any) -> Any:
    return next(Draft202012Validator(schema).iter_errors(instance), None)


def _assert_parity(schema: dict[str, Any], instance: Any) -> None:
    theirs = _theirs(schema, instance)
    ours = _ours(schema, instance)
    assert (ours is None) == (theirs is None), f"acceptance mismatch for {instance!r}: ours={ours} theirs={theirs}"
    if ours is not None and theirs is not None:
        assert (ours.json_path, ours.message) == (theirs.json_path, theirs.message), f"error mismatch for {instance!r}"


_CURATED: dict[str, list[Any]] = {
    "web_search": [
        {},
        {"query": ""},
        {"query": "x"},
        {"query": 123},
        {"query": True},
        {"query": False},
        {"query": None},
        {"query": ["x"]},
        {"query": {"a": 1}},
        {"query": "x" * 2048},
        {"query": "x" * 2049},
        {"query": "\u00e9" * 2048},
        {"query": "\U0001f600" * 2048},
        {"query": "\U0001f600" * 2049},
        {"query": "ok", "extra": 1},
        {"query": "ok", "a": 1, "b": 2},
        {"query": "ok", "zz": 1, "aa": 2, "mm": 3},
        {"query": "ok", "max_results": 0},
        {"query": "ok", "max_results": 1},
        {"query": "ok", "max_results": 20},
        {"query": "ok", "max_results": 21},
        {"query": "ok", "max_results": -5},
        {"query": "ok", "max_results": 1.0},
        {"query": "ok", "max_results": 2.0},
        {"query": "ok", "max_results": 1.5},
        {"query": "ok", "max_results": True},
        {"query": "ok", "max_results": "5"},
        {"query": "ok", "max_results": 10**30},
        {"query": "ok", "max_results": float("nan")},
        {"query": "ok", "max_results": float("inf")},
        {"query": "ok", "max_results": float("-inf")},
        {"query": "ok", "backend": ""},
        {"query": "ok", "backend": "auto"},
        {"query": "ok", "backend": "x" * 256},
        {"query": "ok", "backend": "x" * 257},
        {"query": "ok", "safe_search": "on"},
        {"query": "ok", "safe_search": "moderate"},
        {"query": "ok", "safe_search": "off"},
        {"query": "ok", "safe_search": "maybe"},
        {"query": "ok", "safe_search": 3},
        {"query": "ok", "safe_search": None},
        {"query": "ok", "region": ""},
        {"query": "ok", "region": "us-en"},
        {"query": "ok", "region": "x" * 32},
        {"query": "ok", "region": "x" * 33},
        {"query": "ok", "timelimit": None},
        {"query": "ok", "timelimit": "d"},
        {"query": "ok", "timelimit": "x" * 64},
        {"query": "ok", "timelimit": "x" * 65},
        {"query": "ok", "timelimit": 5},
    ],
    "scrape_web": [
        {},
        {"url": ""},
        {"url": "https://example.com/"},
        {"url": "not-a-url"},
        {"url": 42},
        {"url": True},
        {"url": None},
        {"url": "x" * 8192},
        {"url": "x" * 8193},
        {"url": "https://example.com/", "extra": True},
        {"url": "https://example.com/", "format": "markdown"},
        {"url": "https://example.com/", "format": "text"},
        {"url": "https://example.com/", "format": "pdf"},
        {"url": "https://example.com/", "format": 3},
        {"url": "https://example.com/", "selector": ""},
        {"url": "https://example.com/", "selector": "#main"},
        {"url": "https://example.com/", "selector": None},
        {"url": "https://example.com/", "extract_links": True},
        {"url": "https://example.com/", "extract_links": False},
        {"url": "https://example.com/", "extract_links": 0},
        {"url": "https://example.com/", "extract_links": 1},
        {"url": "https://example.com/", "extract_links": "true"},
        {"url": "https://example.com/", "only_main_content": True},
        {"url": "https://example.com/", "only_main_content": 1},
        {"url": "https://example.com/", "only_main_content": "yes"},
        {"url": "https://example.com/", "include_tags": []},
        {"url": "https://example.com/", "include_tags": ["a"]},
        {"url": "https://example.com/", "include_tags": ["a", "b"]},
        {"url": "https://example.com/", "include_tags": ["a", 1]},
        {"url": "https://example.com/", "include_tags": [True]},
        {"url": "https://example.com/", "include_tags": [None]},
        {"url": "https://example.com/", "include_tags": [[]]},
        {"url": "https://example.com/", "include_tags": "a"},
        {"url": "https://example.com/", "exclude_tags": []},
        {"url": "https://example.com/", "exclude_tags": ["script"]},
        {"url": "https://example.com/", "timeout_ms": 0},
        {"url": "https://example.com/", "timeout_ms": 1},
        {"url": "https://example.com/", "timeout_ms": 120000},
        {"url": "https://example.com/", "timeout_ms": 120001},
        {"url": "https://example.com/", "timeout_ms": 1.0},
        {"url": "https://example.com/", "timeout_ms": 1.5},
        {"url": "https://example.com/", "timeout_ms": True},
        {"url": "https://example.com/", "timeout_ms": float("nan")},
        {"url": "https://example.com/", "timeout_ms": float("inf")},
        {"url": "https://example.com/", "timeout_ms": float("-inf")},
        {"url": "https://example.com/", "max_chars": -1},
        {"url": "https://example.com/", "max_chars": 0},
        {"url": "https://example.com/", "max_chars": 512},
        {"url": "https://example.com/", "max_chars": 1.0},
        {"url": "https://example.com/", "max_chars": 1.5},
        {"url": "https://example.com/", "max_chars": float("nan")},
        {"url": "https://example.com/", "max_chars": float("inf")},
        {"include_tags": [], "max_chars": 3.5},
        {"include_tags": ["a", 1], "max_chars": 3.5},
        {"url": "https://example.com/", "include_tags": ["a", 1], "max_chars": 3.5},
        {"url": "https://example.com/", "extract_links": "x", "format": "pdf"},
        {"url": 5, "max_chars": -1},
        {"url": "https://example.com/", "format": 3, "extra": 1},
    ],
}


@pytest.mark.parametrize("tool", _STATELESS_TOOLS)
def test_curated_instances_match_draft202012(tool: str) -> None:
    schema = _schemas()[tool]
    for instance in _CURATED[tool]:
        _assert_parity(schema, instance)


def _generated_search_instances() -> list[dict[str, Any]]:
    pool: dict[str, list[Any]] = {
        "query": [None, "", "x", "q" * 2048, "q" * 2049, 1, 1.0, True, [], {}],
        "max_results": [None, 0, 1, 20, 21, 1.0, 1.5, True, "5", 10**25, float("nan"), float("inf")],
        "backend": [None, "", "auto", "x" * 256, "x" * 257, 3],
        "region": [None, "", "us-en", "x" * 32, "x" * 33],
        "safe_search": [None, "on", "moderate", "off", "maybe", 3, True],
        "timelimit": [None, "d", "x" * 64, "x" * 65, 5],
    }
    rng = random.Random(20261125)
    instances: list[dict[str, Any]] = []
    for _ in range(300):
        instance: dict[str, Any] = {}
        for name, values in pool.items():
            if rng.random() < 0.75:
                instance[name] = rng.choice(values)
        if rng.random() < 0.1:
            instance["extra"] = rng.choice([1, "x", None, {}])
        if "query" in instance and instance["query"] is None:
            del instance["query"]
        instances.append(instance)
    return instances


def _generated_scrape_instances() -> list[dict[str, Any]]:
    tags = [None, [], ["a"], ["a", "b"], ["a", 1], [True], "a", [""]]
    pool: dict[str, list[Any]] = {
        "url": [None, "", "https://e/", "x" * 8192, "x" * 8193, 42, True],
        "format": [None, "markdown", "text", "html", "links", "json", "pdf", 3],
        "selector": [None, "", "#main", 5],
        "extract_links": [None, True, False, 0, 1, "true"],
        "only_main_content": [None, True, False, 0, 1, "yes"],
        "include_tags": tags,
        "exclude_tags": tags,
        "timeout_ms": [None, 0, 1, 120000, 120001, 1.0, 1.5, True, float("nan"), float("inf")],
        "max_chars": [None, -1, 0, 512, 1.0, 1.5, float("nan"), float("inf")],
    }
    rng = random.Random(20261126)
    instances: list[dict[str, Any]] = []
    for _ in range(300):
        instance: dict[str, Any] = {}
        for name, values in pool.items():
            if rng.random() < 0.75:
                instance[name] = rng.choice(values)
        if rng.random() < 0.1:
            instance["extra"] = rng.choice([1, "x", None, {}])
        if "url" in instance and instance["url"] is None:
            del instance["url"]
        instances.append(instance)
    return instances


@pytest.mark.parametrize("tool", _STATELESS_TOOLS)
def test_generated_instances_match_draft202012(tool: str) -> None:
    schema = _schemas()[tool]
    instances = _generated_search_instances() if tool == "web_search" else _generated_scrape_instances()
    for instance in instances:
        _assert_parity(schema, instance)


def test_integer_semantics_match_json_schema() -> None:
    schema = _schemas()["web_search"]
    assert _ours(schema, {"query": "x", "max_results": 1.0}) is None
    assert _ours(schema, {"query": "x", "max_results": 2.0}) is None

    fractional = _ours(schema, {"query": "x", "max_results": 1.5})
    assert fractional is not None
    assert fractional.json_path == "$.max_results"
    assert fractional.message == "1.5 is not of type 'integer'"

    for value in (True, False):
        rejection = _ours(schema, {"query": "x", "max_results": value})
        assert rejection is not None
        assert rejection.message == f"{value!r} is not of type 'integer'"

    for value, rendered in ((float("nan"), "nan"), (float("inf"), "inf")):
        rejection = _ours(schema, {"query": "x", "max_results": value})
        assert rejection is not None
        assert rejection.message == f"{rendered} is not of type 'integer'"

    negative_infinity = _ours(schema, {"query": "x", "max_results": float("-inf")})
    assert negative_infinity is not None
    assert negative_infinity.message == "-inf is less than the minimum of 1"


@pytest.mark.parametrize("value", [10**400, -(10**400), 10**308, 10**309])
def test_huge_integers_match_draft202012(value: int) -> None:
    schemas = _schemas()
    instances = {
        "web_search": {"query": "x", "max_results": value},
        "scrape_web": {"url": "https://e/", "max_chars": value},
    }
    for tool, instance in instances.items():
        _assert_parity(schemas[tool], instance)

    if value > 0:
        assert _ours(schemas["web_search"], instances["web_search"]) is None
        assert _ours(schemas["scrape_web"], instances["scrape_web"]) is None
    else:
        rejection = _ours(schemas["web_search"], instances["web_search"])
        assert rejection is not None
        assert rejection.message == f"{value!r} is less than the minimum of 1"


def test_unicode_length_bounds_use_code_points() -> None:
    schema = _schemas()["web_search"]
    assert _ours(schema, {"query": "\u00e9" * 2048}) is None
    assert _ours(schema, {"query": "\U0001f600" * 2048}) is None
    rejected = _ours(schema, {"query": "\U0001f600" * 2049})
    assert rejected is not None
    assert rejected.json_path == "$.query"
    assert rejected.message.endswith("' is too long")


def test_additional_properties_message_is_sorted() -> None:
    schema = _schemas()["web_search"]
    rejection = _ours(schema, {"query": "ok", "zz": 1, "aa": 2, "mm": 3})
    assert rejection is not None
    assert rejection.json_path == "$"
    assert rejection.message == "Additional properties are not allowed ('aa', 'mm', 'zz' were unexpected)"

    single = _ours(schema, {"query": "ok", "extra": 1})
    assert single is not None
    assert single.message == "Additional properties are not allowed ('extra' was unexpected)"


def test_additional_properties_plain_sorted_repro() -> None:
    schema = _schemas()["web_search"]
    rejection = _ours(schema, {"query": "ok", "a": 1, "b'quote": 2})
    assert rejection is not None
    assert rejection.json_path == "$"
    assert rejection.message == "Additional properties are not allowed ('a', \"b'quote\" were unexpected)"


@pytest.mark.parametrize(
    "extra_keys",
    [
        ["a", "b'quote"],
        ["a", 'he"llo'],
        ["a", "\x01"],
        ["z", "\u00e9"],
        ["a", "\U0001f600"],
        ["", "a"],
        ["10", "2"],
    ],
)
def test_additional_properties_order_matches_draft202012(extra_keys: list[str]) -> None:
    schema = _schemas()["web_search"]
    instance: dict[str, Any] = {"query": "ok"}
    for key in extra_keys:
        instance[key] = 1
    _assert_parity(schema, instance)


def test_nested_item_error_path_and_min_items() -> None:
    schema = _schemas()["scrape_web"]
    empty = _ours(schema, {"url": "https://e/", "include_tags": []})
    assert empty is not None
    assert (empty.json_path, empty.message) == ("$.include_tags", "[] should be non-empty")

    item = _ours(schema, {"url": "https://e/", "include_tags": ["a", 1]})
    assert item is not None
    assert (item.json_path, item.message) == ("$.include_tags[1]", "1 is not of type 'string'")


def test_bounds_are_inclusive() -> None:
    search = _schemas()["web_search"]
    scrape = _schemas()["scrape_web"]
    assert _ours(search, {"query": "x", "max_results": 1}) is None
    minimum = _ours(search, {"query": "x", "max_results": 0})
    assert minimum is not None
    assert minimum.message == "0 is less than the minimum of 1"

    assert _ours(scrape, {"url": "https://e/", "timeout_ms": 120000}) is None
    maximum = _ours(scrape, {"url": "https://e/", "timeout_ms": 120001})
    assert maximum is not None
    assert maximum.message == "120001 is greater than the maximum of 120000"

    non_negative = _ours(scrape, {"url": "https://e/", "max_chars": 0})
    assert non_negative is None


def test_enum_numbers_are_distinct_from_booleans() -> None:
    schema = {"type": "object", "properties": {"value": {"enum": [1]}}}
    assert _ours(schema, {"value": 1}) is None
    assert _ours(schema, {"value": 1.0}) is None
    rejection = _ours(schema, {"value": True})
    assert rejection is not None
    assert (rejection.json_path, rejection.message) == ("$.value", "True is not one of [1]")


def test_enum_booleans_are_distinct_from_numbers() -> None:
    schema = {"type": "object", "properties": {"value": {"enum": [True]}}}
    assert _ours(schema, {"value": True}) is None
    for value in (1, 0, 1.0, False):
        rejection = _ours(schema, {"value": value})
        assert rejection is not None, value
        assert rejection.message == f"{value!r} is not one of [True]"


def test_enum_none_is_distinct_from_false_and_numbers() -> None:
    schema = {"type": "object", "properties": {"value": {"enum": [None]}}}
    assert _ours(schema, {"value": None}) is None
    for value in (False, 0, ""):
        rejection = _ours(schema, {"value": value})
        assert rejection is not None, value
        assert rejection.message == f"{value!r} is not one of [None]"


def test_enum_strings_compare_exactly() -> None:
    schema = {"type": "object", "properties": {"value": {"enum": ["1"]}}}
    assert _ours(schema, {"value": "1"}) is None
    rejection = _ours(schema, {"value": 1})
    assert rejection is not None
    assert rejection.message == "1 is not one of ['1']"


@pytest.mark.parametrize("entry", [[1], {"a": 1}, ("x",), {1, 2}])
def test_enum_non_scalar_entries_fail_closed(entry: Any) -> None:
    with pytest.raises(UnsupportedSchemaError):
        compile_schema({"type": "object", "properties": {"value": {"enum": [entry]}}})


@pytest.mark.parametrize(
    "unsupported",
    [
        {"type": "object", "patternProperties": {"^x": {"type": "string"}}},
        {"type": "object", "$ref": "#/$defs/x"},
        {"type": "object", "exclusiveMinimum": 0},
        {"type": "object", "additionalProperties": {"type": "string"}},
        {"type": "object", "properties": {"a": {"type": "string", "format": "uri"}}},
        {"type": "object", "properties": {"a": {"type": "string", "items": []}}},
        {"type": "object", "properties": {"a": {"type": "null"}}},
        {"type": "object", "items": [{"type": "string"}]},
        {"type": "object", "minProperties": 1},
        {"type": "object", "uniqueItems": True},
    ],
)
def test_unknown_schema_keywords_fail_closed(unsupported: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedSchemaError):
        compile_schema(unsupported)


def test_unknown_stateless_tool_still_raises() -> None:
    from openbrowser.mcp.stateless import validate_stateless_arguments

    with pytest.raises(ValueError, match="Unknown stateless tool: nope"):
        validate_stateless_arguments("nope", {})


@pytest.mark.parametrize("tool", _STATELESS_TOOLS)
def test_validate_stateless_arguments_message_matches_jsonschema(tool: str) -> None:
    from openbrowser.mcp.stateless import validate_stateless_arguments

    schema = _schemas()[tool]
    for instance in _CURATED[tool]:
        ours = _ours(schema, instance)
        try:
            validate_stateless_arguments(tool, instance)
        except ValueError as error:
            assert ours is not None, instance
            location = f" at {ours.json_path}" if ours.json_path != "$" else ""
            assert str(error) == f"Invalid arguments{location}: {ours.message}"
        else:
            assert ours is None, instance


def test_stateless_validation_does_not_import_jsonschema() -> None:
    script = """
import sys

from openbrowser.mcp.stateless import validate_stateless_arguments

validate_stateless_arguments("web_search", {"query": "x"})
validate_stateless_arguments("scrape_web", {"url": "https://example.com/"})
banned = sorted(
    name
    for name in sys.modules
    if name == "jsonschema" or name.startswith("jsonschema.") or name == "rpds" or name.startswith("rpds.")
)
print("loaded:" + ",".join(banned) if banned else "clean")
"""
    env = {
        **os.environ,
        "ANONYMIZED_TELEMETRY": "false",
        "OPENBROWSER_DISABLE_TELEMETRY": "true",
        "OPENBROWSER_SETUP_LOGGING": "false",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean", completed.stdout


def test_legacy_search_web_alias_uses_canonical_schema():
    from openbrowser.mcp.stateless import validate_stateless_arguments

    validate_stateless_arguments("search_web", {"query": "x", "max_results": 2})
