"""Differential and fail-closed tests for stateful MCP schemas."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, SchemaError

from openbrowser.mcp.catalog import load_tool_definitions
from openbrowser.mcp.schema import UnsupportedSchemaError, compile_schema, iter_errors

ROOT = Path(__file__).parents[1]
_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen


@pytest.fixture(autouse=True)
def _no_native_io(monkeypatch):
    import socket

    import psutil

    def forbidden(*args, **kwargs):
        raise AssertionError("schema test attempted native I/O")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)


def _variants():
    for compact in (False, True):
        for modern in (False, True):
            yield compact, modern, load_tool_definitions(modern=modern, compact=compact)


def _valid_value(schema: dict[str, Any]) -> Any:
    if "enum" in schema:
        return schema["enum"][0]
    type_name = schema.get("type")
    if isinstance(type_name, list):
        type_name = next((value for value in type_name if value != "null"), type_name[0])
    if type_name == "string":
        return "x"
    if type_name == "integer":
        return 1
    if type_name == "boolean":
        return True
    if type_name == "array":
        return ["x"]
    if type_name == "object":
        return {}
    if type_name == "null":
        return None
    return "x"


def _invalid_values(schema: dict[str, Any]) -> list[Any]:
    values: list[Any] = [None, True, False, 0, 1, 1.5, "x", [], {}]
    if "enum" in schema:
        values.append("__not_enum__")
    type_name = schema.get("type")
    if type_name == "string" or (isinstance(type_name, list) and "string" in type_name):
        values.extend(["", "x" * 10000, 5])
    if type_name == "integer":
        values.extend([1.5, "1", 10**30, math.nan, math.inf])
    if type_name == "boolean":
        values.extend([1, 0, "true"])
    if type_name == "array":
        values.extend(["x", [1], [None]])
    if type_name == "null":
        values.extend(["x", 0, False])
    return values


def _instances_for(schema: dict[str, Any]) -> list[Any]:
    if schema.get("type") != "object":
        return [None, True, 0, 1.5, "x", [], {}]
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    instances: list[Any] = [{}, {"extra": 1}]
    valid = {name: _valid_value(subschema) for name, subschema in properties.items()}
    instances.append(valid)
    for name in required:
        missing = dict(valid)
        missing.pop(name, None)
        instances.append(missing)
    for name, subschema in properties.items():
        for value in _invalid_values(subschema):
            instance = dict(valid)
            instance[name] = value
            instances.append(instance)
    for value in (None, True, 1, 1.5, "x", [], {}):
        instances.append(value)
    return instances


def _assert_parity(schema: dict[str, Any], instance: Any, compiled=None) -> None:
    ours = next(iter_errors(instance, compiled or compile_schema(schema)), None)
    theirs = next(Draft202012Validator(schema).iter_errors(instance), None)
    assert (ours is None) == (theirs is None), f"acceptance mismatch for {instance!r}: ours={ours} theirs={theirs}"
    if ours is not None and theirs is not None:
        assert (ours.json_path, ours.message) == (theirs.json_path, theirs.message), (
            f"error mismatch for {instance!r}: ours={(ours.json_path, ours.message)} "
            f"theirs={(theirs.json_path, theirs.message)}"
        )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("modern", [False, True])
def test_every_catalog_schema_compiles_and_meta_validates(compact: bool, modern: bool):
    tools = load_tool_definitions(modern=modern, compact=compact)
    assert tools
    for tool in tools:
        Draft202012Validator.check_schema(tool["inputSchema"])
        compile_schema(tool["inputSchema"])
        output = tool.get("outputSchema")
        if output is not None:
            Draft202012Validator.check_schema(output)
            compile_schema(output, allow_null=True)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("modern", [False, True])
def test_all_catalog_inputs_match_draft202012(compact: bool, modern: bool):
    for tool in load_tool_definitions(modern=modern, compact=compact):
        schema = tool["inputSchema"]
        compiled = compile_schema(schema)
        for instance in _instances_for(schema):
            _assert_parity(schema, instance, compiled)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("modern", [False, True])
def test_all_catalog_outputs_match_draft202012(compact: bool, modern: bool):
    for tool in load_tool_definitions(modern=modern, compact=compact):
        schema = tool.get("outputSchema")
        if schema is None:
            continue
        compiled = compile_schema(schema, allow_null=True)
        instances = _instances_for(schema)
        instances.append({"context_id": "ctx", "status": "closed", "retryable": True, "error_code": None})
        for instance in instances:
            _assert_parity(schema, instance, compiled)


@pytest.mark.parametrize("instance", [None, "x", 1, True, [], {}])
def test_null_type_matches_draft202012(instance: Any):
    schema = {"type": ["string", "null"]}
    _assert_parity(schema, instance, compile_schema(schema, allow_null=True))


@pytest.mark.parametrize("schema", [{"type": "null"}, {"type": ["string", "null"]}])
def test_null_type_is_rejected_without_output_opt_in(schema: dict[str, Any]):
    with pytest.raises(UnsupportedSchemaError):
        compile_schema(schema)


_MALFORMED = (
    {"type": None},
    {"type": []},
    {"type": ["string", "string"]},
    {"type": ["string", 1]},
    {"type": 1},
    {"type": "invalid"},
    {"required": None},
    {"required": ["a", "a"]},
    {"required": [1]},
    {"properties": None},
    {"properties": []},
    {"title": 1},
    {"title": None},
    {"description": []},
    {"description": None},
    {"minimum": None},
    {"minLength": None},
    {"minItems": None},
    {"items": None},
    {"items": []},
    {"enum": None},
    {"enum": "x"},
    {"additionalProperties": None},
)


@pytest.mark.parametrize("schema", _MALFORMED)
def test_malformed_supported_schemas_are_rejected_by_both(schema: dict[str, Any]):
    with pytest.raises(SchemaError):
        Draft202012Validator.check_schema(schema)
    with pytest.raises(UnsupportedSchemaError):
        compile_schema(schema)


_VALID_SUPPORTED = (
    {"type": "string", "title": "title", "description": "description", "default": None},
    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False},
    {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
    {"enum": ["a", "b"]},
    {"type": "integer", "minimum": 0, "maximum": 10},
)

_VALID_OUTPUT_SUPPORTED = (
    {"type": ["string", "null"]},
    {"type": "null"},
)


@pytest.mark.parametrize("schema", _VALID_SUPPORTED)
def test_valid_supported_schemas_are_accepted_by_both(schema: dict[str, Any]):
    Draft202012Validator.check_schema(schema)
    compile_schema(schema)


@pytest.mark.parametrize("schema", _VALID_OUTPUT_SUPPORTED)
def test_valid_output_schemas_are_accepted_by_both(schema: dict[str, Any]):
    Draft202012Validator.check_schema(schema)
    compile_schema(schema, allow_null=True)


_VALID_UNIMPLEMENTED = (
    {"additionalProperties": {}},
    {"items": True},
    {"type": "object", "properties": {"a": True}},
    {"oneOf": [{"type": "string"}]},
)


@pytest.mark.parametrize("schema", _VALID_UNIMPLEMENTED)
def test_valid_but_unimplemented_schemas_fail_closed(schema: dict[str, Any]):
    Draft202012Validator.check_schema(schema)
    with pytest.raises(UnsupportedSchemaError):
        compile_schema(schema)


@pytest.fixture()
def stateful_server(monkeypatch, isolated_config_path):
    from openbrowser.mcp import server as server_module

    monkeypatch.setattr(server_module, "TELEMETRY_AVAILABLE", False)
    return server_module.OpenBrowserServer(install_sdk=False)


@pytest.mark.asyncio
async def test_dispatch_invalid_argument_text_matches_draft(stateful_server):
    from openbrowser.mcp import server as server_module

    outcome = await stateful_server.dispatch_tool(
        "execute_code",
        {"context_id": "ctx", "code": 5},
        server_module.MODERN_PROTOCOL_VERSION,
    )
    assert outcome.is_error
    assert outcome.content[0]["text"] == "Invalid arguments at $.code: 5 is not of type 'string'"

    missing = await stateful_server.dispatch_tool("get_status", {}, server_module.MODERN_PROTOCOL_VERSION)
    assert missing.is_error
    assert missing.content[0]["text"] == "Invalid arguments: 'context_id' is a required property"

    extra = await stateful_server.dispatch_tool(
        "get_status", {"context_id": "ctx", "extra": 1}, server_module.MODERN_PROTOCOL_VERSION
    )
    assert extra.is_error
    assert extra.content[0]["text"] == (
        "Invalid arguments: Additional properties are not allowed ('extra' was unexpected)"
    )


@pytest.mark.asyncio
async def test_dispatch_alias_and_unknown_tool(stateful_server):
    from openbrowser.mcp import server as server_module

    alias = await stateful_server.dispatch_tool("search_web", {"query": ""}, None)
    canonical = await stateful_server.dispatch_tool("web_search", {"query": ""}, None)
    assert alias.is_error and canonical.is_error
    assert alias.content[0]["text"] == canonical.content[0]["text"]
    assert alias.content[0]["text"] == "Invalid arguments at $.query: '' should be non-empty"

    with pytest.raises(server_module.ProtocolRpcError) as raised:
        await stateful_server.dispatch_tool("missing_tool", {}, server_module.MODERN_PROTOCOL_VERSION)
    assert raised.value.code == -32602
    assert str(raised.value) == "Unknown tool: missing_tool"


@pytest.mark.asyncio
async def test_sdk_handler_preserves_invalid_argument_text(monkeypatch, isolated_config_path):
    from mcp import Client

    from openbrowser.mcp import server as server_module

    monkeypatch.setattr(server_module, "TELEMETRY_AVAILABLE", False)
    server = server_module.OpenBrowserServer()
    async with Client(server.server) as client:
        tools = await client.list_tools(cache_mode="refresh")
        names = {tool.name for tool in tools.tools}
        expected = {
            "create_context",
            "execute_code",
            "get_status",
            "reset_browser",
            "close_context",
            "web_search",
            "scrape_web",
        }
        assert expected <= names
        result = await client.call_tool("execute_code", {"context_id": "ctx", "code": 5})
        assert result.is_error
        assert result.content[0].type == "text"
        assert result.content[0].text == "Invalid arguments at $.code: 5 is not of type 'string'"


def test_sdk_raw_schema_is_an_immutable_snapshot(monkeypatch, isolated_config_path):
    from openbrowser.mcp import server as server_module

    monkeypatch.setattr(server_module, "TELEMETRY_AVAILABLE", False)
    server = server_module.OpenBrowserServer()
    raw = server.server.get_tool_input_schema("execute_code")
    assert raw is not None
    raw["properties"]["code"]["type"] = "number"
    fresh = server.server.get_tool_input_schema("execute_code")
    assert fresh is not None
    assert fresh["properties"]["code"]["type"] == "string"
    assert len(server._legacy_tools) == 4
    assert len(server._modern_tools) == 7


def test_fresh_process_constructor_does_not_import_jsonschema(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _REAL_RUN)
    monkeypatch.setattr(subprocess, "Popen", _REAL_POPEN)
    script = """
import sys
sys.argv = ['openbrowser-ai', '--mcp']
import openbrowser.mcp.server as server
server.OpenBrowserServer(install_sdk=False)
banned = sorted(
    name
    for name in sys.modules
    if name == 'jsonschema' or name.startswith('jsonschema.') or name == 'rpds' or name.startswith('rpds.')
)
print('clean' if not banned else 'loaded:' + ','.join(banned))
"""
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(ROOT / "src"),
        "OPENBROWSER_SETUP_LOGGING": "false",
        "OPENBROWSER_LOGGING_LEVEL": "critical",
        "ANONYMIZED_TELEMETRY": "false",
        "OPENBROWSER_DISABLE_TELEMETRY": "true",
    }
    completed = _REAL_RUN(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean", completed.stdout
