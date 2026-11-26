"""Strict JSON Schema subset for the advertised MCP input and output schemas.

The lean stateless path and the stateful server compile their schemas without
importing ``jsonschema``. Draft 2020-12 validation pulls in referencing, rpds, attrs,
and jsonschema-specifications, which costs about 10 MiB of resident memory on
first use. The advertised MCP schemas use a small fixed vocabulary, so this
module compiles that vocabulary once and raises
``UnsupportedSchemaError`` for any keyword it does not implement. An unknown
keyword never silently bypasses a constraint.

Validation order and messages mirror ``Draft202012Validator`` for the
supported keywords so existing error text and JSON paths stay stable.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeGuard

_ANNOTATIONS = frozenset({"default", "description", "title"})
_CONSTRAINTS = frozenset(
    {
        "additionalProperties",
        "enum",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "properties",
        "required",
        "type",
    }
)
_SUPPORTED_KEYWORDS = _ANNOTATIONS | _CONSTRAINTS
_SUPPORTED_TYPES = frozenset({"array", "boolean", "integer", "object", "string"})
_NULL_TYPE = frozenset({"null"})


class UnsupportedSchemaError(ValueError):
    """A schema uses a keyword this validator cannot enforce."""


@dataclass(frozen=True, slots=True)
class SchemaViolation:
    """One validation failure with the JSON path used by MCP error text."""

    json_path: str
    message: str


@dataclass(frozen=True, slots=True)
class CompiledSchema:
    """Immutable validation plan compiled from one JSON Schema mapping."""

    types: tuple[str, ...] = ()
    enum: tuple[Any, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    min_items: int | None = None
    max_items: int | None = None
    items: CompiledSchema | None = None
    properties: tuple[tuple[str, CompiledSchema], ...] = ()
    required: tuple[str, ...] = ()
    allow_additional: bool = True
    property_names: frozenset[str] = field(default_factory=frozenset)


def _unsupported(detail: str) -> UnsupportedSchemaError:
    return UnsupportedSchemaError(f"Unsupported JSON Schema: {detail}")


def _compile_types(schema: Mapping[str, Any], *, allow_null: bool) -> tuple[str, ...]:
    if "type" not in schema:
        return ()
    value = schema["type"]
    names: tuple[Any, ...]
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, list):
        names = tuple(value)
    else:
        raise _unsupported(f"'type' must be a string or non-empty list of strings, got {value!r}")
    if not names:
        raise _unsupported("'type' must not be empty")
    supported = _SUPPORTED_TYPES | (_NULL_TYPE if allow_null else frozenset())
    for name in names:
        if not isinstance(name, str) or name not in supported:
            raise _unsupported(f"'type' value {name!r} is not supported")
    if len(set(names)) != len(names):
        raise _unsupported("'type' values must be unique")
    return tuple(names)


def _compile_bound(schema: Mapping[str, Any], keyword: str) -> int | float | None:
    if keyword not in schema:
        return None
    value = schema[keyword]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _unsupported(f"{keyword!r} must be a number, got {value!r}")
    return value


def _compile_count(schema: Mapping[str, Any], keyword: str) -> int | None:
    if keyword not in schema:
        return None
    value = schema[keyword]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _unsupported(f"{keyword!r} must be a non-negative integer, got {value!r}")
    return value


_SCALAR_ENUM_TYPES = (str, int, float, bool)


def _compile_enum(schema: Mapping[str, Any]) -> tuple[Any, ...] | None:
    if "enum" not in schema:
        return None
    value = schema["enum"]
    if not isinstance(value, list):
        raise _unsupported(f"'enum' must be a list, got {value!r}")
    for entry in value:
        if entry is not None and type(entry) not in _SCALAR_ENUM_TYPES:
            raise _unsupported(f"'enum' entries must be JSON scalars, got {entry!r}")
    return tuple(value)


def _compile_properties(schema: Mapping[str, Any], *, allow_null: bool) -> tuple[tuple[str, CompiledSchema], ...]:
    if "properties" not in schema:
        return ()
    value = schema["properties"]
    if not isinstance(value, Mapping):
        raise _unsupported(f"'properties' must be an object, got {value!r}")
    compiled: list[tuple[str, CompiledSchema]] = []
    for name, subschema in value.items():
        if not isinstance(name, str):
            raise _unsupported(f"'properties' name {name!r} is not a string")
        if not isinstance(subschema, Mapping):
            raise _unsupported(f"'properties.{name}' must be a schema object")
        compiled.append((name, compile_schema(subschema, allow_null=allow_null)))
    return tuple(compiled)


def _compile_required(schema: Mapping[str, Any]) -> tuple[str, ...]:
    if "required" not in schema:
        return ()
    value = schema["required"]
    if not isinstance(value, list) or any(not isinstance(name, str) for name in value):
        raise _unsupported(f"'required' must be a list of strings, got {value!r}")
    if len(set(value)) != len(value):
        raise _unsupported("'required' values must be unique")
    return tuple(value)


def _validate_annotations(schema: Mapping[str, Any]) -> None:
    for keyword in ("title", "description"):
        if keyword in schema and not isinstance(schema[keyword], str):
            raise _unsupported(f"{keyword!r} must be a string, got {schema[keyword]!r}")


def compile_schema(schema: Mapping[str, Any], *, allow_null: bool = False) -> CompiledSchema:
    """Compile a supported schema mapping or raise ``UnsupportedSchemaError``."""
    if not isinstance(schema, Mapping):
        raise _unsupported(f"schema must be an object, got {schema!r}")
    unknown = sorted(set(schema) - _SUPPORTED_KEYWORDS)
    if unknown:
        raise _unsupported(f"keyword(s) {unknown}")

    _validate_annotations(schema)

    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, bool):
        raise _unsupported(f"'additionalProperties' only supports true or false, got {additional!r}")

    if "items" in schema and not isinstance(schema["items"], Mapping):
        raise _unsupported("'items' must be a single schema object")

    properties = _compile_properties(schema, allow_null=allow_null)
    return CompiledSchema(
        types=_compile_types(schema, allow_null=allow_null),
        enum=_compile_enum(schema),
        minimum=_compile_bound(schema, "minimum"),
        maximum=_compile_bound(schema, "maximum"),
        min_length=_compile_count(schema, "minLength"),
        max_length=_compile_count(schema, "maxLength"),
        min_items=_compile_count(schema, "minItems"),
        max_items=_compile_count(schema, "maxItems"),
        items=compile_schema(schema["items"], allow_null=allow_null) if "items" in schema else None,
        properties=properties,
        required=_compile_required(schema),
        allow_additional=additional,
        property_names=frozenset(name for name, _ in properties),
    )


def _is_number(instance: Any) -> TypeGuard[int | float]:
    return isinstance(instance, (int, float)) and not isinstance(instance, bool)


def _enum_matches(instance: Any, entry: Any) -> bool:
    """Compare JSON scalars with booleans distinct from numbers."""
    if isinstance(instance, bool) or isinstance(entry, bool):
        return isinstance(instance, bool) and isinstance(entry, bool) and instance == entry
    if instance is None or entry is None:
        return instance is None and entry is None
    if isinstance(instance, (int, float)) and isinstance(entry, (int, float)):
        return instance == entry
    if isinstance(instance, str) and isinstance(entry, str):
        return instance == entry
    return False


def _matches_type(instance: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(instance, dict)
    if type_name == "array":
        return isinstance(instance, list)
    if type_name == "string":
        return isinstance(instance, str)
    if type_name == "integer":
        if isinstance(instance, bool):
            return False
        if isinstance(instance, int):
            return True
        if isinstance(instance, float):
            return instance.is_integer()
        return False
    if type_name == "boolean":
        return isinstance(instance, bool)
    if type_name == "null":
        return instance is None
    return False


def iter_errors(instance: Any, schema: CompiledSchema) -> Iterator[SchemaViolation]:
    """Yield violations in the keyword order ``Draft202012Validator`` uses."""
    yield from _iter_errors(instance, schema, "$")


def _iter_errors(instance: Any, schema: CompiledSchema, path: str) -> Iterator[SchemaViolation]:
    if isinstance(instance, dict) and not schema.allow_additional:
        extras = sorted(set(instance) - schema.property_names)
        if extras:
            verb = "was" if len(extras) == 1 else "were"
            extras_repr = ", ".join(repr(extra) for extra in extras)
            yield SchemaViolation(path, f"Additional properties are not allowed ({extras_repr} {verb} unexpected)")

    if schema.enum is not None and not any(_enum_matches(instance, entry) for entry in schema.enum):
        yield SchemaViolation(path, f"{instance!r} is not one of {list(schema.enum)!r}")

    if isinstance(instance, list):
        if schema.items is not None:
            for index, item in enumerate(instance):
                yield from _iter_errors(item, schema.items, f"{path}[{index}]")
        if schema.max_items is not None and len(instance) > schema.max_items:
            yield SchemaViolation(path, f"{instance!r} is too long")
        if schema.min_items is not None and len(instance) < schema.min_items:
            if schema.min_items == 1:
                yield SchemaViolation(path, f"{instance!r} should be non-empty")
            else:
                yield SchemaViolation(path, f"{instance!r} is too short")

    if isinstance(instance, str):
        if schema.max_length is not None and len(instance) > schema.max_length:
            yield SchemaViolation(path, f"{instance!r} is too long")
        if schema.min_length is not None and len(instance) < schema.min_length:
            if schema.min_length == 1:
                yield SchemaViolation(path, f"{instance!r} should be non-empty")
            else:
                yield SchemaViolation(path, f"{instance!r} is too short")

    if _is_number(instance):
        if schema.maximum is not None and instance > schema.maximum:
            yield SchemaViolation(path, f"{instance!r} is greater than the maximum of {schema.maximum!r}")
        if schema.minimum is not None and instance < schema.minimum:
            yield SchemaViolation(path, f"{instance!r} is less than the minimum of {schema.minimum!r}")

    if isinstance(instance, dict):
        for name, subschema in schema.properties:
            if name in instance:
                yield from _iter_errors(instance[name], subschema, f"{path}.{name}")

        for name in schema.required:
            if name not in instance:
                yield SchemaViolation(path, f"{name!r} is a required property")

    if schema.types and not any(_matches_type(instance, type_name) for type_name in schema.types):
        joined = ", ".join(repr(type_name) for type_name in schema.types)
        yield SchemaViolation(path, f"{instance!r} is not of type {joined}")
