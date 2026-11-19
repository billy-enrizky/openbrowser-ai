"""Tests for dependency versions required by the MCP server implementation."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def test_mcp_sdk_requirements_require_current_callback_api():
    """Every install group requires the migrated SDK and excludes future majors."""
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())[
        "project"
    ]
    requirements = list(project["dependencies"])
    for extra_requirements in project["optional-dependencies"].values():
        requirements.extend(extra_requirements)

    mcp_requirements = [
        Requirement(requirement)
        for requirement in requirements
        if Requirement(requirement).name == "mcp"
    ]

    assert mcp_requirements
    for requirement in mcp_requirements:
        bounds = {(spec.operator, spec.version) for spec in requirement.specifier}
        assert (">=", "2.3.0") in bounds, requirement
        assert ("<", "3.0.0") in bounds, requirement
        assert Version("2.2.1") not in requirement.specifier, requirement
        assert Version("3.1.0") not in requirement.specifier, requirement
        assert Version("1.26.0") not in requirement.specifier, requirement
    assert any(
        Requirement(requirement).name == "jsonschema"
        for requirement in project["dependencies"]
    )


def test_lock_selects_current_sdk_and_matching_protocol_types():
    lock = tomllib.loads((Path(__file__).parents[1] / "uv.lock").read_text())
    packages = {package["name"]: package.get("version") for package in lock["package"]}
    assert packages["mcp"] == "2.3.0"
    assert packages["mcp-types"] == "2.3.0"
