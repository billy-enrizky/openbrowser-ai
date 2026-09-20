"""Tests for dependency versions required by the MCP server implementation."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def test_mcp_sdk_requirements_exclude_incompatible_v2_api():
    """The server uses the MCP 1.x decorator API, not the MCP 2.x callback API."""
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
    assert all(
        Version("1.26.0") in requirement.specifier for requirement in mcp_requirements
    )
    assert all(
        Version("2.0.0") not in requirement.specifier
        for requirement in mcp_requirements
    )
