"""Keep package and plugin metadata descriptive and version-consistent."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METADATA = (
    ".claude-plugin/marketplace.json",
    "plugin/.claude-plugin/plugin.json",
    "server.json",
    "src/openbrowser/mcp/manifest.json",
)
RATIO = re.compile(r"\b\d+(?:\.\d+)?\s*[x\u00d7]\b|\bfewer\s+tokens\b", re.IGNORECASE)
GENERIC_TOOL_NAME = re.compile(r"\bthe\s+search\s+tool\b", re.IGNORECASE)


def descriptions(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"description", "long_description"} and isinstance(child, str):
                yield child
            else:
                yield from descriptions(child)
    elif isinstance(value, list):
        for child in value:
            yield from descriptions(child)


@pytest.mark.parametrize("name", METADATA)
def test_metadata_descriptions_state_capabilities_with_explicit_tool_names(name):
    data = json.loads((ROOT / name).read_text())
    for text in descriptions(data):
        assert RATIO.search(text) is None, f"{name}: keep measured comparisons in their evidence-backed results pages"
        assert GENERIC_TOOL_NAME.search(text) is None, f"{name}: use the explicit web_search tool name"


def test_mcp_manifest_separates_standard_and_third_party_libraries():
    data = json.loads((ROOT / "src/openbrowser/mcp/manifest.json").read_text())
    block = re.search(r"(?im)^-\s*\*\*Standard Libraries\*\*:\s*(.+)$", data["long_description"])
    assert block is not None
    assert "requests" not in {name.strip() for name in block[1].split(",")}


def test_release_metadata_versions_match_across_interfaces():
    versions = []
    for name in METADATA:
        data = json.loads((ROOT / name).read_text())
        if name == ".claude-plugin/marketplace.json":
            versions.extend(plugin["version"] for plugin in data["plugins"] if plugin["name"] == "openbrowser")
        else:
            versions.append(data["version"])
        if name == "server.json":
            versions.extend(package["version"] for package in data["packages"])
    assert len(versions) >= len(METADATA)
    assert len(set(versions)) == 1
