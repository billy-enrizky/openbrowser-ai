"""Native frontend instructions agree with the package transport contract."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "docs/customize/integrations/native-cli.mdx"
LINK = "/customize/integrations/native-cli"


def reference():
    return (ROOT / REFERENCE).read_text()


def test_reference_states_exact_response_limit_and_recovery():
    text = reference()
    rust = (ROOT / "src/openbrowser/native_client/rust/src/main.rs").read_text()
    match = re.search(r"MAX_RESPONSE_BYTES:\s*usize\s*=\s*(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)", rust)
    assert match is not None
    limit = int(match[1]) * int(match[2]) * int(match[3])
    assert f"{limit:,}" in text
    assert "terminating newline" in text
    for token in ["daemon_response_too_large", "openbrowser-ai-py -c", "inspect the page", "may have already executed"]:
        assert token in text


def test_reference_covers_frontend_selection_and_source_builds():
    text = reference()
    for token in ["0.1.61", "openbrowser-ai -c", "-c -", "--mcp", "OPENBROWSER_BUILD_NATIVE=1", "editable", "Python"]:
        assert token in text
    assert "Native editable builds are rejected" in text


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "docs/quickstart.mdx",
        "docs/development/setup/local-setup.mdx",
        "docs/customize/integrations/cli-daemon.mdx",
        "docs/customize/integrations/mcp-server.mdx",
        "docs/comparison.mdx",
        "benchmarks/interactive/README.md",
    ],
)
def test_entry_guides_link_to_native_frontend_reference(name):
    assert LINK in (ROOT / name).read_text()


def test_reference_is_in_mintlify_navigation():
    config = json.loads((ROOT / "docs/docs.json").read_text())
    assert "customize/integrations/native-cli" in json.dumps(config)
