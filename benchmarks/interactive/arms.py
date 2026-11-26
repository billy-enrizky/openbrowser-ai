"""Native interface identities and framework families for benchmark cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

WARM_METHOD_ID = "native-session-readiness-v3"
WARM_METHOD_LABEL = "Warm-session readiness"


@dataclass(frozen=True)
class ArmSpec:
    name: str
    label: str
    family: str
    color: str
    canonical: bool = True


_SPECS = (
    ArmSpec("openbrowser-mcp-fast", "OpenBrowser MCP", "openbrowser", "#0f6270"),
    ArmSpec("openbrowser-fast", "OpenBrowser CLI", "openbrowser", "#58a69f"),
    ArmSpec("chrome-devtools-mcp", "Chrome DevTools MCP", "chrome-devtools", "#c2852a"),
    ArmSpec("agent-browser", "agent-browser CLI", "agent-browser", "#4f7f5b"),
    ArmSpec("browser-use", "browser-use CLI", "browser-use", "#8871b4"),
    ArmSpec("playwright-mcp", "Playwright MCP", "playwright", "#b25f48"),
    ArmSpec("playwright-cli", "Playwright CLI", "playwright", "#6c7787"),
    ArmSpec("openbrowser", "OpenBrowser CLI (indexed)", "openbrowser", "#58a69f", canonical=False),
    ArmSpec("openbrowser-mcp", "OpenBrowser MCP (indexed)", "openbrowser", "#0f6270", canonical=False),
)
ARM_SPECS = MappingProxyType({spec.name: spec for spec in _SPECS})
CANONICAL_ARMS = tuple(spec.name for spec in _SPECS if spec.canonical)
SUPPORTED_ARMS = tuple(ARM_SPECS)
