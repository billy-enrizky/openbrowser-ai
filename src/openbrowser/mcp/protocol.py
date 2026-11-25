"""Protocol constants shared by the SDK and lean MCP stdio transports."""

from __future__ import annotations

MODERN_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSIONS = frozenset({"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"})
PROTOCOL_VERSION_META = "io.modelcontextprotocol/protocolVersion"
CLIENT_INFO_META = "io.modelcontextprotocol/clientInfo"
CLIENT_CAPABILITIES_META = "io.modelcontextprotocol/clientCapabilities"
SERVER_INFO_META = "io.modelcontextprotocol/serverInfo"
SERVER_NAME = "openbrowser"


class ProtocolRpcError(Exception):
    """A JSON-RPC error returned by the transport-neutral dispatcher."""

    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(message)
