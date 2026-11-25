"""Lean stateless MCP stdio transport.

This transport implements the ``2026-07-28`` JSON-RPC surface directly. It
answers discovery and the tool catalog before importing the MCP SDK, then loads
the application runtime on the first tool call. The SDK transport remains the
default until the lean surface passes the paired compatibility lane.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .catalog import load_tool_definitions
from .protocol import (
    CLIENT_CAPABILITIES_META,
    LEGACY_PROTOCOL_VERSIONS,
    MODERN_PROTOCOL_VERSION,
    PROTOCOL_VERSION_META,
    SERVER_INFO_META,
    SERVER_NAME,
    ProtocolRpcError,
)
from .stateless import STATELESS_TOOL_NAMES, execute_stateless_tool

_CACHE_TTL_MS = 60_000
_CANCELLED = object()
_CONNECTION_CLOSED = -32000


def _request_key(request_id: Any) -> tuple[type[Any], Any]:
    """Keep numeric and string request IDs distinct in the task table."""
    return (type(request_id), request_id)


def _distribution_version() -> str:
    try:
        return version("openbrowser-ai")
    except PackageNotFoundError:
        return "0+unknown"


class LeanStdioServer:
    """One JSON-RPC stdio loop backed by the transport-neutral application."""

    def __init__(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        self._read_fd = os.dup(0)
        self._write_fd = os.dup(1)
        self._reader = os.fdopen(self._read_fd, "rb")
        null_fd = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(null_fd, 0)
            os.dup2(2, 1)
        finally:
            os.close(null_fd)
        self._server_version = _distribution_version()
        self._start_time = time.time()
        self._app: Any = None
        self._app_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._tasks: dict[tuple[type[Any], Any], asyncio.Task[None]] = {}
        self._request_ids: dict[tuple[type[Any], Any], Any] = {}
        self._protocol_version: str | None = None
        self._era: str | None = None

    @property
    def _server_info(self) -> dict[str, str]:
        return {"name": SERVER_NAME, "version": self._server_version}

    @property
    def _meta(self) -> dict[str, dict[str, str]]:
        return {SERVER_INFO_META: self._server_info}

    def _tools_for(self, modern: bool) -> list[dict[str, Any]]:
        compact = os.environ.get("OPENBROWSER_COMPACT_DESCRIPTION", "").lower() in ("1", "true", "yes")
        return list(load_tool_definitions(modern=modern, compact=compact))

    def _modern_request_version(self, params: dict[str, Any]) -> str:
        meta = params.get("_meta")
        if not isinstance(meta, dict):
            raise ProtocolRpcError(-32602, "Modern requests require a _meta envelope")
        version = meta.get(PROTOCOL_VERSION_META)
        if not isinstance(version, str) or not version:
            raise ProtocolRpcError(-32602, "Modern _meta requires a protocolVersion string")
        if not isinstance(meta.get(CLIENT_CAPABILITIES_META), dict):
            raise ProtocolRpcError(-32602, "Modern _meta requires clientCapabilities as an object")
        if version != MODERN_PROTOCOL_VERSION:
            raise ProtocolRpcError(-32022, f"Unsupported protocol version: {version}")
        return version

    def _activate_modern(self, params: dict[str, Any]) -> None:
        version = self._modern_request_version(params)
        if self._era == "legacy":
            raise ProtocolRpcError(-32600, "Modern request arrived on a legacy connection")
        self._era = "modern"
        self._protocol_version = version

    def _require_legacy(self) -> None:
        if self._era is None:
            raise ProtocolRpcError(-32600, "Legacy session is not initialized")
        if self._era != "legacy":
            raise ProtocolRpcError(-32600, "Legacy request arrived on a modern connection")

    async def _ensure_app(self) -> Any:
        if self._app is not None:
            return self._app
        async with self._app_lock:
            if self._app is None:
                from openbrowser.browser.lifecycle import LifecyclePhase
                from openbrowser.mcp.server import OpenBrowserServer

                app = OpenBrowserServer(install_sdk=False)
                app._start_time = self._start_time
                app._lifecycle.transition(LifecyclePhase.STARTING, reason="lean MCP ready")
                app._lifecycle.transition(LifecyclePhase.READY, reason="lean MCP ready")
                await app._start_cleanup_task()
                self._app = app
        return self._app

    def _write_all(self, payload: bytes) -> None:
        view = memoryview(payload)
        try:
            while view:
                written = os.write(self._write_fd, view)
                view = view[written:]
        except BrokenPipeError:
            return

    async def _write(self, message: dict[str, Any]) -> None:
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("ascii")
        async with self._write_lock:
            await asyncio.to_thread(self._write_all, payload)

    async def _write_result(self, request_id: Any, result: Any) -> None:
        await self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _write_error(self, request_id: Any, code: int, message: str) -> None:
        await self._write({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})

    def _metadata(self, *, modern: bool) -> dict[str, Any]:
        if not modern:
            return {}
        return {"resultType": "complete", "_meta": self._meta}

    def _cache_metadata(self, *, modern: bool, ttl_ms: int = _CACHE_TTL_MS) -> dict[str, Any]:
        if not modern:
            return {}
        return {"ttlMs": ttl_ms, "cacheScope": "private", **self._metadata(modern=True)}

    def _catalog_result(self, *, modern: bool) -> dict[str, Any]:
        return {
            "tools": self._tools_for(modern),
            **self._cache_metadata(modern=modern),
        }

    def _discover_result(self) -> dict[str, Any]:
        return {
            "supportedVersions": [MODERN_PROTOCOL_VERSION],
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
                "prompts": {"listChanged": False},
            },
            **self._cache_metadata(modern=True, ttl_ms=0),
        }

    def _initialize_result(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._era == "modern":
            raise ProtocolRpcError(-32022, "initialize is not allowed on a modern connection")
        if not isinstance(params.get("capabilities"), dict):
            raise ProtocolRpcError(-32602, "initialize requires capabilities as an object")
        if not isinstance(params.get("clientInfo"), dict):
            raise ProtocolRpcError(-32602, "initialize requires clientInfo as an object")
        requested = params.get("protocolVersion")
        if requested == MODERN_PROTOCOL_VERSION:
            chosen = "2025-11-25"
        elif isinstance(requested, str) and requested in LEGACY_PROTOCOL_VERSIONS:
            chosen = requested
        else:
            raise ProtocolRpcError(-32602, f"Unsupported initialize protocolVersion: {requested!r}")
        self._era = "legacy"
        self._protocol_version = chosen
        return {
            "protocolVersion": chosen,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": self._server_info,
        }

    async def _call_tool(self, params: dict[str, Any], *, modern: bool) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            raise ValueError("Tool name must be a string")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        if name in STATELESS_TOOL_NAMES:
            try:
                text, is_error = await execute_stateless_tool(name, arguments)
            except ValueError as error:
                raise ProtocolRpcError(-32602, str(error)) from error
            return {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
                **self._metadata(modern=modern),
            }
        app = await self._ensure_app()
        outcome = await app.dispatch_tool(name, arguments, self._protocol_version)
        result: dict[str, Any] = {
            "content": list(outcome.content),
            "isError": outcome.is_error,
            **self._metadata(modern=modern),
        }
        if outcome.structured_content is not None:
            result["structuredContent"] = outcome.structured_content
        return result

    def _request_era(self, params: dict[str, Any]) -> bool:
        """Return True for a modern request and False for an initialized legacy request."""
        if self._era is None:
            if isinstance(params.get("_meta"), dict):
                self._activate_modern(params)
                return True
            raise ProtocolRpcError(-32600, "Session is not initialized")
        if self._era == "legacy":
            if isinstance(params.get("_meta"), dict):
                raise ProtocolRpcError(-32600, "Modern request arrived on a legacy connection")
            self._require_legacy()
            return False
        self._activate_modern(params)
        return True

    async def _handle_request(self, request_id: Any, method: str, params: dict[str, Any]) -> dict[str, Any] | object:
        if method == "server/discover":
            self._activate_modern(params)
            return self._discover_result()
        if method == "initialize":
            return self._initialize_result(params)
        if method == "ping":
            if self._era == "modern":
                self._activate_modern(params)
                return {"resultType": "complete", "_meta": self._meta}
            return {}
        if method == "logging/setLevel":
            raise LookupError(f"Method not found: {method}")
        modern = self._request_era(params)
        if method == "tools/list":
            return self._catalog_result(modern=modern)
        if method == "resources/list":
            return {"resources": [], **self._cache_metadata(modern=modern)}
        if method == "resources/templates/list":
            return {"resourceTemplates": [], **self._cache_metadata(modern=modern)}
        if method == "prompts/list":
            return {"prompts": [], **self._cache_metadata(modern=modern)}
        if method == "tools/call":
            try:
                return await self._call_tool(params, modern=modern)
            except asyncio.CancelledError:
                return _CANCELLED
        raise LookupError(f"Method not found: {method}")

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if message.get("jsonrpc") != "2.0":
            await self._write_error(message.get("id"), -32600, "Invalid Request")
            return
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params")
        if not isinstance(method, str):
            if request_id is not None:
                await self._write_error(request_id, -32600, "Invalid Request")
            return
        if request_id is None:
            if method == "notifications/cancelled" and isinstance(params, dict):
                target = params.get("requestId")
                task = self._tasks.get(_request_key(target))
                if task is not None:
                    task.cancel()
            return
        if not isinstance(params, dict):
            params = {}
        try:
            result = await self._handle_request(request_id, method, params)
        except LookupError as error:
            await self._write_error(request_id, -32601, str(error))
            return
        except ValueError as error:
            await self._write_error(request_id, -32602, str(error))
            return
        except asyncio.CancelledError:
            return
        except ProtocolRpcError as error:
            await self._write_error(request_id, error.code, str(error))
            return
        except Exception:
            await self._write_error(request_id, -32603, "Internal error")
            return
        if result is not _CANCELLED:
            await self._write_result(request_id, result)

    async def _read_line(self) -> bytes:
        return await asyncio.to_thread(self._reader.readline)

    def _discard_task(self, key: tuple[type[Any], Any], task: asyncio.Task[None]) -> None:
        if self._tasks.get(key) is task:
            self._tasks.pop(key, None)
            self._request_ids.pop(key, None)

    async def serve(self) -> None:
        try:
            while True:
                raw = await self._read_line()
                if not raw:
                    break
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await self._write_error(None, -32700, "Parse error")
                    continue
                if not isinstance(message, dict):
                    await self._write_error(None, -32600, "Invalid Request")
                    continue
                if message.get("method") == "initialize":
                    await self._handle_message(message)
                    continue
                request_id = message.get("id")
                if request_id is None:
                    await self._handle_message(message)
                    continue
                task = asyncio.create_task(self._handle_message(message))
                key = _request_key(request_id)
                self._tasks[key] = task
                self._request_ids[key] = request_id
                task.add_done_callback(lambda finished, task_key=key: self._discard_task(task_key, finished))
        finally:
            pending = [
                (self._request_ids[key], task)
                for key, task in list(self._tasks.items())
                if not task.done() and self._request_ids.get(key) is not None
            ]
            for _request_id, task in pending:
                task.cancel()
            if pending:
                await asyncio.wait([task for _request_id, task in pending], timeout=2)
                for request_id, _task in pending:
                    await self._write_error(request_id, _CONNECTION_CLOSED, "Connection closed")
            shutdown_error: BaseException | None = None
            if self._app is not None:
                try:
                    from openbrowser.browser.lifecycle import LifecyclePhase

                    self._app._lifecycle.transition(LifecyclePhase.STOPPING, reason="lean MCP stopping")
                    await self._app._shutdown()
                except Exception as error:
                    shutdown_error = error
                    with contextlib.suppress(Exception):
                        from openbrowser.browser.lifecycle import LifecyclePhase

                        self._app._lifecycle.fail(error, reason="lean MCP shutdown failed")
                else:
                    with contextlib.suppress(Exception):
                        from openbrowser.browser.lifecycle import LifecyclePhase

                        self._app._lifecycle.transition(
                            LifecyclePhase.STOPPED,
                            reason="lean MCP stopped",
                            increment_generation=True,
                        )
            with contextlib.suppress(Exception):
                self._reader.close()
            with contextlib.suppress(OSError):
                os.close(self._write_fd)
            if shutdown_error is not None:
                raise shutdown_error


async def main() -> None:
    """Run the lean transport until stdin reaches EOF."""
    import logging

    logging.disable(logging.CRITICAL)
    await LeanStdioServer().serve()


if __name__ == "__main__":
    asyncio.run(main())
