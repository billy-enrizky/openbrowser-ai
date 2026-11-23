"""Minimal newline-delimited JSON-RPC client for stdio MCP servers.

The harness only needs initialize, tools/list, and tools/call. Keeping the
client local avoids depending on an SDK version and keeps the measured process
tree to the competitor server plus its browser.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any


class McpError(RuntimeError):
    pass


class StdioMcpClient:
    def __init__(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        protocol_version: str = "2025-11-25",
        timeout: float = 120.0,
    ) -> None:
        self.command = command
        self.protocol_version = protocol_version
        self.timeout = timeout
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            text=True,
            bufsize=1,
        )
        self._next_id = 0
        self._pending: dict[int, tuple[threading.Event, dict]] = {}
        self._lock = threading.Lock()
        self.stderr_lines: list[str] = []
        self.notifications: list[dict] = []
        self._closed = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            message_id = message.get("id")
            if message_id is None:
                self.notifications.append(message)
                continue
            with self._lock:
                entry = self._pending.get(int(message_id))
            if entry is not None:
                event, slot = entry
                slot["message"] = message
                event.set()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_lines.append(line.rstrip("\n"))

    def _send(self, payload: dict) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict | None = None, *, timeout: float | None = None) -> Any:
        if self._closed:
            raise McpError("client is closed")
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            event = threading.Event()
            slot: dict = {}
            self._pending[request_id] = (event, slot)
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)
        if not event.wait(timeout or self.timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise McpError(f"timeout waiting for {method} after {timeout or self.timeout}s")
        with self._lock:
            self._pending.pop(request_id, None)
        message = slot.get("message", {})
        if "error" in message:
            raise McpError(f"{method} failed: {json.dumps(message['error'])}")
        return message.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def initialize(self) -> dict:
        result = self.request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "openbrowser-interactive-bench", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})
        return result

    def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict, *, timeout: float | None = None) -> dict:
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )

    def close(self) -> None:
        self._closed = True
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.process.kill()
            except OSError:
                pass
        for thread in (self._reader, self._stderr_reader):
            thread.join(timeout=2)

    def __enter__(self) -> StdioMcpClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
