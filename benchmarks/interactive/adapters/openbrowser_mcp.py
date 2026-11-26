"""OpenBrowser MCP server adapter for the interactive benchmark.

This arm drives the package's own MCP server over the stateless
``2026-07-28`` stdio protocol: one ``server/discover`` probe, then
``create_context``, ``execute_code``, and ``close_context``. It measures the
MCP surface that coding agents use.
"""

from __future__ import annotations

import contextlib
import json
import os
import time

from ..measurement import browser_launch_checkpoint
from .base import ArmResult, RunContext, terminate_matching, verify_mock_keychain
from .launch_proof import LaunchProof, capture_launch_proof, proof_status
from .mcp_modern import ModernMcpError, ModernStdioMcpClient
from .openbrowser import (
    _default_binary,
    _download,
    _env,
    _fast_download,
    _fast_login,
    _fast_modal,
    _fast_tabs,
    _fast_wizard,
    _login,
    _modal,
    _read_account,
    _tabs,
    _wizard,
    _write_config,
)


class OpenBrowserMcpArm:
    name = "openbrowser-mcp"

    def __init__(self, binary: str | None = None, *, fast_selectors: bool = False) -> None:
        self.binary = binary or os.environ.get("OPENBROWSER_BIN") or _default_binary()
        self.fast_selectors = fast_selectors
        self.name = "openbrowser-mcp-fast" if fast_selectors else "openbrowser-mcp"
        self._result = ArmResult(ok=False)
        self._client: ModernStdioMcpClient | None = None
        self._context_id: str | None = None
        self._launch_proof: LaunchProof | None = None

    # -- lifecycle -------------------------------------------------------
    def _start(self, ctx: RunContext) -> ModernStdioMcpClient:
        if self._client is not None:
            return self._client
        self._launch_proof = None
        client = ModernStdioMcpClient([self.binary, "--mcp"], env=_env(ctx))
        self._result.process_starts += 1
        client.discover()
        self._client = client
        return client

    def _create_context(self, client: ModernStdioMcpClient) -> str:
        payload = json.loads(client.call_tool_text("create_context", {}))
        self._result.tool_calls += 1
        return str(payload["context_id"])

    def _ensure_context(self, client: ModernStdioMcpClient) -> str:
        """Reuse the warm-lane context so one browser serves every task."""
        if self._context_id is None:
            self._context_id = self._create_context(client)
        return self._context_id

    def _execute(self, client: ModernStdioMcpClient, context_id: str, code: str, timeout: float) -> str:
        self._result.tool_calls += 1
        return client.call_tool_text("execute_code", {"context_id": context_id, "code": code}, timeout=timeout)

    def _close_context(self, client: ModernStdioMcpClient, context_id: str) -> None:
        self._launch_proof = None
        self._result.tool_calls += 1
        payload = json.loads(client.call_tool_text("close_context", {"context_id": context_id}, timeout=120))
        if payload.get("status") != "closed":
            raise ModernMcpError(f"close_context returned {payload!r}")

    def _program(self, task: str, url: str, state_path: str | None = None) -> str:
        fast = self.fast_selectors
        if task == "wizard":
            return _fast_wizard(url + "/wizard") if fast else _wizard(url + "/wizard")
        if task == "modal":
            return _fast_modal(url + "/modal") if fast else _modal(url + "/modal")
        if task == "download":
            return _fast_download(url + "/download") if fast else _download(url + "/download")
        if task == "tabs":
            return _fast_tabs(url + "/tabs") if fast else _tabs(url + "/tabs")
        if task == "auth" and state_path is not None:
            login = _fast_login if fast else _login
            return login(url + "/login", state_path)
        raise ValueError(f"unknown task {task}")

    def _mock_keychain_profile(self, ctx: RunContext) -> str:
        profiles = ctx.config / "profiles"
        candidates = sorted(profiles.glob("mcp-*"), key=lambda path: path.stat().st_mtime) if profiles.exists() else []
        if not candidates:
            return str(profiles / "missing-mcp-profile")
        return str(candidates[-1])

    # -- benchmark surface ----------------------------------------------
    def run(self, ctx: RunContext, *, cleanup: bool = True, restart_for_auth: bool = True) -> ArmResult:
        self._result = ArmResult(ok=False)
        _write_config(ctx, extensions=False)
        proof = self._launch_proof
        if cleanup:
            self._launch_proof = None
            proof = None
        if ctx.task == "auth":
            self._launch_proof = None
            proof = None
        try:
            client = self._start(ctx)
            if ctx.task == "auth":
                context_id = self._create_context(client)
                self._context_id = context_id
                login_output = self._execute(
                    client, context_id, self._program("auth", ctx.base_url, str(ctx.root / "storage_state.json")), 120
                )
                if restart_for_auth:
                    browser_launch_checkpoint(self._mock_keychain_profile(ctx))
                    self._close_context(client, context_id)
                    context_id = self._create_context(client)
                    self._context_id = context_id
                account_output = self._execute(client, context_id, _read_account(ctx.base_url + "/account"), 120)
                self._result.stdout += login_output + account_output
                verified = verify_mock_keychain(self._mock_keychain_profile(ctx))
                self._result.keychain_verified = verified
                if verified is False:
                    raise RuntimeError("OpenBrowser MCP Chrome launched without --use-mock-keychain")
            else:
                context_id = self._create_context(client) if cleanup else self._ensure_context(client)
                self._context_id = context_id
                server_pid = client.process.pid if client.process is not None else None
                if proof is not None:
                    before = proof_status(
                        proof,
                        profile_marker=self._mock_keychain_profile(ctx),
                        server_pid=server_pid,
                        context_id=context_id,
                    )
                    if before != "valid":
                        self._launch_proof = None
                        raise RuntimeError(f"launch proof invalid before task: {before}")
                output = self._execute(client, context_id, self._program(ctx.task, ctx.base_url), 150)
                self._result.stdout += output
                if proof is not None:
                    after = proof_status(
                        proof,
                        profile_marker=self._mock_keychain_profile(ctx),
                        server_pid=server_pid,
                        context_id=context_id,
                    )
                    if after != "valid":
                        self._launch_proof = None
                        raise RuntimeError(f"launch proof invalid after task: {after}")
                    verified = True
                else:
                    verified = verify_mock_keychain(self._mock_keychain_profile(ctx))
                self._result.keychain_verified = verified
                if verified is False:
                    raise RuntimeError("OpenBrowser MCP Chrome launched without --use-mock-keychain")
                if cleanup:
                    self._close_context(client, context_id)
                    self._context_id = None
            self._result.reported = self._parse_last_json()
            self._result.ok = True
        except Exception as exc:
            self._result.ok = False
            self._result.error = str(exc)
        finally:
            if cleanup:
                self.cleanup(ctx)
        return self._result

    def warmup(self, ctx: RunContext) -> tuple[dict, float]:
        """Start the server and one context so the browser is warm."""
        self._launch_proof = None
        started = time.monotonic()
        _write_config(ctx, extensions=False)
        client = self._start(ctx)
        context_id = self._create_context(client)
        self._context_id = context_id
        self._execute(client, context_id, "pass", 180)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        profile_marker = self._mock_keychain_profile(ctx)
        verified = verify_mock_keychain(profile_marker)
        if verified is False:
            raise RuntimeError("OpenBrowser MCP Chrome launched without --use-mock-keychain")
        server_pid = client.process.pid if client.process is not None else None
        self._launch_proof = capture_launch_proof(
            profile_marker,
            server_pid,
            context_id=context_id,
            keychain_verified=verified,
        )
        return {
            "server": "started",
            "context_id": context_id,
            "keychain_verified": verified,
            "launch_proof_captured": self._launch_proof is not None,
        }, elapsed_ms

    def _parse_last_json(self) -> dict:
        for line in reversed(self._result.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {}

    def cleanup(self, ctx: RunContext) -> None:
        self._launch_proof = None
        if self._client is not None and self._context_id is not None:
            with contextlib.suppress(Exception):
                self._close_context(self._client, self._context_id)
            self._context_id = None
        if self._client is not None:
            self._client.close()
            self._client = None
        terminate_matching(str(ctx.config / "profiles"))
        terminate_matching(str(ctx.downloads))
