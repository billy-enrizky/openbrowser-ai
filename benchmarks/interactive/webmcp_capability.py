"""Separate functional WebMCP tasks through native CLI and MCP script surfaces.

Run without arguments for an inert plan. This lane records capability evidence.
The existing five UI tasks and performance lanes remain unchanged.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

ARMS = ("openbrowser-cli", "openbrowser-mcp", "playwright-cli", "playwright-mcp")
TOOL = {
    "name": "openbrowser_sum",
    "description": "Add two integers for the capability fixture.",
    "inputSchema": {
        "type": "object",
        "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
        "required": ["left", "right"],
        "additionalProperties": False,
    },
}
EXPECTED_RESULT = {"content": [{"type": "text", "text": "42"}]}
FIXTURE = (
    '<!doctype html><html lang="en"><meta charset="utf-8"><title>WebMCP capability fixture</title>'
    "<h1>WebMCP capability fixture</h1><script>"
    "if (document.modelContext) document.modelContext.registerTool({..."
    + json.dumps(TOOL)
    + ", execute: async ({left, right}) => ({content: [{type: 'text', text: String(left + right)}]})});"
    "</script></html>"
)
PAGE_TASK = """(async () => {
  const mc = document.modelContext;
  if (!mc) return {available: false, catalog: null, invoked: null};
  const tools = await mc.getTools();
  const tool = tools.find(t => t.name === 'openbrowser_sum');
  const catalog = tools.map(t => ({name: t.name, description: t.description, inputSchema: t.inputSchema}));
  const raw = await mc.executeTool(tool, {left: 17, right: 25});
  return {available: true, catalog, invoked: typeof raw === 'string' ? JSON.parse(raw) : raw};
})()"""


def python_program(url: str) -> str:
    return (
        f'import json,sys\nawait navigate({url!r})\nresult = await evaluate("""'
        + PAGE_TASK
        + '""")\nsys.stdout.write(json.dumps(result) + "\\n")\n'
    )


def playwright_program(url: str) -> str:
    return (
        f"async (page) => {{ await page.goto({json.dumps(url)});\n"
        + "const result = await page.evaluate(`"
        + PAGE_TASK
        + "`);\nconsole.log(JSON.stringify(result)); return result; }"
    )


def task_results(result: dict[str, Any]) -> dict[str, bool]:
    return {
        "webmcp-discovery": result.get("available") is True and result.get("catalog") == [TOOL],
        "webmcp-invocation": result.get("available") is True and result.get("invoked") == EXPECTED_RESULT,
    }


def last_json(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "available" in value:
            return value
    raise ValueError("Native output contains no WebMCP capability result")


def write_configs(root: Path, chrome: Path, feature: bool) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config"
    config.mkdir()
    args = ["--use-mock-keychain"] + (["--enable-features=WebMCP"] if feature else [])
    ob = config / "config.json"
    ob.write_text(
        json.dumps(
            {
                "browser_profile": {
                    "capability": {
                        "id": "capability",
                        "default": True,
                        "headless": True,
                        "executable_path": str(chrome),
                        "args": args,
                        "enable_default_extensions": False,
                    }
                },
                "llm": {"capability": {"id": "capability", "default": True}},
                "agent": {"capability": {"id": "capability", "default": True}},
            }
        )
    )
    pw = config / "playwright.json"
    pw.write_text(
        json.dumps(
            {
                "browser": {"launchOptions": {"executablePath": str(chrome), "headless": True, "args": args}},
                "timeouts": {"settle": 0},
            }
        )
    )
    return ob, pw


def child_environment(root: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"} and not k.endswith("_API_KEY")}
    env.update(
        HOME=str(root / "home"),
        OPENBROWSER_CONFIG_DIR=str(root / "config"),
        OPENBROWSER_SOCKET=str(root / "daemon.sock"),
        OPENBROWSER_USE_HEADLESS_SHELL="0",
        OPENBROWSER_SETUP_LOGGING="false",
        OPENBROWSER_LOGGING_LEVEL="result",
        PYTHONNOUSERSITE="1",
        ANONYMIZED_TELEMETRY="false",
    )
    return env


def _command(command: list[str], env: dict[str, str], root: Path, label: str) -> str:
    result = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=120, check=False)
    (root / f"{label}.stdout.txt").write_text(result.stdout)
    (root / f"{label}.stderr.txt").write_text(result.stderr)
    (root / f"{label}.exit.txt").write_text(str(result.returncode))
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}")
    return result.stdout


def run_arm(arm: str, root: Path, chrome: Path, feature: bool, binary: str, url: str) -> dict[str, Any]:
    from .adapters.mcp_modern import ModernStdioMcpClient
    from .adapters.mcp_stdio import StdioMcpClient

    _, pw = write_configs(root, chrome, feature)
    (root / "home").mkdir()
    env = child_environment(root)
    if arm == "openbrowser-cli":
        try:
            output = _command([binary, "-c", python_program(url)], env, root, "execute")
        finally:
            _command([binary, "daemon", "stop"], env, root, "cleanup")
    elif arm == "playwright-cli":
        session = "webmcp-" + root.name
        command = [binary, "-s", session]
        try:
            _command(
                [*command, "open", url, "--profile", str(root / "profile"), "--config", str(pw)], env, root, "open"
            )
            program = root / "program.js"
            program.write_text(playwright_program(url))
            output = _command([*command, "run-code", "--filename", str(program), "--raw"], env, root, "execute")
        finally:
            _command([*command, "close"], env, root, "cleanup")
    else:
        command = [binary, "--mcp"] if arm == "openbrowser-mcp" else [binary, "--headless", "--config", str(pw)]
        client_type = ModernStdioMcpClient if arm == "openbrowser-mcp" else StdioMcpClient
        with client_type(command, env=env, cwd=str(root), timeout=120) as client:
            context_id = None
            if isinstance(client, ModernStdioMcpClient):
                client.discover()
                context_id = json.loads(client.call_tool_text("create_context", {}))["context_id"]
                name, arguments = "execute_code", {"context_id": context_id, "code": python_program(url)}
            else:
                client.initialize()
                name, arguments = "browser_run_code_unsafe", {"code": playwright_program(url)}
            try:
                response = client.call_tool(name, arguments)
                (root / "execute.response.json").write_text(json.dumps(response, indent=2))
                output = "\n".join(
                    item.get("text", "") for item in response.get("content", []) if item.get("type") == "text"
                )
                if response.get("isError"):
                    raise RuntimeError("MCP tool reported an error")
            finally:
                cleanup = (
                    client.call_tool("close_context", {"context_id": context_id})
                    if context_id
                    else client.call_tool("browser_close", {})
                )
                (root / "cleanup.response.json").write_text(json.dumps(cleanup, indent=2))
        (root / "server.stderr.txt").write_text("\n".join(client.stderr_lines))
    result = last_json(output)
    return {"arm": arm, "surface": "native scripted page API", "result": result, "tasks": task_results(result)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--chrome-bin", type=Path)
    parser.add_argument("--openbrowser-bin")
    parser.add_argument("--playwright-cli-bin")
    parser.add_argument("--playwright-mcp-bin")
    parser.add_argument("--without-feature", action="store_true")
    options = parser.parse_args(argv)
    plan = {"arms": list(ARMS), "tasks": ["webmcp-discovery", "webmcp-invocation"], "performance_claim": False}
    if options.run_output is None:
        sys.stdout.write(json.dumps(plan) + "\n")
        return 0
    if not all((options.chrome_bin, options.openbrowser_bin, options.playwright_cli_bin, options.playwright_mcp_bin)):
        parser.error("Execution requires the Chrome, OpenBrowser, Playwright CLI, and Playwright MCP binaries")
    root = options.run_output.resolve()
    root.mkdir(exist_ok=False)
    (root / "index.html").write_text(FIXTURE)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rows = []
    try:
        for arm in ARMS:
            binary = (
                options.openbrowser_bin
                if arm.startswith("openbrowser")
                else options.playwright_cli_bin
                if arm.endswith("cli")
                else options.playwright_mcp_bin
            )
            try:
                row = run_arm(
                    arm,
                    root / arm,
                    options.chrome_bin,
                    not options.without_feature,
                    binary,
                    f"http://127.0.0.1:{server.server_port}/",
                )
            except Exception as error:
                logging.exception("WebMCP capability failed for %s", arm)
                row = {"arm": arm, "error_type": type(error).__name__, "tasks": dict.fromkeys(plan["tasks"], False)}
            rows.append(row)
            (root / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    (root / "summary.json").write_text(
        json.dumps({**plan, "feature_enabled": not options.without_feature, "rows": rows}, indent=2) + "\n"
    )
    return 0 if all(all(row["tasks"].values()) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
