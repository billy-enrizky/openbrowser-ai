"""
E2E LLM Performance Benchmark: OpenBrowser vs Playwright vs Chrome DevTools MCP.

Runs identical browser tasks through Claude on Bedrock (via boto3 Converse API)
with MCP servers as tool providers. Measures task success, tool call count, and
wall-clock time.

Requires AWS credentials with Bedrock access:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-west-2

Usage:
    uv run python benchmarks/e2e_llm_benchmark.py
    uv run python benchmarks/e2e_llm_benchmark.py --servers openbrowser
    uv run python benchmarks/e2e_llm_benchmark.py --tasks content_analysis fact_lookup
    uv run python benchmarks/e2e_llm_benchmark.py --model us.anthropic.claude-sonnet-4-6
"""
import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP Client -- communicates with MCP servers via JSON-RPC over stdio
# ---------------------------------------------------------------------------

class MCPClient:
    """Minimal MCP client that starts a server subprocess and communicates
    via JSON-RPC 2.0 over stdin/stdout."""

    def __init__(self, command: str, args: list[str], env: dict | None = None):
        self._command = command
        self._args = args
        self._env = env
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._tools: list[dict] = []

    async def start(self) -> None:
        """Start the MCP server subprocess and initialize."""
        proc_env = os.environ.copy()
        if self._env:
            proc_env.update(self._env)

        self._process = subprocess.Popen(
            [self._command] + self._args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
        )

        # Initialize
        resp = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "benchmark", "version": "1.0"},
        })
        if "error" in resp:
            raise RuntimeError(f"MCP initialize failed: {resp['error']}")

        # Send initialized notification
        self._send_notification("notifications/initialized", {})

        # Get tools
        tools_resp = self._send_request("tools/list", {})
        if "error" in tools_resp:
            raise RuntimeError(f"MCP tools/list failed: {tools_resp['error']}")
        self._tools = tools_resp.get("result", {}).get("tools", [])
        logger.info("    MCP server started with %d tools", len(self._tools))

    def get_bedrock_tools(self) -> list[dict]:
        """Convert MCP tools to Bedrock Converse toolConfig format."""
        bedrock_tools = []
        for tool in self._tools:
            schema = tool.get("inputSchema", {"type": "object", "properties": {}})
            # Bedrock requires properties to exist
            if "properties" not in schema:
                schema["properties"] = {}
            bedrock_tools.append({
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool.get("description", "")[:4096],
                    "inputSchema": {"json": schema},
                }
            })
        return bedrock_tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call an MCP tool and return the text result."""
        resp = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        if "error" in resp:
            return f"Error: {resp['error']}"
        result = resp.get("result", {})
        content = result.get("content", [])
        texts = []
        for item in content:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts) if texts else "(no output)"

    async def stop(self) -> None:
        """Stop the MCP server subprocess and kill any browser it launched."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
        # Kill Chrome to avoid profile lock conflicts between tasks
        for sig_name in ["chromium", "chrome"]:
            subprocess.run(
                ["pkill", "-f", sig_name],
                capture_output=True, timeout=5,
            )
        await asyncio.sleep(1)

    def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and read the response."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }
        return self._send_and_read(request)

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        assert self._process and self._process.stdin
        line = json.dumps(notification) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

    def _send_and_read(self, request: dict) -> dict:
        """Send request and read the matching response by id, skipping
        notifications and out-of-order messages."""
        assert self._process and self._process.stdin and self._process.stdout
        line = json.dumps(request) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

        expected_id = request.get("id")

        # Keep reading lines until we find the response with matching id
        for _ in range(200):  # safety limit
            response_line = self._process.stdout.readline()
            if not response_line:
                return {"error": "No response from MCP server (EOF)"}
            try:
                resp = json.loads(response_line)
            except json.JSONDecodeError:
                continue  # skip non-JSON lines (e.g. log output)

            # Skip JSON-RPC notifications (no id field)
            if "id" not in resp:
                continue

            # Match by request id
            if resp.get("id") == expected_id:
                return resp

            # Wrong id -- skip (stale response from a previous request)
            logger.debug("    Skipped response id=%s (expected %s)", resp.get("id"), expected_id)

        return {"error": f"No matching response for request id={expected_id}"}


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _verify_fact_lookup(result: str) -> bool:
    """Output must contain 'Guido van Rossum' AND '1991'."""
    lower = result.lower()
    return "guido van rossum" in lower and "1991" in lower


def _verify_form_fill(result: str) -> bool:
    """Output must mention form submission or response data."""
    lower = result.lower()
    return any(kw in lower for kw in ["submitted", "response", "custname", "post"])


def _verify_multi_page_extract(result: str) -> bool:
    """Output must contain at least 3 distinct multi-word strings (story titles)."""
    lines = [line.strip() for line in result.split("\n") if len(line.split()) >= 3]
    return len(lines) >= 3


def _verify_search_navigate(result: str) -> bool:
    """Output must contain 'Mozilla'."""
    return "mozilla" in result.lower()


def _verify_deep_navigation(result: str) -> bool:
    """Output must contain a version number pattern (digits.digits.digits)."""
    return bool(re.search(r"\d+\.\d+\.\d+", result))


def _verify_content_analysis(result: str) -> bool:
    """Output must contain numeric counts for headings, links, and paragraphs."""
    lower = result.lower()
    has_numbers = bool(re.search(r"\d+", result))
    has_terms = sum(1 for term in ["heading", "link", "paragraph"] if term in lower)
    return has_numbers and has_terms >= 2


TASKS = [
    {
        "name": "fact_lookup",
        "prompt": (
            "Go to the Python Wikipedia page "
            "(https://en.wikipedia.org/wiki/Python_(programming_language)) "
            "and find who created Python and in what year."
        ),
        "verify": _verify_fact_lookup,
    },
    {
        "name": "form_fill",
        "prompt": (
            "Navigate to httpbin.org/forms/post, fill in Customer name: John Doe, "
            "choose Medium pizza size, select Mushroom topping, and submit the form."
        ),
        "verify": _verify_form_fill,
    },
    {
        "name": "multi_page_extract",
        "prompt": (
            "Go to news.ycombinator.com and extract the titles of the top 5 stories."
        ),
        "verify": _verify_multi_page_extract,
    },
    {
        "name": "search_navigate",
        "prompt": (
            "Go to en.wikipedia.org, search for 'Rust programming language', "
            "click the result, and tell me what company originally developed it."
        ),
        "verify": _verify_search_navigate,
    },
    {
        "name": "deep_navigation",
        "prompt": (
            "Go to github.com/anthropics/claude-code, "
            "find the latest release version number."
        ),
        "verify": _verify_deep_navigation,
    },
    {
        "name": "content_analysis",
        "prompt": (
            "Go to example.com and describe the page structure: "
            "how many headings, links, and paragraphs."
        ),
        "verify": _verify_content_analysis,
    },
]


# ---------------------------------------------------------------------------
# MCP server configurations
# ---------------------------------------------------------------------------

SERVERS = {
    "openbrowser": {
        "command": "uvx",
        "args": ["openbrowser-ai[mcp]", "--mcp"],
    },
    "openbrowser-1tools": {
        "command": "uvx",
        "args": ["openbrowser-ai[mcp]==0.1.21", "--mcp"],
    },
    "playwright": {
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
    },
    "chrome-devtools": {
        "command": "npx",
        "args": ["-y", "chrome-devtools-mcp@latest"],
        "env": {"CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS": "1"},
    },
}

DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MAX_TURNS = 50
SYSTEM_PROMPT = (
    "You are a browser automation agent. Complete the task using the available "
    "browser tools. Be concise in your final answer."
)


# ---------------------------------------------------------------------------
# Agent runner using Bedrock Converse API
# ---------------------------------------------------------------------------

async def run_task(
    bedrock_client,
    server_name: str,
    server_config: dict,
    task: dict,
    model_id: str,
) -> dict:
    """Run a single task against a single MCP server via Bedrock Converse API.

    Returns dict with: name, success, duration_s, tool_calls, result, error.
    """
    task_name = task["name"]
    logger.info("  [%s/%s] Starting...", server_name, task_name)

    tool_call_count = 0
    result_text = ""
    error_msg = None
    start = time.monotonic()

    # Start MCP server
    mcp = MCPClient(
        command=server_config["command"],
        args=server_config["args"],
        env=server_config.get("env"),
    )

    try:
        await mcp.start()
        bedrock_tools = mcp.get_bedrock_tools()

        messages = [
            {"role": "user", "content": [{"text": task["prompt"]}]}
        ]

        for turn in range(MAX_TURNS):
            # Call Bedrock Converse
            converse_kwargs = {
                "modelId": model_id,
                "messages": messages,
                "system": [{"text": SYSTEM_PROMPT}],
            }
            if bedrock_tools:
                converse_kwargs["toolConfig"] = {"tools": bedrock_tools}

            try:
                response = bedrock_client.converse(**converse_kwargs)
            except Exception as e:
                error_msg = f"Bedrock API error: {e}"
                logger.error("  [%s/%s] %s", server_name, task_name, error_msg)
                break

            stop_reason = response.get("stopReason", "")
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])

            # Add assistant message to history
            messages.append({"role": "assistant", "content": content_blocks})

            # Check for tool use
            tool_uses = [b for b in content_blocks if "toolUse" in b]

            if not tool_uses or stop_reason == "end_turn":
                # No tool calls -- extract final text response
                for block in content_blocks:
                    if "text" in block:
                        result_text += block["text"] + "\n"
                break

            # Execute tool calls and collect results
            tool_results = []
            for block in tool_uses:
                tool_use = block["toolUse"]
                tool_call_count += 1
                tool_name = tool_use["name"]
                tool_input = tool_use.get("input", {})
                tool_use_id = tool_use["toolUseId"]

                # Log the code being sent (first 200 chars)
                code_preview = tool_input.get("code", "")[:200]
                logger.info(
                    "    [%s/%s] Turn %d: %s -- %s",
                    server_name, task_name, turn + 1, tool_name,
                    code_preview.replace("\n", " | "),
                )

                # Call MCP tool with timing
                tool_start = time.monotonic()
                tool_output = await mcp.call_tool(tool_name, tool_input)
                tool_duration = time.monotonic() - tool_start
                logger.info(
                    "    [%s/%s] Turn %d result (%.1fs, %d chars): %s",
                    server_name, task_name, turn + 1,
                    tool_duration, len(tool_output),
                    tool_output[:200].replace("\n", " | "),
                )

                # Truncate large outputs to avoid token limits
                if len(tool_output) > 50000:
                    tool_output = tool_output[:50000] + "\n...(truncated)"

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": tool_output}],
                    }
                })

            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})

        else:
            # Hit MAX_TURNS without natural completion
            error_msg = f"Hit max turns ({MAX_TURNS})"
            logger.warning("  [%s/%s] %s", server_name, task_name, error_msg)

    except Exception as exc:
        error_msg = str(exc)
        logger.error("  [%s/%s] Error: %s", server_name, task_name, error_msg)
    finally:
        await mcp.stop()

    duration_s = time.monotonic() - start
    success = task["verify"](result_text) if not error_msg else False

    logger.info(
        "  [%s/%s] %s in %.1fs (%d tool calls)",
        server_name, task_name,
        "PASS" if success else "FAIL",
        duration_s, tool_call_count,
    )

    return {
        "name": task_name,
        "success": success,
        "duration_s": round(duration_s, 1),
        "tool_calls": tool_call_count,
        "result": result_text[:500],
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Results aggregation and output
# ---------------------------------------------------------------------------

def aggregate_results(task_results: list[dict]) -> dict:
    """Compute summary statistics for a list of task results."""
    total = len(task_results)
    passed = sum(1 for t in task_results if t["success"])
    total_duration = sum(t["duration_s"] for t in task_results)
    total_tools = sum(t["tool_calls"] for t in task_results)
    return {
        "total_tasks": total,
        "passed": passed,
        "total_duration_s": round(total_duration, 1),
        "total_tool_calls": total_tools,
        "avg_tool_calls": round(total_tools / total, 1) if total else 0,
    }


def format_summary_table(server_results: dict) -> str:
    """Format a console-friendly comparison table."""
    names = list(server_results.keys())
    col_width = max(len(n) for n in names) + 4

    header = f"{'Metric':<25s}"
    for name in names:
        header += f"{name:>{col_width}s}"

    rows = [header, "=" * len(header)]

    for label, key, fmt in [
        ("Tasks Passed", None, None),
        ("Total Duration (s)", "total_duration_s", ".1f"),
        ("Total Tool Calls", "total_tool_calls", "d"),
        ("Avg Tool Calls/Task", "avg_tool_calls", ".1f"),
    ]:
        row = f"{label:<25s}"
        for name in names:
            s = server_results[name]["summary"]
            if key is None:
                val = f"{s['passed']}/{s['total_tasks']}"
                row += f"{val:>{col_width}s}"
            else:
                val = s[key]
                row += f"{val:>{col_width}{fmt}}"
        rows.append(row)

    return "\n".join(rows)


def write_results(server_results: dict, output_path: str, model: str | None = None):
    """Write structured results to JSON."""
    output = {
        "model": model or DEFAULT_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "servers": {},
    }
    for name, data in server_results.items():
        output["servers"][name] = {
            "tasks": data["tasks"],
            "summary": data["summary"],
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", output_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_benchmark(
    server_names: list[str] | None = None,
    task_names: list[str] | None = None,
    output_path: str = "benchmarks/e2e_llm_results.json",
    model: str | None = None,
):
    """Run the full benchmark suite."""
    model_id = model or DEFAULT_MODEL

    servers_to_run = {
        name: config for name, config in SERVERS.items()
        if server_names is None or name in server_names
    }
    tasks_to_run = [
        t for t in TASKS
        if task_names is None or t["name"] in task_names
    ]

    logger.info("E2E LLM Benchmark (Bedrock Converse API)")
    logger.info("Model: %s", model_id)
    logger.info("Servers: %s", ", ".join(servers_to_run.keys()))
    logger.info("Tasks: %s", ", ".join(t["name"] for t in tasks_to_run))
    logger.info("")

    # Create Bedrock client
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

    server_results = {}

    for server_name, server_config in servers_to_run.items():
        logger.info("=" * 60)
        logger.info("Server: %s", server_name)
        logger.info("=" * 60)

        task_results = []
        for task in tasks_to_run:
            result = await run_task(
                bedrock_client, server_name, server_config, task, model_id,
            )
            task_results.append(result)

        summary = aggregate_results(task_results)
        server_results[server_name] = {
            "tasks": task_results,
            "summary": summary,
        }

        logger.info(
            "  Server %s: %d/%d passed, %.1fs total, %d tool calls",
            server_name, summary["passed"], summary["total_tasks"],
            summary["total_duration_s"], summary["total_tool_calls"],
        )
        logger.info("")

    # Output
    logger.info("=" * 60)
    logger.info("E2E LLM Benchmark Results (%s)", model_id)
    logger.info("=" * 60)
    logger.info("\n%s", format_summary_table(server_results))

    write_results(server_results, output_path, model=model_id)

    return server_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="E2E LLM Performance Benchmark")
    parser.add_argument(
        "--servers", nargs="*", choices=list(SERVERS.keys()),
        help="Servers to benchmark (default: all)",
    )
    parser.add_argument(
        "--tasks", nargs="*", choices=[t["name"] for t in TASKS],
        help="Tasks to run (default: all)",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Bedrock model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", default="benchmarks/e2e_llm_results.json",
        help="Output JSON path (default: benchmarks/e2e_llm_results.json)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        server_names=args.servers,
        task_names=args.tasks,
        output_path=args.output,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
