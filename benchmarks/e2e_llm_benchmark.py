"""
E2E LLM Performance Benchmark: OpenBrowser vs Playwright vs Chrome DevTools MCP.

Runs identical browser tasks through Claude (via claude-agent-sdk) and measures
task success, tool call count, wall-clock time, and cost.

Usage:
    uv run python benchmarks/e2e_llm_benchmark.py
    uv run python benchmarks/e2e_llm_benchmark.py --servers openbrowser
    uv run python benchmarks/e2e_llm_benchmark.py --tasks content_analysis fact_lookup
"""
import argparse
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

# Agent SDK spawns a Claude Code CLI subprocess. If this script is run from
# inside a Claude Code session the CLAUDECODE env var causes the subprocess to
# refuse to start ("cannot be launched inside another Claude Code session").
# Unsetting it before the import is safe -- this script does not need to run
# as a nested session.
os.environ.pop("CLAUDECODE", None)

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    ToolUseBlock,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


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

MODEL = None  # Use CLI default; override with --model
MAX_TURNS = 20
SYSTEM_PROMPT = (
    "You are a browser automation agent. Complete the task using the available "
    "browser tools. Be concise in your final answer."
)


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

async def run_task(
    server_name: str,
    server_config: dict,
    task: dict,
    model: str | None = None,
) -> dict:
    """Run a single task against a single MCP server via Claude Agent SDK.

    Returns dict with: name, success, duration_s, tool_calls, result, error.
    """
    task_name = task["name"]
    logger.info("  [%s/%s] Starting...", server_name, task_name)

    tool_call_count = 0
    result_text = ""
    error_msg = None
    start = time.monotonic()

    opts = ClaudeAgentOptions(
        mcp_servers={server_name: server_config},
        max_turns=MAX_TURNS,
        system_prompt=SYSTEM_PROMPT,
        permission_mode="bypassPermissions",
    )
    if model:
        opts.model = model

    try:
        async for message in query(
            prompt=task["prompt"],
            options=opts,
        ):
            # Count tool calls from assistant messages
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_call_count += 1

            # Capture final result
            if isinstance(message, ResultMessage):
                result_text = message.result or ""
                logger.info(
                    "  [%s/%s] Finished: %d turns, cost=$%.4f",
                    server_name, task_name,
                    message.num_turns,
                    message.total_cost_usd or 0,
                )

    except Exception as exc:
        error_msg = str(exc)
        logger.error("  [%s/%s] Error: %s", server_name, task_name, error_msg)

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
        "model": model or "default",
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
    servers_to_run = {
        name: config for name, config in SERVERS.items()
        if server_names is None or name in server_names
    }
    tasks_to_run = [
        t for t in TASKS
        if task_names is None or t["name"] in task_names
    ]

    logger.info("E2E LLM Benchmark")
    logger.info("Model: %s", model or "default")
    logger.info("Servers: %s", ", ".join(servers_to_run.keys()))
    logger.info("Tasks: %s", ", ".join(t["name"] for t in tasks_to_run))
    logger.info("")

    server_results = {}

    for server_name, server_config in servers_to_run.items():
        logger.info("=" * 60)
        logger.info("Server: %s", server_name)
        logger.info("=" * 60)

        task_results = []
        for task in tasks_to_run:
            result = await run_task(server_name, server_config, task, model=model)
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
    logger.info("E2E LLM Benchmark Results (%s)", model or "default")
    logger.info("=" * 60)
    logger.info("\n%s", format_summary_table(server_results))

    write_results(server_results, output_path, model=model)

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
        help="Model ID to use (default: CLI default)",
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
