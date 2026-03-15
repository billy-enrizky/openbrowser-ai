"""
E2E Benchmark: 4-Way CLI Comparison.

Compares openbrowser-ai, browser-use, playwright-cli, and agent-browser CLIs
driven by Claude Sonnet 4.6 on Bedrock (Converse API). All four approaches
use a single bash tool -- each gets an optimized system prompt.

Measures: task success, tool calls, Bedrock tokens, wall-clock time, response size.

Requires:
    - openbrowser-ai installed (uv pip install -e .)
    - browser-use installed (uvx --from "browser-use[cli]")
    - playwright-cli installed (npm install -g @anthropic-ai/playwright-cli)
    - agent-browser installed (npm install -g agent-browser)
    - AWS credentials with Bedrock access

Usage:
    uv run python benchmarks/e2e_4way_cli_benchmark.py
    uv run python benchmarks/e2e_4way_cli_benchmark.py --approaches openbrowser-ai browser-use
    uv run python benchmarks/e2e_4way_cli_benchmark.py --runs 5
"""
import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone

import boto3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
BEDROCK_REGION = "us-west-1"
MAX_TURNS = 20

ALL_APPROACHES = ["openbrowser-ai", "browser-use", "playwright-cli", "agent-browser"]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from terminal output."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _verify_fact_lookup(result: str) -> bool:
    lower = result.lower()
    return "guido van rossum" in lower and "1991" in lower


def _verify_form_fill(result: str) -> bool:
    lower = result.lower()
    return any(kw in lower for kw in ["submitted", "response", "custname", "post"])


def _verify_multi_page_extract(result: str) -> bool:
    lines = [line.strip() for line in result.split("\n") if len(line.split()) >= 3]
    return len(lines) >= 3


def _verify_search_navigate(result: str) -> bool:
    return "mozilla" in result.lower()


def _verify_deep_navigation(result: str) -> bool:
    return bool(re.search(r"\d+\.\d+\.\d+", result))


def _verify_content_analysis(result: str) -> bool:
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
# Browser cleanup
# ---------------------------------------------------------------------------

def _kill_stale_browsers():
    """Kill all Chrome/Chromium processes and wait until fully dead."""
    for pattern in ["chromium", "chrome", "Chromium", "Google Chrome"]:
        try:
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True, timeout=5)
        except Exception:
            pass
    for _ in range(20):
        result = subprocess.run(["pgrep", "-f", "chrom"], capture_output=True, timeout=5)
        if result.returncode != 0:
            break
        time.sleep(0.5)
    time.sleep(1)


# ---------------------------------------------------------------------------
# Shared bash tool definition
# ---------------------------------------------------------------------------

BASH_TOOL = {
    "toolSpec": {
        "name": "bash",
        "description": (
            "Run a shell command and return stdout+stderr. "
            "Commands can use pipes (|), redirects (>), and standard Unix tools."
        ),
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
            },
            "required": ["command"],
        }},
    }
}


# ---------------------------------------------------------------------------
# System prompts (optimized per approach)
# ---------------------------------------------------------------------------

_COMMON_PREAMBLE = (
    "You are a browser automation agent. Complete browser tasks efficiently. "
    "Give the answer directly, no markdown.\n\n"
    "RULES:\n"
    "1. Be efficient -- combine operations, minimize tool calls.\n"
    "2. After navigating, check page state before interacting.\n"
    "3. Give your final answer as plain text.\n"
    "4. Do NOT make separate calls to verify or summarize what you already found."
)

OPENBROWSER_SYSTEM_PROMPT = (
    _COMMON_PREAMBLE + "\n\n"
    "TOOLS:\n"
    "- openbrowser-ai CLI is available for browser automation.\n"
    "- Run `openbrowser-ai -c '<python code>'` to execute async Python code with browser.\n"
    "- Variables persist between openbrowser-ai -c calls (same browser session).\n"
    "- Key functions: navigate(url), click(index), input_text(index, text), "
    "evaluate(js_code), scroll(), send_keys(keys), get_dom_state()\n"
    "- All functions are async -- use await.\n\n"
    "OPTIMIZATION:\n"
    "- Batch multiple operations into a single openbrowser-ai -c call.\n"
    "- Use print() to output results.\n"
    "- Use evaluate() for direct JavaScript data extraction.\n\n"
    "SHELL ESCAPING:\n"
    "- Wrap code in SINGLE QUOTES: openbrowser-ai -c 'await navigate(\"url\")'\n"
    "- For JS with quotes, escape inner quotes or use different quote styles:\n"
    "  openbrowser-ai -c 'result = await evaluate(\"document.title\"); print(result)'"
)

BROWSER_USE_SYSTEM_PROMPT = (
    _COMMON_PREAMBLE + "\n\n"
    "TOOLS:\n"
    "- browser-use CLI is available for browser automation.\n"
    "- All commands prefixed with: uvx --from \"browser-use[cli]\" browser-use\n\n"
    "COMMANDS:\n"
    "- open <url>: Navigate to URL\n"
    "- state: Get page DOM with indexed elements [123]<tag>. New elements prefixed with *.\n"
    "- click <index>: Click element by index\n"
    "- input <index> \"text\": Click + type in one command (preferred over separate click + type)\n"
    "- type \"text\": Type into focused element\n"
    "- keys \"key\": Press keyboard key (Enter, Tab, Escape, etc.)\n"
    "- scroll down [--amount N]: Scroll page\n"
    "- eval \"js\": Execute JavaScript and return result\n"
    "- get title: Get page title (lightweight check)\n\n"
    "OPTIMIZATION:\n"
    "- Use `input <index> \"text\"` instead of separate click + type.\n"
    "- Use `eval \"js\"` for direct data extraction without full state refresh.\n"
    "- Use `keys \"Tab\"/\"Enter\"` for form navigation when index is not needed.\n"
    "- Minimize state calls -- they are the heaviest command."
)

PLAYWRIGHT_CLI_SYSTEM_PROMPT = (
    _COMMON_PREAMBLE + "\n\n"
    "TOOLS:\n"
    "- playwright-cli is available for browser automation.\n\n"
    "COMMANDS:\n"
    "- open <url>: Navigate to URL\n"
    "- snapshot: Take accessibility snapshot (saves to .playwright-cli/page-*.yml)\n"
    "  IMPORTANT: After running snapshot, use `cat .playwright-cli/page-*.yml` to read it.\n"
    "- click <ref>: Click element by ref (e.g., click @e5)\n"
    "- fill <ref> \"text\": Clear and fill input field\n"
    "- fill --submit <ref> \"text\": Fill and press Enter to submit\n"
    "- type <ref> \"text\": Type without clearing\n"
    "- press <key>: Press key (Enter, Tab, Escape, etc.)\n"
    "- select <ref> \"value\": Select dropdown option\n"
    "- check <ref>: Check checkbox\n"
    "- scroll down: Scroll page down\n"
    "- run-code \"async page => { ... }\": Execute Playwright JS code batch\n\n"
    "ELEMENT REFS:\n"
    "- Snapshot output shows elements with refs like @e1, @e2.\n"
    "- Use these refs in click, fill, type commands.\n\n"
    "OPTIMIZATION:\n"
    "- Use run-code for batching multiple Playwright operations in one call.\n"
    "- Use fill --submit to fill and submit forms in one action.\n"
    "- Snapshot saves to a file -- always read it with cat after running snapshot."
)

AGENT_BROWSER_SYSTEM_PROMPT = (
    _COMMON_PREAMBLE + "\n\n"
    "TOOLS:\n"
    "- agent-browser CLI is available for browser automation.\n\n"
    "COMMANDS:\n"
    "- open <url>: Navigate to URL\n"
    "- snapshot -i: Get compact interactive elements only (always use -i flag)\n"
    "- snapshot: Get full accessibility tree (verbose -- prefer -i)\n"
    "- click <ref>: Click element by ref (e.g., click @e5)\n"
    "- fill <ref> \"text\": Clear and fill input field\n"
    "- type <ref> \"text\": Type without clearing\n"
    "- press <key>: Press key (Enter, Tab, Escape, etc.)\n"
    "- select <ref> \"value\": Select dropdown option\n"
    "- check <ref>: Check checkbox\n"
    "- scroll down: Scroll page down\n"
    "- eval \"js\": Execute JavaScript and return result\n"
    "- find \"text\": Search visible text without full snapshot\n\n"
    "ELEMENT REFS:\n"
    "- Snapshot output shows elements with refs like @e1, @e2.\n"
    "- Use these refs in click, fill, type commands.\n\n"
    "OPTIMIZATION:\n"
    "- Always use `snapshot -i` (not bare snapshot) for 85-95% smaller output.\n"
    "- Chain commands with &&: agent-browser click @e5 && agent-browser snapshot -i\n"
    "- Use eval for direct JavaScript data extraction.\n"
    "- Use find to search for text without full snapshot."
)

# Approach name -> system prompt mapping
SYSTEM_PROMPTS = {
    "openbrowser-ai": OPENBROWSER_SYSTEM_PROMPT,
    "browser-use": BROWSER_USE_SYSTEM_PROMPT,
    "playwright-cli": PLAYWRIGHT_CLI_SYSTEM_PROMPT,
    "agent-browser": AGENT_BROWSER_SYSTEM_PROMPT,
}


# ---------------------------------------------------------------------------
# Client base class (shared by all 4 approaches)
# ---------------------------------------------------------------------------

class BaseBashClient:
    """Base class for all CLI benchmark clients. Provides shared bash tool interface."""

    def __init__(self, env: dict | None = None):
        self._env = env or {}

    def _make_env(self) -> dict:
        env = os.environ.copy()
        env.update(self._env)
        return env

    def get_bedrock_tools(self) -> list[dict]:
        return [BASH_TOOL]

    async def call_tool(self, name: str, arguments: dict) -> str:
        if name != "bash":
            return f"Error: Unknown tool {name}"
        command = arguments.get("command", "")
        try:
            result = subprocess.run(
                command, shell=True,
                env=self._make_env(), capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 300s"
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            stderr_lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
            if stderr_lines:
                output += "\nSTDERR: " + "\n".join(stderr_lines[-10:])
        return _strip_ansi(output.strip()) or "(no output)"

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Client: openbrowser-ai
# ---------------------------------------------------------------------------

class OpenBrowserClient(BaseBashClient):
    """openbrowser-ai CLI via persistent daemon."""

    async def start(self) -> None:
        env = self._make_env()
        # Auto-start daemon via self-documenting -c (no-arg) path
        subprocess.run(
            "openbrowser-ai -c", shell=True, env=env,
            capture_output=True, text=True, timeout=30,
        )
        # Warm up browser -- retry until ready
        for _ in range(5):
            r = subprocess.run(
                'openbrowser-ai -c "print(\'ready\')"', shell=True, env=env,
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0 and "ready" in r.stdout:
                logger.info("    openbrowser-ai ready -- daemon and browser warmed up")
                return
            time.sleep(2)
        logger.warning("    openbrowser-ai warm-up did not confirm -- proceeding anyway")

    async def stop(self) -> None:
        subprocess.run(
            "openbrowser-ai daemon stop",
            shell=True, env=self._make_env(), capture_output=True, timeout=15,
        )


# ---------------------------------------------------------------------------
# Client: browser-use
# ---------------------------------------------------------------------------

class BrowserUseClient(BaseBashClient):
    """browser-use CLI via uvx isolation."""

    _PREFIX = 'uvx --from "browser-use[cli]" browser-use'

    async def start(self) -> None:
        env = self._make_env()
        # Open blank page to spawn server + browser
        subprocess.run(
            f'{self._PREFIX} open about:blank', shell=True, env=env,
            capture_output=True, text=True, timeout=60,
        )
        # Verify with a lightweight check
        r = subprocess.run(
            f'{self._PREFIX} get title', shell=True, env=env,
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            logger.info("    browser-use ready -- server and browser warmed up")
        else:
            logger.warning("    browser-use warm-up did not confirm -- proceeding anyway")

    async def stop(self) -> None:
        env = self._make_env()
        subprocess.run(
            f'{self._PREFIX} close --all', shell=True, env=env,
            capture_output=True, timeout=15,
        )
        subprocess.run(
            f'{self._PREFIX} server stop', shell=True, env=env,
            capture_output=True, timeout=15,
        )


# ---------------------------------------------------------------------------
# Client: playwright-cli
# ---------------------------------------------------------------------------

class PlaywrightCliClient(BaseBashClient):
    """playwright-cli with persistent background process."""

    async def start(self) -> None:
        env = self._make_env()
        # Launch browser via snapshot of blank page
        subprocess.run(
            "playwright-cli snapshot about:blank", shell=True, env=env,
            capture_output=True, text=True, timeout=60,
        )
        logger.info("    playwright-cli ready -- browser launched")

    async def stop(self) -> None:
        subprocess.run(
            "playwright-cli close", shell=True, env=self._make_env(),
            capture_output=True, timeout=15,
        )


# ---------------------------------------------------------------------------
# Client: agent-browser
# ---------------------------------------------------------------------------

class AgentBrowserClient(BaseBashClient):
    """agent-browser CLI with persistent session."""

    async def start(self) -> None:
        env = self._make_env()
        # Open blank page to launch browser
        subprocess.run(
            "agent-browser open about:blank", shell=True, env=env,
            capture_output=True, text=True, timeout=60,
        )
        # Verify with compact snapshot
        r = subprocess.run(
            "agent-browser snapshot -i", shell=True, env=env,
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            logger.info("    agent-browser ready -- browser launched")
        else:
            logger.warning("    agent-browser warm-up did not confirm -- proceeding anyway")

    async def stop(self) -> None:
        subprocess.run(
            "agent-browser close", shell=True, env=self._make_env(),
            capture_output=True, timeout=15,
        )


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _create_client(approach: str) -> BaseBashClient:
    """Create the right client for each approach."""
    headless_env = {"OPENBROWSER_HEADLESS": "true", "TIMEOUT_BrowserStartEvent": "60"}
    if approach == "openbrowser-ai":
        return OpenBrowserClient(env=headless_env)
    if approach == "browser-use":
        return BrowserUseClient()  # headless by default in chromium mode
    if approach == "playwright-cli":
        return PlaywrightCliClient()  # headless by default
    if approach == "agent-browser":
        return AgentBrowserClient()  # headless by default
    raise ValueError(f"Unknown approach: {approach}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(data: list[float], n_bootstrap: int = 10000, ci: float = 0.95) -> dict:
    n = len(data)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    means = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    low_idx = int(math.floor(alpha * n_bootstrap))
    high_idx = int(math.ceil((1 - alpha) * n_bootstrap)) - 1
    sample_mean = sum(data) / n
    sample_std = (sum((x - sample_mean) ** 2 for x in data) / max(n - 1, 1)) ** 0.5
    return {
        "mean": round(sample_mean, 1),
        "std": round(sample_std, 1),
        "ci_low": round(means[low_idx], 1),
        "ci_high": round(means[high_idx], 1),
    }


def aggregate_results(task_results: list[dict]) -> dict:
    total = len(task_results)
    passed = sum(1 for t in task_results if t["success"])
    return {
        "total_tasks": total,
        "passed": passed,
        "total_duration_s": round(sum(t["duration_s"] for t in task_results), 1),
        "total_tool_calls": sum(t["tool_calls"] for t in task_results),
        "avg_tool_calls": round(sum(t["tool_calls"] for t in task_results) / total, 1) if total else 0,
        "bedrock_input_tokens": sum(t["bedrock_input_tokens"] for t in task_results),
        "bedrock_output_tokens": sum(t["bedrock_output_tokens"] for t in task_results),
        "total_bedrock_tokens": sum(t["total_bedrock_tokens"] for t in task_results),
        "response_chars": sum(t["response_chars"] for t in task_results),
    }


# ---------------------------------------------------------------------------
# Task runner (works with any BaseBashClient)
# ---------------------------------------------------------------------------

async def run_task(
    client: BaseBashClient,
    bedrock_client,
    task: dict,
    approach_name: str,
    model_id: str,
    system_prompt: str,
) -> dict:
    """Run a single task against a client via Bedrock Converse API."""
    task_name = task["name"]
    logger.info("  [%s/%s] Starting...", approach_name, task_name)

    tool_call_count = 0
    bedrock_input_tokens = 0
    bedrock_output_tokens = 0
    response_chars = 0
    result_text = ""
    error_msg = None
    start = time.monotonic()

    try:
        bedrock_tools = client.get_bedrock_tools()
        messages = [{"role": "user", "content": [{"text": task["prompt"]}]}]

        for turn in range(MAX_TURNS):
            converse_kwargs = {
                "modelId": model_id,
                "messages": messages,
                "system": [{"text": system_prompt}],
            }
            if bedrock_tools:
                converse_kwargs["toolConfig"] = {"tools": bedrock_tools}

            try:
                response = bedrock_client.converse(**converse_kwargs)
            except Exception as e:
                error_msg = f"Bedrock API error: {e}"
                logger.error("  [%s/%s] %s", approach_name, task_name, error_msg)
                break

            usage = response.get("usage", {})
            bedrock_input_tokens += usage.get("inputTokens", 0)
            bedrock_output_tokens += usage.get("outputTokens", 0)

            stop_reason = response.get("stopReason", "")
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            messages.append({"role": "assistant", "content": content_blocks})

            for block in content_blocks:
                if "text" in block:
                    logger.info(
                        "    [%s/%s] LLM >>> %s",
                        approach_name, task_name, block["text"][:300],
                    )

            tool_uses = [b for b in content_blocks if "toolUse" in b]

            if not tool_uses or stop_reason == "end_turn":
                for block in content_blocks:
                    if "text" in block:
                        result_text += block["text"] + "\n"
                logger.info(
                    "    [%s/%s] FINAL (stop=%s): %s",
                    approach_name, task_name, stop_reason, result_text[:300],
                )
                break

            tool_results = []
            for block in tool_uses:
                tool_use = block["toolUse"]
                tool_call_count += 1
                tool_name = tool_use["name"]
                tool_input = tool_use.get("input", {})
                tool_use_id = tool_use["toolUseId"]

                logger.info(
                    "    [%s/%s] Turn %d: %s(%s)",
                    approach_name, task_name, turn + 1,
                    tool_name, json.dumps(tool_input)[:200],
                )

                tool_output = await client.call_tool(tool_name, tool_input)
                response_chars += len(tool_output)

                logger.info(
                    "    [%s/%s] OUTPUT (%d chars): %s",
                    approach_name, task_name, len(tool_output),
                    tool_output[:500],
                )

                if len(tool_output) > 50000:
                    tool_output = tool_output[:50000] + "\n...(truncated)"

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": tool_output}],
                    }
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            error_msg = f"Hit max turns ({MAX_TURNS})"
            logger.warning("  [%s/%s] %s", approach_name, task_name, error_msg)

    except Exception as exc:
        error_msg = str(exc)
        logger.error("  [%s/%s] Error: %s", approach_name, task_name, error_msg)

    duration_s = time.monotonic() - start
    success = task["verify"](result_text) if not error_msg else False

    logger.info(
        "  [%s/%s] %s in %.1fs (%d calls, %d bedrock tokens, %d response chars)",
        approach_name, task_name,
        "PASS" if success else "FAIL",
        duration_s, tool_call_count,
        bedrock_input_tokens + bedrock_output_tokens, response_chars,
    )

    return {
        "name": task_name,
        "success": success,
        "duration_s": round(duration_s, 1),
        "tool_calls": tool_call_count,
        "bedrock_input_tokens": bedrock_input_tokens,
        "bedrock_output_tokens": bedrock_output_tokens,
        "total_bedrock_tokens": bedrock_input_tokens + bedrock_output_tokens,
        "response_chars": response_chars,
        "result": result_text[:500],
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    approach_names: list[str] | None = None,
    task_names: list[str] | None = None,
    n_runs: int = 3,
    n_bootstrap: int = 10000,
    model: str | None = None,
    output_path: str = "benchmarks/e2e_4way_cli_results.json",
):
    model_id = model or DEFAULT_MODEL
    tasks_to_run = [t for t in TASKS if task_names is None or t["name"] in task_names]
    approaches = [a for a in ALL_APPROACHES if approach_names is None or a in approach_names]

    logger.info("=" * 70)
    logger.info("4-Way CLI Benchmark")
    logger.info("=" * 70)
    logger.info("Model: %s", model_id)
    logger.info("Region: %s", BEDROCK_REGION)
    logger.info("Approaches: %s", ", ".join(approaches))
    logger.info("Tasks: %s", ", ".join(t["name"] for t in tasks_to_run))
    logger.info("Runs: %d", n_runs)
    logger.info("All approaches run headless with persistent daemon")
    logger.info("")

    bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    # Initialize per-approach accumulators
    accumulators = {}
    for approach in approaches:
        accumulators[approach] = {
            "run_durations": [],
            "run_tool_calls": [],
            "run_pass_counts": [],
            "run_input_tokens": [],
            "run_output_tokens": [],
            "run_total_tokens": [],
            "run_response_chars": [],
            "per_task_durations": {t["name"]: [] for t in tasks_to_run},
            "per_task_tool_calls": {t["name"]: [] for t in tasks_to_run},
            "per_task_pass": {t["name"]: [] for t in tasks_to_run},
            "per_task_input_tokens": {t["name"]: [] for t in tasks_to_run},
            "per_task_output_tokens": {t["name"]: [] for t in tasks_to_run},
            "per_task_response_chars": {t["name"]: [] for t in tasks_to_run},
            "raw_runs": [],
        }

    # Run benchmark: randomize approach and task order per run
    for run_idx in range(n_runs):
        logger.info("")
        logger.info("=" * 60)
        logger.info("RUN %d/%d", run_idx + 1, n_runs)
        logger.info("=" * 60)

        shuffled_approaches = approaches.copy()
        random.shuffle(shuffled_approaches)
        logger.info("Approach order: %s", ", ".join(shuffled_approaches))

        for approach in shuffled_approaches:
            shuffled_tasks = tasks_to_run.copy()
            random.shuffle(shuffled_tasks)

            logger.info("")
            logger.info("--- %s (run %d) ---", approach, run_idx + 1)
            logger.info("Task order: %s", ", ".join(t["name"] for t in shuffled_tasks))

            sys_prompt = SYSTEM_PROMPTS[approach]
            client = _create_client(approach)
            task_results = []

            await client.start()
            try:
                for task in shuffled_tasks:
                    result = await run_task(
                        client, bedrock_client, task, approach, model_id,
                        system_prompt=sys_prompt,
                    )
                    task_results.append(result)
            finally:
                await client.stop()
                _kill_stale_browsers()

            # Accumulate results
            acc = accumulators[approach]
            summary = aggregate_results(task_results)
            acc["run_durations"].append(summary["total_duration_s"])
            acc["run_tool_calls"].append(summary["total_tool_calls"])
            acc["run_pass_counts"].append(summary["passed"])
            acc["run_input_tokens"].append(summary["bedrock_input_tokens"])
            acc["run_output_tokens"].append(summary["bedrock_output_tokens"])
            acc["run_total_tokens"].append(summary["total_bedrock_tokens"])
            acc["run_response_chars"].append(summary["response_chars"])

            for tr in task_results:
                acc["per_task_durations"][tr["name"]].append(tr["duration_s"])
                acc["per_task_tool_calls"][tr["name"]].append(tr["tool_calls"])
                acc["per_task_pass"][tr["name"]].append(1 if tr["success"] else 0)
                acc["per_task_input_tokens"][tr["name"]].append(tr["bedrock_input_tokens"])
                acc["per_task_output_tokens"][tr["name"]].append(tr["bedrock_output_tokens"])
                acc["per_task_response_chars"][tr["name"]].append(tr["response_chars"])

            acc["raw_runs"].append({"run": run_idx + 1, "tasks": task_results, "summary": summary})

            logger.info(
                "  Run %d: %d/%d passed, %.1fs, %d calls, %d tokens",
                run_idx + 1, summary["passed"], summary["total_tasks"],
                summary["total_duration_s"], summary["total_tool_calls"],
                summary["total_bedrock_tokens"],
            )

    # Compute statistics
    all_results = {}
    for approach in approaches:
        acc = accumulators[approach]
        per_task_stats = {}
        for tn in acc["per_task_durations"]:
            per_task_stats[tn] = {
                "duration": bootstrap_ci(acc["per_task_durations"][tn], n_bootstrap),
                "tool_calls": bootstrap_ci(acc["per_task_tool_calls"][tn], n_bootstrap),
                "pass_rate": round(
                    sum(acc["per_task_pass"][tn]) / len(acc["per_task_pass"][tn]), 2
                ) if acc["per_task_pass"][tn] else 0,
                "input_tokens": bootstrap_ci(acc["per_task_input_tokens"][tn], n_bootstrap),
                "output_tokens": bootstrap_ci(acc["per_task_output_tokens"][tn], n_bootstrap),
                "response_chars": bootstrap_ci(acc["per_task_response_chars"][tn], n_bootstrap),
            }

        all_results[approach] = {
            "n_runs": n_runs,
            "raw_runs": acc["raw_runs"],
            "stats": {
                "duration_s": bootstrap_ci(acc["run_durations"], n_bootstrap),
                "tool_calls": bootstrap_ci(acc["run_tool_calls"], n_bootstrap),
                "pass_count": bootstrap_ci(acc["run_pass_counts"], n_bootstrap),
                "bedrock_input_tokens": bootstrap_ci(acc["run_input_tokens"], n_bootstrap),
                "bedrock_output_tokens": bootstrap_ci(acc["run_output_tokens"], n_bootstrap),
                "total_bedrock_tokens": bootstrap_ci(acc["run_total_tokens"], n_bootstrap),
                "response_chars": bootstrap_ci(acc["run_response_chars"], n_bootstrap),
            },
            "per_task": per_task_stats,
        }

        stats = all_results[approach]["stats"]
        logger.info("")
        logger.info("Approach %s summary (%d runs):", approach, n_runs)
        logger.info("  Duration: %.1f +/- %.1f s", stats["duration_s"]["mean"], stats["duration_s"]["std"])
        logger.info("  Tool calls: %.1f +/- %.1f", stats["tool_calls"]["mean"], stats["tool_calls"]["std"])
        logger.info("  Pass count: %.1f / %d", stats["pass_count"]["mean"], len(tasks_to_run))
        logger.info(
            "  Bedrock tokens: %.0f +/- %.0f (input: %.0f, output: %.0f)",
            stats["total_bedrock_tokens"]["mean"], stats["total_bedrock_tokens"]["std"],
            stats["bedrock_input_tokens"]["mean"], stats["bedrock_output_tokens"]["mean"],
        )

    # Print comparison
    _print_comparison(all_results, tasks_to_run, n_runs, n_bootstrap)

    # Write results
    output = {
        "model": model_id,
        "region": BEDROCK_REGION,
        "n_runs": n_runs,
        "n_bootstrap": n_bootstrap,
        "headless": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompts": {k: v for k, v in SYSTEM_PROMPTS.items() if k in approaches},
        "approaches": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("")
    logger.info("Results written to %s", output_path)

    return all_results


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def _print_comparison(all_results: dict, tasks: list, n_runs: int, n_bootstrap: int):
    """Print formatted comparison tables."""
    names = list(all_results.keys())
    col_w = max(len(n) for n in names) + 2

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON TABLE (%d runs, %d bootstrap)", n_runs, n_bootstrap)
    logger.info("=" * 70)

    header = f"{'Metric':<30s}"
    for name in names:
        header += f"{name:>{col_w + 14}s}"
    logger.info(header)
    logger.info("-" * len(header))

    for label, stat_key, fmt in [
        ("Accuracy", "pass_count", None),
        ("Duration (s)", "duration_s", ".1f"),
        ("Tool Calls", "tool_calls", ".1f"),
        ("Bedrock Input Tokens", "bedrock_input_tokens", ".0f"),
        ("Bedrock Output Tokens", "bedrock_output_tokens", ".0f"),
        ("Total Bedrock Tokens", "total_bedrock_tokens", ".0f"),
        ("Response Chars", "response_chars", ".0f"),
    ]:
        row = f"{label:<30s}"
        for name in names:
            if label == "Accuracy":
                s = all_results[name]["stats"]["pass_count"]
                n_tasks = len(tasks)
                pct = (s["mean"] / n_tasks * 100) if n_tasks else 0
                val = f"{s['mean']:.1f}/{n_tasks} ({pct:.0f}%)"
                row += f"{val:>{col_w + 14}s}"
            else:
                s = all_results[name]["stats"][stat_key]
                val = f"{s['mean']:{fmt}} +/- {s['std']:{fmt}}"
                row += f"{val:>{col_w + 14}s}"
        logger.info(row)

    # Per-task breakdown
    logger.info("")
    logger.info("PER-TASK PASS RATE:")
    header = f"{'Task':<25s}"
    for name in names:
        header += f"{name:>{col_w + 5}s}"
    logger.info(header)
    logger.info("-" * len(header))
    for task in tasks:
        row = f"{task['name']:<25s}"
        for name in names:
            rate = all_results[name]["per_task"][task["name"]]["pass_rate"]
            val = f"{rate*100:.0f}%"
            row += f"{val:>{col_w + 5}s}"
        logger.info(row)

    # Per-task tokens
    logger.info("")
    logger.info("PER-TASK BEDROCK TOKENS (mean):")
    header = f"{'Task':<25s}"
    for name in names:
        header += f"{name:>{col_w + 5}s}"
    logger.info(header)
    logger.info("-" * len(header))
    for task in tasks:
        row = f"{task['name']:<25s}"
        for name in names:
            it = all_results[name]["per_task"][task["name"]]["input_tokens"]["mean"]
            ot = all_results[name]["per_task"][task["name"]]["output_tokens"]["mean"]
            val = f"{it + ot:.0f}"
            row += f"{val:>{col_w + 5}s}"
        logger.info(row)

    # Error rate
    logger.info("")
    logger.info("ERROR RATE:")
    for name in names:
        s = all_results[name]["stats"]["pass_count"]
        n_tasks = len(tasks)
        error_rate = (1 - s["mean"] / n_tasks) * 100 if n_tasks else 0
        logger.info("  %s: %.1f%%", name, error_rate)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="4-Way CLI Benchmark")
    parser.add_argument(
        "--approaches", nargs="*",
        choices=ALL_APPROACHES,
        help="Approaches to benchmark (default: all four)",
    )
    parser.add_argument(
        "--tasks", nargs="*",
        choices=[t["name"] for t in TASKS],
        help="Tasks to run (default: all 6)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per approach (default: 3)")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap samples (default: 10000)")
    parser.add_argument("--model", default=None, help=f"Bedrock model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", default="benchmarks/e2e_4way_cli_results.json", help="Output JSON path")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        approach_names=args.approaches,
        task_names=args.tasks,
        n_runs=args.runs,
        n_bootstrap=args.bootstrap,
        model=args.model,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
