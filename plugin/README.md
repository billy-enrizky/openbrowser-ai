# OpenBrowser - Claude Code Plugin

AI-powered browser automation for Claude Code. Control real web browsers directly from Claude -- navigate websites, fill forms, extract data, inspect accessibility trees, and automate multi-step workflows.

## Prerequisites

- **Chrome or Chromium** installed on your system
- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) package manager
- **Claude Code** CLI

## Installation

### From GitHub marketplace

```bash
# Add the OpenBrowser marketplace (one-time)
claude plugin marketplace add billy-enrizky/openbrowser-ai

# Install the plugin
claude plugin install openbrowser@openbrowser-ai
```

This installs the MCP server, 5 skills, and auto-enables the plugin. Restart Claude Code to activate.

### Local development

```bash
# Test from a local clone without installing
claude --plugin-dir /path/to/openbrowser-ai/plugin
```

### OpenClaw

[OpenClaw](https://openclaw.ai) does not natively support MCP servers, but the community
[openclaw-mcp-adapter](https://github.com/androidStern-personal/openclaw-mcp-adapter) plugin
bridges MCP servers to OpenClaw agents.

1. Install the MCP adapter plugin (see its README for setup).

2. Add OpenBrowser as an MCP server in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "mcp-adapter": {
        "enabled": true,
        "config": {
          "servers": [
            {
              "name": "openbrowser",
              "transport": "stdio",
              "command": "uvx",
              "args": ["openbrowser-ai[mcp]", "--mcp"]
            }
          ]
        }
      }
    }
  }
}
```

The `execute_code` tool will be registered as a native OpenClaw agent tool.

For OpenClaw plugin documentation, see [docs.openclaw.ai/tools/plugin](https://docs.openclaw.ai/tools/plugin).

### Standalone MCP server (without plugin)

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "openbrowser": {
      "command": "uvx",
      "args": ["openbrowser-ai[mcp]", "--mcp"]
    }
  }
}
```

## Available Tool

The MCP server exposes a single `execute_code` tool that runs Python code in a persistent namespace with browser automation functions. The LLM writes Python code to navigate, interact, and extract data.

**Functions** (all async, use `await`):

| Category | Functions |
|----------|-----------|
| **Navigation** | `navigate(url, new_tab)`, `go_back()`, `wait(seconds)` |
| **Interaction** | `click(index)`, `input_text(index, text, clear)`, `scroll(down, pages, index)`, `send_keys(keys)`, `upload_file(index, path)` |
| **Dropdowns** | `select_dropdown(index, text)`, `dropdown_options(index)` |
| **Tabs** | `switch(tab_id)`, `close(tab_id)` |
| **JavaScript** | `evaluate(code)` -- run JS in page context, returns Python objects |
| **State** | `browser.get_browser_state_summary()` -- page metadata and interactive elements |
| **CSS** | `get_selector_from_index(index)` -- CSS selector for an element |
| **Completion** | `done(text, success)` -- signal task completion |

**Pre-imported libraries**: `json`, `csv`, `re`, `datetime`, `asyncio`, `Path`, `requests`, `numpy`, `pandas`, `matplotlib`, `BeautifulSoup`

## Benchmark: Token Efficiency

### E2E LLM Benchmark (6 Real-World Tasks, N=5 runs)

Six real-world browser tasks run through Claude Sonnet 4.6 on AWS Bedrock (Converse API) with a server-agnostic system prompt. The LLM autonomously decides which tools to call and when the task is complete. 5 runs per server with 10,000-sample bootstrap CIs.

| # | Task | Description | Target Site |
|:-:|------|-------------|-------------|
| 1 | **fact_lookup** | Navigate to a Wikipedia article and extract specific facts | en.wikipedia.org |
| 2 | **form_fill** | Fill out a multi-field form and submit | httpbin.org |
| 3 | **multi_page_extract** | Extract top 5 story titles from a dynamic page | news.ycombinator.com |
| 4 | **search_navigate** | Search Wikipedia, click result, extract info | en.wikipedia.org |
| 5 | **deep_navigation** | Find latest release version from a GitHub repo | github.com |
| 6 | **content_analysis** | Analyze page structure (headings, links, paragraphs) | example.com |

| MCP Server | Pass Rate | Duration (mean +/- std) | Tool Calls | Bedrock API Tokens |
|------------|:---------:|------------------------:|-----------:|-------------------:|
| **Playwright MCP** | 100% | 92.2 +/- 11.4s | 11.0 +/- 1.4 | 150,248 |
| **Chrome DevTools MCP** (Google) | 100% | 128.8 +/- 6.2s | 19.8 +/- 0.4 | 310,856 |
| **OpenBrowser MCP** | 100% | 103.1 +/- 16.4s | 15.0 +/- 3.9 | **49,423** |

OpenBrowser uses **3x fewer tokens** than Playwright and **6.3x fewer** than Chrome DevTools (Bedrock Converse API `usage` -- the actual billed tokens).

### Cost per Benchmark Run (6 Tasks)

Based on Bedrock API token usage (input + output tokens at respective rates).

| Model | Playwright MCP | Chrome DevTools MCP | OpenBrowser MCP |
|-------|---------------:|--------------------:|----------------:|
| Claude Sonnet ($3/$15 per M) | $0.47 | $0.96 | **$0.18** |
| Claude Opus ($15/$75 per M) | $2.35 | $4.78 | **$0.91** |

Playwright and Chrome DevTools return full page accessibility snapshots as tool output, which the LLM must process. OpenBrowser's CodeAgent processes browser state server-side in Python code, returning only extracted results to the LLM.

[Full comparison with methodology](https://docs.openbrowser.me/comparison)

## Configuration

Optional environment variables:

| Variable | Description |
|----------|-------------|
| `OPENBROWSER_HEADLESS` | Set to `true` to run browser without GUI |
| `OPENBROWSER_ALLOWED_DOMAINS` | Comma-separated domain whitelist |

Set these in your `.mcp.json`:

```json
{
  "mcpServers": {
    "openbrowser": {
      "command": "uvx",
      "args": ["openbrowser-ai[mcp]", "--mcp"],
      "env": {
        "OPENBROWSER_HEADLESS": "true"
      }
    }
  }
}
```

## Skills

The plugin includes 5 built-in skills that provide guided workflows for common browser automation tasks. Each skill is triggered automatically when the user's request matches its description.

| Skill | Directory | Description |
|-------|-----------|-------------|
| `web-scraping` | `skills/web-scraping/` | Extract structured data from websites, handle pagination, and multi-tab scraping |
| `form-filling` | `skills/form-filling/` | Fill out web forms, handle login/registration flows, and multi-step wizards |
| `e2e-testing` | `skills/e2e-testing/` | Test web applications end-to-end by simulating user interactions and verifying outcomes |
| `page-analysis` | `skills/page-analysis/` | Analyze page content, structure, metadata, and interactive elements |
| `accessibility-audit` | `skills/accessibility-audit/` | Audit pages for WCAG compliance, heading structure, labels, alt text, ARIA, and landmarks |

Each skill file (`SKILL.md`) contains YAML frontmatter with trigger conditions and a step-by-step workflow using the `execute_code` tool.

## Testing and Benchmarks

```bash
# E2E test the MCP server against the published PyPI package
uv run python benchmarks/e2e_published_test.py

# Run MCP benchmarks (5-step Wikipedia workflow)
uv run python benchmarks/openbrowser_benchmark.py
uv run python benchmarks/playwright_benchmark.py
uv run python benchmarks/cdp_benchmark.py
```

## Troubleshooting

**Browser does not launch**: Ensure Chrome or Chromium is installed and accessible from PATH.

**MCP server not found**: Verify `uvx` is installed (`pip install uv`) and the MCP server starts (`uvx openbrowser-ai[mcp] --mcp`).

**Session timeout**: Browser sessions auto-close after 10 minutes of inactivity. Use any tool to keep the session alive.

## License

MIT
