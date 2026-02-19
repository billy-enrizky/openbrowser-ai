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

All 11 browser tools will be registered as native OpenClaw agent tools.

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

## Available Tools

### Navigation

| Tool | Description |
|------|-------------|
| `browser_navigate` | Navigate to a URL, optionally in a new tab |
| `browser_go_back` | Go back to the previous page |
| `browser_scroll` | Scroll the page. Use `target_text` to find text and scroll to it |

### Interaction

| Tool | Description |
|------|-------------|
| `browser_click` | Click an element by its index |
| `browser_type` | Type text into an input field |

### Content Extraction

| Tool | Description |
|------|-------------|
| `browser_get_state` | Get page metadata and interactive elements. Use `filter_by`/`filter_query` to search elements |
| `browser_get_text` | Get page content as markdown. Use `search` param to grep with regex |
| `browser_get_accessibility_tree` | Get the page accessibility tree |
| `browser_execute_js` | Execute JavaScript in the page context |

### Tab and Session Management

| Tool | Description |
|------|-------------|
| `browser_tab` | Manage tabs: `action=list` / `switch` / `close` |
| `browser_session` | Manage sessions: `action=list` / `close` / `close_all` |

## Benchmark: Token Efficiency

### E2E LLM Benchmark (6 Real-World Tasks)

Six browser tasks run through Claude Sonnet 4.6 on AWS Bedrock. The LLM autonomously decides which tools to call. All three servers pass **6/6 tasks**. Token usage measured from actual MCP tool response sizes.

| MCP Server | Tools | Response Tokens | Tool Calls | vs OpenBrowser |
|------------|------:|----------------:|-----------:|---------------:|
| **Playwright MCP** | 22 | 283,853 | 10 | **170x more tokens** |
| **Chrome DevTools MCP** (Google) | 26 | 301,030 | 21 | **181x more tokens** |
| **OpenBrowser MCP** | 1 | **1,665** | 20 | baseline |

### Cost per Benchmark Run (6 Tasks)

| Model | Playwright MCP | Chrome DevTools MCP | OpenBrowser MCP |
|-------|---------------:|--------------------:|----------------:|
| Claude Sonnet ($3/M) | $0.852 | $0.903 | **$0.005** |
| Claude Opus ($15/M) | $4.258 | $4.515 | **$0.025** |

### Per-Task Response Size

| Task | Playwright MCP | Chrome DevTools MCP | OpenBrowser MCP |
|------|---------------:|--------------------:|----------------:|
| fact_lookup | 477,003 chars | 509,059 chars | 1,041 chars |
| form_fill | 4,075 chars | 3,150 chars | 2,410 chars |
| multi_page_extract | 58,099 chars | 38,593 chars | 513 chars |
| search_navigate | 518,461 chars | 594,458 chars | 1,996 chars |
| deep_navigation | 77,292 chars | 58,359 chars | 113 chars |
| content_analysis | 493 chars | 513 chars | 594 chars |

Playwright completes tasks in fewer tool calls (1-2 per task) because it dumps the full a11y snapshot on every navigation. OpenBrowser takes more round-trips but each response is compact -- the code extracts only what's needed.

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

## MCP Resources

When a browser session is active, three MCP resources are available:

| URI | Type | Description |
|-----|------|-------------|
| `browser://current-page/content` | text/markdown | Page content as markdown |
| `browser://current-page/state` | application/json | Interactive elements and metadata |
| `browser://current-page/accessibility` | application/json | Accessibility tree |

## Skills

The plugin includes 5 built-in skills that provide guided workflows for common browser automation tasks. Each skill is triggered automatically when the user's request matches its description.

| Skill | Directory | Description |
|-------|-----------|-------------|
| `web-scraping` | `skills/web-scraping/` | Extract structured data from websites, handle pagination, and multi-tab scraping |
| `form-filling` | `skills/form-filling/` | Fill out web forms, handle login/registration flows, and multi-step wizards |
| `e2e-testing` | `skills/e2e-testing/` | Test web applications end-to-end by simulating user interactions and verifying outcomes |
| `page-analysis` | `skills/page-analysis/` | Analyze page content, structure, metadata, and interactive elements |
| `accessibility-audit` | `skills/accessibility-audit/` | Audit pages for WCAG compliance, heading structure, labels, alt text, ARIA, and landmarks |

Each skill file (`SKILL.md`) contains YAML frontmatter with trigger conditions and a step-by-step workflow that references the MCP tools listed above.

## Testing and Benchmarks

```bash
# E2E test all 11 MCP tools against the published PyPI package
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
