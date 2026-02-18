# OpenBrowser - Claude Code Plugin

AI-powered browser automation for Claude Code. Control real web browsers directly from Claude -- navigate websites, fill forms, extract data, inspect accessibility trees, and automate multi-step workflows.

## Prerequisites

- **Chrome or Chromium** installed on your system
- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) package manager
- **Claude Code** CLI

## Installation

### From a marketplace (when available)

```bash
# Install via Claude Code plugin system
claude plugins install openbrowser
```

### Manual installation

Clone the plugin into your Claude Code plugins directory:

```bash
git clone https://github.com/billy-enrizky/openbrowser-ai.git
cd openbrowser-ai/plugin

# In your project, add the plugin path to .claude/settings.json:
# {
#   "plugins": ["/path/to/openbrowser-ai/plugin"]
# }
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

Measured on a 5-step workflow (navigate Wikipedia, get state, click, go back, get state) via JSON-RPC stdio. All numbers are real measurements -- no estimates.

| MCP Server | Tools | Response Tokens | Cost (Sonnet) | vs OpenBrowser |
|------------|------:|----------------:|--------------:|---------------:|
| **Playwright MCP** | 22 | 248,016 | $0.744 | 877x more |
| **Chrome DevTools MCP** (Google) | 26 | 134,802 | $0.404 | 476x more |
| **OpenBrowser MCP** | 11 | **283** | **$0.001** | baseline |

**What each server returns for navigate:**

| Server | Navigate Response | Size |
|--------|------------------|-----:|
| Playwright MCP | Full a11y snapshot (entire page tree with `[ref=eXX]` identifiers) | ~496K chars |
| Chrome DevTools MCP | `"Successfully navigated to URL. ## Pages 1: URL [selected]"` | ~136 chars |
| OpenBrowser MCP | `"Navigated to: URL"` | ~105 chars |

OpenBrowser returns minimal confirmations for actions and lets the agent request only the detail level it needs -- from 105 tokens (compact state) to 25K tokens (full page text). Each detail level is opt-in.

[Full comparison](https://docs.openbrowser.me/comparison)

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

**MCP server not found**: Verify `uvx` is installed (`pip install uv`) and `openbrowser-ai` is available (`uvx openbrowser-ai --version`).

**Session timeout**: Browser sessions auto-close after 10 minutes of inactivity. Use any tool to keep the session alive.

## License

MIT
