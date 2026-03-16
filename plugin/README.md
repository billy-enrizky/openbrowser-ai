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

This installs the MCP server, 6 skills, and auto-enables the plugin. Restart Claude Code to activate.

### Local development

```bash
# Test from a local clone without installing
claude --plugin-dir /path/to/openbrowser-ai/plugin
```

### OpenClaw

[OpenClaw](https://openclaw.ai) supports OpenBrowser via the CLI daemon. Install OpenBrowser,
then use `openbrowser-ai -c` from the Bash tool:

```bash
openbrowser-ai -c "await navigate('https://example.com')"
openbrowser-ai -c "print(await evaluate('document.title'))"
```

The daemon starts automatically on first use and persists variables across calls.

For OpenClaw plugin documentation, see [docs.openclaw.ai/tools/plugin](https://docs.openclaw.ai/tools/plugin).

### Standalone MCP server (without plugin)

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "openbrowser": {
      "command": "uvx",
      "args": ["openbrowser-ai", "--mcp"]
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
| **Downloads** | `download_file(url, filename)` -- download a file using browser cookies, `list_downloads()` (sync, no await) -- list downloaded files |
| **State** | `browser.get_browser_state_summary()` -- page metadata and interactive elements |
| **CSS** | `get_selector_from_index(index)` -- CSS selector for an element |

**Pre-imported libraries**: `json`, `csv`, `re`, `datetime`, `asyncio`, `Path`, `requests`

**Available if installed**: `numpy`/`np`, `pandas`/`pd`, `matplotlib`, `BeautifulSoup`, `PdfReader` (requires `pip install openbrowser-ai[pdf]`)

## Benchmark: Token Efficiency

### CLI Benchmark: 4-Way Comparison (6 Tasks, N=3 runs)

Four CLI tools compared with a single Bash tool each. Claude Sonnet 4.6 on Bedrock. Randomized order. All achieve 100% accuracy.

<p align="center">
  <img src="../benchmarks/cli_benchmark_scatter.png" alt="CLI Benchmark: Token Usage vs Duration" width="700" />
</p>

| CLI Tool | Duration | Tool Calls | Bedrock API Tokens |
|----------|----------|-----------|-------------------|
| **openbrowser-ai** | **84.8 +/- 10.9s** | **15.3 +/- 2.3** | **36,010 +/- 6,063** |
| browser-use | 106.0 +/- 9.5s | 20.7 +/- 6.4 | 77,123 +/- 33,354 |
| agent-browser | 99.0 +/- 6.8s | 25.0 +/- 4.0 | 90,107 +/- 3,698 |
| playwright-cli | 118.3 +/- 21.4s | 25.7 +/- 8.1 | 94,130 +/- 35,982 |

openbrowser-ai uses **2.1-2.6x fewer tokens** than all competitors via Python code batching and compact DOM representation. Raw data: [`benchmarks/e2e_4way_cli_results.json`](../benchmarks/e2e_4way_cli_results.json).

<p align="center">
  <img src="../benchmarks/cli_benchmark_per_task.png" alt="CLI Benchmark: Per-Task Token Usage" width="700" />
</p>

[Full 4-way comparison](https://docs.openbrowser.me/cli-comparison)

### E2E LLM Benchmark: MCP Server Comparison (6 Tasks, N=5 runs)

| MCP Server | Tools | Bedrock API Tokens | Tool Calls (mean) | vs OpenBrowser |
|------------|------:|-------------------:|-----------:|---------------:|
| **Playwright MCP** | 22 | 158,787 | 9.4 | **3.2x more tokens** |
| **Chrome DevTools MCP** (Google) | 26 | 299,486 | 19.4 | **6.0x more tokens** |
| **OpenBrowser MCP** | 1 | **50,195** | 13.8 | baseline |

| Model | Playwright MCP | Chrome DevTools MCP | OpenBrowser MCP |
|-------|---------------:|--------------------:|----------------:|
| Claude Sonnet 4.6 ($3/$15 per M) | $0.50 | $0.92 | **$0.18** |
| Claude Opus 4.6 ($5/$25 per M) | $0.83 | $1.53 | **$0.30** |

[Full MCP comparison with methodology](https://docs.openbrowser.me/comparison)

## CLI Execute Mode

Run browser automation directly from Bash without the MCP server:

```bash
# Execute code via persistent daemon
openbrowser-ai -c "await navigate('https://example.com')"
openbrowser-ai -c "print(await evaluate('document.title'))"

# Daemon management
openbrowser-ai daemon start|stop|status|restart
```

Variables persist across `-c` calls while the daemon is running. The daemon starts automatically on first use and shuts down after 10 minutes of inactivity.

## Configuration

Optional environment variables:

| Variable | Description |
|----------|-------------|
| `OPENBROWSER_HEADLESS` | Set to `true` to run browser without GUI |
| `OPENBROWSER_ALLOWED_DOMAINS` | Comma-separated domain whitelist |
| `OPENBROWSER_COMPACT_DESCRIPTION` | Set to `true` for minimal tool description (~500 tokens) |
| `OPENBROWSER_MAX_OUTPUT` | Maximum output characters per execution (default: 10,000) |
| `ANONYMIZED_TELEMETRY` | Set to `false` to disable anonymized usage telemetry (default: `true`) |

Set these in your `.mcp.json`:

```json
{
  "mcpServers": {
    "openbrowser": {
      "command": "uvx",
      "args": ["openbrowser-ai", "--mcp"],
      "env": {
        "OPENBROWSER_HEADLESS": "true"
      }
    }
  }
}
```

## Skills

The plugin includes 6 built-in skills that provide guided workflows for common browser automation tasks. All skills use the CLI-first approach via `openbrowser-ai -c` for direct code execution. Each skill is triggered automatically when the user's request matches its description.

| Skill | Directory | Description |
|-------|-----------|-------------|
| `web-scraping` | `skills/web-scraping/` | Extract structured data from websites, handle pagination, and multi-tab scraping |
| `form-filling` | `skills/form-filling/` | Fill out web forms, handle login/registration flows, and multi-step wizards |
| `e2e-testing` | `skills/e2e-testing/` | Test web applications end-to-end by simulating user interactions and verifying outcomes |
| `page-analysis` | `skills/page-analysis/` | Analyze page content, structure, metadata, and interactive elements |
| `accessibility-audit` | `skills/accessibility-audit/` | Audit pages for WCAG compliance, heading structure, labels, alt text, ARIA, and landmarks |
| `file-download` | `skills/file-download/` | Download files (PDFs, CSVs, images) using the browser's authenticated session and read content |

Each skill file (`SKILL.md`) contains YAML frontmatter with trigger conditions and a step-by-step workflow. All skills use `openbrowser-ai -c` via the Bash tool for CLI-first browser automation.

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

**MCP server not found**: Verify `uvx` is installed (`pip install uv`) and the MCP server starts (`uvx openbrowser-ai --mcp`).

**Session timeout**: Browser sessions auto-close after 10 minutes of inactivity. Use any tool to keep the session alive.

## License

MIT
