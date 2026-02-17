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
| `browser_scroll` | Scroll the page up or down |

### Interaction

| Tool | Description |
|------|-------------|
| `browser_click` | Click an element by its index |
| `browser_type` | Type text into an input field |

### Content Extraction

| Tool | Description |
|------|-------------|
| `browser_get_state` | Get page metadata and interactive elements (compact or full) |
| `browser_get_text` | Get page content as clean markdown |
| `browser_grep` | Search page text with regex or string patterns |
| `browser_extract_content` | Extract structured data using an LLM (requires API key) |

### DOM Inspection

| Tool | Description |
|------|-------------|
| `browser_search_elements` | Search elements by text, tag, id, class, or attribute |
| `browser_find_and_scroll` | Find text on page and scroll to it |
| `browser_get_accessibility_tree` | Get the page accessibility tree |
| `browser_execute_js` | Execute JavaScript in the page context |

### Tab Management

| Tool | Description |
|------|-------------|
| `browser_list_tabs` | List all open tabs |
| `browser_switch_tab` | Switch to a tab by ID |
| `browser_close_tab` | Close a tab by ID |

### Session Management

| Tool | Description |
|------|-------------|
| `browser_list_sessions` | List active browser sessions |
| `browser_close_session` | Close a specific session |
| `browser_close_all` | Close all browser sessions |

### Agent

| Tool | Description |
|------|-------------|
| `retry_with_openbrowser_agent` | Execute a task using the autonomous AI agent (requires API key) |

## Configuration

Some tools require additional configuration via environment variables:

| Variable | Required For | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | `browser_extract_content`, `retry_with_openbrowser_agent` | OpenAI API key for LLM-powered features |
| `ANTHROPIC_API_KEY` | Alternative to OpenAI | Anthropic API key |
| `OPENBROWSER_HEADLESS` | Optional | Set to `true` to run browser without GUI |
| `OPENBROWSER_ALLOWED_DOMAINS` | Optional | Comma-separated domain whitelist |

Set these in your `.mcp.json`:

```json
{
  "mcpServers": {
    "openbrowser": {
      "command": "uvx",
      "args": ["openbrowser-ai[mcp]", "--mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
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

## Troubleshooting

**Browser does not launch**: Ensure Chrome or Chromium is installed and accessible from PATH.

**MCP server not found**: Verify `uvx` is installed (`pip install uv`) and `openbrowser-ai` is available (`uvx openbrowser-ai --version`).

**Tools return errors about LLM**: `browser_extract_content` and `retry_with_openbrowser_agent` require an LLM API key. All other tools work without one.

**Session timeout**: Browser sessions auto-close after 10 minutes of inactivity. Use any tool to keep the session alive.

## License

MIT
