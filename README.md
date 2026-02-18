# OpenBrowser

**Automating Walmart Product Scraping:**

https://github.com/user-attachments/assets/ae5d74ce-0ac6-46b0-b02b-ff5518b4b20d


**OpenBrowserAI Automatic Flight Booking:**

https://github.com/user-attachments/assets/632128f6-3d09-497f-9e7d-e29b9cb65e0f


[![PyPI version](https://badge.fury.io/py/openbrowser-ai.svg)](https://pypi.org/project/openbrowser-ai/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/billy-enrizky/openbrowser-ai/actions/workflows/test.yml/badge.svg)](https://github.com/billy-enrizky/openbrowser-ai/actions)

**AI-powered browser automation using LangGraph and CDP (Chrome DevTools Protocol)**

OpenBrowser is a framework for intelligent browser automation. It combines direct CDP communication with LangGraph orchestration to create AI agents that can navigate, interact with, and extract information from web pages autonomously.

## Table of Contents

- [Documentation](#documentation)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Supported LLM Providers](#supported-llm-providers)
- [Claude Code Plugin](#claude-code-plugin)
- [Codex](#codex)
- [OpenCode](#opencode)
- [OpenClaw](#openclaw)
- [MCP Server](#mcp-server)
- [CLI Usage](#cli-usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Documentation

**Full documentation**: [https://docs.openbrowser.me](https://docs.openbrowser.me)

## Key Features

- **LangGraph-Powered Agents** - Stateful workflow orchestration with perceive-plan-execute loop
- **Raw CDP Communication** - Direct Chrome DevTools Protocol for maximum control and speed
- **Vision Support** - Screenshot analysis for visual understanding of pages
- **12+ LLM Providers** - OpenAI, Anthropic, Google, Groq, AWS Bedrock, Azure OpenAI, Ollama, and more
- **Code Agent Mode** - Jupyter notebook-like code execution for complex automation
- **MCP Server** - Model Context Protocol support for Claude Desktop integration
- **Video Recording** - Record browser sessions as video files

## Installation

```bash
pip install openbrowser-ai
```

### With Optional Dependencies

```bash
# Install with all LLM providers
pip install openbrowser-ai[all]

# Install specific providers
pip install openbrowser-ai[anthropic]  # Anthropic Claude
pip install openbrowser-ai[groq]       # Groq
pip install openbrowser-ai[ollama]     # Ollama (local models)
pip install openbrowser-ai[aws]        # AWS Bedrock
pip install openbrowser-ai[azure]      # Azure OpenAI

# Install with video recording support
pip install openbrowser-ai[video]
```

### Install Browser

```bash
uvx openbrowser install
# or
playwright install chromium
```

## Quick Start

### Basic Usage

```python
import asyncio
from openbrowser import Agent, ChatGoogle

async def main():
    agent = Agent(
        task="Go to google.com and search for 'Python tutorials'",
        llm=ChatGoogle(),
    )
    
    result = await agent.run()
    print(f"Result: {result}")

asyncio.run(main())
```

### With Different LLM Providers

```python
from openbrowser import Agent, ChatOpenAI, ChatAnthropic, ChatGoogle

# OpenAI
agent = Agent(task="...", llm=ChatOpenAI(model="gpt-4o"))

# Anthropic
agent = Agent(task="...", llm=ChatAnthropic(model="claude-sonnet-4-0"))

# Google Gemini
agent = Agent(task="...", llm=ChatGoogle(model="gemini-2.0-flash"))
```

### Using Browser Session Directly

```python
import asyncio
from openbrowser import BrowserSession, BrowserProfile

async def main():
    profile = BrowserProfile(
        headless=True,
        viewport_width=1920,
        viewport_height=1080,
    )
    
    session = BrowserSession(browser_profile=profile)
    await session.start()
    
    await session.navigate_to("https://example.com")
    screenshot = await session.screenshot()
    
    await session.stop()

asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
# Google (recommended)
export GOOGLE_API_KEY="..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Groq
export GROQ_API_KEY="gsk_..."

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-west-2"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Browser-Use LLM (external service)
export BROWSER_USE_API_KEY="..."
```

### BrowserProfile Options

```python
from openbrowser import BrowserProfile

profile = BrowserProfile(
    headless=True,
    viewport_width=1280,
    viewport_height=720,
    disable_security=False,
    extra_chromium_args=["--disable-gpu"],
    record_video_dir="./recordings",
    proxy={
        "server": "http://proxy.example.com:8080",
        "username": "user",
        "password": "pass",
    },
)
```

## Supported LLM Providers

| Provider | Class | Models |
|----------|-------|--------|
| **Google** | `ChatGoogle` | gemini-2.0-flash, gemini-1.5-pro |
| **OpenAI** | `ChatOpenAI` | gpt-4o, o3, gpt-4-turbo |
| **Anthropic** | `ChatAnthropic` | claude-sonnet-4-0, claude-3-opus |
| **Groq** | `ChatGroq` | llama-3.3-70b-versatile, mixtral-8x7b |
| **AWS Bedrock** | `ChatAWSBedrock` | claude-3, amazon.titan |
| **Azure OpenAI** | `ChatAzureOpenAI` | Any Azure-deployed model |
| **Ollama** | `ChatOllama` | llama3, mistral (local) |
| **OCI** | `ChatOCIRaw` | Oracle Cloud GenAI models |
| **Browser-Use** | `ChatBrowserUse` | External LLM service |

## Claude Code Plugin

OpenBrowser is available as a Claude Code plugin with 5 built-in skills:

| Skill | Description |
|-------|-------------|
| `web-scraping` | Extract structured data, handle pagination |
| `form-filling` | Fill forms, login flows, multi-step wizards |
| `e2e-testing` | Test web apps by simulating user interactions |
| `page-analysis` | Analyze page content, structure, metadata |
| `accessibility-audit` | Audit pages for WCAG compliance |

See [plugin/README.md](plugin/README.md) for installation and detailed tool parameter documentation.

## Codex

OpenBrowser works with OpenAI Codex via native skill discovery.

### Quick Install

Tell Codex:

```
Fetch and follow instructions from https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/refs/heads/main/.codex/INSTALL.md
```

### Manual Install

```bash
# Clone the repository
git clone https://github.com/billy-enrizky/openbrowser-ai.git ~/.codex/openbrowser

# Symlink skills for native discovery
mkdir -p ~/.agents/skills
ln -s ~/.codex/openbrowser/plugin/skills ~/.agents/skills/openbrowser

# Restart Codex
```

Then configure the MCP server in your project (see [MCP Server](#mcp-server) below).

Detailed docs: [.codex/INSTALL.md](.codex/INSTALL.md)

## OpenCode

OpenBrowser works with [OpenCode.ai](https://opencode.ai) via plugin and skill symlinks.

### Quick Install

Tell OpenCode:

```
Fetch and follow instructions from https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/refs/heads/main/.opencode/INSTALL.md
```

### Manual Install

```bash
# Clone the repository
git clone https://github.com/billy-enrizky/openbrowser-ai.git ~/.config/opencode/openbrowser

# Create directories
mkdir -p ~/.config/opencode/plugins ~/.config/opencode/skills

# Symlink plugin and skills
ln -s ~/.config/opencode/openbrowser/.opencode/plugins/openbrowser.js ~/.config/opencode/plugins/openbrowser.js
ln -s ~/.config/opencode/openbrowser/plugin/skills ~/.config/opencode/skills/openbrowser

# Restart OpenCode
```

Then configure the MCP server in your project (see [MCP Server](#mcp-server) below).

Detailed docs: [.opencode/INSTALL.md](.opencode/INSTALL.md)

## OpenClaw

OpenBrowser works with [OpenClaw](https://openclaw.ai) via its plugin system.

### Install via Plugin Registry

```bash
openclaw plugin install openbrowser
```

### Manual Install

Clone the repository and register the plugin:

```bash
git clone https://github.com/billy-enrizky/openbrowser-ai.git
openclaw plugin add ./openbrowser-ai/plugin
```

Then configure the MCP server in your project (see [MCP Server](#mcp-server) below).

For OpenClaw plugin documentation, see [https://docs.openclaw.ai/tools/plugin](https://docs.openclaw.ai/tools/plugin).

## MCP Server

OpenBrowser includes an MCP (Model Context Protocol) server that exposes browser automation as tools for AI assistants like Claude. No external LLM API keys required -- the MCP client (Claude) provides the intelligence.

### Quick Setup

**Claude Code** -- add to your project's `.mcp.json`:

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

**Claude Desktop** -- add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

**Run directly:**

```bash
uvx openbrowser-ai[mcp] --mcp
```

### Tools (18)

#### Navigation

| Tool | Description |
|------|-------------|
| `browser_navigate` | Navigate to a URL, optionally in a new tab |
| `browser_go_back` | Go back to the previous page |
| `browser_scroll` | Scroll the page up or down |

#### Interaction

| Tool | Description |
|------|-------------|
| `browser_click` | Click an element by its index |
| `browser_type` | Type text into an input field |

#### Content Extraction

| Tool | Description |
|------|-------------|
| `browser_get_state` | Get page metadata and interactive elements (compact or full) |
| `browser_get_text` | Get page content as clean markdown |
| `browser_grep` | Search page text with regex or string patterns |

#### DOM Inspection

| Tool | Description |
|------|-------------|
| `browser_search_elements` | Search elements by text, tag, id, class, or attribute |
| `browser_find_and_scroll` | Find text on page and scroll to it |
| `browser_get_accessibility_tree` | Get page a11y tree (tree or flat format, depth limit) |
| `browser_execute_js` | Execute JavaScript in page context (await/fire-and-forget, by-value/by-reference) |

#### Tab Management

| Tool | Description |
|------|-------------|
| `browser_list_tabs` | List all open tabs |
| `browser_switch_tab` | Switch to a tab by ID |
| `browser_close_tab` | Close a tab by ID |

#### Session Management

| Tool | Description |
|------|-------------|
| `browser_list_sessions` | List active browser sessions |
| `browser_close_session` | Close a specific session |
| `browser_close_all` | Close all browser sessions |

### MCP Resources

| URI | Type | Description |
|-----|------|-------------|
| `browser://current-page/content` | text/markdown | Current page as markdown |
| `browser://current-page/state` | application/json | Interactive elements and metadata |
| `browser://current-page/accessibility` | application/json | Accessibility tree |
| `browser://sessions/{id}/content` | text/markdown | Specific session page content |
| `browser://sessions/{id}/state` | application/json | Specific session state |
| `browser://sessions/{id}/accessibility` | application/json | Specific session a11y tree |

### Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `OPENBROWSER_HEADLESS` | Run browser without GUI | `false` |
| `OPENBROWSER_ALLOWED_DOMAINS` | Comma-separated domain whitelist | (none) |

## CLI Usage

```bash
# Run a browser automation task
uvx openbrowser run "Search for Python tutorials on Google"

# Install browser
uvx openbrowser install

# Run MCP server
uvx openbrowser mcp
```

## Project Structure

```
openbrowser-ai/
├── .claude-plugin/            # Claude Code marketplace config
├── .codex/                    # Codex integration
│   └── INSTALL.md
├── .opencode/                 # OpenCode integration
│   ├── INSTALL.md
│   └── plugins/openbrowser.js
├── plugin/                    # Plugin package (skills + MCP config)
│   ├── .claude-plugin/
│   ├── .mcp.json
│   └── skills/                # 5 browser automation skills
├── src/openbrowser/
│   ├── __init__.py            # Main exports
│   ├── cli.py                 # CLI commands
│   ├── config.py              # Configuration
│   ├── actor/                 # Element interaction
│   ├── agent/                 # LangGraph agent
│   ├── browser/               # CDP browser control
│   ├── code_use/              # Code agent
│   ├── dom/                   # DOM extraction
│   ├── llm/                   # LLM providers
│   ├── mcp/                   # MCP server
│   └── tools/                 # Action registry
└── tests/                     # Test suite
```

## Testing

```bash
# Run tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Email**: billy.suharno@gmail.com
- **GitHub**: [@billy-enrizky](https://github.com/billy-enrizky)
- **Repository**: [github.com/billy-enrizky/openbrowser-ai](https://github.com/billy-enrizky/openbrowser-ai)
- **Documentation**: [https://docs.openbrowser.me](https://docs.openbrowser.me)

---

**Made with love for the AI automation community**
