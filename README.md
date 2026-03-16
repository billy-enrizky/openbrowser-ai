# OpenBrowser

**Automating Walmart Product Scraping:**

https://github.com/user-attachments/assets/c517c739-9199-47b0-bac7-c2c642a21094

**OpenBrowserAI Automatic Flight Booking:**

https://github.com/user-attachments/assets/632128f6-3d09-497f-9e7d-e29b9cb65e0f


[![PyPI version](https://badge.fury.io/py/openbrowser-ai.svg)](https://pypi.org/project/openbrowser-ai/)
[![Downloads](https://img.shields.io/pypi/dm/openbrowser-ai?color=brightgreen&label=downloads)](https://pepy.tech/projects/openbrowser-ai)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/billy-enrizky/openbrowser-ai/actions/workflows/test.yml/badge.svg)](https://github.com/billy-enrizky/openbrowser-ai/actions)
[![Coverage](https://codecov.io/gh/billy-enrizky/openbrowser-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/billy-enrizky/openbrowser-ai)

<!-- mcp-name: me.openbrowser/openbrowser-ai -->

**AI-powered browser automation using CodeAgent and CDP (Chrome DevTools Protocol)**

OpenBrowser is a framework for intelligent browser automation. It combines direct CDP communication with a CodeAgent architecture, where the LLM writes Python code executed in a persistent namespace, to navigate, interact with, and extract information from web pages autonomously.

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
- [Benchmark: Token Efficiency](#benchmark-token-efficiency)
- [CLI Usage](#cli-usage)
- [Project Structure](#project-structure)
- [Backend and Frontend Deployment](#backend-and-frontend-deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Documentation

**Full documentation**: [https://docs.openbrowser.me](https://docs.openbrowser.me)

## Key Features

- **CodeAgent Architecture** - LLM writes Python code in a persistent Jupyter-like namespace for browser automation
- **Raw CDP Communication** - Direct Chrome DevTools Protocol for maximum control and speed
- **Vision Support** - Screenshot analysis for visual understanding of pages
- **12+ LLM Providers** - OpenAI, Anthropic, Google, Groq, AWS Bedrock, Azure OpenAI, Ollama, and more
- **MCP Server** - Model Context Protocol support for Claude Desktop integration
- **CLI Daemon** - Persistent browser daemon with `-c` flag for direct code execution from Bash
- **Video Recording** - Record browser sessions as video files

## Installation

### Quick install (macOS / Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.sh | sh
```

### Quick install (Windows PowerShell)

```powershell
irm https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.ps1 | iex
```

Detects `uv`, `pipx`, or `pip` and installs OpenBrowser automatically.

Install to `~/.local/bin` without sudo:

```bash
curl -fsSL https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.sh | sh -s -- --local
```

### Homebrew (macOS / Linux)

```bash
brew tap billy-enrizky/openbrowser
brew install openbrowser-ai
```

### pip

```bash
pip install openbrowser-ai
```

### uv (recommended)

```bash
uv pip install openbrowser-ai
```

### uvx (zero install)

Run directly without installing -- `uvx` downloads and caches the package automatically:

```bash
# MCP server mode
uvx openbrowser-ai --mcp

# CLI daemon mode
uvx openbrowser-ai -c "await navigate('https://example.com')"
```

### pipx

```bash
pipx install openbrowser-ai
```

### From source

```bash
git clone https://github.com/billy-enrizky/openbrowser-ai.git
cd openbrowser-ai
uv pip install -e ".[agent]"
```

### Optional Dependencies

```bash
pip install openbrowser-ai[agent]      # LLM agent support (langgraph, langchain, litellm)
pip install openbrowser-ai[all]        # All LLM providers
pip install openbrowser-ai[anthropic]  # Anthropic Claude
pip install openbrowser-ai[groq]       # Groq
pip install openbrowser-ai[ollama]     # Ollama (local models)
pip install openbrowser-ai[aws]        # AWS Bedrock
pip install openbrowser-ai[azure]      # Azure OpenAI
pip install openbrowser-ai[video]      # Video recording support
```

> **No separate browser install needed.** OpenBrowser auto-detects any installed Chromium-based browser (Chrome, Edge, Brave, Chromium) and uses it directly. If none is found and `uvx` is available, Chromium is installed automatically on first run. To pre-install manually (requires `uvx`): `openbrowser-ai install`

## Quick Start

### Basic Usage

```python
import asyncio
from openbrowser import CodeAgent, ChatGoogle

async def main():
    agent = CodeAgent(
        task="Go to google.com and search for 'Python tutorials'",
        llm=ChatGoogle(model="gemini-3-flash"),
    )

    result = await agent.run()
    print(f"Result: {result}")

asyncio.run(main())
```

### With Different LLM Providers

```python
from openbrowser import CodeAgent, ChatOpenAI, ChatAnthropic, ChatGoogle

# OpenAI
agent = CodeAgent(task="...", llm=ChatOpenAI(model="gpt-5.2"))

# Anthropic
agent = CodeAgent(task="...", llm=ChatAnthropic(model="claude-sonnet-4-6"))

# Google Gemini
agent = CodeAgent(task="...", llm=ChatGoogle(model="gemini-3-flash"))
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
| **Google** | `ChatGoogle` | gemini-3-flash, gemini-3-pro |
| **OpenAI** | `ChatOpenAI` | gpt-5.2, o4-mini, o3 |
| **Anthropic** | `ChatAnthropic` | claude-sonnet-4-6, claude-opus-4-6 |
| **Groq** | `ChatGroq` | llama-4-scout, qwen3-32b |
| **AWS Bedrock** | `ChatAWSBedrock` | anthropic.claude-sonnet-4-6, amazon.nova-pro |
| **AWS Bedrock (Anthropic)** | `ChatAnthropicBedrock` | Claude models via Anthropic Bedrock SDK |
| **Azure OpenAI** | `ChatAzureOpenAI` | Any Azure-deployed model |
| **OpenRouter** | `ChatOpenRouter` | Any model on openrouter.ai |
| **DeepSeek** | `ChatDeepSeek` | deepseek-chat, deepseek-r1 |
| **Cerebras** | `ChatCerebras` | llama-4-scout, qwen-3-235b |
| **Ollama** | `ChatOllama` | llama-4-scout, deepseek-r1 (local) |
| **OCI** | `ChatOCIRaw` | Oracle Cloud GenAI models |
| **Browser-Use** | `ChatBrowserUse` | External LLM service |

## Claude Code Plugin

Install OpenBrowser as a Claude Code plugin:

```bash
# Add the marketplace (one-time)
claude plugin marketplace add billy-enrizky/openbrowser-ai

# Install the plugin
claude plugin install openbrowser@openbrowser-ai
```

This installs the MCP server and 6 built-in skills:

| Skill | Description |
|-------|-------------|
| `web-scraping` | Extract structured data, handle pagination |
| `form-filling` | Fill forms, login flows, multi-step wizards |
| `e2e-testing` | Test web apps by simulating user interactions |
| `page-analysis` | Analyze page content, structure, metadata |
| `accessibility-audit` | Audit pages for WCAG compliance |
| `file-download` | Download files (PDFs, CSVs) using browser session |

See [plugin/README.md](plugin/README.md) for detailed tool parameter documentation.

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

[OpenClaw](https://openclaw.ai) supports OpenBrowser via the CLI daemon. Install OpenBrowser,
then use `openbrowser-ai -c` from the Bash tool:

```bash
openbrowser-ai -c "await navigate('https://example.com')"
openbrowser-ai -c "print(await evaluate('document.title'))"
```

The daemon starts automatically on first use and persists variables across calls.

For OpenClaw plugin documentation, see [docs.openclaw.ai/tools/plugin](https://docs.openclaw.ai/tools/plugin).

## MCP Server

[![MCP Registry](https://img.shields.io/badge/MCP-Registry-blue)](https://registry.modelcontextprotocol.io/?q=openbrowser)

OpenBrowser includes an MCP (Model Context Protocol) server that exposes browser automation as tools for AI assistants like Claude. Listed on the [MCP Registry](https://registry.modelcontextprotocol.io/?q=openbrowser) as `me.openbrowser/openbrowser-ai`. No external LLM API keys required -- the MCP client provides the intelligence.

### Quick Setup

**Claude Code**: add to your project's `.mcp.json`:

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

**Claude Desktop**: add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

**Run directly:**

```bash
uvx openbrowser-ai --mcp
```

### Tool

The MCP server exposes a single `execute_code` tool that runs Python code in a persistent namespace with browser automation functions. The LLM writes Python code to navigate, interact, and extract data, returning only what was explicitly requested.

**Available functions** (all async, use `await`):

| Category | Functions |
|----------|-----------|
| **Navigation** | `navigate(url, new_tab)`, `go_back()`, `wait(seconds)` |
| **Interaction** | `click(index)`, `input_text(index, text, clear)`, `scroll(down, pages, index)`, `send_keys(keys)`, `upload_file(index, path)` |
| **Dropdowns** | `select_dropdown(index, text)`, `dropdown_options(index)` |
| **Tabs** | `switch(tab_id)`, `close(tab_id)` |
| **JavaScript** | `evaluate(code)`: run JS in page context, returns Python objects |
| **Downloads** | `download_file(url, filename)`: download a file using browser cookies, `list_downloads()`: list downloaded files |
| **State** | `browser.get_browser_state_summary()`: get page metadata and interactive elements |
| **CSS** | `get_selector_from_index(index)`: get CSS selector for an element |
| **Completion** | `done(text, success)`: signal task completion |

**Pre-imported libraries**: `json`, `csv`, `re`, `datetime`, `asyncio`, `Path`, `requests`, `numpy`, `pandas`, `matplotlib`, `BeautifulSoup`

### Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `OPENBROWSER_HEADLESS` | Run browser without GUI | `false` |
| `OPENBROWSER_ALLOWED_DOMAINS` | Comma-separated domain whitelist | (none) |
| `OPENBROWSER_COMPACT_DESCRIPTION` | Minimal tool description (~500 tokens) | `false` |
| `OPENBROWSER_MAX_OUTPUT` | Max output characters per execution | `10000` |

## Benchmark: Token Efficiency

### CLI Benchmark: 4-Way Comparison (6 Tasks, N=3 runs)

Four CLI tools compared with a single Bash tool each. Claude Sonnet 4.6 on Bedrock. Randomized order. All achieve 100% accuracy.

<p align="center">
  <img src="benchmarks/cli_benchmark_scatter.png" alt="CLI Benchmark: Token Usage vs Duration" width="800" />
</p>

| CLI Tool | Duration (mean +/- std) | Tool Calls | Bedrock API Tokens | Response Chars |
|----------|------------------------:|-----------:|-------------------:|---------------:|
| **openbrowser-ai** | **84.8 +/- 10.9s** | **15.3 +/- 2.3** | **36,010 +/- 6,063** | **9,452 +/- 472** |
| browser-use | 106.0 +/- 9.5s | 20.7 +/- 6.4 | 77,123 +/- 33,354 | 36,241 +/- 12,940 |
| agent-browser | 99.0 +/- 6.8s | 25.0 +/- 4.0 | 90,107 +/- 3,698 | 56,009 +/- 39,733 |
| playwright-cli | 118.3 +/- 21.4s | 25.7 +/- 8.1 | 94,130 +/- 35,982 | 84,065 +/- 49,713 |

openbrowser-ai uses **2.1-2.6x fewer tokens** than all competitors via Python code batching and compact DOM representation.

<p align="center">
  <img src="benchmarks/cli_benchmark_overview.png" alt="CLI Benchmark: Overview" width="800" />
</p>

#### Per-Task Token Usage

<p align="center">
  <img src="benchmarks/cli_benchmark_per_task.png" alt="CLI Benchmark: Per-Task Token Usage" width="800" />
</p>

| Task | openbrowser-ai | browser-use | playwright-cli | agent-browser |
|------|---------------:|------------:|---------------:|--------------:|
| fact_lookup | **2,504** | 4,710 | 16,857 | 9,676 |
| form_fill | **7,887** | 15,811 | 31,757 | 19,226 |
| multi_page_extract | **2,354** | 2,405 | 8,886 | 8,117 |
| search_navigate | **16,539** | 47,936 | 27,779 | 44,367 |
| deep_navigation | **2,178** | 3,747 | 4,705 | 5,534 |
| content_analysis | 4,548 | **2,515** | 4,147 | 3,189 |

openbrowser-ai wins 5 of 6 tasks. The advantage is largest on complex pages (search_navigate: 2.9x fewer tokens than browser-use) where code batching avoids repeated page state dumps.

#### Cost per Benchmark Run (6 Tasks)

| Model | openbrowser-ai | browser-use | playwright-cli | agent-browser |
|-------|---------------:|------------:|---------------:|--------------:|
| Claude Sonnet 4.6 ($3/$15 per M) | **$0.12** | $0.24 | $0.29 | $0.27 |
| Claude Opus 4.6 ($5/$25 per M) | **$0.24** | $0.45 | $0.56 | $0.51 |

Raw results are in [`benchmarks/e2e_4way_cli_results.json`](benchmarks/e2e_4way_cli_results.json). [Full 4-way comparison with methodology](https://docs.openbrowser.me/cli-comparison).

### E2E LLM Benchmark: MCP Server Comparison (6 Tasks, N=5 runs)

<p align="center">
  <img src="benchmarks/benchmark_comparison.png" alt="E2E LLM Benchmark: MCP Server Comparison" width="800" />
</p>

| MCP Server | Pass Rate | Duration (mean +/- std) | Tool Calls | Bedrock API Tokens |
|------------|:---------:|------------------------:|-----------:|-------------------:|
| **Playwright MCP** (Microsoft) | 100% | 62.7 +/- 4.8s | 9.4 +/- 0.9 | 158,787 |
| **Chrome DevTools MCP** (Google) | 100% | 103.4 +/- 2.7s | 19.4 +/- 0.5 | 299,486 |
| **OpenBrowser MCP** | 100% | 77.0 +/- 6.7s | 13.8 +/- 2.0 | **50,195** |

OpenBrowser uses **3.2x fewer tokens** than Playwright and **6.0x fewer** than Chrome DevTools. MCP response sizes: Playwright 1,132,173 chars, Chrome DevTools 1,147,244 chars, OpenBrowser 7,853 chars -- a **144x difference**.

[Full MCP comparison with methodology](https://docs.openbrowser.me/comparison)

## CLI Usage

```bash
# Run a browser automation task with an LLM agent
uvx openbrowser-ai -p "Search for Python tutorials on Google"

# Execute code directly via persistent daemon
uvx openbrowser-ai -c "await navigate('https://example.com')"
uvx openbrowser-ai -c "print(await evaluate('document.title'))"

# Daemon management
uvx openbrowser-ai daemon start     # Start daemon (auto-starts on first -c call)
uvx openbrowser-ai daemon stop      # Stop daemon and browser
uvx openbrowser-ai daemon status    # Show daemon info
uvx openbrowser-ai daemon restart   # Restart daemon

# Install browser
uvx openbrowser-ai install

# Run MCP server
uvx openbrowser-ai --mcp
```

The `-c` flag connects to a persistent browser daemon over a Unix socket (localhost TCP on Windows). Variables persist across calls while the daemon is running. The daemon starts automatically on first use and shuts down after 10 minutes of inactivity.

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
│   └── skills/                # 6 browser automation skills
├── src/openbrowser/
│   ├── __init__.py            # Main exports
│   ├── cli.py                 # CLI commands
│   ├── config.py              # Configuration
│   ├── actor/                 # Element interaction
│   ├── agent/                 # LangGraph agent
│   ├── browser/               # CDP browser control
│   ├── code_use/              # Code agent + shared executor
│   ├── daemon/                # Persistent browser daemon (Unix socket)
│   ├── dom/                   # DOM extraction
│   ├── llm/                   # LLM providers
│   ├── mcp/                   # MCP server
│   └── tools/                 # Action registry
├── benchmarks/                # MCP benchmarks and E2E tests
│   ├── playwright_benchmark.py
│   ├── cdp_benchmark.py
│   ├── openbrowser_benchmark.py
│   └── e2e_published_test.py
└── tests/                     # Test suite
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# E2E test the MCP server against the published PyPI package
uv run python benchmarks/e2e_published_test.py
```

### Benchmarks

Run individual MCP server benchmarks (JSON-RPC stdio, 5-step Wikipedia workflow):

```bash
uv run python benchmarks/openbrowser_benchmark.py   # OpenBrowser MCP
uv run python benchmarks/playwright_benchmark.py     # Playwright MCP
uv run python benchmarks/cdp_benchmark.py            # Chrome DevTools MCP
```

Raw results are in [`benchmarks/e2e_4way_cli_results.json`](benchmarks/e2e_4way_cli_results.json). See [full comparison](https://docs.openbrowser.me/comparison) for methodology.

## Backend and Frontend Deployment

The project includes a FastAPI backend and a Next.js frontend, both containerized with Docker.

### Prerequisites

- Docker and Docker Compose
- A `.env` file in the project root with `POSTGRES_PASSWORD` and any LLM API keys (see `backend/env.example`)

### Local Development (Docker Compose)

```bash
# Start backend + PostgreSQL (frontend runs locally)
docker-compose -f docker-compose.dev.yml up --build

# In a separate terminal, start the frontend
cd frontend && npm install && npm run dev
```

| Service | URL | Description |
|---------|-----|-------------|
| Backend | http://localhost:8000 | FastAPI + WebSocket + VNC |
| Frontend | http://localhost:3000 | Next.js dev server |
| PostgreSQL | localhost:5432 | Chat persistence |
| VNC | ws://localhost:6080 | Live browser view |

The dev compose mounts `backend/app/` and `src/` as volumes for hot-reload. API keys are loaded from `backend/.env` via `env_file`. The `POSTGRES_PASSWORD` is read from the root `.env` file.

### Full Stack (Docker Compose)

```bash
# Start all services (backend + frontend + PostgreSQL)
docker-compose up --build
```

This builds and runs both the backend and frontend containers together with PostgreSQL.

### Backend

The backend is a FastAPI application in `backend/` with a Dockerfile at `backend/Dockerfile`. It includes:

- REST API on port 8000
- WebSocket endpoint at `/ws` for real-time agent communication
- VNC support (Xvfb + x11vnc + websockify) for live browser viewing on ports 6080-6090
- Kiosk security: Openbox window manager, Chromium enterprise policies, X11 key grabber daemon
- Health check at `/health`

```bash
# Build the backend image
docker build -f backend/Dockerfile -t openbrowser-backend .

# Run standalone
docker run -p 8000:8000 -p 6080:6080 \
  --env-file backend/.env \
  -e VNC_ENABLED=true \
  -e AUTH_ENABLED=false \
  --shm-size=2g \
  openbrowser-backend
```

### Frontend

The frontend is a Next.js application in `frontend/` with a Dockerfile at `frontend/Dockerfile`.

```bash
# Build the frontend image
cd frontend && docker build -t openbrowser-frontend .

# Run standalone
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  -e NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws \
  openbrowser-frontend
```

### Environment Variables

Key environment variables for the backend (see `backend/env.example` for the full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google/Gemini API key | (required) |
| `DEFAULT_LLM_MODEL` | Default model for agents | `gemini-3-flash-preview` |
| `AUTH_ENABLED` | Enable Cognito JWT auth | `false` |
| `VNC_ENABLED` | Enable VNC browser viewing | `true` |
| `DATABASE_URL` | PostgreSQL connection string | (optional) |
| `POSTGRES_PASSWORD` | PostgreSQL password (root `.env`) | (required for compose) |

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
