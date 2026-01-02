# openbrowser-ai

[![PyPI version](https://badge.fury.io/py/openbrowser-ai.svg)](https://pypi.org/project/openbrowser-ai/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/billy-enrizky/openbrowser-ai/actions/workflows/test.yml/badge.svg)](https://github.com/billy-enrizky/openbrowser-ai/actions)

**Agentic browser automation using LangGraph and raw CDP (Chrome DevTools Protocol)**

openbrowser-ai is a powerful framework for AI-driven browser automation. It combines the flexibility of direct CDP communication with the orchestration power of LangGraph to create intelligent agents that can navigate, interact with, and extract information from web pages autonomously.

## Key Features

- **LangGraph-Powered Agents** - Stateful workflow orchestration with perceive-step-execute loop
- **Raw CDP Communication** - Direct Chrome DevTools Protocol for maximum control and speed
- **Vision Support** - Screenshot analysis for visual understanding of pages
- **12+ LLM Providers** - OpenAI, Anthropic, Google, Groq, AWS Bedrock, Azure OpenAI, Ollama, and more
- **Code Agent Mode** - Jupyter notebook-like code execution for complex automation
- **MCP Server** - Model Context Protocol support for Claude Desktop integration
- **Video Recording** - Record browser sessions as video files
- **GIF Export** - Export execution history as animated GIFs
- **CAPTCHA Detection** - Automatic detection and alternative routing
- **Conversation Persistence** - Save and resume agent sessions

## Installation

### Basic Installation

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

# Install with CLI interface
pip install openbrowser-ai[cli]

# Install with telemetry
pip install openbrowser-ai[telemetry]
```

### Install Playwright Browsers

```bash
playwright install chromium
```

## Quick Start

### Basic Usage

```python
import asyncio
from openbrowser import BrowserAgent

async def main():
    agent = BrowserAgent(
        task="Go to google.com and search for 'Python tutorials'",
        llm_provider="openai",
        model_name="gpt-4o",
        headless=False,  # Set to True for headless mode
    )
    
    history = await agent.run()
    
    print(f"Task completed: {history.is_successful()}")
    print(f"Final result: {history.final_result()}")

asyncio.run(main())
```

### With Custom LLM

```python
import asyncio
from openbrowser import BrowserAgent, ChatOpenAI

async def main():
    # Create a custom LLM instance
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    agent = BrowserAgent(
        task="Navigate to example.com and extract the main heading",
        llm=llm,
    )
    
    history = await agent.run()
    print(history.final_result())

asyncio.run(main())
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

Set your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."

# Groq
export GROQ_API_KEY="gsk_..."

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-west-2"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# Cerebras
export CEREBRAS_API_KEY="..."
```

### BrowserProfile Options

```python
from openbrowser import BrowserProfile

profile = BrowserProfile(
    # Display settings
    headless=True,
    viewport_width=1280,
    viewport_height=720,
    
    # Browser settings
    disable_security=False,
    extra_chromium_args=["--disable-gpu"],
    
    # Recording
    record_video_dir="./recordings",  # Enable video recording
    
    # Proxy settings
    proxy={
        "server": "http://proxy.example.com:8080",
        "username": "user",
        "password": "pass",
    },
)
```

### AgentSettings Options

```python
from openbrowser import AgentSettings

settings = AgentSettings(
    use_vision=True,           # Enable screenshot analysis
    max_actions_per_step=4,    # Max actions per LLM call
    max_failures=3,            # Max consecutive failures
)
```

## Supported LLM Providers

| Provider | Class | Models |
|----------|-------|--------|
| **OpenAI** | `ChatOpenAI` | gpt-4o, gpt-4-turbo, gpt-3.5-turbo |
| **Anthropic** | `ChatAnthropic` | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| **Google** | `ChatGoogle` | gemini-2.0-flash-exp, gemini-1.5-pro |
| **Groq** | `ChatGroq` | llama-3.3-70b-versatile, mixtral-8x7b |
| **AWS Bedrock** | `ChatAWSBedrock` | claude-3, amazon.titan |
| **Azure OpenAI** | `ChatAzureOpenAI` | Any Azure-deployed model |
| **Ollama** | `ChatOllama` | llama3, mistral, codellama (local) |
| **OpenRouter** | `ChatOpenRouter` | Multi-provider gateway |
| **OCI** | `ChatOCI` | Oracle Cloud GenAI models |
| **Cerebras** | `ChatCerebras` | llama-3.3-70b |
| **DeepSeek** | `ChatDeepSeek` | deepseek-chat, deepseek-coder |
| **BrowserUse** | `ChatBrowserUse` | Cloud-hosted endpoint |

### Provider Examples

```python
from openbrowser import (
    ChatOpenAI,
    ChatAnthropic,
    ChatGoogle,
    ChatGroq,
    ChatAWSBedrock,
    ChatAzureOpenAI,
    ChatOllama,
)

# OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Anthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

# Google Gemini
llm = ChatGoogle(model="gemini-2.0-flash-exp", temperature=0)

# Groq (fast inference)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# AWS Bedrock
llm = ChatAWSBedrock(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region="us-west-2",
)

# Azure OpenAI
llm = ChatAzureOpenAI(
    model="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-15-preview",
)

# Local Ollama
llm = ChatOllama(model="llama3", temperature=0)
```

## Code Agent Mode

For complex automation tasks, use the Code Agent which executes Python code in a Jupyter-like environment:

```python
import asyncio
from openbrowser import BrowserSession
from openbrowser.code_use import CodeAgent, export_to_ipynb

async def main():
    session = BrowserSession()
    await session.start()
    
    agent = CodeAgent(
        task="Scrape the top 10 trending repositories from GitHub",
        llm=my_llm,
        browser=session,
    )
    
    result = await agent.run()
    
    # Export session to Jupyter notebook
    export_to_ipynb(result.session, "github_scraper.ipynb")
    
    await session.stop()

asyncio.run(main())
```

## MCP Server (Claude Desktop Integration)

openbrowser-ai includes an MCP (Model Context Protocol) server for integration with Claude Desktop and other MCP clients.

### Running the MCP Server

```bash
python -m openbrowser.mcp
```

### Claude Desktop Configuration

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "openbrowser-ai": {
      "command": "python",
      "args": ["-m", "openbrowser.mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Available MCP Tools

- `run_browser_agent` - Run an autonomous browser task
- `browser_navigate` - Navigate to a URL
- `browser_click` - Click an element
- `browser_type` - Type text into an element
- `browser_scroll` - Scroll the page
- `browser_screenshot` - Take a screenshot
- `browser_get_elements` - Get interactive elements
- `browser_extract_content` - Extract page content

## CLI Usage

```bash
# Run a browser automation task
openbrowser-ai run "Search for Python tutorials on Google" --provider openai --model gpt-4o

# Run in headless mode
openbrowser-ai run "Get the weather in New York" --headless

# Save execution as GIF
openbrowser-ai run "Navigate to example.com" --save-gif ./recording.gif

# List available models
openbrowser-ai models

# Initialize configuration
openbrowser-ai init
```

## Video Recording

Enable video recording of browser sessions:

```python
from openbrowser import BrowserProfile, BrowserSession

profile = BrowserProfile(
    headless=False,
    record_video_dir="./recordings",
)

session = BrowserSession(browser_profile=profile)
await session.start()

# Perform automation...

await session.stop()  # Video saved to ./recordings/
```

## GIF Export

Export agent execution history as an animated GIF:

```python
from openbrowser import BrowserAgent
from openbrowser.agent import create_history_gif

agent = BrowserAgent(task="...", llm_provider="openai")
history = await agent.run()

# Create GIF from execution history
create_history_gif(history, output_path="execution.gif")
```

## Available Actions

The agent can perform these browser actions:

| Action | Description |
|--------|-------------|
| `navigate` | Navigate to a URL |
| `click` | Click an element by index |
| `type` | Type text into an input field |
| `scroll` | Scroll the page up/down |
| `send_keys` | Send keyboard keys (Enter, Tab, Escape, etc.) |
| `wait` | Wait for a specified duration |
| `go_back` | Navigate back in history |
| `go_forward` | Navigate forward in history |
| `refresh` | Refresh the current page |
| `switch_tab` | Switch to a different tab |
| `new_tab` | Open a new tab |
| `close_tab` | Close the current tab |
| `extract_content` | Extract page content |
| `done` | Mark task as complete |

## Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/openbrowser

# Run specific test file
pytest tests/test_agent_views.py
```

## Project Structure

```
openbrowser-ai/
├── src/openbrowser/
│   ├── __init__.py          # Main exports
│   ├── cli.py                # CLI commands
│   ├── config.py             # Configuration handling
│   ├── actor/                # Element interaction (click, type, scroll)
│   ├── agent/                # LangGraph agent workflow
│   │   ├── graph.py          # BrowserAgent implementation
│   │   ├── prompts.py        # System prompts
│   │   ├── views.py          # Agent data models
│   │   └── message_manager/  # Conversation management
│   ├── browser/              # CDP browser control
│   │   ├── session.py        # BrowserSession
│   │   ├── profile.py        # BrowserProfile
│   │   ├── dom/              # DOM extraction & serialization
│   │   └── watchdogs/        # Browser event handlers
│   ├── code_use/             # Code agent (Jupyter-like)
│   ├── filesystem/           # File operations
│   ├── integrations/         # Third-party integrations (Gmail)
│   ├── llm/                  # LLM provider implementations
│   │   ├── openai/
│   │   ├── anthropic/
│   │   ├── google/
│   │   ├── groq/
│   │   ├── aws/
│   │   ├── azure/
│   │   └── ...
│   ├── mcp/                  # MCP server
│   ├── screenshots/          # Screenshot service
│   ├── telemetry/            # Usage telemetry
│   ├── tokens/               # Token cost tracking
│   └── tools/                # Action registry
└── tests/                    # Test suite
```

## Security Considerations

- **Sandboxed Execution**: File operations are sandboxed to the workspace directory
- **Proxy Support**: Full proxy configuration for network isolation
- **CAPTCHA Detection**: Automatic detection with alternative routing
- **Credential Handling**: Use environment variables, never hardcode API keys
- **Browser Security**: Optional security flags for testing environments

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the repository.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [browser-use](https://github.com/browser-use/browser-use)
- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Uses [Playwright](https://playwright.dev/) for browser orchestration

## Contact

- **Email**: billy.suharno@gmail.com
- **GitHub**: [@billy-enrizky](https://github.com/billy-enrizky)
- **Repository**: [github.com/billy-enrizky/openbrowser-ai](https://github.com/billy-enrizky/openbrowser-ai)

---

**Made with love for the AI automation community**