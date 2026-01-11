# OpenBrowser

Automating Walmart Product Scraping:
https://github.com/user-attachments/assets/ae5d74ce-0ac6-46b0-b02b-ff5518b4b20d


OpenBrowserAI Automatic Flight Booking:
https://github.com/user-attachments/assets/632128f6-3d09-497f-9e7d-e29b9cb65e0f


[![PyPI version](https://badge.fury.io/py/openbrowser-ai.svg)](https://pypi.org/project/openbrowser-ai/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/billy-enrizky/openbrowser-ai/actions/workflows/test.yml/badge.svg)](https://github.com/billy-enrizky/openbrowser-ai/actions)

**AI-powered browser automation using LangGraph and CDP (Chrome DevTools Protocol)**

OpenBrowser is a framework for intelligent browser automation. It combines direct CDP communication with LangGraph orchestration to create AI agents that can navigate, interact with, and extract information from web pages autonomously.

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

## MCP Server (Claude Desktop Integration)

OpenBrowser includes an MCP server for integration with Claude Desktop.

### Running the MCP Server

```bash
python -m openbrowser.mcp
```

### Claude Desktop Configuration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "openbrowser": {
      "command": "uvx",
      "args": ["openbrowser-ai", "mcp"],
      "env": {
        "GOOGLE_API_KEY": "..."
      }
    }
  }
}
```

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
├── src/openbrowser/
│   ├── __init__.py          # Main exports
│   ├── cli.py                # CLI commands
│   ├── config.py             # Configuration
│   ├── actor/                # Element interaction
│   ├── agent/                # LangGraph agent
│   │   ├── graph.py          # Agent workflow
│   │   ├── service.py        # Agent class
│   │   └── views.py          # Data models
│   ├── browser/              # CDP browser control
│   │   ├── session.py        # BrowserSession
│   │   └── profile.py        # BrowserProfile
│   ├── code_use/             # Code agent
│   ├── dom/                  # DOM extraction
│   ├── llm/                  # LLM providers
│   │   ├── openai/
│   │   ├── anthropic/
│   │   ├── google/
│   │   ├── groq/
│   │   ├── aws/
│   │   ├── azure/
│   │   └── ...
│   ├── mcp/                  # MCP server
│   └── tools/                # Action registry
└── tests/                    # Test suite
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
