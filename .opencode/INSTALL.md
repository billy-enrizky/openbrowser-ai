# Installing OpenBrowser for OpenCode

<!-- toc:start -->
## Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Quick install with skills.sh](#2-quick-install-with-skillssh)
- [3. Installation](#3-installation)
  - [3.1. macOS / Linux](#31-macos-linux)
  - [3.2. Windows (PowerShell)](#32-windows-powershell)
- [4. Verify](#4-verify)
- [5. Configure the MCP server](#5-configure-the-mcp-server)
- [6. Available Skills](#6-available-skills)
- [7. Install OpenBrowser CLI](#7-install-openbrowser-cli)
- [8. Usage](#8-usage)
  - [8.1. Tool Mapping](#81-tool-mapping)
- [9. Updating](#9-updating)
- [10. Uninstalling](#10-uninstalling)

<!-- toc:end -->

## 1. Prerequisites

- [OpenCode.ai](https://opencode.ai) installed
- Git installed
- Chrome or Chromium installed, or the ability to install Chromium with `openbrowser-ai install`
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)

MCP and `openbrowser-ai -c` need no LLM key.

## 2. Quick install with skills.sh

```bash
npx skills add billy-enrizky/openbrowser-ai --skill setup -a opencode
npx skills add billy-enrizky/openbrowser-ai -a opencode
```

The first command installs only the setup skill. The second installs all eight OpenBrowser skills. The `setup` skill installs the CLI and Chromium, configures MCP, and verifies a real browser session.

These commands install skill files. They do not run the setup workflow. After installation, ask OpenCode to run the `setup` skill.

## 3. Installation

### 3.1. macOS / Linux

```bash
# 1. Clone the repository
git clone https://github.com/billy-enrizky/openbrowser-ai.git ~/.config/opencode/openbrowser

# 2. Create directories
mkdir -p ~/.config/opencode/plugins ~/.config/opencode/skills

# 3. Create plugin symlink
ln -s ~/.config/opencode/openbrowser/.opencode/plugins/openbrowser.js ~/.config/opencode/plugins/openbrowser.js

# 4. Create skills symlink
ln -s ~/.config/opencode/openbrowser/plugin/skills ~/.config/opencode/skills/openbrowser

# 5. Restart OpenCode
```

### 3.2. Windows (PowerShell)

Run as Administrator or with Developer Mode enabled:

```powershell
# 1. Clone the repository
git clone https://github.com/billy-enrizky/openbrowser-ai.git "$env:USERPROFILE\.config\opencode\openbrowser"

# 2. Create directories
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.config\opencode\plugins"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.config\opencode\skills"

# 3. Create plugin symlink
New-Item -ItemType SymbolicLink -Path "$env:USERPROFILE\.config\opencode\plugins\openbrowser.js" -Target "$env:USERPROFILE\.config\opencode\openbrowser\.opencode\plugins\openbrowser.js"

# 4. Create skills junction
cmd /c mklink /J "$env:USERPROFILE\.config\opencode\skills\openbrowser" "$env:USERPROFILE\.config\opencode\openbrowser\plugin\skills"

# 5. Restart OpenCode
```

## 4. Verify

Verify the CLI entries:

```bash
openbrowser-ai --version
openbrowser-ai-py --version
```

The two commands must report the package version. If `openbrowser-ai-py --version` fails, reinstall OpenBrowser.

Verify the plugin and skills paths:

```bash
ls -l ~/.config/opencode/plugins/openbrowser.js
ls -l ~/.config/opencode/skills/openbrowser
```

Both paths should resolve to the OpenBrowser plugin and skill directories. On Windows, the skills path is a junction.

## 5. Configure the MCP server

Add this entry to `opencode.json` in your project:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "openbrowser": {
      "type": "local",
      "command": ["openbrowser-ai", "--mcp"],
      "enabled": true
    }
  }
}
```

If OpenCode cannot find `openbrowser-ai`, set `command` to `["uvx", "openbrowser-ai", "--mcp"]`.

The default OpenBrowser stdio transport is a lean JSON-RPC loop. It answers discovery and the tool catalog before it imports the MCP SDK. MCP runs through the Python CLI. On a native wheel, `openbrowser-ai --mcp` delegates to `openbrowser-ai-py --mcp`. Set `OPENBROWSER_MCP_TRANSPORT=sdk` in the server environment to use the MCP SDK 2.3 escape hatch. Invalid values exit with code 2. Modern `2026-07-28` clients see seven tools: `create_context`, `execute_code`, `get_status`, `reset_browser`, `close_context`, `web_search`, and `scrape_web`. The legacy handshakes `2024-11-05`, `2025-03-26`, `2025-06-18`, and `2025-11-25` see four tools: `execute_code`, `get_status`, `reset_browser`, and `web_search`. Call `create_context` first, pass its returned `context_id` to stateful tools, and call `close_context` when finished. `scrape_web` is stateless and fetches one HTTP(S) page in OpenBrowser no-browser mode: it starts no Chrome, Blink, or V8 process and stays on the HTTP path. JavaScript, screenshots, and browser cookies are unavailable on this path.

## 6. Available Skills

| Skill | Description |
|-------|-------------|
| `setup` | Install the CLI and Chromium, configure MCP and profiles, and verify a real browser session |
| `deep-research` | Conduct cited, multi-source web research and produce markdown and JSON reports |
| `web-scraping` | Extract structured data from websites, handle pagination |
| `form-filling` | Fill forms, login flows, multi-step wizards |
| `e2e-testing` | Test web apps by simulating user interactions |
| `page-analysis` | Analyze page content, structure, metadata |
| `accessibility-audit` | Audit pages for WCAG compliance |
| `file-download` | Download files (PDFs, CSVs) using browser session |

## 7. Install OpenBrowser CLI

```bash
uv tool install openbrowser-ai
openbrowser-ai install
```

`uv tool install` selects a platform wheel on supported macOS and Linux platforms. That wheel installs a native `openbrowser-ai` launcher and the Python `openbrowser-ai-py` CLI. Native wheels cover macOS arm64 11.0+, macOS x86_64 10.13+, Linux x86_64, and Linux aarch64. The Linux wheels contain static musl binaries and use the manylinux_2_17 tag. Windows uses the pure Python wheel. A source checkout, a `uv sync` environment, an editable install, and a git URL install use the Python CLI by default. To build a native wheel from source, set `OPENBROWSER_BUILD_NATIVE=1`, select a supported target, and use the Rust toolchain. Editable native builds fail. See [Native CLI frontend](https://docs.openbrowser.me/customize/integrations/native-cli) for the public reference.

## 8. Usage

Skills are discovered automatically. Load a specific skill:

```
use skill tool to load openbrowser/web-scraping
```

Before a sequence of CLI calls, start the daemon and browser once:

```bash
openbrowser-ai warm
openbrowser-ai -c "await navigate('https://example.com')"
openbrowser-ai -c "print(await evaluate('document.title'))"
```

On a native wheel, an exact `openbrowser-ai -c CODE` call uses the native launcher to send one JSON request to the daemon. On a pure Python install, the Python thin client sends the same request. Both paths keep variables in the daemon. A cold call starts the daemon and browser.

Every other command form uses `openbrowser-ai-py`. Examples include `--mcp`, `--version`, `--help`, scrape, web-search, prompt runs, daemon commands, `-c -`, and options outside the exact `-c CODE` form. Keep `openbrowser-ai-py` installed. If the Python CLI is missing, delegated commands fail with a reinstall message.

The native launcher accepts a JSON response frame up to 8 MiB (8,388,608 bytes). The limit includes the JSON envelope. The limit excludes the terminating newline. A larger frame exits with code 1 and `daemon_response_too_large`. The code may have run. Inspect the page before you retry. Run the same code with `openbrowser-ai-py -c CODE` for the Python CLI. The Python CLI uses the same daemon and variables. `OPENBROWSER_MAX_OUTPUT` still applies.

For static pages, use OpenBrowser no-browser mode:

```bash
openbrowser-ai scrape https://a.example https://b.example --no-browser --jsonl --concurrency 8
```

`--no-browser` starts no Chrome, Blink, or V8 process and never falls back to a browser. It cannot run JavaScript, take screenshots, or use browser cookies. Use the default `auto` engine or `--engine browser` when a page needs JavaScript.

### 8.1. Tool Mapping

When skills reference Claude Code tools, substitute OpenCode equivalents:
- `Read`, `Write`, `Edit`, `Bash` -- use your native tools
- `Task` with subagents -- use OpenCode's subagent system
- `Skill` tool -- use OpenCode's native `skill` tool

## 9. Updating

```bash
cd ~/.config/opencode/openbrowser && git pull
```

## 10. Uninstalling

```bash
rm ~/.config/opencode/plugins/openbrowser.js
rm -rf ~/.config/opencode/skills/openbrowser
```

Optionally delete the clone: `rm -rf ~/.config/opencode/openbrowser`
