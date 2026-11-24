# Installing OpenBrowser for OpenCode

<!-- toc:start -->
## Table of Contents

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

The OpenBrowser server uses MCP SDK 2.3 and negotiates protocol version `2026-07-28` with modern clients. Call `create_context` first, pass its returned `context_id` to stateful tools, and call `close_context` when finished. Older supported clients use the legacy four-tool catalog, which includes stateless `search_web`.

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

A warm `-c` call uses a standard-library thin client, so it does not import the full CLI or agent stack. A cold call falls back to the full client and starts the daemon.

For static pages, scrape a batch without starting a browser:

```bash
openbrowser-ai scrape https://a.example https://b.example --engine http --jsonl --concurrency 8
```

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
