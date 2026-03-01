# Installing OpenBrowser for OpenCode

## Prerequisites

- [OpenCode.ai](https://opencode.ai) installed
- Git installed
- Chrome or Chromium installed
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)

## Installation

### macOS / Linux

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

### Windows (PowerShell)

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

## Verify

```bash
ls -l ~/.config/opencode/plugins/openbrowser.js
ls -l ~/.config/opencode/skills/openbrowser
```

Both should show symlinks pointing to the openbrowser directories.

## Available Skills

| Skill | Description |
|-------|-------------|
| `web-scraping` | Extract structured data from websites, handle pagination |
| `form-filling` | Fill forms, login flows, multi-step wizards |
| `e2e-testing` | Test web apps by simulating user interactions |
| `page-analysis` | Analyze page content, structure, metadata |
| `accessibility-audit` | Audit pages for WCAG compliance |
| `file-download` | Download files (PDFs, CSVs) using browser session |

## Usage

Skills are discovered automatically. Load a specific skill:

```
use skill tool to load openbrowser/web-scraping
```

### Tool Mapping

When skills reference Claude Code tools, substitute OpenCode equivalents:
- `Read`, `Write`, `Edit`, `Bash` -- use your native tools
- `Task` with subagents -- use OpenCode's subagent system
- `Skill` tool -- use OpenCode's native `skill` tool

## Updating

```bash
cd ~/.config/opencode/openbrowser && git pull
```

## Uninstalling

```bash
rm ~/.config/opencode/plugins/openbrowser.js
rm -rf ~/.config/opencode/skills/openbrowser
```

Optionally delete the clone: `rm -rf ~/.config/opencode/openbrowser`
