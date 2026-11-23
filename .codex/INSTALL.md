# Installing OpenBrowser for Codex

Browser automation skills for Codex via native skill discovery. Clone and symlink.

## Prerequisites

- Git
- Chrome or Chromium installed
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)

## Quick install with skills.sh

```bash
npx skills add billy-enrizky/openbrowser-ai --skill setup -a codex
npx skills add billy-enrizky/openbrowser-ai -a codex
```

The first command installs only the setup skill. The second installs all eight OpenBrowser skills. The `setup` skill installs the CLI and Chromium, configures MCP, and verifies a real browser session.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/billy-enrizky/openbrowser-ai.git ~/.codex/openbrowser
   ```

2. **Create the skills symlink:**
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/openbrowser/plugin/skills ~/.agents/skills/openbrowser
   ```

   **Windows (PowerShell):**
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills"
   cmd /c mklink /J "$env:USERPROFILE\.agents\skills\openbrowser" "$env:USERPROFILE\.codex\openbrowser\plugin\skills"
   ```

3. **Install OpenBrowser CLI:**
   ```bash
   uv tool install openbrowser-ai
   openbrowser-ai install
   ```

4. **Restart Codex** to discover the skills.

## Verify

```bash
ls -la ~/.agents/skills/openbrowser
```

You should see a symlink pointing to the openbrowser plugin skills directory.

## Available Skills

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

## Updating

```bash
cd ~/.codex/openbrowser && git pull
```

Skills update instantly through the symlink.

## Uninstalling

```bash
rm ~/.agents/skills/openbrowser
```

Optionally delete the clone: `rm -rf ~/.codex/openbrowser`
