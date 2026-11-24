# Installing OpenBrowser for Codex

Browser automation skills for Codex via native skill discovery. Clone and symlink.

## Prerequisites

- Git
- Chrome or Chromium installed, or the ability to install Chromium with `openbrowser-ai install`
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)

## Quick install with skills.sh

```bash
npx skills add billy-enrizky/openbrowser-ai --skill setup -a codex
npx skills add billy-enrizky/openbrowser-ai -a codex
```

The first command installs only the setup skill. The second installs all eight OpenBrowser skills. The `setup` skill installs the CLI and Chromium, configures MCP, and verifies a real browser session.

These commands install skill files. They do not run the setup workflow. After installation, ask Codex to run the `openbrowser/setup` skill.

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
   uv tool install --upgrade openbrowser-ai
   openbrowser-ai install
   ```

4. **Restart Codex** to discover the skills.

## Configure the MCP server

Add this entry to `~/.codex/config.toml`:

```toml
[mcp_servers.openbrowser]
command = "openbrowser-ai"
args = ["--mcp"]
```

If Codex cannot find `openbrowser-ai`, use `uvx` instead:

```toml
[mcp_servers.openbrowser]
command = "uvx"
args = ["openbrowser-ai", "--mcp"]
```

The OpenBrowser server uses MCP SDK 2.3 and negotiates protocol version `2026-07-28` with modern clients. Call `create_context` first, pass its returned `context_id` to stateful tools, and call `close_context` when finished. Older supported clients use the legacy four-tool catalog, which includes stateless `search_web`.

## Use the CLI daemon

Before a sequence of `openbrowser-ai -c` calls, run `openbrowser-ai warm` once to start the daemon and browser. Variables persist between calls while the daemon remains running.

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
