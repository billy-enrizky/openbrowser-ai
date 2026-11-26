# Installing OpenBrowser for Codex

<!-- toc:start -->
## Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Quick install with skills.sh](#2-quick-install-with-skillssh)
- [3. Installation](#3-installation)
- [4. Configure the MCP server](#4-configure-the-mcp-server)
- [5. Use the CLI daemon](#5-use-the-cli-daemon)
  - [5.1. Native launcher scope](#51-native-launcher-scope)
- [6. Verify](#6-verify)
- [7. Available Skills](#7-available-skills)
- [8. Updating](#8-updating)
- [9. Uninstalling](#9-uninstalling)

<!-- toc:end -->

Browser automation skills for Codex via native skill discovery. Clone and symlink.

## 1. Prerequisites

- Git
- Chrome or Chromium installed, or the ability to install Chromium with `openbrowser-ai install`
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)

## 2. Quick install with skills.sh

```bash
npx skills add billy-enrizky/openbrowser-ai --skill setup -a codex
npx skills add billy-enrizky/openbrowser-ai -a codex
```

The first command installs only the setup skill. The second installs all eight OpenBrowser skills. The `setup` skill installs the CLI and Chromium, configures MCP, and verifies a real browser session.

These commands install skill files. They do not run the setup workflow. After installation, ask Codex to run the `openbrowser/setup` skill.

## 3. Installation

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

   On supported macOS and Linux platforms, `uv tool install` selects a platform wheel with a native `openbrowser-ai` launcher. Windows uses the pure Python wheel. Source checkouts and git URL installs use the Python CLI by default. Section 5.1 lists the native launcher scope.

4. **Restart Codex** to discover the skills.

## 4. Configure the MCP server

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

The default OpenBrowser stdio transport is a lean JSON-RPC loop. It answers discovery and the tool catalog before it imports the MCP SDK. MCP runs through the Python CLI. On a native wheel, `openbrowser-ai --mcp` delegates to `openbrowser-ai-py --mcp`. Set `OPENBROWSER_MCP_TRANSPORT=sdk` in the server environment to use the MCP SDK 2.3 escape hatch. Invalid values exit with code 2. Modern `2026-07-28` clients see seven tools: `create_context`, `execute_code`, `get_status`, `reset_browser`, `close_context`, `web_search`, and `scrape_web`. The legacy handshakes `2024-11-05`, `2025-03-26`, `2025-06-18`, and `2025-11-25` see four tools: `execute_code`, `get_status`, `reset_browser`, and `web_search`. Call `create_context` first, pass its returned `context_id` to stateful tools, and call `close_context` when finished. `scrape_web` is stateless and fetches one HTTP(S) page in OpenBrowser no-browser mode: it starts no Chrome, Blink, or V8 process and stays on the HTTP path. JavaScript, screenshots, and browser cookies are unavailable on this path.

## 5. Use the CLI daemon

Before a sequence of `openbrowser-ai -c` calls, run `openbrowser-ai warm` once to start the daemon and browser. Variables persist between calls while the daemon remains running.

```bash
openbrowser-ai warm
openbrowser-ai -c "await navigate('https://example.com')"
openbrowser-ai -c "print(await evaluate('document.title'))"
```

On a native wheel, an exact `openbrowser-ai -c CODE` call uses the native launcher to send one JSON request to the daemon. On a pure Python install, the Python thin client sends the same request. Both paths keep variables in the daemon. A cold call starts the daemon and browser.

### 5.1. Native launcher scope

See [Native CLI frontend](https://docs.openbrowser.me/customize/integrations/native-cli) for the public reference.

The native launcher handles the exact command `openbrowser-ai -c CODE`. Every other command form uses the Python CLI. Examples include:

- `--mcp`
- `--version` and `--help`
- `scrape`, `web-search`, and prompt runs
- daemon commands
- `-c -`
- an option before or after `-c`

Keep `openbrowser-ai-py` beside `openbrowser-ai`. The native launcher uses it for every delegated command. If the Python CLI is missing, delegated commands fail with a reinstall message.

Native platform wheels cover:

- macOS arm64, minimum 11.0
- macOS x86_64, minimum 10.13
- Linux x86_64, manylinux_2_17
- Linux aarch64, manylinux_2_17

The Linux wheels contain static musl binaries. Windows uses the pure Python wheel. A source checkout, a `uv sync` environment, an editable install, and a git URL install use the Python CLI by default. To build a native wheel from source, set `OPENBROWSER_BUILD_NATIVE=1`, select a supported target, and use the Rust toolchain. Editable native builds fail.

The native launcher accepts a JSON response frame up to 8 MiB (8,388,608 bytes). The limit includes the JSON envelope. The limit excludes the terminating newline. A larger frame exits with code 1 and `daemon_response_too_large`. The code may have run. Inspect the page before you retry. Run the same code with `openbrowser-ai-py -c CODE` for the Python CLI. The Python CLI uses the same daemon and variables. `OPENBROWSER_MAX_OUTPUT` still applies.

For static pages, use OpenBrowser no-browser mode:

```bash
openbrowser-ai scrape https://a.example https://b.example --no-browser --jsonl --concurrency 8
```

`--no-browser` starts no Chrome, Blink, or V8 process and never falls back to a browser. It cannot run JavaScript, take screenshots, or use browser cookies. Use the default `auto` engine or `--engine browser` when a page needs JavaScript.

## 6. Verify

Verify the CLI entries:

```bash
openbrowser-ai --version
openbrowser-ai-py --version
```

The two commands must report the package version. If `openbrowser-ai-py --version` fails, reinstall OpenBrowser.

Verify the skills link:

```bash
ls -la ~/.agents/skills/openbrowser
```

You should see a symlink pointing to the openbrowser plugin skills directory.

## 7. Available Skills

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

## 8. Updating

```bash
cd ~/.codex/openbrowser && git pull
```

Skills update instantly through the symlink.

## 9. Uninstalling

```bash
rm ~/.agents/skills/openbrowser
```

Optionally delete the clone: `rm -rf ~/.codex/openbrowser`
