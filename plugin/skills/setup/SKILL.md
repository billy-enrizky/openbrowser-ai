---
name: setup
description: |
  Install, configure, repair, or verify OpenBrowser for the first time.
  Trigger when the user asks to: install OpenBrowser, set up OpenBrowser,
  configure the MCP server, install the CLI, set up browser profiles,
  install the plugin skills, verify a browser, or repair a broken install.
allowed-tools: Bash(openbrowser-ai:*) Bash(openbrowser-ai-py:*) Bash(uv:*) Bash(uvx:*) Bash(python:*) Bash(python3:*) Bash(pip:*) Bash(pipx:*) Bash(npm:*) Bash(npx:*) Bash(claude:*) Bash(curl:*) Bash(irm:*) Read Write Edit
---

# OpenBrowser Setup

<!-- toc:start -->
## Contents

- [1. Rules](#1-rules)
- [2. Step 1 -- Inspect the machine](#2-step-1----inspect-the-machine)
- [3. Step 2 -- Install or upgrade OpenBrowser](#3-step-2----install-or-upgrade-openbrowser)
  - [3.1. Native web search](#31-native-web-search)
  - [3.2. Fast CLI paths](#32-fast-cli-paths)
  - [3.3. Native frontend scope](#33-native-frontend-scope)
- [4. Step 3 -- Inspect and repair config](#4-step-3----inspect-and-repair-config)
- [5. Step 4 -- Configure MCP](#5-step-4----configure-mcp)
- [6. Step 5 -- Install the skills](#6-step-5----install-the-skills)
- [7. Step 6 -- Understand the profiles](#7-step-6----understand-the-profiles)
- [8. Step 7 -- Verify a real browser](#8-step-7----verify-a-real-browser)
- [9. Step 8 -- Report](#9-step-8----report)

<!-- toc:end -->

Use this skill once per machine or project. It installs the CLI and browser, configures the MCP server and profiles, installs the plugin skills, and verifies a real browser session.

## 1. Rules

- Never print API keys, cookies, proxy credentials, or storage-state contents. Report only whether a key is set.
- Prefer `uv`, then `pipx`, then `pip`.
- Python 3.12 or newer is required.
- Use `openbrowser-ai -c` for CLI verification and `openbrowser-ai --mcp` for the installed MCP server.
- Run the exact form `openbrowser-ai -c CODE` for the native fast path. Use `openbrowser-ai-py -c CODE` for the Python CLI.
- Keep `openbrowser-ai-py` installed. The native launcher uses it for delegated commands.
- If `openbrowser-ai -c` returns `daemon_response_too_large`, inspect the page before you retry. Use `openbrowser-ai-py -c` or reduce the output.
- For stateful MCP calls, create a context and pass its `context_id`. Call `close_context` when finished. The stateless `web_search` tool needs no context and does not start a browser.
- Verify with a temporary socket and profile. Never stop a user daemon that this setup did not start.
- Do not install optional provider packages unless the user wants that provider.

## 2. Step 1 -- Inspect the machine

```bash
openbrowser-ai --version 2>/dev/null || true
openbrowser-ai-py --version 2>/dev/null || true
openbrowser-ai web-search --help 2>/dev/null || true
command -v openbrowser-ai-py || true
command -v uv || true
command -v pipx || true
python3 --version || python --version
```

Run `openbrowser-ai web-search --help`. This shows whether the installed CLI has web search.
`openbrowser-ai-py --version` reports the Python CLI. Every other command form uses that CLI. Examples include `--mcp`, `--version`, `--help`, scrape, web-search, prompt runs, daemon commands, `-c -`, and options outside the exact `-c CODE` form.
If both version commands and the web-search help command succeed, skip only the CLI upgrade commands in Step 2.
If any of the three commands fails, run the Step 2 install or upgrade commands.
After the CLI is present, always run `openbrowser-ai install` to repair Chromium.
If Python is below 3.12, install a supported Python. Then repeat this step.

## 3. Step 2 -- Install or upgrade OpenBrowser

Preferred:

```bash
uv tool install --upgrade openbrowser-ai
```

With optional providers and the interactive CLI:

```bash
uv tool install --upgrade "openbrowser-ai[all]"
```

Fallbacks:

```bash
pipx install openbrowser-ai || pipx upgrade openbrowser-ai
python3 -m pip install --user --upgrade openbrowser-ai
```

Install script alternatives:

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.sh | sh
```

```powershell
# Windows PowerShell
irm https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.ps1 | iex
```

Pass `--no-browser` to either installer to skip Chromium installation. This installer flag is separate from the `scrape --no-browser` mode. If `install.sh` falls back to pip, pass `--local` to request a user install.

`pip`, `pipx`, and `uv` select a platform wheel on supported macOS and Linux platforms. A source checkout, a `uv sync` environment, an editable install, and a git URL install use the Python CLI by default.

Install or repair Chromium:

```bash
openbrowser-ai install
```

`openbrowser-ai install` uses `uvx playwright install chromium`. If `uvx` is unavailable, install uv first or run `python -m playwright install chromium` with the interpreter that owns the OpenBrowser install. The MCP command also uses `openbrowser-ai --mcp`, so it does not require uvx at runtime.

### 3.1. Native web search

The base OpenBrowser package includes the DDGS client at `openbrowser.web_search`. Do not install DDGS separately and do not shell out to a `ddgs` CLI. The `web-search` command does not start a browser.

OpenBrowser 0.1.61 provides `openbrowser-ai web-search`. To install the source branch, run `uv tool install --force "openbrowser-ai @ git+https://github.com/billy-enrizky/openbrowser-ai.git@main"`. A git URL install uses the Python CLI. To run the source build once without changing the installed CLI, use `uvx --from git+https://github.com/billy-enrizky/openbrowser-ai.git@main openbrowser-ai web-search "OpenBrowser MCP docs"`.

```bash
openbrowser-ai web-search "OpenBrowser MCP docs"
openbrowser-ai web-search "OpenBrowser MCP docs" --json --output web-search-results.json
openbrowser-ai web-search "OpenBrowser MCP docs" --urls-only --output web-search-urls.txt
openbrowser-ai scrape --urls-file web-search-urls.txt --no-browser --output scraped-pages.jsonl --output-format jsonl
```

Search snippets are discovery previews. Scrape and read the source page before citing it. The package caps `max_results` at 20.

`--no-browser` is OpenBrowser no-browser mode. It starts no Chrome, Blink, or V8 process, never falls back to a browser, and cannot run JavaScript, take screenshots, or use browser cookies. Use the default `auto` engine or `--engine browser` when a page needs JavaScript.

### 3.2. Fast CLI paths

On a native wheel, an exact `openbrowser-ai -c CODE` call uses the native launcher to send one JSON request to the daemon. On a pure Python install, the Python thin client sends the same request. Both paths keep variables in the daemon. A cold call starts the daemon and browser.

Every other command form uses `openbrowser-ai-py`. Examples include `--mcp`, `--version`, `--help`, scrape, web-search, prompt runs, daemon commands, `-c -`, and options outside the exact `-c CODE` form.

For static pages, `openbrowser-ai scrape --no-browser` uses OpenBrowser no-browser mode. It uses a light console entry that does not import the browser or configuration stack, starts no Chrome, Blink, or V8 process, and never falls back to a browser. Multi-URL batches use bounded concurrency:

```bash
openbrowser-ai scrape https://example.com --no-browser --format text
openbrowser-ai scrape https://a.example https://b.example --no-browser --jsonl --concurrency 8
```

It cannot run JavaScript, take screenshots, or use browser cookies. Use the default `auto` engine or `--engine browser` when a page needs JavaScript.

### 3.3. Native frontend scope

See [Native CLI frontend](https://docs.openbrowser.me/customize/integrations/native-cli) for the public reference.

Native platform wheels cover:

- macOS arm64, minimum 11.0
- macOS x86_64, minimum 10.13
- Linux x86_64, manylinux_2_17
- Linux aarch64, manylinux_2_17

Windows uses the pure Python wheel. The Linux wheels contain static musl binaries and use the manylinux_2_17 tag. A source checkout, a `uv sync` environment, an editable install, and a git URL install use the Python CLI by default. To build a native wheel from source, set `OPENBROWSER_BUILD_NATIVE=1`, select a supported target, and use the Rust toolchain. Editable native builds fail.

The native launcher accepts a JSON response frame up to 8 MiB (8,388,608 bytes). The limit includes the JSON envelope. The limit excludes the terminating newline. A larger frame exits with code 1 and `daemon_response_too_large`. The code may have run. Inspect the page before you retry. Run the same code with `openbrowser-ai-py -c CODE` for the Python CLI. The Python CLI uses the same daemon and variables. `OPENBROWSER_MAX_OUTPUT` still applies.

## 4. Step 3 -- Inspect and repair config

```bash
openbrowser-ai config status
```

Read `path`, `state`, and `restart_required`. A missing file is created with defaults once. A legacy, invalid, or unreadable file is never rewritten. If the file is invalid, repair it in place or move it to a unique backup name and run `config status` again. Moving it aside does not migrate old settings.

Set an LLM key only when the user wants Agent or CodeAgent runs:

```bash
export GOOGLE_API_KEY=...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
```

MCP and `openbrowser-ai -c` do not need an LLM key. Do not print the value.

## 5. Step 4 -- Configure MCP

The installed server command is `openbrowser-ai --mcp`. The zero-install form is `uvx openbrowser-ai --mcp`. MCP runs through the Python CLI. On a native wheel, the native launcher delegates `--mcp` to `openbrowser-ai-py`. The default stdio transport is a lean JSON-RPC loop that answers discovery and `tools/list` before it imports the MCP SDK. Set `OPENBROWSER_MCP_TRANSPORT=sdk` to use the MCP SDK 2.3 escape hatch. Invalid values exit with code 2. Modern `2026-07-28` clients see seven tools: `create_context`, `execute_code`, `get_status`, `reset_browser`, `close_context`, `web_search`, and `scrape_web`. The legacy handshakes `2024-11-05`, `2025-03-26`, `2025-06-18`, and `2025-11-25` see four tools: `execute_code`, `get_status`, `reset_browser`, and `web_search`. Both catalogs include stateless `web_search`. The modern catalog adds stateless `scrape_web`, which fetches one HTTP(S) page in OpenBrowser no-browser mode: it starts no Chrome, Blink, or V8 process and stays on the HTTP path. JavaScript, screenshots, and browser cookies are unavailable on this path. Neither stateless tool needs a `context_id` or starts a browser. Use the handle returned by `create_context` for stateful calls.

Add it to the agent's native MCP configuration.

Claude Code project `.mcp.json`:

```json
{
  "mcpServers": {
    "openbrowser": {
      "command": "openbrowser-ai",
      "args": ["--mcp"]
    }
  }
}
```

Codex `~/.codex/config.toml`:

```toml
[mcp_servers.openbrowser]
command = "openbrowser-ai"
args = ["--mcp"]
```

Claude Desktop uses `claude_desktop_config.json` with the same `mcpServers` object.

OpenCode `opencode.json`:

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

If `openbrowser-ai` is not on the agent's PATH, replace it with the absolute path to the installed executable. To use uvx instead, set `command` to `uvx` and `args` to `["openbrowser-ai", "--mcp"]`; do not put the whole command in one string.

Useful MCP environment variables:

| Variable | Use | Default |
|---|---|---|
| `OPENBROWSER_MCP_TRANSPORT` | Stdio implementation: `lean` or `sdk`. Invalid values exit with code 2 | `lean` |
| `OPENBROWSER_HEADLESS` | Run without a visible window | `true` |
| `OPENBROWSER_ALLOWED_DOMAINS` | Comma-separated domain whitelist | none |
| `OPENBROWSER_RENDER_MODE` | Asset policy for new contexts: `full`, `lightweight`, or `no-render` | `full` |
| `OPENBROWSER_TEXT_ONLY` | Set to `1` to block external CSS, images, fonts, media, and text tracks while keeping scripts and screenshots | `false` |
| `OPENBROWSER_BLOCK_RESOURCES` | Comma-separated groups to block: `css`, `images`, `fonts`, `media`, `scripts` | none |
| `OPENBROWSER_USER_DATA_DIR` | Explicit shared profile; concurrent owners are rejected | per-context managed profile |
| `OPENBROWSER_STORAGE_STATE` | One-time MCP seed for cookies plus localStorage | per-context copy |
| `OPENBROWSER_MCP_MAX_CONTEXTS` | Open contexts total, including legacy | `16` |
| `OPENBROWSER_MCP_WORKSPACE_DIR` | Base directory for per-context workspaces | `~/Downloads/openbrowser-mcp/workspace` |
| `OPENBROWSER_MCP_START_TIMEOUT` | Browser startup bound in seconds | `30` |
| `OPENBROWSER_MCP_EXEC_TIMEOUT` | Tool call timeout in seconds | `110` |
| `OPENBROWSER_SHUTDOWN_TIMEOUT` | Cleanup bound in seconds | `20` |

## 6. Step 5 -- Install the skills

For skills.sh-compatible agents:

```bash
npx skills add billy-enrizky/openbrowser-ai
```

Install only this setup skill:

```bash
npx skills add billy-enrizky/openbrowser-ai --skill setup -a codex
```

These commands install skill files. They do not execute the setup workflow. Run this skill after installation to configure the CLI and MCP server.

Claude Code plugin:

```bash
claude plugin marketplace add billy-enrizky/openbrowser-ai
claude plugin install openbrowser@openbrowser-ai
```

Codex manual installation is documented at `https://github.com/billy-enrizky/openbrowser-ai/blob/main/.codex/INSTALL.md`. OpenCode manual installation is documented at `https://github.com/billy-enrizky/openbrowser-ai/blob/main/.opencode/INSTALL.md`.

## 7. Step 6 -- Understand the profiles

- CLI daemon profile: `~/.config/openbrowser/profiles/daemon`. This is a Chrome user-data directory. No `storage_state.json` is written unless configured.
- Modern MCP context profile: `~/.config/openbrowser/profiles/mcp-<owner-id>`, seeded once from `profiles/default/storage_state.json`.
- Legacy MCP owner profile: `mcp-<server-instance-id>`.
- MCP workspaces: `~/Downloads/openbrowser-mcp/workspace/<owner-id>`.
- MCP downloads: `~/Downloads/openbrowser-mcp/<owner-id>`.
- Daemon downloads: `~/Downloads/openbrowser-daemon`.
- Set `OPENBROWSER_USER_DATA_DIR` for a shared profile. The profile lease rejects concurrent owners.
- Set `OPENBROWSER_DAEMON_PROFILE_ISOLATION=true` or `OPENBROWSER_DAEMON_PROFILE_DIR` for isolated daemons.

## 8. Step 7 -- Verify a real browser

First verify the CLI entries:

```bash
openbrowser-ai --version
openbrowser-ai-py --version
```

The two commands must report the package version. If `openbrowser-ai-py --version` fails, reinstall OpenBrowser.

Then verify the HTTP-only path. It should print page text without starting Chromium:

```bash
openbrowser-ai scrape https://example.com --engine http --format text
```

Then verify a real browser. Use the direct API so verification never attaches to or stops a daemon:

```bash
python - <<'PY'
import asyncio
from openbrowser import BrowserProfile, BrowserSession


async def main():
    session = BrowserSession(
        browser_profile=BrowserProfile(headless=True, user_data_dir=None)
    )
    try:
        await session.start()
        await session.navigate_to("https://example.com")
        print(await session.get_current_page_title())
        print(session.lifecycle_snapshot()["health"]["phase"])
    finally:
        await session.stop()


asyncio.run(main())
PY
```

`user_data_dir=None` creates a throwaway profile. The script starts a real Chrome, prints the title and health phase, and stops. It does not touch the user's daemon or profile. A document title does not prove application hydration.

If you also want to verify the CLI daemon, run the exact form `openbrowser-ai -c CODE` explicitly. On macOS and Linux, set `OPENBROWSER_SOCKET` and `OPENBROWSER_DAEMON_PROFILE_DIR` to isolate it. On Windows the daemon uses fixed port 19222 and ignores `OPENBROWSER_SOCKET`, so only stop a daemon you started. If the native command returns `daemon_response_too_large`, inspect the page and run the same code with `openbrowser-ai-py -c`.

For a batch of autonomous commands, start the daemon and browser once before the first call:

```bash
openbrowser-ai warm
```

This starts the daemon and browser and prints `{"ok": true, "elapsed_ms": ..., "url": ...}`. Later `openbrowser-ai -c` calls reuse the same process. The command does not change the browser profile or navigate to an external site.

Optional MCP smoke test:

This test exercises the default lean transport. To test the SDK transport, set `OPENBROWSER_MCP_TRANSPORT=sdk` in the child environment.

```python
import asyncio
from mcp import Client, StdioServerParameters

async def main():
    async with Client(StdioServerParameters(command="openbrowser-ai", args=["--mcp"])) as client:
        created = await client.call_tool("create_context", {})
        context_id = created.structured_content["context_id"]
        try:
            await client.call_tool("execute_code", {
                "context_id": context_id,
                "code": "print(await evaluate('document.title'))",
            })
        finally:
            await client.call_tool("close_context", {"context_id": context_id})

asyncio.run(main())
```

## 9. Step 8 -- Report

Print a short setup report:

- Installed version
- CLI entries present: `openbrowser-ai` and `openbrowser-ai-py`
- Config path and state
- Browser path or channel
- MCP command and config location
- Skills installed
- Profile and download paths
- Verification result
- Optional providers skipped
- Confirmation that the isolated verification daemon and profile were removed

Do not print secrets. Do not claim a page is ready beyond the document milestone that was observed.
