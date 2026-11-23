---
name: setup
description: |
  Install, configure, repair, or verify OpenBrowser for the first time.
  Trigger when the user asks to: install OpenBrowser, set up OpenBrowser,
  configure the MCP server, install the CLI, set up browser profiles,
  install the plugin skills, verify a browser, or repair a broken install.
allowed-tools: Bash(openbrowser-ai:*) Bash(uv:*) Bash(uvx:*) Bash(pip:*) Bash(pipx:*) Bash(npm:*) Bash(npx:*) Bash(claude:*) Bash(curl:*) Bash(irm:*) Read Write Edit
---

# OpenBrowser Setup

Use this skill once per machine or project. It installs the CLI and browser, configures the MCP server and profiles, installs the plugin skills, and verifies a real browser session.

## Rules

- Never print API keys, cookies, proxy credentials, or storage-state contents. Report only whether a key is set.
- Prefer `uv`, then `pipx`, then `pip`.
- Python 3.12 or newer is required.
- Use `openbrowser-ai -c` for CLI verification and `uvx openbrowser-ai --mcp` as the portable MCP command.
- A daemon started by this skill must be stopped at the end. Do not stop a daemon that was already running.
- Do not install optional provider packages unless the user wants that provider.

## Step 1 -- Inspect the machine

```bash
openbrowser-ai --version 2>/dev/null || true
command -v uv || true
command -v pipx || true
python3 --version || python --version
```

If `openbrowser-ai --version` reports 0.1.57 or newer, skip Step 2. If Python is below 3.12, install a supported Python first and repeat this step.

## Step 2 -- Install or upgrade OpenBrowser

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
pipx install --upgrade openbrowser-ai
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

Install or repair Chromium:

```bash
openbrowser-ai install
```

## Step 3 -- Inspect and repair config

```bash
openbrowser-ai config status
```

Read `path`, `state`, and `restart_required`. A missing file is created with defaults once. A legacy, invalid, or unreadable file is never rewritten. If the file is invalid, repair it in place or move it to a unique backup name and run `config status` again. Moving it aside does not migrate old settings.

Set an LLM key only when the user wants Agent or CodeAgent runs:

```bash
export GOOGLE_API_KEY=...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
```

MCP and `openbrowser-ai -c` do not need an LLM key. Do not print the value.

## Step 4 -- Configure MCP

The portable server command is:

```bash
uvx openbrowser-ai --mcp
```

Add it to the agent's native MCP configuration.

Claude Code project `.mcp.json`:

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

Codex `~/.codex/config.toml`:

```toml
[mcp_servers.openbrowser]
command = "uvx"
args = ["openbrowser-ai", "--mcp"]
```

Claude Desktop uses `claude_desktop_config.json` with the same `mcpServers` object. OpenCode can use the plugin or the same `.mcp.json` object.

Useful MCP environment variables:

| Variable | Use | Default |
|---|---|---|
| `OPENBROWSER_HEADLESS` | Run without a visible window | `true` |
| `OPENBROWSER_ALLOWED_DOMAINS` | Comma-separated domain whitelist | none |
| `OPENBROWSER_USER_DATA_DIR` | Explicit shared profile; concurrent owners are rejected | per-context managed profile |
| `OPENBROWSER_STORAGE_STATE` | One-time MCP seed for cookies plus localStorage | per-context copy |
| `OPENBROWSER_MCP_MAX_CONTEXTS` | Open contexts total, including legacy | `16` |
| `OPENBROWSER_MCP_WORKSPACE_DIR` | Base directory for per-context workspaces | `~/Downloads/openbrowser-mcp/workspace` |
| `OPENBROWSER_MCP_START_TIMEOUT` | Browser startup bound in seconds | `30` |
| `OPENBROWSER_MCP_EXEC_TIMEOUT` | Tool call timeout in seconds | `110` |
| `OPENBROWSER_SHUTDOWN_TIMEOUT` | Cleanup bound in seconds | `20` |

## Step 5 -- Install the skills

For skills.sh-compatible agents:

```bash
npx skills add billy-enrizky/openbrowser-ai
```

Install only this setup skill:

```bash
npx skills add billy-enrizky/openbrowser-ai --skill setup -a codex
```

Claude Code plugin:

```bash
claude plugin marketplace add billy-enrizky/openbrowser-ai
claude plugin install openbrowser@openbrowser-ai
```

Codex manual installation is documented at `https://github.com/billy-enrizky/openbrowser-ai/blob/main/.codex/INSTALL.md`. OpenCode manual installation is documented at `https://github.com/billy-enrizky/openbrowser-ai/blob/main/.opencode/INSTALL.md`.

## Step 6 -- Understand the profiles

- CLI daemon profile: `~/.config/openbrowser/profiles/daemon`. This is a Chrome user-data directory. No `storage_state.json` is written unless configured.
- Modern MCP context profile: `~/.config/openbrowser/profiles/mcp-<owner-id>`, seeded once from `profiles/default/storage_state.json`.
- Legacy MCP owner profile: `mcp-<server-instance-id>`.
- MCP workspaces: `~/Downloads/openbrowser-mcp/workspace/<owner-id>`.
- MCP downloads: `~/Downloads/openbrowser-mcp/<owner-id>`.
- Daemon downloads: `~/Downloads/openbrowser-daemon`.
- Set `OPENBROWSER_USER_DATA_DIR` for a shared profile. The profile lease rejects concurrent owners.
- Set `OPENBROWSER_DAEMON_PROFILE_ISOLATION=true` or `OPENBROWSER_DAEMON_PROFILE_DIR` for isolated daemons.

## Step 7 -- Verify a real browser

Check ownership before starting:

```bash
openbrowser-ai daemon status
```

If the status reports `daemon_not_running`, this setup owns the daemon and must stop it at the end. If a daemon is already running, reuse it and leave it running.

Run the browser check:

```bash
openbrowser-ai -c "await navigate('https://example.com'); print(await evaluate('document.title'))"
```

Then inspect:

```bash
openbrowser-ai daemon status
```

Read `state`, `browser.health.phase`, and `browser.cdp.phase`. A printed title and a healthy/ready status verify the install. A document title does not prove application hydration.

Optional MCP smoke test:

```python
import asyncio
from mcp import Client, StdioServerParameters

async def main():
    async with Client(StdioServerParameters(command="uvx", args=["openbrowser-ai", "--mcp"])) as client:
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

## Step 8 -- Clean up only what this setup started

If this setup started the daemon:

```bash
openbrowser-ai daemon stop
```

Poll `openbrowser-ai daemon status` until it reports `daemon_not_running`, up to about 30 seconds. If the daemon does not exit, run `openbrowser-ai daemon force-stop`, which waits and verifies listener, owner, and browser cleanup.

Do not stop a daemon that was already running. Profiles, cookies, workspaces, and downloaded files survive cleanup.

## Step 9 -- Report

Print a short setup report:

- Installed version
- Config path and state
- Browser path or channel
- MCP command and config location
- Skills installed
- Profile and download paths
- Verification result
- Optional providers skipped

Do not print secrets. Do not claim a page is ready beyond the document milestone that was observed.
