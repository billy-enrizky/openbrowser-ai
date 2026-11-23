/**
 * OpenBrowser plugin for OpenCode.ai
 *
 * Injects MCP server configuration context via system prompt transform.
 * Skills are discovered via OpenCode's native skill tool from symlinked directory.
 */

import path from 'path';
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const normalizePath = (p, homeDir) => {
  if (!p || typeof p !== 'string') return null;
  let normalized = p.trim();
  if (!normalized) return null;
  if (normalized.startsWith('~/')) {
    normalized = path.join(homeDir, normalized.slice(2));
  } else if (normalized === '~') {
    normalized = homeDir;
  }
  return path.resolve(normalized);
};

export const OpenBrowserPlugin = async ({ client, directory }) => {
  const homeDir = os.homedir();
  const envConfigDir = normalizePath(process.env.OPENCODE_CONFIG_DIR, homeDir);
  const configDir = envConfigDir || path.join(homeDir, '.config/opencode');
  const skillsDir = path.join(configDir, 'skills/openbrowser');

  const getBootstrapContent = () => {
    const skills = [];
    if (fs.existsSync(skillsDir)) {
      try {
        for (const entry of fs.readdirSync(skillsDir, { withFileTypes: true })) {
          if (entry.isDirectory()) {
            const skillPath = path.join(skillsDir, entry.name, 'SKILL.md');
            if (fs.existsSync(skillPath)) {
              skills.push(entry.name);
            }
          }
        }
      } catch {
        // Ignore read errors
      }
    }

    const skillList = skills.length > 0
      ? `Available OpenBrowser skills: ${skills.join(', ')}`
      : 'No OpenBrowser skills found in skills directory.';

    return `<openbrowser-context>
You have access to OpenBrowser -- AI-powered browser automation via MCP.

A 2026-07-28 client calls \`create_context\` first and keeps the returned \`context_id\`.
Pass that handle to \`execute_code\`, \`get_status\`, and \`reset_browser\`, then call
\`close_context\` when finished. \`execute_code\` runs Python code in a persistent
per-context namespace with browser automation functions. Variables and state
persist between calls. Most browser functions are async -- use \`await\`.
\`list_downloads()\`, \`download_outcomes()\`, and \`download_failures()\` are
synchronous. Older clients
that negotiate 2024-11-05 through 2025-11-25 use the legacy three-tool catalog without
a handle.

If a browser call fails, call \`get_status\` before retrying. A page-changing action may
have partially run. Read \`browser.health\` first: wait while recovery is pending, and
reset only when no usable target exists.

Available functions in the execute_code namespace:
- Navigation: navigate(url, new_tab=False), go_back(), wait(seconds)
- Interaction: click(index), input_text(index, text, clear=True), send_keys(keys)
- Coordinate interaction: click_xy(x, y, button, click_count), hover_xy(x, y), scroll_xy(x, y, delta_x, delta_y)
- Dropdowns: select_dropdown(index, text), dropdown_options(index)
- Scrolling: scroll(down=True, pages=1.0, index=None)
- Tabs: switch(tab_id), close(tab_id)
- Files: upload_file(index, path)
- JavaScript: evaluate(code) -- execute JS in page context, returns Python objects
- Downloads: download_file(url, filename=None), wait_for_download(timeout=30, download_ref=None), download_outcomes(), download_failures(), list_downloads() -- start wait_for_download before a click that triggers a download
- State: browser.get_browser_state_summary() -- get page state with interactive elements; get_selector_from_index(index) -- CSS selector for an element index
- Completion: done(text, success=True) -- signal task complete
- Libraries: bundled json, csv, re, datetime, requests, asyncio, Path, numpy, pandas; only if installed: matplotlib, BeautifulSoup, pypdf (openbrowser-ai[pdf]), tabulate

${skillList}

Use OpenCode's native skill tool to load OpenBrowser skills when relevant:
  skill load openbrowser/deep-research
  skill load openbrowser/web-scraping
  skill load openbrowser/form-filling
  skill load openbrowser/e2e-testing
  skill load openbrowser/page-analysis
  skill load openbrowser/accessibility-audit
  skill load openbrowser/file-download

**Tool Mapping:**
- Read, Write, Edit, Bash -- use your native tools
- Task with subagents -- use OpenCode's subagent system
- Skill tool -- use OpenCode's native skill tool
</openbrowser-context>`;
  };

  return {
    'experimental.chat.system.transform': async (_input, output) => {
      const bootstrap = getBootstrapContent();
      if (bootstrap) {
        (output.system ||= []).push(bootstrap);
      }
    }
  };
};
