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

The OpenBrowser MCP server exposes a single \`execute_code\` tool that runs Python code
in a persistent namespace with async browser automation functions. Variables and state
persist between calls. All browser functions are async -- use \`await\`.

Available functions in the execute_code namespace:
- Navigation: navigate(url, new_tab=False), go_back(), wait(seconds)
- Interaction: click(index), input_text(index, text, clear=True), send_keys(keys)
- Dropdowns: select_dropdown(index, text), dropdown_options(index)
- Scrolling: scroll(down=True, pages=1.0, index=None)
- Tabs: switch(tab_id), close(tab_id)
- Files: upload_file(index, path)
- JavaScript: evaluate(code) -- execute JS in page context, returns Python objects
- Downloads: download_file(url, filename=None) -- download files using browser cookies, list_downloads() -- list downloaded files
- State: browser.get_browser_state_summary() -- get page state with interactive elements
- Completion: done(text, success=True) -- signal task complete
- Libraries: json, csv, re, datetime, requests, asyncio, Path, numpy, pandas, matplotlib, BeautifulSoup, pypdf

${skillList}

Use OpenCode's native skill tool to load OpenBrowser skills when relevant:
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
