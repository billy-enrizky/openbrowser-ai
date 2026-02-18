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
You have access to OpenBrowser -- AI-powered browser automation tools via MCP.

The OpenBrowser MCP server provides browser control tools: browser_navigate, browser_click,
browser_type, browser_get_state, browser_get_text, browser_grep, browser_search_elements,
browser_find_and_scroll, browser_get_accessibility_tree, browser_execute_js, browser_scroll,
browser_go_back, browser_list_tabs, browser_switch_tab, browser_close_tab,
browser_list_sessions, browser_close_session, browser_close_all.

${skillList}

Use OpenCode's native skill tool to load OpenBrowser skills when relevant:
  skill load openbrowser/web-scraping
  skill load openbrowser/form-filling
  skill load openbrowser/e2e-testing
  skill load openbrowser/page-analysis
  skill load openbrowser/accessibility-audit

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
