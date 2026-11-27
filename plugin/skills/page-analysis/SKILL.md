---
name: page-analysis
description: |
  Analyze web page content, structure, and layout to understand what a page contains and how it is organized.
  Runs through keyless `openbrowser-ai -c` or local MCP code tools. No separate API key is required.
  Trigger when the user asks to: analyze a page, understand page structure, inspect a website,
  summarize page content, examine page layout, review a web page, or describe what is on a page.
allowed-tools: Bash(openbrowser-ai:*) Bash(curl:*) Bash(uv:*) Bash(irm:*) Read Write
---

# Page Analysis

<!-- toc:start -->
## Contents

- [1. Setup](#1-setup)
- [2. Workflow](#2-workflow)
  - [2.1. Step 1 -- Navigate and get overview](#21-step-1----navigate-and-get-overview)
  - [2.2. Step 2 -- Extract page metadata](#22-step-2----extract-page-metadata)
  - [2.3. Step 3 -- Detect frameworks and technologies](#23-step-3----detect-frameworks-and-technologies)
  - [2.4. Step 4 -- Content summary and statistics](#24-step-4----content-summary-and-statistics)
  - [2.5. Step 5 -- Analyze heading structure](#25-step-5----analyze-heading-structure)
  - [2.6. Step 6 -- Analyze interactive elements](#26-step-6----analyze-interactive-elements)
  - [2.7. Step 7 -- Page dimensions and scroll analysis](#27-step-7----page-dimensions-and-scroll-analysis)
  - [2.8. Step 8 -- Search for specific content patterns](#28-step-8----search-for-specific-content-patterns)
- [3. Tips](#3-tips)
- [4. Cleanup](#4-cleanup)

<!-- toc:end -->

Analyze and understand web page content, structure, and interactive elements using Python code execution. Produces a comprehensive breakdown of what is on the page and how it is organized.

Some page analysis can start from server-rendered HTML. `openbrowser-ai scrape <url> --no-browser --format text` fetches the page in OpenBrowser no-browser mode and starts no Chrome, Blink, or V8 process, but it cannot run JavaScript, take screenshots, or use browser cookies. The rendered DOM and JavaScript checks below need the browser workflow.

All code runs via `openbrowser-ai -c`. The daemon starts automatically and persists variables across calls. Run `openbrowser-ai warm` once before a multi-call workflow to start the daemon and browser before the workflow begins. Most browser functions are async, use `await`. `list_downloads()`, `download_outcomes()`, and `download_failures()` are synchronous.

`openbrowser-ai -c` needs no separate OpenAI, OpenBrowser, or MCP API key.

The CLI daemon persists cookies and login state in the Chrome user-data directory at `~/.config/openbrowser/profiles/daemon`, so authenticated sessions can be reused across later runs. No `storage_state.json` file is written unless you configure one explicitly.

## 1. Setup

Before running, verify openbrowser-ai is installed:

```bash
openbrowser-ai --help
```

If not found, install:

```bash
# macOS/Linux
curl -fsSL https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.ps1 | iex
```

## 2. Workflow

### 2.1. Step 1 -- Navigate and get overview

```bash
openbrowser-ai -c - <<'EOF'
await navigate("https://example.com")
state = await browser.get_browser_state_summary()
print(f"Title: {state.title}")
print(f"URL: {state.url}")
print(f"Interactive elements: {len(state.dom_state.selector_map)}")
print(f"Tabs: {len(state.tabs)}")
EOF
```

### 2.2. Step 2 -- Extract page metadata

```bash
openbrowser-ai -c - <<'EOF'
meta = await evaluate("""
(function(){
  return {
    title: document.title,
    description: document.querySelector("meta[name='description']")?.content,
    canonical: document.querySelector("link[rel='canonical']")?.href,
    ogTitle: document.querySelector("meta[property='og:title']")?.content,
    ogImage: document.querySelector("meta[property='og:image']")?.content,
    lang: document.documentElement.lang,
    charset: document.characterSet
  };
})()
""")

import json
print(json.dumps(meta, indent=2))
EOF
```

### 2.3. Step 3 -- Detect frameworks and technologies

```bash
openbrowser-ai -c - <<'EOF'
tech = await evaluate("""
(function(){
  const t = [];
  if (window.__NEXT_DATA__) t.push("Next.js");
  if (window.__NUXT__) t.push("Nuxt.js");
  if (document.querySelector("[data-reactroot]") || document.querySelector("#__next")) t.push("React");
  if (document.querySelector("[ng-version]")) t.push("Angular");
  if (window.jQuery) t.push("jQuery");
  if (window.Vue) t.push("Vue.js");
  if (document.querySelector("[data-svelte]")) t.push("Svelte");
  return t;
})()
""")
print(f"Technologies detected: {tech}")
EOF
```

### 2.4. Step 4 -- Content summary and statistics

```bash
openbrowser-ai -c - <<'EOF'
stats = await evaluate("""
(function(){
  return {
    headings: document.querySelectorAll("h1,h2,h3,h4,h5,h6").length,
    paragraphs: document.querySelectorAll("p").length,
    images: document.querySelectorAll("img").length,
    links: document.querySelectorAll("a").length,
    forms: document.querySelectorAll("form").length,
    tables: document.querySelectorAll("table").length,
    lists: document.querySelectorAll("ul,ol").length,
    buttons: document.querySelectorAll("button,[role='button']").length,
    inputs: document.querySelectorAll("input,textarea,select").length,
    iframes: document.querySelectorAll("iframe").length,
    scripts: document.querySelectorAll("script").length,
    stylesheets: document.querySelectorAll("link[rel='stylesheet']").length
  };
})()
""")

import json
print("Content statistics:")
print(json.dumps(stats, indent=2))
EOF
```

### 2.5. Step 5 -- Analyze heading structure

```bash
openbrowser-ai -c - <<'EOF'
headings = await evaluate("""
(function(){
  return Array.from(document.querySelectorAll("h1,h2,h3,h4,h5,h6")).map(h => ({
    tag: h.tagName,
    text: h.textContent.trim().substring(0, 80)
  }));
})()
""")

for h in headings:
    htag = h["tag"]
    htext = h["text"]
    indent = "  " * (int(htag[1]) - 1)
    print(f"{indent}{htag}: {htext}")
EOF
```

### 2.6. Step 6 -- Analyze interactive elements

```bash
openbrowser-ai -c - <<'EOF'
state = await browser.get_browser_state_summary()
elements_by_tag = {}
for idx, el in state.dom_state.selector_map.items():
    tag = el.tag_name
    elements_by_tag.setdefault(tag, []).append({
        "index": idx,
        "text": el.get_all_children_text(max_depth=1)[:50],
        "type": el.attributes.get("type", ""),
        "href": el.attributes.get("href", "")[:50] if el.attributes.get("href") else "",
    })

for tag, elems in sorted(elements_by_tag.items()):
    print(f"\n{tag} ({len(elems)} elements):")
    for e in elems[:5]:
        eidx = e["index"]
        etxt = e["text"]
        etype = e["type"]
        ehref = e["href"]
        print(f"  [{eidx}] text=\"{etxt}\" type={etype} href={ehref}")
    if len(elems) > 5:
        print(f"  ... and {len(elems) - 5} more")
EOF
```

### 2.7. Step 7 -- Page dimensions and scroll analysis

```bash
openbrowser-ai -c - <<'EOF'
dims = await evaluate("""
(function(){
  return {
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
    scrollHeight: document.body.scrollHeight,
    scrollWidth: document.body.scrollWidth,
    scrollable: document.body.scrollHeight > window.innerHeight
  };
})()
""")

import json
print(json.dumps(dims, indent=2))
if dims["scrollable"]:
    pages = dims["scrollHeight"] / dims["viewportHeight"]
    print(f"Page is approximately {pages:.1f} viewport heights long")
EOF
```

### 2.8. Step 8 -- Search for specific content patterns

```bash
openbrowser-ai -c - <<'EOF'
import re

# Get page text for Python-side analysis
text_content = await evaluate("document.body.innerText")

# Find emails
emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text_content)
print(f"Emails found: {emails}")

# Find phone numbers
phones = re.findall(r"\+?\d[\d\s()-]{7,}", text_content)
print(f"Phone numbers found: {phones}")

# Find dates
dates = re.findall(r"\d{4}-\d{2}-\d{2}|\w+ \d{1,2},? \d{4}", text_content)
print(f"Dates found: {dates}")
EOF
```

## 3. Tips

- Code is piped via stdin using heredoc (`-c - <<'EOF'`), so all Python syntax works without shell escaping issues.
- Start with `evaluate()` for metadata and DOM statistics -- gives a fast structured overview.
- Use `browser.get_browser_state_summary()` for interactive element analysis.
- Use `get_tabs()` to list tabs without a DOM scan. Pass a returned `tab_id` to `switch()` or `close()` when you need to inspect another tab.
- Use Python regex on extracted text for pattern matching (emails, phones, dates, prices).
- For long pages, use `await scroll(down=True)` and re-extract to analyze below-fold content.
- Variables persist between `-c` calls while the daemon is running, so you can build a comprehensive analysis incrementally.

## 4. Cleanup

Run this step after the task, whether it succeeded or failed.

**Do not stop a daemon you did not start.** Check `openbrowser-ai daemon status` before the first browser call. If it already reports a running daemon, reuse it and leave it running at the end. If it reports `daemon_not_running`, this run starts one and must clean it up.

For a daemon started by this run:

```bash
openbrowser-ai daemon stop
```

`daemon stop` only acknowledges the request and returns. Poll `openbrowser-ai daemon status` until it reports `daemon_not_running`, up to about 30 seconds. If the daemon does not exit, run `openbrowser-ai daemon force-stop`, which waits and verifies listener, owner, and browser cleanup.

If the task can fail mid-workflow, guarantee cleanup with a shell trap that stops, polls, and force-stops if needed:

```bash
cleanup() {
  openbrowser-ai daemon stop >/dev/null 2>&1 || true
  i=0
  while [ "$i" -lt 30 ]; do
    status=$(openbrowser-ai daemon status 2>&1 || true)
    case "$status" in
      *daemon_not_running*) return 0 ;;
    esac
    sleep 1
    i=$((i + 1))
  done
  openbrowser-ai daemon force-stop >/dev/null 2>&1 || true
}
trap cleanup EXIT
# ... openbrowser-ai -c calls here ...
```

Persist results to disk before stopping the daemon. `daemon reset` clears the Python namespace. Call `done()` only to mark the task complete inside the agent loop. Persist results and stop the daemon for cleanup.
