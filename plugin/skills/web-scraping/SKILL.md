---
name: web-scraping
description: |
  Extract structured data from websites, scrape page content, and collect information across multiple pages.
  Trigger when the user asks to: extract data from a website, scrape a page, collect information from URLs,
  pull content from web pages, gather data across multiple pages, or download page content.
allowed-tools: Bash(openbrowser-ai:*) Bash(curl:*) Bash(uv:*) Bash(irm:*) Read Write
---

# Web Scraping

<!-- toc:start -->
## Contents

- [1. Setup](#1-setup)
- [2. Fast path: scrape without an LLM](#2-fast-path-scrape-without-an-llm)
  - [2.1. Site discovery: map and crawl](#21-site-discovery-map-and-crawl)
- [3. Workflow](#3-workflow)
  - [3.1. Step 1 -- Navigate and get content overview](#31-step-1----navigate-and-get-content-overview)
  - [3.2. Step 2 -- Extract data with JavaScript](#32-step-2----extract-data-with-javascript)
  - [3.3. Step 3 -- Process data with Python](#33-step-3----process-data-with-python)
  - [3.4. Step 4 -- Handle pagination](#34-step-4----handle-pagination)
  - [3.5. Step 5 -- Handle infinite scroll](#35-step-5----handle-infinite-scroll)
  - [3.6. Step 6 -- Multi-page scraping](#36-step-6----multi-page-scraping)
- [4. Tips](#4-tips)
- [5. Cleanup](#5-cleanup)

<!-- toc:end -->

Extract structured data from websites using Python code execution with browser automation functions. Handles JavaScript-rendered content, pagination, and multi-page scraping.

All code runs via `openbrowser-ai -c`. The daemon starts automatically and persists variables across calls. Run `openbrowser-ai warm` once before a multi-call workflow to start the daemon and browser before the workflow begins. Most browser functions are async, use `await`. `list_downloads()`, `download_outcomes()`, and `download_failures()` are synchronous.

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

## 2. Fast path: scrape without an LLM

For a single page, prefer the built-in scrape command before writing code:

```bash
openbrowser-ai scrape <url> --format markdown
openbrowser-ai scrape <url> --format json --selector "article"
```

`--engine auto` (default) uses HTTP first and falls back to Chromium for thin JavaScript shells. Use `--engine browser` when JavaScript is required. The browser fallback uses the Playwright headless shell by default; set `OPENBROWSER_USE_HEADLESS_SHELL=0` to force full Chrome.

Add `--only-main-content` to drop navigation, headers, footers, cookie banners, and ad blocks before extraction. `--include-tag` keeps the first matching selector, and `--exclude-tag` removes matches before extraction. They work on the HTTP path and on the browser fallback.

`--render-mode` accepts `full`, `lightweight`, or `no-render`. Lightweight blocks images, fonts, media, and text tracks. No-render adds external stylesheets and disables screenshots. For browser-fallback assets, use the granular flags instead of another mode: `--text-only` blocks CSS, images, fonts, media, and text tracks while keeping scripts and screenshots, and `--no-css`, `--no-images`, `--no-fonts`, and `--no-media` each block one group. `--no-scripts` is available on the one-shot prompt interface, not on `scrape`. Blocking external CSS exposes content the page hides with `display:none`, so prefer `lightweight` when visibility matters. Use `--wait-for <selector>` for client-rendered content.

For several URLs, use batch mode. HTTP requests run concurrently, while browser fallbacks run one at a time:

```bash
openbrowser-ai scrape https://a.example https://b.example --jsonl --concurrency 8
openbrowser-ai scrape --urls-file urls.txt --output results.json
openbrowser-ai scrape --urls-file urls.txt --output results.jsonl
```

Use `--output-format json` or `--output-format jsonl` to choose batch serialization. A `.jsonl` output path selects JSONL by default. Each URL produces a record with `url`, `ok`, `engine_used`, `content`, `error`, `duration_ms`, and `timing`. A failed URL does not stop the batch.

For static pages, add `--no-browser`. This is OpenBrowser no-browser mode: a light console entry that does not import the browser or configuration stack, starts no Chrome, Blink, or V8 process, and never falls back to a browser:

```bash
openbrowser-ai scrape https://a.example https://b.example --no-browser --jsonl --concurrency 8
```

It cannot run JavaScript, take screenshots, or use browser cookies. `--engine http` selects the same HTTP-only path. Use `auto` when a page may need JavaScript, or `browser` when it definitely does. The asset flags `--no-css`, `--no-images`, `--no-fonts`, `--no-media`, and `--render-mode` control the browser fallback, not the no-browser HTTP path.

In CodeAgent, call `await scrape_page(url)` for one page or `await scrape_pages(urls, concurrency=4)` for a batch. Each URL returns its own result, and results keep the input order. HTTP requests run concurrently up to the limit. Browser fallbacks run sequentially. Pass `engine="http"` to `scrape_pages()` when the batch is static and must not start a browser. That is OpenBrowser no-browser mode: it starts no Chrome, Blink, or V8 process and cannot run JavaScript, take screenshots, or use browser cookies. `scrape_page` and `scrape_pages` also accept `only_main_content=True`, `include_tags=[...]`, and `exclude_tags=[...]`.

### 2.1. Site discovery: map and crawl

Use `map` when you need to know which pages exist, and `crawl` when you need their content. Neither starts a browser.

```bash
openbrowser-ai map https://docs.example.com --max-urls 50 --exclude '\.pdf$'
openbrowser-ai crawl https://docs.example.com --max-pages 20 --max-depth 2 --format markdown
```

`map` reads the page, `/sitemap.xml`, and one sitemap index level. It normalizes, deduplicates, and filters by origin and by `--include` / `--exclude` regular expressions. `crawl` follows same-origin links breadth first, fetches and parses each page once, and reports `ok`, `depth`, `title`, `content`, and `links_found` per page.

In CodeAgent, call `await map_urls(url)` and `await crawl_site(url, max_pages=20, max_depth=2)`. Both are HTTP-only, so a JavaScript shell page is reported as skipped rather than rendered.

## 3. Workflow

### 3.1. Step 1 -- Navigate and get content overview

```bash
openbrowser-ai -c - <<'EOF'
await navigate("https://example.com/data")

# Get browser state to see page title, URL, element count
state = await browser.get_browser_state_summary()
print(f"Title: {state.title}")
print(f"URL: {state.url}")
print(f"Elements: {len(state.dom_state.selector_map)}")
EOF
```

### 3.2. Step 2 -- Extract data with JavaScript

Use `evaluate()` to run JS in the browser and return structured data directly as Python objects:

```bash
openbrowser-ai -c - <<'EOF'
data = await evaluate("""
(function(){
  return Array.from(document.querySelectorAll(".product-card")).map(el => ({
    name: el.querySelector(".title")?.textContent?.trim(),
    price: el.querySelector(".price")?.textContent?.trim(),
    url: el.querySelector("a")?.href
  }))
})()
""")

import json
print(json.dumps(data, indent=2))
EOF
```

### 3.3. Step 3 -- Process data with Python

Use pandas, regex, or other Python tools to clean and transform extracted data:

```bash
openbrowser-ai -c - <<'EOF'
import json

# Filter and transform
filtered = [item for item in data if item.get("price")]
for item in filtered:
    # Extract numeric price
    price_str = item["price"].replace("$", "").replace(",", "")
    item["price_float"] = float(price_str)

# Sort by price
filtered.sort(key=lambda x: x["price_float"])
print(json.dumps(filtered, indent=2))
EOF
```

Or with pandas if available:

```bash
openbrowser-ai -c - <<'EOF'
import pandas as pd
df = pd.DataFrame(data)
print(df.to_string())
EOF
```

### 3.4. Step 4 -- Handle pagination

```bash
openbrowser-ai -c - <<'EOF'
results = []
page = 1

while True:
    # Extract data from current page
    page_data = await evaluate("""
    (function(){
      return Array.from(document.querySelectorAll(".item")).map(el => ({
        name: el.textContent.trim()
      }))
    })()
    """)
    results.extend(page_data)
    print(f"Page {page}: {len(page_data)} items")

    # Check for next button
    has_next = await evaluate("""
    (function(){ return !!document.querySelector(".pagination .next:not(.disabled)") })()
    """)

    if not has_next:
        break

    # Replace with the actual index from browser.get_browser_state_summary()
    await click(next_button_index)
    await wait(2)
    page += 1

print(f"Total: {len(results)} items")
EOF
```

### 3.5. Step 5 -- Handle infinite scroll

```bash
openbrowser-ai -c - <<'EOF'
results = []
prev_count = 0

for _ in range(20):  # Max 20 scroll attempts
    # Get current items
    count = await evaluate("""
    (function(){ return document.querySelectorAll(".item").length })()
    """)

    if count == prev_count:
        break  # No new content loaded

    prev_count = count
    await scroll(down=True, pages=3)
    await wait(1)

# Now extract all loaded items
results = await evaluate("""
(function(){
  return Array.from(document.querySelectorAll(".item")).map(el => ({
    text: el.textContent.trim()
  }))
})()
""")
print(f"Extracted {len(results)} items")
EOF
```

### 3.6. Step 6 -- Multi-page scraping

```bash
openbrowser-ai -c - <<'EOF'
urls = [
    "https://example.com/page-1",
    "https://example.com/page-2",
    "https://example.com/page-3",
]

all_data = []
for url in urls:
    await navigate(url)
    await wait(1)

    page_data = await evaluate("""
    (function(){
      return document.querySelector("h1")?.textContent?.trim()
    })()
    """)
    all_data.append({"url": url, "title": page_data})
    print(f"{url}: {page_data}")

import json
print(json.dumps(all_data, indent=2))
EOF
```

## 4. Tips

- Code is piped via stdin using heredoc (`-c - <<'EOF'`), so all Python syntax works without shell escaping issues.
- Use `evaluate()` for structured DOM extraction -- it returns Python objects directly.
- Use Python for post-processing: filtering, sorting, deduplication, format conversion.
- For large datasets, process pages incrementally rather than loading everything into memory.
- Check for rate limiting; add `await wait(2)` between page loads if needed.
- Use `get_tabs()` to list tabs without a DOM scan. Pass a returned `tab_id` to `switch()` or `close()` when a page opens a new tab.
- Variables persist between `-c` calls while the daemon is running, so you can build up results across multiple calls.

## 5. Cleanup

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

Persist results to disk before stopping the daemon. `daemon reset` clears the Python namespace. Do not call `done()` as a substitute for cleanup; it only marks the task complete inside the agent loop.
