---
name: cli-execute
description: Execute browser automation code via OpenBrowser CLI
allowed-tools: Bash(openbrowser-ai -c:*), Bash(openbrowser-ai daemon:*)
---

# OpenBrowser CLI Execute

Run browser automation Python code directly from the command line.
The daemon starts automatically on first use and persists the browser session.

## Quick Start

```bash
# Navigate to a page
openbrowser-ai -c "await navigate('https://example.com')"

# Get page state and interact
openbrowser-ai -c "
state = await browser.get_browser_state_summary()
print(state.url)
print(state.title)
"

# Click an element and extract data
openbrowser-ai -c "
await click(3)
data = await evaluate('document.querySelector(\"h1\").textContent')
print(data)
"
```

## Available Functions

### Navigation
- `await navigate(url, new_tab=False)` -- Navigate to URL
- `await go_back()` -- Browser back
- `await wait(seconds=3)` -- Wait (max 30)

### Interaction
- `await click(index)` -- Click element by [i_N] index
- `await input_text(index, text, clear=True)` -- Type into field
- `await scroll(down=True, pages=1.0, index=None)` -- Scroll
- `await send_keys(keys)` -- Keyboard ("Enter", "Escape", "Control+a")
- `await upload_file(index, path)` -- Upload file
- `await select_dropdown(index, text)` -- Select dropdown option
- `await dropdown_options(index)` -- List dropdown options

### Data Extraction
- `await evaluate(code)` -- Run JavaScript, returns Python objects
- `state = await browser.get_browser_state_summary()` -- Page state with interactive elements
- `await get_selector_from_index(index)` -- CSS selector for element

### Tabs
- `await switch(tab_id)` -- Switch tab (4-char ID from state.tabs)
- `await close(tab_id)` -- Close tab

### File Downloads
- `await download_file(url, filename=None)` -- Download via browser session
- `list_downloads()` -- List downloaded files

### Task Completion
- `await done(text, success=True)` -- Signal done

## Variable Persistence

Variables persist across `-c` calls while the daemon is running:

```bash
openbrowser-ai -c "await navigate('https://example.com')"
openbrowser-ai -c "title = await evaluate('document.title'); print(title)"
openbrowser-ai -c "print(title)"  # still available
```

## Daemon Management

```bash
openbrowser-ai daemon start     # Start daemon (auto-starts on first -c call)
openbrowser-ai daemon stop      # Stop daemon and browser
openbrowser-ai daemon status    # Show daemon info
openbrowser-ai daemon restart   # Restart daemon
```

## Multi-Action Batching

Batch multiple actions in one call for efficiency:

```bash
openbrowser-ai -c "
await navigate('https://example.com/search')
await input_text(1, 'python automation')
await click(2)
await wait(2)
state = await browser.get_browser_state_summary()
print(f'Results page: {state.title}')
"
```

## Pre-imported Libraries

json, asyncio, Path, csv, re, datetime, requests
Optional: numpy/np, pandas/pd, matplotlib/plt, BeautifulSoup, PdfReader, tabulate
