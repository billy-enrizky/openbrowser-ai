# ruff: noqa: E501
# src/openbrowser/code_use/descriptions.py
"""Canonical description strings for the execute_code tool.

Imported by both the MCP server and the CLI so the function reference
is defined in exactly one place.  This module is intentionally
dependency-free (no openbrowser imports) so the CLI -c fast path can
load it without pulling in heavy packages.
"""

EXECUTE_CODE_DESCRIPTION_COMPACT = """Execute Python code with browser automation. Most functions are async (use await).
download_outcomes() and download_failures() are synchronous. Use print() to return output.
Variables and browser login state persist between calls.
After a browser error, use MCP `get_status` or CLI `openbrowser-ai daemon status`; `health` is not page readiness.

## Core Functions
- `await navigate(url, new_tab=False)` -- Go to URL
- `await find_element(role=None, name=None, element_id=None, tag=None, timeout=0)` -- Resolve an index from a screenshotless state
- `await click_element(role=..., name=...)` -- Find and click in one call
- `await fill_element(text, role=..., name=...)` -- Find an input and type in one call
- `await click(index)` -- Click element by index
- `await input_text(index, text, clear=True)` -- Type into input field
- `await scroll(down=True, pages=1.0)` -- Scroll page
- `await click_xy(x, y, button="left", click_count=1)` -- Click a no-index canvas, PDF, embed, or native-rendered target
- `await hover_xy(x, y)` -- Move over a no-index canvas, PDF, embed, or native-rendered target
- `await scroll_xy(x, y, delta_x=0, delta_y=0)` -- Wheel-scroll a no-index surface
- `await send_keys(keys)` -- Keyboard input (e.g. "Enter", "Escape")
- `await get_tabs()` -- List open tabs without a DOM scan
- `await evaluate(code)` -- Run JavaScript, returns Python objects
- `await select_dropdown(index, text)` -- Select dropdown option
- `await done(text, success=True)` -- Signal task complete

## CSS selector fast path
- Prefer indexed actions when the current browser state provides a valid index. Use CSS helpers when a selector is stable.
- `await click_selector(selector, timeout=0.0)`: Click the first CSS match through `evaluate()`. A zero timeout makes one lookup. A positive finite timeout waits for a match. For an `<a>` with `target="_blank"` or `download`, the helper uses `click_element(element_id=...)` when the anchor has an id. Otherwise, use `click_element()` or `click(index=...)`.
- `await fill_selector(selector, text, timeout=0.0)`: Set a standard `<input>` or `<textarea>` with the native value setter and dispatch bubbling `input` and `change` events. Use `input_text(index, text)` for rich editors and other controls.
- `await wait_for_selector(selector, timeout=10.0)`: Wait for a CSS match and return the number of matching elements. The timeout must be positive and finite.
- Invalid CSS selectors raise `ValueError`. Missing matches raise `LookupError` for click and fill helpers, and `TimeoutError` for `wait_for_selector()`.

## Downloads
- `waiter = asyncio.create_task(wait_for_download(timeout=30))` then click, then `await waiter`
  -- returns phase, path, file_size, file_type, stage, error, download_ref
- To wait for a prior outcome, pass its `download_ref`; it resolves only in the attached session's live registry
- `download_outcomes()` / `download_failures()` -- Synchronous recent attempts and failures

## Web Search
- `await search_web(query, max_results=5, backend="auto", region="us-en", safe_search="moderate", timelimit=None)` returns dictionaries with `title`, `url`, and `snippet`.
- Search snippets do not count as evidence. Read the source page before you cite it. The package caps `max_results` at 20.
- `await search_and_scrape(query, max_results=5, concurrency=4, format="markdown", ...)` searches, then passes ordered URLs to `scrape_pages`.
- The result has `query`, `search_results`, and `page_results` fields.
- Use `engine="http"` to prevent browser fallback. The default `auto` mode can use the browser for pages that need it.

## Scraping
- `await scrape_page(url=None, format="markdown", selector=None, wait_for=None, engine="auto")`
  : Scrape one page without an LLM. Formats: markdown, text, html, links, json.
    `engine=auto` uses HTTP first and falls back to the browser.
    `scrape` is an alias for `scrape_page`.
- `await scrape_pages(urls, concurrency=4, format="markdown", wait_for=None, engine="auto")`
  : Scrape URLs without an LLM. HTTP requests run concurrently. Browser fallbacks
    run one at a time. Returns ok, engine_used, content, error, and duration_ms per URL.

## State
- `state = await browser.get_browser_state_summary()` -- Get page URL, title, interactive elements
- Elements shown as `[i_N]` -- use the integer N for click/input_text, not the label text
- Indexes belong to the latest browser state. A missing or invalid index raises `ActionReadinessError`; a stale index fails the action with an `element_index_stale` readiness message. Re-read browser state before retrying
- `evaluate()` can change the DOM without refreshing indexes. Re-read browser state after DOM-changing JavaScript or a navigation
- Prefer indexed actions whenever an index exists. For no-index targets, `x` and `y` are CSS viewport pixels. If converting screenshot pixels, divide by `window.devicePixelRatio` first.

## Libraries
json, asyncio, Path, csv, re, datetime, requests (pre-imported)
Optional: numpy/np, pandas/pd, BeautifulSoup, PdfReader
"""

EXECUTE_CODE_DESCRIPTION = """Execute Python code in a persistent namespace with browser automation functions. Most functions are async -- use `await`. The download helpers `download_outcomes()` and `download_failures()` are synchronous. Use print() to return output. Variables and browser login state persist between calls.

## Navigation

- `await navigate(url: str, new_tab: bool = False)` -- Navigate to a URL. Set new_tab=True to open in a new tab.
- `await go_back()` -- Go back to the previous page in browser history. Returns a status message that includes the final URL. A document traversal waits for load; a same-document traversal waits for its commit.
- `await wait(seconds: float = 3)` -- Wait for the requested duration, capped at 30 seconds. Fractional seconds are accepted. Use after actions that trigger page loads.

## Element Interaction

- `await click(index: int)` -- Click an element by its index from browser state. Index must be >= 1. Works for buttons, links, checkboxes, radio buttons. Does NOT work for <select> elements (use select_dropdown instead).
- `await input_text(index: int, text: str, clear: bool = True)` -- Type text into an input field. clear=True (default) clears the field first; clear=False appends.
- `await find_element(role=None, name=None, element_id=None, tag=None, timeout=0.0, poll_interval=0.1)` -- Return the current interactive index for a matching element. It matches by exact element id, accessibility role and name, or tag and name, and returns the first match in selector-map order. The lookup uses `get_browser_state_summary(include_screenshot=False)`. A positive timeout polls until the element appears.
- `await click_element(role=None, name=None, element_id=None, tag=None, timeout=10.0, **click_kwargs)` -- Find an element and click it in one call. The same filters as find_element() apply.
- `await fill_element(text, role=None, name=None, element_id=None, tag=None, timeout=10.0, **input_kwargs)` -- Find an input and type text into it in one call. The same filters as find_element() apply.
- `await scroll(down: bool = True, pages: float = 1.0, index: int | None = None)` -- Scroll the page. down=True scrolls down, down=False scrolls up. pages=0.5 for half page, 1 for full page, 10 for top/bottom. Pass index to scroll within a specific container element.
- `await click_xy(x: float, y: float, button: str = "left", click_count: int = 1)` -- Click a no-index canvas, PDF, embed, or native-rendered target.
- `await hover_xy(x: float, y: float)` -- Move over a no-index canvas, PDF, embed, or native-rendered target.
- `await scroll_xy(x: float, y: float, delta_x: int = 0, delta_y: int = 0)` -- Wheel-scroll a no-index surface. Positive `delta_y` scrolls down and positive `delta_x` scrolls right.
- Prefer indexed actions whenever an index exists. The `*_xy` coordinates are CSS viewport pixels, not screenshot pixels. If `devicePixelRatio` is not 1, divide screenshot coordinates by `window.devicePixelRatio` first.
- `await send_keys(keys: str)` -- Send keyboard keys or shortcuts. Examples: "Escape", "Enter", "PageDown", "Control+o", "Control+a", "ArrowDown".
- `await upload_file(index: int, path: str)` -- Upload a file to a file input element. index is the file input element index, path is the local file path.

## Dropdowns

- `await select_dropdown(index: int, text: str)` -- Select an option in a <select> dropdown by its visible text. text must be the exact option text.
- `await dropdown_options(index: int)` -- Get all available options for a <select> dropdown. Returns the options as text. Call this first to see what options are available.

## Tab Management

- `await get_tabs()` -- Return open tabs with full target_id, url, title, parent_target_id, and active flag. It uses the CDP tab list instead of reading the full DOM.
- `await switch(tab_id: str)` -- Switch to a different browser tab. Pass the full target_id or its 4-character suffix from the browser state tabs list. If a suffix is ambiguous, retry with one of the returned full IDs.
- `await close(tab_id: str)` -- Close a browser tab. Pass the full target_id or its 4-character suffix. Closing an already closed tab is a no-op.

## JavaScript Execution

- `await evaluate(code: str)` -- Execute JavaScript in the browser page context and return the result as a Python object. Auto-wraps code in an IIFE if not already wrapped. Returns Python dicts/lists/primitives directly. Raises EvaluateError on JS errors.
  Example: `data = await evaluate('document.title')` returns the page title string.
  Example: `items = await evaluate('Array.from(document.querySelectorAll(".item")).map(e => e.textContent)')` returns a Python list of strings.

## File Downloads

- `await download_file(url: str, filename: str | None = None)` -- Download a file from a URL using the browser's cookies and session. Returns the absolute path to the downloaded file. Preserves authentication -- uses the browser's JavaScript fetch internally, so cookies and login sessions carry over. Falls back to Python requests if browser fetch fails.
  IMPORTANT: When you need to download a PDF or any file, use download_file() -- do NOT use navigate() for downloads. navigate() opens the PDF in the browser viewer but does not save the file.
  Example: `path = await download_file('https://example.com/report.pdf')`
  Example: `path = await download_file('https://example.com/data', filename='export.csv')`
  After downloading, read PDFs with: `reader = PdfReader(path); text = reader.pages[0].extract_text()`
- `await wait_for_download(timeout: float = 30.0, url: str | None = None, lookback: float = 0.0,
  download_ref: str | None = None)` -- Wait for one download to reach a terminal phase. Start the wait
  before the click so a fast download cannot finish first. Returns a dict with `phase` (`completed`,
  `failed`, or `cancelled`), `path`, `file_size`, `file_type`, `stage`, `error`, and `download_ref`.
  To wait for a
  specific prior outcome, pass its `download_ref`. A reference resolves only in the attached session's
  live download registry; after the session stops or is replaced, read the completed history instead of
  waiting on that reference. Raises TimeoutError with the last observed phase when nothing finishes in
  time. If the download may already have finished, pass `lookback=5` to accept a terminal outcome from
  the last five seconds.
  Example: `waiter = asyncio.create_task(wait_for_download(timeout=30)); await click(12); outcome = await waiter; print(outcome['phase'], outcome['path'])`
  If it failed, `outcome['stage']` names where it stopped and `outcome['error']` says why.
- `download_outcomes()` -- Synchronous. Return recent download attempts, oldest first, with their last phase and terminal detail.
- `download_failures()` -- Synchronous. Return only the attempts that ended in `failed`, with `stage` and `error`.
- `list_downloads()` -- List all files in the downloads directory. Returns a list of absolute file paths.

## Web Search
- `await search_web(query, max_results=5, backend="auto", region="us-en", safe_search="moderate", timelimit=None)` returns dictionaries with `title`, `url`, and `snippet`.
- Search snippets do not count as evidence. Read the source page before you cite it. The package caps `max_results` at 20.
- `await search_and_scrape(query, max_results=5, concurrency=4, format="markdown", ...)` searches, then passes ordered URLs to `scrape_pages`.
- The result has `query`, `search_results`, and `page_results` fields.
- Use `engine="http"` to prevent browser fallback. The default `auto` mode can use the browser for pages that need it.

## Scraping

- `await scrape_page(url=None, *, format="markdown", selector=None, wait_for=None, ...)`
  Scrape the current page or a URL without an LLM.
  Formats: `markdown`, `text`, `html`, `links`, and selector `json`.
  `engine="auto"` tries HTTP first and falls back to the browser.
  Use `engine="http"` to forbid a browser and `engine="browser"` to require one.
  `scrape` is an alias for `scrape_page`.
- `await scrape_pages(urls, *, concurrency=4, format="markdown", ...)`
  Scrape many URLs without an LLM. HTTP requests run concurrently up to `concurrency`.
  Browser fallbacks run one at a time because one browser session has one global agent focus.
  Results keep input order and each item has `ok`, `engine_used`, `content`, `error`, and `duration_ms`.
  One URL failure does not abort the batch.

## CSS Selectors

- `await get_selector_from_index(index: int)`: Get the CSS selector for an element by its interactive index. Useful for building JS queries targeting specific elements. Returns a CSS selector string.
- `await click_selector(selector: str, *, timeout: float = 0.0)`: Click the first CSS match through `evaluate()`. A zero timeout makes one lookup. A positive finite timeout waits for a match.
- `await fill_selector(selector: str, text: str, *, timeout: float = 0.0)`: Set a standard `<input>` or `<textarea>` with the native value setter. The helper dispatches bubbling `input` and `change` events. Use `input_text()` for rich editors and other controls.
- `await wait_for_selector(selector: str, timeout: float = 10.0)`: Wait for a CSS match and return the match count. The timeout must be positive and finite.

For an `<a>` with `target="_blank"` or a `download` attribute, `click_selector()` uses `click_element(element_id=...)` when the anchor has an id. Use `click_element()` or `click()` directly when the anchor has no id. Invalid selectors raise `ValueError`. Missing click and fill matches raise `LookupError`. A timed-out wait raises `TimeoutError`.

## Task Completion

- `await done(text: str, success: bool = True)` -- Signal that the task is complete. text is the final output/result. success=True if the task completed successfully. Call this only when the task is truly finished.

## Browser State

- `browser` -- The BrowserSession object. Use `state = await browser.get_browser_state_summary()` to get current page state including:
  - For index-only lookup, `find_element()` and both action helpers read `get_browser_state_summary(include_screenshot=False)`. Use a screenshot only when visual confirmation is part of the task.
  - `state.url` -- current URL
  - `state.title` -- page title
  - `state.tabs` -- list of open tabs (each has .target_id, .url, .title). For switch()/close(), use the last 4 chars of target_id or the full target_id.
  - `state.dom_state.selector_map` -- dict of {index: element} for all interactive elements
- Indexes belong to the latest browser state. `click`, `input_text`, `upload_file`, `dropdown_options`, and `select_dropdown` fail when the index is missing, invalid, or not in the current state. A missing or invalid index raises `ActionReadinessError`; a stale index fails the action with an `element_index_stale` readiness message, surfaced as `RuntimeError` in execute_code. `scroll(index=...)` follows the same rule for a supplied index; its index is optional, and 0 means the whole page. Re-read browser state before retrying
- `evaluate()` can change the DOM without refreshing indexes. Re-read browser state after DOM-changing JavaScript or a navigation
  - Each element has: `.tag_name`, `.attributes` (dict), `.get_all_children_text(max_depth=N)` (text content)

## File System

- `file_system` -- FileSystem object for file operations.

## Libraries (pre-imported)

- `json` -- JSON encoding/decoding
- `asyncio` -- async utilities
- `Path` -- pathlib.Path for file paths
- `csv` -- CSV reading/writing
- `re` -- regular expressions
- `datetime` -- date/time operations
- `requests` -- HTTP requests (synchronous)

## Optional Libraries (available if installed)

- `numpy` / `np` -- numerical computing
- `pandas` / `pd` -- data analysis and DataFrames
- `matplotlib` / `plt` -- plotting and charts
- `BeautifulSoup` / `bs4` -- HTML parsing
- `PdfReader` / `pypdf` -- PDF reading
- `tabulate` -- table formatting

## Typical Workflow

1. Navigate: `await navigate('https://example.com')`
2. Get state: `state = await browser.get_browser_state_summary()`
3. Inspect elements: iterate `state.dom_state.selector_map` to find element indices
4. Interact: `await click(index)`, `await input_text(index, 'text')`, etc.
5. Extract data: `data = await evaluate('JS expression')` -- returns Python objects
6. Process with Python: use json, pandas, re, etc.
7. Print results: `print(output)` -- this is what gets returned to the client

## Recovery

If a call fails with a browser connection error, do not replay a page-changing action blindly.
Check `browser.lifecycle_snapshot()` for the current state.
Read the `health` key from that snapshot.
`health.phase` is `unknown`, `healthy`, `degraded`, `crashed`, `stopped`, or `failed`.
`health.healthy=true` means setup or focus recovery completed. It does not prove a page can accept commands.
Read `last_focus_recovery_outcome`, `target_crash_count`, and `process_present` before retrying.
`process_present` is `null` when the published root is not bound to the recorded local owner
or when process identity is inconclusive.
This includes remote, stopped, and ambiguous local sessions.
Read the `cdp` key from that snapshot, for example `cdp = browser.lifecycle_snapshot().get('cdp') or {}`.
Read `cdp.get('healthy')`, `cdp.get('phase')`, and `cdp.get('last_error_code')` when present.
`healthy` is the last accepted connect or probe observation while the lifecycle preserves an active health claim.
An inactive snapshot can report `null` and keeps the history fields.
`last_probe_at` tells you whether a probe ran.
`cdp_connected` is only pointer presence.
Root CDP health does not prove a target session, tab, or page can accept commands.
If crash recovery is `pending`, wait for it before starting another recovery.
If recovery `failed`, inspect `focus.last_error` and reset only when no usable target exists.
If `health.phase` is `crashed` or `failed` and no usable target exists,
use `reset_browser` through MCP or `openbrowser-ai daemon reset` through the CLI.
"""
