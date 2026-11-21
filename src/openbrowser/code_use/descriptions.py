# src/openbrowser/code_use/descriptions.py
"""Canonical description strings for the execute_code tool.

Imported by both the MCP server and the CLI so the function reference
is defined in exactly one place.  This module is intentionally
dependency-free (no openbrowser imports) so the CLI -c fast path can
load it without pulling in heavy packages.
"""

EXECUTE_CODE_DESCRIPTION_COMPACT = """Execute Python code with browser automation. Most functions are async (use await); download_outcomes() and download_failures() are synchronous. Use print() to return output. Variables and browser login state persist between calls.

## Core Functions
- `await navigate(url, new_tab=False)` -- Go to URL
- `await click(index)` -- Click element by index
- `await input_text(index, text, clear=True)` -- Type into input field
- `await scroll(down=True, pages=1.0)` -- Scroll page
- `await click_xy(x, y, button="left", click_count=1)` -- Click a no-index canvas, PDF, embed, or native-rendered target
- `await hover_xy(x, y)` -- Move over a no-index canvas, PDF, embed, or native-rendered target
- `await scroll_xy(x, y, delta_x=0, delta_y=0)` -- Wheel-scroll a no-index surface
- `await send_keys(keys)` -- Keyboard input (e.g. "Enter", "Escape")
- `await evaluate(code)` -- Run JavaScript, returns Python objects
- `await select_dropdown(index, text)` -- Select dropdown option
- `await done(text, success=True)` -- Signal task complete

## Downloads
- `waiter = asyncio.create_task(wait_for_download(timeout=30))` then click, then `await waiter` -- returns phase, path, error
- `download_outcomes()` / `download_failures()` -- Synchronous recent attempts and failures

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
  `failed`, or `cancelled`), `path`, `file_size`, `file_type`, `stage`, and `error`. To wait for a
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

## CSS Selectors

- `await get_selector_from_index(index: int)` -- Get the CSS selector for an element by its interactive index. Useful for building JS queries targeting specific elements. Returns a CSS selector string.

## Task Completion

- `await done(text: str, success: bool = True)` -- Signal that the task is complete. text is the final output/result. success=True if the task completed successfully. Call this only when the task is truly finished.

## Browser State

- `browser` -- The BrowserSession object. Use `state = await browser.get_browser_state_summary()` to get current page state including:
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

If a call fails with a browser connection error, do not replay a page-changing action blindly. Check `browser.lifecycle_snapshot()` for the current state. Through MCP, call `get_status`, then `reset_browser` before retrying. Through the CLI, run `openbrowser-ai daemon status` or `openbrowser-ai daemon reset`.
"""
