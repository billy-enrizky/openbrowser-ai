# Code-Use Mode

<!-- toc:start -->
## Table of Contents

- [1. Problem Solved](#1-problem-solved)
  - [1.1. Namespace](#11-namespace)
  - [1.2. CSS selector fast path](#12-css-selector-fast-path)
  - [1.3. Step 1: Navigate](#13-step-1-navigate)
  - [1.4. Step 2 analyse our DOM state and write code to extract the data we need.](#14-step-2-analyse-our-dom-state-and-write-code-to-extract-the-data-we-need)
  - [1.5. Step 3: Analyse output & save the data to a file](#15-step-3-analyse-output-save-the-data-to-a-file)
- [2. Run lifecycle](#2-run-lifecycle)

<!-- toc:end -->

Code-Use Mode is a Notebook-like code execution system for browser automation. Instead of the agent choosing from a predefined set of actions, the LLM writes Python code that gets executed in a persistent namespace with all browser control functions available.

## 1. Problem Solved

**Code-Use Mode solves this** by giving the agent a Python execution environment where it can:
- Store extracted data in variables
- Loop through pages programmatically
- Combine results from multiple extractions
- Process and filter data before saving
- Use conditional logic to decide what to do next
- Output more tokens than the LLM writes

### 1.1. Namespace
The namespace is initialized with:

**Browser Control Functions:**
- `navigate(url, new_tab=False)` - Navigate to a URL
- `go_back()` - Go back to the previous page
- `wait(seconds=3)` - Wait for specified seconds
- `click(index)` - Click an element by index
- `click_xy(x, y, button='left', click_count=1)` - Click a no-index canvas, PDF, embed, or native-rendered target using CSS viewport pixels
- `hover_xy(x, y)` - Move over a no-index canvas, PDF, embed, or native-rendered target using CSS viewport pixels
- `input_text(index, text, clear=True)` - Type text into an input field
- `find_element(role=None, name=None, element_id=None, tag=None, timeout=0)` - Resolve an index from a screenshotless state
- `click_element(role=..., name=...)` - Find and click in one call
- `fill_element(text, role=..., name=...)` - Find an input and type in one call
- `scroll(down, pages=1.0, index=None)` - Scroll the page or a container. `down` is required.
- `scroll_xy(x, y, delta_x=0, delta_y=0)` - Wheel-scroll a no-index surface using CSS viewport pixels
- `send_keys(keys)` - Send keyboard keys or shortcuts
- `upload_file(index, path)` - Upload a file to a file input element
- `select_dropdown(index, text)` - Select an option in a `<select>` dropdown
- `dropdown_options(index)` - Get all options for a `<select>` dropdown
- `get_tabs()` - List tabs without a DOM scan
- `switch(tab_id)` - Switch to a different browser tab
- `close(tab_id)` - Close a browser tab
- `evaluate(code, variables={})` - Execute JavaScript and return Python objects
- `get_selector_from_index(index)` - Get CSS selector for an element by index
- `done(text, success, files_to_display)` - Mark task complete
- `download_file(url, filename=None)` - Download a file using browser cookies
- `wait_for_download(timeout=30, url=None, lookback=0, download_ref=None)` - Wait for a terminal download phase; start it before the click
- `download_outcomes()` / `download_failures()` - Synchronous recent attempts and failures
- `list_downloads()` - List files in the downloads directory
- `scrape_page(url=None, format='markdown', selector=None, wait_for=None, engine='auto', only_main_content=False, include_tags=None, exclude_tags=None)` - Scrape one page without an LLM. `engine=auto` uses HTTP first and falls back to the browser. `only_main_content` drops navigation, headers, footers, cookie banners, and ad blocks. `include_tags` keeps the first matching selector and `exclude_tags` removes matches
- `scrape_pages(urls, concurrency=4, format='markdown', engine='auto', only_main_content=False, include_tags=None, exclude_tags=None)` - Scrape many URLs without an LLM. HTTP requests run concurrently. Browser fallbacks run sequentially
- `search_web(query, max_results=5, backend='auto', region='us-en', safe_search='moderate', timelimit=None)` - Search with DDGS and return title, URL, and snippet fields
- `search_and_scrape(query, max_results=5, concurrency=4, format='markdown', ...)` - Search, then scrape result URLs through `scrape_pages`
- `map_urls(url, max_urls=100, include=None, exclude=None, timeout_ms=None, use_sitemap=True)` - List URLs from one page, `/sitemap.xml`, and one sitemap index level. Normalizes, deduplicates, and keeps same-origin links. No browser
- `crawl_site(url, max_pages=10, max_depth=2, format='markdown', selector=None, include=None, exclude=None, timeout_ms=None, concurrency=4)` - Follow same-origin links breadth first and return each page's content. No browser

### 1.2. CSS selector fast path
- `click_selector(selector, timeout=0.0)` - Click the first CSS match through `evaluate()`. A zero timeout makes one lookup. A positive finite timeout waits for a match.
- `click_selector_and_wait_for_new_tab(selector, timeout=5.0)` - Click a visible `<a target="_blank">` with CDP mouse input. Return its full page target ID after it opens. The helper does not switch tabs. Call `switch(tab_id=target_id)` to focus the new tab.
- `fill_selector(selector, text, timeout=0.0)` - Set a standard `<input>` or `<textarea>` with the native value setter and dispatch bubbling `input` and `change` events. Use `input_text(index, text)` for rich editors and other controls.
- `wait_for_selector(selector, timeout=10.0)` - Wait for a CSS match and return the number of matching elements. The timeout must be positive and finite.

Use these helpers when the page has a stable CSS selector, such as a fixture id. Use `click_selector_and_wait_for_new_tab()` when a blank-target link must open a tab and you need its target ID. Set `rel="opener"` when the new page needs `window.opener`. Use `click_element()` or `click(index=...)` for download links. Ordinary selector actions do not request a browser-state summary.

Search snippets do not count as evidence. Read each scraped page before you cite it. The package caps `max_results` at 20. Set `engine='http'` to prevent browser fallback.

Indexed actions remain preferred whenever an element index exists. For screenshot-derived coordinates, divide screenshot pixels by `window.devicePixelRatio` before calling an `*_xy` function.

**Custom evaluate() Function:**
```python
# Returns values directly, not wrapped in ActionResult
result = await evaluate('''
(function(){
  return Array.from(document.querySelectorAll('.product')).map(p => ({
    name: p.querySelector('.name').textContent,
    price: p.querySelector('.price').textContent
  }))
})()
''')
# result is now a list of dicts, ready to use!
```

**Utilities:**
The agent can just utilize packages like `requests`, `pandas`, `numpy`, `matplotlib`, `BeautifulSoup`, `tabulate`, `csv`, ...

The agent will write code like:

### 1.3. Step 1: Navigate
```python
# Navigate to first page
await navigate(url='https://example.com/products?page=1')
```
### 1.4. Step 2 analyse our DOM state and write code to extract the data we need.

```js extract_products
(function(){
    return Array.from(document.querySelectorAll('.product')).map(p => ({
        name: p.querySelector('.name')?.textContent || '',
        price: p.querySelector('.price')?.textContent || '',
        rating: p.querySelector('.rating')?.textContent || ''
    }))
})()
```

```python
# Extract products using JavaScript
all_products = []
for page in range(1, 6):
    if page > 1:
        await navigate(url=f'https://example.com/products?page={page}')

    products = await evaluate(extract_products)
    all_products.extend(products)
    print(f'Page {page}: Found {len(products)} products')
```

### 1.5. Step 3: Analyse output & save the data to a file
```python
# Save to file
import json
with open('products.json', 'w') as f:
    json.dump(all_products, f, indent=2)

print(f'Total: {len(all_products)} products saved to products.json')
await done(text='Extracted all products', success=True, files_to_display=['products.json'])
```


## 2. Run lifecycle

`code_agent.status` returns a `CodeAgentRunStatus` with `phase`, `terminal`, `success`, `terminal_reason`, `result`, `error`, `attachments`, `step`, `max_steps`, `cancellation_requested`, `closed`, `started_at`, and `finished_at`. Terminal phases are `succeeded`, `failed`, `max_steps`, and `cancelled`.

`code_agent.stop()` cancels between `await` points. It does not interrupt a cell that is already executing; cancel the `run()` task for that. A caller cancellation does not prove that the worker stopped, so inspect `status` and run required cleanup before starting another run.
