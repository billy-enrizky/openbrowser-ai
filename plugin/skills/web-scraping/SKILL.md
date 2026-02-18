---
name: web-scraping
description: |
  Extract structured data from websites, scrape page content, and collect information across multiple pages.
  Trigger when the user asks to: extract data from a website, scrape a page, collect information from URLs,
  pull content from web pages, gather data across multiple pages, or download page content.
---

# Web Scraping

Extract structured data from websites using browser automation. Handles JavaScript-rendered content, pagination, and multi-tab scraping.

## Workflow

### Step 1 -- Navigate to the target page

Use `browser_navigate` to open the target URL.

```
browser_navigate(url="https://example.com/data")
```

If the page requires interaction before data is visible (e.g., accepting cookies or closing a modal), use `browser_get_state(compact=false)` to find the dismiss button, then `browser_click` to close it.

### Step 2 -- Get a content overview

Use `browser_get_text` to retrieve the full page content as clean markdown. This gives you a quick overview of the page structure and available data.

```
browser_get_text()
```

For pages with links you need to follow, add `extract_links=true` to include href URLs in the output.

```
browser_get_text(extract_links=true)
```

### Step 3 -- Search for specific data

Use `browser_get_text` with the `search` param to find targeted content on the page. Supports regex patterns for flexible matching.

```
browser_get_text(search="Price:\\s*\\$[\\d.]+", case_insensitive=true)
```

Adjust `context_lines` to capture surrounding data:

```
browser_get_text(search="product-name", context_lines=5, max_matches=50)
```

### Step 4 -- Extract structured elements

Use `browser_get_state` with `filter_by`/`filter_query` to find specific DOM elements by text, tag, class, or attribute.

```
browser_get_state(filter_by="class", filter_query="product", max_results=50)
```

For more precise extraction, use `browser_execute_js` to run JavaScript that collects data into a structured format:

```
browser_execute_js(expression="(() => { const items = document.querySelectorAll('.product-card'); return Array.from(items).map(el => ({ name: el.querySelector('.title')?.textContent?.trim(), price: el.querySelector('.price')?.textContent?.trim(), url: el.querySelector('a')?.href })); })()")
```

### Step 5 -- Handle pagination

Check for pagination controls using `browser_get_state`:

```
browser_get_state(filter_by="text", filter_query="Next")
```

Or search for pagination by class:

```
browser_get_state(filter_by="class", filter_query="pagination")
```

Click the next page button using the element index returned:

```
browser_click(index=<next_button_index>)
```

After each page loads, repeat Steps 2-4 to extract data from the new page. Continue until no more pages are available.

### Step 6 -- Multi-tab scraping

For scraping data from multiple URLs, open each in a new tab:

```
browser_navigate(url="https://example.com/page-1", new_tab=true)
browser_navigate(url="https://example.com/page-2", new_tab=true)
```

List open tabs to track progress:

```
browser_tab(action="list")
```

Switch between tabs to extract data from each:

```
browser_tab(action="switch", tab_id="<tab_id>")
browser_get_text()
```

Close tabs when done:

```
browser_tab(action="close", tab_id="<tab_id>")
```

### Step 7 -- Handle infinite scroll pages

Some pages load content dynamically as you scroll. Use a scroll-and-collect loop:

```
browser_scroll(direction="down")
browser_get_text()
```

Use `browser_get_text` with `search` after each scroll to check if new content appeared, and `browser_execute_js` to detect when you have reached the bottom:

```
browser_execute_js(expression="window.innerHeight + window.scrollY >= document.body.scrollHeight")
```

### Step 8 -- Clean up

Close the browser session when scraping is complete:

```
browser_session(action="close_all")
```

## Tips

- Use `browser_get_text` for a quick content overview before targeted extraction.
- Use `browser_get_text` with `search` and regex for pattern-based data extraction (prices, dates, emails).
- Use `browser_execute_js` when you need structured JSON output from complex DOM structures.
- For large datasets, process pages incrementally rather than loading all tabs at once.
- Check for rate limiting or bot detection; add reasonable delays between page loads if needed.
