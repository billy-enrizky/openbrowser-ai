---
name: web-scraping
description: |
  Extract structured data from websites, scrape page content, and collect information across multiple pages.
  Trigger when the user asks to: extract data from a website, scrape a page, collect information from URLs,
  pull content from web pages, gather data across multiple pages, or download page content.
---

# Web Scraping

Extract structured data from websites using Python code execution with browser automation functions. Handles JavaScript-rendered content, pagination, and multi-page scraping.

All code runs in a persistent namespace via `execute_code`. All browser functions are async -- use `await`.

## Workflow

### Step 1 -- Navigate and get content overview

```python
await navigate('https://example.com/data')

# Get browser state to see page title, URL, element count
state = await browser.get_browser_state_summary()
print(f'Title: {state.title}')
print(f'URL: {state.url}')
print(f'Elements: {len(state.dom_state.selector_map)}')
```

### Step 2 -- Extract data with JavaScript

Use `evaluate()` to run JS in the browser and return structured data directly as Python objects:

```python
data = await evaluate('''
(function(){
  return Array.from(document.querySelectorAll('.product-card')).map(el => ({
    name: el.querySelector('.title')?.textContent?.trim(),
    price: el.querySelector('.price')?.textContent?.trim(),
    url: el.querySelector('a')?.href
  }))
})()
''')

import json
print(json.dumps(data, indent=2))
```

### Step 3 -- Process data with Python

Use pandas, regex, or other Python tools to clean and transform extracted data:

```python
import json

# Filter and transform
filtered = [item for item in data if item.get('price')]
for item in filtered:
    # Extract numeric price
    price_str = item['price'].replace('$', '').replace(',', '')
    item['price_float'] = float(price_str)

# Sort by price
filtered.sort(key=lambda x: x['price_float'])
print(json.dumps(filtered, indent=2))
```

Or with pandas if available:

```python
import pandas as pd
df = pd.DataFrame(data)
print(df.to_string())
```

### Step 4 -- Handle pagination

```python
results = []
page = 1

while True:
    # Extract data from current page
    page_data = await evaluate('''
    (function(){
      return Array.from(document.querySelectorAll('.item')).map(el => ({
        name: el.textContent.trim()
      }))
    })()
    ''')
    results.extend(page_data)
    print(f'Page {page}: {len(page_data)} items')

    # Check for next button
    has_next = await evaluate('''
    (function(){ return !!document.querySelector('.pagination .next:not(.disabled)') })()
    ''')

    if not has_next:
        break

    await click(next_button_index)
    await wait(2)
    page += 1

print(f'Total: {len(results)} items')
```

### Step 5 -- Handle infinite scroll

```python
results = []
prev_count = 0

for _ in range(20):  # Max 20 scroll attempts
    # Get current items
    count = await evaluate('''
    (function(){ return document.querySelectorAll('.item').length })()
    ''')

    if count == prev_count:
        break  # No new content loaded

    prev_count = count
    await scroll(down=True, pages=3)
    await wait(1)

# Now extract all loaded items
results = await evaluate('''
(function(){
  return Array.from(document.querySelectorAll('.item')).map(el => ({
    text: el.textContent.trim()
  }))
})()
''')
print(f'Extracted {len(results)} items')
```

### Step 6 -- Multi-page scraping

```python
urls = [
    'https://example.com/page-1',
    'https://example.com/page-2',
    'https://example.com/page-3',
]

all_data = []
for url in urls:
    await navigate(url)
    await wait(1)

    page_data = await evaluate('''
    (function(){
      return document.querySelector('h1')?.textContent?.trim()
    })()
    ''')
    all_data.append({'url': url, 'title': page_data})
    print(f'{url}: {page_data}')

import json
print(json.dumps(all_data, indent=2))
```

## Tips

- Use `evaluate()` for structured DOM extraction -- it returns Python objects directly.
- Use Python for post-processing: filtering, sorting, deduplication, format conversion.
- For large datasets, process pages incrementally rather than loading everything into memory.
- Check for rate limiting; add `await wait(2)` between page loads if needed.
- Variables persist between `execute_code` calls, so you can build up results across multiple calls.
