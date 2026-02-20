# Code-Use Mode

Code-Use Mode is a Notebook-like code execution system for browser automation. Instead of the agent choosing from a predefined set of actions, the LLM writes Python code that gets executed in a persistent namespace with all browser control functions available.

## Problem Solved

**Code-Use Mode solves this** by giving the agent a Python execution environment where it can:
- Store extracted data in variables
- Loop through pages programmatically
- Combine results from multiple extractions
- Process and filter data before saving
- Use conditional logic to decide what to do next
- Output more tokens than the LLM writes

### Namespace
The namespace is initialized with:

**Browser Control Functions:**
- `navigate(url, new_tab=False)` - Navigate to a URL
- `go_back()` - Go back to the previous page
- `wait(seconds=3)` - Wait for specified seconds
- `click(index)` - Click an element by index
- `input_text(index, text, clear=True)` - Type text into an input field
- `scroll(down=True, pages=1.0, index=None)` - Scroll the page or a container
- `send_keys(keys)` - Send keyboard keys or shortcuts
- `upload_file(index, path)` - Upload a file to a file input element
- `select_dropdown(index, text)` - Select an option in a `<select>` dropdown
- `dropdown_options(index)` - Get all options for a `<select>` dropdown
- `switch(tab_id)` - Switch to a different browser tab
- `close(tab_id)` - Close a browser tab
- `evaluate(code, variables={})` - Execute JavaScript and return Python objects
- `get_selector_from_index(index)` - Get CSS selector for an element by index
- `done(text, success)` - Mark task complete

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

### Step 1: Navigate
```python
# Navigate to first page
await navigate(url='https://example.com/products?page=1')
```
### Step 2 analyse our DOM state and write code to extract the data we need.

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

### Step 3: Analyse output & save the data to a file
```python
# Save to file
import json
with open('products.json', 'w') as f:
    json.dump(all_products, f, indent=2)

print(f'Total: {len(all_products)} products saved to products.json')
await done(text='Extracted all products', success=True, files_to_display=['products.json'])
```
