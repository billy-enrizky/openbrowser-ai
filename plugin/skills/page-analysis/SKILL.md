---
name: page-analysis
description: |
  Analyze web page content, structure, and layout to understand what a page contains and how it is organized.
  Trigger when the user asks to: analyze a page, understand page structure, inspect a website,
  summarize page content, examine page layout, review a web page, or describe what is on a page.
---

# Page Analysis

Analyze and understand web page content, structure, and interactive elements using Python code execution. Produces a comprehensive breakdown of what is on the page and how it is organized.

All code runs in a persistent namespace via `execute_code`. All browser functions are async -- use `await`.

## Workflow

### Step 1 -- Navigate and get overview

```python
await navigate('https://example.com')
state = await browser.get_browser_state_summary()
print(f'Title: {state.title}')
print(f'URL: {state.url}')
print(f'Interactive elements: {len(state.dom_state.selector_map)}')
print(f'Tabs: {len(state.tabs)}')
```

### Step 2 -- Extract page metadata

```python
meta = await evaluate('''
(function(){
  return {
    title: document.title,
    description: document.querySelector('meta[name="description"]')?.content,
    canonical: document.querySelector('link[rel="canonical"]')?.href,
    ogTitle: document.querySelector('meta[property="og:title"]')?.content,
    ogImage: document.querySelector('meta[property="og:image"]')?.content,
    lang: document.documentElement.lang,
    charset: document.characterSet
  };
})()
''')

import json
print(json.dumps(meta, indent=2))
```

### Step 3 -- Detect frameworks and technologies

```python
tech = await evaluate('''
(function(){
  const t = [];
  if (window.__NEXT_DATA__) t.push('Next.js');
  if (window.__NUXT__) t.push('Nuxt.js');
  if (document.querySelector('[data-reactroot]') || document.querySelector('#__next')) t.push('React');
  if (document.querySelector('[ng-version]')) t.push('Angular');
  if (window.jQuery) t.push('jQuery');
  if (window.Vue) t.push('Vue.js');
  if (document.querySelector('[data-svelte]')) t.push('Svelte');
  return t;
})()
''')
print(f'Technologies detected: {tech}')
```

### Step 4 -- Content summary and statistics

```python
stats = await evaluate('''
(function(){
  return {
    headings: document.querySelectorAll('h1,h2,h3,h4,h5,h6').length,
    paragraphs: document.querySelectorAll('p').length,
    images: document.querySelectorAll('img').length,
    links: document.querySelectorAll('a').length,
    forms: document.querySelectorAll('form').length,
    tables: document.querySelectorAll('table').length,
    lists: document.querySelectorAll('ul,ol').length,
    buttons: document.querySelectorAll('button,[role="button"]').length,
    inputs: document.querySelectorAll('input,textarea,select').length,
    iframes: document.querySelectorAll('iframe').length,
    scripts: document.querySelectorAll('script').length,
    stylesheets: document.querySelectorAll('link[rel="stylesheet"]').length
  };
})()
''')

import json
print('Content statistics:')
print(json.dumps(stats, indent=2))
```

### Step 5 -- Analyze heading structure

```python
headings = await evaluate('''
(function(){
  return Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6')).map(h => ({
    tag: h.tagName,
    text: h.textContent.trim().substring(0, 80)
  }));
})()
''')

for h in headings:
    indent = '  ' * (int(h['tag'][1]) - 1)
    print(f'{indent}{h["tag"]}: {h["text"]}')
```

### Step 6 -- Analyze interactive elements

```python
state = await browser.get_browser_state_summary()
elements_by_tag = {}
for idx, el in state.dom_state.selector_map.items():
    tag = el.tag_name
    elements_by_tag.setdefault(tag, []).append({
        'index': idx,
        'text': el.get_all_children_text(max_depth=1)[:50],
        'type': el.attributes.get('type', ''),
        'href': el.attributes.get('href', '')[:50] if el.attributes.get('href') else '',
    })

for tag, elems in sorted(elements_by_tag.items()):
    print(f'\n{tag} ({len(elems)} elements):')
    for e in elems[:5]:
        print(f'  [{e["index"]}] text="{e["text"]}" type={e["type"]} href={e["href"]}')
    if len(elems) > 5:
        print(f'  ... and {len(elems) - 5} more')
```

### Step 7 -- Page dimensions and scroll analysis

```python
dims = await evaluate('''
(function(){
  return {
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
    scrollHeight: document.body.scrollHeight,
    scrollWidth: document.body.scrollWidth,
    scrollable: document.body.scrollHeight > window.innerHeight
  };
})()
''')

import json
print(json.dumps(dims, indent=2))
if dims['scrollable']:
    pages = dims['scrollHeight'] / dims['viewportHeight']
    print(f'Page is approximately {pages:.1f} viewport heights long')
```

### Step 8 -- Search for specific content patterns

```python
import re

# Get page text for Python-side analysis
text_content = await evaluate('document.body.innerText')

# Find emails
emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text_content)
print(f'Emails found: {emails}')

# Find phone numbers
phones = re.findall(r'\+?\d[\d\s()-]{7,}', text_content)
print(f'Phone numbers found: {phones}')

# Find dates
dates = re.findall(r'\d{4}-\d{2}-\d{2}|\w+ \d{1,2},? \d{4}', text_content)
print(f'Dates found: {dates}')
```

## Tips

- Start with `evaluate()` for metadata and DOM statistics -- gives a fast structured overview.
- Use `browser.get_browser_state_summary()` for interactive element analysis.
- Use Python regex on extracted text for pattern matching (emails, phones, dates, prices).
- For long pages, use `await scroll(down=True)` and re-extract to analyze below-fold content.
- Variables persist between calls, so you can build a comprehensive analysis incrementally.
