---
name: page-analysis
description: |
  Analyze web page content, structure, and layout to understand what a page contains and how it is organized.
  Trigger when the user asks to: analyze a page, understand page structure, inspect a website,
  summarize page content, examine page layout, review a web page, or describe what is on a page.
---

# Page Analysis

Analyze and understand web page content, structure, and interactive elements. Produces a comprehensive breakdown of what is on the page and how it is organized.

## Workflow

### Step 1 -- Navigate to the page

Open the target page:

```
browser_navigate(url="https://example.com")
```

Get a compact state overview to confirm the page loaded and see basic metadata:

```
browser_get_state(compact=true)
```

This returns the URL, page title, tab count, and total number of interactive elements.

### Step 2 -- Get full page content

Retrieve the page content as markdown for a human-readable overview:

```
browser_get_text()
```

This captures all visible text, headings, paragraphs, lists, and tables in a structured format.

To include links for further analysis:

```
browser_get_text(extract_links=true)
```

### Step 3 -- Search for specific content

Use `browser_grep` to find specific patterns or sections on the page:

```
browser_grep(pattern="pricing|plans|features", case_insensitive=true, context_lines=3)
```

Search for contact information:

```
browser_grep(pattern="[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", case_insensitive=true)
browser_grep(pattern="\\+?\\d[\\d\\s()-]{7,}", case_insensitive=true)
```

Search for dates:

```
browser_grep(pattern="\\d{4}-\\d{2}-\\d{2}|\\w+ \\d{1,2},? \\d{4}", case_insensitive=true)
```

### Step 4 -- Analyze interactive elements

Get the full list of interactive elements to understand what actions are available:

```
browser_get_state(compact=false)
```

Search for specific types of interactive elements:

```
browser_search_elements(query="button", by="tag")
browser_search_elements(query="input", by="tag")
browser_search_elements(query="a", by="tag", max_results=50)
```

### Step 5 -- Inspect page structure with the accessibility tree

Get the accessibility tree for a semantic view of the page structure:

```
browser_get_accessibility_tree()
```

For a shallow overview of the top-level structure:

```
browser_get_accessibility_tree(max_depth=3)
```

This reveals the heading hierarchy, landmark regions (nav, main, footer), and ARIA roles.

### Step 6 -- Analyze page metadata and technical details

Use `browser_execute_js` to extract metadata not visible in the rendered content:

```
browser_execute_js(expression="(() => { const meta = {}; meta.title = document.title; meta.description = document.querySelector('meta[name=\"description\"]')?.content; meta.canonical = document.querySelector('link[rel=\"canonical\"]')?.href; meta.ogTitle = document.querySelector('meta[property=\"og:title\"]')?.content; meta.ogImage = document.querySelector('meta[property=\"og:image\"]')?.content; meta.lang = document.documentElement.lang; return meta; })()")
```

Check for common frameworks and technologies:

```
browser_execute_js(expression="(() => { const tech = []; if (window.__NEXT_DATA__) tech.push('Next.js'); if (window.__NUXT__) tech.push('Nuxt.js'); if (document.querySelector('[data-reactroot]') || document.querySelector('#__next')) tech.push('React'); if (document.querySelector('[ng-version]')) tech.push('Angular'); if (window.jQuery) tech.push('jQuery'); return tech; })()")
```

### Step 7 -- Analyze page layout sections

Use `browser_find_and_scroll` to navigate to specific sections and analyze them:

```
browser_find_and_scroll(text="Footer")
browser_get_text()
```

Check the page dimensions and scroll height:

```
browser_execute_js(expression="({ viewportWidth: window.innerWidth, viewportHeight: window.innerHeight, scrollHeight: document.body.scrollHeight, scrollWidth: document.body.scrollWidth })")
```

### Step 8 -- Count and categorize content

Use JavaScript to generate a content summary:

```
browser_execute_js(expression="(() => { return { headings: document.querySelectorAll('h1,h2,h3,h4,h5,h6').length, paragraphs: document.querySelectorAll('p').length, images: document.querySelectorAll('img').length, links: document.querySelectorAll('a').length, forms: document.querySelectorAll('form').length, tables: document.querySelectorAll('table').length, lists: document.querySelectorAll('ul,ol').length, buttons: document.querySelectorAll('button,[role=\"button\"]').length, inputs: document.querySelectorAll('input,textarea,select').length }; })()")
```

## Tips

- Start with `browser_get_text` for a quick content overview, then drill down with `browser_grep` for specifics.
- Use `browser_get_accessibility_tree(max_depth=3)` for a high-level structural summary without overwhelming detail.
- Use `browser_execute_js` to extract metadata, detect technologies, and gather statistics not available through text extraction.
- Combine `browser_grep` with regex patterns to find structured data like emails, phone numbers, dates, and prices.
- Use `browser_find_and_scroll` to navigate long pages section by section.
