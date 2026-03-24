---
name: file-download
description: |
  Download files from websites, save PDFs, and read downloaded content.
  Trigger when the user asks to: download a file, save a PDF, export a document,
  fetch a file from a URL, grab a report, download and read a PDF, or save page content as a file.
---

# File Download

Download files from websites using the browser's authenticated session. Handles PDFs, CSVs, images, and any downloadable content. Preserves cookies and login sessions for authenticated downloads.

All code runs in a persistent namespace via `execute_code`. All browser functions are async -- use `await`.

## Workflow

### Step 1 -- Navigate and find download links

```python
await navigate('https://example.com/reports')

# Get browser state to find clickable download links
state = await browser.get_browser_state_summary()
for idx, el in state.dom_state.selector_map.items():
    text = el.get_all_children_text(max_depth=1)
    if 'download' in text.lower() or 'pdf' in text.lower() or 'export' in text.lower():
        print(f'[{idx}] {el.tag_name}: {text}')
```

### Step 2 -- Download a file by URL

Use `download_file()` to download directly. This uses the browser's JavaScript `fetch` internally, preserving cookies and authentication:

```python
path = await download_file('https://example.com/reports/annual-report.pdf')
print(f'Saved to: {path}')
```

With a custom filename:

```python
path = await download_file(
    'https://example.com/api/export?format=csv',
    filename='sales-data.csv'
)
print(f'Saved to: {path}')
```

### Step 3 -- Extract a download URL from the page

When the download URL is not directly visible, extract it from a link or button:

```python
# Extract href from a download link
download_url = await evaluate('''
(function(){
  const link = document.querySelector('a[href$=".pdf"]');
  return link ? link.href : null;
})()
''')

if download_url:
    path = await download_file(download_url)
    print(f'Downloaded: {path}')
else:
    print('No PDF link found')
```

### Step 4 -- Read a downloaded PDF

After downloading, use `pypdf` (pre-installed) to extract text:

```python
from pypdf import PdfReader

reader = PdfReader(path)
print(f'Pages: {len(reader.pages)}')

# Extract text from all pages
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f'--- Page {i+1} ---')
    print(text[:500])
```

### Step 5 -- Read other file types

```python
from pathlib import Path

file_path = Path(path)

# CSV
if file_path.suffix == '.csv':
    import pandas as pd
    df = pd.read_csv(file_path)
    print(df.to_string())

# JSON
if file_path.suffix == '.json':
    import json
    data = json.loads(file_path.read_text())
    print(json.dumps(data, indent=2))

# Plain text
if file_path.suffix in ('.txt', '.md', '.log'):
    print(file_path.read_text())
```

### Step 6 -- Download multiple files

```python
urls = [
    'https://example.com/report-q1.pdf',
    'https://example.com/report-q2.pdf',
    'https://example.com/report-q3.pdf',
]

paths = []
for url in urls:
    path = await download_file(url)
    paths.append(path)
    print(f'Downloaded: {path}')

print(f'Total files: {len(paths)}')
```

### Step 7 -- List all downloads

```python
files = list_downloads()
for f in files:
    print(f)
print(f'Total: {len(files)} files')
```

### Step 8 -- Download from authenticated pages

`download_file()` preserves the browser's login session. Log in first, then download:

```python
# Navigate and log in
await navigate('https://portal.example.com/login')
await input_text(username_index, 'user@example.com')
await input_text(password_index, 'password')
await click(login_button_index)
await wait(2)

# Now download an authenticated resource
path = await download_file('https://portal.example.com/api/reports/confidential.pdf')
print(f'Downloaded: {path}')
```

## Tips

- Use `download_file(url)` instead of `navigate(url)` for files. `navigate()` opens PDFs in the browser viewer but does not save them.
- `download_file()` preserves cookies and authentication -- no need to re-authenticate.
- Filename conflicts are handled automatically with `(N)` suffix (e.g., `report (1).pdf`).
- If no filename is provided, it is derived from the URL path.
- Use `list_downloads()` to see all files saved in the downloads directory.
- For large files, `download_file()` has a 120-second timeout.
- The fallback strategy uses Python `requests` if the browser fetch fails (e.g., CORS restrictions), but without browser cookies.
