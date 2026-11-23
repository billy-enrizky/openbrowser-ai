---
name: file-download
description: |
  Download files from websites, save PDFs, and read downloaded content.
  Trigger when the user asks to: download a file, save a PDF, export a document,
  fetch a file from a URL, grab a report, download and read a PDF, or save page content as a file.
allowed-tools: Bash(openbrowser-ai:*) Bash(curl:*) Bash(uv:*) Bash(irm:*) Read Write
---

# File Download

Download files from websites using the browser's authenticated session. Handles PDFs, CSVs, images, and any downloadable content. Preserves cookies and login sessions for authenticated downloads.

All code runs via `openbrowser-ai -c`. The daemon starts automatically and persists variables across calls. Most browser functions are async -- use `await`. `list_downloads()`, `download_outcomes()`, and `download_failures()` are synchronous.

The CLI daemon persists cookies and login state in the Chrome user-data directory at `~/.config/openbrowser/profiles/daemon`, so authenticated sessions can be reused across later runs. No `storage_state.json` file is written unless you configure one explicitly.

## Setup

Before running, verify openbrowser-ai is installed:

```bash
openbrowser-ai --help
```

If not found, install:

```bash
# macOS/Linux
curl -fsSL https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.ps1 | iex
```

## Workflow

### Step 1 -- Navigate and find download links

```bash
openbrowser-ai -c - <<'EOF'
await navigate("https://example.com/reports")

# Get browser state to find clickable download links
state = await browser.get_browser_state_summary()
for idx, el in state.dom_state.selector_map.items():
    text = el.get_all_children_text(max_depth=1)
    if "download" in text.lower() or "pdf" in text.lower() or "export" in text.lower():
        print(f"[{idx}] {el.tag_name}: {text}")
EOF
```

### Step 2 -- Download a file by URL

Use `download_file()` to download directly. This uses the browser's JavaScript `fetch` internally, preserving cookies and authentication:

```bash
openbrowser-ai -c - <<'EOF'
path = await download_file("https://example.com/reports/annual-report.pdf")
print(f"Saved to: {path}")
EOF
```

With a custom filename:

```bash
openbrowser-ai -c - <<'EOF'
path = await download_file(
    "https://example.com/api/export?format=csv",
    filename="sales-data.csv"
)
print(f"Saved to: {path}")
EOF
```

### Step 3 -- Extract a download URL from the page

When the download URL is not directly visible, extract it from a link or button:

```bash
openbrowser-ai -c - <<'EOF'
# Extract href from a download link
download_url = await evaluate("""
(function(){
  const link = document.querySelector("a[href$=\".pdf\"]");
  return link ? link.href : null;
})()
""")

if download_url:
    path = await download_file(download_url)
    print(f"Downloaded: {path}")
else:
    print("No PDF link found")
EOF
```

### Step 4 -- Read a downloaded PDF

After downloading, use `pypdf` to extract text (requires `pip install openbrowser-ai[pdf]`):

```bash
openbrowser-ai -c - <<'EOF'
from pypdf import PdfReader

reader = PdfReader(path)
print(f"Pages: {len(reader.pages)}")

# Extract text from all pages
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"--- Page {i+1} ---")
    print(text[:500])
EOF
```

### Step 5 -- Read other file types

```bash
openbrowser-ai -c - <<'EOF'
from pathlib import Path

file_path = Path(path)

# CSV
if file_path.suffix == ".csv":
    import pandas as pd
    df = pd.read_csv(file_path)
    print(df.to_string())

# JSON
if file_path.suffix == ".json":
    import json
    data = json.loads(file_path.read_text())
    print(json.dumps(data, indent=2))

# Plain text
if file_path.suffix in (".txt", ".md", ".log"):
    print(file_path.read_text())
EOF
```

### Step 6 -- Download multiple files

```bash
openbrowser-ai -c - <<'EOF'
urls = [
    "https://example.com/report-q1.pdf",
    "https://example.com/report-q2.pdf",
    "https://example.com/report-q3.pdf",
]

paths = []
for url in urls:
    path = await download_file(url)
    paths.append(path)
    print(f"Downloaded: {path}")

print(f"Total files: {len(paths)}")
EOF
```

### Step 7 -- List all downloads

```bash
openbrowser-ai -c - <<'EOF'
files = list_downloads()
for f in files:
    print(f)
print(f"Total: {len(files)} files")
EOF
```

### Step 8 -- Wait for a click-triggered download

When a button or link triggers the download, start `wait_for_download` before the click. The waiter captures the terminal outcome even when the file finishes quickly.

```bash
openbrowser-ai -c - <<'EOF'
import asyncio

waiter = asyncio.create_task(wait_for_download(timeout=30))
await click(download_button_index)
outcome = await waiter

print(outcome["phase"], outcome["path"], outcome["file_size"], outcome["stage"])
if outcome["phase"] == "failed":
    print(outcome["error"])
EOF
```

`wait_for_download` returns `phase` (`completed`, `failed`, or `cancelled`), `path`, `file_size`, `file_type`, `stage`, `error`, and `download_ref`. To inspect earlier attempts, call `download_outcomes()`. To list only failures, call `download_failures()`. A `download_ref` resolves only in the current browser session's live registry; after the session stops or is replaced, read the completed history instead.

### Step 9 -- Download from authenticated pages

`download_file()` preserves the browser's login session on its browser-fetch path. Log in first, then download:

```bash
openbrowser-ai -c - <<'EOF'
# Navigate and log in
await navigate("https://portal.example.com/login")
await input_text(username_index, "user@example.com")
await input_text(password_index, "password")
await click(login_button_index)
await wait(2)

# Now download an authenticated resource
path = await download_file("https://portal.example.com/api/reports/confidential.pdf")
print(f"Downloaded: {path}")
EOF
```

## Tips

- Code is piped via stdin using heredoc (`-c - <<'EOF'`), so all Python syntax works without shell escaping issues.
- Prefer `download_file(url)` when you need a deterministic filename and return path. `navigate()` may auto-download a PDF when `auto_download_pdfs` is enabled, but it does not return the saved path.
- The browser-fetch path preserves cookies and authentication. The `requests` fallback does not send browser cookies, so authenticated downloads can fail after fallback.
- Start `wait_for_download` before a click that triggers a download. A fast download can finish before a wait started afterward.
- Filename conflicts are handled automatically with `(N)` suffix (e.g., `report (1).pdf`).
- If no filename is provided, it is derived from the URL path.
- Use `list_downloads()` to see all files saved in the downloads directory. `download_outcomes()` and `download_failures()` are synchronous.
- For large files, `download_file()` has a 120-second timeout.

## Cleanup

Run this step after the task, whether it succeeded or failed.

**Do not stop a daemon you did not start.** Check `openbrowser-ai daemon status` before the first browser call. If it already reports a running daemon, reuse it and leave it running at the end. If it reports `daemon_not_running`, this run starts one and must clean it up.

For a daemon started by this run:

```bash
openbrowser-ai daemon stop
```

`daemon stop` only acknowledges the request and returns. Poll `openbrowser-ai daemon status` until it reports `daemon_not_running`, up to about 30 seconds. If the daemon does not exit, run `openbrowser-ai daemon force-stop`, which waits and verifies listener, owner, and browser cleanup.

If the task can fail mid-workflow, guarantee cleanup with a shell trap that stops, polls, and force-stops if needed:

```bash
cleanup() {
  openbrowser-ai daemon stop >/dev/null 2>&1 || true
  i=0
  while [ "$i" -lt 30 ]; do
    status=$(openbrowser-ai daemon status 2>&1 || true)
    case "$status" in
      *daemon_not_running*) return 0 ;;
    esac
    sleep 1
    i=$((i + 1))
  done
  openbrowser-ai daemon force-stop >/dev/null 2>&1 || true
}
trap cleanup EXIT
# ... openbrowser-ai -c calls here ...
```

Persist results to disk before stopping the daemon. `daemon reset` clears the Python namespace. Do not call `done()` as a substitute for cleanup; it only marks the task complete inside the agent loop.

Downloaded files in `~/Downloads/openbrowser-daemon/` survive `daemon stop` and `daemon reset`. Persist the exact paths you need before cleanup.
