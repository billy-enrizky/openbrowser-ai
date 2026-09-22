# Context Atlas

Context Atlas helps people find the exact words that answer a question about
the webpage they are reading. Every result stays connected to its source
sentence, so the page can be checked immediately.

## Two ways to search

- **Local** keeps search in the browser. The model downloads once, on first
  use, and prefers WebGPU with a bundled WASM fallback. No model server is
  required.
- **Cloud** uses an access key saved locally in Chrome extension storage. Page text is
  sent only after the user chooses Cloud and starts a search.

The last selection is remembered. New installations start with Local.

## Run the separate localhost page

The localhost page is a companion workspace, not the page being searched. The
extension panel stays on the current webpage while the companion page remains
open in its own tab.

Start the static development server from the repository root:

```bash
uv sync --extra dev
uv run python -m examples.context_atlas.server
```

Open <http://127.0.0.1:8765/>. Then:

1. Load the unpacked extension from `examples/context_atlas/extension/`.
2. Open the webpage you want to search and click the Context Atlas toolbar
   action.
3. Leave the extension panel on that webpage and keep the localhost page open
   in a separate tab.
4. Choose Local or Cloud and ask a question. The localhost page receives the
   latest page source through an exact-origin extension bridge.

The page source is captured automatically. If it is not ready, click
**Refresh from current page** after opening the extension on the webpage.

## Build the extension

```bash
npm --prefix examples/context_atlas/ui install
npm --prefix examples/context_atlas/ui run build
```

The build creates local standalone, extension, offscreen, sandbox, and ONNX
Runtime WASM assets. The package contains no remote executable scripts. Only
the pinned model data is downloaded at runtime.

For Chrome testing:

1. Open `chrome://extensions` and enable **Developer mode**.
2. Choose **Load unpacked** and select `examples/context_atlas/extension/`.
3. Open an ordinary HTTP(S) webpage and click the Context Atlas toolbar action.

The extension reads visible text such as paragraphs, lists, headings, quotes,
preformatted blocks, and captions. It ignores hidden content, navigation and
form controls, Chrome internal pages, PDF viewers, cross-origin frames, and
shadow-root content. Page changes refresh the source and invalidate stale
results.

External access is optional: choosing Cloud requests access to the Cloud API,
and starting Local search for the first time requests access to the pinned
model-data hosts. No external host access is required at installation.

## Source-backed results

Search is bounded to 400 question characters, 160 passages, 2,200 characters
per passage, and 60,000 total source characters. Longer pages are planned with
query-aware windows while retaining page boundaries and nearby context. When
several results are similarly relevant, Context Atlas shows up to five close
matches. Choosing one displays the exact source sentence, preserves provenance,
and highlights that sentence on the webpage.

## Tests and packaging

```bash
npm --prefix examples/context_atlas/ui test
node --test examples/context_atlas/tests/*.test.mjs
OPENBROWSER_HEADLESS=true uv run --extra dev pytest examples/context_atlas/tests -q
uv run python -m examples.context_atlas.package_extension --build
```
