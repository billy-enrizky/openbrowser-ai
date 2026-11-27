# Local scraping benchmark

<!-- toc:start -->
## Contents

- [1. How to read these numbers](#1-how-to-read-these-numbers)
- [2. Current results, 2026-11-26, and reference lanes](#2-current-results-2026-11-26-and-reference-lanes)
  - [2.1. OpenBrowser no-browser MCP scrape_web memory lane, 2026-11-26](#21-openbrowser-no-browser-mcp-scrape_web-memory-lane-2026-11-26)
  - [2.2. Firecrawl Local sequential, 2026-11-25, N=10](#22-firecrawl-local-sequential-2026-11-25-n10)
  - [2.3. Firecrawl Local batch, 2026-11-24, N=10](#23-firecrawl-local-batch-2026-11-24-n10)
  - [2.4. Lightpanda no-render latency, 2026-11-25, N=10](#24-lightpanda-no-render-latency-2026-11-25-n10)
- [3. Arms](#3-arms)
- [4. Tasks](#4-tasks)
- [5. Run](#5-run)
- [6. Batch harness](#6-batch-harness)
- [7. Firecrawl lane reference procedure](#7-firecrawl-lane-reference-procedure)
  - [7.1. Install and run](#71-install-and-run)
  - [7.2. Fairness rules](#72-fairness-rules)
  - [7.3. Correctness oracles](#73-correctness-oracles)
  - [7.4. Limits for this lane](#74-limits-for-this-lane)
- [8. Limits](#8-limits)

<!-- toc:end -->

Measure OpenBrowser render modes on deterministic local pages. No external network, cloud browser, or LLM is used.

## 1. How to read these numbers

- Latency is lower is better ↓. Memory is lower is better ↓.
- Latency is in milliseconds. Memory is process-tree RSS in MiB, divided by 1048576.
- The `scrape_web` memory lane uses N=5 for both one call and 25 concurrent calls. The Firecrawl Local and Lightpanda latency lanes use N=10 per arm and task.
- The memory lane is from 2026-11-26. The Firecrawl Local and Lightpanda reference lanes are from 2026-11-24.
- No AWS or cloud browser was used.
- The brief does not report host load for these scraping lanes. Historical sections state their own recorded load where available.
- Published Lightpanda and Firecrawl figures provide context and come from other measurement scopes.

## 2. Current results, 2026-11-26, and reference lanes

The 2026-11-26 MCP memory lane appears first. The Firecrawl Local and Lightpanda lanes are the latest reference runs from 2026-11-24.

### 2.1. OpenBrowser no-browser MCP scrape_web memory lane, 2026-11-26

The `scrape_web` MCP tool runs without Chromium, Blink, or V8. It performs an HTTP fetch and parses the response. It does not run JavaScript, take screenshots, or use browser cookies.

Lower is better for memory ↓.

| Mode | Peak process-tree RSS | Browser processes observed | Correct results |
| --- | ---: | ---: | ---: |
| One `scrape_web` call | 46.188 MiB median (45.859 to 51.641, N=5) | 0 | 5 of 5 |
| 25 concurrent `scrape_web` calls | 63.188 MiB median (60.781 to 65.250, N=5) | 0 | 25 of 25 in 5 of 5 runs |

Each of the ten observations started a fresh MCP server. All 130 calls passed the exact content, truncation, length, and HTTP-engine checks. The sampler observes the MCP server and its descendants every 5 ms. Short-lived peaks can occur between samples.

Raw evidence:

- [results-mcp-scrape-web-resource-2026-11-26-n5.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-mcp-scrape-web-resource-2026-11-26-n5.json)
- [provenance](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-mcp-scrape-web-resource-2026-11-26-n5.provenance.json)

Chart: [native-mcp-http-memory-2026-11-26.png](https://github.com/billy-enrizky/openbrowser-ai/blob/main/docs/images/native-mcp-http-memory-2026-11-26.png). The chart has been regenerated from the same artifact with these medians and is labeled `Lower is better ↓` for memory. It shows only OpenBrowser process-tree RSS and does not overlay Lightpanda or Firecrawl.

Published context from other measurement scopes:

- Lightpanda: 21.2 MB for one automation page, and 123 MB PSS for 25 parallel crawler processes on Ubuntu AWS.
- Firecrawl Local static: about 155 MiB in an earlier local measurement with a different process boundary.

Do not place these figures on one axis with the OpenBrowser RSS values. Do not claim that one `scrape_web` call uses less memory than Lightpanda's 21.2 MB. Do not claim a direct memory win against Lightpanda or Firecrawl.

Scope rules:

- Our numbers are process-tree RSS in MiB, divided by 1048576. RSS can count shared pages more than once. RSS and PSS are not directly comparable.
- Our 25-call number is one server process that handles 25 concurrent calls. Lightpanda's 123 MB is PSS for 25 separate processes crawling 933 pages on Ubuntu AWS.
- The lane uses one local static fixture.
- The Python MCP server process and any observed descendants are included.
- The no-browser mode cannot execute JavaScript, take screenshots, or use browser cookies.
- State our numbers only as OpenBrowser process-tree RSS for this local fixture.

### 2.2. Firecrawl Local sequential, 2026-11-25, N=10

OpenBrowser lightweight won 50 of 50 paired samples across five tasks. The full arm won 49 of 50, with one static sample 0.13 ms slower. Each task used 10 pairs. All 150 measured runs passed the complete ordered oracle. HTTP-only tasks do not start a browser. Browser tasks start their browser before the timed sample. A zero-warmup run also passed all 150 correctness checks, but the first static sample per arm pays the HTTP client setup and can lose one pair. The published lane keeps one warmup so both arms compare steady-state task work.

Lower is better for latency ↓.

| Task | OpenBrowser full p50 ms | OpenBrowser lightweight p50 ms | Firecrawl Local p50 ms |
| --- | ---: | ---: | ---: |
| `static_article` | 0.8 | 0.7 | 1.2 |
| `pagination` | 2.8 | 2.9 | 4.4 |
| `table` | 1.0 | 1.1 | 2.9 |
| `js_list` | 13.1 | 15.1 | 556.4 |
| `asset_heavy` | 56.8 | 17.1 | 570.7 |

Chart: [native-scraping-current-p50.png](https://github.com/billy-enrizky/openbrowser-ai/blob/main/docs/images/native-scraping-current-p50.png). The chart is labeled `Lower is better ↓` for latency.

Raw evidence:

- [results-firecrawl-final6-2026-11-25-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-final6-2026-11-25-n10-interleaved.json)
- [results-firecrawl-final6-2026-11-25-n10-interleaved.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-final6-2026-11-25-n10-interleaved.jsonl)

### 2.3. Firecrawl Local batch, 2026-11-24, N=10

Lower is better for latency ↓.

| Arm | p50 ms |
| --- | ---: |
| OpenBrowser no-browser batch | 2.834 |
| Firecrawl Local batch | 13.448 |

OpenBrowser won 10 of 10 pairs and was 4.75 times faster at p50. Raw evidence:

- [results-batch-firecrawl-final-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-batch-firecrawl-final-2026-11-24-n10.json)

### 2.4. Lightpanda no-render latency, 2026-11-25, N=10

Lower is better for latency ↓.

| Task | OpenBrowser no-render p50 ms | Lightpanda p50 ms |
| --- | ---: | ---: |
| `static_article` | 1.9 | 236.7 |
| `pagination` | 4.9 | 1160.3 |
| `table` | 3.3 | 228.5 |
| `js_list` | 14.5 | 209.9 |
| `asset_heavy` | 16.1 | 219.5 |

All 100 runs passed.

Chart: [native-lightpanda-current-p50.png](https://github.com/billy-enrizky/openbrowser-ai/blob/main/docs/images/native-lightpanda-current-p50.png). The chart is labeled `Lower is better ↓` for latency.

Caveat: Lightpanda starts one process per sample, so its startup dominates static tasks. Use the ordering across arms. Read the Lightpanda absolute value within its per-sample startup scope. OpenBrowser no-render keeps Blink and V8 and blocks CSS, images, fonts, and media. Lightpanda uses its own engine.

Raw evidence:

- [results-lightpanda-final6-2026-11-25-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-lightpanda-final6-2026-11-25-n10-interleaved.json)
- [results-lightpanda-final6-2026-11-25-n10-interleaved.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-lightpanda-final6-2026-11-25-n10-interleaved.jsonl)

## 3. Arms

| Arm | Engine | Render mode |
| --- | --- | --- |
| `openbrowser-full` | Chromium, `engine=auto` | full |
| `openbrowser-lightweight` | Chromium, `engine=auto` | lightweight |
| `openbrowser-no-render` | Chromium, `engine=auto` | no-render |
| `firecrawl-local` | Firecrawl Local 0.1.1, native per-task wait | not applicable |
| `firecrawl-local-auto` | Firecrawl Local 0.1.1, no `waitFor` override | not applicable |

`engine=auto` uses HTTP when the server-rendered text is substantial, and
falls back to Chromium when the page is a thin JavaScript shell.

## 4. Tasks

| Task | What it checks |
| --- | --- |
| `static_article` | 50 article rows, title, links, and a script decoy that must not appear |
| `js_list` | 30 SKU cards rendered by JavaScript after an API fetch |
| `pagination` | 5 server-rendered pages, 100 unique row IDs |
| `table` | 100 rows by 6 columns |
| `asset_heavy` | 20 images, 2 stylesheets, a web font, and 20 JavaScript-rendered headlines |

The fixture server records every request and response body size. Correctness is
checked before latency counts. A failed run counts as a failure.

## 5. Run

```sh
uv run --locked python benchmarks/scraping/scrape_compare.py \
  --samples 3 --warmups 1 \
  --output benchmarks/scraping/results-firecrawl-local-n3.json
```

For a longer run, use `--samples 30`. Report p50 and p95 with confidence
intervals before making a claim.

Use `--interleave` to alternate arm order for each sample. This mode starts each
arm once and keeps it running until the task set finishes. Use
`--raw-jsonl <path>` to write a run record and one record per measured sample.
Warmup samples are not included in the JSONL file. The summary JSON remains at
`--output`.

## 6. Batch harness

```sh
uv run --locked python benchmarks/scraping/batch_compare.py \
  --samples 10 --warmups 1 --concurrency 4 \
  --output /tmp/openbrowser-batch-results.json
```

The batch harness serves five deterministic pages with 20 rows each. Both
arms return the same 100 row IDs. It reports p50, p95, request count, response
bytes, and output bytes for a sequential HTTP loop and for `scrape_pages`.

## 7. Firecrawl lane reference procedure

*This section keeps the install, fairness, and oracle rules for the Firecrawl Local lane. Current results are in section 2.2 and section 2.3.*

Compare Firecrawl Local 0.1.1 with OpenBrowser on the same fixtures, browser
binary, correctness oracles, samples, and warmups. Firecrawl Cloud is not used.
There is no API key, no Docker, and no AWS.

Recorded paths replace the user home directory with `$HOME`.

### 7.1. Install and run

```sh
cargo install firecrawl-local --version 0.1.1 --locked \
  --root "$HOME/.local/firecrawl-tools"
"$HOME/.local/firecrawl-tools/bin/firecrawl-local" --version
# firecrawl-local 0.1.1

CHROME="$HOME/Library/Caches/ms-playwright/chromium-1255/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"

uv run --locked python benchmarks/scraping/scrape_compare.py \
  --arms openbrowser-full openbrowser-lightweight openbrowser-no-render \
    firecrawl-local firecrawl-local-auto \
  --samples 10 --warmups 2 \
  --firecrawl-bin "$HOME/.local/firecrawl-tools/bin/firecrawl-local" \
  --chrome "$CHROME" \
  --output benchmarks/scraping/results-firecrawl-local.json

uv run --locked python benchmarks/scraping/scrape_compare.py \
  --arms openbrowser-full openbrowser-lightweight openbrowser-no-render \
    firecrawl-local firecrawl-local-auto \
  --samples 3 --warmups 1 \
  --firecrawl-bin "$HOME/.local/firecrawl-tools/bin/firecrawl-local" \
  --chrome "$CHROME" \
  --output benchmarks/scraping/results-firecrawl-local-n3.json
```

The harness starts `firecrawl-local serve` on a free loopback port with
`FIRECRAWL_CLOUD=off` and these flags:

```sh
firecrawl-local serve --port <free port> --allow-private --browser auto
```

`--allow-private` is required because the fixture runs on loopback and
Firecrawl Local refuses private targets by default. To use an existing server
instead, pass `--firecrawl-url http://127.0.0.1:3002` and start it yourself with
the same environment and flags.

The Firecrawl adapter posts the documented v2 body:

```json
{"url": "<fixture URL>", "formats": ["markdown"], "maxAge": 0, "storeInCache": false}
```

No cache hit can hide fetch or render work, and no cloud fallback can serve a
blocked page.

### 7.2. Fairness rules

- One browser binary for both arms. `--chrome` sets
  `BrowserProfile.executable_path` for OpenBrowser and `FIRECRAWL_CHROME_PATH`
  for Firecrawl Local. The recorded run used Google Chrome for Testing
  159.0.8154.6, the Playwright chromium-1255 build.
- Headless only. OpenBrowser launches add `--use-mock-keychain`. Firecrawl
  Local already passes that flag. With those flags, neither launch reads or
  writes the macOS login keychain, and no benchmark step resets or modifies the
  Keychain.
- Firecrawl Local runs with `FIRECRAWL_CLOUD=off`, `maxAge: 0`, and
  `storeInCache: false`. OpenBrowser arms run with default extensions disabled
  and a fresh in-process fixture.
- The fair Firecrawl arm passes `waitFor: 1` for `asset_heavy` only. Without a
  request-level wait, Firecrawl Local's router chose the static fetch for that
  page and returned zero JavaScript-rendered headlines. One millisecond forces
  the local browser and adds no measurable sleep. The `firecrawl-local-auto`
  arm keeps the product default and exists to show that behavior.
- Each arm runs its warmups and samples in sequence. The recorded run used 2
  warmups and 10 measured samples per arm and task.
- Firecrawl Local behaves like it launches a browser for each JavaScript
  scrape. Every measured JavaScript run stays near 600 ms after warmup, and the
  `metadata.engine` field reports `local-browser` on every run. OpenBrowser keeps
  one browser per arm. That is a product architecture difference, measured as it
  behaves.
- Firecrawl Local skips the `json` format and ignores `actions`. Every Firecrawl
  task therefore returns markdown, while the OpenBrowser arms can return
  structured JSON rows for `js_list`, `pagination`, and `table`. Output bytes are
  the raw bytes each tool returned.

### 7.3. Correctness oracles

The oracle is exact for every task and runs before latency counts.

| Task | Firecrawl markdown oracle | OpenBrowser oracle |
| --- | --- | --- |
| `static_article` | All `Item 0` to `Item 49` in order, script decoy absent | Same |
| `js_list` | Exact ordered sequence of 30 rendered name plus price pairs, and exactly 30 `Product` matches. Markdown drops `data-sku`, so the card order and count are the proof | 30 `SKU-000` to `SKU-029` rows |
| `pagination` | Complete ordered 100 row names across five pages | Complete ordered 100 row IDs across five pages |
| `table` | Exact ordered 100 by 6 cell matrix | Exact ordered 100 by 6 cell matrix including every row ID |
| `asset_heavy` | `Headline 1` to `Headline 20` in order, each exactly once | Same |

### 7.4. Limits for this lane

- This is Firecrawl Local 0.1.1. It has no `json` or
  `summary` format, no screenshots without Chrome, no page actions, and
  in-memory crawl jobs. Do not compare these numbers to Cloud latency.
- `serve` refuses loopback and private targets unless `--allow-private` is set.
  The harness sets it for the fixture only.
- Ten samples on a shared host are a smoke-scale result. Sub 10 ms ratios are
  order-of-magnitude values.
- Local fixtures do not cover anti-bot behavior, real login providers, or live
  sites.
- Report the split on each task. Scope every claim to the measured task.

## 8. Limits

- Local fixtures do not cover anti-bot pages, real login providers, CAPTCHAs,
  or the full diversity of production sites.
- Three samples are a smoke baseline. Use 30 or more before claiming a
  statistically meaningful difference.
- The HTTP path strips scripts before extraction. A browser page with a
  trivial script but substantial server text correctly uses the HTTP path.
- `no-render` still runs Blink layout internally. It disables screenshots and
  visual-only actions while Blink layout continues. Do not claim Lightpanda-level
  memory or startup.
- The latency lanes do not measure peak RSS or CPU. Use the resource lane and
  the 2026-11-26 MCP memory lane for memory results, and state their process
  boundaries before comparing them.
