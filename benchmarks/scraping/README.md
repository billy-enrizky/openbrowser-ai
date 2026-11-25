# Local scraping benchmark

<!-- toc:start -->
## Table of Contents

- [1. How to read these numbers](#1-how-to-read-these-numbers)
- [2. Current results, 2026-11-25, and reference lanes](#2-current-results-2026-11-25-and-reference-lanes)
  - [2.1. OpenBrowser no-browser MCP scrape_web memory lane, 2026-11-25](#21-openbrowser-no-browser-mcp-scrape_web-memory-lane-2026-11-25)
  - [2.2. Firecrawl Local sequential, 2026-11-24, N=10](#22-firecrawl-local-sequential-2026-11-24-n10)
  - [2.3. Firecrawl Local batch, 2026-11-24, N=10](#23-firecrawl-local-batch-2026-11-24-n10)
  - [2.4. Lightpanda no-render latency, 2026-11-24, N=10](#24-lightpanda-no-render-latency-2026-11-24-n10)
- [3. Arms](#3-arms)
- [4. Tasks](#4-tasks)
- [5. Run](#5-run)
- [6. Batch harness](#6-batch-harness)
- [7. Firecrawl lane](#7-firecrawl-lane)
  - [7.1. Install and run](#71-install-and-run)
  - [7.2. Fairness rules](#72-fairness-rules)
  - [7.3. Correctness oracles](#73-correctness-oracles)
  - [7.4. Historical results: 2026-11-23, N=10](#74-historical-results-2026-11-23-n10)
  - [7.5. Historical analysis: 2026-11-23 Firecrawl lane](#75-historical-analysis-2026-11-23-firecrawl-lane)
  - [7.6. Historical hillclimb targets from the 2026-11-23 lane](#76-historical-hillclimb-targets-from-the-2026-11-23-lane)
  - [7.7. Limits for this lane](#77-limits-for-this-lane)
- [8. Limits](#8-limits)
- [9. Historical results: 2026-11-24, interleaved comparison](#9-historical-results-2026-11-24-interleaved-comparison)
  - [9.1. Firecrawl Local](#91-firecrawl-local)
  - [9.2. Firecrawl batch, 2026-11-24](#92-firecrawl-batch-2026-11-24)
  - [9.3. Lightpanda](#93-lightpanda)
  - [9.4. Charts](#94-charts)
- [10. Historical results: 2026-11-24, HTTP-only CLI batch](#10-historical-results-2026-11-24-http-only-cli-batch)
- [11. Historical results: 2026-11-23, network-level render policy](#11-historical-results-2026-11-23-network-level-render-policy)
- [12. Historical results: 2026-11-23, scrape comparison](#12-historical-results-2026-11-23-scrape-comparison)
- [13. Historical analysis: 2026-11-23 results](#13-historical-analysis-2026-11-23-results)
- [14. Historical results: 2026-11-24, native Lightpanda lane](#14-historical-results-2026-11-24-native-lightpanda-lane)
- [15. Historical results: 2026-11-23, fast static path](#15-historical-results-2026-11-23-fast-static-path)
- [16. Historical diagnostic: 2026-11-24, parsed HTML work reduction](#16-historical-diagnostic-2026-11-24-parsed-html-work-reduction)
  - [16.1. Correctness smoke](#161-correctness-smoke)
  - [16.2. Quiet-host comparison pending](#162-quiet-host-comparison-pending)
- [17. Historical results: 2026-11-24, resource lane](#17-historical-results-2026-11-24-resource-lane)
- [18. Historical results: 2026-11-24, final lanes (OpenBrowser 0.1.59.dev18)](#18-historical-results-2026-11-24-final-lanes-openbrowser-0159dev18)
  - [18.1. Final lanes, 2026-11-24, dev18: Sequential scraping against Firecrawl Local](#181-final-lanes-2026-11-24-dev18-sequential-scraping-against-firecrawl-local)
  - [18.2. Final lanes, 2026-11-24, dev18: No-render against Lightpanda](#182-final-lanes-2026-11-24-dev18-no-render-against-lightpanda)
  - [18.3. Final lanes, 2026-11-24, dev18: Firecrawl batch](#183-final-lanes-2026-11-24-dev18-firecrawl-batch)
  - [18.4. Final lanes, 2026-11-24, dev18: One-shot CLI static path](#184-final-lanes-2026-11-24-dev18-one-shot-cli-static-path)
  - [18.5. Final lanes, 2026-11-24, dev18: Resource lane caveat](#185-final-lanes-2026-11-24-dev18-resource-lane-caveat)
- [19. Historical results: 2026-11-24, second final lanes (OpenBrowser 0.1.59.dev28)](#19-historical-results-2026-11-24-second-final-lanes-openbrowser-0159dev28)
  - [19.1. Second final lanes, 2026-11-24, dev28: Sequential scraping against Firecrawl Local](#191-second-final-lanes-2026-11-24-dev28-sequential-scraping-against-firecrawl-local)
  - [19.2. Second final lanes, 2026-11-24, dev28: Headless shell against full Chrome](#192-second-final-lanes-2026-11-24-dev28-headless-shell-against-full-chrome)
  - [19.3. Second final lanes, 2026-11-24, dev28: One-shot CLI static path](#193-second-final-lanes-2026-11-24-dev28-one-shot-cli-static-path)

<!-- toc:end -->

Measure OpenBrowser render modes on deterministic local pages. No external network, cloud browser, or LLM is used.

## 1. How to read these numbers

- Latency is lower is better ↓. Memory is lower is better ↓.
- Latency is in milliseconds. Memory is process-tree RSS in MiB, divided by 1048576.
- The `scrape_web` memory lane uses N=5 for one call and N=3 for 25 concurrent calls. The Firecrawl Local and Lightpanda latency lanes use N=10 per arm and task.
- The memory lane is from 2026-11-25. The Firecrawl Local and Lightpanda reference lanes are from 2026-11-24.
- No AWS or cloud browser was used.
- The brief does not report host load for these scraping lanes. Historical sections state their own recorded load where available.
- Published Lightpanda and Firecrawl figures are non-comparable context, not measurements from this run.

## 2. Current results, 2026-11-25, and reference lanes

The 2026-11-25 MCP memory lane appears first. The Firecrawl Local and Lightpanda lanes are the latest reference runs from 2026-11-24.

### 2.1. OpenBrowser no-browser MCP scrape_web memory lane, 2026-11-25

The `scrape_web` MCP tool runs without Chromium, Blink, or V8. It performs an HTTP fetch and parses the response. It does not run JavaScript, take screenshots, or use browser cookies.

Lower is better for memory ↓.

| Mode | Peak process-tree RSS | Browser processes | Correct results |
| --- | ---: | ---: | ---: |
| One `scrape_web` call | 48.1 MiB median (47.55 to 50.17), N=5 | 0 | 5 of 5 |
| 25 concurrent `scrape_web` calls | 62.4 MiB median (62.31 to 64.30), N=3 | 0 | 25 of 25 in 3 of 3 runs |

Sequential latency was about 77 to 90 ms per call after discovery in this run. The 25 concurrent calls completed in about 155 to 169 ms. Lower is better for latency ↓.

Raw evidence:

- [results-mcp-scrape-web-resource-2026-11-25-n5.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-mcp-scrape-web-resource-2026-11-25-n5.json)

Chart: [native-mcp-http-memory.png](https://github.com/billy-enrizky/openbrowser-ai/blob/main/docs/images/native-mcp-http-memory.png). The chart has been regenerated from the same artifact with these medians and is labeled `Lower is better ↓` for memory. It shows only OpenBrowser process-tree RSS and does not overlay Lightpanda or Firecrawl.

Non-comparable published context, not measured in this run and not a direct win claim:

- Lightpanda: 21.2 MB for one automation page, and 123 MB PSS for 25 parallel crawler processes on Ubuntu AWS.
- Firecrawl Local static: about 155 MiB in an earlier local measurement with a different process boundary.

Do not place these figures on one axis with the OpenBrowser RSS values. Do not claim that one `scrape_web` call uses less memory than Lightpanda's 21.2 MB. Do not claim a direct memory win against Lightpanda or Firecrawl.

Scope rules:

- Our numbers are process-tree RSS in MiB, divided by 1048576. RSS can count shared pages more than once. RSS and PSS are not directly comparable.
- Our 25-call number is one server process that handles 25 concurrent calls. Lightpanda's 123 MB is PSS for 25 separate processes crawling 933 pages on Ubuntu AWS.
- The lane uses one local static fixture, not the live web.
- The Python MCP server process and any observed descendants are included.
- The no-browser mode cannot execute JavaScript, take screenshots, or use browser cookies.
- State our numbers only as OpenBrowser process-tree RSS for this local fixture.

### 2.2. Firecrawl Local sequential, 2026-11-25, N=10

OpenBrowser won 50 of 50 paired samples across five tasks. Each task used 10 pairs. All 150 measured runs passed the complete ordered oracle. HTTP-only tasks do not start a browser. Browser tasks start their browser before the timed sample. A zero-warmup run also passed all 150 correctness checks, but the first static sample per arm pays the HTTP client setup and can lose one pair. The published lane keeps one warmup so both arms compare steady-state task work.

Lower is better for latency ↓.

| Task | OpenBrowser full p50 ms | OpenBrowser lightweight p50 ms | Firecrawl Local p50 ms |
| --- | ---: | ---: | ---: |
| `static_article` | 0.7 | 0.7 | 1.2 |
| `pagination` | 2.8 | 2.8 | 4.3 |
| `table` | 1.1 | 1.1 | 2.9 |
| `js_list` | 13.8 | 15.6 | 552.2 |
| `asset_heavy` | 50.2 | 16.5 | 564.0 |

Chart: [native-scraping-current-p50.png](https://github.com/billy-enrizky/openbrowser-ai/blob/main/docs/images/native-scraping-current-p50.png). The chart is labeled `Lower is better ↓` for latency.

Raw evidence:

- [results-firecrawl-final4-2026-11-25-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-final4-2026-11-25-n10-interleaved.json)

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
| `static_article` | 1.6 | 269.8 |
| `pagination` | 4.0 | 1378.6 |
| `table` | 2.0 | 446.1 |
| `js_list` | 15.9 | 272.9 |
| `asset_heavy` | 16.3 | 236.7 |

All 100 runs passed.

Chart: [native-lightpanda-current-p50.png](https://github.com/billy-enrizky/openbrowser-ai/blob/main/docs/images/native-lightpanda-current-p50.png). The chart is labeled `Lower is better ↓` for latency.

Caveat: Lightpanda starts one process per sample, so its startup dominates static tasks. Use the ordering, not the absolute Lightpanda value. OpenBrowser no-render keeps Blink and V8 and blocks CSS, images, fonts, and media. It is not the same engine as Lightpanda.

Raw evidence:

- [results-lightpanda-final4-2026-11-25-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-lightpanda-final4-2026-11-25-n10-interleaved.json)

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
checked before latency counts. A failed run is a failure, not a fast success.

## 5. Run

```sh
uv run --locked python benchmarks/scraping/scrape_compare.py \
  --samples 3 --warmups 1 \
  --output benchmarks/scraping/results-2026-11-23.json
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

## 7. Firecrawl lane

*This section documents the historical Firecrawl Local 0.1.1 lane. The install, fairness, and oracle rules remain reference material. The results and analysis subsections are historical.*

Compare Firecrawl Local 0.1.1 with OpenBrowser on the same fixtures, browser
binary, correctness oracles, samples, and warmups. Firecrawl Cloud is not used.
There is no API key, no Docker, and no AWS.

Results files:

- [results-firecrawl-2026-11-23-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-2026-11-23-n10.json): 10 samples and 2 warmups per arm and
  task. This was the authoritative run for that historical lane.
- [results-firecrawl-2026-11-23.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-2026-11-23.json): 3 samples and 1 warmup, matching the
  sample count of the earlier OpenBrowser baseline.

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
  --output benchmarks/scraping/results-firecrawl-2026-11-23-n10.json

uv run --locked python benchmarks/scraping/scrape_compare.py \
  --arms openbrowser-full openbrowser-lightweight openbrowser-no-render \
    firecrawl-local firecrawl-local-auto \
  --samples 3 --warmups 1 \
  --firecrawl-bin "$HOME/.local/firecrawl-tools/bin/firecrawl-local" \
  --chrome "$CHROME" \
  --output benchmarks/scraping/results-firecrawl-2026-11-23.json
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
  the bytes each tool returned, not a normalized artifact.

### 7.3. Correctness oracles

The oracle is exact for every task and runs before latency counts.

| Task | Firecrawl markdown oracle | OpenBrowser oracle |
| --- | --- | --- |
| `static_article` | Page title and `Item 49` present, script decoy absent | Same |
| `js_list` | Exact ordered sequence of 30 rendered name plus price pairs, and exactly 30 `Product` matches. Markdown drops `data-sku`, so the card order and count are the proof | 30 `SKU-000` to `SKU-029` rows |
| `pagination` | 5 pages, names 0 to 99 in order, exactly 100 rows | 100 unique row IDs |
| `table` | Exact 100 by 6 cell matrix | 100 rows with 6 cells each |
| `asset_heavy` | 20 distinct `Headline N` values after the wait | 20 headlines |

### 7.4. Historical results: 2026-11-23, N=10

Local macOS 26.6 arm64, Python 3.12.12, OpenBrowser
`0.1.58.dev4+g36ccf113b`, Firecrawl Local 0.1.1, Google Chrome for Testing
159.0.8154.6 for both arms. 2 warmups and 10 measured samples per arm and task.
The host was shared, so read the sub 10 ms rows as order-of-magnitude values.

Lower is better for latency ↓.

| Task | Arm | OK | p50 ms | p95 ms | Output bytes | Requests | Response bytes | Engine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| static_article | firecrawl-local | 10/10 | 2.2 | 2.4 | 1137 | 1.0 | 3631 | local-fetch |
| static_article | firecrawl-local-auto | 10/10 | 1.8 | 2.1 | 1137 | 1.0 | 3631 | local-fetch |
| static_article | openbrowser-full | 10/10 | 11.7 | 13.8 | 1182 | 1.0 | 3631 | http |
| static_article | openbrowser-lightweight | 10/10 | 9.9 | 12.7 | 1182 | 1.0 | 3631 | http |
| static_article | openbrowser-no-render | 10/10 | 9.7 | 12.6 | 1182 | 1.0 | 3631 | http |
| js_list | firecrawl-local | 10/10 | 606.9 | 707.1 | 438 | 3.0 | 2523 | local-browser |
| js_list | firecrawl-local-auto | 10/10 | 616.3 | 691.0 | 438 | 3.0 | 2523 | local-browser |
| js_list | openbrowser-full | 10/10 | 32.5 | 38.6 | 330 | 3.0 | 2523 | browser |
| js_list | openbrowser-lightweight | 10/10 | 43.8 | 73.0 | 330 | 3.0 | 2523 | browser |
| js_list | openbrowser-no-render | 10/10 | 38.7 | 49.6 | 330 | 3.0 | 2523 | browser |
| pagination | firecrawl-local | 10/10 | 9.7 | 11.7 | 2064 | 5.0 | 6923 | local-fetch |
| pagination | firecrawl-local-auto | 10/10 | 8.0 | 8.6 | 2064 | 5.0 | 6923 | local-fetch |
| pagination | openbrowser-full | 10/10 | 41.3 | 44.6 | 1650 | 5.0 | 6923 | http |
| pagination | openbrowser-lightweight | 10/10 | 39.8 | 56.9 | 1650 | 5.0 | 6923 | http |
| pagination | openbrowser-no-render | 10/10 | 38.3 | 85.6 | 1650 | 5.0 | 6923 | http |
| table | firecrawl-local | 10/10 | 4.5 | 4.6 | 6736 | 1.0 | 13067 | local-fetch |
| table | firecrawl-local-auto | 10/10 | 4.0 | 5.5 | 6736 | 1.0 | 13067 | local-fetch |
| table | openbrowser-full | 10/10 | 15.4 | 61.2 | 2490 | 1.0 | 13067 | http |
| table | openbrowser-lightweight | 10/10 | 16.8 | 18.4 | 2490 | 1.0 | 13067 | http |
| table | openbrowser-no-render | 10/10 | 15.6 | 18.6 | 2490 | 1.0 | 13067 | http |
| asset_heavy | firecrawl-local | 10/10 | 682.4 | 784.8 | 1226 | 24.0 | 2899 | local-browser |
| asset_heavy | firecrawl-local-auto | 0/10 | n/a | n/a | 915 | 1.0 | 1165 | local-fetch |
| asset_heavy | openbrowser-full | 10/10 | 70.7 | 129.3 | 242 | 25.0 | 4064 | browser |
| asset_heavy | openbrowser-lightweight | 10/10 | 33.1 | 50.5 | 242 | 4.0 | 2492 | browser |
| asset_heavy | openbrowser-no-render | 10/10 | 37.8 | 44.5 | 242 | 4.0 | 2492 | browser |

### 7.5. Historical analysis: 2026-11-23 Firecrawl lane

1. The result is a split, not a Firecrawl win and not an OpenBrowser win.
2. Firecrawl Local wins the static fetch lanes. It is faster on
   `static_article` (2.2 ms against 9.9 ms for `openbrowser-lightweight`, 4.45x),
   `pagination` (9.7 ms against 39.8 ms, 4.10x), and `table` (4.5 ms against
   16.8 ms, 3.76x). The gap is in the cleaning and request stack, not only in
   OpenBrowser's wrapper.
3. OpenBrowser wins every JavaScript-rendered lane by a wide margin.
   `js_list` is 43.8 ms against 606.9 ms (13.85x) and `asset_heavy` is 33.1 ms
   against 682.4 ms (20.61x). Firecrawl pays a browser launch per scrape.
4. Firecrawl Local's own router fails `asset_heavy` 0/10. It returns the static
   shell with zero headlines and `engine: local-fetch`. `waitFor: 1` moves the
   same request to `local-browser` and the task passes 10/10. A caller has to
   know to force rendering.
5. Lightweight mode cuts the asset page from 24 requests and 2899 response bytes
   on Firecrawl to 4 requests and 2492 response bytes, and returns 242 output
   bytes against 1226. The fixture assets are tiny, so the request count and the
   output size are the meaningful signals, not the byte ratio.
6. OpenBrowser returns a smaller table artifact, 2490 bytes of structured rows
   against 6736 bytes of Firecrawl markdown, but it spends more time producing
   it. The extra time is the structured table extraction, not the fetch.
7. Firecrawl Local cannot return structured JSON locally. For `js_list`,
   `pagination`, and `table` the OpenBrowser arms deliver typed rows while
   Firecrawl delivers markdown that a caller still has to parse.

### 7.6. Historical hillclimb targets from the 2026-11-23 lane

1. Static HTTP path: OpenBrowser spends 9.7 to 11.7 ms and Firecrawl spends
   2.2 ms. Cut the Python cleaning and wrapper overhead to get under 5 ms.
2. Table extraction: OpenBrowser spends about 12 ms over the raw fetch to build
   structured rows. A faster HTML table parser, or markdown output on request,
   would make the table lane competitive.
3. Keep the JavaScript win. The warm browser session and `wait_for` selector are
   worth 13.9x to 20.6x against Firecrawl Local. Do not trade that away for a
   faster cold start.
4. Make rendering intent explicit. Firecrawl Local needed `waitFor: 1` to render
   a page its router classified as static. OpenBrowser's `scrape_page` already
   exposes `engine` and `wait_for`, and the benchmark should keep testing the
   case where a caller asks for rendering.

### 7.7. Limits for this lane

- This is Firecrawl Local 0.1.1, not Firecrawl Cloud. It has no `json` or
  `summary` format, no screenshots without Chrome, no page actions, and
  in-memory crawl jobs. Do not compare these numbers to Cloud latency.
- `serve` refuses loopback and private targets unless `--allow-private` is set.
  The harness sets it for the fixture only.
- Ten samples on a shared host are a smoke-scale result. Sub 10 ms ratios are
  order-of-magnitude values, not stable rankings.
- Local fixtures do not cover anti-bot behavior, real login providers, or live
  sites.
- Do not claim a universal Firecrawl win or a universal OpenBrowser win from
  this lane. Report the split.

## 8. Limits

- Local fixtures do not cover anti-bot pages, real login providers, CAPTCHAs,
  or the full diversity of production sites.
- Three samples are a smoke baseline. Use 30 or more before claiming a
  statistically meaningful difference.
- The HTTP path strips scripts before extraction. A browser page with a
  trivial script but substantial server text correctly uses the HTTP path.
- `no-render` still runs Blink layout internally. It disables screenshots and
  visual-only actions, not layout. Do not claim Lightpanda-level memory or
  startup.
- The latency lanes do not measure peak RSS or CPU. Use the resource lane and
  the 2026-11-25 MCP memory lane for memory results, and state their process
  boundaries before comparing them.

## 9. Historical results: 2026-11-24, interleaved comparison

*Historical run. It is retained for traceability. The current results above supersede it for current claims.*

That historical run compared the OpenBrowser HTTP path with Firecrawl Local and
Lightpanda on the same fixtures. It used 10 measured samples and 2 warmups per
arm and task. The harness alternated arm order for every sample and recorded one
JSONL row per measured sample. All arms passed their oracle on every sample.

Host load at the start was `7.67 / 7.74 / 8.06` on 18 logical CPUs. Load at the
end was `9.64 / 8.23 / 8.23`. This is above the conservative quiet-host gate,
so read the absolute p50 values as shared-host evidence. The alternating order
puts all arms under the same drift. OpenBrowser was `0.1.58.dev56+g8065994cc`,
Firecrawl Local was `0.1.1`, and both used Chrome for Testing `159.0.8154.6`.

### 9.1. Firecrawl Local

Lower is better for latency ↓.

| Task | OpenBrowser full, p50 ms | OpenBrowser lightweight, p50 ms | Firecrawl Local, p50 ms | Result |
| --- | ---: | ---: | ---: | --- |
| `static_article` | 2.09 | 2.20 | 2.29 | Parity, sample ranges overlap |
| `pagination` | 10.4 | 10.8 | 9.7 | Parity, ranges overlap. Firecrawl p50 7% lower |
| `table` | 2.23 | 2.25 | 4.67 | OpenBrowser full 2.09x faster |
| `js_list` | 29.8 | 32.8 | 616.4 | OpenBrowser full 20.7x faster |
| `asset_heavy` | 59.0 | 32.4 | 619.4 | OpenBrowser lightweight 19.1x faster |

In that historical run, OpenBrowser won three of the five local tasks and drew
static article. Pagination was parity because the sample ranges overlap, but
Firecrawl's p50 was 7 percent lower.

A separate profile of the pagination task measured 1.24 ms for the HTTP fetch
and 0.09 ms for Lexbor extraction per page. The extraction is about 7 percent
of the page cost. The 0.7 ms total difference over five pages is request and
Python orchestration overhead, not a parsing gap. See the batch comparison
below for the multi-page surface.

The JavaScript gap is architectural: OpenBrowser keeps one browser per arm,
while Firecrawl Local starts a browser for each JavaScript scrape.
OpenBrowser returns 2490 structured bytes for the table task. Firecrawl returns
6736 markdown bytes. Those outputs are not semantically identical artifacts.

Raw evidence:

- [results-firecrawl-all-2026-11-24-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-all-2026-11-24-n10-interleaved.json)
- [results-firecrawl-all-2026-11-24-n10-interleaved.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-all-2026-11-24-n10-interleaved.jsonl)

### 9.2. Firecrawl batch, 2026-11-24

Firecrawl Local exposes a native `/v2/batch/scrape` job API. The batch
comparison sends the same five pagination URLs to both products and checks
that each returns the same 100 ordered names. The OpenBrowser arm uses
`scrape_pages(engine="http", concurrency=5)` and never starts a browser. The
Firecrawl arm submits one batch job, polls it every 10 ms, and downloads the
results. All 20 measured runs passed their oracle.

Lower is better for latency ↓.

| Arm | Samples | p50 ms | p95 ms | Range ms | Requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenBrowser HTTP batch | 10 | 4.619 | 6.911 | 3.985 to 6.911 | 5 |
| Firecrawl Local batch | 10 | 14.598 | 15.354 | 13.747 to 15.354 | 5 |

OpenBrowser was faster in 10 of 10 pairs. The median paired difference was
`9.966` ms, and OpenBrowser was 3.16 times faster at p50. The arms return
different artifacts: OpenBrowser returns selected row dictionaries, while
Firecrawl returns markdown pages. Both satisfy the same 100-name oracle.
`openbrowser_browser_started` was false for every OpenBrowser batch.

This is the batch surface. The sequential five-request comparison above remains
parity within sample ranges. Do not substitute one result for the other.

Rerun the batch comparison:

```sh
uv run --locked python -m benchmarks.scraping.batch_firecrawl_compare \
  --samples 10 --warmups 1 \
  --firecrawl-bin ~/.local/firecrawl-tools/bin/firecrawl-local \
  --output benchmarks/scraping/results-batch-firecrawl.json
```

Raw evidence:

- [results-batch-firecrawl-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-batch-firecrawl-2026-11-24-n10.json)
- [results-batch-firecrawl-2026-11-24-n10.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-batch-firecrawl-2026-11-24-n10.summary.json)

### 9.3. Lightpanda

Lower is better for latency ↓.

| Task | OpenBrowser no-render, p50 ms | Lightpanda, p50 ms | Result |
| --- | ---: | ---: | --- |
| `static_article` | 2.44 | 166.2 | OpenBrowser 68.1x faster |
| `pagination` | 9.37 | 837.4 | OpenBrowser 89.4x faster |
| `table` | 3.07 | 160.3 | OpenBrowser 52.2x faster |
| `js_list` | 26.8 | 170.6 | OpenBrowser 6.37x faster |
| `asset_heavy` | 31.2 | 159.9 | OpenBrowser 5.12x faster |

That historical run used 10 samples and 2 warmups per arm and task. Host load
at the start was `5.81 / 7.44 / 7.93`. All 100 measured runs passed their
oracle. The static, pagination, and table tasks used the HTTP path. The
JavaScript and asset-heavy tasks used Chromium because they require client-side
rendering.

That table is a historical latency result, not a memory result. Its old
resource-lane rows used a different process boundary from the 2026-11-25 MCP
memory lane. Do not combine those rows with the current memory numbers.
OpenBrowser no-render still runs Blink layout and does not match Lightpanda's
small engine footprint. Do not convert this table into a memory or full-site
compatibility claim.

Raw evidence:

- [results-lightpanda-after-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-lightpanda-after-n10-interleaved.json)
- [results-lightpanda-after-n10-interleaved.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-lightpanda-after-n10-interleaved.jsonl)

### 9.4. Charts

Generate the documentation charts from the raw result files:

```sh
uv run --with matplotlib python benchmarks/scraping/plot_results.py
uv run --with matplotlib python benchmarks/scraping/plot_batch_results.py
uv run --with matplotlib python benchmarks/scraping/plot_resources.py
```

The first script writes two latency charts. Both are labeled `Lower is better ↓`.

- `docs/images/native-scraping-current-p50.png`
- `docs/images/native-lightpanda-current-p50.png`

The batch script writes a latency chart labeled `Lower is better ↓`:

- `docs/images/native-scraping-batch-current.png`

The resource script writes memory and CPU charts labeled `Lower is better ↓`:

- `docs/images/native-scraping-resources.png`

## 10. Historical results: 2026-11-24, HTTP-only CLI batch

*Historical run. It is retained for traceability. The current results above supersede it for current claims.*

The console entry now handles `scrape --engine http` without importing the
Click command tree, browser modules, or the configuration stack. A single URL
uses `scrape_http()`. Multiple URLs use `scrape_http_batch()` with bounded
concurrency. The fast entry accepts only the documented HTTP flags and falls
back to the full Click command for anything else.

The A/B used the pre-change binary and the branch binary with the same five
local pagination URLs:

```sh
openbrowser-ai scrape URL1 URL2 URL3 URL4 URL5 \
  --engine http --format json --selector 'tbody tr' --jsonl
```

Each sample alternated binary order. Peak RSS was sampled every 5 ms while the
child process ran. Load on 18 logical CPUs was `7.491`, `8.508`, and `9.131`
before the run and `9.162`, `8.815`, and `9.226` after it. All 20 runs passed
the 100-row oracle.

Lower is better for latency and memory ↓.

| Binary | p50 ms | p95 ms | Peak RSS p50 MB | Peak RSS max MB |
| --- | ---: | ---: | ---: | ---: |
| Pre-change | 1144.941 | 1177.547 | 153.844 | 154.094 |
| HTTP fast entry | 205.029 | 212.359 | 47.859 | 47.969 |

The new path was faster in 10 of 10 pairs and used less peak RSS in 10 of 10.
The median paired difference was `936.032` ms and `106.000` MB. Both binaries
returned the same five JSONL rows with the same 100 row IDs; only the measured
`duration_ms` and `timing` values differed.

The Lightpanda lane measured `837.4` ms p50 and about `123` MB peak RSS for the
same five-page pagination task. The command surfaces and harnesses differ, so
treat that comparison as directional. The OpenBrowser CLI numbers above are
from this A/B only.

Rerun the A/B:

```sh
uv run --locked python -m benchmarks.scraping.cli_http_batch_compare \
  --old-binary /path/to/pre-change/openbrowser-ai \
  --new-binary /path/to/branch/openbrowser-ai \
  --samples 10 \
  --output benchmarks/scraping/results-cli-http-batch.json
```

Raw evidence:

- [results-cli-http-batch-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-cli-http-batch-2026-11-24-n10.json)
- [results-cli-http-batch-2026-11-24-n10.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-cli-http-batch-2026-11-24-n10.summary.json)

## 11. Historical results: 2026-11-23, network-level render policy

*Historical run. It is retained for traceability. The current results above supersede it for current claims.*

`scrape_compare.py` measures extraction. `no_render_compare.py` measures the
CDP Fetch policy itself, because a blocked resource type must never reach the
fixture server. Each arm launches a fresh headless browser with a temporary
profile and `--use-mock-keychain`, then records server-observed requests and
bytes.

```sh
uv run --locked python benchmarks/scraping/no_render_compare.py --samples 3
```

The command exits nonzero if a blocked type is served, if lightweight stops
loading CSS, or if document, script, XHR, or fetch requests fail.

Smoke run on 2026-11-23, 3 samples, local macOS arm64:

Lower is better for latency, requests, and bytes served ↓. Higher is better for bytes avoided ↑.

| Arm | p50 ms | Requests | Bytes served | Bytes avoided |
| --- | ---: | ---: | ---: | ---: |
| full | 777.6 | 10 | 70,907 | 0 |
| lightweight | 664.0 | 6 | 1,143 | 69,764 |
| no-render | 687.4 | 5 | 972 | 69,935 |

Latency ranges overlap at this sample count, so only the request and byte
counts are deterministic. Do not claim a speed win from this table.

## 12. Historical results: 2026-11-23, scrape comparison

*Historical run. It is retained for traceability. The current results above supersede it for current claims.*

Local macOS arm64, 3 measured samples after 1 warmup per arm and task. These
are smoke numbers for the harness, not a statistical claim.

Lower is better for latency ↓.

| Task | Arm | OK | p50 ms | p95 ms | Output bytes | Requests | Engine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| static_article | openbrowser-full | 3/3 | 7.4 | 7.9 | 1182 | 1 | http |
| static_article | openbrowser-lightweight | 3/3 | 6.8 | 8.3 | 1182 | 1 | http |
| static_article | openbrowser-no-render | 3/3 | 7.8 | 9.5 | 1182 | 1 | http |
| js_list | openbrowser-full | 3/3 | 18.7 | 19.3 | 330 | 3 | browser |
| js_list | openbrowser-lightweight | 3/3 | 17.5 | 46.0 | 330 | 3 | browser |
| js_list | openbrowser-no-render | 3/3 | 19.7 | 24.6 | 330 | 3 | browser |
| pagination | openbrowser-full | 3/3 | 24.7 | 25.1 | 1650 | 5 | http |
| pagination | openbrowser-lightweight | 3/3 | 22.3 | 22.9 | 1650 | 5 | http |
| pagination | openbrowser-no-render | 3/3 | 25.9 | 26.6 | 1650 | 5 | http |
| table | openbrowser-full | 3/3 | 11.8 | 11.8 | 2490 | 1 | http |
| table | openbrowser-lightweight | 3/3 | 11.6 | 11.7 | 2490 | 1 | http |
| table | openbrowser-no-render | 3/3 | 12.5 | 13.0 | 2490 | 1 | http |
| asset_heavy | openbrowser-full | 3/3 | 57.1 | 57.2 | 242 | 25 | browser |
| asset_heavy | openbrowser-lightweight | 3/3 | 17.5 | 19.6 | 242 | 4 | browser |
| asset_heavy | openbrowser-no-render | 3/3 | 20.8 | 25.1 | 242 | 4 | browser |

## 13. Historical analysis: 2026-11-23 results

*Historical analysis of the named run. It does not describe the current results above.*

1. `lightweight` cuts the asset-heavy page from 57.1 ms and 25 requests to
   17.5 ms and 4 requests. Images, web fonts, and media no longer reach the
   page. This is a 3.3x latency win and a 6.3x request reduction on the local
   fixture.
2. `engine=auto` turns the static article, pagination, and table tasks into
   HTTP paths. Static article latency falls from the full-browser baseline
   measured in the earlier harness runs (about 25 to 27 ms) to 6.8 to 7.8 ms.
3. The table JSON output is 2490 bytes. This is the structured artifact the
   caller receives, without a second HTML parse.
5. Pagination still pays one namespace call per page in the main harness.
   The separate batch harness measures `scrape_pages()` against a sequential
   HTTP loop with the same five-page fixture.

## 14. Historical results: 2026-11-24, native Lightpanda lane

*Historical run. It is retained for traceability. The current results above supersede it for current claims.*

Lightpanda 1.0.5 (macOS arm64, SHA-256 `eec3781086ee512320a04b036f092806a8a959944ea78fc8f4fe7dfe215a84b4`), Chrome for Testing `159.0.8154.6`, Firecrawl Local 0.1.1, and OpenBrowser `0.1.58.dev16+g325d7b469`. That historical run used 10 samples per arm and task and 2 warmups. The host load average was `13.6 / 16.3 / 16.4` at start, so treat the absolute p50 values as shared-host evidence, not a quiet-host record.

Lower is better for latency ↓.

| Task | Lightpanda | OpenBrowser no-render | OpenBrowser lightweight | OpenBrowser full | Firecrawl Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_article | 189.1 ms | 12.3 ms | 17.4 ms | 17.9 ms | 2.9 ms |
| js_list | 207.1 ms | 47.5 ms | 42.0 ms | 44.8 ms | 641.8 ms |
| pagination | 935.5 ms | 47.9 ms | 47.2 ms | 46.0 ms | 9.1 ms |
| table | 169.3 ms | 19.8 ms | 22.2 ms | 25.7 ms | 8.0 ms |
| asset_heavy | 164.1 ms | 41.6 ms | 50.0 ms | 82.7 ms | 639.2 ms |

Lightpanda uses `fetch --dump markdown` or `fetch --dump html`, with one process per URL and its default no-subresource policy. The OpenBrowser full arm loaded 25 requests on asset_heavy, lightweight loaded 4, and no-render loaded 2. The task is text extraction, so the no-subresource Lightpanda default is fair for text, but this is not a rendering or screenshot comparison.

In that historical run, OpenBrowser no-render was faster than Lightpanda on all five local tasks and kept Chromium JavaScript and DOM capability. It is not a lighter browser: it still runs Blink layout and uses more memory than Lightpanda by design. The HTTP path is the efficiency path for server-rendered pages. Do not claim Lightpanda-level memory, startup, or full-site compatibility.

Limitations:

- No peak RSS or CPU sampling yet. This lane measures wall clock and fixture requests only.
- Lightpanda and OpenBrowser have different process models. Lightpanda starts one process per URL; OpenBrowser reuses one browser session for a task.
- The shared host was under high load. The relative ordering was stable in a second clean N=10 run. An earlier run was excluded after transient OpenBrowser full-mode CDP closes and a 3 to 4 times Lightpanda outlier.
- Lightpanda does not provide the same layout, paint, extension, or screenshot semantics as Chromium. Keep capability and latency results separate.

Run:

```sh
uv run --locked python benchmarks/scraping/scrape_compare.py \
  --samples 10 --warmups 2 \
  --arms lightpanda openbrowser-full openbrowser-lightweight openbrowser-no-render firecrawl-local \
  --lightpanda-bin "$LIGHTPANDA_BIN" \
  --firecrawl-bin "$FIRECRAWL_LOCAL_BIN" \
  --chrome "$CHROME_EXECUTABLE" \
  --output benchmarks/scraping/results-lightpanda-2026-11-24-n10.json
```

Raw evidence: [results-lightpanda-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-lightpanda-2026-11-24-n10.json).

## 15. Historical results: 2026-11-23, fast static path

*Historical run. It is retained for traceability. The current results above supersede it for current claims.*

The static HTTP path now uses selectolax with the Lexbor parser and one reusable `httpx.AsyncClient` per namespace or scrape batch. Cookies are disabled on that client, and the namespace owner closes it on agent shutdown.

N=10 with 2 warmups, macOS arm64, Firecrawl Local 0.1.1:

Lower is better for latency ↓.

| Task | OpenBrowser arm | Engine | Firecrawl Local | OpenBrowser | Result |
| --- | --- | --- | ---: | ---: | --- |
| static_article | full | http | 1.3 ms | 1.1 ms | parity, p95 1.5 ms for both |
| pagination | full | http | 6.5 ms | 4.9 ms | OpenBrowser 1.33x |
| table | full | http | 3.4 ms | 1.4 ms | OpenBrowser 2.43x |
| js_list | full | browser | 594.0 ms | 16.5 ms | OpenBrowser 36.0x |
| asset_heavy | lightweight | browser | 614.8 ms | 16.3 ms | OpenBrowser 37.7x |

The table path returns 2490 structured bytes against Firecrawl's 6736 markdown bytes. The static result is parity, not a claim of a clear static latency win. The JavaScript tasks remain the largest OpenBrowser advantage.

The raw `httpx` arm still uses BeautifulSoup and markdownify. It is retained as a baseline, not as the product path.

These numbers are historical. They came from OpenBrowser `0.1.58.dev6+gcf873b951`, before commit `6326786ed` changed the fast HTML renderer and HTTP probe. The result file has no host-load value or per-sample records. Do not use it as the current before measurement for the parsed-document change.

## 16. Historical diagnostic: 2026-11-24, parsed HTML work reduction

*Historical diagnostic. It is retained for traceability and is not a current latency claim.*

The HTTP response path now creates one `ParsedHTML` value. The `auto` engine uses its Lexbor tree for the static probe, title, and content. It returns to the browser before title and content extraction when a page is a thin JavaScript shell.

A cProfile diagnostic ran 100 extraction calls per fixture. The old response path called `_parse` 300 times. The new path called it 100 times. On the article fixture, sibling-search generator calls fell from 21,700 to zero. These are work counts, not latency results. An exact comparison against the `728099213` source output matched the title, probe, and extracted content for all three tasks.

The markdown renderer also skips HTML comments and preserves one visible space around them. It keeps adjacent inline nodes joined when the source has no whitespace. This fixes text such as `A <!-- comment --> B` without changing the benchmark fixtures.

### 16.1. Correctness smoke

The smoke run used one measured sample and one warmup. It ran on macOS 26.6 arm64 with Python 3.12.12, OpenBrowser `0.1.58.dev43+g728099213`, Firecrawl Local 0.1.1, and Chrome for Testing 159.0.8154.6. Host load at start was `10.09 / 12.62 / 12.49`. All six task and arm oracles passed. Do not use this run for a latency claim.

| Task | OpenBrowser passed | Firecrawl passed | OpenBrowser output bytes | Firecrawl output bytes |
| --- | ---: | ---: | ---: | ---: |
| `static_article` | 1/1 | 1/1 | 1168 | 1137 |
| `pagination` | 1/1 | 1/1 | 1650 | 2064 |
| `table` | 1/1 | 1/1 | 2490 | 6736 |

OpenBrowser returns structured rows for `pagination` and `table`. Firecrawl returns markdown. Their output byte counts are not equivalent for those tasks.

### 16.2. Quiet-host comparison pending

The N=10 before and after medians are pending. The host load was above 4, so this change makes no latency claim. Before the next N=10 run, check `uptime`. Run only when all three load averages are below 4.

Run the before measurement on the base revision. Run the after measurement on the merged revision. Keep separate summary and JSONL paths for both runs.

```sh
CHROME="$HOME/Library/Caches/ms-playwright/chromium-1255/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
FIRECRAWL_LOCAL_BIN="$HOME/.local/firecrawl-tools/bin/firecrawl-local"

uv run --locked python benchmarks/scraping/scrape_compare.py \
  --arms openbrowser-full firecrawl-local \
  --tasks static_article pagination table \
  --samples 10 --warmups 2 --interleave \
  --firecrawl-bin "$FIRECRAWL_LOCAL_BIN" \
  --chrome "$CHROME" \
  --output benchmarks/scraping/results-firecrawl-static-before-n10.json \
  --raw-jsonl benchmarks/scraping/results-firecrawl-static-before-n10.jsonl
```

For the merged revision, change `before` to `after` in both output paths. The raw JSONL file starts with run metadata, then has one record per measured sample. The run records the load averages, versions, task and arm order, correctness result, output bytes, and errors.

## 17. Historical results: 2026-11-24, resource lane

*Historical resource lane. It is retained for traceability. Its Lightpanda and Firecrawl rows use different process boundaries from the 2026-11-25 MCP memory lane. Do not combine them on one axis.*

`resource_compare.py` runs the same arms and task oracles with a process-tree
sampler. It records peak RSS, live tree CPU, reaped child CPU, and the maximum
child RSS reported by `getrusage`. The sampler adds overhead, so use
`scrape_compare.py` for clean latency claims.

N=3 with 1 warmup per arm and task, macOS arm64, Chrome for Testing
`159.0.8154.6`, Firecrawl Local 0.1.1, Lightpanda 1.0.5. The host load average
was `15.7 / 16.8 / 18.4`, so rely on CPU time and RSS rather than the sampled
latency. The Firecrawl server is started per arm, so its memory is not
attributed to the other arms.

Peak process-tree RSS, p50 MB:

Lower is better for memory ↓.

| Task | OpenBrowser full | OpenBrowser lightweight | OpenBrowser no-render | Firecrawl Local | Lightpanda |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_article | 780.7 | 741.8 | 744.6 | 155.7 | 123.6 |
| js_list | 741.8 | 762.3 | 764.1 | 1411.0 | 124.9 |
| pagination | 740.0 | 765.3 | 766.5 | 1433.4 | 122.9 |
| table | 740.4 | 765.5 | 766.2 | 1384.2 | 123.4 |
| asset_heavy | 749.2 | 783.2 | 777.4 | 1687.0 | 124.5 |

Live process-tree CPU, p50 ms:

Lower is better for CPU time ↓.

| Task | OpenBrowser full | OpenBrowser lightweight | OpenBrowser no-render | Firecrawl Local | Lightpanda |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_article | 46.7 | 41.0 | 41.2 | 32.5 | 60.5 |
| js_list | 133.8 | 100.0 | 108.5 | 380.4 | 61.6 |
| pagination | 44.9 | 44.3 | 44.7 | 74.7 | 203.3 |
| table | 36.4 | 30.8 | 32.2 | 34.4 | 62.6 |
| asset_heavy | 215.8 | 115.4 | 100.6 | 520.6 | 62.7 |

What the lane shows:

- OpenBrowser browser arms use roughly 740 to 783 MB across tasks. Lightweight
  and no-render keep similar memory while cutting CPU on JavaScript and
  asset-heavy pages. No-render uses 100.6 ms on asset-heavy against 215.8 ms
  for full.
- Firecrawl Local uses 155.7 MB on the static article, but grows to 1.38 to
  1.69 GB on JavaScript and asset-heavy pages. Its CPU reaches 380.4 ms on
  `js_list` and 520.6 ms on `asset_heavy`.
- Lightpanda uses about 123 MB, the lowest browser-like footprint here, but its
  pagination CPU is 203.3 ms against 44.9 ms for OpenBrowser full. It is a
  different engine and not a Chromium memory or capability equivalent.
- OpenBrowser's HTTP-only CLI batch is measured in the CLI batch section:
  205.029 ms and 47.859 MB peak RSS for five pages.

Run:

```sh
uv run --locked python -m benchmarks.scraping.resource_compare \
  --samples 3 --warmups 1 \
  --arms openbrowser-full openbrowser-lightweight openbrowser-no-render firecrawl-local lightpanda \
  --firecrawl-bin "$FIRECRAWL_LOCAL_BIN" \
  --chrome "$CHROME_EXECUTABLE" \
  --lightpanda-bin "$LIGHTPANDA_BIN" \
  --output benchmarks/scraping/results-resources-2026-11-24-n3.json
```

Raw evidence: [results-resources-2026-11-24-n3.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-resources-2026-11-24-n3.json).

## 18. Historical results: 2026-11-24, final lanes (OpenBrowser 0.1.59.dev18)

*Historical run. It is retained for traceability. The current results above supersede it for current claims.*

These were the runs behind the comparison page at the time. All use local
fixtures, exact oracles, and alternating arm order.

### 18.1. Final lanes, 2026-11-24, dev18: Sequential scraping against Firecrawl Local

N=10 per arm and task, 1 warmup, interleaved. Host load `5.81, 7.09, 6.60` on
18 logical CPUs. All 30 runs per task passed.

Lower is better for latency ↓.

| Task | OpenBrowser full | OpenBrowser lightweight | Firecrawl Local | Pairs won |
| --- | ---: | ---: | ---: | ---: |
| `static_article` | 0.80 ms | 0.83 ms | 1.33 ms | 10/10 |
| `pagination` | 2.46 ms | 2.57 ms | 4.00 ms | 10/10 |
| `table` | 0.97 ms | 1.02 ms | 3.04 ms | 10/10 |
| `js_list` | 22.4 ms | 24.0 ms | 565.5 ms | 10/10 |
| `asset_heavy` | 41.8 ms | 28.1 ms | 574.1 ms | 10/10 |

Raw evidence: [results-firecrawl-final-2026-11-24-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-final-2026-11-24-n10-interleaved.json)
and the matching `.jsonl`.

### 18.2. Final lanes, 2026-11-24, dev18: No-render against Lightpanda

N=10 per arm and task, 1 warmup, interleaved. Lightpanda `1.0.5`. All 100 runs
passed.

Lower is better for latency ↓.

| Task | OpenBrowser no-render | Lightpanda |
| --- | ---: | ---: |
| `static_article` | 1.8 ms | 425.8 ms |
| `pagination` | 4.5 ms | 2197.7 ms |
| `table` | 2.1 ms | 479.4 ms |
| `js_list` | 19.5 ms | 479.3 ms |
| `asset_heavy` | 23.1 ms | 503.8 ms |

Lightpanda starts one process per sample, so its startup dominates the static
tasks. A prior interleaved run measured it between 160 and 837 ms. Use the
ordering, not the absolute value.

Raw evidence: [results-lightpanda-final-2026-11-24-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-lightpanda-final-2026-11-24-n10-interleaved.json).

### 18.3. Final lanes, 2026-11-24, dev18: Firecrawl batch

N=10, both arms, 10 of 10 pairs. OpenBrowser 2.834 ms p50 against Firecrawl
Local 13.448 ms p50, a 4.75x difference. No browser started.

Raw evidence: [results-batch-firecrawl-final-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-batch-firecrawl-final-2026-11-24-n10.json).

### 18.4. Final lanes, 2026-11-24, dev18: One-shot CLI static path

Five fresh processes per arm, peak child RSS sampled every 5 ms, identical
1169-byte output.

Lower is better for latency and memory ↓.

| Command | p50 latency | p50 peak RSS |
| --- | ---: | ---: |
| Default engine | 223.4 ms | 47.7 MB |
| `--engine http` | 213.5 ms | 47.6 MB |

Run it with:

```sh
uv run --no-project python -m benchmarks.scraping.cli_static_resource_compare \
  --binary "$(which openbrowser-ai)" --samples 5 \
  --output benchmarks/scraping/results/results-cli-static-final-2026-11-24-n5.json
```

Raw evidence: [results-cli-static-final-2026-11-24-n5.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-cli-static-final-2026-11-24-n5.json).

### 18.5. Final lanes, 2026-11-24, dev18: Resource lane caveat

`resource_compare.py` starts a browser for every arm before it runs any task.
The OpenBrowser static rows therefore include a Chromium process that the real
static path never starts. Read those rows as a persistent browser session. Use
the one-shot CLI table above for static scraping memory.

## 19. Historical results: 2026-11-24, second final lanes (OpenBrowser 0.1.59.dev28)

*Historical run. It is retained for traceability. The current results above supersede it for current claims.*

That historical rerun followed the sync HTTP client, main-content filters,
render asset flags, headless shell, and cold-start work. Load was
`5.40, 5.36, 5.11`.

### 19.1. Second final lanes, 2026-11-24, dev28: Sequential scraping against Firecrawl Local

N=10, 1 warmup, interleaved. All 30 runs per task passed.

Lower is better for latency ↓.

| Task | OpenBrowser full | Lightweight | Firecrawl Local | Pairs won |
| --- | ---: | ---: | ---: | ---: |
| `static_article` | 0.92 ms | 0.94 ms | 1.54 ms | 10/10 |
| `pagination` | 2.93 ms | 3.02 ms | 4.42 ms | 10/10 |
| `table` | 0.86 ms | 0.88 ms | 2.70 ms | 10/10 |
| `js_list` | 18.00 ms | 18.33 ms | 557.65 ms | 10/10 |
| `asset_heavy` | 43.46 ms | 21.66 ms | 575.66 ms | 10/10 |

Raw evidence: [results-firecrawl-final2-2026-11-24-n10-interleaved.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-firecrawl-final2-2026-11-24-n10-interleaved.json).

### 19.2. Second final lanes, 2026-11-24, dev28: Headless shell against full Chrome

N=6 per arm, order alternated, peak process-tree RSS every 20 ms.

Lower is better for latency and memory ↓.

| Task | Shell | Full Chrome | Saved |
| --- | ---: | ---: | ---: |
| `js_list` | 940.1 ms, 355.4 MB | 1296.0 ms, 742.4 MB | 387.0 MB |
| `asset_heavy` | 974.7 ms, 381.6 MB | 1274.3 ms, 754.8 MB | 373.1 MB |

Run it with:

```sh
uv run --no-project python -m benchmarks.scraping.headless_shell_ab 6 \
  benchmarks/scraping/results/results-headless-shell-final2-2026-11-24-n6.json
```

Raw evidence: [results-headless-shell-final2-2026-11-24-n6.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-headless-shell-final2-2026-11-24-n6.json).

### 19.3. Second final lanes, 2026-11-24, dev28: One-shot CLI static path

Five fresh processes per arm, peak child RSS every 5 ms, identical 1169-byte output.

Lower is better for latency and memory ↓.

| Command | p50 latency | p50 peak RSS |
| --- | ---: | ---: |
| Default engine | 148.6 ms | 47.7 MB |
| `--engine http` | 142.7 ms | 47.6 MB |

Raw evidence: [results-cli-static-final2-2026-11-24-n5.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/scraping/results/results-cli-static-final2-2026-11-24-n5.json).
