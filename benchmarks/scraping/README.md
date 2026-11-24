# Local scraping benchmark

Measure OpenBrowser render modes and an HTTP baseline on deterministic local
pages. No external network, cloud browser, or LLM is used.

## Arms

| Arm | Engine | Render mode |
| --- | --- | --- |
| `http` | `httpx` plus BeautifulSoup and markdownify | none |
| `openbrowser-full` | Chromium, `engine=auto` | full |
| `openbrowser-lightweight` | Chromium, `engine=auto` | lightweight |
| `openbrowser-no-render` | Chromium, `engine=auto` | no-render |
| `firecrawl-local` | Firecrawl Local 0.1.1, native per-task wait | not applicable |
| `firecrawl-local-auto` | Firecrawl Local 0.1.1, no `waitFor` override | not applicable |

`engine=auto` uses HTTP when the server-rendered text is substantial, and
falls back to Chromium when the page is a thin JavaScript shell. The
`http` arm never runs JavaScript.

## Tasks

| Task | What it checks |
| --- | --- |
| `static_article` | 50 article rows, title, links, and a script decoy that must not appear |
| `js_list` | 30 SKU cards rendered by JavaScript after an API fetch |
| `pagination` | 5 server-rendered pages, 100 unique row IDs |
| `table` | 100 rows by 6 columns |
| `asset_heavy` | 20 images, 2 stylesheets, a web font, and 20 JavaScript-rendered headlines |

The fixture server records every request and response body size. Correctness is
checked before latency counts. A failed run is a failure, not a fast success.

## Run

```sh
uv run --locked python benchmarks/scraping/scrape_compare.py \
  --samples 3 --warmups 1 \
  --output benchmarks/scraping/results-2026-11-23.json
```

For a longer run, use `--samples 30`. Report p50 and p95 with confidence
intervals before making a claim.

## Network-level render policy

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

| Arm | p50 ms | Requests | Bytes served | Bytes avoided |
| --- | ---: | ---: | ---: | ---: |
| full | 777.6 | 10 | 70,907 | 0 |
| lightweight | 664.0 | 6 | 1,143 | 69,764 |
| no-render | 687.4 | 5 | 972 | 69,935 |

Latency ranges overlap at this sample count, so only the request and byte
counts are deterministic. Do not claim a speed win from this table.

## Results, 2026-11-23

Local macOS arm64, 3 measured samples after 1 warmup per arm and task. These
are smoke numbers for the harness, not a statistical claim.

| Task | Arm | OK | p50 ms | p95 ms | Output bytes | Requests | Engine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| static_article | http | 3/3 | 3.8 | 4.7 | 1182 | 1 | http |
| static_article | openbrowser-full | 3/3 | 7.4 | 7.9 | 1182 | 1 | http |
| static_article | openbrowser-lightweight | 3/3 | 6.8 | 8.3 | 1182 | 1 | http |
| static_article | openbrowser-no-render | 3/3 | 7.8 | 9.5 | 1182 | 1 | http |
| js_list | http | 0/3 | n/a | n/a | 432 | 1 | http |
| js_list | openbrowser-full | 3/3 | 18.7 | 19.3 | 330 | 3 | browser |
| js_list | openbrowser-lightweight | 3/3 | 17.5 | 46.0 | 330 | 3 | browser |
| js_list | openbrowser-no-render | 3/3 | 19.7 | 24.6 | 330 | 3 | browser |
| pagination | http | 3/3 | 2.8 | 2.8 | 1650 | 5 | http |
| pagination | openbrowser-full | 3/3 | 24.7 | 25.1 | 1650 | 5 | http |
| pagination | openbrowser-lightweight | 3/3 | 22.3 | 22.9 | 1650 | 5 | http |
| pagination | openbrowser-no-render | 3/3 | 25.9 | 26.6 | 1650 | 5 | http |
| table | http | 3/3 | 0.7 | 0.7 | 13067 | 1 | http |
| table | openbrowser-full | 3/3 | 11.8 | 11.8 | 2490 | 1 | http |
| table | openbrowser-lightweight | 3/3 | 11.6 | 11.7 | 2490 | 1 | http |
| table | openbrowser-no-render | 3/3 | 12.5 | 13.0 | 2490 | 1 | http |
| asset_heavy | http | 0/3 | n/a | n/a | 1165 | 1 | http |
| asset_heavy | openbrowser-full | 3/3 | 57.1 | 57.2 | 242 | 25 | browser |
| asset_heavy | openbrowser-lightweight | 3/3 | 17.5 | 19.6 | 242 | 4 | browser |
| asset_heavy | openbrowser-no-render | 3/3 | 20.8 | 25.1 | 242 | 4 | browser |

## What this run shows

1. `lightweight` cuts the asset-heavy page from 57.1 ms and 25 requests to
   17.5 ms and 4 requests. Images, web fonts, and media no longer reach the
   page. This is a 3.3x latency win and a 6.3x request reduction on the local
   fixture.
2. `engine=auto` turns the static article, pagination, and table tasks into
   HTTP paths. Static article latency falls from the full-browser baseline
   measured in the earlier harness runs (about 25 to 27 ms) to 6.8 to 7.8 ms.
3. The HTTP arm fails `js_list` and `asset_heavy`, as expected. The browser
   path is still required for JavaScript-rendered content.
4. The table JSON output is 2490 bytes against 13067 bytes for the HTTP raw
   HTML path, a 5.2x output reduction. Smaller output means fewer tokens and
   lower downstream cost.
5. Pagination still pays one namespace call per page in the main harness.
   The separate batch harness measures `scrape_pages()` against a sequential
   HTTP loop with the same five-page fixture.

## Batch harness

```sh
uv run --locked python benchmarks/scraping/batch_compare.py \
  --samples 10 --warmups 1 --concurrency 4 \
  --output /tmp/openbrowser-batch-results.json
```

The batch harness serves five deterministic pages with 20 rows each. Both
arms return the same 100 row IDs. It reports p50, p95, request count, response
bytes, and output bytes for a sequential HTTP loop and for `scrape_pages`.

## Firecrawl lane

Compare Firecrawl Local 0.1.1 with OpenBrowser on the same fixtures, browser
binary, correctness oracles, samples, and warmups. Firecrawl Cloud is not used.
There is no API key, no Docker, and no AWS.

Results files:

- `results-firecrawl-2026-11-23-n10.json`: 10 samples and 2 warmups per arm and
  task. This is the authoritative run.
- `results-firecrawl-2026-11-23.json`: 3 samples and 1 warmup, matching the
  sample count of the earlier OpenBrowser baseline.

Recorded paths replace the user home directory with `$HOME`.

### Install and run

```sh
cargo install firecrawl-local --version 0.1.1 --locked \
  --root "$HOME/.local/firecrawl-tools"
"$HOME/.local/firecrawl-tools/bin/firecrawl-local" --version
# firecrawl-local 0.1.1

CHROME="$HOME/Library/Caches/ms-playwright/chromium-1255/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"

uv run --locked python benchmarks/scraping/scrape_compare.py \
  --arms openbrowser-full openbrowser-lightweight openbrowser-no-render http \
    firecrawl-local firecrawl-local-auto \
  --samples 10 --warmups 2 \
  --firecrawl-bin "$HOME/.local/firecrawl-tools/bin/firecrawl-local" \
  --chrome "$CHROME" \
  --output benchmarks/scraping/results-firecrawl-2026-11-23-n10.json

uv run --locked python benchmarks/scraping/scrape_compare.py \
  --arms openbrowser-full openbrowser-lightweight openbrowser-no-render http \
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

### Fairness rules

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

### Correctness oracles

The oracle is exact for every task and runs before latency counts.

| Task | Firecrawl markdown oracle | OpenBrowser oracle |
| --- | --- | --- |
| `static_article` | Page title and `Item 49` present, script decoy absent | Same |
| `js_list` | Exact ordered sequence of 30 rendered name plus price pairs, and exactly 30 `Product` matches. Markdown drops `data-sku`, so the card order and count are the proof | 30 `SKU-000` to `SKU-029` rows |
| `pagination` | 5 pages, names 0 to 99 in order, exactly 100 rows | 100 unique row IDs |
| `table` | Exact 100 by 6 cell matrix | 100 rows with 6 cells each |
| `asset_heavy` | 20 distinct `Headline N` values after the wait | 20 headlines |

### Results, 2026-11-23, N=10

Local macOS 26.6 arm64, Python 3.12.12, OpenBrowser
`0.1.58.dev4+g36ccf113b`, Firecrawl Local 0.1.1, Google Chrome for Testing
159.0.8154.6 for both arms. 2 warmups and 10 measured samples per arm and task.
The host was shared, so read the sub 10 ms rows as order-of-magnitude values.

| Task | Arm | OK | p50 ms | p95 ms | Output bytes | Requests | Response bytes | Engine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| static_article | firecrawl-local | 10/10 | 2.2 | 2.4 | 1137 | 1.0 | 3631 | local-fetch |
| static_article | firecrawl-local-auto | 10/10 | 1.8 | 2.1 | 1137 | 1.0 | 3631 | local-fetch |
| static_article | openbrowser-full | 10/10 | 11.7 | 13.8 | 1182 | 1.0 | 3631 | http |
| static_article | openbrowser-lightweight | 10/10 | 9.9 | 12.7 | 1182 | 1.0 | 3631 | http |
| static_article | openbrowser-no-render | 10/10 | 9.7 | 12.6 | 1182 | 1.0 | 3631 | http |
| static_article | http | 10/10 | 8.0 | 12.5 | 1182 | 1.0 | 3631 | http |
| js_list | firecrawl-local | 10/10 | 606.9 | 707.1 | 438 | 3.0 | 2523 | local-browser |
| js_list | firecrawl-local-auto | 10/10 | 616.3 | 691.0 | 438 | 3.0 | 2523 | local-browser |
| js_list | openbrowser-full | 10/10 | 32.5 | 38.6 | 330 | 3.0 | 2523 | browser |
| js_list | openbrowser-lightweight | 10/10 | 43.8 | 73.0 | 330 | 3.0 | 2523 | browser |
| js_list | openbrowser-no-render | 10/10 | 38.7 | 49.6 | 330 | 3.0 | 2523 | browser |
| js_list | http | 0/10 | n/a | n/a | 432 | 1.0 | 432 | http |
| pagination | firecrawl-local | 10/10 | 9.7 | 11.7 | 2064 | 5.0 | 6923 | local-fetch |
| pagination | firecrawl-local-auto | 10/10 | 8.0 | 8.6 | 2064 | 5.0 | 6923 | local-fetch |
| pagination | openbrowser-full | 10/10 | 41.3 | 44.6 | 1650 | 5.0 | 6923 | http |
| pagination | openbrowser-lightweight | 10/10 | 39.8 | 56.9 | 1650 | 5.0 | 6923 | http |
| pagination | openbrowser-no-render | 10/10 | 38.3 | 85.6 | 1650 | 5.0 | 6923 | http |
| pagination | http | 10/10 | 8.3 | 19.6 | 1650 | 5.0 | 6923 | http |
| table | firecrawl-local | 10/10 | 4.5 | 4.6 | 6736 | 1.0 | 13067 | local-fetch |
| table | firecrawl-local-auto | 10/10 | 4.0 | 5.5 | 6736 | 1.0 | 13067 | local-fetch |
| table | openbrowser-full | 10/10 | 15.4 | 61.2 | 2490 | 1.0 | 13067 | http |
| table | openbrowser-lightweight | 10/10 | 16.8 | 18.4 | 2490 | 1.0 | 13067 | http |
| table | openbrowser-no-render | 10/10 | 15.6 | 18.6 | 2490 | 1.0 | 13067 | http |
| table | http | 10/10 | 3.2 | 5.6 | 13067 | 1.0 | 13067 | http |
| asset_heavy | firecrawl-local | 10/10 | 682.4 | 784.8 | 1226 | 24.0 | 2899 | local-browser |
| asset_heavy | firecrawl-local-auto | 0/10 | n/a | n/a | 915 | 1.0 | 1165 | local-fetch |
| asset_heavy | openbrowser-full | 10/10 | 70.7 | 129.3 | 242 | 25.0 | 4064 | browser |
| asset_heavy | openbrowser-lightweight | 10/10 | 33.1 | 50.5 | 242 | 4.0 | 2492 | browser |
| asset_heavy | openbrowser-no-render | 10/10 | 37.8 | 44.5 | 242 | 4.0 | 2492 | browser |
| asset_heavy | http | 0/10 | n/a | n/a | 1165 | 1.0 | 1165 | http |

### What the lane shows

1. The result is a split, not a Firecrawl win and not an OpenBrowser win.
2. Firecrawl Local wins the static fetch lanes. It is faster on
   `static_article` (2.2 ms against 9.9 ms for `openbrowser-lightweight`, 4.45x),
   `pagination` (9.7 ms against 39.8 ms, 4.10x), and `table` (4.5 ms against
   16.8 ms, 3.76x). It also beats the raw `http` arm on static pages and
   pagination, which puts the gap in the cleaning and request stack, not only in
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
   it. The raw `http` arm returns 13067 bytes of HTML in 3.2 ms, so the extra
   OpenBrowser time is the structured table extraction, not the fetch.
7. Firecrawl Local cannot return structured JSON locally. For `js_list`,
   `pagination`, and `table` the OpenBrowser arms deliver typed rows while
   Firecrawl delivers markdown that a caller still has to parse.

### Hillclimb targets from this lane

1. Static HTTP path: OpenBrowser spends 9.7 to 11.7 ms where the raw `http` arm
   spends 8.0 ms and Firecrawl spends 2.2 ms. Cut the Python cleaning and
   wrapper overhead to get under 5 ms.
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

### Limits for this lane

- This is Firecrawl Local 0.1.1, not Firecrawl Cloud. It has no `json` or
  `summary` format, no screenshots without Chrome, no page actions, and
  in-memory crawl jobs. Do not compare these numbers to Cloud latency.
- `serve` refuses loopback and private targets unless `--allow-private` is set.
  The harness sets it for the fixture only.
- Ten samples on a shared host are a smoke-scale result. The raw `http` baseline
  moved between 0.7 ms and 8.3 ms p50 across runs on this host, so sub 10 ms
  ratios are order-of-magnitude values.
- Local fixtures do not cover anti-bot behavior, real login providers, or live
  sites.
- Do not claim a universal Firecrawl win or a universal OpenBrowser win from
  this lane. Report the split.

## Limits

- Local fixtures do not cover anti-bot pages, real login providers, CAPTCHAs,
  or the full diversity of production sites.
- Three samples are a smoke baseline. Use 30 or more before claiming a
  statistically meaningful difference.
- The `http` arm strips scripts with BeautifulSoup. A browser page with a
  trivial script but substantial server text correctly uses the HTTP path.
- `no-render` still runs Blink layout internally. It disables screenshots and
  visual-only actions, not layout. Do not claim Lightpanda-level memory or
  startup.
- The benchmark does not yet measure peak RSS or CPU. Add cgroup or process
  sampling before calling a result an efficiency win.

## Native Lightpanda lane, 2026-11-24

Lightpanda 1.0.5 (macOS arm64, SHA-256 `eec3781086ee512320a04b036f092806a8a959944ea78fc8f4fe7dfe215a84b4`), Chrome for Testing `159.0.8154.6`, Firecrawl Local 0.1.1, and OpenBrowser `0.1.58.dev16+g325d7b469`. The run used 10 samples per arm and task and 2 warmups. The host load average was `13.6 / 16.3 / 16.4` at start, so treat the absolute p50 values as shared-host evidence, not a quiet-host record.

| Task | Lightpanda | OpenBrowser no-render | OpenBrowser lightweight | OpenBrowser full | Firecrawl Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_article | 189.1 ms | 12.3 ms | 17.4 ms | 17.9 ms | 2.9 ms |
| js_list | 207.1 ms | 47.5 ms | 42.0 ms | 44.8 ms | 641.8 ms |
| pagination | 935.5 ms | 47.9 ms | 47.2 ms | 46.0 ms | 9.1 ms |
| table | 169.3 ms | 19.8 ms | 22.2 ms | 25.7 ms | 8.0 ms |
| asset_heavy | 164.1 ms | 41.6 ms | 50.0 ms | 82.7 ms | 639.2 ms |

The raw `http` arm was 5.8 ms on static_article, 3.8 ms on pagination, and 1.1 ms on table. It failed `js_list` and `asset_heavy` because those pages need JavaScript.

Lightpanda uses `fetch --dump markdown` or `fetch --dump html`, with one process per URL and its default no-subresource policy. The OpenBrowser full arm loaded 25 requests on asset_heavy, lightweight loaded 4, and no-render loaded 2. The task is text extraction, so the no-subresource Lightpanda default is fair for text, but this is not a rendering or screenshot comparison.

OpenBrowser no-render is faster than Lightpanda on all five local tasks and keeps Chromium JavaScript and DOM capability. It is not a lighter browser: it still runs Blink layout and uses more memory than Lightpanda by design. The HTTP path is the efficiency path for server-rendered pages. Do not claim Lightpanda-level memory, startup, or full-site compatibility.

Limitations:

- No peak RSS or CPU sampling yet. This lane measures wall clock and fixture requests only.
- Lightpanda and OpenBrowser have different process models. Lightpanda starts one process per URL; OpenBrowser reuses one browser session for a task.
- The shared host was under high load. The relative ordering was stable in a second clean N=10 run. An earlier run was excluded after transient OpenBrowser full-mode CDP closes and a 3 to 4 times Lightpanda outlier.
- Lightpanda does not provide the same layout, paint, extension, or screenshot semantics as Chromium. Keep capability and latency results separate.

Run:

```sh
uv run --locked python benchmarks/scraping/scrape_compare.py \
  --samples 10 --warmups 2 \
  --arms lightpanda http openbrowser-full openbrowser-lightweight openbrowser-no-render firecrawl-local \
  --lightpanda-bin "$LIGHTPANDA_BIN" \
  --firecrawl-bin "$FIRECRAWL_LOCAL_BIN" \
  --chrome "$CHROME_EXECUTABLE" \
  --output benchmarks/scraping/results-lightpanda-2026-11-24-n10.json
```

Raw evidence: `results-lightpanda-2026-11-24-n10.json`.

## Fast static path, 2026-11-23

The static HTTP path now uses selectolax with the Lexbor parser and one reusable `httpx.AsyncClient` per namespace or scrape batch. Cookies are disabled on that client, and the namespace owner closes it on agent shutdown.

N=10 with 2 warmups, macOS arm64, Firecrawl Local 0.1.1:

| Task | OpenBrowser arm | Engine | Firecrawl Local | OpenBrowser | Result |
| --- | --- | --- | ---: | ---: | --- |
| static_article | full | http | 1.3 ms | 1.1 ms | parity, p95 1.5 ms for both |
| pagination | full | http | 6.5 ms | 4.9 ms | OpenBrowser 1.33x |
| table | full | http | 3.4 ms | 1.4 ms | OpenBrowser 2.43x |
| js_list | full | browser | 594.0 ms | 16.5 ms | OpenBrowser 36.0x |
| asset_heavy | lightweight | browser | 614.8 ms | 16.3 ms | OpenBrowser 37.7x |

The table path returns 2490 structured bytes against Firecrawl's 6736 markdown bytes. The static result is parity, not a claim of a clear static latency win. The JavaScript tasks remain the largest OpenBrowser advantage.

The raw `httpx` arm still uses BeautifulSoup and markdownify. It is retained as a baseline, not as the product path.

## Resource lane, 2026-11-24

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

| Task | HTTP | OpenBrowser full | OpenBrowser lightweight | OpenBrowser no-render | Firecrawl Local | Lightpanda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_article | 51.6 | 780.7 | 741.8 | 744.6 | 155.7 | 123.6 |
| js_list | n/a | 741.8 | 762.3 | 764.1 | 1411.0 | 124.9 |
| pagination | 51.7 | 740.0 | 765.3 | 766.5 | 1433.4 | 122.9 |
| table | 51.8 | 740.4 | 765.5 | 766.2 | 1384.2 | 123.4 |
| asset_heavy | n/a | 749.2 | 783.2 | 777.4 | 1687.0 | 124.5 |

Live process-tree CPU, p50 ms:

| Task | HTTP | OpenBrowser full | OpenBrowser lightweight | OpenBrowser no-render | Firecrawl Local | Lightpanda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_article | 41.0 | 46.7 | 41.0 | 41.2 | 32.5 | 60.5 |
| js_list | n/a | 133.8 | 100.0 | 108.5 | 380.4 | 61.6 |
| pagination | 31.4 | 44.9 | 44.3 | 44.7 | 74.7 | 203.3 |
| table | 33.8 | 36.4 | 30.8 | 32.2 | 34.4 | 62.6 |
| asset_heavy | n/a | 215.8 | 115.4 | 100.6 | 520.6 | 62.7 |

What the lane shows:

- The HTTP path is the memory and CPU winner for server-rendered pages at
  about 52 MB, but it fails the JavaScript tasks by design.
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
- The HTTP arm's `n/a` rows are expected failures, not missing measurements.

Run:

```sh
uv run --locked python -m benchmarks.scraping.resource_compare \
  --samples 3 --warmups 1 \
  --arms http openbrowser-full openbrowser-lightweight openbrowser-no-render firecrawl-local lightpanda \
  --firecrawl-bin "$FIRECRAWL_LOCAL_BIN" \
  --chrome "$CHROME_EXECUTABLE" \
  --lightpanda-bin "$LIGHTPANDA_BIN" \
  --output benchmarks/scraping/results-resources-2026-11-24-n3.json
```

Raw evidence: `results-resources-2026-11-24-n3.json`.
