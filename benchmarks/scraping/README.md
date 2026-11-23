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
5. Pagination still pays one namespace call per page. A future
   `scrape_pages()` batch helper should remove that overhead.

## Firecrawl lane

The harness is ready for a Firecrawl arm, but Firecrawl Local is not installed
on this machine. A fair Firecrawl run must pin the Firecrawl Local crate or the
self-host tag, run on the same host, use the same fixtures, disable the cache,
and use the same correctness checks. Do not compare Firecrawl Cloud latency
with local OpenBrowser latency.

When a Firecrawl Local or self-host endpoint is available, add an adapter that
posts to `/v2/scrape` with `formats: ["markdown"]` and `maxAge: 0` and feeds the
result into the same `TaskResult` path.

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
