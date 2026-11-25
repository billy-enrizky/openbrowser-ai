# Interactive browser benchmark

<!-- toc:start -->
## Table of Contents

- [1. How to read these numbers](#1-how-to-read-these-numbers)
- [2. Current results, 2026-11-25, rotated N=5](#2-current-results-2026-11-25-rotated-n5)
  - [2.1. Cold interactive](#21-cold-interactive)
  - [2.2. Warm interactive](#22-warm-interactive)
  - [2.3. Remaining measured gap](#23-remaining-measured-gap)
- [3. Interactive resource pilot, 2026-11-25](#3-interactive-resource-pilot-2026-11-25)
- [4. Arms](#4-arms)
- [5. Tasks](#5-tasks)
- [6. Fair-run procedure](#6-fair-run-procedure)
- [7. Metrics](#7-metrics)
- [8. OpenBrowser lean profile](#8-openbrowser-lean-profile)
- [9. Limitations](#9-limitations)
- [10. Setup](#10-setup)
- [11. Run](#11-run)
- [12. Files](#12-files)
- [13. Safety and cleanup](#13-safety-and-cleanup)
- [14. Historical results: 2026-11-23, lean profile](#14-historical-results-2026-11-23-lean-profile)
- [15. Historical results: 2026-11-24, cold five-arm lane](#15-historical-results-2026-11-24-cold-five-arm-lane)
- [16. Historical results: 2026-11-24, warm session lane](#16-historical-results-2026-11-24-warm-session-lane)
- [17. Historical results: 2026-11-24, warm session lane after the popup fix](#17-historical-results-2026-11-24-warm-session-lane-after-the-popup-fix)
- [18. Historical results: 2026-11-24, fast selector arm](#18-historical-results-2026-11-24-fast-selector-arm)
- [19. Historical results: 2026-11-24, warm tabs direct CDP click](#19-historical-results-2026-11-24-warm-tabs-direct-cdp-click)
- [20. Historical results: 2026-11-24, five-arm tabs-only warm lane](#20-historical-results-2026-11-24-five-arm-tabs-only-warm-lane)
- [21. Historical results: 2026-11-24, download wait wakeup A/B](#21-historical-results-2026-11-24-download-wait-wakeup-ab)
- [22. Historical results: 2026-11-24, warm CLI startup A/B](#22-historical-results-2026-11-24-warm-cli-startup-ab)
- [23. Historical results: 2026-11-24, paired tabs and download comparisons](#23-historical-results-2026-11-24-paired-tabs-and-download-comparisons)
- [24. Historical results: 2026-11-23, OpenBrowser extension A/B](#24-historical-results-2026-11-23-openbrowser-extension-ab)
- [25. Historical analysis: 2026-11-23 results](#25-historical-analysis-2026-11-23-results)
- [26. Historical results: 2026-11-24, final lanes (OpenBrowser 0.1.59.dev18)](#26-historical-results-2026-11-24-final-lanes-openbrowser-0159dev18)
  - [26.1. Final lanes, 2026-11-24, dev18: Warm five-arm, N=5](#261-final-lanes-2026-11-24-dev18-warm-five-arm-n5)
  - [26.2. Final lanes, 2026-11-24, dev18: Warm paired, N=10](#262-final-lanes-2026-11-24-dev18-warm-paired-n10)
  - [26.3. Final lanes, 2026-11-24, dev18: Cold six-arm, N=3](#263-final-lanes-2026-11-24-dev18-cold-six-arm-n3)
  - [26.4. Final lanes, 2026-11-24, dev18: MCP arm](#264-final-lanes-2026-11-24-dev18-mcp-arm)
- [27. Historical results: 2026-11-24, second final lanes (OpenBrowser 0.1.59.dev28)](#27-historical-results-2026-11-24-second-final-lanes-openbrowser-0159dev28)
  - [27.1. Second final lanes, 2026-11-24, dev28: Cold six-arm, N=3](#271-second-final-lanes-2026-11-24-dev28-cold-six-arm-n3)
  - [27.2. Second final lanes, 2026-11-24, dev28: Warm five-arm, N=5](#272-second-final-lanes-2026-11-24-dev28-warm-five-arm-n5)
  - [27.3. Second final lanes, 2026-11-24, dev28: Warm paired, N=10](#273-second-final-lanes-2026-11-24-dev28-warm-paired-n10)

<!-- toc:end -->

Deterministic local benchmark for scripted interactive browser tasks. No LLM,
no cloud browser, no AWS, and no external network. Every task is scored by a
server-side oracle, not by the adapter's self-report.

## 1. How to read these numbers

- Cold means a fresh process tree, daemon or server, and browser for every task. It includes startup cost.
- Warm means each arm starts once and the browser stays open. Each task still runs against a fresh fixture server. The `auth` task is cold only because it restarts the browser.
- Latency is lower is better ↓. Memory is lower is better ↓. A speedup value is higher is better ↑.
- The current cold and warm lanes use N=5 per arm and task. Arm order was rotated.
- `OpenBrowser MCP` means the package MCP stdio arm. `OpenBrowser CLI` means the one-cell CLI arm. The historical Arms table lists the fast and semantic variants used in older lanes.
- Each sample used an isolated HOME.
- The host Playwright cache was exposed to every arm.
- Lean MCP transport was the default.
- No AWS or cloud browser was used.
- Cold-lane host load was 6.486, 5.743, 5.360 before and 8.189, 6.929, 5.971 after on 18 logical CPUs. The brief does not report warm-lane host load.
- A paired win means one arm was faster on an individual sample matched against the same task and fixture. It is not a significance test.

## 2. Current results, 2026-11-25, rotated N=5

### 2.1. Cold interactive

Cold means a fresh process tree, daemon or server, and browser for every task. All 125 of 125 runs passed the exact oracle and the mock-keychain check. Transport was lean. The host Playwright cache was exposed to every arm.

Lower is better for latency ↓.

| Task | OpenBrowser MCP p50 ms | OpenBrowser CLI p50 ms | chrome-devtools-mcp p50 ms | agent-browser p50 ms | browser-use p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 715 | 999 | 4335 | 1599 | 1277 |
| modal | 811 | 1206 | 3639 | 2262 | 1571 |
| download | 954 | 1344 | 1633 | 1554 | 1368 |
| tabs | 658 | 1010 | 2426 | 1511 | 940 |
| auth | 1390 | 1915 | 3707 | 2529 | 2585 |

OpenBrowser MCP has the lowest p50 on all five cold tasks against all three competitors and won every paired sample, 5 of 5, on every task against every competitor.

OpenBrowser CLI won every cold task against every competitor except `tabs` against browser-use. CLI `tabs` p50 was 1010 ms against 940 ms. Do not claim that every arm wins every cold task.

Raw evidence:

- [results-cold-five-arm-lean-2026-11-25-n5-final2.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final2.jsonl)
- [results-cold-five-arm-lean-2026-11-25-n5-final2.meta.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final2.meta.json)
- [results-cold-five-arm-lean-2026-11-25-n5-final2.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final2.summary.json)

### 2.2. Warm interactive

Warm means each arm starts once and the browser stays open. Every task runs against a fresh fixture server. All 100 of 100 runs passed.

Lower is better for latency ↓.

| Task | OpenBrowser MCP p50 ms | OpenBrowser CLI p50 ms | chrome-devtools-mcp p50 ms | agent-browser p50 ms | browser-use p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 57 | 122 | 3481 | 524 | 486 |
| modal | 218 | 282 | 2734 | 1126 | 625 |
| download | 350 | 398 | 777 | 440 | 578 |
| tabs | 72 | 130 | 1485 | 382 | 184 |

Both OpenBrowser arms have the lowest warm p50 on all four tasks against all three competitors. OpenBrowser MCP won every paired sample, 5 of 5, on every task against every competitor. OpenBrowser CLI also won every paired sample, 5 of 5, on every task against every competitor.

The `auth` task is cold only because it restarts the browser.

Raw evidence:

- [results-warm-five-arm-lean-2026-11-25-n5-final2.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-five-arm-lean-2026-11-25-n5-final2.jsonl)
- [results-warm-five-arm-lean-2026-11-25-n5-final2.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-five-arm-lean-2026-11-25-n5-final2.summary.json)

### 2.3. Remaining measured gap

OpenBrowser CLI is the only OpenBrowser arm with a measured cold p50 loss in this lane: `tabs` at 1010 ms against browser-use at 940 ms. OpenBrowser MCP has the lowest p50 on every cold task, and both OpenBrowser arms have the lowest warm p50 on every warm task. Do not generalize these results to unmeasured tasks, surfaces, or configurations.

## 3. Interactive resource pilot, 2026-11-25

Memory: **Lower is better** ↓.

This pilot measures one sample per arm and task. All 25 observations passed the exact oracle, keychain check, resource gate, and cleanup. It is descriptive, not a significance claim.

| Task | OpenBrowser MCP footprint MiB | OpenBrowser CLI footprint MiB | chrome-devtools-mcp footprint MiB | agent-browser footprint MiB | browser-use footprint MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 136.2 | 146.5 | 480.5 | 494.2 | 554.7 |
| modal | 133.5 | 149.6 | 473.2 | 480.3 | 550.4 |
| download | 124.9 | 141.7 | 447.2 | 485.3 | 433.4 |
| tabs | 132.0 | 144.2 | 461.7 | 472.9 | 438.7 |
| auth | 132.8 | 144.6 | 455.1 | 475.0 | 547.1 |

| Task | OpenBrowser MCP tree RSS MiB | OpenBrowser CLI tree RSS MiB | chrome-devtools-mcp tree RSS MiB | agent-browser tree RSS MiB | browser-use tree RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 408.9 | 428.2 | 761.9 | 1205.7 | 1201.1 |
| modal | 409.4 | 433.8 | 805.8 | 1204.4 | 1198.2 |
| download | 390.9 | 418.6 | 762.0 | 1179.9 | 960.2 |
| tabs | 408.2 | 431.3 | 763.2 | 1193.4 | 980.9 |
| auth | 411.2 | 419.5 | 731.6 | 1188.3 | 1196.0 |

Both OpenBrowser arms have the lowest child footprint and tree RSS on all five tasks. Observed CPU: OpenBrowser MCP is lowest on all five tasks. OpenBrowser CLI is lowest on four; the one gap is cold `auth` at 3.13 seconds against agent-browser at 3.06 seconds. CPU is a sampled lower bound and includes native driver subprocesses.

Raw data: [results-interactive-resources-pilot-2026-11-25-n1.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-interactive-resources-pilot-2026-11-25-n1.jsonl).

## 4. Arms

| Arm | Version | Native scripted surface | Browser |
| --- | --- | --- | --- |
| `openbrowser` | OpenBrowser CLI `0.1.59.dev47+gbedbbee6b.d20261125` | One `openbrowser-ai -c` Python cell per task | System Chrome 158.0.8127.89 through `channel=chrome` |
| `openbrowser-fast` | OpenBrowser CLI `0.1.59.dev47+gbedbbee6b.d20261125` | One `openbrowser-ai -c` Python cell with CSS helpers for fixture ids | System Chrome 158.0.8127.89 through `channel=chrome` |
| `openbrowser-mcp` | OpenBrowser CLI `0.1.59.dev47+gbedbbee6b.d20261125` | MCP stdio `2026-07-28`: `create_context`, `execute_code`, `close_context` | System Chrome 158.0.8127.89 through `channel=chrome` |
| `openbrowser-mcp-fast` | OpenBrowser CLI `0.1.59.dev47+gbedbbee6b.d20261125` | Same MCP surface with CSS helpers for fixture ids | System Chrome 158.0.8127.89 through `channel=chrome` |
| `chrome-devtools-mcp` | `chrome-devtools-mcp` 1.13.0 | MCP stdio, 33 tools, `pageId` routing | System Chrome 158.0.8127.89 |
| `agent-browser` | `agent-browser` 0.43.0 | CLI commands, one process per command | System Chrome 158.0.8127.89 |
| `browser-use` | `browser-use` 1.0.8 (`browser-harness` 0.1.13) | Piped Python in one CLI process | Dedicated system Chrome 158.0.8127.36 |

Cold runs use a fresh profile per run. Warm runs start each arm once and keep
the browser open. All current lanes use a 1280x800 viewport and local fixtures
only. Every Chrome launch includes `--use-mock-keychain`. The harness checks the
actual browser process command line and fails the run if the flag is missing.

The version column records the arms used in the historical 2026-11-24 runs.
The current 2026-11-25 brief does not record package or browser version strings,
so this table is not a current build manifest.

The earlier four-arm warm results below used OpenBrowser CLI
`0.1.58.dev12+gc421f7eac`. The fast selector results use the build listed
above. Each historical results section is dated and names the build it used.
The newest paired sections are historical. Use the 2026-11-25 current results
above for current competitor claims.

## 5. Tasks

| Task | Route | Exact oracle |
| --- | --- | --- |
| `wizard` | `/wizard` | Exactly one submission with `Alice Bench`, `alice@example.com`, `CA`, `ON`, `pro`, 5 seats, terms accepted, quote `Q-1`, and one post for each of the three steps |
| `modal` | `/modal` | At least two modal opens, at least one Escape close, then `Project Atlas` selected through the ARIA combobox |
| `download` | `/download` | One request and one completion for `report.csv`, file size 2229 bytes, SHA-256 `293ce662323595edec422dd7b0abebc9884971c8d7928de483fb23620c0cf2e3` |
| `tabs` | `/tabs` | Detail tab loaded with `window.opener`, approve clicked once, original tab received `approved:42`, detail tab close observed, original title `Tabs Bench` |
| `auth` | `/login` | Exactly one login POST, at least two account views across a browser restart, account secret `ACCOUNT-SECRET-42` read in phase 2, password absent from output |

The `openbrowser` and `openbrowser-fast` arms poll a JavaScript predicate every
25 ms and notice a modal filter result within about 155 ms of the fixture's
150 ms debounce. The `auth` fast script fills the login form through CSS ids
instead of the semantic selector map, matching the other fast scripts.

Per-run reset: each run starts a new `ThreadingHTTPServer` on an ephemeral
loopback port with new state, a new profile, a new download directory, and a
new `HOME`, `TMPDIR`, and `OPENBROWSER_SOCKET`. The OpenBrowser daemon is
stopped and removed before every run. `agent-browser` sessions are closed with
`close --all`.
`chrome-devtools-mcp` servers are terminated and their Chrome processes are
killed by profile path. The OpenBrowser MCP arm speaks the stateless
`2026-07-28` protocol with `server/discover` and no handshake, so it measures
the same per-request path a coding agent uses. `browser-use` daemons and dedicated Chrome processes
are killed after every run.

## 6. Fair-run procedure

1. Pin every version and record the browser executable. The historical
   2026-11-24 runs used Chrome `158.0.8127.36`. The current brief does not
   record a browser version string.
2. Use the same viewport, 1280x800, for all arms: OpenBrowser `window_size`,
   chrome-devtools-mcp `--viewport 1280x800`, agent-browser
   `set viewport 1280 800`, browser-use `--window-size=1280,800`.
3. Cold runs use a fresh profile and a fresh fixture server per run. Warm runs
   reuse the arm's browser but still use a fresh fixture server per task. No
   external requests.
4. Cold runs use a fresh process tree for each task. Warm runs start each arm
   once. Latency is wall clock from the first adapter call to the adapter
   return. Cold latency includes daemon and browser start. The oracle check runs
   after the timer stops, so verification is not charged to the adapter.
5. Score correctness first. A failed oracle scores zero regardless of latency.
   p50 and p95 are computed over successful runs only. A failure-inclusive
   latency is recorded in the raw JSONL.
6. Keep the raw JSONL. The README table is a summary, not the evidence.

## 7. Metrics

- `p50` and `p95`: nearest-rank over successful runs. The current cold and
  warm lanes use five samples per arm and task. These are descriptive values,
  not a significance test. Older sections state their own sample count.
- `tool calls`: MCP `tools/call` count for chrome-devtools-mcp, CLI command
  count for agent-browser, helper count for browser-use, and `-c` cell count
  for OpenBrowser. OpenBrowser batches many browser actions inside one cell,
  so this number is not directly comparable across architectures.
- `process starts`: adapter process count for the task. Agent-browser starts
  one process per command. OpenBrowser, chrome-devtools-mcp, and browser-use
  use one long-lived server or daemon per phase.
- `output bytes`: UTF-8 bytes written to stdout by the adapter. Agent-browser
  emits a JSON envelope with lifecycle metadata on every command, so its
  output is larger by design.
- `requests` and `request bytes`: fixture request count and response bytes.

## 8. OpenBrowser lean profile

The OpenBrowser arm uses the lean automation profile:

- `enable_default_extensions: false`
- index lookup uses `get_browser_state_summary(include_screenshot=False)`
- `OPENBROWSER_SOCKET` is pinned to the run home so a global daemon cannot
  intercept a run

The one-shot `openbrowser-ai scrape` command already disables default
extensions. The local fixtures cannot use ad blocking, cookie handling, or URL
cleaning, so loading those extensions measures setup cost without task value.
Screenshots are not needed to resolve interactive indexes. The
`--openbrowser-extensions` flag reruns the arm with the default extension
bundle for comparison.

Validating this isolation exposed a daemon diagnostics bug. When another
daemon's environment was unreadable, the diagnostics layer attributed the
caller's default socket to that process. It now treats an empty or unreadable
environment as unknown. When the environment is readable and has no explicit
`OPENBROWSER_SOCKET`, it derives the default socket from that daemon's own
`HOME`. The caller's default socket is never reused for another process.

## 9. Limitations

- The current cold and warm lanes use five samples per arm and task. The p95
  values are indicative, not a significance test. Older sections state their
  own sample count.
- Custom widget fidelity differs. chrome-devtools-mcp does not expose the
  ARIA listbox options in its accessibility snapshot, so the adapter uses its
  native `evaluate_script` fallback to click the option. The task still passes
  through the page's own selection handler.
- Auth mechanisms differ by product. OpenBrowser exports and reloads a
  storage state file; the other arms rely on a persistent Chrome profile.
  This measures each product's native session reuse path, not one identical
  mechanism.
- Tool-call counts reflect architecture. OpenBrowser batches actions in one
  cell, agent-browser uses one process per command, chrome-devtools-mcp uses
  one MCP call per action, and browser-use uses one piped Python process.
- The fixtures are deterministic and local. They do not include anti-bot
  challenges, real credentials, live websites, network latency, or CAPTCHA.
- Output bytes are not context tokens. Agent-browser returns a lifecycle
  envelope per command, which is useful diagnostic data but inflates bytes.
- The harness runs on macOS. Linux results can differ because browser launch
  and filesystem costs differ.

## 10. Setup

Tools live outside the repository:

```sh
mkdir -p ~/.cache/openbrowser-interactive-bench/npm
npm install --prefix ~/.cache/openbrowser-interactive-bench/npm \
  chrome-devtools-mcp@1.13.0 agent-browser@0.43.0

uv venv ~/.cache/openbrowser-interactive-bench/browser-use-venv --python 3.12
uv pip install --python ~/.cache/openbrowser-interactive-bench/browser-use-venv/bin/python \
  browser-use==1.0.8

UV_PROJECT_ENVIRONMENT=~/.cache/openbrowser-interactive-bench/openbrowser-env \
  uv sync --locked --all-extras
```

The local `chrome-devtools-mcp` binary is the same 1.13.0 package that
`npx -y chrome-devtools-mcp@1.13.0` would download. The harness uses the local
install to remove network and npx startup jitter from the measurement.

## 11. Run

```sh
uv run --no-project python -m benchmarks.interactive.run_benchmark \
  --samples 3 \
  --output benchmarks/interactive/results/results-2026-11-23-lean.jsonl
```

A focused task:

```sh
uv run --no-project python -m benchmarks.interactive.run_benchmark \
  --tasks tabs --arms openbrowser agent-browser --samples 1
```

Useful flags:

- `--arms openbrowser chrome-devtools-mcp agent-browser browser-use`
- `--tasks wizard modal download tabs auth`
- `--samples N`
- `--warmups N`
- `--keep-runs` to keep temp profiles under `/tmp/obib-runs`
- `--openbrowser-extensions` to measure the default OpenBrowser extension bundle
- `--output PATH`

The default temp root is `/tmp/obib-runs`. It is outside the repository so the
OpenBrowser daemon socket path stays below the macOS `AF_UNIX` limit.

## 12. Files

| Path | Purpose |
| --- | --- |
| `fixtures/server.py` | Deterministic HTTP server, fixture state, and task oracles' server data |
| `tasks.py` | Task metadata and exact success oracles |
| `adapters/openbrowser.py` | OpenBrowser daemon adapter |
| `adapters/chrome_devtools.py` | chrome-devtools-mcp stdio adapter |
| `adapters/agent_browser.py` | agent-browser CLI adapter |
| `adapters/browser_use.py` | browser-use piped Python adapter |
| `adapters/mcp_stdio.py` | Minimal newline-delimited JSON-RPC MCP client |
| `adapters/base.py` | Run context, process hygiene, dedicated Chrome, keychain verification |
| `run_benchmark.py` | Runner, JSONL writer, summary table |
| [results](https://github.com/billy-enrizky/openbrowser-ai/tree/main/benchmarks/interactive/results) | Raw JSONL and summary JSON |

## 13. Safety and cleanup

- Every browser launch includes `--use-mock-keychain`. The harness reads the
  actual Chrome command line and records `keychain_verified`.
- No Keychain entries are created, reset, or modified.
- Every adapter stops its daemon and kills its browser processes in a
  `finally` block. The runner also terminates any process whose command line
  contains the run's unique temp path.
- No AWS, cloud browser, paid API, or model provider is called.

## 14. Historical results: 2026-11-23, lean profile

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

macOS arm64, Chrome `158.0.8127.36`, 3 measured samples per arm and task, no
warmups. 60 of 60 runs passed their oracle. All 60 runs reported
`keychain_verified: true`. All 15 OpenBrowser runs recorded
`openbrowser_extensions: false`.

Lower is better for latency ↓.

| Task | Arm | OK | p50 ms | p95 ms | Tool calls | Process starts | Output bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | openbrowser | 3/3 | 4837 | 4855 | 1 | 1 | 18 |
| wizard | chrome-devtools-mcp | 3/3 | 4570 | 4633 | 22 | 1 | 2865 |
| wizard | agent-browser | 3/3 | 1789 | 1946 | 17 | 17 | 5607 |
| wizard | browser-use | 3/3 | 1499 | 1567 | 15 | 1 | 48 |
| modal | openbrowser | 3/3 | 3509 | 3744 | 1 | 1 | 31 |
| modal | chrome-devtools-mcp | 3/3 | 3805 | 3955 | 17 | 1 | 1512 |
| modal | agent-browser | 3/3 | 2468 | 2559 | 13 | 13 | 5024 |
| modal | browser-use | 3/3 | 1811 | 2282 | 11 | 1 | 61 |
| download | openbrowser | 3/3 | 3492 | 3619 | 1 | 1 | 550 |
| download | chrome-devtools-mcp | 3/3 | 2054 | 2375 | 5 | 1 | 490 |
| download | agent-browser | 3/3 | 1710 | 1813 | 3 | 3 | 1155 |
| download | browser-use | 3/3 | 1451 | 1542 | 4 | 1 | 54 |
| tabs | openbrowser | 3/3 | 7792 | 7946 | 1 | 1 | 59 |
| tabs | chrome-devtools-mcp | 3/3 | 2426 | 2508 | 12 | 1 | 1077 |
| tabs | agent-browser | 3/3 | 1638 | 1869 | 13 | 13 | 5097 |
| tabs | browser-use | 3/3 | 1111 | 1153 | 12 | 1 | 102 |
| auth | openbrowser | 3/3 | 5110 | 5263 | 2 | 2 | 66 |
| auth | chrome-devtools-mcp | 3/3 | 3963 | 4157 | 14 | 2 | 1064 |
| auth | agent-browser | 3/3 | 2709 | 2777 | 10 | 11 | 3856 |
| auth | browser-use | 3/3 | 3150 | 3300 | 8 | 2 | 124 |

Raw evidence: [results-2026-11-23-lean.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-2026-11-23-lean.jsonl).

The original default-extension run remains at
[results-2026-11-23.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-2026-11-23.jsonl).

## 15. Historical results: 2026-11-24, cold five-arm lane

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

That historical run used the OpenBrowser 0.1.58 build for both OpenBrowser arms
and the pinned competitor versions. It started a fresh process tree, daemon,
and browser for every task. All 75 runs passed. Shared host, no load average
was recorded.

Lower is better for latency ↓.

| Task | OpenBrowser semantic, p50 ms | OpenBrowser selectors, p50 ms | chrome-devtools-mcp, p50 ms | agent-browser, p50 ms | browser-use, p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wizard` | 4349 | 2138 | 4472 | 1652 | 1426 |
| `modal` | 3023 | 2115 | 3730 | 2418 | 1915 |
| `download` | 2883 | 2308 | 1741 | 1727 | 1439 |
| `tabs` | 2697 | 1936 | 2404 | 1665 | 1078 |
| `auth` | 4149 | 4200 | 3926 | 2872 | 2847 |

That historical run is superseded by the 2026-11-25 cold results above. Do not
use its rankings for current claims.

The cold lane is a startup lane. A same-session probe measured a cold
`openbrowser-ai -c` no-op cell at 1381 ms p50 and a warm `-c` cell at 99.8 ms.
`openbrowser-ai warm` took 1454 ms p50. Direct `BrowserSession.start()` took
459 ms p50, so daemon, config, event bus, and namespace setup add about one
second before task code runs. Browser-use non-auth totals were 1.1 to 1.9
seconds, so OpenBrowser startup alone is close to the whole browser-use run.
The auth task starts the browser twice and shows the startup gap twice.

Raw evidence:

- [results-cold-current-five-arm-2026-11-24-n3.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-current-five-arm-2026-11-24-n3.jsonl)
- [results-cold-current-five-arm-2026-11-24-n3.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-current-five-arm-2026-11-24-n3.summary.json)

## 16. Historical results: 2026-11-24, warm session lane

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

The cold lane starts a new process tree, daemon, and browser for every task.
The warm lane starts each arm once, then runs each task against a fresh fixture
server. It keeps the startup cost separate from steady-state task latency. The
`auth` task remains in the cold lane because it deliberately restarts the
browser to prove profile reuse across processes.

The OpenBrowser arm uses `openbrowser-ai warm`. chrome-devtools-mcp starts its
MCP server and Chrome once. agent-browser starts one named session.
browser-use starts one dedicated Chrome and reuses its CDP URL.

Paired N=3 run, macOS arm64, system Chrome `158.0.8127.36`, isolated HOME,
config, profile, and socket, headless, with `--use-mock-keychain`. Cold p50 is
from [results-2026-11-23-lean.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-2026-11-23-lean.jsonl). Warmup wall time by arm: OpenBrowser
6896 ms, chrome-devtools-mcp
3638 ms, agent-browser
3214 ms, browser-use
1874 ms.

Lower is better for cold and warm latency ↓. Higher is better for speedup ↑.

| Task | Arm | Cold p50 ms | Warm p50 ms | Warm p95 ms | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| wizard | openbrowser | 4837 | 3401 | 3477 | 1.42x |
| wizard | chrome-devtools-mcp | 4570 | 4089 | 4403 | 1.12x |
| wizard | agent-browser | 1789 | 1264 | 1670 | 1.42x |
| wizard | browser-use | 1499 | 2111 | 3067 | 0.71x |
| modal | openbrowser | 3509 | 1574 | 1875 | 2.23x |
| modal | chrome-devtools-mcp | 3805 | 2910 | 2936 | 1.31x |
| modal | agent-browser | 2468 | 1754 | 1789 | 1.41x |
| modal | browser-use | 1811 | 1057 | 1258 | 1.71x |
| download | openbrowser | 3492 | 966 | 1391 | 3.62x |
| download | chrome-devtools-mcp | 2054 | 836 | 840 | 2.46x |
| download | agent-browser | 1710 | 629 | 723 | 2.72x |
| download | browser-use | 1451 | 915 | 1001 | 1.59x |
| tabs | openbrowser | 7792 | 6170 | 6668 | 1.26x |
| tabs | chrome-devtools-mcp | 2426 | 1644 | 1689 | 1.48x |
| tabs | agent-browser | 1638 | 1237 | 2480 | 1.32x |
| tabs | browser-use | 1111 | 650 | 651 | 1.71x |

All 48 warm runs passed their oracle. In that historical run, OpenBrowser
improved over its own cold lane but still trailed agent-browser and browser-use
on wizard, download, and tabs. It beat chrome-devtools-mcp on modal and was
close on wizard. The tabs task was the largest gap: OpenBrowser 6170 ms warm
versus browser-use 650 ms. This is not a current warm-lane claim.

Run the lane:

```sh
OPENBROWSER_BIN=/path/to/openbrowser-ai \
uv run --locked python -m benchmarks.interactive.warm_benchmark \
  --samples 3 \
  --arms openbrowser chrome-devtools-mcp agent-browser browser-use \
  --tasks wizard modal download tabs \
  --output benchmarks/interactive/results/results-warm.jsonl \
  --cold-results benchmarks/interactive/results/results-2026-11-23-lean.jsonl
```

Raw evidence:

- [results-warm-4arm-2026-11-24-n3.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-4arm-2026-11-24-n3.jsonl)
- [results-warm-4arm-2026-11-24-n3.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-4arm-2026-11-24-n3.summary.json)

## 17. Historical results: 2026-11-24, warm session lane after the popup fix

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

The popup fix in PR #217 stops a `target="_blank"` click from waiting the full
five-second mouse-release bound. It checks for a new page target and fails
quickly when none appears. This run uses the same four arms, tasks, fixtures,
and Chromium build as the earlier warm lane.

The run uses 3 samples per arm and task. All 48 runs passed their oracle. Host
load was high during the run, and OpenBrowser warmup took 23.6 seconds, so treat
these values as shared-host smoke evidence, not a quiet-host p50.

Lower is better for latency ↓.

| Task | OpenBrowser | chrome-devtools-mcp | agent-browser | browser-use |
| --- | ---: | ---: | ---: | ---: |
| `wizard` | 2607 | 3511 | 1231 | 723 |
| `modal` | 1421 | 2768 | 1604 | 787 |
| `download` | 924 | 785 | 589 | 720 |
| `tabs` | 1124 | 1485 | 918 | 335 |

In that historical run, OpenBrowser reduced warm `tabs` from 6170 ms to
1124 ms, a 5.49x improvement against the earlier run. It beat
chrome-devtools-mcp on `wizard`, `modal`, and `tabs`. It still trailed
browser-use on every task and trailed agent-browser on `wizard`, `download`,
and `tabs`. The result was progress, not a current warm-lane claim.

Generate the documentation chart from the summary:

```sh
uv run --with matplotlib python benchmarks/interactive/plot_results.py
uv run --with matplotlib python benchmarks/interactive/plot_cold_results.py
```

Both charts show latency. Lower is better ↓. The first script writes
`docs/images/native-interactive-warm-current-p50.png`. The cold script writes
`docs/images/native-interactive-cold-p50.png`.

Raw evidence:

- [results-warm-4arm-2026-11-24-after-popup-n3.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-4arm-2026-11-24-after-popup-n3.jsonl)
- [results-warm-4arm-2026-11-24-after-popup-n3.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-4arm-2026-11-24-after-popup-n3.summary.json)

## 18. Historical results: 2026-11-24, fast selector arm

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

`openbrowser-fast` uses the same adapter and lifecycle as `openbrowser`. Its
task code uses CSS helpers for fixture ids. The semantic `openbrowser` scripts
and the earlier result files remain unchanged. The tabs task still routes the
`target="_blank"` anchor through `click_element()`. The oracle verifies that
the detail tab sees `window.opener`.

The paired run used three samples per arm and task on an 18-core Mac. Load
averages were `9.24`, `8.27`, and `7.40` before the run. They were `9.66`,
`8.75`, and `7.63` after the run. The harness ran all samples for `openbrowser`
before it ran `openbrowser-fast`. This was not an interleaved or quiet-host
comparison.

Lower is better for latency ↓.

| Task | `openbrowser` p50 ms | `openbrowser-fast` p50 ms | Observed result |
| --- | ---: | ---: | --- |
| wizard | 2319.702 | 204.657 | Fast lower by 2115.045 ms |
| modal | 1161.486 | 383.861 | Fast lower by 777.625 ms |
| download | 772.710 | 570.671 | Fast lower by 202.039 ms |
| tabs | 975.636 | 990.837 | No measurable difference |

All 24 paired-run samples passed both the adapter check and the task oracle.
The modal oracle confirms that the fill action triggers the search filter. The
helper test checks that its `input` and `change` events bubble. The tabs oracle
confirms that the popup keeps its opener. Treat these N=3 results as
exploratory because the host was busy and the run order was sequential.

The full five-arm lane used the same host and three samples per arm and task.
Load averages were `9.66`, `8.75`, and `7.63` before the run. They were `7.59`,
`8.17`, and `7.52` after the run. All 60 samples passed both checks.

Lower is better for latency ↓.

| Task | `openbrowser` | `openbrowser-fast` | `chrome-devtools-mcp` | `agent-browser` | `browser-use` |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 2164.780 | 204.323 | 3532.686 | 537.348 | 524.218 |
| modal | 1161.091 | 396.123 | 2775.570 | 1205.172 | 1167.143 |
| download | 798.839 | 575.519 | 781.001 | 480.877 | 585.277 |
| tabs | 994.065 | 984.681 | 1492.314 | 440.220 | 217.152 |

The cells show warm p50 milliseconds. In that historical run,
`openbrowser-fast` was lower than the semantic arm on wizard, modal, and
download. The tabs result changed direction between the paired run and the full
lane. Its ranges overlap, so the data show no measurable difference.
`agent-browser` was faster on download and tabs. Download against `browser-use`
was parity because the sample ranges overlap, and `browser-use` was faster on
tabs. This did not establish a universal win. The 2026-11-25 current results
above supersede this section for current competitor claims.

Rerun the paired selector check:

```sh
OPENBROWSER_BIN=/path/to/openbrowser-ai \
uv run --locked python -m benchmarks.interactive.warm_benchmark \
  --samples 3 \
  --arms openbrowser openbrowser-fast \
  --tasks wizard modal download tabs \
  --output benchmarks/interactive/results/results-warm-fast-selector-final-ab-2026-11-24-n3.jsonl \
  --cold-results benchmarks/interactive/results/results-2026-11-23-lean.jsonl
```

Rerun the five-arm lane:

```sh
OPENBROWSER_BIN=/path/to/openbrowser-ai \
uv run --locked python -m benchmarks.interactive.warm_benchmark \
  --samples 3 \
  --arms openbrowser openbrowser-fast chrome-devtools-mcp agent-browser browser-use \
  --tasks wizard modal download tabs \
  --output benchmarks/interactive/results/results-warm-fast-selector-final-all-2026-11-24-n3.jsonl \
  --cold-results benchmarks/interactive/results/results-2026-11-23-lean.jsonl
```

Raw evidence:

- [results-warm-fast-selector-final-ab-2026-11-24-n3.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-fast-selector-final-ab-2026-11-24-n3.jsonl)
- [results-warm-fast-selector-final-ab-2026-11-24-n3.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-fast-selector-final-ab-2026-11-24-n3.summary.json)
- [results-warm-fast-selector-final-all-2026-11-24-n3.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-fast-selector-final-all-2026-11-24-n3.jsonl)
- [results-warm-fast-selector-final-all-2026-11-24-n3.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-fast-selector-final-all-2026-11-24-n3.summary.json)

## 19. Historical results: 2026-11-24, warm tabs direct CDP click

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

`openbrowser-fast` now calls `click_selector_and_wait_for_new_tab()` for the
`#compare` link. The helper reads the link rectangle with `evaluate()`. It sends
CDP mouse input and returns the new page target ID. The adapter then calls the
existing `switch` and `close` actions. The semantic arm keeps its current `_tabs`
script.

The paired run alternated the pre-helper fast script and the direct helper in
one warm browser session. Each variant ran five times with a fresh fixture. All
ten adapter checks and server-side oracles passed. The oracle checks that the
detail page sees `window.opener`, sends an approval event, and sends
`approved:42` to the original page. It also checks the original title and the
detail-page close event.

Load on 18 logical CPUs was `9.48`, `8.20`, and `8.18` before the run. It was
`9.51`, `8.27`, and `8.20` after the run. Load exceeded the declared limit of
4, so the results are exploratory, not quiet-host p50.

Lower is better for latency ↓.

| Path | Samples | p50 ms | Range ms | Adapter and oracle |
| --- | ---: | ---: | ---: | ---: |
| Existing CodeAgent click | 5 | 1,062.497 | 974.280 to 1,216.891 | 5/5 |
| Direct CDP click helper | 5 | 367.326 | 289.933 to 405.063 | 5/5 |

The direct helper lowered p50 by `695.171` ms, or `65.428%`, in this run. This
paired comparison tests two OpenBrowser paths. It does not compare a competitor
or establish a competitor win.

The step timings show where the time went. The baseline values came from a
separate N=5 diagnostic run. The direct values came from the paired run. These
step values are not paired deltas.

Lower is better for latency ↓.

| Step | Baseline p50 ms | Direct p50 ms |
| --- | ---: | ---: |
| Popup click and target wait | 517.223 | 42.772 |
| Fixed focus-settle sleep | 151.925 | Not used |
| Separate target-poll sleep | 101.369 | Not used |
| Switch to detail tab | 6.752 | 13.523 |
| Wait for the approve button | 1.253 | 13.132 |
| Close detail tab | 16.723 | 27.784 |
| Switch to original tab | 4.957 | 8.924 |

The legacy step profile also measured `get_browser_state_summary()` at `18.632`
ms and `get_selector_map()` at `0.486` ms. The `DefaultActionWatchdog` click
handler took `506.184` ms in one diagnostic sample. The click handler, not the
state summary, dominated that step.

Raw evidence:

- [results-warm-tabs-step-baseline-2026-11-24-n5.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-tabs-step-baseline-2026-11-24-n5.jsonl)
- [results-warm-tabs-step-baseline-2026-11-24-n5.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-tabs-step-baseline-2026-11-24-n5.summary.json)
- [results-warm-tabs-direct-cdp-2026-11-24-n5-paired.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-tabs-direct-cdp-2026-11-24-n5-paired.jsonl)
- [results-warm-tabs-direct-cdp-2026-11-24-n5-paired.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-tabs-direct-cdp-2026-11-24-n5-paired.summary.json)

## 20. Historical results: 2026-11-24, five-arm tabs-only warm lane

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

This lane used the direct popup helper and the three interactive competitors.
It warmed each arm once, then ran five fresh tabs fixtures per arm. It used
arm-major order, not interleaved. All 25 runs passed the adapter check and the
exact tabs oracle.

Load on 18 logical CPUs was `16.81`, `10.82`, and `9.78` before the run. It was
`20.09`, `12.91`, and `10.63` after the run. Load stayed above 4, so treat the
latencies as exploratory.

Lower is better for latency ↓.

| Arm | Samples | p50 ms | Range ms | Adapter and oracle |
| --- | ---: | ---: | ---: | ---: |
| `openbrowser` | 5 | 1,171.336 | 1,055.090 to 1,207.928 | 5/5 |
| `openbrowser-fast` | 5 | 484.242 | 377.246 to 539.227 | 5/5 |
| `chrome-devtools-mcp` | 5 | 1,552.286 | 1,534.648 to 1,621.911 | 5/5 |
| `agent-browser` | 5 | 795.759 | 779.441 to 893.941 | 5/5 |
| `browser-use` | 5 | 594.553 | 507.341 to 1,050.248 | 5/5 |

In that historical run, `openbrowser-fast` had a lower p50 than
`agent-browser` and `chrome-devtools-mcp`. Its range overlapped the
`browser-use` range. The run did not show a measurable difference against
`browser-use`. These high-load results are not a current competitor ranking.

Rerun this lane:

```sh
while ! mkdir /tmp/openbrowser-bench.lock 2>/dev/null; do sleep 1; done
trap 'rmdir /tmp/openbrowser-bench.lock 2>/dev/null || true' EXIT HUP INT TERM
OPENBROWSER_BIN=/path/to/openbrowser-ai \
uv run --locked python -m benchmarks.interactive.warm_benchmark \
  --samples 5 \
  --arms openbrowser openbrowser-fast chrome-devtools-mcp agent-browser browser-use \
  --tasks tabs \
  --output benchmarks/interactive/results/results-warm-tabs-five-arm.jsonl \
  --cold-results benchmarks/interactive/results/results-2026-11-23-lean.jsonl
```

Raw evidence:

- [results-warm-tabs-five-arm-direct-cdp-2026-11-24-n5.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-tabs-five-arm-direct-cdp-2026-11-24-n5.jsonl)
- [results-warm-tabs-five-arm-direct-cdp-2026-11-24-n5.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-tabs-five-arm-direct-cdp-2026-11-24-n5.summary.json)

## 21. Historical results: 2026-11-24, download wait wakeup A/B

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

This run compared the 100 ms registry-poll baseline with the event-wakeup
candidate. Both builds used `openbrowser-fast`, the same fixture, and the same
download directory rules. The runner alternated build order by sample. It
measured the full task, click-to-terminal time, and terminal-to-return time.
The exact oracle checked the server request, completion count, file size, and
SHA-256. The baseline binary came from `01c4a2e29`, before the tabs PR merged.
The quiet-host confirmation will use the rebased main build.

Lower is better for latency ↓.

| Build | Task p50 ms, range | Click to terminal p50 ms, range | Terminal to return p50 ms, range |
| --- | --- | --- | --- |
| 100 ms poll | 620.720, 613.253 to 709.969 | 317.844, 308.553 to 326.329 | 87.354, 81.045 to 95.159 |
| Event wakeup | 526.401, 519.624 to 552.766 | 308.707, 307.779 to 317.485 | 0.159, 0.141 to 0.183 |

All 10 samples passed. Each file had 2,229 bytes and SHA-256
`293ce662323595edec422dd7b0abebc9884971c8d7928de483fb23620c0cf2e3`. The task
p50 difference was 94.318 ms. The terminal-to-return p50 difference was
87.195 ms. The task p50 difference is smaller than the largest sample range
of 96.716 ms, so this run did not establish a full-task latency win.

The host had 18 logical CPUs. Load averages were 25.46, 17.03, and 13.29 before
the run. They were 23.55, 17.13, and 13.41 after the measured samples. The
maximum sampled load was 25.456. Treat every latency here as exploratory. Do
not call 526.401 ms a quiet-host p50 or claim a competitor win from this run.
The quiet-host confirmation is pending after merge.

The runner acquires `/tmp/openbrowser-bench.lock` with a `mkdir` spin-wait. It
requires a short work path under `/tmp`. It marks `claimable` false if any
sampled load average exceeds 4. Re-run it with five or more samples:

```sh
uv run --locked python -m benchmarks.interactive.download_wait_ab \
  --baseline-bin /path/to/baseline/bin/openbrowser-ai \
  --treatment-bin /path/to/treatment/bin/openbrowser-ai \
  --baseline-sha 01c4a2e29 \
  --treatment-sha candidate-source-id \
  --workdir /tmp/obab \
  --output benchmarks/interactive/results/results-warm-download-wakeup-ab.jsonl \
  --samples 5
```

Raw evidence: [`A/B JSONL`](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-download-wakeup-ab-2026-11-24-n5.jsonl) and [`summary`](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-download-wakeup-ab-2026-11-24-n5.summary.json).

## 22. Historical results: 2026-11-24, warm CLI startup A/B

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

The `-c` fast path now executes a cell against a warm daemon without importing
asyncio, the logging configuration, pydantic settings, or dotenv. The full
client remains the cold-start fallback for a missing daemon.

The A/B compared the pre-change binary at `03c81dd5e` with the branch binary.
It ran 12 samples per binary in alternating order against the same warm daemon.
Load on 18 logical CPUs was `12.129`, `12.341`, and `10.945` before the run.
It was `12.083`, `12.328`, and `10.957` after the run. Absolute values are
shared-host evidence; the paired result is load-robust.

Lower is better for latency ↓.

| Arm | Samples | p50 ms | p95 ms | Range ms |
| --- | ---: | ---: | ---: | ---: |
| Pre-change `-c` | 12 | 98.730 | 103.334 | 95.060 to 118.617 |
| Warm-path `-c` | 12 | 61.825 | 69.616 | 58.601 to 70.649 |

The warm path was faster in 12 of 12 pairs. The median paired difference was
`34.452` ms, or `34.9` percent of the pre-change p50.

Rerun the A/B:

```sh
uv run --locked python -m benchmarks.interactive.cli_startup_ab \
  --old-binary /path/to/pre-change/openbrowser-ai \
  --new-binary /path/to/branch/openbrowser-ai \
  --samples 12 \
  --output benchmarks/interactive/results/results-cli-startup-ab.jsonl
```

Raw evidence:

- [results-cli-startup-ab-2026-11-24-n12.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cli-startup-ab-2026-11-24-n12.jsonl)
- [results-cli-startup-ab-2026-11-24-n12.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cli-startup-ab-2026-11-24-n12.summary.json)

## 23. Historical results: 2026-11-24, paired tabs and download comparisons

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

The five-arm lane runs every sample for one arm before it runs the next arm.
On a shared host, load drift can dominate a small difference. The paired runner
warms both arms once and alternates their order for every sample. Each run gets
a fresh fixture server and an exact oracle check.

The tabs script now switches to the original tab before it closes the detail
tab. Closing a background tab skips focus repair. The script passes
`timeout=15` to `click_selector()` so the selector wait and click share one
helper call. The popup target poll is bounded at 5 ms instead of 25 ms.

Load on 18 logical CPUs was `8.399`, `11.152`, and `10.625` before the run.
It was `8.170`, `10.786`, and `10.511` after the run. Absolute values are not a
quiet-host estimate. All 40 runs passed their adapter check and exact oracle.

Lower is better for latency ↓.

| Task | OpenBrowser fast p50 ms | Competitor | Competitor p50 ms | Pairs won | Paired delta p50 ms |
| --- | ---: | --- | ---: | ---: | ---: |
| `tabs` | 226.797 | browser-use | 279.402 | 10/10 | 57.794 |
| `download` | 464.463 | agent-browser | 491.526 | 10/10 | 24.587 |

The download fixture holds the response for 300 ms. The `24.587` ms paired
difference is the control-plane margin above that fixed delay, not a claim that
the whole task is three times faster.

Rerun a paired comparison:

```sh
OPENBROWSER_BIN=/path/to/openbrowser-ai \
uv run --locked python -m benchmarks.interactive.paired_compare \
  --arms openbrowser-fast browser-use \
  --task tabs \
  --samples 10 \
  --output benchmarks/interactive/results/results-paired-tabs.jsonl
```

Raw evidence:

- [results-paired-tabs-2026-11-24-n10.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-tabs-2026-11-24-n10.jsonl)
- [results-paired-tabs-2026-11-24-n10.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-tabs-2026-11-24-n10.summary.json)
- [results-paired-download-2026-11-24-n10.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-download-2026-11-24-n10.jsonl)
- [results-paired-download-2026-11-24-n10.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-download-2026-11-24-n10.summary.json)

## 24. Historical results: 2026-11-23, OpenBrowser extension A/B

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

The A/B uses one binary, the same screenshot setting, and 3 samples per task.
Only `enable_default_extensions` changes. The default bundle adds 12.9 to 21.8
percent to the measured p50.

Lower is better for latency ↓.

| Task | Default extensions p50 ms | Lean p50 ms | Lean improvement |
| --- | ---: | ---: | ---: |
| wizard | 5683 | 4837 | 14.9% lower |
| modal | 4490 | 3509 | 21.8% lower |
| download | 4081 | 3492 | 14.4% lower |
| tabs | 8947 | 7792 | 12.9% lower |
| auth | 6273 | 5110 | 18.5% lower |

Raw evidence: [results-2026-11-23-extensions.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-2026-11-23-extensions.jsonl).

## 25. Historical analysis: 2026-11-23 results

*Historical analysis of the named run. It does not describe the 2026-11-25 results above.*

That run established two historical facts that remain useful for traceability:

- All four arms completed all five tasks, including the custom ARIA combobox, the click-triggered download, the multi-tab opener flow, and the profile restart.
- The lean profile cut OpenBrowser p50 by 12.9 to 21.8 percent compared with the same binary using the default extension bundle.

The 2026-11-25 results above supersede every ranking and latency claim from that run.

## 26. Historical results: 2026-11-24, final lanes (OpenBrowser 0.1.59.dev18)

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

These were the runs behind the comparison page at the time. Every run used a
fresh fixture server and an exact oracle check. All 75 cold runs, all 100 warm
runs, and all 40 paired runs passed.

### 26.1. Final lanes, 2026-11-24, dev18: Warm five-arm, N=5

Lower is better for latency ↓.

| Task | OpenBrowser MCP | OpenBrowser CLI | chrome-devtools-mcp | agent-browser | browser-use |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wizard` | 138 | 181 | 3452 | 1609 | 500 |
| `modal` | 267 | 330 | 2717 | 2105 | 1217 |
| `download` | 399 | 448 | 778 | 803 | 567 |
| `tabs` | 159 | 221 | 1458 | 1179 | 226 |

Raw evidence: [results-warm-five-arm-final-2026-11-24-n5.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-five-arm-final-2026-11-24-n5.jsonl).

### 26.2. Final lanes, 2026-11-24, dev18: Warm paired, N=10

Lower is better for latency ↓.

| Task | OpenBrowser MCP | Closest competitor | Pairs won | Paired delta p50 |
| --- | ---: | --- | ---: | ---: |
| `wizard` | 163.087 ms | browser-use 474.067 ms | 10/10 | 322.994 ms |
| `modal` | 255.657 ms | browser-use 803.270 ms | 10/10 | 517.974 ms |
| `tabs` | 137.929 ms | browser-use 221.003 ms | 10/10 | 84.434 ms |
| `download` | 387.938 ms | agent-browser 480.798 ms | 10/10 | 78.613 ms |

Raw evidence:

- [results-paired-wizard-mcp-vs-browseruse-final-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-wizard-mcp-vs-browseruse-final-2026-11-24-n10.json)
- [results-paired-modal-mcp-vs-browseruse-final-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-modal-mcp-vs-browseruse-final-2026-11-24-n10.json)
- [results-paired-tabs-mcp-vs-browseruse-final-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-tabs-mcp-vs-browseruse-final-2026-11-24-n10.json)
- [results-paired-download-mcp-vs-agentbrowser-final-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-download-mcp-vs-agentbrowser-final-2026-11-24-n10.json)

### 26.3. Final lanes, 2026-11-24, dev18: Cold six-arm, N=3

Lower is better for latency ↓.

| Task | OpenBrowser MCP | OpenBrowser CLI | chrome-devtools-mcp | agent-browser | browser-use |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wizard` | 1654 | 2028 | 4439 | 1776 | 1735 |
| `modal` | 1801 | 2682 | 3639 | 2405 | 1570 |
| `download` | 1785 | 1928 | 1733 | 1663 | 1486 |
| `tabs` | 1543 | 1639 | 2412 | 1674 | 1074 |
| `auth` | 2980 | 3210 | 3949 | 2712 | 2996 |

Raw evidence: [results-cold-six-arm-final-2026-11-24-n3.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-six-arm-final-2026-11-24-n3.jsonl).

The cold lane is a historical startup lane with N=3. Do not treat it as a
statistical claim. The 2026-11-25 cold results above supersede it.

### 26.4. Final lanes, 2026-11-24, dev18: MCP arm

`openbrowser-mcp` and `openbrowser-mcp-fast` drive the package's own MCP server
over the stateless `2026-07-28` protocol. The client in
`adapters/mcp_modern.py` sends one `server/discover` probe and stamps every
request with the `_meta` envelope. The arm calls `create_context`,
`execute_code`, and `close_context`. The warm lane keeps one context so one
browser serves every task.

## 27. Historical results: 2026-11-24, second final lanes (OpenBrowser 0.1.59.dev28)

*Historical run. It is retained for traceability. The 2026-11-25 results above supersede it for current claims.*

That historical rerun followed the lazy namespace imports, deferred recording
watchdog, removed redirect sleep, and headless shell. All 75 cold, 100 warm,
and 40 paired runs passed their exact oracles.

### 27.1. Second final lanes, 2026-11-24, dev28: Cold six-arm, N=3

Lower is better for latency ↓.

| Task | OpenBrowser MCP | OpenBrowser CLI | chrome-devtools-mcp | agent-browser | browser-use |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wizard` | 1387 | 1345 | 4409 | 1608 | 1367 |
| `modal` | 1506 | 1559 | 3704 | 2250 | 1628 |
| `download` | 1668 | 1707 | 1690 | 1645 | 1442 |
| `tabs` | 1394 | 1446 | 2401 | 1503 | 1048 |
| `auth` | 2407 | 2644 | 3735 | 2487 | 2711 |

Raw evidence: [results-cold-six-arm-final3-2026-11-24-n3.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-six-arm-final3-2026-11-24-n3.jsonl).

### 27.2. Second final lanes, 2026-11-24, dev28: Warm five-arm, N=5

Lower is better for latency ↓.

| Task | OpenBrowser MCP | OpenBrowser CLI | chrome-devtools-mcp | agent-browser | browser-use |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wizard` | 117 | 160 | 3510 | 573 | 464 |
| `modal` | 255 | 302 | 2776 | 1198 | 1164 |
| `download` | 397 | 443 | 784 | 498 | 551 |
| `tabs` | 130 | 189 | 1494 | 417 | 174 |

Raw evidence: [results-warm-five-arm-final2-2026-11-24-n5.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-five-arm-final2-2026-11-24-n5.jsonl).

### 27.3. Second final lanes, 2026-11-24, dev28: Warm paired, N=10

Lower is better for latency ↓.

| Task | OpenBrowser MCP | Closest competitor | Pairs won | Paired delta p50 |
| --- | ---: | --- | ---: | ---: |
| `wizard` | 121.966 ms | browser-use 446.056 ms | 10/10 | 330.459 ms |
| `modal` | 267.286 ms | browser-use 806.803 ms | 10/10 | 519.428 ms |
| `tabs` | 140.130 ms | browser-use 220.321 ms | 10/10 | 75.767 ms |
| `download` | 393.629 ms | agent-browser 482.813 ms | 10/10 | 89.421 ms |

Raw evidence:

- [results-paired-wizard-mcp-vs-browseruse-final2-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-wizard-mcp-vs-browseruse-final2-2026-11-24-n10.json)
- [results-paired-modal-mcp-vs-browseruse-final2-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-modal-mcp-vs-browseruse-final2-2026-11-24-n10.json)
- [results-paired-tabs-mcp-vs-browseruse-final2-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-tabs-mcp-vs-browseruse-final2-2026-11-24-n10.json)
- [results-paired-download-mcp-vs-agentbrowser-final2-2026-11-24-n10.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-paired-download-mcp-vs-agentbrowser-final2-2026-11-24-n10.json)
