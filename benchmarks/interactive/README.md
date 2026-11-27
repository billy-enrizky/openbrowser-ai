# Interactive browser benchmark

<!-- toc:start -->
## Contents

- [1. How to read these numbers](#1-how-to-read-these-numbers)
- [2. Current results, 2026-11-27, rotated N=5](#2-current-results-2026-11-27-rotated-n5)
  - [2.1. Cold interactive](#21-cold-interactive)
  - [2.2. Warm interactive](#22-warm-interactive)
  - [2.3. Timer and cleanup boundary](#23-timer-and-cleanup-boundary)
- [3. Interactive resource lane, 2026-11-27](#3-interactive-resource-lane-2026-11-27)
  - [3.1. WebMCP capability lane](#31-webmcp-capability-lane)
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

<!-- toc:end -->

Deterministic local benchmark for scripted interactive browser tasks. No LLM,
no cloud browser, no AWS, and no external network. Every task is scored by a
server-side oracle.

## 1. How to read these numbers

- Cold means a fresh process tree, daemon or server, and browser for every task. It includes startup cost.
- Warm means each arm starts once and the browser stays open. Each task still runs against a fresh fixture server. The `auth` task is cold only because it restarts the browser.
- Latency is lower is better ↓. Memory is lower is better ↓. A speedup value is higher is better ↑.
- The current cold and warm lanes use N=5 per arm and task. Arm order was rotated.
- `OpenBrowser MCP` means the package MCP stdio arm. `OpenBrowser CLI` means the one-cell CLI arm. The Arms table lists the supported native variants.
- Each sample used an isolated HOME.
- The host Playwright cache was exposed to every arm.
- Lean MCP transport was the default.
- No AWS or cloud browser was used.
- Cold-lane host load was 4.082, 4.409, 4.002 before and 3.808, 3.759, 3.809 after on 18 logical CPUs. Warm-lane host load was 4.705, 4.282, 4.488 before and 2.818, 3.580, 4.160 after.
- Cold latency is time to result. The timer measures task completion and any in-task restart. Final adapter cleanup runs once after the timed interval for every arm. The task timer excludes that cleanup.
- A paired win means one arm was faster on an individual sample matched against the same task and fixture. Treat it as a sample comparison; run a significance test for statistical claims.

## 2. Current results, 2026-11-27, rotated N=5

Seven interfaces run in both lanes: OpenBrowser MCP, OpenBrowser CLI, Chrome DevTools MCP, agent-browser CLI, browser-use CLI, Playwright MCP, and Playwright CLI. Every cell has N=5, and the arm order rotates between samples. One machine and one pinned browser serve every arm: Chrome for Testing `159.0.8154.6`, revision 1255. OpenBrowser app `0.1.61.dev173+g9f630f7c6` starts at digest `847bbf45`.

### 2.1. Cold interactive

Cold means a fresh process tree, daemon or server, and browser for every task. All 175 of 175 runs passed the exact oracle and the mock-keychain check. Transport was lean. The host Playwright cache was exposed to every arm.

Latency (ms) ↓ **Lower is better**. Each cell is median / p95. N=5, so p95 equals the slowest observation.

| Task | OpenBrowser MCP | OpenBrowser CLI | Chrome DevTools MCP | agent-browser CLI | browser-use CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | 699.861 / 1146.867 | 795.955 / 824.244 | 4195.437 / 4249.135 | 1534.293 / 1681.829 | 1107.908 / 1120.683 | 941.537 / 949.036 | 1732.262 / 1853.461 |
| modal | 823.093 / 841.969 | 945.112 / 970.386 | 3421.781 / 3466.559 | 2132.221 / 2191.821 | 1313.729 / 1921.332 | 850.733 / 934.007 | 1709.790 / 1727.923 |
| download | 978.479 / 1036.389 | 1075.754 / 1105.994 | 1432.855 / 1455.832 | 1450.081 / 1633.871 | 1122.297 / 1159.887 | 938.340 / 951.259 | 1738.421 / 1783.139 |
| tabs | 722.224 / 768.955 | 812.648 / 814.743 | 2151.068 / 2207.647 | 1381.320 / 1450.256 | 766.152 / 779.134 | 669.406 / 691.612 | 1491.717 / 1561.706 |
| auth | 1140.787 / 1188.648 | 1365.922 / 1452.703 | 3475.949 / 3523.551 | 2441.667 / 2545.621 | 2404.065 / 2430.624 | 1038.182 / 1078.800 | 2889.772 / 2955.748 |

Against Playwright, both OpenBrowser surfaces win 63 of 100 cold pairs. OpenBrowser MCP wins 8 of 25 pairs against Playwright MCP and 25 of 25 against Playwright CLI. OpenBrowser CLI wins 5 of 25 pairs against Playwright MCP and 25 of 25 against Playwright CLI. OpenBrowser wins 13 of 20 median and 12 of 20 p95 comparisons against the two Playwright surfaces. Against the two Playwright surfaces, every cold loss is against Playwright MCP. Across the five non-OpenBrowser interfaces, OpenBrowser wins 42 of 50 cold median groups and 40 of 50 cold p95 groups.

Against Chrome DevTools MCP and agent-browser CLI, both OpenBrowser surfaces win every one of 25 pairs. Against browser-use CLI, OpenBrowser MCP wins 24 of 25 pairs and OpenBrowser CLI wins 20 of 25. Three cold group losses go to browser-use: the OpenBrowser CLI `tabs` median (812.648 ms against 766.152 ms), the OpenBrowser CLI `tabs` p95 (814.743 ms against 779.134 ms), and the OpenBrowser MCP `wizard` p95 (1146.867 ms against 1120.683 ms). OpenBrowser MCP `tabs` at 722.224 ms is lower than browser-use.

Retained cold losses against Playwright are listed in [the comparison page](https://docs.openbrowser.me/comparison) and in [results-seven-arm-2026-11-27-n5.scoreboard.md](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-seven-arm-2026-11-27-n5.scoreboard.md).

Raw evidence:

- [results-cold-seven-arm-2026-11-27-n5.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-seven-arm-2026-11-27-n5.jsonl)
- [results-cold-seven-arm-2026-11-27-n5.meta.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-seven-arm-2026-11-27-n5.meta.json)
- [results-cold-seven-arm-2026-11-27-n5.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-seven-arm-2026-11-27-n5.summary.json)
- [results-cold-seven-arm-2026-11-27-n5.provenance.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-seven-arm-2026-11-27-n5.provenance.json)

### 2.2. Warm interactive

Warm-session readiness uses the native session workflow with OpenBrowser app `0.1.61.dev173+g9f630f7c6`. Each arm starts once and every task gets a fresh fixture page. All 140 observations passed the adapter, exact oracle, keychain, and cleanup checks.

Latency (ms) ↓ **Lower is better**. Each cell is median / p95.

| Task | OpenBrowser MCP | OpenBrowser CLI | Chrome DevTools MCP | agent-browser CLI | browser-use CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | 54.394 / 85.800 | 93.174 / 109.643 | 3507.588 / 3521.351 | 445.017 / 451.559 | 522.913 / 631.407 | 359.009 / 380.117 | 561.216 / 565.046 |
| modal | 204.733 / 216.366 | 237.042 / 249.272 | 2770.595 / 2778.484 | 1180.606 / 1191.859 | 1346.820 / 1437.020 | 282.960 / 288.477 | 479.470 / 498.992 |
| download | 342.371 / 359.854 | 364.725 / 371.380 | 778.532 / 787.500 | 422.595 / 425.037 | 543.508 / 658.885 | 371.945 / 378.389 | 570.364 / 587.300 |
| tabs | 78.586 / 84.018 | 114.917 / 124.138 | 1492.026 / 1497.729 | 368.715 / 389.761 | 165.507 / 228.349 | 128.525 / 129.309 | 309.912 / 313.608 |

OpenBrowser MCP has the lowest warm p50 on all four tasks among all seven interfaces. OpenBrowser CLI is second on all four tasks. Against the five other interfaces, OpenBrowser MCP wins all 100 pairs and OpenBrowser CLI wins 98 of 100. The two OpenBrowser CLI losses are one `modal` sample and one `tabs` sample against Playwright MCP. The two OpenBrowser interfaces split their head-to-head pairs 20 to 0 in favor of MCP. OpenBrowser wins 16 of 16 median and 16 of 16 p95 comparisons against the two Playwright surfaces.

Raw data and method provenance:

- [results-warm-seven-arm-2026-11-27-n5.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-seven-arm-2026-11-27-n5.jsonl)
- [results-warm-seven-arm-2026-11-27-n5.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-seven-arm-2026-11-27-n5.summary.json)
- [results-warm-seven-arm-2026-11-27-n5.controller.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-seven-arm-2026-11-27-n5.controller.json)

### 2.3. Timer and cleanup boundary

The timer runs from the start of `arm.run` to its return. Cleanup runs once after that interval in `run_one` for every arm. An earlier harness version also ran cleanup inside `arm.run`, so every arm paid teardown twice and the timed interval included that teardown. The current lane removes the duplicate and keeps one verified cleanup outside the timer.

This is a harness change. The cold table above is the current measurement. An earlier harness version included teardown in the timed interval; the current lane excludes it.

## 3. Interactive resource lane, 2026-11-27

Memory and CPU: **Lower is better** ↓.

The cold lane runs 175 observations, five tasks by seven interfaces by five samples. The warm lane runs 140 observations, four tasks by seven interfaces by five samples. Every cell in both lanes passed the exact oracle, keychain check, resource gate, and cleanup. The launcher binding and application bytes were unchanged before and after each lane.

Cold browser child physical footprint (MiB) ↓ **Lower is better**.

| Task | OpenBrowser MCP | OpenBrowser CLI | Chrome DevTools MCP | agent-browser CLI | browser-use CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | 404.7 | 420.0 | 501.5 | 550.3 | 606.3 | 467.3 | 544.6 |
| modal | 401.8 | 423.3 | 509.6 | 539.4 | 602.4 | 441.7 | 524.5 |
| download | 392.0 | 412.4 | 474.1 | 524.3 | 481.1 | 431.8 | 504.8 |
| tabs | 399.3 | 417.9 | 491.4 | 549.2 | 503.9 | 447.8 | 527.2 |
| auth | 399.3 | 417.5 | 483.7 | 536.4 | 597.5 | 444.7 | 522.7 |

Cold native child-tree RSS (MiB) ↓ **Lower is better**. This is the memory score. The benchmark worker, controller, and fixture server are outside the score.

| Task | OpenBrowser MCP | OpenBrowser CLI | Chrome DevTools MCP | agent-browser CLI | browser-use CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | 720.4 | 746.5 | 862.7 | 1293.2 | 1281.9 | 806.5 | 918.3 |
| modal | 718.9 | 750.1 | 926.8 | 1298.4 | 1276.8 | 788.8 | 901.7 |
| download | 709.9 | 737.5 | 837.3 | 1279.0 | 1059.9 | 772.7 | 865.3 |
| tabs | 723.5 | 752.5 | 853.6 | 1287.1 | 1038.9 | 792.3 | 897.7 |
| auth | 727.4 | 749.1 | 846.4 | 1296.5 | 1273.6 | 797.7 | 902.0 |

Cold sampled native child CPU time (seconds) ↓ **Lower is better**. This is a sampled lower bound. The benchmark worker, controller, and fixture server are outside the score.

| Task | OpenBrowser MCP | OpenBrowser CLI | Chrome DevTools MCP | agent-browser CLI | browser-use CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | 1.151 | 1.239 | 1.722 | 1.644 | 1.720 | 1.199 | 1.918 |
| modal | 1.076 | 1.202 | 1.571 | 1.517 | 1.597 | 1.068 | 1.732 |
| download | 1.072 | 1.205 | 1.370 | 1.465 | 1.351 | 1.016 | 1.776 |
| tabs | 1.148 | 1.327 | 1.603 | 1.589 | 1.356 | 1.081 | 1.889 |
| auth | 1.929 | 2.115 | 2.613 | 2.842 | 2.646 | 1.785 | 3.534 |

Warm browser child physical footprint (MiB) ↓ **Lower is better**.

| Task | OpenBrowser MCP | OpenBrowser CLI | Chrome DevTools MCP | agent-browser CLI | browser-use CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | 404.8 | 402.4 | 494.2 | 485.8 | 574.0 | 451.1 | 534.3 |
| modal | 397.8 | 401.5 | 495.5 | 477.2 | 565.1 | 440.7 | 528.2 |
| download | 394.9 | 398.9 | 489.4 | 472.2 | 542.1 | 438.6 | 451.5 |
| tabs | 400.9 | 403.0 | 501.3 | 500.6 | 568.5 | 451.2 | 521.4 |

Warm process-tree RSS (MiB) ↓ **Lower is better**. This is the memory score.

| Task | OpenBrowser MCP | OpenBrowser CLI | Chrome DevTools MCP | agent-browser CLI | browser-use CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | 662.1 | 667.6 | 860.5 | 1012.0 | 1292.5 | 740.0 | 847.9 |
| modal | 656.2 | 662.4 | 862.2 | 1019.5 | 1243.4 | 730.8 | 850.2 |
| download | 652.0 | 659.1 | 861.0 | 1009.5 | 1156.8 | 713.9 | 841.7 |
| tabs | 658.1 | 664.4 | 861.1 | 1004.6 | 1191.1 | 736.8 | 845.0 |

Warm observed task CPU delta (seconds) ↓ **Lower is better**. This is a sampled lower bound.

| Task | OpenBrowser MCP | OpenBrowser CLI | Chrome DevTools MCP | agent-browser CLI | browser-use CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | 0.13 | 0.13 | 0.38 | 0.23 | 0.32 | 0.25 | 0.43 |
| modal | 0.09 | 0.08 | 0.17 | 0.11 | 0.24 | 0.12 | 0.28 |
| download | 0.07 | 0.07 | 0.09 | 0.07 | 0.19 | 0.07 | 0.24 |
| tabs | 0.13 | 0.13 | 0.21 | 0.18 | 0.28 | 0.17 | 0.35 |

Both OpenBrowser surfaces use less median RSS than each of the five other interfaces on every cold and warm task. OpenBrowser MCP has the lowest median RSS on every task. Median and maximum RSS win all 90 groups against the five non-OpenBrowser interfaces. At pair level, OpenBrowser wins 899 of 900 footprint and RSS pairs. The single loss is one warm `download` sample: OpenBrowser MCP at 634.125 MiB against Chrome DevTools MCP at 632.172 MiB. OpenBrowser MCP has lower cold RSS than OpenBrowser CLI in every pair.

CPU is a retained loss against Playwright MCP in the cold lane. The observed category split places the cold CPU difference in OpenBrowser Python client startup, import, and dispatch work, while Chrome CPU is generally lower for OpenBrowser. OpenBrowser MCP wins 7 of 25 cold CPU pairs and OpenBrowser CLI wins 0 of 25 against Playwright MCP. Against Playwright CLI, both OpenBrowser surfaces win 25 of 25 cold CPU pairs. Warm median CPU wins all 40 groups against the five non-OpenBrowser interfaces. Warm maximum CPU loses 3 groups on `download`.

CPU charts: [cold](https://github.com/billy-enrizky/openbrowser-ai/blob/main/docs/images/native-interactive-cold-cpu.png) and [warm](https://github.com/billy-enrizky/openbrowser-ai/blob/main/docs/images/native-interactive-warm-cpu.png). Raw data: [cold resources JSONL](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-resources-seven-arm-2026-11-27-n5.jsonl), [cold resource summary](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-resources-seven-arm-2026-11-27-n5.summary.json), [warm resources JSONL](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-resources-seven-arm-2026-11-27-n5.jsonl), [warm resource summary](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-resources-seven-arm-2026-11-27-n5.summary.json).

### 3.1. WebMCP capability lane

Pinned Chrome `159.0.8154.6` exposes `document.modelContext` only with `--enable-features=WebMCP`. The capability runner tests two tasks on OpenBrowser CLI `-c`, OpenBrowser MCP, Playwright CLI, and Playwright MCP. All eight task oracles pass for exact catalog discovery and invocation with 17 and 25 returning `{"content":[{"type":"text","text":"42"}]}`. This is a page-API bridge and makes no timing or memory claim. Run [webmcp_capability.py](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/webmcp_capability.py) without arguments to print the inert plan.

## 4. Arms

These interactive figures describe the configured native workflows. Every browser interface in the 2026-11-27 lanes uses the same pinned Chrome for Testing build, revision 1255, and complete executable bindings.

The harness registers seven interfaces. Native frontend selection and the explicit Python command are described in the [CLI frontend reference](https://docs.openbrowser.me/customize/integrations/native-cli). Each artifact records the installed build used for its observations.

| Arm | Native scripted surface | Browser |
| --- | --- | --- |
| `openbrowser-fast` | One `openbrowser-ai -c` Python cell with CSS helpers for fixture ids | Automatic local engine with `channel=chrome` |
| `openbrowser-mcp-fast` | MCP stdio `2026-07-28`: `create_context`, `execute_code`, `close_context`, with CSS helpers for fixture ids | Automatic local engine with `channel=chrome` |
| `chrome-devtools-mcp` | MCP stdio, `pageId` routing | System Chrome |
| `agent-browser` | CLI commands, one process per command | System Chrome |
| `browser-use` | Piped Python in one CLI process | Dedicated system Chrome |
| `playwright-mcp` | Pinned Playwright MCP native code tool | System Chrome |
| `playwright-cli` | Pinned Playwright CLI `run-code` | System Chrome |

Package and browser versions are recorded in each result artifact. They are not listed here because they change between runs. Cold runs use a fresh profile per run. Warm runs start each arm once and keep the browser open. All current lanes use a 1280x800 viewport and local fixtures only. Every Chrome launch includes `--use-mock-keychain`, and the harness checks the actual browser command line.

## 5. Tasks

| Task | Route | Exact oracle |
| --- | --- | --- |
| `wizard` | `/wizard` | Exactly one submission with `Alice Bench`, `alice@example.com`, `CA`, `ON`, `pro`, 5 seats, terms accepted, quote `Q-1`, and one post for each of the three steps |
| `modal` | `/modal` | At least two modal opens, at least one Escape close, then `Project Atlas` selected through the ARIA combobox |
| `download` | `/download` | One request and one completion for `report.csv`, file size 2229 bytes, SHA-256 `293ce662323595edec422dd7b0abebc9884971c8d7928de483fb23620c0cf2e3` |
| `tabs` | `/tabs` | Detail tab loaded with `window.opener`, approve clicked once, original tab received `approved:42`, detail tab close observed, original title `Tabs Bench` |
| `auth` | `/login` | Exactly one login POST, at least two account views across a browser restart, account secret `ACCOUNT-SECRET-42` read in phase 2, password absent from output |

The modal scripts wait for the page's filter result before selecting the option. The `auth` fast script fills the login form through CSS ids.

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

1. Pin every version, the canonical browser executable, and its SHA-256.
   Verify the actual browser process for every arm. A `channel=chrome` setting
   alone leaves executable selection to the package.
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
6. Keep the raw JSONL. The README table summarizes the raw JSONL evidence.

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
  values are indicative. Run a significance test for statistical claims. Older
  sections state their own sample count.
- Custom widget fidelity differs. chrome-devtools-mcp does not expose the
  ARIA listbox options in its accessibility snapshot, so the adapter uses its
  native `evaluate_script` fallback to click the option. The task still passes
  through the page's own selection handler.
- Auth mechanisms differ by product. OpenBrowser exports and reloads a
  storage state file; the other arms rely on a persistent Chrome profile.
  This measures each product's native session reuse path.
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

npm install --prefix \
  ~/.cache/openbrowser-interactive-bench/playwright-native-0.0.86-0.1.25 \
  @playwright/mcp@0.0.86 @playwright/cli@0.1.25

uv venv ~/.cache/openbrowser-interactive-bench/browser-use-venv --python 3.12
uv pip install --python ~/.cache/openbrowser-interactive-bench/browser-use-venv/bin/python \
  browser-use==1.0.8

UV_PROJECT_ENVIRONMENT=~/.cache/openbrowser-interactive-bench/openbrowser-env \
  uv sync --locked --all-extras
```

The local `chrome-devtools-mcp` binary is the same 1.13.0 package that
`npx -y chrome-devtools-mcp@1.13.0` would download. The harness uses the local
install to remove network and npx startup jitter from the measurement.

The Playwright adapters use these pinned local binaries. Override their paths
with `PLAYWRIGHT_MCP_BIN` and `PLAYWRIGHT_CLI_BIN` when needed. Both adapters
use the same system Chrome, 1280x800 viewport, and mock-keychain flag.
The shared configuration sets `timeouts.settle` to zero. Each task waits for
its exact readiness condition.

This setup uses a source installation. To measure the native frontend, set
`OPENBROWSER_BIN` to a verified platform-wheel installation. Keep that
installation and every other input fixed across all four lanes.

## 11. Run

Select the Python environment from the setup step. Read the Chrome version
before any observation so fresh resource workers reuse the same metadata.

```sh
BENCH_PYTHON="$HOME/.cache/openbrowser-interactive-bench/openbrowser-env/bin/python"
export OPENBROWSER_BIN="$HOME/.cache/openbrowser-interactive-bench/openbrowser-env/bin/openbrowser-ai"
export CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
export PLAYWRIGHT_BENCH_BROWSER_VERSION="$("$CHROME_BIN" --version)"
```

Run the four lanes in sequence. Each command includes both Playwright
interfaces. Cold lanes include all five tasks. Warm lanes include the four
tasks that preserve the browser session.

```sh
uv run --no-project --python "$BENCH_PYTHON" python -m benchmarks.interactive.run_benchmark \
  --arms openbrowser-fast openbrowser-mcp-fast chrome-devtools agent-browser \
    browser-use playwright-mcp playwright-cli \
  --tasks wizard modal download tabs auth --samples 5 --rotate-arms \
  --workdir /tmp/ob-cold --output /tmp/ob-cold-latency.jsonl

uv run --no-project --python "$BENCH_PYTHON" python -m benchmarks.interactive.warm_benchmark \
  --arms openbrowser-fast openbrowser-mcp-fast chrome-devtools agent-browser \
    browser-use playwright-mcp playwright-cli \
  --tasks wizard modal download tabs --samples 5 --rotate-arms \
  --workdir /tmp/ob-warm --output /tmp/ob-warm-latency.jsonl

uv run --no-project --python "$BENCH_PYTHON" python -m benchmarks.interactive.resource_compare \
  --arms openbrowser-fast openbrowser-mcp-fast chrome-devtools agent-browser \
    browser-use playwright-mcp playwright-cli \
  --tasks wizard modal download tabs auth --samples 5 \
  --workdir /tmp/ob-resource --output /tmp/ob-cold-resource.jsonl

uv run --no-project --python "$BENCH_PYTHON" python -m benchmarks.interactive.warm_resource_compare \
  --arms openbrowser-fast openbrowser-mcp-fast chrome-devtools agent-browser \
    browser-use playwright-mcp playwright-cli \
  --tasks wizard modal download tabs --samples 5 \
  --run-workdir /tmp/ob-wr --output /tmp/ob-warm-resource.jsonl
```

Require complete task grids, exact oracles, verified cleanup, and unchanged
source and installation identities before publishing. Resource records include
sampler provenance. Resource-instrumented wall time stays in the resource lane.

A focused task:

```sh
uv run --no-project --python "$BENCH_PYTHON" python -m benchmarks.interactive.run_benchmark \
  --tasks tabs --arms openbrowser-fast agent-browser --samples 1
```

Useful flags:

- `--arms openbrowser-fast openbrowser-mcp-fast chrome-devtools agent-browser browser-use playwright-mcp playwright-cli`
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
| `adapters/playwright_mcp.py` | Pinned Playwright MCP adapter and persistent session |
| `adapters/playwright_cli.py` | Pinned Playwright CLI adapter and persistent session |
| `adapters/playwright_common.py` | Shared Chrome configuration, secrets, and download checks |
| `adapters/playwright_tasks.py` | Interactive and scraping programs for both Playwright interfaces |
| `adapters/playwright_scrape_bridge.py` | Five-task scraping bridge with exact output oracles |
| `adapters/mcp_stdio.py` | Minimal newline-delimited JSON-RPC MCP client |
| `adapters/base.py` | Run context, process hygiene, dedicated Chrome, keychain verification |
| `run_benchmark.py` | Cold latency runner, JSONL writer, summary table |
| `warm_benchmark.py` | Warm latency runner with a persistent session |
| `resource_compare.py` | Cold resource runner with identity-fenced process attribution |
| `warm_resource_compare.py` | Warm resource runner with task CPU deltas and persistent workers |
| [results](https://github.com/billy-enrizky/openbrowser-ai/tree/main/benchmarks/interactive/results) | Raw JSONL and summary JSON |

## 13. Safety and cleanup

- Every browser launch includes `--use-mock-keychain`. The harness reads the
  actual Chrome command line and records `keychain_verified`.
- No Keychain entries are created, reset, or modified.
- Every adapter stops its daemon and kills its browser processes in a
  `finally` block. The runner also terminates any process whose command line
  contains the run's unique temp path.
- No AWS, cloud browser, paid API, or model provider is called.
