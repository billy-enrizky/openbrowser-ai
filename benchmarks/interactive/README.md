# Interactive browser benchmark

<!-- toc:start -->
## Contents

- [1. How to read these numbers](#1-how-to-read-these-numbers)
- [2. Current results, 2026-11-25, rotated N=5](#2-current-results-2026-11-25-rotated-n5)
  - [2.1. Cold interactive](#21-cold-interactive)
  - [2.2. Warm interactive](#22-warm-interactive)
  - [2.3. Timer and cleanup boundary](#23-timer-and-cleanup-boundary)
- [3. Interactive resource lane, 2026-11-25](#3-interactive-resource-lane-2026-11-25)
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
- Cold-lane host load was 6.570, 4.896, 4.404 before and 6.729, 6.016, 4.994 after on 18 logical CPUs. Warm v2 host load and prepared-bytecode counts are recorded in its provenance file.
- Cold latency is time to result. The timer measures task completion and any in-task restart. Final adapter cleanup runs once after the timed interval for every arm. The task timer excludes that cleanup.
- A paired win means one arm was faster on an individual sample matched against the same task and fixture. Treat it as a sample comparison; run a significance test for statistical claims.

## 2. Current results, 2026-11-25, rotated N=5

### 2.1. Cold interactive

Cold means a fresh process tree, daemon or server, and browser for every task. All 125 of 125 runs passed the exact oracle and the mock-keychain check. Transport was lean. The host Playwright cache was exposed to every arm.

Lower is better for latency ↓.

| Task | OpenBrowser MCP p50 ms | OpenBrowser CLI p50 ms | chrome-devtools-mcp p50 ms | agent-browser p50 ms | browser-use p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 569 | 677 | 4312 | 1573 | 1111 |
| modal | 725 | 841 | 3471 | 2304 | 1744 |
| download | 860 | 963 | 1513 | 1484 | 1126 |
| tabs | 593 | 683 | 2220 | 1393 | 797 |
| auth | 1305 | 1604 | 3703 | 2506 | 2476 |

Both OpenBrowser arms have the lowest p50 on all five cold tasks against all three other frameworks. OpenBrowser MCP and OpenBrowser CLI each won every paired sample, 5 of 5, on every task against every other framework. The CLI `tabs` p50 is 683 ms against browser-use at 797 ms.

Raw evidence:

- [results-cold-five-arm-lean-2026-11-25-n5-final4.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final4.jsonl)
- [results-cold-five-arm-lean-2026-11-25-n5-final4.meta.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final4.meta.json)
- [results-cold-five-arm-lean-2026-11-25-n5-final4.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final4.summary.json)

### 2.2. Warm interactive

Warm-session readiness uses the native session workflow with OpenBrowser app `5a88fb97`. Each arm starts once and every task gets a fresh fixture server. All 100 observations passed the adapter, exact oracle, and keychain checks.

Latency (ms) ↓ **Lower is better**. Warm tables show milliseconds to three decimal places. The raw file keeps full precision.

| Task | OpenBrowser MCP p50 ms | OpenBrowser CLI p50 ms | chrome-devtools-mcp p50 ms | agent-browser p50 ms | browser-use p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 43.555 | 88.456 | 3441.495 | 451.787 | 471.800 |
| modal | 199.144 | 244.149 | 2716.979 | 1095.635 | 912.795 |
| download | 332.502 | 384.810 | 767.921 | 414.847 | 558.928 |
| tabs | 62.077 | 113.553 | 1453.474 | 359.101 | 194.579 |

Both OpenBrowser arms have the lowest warm p50 on all four tasks. Each arm won all 60 eligible pairs, for 120 of 120 total. The `auth` task is cold only because it restarts the browser.

Cold and resource tables use the release baseline. No cold-to-warm ratio is computed by default. Supply `--cold-results` only for a matching app and method.

Raw data and method provenance:

- [results-warm-five-arm-readiness-v2-2026-11-25-n5.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-five-arm-readiness-v2-2026-11-25-n5.jsonl)
- [results-warm-five-arm-readiness-v2-2026-11-25-n5.summary.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-five-arm-readiness-v2-2026-11-25-n5.summary.json)
- [results-warm-five-arm-readiness-v2-2026-11-25-n5.provenance.json](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-warm-five-arm-readiness-v2-2026-11-25-n5.provenance.json)

### 2.3. Timer and cleanup boundary

The timer runs from the start of `arm.run` to its return. Cleanup runs once after that interval in `run_one` for every arm. An earlier harness version also ran cleanup inside `arm.run`, so every arm paid teardown twice and the timed interval included that teardown. The current lane removes the duplicate and keeps one verified cleanup outside the timer.

This is a harness change. The cold table above is the current measurement. An earlier harness version included teardown in the timed interval; the current lane excludes it.

## 3. Interactive resource lane, 2026-11-25

Memory and CPU: **Lower is better** ↓.

The lane runs 125 observations, five tasks by five arms by five samples. All 125 passed the exact oracle, keychain check, resource gate, and cleanup, so every cell is N=5.

| Task | OpenBrowser MCP footprint MiB | OpenBrowser CLI footprint MiB | chrome-devtools-mcp footprint MiB | agent-browser footprint MiB | browser-use footprint MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 132.9 | 140.4 | 475.5 | 495.7 | 553.6 |
| modal | 134.2 | 150.7 | 472.6 | 481.2 | 553.0 |
| download | 126.7 | 142.8 | 455.9 | 477.2 | 433.9 |
| tabs | 129.6 | 145.8 | 463.2 | 477.5 | 433.4 |
| auth | 133.9 | 141.4 | 460.6 | 479.8 | 544.5 |

| Task | OpenBrowser MCP tree RSS MiB | OpenBrowser CLI tree RSS MiB | chrome-devtools-mcp tree RSS MiB | agent-browser tree RSS MiB | browser-use tree RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 402.7 | 406.9 | 786.0 | 1205.0 | 1201.5 |
| modal | 408.5 | 433.3 | 800.6 | 1202.2 | 1199.7 |
| download | 395.5 | 417.7 | 767.6 | 1188.8 | 977.4 |
| tabs | 407.0 | 431.3 | 773.4 | 1188.7 | 978.3 |
| auth | 411.3 | 416.6 | 772.5 | 1202.6 | 1192.3 |

| Task | OpenBrowser MCP CPU s | OpenBrowser CLI CPU s | chrome-devtools-mcp CPU s | agent-browser CPU s | browser-use CPU s |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 0.78 | 1.03 | 2.22 | 1.81 | 1.91 |
| modal | 0.78 | 0.93 | 2.03 | 1.65 | 1.81 |
| download | 0.76 | 0.95 | 1.83 | 1.65 | 1.53 |
| tabs | 0.79 | 0.95 | 2.08 | 1.74 | 1.45 |
| auth | 1.11 | 1.71 | 3.04 | 2.94 | 2.84 |

Both OpenBrowser arms have the lowest median on every task for all three metrics. Across 150 planned OpenBrowser-to-other-framework pairs per metric, both arms won all 150 footprint pairs, all 150 RSS pairs, and all 150 CPU pairs.

Raw data: [results-interactive-resources-2026-11-25-n5-final.jsonl](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/results/results-interactive-resources-2026-11-25-n5-final.jsonl).

## 4. Arms

The harness registers seven interfaces. Native frontend selection and the explicit Python command are described in the [CLI frontend reference](https://docs.openbrowser.me/customize/integrations/native-cli). Each artifact records the installed build used for its observations.

| Arm | Native scripted surface | Browser |
| --- | --- | --- |
| `openbrowser-fast` | One `openbrowser-ai -c` Python cell with CSS helpers for fixture ids | System Chrome through `channel=chrome` |
| `openbrowser-mcp-fast` | MCP stdio `2026-07-28`: `create_context`, `execute_code`, `close_context`, with CSS helpers for fixture ids | System Chrome through `channel=chrome` |
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
  --tasks wizard modal download tabs auth --samples 7 --rotate-arms \
  --workdir /tmp/ob-cold --output /tmp/ob-cold-latency.jsonl

uv run --no-project --python "$BENCH_PYTHON" python -m benchmarks.interactive.warm_benchmark \
  --arms openbrowser-fast openbrowser-mcp-fast chrome-devtools agent-browser \
    browser-use playwright-mcp playwright-cli \
  --tasks wizard modal download tabs --samples 7 --rotate-arms \
  --workdir /tmp/ob-warm --output /tmp/ob-warm-latency.jsonl

uv run --no-project --python "$BENCH_PYTHON" python -m benchmarks.interactive.resource_compare \
  --arms openbrowser-fast openbrowser-mcp-fast chrome-devtools agent-browser \
    browser-use playwright-mcp playwright-cli \
  --tasks wizard modal download tabs auth --samples 7 \
  --workdir /tmp/ob-resource --output /tmp/ob-cold-resource.jsonl

uv run --no-project --python "$BENCH_PYTHON" python -m benchmarks.interactive.warm_resource_compare \
  --arms openbrowser-fast openbrowser-mcp-fast chrome-devtools agent-browser \
    browser-use playwright-mcp playwright-cli \
  --tasks wizard modal download tabs --samples 7 \
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
