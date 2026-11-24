# Interactive browser benchmark

Deterministic local benchmark for scripted interactive browser tasks. No LLM,
no cloud browser, no AWS, and no external network. Every task is scored by a
server-side oracle, not by the adapter's self-report.

## Arms

| Arm | Version | Native scripted surface | Browser |
| --- | --- | --- | --- |
| `openbrowser` | OpenBrowser CLI `0.1.58.dev67+gfc17d3201.d20261124` | One `openbrowser-ai -c` Python cell per task | System Chrome 158.0.8127.36 through `channel=chrome` |
| `openbrowser-fast` | OpenBrowser CLI `0.1.58.dev67+gfc17d3201.d20261124` | One `openbrowser-ai -c` Python cell with CSS helpers for fixture ids | System Chrome 158.0.8127.36 through `channel=chrome` |
| `chrome-devtools-mcp` | `chrome-devtools-mcp` 1.13.0 | MCP stdio, 33 tools, `pageId` routing | System Chrome 158.0.8127.36 |
| `agent-browser` | `agent-browser` 0.43.0 | CLI commands, one process per command | System Chrome 158.0.8127.36 |
| `browser-use` | `browser-use` 1.0.8 (`browser-harness` 0.1.13) | Piped Python in one CLI process | Dedicated system Chrome 158.0.8127.36 |

All five arms use a fresh profile per run, a 1280x800 viewport, and local
fixtures only. Every Chrome launch includes `--use-mock-keychain`. The harness
checks the actual browser process command line and fails the run if the flag is
missing.

The earlier four-arm warm results below used OpenBrowser CLI
`0.1.58.dev12+gc421f7eac`. The fast selector results use the build listed
above.

## Tasks

| Task | Route | Exact oracle |
| --- | --- | --- |
| `wizard` | `/wizard` | Exactly one submission with `Alice Bench`, `alice@example.com`, `CA`, `ON`, `pro`, 5 seats, terms accepted, quote `Q-1`, and one post for each of the three steps |
| `modal` | `/modal` | At least two modal opens, at least one Escape close, then `Project Atlas` selected through the ARIA combobox |
| `download` | `/download` | One request and one completion for `report.csv`, file size 2229 bytes, SHA-256 `293ce662323595edec422dd7b0abebc9884971c8d7928de483fb23620c0cf2e3` |
| `tabs` | `/tabs` | Detail tab loaded with `window.opener`, approve clicked once, original tab received `approved:42`, detail tab close observed, original title `Tabs Bench` |
| `auth` | `/login` | Exactly one login POST, at least two account views across a browser restart, account secret `ACCOUNT-SECRET-42` read in phase 2, password absent from output |

Per-run reset: each run starts a new `ThreadingHTTPServer` on an ephemeral
loopback port with new state, a new profile, a new download directory, and a
new `HOME`, `TMPDIR`, and `OPENBROWSER_SOCKET`. The OpenBrowser daemon is
stopped and removed before every run. `agent-browser` sessions are closed with
`close --all`.
`chrome-devtools-mcp` servers are terminated and their Chrome processes are
killed by profile path. `browser-use` daemons and dedicated Chrome processes
are killed after every run.

## Fair-run procedure

1. Pin every version and record the browser executable. The Chrome build in
   this run was `158.0.8127.36`.
2. Use the same viewport, 1280x800, for all arms: OpenBrowser `window_size`,
   chrome-devtools-mcp `--viewport 1280x800`, agent-browser
   `set viewport 1280 800`, browser-use `--window-size=1280,800`.
3. Use a fresh profile and a fresh fixture server per run. No cache. No
   external requests.
4. Run each task in a fresh process tree. Latency is wall clock from the first
   adapter call to the adapter return, including daemon and browser start.
   The oracle check runs after the timer stops, so verification is not charged
   to the adapter.
5. Score correctness first. A failed oracle scores zero regardless of latency.
   p50 and p95 are computed over successful runs only. A failure-inclusive
   latency is recorded in the raw JSONL.
6. Keep the raw JSONL. The README table is a summary, not the evidence.

## Metrics

- `p50` and `p95`: nearest-rank over successful runs. Three samples per arm
  and task in this smoke run. These are smoke numbers, not a statistical
  claim.
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

## OpenBrowser lean profile

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

## Results, 2026-11-23, lean profile

macOS arm64, Chrome `158.0.8127.36`, 3 measured samples per arm and task, no
warmups. 60 of 60 runs passed their oracle. All 60 runs reported
`keychain_verified: true`. All 15 OpenBrowser runs recorded
`openbrowser_extensions: false`.

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

Raw evidence: `results/results-2026-11-23-lean.jsonl`.

The original default-extension run remains at
`results/results-2026-11-23.jsonl`.

## Warm session lane, 2026-11-24

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
from `results-2026-11-23-lean.jsonl`. Warmup wall time by arm: OpenBrowser
6896 ms, chrome-devtools-mcp
3638 ms, agent-browser
3214 ms, browser-use
1874 ms.

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

All 48 warm runs passed their oracle. OpenBrowser improves over its own cold
lane, but it still trails agent-browser and browser-use on wizard, download,
and tabs. It beats chrome-devtools-mcp on modal and is close on wizard. The
tabs task is the largest gap: OpenBrowser 6170 ms warm versus browser-use
650 ms. This is a benchmark adapter and product API gap to address next, not a
universal warm-lane win.

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

- `results-warm-4arm-2026-11-24-n3.jsonl`
- `results-warm-4arm-2026-11-24-n3.summary.json`

## Warm session lane after the popup fix, 2026-11-24

The popup fix in PR #217 stops a `target="_blank"` click from waiting the full
five-second mouse-release bound. It checks for a new page target and fails
quickly when none appears. This run uses the same four arms, tasks, fixtures,
and Chromium build as the earlier warm lane.

The run uses 3 samples per arm and task. All 48 runs passed their oracle. Host
load was high during the run, and OpenBrowser warmup took 23.6 seconds, so treat
these values as shared-host smoke evidence, not a quiet-host p50.

| Task | OpenBrowser | chrome-devtools-mcp | agent-browser | browser-use |
| --- | ---: | ---: | ---: | ---: |
| `wizard` | 2607 | 3511 | 1231 | 723 |
| `modal` | 1421 | 2768 | 1604 | 787 |
| `download` | 924 | 785 | 589 | 720 |
| `tabs` | 1124 | 1485 | 918 | 335 |

OpenBrowser reduced warm `tabs` from 6170 ms to 1124 ms, a 5.49x improvement
against the earlier run. It now beats chrome-devtools-mcp on `wizard`, `modal`,
and `tabs`. It still trails browser-use on every task and trails agent-browser
on `wizard`, `download`, and `tabs`. The result is progress, not a universal
warm-lane win.

Generate the documentation chart from the summary:

```sh
uv run --with matplotlib python benchmarks/interactive/plot_results.py
```

The script writes `docs/images/native-interactive-warm-current-p50.png`.

Raw evidence:

- `results/results-warm-4arm-2026-11-24-after-popup-n3.jsonl`
- `results/results-warm-4arm-2026-11-24-after-popup-n3.summary.json`

## Fast selector arm, 2026-11-24

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

| Task | `openbrowser` | `openbrowser-fast` | `chrome-devtools-mcp` | `agent-browser` | `browser-use` |
| --- | ---: | ---: | ---: | ---: | ---: |
| wizard | 2164.780 | 204.323 | 3532.686 | 537.348 | 524.218 |
| modal | 1161.091 | 396.123 | 2775.570 | 1205.172 | 1167.143 |
| download | 798.839 | 575.519 | 781.001 | 480.877 | 585.277 |
| tabs | 994.065 | 984.681 | 1492.314 | 440.220 | 217.152 |

The cells show warm p50 milliseconds. `openbrowser-fast` was lower than the
semantic arm on wizard, modal, and download. The tabs result changed direction
between the paired run and the full lane. Its ranges overlap, so the data show
no measurable difference. `agent-browser` was faster on download and tabs. Download against `browser-use` is parity because the sample ranges overlap. `browser-use` was faster on tabs. This does not establish a universal win.

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

- `results-warm-fast-selector-final-ab-2026-11-24-n3.jsonl`
- `results-warm-fast-selector-final-ab-2026-11-24-n3.summary.json`
- `results-warm-fast-selector-final-all-2026-11-24-n3.jsonl`
- `results-warm-fast-selector-final-all-2026-11-24-n3.summary.json`

## OpenBrowser extension A/B, 2026-11-23

The A/B uses one binary, the same screenshot setting, and 3 samples per task.
Only `enable_default_extensions` changes. The default bundle adds 12.9 to 21.8
percent to the measured p50.

| Task | Default extensions p50 ms | Lean p50 ms | Lean improvement |
| --- | ---: | ---: | ---: |
| wizard | 5683 | 4837 | 14.9% lower |
| modal | 4490 | 3509 | 21.8% lower |
| download | 4081 | 3492 | 14.4% lower |
| tabs | 8947 | 7792 | 12.9% lower |
| auth | 6273 | 5110 | 18.5% lower |

Raw evidence: `results/results-2026-11-23-extensions.jsonl`.

## What the results show

Capability: all four arms completed all five tasks. That includes the custom
ARIA combobox, the click-triggered download, the multi-tab opener flow, and
the profile restart. This is a correctness result, not a universal speed win.

OpenBrowser results after the lean profile change:

- It beats chrome-devtools-mcp on `modal`, at p50 `3509` ms against `3805`
  ms.
- It is within 5.8 percent of chrome-devtools-mcp on `wizard`, at `4837` ms
  against `4570` ms.
- It remains slower than chrome-devtools-mcp on `download`, `tabs`, and
  `auth`, and slower than agent-browser and browser-use on every task.
- It keeps the smallest output on four of five tasks, 18 to 550 bytes.
- It uses one control-plane call per task, but `tabs` still takes `7792` ms
  because tab discovery, focus switching, and per-action overhead dominate.

The lean profile cuts OpenBrowser p50 by 12.9 to 21.8 percent compared with
the same binary using the default extension bundle. That is a real efficiency
gain, but it does not close the gap with the native command and Python arms.

## Top three OpenBrowser product gaps exposed

1. Cold start and per-action latency remain the largest gap. OpenBrowser p50
   is `4837` ms on wizard, `3509` ms on modal, `3492` ms on download, `7792`
   ms on tabs, and `5110` ms on auth. Browser-use is `1111` ms on tabs and
   agent-browser is `1638` ms. The harness uses a fresh daemon and a fresh
   Chrome profile per run. A thin client path, lazy daemon boot, or a
   prewarmed daemon would reduce part of this gap.

2. There is no first-class interactive CLI. Every OpenBrowser task is a
   Python `-c` cell that scans `browser.get_browser_state_summary()` for
   indexes. Agent-browser exposes `open`, `fill`, `click`, `check`, `select`,
   `tab`, `download`, `state`, and `get`. Chrome-devtools-mcp exposes typed
   MCP tools. OpenBrowser scripts are longer and harder for a coding agent to
   compose and verify than a sequence of typed commands.

3. Session and profile lifecycle is not a first-class surface. Auth reuse
   required `await browser.export_storage_state(...)`, a daemon stop, a wait
   for Chrome to release the profile directory, and a second daemon start.
   Early harness runs hit `daemon_start_failed` because `daemon stop` returned
   before Chrome exited. Agent-browser exposes `--profile` and
   `state save/load`; chrome-devtools-mcp exposes `--userDataDir`;
   browser-use persists a Chrome profile directly. OpenBrowser needs a
   supported `state save` and `state load` CLI or MCP surface, and `daemon
   stop` should not report completion before the browser and profile lock are
   released.

## Limitations

- Three samples per arm and task. The p95 values are indicative, not a
  statistical claim.
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

## Setup

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

## Run

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

## Files

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
| `results/` | Raw JSONL and summary JSON |

## Safety and cleanup

- Every browser launch includes `--use-mock-keychain`. The harness reads the
  actual Chrome command line and records `keychain_verified`.
- No Keychain entries are created, reset, or modified.
- Every adapter stops its daemon and kills its browser processes in a
  `finally` block. The runner also terminates any process whose command line
  contains the run's unique temp path.
- No AWS, cloud browser, paid API, or model provider is called.
