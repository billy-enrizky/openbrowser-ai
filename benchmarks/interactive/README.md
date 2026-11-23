# Interactive browser benchmark

Deterministic local benchmark for scripted interactive browser tasks. No LLM,
no cloud browser, no AWS, and no external network. Every task is scored by a
server-side oracle, not by the adapter's self-report.

## Arms

| Arm | Version | Native scripted surface | Browser |
| --- | --- | --- | --- |
| `openbrowser` | OpenBrowser CLI `0.1.58.dev4+g36ccf113b` | One `openbrowser-ai -c` Python cell per task | System Chrome 158.0.8127.36 through `channel=chrome` |
| `chrome-devtools-mcp` | `chrome-devtools-mcp` 1.13.0 | MCP stdio, 33 tools, `pageId` routing | System Chrome 158.0.8127.36 |
| `agent-browser` | `agent-browser` 0.43.0 | CLI commands, one process per command | System Chrome 158.0.8127.36 |
| `browser-use` | `browser-use` 1.0.8 (`browser-harness` 0.1.13) | Piped Python in one CLI process | Dedicated system Chrome 158.0.8127.36 |

All four arms use a fresh profile per run, a 1280x800 viewport, and local
fixtures only. Every Chrome launch includes `--use-mock-keychain`. The harness
checks the actual browser process command line and fails the run if the flag is
missing.

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
new `HOME` and `TMPDIR`. The OpenBrowser daemon is stopped and removed before
every run. `agent-browser` sessions are closed with `close --all`.
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

## Results, 2026-11-23

macOS arm64, Chrome `158.0.8127.36`, 3 measured samples per arm and task, no
warmups. 60 of 60 runs passed their oracle. All 60 runs reported
`keychain_verified: true`.

| Task | Arm | OK | p50 ms | p95 ms | Tool calls | Process starts | Output bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wizard | openbrowser | 3/3 | 6054 | 6428 | 1 | 1 | 18 |
| wizard | chrome-devtools-mcp | 3/3 | 4459 | 4582 | 22 | 1 | 2865 |
| wizard | agent-browser | 3/3 | 1759 | 1795 | 17 | 17 | 5607 |
| wizard | browser-use | 3/3 | 1506 | 1521 | 15 | 1 | 48 |
| modal | openbrowser | 3/3 | 4872 | 4935 | 1 | 1 | 31 |
| modal | chrome-devtools-mcp | 3/3 | 3683 | 3772 | 17 | 1 | 1512 |
| modal | agent-browser | 3/3 | 2473 | 2496 | 13 | 13 | 5024 |
| modal | browser-use | 3/3 | 2265 | 2359 | 11 | 1 | 61 |
| download | openbrowser | 3/3 | 4657 | 4951 | 1 | 1 | 552 |
| download | chrome-devtools-mcp | 3/3 | 1763 | 2087 | 5 | 1 | 490 |
| download | agent-browser | 3/3 | 1812 | 1857 | 3 | 3 | 1155 |
| download | browser-use | 3/3 | 1509 | 1605 | 4 | 1 | 54 |
| tabs | openbrowser | 3/3 | 9141 | 9272 | 1 | 1 | 59 |
| tabs | chrome-devtools-mcp | 3/3 | 2466 | 2506 | 12 | 1 | 1077 |
| tabs | agent-browser | 3/3 | 1663 | 1713 | 13 | 13 | 5110 |
| tabs | browser-use | 3/3 | 1089 | 1404 | 12 | 1 | 102 |
| auth | openbrowser | 3/3 | 5916 | 5985 | 2 | 2 | 66 |
| auth | chrome-devtools-mcp | 3/3 | 3927 | 4152 | 14 | 2 | 1064 |
| auth | agent-browser | 3/3 | 2720 | 2811 | 10 | 11 | 3866 |
| auth | browser-use | 3/3 | 2943 | 2968 | 8 | 2 | 124 |

Raw evidence: `results/results-2026-11-23.jsonl`.

## What the results show

Capability: all four arms completed all five tasks. That includes the custom
ARIA combobox, the click-triggered download, the multi-tab opener flow, and
the profile restart. This is a correctness result, not a speed result.

OpenBrowser strengths in this smoke run:

- One control-plane call per task. The whole wizard, modal, tabs, and auth
  flow runs inside one `-c` cell, so the agent does not pay a round trip per
  action.
- Smallest output on four of five tasks. The final JSON line is 18 to 552
  bytes. That is the useful part for an agent, even though it moves work into
  the Python cell.
- Download and auth passed without additional tools. Chrome-devtools-mcp
  needed Chrome download preferences; agent-browser and browser-use passed
  through their own native download and profile mechanisms.

## Top three OpenBrowser product gaps exposed

1. Cold-start latency is the largest gap. OpenBrowser is the slowest arm on
   all five tasks: p50 `6054` ms on wizard, `4872` ms on modal, `4657` ms on
   download, `9141` ms on tabs, and `5916` ms on auth. Browser-use is
   `1089` ms on tabs and agent-browser is `1663` ms. The harness uses a fresh
   daemon and a fresh Chrome profile per run, so the number includes the full
   Python daemon boot and Chrome start. A thin client path, lazy daemon boot,
   or a prewarmed daemon would close part of this gap without changing task
   logic.

2. There is no first-class interactive CLI. Every OpenBrowser task is a
   Python `-c` cell that scans `browser.get_browser_state_summary()` for
   indexes. Agent-browser exposes `open`, `fill`, `click`, `check`, `select`,
   `tab`, `download`, `state`, and `get`. Chrome-devtools-mcp exposes typed
   MCP tools. The gap is developer and agent ergonomics: OpenBrowser scripts
   are longer, harder to compose, and harder for a coding agent to verify than
   a sequence of typed commands.

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
  --output benchmarks/interactive/results/results-2026-11-23.jsonl
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
