# Compare local navigation contracts

<!-- toc:start -->
## Contents

- [1. Prepare the runtimes](#1-prepare-the-runtimes)
- [2. Claim tiers and pins](#2-claim-tiers-and-pins)
- [3. Run after the source stops changing](#3-run-after-the-source-stops-changing)
- [4. Read the result](#4-read-the-result)
- [5. Limit the performance claim](#5-limit-the-performance-claim)

<!-- toc:end -->

Run the same navigation, JavaScript input assignment, and POST through OpenBrowser, raw CDP, Lightpanda, agent-browser, and browser-use. This diagnostic uses native local engines and fresh private profiles. It does not invoke an LLM, AWS, Docker, or a user's browser session.

## 1. Prepare the runtimes

Use Python 3.12 or later. The coordinator and each worker need `psutil` and `websockets>=15.0.1`. The OpenBrowser worker also needs the selected source revision's dependencies. Use an existing environment or prepare an isolated environment before measurement. The script never installs packages or downloads browsers.

The adapter versions used for the 2026-11-15 diagnostic were Lightpanda `1.0.4`, agent-browser `0.43.0`, browser-use `1.0.5`, and browser-harness `0.1.13`. That run is historical. The checked-in example pin uses browser-use `1.0.6`. A pinned run must refresh the pin from official metadata before anyone calls the result current. The browser-use adapter uses the `Browser('chromium')` API introduced in 1.0.5. A later incompatible API fails startup.

To prepare a separate browser-use environment, run these commands outside the measurement. The 2026-11-15 historical run used browser-use 1.0.5; to rebuild that historical environment, replace `browser-use==1.0.6` with `browser-use==1.0.5` in the command below. The current example pin uses 1.0.6.

```sh
uv venv "$TOOLS/browser-use-env" --python 3.12
uv pip install --python "$TOOLS/browser-use-env/bin/python" \
	'browser-use==1.0.6' 'browser-harness==0.1.13' \
	'psutil==7.2.2' 'websockets==15.0.1'
uv pip freeze --python "$TOOLS/browser-use-env/bin/python" > "$TOOLS/browser-use-freeze.txt"
```

Set `TOOLS` to a dedicated tools directory first. Obtain native Lightpanda and agent-browser binaries from their official release assets. Obtain Chrome for Testing from the Stable channel download manifest. Select assets for your operating system and architecture. Verify the release digests before execution. Do not point any profile or runtime directory at an existing user profile. The script exposes no profile option.

Use these official metadata endpoints to resolve and retain release pins before a run:

- [Lightpanda latest release and asset digests](https://api.github.com/repos/lightpanda-io/browser/releases/latest).
- [agent-browser npm release and integrity](https://registry.npmjs.org/agent-browser/latest).
- [agent-browser native release assets](https://api.github.com/repos/vercel-labs/agent-browser/releases/latest).
- [browser-use package version and file hashes](https://pypi.org/pypi/browser-use/json).
- [browser-harness 0.1.13 package metadata](https://pypi.org/pypi/browser-harness/0.1.13/json).
- [Chrome for Testing downloadable Stable channel](https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json).

For example, `curl -fsSL URL -o FILE.json` saves an endpoint response. Combine retained responses into a JSON object and pass its path with `--version-manifest`. The script records that object's contents and SHA-256. This evidence is optional and does not certify that a release remains current. Runtime versions, executable hashes, and installed worker package versions are recorded separately. Retain your full dependency freeze with the results.

## 2. Claim tiers and pins

A result is one of two tiers:

- `historical` (default): the run is retained, but it is never called current. No pin file is required.
- `pinned`: every requested, ready arm must match a pin. A pinned run is not automatically current. Calling it current requires checking the pins against official release metadata at capture time. The run fails verification if a pin is missing, a recorded version is missing, or a recorded version does not match.

A pinned run requires `--pins` with a canonical pin file:

```json
{
  "captured_utc": "2026-11-20T10:00:00Z",
  "pins": {
    "lightpanda": {"version": "1.0.4"},
    "agent-browser": {"version": "0.43.0"},
    "browser-use": {"version": "1.0.6"},
    "chrome": {"version": "130.0.6723.58"},
    "openbrowser": {"source_commit": "a9022ad4"}
  }
}
```

Every pin has a `version`, except OpenBrowser, which must use a `source_commit` prefix. OpenBrowser uses a `source_commit` prefix, because the worker imports the selected source tree. Chrome is a shared pin for every Chrome-based arm. A pinned run requires `captured_utc` in the pin file, exact version-token matches, and a verified clean OpenBrowser source tree for a pinned OpenBrowser run. The commit prefix must be at least seven characters. The harness retains the raw pin file, its SHA-256, the recorded executable and package versions, and the verification result. A pin only proves that the run matches the recorded versions. It does not prove that those versions are still the latest. Refresh the pin file from official metadata before a pinned run, and check the pins before calling the result current.

A pinned run:

```sh
uv run --no-project --offline --python "$NAV_BENCH_OPENBROWSER_PYTHON" \
	benchmarks/navigation_compare.py --output "$RESULTS/new-run" --samples 5 \
	--claim-tier pinned --pins "$RESULTS/pins.json"
```

Without `--claim-tier pinned`, the result is historical even when the versions happen to be current.

## 3. Run after the source stops changing

Run from the repository root. Set these variables to your existing native runtimes:

```sh
export NAV_BENCH_SOURCE="$PWD"
export NAV_BENCH_OPENBROWSER_PYTHON="$PWD/.venv/bin/python"
export NAV_BENCH_CHROME="$CHROME_EXECUTABLE"
export NAV_BENCH_LIGHTPANDA="$LIGHTPANDA_EXECUTABLE"
export NAV_BENCH_AGENT_BROWSER="$AGENT_BROWSER_EXECUTABLE"
export NAV_BENCH_BROWSER_USE_PYTHON="$TOOLS/browser-use-env/bin/python"

uv run --no-project --offline --python "$NAV_BENCH_OPENBROWSER_PYTHON" \
	benchmarks/navigation_compare.py --output "$RESULTS/new-run" --samples 5
uv run --no-project --offline --python "$NAV_BENCH_OPENBROWSER_PYTHON" \
	benchmarks/navigation_compare.py --verify "$RESULTS/new-run"
```

Set `RESULTS` to a results parent directory. Each output directory must be new. These commands use the existing Python environments without dependency synchronization. The script prepends the chosen worker interpreter's directory to its `PATH` and checks that OpenBrowser imports from `--source`.

Every executable variable also has a CLI option. Use `--source`, `--openbrowser-python`, `--chrome`, `--lightpanda`, `--agent-browser`, and `--browser-use-python` to override the corresponding environment variable.

All five arms are requested by default. Missing optional tools are recorded as `unavailable`, with zero successes and no latency value. They make the run and verification exit with code 1. An installed tool that fails startup is recorded as `failed`. Its unexecuted samples remain `not_run`. Invalid configuration exits with code 2.

To test only OpenBrowser and raw CDP, add `--arms raw-cdp openbrowser`. This produces a two-arm result. For a smoke check, use `--samples 1`. Use at least five samples per workload and arm before interpreting latency. The pure contract tests require no browsers:

```sh
uv run --no-sync --offline pytest tests/test_navigation_benchmark.py -q
```

## 4. Read the result

Each arm starts once and receives one fast-page warmup. Measured rounds rotate the arm order. Each round includes a fast page and a page with a deferred external script. The deferred script response waits `--delay` seconds, which defaults to `0.15`. Every page has a fresh nonce and `Cache-Control: no-store`.

Verification requires exactly one successful fast-page warmup per ready arm, including the same document, side-effect, and timed server-evidence checks as measured trials. Missing, duplicate, failed, or invalid warmups fail verification. Warmups remain excluded from measured latency summaries.

The timed interval includes all of this work:

1. Navigate through the adapter's load wait.
2. Assert the exact URL, title, nonce, complete document state, load-event marker, and deferred-script marker.
3. Assign the nonce to an input through JavaScript and await exactly one POST response.
4. Recheck the final document and retained input value.
5. Read the loopback server's audit snapshot and assert one successful document response, one POST response, and the delayed script response when required. Each required request must begin and finish inside the timed interval.

Assertions finish before the timer stops. The server audit and final observation add work beyond navigation latency. Later settling cannot repair a failed sample. A final server snapshot also detects duplicate side effects after a sample returns. Startup, warmup, ownership scans, and shutdown are outside measured latency. The input operation is DOM assignment.

`results.json` retains integer nanoseconds, observations, failures, timeout flags, server timestamps, source manifests, runtime versions, executable hashes, load averages, and process ownership. Load averages are `null` when the platform has no load-average API or the operating system reports that the value is unavailable. `samples.jsonl` retains each immediate trial result. `results.json` contains the final server snapshots. `sha256.json` covers the result files and worker logs. The verifier checks hashes, measured-attempt accounting, workload assertions, source stability, and recorded cleanup identities. Browser-engine CPU limiting and third-party release authentication sit outside its scope.

Each workload summary separates `expected`, `attempted`, `successful`, `failures`, `timeouts`, `unavailable`, and `not_run`. Timeouts are a subset of failures. Successful samples alone enter the median and range. Failures remain in the attempted denominator. Missing samples prevent successful verification.

The script creates a private run directory, sanitizes browser configuration variables, and gives each worker private home, configuration, socket, and profile directories. On POSIX, the run directory uses `/tmp` to keep Unix socket paths short. Chrome uses `--use-mock-keychain` so an isolated home cannot block on the host keychain. The agent-browser and browser-use releases already set that flag. The raw-CDP and OpenBrowser adapters add it explicitly.

Cleanup captures descendants and private PID-file identities without treating ancestry alone as ownership. Each process must carry the actual run token and private runtime and home paths. The run directory's ownership marker must match. An explicit Chrome profile argument must identify that runtime's private profile. Graceful shutdown has a ten-second deadline. Before each terminate or kill signal, the fallback rechecks that ownership evidence, the private runtime root, PID creation time, and exact command line. Unverified live processes are not signaled and prevent successful cleanup verification. The directory is deleted only after recorded processes are absent or their PIDs have been reused. On uncertain cleanup, the script retains the directory and reports its path. It never runs a global browser kill or daemon reap. Raw artifacts contain local paths and process command lines. Review them before publishing.

## 5. Limit the performance claim

The adapters do not have the same engine, IPC cost, or wait policy:

- `raw-cdp` uses a persistent Python WebSocket to Chrome and waits for `Page.loadEventFired`.
- `lightpanda` uses the same CDP adapter against the Lightpanda engine. Its Chrome compatibility version string is not its release version.
- `agent-browser` uses its native CLI and private daemon. `open` uses the release's default load strategy. Per-command CLI startup and IPC are inside the timer.
- `browser-use` uses `Browser('chromium')`, `goto_url`, and the prescribed `wait_for_load`. Browser-harness `0.1.13` polls document readiness every `0.3` seconds.
- `openbrowser` uses the selected local source's `BrowserSession.navigate_to` default wait and direct CDP evaluation.

The Chrome executable and mock-keychain isolation flag are shared across the Chrome arms. Other library launch defaults are not forced into one common configuration. Effective command lines remain in the ownership records. The deferred-script workload avoids making rendering-engine image support a prerequisite.

The script labels cross-tool performance inconclusive. No universal winner follows from these numbers. This is the `native-navigation` lane only. Its scope covers navigation latency. CodeAgent task success, MCP tool behavior, CLI or daemon behavior, agent framework behavior, cold-start latency, public-site compatibility, and production behavior belong to other lanes. Each result records `claim_scope` with those surfaces. Record unrelated host load, profile the limiter in a separate untimed diagnostic, and alternate source revisions for a before-and-after claim.

Compare only records with identical claim scope, harness revision, and timer boundaries. Smoke checks validate plumbing. Use at least five samples for timing evidence.
