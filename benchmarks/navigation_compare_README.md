# Compare latest local browser adapters

This benchmark measures a shared local navigation task through OpenBrowser, raw Chrome DevTools Protocol (CDP), Lightpanda, agent-browser, and browser-use. A separate suite tests a small screenshot pixel contract. It does not use AWS, public websites, an LLM, Docker, or a user's browser profile.

The numbers cover only these versions, this local host, these adapters, and the documented fixture tasks. They cannot establish a win across every browser, task, or product dimension. Do not use these numbers to update the public MCP or CLI comparison page.

## What the results measure

The navigation suite uses a local loopback fixture. Each sample proves one successful fixture page request, one JavaScript field mutation, and one POST side effect. The delayed workload also proves one delayed script request.

- `navigation_ns` ends when the adapter returns after the shared `load` milestone.
- `elapsed_ns` includes the page checks, field mutation, POST, audit read, and task assertions.
- `post_navigation_ns` is the part of `elapsed_ns` after the adapter returns.
- The report gives the median, minimum, maximum, attempted work, successful work, and error count.
- A comparison is a clear pairwise win only when OpenBrowser's p50 is lower and its measured range does not overlap the competitor's range in both rounds.
- The screenshot suite reports capability only. It does not time screenshot calls or rank screenshot speed.

A navigation run uses two rounds, four competitors, six samples per workload and arm, and one warmup per arm. Round two reverses competitor order. Each pair keeps OpenBrowser and one competitor resident. The screenshot suite runs six nonce-bound attempts for OpenBrowser, raw CDP, and Lightpanda. It checks a 128 by 128 PNG and four hidden RGB probes. `unsupported` applies only when all six Lightpanda attempts fail the same pixel assertion, all six image requests pass, and both controls pass all six attempts.

## Prepare a private local result root

Run the commands from a clean candidate checkout. Commit the benchmark scripts before creating the candidate identity. Use a new `RESULT_ROOT` for every benchmark campaign.

```sh
export PROJECT_ROOT="$PWD"
export CANDIDATE_ROOT="$PROJECT_ROOT"
export RESULT_ROOT="$(mktemp -d /private/tmp/openbrowser-latest-navigation.XXXXXX)"
export SYSTEM_PATH="/usr/bin:/bin:/usr/sbin:/sbin"
export UV_BIN="$(command -v uv)"
chmod 700 "$RESULT_ROOT"
mkdir -m 700 "$RESULT_ROOT/home" "$RESULT_ROOT/home/config" \
  "$RESULT_ROOT/home/cache" "$RESULT_ROOT/import-probe"

safe() {
  env -i HOME="$RESULT_ROOT/home" PATH="$SYSTEM_PATH" TMPDIR=/private/tmp LANG=C \
    XDG_CONFIG_HOME="$RESULT_ROOT/home/config" \
    XDG_CACHE_HOME="$RESULT_ROOT/home/cache" \
    UV_CACHE_DIR="$RESULT_ROOT/uv-cache" \
    AWS_EC2_METADATA_DISABLED=true AWS_CONFIG_FILE=/dev/null \
    AWS_SHARED_CREDENTIALS_FILE=/dev/null "$@"
}
```

The `safe` function removes inherited credentials and cloud configuration. Its AWS variables disable accidental SDK lookup. The benchmark makes no AWS calls.

## Fetch the current competitor versions

Fetch official release metadata before each run. The metadata command retains the source responses, response hashes, release versions, and official asset digests. The download command checks each published digest before it accepts an executable. Chrome archive and executable hashes are local hashes, not official digests.

```sh
export BOOTSTRAP_PYTHON="/absolute/path/to/python3.12"
safe "$BOOTSTRAP_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_competitor_metadata.py" fetch \
  --result-root "$RESULT_ROOT" --output "$RESULT_ROOT/metadata"
safe "$BOOTSTRAP_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_competitor_metadata.py" download-tools \
  --result-root "$RESULT_ROOT" --metadata "$RESULT_ROOT/metadata" \
  --tools "$RESULT_ROOT/tools" --manifest "$RESULT_ROOT/tool-manifest.json"
```

Create isolated Python environments for OpenBrowser and browser-use. Install browser-use and browser-harness at the exact versions recorded by `metadata`. Freeze both environments. Let preflight reject any version or hash that differs from the saved metadata.

```sh
export CANDIDATE_PYTHON="$RESULT_ROOT/candidate-env/bin/python"
export BROWSER_USE_PYTHON="$RESULT_ROOT/browser-use-env/bin/python"

safe /usr/bin/env UV_PROJECT_ENVIRONMENT="$RESULT_ROOT/candidate-env" \
  "$UV_BIN" sync --frozen --project "$CANDIDATE_ROOT" --extra dev --python 3.12.12 --no-install-project
safe "$UV_BIN" venv --python 3.12.12 "$RESULT_ROOT/browser-use-env"
```

Set `CHROME`, `LIGHTPANDA`, and `AGENT_BROWSER` to the verified executable paths under `RESULT_ROOT/tools`. If the agent-browser release needs Node, use the absolute Node executable recorded by preflight.

For the macOS arm64 assets downloaded by the metadata command, use these paths:

```sh
export CHROME="$RESULT_ROOT/tools/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
export LIGHTPANDA="$RESULT_ROOT/tools/lightpanda"
export AGENT_BROWSER="$RESULT_ROOT/tools/agent-browser"
```

Create `candidate-freeze.txt` and `browser-use-freeze.txt` with `uv pip freeze`. The browser-use environment must pin the current metadata versions of browser-use and browser-harness. The candidate environment must use the selected source revision's lockfile.

```sh
(umask 077; safe "$UV_BIN" pip freeze --python "$CANDIDATE_PYTHON" > "$RESULT_ROOT/candidate-freeze.txt")
(umask 077; safe "$UV_BIN" pip freeze --python "$BROWSER_USE_PYTHON" > "$RESULT_ROOT/browser-use-freeze.txt")
```

## Seal the candidate and preflight

Create `candidate-identity.json` only after the benchmark files are committed and the source working tree is clean. The identity binds the candidate commit, runtime tree, interpreter paths, tool identities, and exact runner and verifier file hashes.

Create the candidate identity, prove the import, then seal the version manifest. Create an empty import-probe directory below `RESULT_ROOT` before the second command.

```sh
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_benchmark_preflight.py" identity \
  --source "$CANDIDATE_ROOT" --python "$CANDIDATE_PYTHON" \
  --browser-use-python "$BROWSER_USE_PYTHON" \
  --candidate-freeze "$RESULT_ROOT/candidate-freeze.txt" \
  --browser-use-freeze "$RESULT_ROOT/browser-use-freeze.txt" \
  --tool-manifest "$RESULT_ROOT/tool-manifest.json" \
  --output "$RESULT_ROOT/candidate-identity.json"
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_benchmark_preflight.py" source-import-proof \
  --source "$CANDIDATE_ROOT" --python "$CANDIDATE_PYTHON" \
  --cwd "$RESULT_ROOT/import-probe" --identity "$RESULT_ROOT/candidate-identity.json" \
  --output "$RESULT_ROOT/source-import-proof.json"
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_benchmark_preflight.py" seal \
  --source "$CANDIDATE_ROOT" --python "$CANDIDATE_PYTHON" \
  --browser-use-python "$BROWSER_USE_PYTHON" \
  --metadata "$RESULT_ROOT/metadata" --tools "$RESULT_ROOT/tools" \
  --candidate-freeze "$RESULT_ROOT/candidate-freeze.txt" \
  --browser-use-freeze "$RESULT_ROOT/browser-use-freeze.txt" \
  --tool-manifest "$RESULT_ROOT/tool-manifest.json" \
  --identity "$RESULT_ROOT/candidate-identity.json" \
  --source-import-proof "$RESULT_ROOT/source-import-proof.json" \
  --version-manifest "$RESULT_ROOT/version-manifest.json"
```

Run `latest_benchmark_preflight.py verify` before every smoke or full suite. Use the same sealed inputs and a new output named `preflight-<phase>.json` for each phase.

```sh
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_benchmark_preflight.py" verify \
  --source "$CANDIDATE_ROOT" --python "$CANDIDATE_PYTHON" \
  --browser-use-python "$BROWSER_USE_PYTHON" \
  --metadata "$RESULT_ROOT/metadata" --tools "$RESULT_ROOT/tools" \
  --candidate-freeze "$RESULT_ROOT/candidate-freeze.txt" \
  --browser-use-freeze "$RESULT_ROOT/browser-use-freeze.txt" \
  --tool-manifest "$RESULT_ROOT/tool-manifest.json" \
  --identity "$RESULT_ROOT/candidate-identity.json" \
  --version-manifest "$RESULT_ROOT/version-manifest.json" \
  --phase navigation-smoke --output "$RESULT_ROOT/preflight-navigation-smoke.json"
```

Use phase `screenshot-smoke` and output `preflight-screenshot-smoke.json` for the screenshot control. For a full baseline, fetch metadata into a new `metadata-check/navigation` directory, pass it with `--metadata-check`, and use phase `navigation-run` or `screenshot-run` with the matching output filename.

The preflight command exits nonzero if the source, interpreter, version, download digest, executable, freeze, or artifact path differs from the sealed identity. It rejects AWS credentials and writes only below `RESULT_ROOT`.

## Run and freeze both control smokes

Before the full baseline, run one OpenBrowser and raw-CDP navigation smoke and one OpenBrowser and raw-CDP screenshot smoke. Verify each smoke offline. The navigation smoke checks that the delayed fixture adds at least 100 milliseconds to each control's `navigation_ns`. It is sensitivity evidence, not a competitor ranking.

Before each smoke, run `latest_benchmark_preflight.py verify` with the matching phase name and output path. For example, create `preflight-navigation-smoke.json` with phase `navigation-smoke`. Create `preflight-screenshot-smoke.json` with phase `screenshot-smoke`.

```sh
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_navigation_baseline.py" smoke \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-navigation-smoke.json" \
  --version-manifest "$RESULT_ROOT/version-manifest.json" \
  --source "$CANDIDATE_ROOT" --python "$CANDIDATE_PYTHON" --chrome "$CHROME" \
  --output "$RESULT_ROOT/smoke/navigation"
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/summarize_latest_navigation.py" verify-smoke \
  --run-root "$RESULT_ROOT/smoke/navigation" \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-navigation-smoke.json"
```

Run the screenshot smoke with `screenshot_capability.py smoke`. Verify it with `screenshot_capability.py verify-smoke`. The screenshot smoke has no Lightpanda arm and produces no capability classification.

```sh
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/screenshot_capability.py" smoke \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-screenshot-smoke.json" \
  --version-manifest "$RESULT_ROOT/version-manifest.json" \
  --source "$CANDIDATE_ROOT" --python "$CANDIDATE_PYTHON" --chrome "$CHROME" \
  --output "$RESULT_ROOT/smoke/screenshot"
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/screenshot_capability.py" verify-smoke \
  --run-root "$RESULT_ROOT/smoke/screenshot" \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-screenshot-smoke.json"
```

After both smoke verifiers pass, create the immutable harness record. This command reruns both offline verifiers and records the hashes of the two smoke preflights and the two smoke manifests.

```sh
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_benchmark_preflight.py" freeze-harness \
  --source "$CANDIDATE_ROOT" \
  --identity "$RESULT_ROOT/candidate-identity.json" \
  --version-manifest "$RESULT_ROOT/version-manifest.json" \
  --tool-manifest "$RESULT_ROOT/tool-manifest.json" \
  --candidate-freeze "$RESULT_ROOT/candidate-freeze.txt" \
  --browser-use-freeze "$RESULT_ROOT/browser-use-freeze.txt" \
  --navigation-preflight "$RESULT_ROOT/preflight-navigation-smoke.json" \
  --screenshot-preflight "$RESULT_ROOT/preflight-screenshot-smoke.json" \
  --navigation-smoke "$RESULT_ROOT/smoke/navigation" \
  --screenshot-smoke "$RESULT_ROOT/smoke/screenshot" \
  --output "$RESULT_ROOT/frozen-harness.json"
```

Any runner, verifier, tool, dependency freeze, smoke, or smoke-preflight change requires a new candidate identity, both new smokes, and a new frozen harness.

## Run the latest-version baseline

Fetch current metadata into a new `metadata-check/navigation` directory immediately before the run. Run `latest_benchmark_preflight.py verify` with phase `navigation-run`, and pass that new metadata check directory with `--metadata-check`. Then run the fixed pairwise schedule:

```sh
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/latest_navigation_baseline.py" run \
  --frozen-harness "$RESULT_ROOT/frozen-harness.json" \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-navigation-run.json" \
  --version-manifest "$RESULT_ROOT/version-manifest.json" \
  --source "$CANDIDATE_ROOT" --python "$CANDIDATE_PYTHON" --chrome "$CHROME" \
  --lightpanda "$LIGHTPANDA" --agent-browser "$AGENT_BROWSER" \
  --browser-use-python "$BROWSER_USE_PYTHON" \
  --rounds 2 --samples 6 --output "$RESULT_ROOT/navigation"
```

Verify the raw artifacts offline and write a sanitized summary only after all eight pair blocks verify:

```sh
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/summarize_latest_navigation.py" verify \
  --run-root "$RESULT_ROOT/navigation" \
  --frozen-harness "$RESULT_ROOT/frozen-harness.json" \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-navigation-run.json" \
  --output "$RESULT_ROOT/navigation/summary.json" \
  --project-root "$PROJECT_ROOT" \
  --sanitized-output "$PROJECT_ROOT/local_docs/specs/2026-11-17-latest-navigation-benchmark/candidate-baseline-summary.md"
```

Run `screenshot_capability.py run` separately with six attempts per arm. Verify it with both `screenshot_capability.py verify` and `summarize_screenshot_capability.py verify`. The sanitized screenshot report contains no latency field. Do not merge it into the navigation score.

```sh
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/screenshot_capability.py" run \
  --frozen-harness "$RESULT_ROOT/frozen-harness.json" \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-screenshot-run.json" \
  --version-manifest "$RESULT_ROOT/version-manifest.json" \
  --source "$CANDIDATE_ROOT" --python "$CANDIDATE_PYTHON" \
  --chrome "$CHROME" --lightpanda "$LIGHTPANDA" \
  --output "$RESULT_ROOT/screenshot" --attempts-per-arm 6
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/screenshot_capability.py" verify \
  --run-root "$RESULT_ROOT/screenshot" \
  --frozen-harness "$RESULT_ROOT/frozen-harness.json" \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-screenshot-run.json"
safe "$CANDIDATE_PYTHON" "$CANDIDATE_ROOT/benchmarks/summarize_screenshot_capability.py" verify \
  --run-root "$RESULT_ROOT/screenshot" \
  --frozen-harness "$RESULT_ROOT/frozen-harness.json" \
  --candidate-identity "$RESULT_ROOT/candidate-identity.json" \
  --preflight "$RESULT_ROOT/preflight-screenshot-run.json" \
  --project-root "$PROJECT_ROOT" \
  --sanitized-output "$PROJECT_ROOT/local_docs/specs/2026-11-17-latest-navigation-benchmark/candidate-baseline-screenshot-capability.md"
```

The run artifacts stay private below `RESULT_ROOT`. Sanitized reports omit process identities, owner tokens, local result paths, environment values, and raw logs. Keep failed attempts and incomplete manifests. Never overwrite raw results.

## Optimize only from measured evidence

Do not edit `src/openbrowser` during benchmark construction or the baseline. If repeated unprofiled data shows a clear OpenBrowser loss, run an unreported profile and map its hot path to OpenBrowser source. A profile alone is not a latency measurement.

Test one source-backed hypothesis at a time. Keep a runtime change only if two independent counterbalanced runs each improve fast-task `navigation_ns` p50 by at least 10 percent, both runs pass the no-regression gate, and the full regression suite stays green. If no source-mapped bottleneck exists or the measurement spread exceeds the effect, stop without a runtime edit or speedup claim.

A benchmark-only change does not create a release. A common-navigation result is not an MCP or CLI result. Do not update `https://docs.openbrowser.me/comparison` from this harness.
