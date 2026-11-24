---
name: deep-research
description: |
  Conduct deep web research using the openbrowser-ai agent. Discover sources with ddgs, investigate sub-questions across multiple sources, and produce cited markdown and JSON reports.
  Trigger for /search, web search, source discovery, or requests to find sources. Also trigger for research, deep dives, investigations, evidence gathering, comparisons, literature reviews, briefings, or cited reports.
allowed-tools: Bash(openbrowser-ai:*) Bash(curl:*) Bash(uv:*) Bash(uvx:*) Bash(python3:*) Bash(sleep:*) Bash(cat:*) Bash(irm:*) Bash(mkdir:*) Bash(date:*) Read Write
---

# Deep Research

Drive `openbrowser-ai` to investigate a topic across multiple web sources and produce a cited markdown report plus structured JSON. Two modes:

- **flat synthesis** (default) -- decompose query into 3-7 sub-questions, dispatch one parallel sub-agent per sub-question (each owns one tab), merge into one cited report.
- **drilldown** (auto-detected from prompt phrasing: "deep dive", "exhaustive", "recursive", "drilldown", "thorough") -- same as flat, plus a second wave of parallel sub-agents on findings flagged `needs_depth=true`. Hard cap depth=2, max 3 follow-up sub-agents per parent.

Output paths (relative to current project root):

- `local_docs/research/YYYY-MM-DD-<slug>.md`
- `local_docs/research/YYYY-MM-DD-<slug>.json`

**Architecture (mandatory):** the orchestrating Claude session (the one running this skill) MUST dispatch parallel sub-agents via `/dispatching-parallel-agents`, one sub-agent per sub-question. Each sub-agent owns exactly ONE tab. Sub-agents do not open additional tabs. The orchestrator merges per-agent findings into one report.

Why one tab per sub-agent and not `asyncio.gather` over tabs in a single `-c` call: a single Python coroutine driving N tabs through one daemon serializes navigation events at the CDP layer, contends for the LLM-extraction worker, and cannot make independent decisions about pagination or follow-up clicks per tab. Dispatching real Claude sub-agents (each with its own context window and its own browser tab) gives true parallelism, independent reasoning per tab, and isolates failures so one bad page doesn't poison the rest.

Hard rules:

- One sub-agent = one tab. Sub-agents must NOT call `navigate(url, new_tab=True)` to spawn additional tabs.
- All sub-agents share the same dedicated run daemon and Chrome process. Browser tabs are isolated targets, but the daemon has one global agent focus. Every sub-agent call must switch to its own tab before navigating.
- Each sub-agent writes its findings to its own JSON file under `$DEEP_RESEARCH_PARTIAL_DIR/<slug>-NN.json`. The orchestrator reads and merges these.
- The orchestrator never drives tabs itself. It only plans, dispatches, merges, renders, verifies, cleans up.

If a first-wave sub-agent returns <2 findings, the orchestrator dispatches a Step 2b retry sub-agent with broader search strategy (alternative engines, query reformulation, lower thresholds). Still `-c`-only: the skill never calls `openbrowser-ai -p`.

Variables persist across `-c` calls in the daemon namespace.

**Session reuse:** Step 0 creates a dedicated socket and profile for this research run. If that exact daemon is already running from an earlier step, the skill reuses it and operates in NEW tabs. It never attaches to or restarts the user's default daemon. If no daemon exists for this socket, the skill auto-starts one on the first `-c` call.

Every factual claim in the report carries a footnote citation `[N]`. Verifier fails the run if uncited prose is found.

## Setup

Verify install:

```bash
openbrowser-ai --help
```

Install if missing:

```bash
# macOS / Linux
curl -fsSL https://openbrowser.me/install.sh | sh

# Windows PowerShell
irm https://openbrowser.me/install.ps1 | iex
```

No LLM API key required. The skill drives the daemon via `openbrowser-ai -c` only, which executes raw CDP / JS through the daemon's Python namespace and never invokes a model. (The `-p` "prompt mode" of the CLI is a separate code path that loads `get_llm()` and requires an OpenAI / Anthropic / Google key. This skill explicitly avoids `-p`.)

Set the headless env var so the daemon starts without a visible browser window. The daemon forces `headless=True` unless `OPENBROWSER_HEADLESS` is explicitly false; a config file alone does not opt into a visible window:

```bash
export OPENBROWSER_HEADLESS=true
```

Prepare output dir at the project root (NOT user home):

```bash
mkdir -p local_docs/research
```

## Workflow

### Step 0 -- Session check

Use a dedicated socket and profile for the research run so the user's existing daemon, tabs, and namespace are never restarted or touched. If this research run already started its own daemon, reuse it. Otherwise the next `-c` call auto-starts one.

Each shell block starts a new process, so bare `export` values do not carry across blocks. Step 0 creates a per-invocation run directory, writes a shell-escaped environment file, and prints its absolute path as `RUN_ENV_FILE=...`.

Every later shell block must begin by sourcing that exact path:

```bash
. "<RUN_ENV_FILE>"
```

Copy the printed path from Step 0. Do not substitute the project path or reuse another run's file.

```bash
RESEARCH_ROOT="$(pwd)"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$RESEARCH_ROOT/local_docs/research/.runs/$RUN_STAMP"
ENV_FILE="$RUN_DIR/env.sh"
mkdir -p "$RUN_DIR/partial" || { echo "Cannot create $RUN_DIR"; exit 1; }
{
    printf 'export OPENBROWSER_HEADLESS=true\n'
    printf 'export OPENBROWSER_SOCKET=%q\n' "${TMPDIR:-/tmp}/openbrowser-deep-research-${RUN_STAMP}.sock"
    printf 'export OPENBROWSER_DAEMON_PROFILE_DIR=%q\n' "${TMPDIR:-/tmp}/openbrowser-deep-research-profile-${RUN_STAMP}"
    printf 'export OPENBROWSER_DAEMON_PROFILE_ISOLATION=true\n'
    printf 'export OPENBROWSER_USER_DATA_DIR=\n'
    printf 'export DEEP_RESEARCH_RUN_DIR=%q\n' "$RUN_DIR"
    printf 'export DEEP_RESEARCH_PARTIAL_DIR=%q\n' "$RUN_DIR/partial"
    printf 'export DEEP_RESEARCH_PROJECT_ROOT=%q\n' "$RESEARCH_ROOT"
    printf 'export DEEP_RESEARCH_ENV_FILE=%q\n' "$ENV_FILE"
} > "$ENV_FILE" || { echo "Cannot write $ENV_FILE"; exit 1; }
[ -s "$ENV_FILE" ] || { echo "Empty run environment: $ENV_FILE"; exit 1; }
. "$ENV_FILE" || { echo "Cannot source $ENV_FILE"; exit 1; }
: "${OPENBROWSER_SOCKET:?dedicated socket was not loaded}"
case "$OPENBROWSER_SOCKET" in
    *openbrowser-deep-research-*) ;;
    *) echo "Refusing to continue without a dedicated research socket"; exit 1 ;;
esac

ensure_output=$(openbrowser-ai daemon ensure 2>&1) || { echo "$ensure_output"; exit 1; }
if printf '%s' "$ensure_output" | grep -q 'already matches'; then
    echo "Reusing this research run's daemon"
    DEEP_RESEARCH_REUSED=1
    DEEP_RESEARCH_OWNED=0
else
    echo "Started this research run's isolated daemon"
    DEEP_RESEARCH_REUSED=0
    DEEP_RESEARCH_OWNED=1
fi
printf 'export DEEP_RESEARCH_REUSED=%s\nexport DEEP_RESEARCH_OWNED=%s\n' \
    "$DEEP_RESEARCH_REUSED" "$DEEP_RESEARCH_OWNED" >> "$ENV_FILE"
export DEEP_RESEARCH_REUSED DEEP_RESEARCH_OWNED
warm_output=$(openbrowser-ai warm 2>&1) || { echo "$warm_output"; exit 1; }
printf 'Research browser warm: %s\n' "$warm_output"
echo "RUN_ENV_FILE=$ENV_FILE"
```

`OPENBROWSER_USER_DATA_DIR=` clears an inherited explicit profile so it cannot override the dedicated research profile. `daemon ensure` reuses a daemon only when its effective headless and profile settings match; otherwise it restarts this run's daemon.

Snapshot existing tabs so cleanup leaves them untouched:

```bash
openbrowser-ai -c - <<'EOF'
state = await browser.get_browser_state_summary()
_preexisting_tab_ids = {t.target_id for t in state.tabs} if state.tabs else set()
print(f"Pre-existing tabs: {len(_preexisting_tab_ids)}")
EOF
```

### Step 0.5 -- Run `/search` to discover sources

Run this step after Step 0 for `/search`, web search, or source-discovery requests. It uses the run directory that Step 0 created. It does not change the dedicated daemon or profile.

Set `SEARCH_QUERY` to the user's query. Run DDGS in an isolated `uvx` environment. The command does not install DDGS globally.

```bash
set -e
. "<RUN_ENV_FILE>"
SEARCH_QUERY="<USER_QUERY>"
DDGS_RAW_JSON="$DEEP_RESEARCH_RUN_DIR/ddgs-search-raw.json"
SEARCH_RESULTS="$DEEP_RESEARCH_RUN_DIR/search-results.json"
SEARCH_URLS="$DEEP_RESEARCH_RUN_DIR/search-urls.txt"
SCRAPE_JSONL="$DEEP_RESEARCH_RUN_DIR/scraped-pages.jsonl"
NORMALIZER="$DEEP_RESEARCH_RUN_DIR/normalize-ddgs-search.py"

cat > "$NORMALIZER" <<'PY'
import json
import sys
from urllib.parse import urlsplit, urlunsplit

raw_path, results_path, urls_path = sys.argv[1:]
with open(raw_path, encoding="utf-8") as source:
    rows = json.load(source)
if isinstance(rows, dict):
    rows = rows.get("results", [])
if not isinstance(rows, list):
    raise SystemExit("DDGS output is not a JSON result list")

results = []
seen = set()
for row in rows:
    if not isinstance(row, dict):
        continue
    candidate = str(row.get("href") or row.get("url") or "").strip()
    parts = urlsplit(candidate)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        continue
    url = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))
    if url in seen:
        continue
    seen.add(url)
    results.append({
        "title": str(row.get("title") or "").strip(),
        "url": url,
        "snippet": str(row.get("body") or row.get("snippet") or row.get("description") or "").strip(),
    })

with open(results_path, "w", encoding="utf-8") as output:
    json.dump(results, output, ensure_ascii=False, indent=2)
    output.write("\n")
with open(urls_path, "w", encoding="utf-8") as output:
    for result in results[:5]:
        output.write(result["url"] + "\n")
print(f"Saved {len(results)} unique results to {results_path}")
print(f"Selected {min(5, len(results))} page URLs in {urls_path}")
PY

[ -n "$SEARCH_QUERY" ] || { echo "Set SEARCH_QUERY before searching"; exit 1; }
if DDGS_OUTPUT=$(uvx ddgs text --query "$SEARCH_QUERY" --max_results 10 --output "$DDGS_RAW_JSON" 2>&1); then
    printf '%s\n' "$DDGS_OUTPUT"
else
    printf '%s\n' "$DDGS_OUTPUT"
    case "$DDGS_OUTPUT" in
        *"No results found"*)
            printf '[]\n' > "$SEARCH_RESULTS"
            : > "$SEARCH_URLS"
            echo "DDGS returned no results."
            ;;
        *)
            echo "DDGS search failed. Check the error before continuing."
            exit 1
            ;;
    esac
fi

if [ -s "$DDGS_RAW_JSON" ]; then
    python3 "$NORMALIZER" "$DDGS_RAW_JSON" "$SEARCH_RESULTS" "$SEARCH_URLS"
elif [ ! -f "$SEARCH_RESULTS" ]; then
    printf '[]\n' > "$SEARCH_RESULTS"
    : > "$SEARCH_URLS"
fi

if [ ! -s "$SEARCH_URLS" ]; then
    echo "DDGS returned no usable URLs. Do not scrape an empty URL list."
else
    openbrowser-ai scrape --urls-file "$SEARCH_URLS" --output "$SCRAPE_JSONL" --output-format jsonl
fi
```

The search result file is a JSON array. Each item has `title`, `url`, and `snippet`. The URL list contains the first five unique results. Edit that list before scraping if you want to select different pages.

If DDGS reports `429`, `rate limit`, or `too many requests`, wait 30 seconds. Retry once with five results and a new raw file:

```bash
set -e
. "<RUN_ENV_FILE>"
SEARCH_QUERY="<USER_QUERY>"
NORMALIZER="$DEEP_RESEARCH_RUN_DIR/normalize-ddgs-search.py"
SEARCH_RESULTS="$DEEP_RESEARCH_RUN_DIR/search-results.json"
SEARCH_URLS="$DEEP_RESEARCH_RUN_DIR/search-urls.txt"
SCRAPE_JSONL="$DEEP_RESEARCH_RUN_DIR/scraped-pages.jsonl"
sleep 30
DDGS_RAW_JSON="$DEEP_RESEARCH_RUN_DIR/ddgs-search-retry.json"
if DDGS_OUTPUT=$(uvx ddgs text --query "$SEARCH_QUERY" --max_results 5 --output "$DDGS_RAW_JSON" 2>&1); then
    printf '%s\n' "$DDGS_OUTPUT"
    python3 "$NORMALIZER" "$DDGS_RAW_JSON" "$SEARCH_RESULTS" "$SEARCH_URLS"
else
    printf '%s\n' "$DDGS_OUTPUT"
    case "$DDGS_OUTPUT" in
        *"No results found"*)
            printf '[]\n' > "$SEARCH_RESULTS"
            : > "$SEARCH_URLS"
            echo "DDGS returned no results."
            ;;
        *)
            echo "DDGS retry failed. Stop and report the error."
            exit 1
            ;;
    esac
fi
if [ -s "$SEARCH_URLS" ]; then
    openbrowser-ai scrape --urls-file "$SEARCH_URLS" --output "$SCRAPE_JSONL" --output-format jsonl
else
    echo "DDGS returned no usable URLs. Do not scrape an empty URL list."
fi
```

If the retry fails, stop the search workflow and report the error. Do not retry again. If DDGS returns no results, keep the empty result file and do not call the scraper.

The scraper writes one JSON object per line. Check each row's `ok`, `url`, `content`, and `error` fields before using it:

```bash
python3 - "$SCRAPE_JSONL" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    for line in source:
        if not line.strip():
            continue
        result = json.loads(line)
        if result.get("ok") and result.get("content"):
            print(f"OK {result.get('url')}: {len(result['content'])} characters")
        else:
            print(f"FAILED {result.get('url')}: {result.get('error')}")
PY
```

Read the scraped `content` before you cite a page. Do not cite search snippets. A snippet is not evidence until you scrape and read its source page. Cite the source page URL in the report.

### Step 1 -- Plan

Decompose the user query into sub-questions and pick the mode. Daemon namespace persists `_plan` across later `-c` calls.

```bash
openbrowser-ai -c - <<'EOF'
import json, re, datetime, os

QUERY = """<USER_QUERY>"""  # paste exact user query here

# Daemon CWD often != shell CWD. Hard-code the absolute project root.
# Set this to the shell CWD at the start of the run; do NOT rely on os.getcwd().
PROJECT_ROOT = "<ABSOLUTE_PATH_TO_PROJECT_ROOT>"  # e.g. /Users/foo/myproject

# Auto-detect mode
DRILL_RE = re.compile(r"\b(deep ?dive|exhaustive|recursive|drill ?down|thorough)\b", re.I)
mode = "drilldown" if DRILL_RE.search(QUERY) else "flat"

# Slug = first 60 chars, lowercase, non-alnum -> '-', collapse repeats
def slugify(s):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower())[:60]
    return s.strip("-") or "research"

today = datetime.date.today().isoformat()
slug = slugify(QUERY)
research_dir = os.path.join(PROJECT_ROOT, "local_docs", "research")
os.makedirs(research_dir, exist_ok=True)
base = os.path.join(research_dir, f"{today}-{slug}")
md_path, json_path = f"{base}.md", f"{base}.json"

# Bump suffix if collision
n = 2
while os.path.exists(md_path):
    md_path, json_path = f"{base}-{n}.md", f"{base}-{n}.json"
    n += 1

# Decompose: 3-7 sub-questions. Keep tight, non-overlapping, each answerable from web.
# This is a heuristic split; replace with your own decomposition for the actual query.
sub_questions = [
    # e.g.  "What is X?",
    #       "Who are the main actors / vendors / authors?",
    #       "What are recent (last 12 months) developments?",
    #       "What are the trade-offs / criticisms?",
    #       "What concrete numbers / benchmarks exist?",
]

assert 3 <= len(sub_questions) <= 7, "need 3-7 sub-questions"

_plan = {
    "query": QUERY,
    "mode": mode,
    "generated_at": datetime.datetime.now().astimezone().isoformat(),
    "sub_questions": sub_questions,
    "md_path": md_path,
    "json_path": json_path,
}
print(json.dumps(_plan, indent=2))
EOF
```

Edit the `QUERY`, `PROJECT_ROOT`, and `sub_questions` list before running. `PROJECT_ROOT` MUST be an absolute path: the daemon runs in its own working directory (usually wherever the daemon was first started), so relative paths land in the wrong place. Use the shell `pwd` output as the value. Verify output looks right before continuing.

### Step 2a -- Dispatch parallel sub-agents (one tab per agent)

The orchestrator (the Claude session running this skill) MUST invoke `/dispatching-parallel-agents` and dispatch one sub-agent per sub-question. All sub-agents share the same `openbrowser-ai` daemon. Each sub-agent owns exactly one tab, the one it opens at the start of its run.

Give every sub-agent the exact absolute `RUN_ENV_FILE` path printed by Step 0 and require it to source that file before any `openbrowser-ai` command. Without those values a sub-agent would talk to the user's default daemon.

**Hard rules for the orchestrator:**

- Send ONE message containing N parallel `Agent` tool calls, where N = `len(_plan["sub_questions"])`.
- Each sub-agent gets the prompt template below, parameterized with: `SUB_QUESTION`, `AGENT_INDEX` (zero-padded 2 digits, used in output filename), `PROJECT_ROOT` (absolute path).
- After dispatch, wait for all sub-agents to return. Do not begin Step 4 until every partial JSON file under `$DEEP_RESEARCH_PARTIAL_DIR/` is on disk.
- The orchestrator does NOT drive any tab in this step.

**Sub-agent prompt template** (copy into each `Agent` tool call's `prompt` argument):

```
You are a deep-research sub-agent. Your job: investigate ONE sub-question
in ONE Chrome tab and write findings to a JSON file.

Sub-question: <SUB_QUESTION>
Agent index: <AGENT_INDEX>
Project root: <PROJECT_ROOT>

Hard constraints:

- You own ONE tab. The tab is the one you open at the start of this task.
- NEVER pass new_tab=True to navigate(). Reuse your one tab for every page.
- Do not switch to other tabs.
- Do not call openbrowser-ai daemon stop. The orchestrator owns daemon lifecycle.
- Visit at least 3 result URLs from at least 3 different domains.
- For each URL, extract one finding with: claim (one sentence), exact url,
  supporting quote (verbatim, max 40 words), domain, confidence (low|medium|high),
  needs_depth (bool, true if a deeper follow-up would meaningfully sharpen the claim).
- Return at least 2 findings; 3 is ideal.
- Write the findings JSON array to:
  <DEEP_RESEARCH_PARTIAL_DIR>/agent-<AGENT_INDEX>.json

Workflow (run via openbrowser-ai -c - heredocs, all in this one bash session):

1. Open exactly one tab on Google search:
   openbrowser-ai -c - <<'EOF'
   await navigate("https://www.google.com/search?q=<URL_ENCODED_SUB_QUESTION>", new_tab=True)
   await wait(2)
   tab_id = browser.agent_focus.target_id
   _MY_TAB_<AGENT_INDEX> = tab_id[-4:]
   print(f"my tab: {_MY_TAB_<AGENT_INDEX>}")
   EOF

2. Scrape the SERP for top 5 result URLs.

3. For each of the top 3 results, switch to your tab first, then navigate IN THE
   SAME TAB (no new_tab=True), wait 2s, capture body innerText, pick the sentence
   with the most word-overlap vs the sub-question (40-280 chars, >3-letter keyword
   overlap >= 1).
   Every later -c call must begin with:
   await switch(_MY_TAB_<AGENT_INDEX>)

4. Write the findings JSON array to the partial file.

5. Print a one-line summary: "agent <AGENT_INDEX>: <N> findings written".

Return: a one-line confirmation that the partial JSON file was written, and
its absolute path. Do not include the findings in your text reply -- the
orchestrator will read them from disk.
```

The orchestrator's pre-step (run once before dispatching agents):

```bash
openbrowser-ai -c - <<'EOF'
import os
partial_dir = os.environ["DEEP_RESEARCH_PARTIAL_DIR"]
os.makedirs(partial_dir, exist_ok=True)
# Wipe stale partials from prior runs of the same query
for f in os.listdir(partial_dir):
    if f.startswith("agent-") and f.endswith(".json"):
        os.remove(os.path.join(partial_dir, f))
print(f"partial dir ready: {partial_dir}")
EOF
```

After all sub-agents return, the orchestrator collects findings:

```bash
openbrowser-ai -c - <<'EOF'
import os, json
global _findings
partial_dir = os.environ["DEEP_RESEARCH_PARTIAL_DIR"]

_findings = []
for fname in sorted(os.listdir(partial_dir)):
    if not (fname.startswith("agent-") and fname.endswith(".json")):
        continue
    path = os.path.join(partial_dir, fname)
    try:
        with open(path) as fh:
            arr = json.load(fh)
        if isinstance(arr, list):
            # normalize exact_url -> url (sub-agents may use either key)
            for item in arr:
                if "exact_url" in item and "url" not in item:
                    item["url"] = item.pop("exact_url")
            _findings.extend(arr)
    except Exception as e:
        print(f"skip {path}: {e}")

print(f"merged {len(_findings)} findings from {len(os.listdir(partial_dir))} partials")
EOF
```

If any sub-question's partial file is missing or has <2 findings, fall through to Step 2b for that one.

### Step 2b -- Retry weak sub-questions with broader sub-agents

If a sub-question came back with <2 findings, do NOT use `openbrowser-ai -p`. The `-p` mode requires an LLM API key (OpenAI / Anthropic / Google) because it runs the full Browser Agent loop with LLM-driven navigation,. This skill is `-c`-only by design: the daemon executes raw CDP / JS and needs no API key.

Identify weak sub-questions, then dispatch a fresh round of parallel sub-agents (same `/dispatching-parallel-agents` pattern as Step 2a) with a broader prompt. The retry sub-agents reuse the same single-tab discipline and write to `agent-retry-NN.json` partials.

```bash
openbrowser-ai -c - <<'EOF'
from collections import Counter
global _retry_subqs
counts = Counter(f.get("sub_question") for f in _findings)
_retry_subqs = [sq for sq in _plan["sub_questions"] if counts.get(sq, 0) < 2]
print(f"Retry targets: {len(_retry_subqs)}")
for sq in _retry_subqs:
    print(f"  - {sq}")
EOF
```

The orchestrator then dispatches one parallel sub-agent per weak sub-question with this retry prompt template (same single-tab rule, broader search strategy):

```
You are a deep-research RETRY sub-agent. Your job: investigate ONE sub-question
that the first-wave sub-agent could not satisfy. Use ONE Chrome tab and write
findings to a JSON file.

Sub-question: <SUB_QUESTION>
Agent index: <AGENT_INDEX> (filename agent-retry-<AGENT_INDEX>.json)
Project root: <PROJECT_ROOT>

Hard constraints:

- You own ONE tab. NEVER pass new_tab=True to navigate() after the first call.
- Use openbrowser-ai -c only. NEVER use openbrowser-ai -p (it requires an LLM
  API key and we explicitly avoid that path).
- The first-wave attempt failed: heuristic sentence picker returned <2 findings.
  This means the page text either had no high-overlap sentences for the question,
  or the SERP returned thin sources. Try one or more of:
  1. Reformulate the search query (try 2-3 alternative phrasings, pick the one
     with the best SERP).
  2. Search a different engine: try Bing or DuckDuckGo if Google was thin.
     URLs: https://www.bing.com/search?q=... or https://duckduckgo.com/?q=...
  3. Lower the sentence-length floor for the heuristic (e.g. 30 chars instead
     of 40), or accept partial-match sentences with overlap >= 1 word.
  4. For factual sub-questions ("when was X released"), check Wikipedia
     directly: https://en.wikipedia.org/wiki/Special:Search?search=...
- Same JSON shape as Step 2a sub-agents.
- Write to: <DEEP_RESEARCH_PARTIAL_DIR>/agent-retry-<AGENT_INDEX>.json
- Return at least 2 findings. If still <2 after the broader strategy, write
  whatever you got (even 0-1) and surface the limitation in your reply.
```

After all retry sub-agents return, merge their partials into `_findings`:

```bash
openbrowser-ai -c - <<'EOF'
import os, json
global _findings
partial_dir = os.environ["DEEP_RESEARCH_PARTIAL_DIR"]
extra = []
for fname in sorted(os.listdir(partial_dir)):
    if not (fname.startswith("agent-retry-") and fname.endswith(".json")):
        continue
    path = os.path.join(partial_dir, fname)
    try:
        with open(path) as fh:
            arr = json.load(fh)
        if isinstance(arr, list):
            for item in arr:
                if "exact_url" in item and "url" not in item:
                    item["url"] = item.pop("exact_url")
            extra.extend(arr)
    except Exception as e:
        print(f"skip {path}: {e}")
_findings.extend(extra)
print(f"Retry added {len(extra)} findings -> total {len(_findings)}")
EOF
```

`_findings` lives in the daemon namespace for the next steps.

### Step 3 -- Drilldown (drilldown mode only): second wave of parallel sub-agents

Skip this step in flat mode. In drilldown mode, dispatch a second wave of `/dispatching-parallel-agents` -- one sub-agent per `needs_depth=true` finding, max 3 follow-ups per parent sub-question. Each follow-up sub-agent owns ONE tab, exactly like Step 2a.

The orchestrator first picks targets:

```bash
openbrowser-ai -c - <<'EOF'
global _drill_targets
if _plan["mode"] != "drilldown":
    print("flat mode, skipping drilldown")
    _drill_targets = []
else:
    by_sq = {}
    for f in _findings:
        if f.get("needs_depth"):
            by_sq.setdefault(f["sub_question"], []).append(f)
    _drill_targets = []
    for sq, fs in by_sq.items():
        _drill_targets.extend(fs[:3])
    print(f"Drilldown targets: {len(_drill_targets)}")
    for t in _drill_targets:
        print(f"  - {t['claim'][:80]}  ({t['url']})")
EOF
```

Then dispatch one parallel sub-agent per drilldown target. Use the same single-tab-per-agent rule and the same partial-file output convention, but with this drilldown prompt template:

```
You are a deep-research drilldown sub-agent. Your job: investigate ONE claim
in ONE Chrome tab and write supporting/contradicting evidence to a JSON file.

Parent claim: <CLAIM>
Original source URL: <URL>
Agent index: <AGENT_INDEX> (use prefix "drill-" -> filename agent-drill-<AGENT_INDEX>.json)
Project root: <PROJECT_ROOT>

Hard constraints:

- You own ONE tab. NEVER pass new_tab=True to navigate() after the first call.
- Find at least 2 additional sources from at least 2 different domains
  (different from the original source URL above).
- needs_depth must be false on every finding you return (depth cap reached).
- Same JSON shape as Step 2a sub-agents: claim, url, quote, domain, confidence,
  needs_depth, plus add sub_question = "<PARENT_SUB_QUESTION>".
- Write to: <DEEP_RESEARCH_PARTIAL_DIR>/agent-drill-<AGENT_INDEX>.json
```

After all drilldown sub-agents return, merge their partials into `_findings`:

```bash
openbrowser-ai -c - <<'EOF'
import os, json
global _findings
partial_dir = os.environ["DEEP_RESEARCH_PARTIAL_DIR"]
extra = []
for fname in sorted(os.listdir(partial_dir)):
    if not (fname.startswith("agent-drill-") and fname.endswith(".json")):
        continue
    path = os.path.join(partial_dir, fname)
    try:
        with open(path) as fh:
            arr = json.load(fh)
        if isinstance(arr, list):
            for f in arr:
                f["needs_depth"] = False  # cap reached
                if "exact_url" in f and "url" not in f:
                    f["url"] = f.pop("exact_url")
            extra.extend(arr)
    except Exception as e:
        print(f"skip {path}: {e}")
_findings.extend(extra)
print(f"Drilldown added {len(extra)} findings -> total {len(_findings)}")
EOF
```

### Step 4 -- Aggregate and dedup

Merge findings, dedup by URL, build the source index that footnote numbers will point to.

```bash
openbrowser-ai -c - <<'EOF'
from urllib.parse import urlparse

# `global` so reassignment of daemon-namespace var doesn't shadow it
global _findings, _sources

# Normalize exact_url -> url (sub-agents may use either key)
for f in _findings:
    if "exact_url" in f and "url" not in f:
        f["url"] = f.pop("exact_url")

# Dedup by URL, keep first occurrence
seen_urls = {}
deduped = []
for f in _findings:
    u = (f.get("url") or "").strip()
    if not u:
        continue  # drop uncited findings entirely
    if u in seen_urls:
        continue
    seen_urls[u] = True
    deduped.append(f)

# Build sources index, 1-based ids
_sources = []
url_to_id = {}
for f in deduped:
    u = f["url"]
    if u not in url_to_id:
        sid = len(_sources) + 1
        url_to_id[u] = sid
        domain = f.get("domain") or urlparse(u).netloc
        _sources.append({"id": sid, "url": u, "domain": domain, "title": f.get("title", "")})
    f["source_id"] = url_to_id[u]

_findings = deduped
print(f"After dedup: {len(_findings)} findings, {len(_sources)} unique sources")
EOF
```

### Step 5 -- Render report

Write paired markdown + JSON. Every claim line in markdown ends with `[N]` cite. Summary cites top 3-5 sources.

```bash
openbrowser-ai -c - <<'EOF'
import json, datetime
from collections import defaultdict

# Group findings by sub-question (preserve plan order)
groups = defaultdict(list)
for f in _findings:
    groups[f["sub_question"]].append(f)

lines = []
lines.append(f"# Research: {_plan['query']}")
lines.append("")
gen = datetime.datetime.fromisoformat(_plan["generated_at"]).strftime("%Y-%m-%d %H:%M %Z").strip()
lines.append(f"Generated: {gen}  *  Mode: {_plan['mode']}  *  Sources: {len(_sources)}")
lines.append("")

# Summary: pick first finding from each sub-question, max 5 sentences
lines.append("## Summary")
lines.append("")
summary_sents = []
for sq in _plan["sub_questions"]:
    fs = groups.get(sq, [])
    if fs:
        f = fs[0]
        claim = " ".join(f["claim"].split()).rstrip(".")
        summary_sents.append(f"{claim}[{f['source_id']}].")
    if len(summary_sents) >= 5:
        break
lines.append(" ".join(summary_sents) or "_No findings._")
lines.append("")

# Per sub-question section
for sq in _plan["sub_questions"]:
    lines.append(f"## {sq}")
    lines.append("")
    fs = groups.get(sq, [])
    if not fs:
        lines.append("_No findings for this sub-question._")
        lines.append("")
        continue
    for f in fs:
        claim = " ".join(f["claim"].split()).rstrip(".")  # collapse all whitespace incl newlines
        lines.append(f"- {claim}[{f['source_id']}].")
        q = " ".join((f.get("quote") or "").split()).strip()
        if q:
            lines.append(f"  > {q}")
    lines.append("")

# Sources
lines.append("## Sources")
lines.append("")
for s in _sources:
    title = s.get("title") or s["domain"]
    lines.append(f"{s['id']}. [{title}]({s['url']})")
lines.append("")

md = "\n".join(lines)

with open(_plan["md_path"], "w") as fh:
    fh.write(md)

payload = {
    "query": _plan["query"],
    "mode": _plan["mode"],
    "generated_at": _plan["generated_at"],
    "sub_questions": _plan["sub_questions"],
    "findings": _findings,
    "sources": _sources,
}
with open(_plan["json_path"], "w") as fh:
    json.dump(payload, fh, indent=2, ensure_ascii=False)

print(f"Wrote {_plan['md_path']}")
print(f"Wrote {_plan['json_path']}")
EOF
```

### Step 6 -- Verify citations

Fail loud if any prose sentence outside headings, quotes, code, or source list lacks a `[N]` cite. Fix by adding the missing source then rerendering, never by deleting prose silently.

Do the verify in plain shell `python3` (not the daemon). `sys.exit` inside the daemon namespace kills the daemon.

```bash
python3 - "$PROJECT_ROOT" <<'EOF'
import re, sys, os

project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
research_dir = os.path.join(project_root, "local_docs", "research")

# Find latest report file -- use os.listdir + absolute path (glob fails on non-shell CWD)
files = sorted(
    [os.path.join(research_dir, f) for f in os.listdir(research_dir) if f.endswith(".md")],
    key=os.path.getmtime, reverse=True
)
if not files:
    print("No report found"); sys.exit(1)
md_path = files[0]

with open(md_path) as fh: md = fh.read()
md_no_code = re.sub(r"```.*?```", "", md, flags=re.S)

violations = []
in_sources = False
for i, line in enumerate(md_no_code.splitlines(), 1):
    s = line.strip()
    if not s: continue
    if s.startswith("## Sources"):
        in_sources = True
        continue
    if in_sources: continue
    if s.startswith("#"): continue
    if s.startswith(">"): continue
    if re.match(r"^\d+\.\s+\[", s): continue
    if s.startswith("Generated:"): continue
    if s in ("_No findings._", "_No findings for this sub-question._"): continue
    if not re.search(r"\[\d+\]", s):
        violations.append((i, s[:120]))

if violations:
    print("CITATION VIOLATIONS:")
    for ln, txt in violations:
        print(f"  L{ln}: {txt}")
    sys.exit(1)
print(f"OK: all prose cited in {md_path}")
EOF
```

### Step 7 -- Cleanup

Two cases:

**Case A: daemon was already running before the skill started** (`DEEP_RESEARCH_REUSED=1`). Close ONLY the research tabs and leave the daemon + pre-existing tabs alone:

```bash
if [ "$DEEP_RESEARCH_REUSED" = "1" ]; then
openbrowser-ai -c - <<'EOF'
state = await browser.get_browser_state_summary()
to_close = [t.target_id for t in state.tabs if t.target_id not in _preexisting_tab_ids]
print(f"Closing {len(to_close)} research tabs, preserving {len(_preexisting_tab_ids)} pre-existing")
for tid in to_close:
    try:
        await close(tab_id=tid)
    except Exception as e:
        print(f"  skip {tid[-4:]}: {e}")
EOF
fi
```

**Case B: daemon was started fresh by this skill** (`DEEP_RESEARCH_REUSED=0`). Full daemon stop, freeing the Chrome process. See Cleanup section below.

**Always:** the final cleanup below removes this run's partial directory and run directory after the report is written. If an owned daemon cannot be shut down, it keeps the run directory intact so the partials and environment remain available for recovery.

## Tips

- **Source the run environment first:** every later Bash call must begin with `. "<RUN_ENV_FILE>"`, using the exact absolute path printed by Step 0. Bare exports do not survive separate shell blocks.
- **Headless by default:** `OPENBROWSER_HEADLESS=true` is written to `RUN_ENV_FILE` in Step 0 so the daemon starts without a visible browser window. The daemon forces `headless=True` unless `OPENBROWSER_HEADLESS` is explicitly false; a config file alone does not opt into a visible window. To debug with a visible window, edit `RUN_ENV_FILE` before the first `-c` call and change that line to `export OPENBROWSER_HEADLESS=false`. Changing it after the daemon starts does not reconfigure the running browser.
- **One tab per sub-agent:** the orchestrator dispatches via `/dispatching-parallel-agents` and each sub-agent must own exactly one tab. Multiple-tab sub-agents serialize navigation at the CDP layer and lose the parallelism benefit. Enforce in the agent prompt: "NEVER pass new_tab=True to navigate() after the first call." The daemon has one global agent focus, so every later sub-agent call must start with `await switch(_MY_TAB_<AGENT_INDEX>)` before navigating.
- **Single message, multiple Agent calls:** to actually parallelize, the orchestrator must put all N `Agent` tool calls in ONE message. Sequential `Agent` invocations across messages run serially.
- **Heredoc quoting:** always use `<<'EOF'` (single-quoted) so `$`, backticks, and `!` inside Python don't expand in the shell.
- **Daemon namespace:** `_plan`, `_findings`, `_sources` persist across orchestrator `-c` calls. Sub-agents share the same daemon and so see the same namespace, but each operates in its own tab. Don't restart the daemon mid-run.
- **Partial files as the contract:** sub-agents write to `$DEEP_RESEARCH_PARTIAL_DIR/agent-NN.json`, the orchestrator reads them back. Sub-agents do NOT communicate findings via stdout. If a partial file is missing or empty, that sub-agent failed; rerun it or fall through to Step 2b.
- **`exact_url` normalization:** sub-agents may write `exact_url` instead of `url`. All merge steps (Step 2a, 2b, 3) normalize this automatically. Step 4 has a final safety pass. Never rely on `url` being set before the merge step runs.
- **Daemon restart recovery:** if the daemon restarts mid-session, `_plan`/`_findings`/`_sources` are lost from namespace. Partial JSON files under `$DEEP_RESEARCH_PARTIAL_DIR` survive. Source the same run environment and re-run the Step 2a merge block. Re-run Step 1 with the same `QUERY` and project root to restore `_plan`; the slug-collision bump avoids overwriting the existing output file.
- **No `-p`, ever:** this skill never shells out to `openbrowser-ai -p`. The `-p` mode runs the full Browser Agent loop with LLM-driven navigation and requires an OpenAI / Anthropic / Google API key. The `-c` daemon path needs no API key. If you see `-p` anywhere in skill code, that is a bug.
- **Source diversity:** if dedup leaves <60% of findings (heavy domain repetition), rerun the affected sub-agent with explicit "prefer different domains than: <list>" appended to its prompt.
- **Drilldown cost:** drilldown dispatches up to `len(sub_questions) * 3` extra parallel sub-agents. Reserve for genuinely deep topics.
- **Rerun semantics:** rerunning Step 5 alone re-renders from current `_findings`, useful after manual edits.
- **Slug collisions:** Step 1 auto-bumps `-2`, `-3` if file exists, so rerunning the same query never overwrites.

## Cleanup

**Mandatory.** Run after every workflow, success or failure.

If this skill started the daemon (`DEEP_RESEARCH_REUSED=0`), use `daemon force-stop`. It targets the current daemon endpoint (the configured Unix socket on macOS/Linux, the fixed loopback port on Windows) and verifies listener and owner absence before it reports success. Delete the profile and run directory only after that success:

```bash
. "$DEEP_RESEARCH_ENV_FILE"
DEEP_RESEARCH_CLEANUP_OK=0
if [ "$DEEP_RESEARCH_REUSED" = "0" ]; then
    if openbrowser-ai daemon force-stop; then
        rm -rf "$OPENBROWSER_DAEMON_PROFILE_DIR" 2>/dev/null || true
        rm -f "${OPENBROWSER_SOCKET%.sock}.log" "${OPENBROWSER_SOCKET%.sock}.pid" 2>/dev/null || true
        DEEP_RESEARCH_CLEANUP_OK=1
    else
        echo "Daemon shutdown could not be verified. Keeping $OPENBROWSER_DAEMON_PROFILE_DIR and $DEEP_RESEARCH_RUN_DIR for manual cleanup."
    fi
else
    DEEP_RESEARCH_CLEANUP_OK=1
fi
if [ "$DEEP_RESEARCH_CLEANUP_OK" = "1" ]; then
    rm -rf "$DEEP_RESEARCH_RUN_DIR" 2>/dev/null || true
fi
```

If daemon was reused (`DEEP_RESEARCH_REUSED=1`), Step 7 already closed only the research tabs. Do NOT call `daemon stop` or `daemon force-stop`, that kills the user's pre-existing browser session.

Install this trap only after Step 0 has sourced the dedicated environment and set `DEEP_RESEARCH_OWNED`. It never fires for an unset or default environment and never deletes the profile:

```bash
trap 'if [ "${DEEP_RESEARCH_OWNED:-0}" = "1" ] && [ -n "${OPENBROWSER_SOCKET:-}" ]; then openbrowser-ai daemon force-stop >/dev/null 2>&1 || true; fi' EXIT
```

Anti-patterns:

- Do NOT `daemon stop` when reusing an existing session; that kills the user's open tabs.
- Do NOT rely on the 600s idle timeout; that wastes a Chrome process for 10 minutes.
- Do NOT use `done()` as a substitute; it only ends the agent loop, browser stays open.
- Do NOT mix `--mcp` mode with the daemon in one workflow; they have separate persistent namespaces and browser profiles.
