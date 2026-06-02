---
name: deep-research
description: |
  Conduct deep web research using the openbrowser-ai agent: decompose a query, investigate sub-questions across multiple sources, and produce a cited markdown report plus structured JSON under local_docs/research/.
  Trigger when the user asks to: research a topic, do a deep dive, investigate, gather evidence, compare options, write a literature review, build a briefing, or produce a cited report.
allowed-tools: Bash(openbrowser-ai:*) Bash(curl:*) Bash(uv:*) Bash(irm:*) Bash(mkdir:*) Bash(date:*) Read Write
---

# Deep Research

Drive `openbrowser-ai` to investigate a topic across multiple web sources and produce a cited markdown report plus structured JSON. Two modes:

- **flat synthesis** (default) -- decompose query into 3-7 sub-questions, investigate each, merge into one cited report.
- **drilldown** (auto-detected from prompt phrasing: "deep dive", "exhaustive", "recursive", "drilldown", "thorough") -- same as flat, plus one recursive level on findings flagged `needs_depth=true`. Hard cap depth=2.

Output paths (relative to current project root):

- `local_docs/research/YYYY-MM-DD-<slug>.md`
- `local_docs/research/YYYY-MM-DD-<slug>.json`

Skill orchestrates via `openbrowser-ai -c -` (heredoc, daemon-backed Python). Two investigation paths, used together:

- **Multi-tab parallel** inside the existing daemon: open one tab per sub-question with `navigate(url, new_tab=True)`, fan out search-and-extract via `asyncio.gather`, dedup. Fast, shares one browser session.
- **Autonomous `-p` agent** for sub-questions that need free-form navigation (paginated archives, JS-heavy sites): `openbrowser-ai -p "<scoped prompt>"`. Returns JSON-only findings.

Variables persist across `-c` calls in the daemon namespace.

**Session reuse:** Step 0 checks `openbrowser-ai daemon status`. If a daemon is already running (warm browser), the skill reuses it and operates in NEW tabs (never disturbs the user's existing tabs). If no daemon, the skill auto-starts one on first `-c` call.

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

Set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`) so spawned `-p` agents can run.

Prepare output dir at the project root (NOT user home):

```bash
mkdir -p local_docs/research
```

## Workflow

### Step 0 -- Session check

Detect if a daemon is already running. If yes, reuse it (and operate in NEW tabs only). If no, the next `-c` call will auto-start one.

```bash
if openbrowser-ai daemon status 2>&1 | grep -qi 'running\|listening\|pid'; then
    echo "Reusing existing daemon -- will work in new tabs"
    export DEEP_RESEARCH_REUSED=1
else
    echo "No daemon running -- will start fresh session"
    export DEEP_RESEARCH_REUSED=0
fi
```

Snapshot existing tabs so cleanup leaves them untouched:

```bash
openbrowser-ai -c - <<'EOF'
state = await browser.get_browser_state_summary()
_preexisting_tab_ids = {t.target_id for t in state.tabs} if state.tabs else set()
print(f"Pre-existing tabs: {len(_preexisting_tab_ids)}")
EOF
```

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

### Step 2a -- Multi-tab parallel investigation (preferred)

Open one new tab per sub-question, scrape SERP via `evaluate`, visit top results, capture body text. Uses `asyncio.gather` so all tabs work in parallel inside the single daemon session. Pre-existing tabs are left alone.

`evaluate` runs raw JS in the page; no extra LLM calls inside the loop, so the multi-tab path is fast and cheap. The Python orchestrator then selects best-fit sentences (or hands off to Step 2b for free-form synthesis).

```bash
openbrowser-ai -c - <<'EOF'
import asyncio, json, re
from urllib.parse import urlparse

SERP_JS = """(function(){
    const out = [];
    document.querySelectorAll('a').forEach(a => {
        const h3 = a.querySelector('h3');
        if (h3 && a.href && a.href.startsWith('http') && !a.href.includes('google.com')) {
            out.push({url: a.href, title: h3.textContent});
        }
    });
    return out.slice(0, 5);
})()"""

PAGE_JS = """(function(){ return (document.body.innerText || '').slice(0, 4000); })()"""

KEYWORDS_RE = re.compile(r"[A-Za-z][A-Za-z0-9 ,'\-]{20,200}")

def best_sentence(text, sub_q):
    # Pick the sentence with the most word-overlap vs sub_q
    q_words = {w.lower() for w in re.findall(r"[a-zA-Z]+", sub_q) if len(w) > 3}
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for s in sentences:
        s = s.strip()
        if not (40 <= len(s) <= 280): continue
        s_words = {w.lower() for w in re.findall(r"[a-zA-Z]+", s)}
        overlap = len(q_words & s_words)
        if overlap == 0: continue
        scored.append((overlap, s))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None

async def investigate_one_tab(sub_q):
    search_url = f"https://www.google.com/search?q={sub_q.replace(' ', '+')}"
    await navigate(search_url, new_tab=True)
    await wait(2)
    state = await browser.get_browser_state_summary()
    tab_id = state.tabs[-1].target_id

    serp = await evaluate(SERP_JS) or []
    findings = []
    for hit in serp[:3]:
        url = hit.get("url", "")
        if not url.startswith("http"): continue
        try:
            await switch(tab_id=tab_id[-4:])
            await navigate(url)
            await wait(2)
            text = await evaluate(PAGE_JS) or ""
            sentence = best_sentence(text, sub_q)
            if not sentence: continue
            words = sentence.split()
            quote = " ".join(words[:40])
            findings.append({
                "claim": sentence,
                "quote": quote,
                "url": url,
                "domain": urlparse(url).netloc,
                "title": hit.get("title", ""),
                "sub_question": sub_q,
                "confidence": "medium",
                "needs_depth": False,
            })
        except Exception as e:
            print(f"  skip {url}: {type(e).__name__}: {e}")
    return tab_id, findings

results = await asyncio.gather(*[investigate_one_tab(sq) for sq in _plan["sub_questions"]])

_findings = []
_research_tab_ids = []
for tab_id, fs in results:
    _research_tab_ids.append(tab_id)
    _findings.extend(fs)

print(f"Opened {len(_research_tab_ids)} research tabs, gathered {len(_findings)} findings")
EOF
```

If the heuristic sentence picker returns <2 findings for any sub-question, fall through to Step 2b for that one.

### Step 2b -- `-p` agent fallback (optional, per sub-question)

For sub-questions that returned <2 findings in Step 2a, spawn an autonomous `-p` agent. Runs in a separate browser process so it doesn't touch the multi-tab session. Sequential, since `-p` agents are heavy.

The scoped prompt below forces JSON-only output with citation-ready fields:

```bash
openbrowser-ai -c - <<'EOF'
import json, subprocess, re

PROMPT_TEMPLATE = """Research the following question and return findings.

Question: {sub_question}

Constraints:
- Visit at least 3 sources from 3 different domains.
- For each finding capture: claim (one sentence), exact URL, supporting quote (verbatim, max 40 words), domain, confidence (low/medium/high), needs_depth (true if claim warrants deeper investigation).
- Output ONLY a JSON array of objects with keys: claim, url, quote, domain, confidence, needs_depth.
- No prose, no markdown fences, no preamble. Just the JSON array."""

def extract_json_array(text):
    # Find first '[' ... matching ']' block
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def investigate(sub_q):
    prompt = PROMPT_TEMPLATE.format(sub_question=sub_q)
    result = subprocess.run(
        ["openbrowser-ai", "-p", prompt],
        capture_output=True, text=True, timeout=600,
    )
    findings = extract_json_array(result.stdout)
    if findings is None:
        # Retry once with stricter framing
        retry = subprocess.run(
            ["openbrowser-ai", "-p", prompt + "\n\nReturn ONLY the JSON array. No other text."],
            capture_output=True, text=True, timeout=600,
        )
        findings = extract_json_array(retry.stdout) or []
    # Tag each finding with the sub-question
    for f in findings:
        f["sub_question"] = sub_q
    return findings

from collections import Counter
counts = Counter(f["sub_question"] for f in _findings)
weak = [sq for sq in _plan["sub_questions"] if counts[sq] < 2]

print(f"Sub-questions needing fallback: {len(weak)}")
for sq in weak:
    print(f"--> {sq}")
    fs = investigate(sq)
    print(f"    got {len(fs)} findings via -p agent")
    _findings.extend(fs)

print(f"\nTotal findings after fallback: {len(_findings)}")
EOF
```

`_findings` lives in the daemon namespace for the next steps.

### Step 3 -- Drilldown (drilldown mode only)

Skip this step in flat mode. In drilldown mode, recurse one level on findings flagged `needs_depth=true`, max 3 targets per parent sub-question, total depth cap = 2.

```bash
openbrowser-ai -c - <<'EOF'
import json, subprocess, re

if _plan["mode"] != "drilldown":
    print("flat mode, skipping drilldown")
else:
    PROMPT_TEMPLATE = """Research the following claim in depth and return supporting / contradicting evidence.

Original claim: {claim}
Source URL: {url}

Constraints:
- Find at least 2 additional sources from different domains.
- For each finding capture: claim (one sentence), exact URL, supporting quote (verbatim, max 40 words), domain, confidence (low/medium/high), needs_depth (always false at this depth).
- Output ONLY a JSON array. No prose."""

    def extract_json_array(text):
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    # Pick top 3 needs_depth=true per sub-question
    by_sq = {}
    for f in _findings:
        if f.get("needs_depth"):
            by_sq.setdefault(f["sub_question"], []).append(f)
    targets = []
    for sq, fs in by_sq.items():
        targets.extend(fs[:3])

    print(f"Drilldown targets: {len(targets)}")
    extra = []
    for t in targets:
        prompt = PROMPT_TEMPLATE.format(claim=t["claim"], url=t["url"])
        result = subprocess.run(
            ["openbrowser-ai", "-p", prompt],
            capture_output=True, text=True, timeout=600,
        )
        more = extract_json_array(result.stdout) or []
        for m in more:
            m["sub_question"] = t["sub_question"]
            m["needs_depth"] = False  # cap reached
        print(f"  {t['claim'][:60]}... -> {len(more)} findings")
        extra.extend(more)

    _findings.extend(extra)
    print(f"Total after drilldown: {len(_findings)}")
EOF
```

### Step 4 -- Aggregate and dedup

Merge findings, dedup by URL, build the source index that footnote numbers will point to.

```bash
openbrowser-ai -c - <<'EOF'
from urllib.parse import urlparse

# `global` so reassignment of daemon-namespace var doesn't shadow it
global _findings, _sources

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
python3 - <<'EOF'
import re, sys, json, glob, os

# Find latest report file under local_docs/research/
files = sorted(glob.glob("local_docs/research/*.md"), key=os.path.getmtime, reverse=True)
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
        await close(tab_id=tid[-4:])
    except Exception as e:
        print(f"  skip {tid[-4:]}: {e}")
EOF
fi
```

**Case B: daemon was started fresh by this skill** (`DEEP_RESEARCH_REUSED=0`). Full daemon stop, freeing the Chrome process. See Cleanup section below.

## Tips

- **Heredoc quoting:** always use `<<'EOF'` (single-quoted) so `$`, backticks, and `!` inside Python don't expand in the shell.
- **Daemon namespace:** `_plan`, `_findings`, `_sources` persist across the `-c` calls in this workflow. Don't restart the daemon mid-run.
- **JSON-only `-p` prompts:** the agent occasionally emits a leading sentence. The regex `\[.*\]` extractor + retry handles most cases. If both fail, inspect the raw stdout (`subprocess.run` keeps it) and tighten the sub-question.
- **Source diversity:** if dedup leaves <60% of findings (heavy domain repetition), rerun investigate for those sub-questions with explicit `prefer different domains than: <list>` appended.
- **Drilldown cost:** drilldown spawns up to `len(sub_questions) * 3` extra `-p` agents. Reserve for genuinely deep topics.
- **Rerun semantics:** rerunning Step 5 alone re-renders from current `_findings`, useful after manual edits.
- **Slug collisions:** Step 1 auto-bumps `-2`, `-3` if file exists, so rerunning the same query never overwrites.

## Cleanup

**Mandatory.** Run after every workflow, success or failure.

If this skill started the daemon (`DEEP_RESEARCH_REUSED=0`), stop it fully:

```bash
if [ "$DEEP_RESEARCH_REUSED" = "0" ]; then
    openbrowser-ai daemon stop
    openbrowser-ai daemon status
fi
```

If daemon was reused (`DEEP_RESEARCH_REUSED=1`), Step 7 already closed only the research tabs. Do NOT call `daemon stop`, that kills the user's pre-existing browser session.

Force-kill fallback (only if daemon was started by skill and stop failed):

```bash
[ "$DEEP_RESEARCH_REUSED" = "0" ] && pkill -f 'openbrowser.*daemon' || true
```

Trap for failure-safe cleanup at script top:

```bash
trap '[ "$DEEP_RESEARCH_REUSED" = "0" ] && openbrowser-ai daemon stop >/dev/null 2>&1 || true' EXIT
```

Anti-patterns:

- Do NOT `daemon stop` when reusing an existing session; that kills the user's open tabs.
- Do NOT rely on the 600s idle timeout; that wastes a Chrome process for 10 minutes.
- Do NOT use `done()` as a substitute; it only ends the agent loop, browser stays open.
- Do NOT mix `--mcp` mode with the daemon; separate profiles, browser contention.
