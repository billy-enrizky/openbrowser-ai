# Browser MCP Server Comparison

Benchmark date: 2026-02-18

## Competitive Landscape

| Server | Stars | npm/week | Tools | Approach | Token Efficiency | Setup |
|--------|-------|----------|-------|----------|-----------------|-------|
| **Playwright MCP** (Microsoft) | 27,300 | 1,433,000 | 22 core (39 total) | Text (a11y snapshot) | High | Low |
| **mcp-chrome** (hangwin) | 10,400 | 9 | 27 | Hybrid (a11y + screenshot) | Medium | Moderate |
| **BrowserMCP** (browsermcp.io) | 5,800 | 9 | 14 | Hybrid (a11y + screenshot) | Medium | Moderate |
| **mcp-playwright** (ExecuteAutomation) | 5,200 | 701 | 28 | Screenshot-primary | Low-Medium | Low |
| **Browserbase MCP** | 3,100 | 1,076 | 9 | Screenshot (vision AI) | Low | High |
| **Fetcher MCP** | 984 | 3,438 | 3 | Text-only (Readability) | Very High | Low |
| **browser-use MCP** (kontext) | 805 | n/a | 1-2 | Screenshot (agent) | Low | High |
| **OpenBrowser MCP** | -- | -- | 18 | Text-first (CDP direct) | High | Low |

## Apple-to-Apple Benchmark: Playwright MCP vs OpenBrowser MCP

### Methodology

Both MCP servers were started as subprocesses and tested via JSON-RPC stdio transport.
Same tasks, same pages, same measurement method.

- **Playwright MCP** v0.0.68 (published 2026-02-14): `npx @playwright/mcp@latest`
- **OpenBrowser MCP** v0.1.16 (published 2026-02-18): `uvx openbrowser-ai[mcp]@0.1.16 --mcp`

Response sizes measured as total JSON-RPC response character count.
Estimated tokens = chars / 4 (standard approximation for mixed English/JSON content).

### Result: httpbin.org/forms/post (Small Page)

| Operation | Playwright Chars | OpenBrowser Chars | Ratio |
|-----------|-----------------|-------------------|-------|
| Navigate | 1,985 | 105 | 19x |
| Page state/snapshot | 1,896 | 1,488 (full) / 176 (compact) | 1.3x / 10.8x |
| Type into form | 526 | 91 | 5.8x |
| List tabs | 367 | 63 | 5.8x |

**Key finding on small pages:** Playwright returns an accessibility snapshot with every navigation
and interaction (including navigate, click, type). OpenBrowser returns only a confirmation message
and lets the agent decide when to request page state.

### Result: Wikipedia Python Page (Large Page, 327 Interactive Elements)

| Operation | Playwright Chars | Playwright Est. Tokens |
|-----------|-----------------|----------------------|
| Navigate | 493,604 | ~123,400 |
| Snapshot (browser_snapshot) | 493,486 | ~123,370 |
| Click | 4,107 | ~1,027 |

| Operation | OpenBrowser Chars | OpenBrowser Est. Tokens |
|-----------|------------------|------------------------|
| Navigate | 272 | ~68 |
| Get state (compact) | 176 | ~44 |
| Get text (full markdown) | 97,712 | ~24,428 |
| Grep "Guido van Rossum" | ~5,000 | ~1,250 |
| Click | 91 | ~23 |

**Critical finding on large pages:** Playwright's a11y snapshot for Wikipedia is 477K chars (~119K tokens).
This is returned with EVERY navigation and snapshot call. OpenBrowser gives the agent control over
data granularity -- from 44 tokens (compact state) to 1,250 tokens (targeted grep) to 24K tokens
(full text extraction).

### Total Token Comparison: 5-Step Workflow

Workflow: Navigate to Wikipedia, get page state, click link, go back, get state again.

| MCP Server | Tool Calls | Total Response Chars | Est. Response Tokens |
|------------|-----------|---------------------|---------------------|
| **Playwright MCP** | 9 | 996,308 | **249,077** |
| **OpenBrowser MCP** | 12 | 1,631 | **408** |

**OpenBrowser used 610x fewer tokens** for the same workflow (compact mode).

Even using OpenBrowser's most verbose mode (full state + get_text), the token
consumption would be approximately 25,000-30,000 tokens -- still **8-10x fewer**
than Playwright for the same information.

### Why the Difference

| Design Decision | Playwright MCP | OpenBrowser MCP |
|----------------|----------------|-----------------|
| Navigate response | Full a11y snapshot included | Confirmation only ("Navigated to: URL") |
| State query | Always full a11y snapshot | Compact (44 tokens) or full (varies) |
| Text extraction | No dedicated tool (use snapshot) | `browser_get_text` returns clean markdown |
| Content search | No grep (dump full snapshot, LLM searches) | `browser_grep` returns only matching lines |
| Element search | Part of snapshot | `browser_search_elements` by tag/class/id/text |
| Click response | Updated a11y snapshot included | Confirmation only |
| Type response | Updated a11y snapshot included | Confirmation only |

**Playwright's approach:** Every tool returns the full page snapshot. This is simple and consistent
but forces the LLM to process ~120K tokens per interaction on large pages.

**OpenBrowser's approach:** Tools return minimal confirmations. The agent explicitly requests
the level of detail it needs via separate tools (compact state, full state, get_text, grep).
This is more tool calls but dramatically fewer tokens.

## Token Cost Reference

### Screenshot-Based Approaches (for comparison)

| Resolution | Claude Tokens | GPT-4o Tokens (high) | GPT-4o Tokens (low) |
|-----------|--------------|---------------------|---------------------|
| 1024x768 | ~1,398 | ~765 | 85 |
| 1280x720 | ~1,229 | ~1,105 | 85 |
| 1920x1080 | ~1,600 (capped) | ~1,105 | 85 |

Formula: Claude = (width * height) / 750. GPT-4o = (170 * tile_count) + 85.

### Text-Based Approaches

| Content Type | Typical Tokens |
|-------------|---------------|
| Compact page state (URL + element count) | 50-100 |
| Full element list (50 elements) | 500-1,000 |
| Full element list (327 elements, Wikipedia) | 5,000-10,000 |
| Accessibility tree (medium page) | 800-2,000 |
| Accessibility tree (Wikipedia, depth=3) | 50,000+ |
| Page as markdown (simple page) | 500-1,500 |
| Page as markdown (Wikipedia) | ~24,400 |
| Targeted grep result | 50-200 |

### Per-Workflow Comparison (10-Step Task)

| Approach | Total Input Tokens | Ratio |
|----------|-------------------|-------|
| Screenshot-based (10 images at 1024x768) | ~14,000 | 1x |
| Playwright MCP (a11y snapshots, complex page) | ~500,000-1,200,000 | 35-86x |
| Playwright MCP (a11y snapshots, simple page) | ~5,000-20,000 | 0.4-1.4x |
| OpenBrowser (selective text tools, complex page) | ~2,000-30,000 | 0.14-2.1x |
| OpenBrowser (targeted grep/search) | ~500-2,000 | 0.04-0.14x |

**Key insight:** Playwright's a11y snapshots can be LARGER than screenshots for complex pages.
A Wikipedia accessibility snapshot (~120K tokens) dwarfs a screenshot (~1.4K tokens).
OpenBrowser's selective approach avoids this bloat entirely.

## Architecture Comparison

### Playwright MCP (Microsoft)

- **Engine:** Playwright (Chromium, Firefox, WebKit, Edge)
- **Tools:** 22 core, 39 total (with opt-in capabilities)
- **Interaction model:** Ref-based (elements identified by `[ref=e25]` from snapshot)
- **Snapshot mode:** Incremental (sends only changes, default), full, or none
- **Codegen:** Yes (generates Playwright TypeScript test code)
- **Session:** Spawns new browser (no reuse of existing sessions)
- **Strengths:** Cross-browser, massive adoption, incremental snapshots, assertion tools
- **Weaknesses:** Always returns snapshot (even when not needed), large tool surface, no grep/search

### OpenBrowser MCP

- **Engine:** Chrome DevTools Protocol (CDP) direct
- **Tools:** 18 total
- **Interaction model:** Index-based (elements identified by numeric index from state)
- **Text extraction:** Dedicated `get_text`, `grep`, `search_elements`, `find_and_scroll`
- **Session:** Connects to existing Chrome instance (reuses sessions/logins)
- **Strengths:** Progressive disclosure, targeted search, session reuse, minimal token overhead
- **Weaknesses:** Chrome-only, no incremental snapshots, no screenshot tool, no codegen

### Puppeteer MCP (Anthropic) -- ARCHIVED

- **Engine:** Puppeteer (Chromium only)
- **Tools:** 7 total
- **Interaction model:** CSS selector-based
- **Status:** Archived, unmaintained. Superseded by Playwright MCP.
- **Approach:** Screenshot-first (requires vision model)

### Browserbase MCP

- **Engine:** Stagehand (cloud-hosted Playwright)
- **Tools:** 9 total
- **Approach:** Screenshot-based with AI vision model interpretation
- **Cost:** Requires paid cloud service + external AI API keys
- **Double inference:** The MCP server calls its own vision model, then returns results to calling LLM

## OpenBrowser Differentiation

### Unique capabilities (no competitor has these)

1. **`browser_grep`** -- regex search within page content, returns only matching lines with context
2. **`browser_search_elements`** -- search interactive elements by text/tag/class/id/attribute
3. **`browser_find_and_scroll`** -- find text and scroll to it
4. **Progressive disclosure** -- compact state (44 tokens) vs full state vs text vs grep
5. **MCP Resources** -- `browser://current-page/content`, `state`, `accessibility`
6. **Multi-session resource templates** -- read from specific sessions by ID
7. **Resource subscriptions** -- subscribe to specific resource URIs for notifications
8. **5 built-in skills** -- web-scraping, form-filling, e2e-testing, page-analysis, accessibility-audit

### Gaps vs competitors

- No screenshot capability (deliberate text-first design)
- No cross-browser support (Chrome/Chromium only via CDP)
- No file upload tool
- No PDF export
- No network interception
- No test assertion tools
- No incremental snapshots (Playwright has this)
- No codegen (Playwright generates TypeScript)

## Next Steps: webmcp Alignment

The [webmcp specification](https://github.com/webmachinelearning/webmcp) proposes a standard
tool interface for browser MCP servers. OpenBrowser should evaluate alignment with this spec
to reduce tool count and avoid context bloat from loading 18 tool schemas into the LLM context.

Key areas to investigate:
- Which tools can be consolidated (e.g., merge session management tools)
- Which tools should be removed (redundant with webmcp standard)
- How to align tool names and schemas with webmcp conventions
- Whether to adopt webmcp's resource-based patterns over tool-based patterns

See TODO: webmcp alignment task in docs/todo.md.
