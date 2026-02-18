# Session Summary & TODO

Last updated: 2026-02-18 03:28:00

---

## Current Session: Competitor Benchmarks, Comparison Doc, Tool Consolidation Plan

**Date**: 2026-02-18 03:28:00
**Branch**: feat/competitor-research
**Duration context**: Researched 8 browser MCP servers. Ran apple-to-apple benchmarks (Playwright MCP v0.0.68 vs OpenBrowser MCP v0.1.16). Created comparison document. Identified tool consolidation path from 18 to 11 tools.

### What Was Accomplished

- Researched 8 browser MCP servers: Playwright MCP, mcp-chrome, BrowserMCP, mcp-playwright, Browserbase MCP, Fetcher MCP, browser-use MCP
- Installed Playwright MCP v0.0.68 (`npx @playwright/mcp@latest`) for head-to-head benchmarking
- Created `benchmarks/playwright_benchmark.py` and `benchmarks/openbrowser_benchmark.py` -- JSON-RPC stdio transport, same tasks, same pages
- Ran apple-to-apple benchmark: Playwright vs OpenBrowser on Wikipedia + httpbin.org
- Key result: **OpenBrowser used 610x fewer response tokens** than Playwright for same 5-step workflow (compact mode), 8-10x fewer even in verbose mode
- Playwright returns ~494K chars per Wikipedia snapshot; OpenBrowser gives 176 chars (compact) to 97K chars (full text)
- Created `docs/comparison.md` with competitive landscape, benchmark data, architecture comparison, token cost reference
- Ran 18/18 integration tests: all PASS
- Researched webmcp specification (W3C Community Group Draft) -- found it is NOT a browser automation tool spec but provides design guidance for tool consolidation
- Identified tool consolidation path: 18 tools -> 11 tools to reduce context bloat

### Files Created/Modified

| File | Change |
|------|--------|
| `benchmarks/playwright_benchmark.py` | NEW: Playwright MCP benchmark via JSON-RPC stdio transport |
| `benchmarks/openbrowser_benchmark.py` | NEW: OpenBrowser MCP benchmark via JSON-RPC stdio transport |
| `benchmarks/playwright_results.json` | NEW: Raw benchmark data (9 calls, 996K resp chars, ~249K tokens) |
| `benchmarks/openbrowser_results.json` | NEW: Raw benchmark data (12 calls, 1.6K resp chars, ~408 tokens) |
| `docs/comparison.md` | NEW: Competitive landscape, benchmarks, architecture comparison |
| `docs/todo.md` | Updated with benchmark results and tool consolidation plan |
| `local_docs/CHANGELOG.md` | Updated with session changes |

### Important Commands

```bash
# Run MCP unit tests (117 tests, no browser needed)
uv run python -m pytest tests/test_mcp_server.py -v

# Run integration tests (requires Chrome/Chromium)
uv run python -m pytest tests/test_mcp_integration.py -v -m integration

# Test published MCP server via uvx
uvx "openbrowser-ai[mcp]@0.1.16" --mcp

# Clear uvx cache if version resolution fails
uvx --refresh --from "openbrowser-ai[mcp]==0.1.16" openbrowser-ai --mcp
```

### E2E Test Results v0.1.16 (18 tools via uvx, Claude Code MCP client)

| Tool | Status | Notes |
|------|--------|-------|
| `browser_navigate` | PASS | Standard + new_tab modes |
| `browser_click` | PASS | By element index, navigated to Toni Morrison article |
| `browser_type` | PASS | Input field + textarea on httpbin forms |
| `browser_get_state` | PASS | Compact (default) + full with 183 interactive elements |
| `browser_scroll` | PASS | Down + up on Wikipedia |
| `browser_go_back` | PASS | Returns to previous page |
| `browser_list_tabs` | PASS | Lists all with 4-char IDs |
| `browser_switch_tab` | PASS | By tab_id, confirmed switch |
| `browser_close_tab` | PASS | Two-phase settle fix verified, no about:blank landing (2 rounds) |
| `browser_get_text` | PASS | Clean markdown extraction |
| `browser_grep` | PASS | Regex pattern with context lines |
| `browser_search_elements` | PASS | By tag + by text |
| `browser_find_and_scroll` | PASS | Finds and scrolls to text |
| `browser_get_accessibility_tree` | PASS | Tree format + flat format with max_depth |
| `browser_execute_js` | PASS | Simple, IIFE, await_promise, return_by_value (RemoteObject) |
| `browser_list_sessions` | PASS | Lists active sessions with metadata |
| `browser_close_session` | PASS | Valid ID closes, invalid ID returns clean error |
| `browser_close_all` | PASS | Closes all sessions, server remains functional |

### Current State

**Git**:
- On `main`, changes not yet committed
- Modified: `tests/conftest.py`, `tests/test_mcp_server.py`, `tests/test_mcp_integration.py`
- Untracked: `.mcp.json`, `docs/plans/`, `docs/todo.md`, `extension.zip`
- Latest tag: `v0.1.16`

**Tests**:
- 117 unit tests: PASS
- 18 integration tests: collected (requires Chrome)
- E2E via Claude Code (v0.1.16 uvx): 18/18 PASS

### Next Steps

#### Immediate -- Tool Consolidation (reduce context bloat)

**Priority 1: Consolidate 18 tools down to 11 tools**

The benchmark showed OpenBrowser has strong token efficiency per-tool, but **18 tool schemas loaded into LLM context is itself a source of bloat**. Each tool schema costs ~200-400 tokens in the system prompt. Consolidating tools reduces context overhead by ~1,400-2,800 tokens per session.

Consolidation plan (informed by webmcp design principles and benchmark findings):

| Current Tools | Consolidated Tool | Approach |
|--------------|-------------------|----------|
| `browser_list_tabs` + `browser_switch_tab` + `browser_close_tab` | `browser_tab` | Single tool with `action` param: `list`, `switch`, `close` |
| `browser_list_sessions` + `browser_close_session` + `browser_close_all` | `browser_session` | Single tool with `action` param: `list`, `close`, `close_all` |
| `browser_grep` | Merge into `browser_get_text` | Add `search` param (regex pattern) and `context_lines` param |
| `browser_find_and_scroll` | Merge into `browser_scroll` | Add `target_text` param for find-and-scroll behavior |
| `browser_search_elements` | Merge into `browser_get_state` | Add `filter` param with `by` and `query` sub-params |

Result: **11 tools** (down from 18)
1. `browser_navigate`
2. `browser_click`
3. `browser_type`
4. `browser_get_state` (with optional `filter` for element search)
5. `browser_scroll` (with optional `target_text` for find-and-scroll)
6. `browser_go_back`
7. `browser_get_text` (with optional `search` for grep)
8. `browser_tab` (merged list/switch/close)
9. `browser_session` (merged list/close/close_all)
10. `browser_get_accessibility_tree`
11. `browser_execute_js`

Implementation steps:
- [ ] Create new `browser_tab` tool with action param, deprecate 3 individual tools
- [ ] Create new `browser_session` tool with action param, deprecate 3 individual tools
- [ ] Add `search`/`context_lines` params to `browser_get_text`, deprecate `browser_grep`
- [ ] Add `target_text` param to `browser_scroll`, deprecate `browser_find_and_scroll`
- [ ] Add `filter` param to `browser_get_state`, deprecate `browser_search_elements`
- [ ] Update manifest.json with consolidated tool definitions
- [ ] Update plugin skill SKILL.md files to reference new tool names
- [ ] Update all unit tests for merged tools
- [ ] Update integration tests
- [ ] Update README and comparison doc
- [ ] Bump to v0.1.18 and publish

Reference: [webmcp specification](https://github.com/webmachinelearning/webmcp) -- W3C Community Group Draft for web apps exposing tools to AI agents. While not a browser automation spec, its design philosophy around minimal tool surface area and progressive disclosure aligns with OpenBrowser's approach.

#### Short-term
- [ ] Commit and tag v0.1.17 with benchmark and comparison work
- [ ] Test with Claude Desktop as MCP client
- [ ] Test Codex integration: clone repo, symlink skills, verify discovery

#### Completed (this session)
- [x] Research 8 competitor browser MCPs
- [x] Install Playwright MCP for benchmarking
- [x] Run apple-to-apple benchmark (Playwright vs OpenBrowser)
- [x] Create comparison document (docs/comparison.md)
- [x] Run integration tests (18/18 PASS)
- [x] Research webmcp specification

#### Future
- [ ] Performance benchmarking for large page a11y trees (flat vs tree, Wikipedia was 373K at depth=2)
- [ ] Explore MCP sampling/prompts capabilities
- [ ] Multi-browser support (Firefox via Playwright)
- [ ] Wait for `claude plugin validate` CLI command to do formal marketplace validation

---

### Comprehensive Browser MCP Comparison -- COMPLETED

See `docs/comparison.md` for the full comparison document with:
- [x] Competitive landscape table (8 browser MCPs)
- [x] Apple-to-apple benchmark: Playwright MCP vs OpenBrowser MCP (Wikipedia + httpbin)
- [x] Token consumption benchmarks with raw numbers (610x efficiency in compact mode)
- [x] Architecture comparison (design philosophy, tool surface, session model)
- [x] OpenBrowser differentiation and gaps vs competitors
- [x] Token cost reference (screenshots vs text approaches)
- [x] webmcp alignment analysis and tool consolidation recommendations

Raw benchmark data in `benchmarks/playwright_results.json` and `benchmarks/openbrowser_results.json`.

---

## Previous Sessions (Consolidated)

### 2026-02-18 01:04 -- E2E Testing v0.1.15 via uvx, Second close_tab Fix, Release v0.1.16
- Full E2E test of all 18 MCP tools using published `uvx openbrowser-ai[mcp]@0.1.15 --mcp`: 18/18 PASS
- Found and fixed second close_tab race condition (AboutBlankWatchdog async focus steal)
- Tagged v0.1.16, published to PyPI, updated .mcp.json to v0.1.16

### 2026-02-18 00:38 -- E2E v0.1.15 via local source, first close_tab fix, release v0.1.15
- E2E tested 18 tools via local source MCP: 17/18 PASS, 1 SKIP (close_all)
- Found and fixed first close_tab race condition (c258e93)
- Tagged v0.1.15, published to PyPI

### 2026-02-17 23:43 -- Fix uvx import chain, release v0.1.14
- Fixed 3 import chain bugs preventing `uvx openbrowser-ai[mcp] --mcp` from working
- PR #47 merged, v0.1.14 released to PyPI

### 2026-02-17 22:50 -- CI fix, multi-platform plugin docs
- Fixed CI: Added ResourceTemplate stub to DummyTypes
- Created Codex, OpenCode integration docs; PR #46 merged

### 2026-02-17 19:33-21:25 -- Feature batch and docs
- await_promise/return_by_value for execute_js, flat a11y format, resource notifications/subscriptions, multi-session templates, 5 plugin skills, marketplace.json, README overhaul

### 2026-02-17 14:40 -- Full tool suite implementation
- Text-first tools, a11y tree, JS execution, MCP resources; 87 unit + 18 integration tests
