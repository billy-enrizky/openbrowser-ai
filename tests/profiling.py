"""
Profiling script for 5 important functions in the OpenBrowser-AI codebase.

Uses Python's cProfile and pstats modules to profile:
1. CodeExecutor.execute()       - core code execution path
2. _strip_js_comments()         - JS comment stripping on every evaluate() call
3. BrowserStateSummary.__str__  - state serialization for LLM consumption
4. CodeExecutor._truncate()     - output truncation
5. _read_pid()                  - daemon PID management

Run with: uv run python tests/profiling.py
"""

import asyncio
import cProfile
import logging
import os
import pstats
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Ensure the project source is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from the OpenBrowser codebase
# ---------------------------------------------------------------------------
from openbrowser.code_use.executor import CodeExecutor  # noqa: E402
from openbrowser.code_use.namespace import _strip_js_comments  # noqa: E402
from openbrowser.daemon.server import _read_pid  # noqa: E402

# BrowserStateSummary and its dependencies
from openbrowser.browser.views import (  # noqa: E402
    BrowserStateSummary,
    PageInfo,
    TabInfo,
)
from openbrowser.dom.views import (  # noqa: E402
    DOMSelectorMap,
    EnhancedDOMTreeNode,
    NodeType,
    SerializedDOMState,
    SimplifiedNode,
)

# ============================================================================
# Helper: build minimal test fixtures
# ============================================================================

ITERATIONS_FAST = 1000      # for cheap / synchronous functions
ITERATIONS_MEDIUM = 500     # for moderate functions
ITERATIONS_ASYNC = 200      # for async functions (event-loop overhead)


def _make_executor(max_chars: int = 10_000) -> CodeExecutor:
    """Create a CodeExecutor with a minimal namespace (no real browser)."""
    executor = CodeExecutor(max_output_chars=max_chars)
    namespace: dict = {"__builtins__": __builtins__}
    executor.set_namespace(namespace)
    return executor


def _make_dom_tree_node() -> EnhancedDOMTreeNode:
    """Build a small but realistic EnhancedDOMTreeNode tree for profiling."""
    root = EnhancedDOMTreeNode(
        node_id=1,
        backend_node_id=100,
        node_type=NodeType.ELEMENT_NODE,
        node_name="HTML",
        node_value="",
        attributes={"lang": "en"},
        is_scrollable=False,
        is_visible=True,
        absolute_position=None,
        target_id="ABCDEF1234567890ABCDEF1234567890",
        frame_id="FRAME1234567890AB",
        session_id=None,
        content_document=None,
        shadow_root_type=None,
        shadow_roots=None,
        parent_node=None,
        children_nodes=[],
        ax_node=None,
        snapshot_node=None,
    )
    # Add a few child elements to make the tree non-trivial
    for i in range(10):
        child = EnhancedDOMTreeNode(
            node_id=10 + i,
            backend_node_id=200 + i,
            node_type=NodeType.ELEMENT_NODE,
            node_name="DIV",
            node_value="",
            attributes={"id": f"item-{i}", "class": "card"},
            is_scrollable=False,
            is_visible=True,
            absolute_position=None,
            target_id="ABCDEF1234567890ABCDEF1234567890",
            frame_id="FRAME1234567890AB",
            session_id=None,
            content_document=None,
            shadow_root_type=None,
            shadow_roots=None,
            parent_node=root,
            children_nodes=[],
            ax_node=None,
            snapshot_node=None,
        )
        # Add a text node child
        text_node = EnhancedDOMTreeNode(
            node_id=100 + i,
            backend_node_id=300 + i,
            node_type=NodeType.TEXT_NODE,
            node_name="#text",
            node_value=f"Item {i} content with some realistic text length for profiling",
            attributes={},
            is_scrollable=False,
            is_visible=True,
            absolute_position=None,
            target_id="ABCDEF1234567890ABCDEF1234567890",
            frame_id="FRAME1234567890AB",
            session_id=None,
            content_document=None,
            shadow_root_type=None,
            shadow_roots=None,
            parent_node=child,
            children_nodes=None,
            ax_node=None,
            snapshot_node=None,
        )
        child.children_nodes = [text_node]
        root.children_nodes.append(child)
    return root


def _make_simplified_tree(root: EnhancedDOMTreeNode) -> SimplifiedNode:
    """Convert an EnhancedDOMTreeNode tree into a SimplifiedNode tree."""
    children = []
    if root.children_nodes:
        for child_node in root.children_nodes:
            children.append(_make_simplified_tree(child_node))
    return SimplifiedNode(
        original_node=root,
        children=children,
        should_display=True,
        is_interactive=(root.node_type == NodeType.ELEMENT_NODE and root.node_name == "DIV"),
    )


def _make_selector_map(root: EnhancedDOMTreeNode) -> DOMSelectorMap:
    """Build a selector map from child elements of root."""
    smap: DOMSelectorMap = {}
    if root.children_nodes:
        for idx, child in enumerate(root.children_nodes):
            smap[idx] = child
    return smap


def _make_browser_state_summary() -> BrowserStateSummary:
    """Build a realistic BrowserStateSummary for profiling __str__."""
    root_node = _make_dom_tree_node()
    simplified_root = _make_simplified_tree(root_node)
    selector_map = _make_selector_map(root_node)

    dom_state = SerializedDOMState(
        _root=simplified_root,
        selector_map=selector_map,
    )

    tabs = [
        TabInfo(
            url="https://example.com/page1",
            title="Example Page 1 - A Fairly Long Title For Realism",
            target_id="ABCDEF1234567890ABCDEF1234567890",
        ),
        TabInfo(
            url="https://example.com/page2",
            title="Example Page 2 - Another Tab With Content",
            target_id="0987654321FEDCBA0987654321FEDCBA",
        ),
        TabInfo(
            url="https://example.com/page3",
            title="Example Page 3 - Third Tab For Multi-Tab Scenario",
            target_id="1111222233334444AAAABBBBCCCCDDDD",
        ),
    ]

    return BrowserStateSummary(
        dom_state=dom_state,
        url="https://example.com/page1",
        title="Example Page 1 - A Fairly Long Title For Realism",
        tabs=tabs,
        pixels_above=150,
        pixels_below=2400,
    )


# Sample JS code with various comment styles for _strip_js_comments profiling
JS_CODE_SAMPLE = """\
/* Multi-line comment
   spanning several lines
   with various content */
(function() {
    // This is a single-line comment
    const elements = document.querySelectorAll('.product');
    // Another comment
    const results = [];
    /* inline block comment */ const url = "https://example.com/path";
    for (const el of elements) {
        // Extract data from each element
        results.push({
            name: el.querySelector('.name')?.textContent,
            price: el.querySelector('.price')?.textContent,
            /* nested comment */
            link: el.querySelector('a')?.href
        });
    }
    // Final comment before return
    return JSON.stringify(results);
})()
"""


# ============================================================================
# Profiling functions
# ============================================================================

def profile_executor_execute() -> tuple[pstats.Stats, float]:
    """
    Profile CodeExecutor.execute() -- the core code execution path.

    This wraps user Python code in an async function, compiles it, executes
    it against a namespace, and captures stdout. It is the single most
    important hot path in code-use mode.

    Setup: minimal namespace with builtins only (no real browser needed).
    We execute a simple arithmetic expression to isolate executor overhead
    from user-code complexity.
    """
    executor = _make_executor()
    code_snippet = "result = sum(range(100))\n__ob_result__ = result"

    profiler = cProfile.Profile()

    async def run_iterations():
        for _ in range(ITERATIONS_ASYNC):
            await executor.execute(code_snippet)

    start = time.perf_counter()
    profiler.enable()
    asyncio.run(run_iterations())
    profiler.disable()
    elapsed = time.perf_counter() - start

    stats = pstats.Stats(profiler)
    return stats, elapsed


def profile_strip_js_comments() -> tuple[pstats.Stats, float]:
    """
    Profile _strip_js_comments() -- called on every evaluate() invocation.

    This function uses two regex substitutions:
    1. re.sub for multi-line comments (/* ... */)
    2. re.sub for single-line comments (// at line start)

    Analysis: The regex patterns compile on each call because re.sub compiles
    internally unless pre-compiled. For high-frequency usage, pre-compiling
    the patterns as module-level constants would eliminate repeated compilation.

    Additionally, the DOTALL flag on the multi-line pattern means the regex
    engine must scan the entire input. For very large JS payloads, this
    could become a bottleneck.
    """
    profiler = cProfile.Profile()

    start = time.perf_counter()
    profiler.enable()
    for _ in range(ITERATIONS_FAST):
        _strip_js_comments(JS_CODE_SAMPLE)
    profiler.disable()
    elapsed = time.perf_counter() - start

    stats = pstats.Stats(profiler)
    return stats, elapsed


def profile_browser_state_str() -> tuple[pstats.Stats, float]:
    """
    Profile BrowserStateSummary.__str__() -- state serialization for LLM.

    This is called every agent step to produce the text representation of
    browser state that gets injected into the LLM prompt. It involves:
    1. Building tab list strings with truncation
    2. Counting interactive elements from selector_map
    3. Calling dom_state.eval_representation() which triggers the full
       DOMEvalSerializer.serialize_tree() pass

    The eval_representation() call is the most expensive part -- it walks
    the entire simplified DOM tree. Caching or incremental serialization
    could reduce cost when the DOM has not changed between steps.
    """
    state = _make_browser_state_summary()
    profiler = cProfile.Profile()

    start = time.perf_counter()
    profiler.enable()
    for _ in range(ITERATIONS_MEDIUM):
        str(state)
    profiler.disable()
    elapsed = time.perf_counter() - start

    stats = pstats.Stats(profiler)
    return stats, elapsed


def profile_truncate() -> tuple[pstats.Stats, float]:
    """
    Profile CodeExecutor._truncate() -- output truncation.

    This function checks if text exceeds max_output_chars and slices if so.
    It is called on every execute() result. The implementation is O(1) for
    the length check and O(n) for the slice when truncation occurs.

    We profile two scenarios:
    - Short text (no truncation): just a len() check
    - Long text (triggers truncation): slice + f-string formatting

    Optimization note: The current implementation is already minimal.
    The only potential improvement would be avoiding the f-string allocation
    when truncation is not needed (already handled by the early return).
    For very large outputs, the string slice dominates cost.
    """
    executor = _make_executor(max_chars=5000)
    short_text = "Hello, world! " * 10          # ~150 chars, no truncation
    long_text = "x" * 20_000                     # 20K chars, triggers truncation

    profiler = cProfile.Profile()

    start = time.perf_counter()
    profiler.enable()
    for _ in range(ITERATIONS_FAST):
        executor._truncate(short_text)
        executor._truncate(long_text)
    profiler.disable()
    elapsed = time.perf_counter() - start

    stats = pstats.Stats(profiler)
    return stats, elapsed


def profile_read_pid() -> tuple[pstats.Stats, float]:
    """
    Profile _read_pid() -- daemon PID management.

    This function:
    1. Resolves the PID file path via get_pid_path()
    2. Checks if the file exists (Path.exists())
    3. If present, reads the PID text and parses as int
    4. Sends signal 0 to check if the process is alive (os.kill(pid, 0))
    5. On any error, unlinks the stale PID file

    We profile two scenarios by using a temp directory:
    - Missing PID file: early return after Path.exists() check
    - Present but stale PID file: read + parse + os.kill fails + cleanup

    The function does filesystem I/O on every call, so it is inherently
    slower than pure computation. Caching the PID value with a short TTL
    could reduce repeated disk reads in tight loops (though in practice
    _read_pid is only called at daemon startup).
    """
    profiler = cProfile.Profile()

    # Use a temp directory so we don't interfere with a real running daemon
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pid_path = Path(tmp) / "daemon.pid"

        # Scenario 1: PID file does not exist (most common fast path)
        start = time.perf_counter()
        profiler.enable()
        with patch("openbrowser.daemon.server.get_pid_path", return_value=tmp_pid_path):
            for _ in range(ITERATIONS_FAST):
                _read_pid()
        profiler.disable()
        elapsed_missing = time.perf_counter() - start

        # Scenario 2: PID file exists but PID is stale (process does not exist)
        # Use PID 99999999 which almost certainly does not exist
        stale_pid = 99999999
        tmp_pid_path.write_text(str(stale_pid))

        profiler2 = cProfile.Profile()
        start2 = time.perf_counter()
        profiler2.enable()
        with patch("openbrowser.daemon.server.get_pid_path", return_value=tmp_pid_path):
            for _ in range(ITERATIONS_MEDIUM):
                # Re-create the file each iteration since _read_pid unlinks stale files
                if not tmp_pid_path.exists():
                    tmp_pid_path.write_text(str(stale_pid))
                _read_pid()
        profiler2.disable()
        elapsed_stale = time.perf_counter() - start2

    # Merge both profiles
    combined = pstats.Stats(profiler)
    combined.add(profiler2)
    total_elapsed = elapsed_missing + elapsed_stale

    return combined, total_elapsed


# ============================================================================
# Summary table and main runner
# ============================================================================

def _get_total_calls_and_time(stats: pstats.Stats) -> tuple[int, float]:
    """Extract total call count and total time from pstats.Stats."""
    total_calls = 0
    total_time = 0.0
    for (_file, _line, _name), (cc, nc, tt, ct, callers) in stats.stats.items():
        total_calls += nc
        total_time += tt
    return total_calls, total_time


def print_section_header(title: str, description: str) -> None:
    """Print a formatted section header for a profiled function."""
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"  {title}")
    print(f"  {description}")
    print(separator)


def main():
    """Run all profiles and print results."""
    print("=" * 80)
    print("  OpenBrowser-AI Function Profiling Report")
    print("  Using cProfile + pstats (https://docs.python.org/3/library/profile.html)")
    print("=" * 80)

    results: list[tuple[str, int, pstats.Stats, float]] = []

    # ---- 1. CodeExecutor.execute() ----
    print_section_header(
        "1. CodeExecutor.execute()",
        f"Core code execution path ({ITERATIONS_ASYNC} iterations, async)",
    )
    stats1, elapsed1 = profile_executor_execute()
    stats1.sort_stats("cumulative")
    stats1.print_stats(20)
    results.append(("CodeExecutor.execute()", ITERATIONS_ASYNC, stats1, elapsed1))

    # ---- 2. _strip_js_comments() ----
    print_section_header(
        "2. _strip_js_comments()",
        f"JS comment stripping ({ITERATIONS_FAST} iterations)",
    )
    stats2, elapsed2 = profile_strip_js_comments()
    stats2.sort_stats("cumulative")
    stats2.print_stats(20)
    results.append(("_strip_js_comments()", ITERATIONS_FAST, stats2, elapsed2))

    # ---- 3. BrowserStateSummary.__str__() ----
    print_section_header(
        "3. BrowserStateSummary.__str__()",
        f"State serialization for LLM ({ITERATIONS_MEDIUM} iterations)",
    )
    stats3, elapsed3 = profile_browser_state_str()
    stats3.sort_stats("cumulative")
    stats3.print_stats(20)
    results.append(("BrowserStateSummary.__str__()", ITERATIONS_MEDIUM, stats3, elapsed3))

    # ---- 4. CodeExecutor._truncate() ----
    print_section_header(
        "4. CodeExecutor._truncate()",
        f"Output truncation ({ITERATIONS_FAST} iterations, 2 calls each)",
    )
    stats4, elapsed4 = profile_truncate()
    stats4.sort_stats("cumulative")
    stats4.print_stats(20)
    results.append(("CodeExecutor._truncate()", ITERATIONS_FAST * 2, stats4, elapsed4))

    # ---- 5. _read_pid() ----
    print_section_header(
        "5. _read_pid()",
        f"Daemon PID management ({ITERATIONS_FAST} missing + {ITERATIONS_MEDIUM} stale iterations)",
    )
    stats5, elapsed5 = profile_read_pid()
    stats5.sort_stats("cumulative")
    stats5.print_stats(20)
    results.append(("_read_pid()", ITERATIONS_FAST + ITERATIONS_MEDIUM, stats5, elapsed5))

    # ========================================================================
    # Summary Table
    # ========================================================================
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Function':<35} {'Iterations':>10} {'Total Calls':>12} {'Total Time (s)':>15} {'Time/Call (us)':>15}"
    print(header)
    print("-" * len(header))

    for name, iterations, stats, elapsed in results:
        total_calls, total_time = _get_total_calls_and_time(stats)
        time_per_call_us = (elapsed / iterations) * 1_000_000 if iterations > 0 else 0
        print(
            f"{name:<35} {iterations:>10} {total_calls:>12} {elapsed:>15.6f} {time_per_call_us:>15.2f}"
        )

    print("-" * len(header))

    # ========================================================================
    # Analysis and Optimization Opportunities
    # ========================================================================
    print("\n" + "=" * 80)
    print("  ANALYSIS AND OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)

    analysis = """
1. CodeExecutor.execute()
   - Hottest path: compile() + exec() + await on every call.
   - The code wrapping (string concatenation to build the async wrapper)
     runs on every invocation. Pre-compiling a template or using
     ast.parse + ast.fix_missing_locations could reduce overhead.
   - The asyncio.Lock acquisition adds contention overhead even when
     there is no concurrent access. A fast-path check could skip locking
     in single-client scenarios.
   - The namespace cleanup (pop __ob_exec__, __ob_result__) in the finally
     block is minimal cost but runs every call.

2. _strip_js_comments()
   - Two re.sub() calls with patterns that are recompiled on each call.
     Python's re module caches compiled patterns (up to 512), but
     pre-compiling as module-level constants (_RE_MULTILINE_COMMENT,
     _RE_SINGLELINE_COMMENT) would make the intent explicit and avoid
     cache lookup overhead.
   - The DOTALL flag on the multi-line pattern causes a full-string scan.
     For very large JS payloads (>100KB), this could become noticeable.
   - Consider a single-pass approach using re.sub with an alternation
     pattern: r'/\\*.*?\\*/|^\\s*//.*$' (with DOTALL|MULTILINE flags)
     to avoid scanning the string twice.

3. BrowserStateSummary.__str__()
   - The dominant cost is dom_state.eval_representation() which walks the
     entire SimplifiedNode tree via DOMEvalSerializer.serialize_tree().
   - This is called every agent step, even when the DOM has not changed.
     Caching the serialized output (invalidated on DOM mutation) would
     eliminate redundant serialization.
   - Tab title truncation ([:60]) runs per tab per call -- negligible cost.
   - String joining via '\\n'.join(lines) is efficient; no optimization needed.

4. CodeExecutor._truncate()
   - Already minimal: a len() check + conditional slice.
   - The f-string for the truncation message allocates a new string each time.
     This is unavoidable and costs are negligible.
   - No meaningful optimization opportunities -- this is effectively O(1)
     for the common case (no truncation) and O(n) for the truncation slice.

5. _read_pid()
   - Filesystem I/O dominates: Path.exists(), Path.read_text(), os.kill().
   - The stale-PID path additionally calls Path.unlink(missing_ok=True).
   - Since _read_pid() is only called at daemon startup (not in a hot loop),
     its absolute cost is acceptable.
   - If it were called frequently, caching the result with a short TTL
     (e.g., 1-5 seconds) would be beneficial.
"""
    print(analysis)


if __name__ == "__main__":
    main()
