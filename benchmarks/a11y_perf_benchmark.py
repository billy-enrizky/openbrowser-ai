"""
A11y Tree Performance Benchmark: measures how OpenBrowser's accessibility
tree processing scales with page complexity.

Navigates to pages of increasing DOM complexity and measures:
- CDP call latency (DOM snapshot, AX tree, layout metrics)
- DOM tree construction time
- Serialization pipeline breakdown (simplify, paint order, optimize, bbox, indices)
- Output size (interactive elements, serialized text length)
- Process memory usage

No LLM required -- this is a pure infrastructure benchmark.

Usage:
    uv run python benchmarks/a11y_perf_benchmark.py
    uv run python benchmarks/a11y_perf_benchmark.py --pages tiny medium_list large_article
    uv run python benchmarks/a11y_perf_benchmark.py --iterations 5
    uv run python benchmarks/a11y_perf_benchmark.py --headed --iterations 1
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.browser.profile import ViewportSize
from openbrowser.dom.service import DomService
from openbrowser.dom.views import DEFAULT_INCLUDE_ATTRIBUTES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test pages (ordered by expected complexity)
# ---------------------------------------------------------------------------

TEST_PAGES = [
    {
        "label": "tiny",
        "url": "https://example.com",
        "description": "Minimal static page (~10 elements)",
    },
    {
        "label": "small_form",
        "url": "https://httpbin.org/forms/post",
        "description": "Small form with several inputs (~30 elements)",
    },
    {
        "label": "medium_list",
        "url": "https://news.ycombinator.com",
        "description": "List-based page with many links (~200+ elements)",
    },
    {
        "label": "large_article",
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "description": "Long article with sections, links, references (~1000+ elements)",
    },
    {
        "label": "complex_spa",
        "url": "https://github.com/anthropics/claude-code",
        "description": "Complex SPA with shadow DOM, interactive elements (~500+ elements)",
    },
    {
        "label": "heavy_ecommerce",
        "url": "https://www.ebay.com/",
        "description": "Heavy e-commerce page with ads, carousels, iframes (~1500+ elements)",
    },
]

LABEL_TO_PAGE = {p["label"]: p for p in TEST_PAGES}


def _kill_stale_browsers():
    """Kill all Chrome/Chromium processes and wait until fully dead."""
    for pattern in ["chromium", "chrome", "Chromium", "Google Chrome"]:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    for _ in range(20):
        result = subprocess.run(
            ["pgrep", "-f", "chrom"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            break
        time.sleep(0.5)
    else:
        logger.warning("Chrome processes still alive after 10s wait")

    time.sleep(1)


def count_tree_nodes(node) -> int:
    """Recursively count nodes in an EnhancedDOMTreeNode tree."""
    count = 1
    if hasattr(node, "children_nodes") and node.children_nodes:
        for child in node.children_nodes:
            count += count_tree_nodes(child)
    if hasattr(node, "shadow_roots") and node.shadow_roots:
        for shadow in node.shadow_roots:
            count += count_tree_nodes(shadow)
    if hasattr(node, "content_document") and node.content_document:
        count += count_tree_nodes(node.content_document)
    return count


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


async def run_single_iteration(
    browser_session: BrowserSession,
    dom_service: DomService,
    page: dict,
    iteration: int,
) -> dict:
    """Run a single measurement iteration for one page."""
    url = page["url"]
    label = page["label"]
    process = psutil.Process(os.getpid())

    try:
        # Navigate and wait for page load
        pipeline_start = time.time()
        await browser_session._cdp_navigate(url)
        await asyncio.sleep(2)

        # Measure the full DOM processing pipeline
        state, tree, timing_info = await dom_service.get_serialized_dom_tree()
        pipeline_end = time.time()

        # Collect output metrics
        interactive_count = len(state.selector_map)
        serialized_text = state.llm_representation(include_attributes=DEFAULT_INCLUDE_ATTRIBUTES)
        serialized_text_length = len(serialized_text)
        enhanced_node_count = count_tree_nodes(tree)
        memory_mb = process.memory_info().rss / (1024 * 1024)

        result = {
            "url": url,
            "label": label,
            "iteration": iteration,
            # Node counts
            "ax_tree_node_count": dom_service._last_ax_node_count,
            "dom_snapshot_node_count": dom_service._last_snapshot_node_count,
            "enhanced_node_count": enhanced_node_count,
            "interactive_element_count": interactive_count,
            # Output size
            "serialized_text_length": serialized_text_length,
            # Timing (seconds)
            "cdp_calls_total_s": timing_info.get("cdp_calls_total", 0),
            "get_dom_tree_total_s": timing_info.get("get_dom_tree_total", 0),
            "create_simplified_tree_s": timing_info.get("create_simplified_tree", 0),
            "calculate_paint_order_s": timing_info.get("calculate_paint_order", 0),
            "optimize_tree_s": timing_info.get("optimize_tree", 0),
            "bbox_filtering_s": timing_info.get("bbox_filtering", 0),
            "assign_interactive_indices_s": timing_info.get("assign_interactive_indices", 0),
            "clickable_detection_time_s": timing_info.get("clickable_detection_time", 0),
            "serialize_accessible_elements_total_s": timing_info.get(
                "serialize_accessible_elements_total", 0
            ),
            "serialize_dom_tree_total_s": timing_info.get("serialize_dom_tree_total", 0),
            "total_pipeline_s": round(pipeline_end - pipeline_start, 4),
            # Memory
            "process_memory_mb": round(memory_mb, 1),
        }

        logger.info(
            "  Iteration %d: %d interactive, %d AX nodes, %d chars, %.2fs total",
            iteration + 1,
            interactive_count,
            dom_service._last_ax_node_count,
            serialized_text_length,
            result["total_pipeline_s"],
        )

        return result

    except Exception as e:
        logger.error("  Iteration %d failed for %s: %s", iteration + 1, label, e)
        return {
            "url": url,
            "label": label,
            "iteration": iteration,
            "error": str(e),
        }


async def run_page_benchmark(
    browser_session: BrowserSession,
    dom_service: DomService,
    page: dict,
    iterations: int,
) -> dict:
    """Run multiple iterations for one page and compute summary stats."""
    label = page["label"]
    url = page["url"]
    logger.info("[%s] %s", label, url)

    results = []
    for i in range(iterations):
        result = await run_single_iteration(browser_session, dom_service, page, i)
        # Patch the log message with total iterations
        results.append(result)

    # Compute summary stats from successful iterations
    successful = [r for r in results if "error" not in r]

    summary = {}
    if successful:
        numeric_keys = [
            "ax_tree_node_count",
            "dom_snapshot_node_count",
            "enhanced_node_count",
            "interactive_element_count",
            "serialized_text_length",
            "cdp_calls_total_s",
            "get_dom_tree_total_s",
            "create_simplified_tree_s",
            "calculate_paint_order_s",
            "optimize_tree_s",
            "bbox_filtering_s",
            "assign_interactive_indices_s",
            "clickable_detection_time_s",
            "serialize_accessible_elements_total_s",
            "serialize_dom_tree_total_s",
            "total_pipeline_s",
            "process_memory_mb",
        ]
        for key in numeric_keys:
            values = [r[key] for r in successful if key in r]
            if values:
                summary[f"{key}_mean"] = round(statistics.mean(values), 4)
                if len(values) > 1:
                    summary[f"{key}_std"] = round(statistics.stdev(values), 4)

    return {
        "url": url,
        "label": label,
        "description": page["description"],
        "iterations_completed": len(successful),
        "iterations_failed": len(results) - len(successful),
        "iterations": results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def format_results_table(all_page_results: list[dict]) -> str:
    """Format a console-friendly results table."""
    lines = []
    lines.append("")
    lines.append("=" * 120)
    lines.append("A11Y TREE PERFORMANCE BENCHMARK RESULTS")
    lines.append("=" * 120)
    lines.append("")

    # Summary table
    header = (
        f"{'Page':<18s} {'Elements':>8s} {'AX Nodes':>9s} "
        f"{'DOM Nodes':>10s} {'Text Len':>9s} "
        f"{'CDP(ms)':>9s} {'Tree(ms)':>9s} {'Serial(ms)':>10s} {'Total(ms)':>10s} "
        f"{'Mem(MB)':>8s}"
    )
    lines.append(header)
    lines.append("-" * 120)

    for page_result in all_page_results:
        s = page_result.get("summary", {})
        if not s:
            lines.append(f"{page_result['label']:<18s} {'FAILED':>8s}")
            continue

        lines.append(
            f"{page_result['label']:<18s} "
            f"{s.get('interactive_element_count_mean', 0):>8.0f} "
            f"{s.get('ax_tree_node_count_mean', 0):>9.0f} "
            f"{s.get('dom_snapshot_node_count_mean', 0):>10.0f} "
            f"{s.get('serialized_text_length_mean', 0):>9.0f} "
            f"{s.get('cdp_calls_total_s_mean', 0) * 1000:>9.1f} "
            f"{s.get('get_dom_tree_total_s_mean', 0) * 1000:>9.1f} "
            f"{s.get('serialize_dom_tree_total_s_mean', 0) * 1000:>10.1f} "
            f"{s.get('total_pipeline_s_mean', 0) * 1000:>10.1f} "
            f"{s.get('process_memory_mb_mean', 0):>8.1f}"
        )

    # Timing breakdown table
    lines.append("")
    lines.append("SERIALIZATION BREAKDOWN (ms, mean)")
    lines.append("-" * 120)

    breakdown_header = (
        f"{'Page':<18s} {'Simplify':>10s} {'PaintOrd':>10s} "
        f"{'Optimize':>10s} {'BBox':>10s} {'Indices':>10s} "
        f"{'Clickable':>10s} {'SerTotal':>10s}"
    )
    lines.append(breakdown_header)
    lines.append("-" * 120)

    for page_result in all_page_results:
        s = page_result.get("summary", {})
        if not s:
            continue

        lines.append(
            f"{page_result['label']:<18s} "
            f"{s.get('create_simplified_tree_s_mean', 0) * 1000:>10.2f} "
            f"{s.get('calculate_paint_order_s_mean', 0) * 1000:>10.2f} "
            f"{s.get('optimize_tree_s_mean', 0) * 1000:>10.2f} "
            f"{s.get('bbox_filtering_s_mean', 0) * 1000:>10.2f} "
            f"{s.get('assign_interactive_indices_s_mean', 0) * 1000:>10.2f} "
            f"{s.get('clickable_detection_time_s_mean', 0) * 1000:>10.2f} "
            f"{s.get('serialize_accessible_elements_total_s_mean', 0) * 1000:>10.2f}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------


async def run_benchmark(
    pages: list[dict],
    iterations: int,
    output_path: str,
    headless: bool,
) -> dict:
    """Run the full benchmark suite."""
    _kill_stale_browsers()

    logger.info("Starting a11y tree performance benchmark")
    logger.info("Pages: %d, Iterations per page: %d, Headless: %s", len(pages), iterations, headless)

    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            window_size=ViewportSize(width=1280, height=720),
            headless=headless,
            disable_security=False,
            wait_for_network_idle_page_load_time=1,
            paint_order_filtering=True,
        ),
    )

    try:
        await browser_session.start()
        dom_service = DomService(browser_session)

        all_page_results = []
        for idx, page in enumerate(pages):
            logger.info("")
            logger.info("[%d/%d] %s (%s)", idx + 1, len(pages), page["label"], page["url"])
            page_result = await run_page_benchmark(browser_session, dom_service, page, iterations)
            all_page_results.append(page_result)

        # Assemble final results
        results = {
            "benchmark": "a11y_perf",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "iterations_per_page": iterations,
                "headless": headless,
                "paint_order_filtering": True,
                "viewport": {"width": 1280, "height": 720},
            },
            "pages": all_page_results,
        }

        # Write JSON
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", output)

        # Print table
        table = format_results_table(all_page_results)
        for line in table.split("\n"):
            logger.info(line)

        # Force-kill browsers and exit. asyncio.run() cleanup hangs on
        # dangling websocket tasks from the browser session, so we exit
        # from here. Results are already written to disk.
        _kill_stale_browsers()
        os._exit(0)

    finally:
        _kill_stale_browsers()
        os._exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="A11y tree performance benchmark for OpenBrowser DOM processing pipeline",
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        choices=[p["label"] for p in TEST_PAGES],
        default=None,
        help="Specific pages to benchmark (default: all)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per page (default: 3)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/a11y_perf_results.json",
        help="Output JSON path (default: benchmarks/a11y_perf_results.json)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (visible window)",
    )
    args = parser.parse_args()

    pages = TEST_PAGES
    if args.pages:
        pages = [LABEL_TO_PAGE[label] for label in args.pages]

    asyncio.run(
        run_benchmark(
            pages=pages,
            iterations=args.iterations,
            output_path=args.output,
            headless=not args.headed,
        )
    )


if __name__ == "__main__":
    main()
