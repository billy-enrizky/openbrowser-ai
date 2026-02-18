"""
E2E test of published openbrowser-ai[mcp]@0.1.17 via JSON-RPC stdio.
Tests all 11 consolidated tools against the published PyPI package.
"""
import asyncio
import json
import logging
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

class MCPClient:
    def __init__(self, command: list[str]):
        self.command = command
        self.process = None
        self._id = 0

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Initialize
        resp = await self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "1.0"},
        })
        # Send initialized notification
        self._write({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    async def call_tool(self, name: str, arguments: dict = None) -> dict:
        return await self._send("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })

    async def list_tools(self) -> dict:
        return await self._send("tools/list", {})

    async def _send(self, method: str, params: dict) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        self._write(msg)
        return await self._read_response(self._id)

    def _write(self, msg: dict):
        line = json.dumps(msg) + "\n"
        self.process.stdin.write(line.encode())

    async def _read_response(self, expected_id: int) -> dict:
        while True:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=60)
            if not line:
                raise RuntimeError("Server closed stdout")
            data = json.loads(line.decode())
            if data.get("id") == expected_id:
                return data
            # Skip notifications

    async def stop(self):
        if self.process:
            self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()


def get_text(resp: dict) -> str:
    """Extract text from tool call response."""
    result = resp.get("result", {})
    content = result.get("content", [])
    if content:
        return content[0].get("text", "")
    return ""


async def run_tests():
    results = []

    logger.info("=" * 60)
    logger.info("E2E Test: openbrowser-ai[mcp]@0.1.17 (published)")
    logger.info("=" * 60)

    client = MCPClient(["uvx", "openbrowser-ai[mcp]@0.1.17", "--mcp"])

    try:
        logger.info("\nStarting MCP server...")
        init = await client.start()
        server_info = init.get("result", {}).get("serverInfo", {})
        logger.info(f"Server: {server_info.get('name')} v{server_info.get('version')}")

        # Test 0: Verify tool count and annotations
        logger.info("\n--- Tool List & Annotations ---")
        tools_resp = await client.list_tools()
        tools = tools_resp.get("result", {}).get("tools", [])
        logger.info(f"Tool count: {len(tools)}")
        assert len(tools) == 11, f"Expected 11 tools, got {len(tools)}"

        annotations_ok = True
        for t in tools:
            a = t.get("annotations", {})
            if not a or "readOnlyHint" not in a:
                logger.info(f"  MISSING annotations: {t['name']}")
                annotations_ok = False
            else:
                ro = a.get("readOnlyHint")
                dest = a.get("destructiveHint")
                idem = a.get("idempotentHint")
                logger.info(f"  {t['name']:40s} readOnly={str(ro):5s} destructive={str(dest):5s} idempotent={str(idem):5s}")

        status = PASS if annotations_ok else FAIL
        results.append(("ToolAnnotations (11 tools)", status))
        logger.info(f"  [{status}] ToolAnnotations on all 11 tools")

        # Test 1: browser_navigate
        logger.info("\n--- 1/11: browser_navigate ---")
        resp = await client.call_tool("browser_navigate", {"url": "https://httpbin.org"})
        text = get_text(resp)
        status = PASS if "Navigated to" in text or "httpbin" in text.lower() else FAIL
        results.append(("browser_navigate", status))
        logger.info(f"  [{status}] {text[:80]}")

        # Give browser time to load
        await asyncio.sleep(2)

        # Test 2: browser_get_state (compact)
        logger.info("\n--- 2/11: browser_get_state ---")
        resp = await client.call_tool("browser_get_state", {"compact": True})
        text = get_text(resp)
        state = json.loads(text) if text.startswith("{") else {}
        status = PASS if "httpbin" in state.get("url", "") else FAIL
        results.append(("browser_get_state (compact)", status))
        logger.info(f"  [{status}] url={state.get('url', 'N/A')}, elements={state.get('interactive_element_count', 'N/A')}")

        # browser_get_state (full)
        resp = await client.call_tool("browser_get_state", {"compact": False})
        text = get_text(resp)
        state_full = json.loads(text) if text.startswith("{") else {}
        has_elements = "interactive_elements" in state_full
        status = PASS if has_elements else FAIL
        results.append(("browser_get_state (full)", status))
        logger.info(f"  [{status}] interactive_elements present={has_elements}")

        # browser_get_state (filter)
        resp = await client.call_tool("browser_get_state", {"filter_by": "tag", "filter_query": "a"})
        text = get_text(resp)
        filter_result = json.loads(text) if text.startswith("{") else {}
        status = PASS if filter_result.get("count", 0) > 0 else FAIL
        results.append(("browser_get_state (filter)", status))
        logger.info(f"  [{status}] filter by tag 'a': count={filter_result.get('count', 0)}")

        # Test 3: browser_get_text (plain)
        logger.info("\n--- 3/11: browser_get_text ---")
        resp = await client.call_tool("browser_get_text", {})
        text = get_text(resp)
        status = PASS if "httpbin" in text.lower() or "HTTP" in text else FAIL
        results.append(("browser_get_text (plain)", status))
        logger.info(f"  [{status}] text length={len(text)}, contains httpbin")

        # browser_get_text (search)
        resp = await client.call_tool("browser_get_text", {
            "search": "HTTP Methods",
            "context_lines": 1,
            "case_insensitive": True,
        })
        text = get_text(resp)
        search_result = json.loads(text) if text.startswith("{") else {}
        status = PASS if search_result.get("total_matches", 0) > 0 else FAIL
        results.append(("browser_get_text (search)", status))
        logger.info(f"  [{status}] search 'HTTP Methods': matches={search_result.get('total_matches', 0)}")

        # Test 4: browser_click
        logger.info("\n--- 4/11: browser_click ---")
        # Find a link to click
        resp = await client.call_tool("browser_get_state", {"filter_by": "text", "filter_query": "HTTP Methods"})
        text = get_text(resp)
        filter_r = json.loads(text) if text.startswith("{") else {}
        click_results = filter_r.get("results", [])
        if click_results:
            # Find the <a> tag
            link = next((r for r in click_results if r.get("tag") == "a"), click_results[0])
            idx = link["index"]
            resp = await client.call_tool("browser_click", {"index": idx})
            text = get_text(resp)
            status = PASS if "Clicked" in text else FAIL
            results.append(("browser_click", status))
            logger.info(f"  [{status}] {text[:80]}")
        else:
            results.append(("browser_click", SKIP))
            logger.info(f"  [SKIP] No clickable element found")

        await asyncio.sleep(1)

        # Test 5: browser_go_back
        logger.info("\n--- 5/11: browser_go_back ---")
        resp = await client.call_tool("browser_go_back", {})
        text = get_text(resp)
        status = PASS if "Navigated back" in text else FAIL
        results.append(("browser_go_back", status))
        logger.info(f"  [{status}] {text[:80]}")

        await asyncio.sleep(1)

        # Test 6: browser_scroll (direction + target_text)
        logger.info("\n--- 6/11: browser_scroll ---")
        resp = await client.call_tool("browser_scroll", {"direction": "down"})
        text = get_text(resp)
        status = PASS if "Scrolled" in text else FAIL
        results.append(("browser_scroll (down)", status))
        logger.info(f"  [{status}] {text[:80]}")

        resp = await client.call_tool("browser_scroll", {"direction": "up"})
        text = get_text(resp)
        status = PASS if "Scrolled" in text else FAIL
        results.append(("browser_scroll (up)", status))
        logger.info(f"  [{status}] {text[:80]}")

        resp = await client.call_tool("browser_scroll", {"target_text": "Other Utilities"})
        text = get_text(resp)
        status = PASS if "Found and scrolled" in text or "scrolled" in text.lower() else FAIL
        results.append(("browser_scroll (target_text)", status))
        logger.info(f"  [{status}] {text[:80]}")

        # Test 7: browser_type
        logger.info("\n--- 7/11: browser_type ---")
        # Navigate to form page
        await client.call_tool("browser_navigate", {"url": "https://httpbin.org/forms/post"})
        await asyncio.sleep(2)

        resp = await client.call_tool("browser_get_state", {"filter_by": "tag", "filter_query": "input"})
        text = get_text(resp)
        inputs = json.loads(text) if text.startswith("{") else {}
        input_results = inputs.get("results", [])
        # Find the text input (first one without a type or type=text)
        text_input = next((r for r in input_results if r.get("tag") == "input" and not r.get("type")), None)
        if text_input:
            resp = await client.call_tool("browser_type", {"index": text_input["index"], "text": "E2E test v0.1.17"})
            text = get_text(resp)
            status = PASS if "Typed" in text else FAIL
            results.append(("browser_type", status))
            logger.info(f"  [{status}] {text[:80]}")
        else:
            results.append(("browser_type", SKIP))
            logger.info(f"  [SKIP] No text input found")

        # Test 8: browser_tab (list + switch + close)
        logger.info("\n--- 8/11: browser_tab ---")
        # Open new tab
        await client.call_tool("browser_navigate", {"url": "https://httpbin.org/get", "new_tab": True})
        await asyncio.sleep(2)

        # List tabs
        resp = await client.call_tool("browser_tab", {"action": "list"})
        text = get_text(resp)
        tabs = json.loads(text) if text.startswith("[") else []
        status = PASS if len(tabs) >= 2 else FAIL
        results.append(("browser_tab (list)", status))
        logger.info(f"  [{status}] tab count={len(tabs)}")

        if len(tabs) >= 2:
            # Switch to first tab
            first_tab_id = tabs[0]["tab_id"]
            resp = await client.call_tool("browser_tab", {"action": "switch", "tab_id": first_tab_id})
            text = get_text(resp)
            status = PASS if "Switched" in text else FAIL
            results.append(("browser_tab (switch)", status))
            logger.info(f"  [{status}] {text[:80]}")

            # Close second tab
            second_tab_id = tabs[1]["tab_id"]
            resp = await client.call_tool("browser_tab", {"action": "close", "tab_id": second_tab_id})
            text = get_text(resp)
            status = PASS if "Closed" in text else FAIL
            results.append(("browser_tab (close)", status))
            logger.info(f"  [{status}] {text[:80]}")
        else:
            results.append(("browser_tab (switch)", SKIP))
            results.append(("browser_tab (close)", SKIP))

        # Test 9: browser_get_accessibility_tree (tree + flat)
        logger.info("\n--- 9/11: browser_get_accessibility_tree ---")
        resp = await client.call_tool("browser_get_accessibility_tree", {"format": "tree", "max_depth": 2})
        text = get_text(resp)
        tree_result = json.loads(text) if text.startswith("{") else {}
        status = PASS if "tree" in tree_result and tree_result.get("total_nodes_in_page", 0) > 0 else FAIL
        results.append(("browser_get_accessibility_tree (tree)", status))
        logger.info(f"  [{status}] tree nodes={tree_result.get('total_nodes', 0)}, page_nodes={tree_result.get('total_nodes_in_page', 0)}")

        resp = await client.call_tool("browser_get_accessibility_tree", {"format": "flat", "max_depth": 1})
        text = get_text(resp)
        flat_result = json.loads(text) if text.startswith("{") else {}
        has_parent_id = any("parent_id" in n for n in flat_result.get("nodes", []))
        status = PASS if "nodes" in flat_result and has_parent_id else FAIL
        results.append(("browser_get_accessibility_tree (flat)", status))
        logger.info(f"  [{status}] flat nodes={len(flat_result.get('nodes', []))}, has parent_id={has_parent_id}")

        # Test 10: browser_execute_js
        logger.info("\n--- 10/11: browser_execute_js ---")
        # Simple expression
        resp = await client.call_tool("browser_execute_js", {"expression": "window.location.href"})
        text = get_text(resp)
        js_result = json.loads(text) if text.startswith("{") else {}
        status = PASS if js_result.get("type") == "string" else FAIL
        results.append(("browser_execute_js (simple)", status))
        logger.info(f"  [{status}] result={js_result.get('result', 'N/A')[:60]}")

        # IIFE
        resp = await client.call_tool("browser_execute_js", {
            "expression": "(()=>{ return {count: document.querySelectorAll('input').length}; })()"
        })
        text = get_text(resp)
        js_result = json.loads(text) if text.startswith("{") else {}
        status = PASS if js_result.get("type") == "object" and isinstance(js_result.get("result"), dict) else FAIL
        results.append(("browser_execute_js (IIFE)", status))
        logger.info(f"  [{status}] result={js_result.get('result', 'N/A')}")

        # await_promise=false
        resp = await client.call_tool("browser_execute_js", {
            "expression": "new Promise(r => setTimeout(() => r('done'), 5000))",
            "await_promise": False,
        })
        text = get_text(resp)
        js_result = json.loads(text) if text.startswith("{") else {}
        status = PASS if js_result.get("type") == "object" else FAIL
        results.append(("browser_execute_js (await_promise=false)", status))
        logger.info(f"  [{status}] returned immediately, type={js_result.get('type')}")

        # return_by_value=false
        resp = await client.call_tool("browser_execute_js", {
            "expression": "document.querySelector('button')",
            "return_by_value": False,
        })
        text = get_text(resp)
        js_result = json.loads(text) if text.startswith("{") else {}
        has_object_id = "objectId" in js_result
        status = PASS if has_object_id else FAIL
        results.append(("browser_execute_js (return_by_value=false)", status))
        logger.info(f"  [{status}] objectId present={has_object_id}, className={js_result.get('className', 'N/A')}")

        # Test 11: browser_session (list + close + close_all)
        logger.info("\n--- 11/11: browser_session ---")
        resp = await client.call_tool("browser_session", {"action": "list"})
        text = get_text(resp)
        sessions = json.loads(text) if text.startswith("[") else []
        status = PASS if len(sessions) >= 1 else FAIL
        results.append(("browser_session (list)", status))
        logger.info(f"  [{status}] active sessions={len(sessions)}")

        if sessions:
            session_id = sessions[0]["session_id"]
            resp = await client.call_tool("browser_session", {"action": "close", "session_id": session_id})
            text = get_text(resp)
            status = PASS if "Successfully closed" in text else FAIL
            results.append(("browser_session (close)", status))
            logger.info(f"  [{status}] {text[:80]}")

        # Create new session for close_all test
        await client.call_tool("browser_navigate", {"url": "https://httpbin.org/get"})
        await asyncio.sleep(2)

        resp = await client.call_tool("browser_session", {"action": "close_all"})
        text = get_text(resp)
        status = PASS if "Closed" in text else FAIL
        results.append(("browser_session (close_all)", status))
        logger.info(f"  [{status}] {text[:80]}")

    except Exception as e:
        logger.info(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.stop()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    pass_count = sum(1 for _, s in results if s == PASS)
    fail_count = sum(1 for _, s in results if s == FAIL)
    skip_count = sum(1 for _, s in results if s == SKIP)
    for name, status in results:
        logger.info(f"  [{status}] {name}")
    logger.info(f"\nTotal: {pass_count} PASS, {fail_count} FAIL, {skip_count} SKIP out of {len(results)} tests")
    logger.info("=" * 60)

    return fail_count == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
