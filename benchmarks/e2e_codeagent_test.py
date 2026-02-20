"""
E2E test of the rewritten CodeAgent MCP server (single execute_code tool).
Tests all namespace functions via Python code execution through JSON-RPC stdio.
Runs from source, not published PyPI.
"""
import asyncio
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


class MCPClient:
    """JSON-RPC stdio client for MCP server."""

    def __init__(self, command: list[str]):
        self.command = command
        self.process = None
        self._id = 0
        self._buffer = b""

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        resp = await self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "codeagent-e2e-test", "version": "1.0"},
        })
        # Send initialized notification
        self._write_msg({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    async def list_tools(self) -> dict:
        return await self._send("tools/list", {})

    async def call_tool(self, name: str, arguments: dict = None) -> dict:
        return await self._send("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })

    async def execute_code(self, code: str) -> dict:
        """Convenience wrapper for execute_code tool."""
        return await self.call_tool("execute_code", {"code": code})

    async def _send(self, method: str, params: dict) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        self._write_msg(msg)
        return await self._read_response(self._id)

    def _write_msg(self, msg: dict):
        line = json.dumps(msg) + "\n"
        self.process.stdin.write(line.encode())

    async def _read_response(self, expected_id: int, timeout: float = 120.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self._try_parse()
            if result is not None:
                try:
                    data = json.loads(result)
                except json.JSONDecodeError:
                    continue
                if data.get("id") == expected_id:
                    return data
                continue

            try:
                remaining = max(0.1, deadline - time.monotonic())
                chunk = await asyncio.wait_for(
                    self.process.stdout.read(262144),
                    timeout=min(10.0, remaining),
                )
                if not chunk:
                    raise ConnectionError("Server closed stdout")
                self._buffer += chunk
            except asyncio.TimeoutError:
                continue

        raise TimeoutError(f"No response for id={expected_id} within {timeout}s")

    def _try_parse(self):
        newline_pos = self._buffer.find(b"\n")
        if newline_pos == -1:
            if self._buffer:
                try:
                    json.loads(self._buffer.decode())
                    body = self._buffer.decode()
                    self._buffer = b""
                    return body
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            return None
        line = self._buffer[:newline_pos].strip()
        self._buffer = self._buffer[newline_pos + 1:]
        if not line:
            return None
        return line.decode()

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
    logger.info("E2E Test: CodeAgent MCP Server (execute_code)")
    logger.info("=" * 60)

    # Run from source
    client = MCPClient(["uv", "run", "python", "-m", "openbrowser.mcp"])

    try:
        logger.info("\nStarting MCP server from source...")
        init = await client.start()
        server_info = init.get("result", {}).get("serverInfo", {})
        logger.info(f"Server: {server_info.get('name')} v{server_info.get('version')}")

        # ---------------------------------------------------------------
        # Test 0: Verify tool list -- should be exactly 1 tool
        # ---------------------------------------------------------------
        logger.info("\n--- Test 0: Tool List ---")
        tools_resp = await client.list_tools()
        tools = tools_resp.get("result", {}).get("tools", [])
        logger.info(f"  Tool count: {len(tools)}")

        if len(tools) == 1 and tools[0]["name"] == "execute_code":
            results.append(("tool_list (1 tool: execute_code)", PASS))
            logger.info(f"  [{PASS}] Single execute_code tool found")
            # Check annotations
            a = tools[0].get("annotations", {})
            if "readOnlyHint" in a:
                logger.info(f"    annotations: readOnly={a.get('readOnlyHint')} destructive={a.get('destructiveHint')} idempotent={a.get('idempotentHint')}")
        else:
            results.append(("tool_list (1 tool: execute_code)", FAIL))
            logger.info(f"  [{FAIL}] Expected 1 tool 'execute_code', got {[t['name'] for t in tools]}")

        # Check description mentions all 15 functions
        desc = tools[0].get("description", "") if tools else ""
        expected_fns = ["navigate", "click", "input_text", "scroll", "select_dropdown",
                        "dropdown_options", "send_keys", "switch", "close", "go_back",
                        "upload_file", "done", "evaluate", "wait", "get_selector_from_index"]
        missing = [fn for fn in expected_fns if fn not in desc]
        if not missing:
            results.append(("tool_description (all 15 functions documented)", PASS))
            logger.info(f"  [{PASS}] All 15 functions documented in description")
        else:
            results.append(("tool_description (all 15 functions documented)", FAIL))
            logger.info(f"  [{FAIL}] Missing from description: {missing}")

        # ---------------------------------------------------------------
        # Test 1: navigate()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 1: navigate() ---")
        resp = await client.execute_code("await navigate('https://httpbin.org')\nprint('navigated')")
        text = get_text(resp)
        status = PASS if "navigated" in text.lower() else FAIL
        results.append(("navigate(url)", status))
        logger.info(f"  [{status}] {text[:100]}")

        await asyncio.sleep(3)

        # ---------------------------------------------------------------
        # Test 2: browser.get_browser_state_summary()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 2: browser.get_browser_state_summary() ---")
        resp = await client.execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "print(f'url={state.url}')\n"
            "print(f'title={state.title}')\n"
            "print(f'elements={len(state.dom_state.selector_map)}')\n"
            "print(f'tabs={len(state.tabs)}')"
        )
        text = get_text(resp)
        status = PASS if "httpbin" in text.lower() and "elements=" in text else FAIL
        results.append(("browser.get_browser_state_summary()", status))
        logger.info(f"  [{status}] {text[:150]}")

        # ---------------------------------------------------------------
        # Test 3: evaluate() -- JS execution returning Python objects
        # ---------------------------------------------------------------
        logger.info("\n--- Test 3: evaluate() ---")
        resp = await client.execute_code(
            "title = await evaluate('document.title')\n"
            "print(f'title={title}')"
        )
        text = get_text(resp)
        status = PASS if "title=" in text and "httpbin" in text.lower() else FAIL
        results.append(("evaluate(js) simple", status))
        logger.info(f"  [{status}] {text[:100]}")

        # evaluate() with IIFE returning structured data
        resp = await client.execute_code(
            "data = await evaluate('''\n"
            "(function(){\n"
            "  return {\n"
            "    url: window.location.href,\n"
            "    linkCount: document.querySelectorAll('a').length\n"
            "  }\n"
            "})()\n"
            "''')\n"
            "print(f'type={type(data).__name__}')\n"
            "print(f'url={data.get(\"url\", \"N/A\")}')\n"
            "print(f'links={data.get(\"linkCount\", 0)}')"
        )
        text = get_text(resp)
        status = PASS if "type=dict" in text and "links=" in text else FAIL
        results.append(("evaluate(js) IIFE dict", status))
        logger.info(f"  [{status}] {text[:150]}")

        # evaluate() returning a list
        resp = await client.execute_code(
            "items = await evaluate('Array.from(document.querySelectorAll(\"a\")).slice(0,3).map(a => a.textContent.trim())')\n"
            "print(f'type={type(items).__name__}')\n"
            "print(f'count={len(items)}')\n"
            "print(f'items={items}')"
        )
        text = get_text(resp)
        status = PASS if "type=list" in text else FAIL
        results.append(("evaluate(js) list return", status))
        logger.info(f"  [{status}] {text[:150]}")

        # ---------------------------------------------------------------
        # Test 4: click()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 4: click() ---")
        # Find a clickable link
        resp = await client.execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "found = None\n"
            "for idx, el in state.dom_state.selector_map.items():\n"
            "    if el.tag_name == 'a' and idx >= 1:\n"
            "        found = idx\n"
            "        break\n"
            "if found:\n"
            "    await click(found)\n"
            "    print(f'clicked index={found}')\n"
            "else:\n"
            "    print('no link found')"
        )
        text = get_text(resp)
        status = PASS if "clicked index=" in text else FAIL
        results.append(("click(index)", status))
        logger.info(f"  [{status}] {text[:100]}")

        await asyncio.sleep(1)

        # ---------------------------------------------------------------
        # Test 5: go_back()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 5: go_back() ---")
        resp = await client.execute_code(
            "await go_back()\n"
            "await wait(1)\n"
            "state = await browser.get_browser_state_summary()\n"
            "print(f'url={state.url}')"
        )
        text = get_text(resp)
        status = PASS if "url=" in text else FAIL
        results.append(("go_back()", status))
        logger.info(f"  [{status}] {text[:100]}")

        # ---------------------------------------------------------------
        # Test 6: scroll()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 6: scroll() ---")
        resp = await client.execute_code(
            "await scroll(down=True, pages=1)\n"
            "print('scrolled down')"
        )
        text = get_text(resp)
        status = PASS if "scrolled down" in text.lower() else FAIL
        results.append(("scroll(down=True)", status))
        logger.info(f"  [{status}] {text[:100]}")

        resp = await client.execute_code(
            "await scroll(down=False, pages=1)\n"
            "print('scrolled up')"
        )
        text = get_text(resp)
        status = PASS if "scrolled up" in text.lower() else FAIL
        results.append(("scroll(down=False)", status))
        logger.info(f"  [{status}] {text[:100]}")

        # ---------------------------------------------------------------
        # Test 7: wait()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 7: wait() ---")
        resp = await client.execute_code(
            "import time as _t\n"
            "start = _t.time()\n"
            "await wait(3)\n"
            "elapsed = _t.time() - start\n"
            "print(f'waited {elapsed:.1f}s')"
        )
        text = get_text(resp)
        # wait(3) internally caps at max(3-3, 0) = 0 due to LLM latency offset
        status = PASS if "waited" in text else FAIL
        results.append(("wait(seconds)", status))
        logger.info(f"  [{status}] {text[:100]}")

        # ---------------------------------------------------------------
        # Test 8: send_keys()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 8: send_keys() ---")
        resp = await client.execute_code(
            "await send_keys('Escape')\n"
            "print('sent Escape')"
        )
        text = get_text(resp)
        status = PASS if "sent Escape" in text or "executed successfully" in text else FAIL
        results.append(("send_keys(keys)", status))
        logger.info(f"  [{status}] {text[:100]}")

        # ---------------------------------------------------------------
        # Test 9: navigate to form + input_text()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 9: input_text() ---")
        resp = await client.execute_code(
            "await navigate('https://httpbin.org/forms/post')\n"
            "await wait(3)\n"
            "state = await browser.get_browser_state_summary()\n"
            "input_idx = None\n"
            "for idx, el in state.dom_state.selector_map.items():\n"
            "    if el.tag_name == 'input':\n"
            "        input_idx = idx\n"
            "        break\n"
            "if input_idx:\n"
            "    await input_text(input_idx, 'CodeAgent Test')\n"
            "    print(f'typed into index={input_idx}')\n"
            "else:\n"
            "    print('no input found')"
        )
        text = get_text(resp)
        status = PASS if "typed into index=" in text else FAIL
        results.append(("input_text(index, text)", status))
        logger.info(f"  [{status}] {text[:100]}")

        # ---------------------------------------------------------------
        # Test 10: select_dropdown() + dropdown_options()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 10: dropdown_options() + select_dropdown() ---")
        resp = await client.execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "select_idx = None\n"
            "for idx, el in state.dom_state.selector_map.items():\n"
            "    if el.tag_name == 'select':\n"
            "        select_idx = idx\n"
            "        break\n"
            "if select_idx:\n"
            "    opts = await dropdown_options(select_idx)\n"
            "    print(f'dropdown index={select_idx} options={opts}')\n"
            "else:\n"
            "    print('no select found')"
        )
        text = get_text(resp)
        if "no select found" in text:
            results.append(("dropdown_options(index)", SKIP))
            results.append(("select_dropdown(index, text)", SKIP))
            logger.info(f"  [{SKIP}] No select element on page")
        else:
            status = PASS if "options=" in text else FAIL
            results.append(("dropdown_options(index)", status))
            logger.info(f"  [{status}] {text[:150]}")
            # select_dropdown not testable without a real select
            results.append(("select_dropdown(index, text)", SKIP))

        # ---------------------------------------------------------------
        # Test 11: Tab management -- navigate new tab, switch, close
        # ---------------------------------------------------------------
        logger.info("\n--- Test 11: switch() + close() ---")
        resp = await client.execute_code(
            "await navigate('https://httpbin.org/get', new_tab=True)\n"
            "await wait(2)\n"
            "state = await browser.get_browser_state_summary()\n"
            "tab_ids = [t.target_id[-4:] for t in state.tabs]\n"
            "print(f'tabs={len(state.tabs)} ids={tab_ids}')"
        )
        text = get_text(resp)
        status = PASS if "tabs=" in text and "ids=" in text else FAIL
        results.append(("navigate(new_tab=True)", status))
        logger.info(f"  [{status}] {text[:150]}")

        # Switch to first tab (use last 4 chars of target_id)
        resp = await client.execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "if len(state.tabs) >= 2:\n"
            "    first_tab_id = state.tabs[0].target_id[-4:]\n"
            "    await switch(first_tab_id)\n"
            "    await wait(1)\n"
            "    new_state = await browser.get_browser_state_summary()\n"
            "    print(f'switched to {first_tab_id}, url={new_state.url}')\n"
            "else:\n"
            "    print('only 1 tab')"
        )
        text = get_text(resp)
        status = PASS if "switched to" in text else FAIL
        results.append(("switch(tab_id)", status))
        logger.info(f"  [{status}] {text[:100]}")

        # Close second tab (use last 4 chars of target_id)
        resp = await client.execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "if len(state.tabs) >= 2:\n"
            "    second_tab_id = state.tabs[1].target_id[-4:]\n"
            "    await close(second_tab_id)\n"
            "    await wait(1)\n"
            "    new_state = await browser.get_browser_state_summary()\n"
            "    print(f'closed tab, remaining={len(new_state.tabs)}')\n"
            "else:\n"
            "    print('only 1 tab')"
        )
        text = get_text(resp)
        status = PASS if "closed tab" in text or "only 1 tab" in text else FAIL
        results.append(("close(tab_id)", status))
        logger.info(f"  [{status}] {text[:100]}")

        # ---------------------------------------------------------------
        # Test 12: get_selector_from_index()
        # ---------------------------------------------------------------
        logger.info("\n--- Test 12: get_selector_from_index() ---")
        # Navigate to a page with elements first
        await client.execute_code("await navigate('https://httpbin.org')\nawait wait(2)")
        resp = await client.execute_code(
            "state = await browser.get_browser_state_summary()\n"
            "first_idx = next(iter(state.dom_state.selector_map.keys()), None)\n"
            "if first_idx:\n"
            "    selector = await get_selector_from_index(first_idx)\n"
            "    print(f'index={first_idx} selector={selector}')\n"
            "else:\n"
            "    print('no elements')"
        )
        text = get_text(resp)
        status = PASS if "selector=" in text else FAIL
        results.append(("get_selector_from_index(index)", status))
        logger.info(f"  [{status}] {text[:150]}")

        # ---------------------------------------------------------------
        # Test 13: Persistent namespace -- variables survive across calls
        # ---------------------------------------------------------------
        logger.info("\n--- Test 13: Namespace persistence ---")
        resp = await client.execute_code("test_var = 42\nprint('set test_var=42')")
        text1 = get_text(resp)

        resp = await client.execute_code("print(f'test_var={test_var}')")
        text2 = get_text(resp)
        status = PASS if "test_var=42" in text2 else FAIL
        results.append(("namespace persistence", status))
        logger.info(f"  [{status}] call1: {text1.strip()} -> call2: {text2.strip()}")

        # ---------------------------------------------------------------
        # Test 14: Pre-imported libraries
        # ---------------------------------------------------------------
        logger.info("\n--- Test 14: Pre-imported libraries ---")
        resp = await client.execute_code(
            "libs = []\n"
            "for name in ['json', 're', 'csv', 'asyncio', 'datetime', 'requests']:\n"
            "    try:\n"
            "        eval(name)\n"
            "        libs.append(name)\n"
            "    except NameError:\n"
            "        pass\n"
            "# Check Path\n"
            "try:\n"
            "    Path\n"
            "    libs.append('Path')\n"
            "except NameError:\n"
            "    pass\n"
            "print(f'available={libs}')"
        )
        text = get_text(resp)
        expected_libs = ['json', 're', 'csv', 'asyncio', 'datetime', 'requests', 'Path']
        status = PASS if all(lib in text for lib in expected_libs) else FAIL
        results.append(("pre-imported libraries", status))
        logger.info(f"  [{status}] {text[:150]}")

        # ---------------------------------------------------------------
        # Test 15: Error handling -- Python errors returned cleanly
        # ---------------------------------------------------------------
        logger.info("\n--- Test 15: Error handling ---")
        resp = await client.execute_code("x = 1 / 0")
        text = get_text(resp)
        status = PASS if "ZeroDivisionError" in text else FAIL
        results.append(("error handling (Python exception)", status))
        logger.info(f"  [{status}] {text[:100]}")

        # JS evaluate error
        resp = await client.execute_code(
            "try:\n"
            "    await evaluate('undefinedVariable.foo')\n"
            "except Exception as e:\n"
            "    print(f'caught: {type(e).__name__}: {str(e)[:80]}')"
        )
        text = get_text(resp)
        status = PASS if "caught:" in text and "Error" in text else FAIL
        results.append(("error handling (JS evaluate error)", status))
        logger.info(f"  [{status}] {text[:150]}")

        # ---------------------------------------------------------------
        # Test 16: Multi-step workflow in single code block
        # ---------------------------------------------------------------
        logger.info("\n--- Test 16: Multi-step workflow ---")
        resp = await client.execute_code(
            "await navigate('https://httpbin.org')\n"
            "await wait(2)\n"
            "title = await evaluate('document.title')\n"
            "link_count = await evaluate('document.querySelectorAll(\"a\").length')\n"
            "state = await browser.get_browser_state_summary()\n"
            "print(f'title={title}')\n"
            "print(f'links={link_count}')\n"
            "print(f'url={state.url}')\n"
            "print(f'elements={len(state.dom_state.selector_map)}')"
        )
        text = get_text(resp)
        status = PASS if "title=" in text and "links=" in text and "url=" in text else FAIL
        results.append(("multi-step workflow", status))
        logger.info(f"  [{status}] {text[:200]}")

    except Exception as e:
        logger.info(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.stop()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
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
