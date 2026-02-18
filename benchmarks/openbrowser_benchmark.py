"""
Benchmark OpenBrowser MCP: measure response sizes for identical tasks.
Communicates via JSON-RPC stdio with the OpenBrowser MCP server.
Same tasks as playwright_benchmark.py for apple-to-apple comparison.
"""

import asyncio
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class OpenBrowserMCPClient:
    def __init__(self):
        self.proc = None
        self._id = 0
        self._buffer = b""

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            "uvx", "openbrowser-ai[mcp]@0.1.17", "--mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        resp = await self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "benchmark", "version": "1.0"},
        })
        logger.info("Initialized: %s", resp.get("serverInfo", {}).get("name", "unknown"))
        await self._notify("notifications/initialized", {})
        return resp

    async def stop(self):
        if self.proc:
            self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()

    async def list_tools(self):
        resp = await self._call("tools/list", {})
        tools = resp.get("tools", [])
        return tools

    async def call_tool(self, name, arguments):
        request_str = json.dumps({"name": name, "arguments": arguments})
        request_chars = len(request_str)

        start = time.monotonic()
        result = await self._call("tools/call", {"name": name, "arguments": arguments})
        latency_ms = (time.monotonic() - start) * 1000

        response_str = json.dumps(result)
        response_chars = len(response_str)

        return result, latency_ms, request_chars, response_chars

    async def _call(self, method, params):
        self._id += 1
        msg = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        # OpenBrowser MCP uses bare JSON (newline-delimited)
        self.proc.stdin.write(msg.encode() + b"\n")
        await self.proc.stdin.drain()
        return await self._read_response(self._id)

    async def _notify(self, method, params):
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        self.proc.stdin.write(msg.encode() + b"\n")
        await self.proc.stdin.drain()

    async def _read_response(self, expected_id, timeout=60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self._try_parse()
            if result is not None:
                try:
                    data = json.loads(result)
                except json.JSONDecodeError:
                    continue
                if data.get("id") == expected_id:
                    if "error" in data:
                        logger.warning("RPC error for id=%d: %s", expected_id, data["error"])
                    return data.get("result", data.get("error", {}))
                continue

            try:
                chunk = await asyncio.wait_for(
                    self.proc.stdout.read(262144),
                    timeout=min(10.0, max(0.1, deadline - time.monotonic())),
                )
                if not chunk:
                    raise ConnectionError("OpenBrowser MCP stdout closed")
                self._buffer += chunk
            except asyncio.TimeoutError:
                continue

        raise TimeoutError(f"No response for id={expected_id}")

    def _try_parse(self):
        """Try to extract a complete JSON message (newline-delimited)."""
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


async def main():
    client = OpenBrowserMCPClient()
    results = []

    try:
        await client.start()

        tools = await client.list_tools()
        tool_names = [t["name"] for t in tools]
        logger.info("OpenBrowser MCP tools: %d", len(tools))

        # --- Task 1: Navigate (same as Playwright) ---
        logger.info("=== Task 1: Navigate to Wikipedia Python page ===")
        result, lat, req_c, res_c = await client.call_tool("browser_navigate", {
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"
        })
        results.append(("navigate", req_c, res_c, lat))
        logger.info("  navigate: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # Wait for page to fully load (browser startup + page render)
        logger.info("  Waiting for page to fully load...")
        await asyncio.sleep(5)

        # --- Task 2: Get page state full (equivalent to Playwright snapshot) ---
        logger.info("=== Task 2: Get page state (full) ===")
        result, lat, req_c, res_c = await client.call_tool("browser_get_state", {"compact": False})
        results.append(("get_state_full", req_c, res_c, lat))
        logger.info("  get_state_full: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 2b: Also get compact state for comparison ---
        logger.info("=== Task 2b: Get page state (compact) ===")
        result_compact, lat_c, req_cc, res_cc = await client.call_tool("browser_get_state", {"compact": True})
        results.append(("get_state_compact", req_cc, res_cc, lat_c))
        logger.info("  get_state_compact: req=%d chars, resp=%d chars, latency=%.0f ms", req_cc, res_cc, lat_c)

        # --- Task 2c: Get text (OpenBrowser exclusive) ---
        logger.info("=== Task 2c: Get text ===")
        result_text, lat_t, req_ct, res_ct = await client.call_tool("browser_get_text", {})
        results.append(("get_text", req_ct, res_ct, lat_t))
        logger.info("  get_text: req=%d chars, resp=%d chars, latency=%.0f ms", req_ct, res_ct, lat_t)

        # --- Task 2d: Search (OpenBrowser exclusive -- consolidated into browser_get_text) ---
        logger.info("=== Task 2d: Search for specific content ===")
        result_grep, lat_g, req_cg, res_cg = await client.call_tool("browser_get_text", {
            "search": "Guido van Rossum",
            "context_lines": 2,
        })
        results.append(("search", req_cg, res_cg, lat_g))
        logger.info("  search: req=%d chars, resp=%d chars, latency=%.0f ms", req_cg, res_cg, lat_g)

        # --- Task 3: Click an element ---
        logger.info("=== Task 3: Click element ===")
        # OpenBrowser uses index-based clicking
        # Get the first clickable link from the state
        state_data = json.loads(json.dumps(result))  # deep copy
        first_link_index = 0
        if isinstance(result, dict) and "content" in result:
            for item in result.get("content", []):
                if item.get("type") == "text":
                    try:
                        state = json.loads(item.get("text", "{}"))
                        for elem in state.get("interactive_elements", []):
                            if elem.get("tag") == "a" and elem.get("href", "").startswith("/wiki/"):
                                first_link_index = elem["index"]
                                break
                    except json.JSONDecodeError:
                        pass

        result, lat, req_c, res_c = await client.call_tool("browser_click", {
            "index": first_link_index,
        })
        results.append(("click", req_c, res_c, lat))
        logger.info("  click: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 4: Navigate back ---
        logger.info("=== Task 4: Go back ===")
        result, lat, req_c, res_c = await client.call_tool("browser_go_back", {})
        results.append(("go_back", req_c, res_c, lat))
        logger.info("  go_back: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 5: Get state again ---
        logger.info("=== Task 5: Get state after go back ===")
        result, lat, req_c, res_c = await client.call_tool("browser_get_state", {"compact": True})
        results.append(("get_state_after_back", req_c, res_c, lat))
        logger.info("  get_state_after_back: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 6: Navigate to form ---
        logger.info("=== Task 6: Navigate to httpbin form ===")
        result, lat, req_c, res_c = await client.call_tool("browser_navigate", {
            "url": "https://httpbin.org/forms/post"
        })
        results.append(("navigate_form", req_c, res_c, lat))
        logger.info("  navigate_form: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # Wait for page load
        await asyncio.sleep(3)

        # --- Task 7: Get state of form ---
        logger.info("=== Task 7: Get state of form ===")
        result, lat, req_c, res_c = await client.call_tool("browser_get_state", {"compact": False})
        results.append(("get_state_form", req_c, res_c, lat))
        logger.info("  get_state_form: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 8: Type into form ---
        logger.info("=== Task 8: Type into form ===")
        # Find first input from state
        input_index = 0
        if isinstance(result, dict) and "content" in result:
            for item in result.get("content", []):
                if item.get("type") == "text":
                    try:
                        state = json.loads(item.get("text", "{}"))
                        for elem in state.get("interactive_elements", []):
                            if elem.get("tag") == "input":
                                input_index = elem["index"]
                                break
                    except json.JSONDecodeError:
                        pass

        result, lat, req_c, res_c = await client.call_tool("browser_type", {
            "index": input_index,
            "text": "John Doe",
        })
        results.append(("type", req_c, res_c, lat))
        logger.info("  type: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 9: List tabs (consolidated into browser_tab) ---
        logger.info("=== Task 9: List tabs ===")
        result, lat, req_c, res_c = await client.call_tool("browser_tab", {"action": "list"})
        results.append(("list_tabs", req_c, res_c, lat))
        logger.info("  list_tabs: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Summary ---
        logger.info("=== SUMMARY ===")
        logger.info("%-25s %10s %10s %10s", "Tool", "Req Chars", "Resp Chars", "Latency(ms)")
        total_req = 0
        total_resp = 0
        total_lat = 0.0
        for name, req_c, res_c, lat in results:
            logger.info("%-25s %10d %10d %10.0f", name, req_c, res_c, lat)
            total_req += req_c
            total_resp += res_c
            total_lat += lat
        logger.info("%-25s %10d %10d %10.0f", "TOTAL", total_req, total_resp, total_lat)
        logger.info("Total tool calls: %d", len(results))
        logger.info("Est. response tokens (chars/4): %d", total_resp // 4)

        # Write raw JSON results
        with open("benchmarks/openbrowser_results.json", "w") as f:
            json.dump({
                "tool_count": len(tools),
                "tool_names": tool_names,
                "measurements": [
                    {"tool": name, "request_chars": req_c, "response_chars": res_c, "latency_ms": round(lat, 1)}
                    for name, req_c, res_c, lat in results
                ],
                "totals": {
                    "tool_calls": len(results),
                    "total_request_chars": total_req,
                    "total_response_chars": total_resp,
                    "total_latency_ms": round(total_lat, 1),
                    "est_response_tokens": total_resp // 4,
                }
            }, f, indent=2)
        logger.info("Results written to benchmarks/openbrowser_results.json")

    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
