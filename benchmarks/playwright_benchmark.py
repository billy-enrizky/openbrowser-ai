"""
Benchmark Playwright MCP: measure response sizes for identical tasks.
Communicates via JSON-RPC stdio with the Playwright MCP server.
"""

import asyncio
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PlaywrightMCPClient:
    def __init__(self):
        self.proc = None
        self._id = 0
        self._buffer = b""

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            "npx", "@playwright/mcp@latest",
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
        # Playwright MCP uses bare JSON (no Content-Length headers)
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
            # Try to parse complete JSON lines from buffer
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
                # Notification or wrong id, keep reading
                continue

            try:
                chunk = await asyncio.wait_for(
                    self.proc.stdout.read(262144),
                    timeout=min(10.0, max(0.1, deadline - time.monotonic())),
                )
                if not chunk:
                    raise ConnectionError("Playwright MCP stdout closed")
                self._buffer += chunk
            except asyncio.TimeoutError:
                continue

        raise TimeoutError(f"No response for id={expected_id}")

    def _try_parse(self):
        """Try to extract a complete JSON message from the buffer (newline-delimited)."""
        newline_pos = self._buffer.find(b"\n")
        if newline_pos == -1:
            # Try to parse the whole buffer as JSON (no trailing newline yet)
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
    client = PlaywrightMCPClient()
    results = []

    try:
        await client.start()

        tools = await client.list_tools()
        tool_names = [t["name"] for t in tools]
        logger.info("Playwright MCP tools: %d", len(tools))

        # --- Task 1: Navigate ---
        logger.info("=== Task 1: Navigate to Wikipedia Python page ===")
        result, lat, req_c, res_c = await client.call_tool("browser_navigate", {
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"
        })
        results.append(("navigate", req_c, res_c, lat))
        logger.info("  navigate: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 2: Get page snapshot (equivalent to get_state) ---
        logger.info("=== Task 2: Get page snapshot ===")
        result, lat, req_c, res_c = await client.call_tool("browser_snapshot", {})
        results.append(("snapshot", req_c, res_c, lat))
        logger.info("  snapshot: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # Save snapshot content for analysis
        if "content" in result:
            for item in result.get("content", []):
                if item.get("type") == "text":
                    text_len = len(item.get("text", ""))
                    logger.info("  snapshot text content: %d chars", text_len)

        # --- Task 3: Click first link ---
        logger.info("=== Task 3: Click element ===")
        # Playwright uses ref-based clicking from the snapshot
        # We need to extract a ref from the snapshot first
        snapshot_text = ""
        if isinstance(result, dict) and "content" in result:
            for item in result.get("content", []):
                if item.get("type") == "text":
                    snapshot_text = item.get("text", "")
                    break

        # Find a ref in the snapshot (format: [ref=XX])
        import re
        refs = re.findall(r'\[ref=(\w+)\]', snapshot_text)
        if refs:
            first_ref = refs[0]
            logger.info("  Using ref: %s", first_ref)
            result, lat, req_c, res_c = await client.call_tool("browser_click", {
                "element": "First clickable element",
                "ref": first_ref,
            })
            results.append(("click", req_c, res_c, lat))
            logger.info("  click: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)
        else:
            logger.warning("  No refs found in snapshot, skipping click")

        # --- Task 4: Navigate back ---
        logger.info("=== Task 4: Navigate back ===")
        result, lat, req_c, res_c = await client.call_tool("browser_navigate_back", {})
        results.append(("navigate_back", req_c, res_c, lat))
        logger.info("  navigate_back: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 5: Get snapshot again (after navigate back) ---
        logger.info("=== Task 5: Snapshot after navigate back ===")
        result, lat, req_c, res_c = await client.call_tool("browser_snapshot", {})
        results.append(("snapshot_2", req_c, res_c, lat))
        logger.info("  snapshot_2: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 6: Navigate to form page ---
        logger.info("=== Task 6: Navigate to httpbin form ===")
        result, lat, req_c, res_c = await client.call_tool("browser_navigate", {
            "url": "https://httpbin.org/forms/post"
        })
        results.append(("navigate_form", req_c, res_c, lat))
        logger.info("  navigate_form: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 7: Snapshot of form page ---
        logger.info("=== Task 7: Snapshot of form ===")
        result, lat, req_c, res_c = await client.call_tool("browser_snapshot", {})
        results.append(("snapshot_form", req_c, res_c, lat))
        logger.info("  snapshot_form: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # Extract form refs
        snapshot_text = ""
        if isinstance(result, dict) and "content" in result:
            for item in result.get("content", []):
                if item.get("type") == "text":
                    snapshot_text = item.get("text", "")
                    break

        # --- Task 8: Type into form field ---
        logger.info("=== Task 8: Type into form ===")
        input_refs = re.findall(r'textbox.*?\[ref=(\w+)\]', snapshot_text, re.IGNORECASE)
        if not input_refs:
            input_refs = re.findall(r'\[ref=(\w+)\]', snapshot_text)
        if input_refs:
            result, lat, req_c, res_c = await client.call_tool("browser_type", {
                "element": "Customer name input",
                "ref": input_refs[0],
                "text": "John Doe",
            })
            results.append(("type", req_c, res_c, lat))
            logger.info("  type: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)
        else:
            logger.warning("  No input refs found in form snapshot")

        # --- Task 9: Get tabs ---
        logger.info("=== Task 9: List tabs ===")
        result, lat, req_c, res_c = await client.call_tool("browser_tabs", {})
        results.append(("tabs", req_c, res_c, lat))
        logger.info("  tabs: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Summary ---
        logger.info("=== SUMMARY ===")
        logger.info("%-20s %10s %10s %10s", "Tool", "Req Chars", "Resp Chars", "Latency(ms)")
        total_req = 0
        total_resp = 0
        total_lat = 0.0
        for name, req_c, res_c, lat in results:
            logger.info("%-20s %10d %10d %10.0f", name, req_c, res_c, lat)
            total_req += req_c
            total_resp += res_c
            total_lat += lat
        logger.info("%-20s %10d %10d %10.0f", "TOTAL", total_req, total_resp, total_lat)
        logger.info("Total tool calls: %d", len(results))
        logger.info("Est. response tokens (chars/4): %d", total_resp // 4)

        # Write raw JSON results
        with open("benchmarks/playwright_results.json", "w") as f:
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
        logger.info("Results written to benchmarks/playwright_results.json")

    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
