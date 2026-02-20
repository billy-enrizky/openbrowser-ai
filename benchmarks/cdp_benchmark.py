"""
Benchmark Chrome DevTools MCP (official): measure response sizes for identical tasks.
Communicates via JSON-RPC stdio with Content-Length framing.
Same tasks as playwright_benchmark.py and openbrowser_benchmark.py for apple-to-apple comparison.

GitHub: https://github.com/ChromeDevTools/chrome-devtools-mcp
npm: chrome-devtools-mcp
"""

import asyncio
import json
import logging
import re
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CDPMCPClient:
    """JSON-RPC client for Chrome DevTools MCP (Content-Length framing)."""

    def __init__(self):
        self.proc = None
        self._id = 0
        self._buffer = b""

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            "npx", "-y", "chrome-devtools-mcp@latest",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **__import__("os").environ,
                "CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS": "1",
            },
        )
        resp = await self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "benchmark", "version": "1.0"},
        }, timeout=120.0)
        logger.info("Initialized: %s", resp.get("serverInfo", {}).get("name", "unknown"))
        await self._notify("notifications/initialized", {})
        return resp

    async def stop(self):
        if self.proc:
            self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.proc.kill()

    async def list_tools(self):
        resp = await self._call("tools/list", {})
        tools = resp.get("tools", [])
        return tools

    async def call_tool(self, name, arguments, timeout=120.0):
        request_str = json.dumps({"name": name, "arguments": arguments})
        request_chars = len(request_str)

        start = time.monotonic()
        result = await self._call("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
        latency_ms = (time.monotonic() - start) * 1000

        response_str = json.dumps(result)
        response_chars = len(response_str)

        return result, latency_ms, request_chars, response_chars

    async def _call(self, method, params, timeout=60.0):
        self._id += 1
        msg = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        # Chrome DevTools MCP uses bare newline-delimited JSON (no Content-Length)
        self.proc.stdin.write(msg.encode() + b"\n")
        await self.proc.stdin.drain()
        return await self._read_response(self._id, timeout=timeout)

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
                    raise ConnectionError("Chrome DevTools MCP stdout closed")
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


def extract_uid_from_snapshot(result, tag_filter=None, text_filter=None):
    """Extract a uid from a take_snapshot response.

    Chrome DevTools MCP snapshot format:
      uid=1_0 RootWebArea url="..."
      uid=1_1 link "Article title"
      uid=1_2 textbox "Customer name: "

    UIDs are at the start of lines as 'uid=VALUE ROLE "label"'.
    """
    snapshot_text = ""
    if isinstance(result, dict) and "content" in result:
        for item in result.get("content", []):
            if item.get("type") == "text":
                snapshot_text += item.get("text", "")

    if not snapshot_text:
        return None

    # Match lines like: uid=1_5 link "Some text"
    if text_filter:
        pattern = re.compile(
            rf'uid=(\S+)\s+\S+\s+[^\n]*{re.escape(text_filter)}',
            re.IGNORECASE,
        )
        match = pattern.search(snapshot_text)
        if match:
            return match.group(1)

    if tag_filter == "link":
        # Find first link element
        link_pattern = re.compile(r'uid=(\S+)\s+link\s', re.IGNORECASE)
        match = link_pattern.search(snapshot_text)
        if match:
            return match.group(1)

    if tag_filter == "textbox":
        # Find first textbox element
        input_pattern = re.compile(r'uid=(\S+)\s+textbox\s', re.IGNORECASE)
        match = input_pattern.search(snapshot_text)
        if match:
            return match.group(1)

    # Last fallback: first non-root uid
    all_uids = re.findall(r'uid=(\S+)\s+(?!RootWebArea)', snapshot_text)
    if all_uids:
        return all_uids[0]

    return None


async def main():
    client = CDPMCPClient()
    results = []

    try:
        await client.start()

        tools = await client.list_tools()
        tool_names = [t["name"] for t in tools]
        logger.info("Chrome DevTools MCP tools: %d -- %s", len(tools), ", ".join(tool_names))

        # --- Task 1: Navigate (same as Playwright/OpenBrowser) ---
        logger.info("=== Task 1: Navigate to Wikipedia Python page ===")
        result, lat, req_c, res_c = await client.call_tool("navigate_page", {
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "type": "url",
        }, timeout=120.0)
        results.append(("navigate", req_c, res_c, lat))
        logger.info("  navigate: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # Wait for page to fully load
        logger.info("  Waiting for page to fully load...")
        await asyncio.sleep(5)

        # --- Task 2: Get page state (take_snapshot) ---
        logger.info("=== Task 2: Take snapshot (full page state) ===")
        result, lat, req_c, res_c = await client.call_tool("take_snapshot", {})
        results.append(("take_snapshot", req_c, res_c, lat))
        logger.info("  take_snapshot: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # Extract a link uid for clicking
        link_uid = extract_uid_from_snapshot(result, tag_filter="link")
        logger.info("  Extracted link uid for click: %s", link_uid)

        # --- Task 3: Click an element ---
        logger.info("=== Task 3: Click element ===")
        if link_uid:
            result, lat, req_c, res_c = await client.call_tool("click", {"uid": link_uid})
            results.append(("click", req_c, res_c, lat))
            logger.info("  click: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)
        else:
            logger.warning("  No uid found for click, skipping")

        # Wait for navigation after click
        await asyncio.sleep(5)

        # --- Task 4: Navigate back ---
        logger.info("=== Task 4: Navigate back ===")
        result, lat, req_c, res_c = await client.call_tool("navigate_page", {"type": "back"})
        results.append(("navigate_back", req_c, res_c, lat))
        logger.info("  navigate_back: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # Wait for page to fully reload after back
        await asyncio.sleep(5)

        # --- Task 5: Take snapshot again ---
        logger.info("=== Task 5: Take snapshot after go back ===")
        result, lat, req_c, res_c = await client.call_tool("take_snapshot", {})
        results.append(("take_snapshot_2", req_c, res_c, lat))
        logger.info("  take_snapshot_2: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # --- Task 6: Navigate to form ---
        logger.info("=== Task 6: Navigate to httpbin form ===")
        result, lat, req_c, res_c = await client.call_tool("navigate_page", {
            "url": "https://httpbin.org/forms/post",
            "type": "url",
        })
        results.append(("navigate_form", req_c, res_c, lat))
        logger.info("  navigate_form: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        await asyncio.sleep(3)

        # --- Task 7: Snapshot of form ---
        logger.info("=== Task 7: Take snapshot of form ===")
        result, lat, req_c, res_c = await client.call_tool("take_snapshot", {})
        results.append(("take_snapshot_form", req_c, res_c, lat))
        logger.info("  take_snapshot_form: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

        # Extract input uid for filling
        input_uid = extract_uid_from_snapshot(result, tag_filter="textbox")
        logger.info("  Extracted input uid for fill: %s", input_uid)

        # --- Task 8: Fill form field ---
        logger.info("=== Task 8: Fill form field ===")
        if input_uid:
            result, lat, req_c, res_c = await client.call_tool("fill", {
                "uid": input_uid,
                "value": "John Doe",
            })
            results.append(("fill", req_c, res_c, lat))
            logger.info("  fill: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)
        else:
            logger.warning("  No input uid found, skipping fill")

        # --- Task 9: List pages ---
        logger.info("=== Task 9: List pages ===")
        result, lat, req_c, res_c = await client.call_tool("list_pages", {})
        results.append(("list_pages", req_c, res_c, lat))
        logger.info("  list_pages: req=%d chars, resp=%d chars, latency=%.0f ms", req_c, res_c, lat)

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
        with open("benchmarks/cdp_results.json", "w") as f:
            json.dump({
                "mcp_server": "chrome-devtools-mcp",
                "mcp_version": "latest",
                "github": "https://github.com/ChromeDevTools/chrome-devtools-mcp",
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
        logger.info("Results written to benchmarks/cdp_results.json")

    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
