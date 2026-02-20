"""
E2E test for the published openbrowser-ai[mcp]==0.1.22 package.

Starts the MCP server via uvx as a subprocess, communicates via JSON-RPC 2.0
over stdio, and validates:
  1. initialize handshake
  2. tools/list returns exactly 1 tool named "execute_code"
  3. execute_code with navigation to example.com
  4. execute_code with variable persistence across calls
  5. execute_code with evaluate() for JS execution

Usage:
    uv run python benchmarks/e2e_published_test.py
"""

import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("e2e_published_test")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kill_browsers():
    """Kill all Chrome/Chromium processes and wait until they are fully gone."""
    for pattern in ["chromium", "chrome", "Chromium", "Google Chrome"]:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass

    for _ in range(20):
        result = subprocess.run(
            ["pgrep", "-f", "chrom"],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            break
        time.sleep(0.5)
    else:
        logger.warning("Chrome processes still alive after 10s wait")

    time.sleep(1)


# ---------------------------------------------------------------------------
# Minimal MCP/JSON-RPC client over stdio
# ---------------------------------------------------------------------------

class MCPStdioClient:
    """Communicates with an MCP server via JSON-RPC 2.0 over stdin/stdout."""

    def __init__(self, command: list[str]):
        self._command = command
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    def start_process(self) -> None:
        """Launch the server subprocess."""
        env = os.environ.copy()
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        logger.info("Server process started (pid=%s)", self._process.pid)

    def stop(self) -> None:
        """Terminate the server subprocess."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            logger.info("Server process stopped")
            self._process = None

    def send_request(self, method: str, params: dict, timeout: float = 60.0) -> dict:
        """Send a JSON-RPC request and return the parsed response dict."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }
        return self._write_and_read(request, timeout=timeout)

    def send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        assert self._process and self._process.stdin
        line = json.dumps(notification) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

    # ------------------------------------------------------------------

    def _write_and_read(self, request: dict, timeout: float = 60.0) -> dict:
        """Write a JSON-RPC line and read the next response line.

        Skips any notification lines (lines without an 'id' field) that the
        server may emit before the actual response.
        """
        assert self._process and self._process.stdin and self._process.stdout
        line = json.dumps(request) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response_line = self._process.stdout.readline()
            if not response_line:
                return {"error": "No response from server (EOF)"}
            try:
                data = json.loads(response_line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON line from server: %s", response_line[:200])
                continue

            # Skip server-initiated notifications (no 'id')
            if "id" not in data:
                logger.debug("Skipping server notification: %s", data.get("method", "?"))
                continue

            return data

        return {"error": f"Timeout waiting for response to request {request.get('id')}"}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail


def run_tests() -> list[TestResult]:
    results: list[TestResult] = []
    client = MCPStdioClient(["uvx", "openbrowser-ai[mcp]==0.1.22", "--mcp"])

    try:
        client.start_process()
        # Give the server a moment to boot
        time.sleep(2)

        # ---------------------------------------------------------------
        # Test 1: initialize handshake
        # ---------------------------------------------------------------
        test_name = "1. initialize handshake"
        logger.info("--- %s ---", test_name)
        try:
            resp = client.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e_published_test", "version": "1.0"},
            })
            if "error" in resp:
                results.append(TestResult(test_name, False, f"Error: {resp['error']}"))
            elif "result" in resp:
                proto = resp["result"].get("protocolVersion", "?")
                server_name = resp["result"].get("serverInfo", {}).get("name", "?")
                logger.info("Server: %s, protocol: %s", server_name, proto)
                results.append(TestResult(test_name, True, f"server={server_name}, proto={proto}"))
            else:
                results.append(TestResult(test_name, False, f"Unexpected response: {resp}"))
        except Exception as exc:
            results.append(TestResult(test_name, False, str(exc)))

        # Send initialized notification
        client.send_notification("notifications/initialized", {})
        time.sleep(0.5)

        # ---------------------------------------------------------------
        # Test 2: tools/list -- exactly 1 tool named "execute_code"
        # ---------------------------------------------------------------
        test_name = "2. tools/list returns execute_code"
        logger.info("--- %s ---", test_name)
        try:
            resp = client.send_request("tools/list", {})
            if "error" in resp:
                results.append(TestResult(test_name, False, f"Error: {resp['error']}"))
            else:
                tools = resp.get("result", {}).get("tools", [])
                tool_names = [t["name"] for t in tools]
                logger.info("Tools: %s", tool_names)
                if len(tools) == 1 and tools[0]["name"] == "execute_code":
                    results.append(TestResult(test_name, True, "1 tool: execute_code"))
                else:
                    results.append(TestResult(
                        test_name, False,
                        f"Expected exactly 1 tool 'execute_code', got {tool_names}"
                    ))
        except Exception as exc:
            results.append(TestResult(test_name, False, str(exc)))

        # ---------------------------------------------------------------
        # Test 3: execute_code -- navigate to example.com and get title
        # Uses the namespace-level navigate() and browser.get_browser_state_summary()
        # ---------------------------------------------------------------
        test_name = "3. execute_code: navigate to example.com"
        logger.info("--- %s ---", test_name)
        try:
            code = (
                'await navigate("https://example.com")\n'
                'state = await browser.get_browser_state_summary()\n'
                'print(state.title)'
            )
            resp = client.send_request("tools/call", {
                "name": "execute_code",
                "arguments": {"code": code},
            }, timeout=60)
            if "error" in resp:
                results.append(TestResult(test_name, False, f"Error: {resp['error']}"))
            else:
                content = resp.get("result", {}).get("content", [])
                text = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                logger.info("Response text: %s", text[:500])
                if "Example Domain" in text:
                    results.append(TestResult(test_name, True, "Title contains 'Example Domain'"))
                else:
                    results.append(TestResult(test_name, False, f"Expected 'Example Domain' in: {text[:300]}"))
        except Exception as exc:
            results.append(TestResult(test_name, False, str(exc)))

        # ---------------------------------------------------------------
        # Test 4: execute_code -- variable persistence across calls
        # Set a variable in one call, read it back in the next call.
        # ---------------------------------------------------------------
        test_name = "4. execute_code: variable persistence"
        logger.info("--- %s ---", test_name)
        try:
            # Call 1: set a variable
            set_code = 'my_test_var = "hello_from_e2e_test"'
            resp1 = client.send_request("tools/call", {
                "name": "execute_code",
                "arguments": {"code": set_code},
            }, timeout=30)
            if "error" in resp1:
                results.append(TestResult(test_name, False, f"Set var error: {resp1['error']}"))
            else:
                # Call 2: read the variable back using print()
                get_code = 'print(my_test_var)'
                resp2 = client.send_request("tools/call", {
                    "name": "execute_code",
                    "arguments": {"code": get_code},
                }, timeout=30)
                if "error" in resp2:
                    results.append(TestResult(test_name, False, f"Get var error: {resp2['error']}"))
                else:
                    content = resp2.get("result", {}).get("content", [])
                    text = " ".join(
                        item.get("text", "") for item in content if item.get("type") == "text"
                    )
                    logger.info("Persistence result: %s", text[:300])
                    if "hello_from_e2e_test" in text:
                        results.append(TestResult(test_name, True, "Variable persisted across calls"))
                    else:
                        results.append(TestResult(
                            test_name, False,
                            f"Expected 'hello_from_e2e_test' in: {text[:300]}"
                        ))
        except Exception as exc:
            results.append(TestResult(test_name, False, str(exc)))

        # ---------------------------------------------------------------
        # Test 5: execute_code -- evaluate() for JS execution
        # Uses the namespace-level evaluate() which runs JS via CDP.
        # Page should still be on example.com from test 3.
        # ---------------------------------------------------------------
        test_name = "5. execute_code: evaluate() JS execution"
        logger.info("--- %s ---", test_name)
        try:
            js_code = (
                'result = await evaluate("document.title")\n'
                'print(result)'
            )
            resp = client.send_request("tools/call", {
                "name": "execute_code",
                "arguments": {"code": js_code},
            }, timeout=30)
            if "error" in resp:
                results.append(TestResult(test_name, False, f"Error: {resp['error']}"))
            else:
                content = resp.get("result", {}).get("content", [])
                text = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                logger.info("JS evaluate result: %s", text[:300])
                # Page should still be on example.com from test 3
                if "Example Domain" in text:
                    results.append(TestResult(test_name, True, "JS evaluate returned page title"))
                else:
                    results.append(TestResult(
                        test_name, False,
                        f"Expected 'Example Domain' in: {text[:300]}"
                    ))
        except Exception as exc:
            results.append(TestResult(test_name, False, str(exc)))

    finally:
        client.stop()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== E2E Published Package Test (openbrowser-ai[mcp]==0.1.22) ===")

    # Kill any pre-existing browser processes
    kill_browsers()

    results = run_tests()

    # Kill browsers after tests
    kill_browsers()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    passed = 0
    failed = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.passed:
            passed += 1
        else:
            failed += 1
        logger.info("  [%s] %s -- %s", status, r.name, r.detail)

    logger.info("-" * 60)
    logger.info("  Total: %d | Passed: %d | Failed: %d", len(results), passed, failed)
    logger.info("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
