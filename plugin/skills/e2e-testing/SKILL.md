---
name: e2e-testing
description: |
  Test web applications end-to-end by simulating user interactions and verifying expected outcomes.
  Runs through keyless `openbrowser-ai -c` or local MCP code tools. No separate API key is required.
  Trigger when the user asks to: test a web app, verify a user flow, run end-to-end tests,
  QA a feature, check that a page works correctly, validate user journeys, or test a deployment.
allowed-tools: Bash(openbrowser-ai:*) Bash(curl:*) Bash(uv:*) Bash(irm:*) Read Write
---

# End-to-End Testing

<!-- toc:start -->
## Contents

- [1. Setup](#1-setup)
- [2. Workflow](#2-workflow)
  - [2.1. Step 1 -- Navigate and verify page load](#21-step-1----navigate-and-verify-page-load)
  - [2.2. Step 2 -- Content assertions](#22-step-2----content-assertions)
  - [2.3. Step 3 -- Test user interactions (login flow)](#23-step-3----test-user-interactions-login-flow)
  - [2.4. Step 4 -- Test navigation flows](#24-step-4----test-navigation-flows)
  - [2.5. Step 5 -- Test error handling](#25-step-5----test-error-handling)
  - [2.6. Step 6 -- Test responsive behavior](#26-step-6----test-responsive-behavior)
  - [2.7. Step 7 -- Test multi-page flows](#27-step-7----test-multi-page-flows)
- [3. Tips](#3-tips)
- [4. Cleanup](#4-cleanup)

<!-- toc:end -->

Simulate real user interactions and verify web application behavior using Python code execution. Covers navigation, form interaction, content assertions, and multi-page flows.

These tests need a browser because they navigate, click, and inspect rendered state. OpenBrowser no-browser mode starts no Chrome, Blink, or V8 process, but it cannot run JavaScript, take screenshots, or use browser cookies, so it cannot replace the workflow below.

All code runs via `openbrowser-ai -c`. The daemon starts automatically and persists variables across calls. Run `openbrowser-ai warm` once before a multi-call workflow to start the daemon and browser before the workflow begins. Most browser functions are async, use `await`. `list_downloads()`, `download_outcomes()`, and `download_failures()` are synchronous.

`openbrowser-ai -c` needs no separate OpenAI, OpenBrowser, or MCP API key.

The CLI daemon persists cookies and login state in the Chrome user-data directory at `~/.config/openbrowser/profiles/daemon`, so authenticated sessions can be reused across later runs. No `storage_state.json` file is written unless you configure one explicitly.

## 1. Setup

Before running, verify openbrowser-ai is installed:

```bash
openbrowser-ai --help
```

If not found, install:

```bash
# macOS/Linux
curl -fsSL https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/billy-enrizky/openbrowser-ai/main/install.ps1 | iex
```

## 2. Workflow

### 2.1. Step 1 -- Navigate and verify page load

```bash
openbrowser-ai -c - <<'EOF'
await navigate("https://staging.example.com")
state = await browser.get_browser_state_summary()

assert "example" in state.url.lower(), f"Unexpected URL: {state.url}"
assert state.title, "Page title is empty"
print(f"Page loaded: {state.title} ({state.url})")
EOF
```

### 2.2. Step 2 -- Content assertions

```bash
openbrowser-ai -c - <<'EOF'
# Check for expected text using JS
has_welcome = await evaluate("""
(function(){ return !!document.body.textContent.match(/Welcome to Example App/i) })()
""")
assert has_welcome, "Welcome message not found"

# Check specific element content
h1_text = await evaluate("document.querySelector('h1')?.textContent?.trim()")
assert h1_text == "Example App", f'Expected "Example App", got "{h1_text}"'

# Check no error messages
error_count = await evaluate("document.querySelectorAll('.error-message').length")
assert error_count == 0, f"Found {error_count} error messages on page"

print("All content assertions passed")
EOF
```

### 2.3. Step 3 -- Test user interactions (login flow)

For stable role or name locators, resolve and act in one step:

```bash
openbrowser-ai -c - <<'EOF'
await fill_element("test@example.com", role="textbox", name="Email")
await fill_element("test-password", role="textbox", name="Password")
await click_element(role="button", name="Sign in")
await wait_for_load()
logged_in = await evaluate("!!document.querySelector('[data-testid=dashboard], main.dashboard, #dashboard')")
assert logged_in, "Login did not reach the dashboard"
EOF
```

Use `find_element(...)` when a later indexed action needs the resolved index. It reads a screenshotless browser state and can wait for an element after navigation.

You can also inspect and use indices directly:

```bash
openbrowser-ai -c - <<'EOF'
# Get form fields
state = await browser.get_browser_state_summary()
for idx, el in state.dom_state.selector_map.items():
    if el.attributes.get("type") in ("email", "text", "password") or el.tag_name == "button":
        etype = el.attributes.get("type", "")
        placeholder = el.attributes.get("placeholder", "")
        print(f"[{idx}] <{el.tag_name}> type={etype} placeholder=\"{placeholder}\"")

# Fill and submit
await input_text(index=3, text="test@example.com")
await input_text(index=4, text="test-password")
await click(index=5)  # Login button
await wait(2)

# Assert logged in
state = await browser.get_browser_state_summary()
assert "dashboard" in state.url.lower() or "welcome" in state.title.lower(), \
    f"Login may have failed. URL: {state.url}, Title: {state.title}"
print("Login test passed")
EOF
```

### 2.4. Step 4 -- Test navigation flows

```bash
openbrowser-ai -c - <<'EOF'
# Click settings link
state = await browser.get_browser_state_summary()
for idx, el in state.dom_state.selector_map.items():
    if "settings" in el.get_all_children_text(max_depth=1).lower():
        await click(index=idx)
        break

await wait(1)

# Assert URL changed
path = await evaluate("window.location.pathname")
assert "/settings" in path, f"Expected /settings path, got {path}"

# Test back button
await go_back()
await wait(1)
state = await browser.get_browser_state_summary()
print(f"After back: {state.url}")
EOF
```

### 2.5. Step 5 -- Test error handling

```bash
openbrowser-ai -c - <<'EOF'
# Submit form with invalid data
await input_text(index=3, text="not-an-email")
await click(index=5)
await wait(1)

# Assert validation errors appear
errors = await evaluate("""
(function(){
  const errs = document.querySelectorAll(".error, .invalid-feedback, [aria-invalid=\"true\"]");
  return Array.from(errs).map(e => e.textContent.trim());
})()
""")
assert len(errors) > 0, "Expected validation errors but found none"
print(f"Validation errors shown: {errors}")

# Assert page did not navigate
path = await evaluate("window.location.pathname")
print(f"Still on: {path}")
EOF
```

### 2.6. Step 6 -- Test responsive behavior

```bash
openbrowser-ai -c - <<'EOF'
viewport = await evaluate("""
(function(){
  return {
    width: window.innerWidth,
    height: window.innerHeight
  }
})()
""")
vw = viewport["width"]
vh = viewport["height"]
print(f"Viewport: {vw}x{vh}")

# Check mobile menu visibility
mobile_display = await evaluate("""
(function(){
  const el = document.querySelector(".mobile-menu");
  return el ? window.getComputedStyle(el).display : "not found";
})()
""")
print(f"Mobile menu display: {mobile_display}")
EOF
```

### 2.7. Step 7 -- Test multi-page flows

```bash
openbrowser-ai -c - <<'EOF'
test_results = []

# Cart page
await navigate("https://staging.example.com/cart")
await wait(1)
cart_title = await evaluate("document.querySelector('h1')?.textContent?.trim()")
test_results.append({"test": "cart_page_loads", "passed": cart_title is not None, "detail": cart_title})

# Checkout
cart_count = await evaluate("JSON.parse(localStorage.getItem('cart'))?.items?.length || 0")
test_results.append({"test": "cart_has_items", "passed": cart_count > 0, "detail": f"{cart_count} items"})

# Print results
import json
passed = sum(1 for t in test_results if t["passed"])
total = len(test_results)
print(f"\nResults: {passed}/{total} passed")
print(json.dumps(test_results, indent=2))
EOF
```

## 3. Tips

- Code is piped via stdin using heredoc (`-c - <<'EOF'`), so all Python syntax works without shell escaping issues.
- Use Python `assert` statements for test assertions -- they produce clear error messages on failure.
- Always verify page state after each interaction before proceeding.
- Test both happy paths and error paths for thorough coverage.
- Use `await wait(N)` after interactions that trigger page loads or animations.
- Variables persist between `-c` calls while the daemon is running, so you can accumulate test results across calls.
- Use `evaluate()` for DOM state assertions and `browser.get_browser_state_summary()` for page metadata.
- Use `send_keys()` for keyboard-only paths. Use `get_tabs()` to inspect open tabs, then pass a `tab_id` to `switch()` or `close()`.

## 4. Cleanup

Run this step after the task, whether it succeeded or failed.

**Do not stop a daemon you did not start.** Check `openbrowser-ai daemon status` before the first browser call. If it already reports a running daemon, reuse it and leave it running at the end. If it reports `daemon_not_running`, this run starts one and must clean it up.

For a daemon started by this run:

```bash
openbrowser-ai daemon stop
```

`daemon stop` only acknowledges the request and returns. Poll `openbrowser-ai daemon status` until it reports `daemon_not_running`, up to about 30 seconds. If the daemon does not exit, run `openbrowser-ai daemon force-stop`, which waits and verifies listener, owner, and browser cleanup.

If the task can fail mid-workflow, guarantee cleanup with a shell trap that stops, polls, and force-stops if needed:

```bash
cleanup() {
  openbrowser-ai daemon stop >/dev/null 2>&1 || true
  i=0
  while [ "$i" -lt 30 ]; do
    status=$(openbrowser-ai daemon status 2>&1 || true)
    case "$status" in
      *daemon_not_running*) return 0 ;;
    esac
    sleep 1
    i=$((i + 1))
  done
  openbrowser-ai daemon force-stop >/dev/null 2>&1 || true
}
trap cleanup EXIT
# ... openbrowser-ai -c calls here ...
```

Persist results to disk before stopping the daemon. `daemon reset` clears the Python namespace. Call `done()` only to mark the task complete inside the agent loop. Persist results and stop the daemon for cleanup.
