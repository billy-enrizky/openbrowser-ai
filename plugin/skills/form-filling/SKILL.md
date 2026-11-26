---
name: form-filling
description: |
  Fill out web forms, submit data, and handle login or registration flows.
  Trigger when the user asks to: fill a form, submit data on a website, log in to a site,
  register an account, complete a checkout, enter information into fields, or automate form submission.
allowed-tools: Bash(openbrowser-ai:*) Bash(curl:*) Bash(uv:*) Bash(irm:*) Read Write
---

# Form Filling

<!-- toc:start -->
## Contents

- [1. Setup](#1-setup)
- [2. Workflow](#2-workflow)
  - [2.1. Step 1 -- Navigate to the form page](#21-step-1----navigate-to-the-form-page)
  - [2.2. Step 2 -- Discover form fields](#22-step-2----discover-form-fields)
  - [2.3. Step 3 -- Fill text inputs](#23-step-3----fill-text-inputs)
  - [2.4. Step 4 -- Handle dropdowns](#24-step-4----handle-dropdowns)
  - [2.5. Step 5 -- Handle checkboxes and radio buttons](#25-step-5----handle-checkboxes-and-radio-buttons)
  - [2.6. Step 6 -- Submit the form](#26-step-6----submit-the-form)
  - [2.7. Step 7 -- Verify submission result](#27-step-7----verify-submission-result)
  - [2.8. Step 8 -- Handle multi-step forms](#28-step-8----handle-multi-step-forms)
- [3. Tips](#3-tips)
- [4. Cleanup](#4-cleanup)

<!-- toc:end -->

Automate filling web forms including login, registration, checkout, and multi-step form wizards using Python code execution.

Form filling needs a browser because it types into fields and submits pages. OpenBrowser no-browser mode starts no Chrome, Blink, or V8 process, but it cannot run JavaScript, take screenshots, or use browser cookies, so it cannot replace the workflow below.

All code runs via `openbrowser-ai -c`. The daemon starts automatically and persists variables across calls. Run `openbrowser-ai warm` once before a multi-call workflow to start the daemon and browser before the workflow begins. Most browser functions are async, use `await`. `list_downloads()`, `download_outcomes()`, and `download_failures()` are synchronous.

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

### 2.1. Step 1 -- Navigate to the form page

```bash
openbrowser-ai -c - <<'EOF'
await navigate("https://example.com/login")
state = await browser.get_browser_state_summary()
print(f"Page: {state.title} ({state.url})")
print(f"Interactive elements: {len(state.dom_state.selector_map)}")
EOF
```

### 2.2. Step 2 -- Discover form fields

```bash
openbrowser-ai -c - <<'EOF'
# List all interactive elements with their indices
state = await browser.get_browser_state_summary()
for index, element in state.dom_state.selector_map.items():
    tag = element.tag_name
    text = element.get_all_children_text(max_depth=2)[:60]
    placeholder = element.attributes.get("placeholder", "")
    input_type = element.attributes.get("type", "")
    name = element.attributes.get("name", "")
    print(f"[{index}] <{tag}> type={input_type} name={name} placeholder=\"{placeholder}\" text=\"{text}\"")
EOF
```

### 2.3. Step 3 -- Fill text inputs

If the field has a stable accessibility name or id, avoid hard-coded indices:

```bash
openbrowser-ai -c - <<'EOF'
await fill_element("user@example.com", role="textbox", name="Email")
await fill_element("secure-password", role="textbox", name="Password")
EOF
```

`find_element(role=..., name=..., element_id=..., tag=..., timeout=...)` returns the index when a later indexed action needs it. The helper reads a screenshotless browser state and waits for the element when a timeout is set.

You can also use indices from Step 2:

```bash
openbrowser-ai -c - <<'EOF'
await input_text(index=5, text="user@example.com")
await input_text(index=7, text="secure-password")
EOF
```

For fields that need clearing first:

```bash
openbrowser-ai -c - <<'EOF'
await click(index=5)
await evaluate("document.activeElement.select()")
await input_text(index=5, text="new-value")
EOF
```

### 2.4. Step 4 -- Handle dropdowns

Standard HTML select elements:

```bash
openbrowser-ai -c - <<'EOF'
await select_dropdown(index=12, text="United States")
EOF
```

To see available options first:

```bash
openbrowser-ai -c - <<'EOF'
options = await dropdown_options(index=12)
print(options)
EOF
```

Custom dropdown components:

```bash
openbrowser-ai -c - <<'EOF'
await evaluate("""
(function(){
  const select = document.querySelector("select#country");
  select.value = "US";
  select.dispatchEvent(new Event("change", { bubbles: true }));
})()
""")
EOF
```

### 2.5. Step 5 -- Handle checkboxes and radio buttons

```bash
openbrowser-ai -c - <<'EOF'
await click(index=15)  # Click checkbox/radio

# Verify state
checked = await evaluate("""document.querySelector("input[name=agree]").checked""")
print(f"Checkbox checked: {checked}")
EOF
```

### 2.6. Step 6 -- Submit the form

```bash
openbrowser-ai -c - <<'EOF'
await click(index=20)  # Click submit button
await wait(2)

# Verify submission
state = await browser.get_browser_state_summary()
print(f"After submit: {state.url}")
EOF
```

Or submit via JavaScript:

```bash
openbrowser-ai -c - <<'EOF'
await evaluate("document.querySelector(\"form\").submit()")
EOF
```

### 2.7. Step 7 -- Verify submission result

```bash
openbrowser-ai -c - <<'EOF'
# Check for success/error messages
result = await evaluate("""
(function(){
  const success = document.querySelector(".success, .alert-success, [role=\"alert\"]");
  const error = document.querySelector(".error, .alert-danger, .validation-error");
  return {
    success: success?.textContent?.trim(),
    error: error?.textContent?.trim(),
    url: window.location.href
  };
})()
""")
print(result)
EOF
```

### 2.8. Step 8 -- Handle multi-step forms

```bash
openbrowser-ai -c - <<'EOF'
for step in range(1, 5):
    # Discover fields for current step
    state = await browser.get_browser_state_summary()
    print(f"Step {step}: {len(state.dom_state.selector_map)} elements")

    # Fill fields (indices vary per step)
    # ... fill fields here ...

    # Click Next/Continue
    # Find the next button
    for idx, el in state.dom_state.selector_map.items():
        text = el.get_all_children_text(max_depth=1).lower()
        if "next" in text or "continue" in text:
            await click(index=idx)
            await wait(2)
            break
EOF
```

## 3. Tips

- Code is piped via stdin using heredoc (`-c - <<'EOF'`), so all Python syntax works without shell escaping issues.
- Always discover fields with `browser.get_browser_state_summary()` before typing. Do not guess element indices.
- Use `click_element(role=..., name=...)` for a button with a stable accessible name. Use `send_keys()` to test keyboard-only form flows.
- For sensitive data (passwords, tokens), confirm with the user before entering values.
- Use `evaluate()` to bypass custom components that do not respond to standard click/type.
- Variables persist between `-c` calls while the daemon is running, so you can store field indices in one call and use them in the next.
- Check for CAPTCHA or bot detection; notify the user if manual intervention is needed.

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
