---
name: accessibility-audit
description: |
  Audit web pages for accessibility issues, WCAG compliance, and screen reader compatibility.
  Trigger when the user asks to: check accessibility, run an a11y audit, test WCAG compliance,
  check screen reader support, audit ARIA attributes, verify keyboard navigation,
  find accessibility issues, or check for missing alt text or labels.
allowed-tools: Bash(openbrowser-ai:*) Bash(curl:*) Bash(uv:*) Bash(irm:*) Read Write
---

# Accessibility Audit

<!-- toc:start -->
## Contents

- [1. Setup](#1-setup)
- [2. Workflow](#2-workflow)
  - [2.1. Step 1 -- Navigate and initialize audit](#21-step-1----navigate-and-initialize-audit)
  - [2.2. Step 2 -- Check heading structure](#22-step-2----check-heading-structure)
  - [2.3. Step 3 -- Check images for alt text](#23-step-3----check-images-for-alt-text)
  - [2.4. Step 4 -- Check form labels](#24-step-4----check-form-labels)
  - [2.5. Step 5 -- Check ARIA attributes](#25-step-5----check-aria-attributes)
  - [2.6. Step 6 -- Check landmark regions](#26-step-6----check-landmark-regions)
  - [2.7. Step 7 -- Check links and buttons](#27-step-7----check-links-and-buttons)
  - [2.8. Step 8 -- Check keyboard navigation](#28-step-8----check-keyboard-navigation)
  - [2.9. Step 9 -- Compile audit report](#29-step-9----compile-audit-report)
- [3. WCAG Quick Reference](#3-wcag-quick-reference)
- [4. Tips](#4-tips)
- [5. Cleanup](#5-cleanup)

<!-- toc:end -->

Audit web pages for accessibility issues following WCAG 2.1 guidelines using Python code execution. Checks heading structure, form labels, image alt text, ARIA attributes, landmark regions, and keyboard navigation.

This audit reads the rendered DOM and runs JavaScript, so it needs the browser workflow. OpenBrowser no-browser mode (`openbrowser-ai scrape <url> --no-browser`) starts no Chrome, Blink, or V8 process, but it cannot run JavaScript, take screenshots, or use browser cookies, so it cannot perform these checks.

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

### 2.1. Step 1 -- Navigate and initialize audit

```bash
openbrowser-ai -c - <<'EOF'
await navigate("https://example.com")
state = await browser.get_browser_state_summary()
print(f"Auditing: {state.title} ({state.url})")

# Store all findings
audit = {
    "url": state.url,
    "title": state.title,
    "issues": [],
    "checks": {}
}
EOF
```

### 2.2. Step 2 -- Check heading structure

```bash
openbrowser-ai -c - <<'EOF'
headings_result = await evaluate("""
(function(){
  const headings = Array.from(document.querySelectorAll("h1,h2,h3,h4,h5,h6"));
  const issues = [];
  let prevLevel = 0;
  const h1Count = headings.filter(h => h.tagName === "H1").length;

  if (h1Count === 0) issues.push("No h1 element found");
  if (h1Count > 1) issues.push("Multiple h1 elements: " + h1Count);

  headings.forEach(h => {
    const level = parseInt(h.tagName[1]);
    if (prevLevel > 0 && level > prevLevel + 1)
      issues.push("Skipped level: h" + prevLevel + " -> h" + level + " (\"" + h.textContent.trim().substring(0, 50) + "\")");
    if (!h.textContent.trim())
      issues.push("Empty heading: " + h.tagName);
    prevLevel = level;
  });

  return {
    total: headings.length,
    h1Count,
    hierarchy: headings.map(h => ({ tag: h.tagName, text: h.textContent.trim().substring(0, 80) })),
    issues
  };
})()
""")

audit["checks"]["headings"] = headings_result
for issue in headings_result.get("issues", []):
    audit["issues"].append({"check": "headings", "wcag": "1.3.1", "issue": issue})
    print(f"[HEADINGS] {issue}")

if not headings_result.get("issues"):
    print("[HEADINGS] PASS")
EOF
```

### 2.3. Step 3 -- Check images for alt text

```bash
openbrowser-ai -c - <<'EOF'
images_result = await evaluate("""
(function(){
  const images = Array.from(document.querySelectorAll("img"));
  const issues = [];
  let withAlt = 0, withEmptyAlt = 0, missingAlt = 0;

  images.forEach(img => {
    const alt = img.getAttribute("alt");
    const src = img.src?.substring(0, 100);
    if (alt === null) {
      missingAlt++;
      issues.push("Missing alt: " + src);
    } else if (alt === "") {
      withEmptyAlt++;
    } else {
      withAlt++;
    }
  });

  return { total: images.length, withAlt, withEmptyAlt, missingAlt, issues };
})()
""")

audit["checks"]["images"] = images_result
for issue in images_result.get("issues", []):
    audit["issues"].append({"check": "images", "wcag": "1.1.1", "issue": issue})
    print(f"[IMAGES] {issue}")

if not images_result.get("issues"):
    total = images_result["total"]
    with_alt = images_result["withAlt"]
    print(f"[IMAGES] PASS ({total} images, {with_alt} with alt)")
EOF
```

### 2.4. Step 4 -- Check form labels

```bash
openbrowser-ai -c - <<'EOF'
forms_result = await evaluate("""
(function(){
  const inputs = Array.from(document.querySelectorAll("input:not([type=\"hidden\"]),select,textarea"));
  const issues = [];

  inputs.forEach(input => {
    const id = input.id;
    const ariaLabel = input.getAttribute("aria-label");
    const ariaLabelledBy = input.getAttribute("aria-labelledby");
    const title = input.getAttribute("title");
    const label = id ? document.querySelector("label[for=\"" + id + "\"]") : null;
    const parentLabel = input.closest("label");
    const hasLabel = label || parentLabel || ariaLabel || ariaLabelledBy || title;

    if (!hasLabel) {
      issues.push({
        tag: input.tagName,
        type: input.type || "text",
        name: input.name || "(none)",
        placeholder: input.getAttribute("placeholder") || "(none)",
        issue: "No label or aria-label"
      });
    }
  });

  return { totalInputs: inputs.length, unlabeled: issues.length, issues };
})()
""")

audit["checks"]["forms"] = forms_result
for issue in forms_result.get("issues", []):
    tag = issue["tag"]
    name = issue["name"]
    itype = issue["type"]
    audit["issues"].append({"check": "forms", "wcag": "1.3.1", "issue": f"Unlabeled {tag} name={name}"})
    print(f"[FORMS] Unlabeled: <{tag}> type={itype} name={name}")

if not forms_result.get("issues"):
    total_inputs = forms_result["totalInputs"]
    print(f"[FORMS] PASS ({total_inputs} inputs, all labeled)")
EOF
```

### 2.5. Step 5 -- Check ARIA attributes

```bash
openbrowser-ai -c - <<'EOF'
aria_result = await evaluate("""
(function(){
  const issues = [];
  const ariaElements = document.querySelectorAll("[role],[aria-label],[aria-labelledby],[aria-describedby],[aria-hidden]");

  ariaElements.forEach(el => {
    const ariaLabelledBy = el.getAttribute("aria-labelledby");
    const ariaDescribedBy = el.getAttribute("aria-describedby");

    if (ariaLabelledBy) {
      ariaLabelledBy.split(/\s+/).forEach(id => {
        if (!document.getElementById(id))
          issues.push({ element: el.tagName, issue: "aria-labelledby references missing id: " + id });
      });
    }
    if (ariaDescribedBy) {
      ariaDescribedBy.split(/\s+/).forEach(id => {
        if (!document.getElementById(id))
          issues.push({ element: el.tagName, issue: "aria-describedby references missing id: " + id });
      });
    }
    if (el.getAttribute("role") === "button" && !el.getAttribute("aria-label") && !el.textContent.trim())
      issues.push({ element: el.tagName, issue: "Button role with no accessible name" });
    if (el.getAttribute("aria-hidden") === "true" && el.querySelector("a,button,input,select,textarea,[tabindex]"))
      issues.push({ element: el.tagName, issue: "aria-hidden on element with focusable children" });
  });

  return { totalAriaElements: ariaElements.length, issues };
})()
""")

audit["checks"]["aria"] = aria_result
for issue in aria_result.get("issues", []):
    msg = issue["issue"]
    audit["issues"].append({"check": "aria", "wcag": "4.1.2", "issue": msg})
    print(f"[ARIA] {msg}")

if not aria_result.get("issues"):
    total_aria = aria_result["totalAriaElements"]
    print(f"[ARIA] PASS ({total_aria} ARIA elements)")
EOF
```

### 2.6. Step 6 -- Check landmark regions

```bash
openbrowser-ai -c - <<'EOF'
landmarks_result = await evaluate("""
(function(){
  const landmarks = {
    banner: document.querySelectorAll("header,[role=\"banner\"]").length,
    navigation: document.querySelectorAll("nav,[role=\"navigation\"]").length,
    main: document.querySelectorAll("main,[role=\"main\"]").length,
    contentinfo: document.querySelectorAll("footer,[role=\"contentinfo\"]").length,
    complementary: document.querySelectorAll("aside,[role=\"complementary\"]").length,
    search: document.querySelectorAll("[role=\"search\"]").length
  };
  const issues = [];
  if (landmarks.main === 0) issues.push("No main landmark");
  if (landmarks.main > 1) issues.push("Multiple main landmarks: " + landmarks.main);
  if (landmarks.banner === 0) issues.push("No banner/header landmark");
  if (landmarks.navigation === 0) issues.push("No navigation landmark");
  if (landmarks.contentinfo === 0) issues.push("No footer/contentinfo landmark");
  return { landmarks, issues };
})()
""")

audit["checks"]["landmarks"] = landmarks_result
for issue in landmarks_result.get("issues", []):
    audit["issues"].append({"check": "landmarks", "wcag": "1.3.1", "issue": issue})
    print(f"[LANDMARKS] {issue}")

if not landmarks_result.get("issues"):
    print("[LANDMARKS] PASS")
EOF
```

### 2.7. Step 7 -- Check links and buttons

```bash
openbrowser-ai -c - <<'EOF'
links_result = await evaluate("""
(function(){
  const issues = [];
  document.querySelectorAll("a").forEach(a => {
    const name = a.textContent.trim() || a.getAttribute("aria-label") || a.getAttribute("title") || a.querySelector("img[alt]")?.alt;
    if (!name) issues.push({ tag: "a", href: a.href?.substring(0, 80), issue: "No accessible name" });
    else if (["click here", "here", "read more", "more", "link"].includes(name.toLowerCase()))
      issues.push({ tag: "a", text: name, issue: "Non-descriptive link text" });
  });
  document.querySelectorAll("button,[role=\"button\"]").forEach(btn => {
    const name = btn.textContent.trim() || btn.getAttribute("aria-label") || btn.getAttribute("title");
    if (!name) issues.push({ tag: btn.tagName, issue: "Button with no accessible name" });
  });
  return { issues };
})()
""")

audit["checks"]["links_buttons"] = links_result
for issue in links_result.get("issues", []):
    msg = issue["issue"]
    audit["issues"].append({"check": "links_buttons", "wcag": "2.4.4", "issue": msg})
    print(f"[LINKS/BUTTONS] {msg}")

if not links_result.get("issues"):
    print("[LINKS/BUTTONS] PASS")
EOF
```

### 2.8. Step 8 -- Check keyboard navigation

```bash
openbrowser-ai -c - <<'EOF'
keyboard_result = await evaluate("""
(function(){
  const focusable = Array.from(document.querySelectorAll("a[href],button,input:not([type=\"hidden\"]),select,textarea,[tabindex]"));
  const issues = [];
  const positiveTabindex = focusable.filter(el => parseInt(el.getAttribute("tabindex")) > 0);
  const negativeTabindex = focusable.filter(el => {
    const ti = parseInt(el.getAttribute("tabindex"));
    return ti < 0 && ["A","BUTTON","INPUT","SELECT","TEXTAREA"].includes(el.tagName);
  });

  if (positiveTabindex.length > 0)
    issues.push("Elements with positive tabindex (disrupts tab order): " + positiveTabindex.length);
  if (negativeTabindex.length > 0)
    issues.push("Interactive elements removed from tab order: " + negativeTabindex.length);

  return { totalFocusable: focusable.length, positiveTabindex: positiveTabindex.length, negativeTabindex: negativeTabindex.length, issues };
})()
""")

audit["checks"]["keyboard"] = keyboard_result
for issue in keyboard_result.get("issues", []):
    audit["issues"].append({"check": "keyboard", "wcag": "2.4.3", "issue": issue})
    print(f"[KEYBOARD] {issue}")

if not keyboard_result.get("issues"):
    total_focusable = keyboard_result["totalFocusable"]
    print(f"[KEYBOARD] PASS ({total_focusable} focusable elements)")
EOF
```

### 2.9. Step 9 -- Compile audit report

```bash
openbrowser-ai -c - <<'EOF'
import json

total_issues = len(audit["issues"])
checks_passed = sum(1 for c in audit["checks"].values() if not c.get("issues"))
checks_total = len(audit["checks"])

url = audit["url"]
title = audit["title"]
print(f"\n=== Accessibility Audit Report ===")
print(f"URL: {url}")
print(f"Title: {title}")
print(f"Checks: {checks_passed}/{checks_total} passed")
print(f"Total issues: {total_issues}")

if total_issues > 0:
    print(f"\nIssues by WCAG criterion:")
    by_wcag = {}
    for issue in audit["issues"]:
        by_wcag.setdefault(issue["wcag"], []).append(issue)
    for wcag, issues in sorted(by_wcag.items()):
        print(f"  {wcag}: {len(issues)} issues")
        for i in issues[:3]:
            msg = i["issue"]
            print(f"    - {msg}")
        if len(issues) > 3:
            print(f"    ... and {len(issues) - 3} more")
EOF
```

## 3. WCAG Quick Reference

| Check | WCAG Criterion | Level |
|-------|---------------|-------|
| Images have alt text | 1.1.1 Non-text Content | A |
| Heading hierarchy is logical | 1.3.1 Info and Relationships | A |
| Form inputs have labels | 1.3.1 Info and Relationships | A |
| Link purpose is clear | 2.4.4 Link Purpose (In Context) | A |
| Landmark regions present | 1.3.1 Info and Relationships | A |
| Focus order is logical | 2.4.3 Focus Order | A |
| ARIA attributes valid | 4.1.2 Name, Role, Value | A |

## 4. Tips

- Code is piped via stdin using heredoc (`-c - <<'EOF'`), so all Python syntax works without shell escaping issues.
- Store results in the `audit` dict -- variables persist between `-c` calls while the daemon is running.
- Run audits on multiple pages, not just the homepage.
- ARIA misuse is often worse than no ARIA at all.
- The heading check catches the most common structural issues.
- Missing form labels are the most common form accessibility failure.

## 5. Cleanup

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

Persist results to disk before stopping the daemon. `daemon reset` clears the Python namespace. Do not call `done()` as a substitute for cleanup; it only marks the task complete inside the agent loop.
