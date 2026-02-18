---
name: accessibility-audit
description: |
  Audit web pages for accessibility issues, WCAG compliance, and screen reader compatibility.
  Trigger when the user asks to: check accessibility, run an a11y audit, test WCAG compliance,
  check screen reader support, audit ARIA attributes, verify keyboard navigation,
  find accessibility issues, or check for missing alt text or labels.
---

# Accessibility Audit

Audit web pages for accessibility issues following WCAG 2.1 guidelines. Checks heading structure, form labels, image alt text, ARIA attributes, color contrast indicators, and keyboard navigation.

## Workflow

### Step 1 -- Navigate to the page

Open the page to audit:

```
browser_navigate(url="https://example.com")
```

Confirm the page loaded:

```
browser_get_state(compact=true)
```

### Step 2 -- Get the accessibility tree

The accessibility tree is the primary data source for an a11y audit. It shows how assistive technologies interpret the page.

```
browser_get_accessibility_tree()
```

Review the tree for:
- Missing roles on interactive elements
- Empty or missing names on buttons and links
- Incorrect heading hierarchy
- Missing landmark regions (banner, navigation, main, contentinfo)

### Step 3 -- Check heading structure

Verify headings follow a logical hierarchy (h1 -> h2 -> h3, no skipped levels):

```
browser_execute_js(expression="(() => { const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6')); const issues = []; let prevLevel = 0; const h1Count = headings.filter(h => h.tagName === 'H1').length; if (h1Count === 0) issues.push('No h1 element found'); if (h1Count > 1) issues.push('Multiple h1 elements found: ' + h1Count); headings.forEach(h => { const level = parseInt(h.tagName[1]); if (prevLevel > 0 && level > prevLevel + 1) { issues.push('Skipped heading level: h' + prevLevel + ' -> h' + level + ' (\"' + h.textContent.trim().substring(0, 50) + '\")'); } if (!h.textContent.trim()) { issues.push('Empty heading: ' + h.tagName); } prevLevel = level; }); return { total: headings.length, h1Count, hierarchy: headings.map(h => ({ tag: h.tagName, text: h.textContent.trim().substring(0, 80) })), issues }; })()")
```

### Step 4 -- Check images for alt text

Find images missing alt attributes or with empty alt text that should have descriptions:

```
browser_execute_js(expression="(() => { const images = Array.from(document.querySelectorAll('img')); const issues = []; const results = { total: images.length, withAlt: 0, withEmptyAlt: 0, missingAlt: 0, details: [] }; images.forEach(img => { const alt = img.getAttribute('alt'); const src = img.src?.substring(0, 100); if (alt === null) { results.missingAlt++; issues.push('Missing alt attribute: ' + src); results.details.push({ src, alt: null, issue: 'missing alt' }); } else if (alt === '') { results.withEmptyAlt++; results.details.push({ src, alt: '', issue: 'empty alt (decorative)' }); } else { results.withAlt++; } }); return { ...results, issues }; })()")
```

### Step 5 -- Check form labels

Verify all form inputs have associated labels:

```
browser_execute_js(expression="(() => { const inputs = Array.from(document.querySelectorAll('input:not([type=\"hidden\"]),select,textarea')); const issues = []; inputs.forEach(input => { const id = input.id; const ariaLabel = input.getAttribute('aria-label'); const ariaLabelledBy = input.getAttribute('aria-labelledby'); const title = input.getAttribute('title'); const placeholder = input.getAttribute('placeholder'); const label = id ? document.querySelector('label[for=\"' + id + '\"]') : null; const parentLabel = input.closest('label'); const hasLabel = label || parentLabel || ariaLabel || ariaLabelledBy || title; if (!hasLabel) { issues.push({ tag: input.tagName, type: input.type || 'text', id: id || '(none)', name: input.name || '(none)', placeholder: placeholder || '(none)', issue: 'No label, aria-label, aria-labelledby, or title' }); } }); return { totalInputs: inputs.length, unlabeled: issues.length, issues }; })()")
```

### Step 6 -- Check ARIA attributes

Verify ARIA attributes are used correctly:

```
browser_execute_js(expression="(() => { const issues = []; const ariaElements = document.querySelectorAll('[role],[aria-label],[aria-labelledby],[aria-describedby],[aria-hidden]'); const results = { totalAriaElements: ariaElements.length, issues: [] }; ariaElements.forEach(el => { const role = el.getAttribute('role'); const ariaLabel = el.getAttribute('aria-label'); const ariaLabelledBy = el.getAttribute('aria-labelledby'); const ariaDescribedBy = el.getAttribute('aria-describedby'); if (ariaLabelledBy) { const ids = ariaLabelledBy.split(/\\s+/); ids.forEach(id => { if (!document.getElementById(id)) { results.issues.push({ element: el.tagName, issue: 'aria-labelledby references missing id: ' + id }); } }); } if (ariaDescribedBy) { const ids = ariaDescribedBy.split(/\\s+/); ids.forEach(id => { if (!document.getElementById(id)) { results.issues.push({ element: el.tagName, issue: 'aria-describedby references missing id: ' + id }); } }); } if (role === 'button' && !ariaLabel && !el.textContent.trim()) { results.issues.push({ element: el.tagName, issue: 'Button role with no accessible name' }); } if (el.getAttribute('aria-hidden') === 'true' && el.querySelector('a,button,input,select,textarea,[tabindex]')) { results.issues.push({ element: el.tagName, issue: 'aria-hidden=true on element containing focusable children' }); } }); return results; })()")
```

### Step 7 -- Check landmark regions

Verify the page has proper landmark structure:

```
browser_execute_js(expression="(() => { const landmarks = { banner: document.querySelectorAll('header,[role=\"banner\"]').length, navigation: document.querySelectorAll('nav,[role=\"navigation\"]').length, main: document.querySelectorAll('main,[role=\"main\"]').length, contentinfo: document.querySelectorAll('footer,[role=\"contentinfo\"]').length, complementary: document.querySelectorAll('aside,[role=\"complementary\"]').length, search: document.querySelectorAll('[role=\"search\"]').length }; const issues = []; if (landmarks.main === 0) issues.push('No main landmark found'); if (landmarks.main > 1) issues.push('Multiple main landmarks found: ' + landmarks.main); if (landmarks.banner === 0) issues.push('No banner/header landmark found'); if (landmarks.navigation === 0) issues.push('No navigation landmark found'); if (landmarks.contentinfo === 0) issues.push('No contentinfo/footer landmark found'); return { landmarks, issues }; })()")
```

### Step 8 -- Check link and button accessibility

Verify links and buttons have descriptive text:

```
browser_execute_js(expression="(() => { const issues = []; const links = Array.from(document.querySelectorAll('a')); links.forEach(a => { const text = a.textContent.trim(); const ariaLabel = a.getAttribute('aria-label'); const title = a.getAttribute('title'); const img = a.querySelector('img[alt]'); const name = text || ariaLabel || title || img?.alt; if (!name) { issues.push({ tag: 'a', href: a.href?.substring(0, 80), issue: 'Link with no accessible name' }); } else if (['click here', 'here', 'read more', 'more', 'link'].includes(name.toLowerCase())) { issues.push({ tag: 'a', text: name, issue: 'Non-descriptive link text' }); } }); const buttons = Array.from(document.querySelectorAll('button,[role=\"button\"]')); buttons.forEach(btn => { const text = btn.textContent.trim(); const ariaLabel = btn.getAttribute('aria-label'); const title = btn.getAttribute('title'); if (!text && !ariaLabel && !title) { issues.push({ tag: btn.tagName, issue: 'Button with no accessible name' }); } }); return { totalLinks: links.length, totalButtons: buttons.length, issues }; })()")
```

### Step 9 -- Check tab order and keyboard access

Verify interactive elements have a logical tab order:

```
browser_execute_js(expression="(() => { const focusable = Array.from(document.querySelectorAll('a[href],button,input:not([type=\"hidden\"]),select,textarea,[tabindex]')); const issues = []; const positiveTabindex = focusable.filter(el => { const ti = parseInt(el.getAttribute('tabindex')); return ti > 0; }); if (positiveTabindex.length > 0) { issues.push('Elements with positive tabindex (disrupts natural tab order): ' + positiveTabindex.length); positiveTabindex.forEach(el => { issues.push('  tabindex=' + el.getAttribute('tabindex') + ' on ' + el.tagName + (el.id ? '#' + el.id : '')); }); } const negativeTabindex = focusable.filter(el => { const ti = parseInt(el.getAttribute('tabindex')); return ti < 0 && ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName); }); if (negativeTabindex.length > 0) { issues.push('Interactive elements removed from tab order (tabindex=-1): ' + negativeTabindex.length); } return { totalFocusable: focusable.length, positiveTabindex: positiveTabindex.length, negativeTabindex: negativeTabindex.length, issues }; })()")
```

### Step 10 -- Compile the audit report

Use `browser_get_text` to capture the full page content for reference:

```
browser_get_text()
```

Compile all findings from Steps 2-9 into a structured report with:
- Summary of issues by severity (critical, serious, moderate, minor)
- Specific elements and their issues
- WCAG success criteria references where applicable
- Recommended fixes for each issue

### Step 11 -- Clean up

Close the browser session:

```
browser_session(action="close_all")
```

## WCAG Quick Reference

| Check | WCAG Criterion | Level |
|-------|---------------|-------|
| Images have alt text | 1.1.1 Non-text Content | A |
| Heading hierarchy is logical | 1.3.1 Info and Relationships | A |
| Form inputs have labels | 1.3.1 Info and Relationships | A |
| Link purpose is clear | 2.4.4 Link Purpose (In Context) | A |
| Page has landmark regions | 1.3.1 Info and Relationships | A |
| Focus order is logical | 2.4.3 Focus Order | A |
| ARIA attributes are valid | 4.1.2 Name, Role, Value | A |
| Page has a single h1 | 1.3.1 Info and Relationships | A |
| Interactive elements are keyboard accessible | 2.1.1 Keyboard | A |

## Tips

- Start with `browser_get_accessibility_tree` for a quick overview of how assistive technologies see the page.
- The heading check (Step 3) catches the most common structural issues.
- Missing form labels (Step 5) are the most common cause of form accessibility failures.
- Run audits on multiple pages of a site, not just the homepage.
- ARIA misuse is often worse than no ARIA at all; verify that ARIA attributes reference valid IDs and use correct roles.
