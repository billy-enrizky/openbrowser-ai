---
name: accessibility-audit
description: |
  Audit web pages for accessibility issues, WCAG compliance, and screen reader compatibility.
  Trigger when the user asks to: check accessibility, run an a11y audit, test WCAG compliance,
  check screen reader support, audit ARIA attributes, verify keyboard navigation,
  find accessibility issues, or check for missing alt text or labels.
---

# Accessibility Audit

Audit web pages for accessibility issues following WCAG 2.1 guidelines using Python code execution. Checks heading structure, form labels, image alt text, ARIA attributes, landmark regions, and keyboard navigation.

All code runs in a persistent namespace via `execute_code`. All browser functions are async -- use `await`.

## Workflow

### Step 1 -- Navigate and initialize audit

```python
await navigate('https://example.com')
state = await browser.get_browser_state_summary()
print(f'Auditing: {state.title} ({state.url})')

# Store all findings
audit = {
    'url': state.url,
    'title': state.title,
    'issues': [],
    'checks': {}
}
```

### Step 2 -- Check heading structure

```python
headings_result = await evaluate('''
(function(){
  const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'));
  const issues = [];
  let prevLevel = 0;
  const h1Count = headings.filter(h => h.tagName === 'H1').length;

  if (h1Count === 0) issues.push('No h1 element found');
  if (h1Count > 1) issues.push('Multiple h1 elements: ' + h1Count);

  headings.forEach(h => {
    const level = parseInt(h.tagName[1]);
    if (prevLevel > 0 && level > prevLevel + 1)
      issues.push('Skipped level: h' + prevLevel + ' -> h' + level + ' ("' + h.textContent.trim().substring(0, 50) + '")');
    if (!h.textContent.trim())
      issues.push('Empty heading: ' + h.tagName);
    prevLevel = level;
  });

  return {
    total: headings.length,
    h1Count,
    hierarchy: headings.map(h => ({ tag: h.tagName, text: h.textContent.trim().substring(0, 80) })),
    issues
  };
})()
''')

audit['checks']['headings'] = headings_result
for issue in headings_result.get('issues', []):
    audit['issues'].append({'check': 'headings', 'wcag': '1.3.1', 'issue': issue})
    print(f'[HEADINGS] {issue}')

if not headings_result.get('issues'):
    print('[HEADINGS] PASS')
```

### Step 3 -- Check images for alt text

```python
images_result = await evaluate('''
(function(){
  const images = Array.from(document.querySelectorAll('img'));
  const issues = [];
  let withAlt = 0, withEmptyAlt = 0, missingAlt = 0;

  images.forEach(img => {
    const alt = img.getAttribute('alt');
    const src = img.src?.substring(0, 100);
    if (alt === null) {
      missingAlt++;
      issues.push('Missing alt: ' + src);
    } else if (alt === '') {
      withEmptyAlt++;
    } else {
      withAlt++;
    }
  });

  return { total: images.length, withAlt, withEmptyAlt, missingAlt, issues };
})()
''')

audit['checks']['images'] = images_result
for issue in images_result.get('issues', []):
    audit['issues'].append({'check': 'images', 'wcag': '1.1.1', 'issue': issue})
    print(f'[IMAGES] {issue}')

if not images_result.get('issues'):
    print(f'[IMAGES] PASS ({images_result["total"]} images, {images_result["withAlt"]} with alt)')
```

### Step 4 -- Check form labels

```python
forms_result = await evaluate('''
(function(){
  const inputs = Array.from(document.querySelectorAll('input:not([type="hidden"]),select,textarea'));
  const issues = [];

  inputs.forEach(input => {
    const id = input.id;
    const ariaLabel = input.getAttribute('aria-label');
    const ariaLabelledBy = input.getAttribute('aria-labelledby');
    const title = input.getAttribute('title');
    const label = id ? document.querySelector('label[for="' + id + '"]') : null;
    const parentLabel = input.closest('label');
    const hasLabel = label || parentLabel || ariaLabel || ariaLabelledBy || title;

    if (!hasLabel) {
      issues.push({
        tag: input.tagName,
        type: input.type || 'text',
        name: input.name || '(none)',
        placeholder: input.getAttribute('placeholder') || '(none)',
        issue: 'No label or aria-label'
      });
    }
  });

  return { totalInputs: inputs.length, unlabeled: issues.length, issues };
})()
''')

audit['checks']['forms'] = forms_result
for issue in forms_result.get('issues', []):
    audit['issues'].append({'check': 'forms', 'wcag': '1.3.1', 'issue': f'Unlabeled {issue["tag"]} name={issue["name"]}'})
    print(f'[FORMS] Unlabeled: <{issue["tag"]}> type={issue["type"]} name={issue["name"]}')

if not forms_result.get('issues'):
    print(f'[FORMS] PASS ({forms_result["totalInputs"]} inputs, all labeled)')
```

### Step 5 -- Check ARIA attributes

```python
aria_result = await evaluate('''
(function(){
  const issues = [];
  const ariaElements = document.querySelectorAll('[role],[aria-label],[aria-labelledby],[aria-describedby],[aria-hidden]');

  ariaElements.forEach(el => {
    const ariaLabelledBy = el.getAttribute('aria-labelledby');
    const ariaDescribedBy = el.getAttribute('aria-describedby');

    if (ariaLabelledBy) {
      ariaLabelledBy.split(/\\s+/).forEach(id => {
        if (!document.getElementById(id))
          issues.push({ element: el.tagName, issue: 'aria-labelledby references missing id: ' + id });
      });
    }
    if (ariaDescribedBy) {
      ariaDescribedBy.split(/\\s+/).forEach(id => {
        if (!document.getElementById(id))
          issues.push({ element: el.tagName, issue: 'aria-describedby references missing id: ' + id });
      });
    }
    if (el.getAttribute('role') === 'button' && !el.getAttribute('aria-label') && !el.textContent.trim())
      issues.push({ element: el.tagName, issue: 'Button role with no accessible name' });
    if (el.getAttribute('aria-hidden') === 'true' && el.querySelector('a,button,input,select,textarea,[tabindex]'))
      issues.push({ element: el.tagName, issue: 'aria-hidden on element with focusable children' });
  });

  return { totalAriaElements: ariaElements.length, issues };
})()
''')

audit['checks']['aria'] = aria_result
for issue in aria_result.get('issues', []):
    audit['issues'].append({'check': 'aria', 'wcag': '4.1.2', 'issue': issue['issue']})
    print(f'[ARIA] {issue["issue"]}')

if not aria_result.get('issues'):
    print(f'[ARIA] PASS ({aria_result["totalAriaElements"]} ARIA elements)')
```

### Step 6 -- Check landmark regions

```python
landmarks_result = await evaluate('''
(function(){
  const landmarks = {
    banner: document.querySelectorAll('header,[role="banner"]').length,
    navigation: document.querySelectorAll('nav,[role="navigation"]').length,
    main: document.querySelectorAll('main,[role="main"]').length,
    contentinfo: document.querySelectorAll('footer,[role="contentinfo"]').length,
    complementary: document.querySelectorAll('aside,[role="complementary"]').length,
    search: document.querySelectorAll('[role="search"]').length
  };
  const issues = [];
  if (landmarks.main === 0) issues.push('No main landmark');
  if (landmarks.main > 1) issues.push('Multiple main landmarks: ' + landmarks.main);
  if (landmarks.banner === 0) issues.push('No banner/header landmark');
  if (landmarks.navigation === 0) issues.push('No navigation landmark');
  if (landmarks.contentinfo === 0) issues.push('No footer/contentinfo landmark');
  return { landmarks, issues };
})()
''')

audit['checks']['landmarks'] = landmarks_result
for issue in landmarks_result.get('issues', []):
    audit['issues'].append({'check': 'landmarks', 'wcag': '1.3.1', 'issue': issue})
    print(f'[LANDMARKS] {issue}')

if not landmarks_result.get('issues'):
    print(f'[LANDMARKS] PASS')
```

### Step 7 -- Check links and buttons

```python
links_result = await evaluate('''
(function(){
  const issues = [];
  document.querySelectorAll('a').forEach(a => {
    const name = a.textContent.trim() || a.getAttribute('aria-label') || a.getAttribute('title') || a.querySelector('img[alt]')?.alt;
    if (!name) issues.push({ tag: 'a', href: a.href?.substring(0, 80), issue: 'No accessible name' });
    else if (['click here', 'here', 'read more', 'more', 'link'].includes(name.toLowerCase()))
      issues.push({ tag: 'a', text: name, issue: 'Non-descriptive link text' });
  });
  document.querySelectorAll('button,[role="button"]').forEach(btn => {
    const name = btn.textContent.trim() || btn.getAttribute('aria-label') || btn.getAttribute('title');
    if (!name) issues.push({ tag: btn.tagName, issue: 'Button with no accessible name' });
  });
  return { issues };
})()
''')

audit['checks']['links_buttons'] = links_result
for issue in links_result.get('issues', []):
    audit['issues'].append({'check': 'links_buttons', 'wcag': '2.4.4', 'issue': issue['issue']})
    print(f'[LINKS/BUTTONS] {issue["issue"]}')

if not links_result.get('issues'):
    print('[LINKS/BUTTONS] PASS')
```

### Step 8 -- Check keyboard navigation

```python
keyboard_result = await evaluate('''
(function(){
  const focusable = Array.from(document.querySelectorAll('a[href],button,input:not([type="hidden"]),select,textarea,[tabindex]'));
  const issues = [];
  const positiveTabindex = focusable.filter(el => parseInt(el.getAttribute('tabindex')) > 0);
  const negativeTabindex = focusable.filter(el => {
    const ti = parseInt(el.getAttribute('tabindex'));
    return ti < 0 && ['A','BUTTON','INPUT','SELECT','TEXTAREA'].includes(el.tagName);
  });

  if (positiveTabindex.length > 0)
    issues.push('Elements with positive tabindex (disrupts tab order): ' + positiveTabindex.length);
  if (negativeTabindex.length > 0)
    issues.push('Interactive elements removed from tab order: ' + negativeTabindex.length);

  return { totalFocusable: focusable.length, positiveTabindex: positiveTabindex.length, negativeTabindex: negativeTabindex.length, issues };
})()
''')

audit['checks']['keyboard'] = keyboard_result
for issue in keyboard_result.get('issues', []):
    audit['issues'].append({'check': 'keyboard', 'wcag': '2.4.3', 'issue': issue})
    print(f'[KEYBOARD] {issue}')

if not keyboard_result.get('issues'):
    print(f'[KEYBOARD] PASS ({keyboard_result["totalFocusable"]} focusable elements)')
```

### Step 9 -- Compile audit report

```python
import json

total_issues = len(audit['issues'])
checks_passed = sum(1 for c in audit['checks'].values() if not c.get('issues'))
checks_total = len(audit['checks'])

print(f'\n=== Accessibility Audit Report ===')
print(f'URL: {audit["url"]}')
print(f'Title: {audit["title"]}')
print(f'Checks: {checks_passed}/{checks_total} passed')
print(f'Total issues: {total_issues}')

if total_issues > 0:
    print(f'\nIssues by WCAG criterion:')
    by_wcag = {}
    for issue in audit['issues']:
        by_wcag.setdefault(issue['wcag'], []).append(issue)
    for wcag, issues in sorted(by_wcag.items()):
        print(f'  {wcag}: {len(issues)} issues')
        for i in issues[:3]:
            print(f'    - {i["issue"]}')
        if len(issues) > 3:
            print(f'    ... and {len(issues) - 3} more')
```

## WCAG Quick Reference

| Check | WCAG Criterion | Level |
|-------|---------------|-------|
| Images have alt text | 1.1.1 Non-text Content | A |
| Heading hierarchy is logical | 1.3.1 Info and Relationships | A |
| Form inputs have labels | 1.3.1 Info and Relationships | A |
| Link purpose is clear | 2.4.4 Link Purpose (In Context) | A |
| Landmark regions present | 1.3.1 Info and Relationships | A |
| Focus order is logical | 2.4.3 Focus Order | A |
| ARIA attributes valid | 4.1.2 Name, Role, Value | A |

## Tips

- Store results in the `audit` dict -- variables persist between `execute_code` calls.
- Run audits on multiple pages, not just the homepage.
- ARIA misuse is often worse than no ARIA at all.
- The heading check catches the most common structural issues.
- Missing form labels are the most common form accessibility failure.
