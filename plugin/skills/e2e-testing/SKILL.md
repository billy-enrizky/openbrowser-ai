---
name: e2e-testing
description: |
  Test web applications end-to-end by simulating user interactions and verifying expected outcomes.
  Trigger when the user asks to: test a web app, verify a user flow, run end-to-end tests,
  QA a feature, check that a page works correctly, validate user journeys, or test a deployment.
---

# End-to-End Testing

Simulate real user interactions and verify web application behavior using Python code execution. Covers navigation, form interaction, content assertions, and multi-page flows.

All code runs in a persistent namespace via `execute_code`. All browser functions are async -- use `await`.

## Workflow

### Step 1 -- Navigate and verify page load

```python
await navigate('https://staging.example.com')
state = await browser.get_browser_state_summary()

assert 'example' in state.url.lower(), f'Unexpected URL: {state.url}'
assert state.title, 'Page title is empty'
print(f'Page loaded: {state.title} ({state.url})')
```

### Step 2 -- Content assertions

```python
# Check for expected text using JS
has_welcome = await evaluate('''
(function(){ return !!document.body.textContent.match(/Welcome to Example App/i) })()
''')
assert has_welcome, 'Welcome message not found'

# Check specific element content
h1_text = await evaluate('document.querySelector("h1")?.textContent?.trim()')
assert h1_text == 'Example App', f'Expected "Example App", got "{h1_text}"'

# Check no error messages
error_count = await evaluate('document.querySelectorAll(".error-message").length')
assert error_count == 0, f'Found {error_count} error messages on page'

print('All content assertions passed')
```

### Step 3 -- Test user interactions (login flow)

```python
# Get form fields
state = await browser.get_browser_state_summary()
for idx, el in state.dom_state.selector_map.items():
    if el.attributes.get('type') in ('email', 'text', 'password') or el.tag_name == 'button':
        print(f'[{idx}] <{el.tag_name}> type={el.attributes.get("type", "")} placeholder="{el.attributes.get("placeholder", "")}"')

# Fill and submit
await input_text(index=3, text='test@example.com')
await input_text(index=4, text='test-password')
await click(index=5)  # Login button
await wait(2)

# Assert logged in
state = await browser.get_browser_state_summary()
assert 'dashboard' in state.url.lower() or 'welcome' in state.title.lower(), \
    f'Login may have failed. URL: {state.url}, Title: {state.title}'
print('Login test passed')
```

### Step 4 -- Test navigation flows

```python
# Click settings link
state = await browser.get_browser_state_summary()
for idx, el in state.dom_state.selector_map.items():
    if 'settings' in el.get_all_children_text(max_depth=1).lower():
        await click(index=idx)
        break

await wait(1)

# Assert URL changed
path = await evaluate('window.location.pathname')
assert '/settings' in path, f'Expected /settings path, got {path}'

# Test back button
await go_back()
await wait(1)
state = await browser.get_browser_state_summary()
print(f'After back: {state.url}')
```

### Step 5 -- Test error handling

```python
# Submit form with invalid data
await input_text(index=3, text='not-an-email')
await click(index=5)
await wait(1)

# Assert validation errors appear
errors = await evaluate('''
(function(){
  const errs = document.querySelectorAll('.error, .invalid-feedback, [aria-invalid="true"]');
  return Array.from(errs).map(e => e.textContent.trim());
})()
''')
assert len(errors) > 0, 'Expected validation errors but found none'
print(f'Validation errors shown: {errors}')

# Assert page did not navigate
path = await evaluate('window.location.pathname')
print(f'Still on: {path}')
```

### Step 6 -- Test responsive behavior

```python
viewport = await evaluate('''
(function(){
  return {
    width: window.innerWidth,
    height: window.innerHeight
  }
})()
''')
print(f'Viewport: {viewport["width"]}x{viewport["height"]}')

# Check mobile menu visibility
mobile_display = await evaluate('''
(function(){
  const el = document.querySelector('.mobile-menu');
  return el ? window.getComputedStyle(el).display : 'not found';
})()
''')
print(f'Mobile menu display: {mobile_display}')
```

### Step 7 -- Test multi-page flows

```python
test_results = []

# Cart page
await navigate('https://staging.example.com/cart')
await wait(1)
cart_title = await evaluate('document.querySelector("h1")?.textContent?.trim()')
test_results.append({'test': 'cart_page_loads', 'passed': cart_title is not None, 'detail': cart_title})

# Checkout
cart_count = await evaluate('JSON.parse(localStorage.getItem("cart"))?.items?.length || 0')
test_results.append({'test': 'cart_has_items', 'passed': cart_count > 0, 'detail': f'{cart_count} items'})

# Print results
import json
passed = sum(1 for t in test_results if t['passed'])
total = len(test_results)
print(f'\nResults: {passed}/{total} passed')
print(json.dumps(test_results, indent=2))
```

## Tips

- Use Python `assert` statements for test assertions -- they produce clear error messages on failure.
- Always verify page state after each interaction before proceeding.
- Test both happy paths and error paths for thorough coverage.
- Use `await wait(N)` after interactions that trigger page loads or animations.
- Variables persist between `execute_code` calls, so you can accumulate test results across calls.
- Use `evaluate()` for DOM state assertions and `browser.get_browser_state_summary()` for page metadata.
