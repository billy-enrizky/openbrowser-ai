---
name: e2e-testing
description: |
  Test web applications end-to-end by simulating user interactions and verifying expected outcomes.
  Trigger when the user asks to: test a web app, verify a user flow, run end-to-end tests,
  QA a feature, check that a page works correctly, validate user journeys, or test a deployment.
---

# End-to-End Testing

Simulate real user interactions in a browser and verify that web applications behave correctly. Covers navigation, form interaction, content assertions, and multi-page flows.

## Workflow

### Step 1 -- Navigate to the application

Open the application under test:

```
browser_navigate(url="https://staging.example.com")
```

Verify the page loaded correctly by checking title and URL:

```
browser_get_state(compact=true)
```

### Step 2 -- Define test assertions

Before interacting, establish what success looks like. Use `browser_grep` to assert that expected content is present:

```
browser_grep(pattern="Welcome to Example App", case_insensitive=false)
```

Use `browser_search_elements` to assert that required UI elements exist:

```
browser_search_elements(query="Sign In", by="text")
browser_search_elements(query="nav", by="tag")
```

Use `browser_execute_js` for precise assertions on DOM state:

```
browser_execute_js(expression="document.querySelector('h1')?.textContent?.trim()")
browser_execute_js(expression="document.querySelectorAll('.error-message').length === 0")
```

### Step 3 -- Test user interactions

Simulate a typical user flow. For example, a login flow:

1. Find and fill the email field:
   ```
   browser_get_state(compact=false)
   browser_type(index=<email_index>, text="test@example.com")
   ```

2. Fill the password field:
   ```
   browser_type(index=<password_index>, text="test-password")
   ```

3. Click the login button:
   ```
   browser_search_elements(query="Log In", by="text")
   browser_click(index=<login_button_index>)
   ```

4. Assert the user is now logged in:
   ```
   browser_grep(pattern="Dashboard|Welcome back", case_insensitive=true)
   ```

### Step 4 -- Test navigation flows

Verify that links and navigation work correctly:

```
browser_search_elements(query="Settings", by="text")
browser_click(index=<settings_link_index>)
browser_get_state(compact=true)
```

Assert the URL changed to the expected path:

```
browser_execute_js(expression="window.location.pathname")
```

Test the back button:

```
browser_go_back()
browser_get_state(compact=true)
```

### Step 5 -- Test error states

Verify that the application handles errors gracefully. For example, submit a form with invalid data:

```
browser_type(index=<email_index>, text="not-an-email")
browser_click(index=<submit_index>)
```

Assert that validation errors appear:

```
browser_grep(pattern="valid email|invalid|required", case_insensitive=true)
browser_search_elements(query="error", by="class")
```

Assert that the page did not navigate away:

```
browser_execute_js(expression="window.location.pathname")
```

### Step 6 -- Test responsive behavior

Use `browser_execute_js` to check viewport-dependent behavior:

```
browser_execute_js(expression="window.innerWidth")
```

Check that mobile navigation elements exist or are hidden as expected:

```
browser_execute_js(expression="window.getComputedStyle(document.querySelector('.mobile-menu')).display")
```

### Step 7 -- Test across multiple pages

For multi-page flows (e.g., checkout), open pages in sequence and verify state at each step:

```
browser_navigate(url="https://staging.example.com/cart")
browser_grep(pattern="Your Cart", case_insensitive=false)
browser_click(index=<checkout_button_index>)
browser_grep(pattern="Shipping Address", case_insensitive=false)
```

Use `browser_execute_js` to verify application state (e.g., cart contents, session data):

```
browser_execute_js(expression="JSON.parse(localStorage.getItem('cart'))?.items?.length")
```

### Step 8 -- Report results

After running all test steps, compile results. Use `browser_get_state` and `browser_grep` to collect final state. Summarize:

- Which assertions passed
- Which assertions failed (with details)
- Screenshots or page content at failure points (use `browser_get_text` to capture page state)

### Step 9 -- Clean up

Close all browser sessions after testing:

```
browser_close_all()
```

## Tips

- Always verify page state after each interaction before proceeding to the next step.
- Use `browser_grep` for content assertions and `browser_execute_js` for DOM state assertions.
- Test both happy paths and error paths for thorough coverage.
- For flaky elements, use `browser_find_and_scroll` to ensure elements are in view before clicking.
- Use `browser_list_tabs` to verify no unexpected popups or new tabs opened during testing.
- Capture page content with `browser_get_text` at failure points for debugging.
