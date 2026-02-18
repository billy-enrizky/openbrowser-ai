---
name: form-filling
description: |
  Fill out web forms, submit data, and handle login or registration flows.
  Trigger when the user asks to: fill a form, submit data on a website, log in to a site,
  register an account, complete a checkout, enter information into fields, or automate form submission.
---

# Form Filling

Automate filling web forms including login, registration, checkout, and multi-step form wizards.

## Workflow

### Step 1 -- Navigate to the form page

Use `browser_navigate` to open the page containing the form.

```
browser_navigate(url="https://example.com/login")
```

### Step 2 -- Discover form fields

Use `browser_get_state` with `compact=false` to get a full list of interactive elements on the page, including input fields, dropdowns, checkboxes, and buttons.

```
browser_get_state(compact=false)
```

This returns each element with an index number, tag type, and any label or placeholder text.

For complex forms, use `browser_get_state` with filter params to find specific fields:

```
browser_get_state(filter_by="tag", filter_query="input")
browser_get_state(filter_by="text", filter_query="email")
browser_get_state(filter_by="attribute", filter_query="password")
```

### Step 3 -- Fill text inputs

Use `browser_type` with the element index from Step 2 to enter text into each field.

```
browser_type(index=<email_field_index>, text="user@example.com")
browser_type(index=<password_field_index>, text="secure-password")
```

For fields that need to be cleared before typing, click the field first, select all, then type:

```
browser_click(index=<field_index>)
browser_execute_js(expression="document.activeElement.select()")
browser_type(index=<field_index>, text="new value")
```

### Step 4 -- Handle dropdowns and select elements

For standard HTML `<select>` elements, use `browser_click` to open the dropdown, then click the desired option:

```
browser_click(index=<select_index>)
browser_get_state(filter_by="text", filter_query="Option Text")
browser_click(index=<option_index>)
```

For custom dropdown components, use `browser_execute_js`:

```
browser_execute_js(expression="document.querySelector('select#country').value = 'US'; document.querySelector('select#country').dispatchEvent(new Event('change', { bubbles: true }))")
```

### Step 5 -- Handle checkboxes and radio buttons

Click the checkbox or radio button element directly:

```
browser_click(index=<checkbox_index>)
```

Verify the state after clicking:

```
browser_execute_js(expression="document.querySelector('input[name=\"agree\"]').checked")
```

### Step 6 -- Submit the form

Find and click the submit button:

```
browser_get_state(filter_by="text", filter_query="Submit")
browser_click(index=<submit_button_index>)
```

Alternatively, submit via JavaScript if the button is hard to locate:

```
browser_execute_js(expression="document.querySelector('form').submit()")
```

### Step 7 -- Verify submission

After submission, verify the result by checking the page content:

```
browser_get_state(compact=true)
```

Check for success or error messages:

```
browser_get_text(search="success|thank you|welcome", case_insensitive=true)
browser_get_text(search="error|invalid|failed", case_insensitive=true)
```

For redirects after login, verify the URL changed:

```
browser_get_state(compact=true)
```

### Step 8 -- Handle multi-step forms

For form wizards with multiple pages:

1. Fill fields on the current step (Steps 3-5).
2. Click "Next" or "Continue":
   ```
   browser_get_state(filter_by="text", filter_query="Next")
   browser_click(index=<next_button_index>)
   ```
3. Wait for the next step to load, then call `browser_get_state(compact=false)` again to discover new fields.
4. Repeat until all steps are complete, then submit on the final step.

### Step 9 -- Clean up

Close the browser session when done:

```
browser_session(action="close_all")
```

## Tips

- Always use `browser_get_state(compact=false)` to discover fields before typing; do not guess element indices.
- For sensitive data (passwords, tokens), confirm with the user before entering values.
- Check for CAPTCHA or bot detection; notify the user if manual intervention is needed.
- For forms with client-side validation, use `browser_get_text` with `search` to check for validation error messages after each field.
- Use `browser_execute_js` to bypass tricky custom components that do not respond to standard click/type interactions.
