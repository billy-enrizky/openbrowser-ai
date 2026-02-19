---
name: form-filling
description: |
  Fill out web forms, submit data, and handle login or registration flows.
  Trigger when the user asks to: fill a form, submit data on a website, log in to a site,
  register an account, complete a checkout, enter information into fields, or automate form submission.
---

# Form Filling

Automate filling web forms including login, registration, checkout, and multi-step form wizards using Python code execution.

All code runs in a persistent namespace via `execute_code`. All browser functions are async -- use `await`.

## Workflow

### Step 1 -- Navigate to the form page

```python
await navigate('https://example.com/login')
state = await browser.get_browser_state_summary()
print(f'Page: {state.title} ({state.url})')
print(f'Interactive elements: {len(state.dom_state.selector_map)}')
```

### Step 2 -- Discover form fields

```python
# List all interactive elements with their indices
state = await browser.get_browser_state_summary()
for index, element in state.dom_state.selector_map.items():
    tag = element.tag_name
    text = element.get_all_children_text(max_depth=2)[:60]
    placeholder = element.attributes.get('placeholder', '')
    input_type = element.attributes.get('type', '')
    name = element.attributes.get('name', '')
    print(f'[{index}] <{tag}> type={input_type} name={name} placeholder="{placeholder}" text="{text}"')
```

### Step 3 -- Fill text inputs

```python
# Fill fields using their indices from Step 2
await input_text(index=5, text='user@example.com')
await input_text(index=7, text='secure-password')
```

For fields that need clearing first:

```python
await click(index=5)
await evaluate('document.activeElement.select()')
await input_text(index=5, text='new-value')
```

### Step 4 -- Handle dropdowns

Standard HTML select elements:

```python
await select_dropdown(index=12, text='United States')
```

To see available options first:

```python
options = await dropdown_options(index=12)
print(options)
```

Custom dropdown components:

```python
await evaluate('''
(function(){
  const select = document.querySelector('select#country');
  select.value = 'US';
  select.dispatchEvent(new Event('change', { bubbles: true }));
})()
''')
```

### Step 5 -- Handle checkboxes and radio buttons

```python
await click(index=15)  # Click checkbox/radio

# Verify state
checked = await evaluate('document.querySelector("input[name=\\"agree\\"]").checked')
print(f'Checkbox checked: {checked}')
```

### Step 6 -- Submit the form

```python
await click(index=20)  # Click submit button
await wait(2)

# Verify submission
state = await browser.get_browser_state_summary()
print(f'After submit: {state.url}')
```

Or submit via JavaScript:

```python
await evaluate('document.querySelector("form").submit()')
```

### Step 7 -- Verify submission result

```python
# Check for success/error messages
result = await evaluate('''
(function(){
  const success = document.querySelector('.success, .alert-success, [role="alert"]');
  const error = document.querySelector('.error, .alert-danger, .validation-error');
  return {
    success: success?.textContent?.trim(),
    error: error?.textContent?.trim(),
    url: window.location.href
  };
})()
''')
print(result)
```

### Step 8 -- Handle multi-step forms

```python
for step in range(1, 5):
    # Discover fields for current step
    state = await browser.get_browser_state_summary()
    print(f'Step {step}: {len(state.dom_state.selector_map)} elements')

    # Fill fields (indices vary per step)
    # ... fill fields here ...

    # Click Next/Continue
    # Find the next button
    for idx, el in state.dom_state.selector_map.items():
        text = el.get_all_children_text(max_depth=1).lower()
        if 'next' in text or 'continue' in text:
            await click(index=idx)
            await wait(2)
            break
```

## Tips

- Always discover fields with `browser.get_browser_state_summary()` before typing -- do not guess element indices.
- For sensitive data (passwords, tokens), confirm with the user before entering values.
- Use `evaluate()` to bypass custom components that do not respond to standard click/type.
- Variables persist between calls, so you can store field indices in one call and use them in the next.
- Check for CAPTCHA or bot detection; notify the user if manual intervention is needed.
