"""Shared trusted Playwright task programs for the native MCP and CLI arms.

Each program is a JavaScript function invoked with one ``page`` argument by
the official code entry points (``browser_run_code_unsafe`` for MCP and
``run-code`` for the CLI). Programs use only the public Playwright page API.
They do not import the harness, read fixture state, or contain credentials.
"""

from __future__ import annotations

from pathlib import Path

from ..fixtures.server import BENCH_USER

SECRET_PASSWORD = "SECRET_BENCH_PASSWORD"
INTERACTIVE_TASKS = ("wizard", "modal", "download", "tabs", "auth")
SCRAPING_TASKS = ("static_article", "js_list", "pagination", "table", "asset_heavy")


def _program(body: str) -> str:
    return "async (page) => {\n" + body.strip() + "\n}"


def _url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def wizard_program(base_url: str) -> str:
    return _program(
        f"""
  await page.goto({_url(base_url, "/wizard")!r});
  await page.fill('#name', 'Alice Bench');
  await page.fill('#email', 'alice@example.com');
  await page.check('#country-ca');
  await page.waitForSelector('#province-group:not([hidden])');
  await page.check('#province-on');
  await page.click('#step1-next');
  await page.waitForSelector('#step2');
  await page.check('#plan-pro');
  await page.fill('#seats', '5');
  await page.click('#step2-next');
  await page.waitForSelector('#step3');
  await page.check('#terms');
  await page.click('#submit');
  await page.waitForSelector('#quote');
  return {{ quote: await page.textContent('#quote') }};
"""
    )


def modal_program(base_url: str) -> str:
    return _program(
        f"""
  await page.goto({_url(base_url, "/modal")!r});
  await page.click('#open-modal');
  await page.waitForSelector('#modal:not([hidden])');
  await page.keyboard.press('Escape');
  await page.waitForSelector('#modal', {{state: 'hidden' }});
  await page.click('#open-modal');
  await page.fill('#project-search', 'Atlas');
  await page.waitForFunction(() => [...document.querySelectorAll('#project-list [role=option]')]
    .filter((option) => !option.hidden).length === 1);
  const optionId = await page.evaluate(() => Array.from(
    document.querySelectorAll('#project-list [role=option]'))
    .find((option) => !option.hidden && option.textContent.trim() === 'Project Atlas')?.id);
  if (!optionId) throw new Error('Project Atlas option has no stable fixture id');
  await page.click('#' + optionId);
  await page.waitForFunction(() =>
    document.getElementById('selected-value').textContent === 'Project Atlas');
  return {{ selected: await page.textContent('#selected-value') }};
"""
    )


def download_program(base_url: str) -> str:
    return _program(
        f"""
  await page.goto({_url(base_url, "/download")!r});
  await page.click('#download-report');
  return {{ clicked: true }};
"""
    )


def tabs_program(base_url: str) -> str:
    return _program(
        f"""
  await page.goto({_url(base_url, "/tabs")!r});
  const [detail] = await Promise.all([
    page.waitForEvent('popup'),
    page.click('#compare'),
  ]);
  await detail.waitForLoadState('load');
  await detail.click('#approve');
  await detail.waitForSelector('#approved');
  await page.waitForFunction(() =>
    document.getElementById('status').textContent === 'approved:42');
  const title = await page.title();
  await detail.close();
  return {{ original_title: title }};
"""
    )


def login_program(base_url: str) -> str:
    """Fill the non-secret field. The adapter types the password natively."""
    return _program(
        f"""
  await page.goto({_url(base_url, "/login")!r});
  await page.fill('#username', {BENCH_USER!r});
  await page.waitForSelector('#password');
  return {{ ready: true }};
"""
    )


def account_program(base_url: str) -> str:
    return _program(
        f"""
  await page.goto({_url(base_url, "/account")!r});
  await page.waitForSelector('#secret');
  return {{ secret: await page.textContent('#secret') }};
"""
    )


def scraping_program(task: str, base_url: str) -> str:
    if task == "static_article":
        return _program(
            f"""
  await page.goto({_url(base_url, "/static")!r});
  return {{ content: await page.evaluate(() => document.body.innerText) }};
"""
        )
    if task == "js_list":
        return _program(
            f"""
  await page.goto({_url(base_url, "/js-list")!r});
  await page.waitForSelector('#items[data-count="30"]');
  return {{ skus: await page.$$eval('#items [data-sku]',
    (nodes) => nodes.map((node) => node.dataset.sku)) }};
"""
        )
    if task == "pagination":
        return _program(
            f"""
  await page.goto({_url(base_url, "/pagination?page=1")!r});
  const ids = [];
  for (;;) {{
    ids.push(...await page.$$eval('tbody tr[data-id]',
      (rows) => rows.map((row) => row.dataset.id)));
    const next = await page.$('#next');
    if (!next) break;
    const current = new URL(page.url()).searchParams.get('page');
    await Promise.all([
      page.waitForURL((url) => new URL(url).searchParams.get('page') !== current),
      next.click(),
    ]);
  }}
  return {{ ids }};
"""
        )
    if task == "table":
        return _program(
            f"""
  await page.goto({_url(base_url, "/table")!r});
  return {{ rows: await page.$$eval('tbody tr', (rows) => rows.map((row) =>
    [row.dataset.id, ...Array.from(row.cells).map((cell) => cell.textContent.trim())])) }};
"""
        )
    if task == "asset_heavy":
        return _program(
            f"""
  await page.goto({_url(base_url, "/assets")!r});
  await page.waitForSelector('#headlines[data-count="20"]');
  return {{ content: await page.evaluate(() => document.body.innerText) }};
"""
        )
    raise ValueError(f"unknown scraping task {task}")


def task_program(task: str, base_url: str, downloads: Path | None = None) -> str:
    """Return one native code program for an interactive or scraping task.

    ``downloads`` is accepted for the shared adapter bridge. The download task
    relies on the native automatic download save, so the program only clicks.
    The auth task returns the post-restart account read. Use ``login_program``
    for the pre-restart step; the password is typed through the native secret
    input, never embedded here.
    """
    del downloads
    if task == "wizard":
        return wizard_program(base_url)
    if task == "modal":
        return modal_program(base_url)
    if task == "download":
        return download_program(base_url)
    if task == "tabs":
        return tabs_program(base_url)
    if task == "auth":
        return account_program(base_url)
    return scraping_program(task, base_url)
