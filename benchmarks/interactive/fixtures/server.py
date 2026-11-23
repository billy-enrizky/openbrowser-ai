"""Deterministic local fixture server for the interactive browser benchmark.

Every task gets its own route family and its own state bucket. The server is
started once per run on an ephemeral loopback port, so a run cannot inherit
state from an earlier run. All success checks read server state, not the
adapter's self-report.

No external network, no third-party dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BENCH_USER = "bench"
BENCH_PASSWORD = "bench-pass-42"
SESSION_TOKEN = "bench-session-token-42"
ACCOUNT_SECRET = "ACCOUNT-SECRET-42"
REPORT_ROWS = 100
MODAL_OPTIONS = 200


def report_csv() -> bytes:
    lines = ["id,name,quantity,price"]
    for i in range(1, REPORT_ROWS + 1):
        lines.append(f"{i},Widget {i:03d},{i % 9 + 1},{i * 125 / 100:.2f}")
    return ("\n".join(lines) + "\n").encode()


REPORT_BYTES = report_csv()
REPORT_SHA256 = hashlib.sha256(REPORT_BYTES).hexdigest()
REPORT_SIZE = len(REPORT_BYTES)


def modal_names() -> list[str]:
    seed = [
        "Project Alpha",
        "Project Aurora",
        "Project Atlas",
        "Project Beacon",
        "Project Cobalt",
        "Project Delta",
        "Project Ember",
        "Project Fjord",
        "Project Granite",
        "Project Harbor",
        "Project Ion",
        "Project Juniper",
        "Project Kestrel",
        "Project Lumen",
        "Project Meridian",
        "Project Nimbus",
        "Project Orbit",
        "Project Pine",
        "Project Quartz",
        "Project Ridge",
    ]
    names = list(seed)
    index = len(seed)
    while len(names) < MODAL_OPTIONS:
        names.append(f"Project Node {index:03d}")
        index += 1
    return names


MODAL_NAMES = modal_names()


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass
class BenchState:
    wizard_step1: int = 0
    wizard_step2: int = 0
    wizard_submits: int = 0
    wizard_submissions: list[dict] = field(default_factory=list)
    modal_events: list[dict] = field(default_factory=list)
    download_requests: int = 0
    download_completed: int = 0
    download_started_at: float | None = None
    tab_events: list[dict] = field(default_factory=list)
    auth_login_posts: int = 0
    auth_login_failures: int = 0
    auth_account_views: int = 0
    request_log: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_request(self, path: str, status: int, body: bytes) -> None:
        with self.lock:
            self.request_log.append(
                {"path": path, "status": status, "bytes": len(body), "at": time.monotonic()}
            )

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "wizard_step1": self.wizard_step1,
                "wizard_step2": self.wizard_step2,
                "wizard_submits": self.wizard_submits,
                "wizard_submissions": list(self.wizard_submissions),
                "modal_events": list(self.modal_events),
                "download_requests": self.download_requests,
                "download_completed": self.download_completed,
                "download_started_at": self.download_started_at,
                "tab_events": list(self.tab_events),
                "auth_login_posts": self.auth_login_posts,
                "auth_login_failures": self.auth_login_failures,
                "auth_account_views": self.auth_account_views,
                "request_count": len(self.request_log),
                "request_bytes": sum(item["bytes"] for item in self.request_log),
                "requests": list(self.request_log),
            }


def _page(title: str, body: str, script: str = "") -> bytes:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_esc(title)}</title></head>
<body>{body}<script>{script}</script></body></html>""".encode()


def _wizard_step1(error: str = "") -> bytes:
    body = f"""
<h1>Quote wizard</h1>
<p>Step 1 of 3</p>
{'<p id="error" class="error">' + _esc(error) + '</p>' if error else ''}
<form id="step1" method="post" action="/wizard/step1">
  <label for="name">Full name</label>
  <input id="name" name="name" autocomplete="off">
  <label for="email">Email</label>
  <input id="email" name="email" type="email" autocomplete="off">
  <fieldset>
    <legend>Country</legend>
    <label><input type="radio" id="country-us" name="country" value="US"> United States</label>
    <label><input type="radio" id="country-ca" name="country" value="CA"> Canada</label>
  </fieldset>
  <fieldset id="province-group" hidden>
    <legend>Province</legend>
    <label><input type="radio" id="province-on" name="province" value="ON"> Ontario</label>
    <label><input type="radio" id="province-bc" name="province" value="BC"> British Columbia</label>
  </fieldset>
  <fieldset id="state-group" hidden>
    <legend>State</legend>
    <label for="state">State</label>
    <input id="state" name="state" autocomplete="off">
  </fieldset>
  <button id="step1-next" type="submit">Next</button>
</form>"""
    script = """
const country = document.querySelectorAll('input[name="country"]');
const provinceGroup = document.getElementById('province-group');
const stateGroup = document.getElementById('state-group');
for (const radio of country) {
  radio.addEventListener('change', () => {
    const value = document.querySelector('input[name="country"]:checked')?.value;
    provinceGroup.hidden = value !== 'CA';
    stateGroup.hidden = value !== 'US';
  });
}
document.getElementById('step1').addEventListener('submit', (event) => {
  const name = document.getElementById('name').value.trim();
  const email = document.getElementById('email').value.trim();
  const picked = document.querySelector('input[name="country"]:checked');
  if (!name || !email.includes('@') || !picked) {
    event.preventDefault();
    let box = document.getElementById('error');
    if (!box) {
      box = document.createElement('p');
      box.id = 'error';
      box.className = 'error';
      document.querySelector('h1').after(box);
    }
    box.textContent = 'Complete every required field';
  }
});
"""
    return _page("Quote wizard", body, script)


def _wizard_step2(token: str, error: str = "") -> bytes:
    body = f"""
<h1>Quote wizard</h1>
<p>Step 2 of 3</p>
{'<p id="error" class="error">' + _esc(error) + '</p>' if error else ''}
<form id="step2" method="post" action="/wizard/step2">
  <input type="hidden" name="token" value="{_esc(token)}">
  <fieldset>
    <legend>Plan</legend>
    <label><input type="radio" id="plan-starter" name="plan" value="starter"> Starter</label>
    <label><input type="radio" id="plan-pro" name="plan" value="pro"> Pro</label>
    <label><input type="radio" id="plan-enterprise" name="plan" value="enterprise"> Enterprise</label>
  </fieldset>
  <label for="seats">Seats</label>
  <input id="seats" name="seats" type="number" min="1" max="50" autocomplete="off">
  <button id="step2-next" type="submit">Next</button>
</form>"""
    script = """
document.getElementById('step2').addEventListener('submit', (event) => {
  const plan = document.querySelector('input[name="plan"]:checked');
  const seats = Number(document.getElementById('seats').value);
  if (!plan || !Number.isInteger(seats) || seats < 1) {
    event.preventDefault();
    let box = document.getElementById('error');
    if (!box) {
      box = document.createElement('p');
      box.id = 'error';
      box.className = 'error';
      document.querySelector('h1').after(box);
    }
    box.textContent = 'Choose a plan and at least one seat';
  }
});
"""
    return _page("Quote wizard", body, script)


def _wizard_step3(token: str, summary: dict) -> bytes:
    body = f"""
<h1>Quote wizard</h1>
<p>Step 3 of 3</p>
<form id="step3" method="post" action="/wizard/step3">
  <input type="hidden" name="token" value="{_esc(token)}">
  <h2>Review</h2>
  <dl id="summary">
    <dt>Name</dt><dd id="summary-name">{_esc(summary.get('name', ''))}</dd>
    <dt>Email</dt><dd id="summary-email">{_esc(summary.get('email', ''))}</dd>
    <dt>Country</dt><dd id="summary-country">{_esc(summary.get('country', ''))}</dd>
    <dt>Province</dt><dd id="summary-province">{_esc(summary.get('province', ''))}</dd>
    <dt>Plan</dt><dd id="summary-plan">{_esc(summary.get('plan', ''))}</dd>
    <dt>Seats</dt><dd id="summary-seats">{_esc(summary.get('seats', ''))}</dd>
  </dl>
  <label><input type="checkbox" id="terms" name="terms" value="yes"> I accept the terms</label>
  <button id="submit" type="submit">Submit</button>
</form>"""
    script = """
document.getElementById('step3').addEventListener('submit', (event) => {
  if (!document.getElementById('terms').checked) {
    event.preventDefault();
    let box = document.getElementById('error');
    if (!box) {
      box = document.createElement('p');
      box.id = 'error';
      box.className = 'error';
      document.querySelector('h1').after(box);
    }
    box.textContent = 'Accept the terms';
  }
});
"""
    return _page("Quote wizard", body, script)


def _wizard_done(quote: str) -> bytes:
    body = f"""
<h1>Quote ready</h1>
<p id="quote">{_esc(quote)}</p>
<p>Your quote is confirmed.</p>"""
    return _page("Quote ready", body)


def _modal_page() -> bytes:
    options = "\n".join(
        f'<li role="option" id="option-{i}" data-value="{_esc(name)}">{_esc(name)}</li>'
        for i, name in enumerate(MODAL_NAMES)
    )
    body = f"""
<h1>Project picker</h1>
<button id="open-modal" type="button">Choose project</button>
<p id="selected-value"></p>
<div id="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title" hidden>
  <h2 id="modal-title">Choose a project</h2>
  <input id="project-search" role="combobox" aria-expanded="false" aria-controls="project-list" autocomplete="off">
  <ul id="project-list" role="listbox">{options}</ul>
  <button id="modal-close" type="button">Cancel</button>
</div>"""
    script = """
const modal = document.getElementById('modal');
const openButton = document.getElementById('open-modal');
const closeButton = document.getElementById('modal-close');
const search = document.getElementById('project-search');
const list = document.getElementById('project-list');
const selected = document.getElementById('selected-value');
let escapes = 0;
let debounce = null;

function report(event, extra) {
  fetch('/api/modal-event', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({event: event}, extra || {})),
  });
}

function visibleOptions() {
  return Array.from(list.querySelectorAll('[role="option"]')).filter(
    (option) => option.hidden !== true
  );
}

function applyFilter() {
  const query = search.value.trim().toLowerCase();
  for (const option of list.querySelectorAll('[role="option"]')) {
    option.hidden = query.length > 0 && !option.textContent.toLowerCase().includes(query);
  }
  search.setAttribute('aria-expanded', 'true');
}

function openModal() {
  modal.hidden = false;
  openButton.setAttribute('aria-expanded', 'true');
  search.value = '';
  applyFilter();
  search.focus();
  report('open', {opens: 1});
}

function closeModal() {
  modal.hidden = true;
  openButton.setAttribute('aria-expanded', 'false');
  openButton.focus();
}

openButton.addEventListener('click', openModal);
closeButton.addEventListener('click', () => {
  closeModal();
});
search.addEventListener('input', () => {
  clearTimeout(debounce);
  debounce = setTimeout(applyFilter, 150);
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !modal.hidden) {
    escapes += 1;
    closeModal();
    report('escape', {escapes: escapes});
  }
});
list.addEventListener('click', (event) => {
  const option = event.target.closest('[role="option"]');
  if (!option) return;
  const value = option.dataset.value;
  selected.textContent = value;
  closeModal();
  report('select', {value: value, escapes: escapes});
});
"""
    return _page("Project picker", body, script)


def _download_page() -> bytes:
    body = """
<h1>Download center</h1>
<p>Generate the quarterly report.</p>
<button id="download-report" type="button">Download report</button>
<p id="download-status">idle</p>"""
    script = """
document.getElementById('download-report').addEventListener('click', () => {
  document.getElementById('download-status').textContent = 'requested';
  const a = document.createElement('a');
  a.href = '/files/report.csv';
  a.download = 'report.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
});
"""
    return _page("Download center", body, script)


def _tabs_page() -> bytes:
    body = """
<h1>Tabs Bench</h1>
<a id="compare" href="/detail?item=42" target="_blank" rel="opener">Compare plans</a>
<p id="status">waiting</p>"""
    script = """
const channel = new BroadcastChannel('bench-tabs');
channel.onmessage = (event) => {
  const status = 'approved:' + event.data.item;
  document.getElementById('status').textContent = status;
  fetch('/api/tab-event', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({event: 'status', status: status}),
  });
};
"""
    return _page("Tabs Bench", body, script)


def _detail_page() -> bytes:
    body = """
<h1>Detail 42</h1>
<button id="approve" type="button">Approve</button>
<p id="approved"></p>"""
    script = """
fetch('/api/tab-event', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({event: 'detail_loaded', opener: !!window.opener, item: 42}),
});
document.getElementById('approve').addEventListener('click', () => {
  localStorage.setItem('bench-approval', '42');
  new BroadcastChannel('bench-tabs').postMessage({item: 42});
  document.getElementById('approved').textContent = 'approved';
  fetch('/api/tab-event', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({event: 'approve', item: 42}),
  });
});
window.addEventListener('pagehide', () => {
  navigator.sendBeacon(
    '/api/tab-event',
    JSON.stringify({event: 'detail_closed', item: 42})
  );
});
"""
    return _page("Detail 42", body, script)


def _login_page(error: str = "") -> bytes:
    body = f"""
<h1>Sign in</h1>
{'<p id="error" class="error">' + _esc(error) + '</p>' if error else ''}
<form id="login" method="post" action="/login">
  <label for="username">Username</label>
  <input id="username" name="username" autocomplete="username">
  <label for="password">Password</label>
  <input id="password" name="password" type="password" autocomplete="current-password">
  <button id="login-submit" type="submit">Sign in</button>
</form>"""
    return _page("Sign in", body)


def _account_page() -> bytes:
    body = f"""
<h1>Account</h1>
<p id="secret">{ACCOUNT_SECRET}</p>"""
    return _page("Account", body)


def _sign(payload: dict, secret: bytes) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(secret, raw, hashlib.sha256).hexdigest()


def _verify(token: str, payload: dict, secret: bytes) -> bool:
    return hmac.compare_digest(token, _sign(payload, secret))


class FixtureHandler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, *_args, **_kwargs) -> None:
        return

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
        extra_headers: dict | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        self.server.state.record_request(self.path, status, body)

    def _redirect(self, location: str, extra_headers: dict | None = None) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.server.state.record_request(self.path, 302, b"")

    def _read_form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode()
        return {key: values[0] for key, values in parse_qs(raw).items()}

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode()
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/__state":
            self._send(200, json.dumps(self.server.state.snapshot()).encode(), "application/json")
        elif path == "/wizard":
            self._send(200, _wizard_step1())
        elif path == "/modal":
            self._send(200, _modal_page())
        elif path == "/download":
            self._send(200, _download_page())
        elif path == "/files/report.csv":
            with self.server.state.lock:
                self.server.state.download_requests += 1
                self.server.state.download_started_at = time.monotonic()
            time.sleep(0.3)
            with self.server.state.lock:
                self.server.state.download_completed += 1
            self._send(
                200,
                REPORT_BYTES,
                "text/csv; charset=utf-8",
                {"Content-Disposition": 'attachment; filename="report.csv"'},
            )
        elif path == "/tabs":
            self._send(200, _tabs_page())
        elif path == "/detail":
            self._send(200, _detail_page())
        elif path == "/login":
            self._send(200, _login_page())
        elif path == "/account":
            cookie = self.headers.get("Cookie", "")
            if f"bench_session={SESSION_TOKEN}" in cookie:
                with self.server.state.lock:
                    self.server.state.auth_account_views += 1
                self._send(200, _account_page())
            else:
                self._send(403, b"forbidden", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/__reset":
            self.server.state = BenchState()
            self._send(200, b'{"ok":true}', "application/json")
            return
        if path == "/wizard/step1":
            form = self._read_form()
            name = form.get("name", "").strip()
            email = form.get("email", "").strip()
            country = form.get("country", "")
            province = form.get("province", "")
            if path == "/wizard/step1":
                with self.server.state.lock:
                    self.server.state.wizard_step1 += 1
            if not name or "@" not in email or country not in {"US", "CA"}:
                self._send(400, _wizard_step1("Complete every required field"))
                return
            if country == "CA" and province != "ON":
                self._send(400, _wizard_step1("Choose Ontario"))
                return
            summary = {"name": name, "email": email, "country": country, "province": province}
            token = _sign(summary, self.server.secret)
            self.server.token_payloads[token] = summary
            self._send(200, _wizard_step2(token))
            return
        if path == "/wizard/step2":
            form = self._read_form()
            # The summary is carried in the signed token, not the form, so that
            # a client cannot rewrite earlier answers.
            token = form.get("token", "")
            plan = form.get("plan", "")
            try:
                seats = int(form.get("seats", "0"))
            except ValueError:
                seats = 0
            summary = self.server.token_payloads.get(token)
            with self.server.state.lock:
                self.server.state.wizard_step2 += 1
            if summary is None or not _verify(token, summary, self.server.secret):
                self._send(400, _wizard_step1("Session expired"))
                return
            if plan not in {"starter", "pro", "enterprise"} or seats < 1:
                self._send(400, _wizard_step2(token, "Choose a plan and at least one seat"))
                return
            full = dict(summary)
            full.update({"plan": plan, "seats": seats})
            final_token = _sign(full, self.server.secret)
            self.server.token_payloads[final_token] = full
            self._send(200, _wizard_step3(final_token, full))
            return
        if path == "/wizard/step3":
            form = self._read_form()
            token = form.get("token", "")
            payload = self.server.token_payloads.get(token)
            with self.server.state.lock:
                self.server.state.wizard_submits += 1
            if payload is None or not _verify(token, payload, self.server.secret):
                self._send(400, _wizard_step1("Session expired"))
                return
            if form.get("terms") != "yes":
                self._send(400, _wizard_step3(token, payload))
                return
            with self.server.state.lock:
                quote = f"Q-{len(self.server.state.wizard_submissions) + 1}"
                record = dict(payload)
                record["terms"] = True
                record["quote"] = quote
                self.server.state.wizard_submissions.append(record)
            self._send(200, _wizard_done(quote))
            return
        if path == "/api/modal-event":
            event = self._read_json()
            with self.server.state.lock:
                self.server.state.modal_events.append(event)
            self._send(200, b'{"ok":true}', "application/json")
            return
        if path == "/api/tab-event":
            event = self._read_json()
            with self.server.state.lock:
                self.server.state.tab_events.append(event)
            self._send(200, b'{"ok":true}', "application/json")
            return
        if path == "/login":
            form = self._read_form()
            with self.server.state.lock:
                self.server.state.auth_login_posts += 1
            if form.get("username") == BENCH_USER and form.get("password") == BENCH_PASSWORD:
                self._redirect(
                    "/account",
                    {"Set-Cookie": f"bench_session={SESSION_TOKEN}; Path=/; Max-Age=3600; HttpOnly; SameSite=Lax"},
                )
            else:
                with self.server.state.lock:
                    self.server.state.auth_login_failures += 1
                self._send(401, _login_page("Wrong username or password"))
            return
        self._send(404, b"not found", "text/plain")


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int] = ("127.0.0.1", 0)):
        super().__init__(address, FixtureHandler)
        self.state = BenchState()
        self.secret = hashlib.sha256(f"fixture-{time.monotonic_ns()}".encode()).digest()
        self.token_payloads: dict[str, dict] = {}

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class running_fixture:
    """Context manager that starts a fresh fixture server and stops it."""

    def __init__(self) -> None:
        self.server = FixtureServer()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> FixtureServer:
        self.thread.start()
        return self.server

    def __exit__(self, *_exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def expected_download_path(directory: Path, name: str = "report.csv") -> Path:
    return directory / name
