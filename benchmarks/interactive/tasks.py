"""Task definitions and exact success oracles for the interactive benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .fixtures.server import ACCOUNT_SECRET, REPORT_SHA256, REPORT_SIZE


@dataclass(frozen=True)
class TaskSpec:
    name: str
    title: str
    route: str
    description: str
    timeout_s: float


TASKS: dict[str, TaskSpec] = {
    "wizard": TaskSpec(
        name="wizard",
        title="Multi-step form fill",
        route="/wizard",
        description=(
            "Complete a three-step quote wizard: name, email, Canada plus Ontario, "
            "Pro plan with 5 seats, accept terms, submit."
        ),
        timeout_s=150.0,
    ),
    "modal": TaskSpec(
        name="modal",
        title="Modal plus custom dropdown",
        route="/modal",
        description=(
            "Open a modal, close it with Escape, reopen it, type Atlas into the ARIA "
            "combobox, and select Project Atlas."
        ),
        timeout_s=150.0,
    ),
    "download": TaskSpec(
        name="download",
        title="Click-triggered download",
        route="/download",
        description=(
            "Click Download report and wait until report.csv is complete in the run "
            "download directory."
        ),
        timeout_s=150.0,
    ),
    "tabs": TaskSpec(
        name="tabs",
        title="Multi-tab flow",
        route="/tabs",
        description=(
            "Click Compare plans to open a detail tab, approve there, switch back, "
            "read the updated status, and close the detail tab."
        ),
        timeout_s=150.0,
    ),
    "auth": TaskSpec(
        name="auth",
        title="Authenticated session reuse",
        route="/login",
        description=(
            "Sign in once, restart the browser with the same profile, and read the "
            "account secret without a second login."
        ),
        timeout_s=240.0,
    ),
}


@dataclass
class OracleResult:
    ok: bool
    detail: str


def _events(state: dict, name: str) -> list[dict]:
    return [event for event in state.get("modal_events", []) if event.get("event") == name]


def _tab_events(state: dict, name: str) -> list[dict]:
    return [event for event in state.get("tab_events", []) if event.get("event") == name]


def _download_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [path for path in directory.rglob("*") if path.is_file()]


def oracle_wizard(state: dict, reported: dict, *_args) -> OracleResult:
    submissions = state.get("wizard_submissions", [])
    expected = {
        "name": "Alice Bench",
        "email": "alice@example.com",
        "country": "CA",
        "province": "ON",
        "plan": "pro",
        "seats": 5,
        "terms": True,
        "quote": "Q-1",
    }
    if len(submissions) != 1:
        return OracleResult(False, f"submissions={len(submissions)}")
    actual = submissions[0]
    for key, value in expected.items():
        if actual.get(key) != value:
            return OracleResult(False, f"{key}={actual.get(key)!r} expected {value!r}")
    if state.get("wizard_step1") != 1 or state.get("wizard_step2") != 1 or state.get("wizard_submits") != 1:
        return OracleResult(
            False,
            "step posts="
            f"{state.get('wizard_step1')}/{state.get('wizard_step2')}/{state.get('wizard_submits')}",
        )
    return OracleResult(True, "quote Q-1 with exact payload")


def oracle_modal(state: dict, reported: dict, *_args) -> OracleResult:
    opens = _events(state, "open")
    escapes = _events(state, "escape")
    selects = _events(state, "select")
    if len(opens) < 2:
        return OracleResult(False, f"opens={len(opens)}")
    if len(escapes) < 1:
        return OracleResult(False, f"escapes={len(escapes)}")
    if not selects:
        return OracleResult(False, "no select event")
    selected = selects[-1].get("value")
    if selected != "Project Atlas":
        return OracleResult(False, f"selected={selected!r}")
    if int(selects[-1].get("escapes", 0)) < 1:
        return OracleResult(False, "page did not observe Escape before select")
    return OracleResult(True, "Project Atlas selected after an Escape cycle")


def oracle_download(state: dict, reported: dict, downloads_dir: Path) -> OracleResult:
    if state.get("download_requests") != 1:
        return OracleResult(False, f"download_requests={state.get('download_requests')}")
    if state.get("download_completed") != 1:
        return OracleResult(False, f"download_completed={state.get('download_completed')}")
    files = _download_files(downloads_dir)
    matches = []
    for path in files:
        try:
            data = path.read_bytes()
        except OSError as exc:
            return OracleResult(False, f"read failed for {path}: {exc}")
        if len(data) == REPORT_SIZE and hashlib.sha256(data).hexdigest() == REPORT_SHA256:
            matches.append(path)
    if not matches:
        return OracleResult(
            False,
            f"no matching report.csv in {downloads_dir}; files={[str(p.name) for p in files][:10]}",
        )
    return OracleResult(True, f"report.csv sha256={REPORT_SHA256[:12]} size={REPORT_SIZE}")


def oracle_tabs(state: dict, reported: dict, *_args) -> OracleResult:
    loaded = _tab_events(state, "detail_loaded")
    approvals = _tab_events(state, "approve")
    statuses = [event for event in _tab_events(state, "status") if event.get("status") == "approved:42"]
    closed = _tab_events(state, "detail_closed")
    if not loaded:
        return OracleResult(False, "detail tab never loaded")
    if not any(event.get("opener") is True for event in loaded):
        return OracleResult(False, "detail tab did not see window.opener")
    if not approvals:
        return OracleResult(False, "approve click missing")
    if not statuses:
        return OracleResult(False, "original tab never received approved:42")
    if not closed:
        return OracleResult(False, "detail tab close was not observed")
    title = str(reported.get("original_title", ""))
    if "Tabs Bench" not in title:
        return OracleResult(False, f"original_title={title!r}")
    return OracleResult(True, "detail approved, original tab updated, detail closed")


def oracle_auth(state: dict, reported: dict, *_args) -> OracleResult:
    if state.get("auth_login_posts") != 1:
        return OracleResult(False, f"login_posts={state.get('auth_login_posts')}")
    if state.get("auth_login_failures") != 0:
        return OracleResult(False, f"login_failures={state.get('auth_login_failures')}")
    if state.get("auth_account_views", 0) < 2:
        return OracleResult(False, f"account_views={state.get('auth_account_views')}")
    secret = str(reported.get("secret", ""))
    if secret != ACCOUNT_SECRET:
        return OracleResult(False, f"secret={secret!r}")
    combined = str(reported.get("stdout", "")) + str(reported.get("stderr", ""))
    if "bench-pass-42" in combined:
        return OracleResult(False, "password leaked into output")
    return OracleResult(True, "one login, profile reuse, secret read")


ORACLES = {
    "wizard": oracle_wizard,
    "modal": oracle_modal,
    "download": oracle_download,
    "tabs": oracle_tabs,
    "auth": oracle_auth,
}
